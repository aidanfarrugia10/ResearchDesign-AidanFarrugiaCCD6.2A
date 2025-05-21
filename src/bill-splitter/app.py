from flask import Flask, render_template, request, jsonify
import os
import base64
import json
import re
from io import BytesIO

from PIL import Image
import pytesseract
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def preprocess_image(img_pil, contrast_alpha=0.9, contrast_beta=10):
    """
    Receipt segmentation and cleanup based on OpenCV notebook approach:
    - Grayscale & bilateral filter
    - Adaptive threshold + morphology
    - Contour detection for perspective warp
    
    Parameters:
    - contrast_alpha: contrast control (1.0 means no change, <1.0 lighter contrast)
    - contrast_beta: brightness control (0 means no change, >0 increases brightness)
    """
    # Convert PIL to OpenCV format
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Adjust contrast and brightness
    adjusted = cv2.convertScaleAbs(gray, alpha=contrast_alpha, beta=contrast_beta)
    
    # 2. Denoise while preserving edges
    denoised = cv2.bilateralFilter(adjusted, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. Adaptive threshold with reduced block size for finer details
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Invert for better text extraction
        9, 2  # Smaller block size (9 instead of 11)
    )
    
    # 4. Invert back for processing
    thresh = cv2.bitwise_not(thresh)

    # 5. Morphological open to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6. Perspective warp if possible
    contours, _ = cv2.findContours(
        opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    processed = opening
    if contours:
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxW = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxH = int(max(heightA, heightB))

            dst = np.array([
                [0, 0],
                [maxW - 1, 0],
                [maxW - 1, maxH - 1],
                [0, maxH - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(img, M, (maxW, maxH))
            warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
            
            # Apply contrast adjustment to warped image
            warp_adjusted = cv2.convertScaleAbs(warp_gray, alpha=contrast_alpha, beta=contrast_beta)
            
            # Apply bilateral filter with reduced parameters
            warp_den = cv2.bilateralFilter(warp_adjusted, d=7, sigmaColor=50, sigmaSpace=50)
            
            # Apply adaptive threshold with smaller block size
            processed = cv2.adaptiveThreshold(
                warp_den, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                9, 2
            )

    # Return PIL image
    return Image.fromarray(processed)


# Updated regex for better name & price extraction
_LINE_RE = re.compile(r'''
    ^\s*
    (?:(?P<qty>\d+)[-xX]?\s*)?           # optional quantity with '-' or 'x'
    (?P<name>[a-zA-Z][-a-zA-Z0-9\s.]+?)  # item name (must start with a letter, allow hyphens within)
    \s+
    (?P<price>€?\$?£?\d+(?:[,.]\d{1,2})?) # price with currency and decimal . or ,
    \s*$
''', re.VERBOSE)


def clean_item_name(raw_name):
    """Clean up item names by removing quantity prefixes and other unwanted patterns"""
    # Remove common quantity patterns at the beginning
    name = re.sub(r'^(\d+[-xX])', '', raw_name).strip()
    # Remove any standalone digits at the beginning
    name = re.sub(r'^\d+\s+', '', name).strip()
    # Remove known patterns like "1-" prefix
    name = re.sub(r'^1-', '', name).strip()
    # Remove other numeric prefixes
    name = re.sub(r'^\d+[A-Z]', '', name).strip()
    return name


def extract_items(text):
    items = []
    
    # Debug: Save the exact OCR text to a file for inspection
    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'ocr_output.txt'), 'w') as f:
        f.write("--- ORIGINAL OCR TEXT ---\n")
        f.write(text)
        f.write("\n\n--- LINE BY LINE ANALYSIS ---\n")
    
    # Process each line
    for line in text.splitlines():
        line_stripped = line.strip()
        
        # Skip empty lines
        if not line_stripped:
            continue
            
        # Log line analysis
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'ocr_output.txt'), 'a') as f:
            f.write(f"\nLine: '{line_stripped}'\n")
        
        # Try to match with regex
        m = _LINE_RE.match(line_stripped)
        if not m:
            with open(os.path.join(app.config['UPLOAD_FOLDER'], 'ocr_output.txt'), 'a') as f:
                f.write("  Result: No match with regex\n")
            continue
            
        # Extract components
        qty = int(m.group('qty')) if m.group('qty') else 1
        raw_name = m.group('name').strip()
        
        # Clean up the name to remove quantity prefixes
        name = clean_item_name(raw_name)
        
        raw_price = m.group('price')
        
        # Log extracted components
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'ocr_output.txt'), 'a') as f:
            f.write(f"  Quantity: {qty}\n")
            f.write(f"  Raw Name: '{raw_name}'\n")
            f.write(f"  Cleaned Name: '{name}'\n")
            f.write(f"  Raw Price: {raw_price}\n")
        
        # Handle currency symbols in price
        price_str = raw_price
        for symbol in ['€', '$', '£']:
            price_str = price_str.replace(symbol, '')
        
        # unify comma to dot and strip other chars
        price_str = price_str.replace(',', '.')
        price_str = re.sub(r'[^0-9.]', '', price_str)
        
        try:
            price = float(price_str)
            with open(os.path.join(app.config['UPLOAD_FOLDER'], 'ocr_output.txt'), 'a') as f:
                f.write(f"  Processed Price: {price}\n")
                f.write(f"  Status: Successfully added\n")
            
            items.append({
                'quantity': qty,
                'name': name,
                'amount': price,
                'total': round(qty * price, 2)
            })
        except ValueError:
            with open(os.path.join(app.config['UPLOAD_FOLDER'], 'ocr_output.txt'), 'a') as f:
                f.write(f"  Error: Could not convert '{price_str}' to float\n")
                f.write(f"  Status: Skipped\n")
            continue
            
    return items


def calculate_total(items):
    return round(sum(item['total'] for item in items), 2)


@app.route('/', methods=['GET'])
def upload_page():
    return render_template('index.html')


@app.route('/process-cropped-image', methods=['POST'])
def process_cropped_image():
    payload = request.get_json(force=True)
    b64 = payload.get('image', '').split(',', 1)[1]
    img_raw = Image.open(BytesIO(base64.b64decode(b64)))
    
    # Get contrast parameters from request, or use defaults
    contrast_alpha = float(payload.get('contrast_alpha', 0.9))  # Default: slightly reduced contrast
    contrast_beta = float(payload.get('contrast_beta', 10))     # Default: slightly increased brightness

    raw_path = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_debug.png')
    pre_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pre_debug.png')
    img_raw.save(raw_path)

    # Apply preprocessing with contrast adjustment
    img_pre = preprocess_image(img_raw, contrast_alpha=contrast_alpha, contrast_beta=contrast_beta)
    img_pre.save(pre_path)

    # Try different OCR configurations for better results
    # Configuration optimized for receipt text
    tess_config = '--oem 1 --psm 6 -l eng'  # Using English language model without character restrictions
    text = pytesseract.image_to_string(img_pre, config=tess_config)
    
    # If we get very little text, try with a more specialized config
    if len(text.strip()) < 10:
        # Try a different page segmentation mode
        tess_config = '--oem 1 --psm 4 -l eng'  # Assume single column of text
        text = pytesseract.image_to_string(img_pre, config=tess_config)

    items = extract_items(text)
    total = calculate_total(items)

    # Add the debug OCR output to the response
    ocr_analysis_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ocr_output.txt')
    ocr_analysis = "OCR analysis not available"
    if os.path.exists(ocr_analysis_path):
        with open(ocr_analysis_path, 'r') as f:
            ocr_analysis = f.read()

    return jsonify({
        'items': items,
        'total': total,
        'debug_raw': os.path.basename(raw_path),
        'debug_pre': os.path.basename(pre_path),
        'ocr_text': text,
        'ocr_analysis': ocr_analysis,
        'applied_contrast': {
            'alpha': contrast_alpha,
            'beta': contrast_beta
        }
    })


# Route to view OCR debug info in browser
@app.route('/ocr-debug', methods=['GET'])
def ocr_debug():
    ocr_file = os.path.join(app.config['UPLOAD_FOLDER'], 'ocr_output.txt')
    if os.path.exists(ocr_file):
        with open(ocr_file, 'r') as f:
            content = f.read()
        return render_template('ocr_debug.html', ocr_text=content)
    else:
        return "No OCR debug information available. Process an image first."


if __name__ == "__main__":
    app.run(debug=True)