from flask import Flask, render_template, request, jsonify
import os
import pytesseract
from PIL import Image
from io import BytesIO
import base64
import json
import re

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    return render_template('index.html')

# Process image that is cropped by the user
@app.route('/process-cropped-image', methods=['POST'])
def process_cropped_image():
    if request.content_type == 'application/json':
        try:
            data = json.loads(request.data)
            cropped_image_data = data['image']
            
            img_data = base64.b64decode(cropped_image_data.split(',')[1])
            img = Image.open(BytesIO(img_data))
            
            text = pytesseract.image_to_string(img)

            items = extract_items(text)
            total = calculate_total(items)

            return jsonify({'items': items, 'total': total})
        
        except Exception as e:
            return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

    return jsonify({'error': 'Request body must be JSON'}), 400

# extraction of items using tesseract
def extract_items(text):
    item_pattern = re.compile(r'(\d+)\s+([a-zA-Z\s]+)\s+(\$?\d+(\.\d{1,2})?)')
    items = []

    lines = text.split('\n')
    for line in lines:
        match = item_pattern.search(line)
        if match:
            quantity = int(match.group(1)) 
            name = match.group(2).strip()
            amount = float(match.group(3).replace('$', '').strip())
            total_per_item = quantity * amount
            items.append({
                'quantity': quantity,
                'name': name,
                'amount': amount,
                'total': total_per_item
            })

    return items

def calculate_total(items):
    return sum(item['total'] for item in items)

if __name__ == "__main__":
    app.run(debug=True)
