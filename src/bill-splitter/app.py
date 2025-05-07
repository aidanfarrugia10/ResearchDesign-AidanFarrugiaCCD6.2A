from flask import Flask, render_template, request, jsonify
import os
import pytesseract
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    return render_template('index.html')


@app.route('/process-cropped-image', methods=['POST'])
def process_cropped_image():
    data = request.get_json()
    cropped_image_data = data['image']

    img_data = base64.b64decode(cropped_image_data.split(',')[1])
    img = Image.open(BytesIO(img_data))

    text = pytesseract.image_to_string(img)

    items = extract_items(text)
    total = extract_total(text)
    
    return jsonify({'items': items, 'total': total})


def extract_items(text):
    items = text.split('\n')
    return [item.strip() for item in items if item.strip() != '']

def extract_total(text):
    lines = text.split('\n')
    total_line = [line for line in lines if 'total' in line.lower()]
    return total_line[-1] if total_line else 'Total not found'


if __name__ == "__main__":
    app.run(debug=True)
