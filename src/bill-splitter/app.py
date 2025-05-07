from flask import Flask, render_template, request, redirect
import os
import pytesseract
from PIL import Image

app = Flask(__name__)

# Directory to store uploaded files
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Allowed file extensions for the receipt images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'receipt' not in request.files:
            print("No file part")  # Debugging message if no file is found
            return redirect(request.url)
        
        file = request.files['receipt']
        
        if file.filename == '':
            print("No selected file")  # Debugging message if no file is selected
            return redirect(request.url)

        # Check if the file is allowed
        if file and allowed_file(file.filename):
            # Save the file in the 'uploads' directory
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            print(f"File saved to: {filename}")  # Debugging message for file path
            
            # Open the uploaded image
            img = Image.open(filename)
            
            # Use Tesseract to extract text
            text = pytesseract.image_to_string(img)

            return render_template('index.html', text=text)
        else:
            print("File type not allowed")
            return redirect(request.url)  # Redirect back to the upload page if invalid file type

    return render_template('index.html', text=None)

if __name__ == "__main__":
    app.run(debug=True)
