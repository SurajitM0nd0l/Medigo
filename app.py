from flask import Flask, request, jsonify, render_template
import easyocr
import os
import logging

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configure logging
logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def perform_ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    if not allowed_file(image.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    # Perform OCR
    try:
        result = reader.readtext(image_path)
        text_output = [detection[1] for detection in result]
        return jsonify({"text": text_output})
    except Exception as e:
        app.logger.error(f"OCR error: {e}")
        return jsonify({"error": "Failed to process the image"}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5000)


