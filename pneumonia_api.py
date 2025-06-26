from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'chest_xray_pneumonia_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['Normal', 'Pneumonia']
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index1.html')  # Serves the HTML file

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    image = preprocess_image(file_path)
    prediction = model.predict(image)[0][0]
    label = CLASS_NAMES[int(prediction > 0.5)]
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)
    
    return jsonify({'prediction': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True,port=5001)
