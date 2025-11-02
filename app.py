from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'best_waste_classifier.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)

# Define your class names (must match your training dataset order)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Map each class to degradability
DEGRADABILITY_MAP = {
    'cardboard': '♻️ Degradable',
    'paper': '♻️ Degradable',
    'trash': '🚯 Non-Degradable',
    'plastic': '🚯 Non-Degradable',
    'glass': '🚯 Non-Degradable',
    'metal': '🚯 Non-Degradable'
}


# -----------------------------
# Helper function for prediction
# -----------------------------
def predict_waste(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    preds = model.predict(img_array, verbose=0)

    # Safety checks for shape
    if preds.ndim == 1:  # single output
        preds = np.expand_dims(preds, axis=0)
    if preds.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"Model output size {preds.shape[1]} doesn’t match number of CLASS_NAMES ({len(CLASS_NAMES)}).")

    # Get top prediction
    predicted_index = np.argmax(preds[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(preds[0]) * 100)

    # Get degradability meaning
    degradability = DEGRADABILITY_MAP.get(predicted_class, "Unknown")

    return predicted_class, degradability, confidence


# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "<h3 style='color:red;'>❌ No file uploaded</h3>", 400

    file = request.files['file']
    if file.filename == '':
        return "<h3 style='color:red;'>⚠️ No file selected</h3>", 400

    # Save uploaded file
    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    try:
        predicted_class, degradability, confidence = predict_waste(file_path)
    except Exception as e:
        return f"<h3 style='color:red;'>⚠️ Error: {e}</h3>"

    # Show results
    return f"""
    <div style='font-family: Arial; margin: 50px; text-align: center;'>
        <h1>🧠 Waste Classification Result</h1>
        <div style='border:2px solid #ccc; border-radius:10px; padding:20px; display:inline-block;'>
            <img src='/{file_path}' alt='Uploaded Image' width='300' style='border-radius:10px;'><br><br>
            <h2>Class: {predicted_class}</h2>
            <h3>Category: {degradability}</h3>
        </div><br><br>
        <a href='/upload' style='text-decoration:none; background:#007bff; color:white; padding:10px 20px; border-radius:5px;'>🔙 Upload Another Image</a>
    </div>
    """


# -----------------------------
# Run the Flask app
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)

