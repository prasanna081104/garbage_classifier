import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ✅ Load the trained model
model = load_model("waste_classifier.h5")
print("✅ Model loaded successfully!")

# ✅ Test dataset path
test_base_folder = "dataset/test"

# ✅ Class names (same order as used during training)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ✅ Degradability mapping
degradability_map = {
    'cardboard': 'Degradable',
    'paper': 'Degradable',
    'glass': 'Non-degradable',
    'metal': 'Non-degradable',
    'plastic': 'Non-degradable',
    'trash': 'Non-degradable'
}

# ✅ Predict function
def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions))

        degradability = degradability_map[predicted_class]

        print(f"🧾 File: {os.path.basename(img_path)}")
        print(f"   🧠 Predicted: {predicted_class.upper()} ({confidence:.2f})")
        print(f"   ♻️ Degradability: {degradability}\n")

    except Exception as e:
        print(f"⚠️ Error with {img_path}: {e}")

# ✅ Loop through each class folder in test set
for class_folder in os.listdir(test_base_folder):
    class_path = os.path.join(test_base_folder, class_folder)
    if os.path.isdir(class_path):
        print(f"\n🔹 Predicting for class folder: {class_folder.upper()}")
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, filename)
                predict_image(img_path)
