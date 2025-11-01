# test_cnn.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random

# Load trained model
model = tf.keras.models.load_model('garbage_cnn_model.h5')
print("✅ Model loaded successfully!")

# Path to test folder
test_dir = 'dataset/test'

# Get class names from test directory
class_labels = sorted(os.listdir(test_dir))
print(f"📁 Found classes: {class_labels}\n")

# Pick one random image from each class
for class_name in class_labels:
    class_path = os.path.join(test_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        continue

    img_name = random.choice(images)
    img_path = os.path.join(class_path, img_name)

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array, verbose=0)
    predicted_class = class_labels[np.argmax(pred)]

    print(f"🧠 Actual: {class_name:10s} | Predicted: {predicted_class:10s} | File: {img_name}")
