import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from flask import Flask, render_template, request, jsonify
import cv2
from deepface import DeepFace
from io import BytesIO
from PIL import Image
import base64
import tensorflow as tf

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

# Flask application
app = Flask(__name__)

# Load the trained model and label encoder
try:
    model = load_model('emotion_detection_model.h5')
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

try:
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)
except Exception as e:
    print(f"Error loading the label encoder: {e}")
    label_encoder = None

# Function to analyze emotion from uploaded image
def analyze_emotion_from_image(image_path):
    try:
        # Load image and preprocess it
        image = load_img(image_path, target_size=(64, 64))  # Make sure target size matches your model's expected input
        image = img_to_array(image)
        image = image / 255.0  # Normalize image
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict emotion
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)

        # Decode the prediction class to the corresponding label
        predicted_label = label_encoder.inverse_transform(predicted_class)

        return predicted_label[0]  # Return the predicted label
    except Exception as e:
        print(f"Error analyzing emotion from image: {e}")
        return None

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle emotion detection from uploaded image
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_route():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image data provided!"}), 400

        # Decode the base64 image data
        image_data = image_data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))

        # Convert RGBA to RGB (if necessary)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        frame = np.array(image)

        # Save image temporarily to predict
        temp_image_path = "temp_user_image.jpg"
        image.save(temp_image_path, 'JPEG')  # Save the image as JPEG

        # Analyze the emotion using the trained model
        emotion = analyze_emotion_from_image(temp_image_path)

        # Delete temporary image file
        os.remove(temp_image_path)

        if emotion is None:
            return jsonify({"error": "Error analyzing emotion from the image!"}), 500

        return jsonify({"emotion": emotion})

    except Exception as e:
        print(f"Error in emotion detection route: {e}")
        return jsonify({"error": "An error occurred while processing the image!"}), 500

if __name__ == "__main__":
    app.run(debug=True)
