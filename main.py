import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from flask import Flask, render_template, request, jsonify
import cv2
from deepface import DeepFace
from io import BytesIO
from PIL import Image
import base64
import tensorflow as tf

# Disable GPU
tf.config.set_visible_devices([], 'GPU')


# # Step 1: Load the dataset and images for model training
# df = pd.read_csv('/home/pc-25/Downloads/AI-project/archive/labels.csv')  # Update path if necessary

# # Step 2: Load images and labels for training
# images = []
# labels = []

# for index, row in df.iterrows():
#     image_path = f'archive/Train/{row["pth"]}'  # Adjust the path if necessary

#     # Check if the image exists
#     if os.path.exists(image_path):
#         # Load and preprocess image
#         image = load_img(image_path, target_size=(64, 64))  # Resize to 64x64
#         image = img_to_array(image)
#         image = image / 255.0  # Normalize to range 0-1
#         images.append(image)
#         labels.append(row['label'])
#     else:
#         print(f"Image not found: {image_path}")

# # Convert to numpy arrays
# X = np.array(images)
# y = np.array(labels)

# # Step 3: Encode labels
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)
# y = to_categorical(y)

# # Step 4: Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 5: Build the CNN model
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# # Flatten and dense layers
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer for multi-class

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Step 6: Train the model
# history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# # Step 7: Evaluate the model
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f"Test accuracy: {test_acc}")

# # Step 8: Save the model and label encoder
# model.save('emotion_detection_model.h5')
# np.save('label_encoder_classes.npy', label_encoder.classes_)

# Flask application
app = Flask(__name__)

# Load the trained model and label encoder
model = load_model('emotion_detection_model.h5')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

#
# Function to analyze emotion from uploaded image
def analyze_emotion_from_image(image_path):
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

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle emotion detection from uploaded image
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_route():
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

    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)