


from flask import Flask, render_template, request, jsonify
import cv2
from deepface import DeepFace
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

# Step 1: Load the dataset
'''df = pd.read_csv('/home/pc-25/Downloads/AI-project/archive/labels.csv')

# Step 2: Load images and labels
images = []
labels = []

for index, row in df.iterrows():
    image_path = f'archive/Train/{row["pth"]}'

    # Check if the image exists
    if os.path.exists(image_path):
        # Load image from the path in the dataset
        image = load_img(image_path, target_size=(64, 64))  # Resize image to 64x64
        image = img_to_array(image)  # Convert to numpy array
        image = image / 255.0  # Normalize pixel values to range 0-1

        images.append(image)
        labels.append(row['label'])
    else:
        print(f"Image not found: {image_path}")

# Convert lists to numpy arrays
X = np.array(images)
y = np.array(labels)

# Step 3: Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert labels to integers
y = to_categorical(y)  # One-hot encode labels for classification

# Step 4: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build the CNN model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Flatten and dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer with softmax for multi-class classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Step 7: Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Step 8: Save the model
model.save('emotion_detection_model.h5')'''

# # Initialize Flask app
# app = Flask(__name__)

# def initialize_webcam():
#     cap = cv2.VideoCapture(0)  # Use webcam index 0 for the default camera
#     return cap

# def detect_faces(frame, face_cascade):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     return faces

# def analyze_emotion(face_crop):
#     try:
#         result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
#         return result[0]['dominant_emotion']
#     except Exception as e:
#         print(f"Error analyzing emotion: {e}")
#         return None

# @app.route('/')
# def index():
#     # Render the index.html page
#     return render_template('index.html')

# @app.route('/detect_emotion', methods=['POST'])
# def detect_emotion_route():
#     # Fetch image data from the frontend
#     data = request.json
#     image_data = data.get('image')

#     if not image_data:
#         return jsonify({"error": "No image data provided!"}), 400

#     # Decode the image from base64 to numpy array
#     image_data = image_data.split(',')[1]  # Remove the 'data:image/png;base64,' part
#     image = Image.open(BytesIO(base64.b64decode(image_data)))
#     frame = np.array(image)

#     # Initialize the face cascade
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Detect faces in the image
#     faces = detect_faces(frame, face_cascade)

#     if len(faces) > 0:
#         for (x, y, w, h) in faces:
#             face_crop = frame[y:y + h, x:x + w]
#             dominant_emotion = analyze_emotion(face_crop)

#             if dominant_emotion:
#                 return jsonify({"emotion": dominant_emotion})
#     else:
#         return jsonify({"emotion": "No faces detected!"})

#     return jsonify({"emotion": "Error capturing video!"})

# if __name__ == "__main__":
#     app.run(debug=True)
