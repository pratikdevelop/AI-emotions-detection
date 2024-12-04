'''
import cv2
from deepface import DeepFace

# Load the pre-trained Haar Cascade face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to grayscale for better performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # Process each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop the detected face from the frame for emotion analysis
            face_crop = frame[y:y + h, x:x + w]
            
            try:
                # Analyze the emotion of the detected face (use enforce_detection=False)
                result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)

                # Get the dominant emotion
                dominant_emotion = result[0]['dominant_emotion']
                print(f"Detected emotion: {dominant_emotion}")

                # Display the emotion on the frame
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error analyzing emotion: {e}")
    else:
        # No faces detected
        print("No faces detected!")

    # Display the image with detected faces and emotions
    cv2.imshow('Face Detection and Emotion Recognition', frame)

    # Exit when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
'''

import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

# Step 1: Load the dataset
df = pd.read_csv('/home/pc-25/Downloads/AI-project/archive/labels.csv')

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
model.save('emotion_detection_model.h5')

# Step 9: Load and use the trained model for predictions
def predict_emotion(image_path):
    # Check if the image exists before loading
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    # Load the trained model
    model = load_model('emotion_detection_model.h5')

    # Load and preprocess the new image
    image = load_img(image_path, target_size=(64, 64))
    image = img_to_array(image)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict emotion
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)

    print(f"Predicted Emotion: {predicted_label[0]}")
    return predicted_label[0]


# Example usage:
# For predicting the emotion of a new image:
new_image_path = 'path_to_new_image.jpg'  # Update with your image path
predicted_emotion = predict_emotion(new_image_path)
