import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image

# Define emotions based on FER-2013 labeling
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
NUM_CLASSES = len(emotions)

# Function to load and preprocess your dataset
def load_dataset(csv_path='/home/pc-25/Music/app/ckextended.csv', img_size=(64, 64)):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Please provide the correct path.")
    
    df = pd.read_csv(csv_path)
    X = []
    y = []
    
    for index, row in df.iterrows():
        # Parse pixel string into a 48x48 grayscale image
        pixels = np.array(row['pixels'].split(), dtype='float32')
        img = pixels.reshape(48, 48)  # FER-2013 images are 48x48
        
        # Convert grayscale to RGB by repeating the channel
        img_rgb = np.stack([img] * 3, axis=-1)
        
        # Resize to 64x64 (to match your existing model)
        img_pil = Image.fromarray(img_rgb.astype('uint8'), 'RGB')
        img_resized = img_pil.resize(img_size)
        
        # Convert to array and normalize
        img_array = img_to_array(img_resized) / 255.0
        X.append(img_array)
        y.append(row['emotion'])  # Emotion label (0-6)
    
    X = np.array(X, dtype='float32')
    y = np.array(y)
    return X, y

# Define CNN model (matching your existing 64x64 input)
def create_model(input_shape=(64, 64, 3), num_classes=NUM_CLASSES):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main training function
def train_model(csv_path='/home/pc-25/Music/app/ckextended.csv'):
    # Load and preprocess the dataset
    X, y = load_dataset(csv_path)
    print(f"Loaded {len(X)} samples from the dataset.")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=NUM_CLASSES)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # Create and train the model
    model = create_model()
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Save the enhanced model and label encoder
    model.save('emotion_detection_model_enhanced.h5')
    np.save('label_encoder_classes_enhanced.npy', label_encoder.classes_)
    print("Enhanced model and label encoder saved.")

    # Optional: Plot training history
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Replace with the path to your full CSV file
    train_model(csv_path='/home/pc-25/Music/app/ckextended.csv')