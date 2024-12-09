# # import pandas as pd
# # import numpy as np
# # import os
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
# # from tensorflow.keras.utils import to_categorical
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import LabelEncoder

# # # Step 1: Load the CSV file
# # df = pd.read_csv('/home/pc-25/Music/AI-emotions-detection-master/fer2013.csv')  # Update with actual path

# # # Step 2: Preprocess the data
# # images = []
# # labels = []

# # for index, row in df.iterrows():
# #     # Extract the pixel data and emotion label
# #     pixels = np.array(row['pixels'].split(), dtype='float32')  # Split the string into pixel values
# #     pixels = pixels.reshape(48, 48, 1)  # Reshape into a 48x48 grayscale image
# #     pixels = pixels / 255.0  # Normalize pixel values to range 0-1
    
# #     images.append(pixels)
# #     labels.append(row['emotion'])

# # # Convert lists to numpy arrays
# # X = np.array(images)
# # y = np.array(labels)

# # # Step 3: Encode the labels
# # label_encoder = LabelEncoder()
# # y = label_encoder.fit_transform(y)  # Convert labels to integers
# # y = to_categorical(y)  # One-hot encode labels for classification

# # # Step 4: Split dataset into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Step 5: Build the CNN model
# # model = Sequential()

# # # Convolutional layers
# # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.2))

# # model.add(Conv2D(64, (3, 3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.2))

# # model.add(Conv2D(128, (3, 3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.2))

# # # Flatten and dense layers
# # model.add(Flatten())
# # model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer with softmax for multi-class classification

# # # Compile the model
# # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # # Step 6: Train the model
# # history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# # # Step 7: Evaluate the model
# # test_loss, test_acc = model.evaluate(X_test, y_test)
# # print(f"Test accuracy: {test_acc}")

# # # Step 8: Save the model
# # model.save('emotion_detection_model.h5')

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Define the directory containing the dataset
image_dir = '/home/pc-25/Music/AI-emotions-detection-master/AffectNet/test'  # Update path

# Step 2: Load images and labels from directories
images = []
labels = []

for label_folder in os.listdir(image_dir):
    folder_path = os.path.join(image_dir, label_folder)
    if os.path.isdir(folder_path):  # Check if it's a directory
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            try:
                # Load and preprocess the image
                image = load_img(image_path, target_size=(48, 48), color_mode='grayscale')
                image = img_to_array(image) / 255.0  # Normalize pixel values to range 0-1
                images.append(image)
                labels.append(label_folder)  # Use the folder name as the label
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

# Convert lists to numpy arrays
X = np.array(images)
y = np.array(labels)

# Step 3: Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert labels to integers
y = to_categorical(y)  # One-hot encode labels for classification

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build the CNN model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
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
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Step 7: Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Step 8: Save the model
model.save('emotion_detection_model.h5')

