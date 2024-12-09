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
import requests
import json
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
import os


# Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

# Flask application
app = Flask(__name__)

# Load the trained model and label encoder
try:
    model = load_model('emotion_detection_model.h5')
    print('gfkgjfjgkfjj')
    print(model)
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

def get_spotify_access_token():
    try:
        auth_url = "https://accounts.spotify.com/api/token"
        response = requests.post(auth_url, {
            'grant_type': 'client_credentials',
            'client_id': SPOTIFY_CLIENT_ID,
            'client_secret': SPOTIFY_CLIENT_SECRET
        })
        response_data = response.json()
        return response_data.get("access_token")
    except Exception as e:
        print(f"Error fetching Spotify token: {e}")
        return None


def get_songs_for_emotion(emotion):
    try:
        # Emotion to genre mapping
        emotion_to_genre = {
            "happy": "Indie Pop",
            "sad": "acoustic",
            "anger": "rock",
            "angry": "rock",
            "fear":'horror',
            "relaxed": "chill",
            "excited": "party",
            "surprise": "electronic",
            "neutral": "classical",
            "disgust": "industrial"  # or grunge, dark ambient, etc.

        }

        genre = emotion_to_genre[emotion]
        # Get Spotify access token
        token = get_spotify_access_token()
        if not token:
            print("Spotify access token could not be retrieved.")
            return []
        # Spotify Search API URL
        search_url = f"https://api.spotify.com/v1/search?q={str(genre)}&type=playlist&limit=1"
        headers = {"Authorization": f"Bearer {token}"}

        # Fetch playlist
        response = requests.get(search_url, headers=headers)
        response_data = response.json()

        # Check if playlists exist in the response
        playlists = response_data.get("playlists", {}).get("items", [])

        if not playlists or len(playlists) == 0:
            print("No playlists found for the emotion:", emotion)
            return []

        # Use the first playlist to get tracks
        print('ddddd')
        playlist_id = playlists[0]["id"]
        tracks_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
        tracks_response = requests.get(tracks_url, headers=headers)

        tracks_data = tracks_response.json()
        # Extract song names and URLs
        songs = [
            {
                "name": track["track"]["name"],
                "url": track["track"]["external_urls"]["spotify"]
            }
            for track in tracks_data.get("items", [])
            if track.get("track") and track["track"].get("external_urls")
        ]
        return songs
    except Exception as e:
        print(f"Error fetching songs for emotion: {e}")
        return []

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle emotion detection from uploaded image
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_route():
    try:
        # Get the image data from the frontend
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

        # Convert the image to a NumPy array (DeepFace requires this format)
        frame = np.array(image)

        # Use DeepFace to analyze the emotion
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        print(analysis[0]['dominant_emotion'])
        if not analysis or 'emotion' not in analysis[0]:
            return jsonify({"error": "Error analyzing emotion from the image!"}), 500

        # Extract the dominant emotion
        dominant_emotion = analysis[0]['dominant_emotion']

        # Fetch song recommendations for the detected emotion
        songs = get_songs_for_emotion(dominant_emotion)  # Assuming you have this function
        if not songs:
            return jsonify({"emotion": dominant_emotion, "songs": [], "message": "No songs found for this emotion."})

        # Return the response with emotion and song recommendations
        return jsonify({"emotion": dominant_emotion, "songs": songs})
    except Exception as e:
        print(f"Error in emotion detection route: {e}")
        return jsonify({"error": "An error occurred while processing the image!"}), 500


# def detect_emotion_route():
#     try:
#         data = request.json
#         image_data = data.get('image')
#         if not image_data:
#             return jsonify({"error": "No image data provided!"}), 400

#         # Decode the base64 image data
#         image_data = image_data.split(',')[1]
#         image = Image.open(BytesIO(base64.b64decode(image_data)))

#         # Convert RGBA to RGB (if necessary)
#         if image.mode == 'RGBA':
#             image = image.convert('RGB')

#         frame = np.array(image)

#         # Save image temporarily to predict
#         temp_image_path = "temp_user_image.jpg"
#         image.save(temp_image_path, 'JPEG')  # Save the image as JPEG

#         # Analyze the emotion using the trained model
#         emotion = analyze_emotion_from_image(temp_image_path)

#         # Delete temporary image file
#         os.remove(temp_image_path)

#         if emotion is None:
#             return jsonify({"error": "Error analyzing emotion from the image!"}), 500

#         # Fetch song recommendations for the detected emotion
#         songs = get_songs_for_emotion(emotion)
#         if not songs:
#             return jsonify({"emotion": emotion, "songs": [], "message": "No songs found for this emotion."})
#         return jsonify({"emotion": emotion, "songs": songs})

#     except Exception as e:
#         print(f"Error in emotion detection route: {e}")
#         return jsonify({"error": "An error occurred while processing the image!"}), 500

if __name__ == "__main__":
    app.run(debug=True)
