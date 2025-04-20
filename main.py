# import os
# import numpy as np
# from flask import Flask, request, jsonify
# from flask_socketio import SocketIO, emit
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
# from deepface import DeepFace
# from io import BytesIO
# from PIL import Image
# import base64
# import requests
# from dotenv import load_dotenv
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from sklearn.preprocessing import LabelEncoder
# from collections import defaultdict
# import time
# import random
# from datetime import datetime
# from werkzeug.security import generate_password_hash, check_password_hash
# from flask_cors import CORS 


# # Load environment variables
# load_dotenv()

# # Environment variables
# SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
# SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
# UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY')

# # Disable GPU to reduce resource usage
# tf.config.set_visible_devices([], 'GPU')

# # Flask app setup
# app = Flask(__name__)
# app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key')
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///moodcraft.db'
# db = SQLAlchemy(app)
# socketio = SocketIO(app)
# login_manager = LoginManager(app)
# login_manager.login_view = 'login'
# CORS(app)

# # Load enhanced custom-trained model and label encoder
# try:
#     custom_model = load_model('emotion_detection_model.h5')
#     print("Enhanced custom model loaded successfully")
# except Exception as e:
#     print(f"Error loading enhanced custom model: {e}")
#     custom_model = None

# try:
#     label_encoder = LabelEncoder()
#     label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)
#     print("Enhanced label encoder loaded successfully")
# except Exception as e:
#     print(f"Error loading enhanced label encoder: {e}")
#     label_encoder = None

# # Database models
# class User(db.Model, UserMixin):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(150), unique=True, nullable=False)
#     email = db.Column(db.String(150), unique=True, nullable=False)
#     password = db.Column(db.String(150), nullable=False)

# class MoodLog(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     emotion = db.Column(db.String(50), nullable=False)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# class Badge(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100), nullable=False)
#     description = db.Column(db.String(255), nullable=False)

# class UserBadge(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     badge_id = db.Column(db.Integer, db.ForeignKey('badge.id'), nullable=False)
#     awarded_at = db.Column(db.DateTime, default=datetime.utcnow)

# # User loader for Flask-Login
# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))

# # Content cache setup
# content_cache = defaultdict(dict)
# CACHE_TIMEOUT = 300  # 5 minutes

# # Function to analyze emotion using enhanced custom model
# def analyze_emotion_custom(image):
#     try:
#         if custom_model is None or label_encoder is None:
#             return None
#         # Preprocess image for custom model (64x64 input size)
#         image = image.resize((64, 64))
#         image_array = img_to_array(image) / 255.0
#         image_array = np.expand_dims(image_array, axis=0)
        
#         # Predict emotion
#         prediction = custom_model.predict(image_array)
#         predicted_class = np.argmax(prediction, axis=1)
#         predicted_label = label_encoder.inverse_transform(predicted_class)
#         return predicted_label[0]
#     except Exception as e:
#         print(f"Error with enhanced custom model prediction: {e}")
#         return None

# # Function to analyze emotion with hybrid approach
# def analyze_emotion_hybrid(image):
#     custom_result = analyze_emotion_custom(image)
#     frame = np.array(image)
#     deepface_result = None
#     try:
#         analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
#         if analysis and 'emotion' in analysis[0]:
#             deepface_result = analysis[0]['dominant_emotion']
#     except Exception as e:
#         print(f"Error with DeepFace prediction: {e}")

#     # Hybrid logic: Use custom model if available, fall back to DeepFace
#     if custom_result:
#         return custom_result
#     elif deepface_result:
#         return deepface_result
#     else:
#         return None

# # Spotify API functions
# def get_spotify_access_token():
#     try:
#         auth_url = "https://accounts.spotify.com/api/token"
#         response = requests.post(auth_url, {
#             'grant_type': 'client_credentials',
#             'client_id': SPOTIFY_CLIENT_ID,
#             'client_secret': SPOTIFY_CLIENT_SECRET
#         })
#         return response.json().get("access_token")
#     except Exception as e:
#         print(f"Error fetching Spotify token: {e}")
#         return None

# def get_songs_for_emotion(emotion):
#     try:
#         # Updated to match FER-2013 emotions
#         emotion_to_genre = {
#             "angry": "rock", "disgust": "industrial", "fear": "horror",
#             "happy": "Indie Pop", "sad": "acoustic", "surprise": "electronic",
#             "neutral": "classical"
#         }
#         genre = emotion_to_genre.get(emotion, "pop")
#         token = get_spotify_access_token()
#         if not token:
#             return []
#         search_url = f"https://api.spotify.com/v1/search?q={genre}&type=playlist&limit=1"
#         headers = {"Authorization": f"Bearer {token}"}
#         response = requests.get(search_url, headers=headers)
#         playlists = response.json().get("playlists", {}).get("items", [])
#         if not playlists:
#             return []
#         playlist_id = playlists[0]["id"]
#         tracks_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
#         tracks_response = requests.get(tracks_url, headers=headers)
#         tracks_data = tracks_response.json()
#         songs = [
#             {"name": track["track"]["name"], "url": track["track"]["external_urls"]["spotify"]}
#             for track in tracks_data.get("items", [])
#             if track.get("track") and track["track"].get("external_urls")
#         ]
#         return songs
#     except Exception as e:
#         print(f"Error fetching songs: {e}")
#         return []

# # Unsplash API function
# def get_visuals_for_emotion(emotion, count=5):
#     try:
#         url = f"https://api.unsplash.com/search/photos?query={emotion}&per_page={count}&client_id={UNSPLASH_ACCESS_KEY}"
#         response = requests.get(url)
#         data = response.json()
#         images = [photo['urls']['small'] for photo in data.get('results', [])]
#         return images
#     except Exception as e:
#         print(f"Error fetching visuals: {e}")
#         return []

# # Motivational texts (updated for FER-2013 emotions)
# emotion_texts = {
#     "angry": ["Take a deep breath.", "Let it out slowly."],
#     "disgust": ["Shake it off!", "Focus on the good."],
#     "fear": ["You’re stronger than you think.", "Face it one step at a time."],
#     "happy": ["Keep smiling!", "You're doing great!"],
#     "sad": ["It's okay to feel down.", "Tomorrow is a new day."],
#     "surprise": ["Wow, what a twist!", "Embrace the unexpected!"],
#     "neutral": ["Steady as you go.", "Balance is key."]
# }

# def get_text_for_emotion(emotion):
#     texts = emotion_texts.get(emotion, ["Stay positive!"])
#     return random.choice(texts)

# # Fetch content with caching
# def get_content_for_emotion(emotion):
#     current_time = time.time()
#     if emotion in content_cache and current_time - content_cache[emotion]['timestamp'] < CACHE_TIMEOUT:
#         return content_cache[emotion]['content']
#     songs = get_songs_for_emotion(emotion)
#     visuals = get_visuals_for_emotion(emotion)
#     text = get_text_for_emotion(emotion)
#     content = {'songs': songs, 'visuals': visuals, 'text': text}
#     content_cache[emotion] = {'content': content, 'timestamp': current_time}
#     return content

# # API Routes
# @app.route('/api/signup', methods=['POST'])
# def register():
#     data = request.json
#     name = data.get('name')
#     password = data.get('password')
#     email = data.get('email')
#     if not name or not password or not email:
#         return jsonify({"error": "name and password are required"}), 400
#     if User.query.filter_by(name=name).first():
#         return jsonify({"error": "name already exists"}), 400
#     hashed_password = generate_password_hash(password, method='sha256')
#     new_user = User(name=name, password=hashed_password, email=email)
#     db.session.add(new_user)
#     db.session.commit()
#     return jsonify({"message": "Registration successful"}), 201

# @app.route('/api/login', methods=['POST'])
# def login():
#     data = request.json
#     name = data.get('name')
#     password = data.get('password')
#     user = User.query.filter_by(name=name).first()
#     if user and check_password_hash(user.password, password):
#         login_user(user)
#         return jsonify({"message": "Login successful"}), 200
#     return jsonify({"error": "Invalid name or password"}), 401

# @app.route('/api/logout', methods=['POST'])
# @login_required
# def logout():
#     logout_user()
#     return jsonify({"message": "Logout successful"}), 200

# @app.route('/api/detect_emotion', methods=['POST'])
# def detect_emotion_route():
#     try:
#         data = request.json
#         image_data = data.get('image')
#         if not image_data:
#             return jsonify({"error": "No image data provided"}), 400
#         image_data = image_data.split(',')[1]
#         image = Image.open(BytesIO(base64.b64decode(image_data)))
#         if image.mode == 'RGBA':
#             image = image.convert('RGB')
        
#         # Use hybrid emotion detection with enhanced model
#         dominant_emotion = analyze_emotion_hybrid(image)
#         if not dominant_emotion:
#             return jsonify({"error": "Error analyzing emotion"}), 500
        
#         if current_user.is_authenticated:
#             mood_log = MoodLog(user_id=current_user.id, emotion=dominant_emotion)
#             db.session.add(mood_log)
#             db.session.commit()
#         content = get_content_for_emotion(dominant_emotion)
#         return jsonify({"emotion": dominant_emotion, **content})
#     except Exception as e:
#         print(f"Error in emotion detection: {e}")
#         return jsonify({"error": "Processing error"}), 500

# # WebSocket Event Handlers
# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')

# @socketio.on('analyze_frame')
# def handle_analyze_frame(data):
#     try:
#         image_data = data.get('image')
#         if not image_data:
#             emit('error', {'message': 'No image data provided'})
#             return
#         image_data = image_data.split(',')[1]
#         image = Image.open(BytesIO(base64.b64decode(image_data)))
#         if image.mode == 'RGBA':
#             image = image.convert('RGB')
        
#         # Use hybrid emotion detection with enhanced model
#         dominant_emotion = analyze_emotion_hybrid(image)
#         if not dominant_emotion:
#             emit('error', {'message': 'Error analyzing emotion'})
#             return
        
#         if current_user.is_authenticated:
#             mood_log = MoodLog(user_id=current_user.id, emotion=dominant_emotion)
#             db.session.add(mood_log)
#             db.session.commit()
#             mood_count = MoodLog.query.filter_by(user_id=current_user.id).count()
#             if mood_count >= 10:
#                 badge = Badge.query.filter_by(name='Mood Explorer').first()
#                 if badge and not UserBadge.query.filter_by(user_id=current_user.id, badge_id=badge.id).first():
#                     user_badge = UserBadge(user_id=current_user.id, badge_id=badge.id)
#                     db.session.add(user_badge)
#                     db.session.commit()
#                     emit('badge_awarded', {'badge': badge.name})
        
#         content = get_content_for_emotion(dominant_emotion)
#         emit('emotion_detected', {'emotion': dominant_emotion, **content})
#     except Exception as e:
#         print(f"Error analyzing frame: {e}")
#         emit('error', {'message': 'Processing error'})

# # Database Initialization
# with app.app_context():
#     db.create_all()
#     if not Badge.query.first():
#         badges = [
#             Badge(name='Mood Explorer', description='Logged 10 moods'),
#             Badge(name='Happy Streak', description='5 consecutive happy moods')
#         ]
#         db.session.add_all(badges)
#         db.session.commit()

# # Run the application
# if __name__ == "__main__":
#     socketio.run(app, debug=True)