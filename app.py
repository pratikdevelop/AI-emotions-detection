import cv2
from deepface import DeepFace

def initialize_webcam():
    cap = cv2.VideoCapture(0)
    return cap

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def analyze_emotion(face_crop):
    try:
        result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return None

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = initialize_webcam()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, face_cascade)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_crop = frame[y:y + h, x:x + w]
                dominant_emotion = analyze_emotion(face_crop)

                if dominant_emotion:
                    print(f"Detected emotion: {dominant_emotion}")
                    cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No faces detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Face Detection and Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()