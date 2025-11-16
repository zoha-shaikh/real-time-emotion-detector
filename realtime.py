import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from collections import deque

# Load model
model = load_model("emotion_model.h5")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Webcam
cap = cv2.VideoCapture(0)

# Queue for smoothing predictions
prediction_queue = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict emotion
        prediction = model.predict(roi_gray, verbose=0)
        predicted_label = le.inverse_transform([np.argmax(prediction)])[0]

        # Smooth prediction
        prediction_queue.append(predicted_label)
        smooth_label = max(set(prediction_queue), key=prediction_queue.count)

        # Draw yellow rectangle that moves with face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)  # Yellow box (BGR: 0,255,255)
        cv2.putText(frame, smooth_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
