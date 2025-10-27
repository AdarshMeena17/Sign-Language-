import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load MediaPipe and model
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
model = load_model('sign_model.h5')  # Your trained model file

# Labels for signs
labels = ['A', 'B ', 'C', 'D', 'E']

# Start webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert image
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                data = []
                for lm in hand_landmarks.landmark:
                    data.append(lm.x)
                    data.append(lm.y)
                
                prediction = model.predict(np.array([data]))
                sign = labels[np.argmax(prediction)]
                cv2.putText(frame, f'Sign: {sign}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow('Sign Language Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()

