import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Choose sign to collect data for
SIGN = input("Enter the sign letter (A / B / C / D / E): ").upper()
DATA_DIR = f'data/{SIGN}'
os.makedirs(DATA_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    sample_count = 0
    print(f"Collecting data for sign '{SIGN}' ... Press 'q' to stop")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

                # Save landmarks to CSV
                with open(f'{DATA_DIR}/{sample_count}.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)

                sample_count += 1

        cv2.putText(frame, f'Samples: {sample_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Collect Data', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Collected {sample_count} samples for '{SIGN}'")

