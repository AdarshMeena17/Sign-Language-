import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load dataset
data = []
labels = []
signs = ['A', 'B', 'C', 'D', 'E']

for i, sign in enumerate(signs):
    folder = f'data/{sign}'
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        landmarks = np.loadtxt(path, delimiter=',')
        data.append(landmarks)
        labels.append(i)

data = np.array(data)
labels = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(len(X_train[0]),)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(signs), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

# Save model
model.save('sign_model.h5')

print("✅ Model saved as sign_model.h5")
