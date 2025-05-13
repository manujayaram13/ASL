import cv2
import mediapipe as mp
import numpy as np
import os
print('hello')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = "Dataset"
SEQUENCE_LENGTH = 30
features = []
labels = []
label_dict = {chr(i+65): i for i in range(28)}  # {'A': 0, ..., 'Z': 25}

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        data = []
        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
        return data
    return None

# Go through each folder and image
for label in os.listdir(DATA_PATH):
    label_folder = os.path.join(DATA_PATH, label)
    for image_file in os.listdir(label_folder):
        image_path = os.path.join(label_folder, image_file)
        image = cv2.imread(image_path)
        landmarks = extract_landmarks(image)
        if landmarks:
            features.append(landmarks)
            labels.append(label_dict[label.upper()])

hands.close()
