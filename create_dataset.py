import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'


data_42 = []
labels_42 = []
data_84 = []
labels_84 = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            if len(data_aux) == 42:
                data_42.append(data_aux)
                labels_42.append(dir_)
            elif len(data_aux) == 84:
                data_84.append(data_aux)
                labels_84.append(dir_)
            else:
                continue
                      

f_42 = open('data_42.pickle', 'wb')
f_84 = open('data_84.pickle', 'wb')
pickle.dump({'data': data_42, 'labels': labels_42}, f_42)
pickle.dump({'data': data_84, 'labels': labels_84}, f_84)
f_42.close()
f_84.close()



