import os 
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle


mp_drawing = mp.solutions.drawing_utils 
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.5,
    max_num_hands=1
)

DATA_DIR = "./imagenes_captura"

data=[]
labels=[]

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux=[]
        img = cv2.imread( os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_ = []
                y_ = []
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)
                    
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data':data, 'labels':labels}, f)