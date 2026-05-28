from flask import Flask, render_template, Response, jsonify, request
import os
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import tensorflow as tf
import threading

# --- Configuración estricta de rutas ---
directorio_base = os.path.abspath(os.path.dirname(__file__))
directorio_templates = os.path.join(directorio_base, 'templates')
app = Flask(__name__, template_folder=directorio_templates)

# --- Configuración ---
ruta_modelo = os.path.join(directorio_base, 'model.keras')
model = tf.keras.models.load_model(ruta_modelo)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

engine = pyttsx3.init()
id_voz_actual = None
for voice in engine.getProperty('voices'):
    if "es" in str(voice.languages).lower() or "spanish" in voice.name.lower():
        id_voz_actual = voice.id
        break

labels_dict = {i: chr(65 + i) for i in range(26)}
labels_dict[26] = ' ' # Espacio

# --- Variables Globales de Estado ---
palabra = ''
ultima_letra = ''
letra_predicha = ''
tiempo_ultimo = time.time()
cap = cv2.VideoCapture(0)

def pronunciar_palabra(texto, id_voz):
    motor = pyttsx3.init()
    if id_voz:
        motor.setProperty('voice', id_voz)
    motor.say(texto)
    motor.runAndWait()

def gen_frames():
    global palabra, ultima_letra, letra_predicha, tiempo_ultimo
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        
        data_aux = []
        x_ = []
        y_ = []
        letra_actual = ''

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            if data_aux and len(data_aux) == 42:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                prediction = model.predict(np.asarray([data_aux]), verbose=0)
                letra_actual = labels_dict[int(np.argmax(prediction[0]))]
                letra_predicha = letra_actual

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, letra_actual, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)

                if time.time() - tiempo_ultimo > 2:
                    if letra_actual != ultima_letra:
                        palabra += letra_actual
                        ultima_letra = letra_actual
                        tiempo_ultimo = time.time()
        else:
            letra_predicha = ''

        # Codificar a JPEG para enviar a la web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_state')
def get_state():
    return jsonify({'palabra': palabra, 'letra': letra_predicha})

@app.route('/action', methods=['POST'])
def action():
    global palabra, ultima_letra
    action_type = request.json.get('type')
    if action_type == 'speak' and palabra:
        threading.Thread(target=pronunciar_palabra, args=(palabra, id_voz_actual)).start()
    elif action_type == 'reset':
        palabra, ultima_letra = '', ''
    elif action_type == 'delete' and palabra:
        palabra, ultima_letra = palabra[:-1], ''
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)