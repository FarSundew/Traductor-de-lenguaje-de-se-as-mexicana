import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import pyttsx3
import tensorflow as tf
import threading

# Cargar modelo entrenado
model = tf.keras.models.load_model('model.keras')

# Inicializar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Inicializar motor de texto a voz
engine = pyttsx3.init()
voices = engine.getProperty('voices')

# Buscar una voz en español
voz_encontrada = False
for voice in voices:
    if "es" in str(voice.languages).lower() or "spanish" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        voz_encontrada = True
        break

# Función para hablar en un plano secundario (Background Thread)
def pronunciar_palabra(texto, id_voz):
    motor = pyttsx3.init()
    if id_voz:
        motor.setProperty('voice', id_voz)
    motor.say(texto)
    motor.runAndWait()

# Diccionario de etiquetas
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
               18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' '}

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Variables de control
ultima_letra = ''
palabra = ''
tiempo_ultimo = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    # Crear panel blanco a la derecha
    espacio_derecho = 300
    panel = np.ones((H, espacio_derecho, 3), dtype=np.uint8) * 255

    # Procesar imagen
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []
    letra_predicha = ''

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

        if data_aux and len(data_aux) == 42:  # Solo una mano (21 puntos * 2 coordenadas)
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # TensorFlow devuelve una matriz de probabilidades, tomamos la de mayor valor
            prediction = model.predict(np.asarray([data_aux]), verbose=0)
            letra_predicha = labels_dict[int(np.argmax(prediction[0]))]

            # Mostrar letra
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, letra_predicha, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (255, 0, 0), 3, cv2.LINE_AA)

            # Temporizador: cada 2 segundos
            if time.time() - tiempo_ultimo > 2:
                if letra_predicha != ultima_letra:
                    palabra += letra_predicha
                    ultima_letra = letra_predicha
                    tiempo_ultimo = time.time()

    # Dibujar texto en el panel blanco
    cv2.putText(panel, 'Letra:', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(panel, letra_predicha, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    cv2.putText(panel, 'Palabra:', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(panel, palabra, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 100, 255), 3)

    # Unir panel blanco con cámara
    frame_con_panel = np.hstack((frame, panel))

    # Mostrar en ventana
    cv2.imshow("Lenguaje - Palabra", frame_con_panel)

    # Teclas
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 13:  # ENTER
        if palabra:
            # Lanzar el audio en un hilo separado para no congelar la cámara
            id_voz_actual = engine.getProperty('voice')
            hilo = threading.Thread(target=pronunciar_palabra, args=(palabra, id_voz_actual))
            hilo.start()
    elif key == ord('r'):  # Tecla R para reiniciar
        palabra = ''
        ultima_letra = ''
    elif key == ord('d'):  # Tecla D para borrar la última letra
        if palabra:
            palabra = palabra[:-1]
            ultima_letra = ''


cap.release()
cv2.destroyAllWindows()
