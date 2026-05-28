import cv2
import os
import time

# Carpeta raíz donde se crearán las subcarpetas
carpeta_raiz = 'imagenes_captura'

# Crear la carpeta raíz si no existe 
if not os.path.exists(carpeta_raiz):
    os.makedirs(carpeta_raiz)

# Función para obtener el siguiente número de carpeta disponible (solo números)
def obtener_numero_carpeta(base_path):
    numeros = []
    for nombre in os.listdir(base_path):
        ruta = os.path.join(base_path, nombre)
        if os.path.isdir(ruta):
            try:
                num = int(nombre)  # Intentamos convertir el nombre directamente a entero
                numeros.append(num)
            except ValueError:
                pass
    return max(numeros) + 1 if numeros else 0  # Empieza en 0 si no hay carpetas

# Obtener número para la nueva carpeta
num_carpeta = obtener_numero_carpeta(carpeta_raiz)
carpeta_actual = os.path.join(carpeta_raiz, str(num_carpeta))
os.makedirs(carpeta_actual)

cap = cv2.VideoCapture(0)
# URL_CELULAR = 'http://192.168.1.XX:8080/video' # Descomenta y pon tu IP si usas IP Webcam en tu celular
# cap = cv2.VideoCapture(URL_CELULAR)

cap = cv2.VideoCapture(0) # Cambia el 0 por URL_CELULAR para usar tu teléfono

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

intervalo_segundos = 0.1
ultimo_guardado = 0
contador = 0
grabando = False

print("Presiona 'q' para comenzar la captura de imágenes.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Webcam', frame)
    key = cv2.waitKey(1) & 0xFF

    if not grabando:
        # Esperar a que el usuario presione 'q' para iniciar la captura
        if key == ord('q'):
            grabando = True
            ultimo_guardado = time.time()
            print(f"Iniciando captura de imágenes en {carpeta_actual}...")
    else:
        ahora = time.time()
        if ahora - ultimo_guardado >= intervalo_segundos:
            nombre_imagen = os.path.join(carpeta_actual, f'frame_{contador+1}.jpg')
            cv2.imwrite(nombre_imagen, frame)
            contador += 1
            print(f'Guardado {nombre_imagen} ({contador}/100)')
            ultimo_guardado = ahora

        # Parar automáticamente después de 100 imágenes
        if contador >= 100:
            print("Se alcanzaron 100 imágenes. Finalizando captura.")
            break

    # Permitir salir en cualquier momento con la tecla 'ESC'
    if key == 27:  # tecla ESC
        print("Captura interrumpida por el usuario.")
        break

cap.release()
cv2.destroyAllWindows()
