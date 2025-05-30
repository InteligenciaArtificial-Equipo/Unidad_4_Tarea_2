import cv2
import numpy as np
from keras.models import load_model
import os

def run_realtime_detector(model_path='models/emotion_detector_cnn.keras', emotion_labels=None):
    # Cargar el modelo entrenado
    try:
        model = load_model(model_path)
        print(f"Modelo '{model_path}' cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}.")
        return
    
    if emotion_labels is None:
        print("Usando etiquetas por defecto:")
        
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        print(emotion_labels)

    # Cargar el clasificador de cascada para detección de rostros de OpenCV
    face_cascade_path = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        print(f"Error: No se pudo cargar {face_cascade_path}.")
        return

    # Inicializar la webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return

    print("Iniciando detección de emociones en tiempo real. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la webcam.")
            break

        frame = cv2.flip(frame, 1) # Voltear el frame horizontalmente
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            resized_face = cv2.resize(roi_gray, (48, 48))
            processed_face = np.expand_dims(np.expand_dims(resized_face / 255.0, -1), 0)

            # 1. Obtener las probabilidades de prediccion
            predictions = model.predict(processed_face, verbose=0)[0] 
            
            # 2. Encontrar el indice de la emocion con la mayor probabilidad
            emotion_index = np.argmax(predictions)
            
            # 3. Mapear el indice a la etiqueta de texto usando emotion_labels
            emotion = emotion_labels[emotion_index]
            
            # 4. Obtener la confianza de esa prediccion
            confidence = np.max(predictions) * 100

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{emotion}: {confidence:.2f}%"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Detector de Emociones', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detector de emociones detenido.")

if __name__ == '__main__':
    known_emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    run_realtime_detector(emotion_labels=known_emotion_labels)
