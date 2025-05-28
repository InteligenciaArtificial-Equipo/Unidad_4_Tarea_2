import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os
import sys
from collections import deque

def check_requirements():
    """Verifica que todos los requisitos estén cumplidos"""
    # Verificar que el modelo existe
    model_path = 'v1_emotion_detector_resnet50.h5'
    if not os.path.exists(model_path):
        print(f"\nError: El modelo '{model_path}' no existe.")
        print("\nPor favor, asegúrate de que el archivo del modelo esté en el directorio correcto.")
        sys.exit(1)

    # Verificar que OpenCV puede acceder a la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\nError: No se puede acceder a la cámara web.")
        print("Por favor, verifica que tu cámara esté conectada y funcionando.")
        sys.exit(1)
    cap.release()

# Cargar múltiples clasificadores de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
alt2_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Etiquetas de emociones
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Colores para cada emoción (en BGR)
emotion_colors = {
    'angry': (0, 0, 255),      # Rojo
    'disgust': (0, 128, 0),    # Verde oscuro
    'fear': (128, 0, 128),     # Púrpura
    'happy': (0, 255, 255),    # Amarillo
    'neutral': (255, 255, 255),# Blanco
    'sad': (255, 0, 0),        # Azul
    'surprise': (0, 255, 0)    # Verde
}

def preprocess_face(face_img):
    try:
        # Convertir a escala de grises
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar CLAHE para mejorar el contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Aplicar suavizado para reducir ruido
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Redimensionar a 48x48 con interpolación de área
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Normalizar
        normalized = resized / 255.0
        
        # Convertir a 3 canales
        rgb = cv2.merge([normalized, normalized, normalized])
        
        # Añadir dimensión del batch
        return np.expand_dims(rgb, axis=0)
    except Exception as e:
        print(f"Error en preprocesamiento: {e}")
        return None

def detect_faces(gray, frame):
    """Detecta rostros usando múltiples clasificadores"""
    faces = []
    
    # Detectar rostros frontales
    frontal_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    faces.extend(frontal_faces)
    
    # Detectar rostros alternativos
    alt_faces = alt_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    faces.extend(alt_faces)
    
    # Detectar rostros alternativos 2
    alt2_faces = alt2_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    faces.extend(alt2_faces)
    
    # Detectar rostros de perfil
    profile_faces = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    faces.extend(profile_faces)
    
    # Detectar rostros de perfil en el espejo
    mirrored_gray = cv2.flip(gray, 1)
    mirrored_profile_faces = profile_cascade.detectMultiScale(
        mirrored_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Ajustar coordenadas de rostros espejados
    for (x, y, w, h) in mirrored_profile_faces:
        x = frame.shape[1] - x - w
        faces.append((x, y, w, h))
    
    # Eliminar detecciones duplicadas
    return remove_overlapping_faces(faces)

def remove_overlapping_faces(faces, overlap_thresh=0.3):
    """Elimina detecciones de rostros que se superponen"""
    if len(faces) == 0:
        return []
    
    # Convertir a array numpy
    faces = np.array(faces)
    
    # Calcular áreas
    areas = faces[:, 2] * faces[:, 3]
    
    # Ordenar por área (de mayor a menor)
    idxs = np.argsort(areas)[::-1]
    
    keep = []
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        
        # Calcular IoU con el resto de detecciones
        xx1 = np.maximum(faces[current][0], faces[idxs[1:]][:, 0])
        yy1 = np.maximum(faces[current][1], faces[idxs[1:]][:, 1])
        xx2 = np.minimum(faces[current][0] + faces[current][2], 
                        faces[idxs[1:]][:, 0] + faces[idxs[1:]][:, 2])
        yy2 = np.minimum(faces[current][1] + faces[current][3], 
                        faces[idxs[1:]][:, 1] + faces[idxs[1:]][:, 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / areas[idxs[1:]]
        
        # Eliminar detecciones con superposición
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
    
    return faces[keep].tolist()

def get_dominant_emotion(emotion_buffer):
    """Obtiene la emoción dominante del buffer"""
    if not emotion_buffer:
        return None, 0.0
    
    # Contar ocurrencias de cada emoción
    emotion_counts = {}
    for emotion, conf in emotion_buffer:
        if emotion not in emotion_counts:
            emotion_counts[emotion] = []
        emotion_counts[emotion].append(conf)
    
    # Encontrar la emoción con más ocurrencias
    max_count = 0
    dominant_emotion = None
    for emotion, confs in emotion_counts.items():
        if len(confs) > max_count:
            max_count = len(confs)
            dominant_emotion = emotion
    
    # Calcular la confianza promedio para la emoción dominante
    avg_confidence = sum(emotion_counts[dominant_emotion]) / len(emotion_counts[dominant_emotion])
    
    return dominant_emotion, avg_confidence

def main():
    # Verificar requisitos
    check_requirements()
    
    print("\nCargando el modelo de detección de emociones...")
    model = load_model('v1_emotion_detector_resnet50.h5')
    print("Modelo cargado exitosamente!")
    
    print("\nIniciando la cámara...")
    # Iniciar la cámara
    cap = cv2.VideoCapture(0)
    
    # Configurar la ventana
    cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Emotion Detection', 800, 600)
    
    print("\nPresiona 'q' para salir")
    print("Detectando emociones en tiempo real...")
    
    # Variables para FPS
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Buffer para suavizar predicciones
    emotion_buffer = deque(maxlen=5)  # Mantener las últimas 5 predicciones
    
    # Umbral de confianza mínimo
    CONFIDENCE_THRESHOLD = 0.2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nError: No se puede leer de la cámara")
            break
            
        # Convertir a escala de grises para la detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros usando múltiples clasificadores
        faces = detect_faces(gray, frame)
        
        # Procesar cada rostro detectado
        for (x, y, w, h) in faces:
            # Extraer el rostro con un margen
            margin = int(min(w, h) * 0.2)  # 20% de margen
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            face_roi = frame[y1:y2, x1:x2]
            
            # Preprocesar el rostro para el modelo
            processed_face = preprocess_face(face_roi)
            if processed_face is None:
                continue
            
            # Predecir la emoción
            predictions = model.predict(processed_face, verbose=0)
            
            emotion_idx = np.argmax(predictions[0])
            emotion = emotions[emotion_idx]
            confidence = predictions[0][emotion_idx]
            
            # Añadir predicción al buffer
            emotion_buffer.append((emotion, confidence))
            
            # Obtener la emoción dominante
            dominant_emotion, avg_confidence = get_dominant_emotion(emotion_buffer)
            
            # Solo mostrar predicciones con confianza suficiente
            if avg_confidence >= CONFIDENCE_THRESHOLD:
                # Dibujar el rectángulo alrededor del rostro
                color = emotion_colors[dominant_emotion]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Mostrar la emoción y la confianza
                text = f"{dominant_emotion}: {avg_confidence:.2f}"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Mostrar todas las probabilidades en la parte superior
                y_offset = 60
                for i, (emotion_name, prob) in enumerate(zip(emotions, predictions[0])):
                    text = f"{emotion_name}: {prob:.2f}"
                    cv2.putText(frame, text, (10, y_offset + i*20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              emotion_colors[emotion_name], 1)
        
        # Calcular y mostrar FPS
        frame_count += 1
        if frame_count >= 30:  # Actualizar FPS cada 30 frames
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mostrar el frame
        cv2.imshow('Emotion Detection', frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("\nPrograma finalizado.")

if __name__ == "__main__":
    main() 