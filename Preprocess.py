import cv2
import numpy as np
from skimage.feature import blob_dog
from pathlib import Path
import random

# Ruta base del dataset
dataset_path = Path("Fer2013_Dataset")
output_path = Path("preprocessed_data")

# Crear directorio de salida si no existe
output_path.mkdir(exist_ok=True)

# Funciones de preprocesamiento
def preprocess_image(image_path, label, split):
    # Leer imagen en escala de grises
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 1. Extracción de características básicas (normalización)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. Detector de bordes (Canny)
    edges = cv2.Canny(img, 100, 200)

    # 3. Detector de blobs (Difference of Gaussian)
    blobs = blob_dog(img, max_sigma=30, threshold=.1)
    blobs = blobs[:, 2] if blobs.size else np.array([])  # Manejo de caso vacío

    # 4. Detector de esquinas (Shi-Tomasi)
    corners = cv2.goodFeaturesToTrack(img, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners = corners.squeeze() if corners is not None else np.array([])

    # Redimensionar a 48x48
    img_resized = cv2.resize(img, (48, 48))
    edges_resized = cv2.resize(edges, (48, 48))

    # Convertir a 3 canales para compatibilidad con OpenCV (replicar canales)
    processed_img = cv2.merge([img_resized, img_resized, img_resized])  # RGB from grayscale

    # Guardar la imagen procesada
    output_dir = output_path / split / label
    output_dir.mkdir(parents=True, exist_ok=True)  # Añadir parents=True para crear directorios padre
    cv2.imwrite(str(output_dir / image_path.name), processed_img)

# Recolectar todas las imágenes
all_images = []
for split in ["train", "test"]:
    for label in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]:
        label_path = dataset_path / split / label
        for img_path in label_path.glob("*.jpg"):
            all_images.append((img_path, label))

# Mezclar aleatoriamente las imágenes
random.shuffle(all_images)

# Calcular índices para la división
total_images = len(all_images)
train_end = int(0.7 * total_images)  # 70% para entrenamiento
test_end = int(0.9 * total_images)   # 20% para prueba (70% + 20% = 90%), 10% para validación

# Dividir las imágenes
train_images = all_images[:train_end]
test_images = all_images[train_end:test_end]
val_images = all_images[test_end:]

# Procesar y guardar según la división
for img_path, label in train_images:
    preprocess_image(img_path, label, "train")
for img_path, label in test_images:
    preprocess_image(img_path, label, "test")
for img_path, label in val_images:
    preprocess_image(img_path, label, "val")

print("Preprocesamiento completado con división: 70% train, 20% test, 10% val.")