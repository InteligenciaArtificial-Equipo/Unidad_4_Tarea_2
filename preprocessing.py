import numpy as np
import pandas as pd
import cv2 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

def load_and_preprocess_fer2013(file_path='fer2013.csv'):
    try:
        data = pd.read_csv(file_path)
        print("Dataset FER-2013 cargado exitosamente.")
    except FileNotFoundError:
        print(f"Error: {file_path} no encontrado.")
        return None, None, None, None, None, None, None, None

    pixels = data['pixels'].tolist()
    emotions = data['emotion'].tolist()

    images = []
    for pixel_sequence in pixels:
        pixel_values = [int(p) for p in pixel_sequence.split(' ')]
        image = np.array(pixel_values).reshape(48, 48)
        images.append(image)

    images = np.array(images)
    # Añadir una dimension para el canal (escala de grises)
    images = np.expand_dims(images, axis=-1)

    # Normalizar los pixeles (importante para las CNN)
    images_normalized = images / 255.0

    # Codificar las etiquetas de emocion
    le = LabelEncoder()
    emotions_encoded = le.fit_transform(emotions)
    emotion_labels = list(le.classes_) # Guardar las etiquetas originales
    num_classes = len(emotion_labels)
    emotions_categorical = to_categorical(emotions_encoded, num_classes=num_classes)

    print(f"Numero de clases de emocion: {num_classes}")
    print(f"Forma de las imagenes: {images_normalized.shape}")
    print(f"Forma de las etiquetas categoricas: {emotions_categorical.shape}")
    print(f"Etiquetas de emocion: {emotion_labels}")

    # División del dataset (70% entrenamiento, 20% prueba, 10% validación)
    X_train, X_temp, y_train, y_temp = train_test_split(
        images_normalized, emotions_categorical, test_size=0.3, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42
    )

    print(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Forma de X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Forma de X_val: {X_val.shape}, y_val: {y_val.shape}")

    return X_train, X_test, X_val, y_train, y_test, y_val, num_classes, emotion_labels

def visualize_feature_extraction_example(image):
    img_8bit = cv2.convertScaleAbs(image)

    # 1. Detector de Bordes (Canny)
    edges = cv2.Canny(img_8bit, 50, 150)

    # 2. Detector de Blobs (SimpleBlobDetector)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints_blobs = detector.detect(img_8bit)
    blobs_image = np.zeros_like(img_8bit)
    for kp in keypoints_blobs:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(blobs_image, (x, y), int(kp.size / 2), 255, -1)

    # 3. Detector de Esquinas (Harris Corner Detector)
    corners_harris = cv2.cornerHarris(np.float32(img_8bit), 2, 3, 0.04)
    corners_harris = cv2.dilate(corners_harris, None)
    corners_image = np.zeros_like(img_8bit)
    corners_image[corners_harris > 0.01 * corners_harris.max()] = 255

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(img_8bit, cmap='gray')
    plt.title('Original')
    plt.subplot(1, 4, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Bordes (Canny)')
    plt.subplot(1, 4, 3)
    plt.imshow(blobs_image, cmap='gray')
    plt.title('Blobs')
    plt.subplot(1, 4, 4)
    plt.imshow(corners_image, cmap='gray')
    plt.title('Esquinas (Harris)')
    plt.suptitle('Ejemplo de Extraccion de Características')
    plt.show()

if __name__ == '__main__':
    # Ejecutar la carga y división del dataset
    X_train, X_test, X_val, y_train, y_test, y_val, num_classes, emotion_labels = \
        load_and_preprocess_fer2013()

    if X_train is not None:
        print("\nMostrando ejemplo de extracción de características.")
        sample_image_for_viz = X_train[0].squeeze() * 255 
        visualize_feature_extraction_example(sample_image_for_viz)