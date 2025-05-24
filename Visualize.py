import cv2
import numpy as np
from skimage.feature import blob_dog
from pathlib import Path
import matplotlib.pyplot as plt

# Rutas
dataset_path = Path("Fer2013_Dataset")

# Función para visualizar
def visualize_image(image_path, label, max_visualizations=1):
    # Leer imagen original en escala de grises
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 1. Extracción de características básicas (normalización)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. Detector de bordes (Canny)
    edges = cv2.Canny(img, 100, 200)

    # 3. Detector de blobs (Difference of Gaussian)
    blobs = blob_dog(img, max_sigma=30, threshold=.1)
    blobs = blobs if blobs.size else np.array([])

    # 4. Detector de esquinas (Shi-Tomasi)
    corners = cv2.goodFeaturesToTrack(img, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners = corners.squeeze() if corners is not None else np.array([])

    # Redimensionar a 48x48
    img_resized = cv2.resize(img, (48, 48))

    # Crear figura para visualización con 5 paneles
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Añadir título general con la emoción
    fig.suptitle(f"Preprocesamiento de {label.capitalize()}", fontsize=16, y=1.1)  # Ajuste de y para más espacio

    # Panel 1: Imagen original
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f"Original - {label.capitalize()}")
    axes[0].axis('off')

    # Panel 2: Imagen con blobs
    axes[1].imshow(img, cmap='gray')
    if blobs.size:
        for blob in blobs:
            y, x, r = blob
            circle = plt.Circle((x, y), r, color='red', fill=False)
            axes[1].add_patch(circle)
    axes[1].set_title(f"Blobs (Red) - {label.capitalize()}")
    axes[1].axis('off')

    # Panel 3: Imagen con esquinas
    axes[2].imshow(img, cmap='gray')
    if corners.size:
        if corners.ndim == 1:  # Caso de una sola esquina
            corners = corners.reshape(1, -1)
        for corner in corners:
            x, y = corner
            axes[2].plot(x, y, 'b.')
    axes[2].set_title(f"Corners (Blue) - {label.capitalize()}")
    axes[2].axis('off')

    # Panel 4: Bordes (Canny)
    axes[3].imshow(edges, cmap='gray')
    axes[3].set_title(f"Edges (Canny) - {label.capitalize()}")
    axes[3].axis('off')

    # Panel 5: Imagen redimensionada
    axes[4].imshow(img_resized, cmap='gray')
    axes[4].set_title(f"Resized (48x48) - {label.capitalize()}")
    axes[4].axis('off')

    # Ajustar diseño y mostrar ventana
    plt.tight_layout()
    plt.show()

# Visualizar imágenes (limitamos a 1 por categoría)
for label in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]:
    label_path = dataset_path / "test" / label
    count = 0
    for img_path in label_path.glob("*.jpg"):
        if count >= 1:  # Solo una imagen por categoría
            break
        print(f"Mostrando visualización para {label}/{img_path.name}")
        visualize_image(img_path, label)
        count += 1

print("Visualización completada.")