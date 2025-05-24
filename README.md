## Unidad 4 - Tarea 1

## Preprocesamiento para un detector de emociones con fer2013 con el modelo CNN (Red Neuronal Convolucional)

## Integrantes

### Chaparro Castillo Christopher
### Peñuelas López Luis Antonio

## Como Funciona?

## Imports
Tenemos los siguientes **imports**, a la hora de ejecutar el programa:

- ### Cv2
  
```py
import cv2
```
Librería para procesar imágenes y videos. Se usa para leer imágenes (como cv2.imread()), detectar bordes (cv2.Canny()), redimensionarlas (cv2.resize()).

- ### Numpy

```py
import numpy as np
```
Librería para cálculos numéricos con arreglos. Sirve para manejar imágenes como matrices y realizar operaciones como normalizar o cambiar la forma de los datos.

- ### Blob_dog

```py
from skimage.feature import blob_dog
```
Parte de Scikit-Image, esta función detecta regiones circulares (blobs) en imágenes usando el método Difference of Gaussian. Se emplea para identificar características como ojos o boca en las caras.

- ### Path

```py
from pathlib import Path
```
Es un módulo de Python para trabajar con rutas de archivos y directorios de forma sencilla. Se usa para crear carpetas (como preprocessed_data) y buscar imágenes (Path.glob("*.jpg")).

- ### Random

```py
import random
```
Es una Librería para generar aleatoriedad. Sirve para mezclar las imágenes aleatoriamente antes de dividirlas en entrenamiento, prueba y validación, asegurando una distribución justa.

- ### Matplotlib

```py
import matplotlib.pyplot as pl
```
Es una Librería de visualización que permite generar gráficos de alta calidad, como líneas, barras, histogramas, imágenes, y más.

## Preprocess.py
Código para llevar a cabo el preprocesamiento de imágenes para nuestro modelo CNN.

- ### Ruta base del dataset
  
```py
dataset_path = Path("Fer2013_Dataset")
output_path = Path("preprocessed_data")

output_path.mkdir(exist_ok=True)
```
Implementamos el dataset Fer2013 para preprocesarlas para el entrenamiento de nuestra CNN.

- ### Función para el preprocesamiento
```py
def preprocess_image(image_path, label, split):

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    edges = cv2.Canny(img, 100, 200)

    blobs = blob_dog(img, max_sigma=30, threshold=.1)
    blobs = blobs[:, 2] if blobs.size else np.array([])  

    corners = cv2.goodFeaturesToTrack(img, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners = corners.squeeze() if corners is not None else np.array([])

    img_resized = cv2.resize(img, (48, 48))
    edges_resized = cv2.resize(edges, (48, 48))

    processed_img = cv2.merge([img_resized, img_resized, img_resized])  

    output_dir = output_path / split / label
    output_dir.mkdir(parents=True, exist_ok=True)  
    cv2.imwrite(str(output_dir / image_path.name), processed_img)
```
Función para el preprocesamiento de imágenes del dataset fer2013 (Extracción de características, Detector de bordes, Detector de blobs y Detector de esquinas), además de redimensionar y convertir a 3 canales para el modelo CNN.

- ### Recolectar todas las imágenes
```py
all_images = []
for split in ["train", "test"]:
    for label in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]:
        label_path = dataset_path / split / label
        for img_path in label_path.glob("*.jpg"):
            all_images.append((img_path, label))
```
Busca todas las imágenes en las carpetas train y test del dataset, que están organizadas por emociones (como "happy", "sad", etc.).

- ### Mezcla y División

```py
random.shuffle(all_images)

total_images = len(all_images)
train_end = int(0.7 * total_images)  
test_end = int(0.9 * total_images)   

train_images = all_images[:train_end]
test_images = all_images[train_end:test_end]
val_images = all_images[test_end:]
```

Mezcla aleatoriamente todas las imágenes para que no estén en un orden predecible y las divide en tres partes: 70% para entrenamiento (para enseñar a la red neuronal), 20% para prueba (para verificar cómo aprende), y 10% para validación (para ajustar y evaluar el modelo).

- ### Guardado y Finalización

```py

for img_path, label in train_images:
    preprocess_image(img_path, label, "train")
for img_path, label in test_images:
    preprocess_image(img_path, label, "test")
for img_path, label in val_images:
    preprocess_image(img_path, label, "val")

print("Preprocesamiento completado con división: 70% train, 20% test, 10% val.")
```

Aplica el procesamiento a cada imagen y las guarda en subcarpetas dentro de preprocessed_data (como train/happy, test/sad, etc.).

- ### Resultado
![Image](https://github.com/user-attachments/assets/0b954a51-eb0a-4ef0-9746-7aba34a5156d)

Nos genera la carpeta preprocessed_data con las imágenes preprocesadas del dataset.
