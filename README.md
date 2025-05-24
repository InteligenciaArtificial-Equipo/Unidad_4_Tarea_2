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

## Visualize.py
Código para visualizar las imágenes preprocesadas del dataset (Con motivo de verificación).

- ### Definición de Rutas
  
```py
dataset_path = Path("preprocessed_data")
```
Indica dónde están las imágenes, en la carpeta llamada preprocessed_data, específicamente en la sección test, que contiene ejemplos para verificar el modelo.

- ### Función para Visualizar
  
```py
def visualize_image(image_path, label, max_visualizations=1):

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    edges = cv2.Canny(img, 100, 200)

    blobs = blob_dog(img, max_sigma=30, threshold=.1)
    blobs = blobs if blobs.size else np.array([])

    corners = cv2.goodFeaturesToTrack(img, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners = corners.squeeze() if corners is not None else np.array([])

    img_resized = cv2.resize(img, (48, 48))
```
Crea una función que toma cada imagen, la ajusta (como cambiar su tamaño o mejorar su claridad), y detecta detalles como bordes, áreas redondeadas (como ojos), y puntos importantes (como esquinas).

- ### Creación y Configuración de la Visualización

```py
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f"Preprocesamiento de {label.capitalize()}", fontsize=16, y=1.1) 
```
Prepara una ventana en la pantalla con cinco secciones (paneles) para mostrar diferentes versiones de la imagen procesada, y pone un título en la parte superior que indica la emoción de la cara.

- ### Mostrar los Resultados

```py
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f"Original - {label.capitalize()}")
    axes[0].axis('off')

    axes[1].imshow(img, cmap='gray')
    if blobs.size:
        for blob in blobs:
            y, x, r = blob
            circle = plt.Circle((x, y), r, color='red', fill=False)
            axes[1].add_patch(circle)
    axes[1].set_title(f"Blobs (Red) - {label.capitalize()}")
    axes[1].axis('off')

    axes[2].imshow(img, cmap='gray')
    if corners.size:
        if corners.ndim == 1: 
            corners = corners.reshape(1, -1)
        for corner in corners:
            x, y = corner
            axes[2].plot(x, y, 'b.')
    axes[2].set_title(f"Corners (Blue) - {label.capitalize()}")
    axes[2].axis('off')

    axes[3].imshow(edges, cmap='gray')
    axes[3].set_title(f"Edges (Canny) - {label.capitalize()}")
    axes[3].axis('off')

    axes[4].imshow(img_resized, cmap='gray')
    axes[4].set_title(f"Resized (48x48) - {label.capitalize()}")
    axes[4].axis('off')
```
Muestra la imagen original, los bordes detectados, las áreas redondeadas, los puntos clave, y la imagen ajustada a un tamaño fijo, todo en los cinco paneles para que puedas compararlos.

- ### Revisión de Imágenes

```py
    plt.tight_layout()
    plt.show()
```
Busca una imagen de cada emoción (como "happy", "sad", etc.) en la carpeta test, procesa solo una por categoría, y la muestra en la ventana con un mensaje que indica qué imagen se está viendo.

- ### Finalización

```py
for label in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]:
    label_path = dataset_path / "test" / label
    count = 0
    for img_path in label_path.glob("*.jpg"):
        if count >= 1:  
            break
        print(f"Mostrando visualización para {label}/{img_path.name}")
        visualize_image(img_path, label)
        count += 1

print("Visualización completada.")
```
Cuando termina de mostrar todas las imágenes seleccionadas, imprime un mensaje para confirmar que el proceso ha concluido.

## Ejecución

- ### Angry
  
![Image](https://github.com/user-attachments/assets/786068db-5f04-412c-9dda-1788713bb1d1)

- ### Disgust

![Image](https://github.com/user-attachments/assets/9aaa54d7-eb80-498d-a47d-8f5aa1ffe614)

- ### Fear

![image](https://github.com/user-attachments/assets/c49afe62-dd76-40f9-91d0-b902317e743b)

- ### Happy
  
![image](https://github.com/user-attachments/assets/b3b49207-c884-4caf-a6f0-699588a4a3db)

- ### Neutral

![image](https://github.com/user-attachments/assets/fa28dca9-669f-473e-a30c-c63f819a2a32)

- ### Sad

![image](https://github.com/user-attachments/assets/b14a8ee8-debe-409a-bcbe-d99de3c75d23)

- ### Surprise

![image](https://github.com/user-attachments/assets/975588fc-62f7-4113-b5b3-9d38f06f5621)

## Seleccionar un modelo
En este proyecto se utilizara un CNN (Red Neuronal Convolucional) porque puede aprender automáticamente cómo distinguir emociones en imágenes de caras, funciona bien con las imágenes que preparamos, y nos permite usar lo que aprende para detectar emociones en tiempo real con una cámara web.

![image](https://github.com/user-attachments/assets/b5a75c09-43f1-48c0-8510-d8ae6ec9975c)

## Parametros para el modelo CNN
Utilizaremos el 70% para el entrenamiento, 20% para el testeo y 10% para la evaluación.

![image](https://github.com/user-attachments/assets/1f21a70f-ace2-4da7-89f0-e99208c989e3)

## Evaluación 

Para evaluar se utilizará la matriz de confusión, la cual es una herramienta que nos ayuda a entender cómo está funcionando nuestro modelo CNN al predecir emociones (como "happy", "sad", "angry") con las imágenes del dataset FER 2013.

![image](https://github.com/user-attachments/assets/ab4e373c-b61b-4997-be65-e34ebd87b3f2)
