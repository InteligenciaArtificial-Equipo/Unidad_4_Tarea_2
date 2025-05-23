## Unidad 4 - Tarea 1

## Preprocesamiento para un detector de emociones con Fer - 2013 con el modelo CNN

## Integrantes

### Chaparro Castillo Christopher
### Peñuelas López Luis Antonio

## Como Funciona?

## Preprocess.py
Código para llevar a cabo el preprocesamiento de imágenes para nuestro modelo CNN.

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
Es una biblioteca para generar aleatoriedad. Sirve para mezclar las imágenes aleatoriamente antes de dividirlas en entrenamiento, prueba y validación, asegurando una distribución justa.

- ### Ruta base del dataset
  
```py
dataset_path = Path("Fer2013_Dataset")
output_path = Path("preprocessed_data")
```
Recibe una lista de listas (matriz) que inicializarán como el estado inicial.
