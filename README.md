# Sistema de Reconocimiento de Expresiones Faciales

## Descripción General
Este proyecto implementa un sistema de reconocimiento de expresiones faciales en tiempo real utilizando aprendizaje profundo. Está entrenado con el dataset FER2013 y puede detectar siete emociones básicas: enojo, disgusto, miedo, felicidad, tristeza, sorpresa y neutral.

## Características
- Detección de expresiones faciales en tiempo real usando webcam
- Modelo CNN pre-entrenado para clasificación de emociones
- Preprocesamiento y aumento de datos
- Pipeline de entrenamiento y evaluación del modelo
- Visualización de extracción de características

## Estructura del Proyecto
```
.
├── main.py                 # Script principal de ejecución
├── preprocessing.py        # Utilidades de preprocesamiento de datos
├── model_training.py       # Entrenamiento y evaluación del modelo CNN
├── realtime_detector.py    # Detección de emociones en tiempo real
├── haarcascade_frontalface_default.xml  # Cascada de detección facial
├── models/                 # Directorio para modelos guardados
├── preprocessed_data/      # Almacenamiento de datos procesados
└── Fer2013_Dataset/       # Directorio del dataset original
```

## Requisitos
- Python 3.x
- OpenCV
- TensorFlow/Keras
- NumPy
- Matplotlib
- scikit-learn

## Instalación
1. Clonar el repositorio:
```bash
git clone git@github.com:InteligenciaArtificial-Equipo/Unidad_4_Tarea_2.git
cd Unidad_4_Tarea_2
```

2. Instalar paquetes requeridos:
```bash
pip install -r requirements.txt
```

3. **Importante**: El dataset FER2013 no está incluido en este repositorio debido a su tamaño y licencia. Para obtenerlo:
   - Visita la página oficial del [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
   - Descarga el dataset desde Kaggle
   - Coloca el archivo `fer2013.csv` en el directorio `Fer2013_Dataset/`

> **Nota**: El dataset FER2013 es necesario para el entrenamiento del modelo. Sin él, el sistema no podrá funcionar correctamente.

## Uso
1. Ejecutar el script principal:
```bash
python main.py
```

El script realizará:
1. Preprocesamiento del dataset FER2013
2. Entrenamiento del modelo CNN
3. Inicio del detector de emociones en tiempo real

## Funcionamiento del Script Principal
El archivo `main.py` es el script principal que orquesta todo el flujo del proyecto. Su funcionamiento se divide en tres etapas principales:

### 1. Preprocesamiento de Datos
- Carga el dataset FER2013 desde el archivo CSV
- Divide los datos en conjuntos de entrenamiento, prueba y validación
- Realiza el preprocesamiento necesario de las imágenes
- Visualiza un ejemplo de extracción de características para verificación

### 2. Entrenamiento y Evaluación
- Entrena el modelo CNN con los datos procesados
- Evalúa el rendimiento del modelo en el conjunto de prueba
- Genera y muestra gráficas de precisión y pérdida
- Crea la matriz de confusión
- Guarda el modelo entrenado en el directorio `models/`

### 3. Detección en Tiempo Real
- Carga el modelo entrenado
- Inicia la captura de video desde la webcam
- Detecta rostros en tiempo real
- Clasifica las expresiones faciales
- Muestra los resultados en una ventana con las etiquetas de emociones

Cada etapa muestra mensajes informativos en la consola para seguir el progreso del proceso.

## Arquitectura del Modelo
El sistema utiliza una Red Neuronal Convolucional (CNN) con la siguiente arquitectura:

### Capas de la Red
1. **Primer Bloque Convolucional**
   - Conv2D (32 filtros, 3x3) + ReLU
   - BatchNormalization
   - Conv2D (32 filtros, 3x3) + ReLU
   - BatchNormalization
   - MaxPooling2D (2x2)
   - Dropout (0.25)

2. **Segundo Bloque Convolucional**
   - Conv2D (64 filtros, 3x3) + ReLU
   - BatchNormalization
   - Conv2D (64 filtros, 3x3) + ReLU
   - BatchNormalization
   - MaxPooling2D (2x2)
   - Dropout (0.25)

3. **Tercer Bloque Convolucional**
   - Conv2D (128 filtros, 3x3) + ReLU
   - BatchNormalization
   - Conv2D (128 filtros, 3x3) + ReLU
   - BatchNormalization
   - MaxPooling2D (2x2)
   - Dropout (0.25)

4. **Capas Densas**
   - Flatten
   - Dense (256 unidades) + ReLU
   - BatchNormalization
   - Dropout (0.5)
   - Dense (7 unidades) + Softmax

### Características del Modelo
- **Entrada**: Imágenes en escala de grises de 48x48 píxeles
- **Optimizador**: Adam con learning rate inicial de 0.001
- **Función de Pérdida**: Categorical Crossentropy
- **Métrica**: Accuracy
- **Regularización**: Dropout y BatchNormalization
- **Callbacks**: Early Stopping y ReduceLROnPlateau

### Hiperparámetros de Entrenamiento
- **Batch Size**: 64
- **Épocas**: 50 (con early stopping)
- **Learning Rate**: Reducción automática cuando la pérdida de validación se estanca

## Detector en Tiempo Real
El archivo `realtime_detector.py` implementa la funcionalidad de detección de emociones en tiempo real. Sus características principales son:

### Componentes Principales
- **Detección Facial**: Utiliza el clasificador Haar Cascade de OpenCV para detectar rostros en el video
- **Preprocesamiento**: Convierte cada rostro detectado a escala de grises y lo redimensiona a 48x48 píxeles
- **Clasificación**: Utiliza el modelo CNN entrenado para predecir la emoción
- **Visualización**: Muestra el video en tiempo real con rectángulos alrededor de los rostros y etiquetas de emociones

### Funcionalidades
- Captura de video en tiempo real desde la webcam
- Detección múltiple de rostros en cada frame
- Visualización de la emoción detectada y su nivel de confianza
- Interfaz gráfica con OpenCV
- Control de salida con la tecla 'q'

### Proceso de Detección
1. Captura de frame desde la webcam
2. Conversión a escala de grises
3. Detección de rostros usando Haar Cascade
4. Para cada rostro detectado:
   - Redimensionamiento a 48x48 píxeles
   - Normalización de valores (0-1)
   - Predicción de emoción usando el modelo CNN
   - Visualización de resultados

### Manejo de Errores
- Verificación de carga exitosa del modelo
- Comprobación de disponibilidad de la webcam
- Validación del clasificador Haar Cascade
- Manejo de errores en la captura de frames

## Rendimiento
El modelo logra una precisión competitiva en el conjunto de prueba de FER2013. La detección en tiempo real está optimizada para un rendimiento fluido en hardware estándar.

## Ejemplos de Ejecución

### Detección en Tiempo Real
![Screenshot from 2025-05-29 18-10-24](https://github.com/user-attachments/assets/01d5b6c9-86c3-412a-aefc-9f346bc518b8)
![Screenshot from 2025-05-29 18-10-02](https://github.com/user-attachments/assets/0816524e-e78e-4922-9e42-551910a124dc)
![Screenshot from 2025-05-29 18-02-26](https://github.com/user-attachments/assets/20e03e05-9f20-43f2-b0cd-850fb06be14c)

### Visualización de Características
![Screenshot from 2025-05-29 18-11-17](https://github.com/user-attachments/assets/ec461db6-a509-4e42-a928-cf55aa0e678b)

### Resultados del Entrenamiento
![image](https://github.com/user-attachments/assets/d149edf9-e266-4098-983f-6c7f32073aa0)
![image](https://github.com/user-attachments/assets/b647fc6e-77ec-40c1-8830-32460ad3a010)

