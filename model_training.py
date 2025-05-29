import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import numpy as np
import os

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, emotion_labels,
                             model_save_path='models/emotion_detector_cnn.keras'):
    
    input_shape = (48, 48, 1)
    model = create_cnn_model(input_shape, num_classes)
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    epochs = 50
    batch_size = 64

    print("\nIniciando entrenamiento del modelo.")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )
    print("Entrenamiento completado.")

    # Guardar el modelo
    model.save(model_save_path)
    print(f"Modelo guardado en: {model_save_path}")

    # Visualizar historial de entrenamiento
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nEvaluando el modelo en el conjunto de prueba.")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Perdida en el conjunto de prueba: {loss:.4f}")
    print(f"Precision en el conjunto de prueba: {accuracy:.4f}")

    print("\nGenerando Matriz de Confusion.")
    # 1. Obtener las predicciones del modelo en el conjunto de prueba
    y_pred_probs = model.predict(X_test)
    # 2. Convertir las probabilidades a clases
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    # 3. Convertir las etiquetas verdaderas de one-hot a clases
    y_true_classes = np.argmax(y_test, axis=1)

    # 4. Calcular la matriz de confusión
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # 5. Visualizar la matriz de confusión
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title('Matriz de Confusion')
    plt.show()

if __name__ == '__main__':
    print("Ejecutar despues de data_preprocessing.py.")