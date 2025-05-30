import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
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

def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, num_classes,
                             model_save_path='models/emotion_detector_cnn.keras'):
    input_shape = (48, 48, 1)
    model = create_cnn_model(input_shape, num_classes)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    epochs = 50
    batch_size = 64

    print("\nIniciando entrenamiento del modelo...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )
    print("Entrenamiento completado.")

    model.save(model_save_path)
    print(f"Modelo guardado en: {model_save_path}")

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

    print("\nEvaluando el modelo en el conjunto de prueba...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Pérdida en el conjunto de prueba: {loss:.4f}")
    print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")

if __name__ == '__main__':
    print("Este script está diseñado para ser importado o ejecutado después de data_preprocessing.py.")
