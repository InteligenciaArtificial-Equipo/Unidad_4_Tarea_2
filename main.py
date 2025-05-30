import os
from data_preprocessing import load_and_preprocess_fer2013, visualize_feature_extraction_example
from model_training import train_and_evaluate_model
from realtime_detector import run_realtime_detector

FER2013_DATA_PATH = 'fer2013.csv'
MODEL_SAVE_PATH = 'models/emotion_detector_cnn.keras'

def main():
    print("--- Inicio ---")

    print("\n## 1. Procesando y dividiendo el dataset FER-2013.")
    X_train, X_test, X_val, y_train, y_test, y_val, num_classes, emotion_labels = \
        load_and_preprocess_fer2013(FER2013_DATA_PATH)

    if X_train is None:
        print("Error: No se pudo cargar el dataset.")
        return

    print("\nVisualizando un ejemplo de extracción de características.")
    sample_image_for_viz = X_train[0].squeeze() * 255
    visualize_feature_extraction_example(sample_image_for_viz)

    print("\n---")
    print("## 2. Entrenando y evaluando el modelo CNN.")
    train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, MODEL_SAVE_PATH)

    print("\n---")
    print("## 3. Iniciando el detector de emociones en tiempo real.")
    run_realtime_detector(MODEL_SAVE_PATH, emotion_labels)

    print("\n--- Pipeline completado. ---")

if __name__ == '__main__':
    main()
