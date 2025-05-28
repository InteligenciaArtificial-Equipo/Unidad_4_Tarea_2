import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Parámetros básicos
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
NUM_CLASSES = 7
EPOCHS = 100
INITIAL_LR = 1e-4

# 1. Creación de datasets desde directorios
train_ds = image_dataset_from_directory(
    'preprocessed_data/train',             # Directorio de entrenamiento
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    validation_split=0.1,                  # 10% de validación interna
    subset='training',
    seed=42
)                                         # :contentReference[oaicite:6]{index=6}

val_ds = image_dataset_from_directory(
    'preprocessed_data/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    validation_split=0.1,
    subset='validation',
    seed=42
)                                         # :contentReference[oaicite:7]{index=7}

test_ds = image_dataset_from_directory(
    'preprocessed_data/test',
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=False
)                                         # :contentReference[oaicite:8]{index=8}

# 2. Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),       # :contentReference[oaicite:9]{index=9}
    layers.RandomRotation(0.1),            # :contentReference[oaicite:10]{index=10}
    layers.RandAugment(),                  # :contentReference[oaicite:11]{index=11}
])

def mixup(batch_images, batch_labels, alpha=0.2):
    """Aplica MixUp a un batch."""
    lam = np.random.beta(alpha, alpha)
    idx = tf.random.shuffle(tf.range(tf.shape(batch_images)[0]))
    mixed_images = lam * batch_images + (1 - lam) * tf.gather(batch_images, idx)
    mixed_labels = lam * batch_labels + (1 - lam) * tf.gather(batch_labels, idx)
    return mixed_images, mixed_labels

def augment(ds):
    for images, labels in ds:
        images = data_augmentation(images)
        images, labels = mixup(images, labels)
        yield images, labels

train_ds_aug = tf.data.Dataset.from_generator(
    lambda: augment(train_ds),
    output_signature=(
        tf.TensorSpec(shape=(None, *IMAGE_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

# 3. Cálculo de class weights para balance
labels_list = np.concatenate([y for x, y in train_ds], axis=0)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=np.argmax(labels_list, axis=1)
)                                         # :contentReference[oaicite:12]{index=12}
class_weights = dict(enumerate(class_weights))

# 4. Construcción del modelo con EfficientNetB0
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)                                         # :contentReference[oaicite:13]{index=13}
# Descongelar últimas capas para fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)

# 5. Compilación con Focal Loss
model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss=CategoricalFocalCrossentropy(gamma=2.0),    # :contentReference[oaicite:14]{index=14}
    metrics=['accuracy']
)

# 6. Callbacks
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
)                                         # :contentReference[oaicite:15]{index=15}

early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)                                         # :contentReference[oaicite:16]{index=16}

# 7. Entrenamiento
model.fit(
    train_ds_aug,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=[reduce_lr, early_stop]
)

# 8. Evaluación y guardado
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")

model.save('emotion_detector.keras')
