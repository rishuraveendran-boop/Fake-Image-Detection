
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH = r"D:\mlproject\universal_fake_detector"
IMG_SIZE = (128, 128)  # Higher res to handle diverse content
BATCH_SIZE = 32
EPOCHS = 25

print("🌍 TRAINING UNIVERSAL FAKE DETECTOR")
print("="*60)
print(f"📊 Image size: {IMG_SIZE}")
print(f"📁 Dataset: {DATASET_PATH}\n")

# ─────────────────────────────────────────────
# DATA GENERATORS
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"✅ Data loaded!")
print(f"   Train: {train_generator.samples:,} images")
print(f"   Test: {test_generator.samples:,} images\n")

# ─────────────────────────────────────────────
# BUILD ROBUST CNN
# ─────────────────────────────────────────────
print("🏗️  Building universal detector CNN...\n")

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    # Block 4
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    # Dense layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
os.makedirs("universal_models", exist_ok=True)

callbacks = [
    ModelCheckpoint(
        'universal_models/best_universal_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("\n🚀 Starting training...\n")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training complete!")

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
print("\n📊 Evaluating...")

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n🎯 Final Test Accuracy: {test_accuracy * 100:.2f}%")

test_generator.reset()
y_pred_prob = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_test = test_generator.classes[:len(y_pred)]

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

# Visualizations (same as before)
# ... [rest of visualization code]

model.save("universal_models/final_universal_model.keras")
print("\n🎉 Universal detector ready for ANY image type!")

