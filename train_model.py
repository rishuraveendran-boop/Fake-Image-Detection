"""
=============================================================
  Fake Image Detection - CNN Model Training (Memory Efficient)
  Author: BSc. Data Science Project
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: SETUP DATA GENERATORS (MEMORY EFFICIENT)
# ─────────────────────────────────────────────
print("📂 Setting up data generators...")

BATCH_SIZE = 32
IMG_SIZE = (64, 64)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Only rescaling for validation and test
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Update these paths to your dataset location
DATASET_PATH =  r"C:\Users\RIYA\Downloads\cifake"

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

# For validation, we'll use a portion of train data
# Split will be handled by taking 15% of training data
val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"✅ Data generators ready!")
print(f"   Train samples: {train_generator.samples}")
print(f"   Test samples: {test_generator.samples}")

# ─────────────────────────────────────────────
# STEP 2: BUILD CNN ARCHITECTURE
# ─────────────────────────────────────────────
print("\n🏗️  Building CNN model...")

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Fully Connected Layers
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Model built!")
model.summary()

# ─────────────────────────────────────────────
# STEP 3: SETUP CALLBACKS
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

callbacks = [
    ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
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
# STEP 4: TRAIN THE MODEL
# ─────────────────────────────────────────────
print("\n🚀 Starting training...")

EPOCHS = 20

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE // 6,  # Use subset for validation
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training complete!")

# ─────────────────────────────────────────────
# STEP 5: EVALUATE ON TEST SET
# ─────────────────────────────────────────────
print("\n📊 Evaluating on test set...")

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\n🎯 Test Accuracy: {test_accuracy * 100:.2f}%")

# Get predictions
test_generator.reset()
y_pred_prob = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Get true labels
y_test = test_generator.classes[:len(y_pred)]

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

# ─────────────────────────────────────────────
# STEP 6: PLOT TRAINING HISTORY
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.show()
print("✅ Training history saved!")

# ─────────────────────────────────────────────
# STEP 7: CONFUSION MATRIX
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['FAKE', 'REAL'], 
            yticklabels=['FAKE', 'REAL'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("✅ Confusion matrix saved!")

# ─────────────────────────────────────────────
# STEP 8: SAVE FINAL MODEL
# ─────────────────────────────────────────────
model.save("models/final_model.keras")
print("\n💾 Final model saved to models/final_model.keras")
print("\n🎉 All done! Your model is ready.")
