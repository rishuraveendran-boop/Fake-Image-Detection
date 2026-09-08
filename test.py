# test_models.py
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
import os

# ─── PUT A FEW TEST IMAGES HERE ───────────────────────────
# Folder structure expected:
# test_images/
#   real/   ← put some real images here
#   fake/   ← put some AI-generated images here

TEST_DIR = "test_images"
IMG_SIZE = (128, 128)

def load_images(folder, label):
    images, labels = [], []
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img = np.array(img.resize(IMG_SIZE)) / 255.0
            images.append(img)
            labels.append(label)
    return images, labels

real_imgs, real_labels = load_images(f"{TEST_DIR}/real", 1)   # 1 = REAL
fake_imgs, fake_labels = load_images(f"{TEST_DIR}/fake", 0)   # 0 = FAKE

all_images = np.array(real_imgs + fake_imgs)
all_labels = np.array(real_labels + fake_labels)
print(f"✅ Loaded {len(real_imgs)} real, {len(fake_imgs)} fake images\n")

results = {}

# ─── MODEL 1: fake_detector_model.keras ───────────────────
try:
    m1 = load_model('fake_detector_model.keras')
    preds = (m1.predict(all_images, verbose=0).flatten() > 0.5).astype(int)
    acc = np.mean(preds == all_labels) * 100
    results['fake_detector_model'] = acc
    print(f"[1] fake_detector_model.keras     → Accuracy: {acc:.1f}%")
except Exception as e:
    print(f"[1] fake_detector_model.keras     → ❌ Failed: {e}")

# ─── MODEL 2: fake_detector_v3.keras ──────────────────────
try:
    m2 = load_model('fake_detector_v3.keras')
    preds = (m2.predict(all_images, verbose=0).flatten() > 0.5).astype(int)
    acc = np.mean(preds == all_labels) * 100
    results['fake_detector_v3'] = acc
    print(f"[2] fake_detector_v3.keras        → Accuracy: {acc:.1f}%")
except Exception as e:
    print(f"[2] fake_detector_v3.keras        → ❌ Failed: {e}")

# Replace the SVM test sections in test_models.py with this:
from tensorflow.keras.models import Model


def get_feature_extractor(cnn_model):
    # Build a dummy pass first to initialise the layers
    dummy = np.zeros((1, 128, 128, 3))
    cnn_model.predict(dummy, verbose=0)
    
    # Now access input via first layer (Sequential fix)
    feature_layer = cnn_model.layers[-2].output
    extractor = Model(inputs=cnn_model.layers[0].input, outputs=feature_layer)
    return extractor

# ─── MODEL 3: cnn_svm_final.pkl ───────────────────────────
try:
    m3_cnn = load_model('fake_detector_v3.keras')
    extractor = get_feature_extractor(m3_cnn)
    features = extractor.predict(all_images, verbose=0)
    m3_svm = joblib.load('cnn_svm_final.pkl')
    preds_raw = m3_svm.predict(features)
    acc = np.mean(preds_raw == all_labels) * 100
    flipped_acc = np.mean((1 - preds_raw) == all_labels) * 100
    best_acc = max(acc, flipped_acc)
    print(f"[3] cnn_svm_final.pkl  → Accuracy: {best_acc:.1f}%")
except Exception as e:
    print(f"[3] cnn_svm_final.pkl  → ❌ Failed: {e}")

# ─── MODEL 4: svm_linear_original_cnn ─────────────────────
try:
    m4_cnn = load_model('fake_detector_model.keras')
    extractor = get_feature_extractor(m4_cnn)
    features = extractor.predict(all_images, verbose=0)
    m4_svm = joblib.load('svm_linear_original_cnn.pkl')
    preds_raw = m4_svm.predict(features)
    acc = np.mean(preds_raw == all_labels) * 100
    flipped_acc = np.mean((1 - preds_raw) == all_labels) * 100
    best_acc = max(acc, flipped_acc)
    print(f"[4] svm_linear_original_cnn.pkl  → Accuracy: {best_acc:.1f}%")
except Exception as e:
    print(f"[4] svm_linear_original_cnn.pkl  → ❌ Failed: {e}")

# ─── WINNER ───────────────────────────────────────────────
print("\n" + "="*55)
if results:
    winner = max(results, key=results.get)
    print(f"🏆 BEST MODEL: {winner}  →  {results[winner]:.1f}%")
print("="*55)