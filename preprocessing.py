

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")



DATASET_PATH = r"C:\Users\RIYA\Downloads\cifake"          
TRAIN_PATH   = os.path.join(DATASET_PATH, "train")
TEST_PATH    = os.path.join(DATASET_PATH, "test")

IMG_SIZE     = (64, 64)               # Resize all images to 64x64
CHANNELS     = 3                      # RGB

# ─────────────────────────────────────────────
# STEP 2: EXPLORE DATASET STRUCTURE
# ─────────────────────────────────────────────
def explore_dataset(path, split_name="Train"):
    real_count = len(os.listdir(os.path.join(path, "REAL")))
    fake_count = len(os.listdir(os.path.join(path, "FAKE")))
    print(f"\n📁 {split_name} | REAL: {real_count} | FAKE: {fake_count} | Total: {real_count+fake_count}")
    return real_count, fake_count

train_real, train_fake = explore_dataset(TRAIN_PATH, "Train")
test_real,  test_fake  = explore_dataset(TEST_PATH,  "Test")

# ─────────────────────────────────────────────
# STEP 3: VISUALISE SAMPLE IMAGES
# ─────────────────────────────────────────────
def show_sample_images(path, label, n=5):
    folder = os.path.join(path, label)
    images = os.listdir(folder)[:n]
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    fig.suptitle(f"Sample {label} Images", fontsize=14, fontweight="bold")
    for ax, img_name in zip(axes, images):
        img = mpimg.imread(os.path.join(folder, img_name))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, color="green" if label == "REAL" else "red")
    plt.tight_layout()
    plt.savefig(f"sample_{label.lower()}_images.png", dpi=150)
    plt.show()

show_sample_images(TRAIN_PATH, "REAL")
show_sample_images(TRAIN_PATH, "FAKE")

# ─────────────────────────────────────────────
# STEP 4: CLASS DISTRIBUTION CHART
# ─────────────────────────────────────────────
def plot_class_distribution(train_real, train_fake, test_real, test_fake):
    x = np.arange(2)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width/2, [train_real, train_fake], width, label="Train", color=["#4CAF50","#F44336"])
    bars2 = ax.bar(x + width/2, [test_real,  test_fake],  width, label="Test",  color=["#81C784","#E57373"])
    ax.set_title("Class Distribution: Real vs Fake", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["REAL", "FAKE"])
    ax.set_ylabel("Number of Images"); ax.legend()
    for bar in bars1 + bars2:
        ax.annotate(f'{bar.get_height():,}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha='center')
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.show()

plot_class_distribution(train_real, train_fake, test_real, test_fake)

# ─────────────────────────────────────────────
# STEP 5: LOAD & PREPROCESS IMAGES
# ─────────────────────────────────────────────
def load_and_preprocess_images(path, img_size=IMG_SIZE):
    """
    For each image:
      1. Resize to 64x64
      2. Convert BGR → RGB
      3. Apply Gaussian Blur (noise reduction)
      4. Normalize pixels to [0.0, 1.0]
    Labels: REAL=0, FAKE=1
    """
    X, y = [], []
    for label, value in {"REAL": 0, "FAKE": 1}.items():
        class_path = os.path.join(path, label)
        files = os.listdir(class_path)
        print(f"\n🔄 Loading {label} ({len(files)} images)...")
        for img_name in files:
            img = cv2.imread(os.path.join(class_path, img_name))
            if img is None: continue
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            img = img.astype(np.float32) / 255.0
            X.append(img); y.append(value)
    X, y = np.array(X), np.array(y)
    print(f"\n✅ X shape: {X.shape} | y shape: {y.shape}")
    return X, y

X_train, y_train = load_and_preprocess_images(TRAIN_PATH)
X_test,  y_test  = load_and_preprocess_images(TEST_PATH)

# ─────────────────────────────────────────────
# STEP 6: TRAIN → TRAIN + VALIDATION SPLIT
# ─────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)
print(f"\n📐 Final Splits → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 7: PIXEL STATISTICS
# ─────────────────────────────────────────────
print(f"\n📊 Pixel Stats → Min:{X_train.min():.2f} Max:{X_train.max():.2f} Mean:{X_train.mean():.4f} Std:{X_train.std():.4f}")

# ─────────────────────────────────────────────
# STEP 8: VISUALISE PREPROCESSING STEPS
# ─────────────────────────────────────────────
def show_preprocessing_steps(path, label="REAL"):
    folder   = os.path.join(path, label)
    img_path = os.path.join(folder, os.listdir(folder)[0])
    original = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    resized  = cv2.resize(original, IMG_SIZE)
    blurred  = cv2.GaussianBlur(resized, (3, 3), 0)
    normed   = blurred.astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Preprocessing Steps — {label}", fontsize=13, fontweight="bold")
    for ax, (img, title) in zip(axes, [(original,"1. Original"),(resized,"2. Resized"),(blurred,"3. Denoised"),(normed,"4. Normalized")]):
        ax.imshow(img); ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"preprocessing_{label.lower()}.png", dpi=150)
    plt.show()

show_preprocessing_steps(TRAIN_PATH, "REAL")
show_preprocessing_steps(TRAIN_PATH, "FAKE")

# ─────────────────────────────────────────────
# STEP 9: SAVE PREPROCESSED DATA
# ─────────────────────────────────────────────
os.makedirs("processed_data", exist_ok=True)
np.save("processed_data/X_train.npy", X_train)
np.save("processed_data/X_val.npy",   X_val)
np.save("processed_data/X_test.npy",  X_test)
np.save("processed_data/y_train.npy", y_train)
np.save("processed_data/y_val.npy",   y_val)
np.save("processed_data/y_test.npy",  y_test)

print("\n🎉 Preprocessing Complete! Data saved to processed_data/")