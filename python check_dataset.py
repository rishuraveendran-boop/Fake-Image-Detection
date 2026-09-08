
import os

DATASET_PATH = r"D:\mlproject\universal_fake_detector"

train_real = len(os.listdir(os.path.join(DATASET_PATH, "train", "REAL")))
train_fake = len(os.listdir(os.path.join(DATASET_PATH, "train", "FAKE")))
test_real = len(os.listdir(os.path.join(DATASET_PATH, "test", "REAL")))
test_fake = len(os.listdir(os.path.join(DATASET_PATH, "test", "FAKE")))

print("\n📊 DATASET STATISTICS")
print("="*50)
print(f"TRAIN → REAL: {train_real:,} | FAKE: {train_fake:,} | Total: {train_real + train_fake:,}")
print(f"TEST  → REAL: {test_real:,} | FAKE: {test_fake:,} | Total: {test_real + test_fake:,}")
print("="*50)
print(f"GRAND TOTAL: {train_real + train_fake + test_real + test_fake:,} images")
print("="*50)