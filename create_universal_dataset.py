"""
Create Universal Fake Image Detector Dataset
Combines CIFAKE + 140k Faces for all-purpose detection
"""

import os
import shutil
import random
from pathlib import Path

# ─────────────────────────────────────────────
# SOURCE PATHS
# ─────────────────────────────────────────────
# CIFAKE paths (already split)
CIFAKE_TRAIN_REAL = r"C:\Users\RIYA\Downloads\cifake\train\REAL"
CIFAKE_TRAIN_FAKE = r"C:\Users\RIYA\Downloads\cifake\train\FAKE"
CIFAKE_TEST_REAL = r"C:\Users\RIYA\Downloads\cifake\test\REAL"
CIFAKE_TEST_FAKE = r"C:\Users\RIYA\Downloads\cifake\test\FAKE"

# 140k Face dataset
FACES_REAL = r"D:\downloads\archive\real_vs_fake\real-vs-fake\train\real"
FACES_FAKE = r"D:\downloads\archive\real_vs_fake\real-vs-fake\train\fake"

# OUTPUT - New universal dataset
OUTPUT_BASE = r"D:\mlproject\universal_fake_detector"

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
# How many face images to add
FACES_TO_ADD_TRAIN = 20000  # Add 20k faces to training
FACES_TO_ADD_TEST = 5000    # Add 5k faces to testing

# ─────────────────────────────────────────────
# CREATE FOLDER STRUCTURE
# ─────────────────────────────────────────────
folders = [
    os.path.join(OUTPUT_BASE, "train", "REAL"),
    os.path.join(OUTPUT_BASE, "train", "FAKE"),
    os.path.join(OUTPUT_BASE, "test", "REAL"),
    os.path.join(OUTPUT_BASE, "test", "FAKE"),
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("📦 CREATING UNIVERSAL FAKE DETECTOR DATASET")
print("="*60)

# ─────────────────────────────────────────────
# STEP 1: COPY ALL CIFAKE DATA
# ─────────────────────────────────────────────
def copy_all_files(source, dest, label):
    files = [f for f in os.listdir(source) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"\n📁 Copying {len(files):,} {label} images from CIFAKE...")
    
    for i, f in enumerate(files):
        shutil.copy(
            os.path.join(source, f),
            os.path.join(dest, f"cifake_{i:06d}{Path(f).suffix}")
        )
        if (i + 1) % 5000 == 0:
            print(f"   Progress: {i+1}/{len(files)}")
    
    print(f"   ✅ Copied {len(files):,} files")
    return len(files)

print("\n🔄 PHASE 1: Adding CIFAKE Dataset (Diverse Content)")
print("-" * 60)

cifake_train_real = copy_all_files(CIFAKE_TRAIN_REAL, os.path.join(OUTPUT_BASE, "train", "REAL"), "train/REAL")
cifake_train_fake = copy_all_files(CIFAKE_TRAIN_FAKE, os.path.join(OUTPUT_BASE, "train", "FAKE"), "train/FAKE")
cifake_test_real = copy_all_files(CIFAKE_TEST_REAL, os.path.join(OUTPUT_BASE, "test", "REAL"), "test/REAL")
cifake_test_fake = copy_all_files(CIFAKE_TEST_FAKE, os.path.join(OUTPUT_BASE, "test", "FAKE"), "test/FAKE")

# ─────────────────────────────────────────────
# STEP 2: ADD FACE IMAGES
# ─────────────────────────────────────────────
def add_random_faces(source, dest, num_to_add, label):
    files = [f for f in os.listdir(source) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(files) < num_to_add:
        print(f"⚠️  Only {len(files):,} available, using all")
        num_to_add = len(files)
    
    selected = random.sample(files, num_to_add)
    
    print(f"\n📁 Adding {num_to_add:,} {label} face images...")
    
    for i, f in enumerate(selected):
        shutil.copy(
            os.path.join(source, f),
            os.path.join(dest, f"face_{i:06d}{Path(f).suffix}")
        )
        if (i + 1) % 2000 == 0:
            print(f"   Progress: {i+1}/{num_to_add}")
    
    print(f"   ✅ Added {num_to_add:,} face images")
    return num_to_add

print("\n🔄 PHASE 2: Adding High-Res Face Dataset")
print("-" * 60)

faces_train_real = add_random_faces(FACES_REAL, os.path.join(OUTPUT_BASE, "train", "REAL"), FACES_TO_ADD_TRAIN, "train/REAL")
faces_train_fake = add_random_faces(FACES_FAKE, os.path.join(OUTPUT_BASE, "train", "FAKE"), FACES_TO_ADD_TRAIN, "train/FAKE")
faces_test_real = add_random_faces(FACES_REAL, os.path.join(OUTPUT_BASE, "test", "REAL"), FACES_TO_ADD_TEST, "test/REAL")
faces_test_fake = add_random_faces(FACES_FAKE, os.path.join(OUTPUT_BASE, "test", "FAKE"), FACES_TO_ADD_TEST, "test/FAKE")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("🎉 UNIVERSAL DATASET CREATED!")
print("="*60)

print("\n📊 DATASET COMPOSITION:")
print("-" * 60)

print("\nTRAIN SET:")
train_real_total = cifake_train_real + faces_train_real
train_fake_total = cifake_train_fake + faces_train_fake
print(f"  REAL: {train_real_total:,} ({cifake_train_real:,} CIFAKE + {faces_train_real:,} faces)")
print(f"  FAKE: {train_fake_total:,} ({cifake_train_fake:,} CIFAKE + {faces_train_fake:,} faces)")
print(f"  TOTAL: {train_real_total + train_fake_total:,}")

print("\nTEST SET:")
test_real_total = cifake_test_real + faces_test_real
test_fake_total = cifake_test_fake + faces_test_fake
print(f"  REAL: {test_real_total:,} ({cifake_test_real:,} CIFAKE + {faces_test_real:,} faces)")
print(f"  FAKE: {test_fake_total:,} ({cifake_test_fake:,} CIFAKE + {faces_test_fake:,} faces)")
print(f"  TOTAL: {test_real_total + test_fake_total:,}")

print("\n" + "="*60)
print(f"📁 Location: {OUTPUT_BASE}")
print("="*60)
print("\n✅ Ready to train universal detector!")
print("   Run: python train_universal_model.py")