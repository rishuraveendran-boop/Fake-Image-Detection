"""
=============================================================
  Fake Video Detection - Frame-by-Frame Analysis
  Uses the trained CNN model to detect fake videos
=============================================================
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from collections import Counter

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH = "models/best_model.keras"
VIDEO_PATH = "test_video.mp4"  # Change this to your video path
OUTPUT_DIR = "video_analysis"
IMG_SIZE = (64, 64)
FRAMES_PER_SECOND = 1  # Extract 1 frame per second

# ─────────────────────────────────────────────
# LOAD TRAINED MODEL
# ─────────────────────────────────────────────
print("📂 Loading trained model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded!")

# ─────────────────────────────────────────────
# EXTRACT FRAMES FROM VIDEO
# ─────────────────────────────────────────────
def extract_frames(video_path, frames_per_sec=1):
    """Extracts frames from video at specified rate."""
    print(f"\n🎬 Extracting frames from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps // frames_per_sec  # Skip frames to get desired rate
    
    frames = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract frame at intervals
        if frame_count % frame_interval == 0:
            frames.append(frame)
            extracted_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"✅ Extracted {extracted_count} frames from {frame_count} total frames")
    return frames

# ─────────────────────────────────────────────
# PREPROCESS FRAME FOR MODEL
# ─────────────────────────────────────────────
def preprocess_frame(frame):
    """Preprocesses frame same way as training images."""
    # Resize
    img = cv2.resize(frame, IMG_SIZE)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize
    img = img.astype(np.float32) / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# ─────────────────────────────────────────────
# ANALYZE VIDEO
# ─────────────────────────────────────────────
def analyze_video(video_path):
    """Analyzes entire video for fake detection."""
    
    # Extract frames
    frames = extract_frames(video_path, FRAMES_PER_SECOND)
    
    if len(frames) == 0:
        print("❌ No frames extracted! Check video path.")
        return
    
    # Analyze each frame
    print("\n🔍 Analyzing frames...")
    predictions = []
    
    for i, frame in enumerate(frames):
        # Preprocess
        processed = preprocess_frame(frame)
        
        # Predict
        pred_prob = model.predict(processed, verbose=0)[0][0]
        pred_label = "FAKE" if pred_prob > 0.5 else "REAL"
        
        predictions.append({
            'frame_num': i,
            'probability': pred_prob,
            'label': pred_label
        })
        
        print(f"Frame {i+1}/{len(frames)}: {pred_label} (confidence: {abs(pred_prob - 0.5) * 200:.1f}%)")
    
    # ─────────────────────────────────────────────
    # AGGREGATE RESULTS
    # ─────────────────────────────────────────────
    labels = [p['label'] for p in predictions]
    label_counts = Counter(labels)
    
    fake_count = label_counts.get('FAKE', 0)
    real_count = label_counts.get('REAL', 0)
    total = len(predictions)
    
    fake_percentage = (fake_count / total) * 100
    real_percentage = (real_count / total) * 100
    
    print("\n" + "="*50)
    print("📊 VIDEO ANALYSIS RESULTS")
    print("="*50)
    print(f"Total frames analyzed: {total}")
    print(f"FAKE frames: {fake_count} ({fake_percentage:.1f}%)")
    print(f"REAL frames: {real_count} ({real_percentage:.1f}%)")
    print("-"*50)
    
    # Final verdict
    if fake_percentage > 60:
        verdict = "🚨 FAKE VIDEO DETECTED"
        confidence = fake_percentage
    elif real_percentage > 60:
        verdict = "✅ REAL VIDEO DETECTED"
        confidence = real_percentage
    else:
        verdict = "⚠️  UNCERTAIN - Mixed signals"
        confidence = max(fake_percentage, real_percentage)
    
    print(f"\n{verdict}")
    print(f"Confidence: {confidence:.1f}%")
    print("="*50)
    
    return predictions

# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video not found: {VIDEO_PATH}")
        print("Please update VIDEO_PATH in the script to point to your video file.")
    else:
        results = analyze_video(VIDEO_PATH)