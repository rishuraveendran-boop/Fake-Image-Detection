import streamlit as st
import cv2
import numpy as np
import tempfile
import plotly.graph_objects as go
import joblib
from PIL import Image
from tensorflow.keras.models import load_model

# ─── CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── LOAD MODEL ───────────────────────────────────────────
@st.cache_resource
def load_detector():
    return joblib.load('cnn_svm_final.pkl')

model = load_detector()

# ─── HELPER FUNCTIONS ─────────────────────────────────────
def predict_image(img):
    img_resized = img.resize((128, 128))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]
    label = "REAL" if pred > 0.5 else "FAKE"
    confidence = float(pred if pred > 0.5 else 1 - pred)
    return label, confidence

def confidence_meter(confidence, label):
    color = "#00C853" if label == "REAL" else "#D50000"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': f"Confidence: {label}", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "#ffebee"},
                {'range': [50, 75], 'color': "#fff9c4"},
                {'range': [75, 100], 'color': "#e8f5e9"}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100
            }
        },
        number={'suffix': "%", 'font': {'size': 28}}
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=0, l=30, r=30))
    return fig

def analyze_video(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_indices, predictions, suspicious_frames = [], [], []
    frame_count = 0
    progress = st.progress(0)
    status = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_array = np.expand_dims(cv2.resize(frame_rgb, (128, 128)) / 255.0, axis=0)
            pred = float(model.predict(img_array, verbose=0)[0][0])
            predictions.append(pred)
            frame_indices.append(frame_count)
            if pred < 0.3:
                suspicious_frames.append((frame_count, frame_rgb, pred))
            progress.progress(min(frame_count / total_frames, 1.0))
            status.text(f"Analyzing frame {frame_count}/{total_frames}...")
        frame_count += 1

    cap.release()
    progress.empty()
    status.empty()
    return frame_indices, predictions, suspicious_frames, fps

def frame_chart(frame_indices, predictions, fps):
    timestamps = [f / fps for f in frame_indices]
    colors = ["#00C853" if p > 0.5 else "#D50000" for p in predictions]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=predictions,
        mode='lines+markers',
        line=dict(color='#2196F3', width=2),
        marker=dict(color=colors, size=8),
        name='Frame Score'
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                  annotation_text="Decision Boundary (0.5)")
    fig.update_layout(
        title="Frame-by-Frame Analysis",
        xaxis_title="Time (seconds)",
        yaxis_title="REAL probability (1=REAL, 0=FAKE)",
        yaxis=dict(range=[0, 1]),
        height=400,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white")
    )
    return fig

# ─── SIDEBAR ──────────────────────────────────────────────
st.sidebar.title("🔍 AI Fake Detector")
st.sidebar.markdown("Detect AI-generated **images** and **videos**")
st.sidebar.divider()
mode = st.sidebar.radio("Select Mode", ["🖼️ Image Detection", "🎥 Video Detection"])
st.sidebar.divider()
st.sidebar.markdown("**Model Info**")
st.sidebar.info("CNN Model (fake_detector_v3)\nTrained on Universal Fake Detector Dataset\nTest Accuracy: **90.93%**")

# ─── IMAGE MODE ───────────────────────────────────────────
if mode == "🖼️ Image Detection":
    st.title("🖼️ Image Fake Detection")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        with col2:
            with st.spinner("Analyzing..."):
                label, confidence = predict_image(img)
            if label == "REAL":
                st.success("## ✅ REAL IMAGE")
            else:
                st.error("## ❌ FAKE IMAGE")
            st.plotly_chart(confidence_meter(confidence, label), use_container_width=True)

# ─── VIDEO MODE ───────────────────────────────────────────
elif mode == "🎥 Video Detection":
    st.title("🎥 Video Fake Detection")
    frame_skip = st.slider("Analyze every N frames", 5, 30, 10)
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        st.video(uploaded)
        with st.spinner("Analyzing video..."):
            frame_indices, predictions, suspicious_frames, fps = analyze_video(tfile.name, frame_skip)
        avg_score = np.mean(predictions)
        label = "REAL" if avg_score > 0.5 else "FAKE"
        confidence = float(avg_score if avg_score > 0.5 else 1 - avg_score)
        col1, col2 = st.columns(2)
        with col1:
            if label == "REAL":
                st.success("## ✅ REAL VIDEO")
            else:
                st.error("## ❌ FAKE VIDEO")
            st.metric("Frames Analyzed", len(predictions))
            st.metric("Suspicious Frames", len(suspicious_frames))
        with col2:
            st.plotly_chart(confidence_meter(confidence, label), use_container_width=True)
        st.divider()
        st.subheader("📈 Frame-by-Frame Analysis")
        st.plotly_chart(frame_chart(frame_indices, predictions, fps), use_container_width=True)
        if suspicious_frames:
            st.divider()
            st.subheader(f"🚨 Suspicious Frames ({len(suspicious_frames)} detected)")
            cols = st.columns(min(len(suspicious_frames), 4))
            for i, (frame_num, frame_img, pred) in enumerate(suspicious_frames[:8]):
                with cols[i % 4]:
                    st.image(frame_img,
                             caption=f"Frame {frame_num} | {frame_num/fps:.1f}s\nFAKE: {(1-pred)*100:.1f}%",
                             use_container_width=True)
        else:
            st.info("No highly suspicious frames detected.")