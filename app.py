# ==========================================
# SmartVision AI - Apple Style Minimal UI
# Mac M2 Optimized | YOLO Live Camera
# ==========================================

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import tempfile
import os
import cv2
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🍎",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS (Apple Minimal)
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f5f7;
}
h1, h2, h3 {
    color: #111111;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}
.stButton>button {
    border-radius: 12px;
    border: none;
    background-color: #111111;
    color: white;
    padding: 0.6rem 1rem;
}
.stButton>button:hover {
    background-color: #333333;
}
div[data-testid="stSidebar"] {
    background-color: white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = "/Users/sabari/Documents/DS/GUVI/SmartVision-AI"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_classification_models():
    return {
        "MobileNet": tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "mobilenet_model.h5")
        ),
        "ResNet50": tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "resnet50_model.h5")
        ),
        "EfficientNet": tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "efficientnet_model.h5")
        ),
        "VGG16": tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "vgg16_model.h5")
        )
    }

@st.cache_resource
def load_yolo():
    return YOLO(os.path.join(MODEL_DIR, "yolo_best.pt"))

# -----------------------------
# CLASS NAMES
# -----------------------------
CLASS_NAMES = [
    "airplane","bed","bench","bicycle","bird",
    "bottle","bowl","bus","cake","car",
    "cat","chair","couch","cow","cup",
    "dog","elephant","horse","motorcycle",
    "person","pizza","potted plant",
    "stop sign","traffic light","truck"
]

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0

    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    arr = np.expand_dims(arr, axis=0)
    return arr

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("👁️ SmartVision AI")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Image Classification",
        "Object Detection",
        "Live Camera",
        "About"
    ]
)

# -----------------------------
# HOME
# -----------------------------
if page == "Home":

    st.title("SmartVision AI")
    st.subheader("Minimal Computer Vision Experience")

    st.markdown("""
    ### Features

    • Image Classification  
    • YOLO Object Detection  
    • Real-Time Live Camera Detection  
    • Apple Style Minimal UI  
    • Mac M2 Optimized
    """)

# -----------------------------
# IMAGE CLASSIFICATION
# -----------------------------
elif page == "Image Classification":

    st.title("Image Classification")

    models = load_classification_models()

    model_name = st.selectbox(
        "Choose Model",
        list(models.keys())
    )

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=250)

        x = preprocess_image(img)

        pred = models[model_name].predict(x)

        idx = np.argmax(pred)
        conf = np.max(pred) * 100

        st.success(f"Prediction: {CLASS_NAMES[idx]}")
        st.write(f"Confidence: {conf:.2f}%")

# -----------------------------
# OBJECT DETECTION
# -----------------------------
elif page == "Object Detection":

    st.title("YOLO Object Detection")

    yolo = load_yolo()

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=300)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            path = tmp.name

        results = yolo(path)

        out = results[0].plot()

        st.image(out, caption="Detected", width=450)

        os.remove(path)

# -----------------------------
# LIVE CAMERA
# -----------------------------
elif page == "Live Camera":

    st.title("Live Camera Detection")

    st.caption("Compact Real-Time Detection")

    run = st.checkbox("Start Camera")

    frame_window = st.image([])

    yolo = load_yolo()

    cam = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

    cam.set(3, 480)   # width
    cam.set(4, 320)   # height

    while run:

        ret, frame = cam.read()

        if not ret:
            st.error("Unable to access camera")
            break

        results = yolo(frame)

        annotated = results[0].plot()

        frame_window.image(
            annotated,
            channels="BGR",
            width=420
        )

        time.sleep(0.03)

    cam.release()

# -----------------------------
# ABOUT
# -----------------------------
elif page == "About":

    st.title("About")

    st.markdown("""
    **SmartVision AI** is a portfolio-ready computer vision app built using:

    - TensorFlow
    - YOLOv8
    - OpenCV
    - Streamlit

    Designed for smooth performance on Mac M2.
    
    -SabariRam.
    """)