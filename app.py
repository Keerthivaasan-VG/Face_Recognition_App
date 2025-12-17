import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Face Recognition", page_icon="🧑")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_model.h5")

model = load_model()

# Load labels
with open("labels.json") as f:
    labels = json.load(f)

# ---------------- PREPROCESS ----------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))  # must match training
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- UI ----------------
st.title("🧑 Face Recognition System")
st.caption("Auto-recognizes face from camera or image upload")

option = st.radio(
    "Select Input Method",
    ["Upload Image", "Use Camera"],
    horizontal=True
)

image = None

if option == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file)

else:
    cam = st.camera_input("Capture Image")
    if cam:
        image = Image.open(cam)

# ---------------- AUTO PREDICTION ----------------
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    img = preprocess_image(image)
    preds = model.predict(img)

    class_id = np.argmax(preds)
    confidence = float(np.max(preds))

    # Confidence threshold
    if confidence < 0.75:
        st.error("❌ Unknown Person")
    else:
        st.success(f"👤 {labels[str(class_id)]}")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
