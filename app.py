import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import json

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🧑",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #141e30, #243b55);
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00E5FF;
}
.subtitle {
    text-align: center;
    color: #E0F7FA;
    font-size: 18px;
}
.card {
    background-color: #1c1c1c;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(0,229,255,0.3);
}
.result {
    font-size: 28px;
    color: #76FF03;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown('<div class="title">🧑 Face Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload or capture an image to identify the person</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/face_model.h5")

model = load_model()

# Load labels
with open("labels.json", "r") as f:
    labels = json.load(f)

# ------------------ IMAGE PREPROCESS ------------------
def preprocess_image(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))   # change if needed
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ------------------ UI CARD ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

input_method = st.radio(
    "📸 Select Input Method",
    ["Upload Image", "Use Camera"],
    horizontal=True
)

image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "📂 Upload an Image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file)

else:
    camera_image = st.camera_input("📷 Capture Image")
    if camera_image:
        image = Image.open(camera_image)

# ------------------ PREDICTION ------------------
if image:
    st.image(image, caption="Input Image", use_column_width=True)

    if st.button("🔍 Recognize Face"):
        with st.spinner("Analyzing face..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            person_name = labels[str(class_index)]

        st.markdown(
            f'<div class="result">👤 {person_name}<br>Confidence: {confidence*100:.2f}%</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

