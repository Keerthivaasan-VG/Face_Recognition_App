import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(
    page_title="Unsupervised Learning Studio",
    page_icon="📊",
    layout="wide"
)

# -------------------- CUSTOM CSS -------------------- #
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
h1, h2, h3 {
    color: #1f2937;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
}
.stFileUploader {
    background-color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER -------------------- #
st.markdown(
    "<h1 style='text-align: center;'>📊 Unsupervised Learning Studio</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Clustering & Association Rule Analysis with Camera and File Input</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------- SIDEBAR -------------------- #
st.sidebar.header("📥 Input Options")

input_mode = st.sidebar.radio(
    "Select Input Method",
    ["📁 File Upload", "📷 Camera Input"]
)

data = None
image = None

# -------------------- FILE UPLOAD -------------------- #
if input_mode == "📁 File Upload":
    file_type = st.sidebar.selectbox(
        "Select File Type",
        ["CSV Dataset", "Image"]
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload File",
        type=["csv", "png", "jpg", "jpeg"]
    )

    if uploaded_file:
        if file_type == "CSV Dataset":
            data = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV file uploaded successfully!")

        else:
            image = Image.open(uploaded_file)
            st.sidebar.success("Image uploaded successfully!")

# -------------------- CAMERA INPUT -------------------- #
else:
    st.sidebar.info("Allow camera access to capture image")
    camera_image = st.camera_input("Capture Image")

    if camera_image:
        image = Image.open(camera_image)
        st.sidebar.success("Image captured successfully!")

# -------------------- MAIN CONTENT -------------------- #
tab1, tab2 = st.tabs(["🔵 Clustering", "🟢 Association Rules"])

# -------------------- CLUSTERING TAB -------------------- #
with tab1:
    st.subheader("🔵 Clustering Module")

    if data is not None:
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        numeric_data = data.select_dtypes(include=np.number)

        if numeric_data.empty:
            st.warning("No numeric columns available for clustering.")
        else:
            st.success("Numeric data detected. Ready for clustering.")
            st.info("Clustering logic can be integrated here.")

    elif image is not None:
        st.write("### Image Input")
        st.image(image, caption="Input Image", use_container_width=True)
        st.info("Image-based clustering or feature extraction can be applied.")

    else:
        st.info("Please upload a dataset or capture an image.")

# -------------------- ASSOCIATION TAB -------------------- #
with tab2:
    st.subheader("🟢 Association Rule Mining")

    if data is not None:
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        st.info(
            "Dataset should contain transactional or binary values "
            "(0/1 or True/False) for association rule mining."
        )

        st.success("Ready for Apriori / ECLAT integration.")

    else:
        st.warning("Association rule mining requires a CSV dataset.")

# -------------------- FOOTER -------------------- #
st.divider()
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>"
    "© 2025 Unsupervised Learning Studio | Built with Streamlit"
    "</p>",
    unsafe_allow_html=True
)
