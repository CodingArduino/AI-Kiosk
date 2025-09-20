import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io

st.title("🩺 Skin Disease Detection Demo (Fast & Online)")

# Upload image
uploaded_file = st.file_uploader("Upload a skin image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Hugging Face inference
    client = InferenceClient(repo_id="soumickmj/skin-cancer-detection")  # hosted model
    img_bytes = uploaded_file.read()
    result = client.predict(img_bytes)

    st.subheader("Prediction:")
    st.json(result)
