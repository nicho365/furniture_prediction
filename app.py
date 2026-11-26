import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import gdown
import os
import requests
from io import BytesIO

st.set_page_config(page_title="Furniture Classfication", layout="centered")

@st.cache_resource
def load_model_from_drive():
    file_id = '1198o-g9NyAUk398WzCA7fVuruna_1V_V'
    output_path = 'furniture_vgg16.keras'

    if not os.path.exists(output_path):
        with st.spinner("Downloading model...(only for first time)"):
            gdown.download(id=file_id, output=output_path, quiet=False)

    return tf.keras.models.load_model(output_path)

try:
    model = load_model_from_drive()
except Exception as e:
    st.error(f"Error while load model: {e}")
    st.stop()
class_names = ['bed', 'chair', 'sofa', 'swivelchair', 'table']

def predict(image, model):
    input_shape = model.input_shape
    target_size = (input_shape[1], input_shape[2]) 
    img = image.resize(target_size)
    img_array = img_to_array(img)    
    img_array = np.expand_dims(img_array, axis=0)    
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence

MAX_BYTES = 10 * 1024 * 1024  # batas 10 MB

@st.cache_data(show_spinner=False)
def fetch_image_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0 (FurnitureClassifier)"}
    with requests.get(url, headers=headers, timeout=10, stream=True) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError("URL bukan gambar langsung (Content-Type: " + content_type + "). Gunakan link yang berakhir .jpg/.png, bukan link halaman web.")
        data = b""
        for chunk in r.iter_content(8192):
            data += chunk
            if len(data) > MAX_BYTES:
                raise ValueError("Ukuran gambar terlalu besar (maks 10 MB).")
    return Image.open(BytesIO(data)).convert("RGB")

st.title("Furniture Classification")
st.write("This model prediction use Convolutional Neural Network VGG16 to predict.")

st.header("Performance Analysis Model (Classification Report)")
st.write("This model was trained using the *VGG16* architecture with Transfer Learning. Here is a summary of the evaluation results:")
st.write("* Overall accuracy: `96%`")
st.write("* Key Strengths: The model has the highest Precision and Recall scores in the bed and sofa class. This is because the object has a massive shape and very distinctive visual features compared to other categories..")
st.write("* Limitations: This model can make wrong predictions for images that are not furniture or even furniture images.")

st.markdown("---")

st.info("""
   **Photo Upload Guide for More Accurate Results:**
1. Focus: Make sure the furniture (Bed, Chair, Sofa, Swivelchair, or Table) is the main object and appears the largest in the photo.
2. One Object: Avoid photos that show lots of furniture stacked on top of each other.
3. Lighting and Angles: Use bright lighting. Take photos from a natural angle (e.g., photographing a chair from the front/side, not from above).
4. Not Random Images: The system has a tolerance limit. Images outside the five furniture categories above will be automatically rejected.
""")

tab_upload, tab_url = st.tabs(["Upload photo", "From image URL"])
image = None

with tab_upload:
    uploaded_file = st.file_uploader("Choose or drag photo file here", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

with tab_url:
    image_url = st.text_input("Paste direct image URL", placeholder="https://example.com/sofa.jpg")
    if image_url and image is None:
        try:
            with st.spinner("Fetching image..."):
                image = fetch_image_from_url(image_url.strip())
        except Exception as e:
            st.error(f"Could not load image from URL: {e}")

if image is not None:
    st.image(image, caption="Selected photo", use_container_width=True)

    if st.button("PREDICT"):
        with st.spinner("Loading..."):
            predicted_class, confidence = predict(image, model)

            THRESHOLD = 0.85

            if confidence < THRESHOLD:
                st.warning("Please upload furniture picture")
                st.info(f"Our model can't detect this furniture image valid because of **low confidence level ({confidence:.2%}).** Please ensure that image are furniture picture..")
            else:
                st.success(f"**Prediction results:** {predicted_class.capitalize()}")
                st.info(f"**Confidence level (Confidence):** {confidence:.2%}")

st.markdown("---")
st.markdown(
    "More infos and :star: at [github.com/nicho365/furniture_prediction](https://github.com/nicho365/furniture_prediction)"
)