import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import pickle
import gdown
import os

# Konfigurasi Halaman (Tab browser)
st.set_page_config(page_title="Furniture Classfication", layout="centered")

@st.cache_resource
def load_model_from_drive():
    url = f'https://drive.google.com/file/d/1RDWBwvl9EECysd4Kjy6JaSJA0SzYhHs8/view?usp=sharing'
    output_path = 'model_furniture.pkl'
    
    if not os.path.exists(output_path):
        with st.spinner("Mengunduh model dari Google Drive (Hanya dilakukan sekali)..."):
            gdown.download(url, output_path, quiet=False)
    
    with open(output_path, 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model_from_drive()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()
class_names = ['bed', 'chair', 'sofa', 'swivelchair', 'table']

def preprocess_and_predict(image, model):
    # 1. Mendapatkan ukuran yang diharapkan model (misal: (None, 150, 150, 3))
    # Kita ambil indeks ke-1 (tinggi) dan ke-2 (lebar)
    input_shape = model.input_shape
    target_size = (input_shape[1], input_shape[2]) 
    
    # 2. Resize gambar secara otomatis mengikuti target_size dari model
    img = image.resize(target_size)
    img_array = img_to_array(img)
    
    # 3. Menambah dimensi untuk batch (1, tinggi, lebar, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. Normalisasi (membagi nilai piksel dengan 255)
    img_array = img_array / 255.0
    
    # 5. Melakukan prediksi
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    return predicted_class, confidence

# UI Aplikasi
st.title("Furniture Classification")
st.write("This model prediction use Convolutional Neural Network VGG16 to predict.")

st.markdown("---")

uploaded_file = st.file_uploader("Choose or drag photo file here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded photo', use_column_width=True)
    
    st.write("")
    
    if st.button("PREDICT"):
        with st.spinner("Loading..."):
            predicted_class, confidence = preprocess_and_predict(image, model)
    
            THRESHOLD = 0.85 
            
            if confidence < THRESHOLD:
                st.warning(f"Please upload furniture picture")
                st.info(f"Our model can't detect this furniture image valid because of **low confidence level ({confidence:.2%}).** Please ensure that image are furniture picture..")
            else:
                st.success(f"**Prediction results:** {predicted_class.capitalize()}")
                st.info(f"**Confidence level (Confidence):** {confidence:.2%}")
