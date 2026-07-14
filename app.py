import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import pickle
import gdown
import os

st.set_page_config(page_title="Furniture Classfication", layout="centered")

@st.cache_resource
def load_model_from_drive():
    url = f'https://drive.google.com/file/d/1RDWBwvl9EECysd4Kjy6JaSJA0SzYhHs8/view?usp=sharing'
    output_path = 'model_furniture.pkl'
    
    if not os.path.exists(output_path):
        with st.spinner("Downloading model...(only for first time)"):
            gdown.download(url, output_path, quiet=False)
    
    with open(output_path, 'rb') as file:
        model = pickle.load(file)
    return model

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
    img_array = img_array / 255.0
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence

st.title("Furniture Classification")
st.write("This model prediction use Convolutional Neural Network VGG16 to predict.")

st.header("Performance Analysis Model (Classification Report)")
st.write("""
This model was trained using the *VGG16* architecture with Transfer Learning. Here is a summary of the evaluation results:
    
    *   Overall accuracy: `95%`
    *   Key Strengths: The model has the highest Precision and Recall scores in the bed and sofa class. This is because the object has a massive shape and very distinctive visual features compared to other categories..
    *   Limitations: This model can make wrong predictions for images that are not furniture or even furniture images.
         """)

st.markdown("---")

st.info("""
   **Photo Upload Guide for More Accurate Results:**
1. Focus: Make sure the furniture (Bed, Chair, Sofa, Swivelchair, or Table) is the main object and appears the largest in the photo.
2. One Object: Avoid photos that show lots of furniture stacked on top of each other.
3. Lighting and Angles: Use bright lighting. Take photos from a natural angle (e.g., photographing a chair from the front/side, not from above).
4. Not Random Images: The system has a tolerance limit. Images outside the five furniture categories above will be automatically rejected.
""")

uploaded_file = st.file_uploader("Choose or drag photo file here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded photo', use_column_width=True)
    
    st.write("")
    
    if st.button("PREDICT"):
        with st.spinner("Loading..."):
            predicted_class, confidence = predict(image, model)
    
            THRESHOLD = 0.85 
            
            if confidence < THRESHOLD:
                st.warning(f"Please upload furniture picture")
                st.info(f"Our model can't detect this furniture image valid because of **low confidence level ({confidence:.2%}).** Please ensure that image are furniture picture..")
            else:
                st.success(f"**Prediction results:** {predicted_class.capitalize()}")
                st.info(f"**Confidence level (Confidence):** {confidence:.2%}")

st.markdown("---")
st.markdown(
    "More infos and :star: at [github.com/nicho365/furniture_prediction](https://github.com/nicho365/furniture_prediction)"
)