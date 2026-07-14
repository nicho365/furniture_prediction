# Furniture Image Classification using VGG16

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://furnitureprediction.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/nicho365/furniture_prediction/blob/main/training_model.ipynb)

## Project Overview
This project is a *Computer Vision* implementation using a *Convolutional Neural Network* (CNN) to detect and classify furniture images. The model is trained using the *VGG16* architecture through the *Transfer Learning* method. This application has been publicly deployed with an easy-to-use interactive interface.

## Live Demo & Resources
- **Aplikasi Web (Streamlit):** [Furniture Prediction App](https://furnitureprediction.streamlit.app/)
- **Model Training (Google Colab):** [VGG16 Notebook](https://github.com/nicho365/furniture_prediction/blob/main/training_model.ipynb)
- **Dataset (Kaggle):** [Furniture Detector Dataset](https://www.kaggle.com/datasets/akkithetechie/furniture-detector/data?select=img)

## Furniture Category (Classes)
This model is trained to recognize 5 categories of objects, such as:
1. `Bed` 
2. `Chair`
3. `Sofa`
4. `Swivelchair`
5. `Table`

## Application Feature
- **Interactive Detection:** Users can upload photos (`.jpg`, `.jpeg`, `.png`) and get prediction results along with confidence level.
- **Out-of-Distribution Handling:** Equipped with a *Confidence Threshold* of 85%. If the uploaded image is not furniture (e.g., random images or scenery), the system will give a warning and refuse to classify it forcibly.
- **Performance Insights:** There is a *Classification Report* panel that provides an evaluation of the model's performance, key detection strengths, and model weaknesses (e.g., visual confusion between *Chair* and *Swivelchair*).
- **Cloud Storage Integration:** Due to the large size of the models, the model *files* (`.pkl`) are stored in Google Drive and downloaded automatically using `gdown` when the app is first run in Streamlit Cloud.

## Libraries Used
- **Deep Learning Framework:** TensorFlow & Keras
- **Pre-trained Model:** VGG16
- **Web App Framework:** Streamlit
- **Data & Image Processing:** NumPy, PIL (Pillow)
- **Deployment:** Streamlit Community Cloud
