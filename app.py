import efficientnet.keras as efn
import streamlit as st
import numpy as np
import os

from PIL import Image
from constants import *
from datetime import datetime
from tensorflow.keras.models import load_model

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    # Convert the image to a numpy array
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize image by dividing by 255
    img_array = np.expand_dims(image_array, axis=0)  # Expand dimensions to match model input
    return img_array


# Function to run inference
def run_inference(model, img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)  # Run model prediction
    # Interpret the prediction
    return 'Lagophthalmos' if prediction[0][0] < 0.5 else 'Normal'

def download_model_wandb(root_path):
    import wandb
    os.system(f'wandb login {wandb_api_key}')
    # wandb.init()
    run = wandb.init(settings=wandb.Settings(start_method="fork"))
    artifact = run.use_artifact(wandb_artifact_path, type='model')
    artifact_dir = artifact.download(root_path)


def download_model(model_link):
    os.system(f'gdown {model_link}')

if not os.path.exists(model_path):
    download_model(model_link)

import time
t1 = time.time()
# Load the model
model = load_model(model_path)

# Streamlit app
st.title("Lagophthalmos Classification Portal")

# Initial placeholder content
placeholder = st.empty()

with placeholder.container():
    st.write("""
        This portal allows you to upload images and classify them using a pre-trained deep learning model.
    """)

    st.header("Image Upload Guidelines")
    st.write("""
        Please ensure your images meet the following requirements for optimal classification:
        - Format: JPEG, JPG, PNG
        - Close-up shot of the Person's eye in well-lit environment
    """)

    st.subheader("Example Images")
    cols = st.columns(3)
    for i, img_path in enumerate(example_images):
        with cols[i]:
            example_image = Image.open(img_path)
            example_image = example_image.resize((190, 120))
            st.image(example_image, use_column_width=True)
print('Time Taken ', time.time() - t1)
# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Clear placeholder content
    placeholder.empty()

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify image with spinner
    with st.spinner('Processing the image...'):
        prediction = run_inference(model, image)

    st.write(f"Status: **{prediction}**")

    # Save the uploaded image with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"{output_folder}uploaded_image_{timestamp}.png"
    image.save(image_filename)

    # Add footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<footer><p>Developed by <strong>Dr Maram Alnefaie - Ophthalmologist</strong></p></footer>", unsafe_allow_html=True)
