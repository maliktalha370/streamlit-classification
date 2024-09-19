import efficientnet.keras as efn
import streamlit as st
import numpy as np
import os
from PIL import Image
from constants import *
from datetime import datetime
from tensorflow.keras.models import load_model
from utils import test_and_limit_tensorflow, download_model


def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    img_array = np.expand_dims(image_array, axis=0)
    return img_array


def run_inference(model, img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    return 'Lagophthalmos' if prediction[0][0] < 0.5 else 'Normal'


def download_model_wandb(root_path):
    import wandb
    os.system(f'wandb login {wandb_api_key}')
    run = wandb.init(settings=wandb.Settings(start_method="fork"))
    artifact = run.use_artifact(wandb_artifact_path, type='model')
    artifact_dir = artifact.download(root_path)




# Main function to run the app
def run_app():
    test_and_limit_tensorflow()

    if not os.path.exists(model_path):
        download_model(model_link)

    import time
    t1 = time.time()
    model = load_model(model_path)
    os.makedirs(lagophthalmos_output_folder, exist_ok=True)

    st.title("Lagophthalmos Classification Portal")

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

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        placeholder.empty()
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner('Processing the image...'):
            prediction = run_inference(model, image)

        st.write(f"Status: **{prediction}**")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{lagophthalmos_output_folder}uploaded_image_{timestamp}.png"
        image.save(image_filename)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<footer><p>Developed by <strong>Dr Maram Alnefaie - Ophthalmologist</strong></p></footer>",
                    unsafe_allow_html=True)
