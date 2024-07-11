import efficientnet.keras as efn
import numpy as np
import os

from constants import *
from PIL import Image
from tensorflow.keras.models import load_model

def test_tensorflow():
    import tensorflow as tf
    print('GPU Available ', tf.test.is_gpu_available())
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
def download_model(model_link):
    os.system(f'gdown {model_link}')

test_tensorflow()
if not os.path.exists(model_path):
    download_model(model_link)

# Load the model
model = load_model(model_path)
uploaded_file = 'demo/IMG_3085.JPG'
# Display uploaded image
image = Image.open(uploaded_file)
# Classify image
prediction = run_inference(model, image)
print('Prediction ', prediction)
