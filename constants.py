# Lagophthalmos
wandb_artifact_path = 'malitkalha370/Maram-Lagophthalmos/Models:v0'
# wandb_api_key = 'c8b9a1d4f827ecb145bb2bee882475b5bd053cca'
model_path = './lagophthalmos/keras_model.h5'
model_link = '1-Crx8B9eOnM6DSmYBHPOBFVpQO7qCufd'

lagophthalmos_output_folder= './lagophthalmos/inferenced_images/'
bells_output_folder= './bells/inferenced_images/'

# Display example images with captions
example_images = [
    "./lagophthalmos/demo/IMG_3084.jpeg",
    "./lagophthalmos/demo/IMG_3085.JPG",
    "./lagophthalmos/demo/IMG_3086.jpeg",
]
gpu_limit = 6144


# Bells
YOLO_MODEL_LINK = '1JyN-C58ieUXQu3djf8A2MLDWDOwvNJFU'
SAM_MODEL_LINK = '1mB3S47w_qIzB5z4x8SEXsORgSPgXqBRc'

YOLO_MODEL_PATH = './bells/models/locator.pt'
SAM_MODEL_PATH =  './bells/models/sam_b.pt'

bells_output_folder= './bells/inferenced_images/'
