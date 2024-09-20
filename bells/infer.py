import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models import SAM
import logging
import argparse
import os
from datetime import datetime
import streamlit as st
from utils import test_and_limit_tensorflow, download_model
from constants import bells_output_folder, YOLO_MODEL_PATH, YOLO_MODEL_LINK, SAM_MODEL_PATH, \
    SAM_MODEL_LINK

# Configure logging
def setup_logging(debug_mode):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=level)


def detect_bells(detection, image, segment):
    trueEye = False
    lower_hsv = np.array([0, 0, 107])
    upper_hsv = np.array([179, 95, 245])
    big_box = []
    for i, r in enumerate(detection.copy()):
        boxes = r.boxes
        if (np.array(boxes.cls.cpu())[0] == 0 and np.array(boxes.conf.cpu())[0] > 0.8):
            logging.debug('TRUE EYE-----')
            logging.debug(f"Confidence: {np.array(boxes.conf)[0]}")
            trueEye = True

        for i, b in enumerate(boxes.xyxy):
            big_box.append(b)
    big_box = [tensor.cpu().numpy() for tensor in big_box]

    boxes_np = np.stack(big_box)
    x_min = np.min(boxes_np[:, 0])
    y_min = np.min(boxes_np[:, 1])
    x_max = np.max(boxes_np[:, 2])
    y_max = np.max(boxes_np[:, 3])

    combined_box = [int(x_min), int(y_min), int(x_max), int(y_max)]

    logging.debug(f"Combined bounding box: {combined_box}")

    b = combined_box
    c = boxes.cls
    logging.debug(f"Classes: {c}")
    if True:  # c in [0,1,2,3]:
        masks = segment(image, bboxes=np.array(b))

        for m in masks:
            logging.debug(f"masks {m.masks.data[0]}")
            mask_org = np.array(m.masks.data[0].cpu() * 1)  # Take the first mask, shape is now (h, w)
            mask_org = mask_org.astype('uint8')

            coords = [int(f) for f in b]
            left, top, right, bottom = coords
            mask = mask_org[top:bottom, left:right]

            kernel = np.ones((15, 15), np.uint8)  # Kernel for morphological operations
            dilated = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            original_image = image[top:bottom, left:right]
            filled_mask = np.zeros_like(original_image)
            cv2.fillPoly(filled_mask, contours, (255, 255, 255))

            masked_image = cv2.bitwise_and(original_image, filled_mask)
            masked_image = cv2.bilateralFilter(masked_image, d=25, sigmaColor=75, sigmaSpace=75)
            hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
            mask_hsv = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

            sam_mask_cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largest_area = max(cv2.contourArea(contour) for contour in sam_mask_cont)
            logging.debug(f"The largest contour has an area of {largest_area} pixels")

            ratio = 0
            if sam_mask_cont:
                largest_contour = max(sam_mask_cont, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                white_pixels, black_pixels, total_pixels = 0, 0, 0
                for i in range(x, x + w):
                    for j in range(y, y + h):
                        total_pixels += 1
                        if cv2.pointPolygonTest(largest_contour, (i, j), False) >= 0:
                            if mask_hsv[j, i] == 255:  # White pixel
                                white_pixels += 1
                            else:  # Black pixel
                                black_pixels += 1

                if black_pixels > 0:
                    ratio = white_pixels / total_pixels
                else:
                    ratio = float('inf')  # Avoid division by zero

                logging.debug(f"White pixels: {white_pixels}")
                logging.debug(f"Black pixels: {black_pixels}")
                logging.debug(f"Ratio of white to black pixels: {ratio:.2f}")

            ratio = ratio - .50 if trueEye else ratio + 0.10
            logging.debug(f"Adjusted ratio: {ratio:.2f}")

            if not np.isinf(ratio) and ratio > .32:
                logging.info('Good Bells found....')
                return True
            else:
                logging.info('Bad bells found....')
                return False


def main(testset_dir, debug_mode=False):
    setup_logging(debug_mode)

    accuracy_list = 0
    images_list = [f for f in os.listdir(testset_dir) if f.endswith('jpeg')]

    for i in images_list:
        logging.info(f"Processing image: {i}")
        image_path = os.path.join(testset_dir, i)
        image = cv2.imread(image_path)

        detection = model(image, verbose=False)
        if detect_bells(detection=detection, image=image):
            accuracy_list += 1

    logging.info(f"Number of detected bells: {accuracy_list}")

# Main function to run the app
def run_app():
    # Initialize models
    test_and_limit_tensorflow()
    setup_logging(debug_mode=False)

    if not os.path.exists(YOLO_MODEL_PATH):
        logging.info(f"Downloading YOLO Model")
        download_model(YOLO_MODEL_LINK, file_path = YOLO_MODEL_PATH)

    if not os.path.exists(SAM_MODEL_PATH):
        logging.info(f"Downloading SAM Model")
        download_model(SAM_MODEL_LINK, file_path = SAM_MODEL_PATH)

    model = YOLO(YOLO_MODEL_PATH)
    segment = SAM(SAM_MODEL_PATH)

    os.makedirs(bells_output_folder, exist_ok=True)

    st.title("Bells Detection Portal")

    placeholder = st.empty()

    with placeholder.container():
        st.write("""
            This portal allows you to upload images and checks whether person is suffereing from Good / Bad Bells.
        """)

        st.header("Image Upload Guidelines")
        st.write("""
            Please ensure your images meet the following requirements for optimal classification:
            - Format: JPEG, JPG, PNG
            - Close-up shot of the Person's eye in well-lit environment
        """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        placeholder.empty()
        # Convert the uploaded file to a byte stream and then to a NumPy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        # Decode the image as a NumPy array that OpenCV can process
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        with st.spinner('Processing the image...'):
            detection = model(image, verbose=False)
            if detect_bells(detection=detection, image=image, segment=segment):
                st.write(f"Status: **Bad Bells Found !**")
            else:
                st.write(f"Status: **Good Bells Found !**")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{bells_output_folder}uploaded_image_{timestamp}.png"
        cv2.imwrite(image_filename, image )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<footer><p>Developed by <strong>Dr Maram Alnefaie - Ophthalmologist</strong></p></footer>",
                    unsafe_allow_html=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images in the testset directory.')
    parser.add_argument('--folder', type=str, required=True, help='Path to the testset directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for verbose output')

    args = parser.parse_args()

    main(args.folder, debug_mode=args.debug)
