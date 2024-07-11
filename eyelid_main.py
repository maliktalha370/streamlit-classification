import cv2
import dlib, os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# Function to detect eyelid closure
def detect_eyelid_closure(image_name):
    image = cv2.imread(image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        height, width, _ = image.shape
        top = 0
        left = 0
        bottom = width
        right = height
        faces = [dlib.rectangle(top, left, bottom, right)]
    for face in faces:
        landmarks = predictor(gray, face)

        # Extract eyelid landmarks
        left_eye_top = landmarks.part(37).y
        left_eye_bottom = landmarks.part(41).y
        right_eye_top = landmarks.part(44).y
        right_eye_bottom = landmarks.part(46).y

        left_eye_open = left_eye_bottom - left_eye_top
        right_eye_open = right_eye_bottom - right_eye_top

        image_cp = cv2.circle(image, (landmarks.part(37).x,landmarks.part(37).y), radius=0, color=(255, 255, 255), thickness=5)
        image_cp = cv2.circle(image_cp, (landmarks.part(41).x,landmarks.part(41).y), radius=0, color=(255, 255, 255), thickness=5)
        image_cp = cv2.circle(image_cp, (landmarks.part(44).x, landmarks.part(44).y), radius=0, color=(255, 255, 255),
                              thickness=5)
        image_cp = cv2.circle(image_cp, (landmarks.part(46).x, landmarks.part(46).y), radius=0, color=(255, 255, 255),
                              thickness=5)



    # cv2.imshow('draw image', image_cp)
    # cv2.waitKey(0)
    extracted_name = image_name.split('/')[-1]
    cv2.imwrite(f'Humans_drawn/{extracted_name}', image_cp)

# detect_eyelid_closure('IMG_4605.jpeg')
data_dir = 'Humans'
for image in tqdm(os.listdir(data_dir)):
    detect_eyelid_closure(data_dir + '/'  + image)

