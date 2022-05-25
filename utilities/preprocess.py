import numpy as np
import cv2
import utitilies.gaze
from scipy.spatial.transform import Rotation as R

def preprocess_eye_image(image, json_data):
    output_width = 160
    output_height = 96

    #