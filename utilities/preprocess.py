import numpy as np
import cv2
import utitilies.gaze
from scipy.spatial.transform import Rotation as R

def preprocess_eye_image(image, json_data):
    output_width = 160
    output_height = 96

    #Prepring to segment the eye image
    input_height, input_weight = image.shape[:2]
    input_height, input_weight = input_height/2.0, input_weight/2.0

    heatmap_weight = int(output_width/2)
    heatmap_height = int(output_height/2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)