import numpy as np
import cv2
import utitilies.gaze
from scipy.spatial.transform import Rotation as R


def preprocess_eye_image(image, json_data):
    output_width = 160
    output_height = 96

    # Prepring to segment the eye image
    input_height, input_width = image.shape[:2]
    input_height_2, input_width_2 = input_height / 2.0, input_width / 2.0

    heatmap_weight = int(output_width / 2)
    heatmap_height = int(output_height / 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def process_coords(coords_list):
        coords = [eval(l) for l in coords_list]
        return np.array([x, input_height-y, z] for (x, y, z) in coords)

    interior_landmarks = process_coords(json_data['interior_margin_2d'])
    caruncle_landmarks = process_coords(json_data['caruncle_2d'])
    iris_landmarks = process_coords(json_data['iris_2d'])

    left_corner = np.mean(caruncle_landmarks[:, :2], axis = 0)
    right_corner = interior_landmarks[8, :2]
    eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
    eye_middle = np.mean([np.amin(interior_landmarks[:, :2], axis = 0)], axis = 0)

    #Normalizing the eye width

    scale = output_width/eye_width

    translate = np.asmatrix(np.eye(3))
    translate[0, 2] = -eye_middle[0] * scale
    translate[1, 2] = -eye_middle[1] * scale

    rand_x = np.random.uniform(low=-10, high=10)
    rand_y = np.random.uniform(low=-10, high=10)

    recenter = np.asmatrix(np.eye(3))
    recenter[0, 2] = output_width/2 + rand_x
    recenter[1, 2] = output_height/2 + rand_y

    scale_mat = np.asmatrix(np.eye(3))
    scale_mat[0, 0] = scale
    scale_mat[1, 1] = scale

    angle = 0
    rotation = R.from_rotvec([0, 0, angle]).as_matrix()

    transform = recenter * rotation * translate * scale_mat
    transform_inv = np.linalg.inv(transform)

    #Apply transforms

    eye = cv2.warpAffine(image, transform[:2], (output_width, output_height))

    rand_blur = np.random.uniform(low = 0, high=20)
    eye = cv2.GaussianBlur(eye, {5,5}, rand_blur)

    #Normalize eye image
    eye = cv2.equalizeHist(eye)
    eye = eye.astype(np.float32)
    eye = eye / 255.0

    #Gaze; convert look vector to gaze direction in polar angles
    look_vec = np.array(eval(json_data['eye_details']['look_vec']))[:3].reshape((1, 3))

    gaze = utilities.gaze.vector_to_pitchyaw(-look_vec).flatten()
    gaze = gaze.astype(np.float32)

    iris_center = np.mean(iris_landmarks[:, :2], axis = 0)
    landmarks = np.concatenate([interior_landmarks[:, :2],
                                iris_landmarks[::2, :2],
                                iris_center.reshape((1, 2)),
                                [[input_width_2, input_height_2]]])
    landmarks = np.asmatrix(np.pad(landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1))
    landmarks = np.asarray(landmarks * transform[:2].T)*np.array([heatmap_weight/output_width, heatmap_height/output_height])
    landmarks = landmarks.astype(np.float32)

    #swap columns so that landmarks are in (y, x) and not (x, y) this is because the network outputs landmarks as (y, x values

    temp = np.zeros((34, 2), dtype=np.float32)
    temp[:, 0] = landmarks[:, 1]
    temp[:, 1] = landmarks[:, 0]
    landmarks = temp

    heatmaps = get_heatmaps(w=heatmap_weight, h=heatmap_height, landmarks = landmarks)
    assert heatmaps.shape == (34, heatmap_height, heatmap_weight)

    return {
        'img': eye,
        'transform': np.asarray(transform_inv),
        'transform_inv': np.asarray(transform_inv)
        'eye_middle': np.asarray(eye_middle),
        'heatmaps': np.asarray(heatmaps),
        'landmarks': np.asarray(landmarks),
        'gaze': np.asarray(gaze)
    }

def gaussian_2d(width, height, cx, cy, sigma=1.0):
    """
    Function generates heatmap with a Single 2D guassian
    :param width: image width
    :param height: image height
    :param cx:
    :param cy:
    :param sigma:
    :return:
    """
    xs, ys = np.meshgrid(
        np.linspace(0, width-1, width, dtype = np.float32),
        np.linspace(0, height-1, height, dtype=np.float32)
    )
    assert xs.shape == (height, width)
    alpha = -0.5 /(sigma ** 2)
    heatmap = np.exp(alpha * ((xs - cx) ** 2 + (ys - cy) **2))
    return heatmap

def get_heatmaps(width, height, landmarks):
    heatmaps = []
    for (y, x) in landmarks:
        heatmaps.append(gaussian_2d(width, height, cx = x, cy = y, sigma=2.0))
    return np.array(heatmaps)