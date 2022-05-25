import cv2 as cv
import numpy as np


def pitchyaw_to_vector(pitchyaws):
    """Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.
        Args:
            pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.
        Returns:
            :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
        """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def vector_to_pitchyaw(vectors):
    """
    Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.
    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.
    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    :param vectors: vectors
    :return: angular
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def radians_to_degree():
    return 180.0 / np.pi


def angular_error(a, b):
    """
    Calculates angular vector error (via cosine similarity)
    :param a:
    :param b:
    :return:
    """
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zeros-values
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiple(a_norm, b_norm))
    return np.arccos(similarity) * radians_to_degree()


def mean_angular_error(a, b):
    """
    Calculatin for mean angular error using cosine similarity
    :param a:
    :param b:
    :return:
    """
    return np.mean(angular_error(a, b))

def draw_gaze(image_input, eye_position, pitchyaw, length = 40.0, thickness = 2, color=(0, 0, 255)):
    """
    Draws gaze angle on given image with a given eye position.
    :param image_input:
    :param eye_position:
    :param pitchyaw:
    :param length:
    :param thickness:
    :param color:
    :return:
    """
    image_output = image_input
    if len(image_output.shape) == 2 or image_output.shape[2] ==1:
        image_output = cv.cvtColor(image_output, cv.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = length * np.sin(pitchyaw[0])
    cv.arrowedLine(image_output, tuple(np.round(eye_position). astype(np.int32)),
                   tuple(np.round([eye_position[0] + dx, eye_position[1] + dy]).astype(int)), color,
                   thickness, cv.LINE_AA, tipLength=0.2)
    return image_output
