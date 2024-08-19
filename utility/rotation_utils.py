import numpy as np
import utility.transformations as transformations


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a 4x4 homogeneous transformation matrix
    :param transform: the input 4x4 matrix
    :return:
    """
    rot = transform[0:3, 0:3].T
    translation = -rot.dot(transform[0:3, 3])
    inverted = np.eye(4)
    inverted[0:3, 0:3] = rot
    inverted[0:3, 3] = translation
    return inverted


def angle_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    return transformations.rotation_matrix(angle, axis)


def rotation_z(theta: float) -> np.ndarray:
    """
        4x4 rotation matrix in z
        :param theta: rotation angle
        :return: 4x4 rotation matrix
        """
    t = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return t


def rotation_x(theta: float) -> np.ndarray:
    """
    4x4 rotation matrix in x
    :param theta: rotation angle
    :return: 4x4 rotation matrix
    """
    t = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta),  np.cos(theta), 0],
        [0, 0, 0, 1]
    ])
    return t


def rotation_y(theta: float) -> np.ndarray:
    """
    4x4 rotation matrix in x
    :param theta: rotation angle
    :return: 4x4 rotation matrix
    """
    t = np.array([
        [np.cos(theta),   0, np.sin(theta),  0],
        [0,               1, 0,              0],
        [- np.sin(theta), 0, np.cos(theta),  0],
        [0, 0, 0, 1]
    ])
    return t


def mmind_rpy(rpy: np.ndarray) -> np.ndarray:
    assert rpy.size == 3
    r = rpy[0]
    r = rotation_x(r)
    p = rpy[1]
    p = rotation_y(p)
    y = rpy[2]
    y = rotation_z(y)
    return r.dot(p.dot(y))


def mmind_transform(rpy: np.ndarray, xyz: np.ndarray):
    transform_matrix = mmind_rpy(rpy)
    transform_matrix[0, 3] = xyz[0]
    transform_matrix[1, 3] = xyz[1]
    transform_matrix[2, 3] = xyz[2]
    return transform_matrix


def translation_z(d: float) -> np.ndarray:
    """
    4x4 translation matrix in z
    :param d: translation distance
    :return: 4x4 translation matrix
    """
    t = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]
    ])
    return t


def translation_x(a: float) -> np.ndarray:
    """
    4x4 translation matrix in x
    :param a: translation distance
    :return: 4x4 translation matrix
    """
    t = np.array([
        [1, 0, 0, a],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return t


def twist_movement(transform_to_update: np.ndarray, twist: np.ndarray) -> np.ndarray:
    """
    Suppose the frame located at :param transform are moving with velocity/displacement twist,
    Return the new transform after movement.
    :param transform_to_update:
    :param twist: Rotation first twist as 6-vector
    :return:
    """
    assert transform_to_update.shape[0] == 4
    assert transform_to_update.shape[1] == 4
    assert twist.size == 6
    alpha, beta, gamma = twist[0], twist[1], twist[2]
    a, b, c = twist[3], twist[4], twist[5]

    # The rotation
    twist_rotation = np.array([alpha, beta, gamma])
    angle = np.linalg.norm(twist_rotation)
    if abs(angle) < 1e-10:
        rotation_mat = np.eye(4)
    else:
        axis = twist_rotation / angle
        rotation_mat = transformations.rotation_matrix(angle, axis)

    delta_transform = np.array([[1.,     -gamma, beta,   a],
                                [gamma, 1.,      -alpha, b],
                                [-beta, alpha,  1.,      c],
                                [0,     0,      0,      1]])
    delta_transform[0:3, 0:3] = rotation_mat[0:3, 0:3]
    new_transform = delta_transform.dot(transform_to_update)
    return new_transform
