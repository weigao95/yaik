import sympy as sp
from typing import List, Optional


def inverse_transform(T: sp.Matrix) -> sp.Matrix:
    """
    Compute the inverse of a 4x4 homogeneous transformation matrix
    :param T: the input 4x4 matrix
    :return:
    """
    rot = T[0:3, 0:3].T
    translation = -rot * T[0:3, 3]
    invT = rot.row_join(translation)
    invT = invT.col_join(sp.Matrix([0, 0, 0, 1]).T)
    return invT


def inverse_rotation(rotation_mat: sp.Matrix) -> sp.Matrix:
    """
    Compute the inverse of a rotation matrix
    :param rotation_mat:
    :return:
    """
    return rotation_mat.T


def rotation_z(theta: sp.Symbol) -> sp.Matrix:
    """
    4x4 rotation matrix in z
    :param theta: rotation angle
    :return: 4x4 rotation matrix
    """
    t = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta), 0, 0],
        [sp.sin(theta),  sp.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return t


def rotation_x(theta: sp.Symbol) -> sp.Matrix:
    """
    4x4 rotation matrix in x
    :param theta: rotation angle
    :return: 4x4 rotation matrix
    """
    t = sp.Matrix([
        [1, 0, 0, 0],
        [0, sp.cos(theta), -sp.sin(theta), 0],
        [0, sp.sin(theta),  sp.cos(theta), 0],
        [0, 0, 0, 1]
    ])
    return t


def translation_z(d: sp.Symbol) -> sp.Matrix:
    """
    4x4 translation matrix in z
    :param d: translation distance
    :return: 4x4 translation matrix
    """
    t = sp.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]
    ])
    return t


def translation_x(a: sp.Symbol) -> sp.Matrix:
    """
    4x4 translation matrix in x
    :param a: translation distance
    :return: 4x4 translation matrix
    """
    t = sp.Matrix([
        [1, 0, 0, a],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return t


def is_identity_matrix(matrix: sp.Matrix) -> bool:
    """
    Determine wheter a matrix is a identity matrix
    :param matrix:
    :return:
    """
    if matrix.rows != matrix.cols:
        return False
    for i in range(matrix.rows):
        if matrix[i, i] != sp.S.One:
            return False
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            if i == j:
                continue
            if matrix[i, j] != sp.S.Zero:
                return False
    return True


def multiple_list_of_transforms(transform_list: List[Optional[sp.Matrix]]) -> sp.Matrix:
    """
    Given a list of transforms, where None represents identify, multiple them together
    to get the final transform.
    :param transform_list:
    :return:
    """
    result = sp.eye(4)
    for i in range(len(transform_list)):
        if transform_list[i] is not None:
            result = result * transform_list[i]
    return result
