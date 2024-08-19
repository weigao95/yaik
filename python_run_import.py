import numpy as np
import math
from typing import Tuple, Optional, List


# General utility
def safe_sqrt(a):
    return math.sqrt(a) if a >= 0 else 0


def safe_asin(a):
    if a > 1:
        return 0.5 * math.pi
    elif a < -1:
        return -0.5 * math.pi
    else:
        return math.asin(a)


def safe_acos(a):
    if a > 1:
        return 0.0
    elif a < -1:
        return math.pi
    else:
        return math.acos(a)


# For general 6-dof
def compute_solution_from_tanhalf_LME(
        A_x2: np.ndarray,
        B_x: np.ndarray,
        C: np.ndarray,
        return_coupled_3_solution: bool = False):
    """
    The solver in Equation 15 of Efficient Inverse Kinematics for General 6R Manipulators.
    """
    # To 12x12 matrix
    A_12x12 = np.zeros(shape=(12, 12))
    A_12x12[0:6, 0:9] = A_x2
    A_12x12[6:12, 3:12] = A_x2
    A_inv = None
    try:
        A_inv = np.linalg.inv(A_12x12)
    except:
        # print('Cannot inverse the matrix A', print(np.linalg.det(A_12x12)))
        return None
    assert A_inv is not None
    B_12x12 = np.zeros(shape=(12, 12))
    B_12x12[0:6, 0:9] = B_x
    B_12x12[6:12, 3:12] = B_x
    C_12x12 = np.zeros(shape=(12, 12))
    C_12x12[0:6, 0:9] = C
    C_12x12[6:12, 3:12] = C

    M = np.zeros(shape=(24, 24))
    M[0:12, 12:24] = np.eye(12)
    M[12:24, 0:12] = - A_inv.dot(C_12x12)
    M[12:24, 12:24] = - A_inv.dot(B_12x12)

    # The case only need one solution
    if not return_coupled_3_solution:
        eigen_values = np.linalg.eigvals(M)
        imag_part = np.imag(eigen_values)
        real_part = np.real(eigen_values)
        solution: List[float] = list()
        for i in range(eigen_values.size):
            if abs(imag_part[i]) < 1e-6:
                atan_value = np.arctan(real_part[i])
                solution.append(2 * atan_value)
        return solution

    # We need all three coupled solution
    solution: List[Tuple[float]] = list()
    eigen_values, eigen_vectors = np.linalg.eig(M)
    for i in range(eigen_values.size):
        if abs(np.imag(eigen_values[i])) > 1e-6:
            continue
        tanhalf_x3 = np.real(eigen_values[i])
        theta3_value = 2.0 * np.arctan(tanhalf_x3)

        # The remaining part
        full_ev_i = eigen_vectors[:, i]
        ev_i = full_ev_i[0:12]
        tanhalf_x4 = ev_i[3] / ev_i[0]
        tanhalf_x5 = ev_i[1] / ev_i[0]
        if np.imag(tanhalf_x4) > 1e-6 or np.imag(tanhalf_x5) > 1e-6:
            continue
        theta4_value = 2.0 * np.arctan(np.real(tanhalf_x4))
        theta5_value = 2.0 * np.arctan(np.real(tanhalf_x5))
        solution.append((float(theta3_value), float(theta4_value), float(theta5_value)))
    return solution


# For linear solver type2
def try_solve_linear_type2(A: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Solve y from three linear equations in the form
    A sin(x) + B cos(x) + C sin(y) + D cos(y) == 0
    Each row of matrix corresponds to A, B, C, D
    """
    assert A.shape[0] == 3
    assert A.shape[1] == 4
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    ad_minus_bc = a * d - b * c
    zero_tolerance = 1e-10
    if abs(ad_minus_bc) < zero_tolerance:
        return None
    A_inv = np.zeros(shape=(2, 2))
    A_inv[0, 0] = d
    A_inv[0, 1] = -b
    A_inv[1, 0] = -c
    A_inv[1, 1] = a
    A_inv /= ad_minus_bc
    # [sin(x), cos(x)].T = B * [sin(y), cos(y)].T
    B = - A_inv.dot(A[0:2, 2:4])

    # e sin(y) + f * cos(y) == 0
    e = A[2, 2] + A[2, 0] * B[0, 0] + A[2, 1] * B[1, 0]
    f = A[2, 3] + A[2, 0] * B[0, 1] + A[2, 1] * B[1, 1]
    if abs(e) < zero_tolerance and abs(f) < zero_tolerance:
        return None
    solution_0 = math.atan2(- f, e)
    solution_1 = solution_0 + np.pi
    if solution_1 > np.pi:
        solution_1 -= 2 * np.pi
    return solution_0, solution_1


def try_solve_linear_type2_specific_rows(
        A_combined: np.ndarray, row_0: int, row_1: int, row_2: int) -> Optional[Tuple[float, float]]:
    """
    For a m x 4 matrix A, each row of whom corresponds to a linear equation in the form of
    A sin(x) + B cos(x) + C sin(y) + D cos(y) == 0
    Select three equations from row_0/1/2 and solve it
    """
    A_tmp = np.zeros(shape=(3, 4))
    A_tmp[0, :] = A_combined[row_0, :]
    A_tmp[1, :] = A_combined[row_1, :]
    A_tmp[2, :] = A_combined[row_2, :]
    A_tmp_sol = try_solve_linear_type2(A_tmp)
    if A_tmp_sol is not None:
        return A_tmp_sol
    return None
