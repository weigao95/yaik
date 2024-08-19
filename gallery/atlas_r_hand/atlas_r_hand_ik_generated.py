import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 7
n_tree_nodes: int = 22
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_1: float = 0.11
a_2: float = 0.016
a_3: float = 0.0092
a_4: float = 0.00921
d_2: float = -0.306
d_4: float = -0.29955
pre_transform_s0: float = 0.1406
pre_transform_s1: float = -0.2256
pre_transform_s2: float = 0.2326

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = 1.5707963267948966
th_2_offset_original2raw: float = 3.141592653589793
th_3_offset_original2raw: float = 3.141592653589793
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 3.141592653589793
th_6_offset_original2raw: float = 1.5707963267948966


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def atlas_r_hand_ik_target_original_to_raw(T_ee: np.ndarray):
    r_11: float = T_ee[0, 0]
    r_12: float = T_ee[0, 1]
    r_13: float = T_ee[0, 2]
    Px: float = T_ee[0, 3]
    r_21: float = T_ee[1, 0]
    r_22: float = T_ee[1, 1]
    r_23: float = T_ee[1, 2]
    Py: float = T_ee[1, 3]
    r_31: float = T_ee[2, 0]
    r_32: float = T_ee[2, 1]
    r_33: float = T_ee[2, 2]
    Pz: float = T_ee[2, 3]
    ee_transformed = np.eye(4)
    ee_transformed[0, 0] = r_21
    ee_transformed[0, 1] = -r_23
    ee_transformed[0, 2] = -r_22
    ee_transformed[0, 3] = -Py + pre_transform_s1
    ee_transformed[1, 0] = -r_11
    ee_transformed[1, 1] = r_13
    ee_transformed[1, 2] = r_12
    ee_transformed[1, 3] = Px - pre_transform_s0
    ee_transformed[2, 0] = -r_31
    ee_transformed[2, 1] = r_33
    ee_transformed[2, 2] = r_32
    ee_transformed[2, 3] = Pz - pre_transform_s2
    return ee_transformed


def atlas_r_hand_ik_target_raw_to_original(T_ee: np.ndarray):
    r_11: float = T_ee[0, 0]
    r_12: float = T_ee[0, 1]
    r_13: float = T_ee[0, 2]
    Px: float = T_ee[0, 3]
    r_21: float = T_ee[1, 0]
    r_22: float = T_ee[1, 1]
    r_23: float = T_ee[1, 2]
    Py: float = T_ee[1, 3]
    r_31: float = T_ee[2, 0]
    r_32: float = T_ee[2, 1]
    r_33: float = T_ee[2, 2]
    Pz: float = T_ee[2, 3]
    ee_transformed = np.eye(4)
    ee_transformed[0, 0] = -r_21
    ee_transformed[0, 1] = r_23
    ee_transformed[0, 2] = r_22
    ee_transformed[0, 3] = Py + pre_transform_s0
    ee_transformed[1, 0] = r_11
    ee_transformed[1, 1] = -r_13
    ee_transformed[1, 2] = -r_12
    ee_transformed[1, 3] = -Px + pre_transform_s1
    ee_transformed[2, 0] = -r_31
    ee_transformed[2, 1] = r_33
    ee_transformed[2, 2] = r_32
    ee_transformed[2, 3] = Pz + pre_transform_s2
    return ee_transformed


def atlas_r_hand_fk(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_6)
    x1 = math.cos(th_4)
    x2 = math.cos(th_0)
    x3 = math.cos(th_2)
    x4 = x2*x3
    x5 = math.cos(th_1)
    x6 = math.sin(th_0)
    x7 = math.sin(th_2)
    x8 = x6*x7
    x9 = -x4 - x5*x8
    x10 = math.sin(th_4)
    x11 = math.sin(th_3)
    x12 = math.sin(th_1)
    x13 = x12*x6
    x14 = math.cos(th_3)
    x15 = x2*x7
    x16 = x3*x6
    x17 = -x15 + x16*x5
    x18 = x11*x13 + x14*x17
    x19 = -x1*x9 - x10*x18
    x20 = math.cos(th_6)
    x21 = math.sin(th_5)
    x22 = -x11*x17 + x13*x14
    x23 = math.cos(th_5)
    x24 = x1*x18 - x10*x9
    x25 = -x21*x22 + x23*x24
    x26 = a_2*x5
    x27 = -x15*x5 + x16
    x28 = x12*x2
    x29 = x4*x5 + x8
    x30 = x11*x28 + x14*x29
    x31 = -x1*x27 - x10*x30
    x32 = -x11*x29 + x14*x28
    x33 = x1*x30 - x10*x27
    x34 = -x21*x32 + x23*x33
    x35 = x12*x7
    x36 = x12*x3
    x37 = x11*x5 - x14*x36
    x38 = -x1*x35 - x10*x37
    x39 = x11*x36 + x14*x5
    x40 = x1*x37 - x10*x35
    x41 = -x21*x39 + x23*x40
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = x0*x19 - x20*x25
    ee_pose[0, 1] = -x21*x24 - x22*x23
    ee_pose[0, 2] = -x0*x25 - x19*x20
    ee_pose[0, 3] = a_1*x6 + a_3*x17 + a_4*x18 - d_2*x13 + d_4*x22 + pre_transform_s0 + x26*x6
    ee_pose[1, 0] = -x0*x31 + x20*x34
    ee_pose[1, 1] = x21*x33 + x23*x32
    ee_pose[1, 2] = x0*x34 + x20*x31
    ee_pose[1, 3] = -a_1*x2 - a_3*x29 - a_4*x30 + d_2*x28 - d_4*x32 + pre_transform_s1 - x2*x26
    ee_pose[2, 0] = x0*x38 - x20*x41
    ee_pose[2, 1] = -x21*x40 - x23*x39
    ee_pose[2, 2] = -x0*x41 - x20*x38
    ee_pose[2, 3] = -a_2*x12 - a_3*x36 + a_4*x37 - d_2*x5 + d_4*x39 + pre_transform_s2
    return ee_pose


def atlas_r_hand_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.cos(th_0)
    x1 = math.sin(th_0)
    x2 = math.sin(th_1)
    x3 = x1*x2
    x4 = math.cos(th_2)
    x5 = x0*x4
    x6 = math.cos(th_1)
    x7 = math.sin(th_2)
    x8 = x1*x7
    x9 = -x5 - x6*x8
    x10 = math.cos(th_3)
    x11 = math.sin(th_3)
    x12 = x0*x7
    x13 = x1*x4
    x14 = -x12 + x13*x6
    x15 = x10*x3 - x11*x14
    x16 = math.cos(th_4)
    x17 = math.sin(th_4)
    x18 = x10*x14 + x11*x3
    x19 = -x16*x9 - x17*x18
    x20 = math.cos(th_5)
    x21 = math.sin(th_5)
    x22 = -x15*x20 - x21*(x16*x18 - x17*x9)
    x23 = x0*x2
    x24 = x12*x6 - x13
    x25 = -x5*x6 - x8
    x26 = -x10*x23 - x11*x25
    x27 = x10*x25 - x11*x23
    x28 = -x16*x24 - x17*x27
    x29 = -x20*x26 - x21*(x16*x27 - x17*x24)
    x30 = x2*x7
    x31 = x2*x4
    x32 = x10*x6 + x11*x31
    x33 = -x10*x31 + x11*x6
    x34 = -x16*x30 - x17*x33
    x35 = -x20*x32 - x21*(x16*x33 - x17*x30)
    x36 = -a_2*x2 - d_2*x6 + pre_transform_s2
    x37 = a_2*x6
    x38 = -a_1*x0 + pre_transform_s1
    x39 = d_2*x23 - x0*x37 + x38
    x40 = -a_3*x31 + x36
    x41 = a_3*x25 + x39
    x42 = a_4*x33 + d_4*x32 + x40
    x43 = a_4*x27 + d_4*x26 + x41
    x44 = a_1*x1 + pre_transform_s0
    x45 = -d_2*x3 + x1*x37 + x44
    x46 = a_3*x14 + x45
    x47 = a_4*x18 + d_4*x15 + x46
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 7))
    jacobian_output[0, 1] = x0
    jacobian_output[0, 2] = -x3
    jacobian_output[0, 3] = x9
    jacobian_output[0, 4] = x15
    jacobian_output[0, 5] = x19
    jacobian_output[0, 6] = x22
    jacobian_output[1, 1] = x1
    jacobian_output[1, 2] = x23
    jacobian_output[1, 3] = x24
    jacobian_output[1, 4] = x26
    jacobian_output[1, 5] = x28
    jacobian_output[1, 6] = x29
    jacobian_output[2, 0] = 1
    jacobian_output[2, 2] = -x6
    jacobian_output[2, 3] = x30
    jacobian_output[2, 4] = x32
    jacobian_output[2, 5] = x34
    jacobian_output[2, 6] = x35
    jacobian_output[3, 0] = pre_transform_s1
    jacobian_output[3, 1] = -pre_transform_s2*x1
    jacobian_output[3, 2] = -x23*x36 - x39*x6
    jacobian_output[3, 3] = -x24*x40 + x30*x41
    jacobian_output[3, 4] = -x26*x42 + x32*x43
    jacobian_output[3, 5] = -x28*x42 + x34*x43
    jacobian_output[3, 6] = -x29*x42 + x35*x43
    jacobian_output[4, 0] = -pre_transform_s0
    jacobian_output[4, 1] = pre_transform_s2*x0
    jacobian_output[4, 2] = -x3*x36 + x45*x6
    jacobian_output[4, 3] = -x30*x46 + x40*x9
    jacobian_output[4, 4] = x15*x42 - x32*x47
    jacobian_output[4, 5] = x19*x42 - x34*x47
    jacobian_output[4, 6] = x22*x42 - x35*x47
    jacobian_output[5, 1] = -x0*x38 + x1*x44
    jacobian_output[5, 2] = x23*x45 + x3*x39
    jacobian_output[5, 3] = x24*x46 - x41*x9
    jacobian_output[5, 4] = -x15*x43 + x26*x47
    jacobian_output[5, 5] = -x19*x43 + x28*x47
    jacobian_output[5, 6] = -x22*x43 + x29*x47
    return jacobian_output


def atlas_r_hand_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.cos(th_0)
    x1 = math.sin(th_0)
    x2 = math.sin(th_1)
    x3 = x1*x2
    x4 = math.cos(th_2)
    x5 = x0*x4
    x6 = math.cos(th_1)
    x7 = math.sin(th_2)
    x8 = x1*x7
    x9 = -x5 - x6*x8
    x10 = math.cos(th_3)
    x11 = math.sin(th_3)
    x12 = x0*x7
    x13 = x1*x4
    x14 = -x12 + x13*x6
    x15 = x10*x3 - x11*x14
    x16 = math.cos(th_4)
    x17 = math.sin(th_4)
    x18 = x10*x14 + x11*x3
    x19 = math.cos(th_5)
    x20 = math.sin(th_5)
    x21 = x0*x2
    x22 = x12*x6 - x13
    x23 = -x5*x6 - x8
    x24 = -x10*x21 - x11*x23
    x25 = x10*x23 - x11*x21
    x26 = x2*x7
    x27 = x2*x4
    x28 = x10*x6 + x11*x27
    x29 = -x10*x27 + x11*x6
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 1] = x0
    jacobian_output[0, 2] = -x3
    jacobian_output[0, 3] = x9
    jacobian_output[0, 4] = x15
    jacobian_output[0, 5] = -x16*x9 - x17*x18
    jacobian_output[0, 6] = -x15*x19 - x20*(x16*x18 - x17*x9)
    jacobian_output[1, 1] = x1
    jacobian_output[1, 2] = x21
    jacobian_output[1, 3] = x22
    jacobian_output[1, 4] = x24
    jacobian_output[1, 5] = -x16*x22 - x17*x25
    jacobian_output[1, 6] = -x19*x24 - x20*(x16*x25 - x17*x22)
    jacobian_output[2, 0] = 1
    jacobian_output[2, 2] = -x6
    jacobian_output[2, 3] = x26
    jacobian_output[2, 4] = x28
    jacobian_output[2, 5] = -x16*x26 - x17*x29
    jacobian_output[2, 6] = -x19*x28 - x20*(x16*x29 - x17*x26)
    return jacobian_output


def atlas_r_hand_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw
    p_on_ee_x: float = point_on_ee[0]
    p_on_ee_y: float = point_on_ee[1]
    p_on_ee_z: float = point_on_ee[2]

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = p_on_ee_z*x0
    x2 = math.cos(th_1)
    x3 = math.sin(th_1)
    x4 = math.cos(th_0)
    x5 = p_on_ee_z*x4
    x6 = -a_2*x3 - d_2*x2 + pre_transform_s2
    x7 = x3*x4
    x8 = a_2*x2
    x9 = -a_1*x4 + pre_transform_s1
    x10 = d_2*x7 - x4*x8 + x9
    x11 = math.sin(th_2)
    x12 = x11*x3
    x13 = math.cos(th_2)
    x14 = x0*x13
    x15 = x11*x4
    x16 = -x14 + x15*x2
    x17 = x13*x3
    x18 = -a_3*x17 + x6
    x19 = x0*x11
    x20 = x13*x4
    x21 = -x19 - x2*x20
    x22 = a_3*x21 + x10
    x23 = math.cos(th_3)
    x24 = math.sin(th_3)
    x25 = x17*x24 + x2*x23
    x26 = -x21*x24 - x23*x7
    x27 = -x17*x23 + x2*x24
    x28 = a_4*x27 + d_4*x25 + x18
    x29 = x21*x23 - x24*x7
    x30 = a_4*x29 + d_4*x26 + x22
    x31 = math.cos(th_4)
    x32 = math.sin(th_4)
    x33 = -x12*x31 - x27*x32
    x34 = -x16*x31 - x29*x32
    x35 = math.cos(th_5)
    x36 = math.sin(th_5)
    x37 = -x25*x35 - x36*(-x12*x32 + x27*x31)
    x38 = -x26*x35 - x36*(-x16*x32 + x29*x31)
    x39 = x0*x3
    x40 = a_1*x0 + pre_transform_s0
    x41 = -d_2*x39 + x0*x8 + x40
    x42 = -x19*x2 - x20
    x43 = x14*x2 - x15
    x44 = a_3*x43 + x41
    x45 = x23*x39 - x24*x43
    x46 = x23*x43 + x24*x39
    x47 = a_4*x46 + d_4*x45 + x44
    x48 = -x31*x42 - x32*x46
    x49 = -x35*x45 - x36*(x31*x46 - x32*x42)
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 0] = -p_on_ee_y + pre_transform_s1
    jacobian_output[0, 1] = -pre_transform_s2*x0 + x1
    jacobian_output[0, 2] = p_on_ee_y*x2 - x10*x2 + x3*x5 - x6*x7
    jacobian_output[0, 3] = -p_on_ee_y*x12 + p_on_ee_z*x16 + x12*x22 - x16*x18
    jacobian_output[0, 4] = -p_on_ee_y*x25 + p_on_ee_z*x26 + x25*x30 - x26*x28
    jacobian_output[0, 5] = -p_on_ee_y*x33 + p_on_ee_z*x34 - x28*x34 + x30*x33
    jacobian_output[0, 6] = -p_on_ee_y*x37 + p_on_ee_z*x38 - x28*x38 + x30*x37
    jacobian_output[1, 0] = p_on_ee_x - pre_transform_s0
    jacobian_output[1, 1] = pre_transform_s2*x4 - x5
    jacobian_output[1, 2] = -p_on_ee_x*x2 + x1*x3 + x2*x41 - x39*x6
    jacobian_output[1, 3] = p_on_ee_x*x12 - p_on_ee_z*x42 - x12*x44 + x18*x42
    jacobian_output[1, 4] = p_on_ee_x*x25 - p_on_ee_z*x45 - x25*x47 + x28*x45
    jacobian_output[1, 5] = p_on_ee_x*x33 - p_on_ee_z*x48 + x28*x48 - x33*x47
    jacobian_output[1, 6] = p_on_ee_x*x37 - p_on_ee_z*x49 + x28*x49 - x37*x47
    jacobian_output[2, 1] = -p_on_ee_x*x0 + p_on_ee_y*x4 + x0*x40 - x4*x9
    jacobian_output[2, 2] = -p_on_ee_x*x7 - p_on_ee_y*x39 + x10*x39 + x41*x7
    jacobian_output[2, 3] = -p_on_ee_x*x16 + p_on_ee_y*x42 + x16*x44 - x22*x42
    jacobian_output[2, 4] = -p_on_ee_x*x26 + p_on_ee_y*x45 + x26*x47 - x30*x45
    jacobian_output[2, 5] = -p_on_ee_x*x34 + p_on_ee_y*x48 - x30*x48 + x34*x47
    jacobian_output[2, 6] = -p_on_ee_x*x38 + p_on_ee_y*x49 - x30*x49 + x38*x47
    return jacobian_output


def atlas_r_hand_ik_solve_raw(T_ee: np.ndarray, th_0):
    # Extracting the ik target symbols
    r_11 = T_ee[0, 0]
    r_12 = T_ee[0, 1]
    r_13 = T_ee[0, 2]
    Px = T_ee[0, 3]
    r_21 = T_ee[1, 0]
    r_22 = T_ee[1, 1]
    r_23 = T_ee[1, 2]
    Py = T_ee[1, 3]
    r_31 = T_ee[2, 0]
    r_32 = T_ee[2, 1]
    r_33 = T_ee[2, 2]
    Pz = T_ee[2, 3]
    inv_ee_translation = - T_ee[0:3, 0:3].T.dot(T_ee[0:3, 3])
    inv_Px = inv_ee_translation[0]
    inv_Py = inv_ee_translation[1]
    inv_Pz = inv_ee_translation[2]
    
    # A new ik type. Should be a fixed array in C++
    IkSolution = NewType("IkSolution", List[float])
    def make_ik_solution():
        tmp_sol = IkSolution(list())
        for tmp_sol_idx in range(8):
            tmp_sol.append(100000.0)
        return tmp_sol
    
    solution_queue: List[IkSolution] = list()
    queue_element_validity: List[bool] = list()
    def append_solution_to_queue(solution_2_add: IkSolution):
        index_4_appended = len(solution_queue)
        solution_queue.append(solution_2_add)
        queue_element_validity.append(True)
        return index_4_appended
    
    # Init for workspace as empty list. A list of fixed size array for each node
    max_n_solutions: int = 16
    node_input_index: List[List[int]] = list()
    node_input_validity: List[bool] = list()
    for i in range(22):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True
    
    # Code for non-branch dispatcher node 0
    # Actually, there is no code
    
    # Code for explicit solution node 1, solved variable is th_2
    def General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return
        
        # The general 6-dof solution of root node with semi-symbolic reduce
        R_l = np.zeros(shape=(8, 8))
        R_l[0, 4] = -Pz
        R_l[0, 5] = Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2
        R_l[1, 4] = Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2
        R_l[1, 5] = Pz
        R_l[2, 0] = r_31
        R_l[2, 1] = r_32
        R_l[2, 2] = -r_11*math.cos(th_0) - r_21*math.sin(th_0)
        R_l[2, 3] = -r_12*math.cos(th_0) - r_22*math.sin(th_0)
        R_l[3, 0] = -r_11*math.cos(th_0) - r_21*math.sin(th_0)
        R_l[3, 1] = -r_12*math.cos(th_0) - r_22*math.sin(th_0)
        R_l[3, 2] = -r_31
        R_l[3, 3] = -r_32
        R_l[4, 6] = r_11*math.sin(th_0) - r_21*math.cos(th_0)
        R_l[4, 7] = r_12*math.sin(th_0) - r_22*math.cos(th_0)
        R_l[5, 6] = -Pz*r_31 - (-Px*math.sin(th_0) + Py*math.cos(th_0))*(-r_11*math.sin(th_0) + r_21*math.cos(th_0)) - (r_11*math.cos(th_0) + r_21*math.sin(th_0))*(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2)
        R_l[5, 7] = -Pz*r_32 - (-Px*math.sin(th_0) + Py*math.cos(th_0))*(-r_12*math.sin(th_0) + r_22*math.cos(th_0)) - (r_12*math.cos(th_0) + r_22*math.sin(th_0))*(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2)
        R_l[6, 0] = (-Px*math.sin(th_0) + Py*math.cos(th_0))*(r_11*math.cos(th_0) + r_21*math.sin(th_0)) - (-r_11*math.sin(th_0) + r_21*math.cos(th_0))*(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2)
        R_l[6, 1] = (-Px*math.sin(th_0) + Py*math.cos(th_0))*(r_12*math.cos(th_0) + r_22*math.sin(th_0)) - (-r_12*math.sin(th_0) + r_22*math.cos(th_0))*(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2)
        R_l[6, 2] = -Pz*(-r_11*math.sin(th_0) + r_21*math.cos(th_0)) + r_31*(-Px*math.sin(th_0) + Py*math.cos(th_0))
        R_l[6, 3] = -Pz*(-r_12*math.sin(th_0) + r_22*math.cos(th_0)) + r_32*(-Px*math.sin(th_0) + Py*math.cos(th_0))
        R_l[7, 0] = -Pz*(-r_11*math.sin(th_0) + r_21*math.cos(th_0)) + r_31*(-Px*math.sin(th_0) + Py*math.cos(th_0))
        R_l[7, 1] = -Pz*(-r_12*math.sin(th_0) + r_22*math.cos(th_0)) + r_32*(-Px*math.sin(th_0) + Py*math.cos(th_0))
        R_l[7, 2] = (Px*math.sin(th_0) - Py*math.cos(th_0))*(r_11*math.cos(th_0) + r_21*math.sin(th_0)) + (-r_11*math.sin(th_0) + r_21*math.cos(th_0))*(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2)
        R_l[7, 3] = (Px*math.sin(th_0) - Py*math.cos(th_0))*(r_12*math.cos(th_0) + r_22*math.sin(th_0)) + (-r_12*math.sin(th_0) + r_22*math.cos(th_0))*(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1*math.sin(th_0)**2 - a_1*math.cos(th_0)**2)
        try:
            R_l_mat_inv = np.linalg.inv(R_l)
        except:
            return
        R_l_inv_00 = R_l_mat_inv[0, 0]
        R_l_inv_01 = R_l_mat_inv[0, 1]
        R_l_inv_02 = R_l_mat_inv[0, 2]
        R_l_inv_03 = R_l_mat_inv[0, 3]
        R_l_inv_04 = R_l_mat_inv[0, 4]
        R_l_inv_05 = R_l_mat_inv[0, 5]
        R_l_inv_06 = R_l_mat_inv[0, 6]
        R_l_inv_07 = R_l_mat_inv[0, 7]
        R_l_inv_10 = R_l_mat_inv[1, 0]
        R_l_inv_11 = R_l_mat_inv[1, 1]
        R_l_inv_12 = R_l_mat_inv[1, 2]
        R_l_inv_13 = R_l_mat_inv[1, 3]
        R_l_inv_14 = R_l_mat_inv[1, 4]
        R_l_inv_15 = R_l_mat_inv[1, 5]
        R_l_inv_16 = R_l_mat_inv[1, 6]
        R_l_inv_17 = R_l_mat_inv[1, 7]
        R_l_inv_20 = R_l_mat_inv[2, 0]
        R_l_inv_21 = R_l_mat_inv[2, 1]
        R_l_inv_22 = R_l_mat_inv[2, 2]
        R_l_inv_23 = R_l_mat_inv[2, 3]
        R_l_inv_24 = R_l_mat_inv[2, 4]
        R_l_inv_25 = R_l_mat_inv[2, 5]
        R_l_inv_26 = R_l_mat_inv[2, 6]
        R_l_inv_27 = R_l_mat_inv[2, 7]
        R_l_inv_30 = R_l_mat_inv[3, 0]
        R_l_inv_31 = R_l_mat_inv[3, 1]
        R_l_inv_32 = R_l_mat_inv[3, 2]
        R_l_inv_33 = R_l_mat_inv[3, 3]
        R_l_inv_34 = R_l_mat_inv[3, 4]
        R_l_inv_35 = R_l_mat_inv[3, 5]
        R_l_inv_36 = R_l_mat_inv[3, 6]
        R_l_inv_37 = R_l_mat_inv[3, 7]
        R_l_inv_40 = R_l_mat_inv[4, 0]
        R_l_inv_41 = R_l_mat_inv[4, 1]
        R_l_inv_42 = R_l_mat_inv[4, 2]
        R_l_inv_43 = R_l_mat_inv[4, 3]
        R_l_inv_44 = R_l_mat_inv[4, 4]
        R_l_inv_45 = R_l_mat_inv[4, 5]
        R_l_inv_46 = R_l_mat_inv[4, 6]
        R_l_inv_47 = R_l_mat_inv[4, 7]
        R_l_inv_50 = R_l_mat_inv[5, 0]
        R_l_inv_51 = R_l_mat_inv[5, 1]
        R_l_inv_52 = R_l_mat_inv[5, 2]
        R_l_inv_53 = R_l_mat_inv[5, 3]
        R_l_inv_54 = R_l_mat_inv[5, 4]
        R_l_inv_55 = R_l_mat_inv[5, 5]
        R_l_inv_56 = R_l_mat_inv[5, 6]
        R_l_inv_57 = R_l_mat_inv[5, 7]
        R_l_inv_60 = R_l_mat_inv[6, 0]
        R_l_inv_61 = R_l_mat_inv[6, 1]
        R_l_inv_62 = R_l_mat_inv[6, 2]
        R_l_inv_63 = R_l_mat_inv[6, 3]
        R_l_inv_64 = R_l_mat_inv[6, 4]
        R_l_inv_65 = R_l_mat_inv[6, 5]
        R_l_inv_66 = R_l_mat_inv[6, 6]
        R_l_inv_67 = R_l_mat_inv[6, 7]
        R_l_inv_70 = R_l_mat_inv[7, 0]
        R_l_inv_71 = R_l_mat_inv[7, 1]
        R_l_inv_72 = R_l_mat_inv[7, 2]
        R_l_inv_73 = R_l_mat_inv[7, 3]
        R_l_inv_74 = R_l_mat_inv[7, 4]
        R_l_inv_75 = R_l_mat_inv[7, 5]
        R_l_inv_76 = R_l_mat_inv[7, 6]
        R_l_inv_77 = R_l_mat_inv[7, 7]
        
        # Temp variable for efficiency
        x0 = math.sin(th_0)
        x1 = Px*x0
        x2 = math.cos(th_0)
        x3 = Py*x2
        x4 = x1 - x3
        x5 = Pz**2
        x6 = -x1 + x3
        x7 = x6**2
        x8 = Px*x2 + Py*x0 - a_1*x0**2 - a_1*x2**2
        x9 = x8**2
        x10 = -x5 - x7 - x9
        x11 = 2*a_3
        x12 = a_2*x11
        x13 = -x12
        x14 = 2*a_4
        x15 = a_2*x14
        x16 = x13 + x15
        x17 = a_4*x11
        x18 = 2*d_2
        x19 = d_4*x18
        x20 = -x17 + x19
        x21 = a_2**2
        x22 = a_3**2
        x23 = a_4**2
        x24 = d_2**2
        x25 = d_4**2
        x26 = x21 + x22 + x23 + x24 + x25
        x27 = x20 + x26
        x28 = x16 + x27
        x29 = x10 + x28
        x30 = 4*a_2
        x31 = d_4*x30
        x32 = 4*a_3
        x33 = d_4*x32
        x34 = 4*a_4
        x35 = d_2*x34
        x36 = -x35
        x37 = -x33 + x36
        x38 = x31 + x37
        x39 = -x15
        x40 = x10 + x39
        x41 = x17 - x19
        x42 = x26 + x41
        x43 = x13 + x42
        x44 = x40 + x43
        x45 = r_11*x2 + r_21*x0
        x46 = Pz*x45
        x47 = r_31*x8
        x48 = x46 - x47
        x49 = r_12*x2 + r_22*x0
        x50 = Pz*x49
        x51 = r_32*x8
        x52 = x50 - x51
        x53 = R_l_inv_60*x48 + R_l_inv_70*x52
        x54 = a_3*x53
        x55 = R_l_inv_61*x48 + R_l_inv_71*x52
        x56 = d_2*x55
        x57 = d_4*x55
        x58 = -a_2*x53
        x59 = a_4*x53
        x60 = -x59
        x61 = x54 + x56 + x57 + x58 + x60
        x62 = R_l_inv_67*x48 + R_l_inv_77*x52
        x63 = a_4*x62
        x64 = R_l_inv_66*x48 + R_l_inv_76*x52
        x65 = d_4*x64
        x66 = x63 + x65
        x67 = a_2*x62
        x68 = d_2*x64
        x69 = R_l_inv_64*x48
        x70 = -x69
        x71 = R_l_inv_74*x52
        x72 = -x71
        x73 = a_3*x62
        x74 = -x73
        x75 = x67 + x68 + x70 + x72 + x74
        x76 = 2*d_4
        x77 = -x76
        x78 = 2*R_l_inv_62
        x79 = x48*x78
        x80 = 2*R_l_inv_72
        x81 = x52*x80
        x82 = R_l_inv_65*x48 + R_l_inv_75*x52
        x83 = 2*a_2
        x84 = x82*x83
        x85 = -x18 + x79 + x81 + x84
        x86 = x77 + x85
        x87 = x11*x82
        x88 = x14*x82
        x89 = -x87 + x88
        x90 = -x63
        x91 = -x65
        x92 = x90 + x91
        x93 = -x67
        x94 = -x68
        x95 = x69 + x71 + x73 + x93 + x94
        x96 = x14*x64
        x97 = -x96
        x98 = x62*x76
        x99 = -x14*x55
        x100 = x53*x76
        x101 = -x100 + x99
        x102 = 4*d_2
        x103 = 4*R_l_inv_63
        x104 = 4*R_l_inv_73
        x105 = -x102*x82 + x103*x48 + x104*x52 - x30
        x106 = -x98
        x107 = -x57
        x108 = x56 + x58 + x59
        x109 = x107 + x108 + x54
        x110 = x18 - x79 - x81 - x84
        x111 = x110 + x77
        x112 = x87 + x88
        x113 = -r_11*x0 + r_21*x2
        x114 = 2*Pz
        x115 = x114*x6
        x116 = 2*x8
        x117 = -r_31*x5 + r_31*x7 + r_31*x9 - x113*x115 - x116*x46
        x118 = -r_12*x0 + r_22*x2
        x119 = -r_32*x5 + r_32*x7 + r_32*x9 - x115*x118 - x116*x50
        x120 = x8*(-2*x1 + 2*x3)
        x121 = x113*x120 + x114*x47 - x45*x5 - x45*x7 + x45*x9
        x122 = x114*x51 + x118*x120 - x49*x5 - x49*x7 + x49*x9
        x123 = R_l_inv_00*x117 + R_l_inv_10*x119 + R_l_inv_20*x121 + R_l_inv_30*x122
        x124 = a_3*x123
        x125 = R_l_inv_01*x117 + R_l_inv_11*x119 + R_l_inv_21*x121 + R_l_inv_31*x122
        x126 = d_2*x125
        x127 = d_4*x125
        x128 = -a_2*x123
        x129 = a_4*x123
        x130 = -x129
        x131 = x124 + x126 + x127 + x128 + x130
        x132 = R_l_inv_07*x117 + R_l_inv_17*x119 + R_l_inv_27*x121 + R_l_inv_37*x122
        x133 = a_4*x132
        x134 = R_l_inv_06*x117 + R_l_inv_16*x119 + R_l_inv_26*x121 + R_l_inv_36*x122
        x135 = d_4*x134
        x136 = x133 + x135
        x137 = a_2*x132
        x138 = d_2*x134
        x139 = R_l_inv_04*x117
        x140 = -x139
        x141 = R_l_inv_14*x119
        x142 = -x141
        x143 = R_l_inv_24*x121
        x144 = -x143
        x145 = R_l_inv_34*x122
        x146 = -x145
        x147 = a_3*x132
        x148 = -x147
        x149 = x137 + x138 + x140 + x142 + x144 + x146 + x148
        x150 = 4*d_4
        x151 = d_2*x150
        x152 = -x151
        x153 = a_4*x32
        x154 = -x153
        x155 = R_l_inv_05*x117 + R_l_inv_15*x119 + R_l_inv_25*x121 + R_l_inv_35*x122
        x156 = x14*x155
        x157 = a_4*x30
        x158 = x152 + x154 + x156 + x157
        x159 = a_3*x30
        x160 = x11*x155
        x161 = -x159 - x160
        x162 = 2*x24
        x163 = 2*x25
        x164 = 2*x21
        x165 = 2*x117
        x166 = R_l_inv_02*x165
        x167 = 2*x119
        x168 = R_l_inv_12*x167
        x169 = 2*x121
        x170 = R_l_inv_22*x169
        x171 = 2*x122
        x172 = R_l_inv_32*x171
        x173 = x155*x83
        x174 = -x162 - x163 + x164 + x166 + x168 + x170 + x172 + x173
        x175 = 2*x22
        x176 = 2*x23
        x177 = x175 + x176
        x178 = x174 + x177
        x179 = -x133
        x180 = -x135
        x181 = x179 + x180
        x182 = -x137
        x183 = -x138
        x184 = x139 + x141 + x143 + x145 + x147 + x182 + x183
        x185 = x134*x14
        x186 = -x185
        x187 = x132*x76
        x188 = -x125*x14
        x189 = x123*x76
        x190 = x188 - x189
        x191 = 8*d_2
        x192 = a_3*x191
        x193 = 8*d_4
        x194 = a_4*x193
        x195 = 4*x117
        x196 = 4*x119
        x197 = 4*x121
        x198 = 4*x122
        x199 = R_l_inv_03*x195 + R_l_inv_13*x196 + R_l_inv_23*x197 + R_l_inv_33*x198 - a_2*x191 - x102*x155
        x200 = -x187
        x201 = -x127
        x202 = x126 + x128 + x129
        x203 = x124 + x201 + x202
        x204 = -x175 - x176
        x205 = x162 + x163 - x164 - x166 - x168 - x170 - x172 - x173
        x206 = x204 + x205
        x207 = x159 + x160
        x208 = R_l_inv_00*x121 + R_l_inv_10*x122 - R_l_inv_20*x117 - R_l_inv_30*x119
        x209 = a_3*x208
        x210 = R_l_inv_01*x121 + R_l_inv_11*x122 - R_l_inv_21*x117 - R_l_inv_31*x119
        x211 = d_2*x210
        x212 = d_4*x210
        x213 = -a_2*x208
        x214 = a_4*x208
        x215 = -x214
        x216 = x209 + x211 + x212 + x213 + x215
        x217 = R_l_inv_07*x121 + R_l_inv_17*x122 - R_l_inv_27*x117 - R_l_inv_37*x119
        x218 = a_4*x217
        x219 = R_l_inv_06*x121 + R_l_inv_16*x122 - R_l_inv_26*x117 - R_l_inv_36*x119
        x220 = d_4*x219
        x221 = x218 + x220
        x222 = R_l_inv_24*x117
        x223 = R_l_inv_34*x119
        x224 = a_2*x217
        x225 = d_2*x219
        x226 = R_l_inv_04*x121
        x227 = -x226
        x228 = R_l_inv_14*x122
        x229 = -x228
        x230 = a_3*x217
        x231 = -x230
        x232 = x222 + x223 + x224 + x225 + x227 + x229 + x231
        x233 = d_2*x30
        x234 = R_l_inv_22*x165
        x235 = R_l_inv_32*x167
        x236 = R_l_inv_02*x169
        x237 = R_l_inv_12*x171
        x238 = R_l_inv_05*x121 + R_l_inv_15*x122 - R_l_inv_25*x117 - R_l_inv_35*x119
        x239 = x238*x83
        x240 = -x233 - x234 - x235 + x236 + x237 + x239
        x241 = -x31
        x242 = x241 + x33
        x243 = x14*x238 + x36
        x244 = x242 + x243
        x245 = a_4*x150
        x246 = x11*x238
        x247 = d_2*x32
        x248 = -x245 - x246 + x247
        x249 = -x218
        x250 = -x220
        x251 = x249 + x250
        x252 = -x222
        x253 = -x223
        x254 = -x224
        x255 = -x225
        x256 = x226 + x228 + x230 + x252 + x253 + x254 + x255
        x257 = x14*x219
        x258 = -x257
        x259 = x217*x76
        x260 = -x14*x210
        x261 = x208*x76
        x262 = x260 - x261
        x263 = 8*a_3
        x264 = a_2*x263
        x265 = 4*x22
        x266 = 4*x24
        x267 = -x265 + x266
        x268 = 4*x21
        x269 = 4*x25
        x270 = 4*x23
        x271 = -x268 - x269 + x270
        x272 = R_l_inv_03*x197 + R_l_inv_13*x198 - R_l_inv_23*x195 - R_l_inv_33*x196 - x102*x238 + x267 + x271
        x273 = -x259
        x274 = -x212
        x275 = x211 + x213 + x214
        x276 = x209 + x274 + x275
        x277 = x233 + x234 + x235 - x236 - x237 - x239
        x278 = x245 + x246 - x247
        x279 = r_31*x115 - x113*x5 + x113*x7 - x113*x9 + x120*x45
        x280 = r_32*x115 - x118*x5 + x118*x7 - x118*x9 + x120*x49
        x281 = R_l_inv_60*x279 + R_l_inv_70*x280
        x282 = a_3*x281
        x283 = R_l_inv_61*x279 + R_l_inv_71*x280
        x284 = d_2*x283
        x285 = d_4*x283
        x286 = -a_2*x281
        x287 = a_4*x281
        x288 = -x287
        x289 = x282 + x284 + x285 + x286 + x288
        x290 = R_l_inv_67*x279 + R_l_inv_77*x280
        x291 = a_4*x290
        x292 = R_l_inv_66*x279 + R_l_inv_76*x280
        x293 = d_4*x292
        x294 = x291 + x293
        x295 = a_2*x290
        x296 = d_2*x292
        x297 = R_l_inv_64*x279
        x298 = -x297
        x299 = R_l_inv_74*x280
        x300 = -x299
        x301 = a_3*x290
        x302 = -x301
        x303 = x295 + x296 + x298 + x300 + x302
        x304 = x279*x78
        x305 = x280*x80
        x306 = R_l_inv_65*x279 + R_l_inv_75*x280
        x307 = x306*x83
        x308 = x304 + x305 + x307
        x309 = x11*x306
        x310 = x14*x306
        x311 = -x309 + x310
        x312 = -x293
        x313 = x301 + x312
        x314 = -x291 + x39
        x315 = -x21 - x22 - x23 - x24 - x25 - x295 - x296 + x297 + x299
        x316 = x315 + x41
        x317 = x14*x292
        x318 = -x317
        x319 = x290*x76
        x320 = -x14*x283
        x321 = x281*x76
        x322 = x320 - x321
        x323 = -x102*x306 + x103*x279 + x104*x280
        x324 = -x319
        x325 = x317 + x35
        x326 = x287 + x312
        x327 = x284 - x285 + x286
        x328 = x314 + x327
        x329 = -x304 - x305 - x307
        x330 = x309 + x310
        x331 = x12 + x15
        x332 = x287 + x301
        x333 = x20 + x315
        x334 = -x11
        x335 = x14 + x334
        x336 = -x14 + x334
        x337 = 4*x69
        x338 = 4*x71
        x339 = 4*x67
        x340 = 4*x68
        x341 = -4*x65
        x342 = x110 + x76
        x343 = x76 + x85
        x344 = x152 + x153
        x345 = x174 + x204
        x346 = 4*x139
        x347 = 4*x141
        x348 = 4*x143
        x349 = 4*x145
        x350 = x132*x30
        x351 = 4*x138
        x352 = -4*x135
        x353 = x151 + x154
        x354 = x177 + x205
        x355 = a_3*x193
        x356 = a_4*x191
        x357 = 4*x222
        x358 = 4*x223
        x359 = 4*x226
        x360 = 4*x228
        x361 = x217*x30
        x362 = 4*x225
        x363 = -4*x220
        x364 = x277 + x31
        x365 = 8*a_2*a_4
        x366 = x240 + x31
        x367 = x290*x30
        x368 = 4*x296
        x369 = 4*x297
        x370 = 4*x299
        x371 = -a_4*x263 - d_4*x191 - 4*x293
        x372 = x12 + x27
        x373 = x372 + x40
        x374 = x241 + x37
        x375 = x331 + x42
        x376 = x10 + x375
        x377 = -x54
        x378 = x108 + x377 + x57
        x379 = x63 + x91
        x380 = x69 + x71 + x74 + x93 + x94
        x381 = x65 + x90
        x382 = x67 + x68 + x70 + x72 + x73
        x383 = x100 + x99
        x384 = x107 + x377 + x56 + x58 + x60
        x385 = -x124
        x386 = x127 + x202 + x385
        x387 = x133 + x180
        x388 = x139 + x141 + x143 + x145 + x148 + x182 + x183
        x389 = x151 + x153 + x156 + x157
        x390 = x135 + x179
        x391 = x137 + x138 + x140 + x142 + x144 + x146 + x147
        x392 = x188 + x189
        x393 = x126 + x128 + x130 + x201 + x385
        x394 = -x209
        x395 = x212 + x275 + x394
        x396 = x218 + x250
        x397 = x226 + x228 + x231 + x252 + x253 + x254 + x255
        x398 = x243 + x33
        x399 = x220 + x249
        x400 = x222 + x223 + x224 + x225 + x227 + x229 + x230
        x401 = x260 + x261
        x402 = x211 + x213 + x215 + x274 + x394
        x403 = -x282
        x404 = x302 + x403
        x405 = x284 + x285 + x286
        x406 = x295 + x296 + x298 + x300 + x403
        x407 = x320 + x321
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = x4
        A[0, 2] = x4
        A[0, 6] = x4
        A[0, 8] = x4
        A[1, 0] = x29
        A[1, 2] = x29
        A[1, 3] = x38
        A[1, 5] = x38
        A[1, 6] = x44
        A[1, 8] = x44
        A[2, 0] = x61 + x66 + x75
        A[2, 1] = x86 + x89
        A[2, 2] = x61 + x92 + x95
        A[2, 3] = x101 + x97 + x98
        A[2, 4] = x105 + x32
        A[2, 5] = x101 + x106 + x96
        A[2, 6] = x109 + x75 + x92
        A[2, 7] = x111 + x112
        A[2, 8] = x109 + x66 + x95
        A[3, 0] = x131 + x136 + x149
        A[3, 1] = x158 + x161 + x178
        A[3, 2] = x131 + x181 + x184
        A[3, 3] = x186 + x187 + x190
        A[3, 4] = x192 + x194 + x199
        A[3, 5] = x185 + x190 + x200
        A[3, 6] = x149 + x181 + x203
        A[3, 7] = x158 + x206 + x207
        A[3, 8] = x136 + x184 + x203
        A[4, 0] = x216 + x221 + x232
        A[4, 1] = x240 + x244 + x248
        A[4, 2] = x216 + x251 + x256
        A[4, 3] = x258 + x259 + x262
        A[4, 4] = x264 + x272
        A[4, 5] = x257 + x262 + x273
        A[4, 6] = x232 + x251 + x276
        A[4, 7] = x244 + x277 + x278
        A[4, 8] = x221 + x256 + x276
        A[5, 0] = x28 + x289 + x294 + x303
        A[5, 1] = x308 + x311
        A[5, 2] = x12 + x289 + x313 + x314 + x316
        A[5, 3] = x318 + x319 + x322 + x38
        A[5, 4] = x323
        A[5, 5] = x242 + x322 + x324 + x325
        A[5, 6] = x282 + x303 + x326 + x328 + x43
        A[5, 7] = x329 + x330
        A[5, 8] = x282 + x294 + x327 + x331 + x332 + x333
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x335
        B[0, 2] = x335
        B[0, 3] = x150
        B[0, 5] = x150
        B[0, 6] = x336
        B[0, 8] = x336
        B[2, 0] = x86
        B[2, 1] = x337 + x338 - x339 - x340 + x341
        B[2, 2] = x342
        B[2, 3] = x34
        B[2, 4] = x263*x64
        B[2, 5] = -x34
        B[2, 6] = x343
        B[2, 7] = -x337 - x338 + x339 + x340 + x341
        B[2, 8] = x111
        B[3, 0] = x344 + x345
        B[3, 1] = x346 + x347 + x348 + x349 - x350 - x351 + x352
        B[3, 2] = x353 + x354
        B[3, 3] = x355 + x356
        B[3, 4] = x134*x263
        B[3, 5] = -x355 - x356
        B[3, 6] = x345 + x353
        B[3, 7] = -x346 - x347 - x348 - x349 + x350 + x351 + x352
        B[3, 8] = x344 + x354
        B[4, 0] = x240 + x241
        B[4, 1] = -x357 - x358 + x359 + x360 - x361 - x362 + x363
        B[4, 2] = x364
        B[4, 3] = x365
        B[4, 4] = x219*x263
        B[4, 5] = -x365
        B[4, 6] = x366
        B[4, 7] = x357 + x358 - x359 - x360 + x361 + x362 + x363
        B[4, 8] = x241 + x277
        B[5, 0] = x308
        B[5, 1] = x265 - x266 + x271 - x367 - x368 + x369 + x370 + x371
        B[5, 2] = x329
        B[5, 4] = 16*a_3*d_2 + 16*a_4*d_4 + x263*x292
        B[5, 6] = x308
        B[5, 7] = x267 + x268 + x269 - x270 + x367 + x368 - x369 - x370 + x371
        B[5, 8] = x329
        C = np.zeros(shape=(6, 9))
        C[0, 0] = x4
        C[0, 2] = x4
        C[0, 6] = x4
        C[0, 8] = x4
        C[1, 0] = x373
        C[1, 2] = x373
        C[1, 3] = x374
        C[1, 5] = x374
        C[1, 6] = x376
        C[1, 8] = x376
        C[2, 0] = x378 + x379 + x380
        C[2, 1] = x342 + x89
        C[2, 2] = x378 + x381 + x382
        C[2, 3] = x383 + x96 + x98
        C[2, 4] = x105 - x32
        C[2, 5] = x106 + x383 + x97
        C[2, 6] = x380 + x381 + x384
        C[2, 7] = x112 + x343
        C[2, 8] = x379 + x382 + x384
        C[3, 0] = x386 + x387 + x388
        C[3, 1] = x161 + x206 + x389
        C[3, 2] = x386 + x390 + x391
        C[3, 3] = x185 + x187 + x392
        C[3, 4] = -x192 - x194 + x199
        C[3, 5] = x186 + x200 + x392
        C[3, 6] = x388 + x390 + x393
        C[3, 7] = x178 + x207 + x389
        C[3, 8] = x387 + x391 + x393
        C[4, 0] = x395 + x396 + x397
        C[4, 1] = x248 + x364 + x398
        C[4, 2] = x395 + x399 + x400
        C[4, 3] = x257 + x259 + x401
        C[4, 4] = -x264 + x272
        C[4, 5] = x258 + x273 + x401
        C[4, 6] = x397 + x399 + x402
        C[4, 7] = x278 + x366 + x398
        C[4, 8] = x396 + x400 + x402
        C[5, 0] = x16 + x291 + x316 + x326 + x404 + x405
        C[5, 1] = x311 + x329
        C[5, 2] = x293 + x314 + x332 + x372 + x405 + x406
        C[5, 3] = x31 + x319 + x325 + x33 + x407
        C[5, 4] = x323
        C[5, 5] = x318 + x324 + x374 + x407
        C[5, 6] = x13 + x288 + x293 + x328 + x333 + x404
        C[5, 7] = x308 + x330
        C[5, 8] = x288 + x291 + x313 + x327 + x375 + x406
        local_solutions = compute_solution_from_tanhalf_LME(A, B, C)
        for local_solutions_i in local_solutions:
            solution_i: IkSolution = make_ik_solution()
            solution_i[2] = local_solutions_i
            appended_idx = append_solution_to_queue(solution_i)
            add_input_index_to(2, appended_idx)
    # Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor()
    # Finish code for explicit solution node 0
    
    # Code for equation all-zero dispatcher node 2
    def EquationAllZeroDispatcherNode_node_2_processor():
        this_node_input_index: List[int] = node_input_index[2]
        this_input_valid: bool = node_input_validity[2]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[2]
            checked_result: bool = (abs(2*a_2*a_4*math.cos(th_2) + 2*a_3*a_4 - 2*d_2*d_4) <= 1.0e-6) and (abs(2*a_2*d_4*math.cos(th_2) + 2*a_3*d_4 + 2*a_4*d_2) <= 1.0e-6) and (abs(-Px**2 + 2*Px*a_1*math.cos(th_0) - Py**2 + 2*Py*a_1*math.sin(th_0) - Pz**2 - a_1**2 + a_2**2 + 2*a_2*a_3*math.cos(th_2) + a_3**2 + a_4**2 + d_2**2 + d_4**2) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor()
    # Finish code for equation all-zero dispatcher node 2
    
    # Code for explicit solution node 3, solved variable is th_3
    def ExplicitSolutionNode_node_3_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[3]
        this_input_valid: bool = node_input_validity[3]
        if not this_input_valid:
            return
        
        # The solution of non-root node 3
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[2]
            condition_0: bool = (abs(2*a_2*a_4*math.cos(th_2) + 2*a_3*a_4 - 2*d_2*d_4) >= zero_tolerance) or (abs(2*a_2*d_4*math.cos(th_2) + 2*a_3*d_4 + 2*a_4*d_2) >= zero_tolerance) or (abs(-Px**2 + 2*Px*a_1*math.cos(th_0) - Py**2 + 2*Py*a_1*math.sin(th_0) - Pz**2 - a_1**2 + a_2**2 + 2*a_2*a_3*math.cos(th_2) + a_3**2 + a_4**2 + d_2**2 + d_4**2) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*d_4
                x1 = 2*a_4
                x2 = a_2*math.cos(th_2)
                x3 = -a_3*x0 - d_2*x1 - x0*x2
                x4 = a_3*x1 - d_2*x0 + x1*x2
                x5 = math.atan2(x3, x4)
                x6 = 2*a_1
                x7 = Px**2 - Px*x6*math.cos(th_0) + Py**2 - Py*x6*math.sin(th_0) + Pz**2 + a_1**2 - a_2**2 - a_3**2 - 2*a_3*x2 - a_4**2 - d_2**2 - d_4**2
                x8 = safe_sqrt(x3**2 + x4**2 - x7**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = x5 + math.atan2(x8, x7)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs(2*a_2*a_4*math.cos(th_2) + 2*a_3*a_4 - 2*d_2*d_4) >= zero_tolerance) or (abs(2*a_2*d_4*math.cos(th_2) + 2*a_3*d_4 + 2*a_4*d_2) >= zero_tolerance) or (abs(-Px**2 + 2*Px*a_1*math.cos(th_0) - Py**2 + 2*Py*a_1*math.sin(th_0) - Pz**2 - a_1**2 + a_2**2 + 2*a_2*a_3*math.cos(th_2) + a_3**2 + a_4**2 + d_2**2 + d_4**2) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*d_4
                x1 = 2*a_4
                x2 = a_2*math.cos(th_2)
                x3 = -a_3*x0 - d_2*x1 - x0*x2
                x4 = a_3*x1 - d_2*x0 + x1*x2
                x5 = math.atan2(x3, x4)
                x6 = 2*a_1
                x7 = Px**2 - Px*x6*math.cos(th_0) + Py**2 - Py*x6*math.sin(th_0) + Pz**2 + a_1**2 - a_2**2 - a_3**2 - 2*a_3*x2 - a_4**2 - d_2**2 - d_4**2
                x8 = safe_sqrt(x3**2 + x4**2 - x7**2)
                # End of temp variables
                this_solution[3] = x5 + math.atan2(-x8, x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_3_processor()
    # Finish code for explicit solution node 3
    
    # Code for equation all-zero dispatcher node 4
    def EquationAllZeroDispatcherNode_node_4_processor():
        this_node_input_index: List[int] = node_input_index[4]
        this_input_valid: bool = node_input_validity[4]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            checked_result: bool = (abs(Pz) <= 1.0e-6) and (abs(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor()
    # Finish code for equation all-zero dispatcher node 4
    
    # Code for explicit solution node 5, solved variable is th_1
    def ExplicitSolutionNode_node_5_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[5]
        this_input_valid: bool = node_input_validity[5]
        if not this_input_valid:
            return
        
        # The solution of non-root node 5
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            condition_0: bool = (abs(Pz) >= 1.0e-6) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) - a_1) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_2)
                x1 = math.sin(th_3)
                x2 = math.cos(th_3)
                x3 = -a_2 - a_3*x0 - a_4*x0*x2 + d_4*x0*x1
                x4 = -Px*math.cos(th_0) - Py*math.sin(th_0) + a_1
                x5 = a_4*x1 - d_2 + d_4*x2
                # End of temp variables
                this_solution[1] = math.atan2(Pz*x3 - x4*x5, Pz*x5 + x3*x4)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_1_processor()
    # Finish code for explicit solution node 5
    
    # Code for non-branch dispatcher node 6
    # Actually, there is no code
    
    # Code for explicit solution node 7, solved variable is th_5
    def ExplicitSolutionNode_node_7_solve_th_5_processor():
        this_node_input_index: List[int] = node_input_index[6]
        this_input_valid: bool = node_input_validity[6]
        if not this_input_valid:
            return
        
        # The solution of non-root node 7
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            condition_0: bool = (abs(r_13*((-math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + math.sin(th_0)*math.sin(th_2)*math.sin(th_3)) + r_23*((-math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - math.sin(th_2)*math.sin(th_3)*math.cos(th_0)) - r_33*(math.sin(th_1)*math.sin(th_3)*math.cos(th_2) + math.cos(th_1)*math.cos(th_3))) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.cos(th_3)
                x2 = math.sin(th_1)
                x3 = math.sin(th_3)
                x4 = x3*math.cos(th_2)
                x5 = math.sin(th_0)
                x6 = x3*math.sin(th_2)
                x7 = math.cos(th_0)
                x8 = x0*x4 - x1*x2
                x9 = safe_acos(r_13*(x5*x6 + x7*x8) + r_23*(x5*x8 - x6*x7) - r_33*(x0*x1 + x2*x4))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[6] = x9
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
            condition_1: bool = (abs(r_13*((-math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + math.sin(th_0)*math.sin(th_2)*math.sin(th_3)) + r_23*((-math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - math.sin(th_2)*math.sin(th_3)*math.cos(th_0)) - r_33*(math.sin(th_1)*math.sin(th_3)*math.cos(th_2) + math.cos(th_1)*math.cos(th_3))) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.cos(th_3)
                x2 = math.sin(th_1)
                x3 = math.sin(th_3)
                x4 = x3*math.cos(th_2)
                x5 = math.sin(th_0)
                x6 = x3*math.sin(th_2)
                x7 = math.cos(th_0)
                x8 = x0*x4 - x1*x2
                x9 = safe_acos(r_13*(x5*x6 + x7*x8) + r_23*(x5*x8 - x6*x7) - r_33*(x0*x1 + x2*x4))
                # End of temp variables
                this_solution[6] = -x9
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_5_processor()
    # Finish code for explicit solution node 6
    
    # Code for solved_variable dispatcher node 8
    def SolvedVariableDispatcherNode_node_8_processor():
        this_node_input_index: List[int] = node_input_index[8]
        this_input_valid: bool = node_input_validity[8]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_5 = this_solution[6]
            degenerate_valid_0 = (abs(th_5) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(12, node_input_i_idx_in_queue)
            
            th_5 = this_solution[6]
            degenerate_valid_1 = (abs(th_5 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(17, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(9, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor()
    # Finish code for solved_variable dispatcher node 8
    
    # Code for explicit solution node 17, solved variable is th_4th_6_soa
    def ExplicitSolutionNode_node_17_solve_th_4th_6_soa_processor():
        this_node_input_index: List[int] = node_input_index[17]
        this_input_valid: bool = node_input_validity[17]
        if not this_input_valid:
            return
        
        # The solution of non-root node 17
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_21*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_31*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance) or (abs(r_12*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_22*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_32*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_2)
                x1 = x0*math.sin(th_1)
                x2 = math.sin(th_0)
                x3 = math.cos(th_2)
                x4 = math.cos(th_0)
                x5 = x0*math.cos(th_1)
                x6 = x2*x3 - x4*x5
                x7 = x2*x5 + x3*x4
                # End of temp variables
                this_solution[5] = math.atan2(r_11*x6 - r_21*x7 + r_31*x1, r_12*x6 - r_22*x7 + r_32*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(18, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_4th_6_soa_processor()
    # Finish code for explicit solution node 17
    
    # Code for non-branch dispatcher node 18
    # Actually, there is no code
    
    # Code for explicit solution node 19, solved variable is th_4
    def ExplicitSolutionNode_node_19_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[18]
        this_input_valid: bool = node_input_validity[18]
        if not this_input_valid:
            return
        
        # The solution of non-root node 19
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[4] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(20, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_4_processor()
    # Finish code for explicit solution node 18
    
    # Code for non-branch dispatcher node 20
    # Actually, there is no code
    
    # Code for explicit solution node 21, solved variable is th_6
    def ExplicitSolutionNode_node_21_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[20]
        this_input_valid: bool = node_input_validity[20]
        if not this_input_valid:
            return
        
        # The solution of non-root node 21
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_4 = this_solution[4]
            th_4th_6_soa = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[7] = -th_4 + th_4th_6_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_6_processor()
    # Finish code for explicit solution node 20
    
    # Code for explicit solution node 12, solved variable is negative_th_6_positive_th_4__soa
    def ExplicitSolutionNode_node_12_solve_negative_th_6_positive_th_4__soa_processor():
        this_node_input_index: List[int] = node_input_index[12]
        this_input_valid: bool = node_input_validity[12]
        if not this_input_valid:
            return
        
        # The solution of non-root node 12
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_21*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_31*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance) or (abs(r_12*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_22*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_32*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_2)
                x1 = x0*math.sin(th_1)
                x2 = math.cos(th_0)
                x3 = math.cos(th_2)
                x4 = math.sin(th_0)
                x5 = x0*math.cos(th_1)
                x6 = x2*x3 + x4*x5
                x7 = -x2*x5 + x3*x4
                # End of temp variables
                this_solution[0] = math.atan2(-r_11*x7 + r_21*x6 - r_31*x1, r_12*x7 - r_22*x6 + r_32*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(13, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_12_solve_negative_th_6_positive_th_4__soa_processor()
    # Finish code for explicit solution node 12
    
    # Code for non-branch dispatcher node 13
    # Actually, there is no code
    
    # Code for explicit solution node 14, solved variable is th_4
    def ExplicitSolutionNode_node_14_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[13]
        this_input_valid: bool = node_input_validity[13]
        if not this_input_valid:
            return
        
        # The solution of non-root node 14
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[4] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_4_processor()
    # Finish code for explicit solution node 13
    
    # Code for non-branch dispatcher node 15
    # Actually, there is no code
    
    # Code for explicit solution node 16, solved variable is th_6
    def ExplicitSolutionNode_node_16_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[15]
        this_input_valid: bool = node_input_validity[15]
        if not this_input_valid:
            return
        
        # The solution of non-root node 16
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            negative_th_6_positive_th_4__soa = this_solution[0]
            th_4 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[7] = -negative_th_6_positive_th_4__soa + th_4
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_6_processor()
    # Finish code for explicit solution node 15
    
    # Code for explicit solution node 9, solved variable is th_4
    def ExplicitSolutionNode_node_9_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[9]
        this_input_valid: bool = node_input_validity[9]
        if not this_input_valid:
            return
        
        # The solution of non-root node 9
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_5 = this_solution[6]
            condition_0: bool = (abs(-r_13*((math.sin(th_1)*math.sin(th_3) + math.cos(th_1)*math.cos(th_2)*math.cos(th_3))*math.cos(th_0) + math.sin(th_0)*math.sin(th_2)*math.cos(th_3)) - r_23*((math.sin(th_1)*math.sin(th_3) + math.cos(th_1)*math.cos(th_2)*math.cos(th_3))*math.sin(th_0) - math.sin(th_2)*math.cos(th_0)*math.cos(th_3)) - r_33*(-math.sin(th_1)*math.cos(th_2)*math.cos(th_3) + math.sin(th_3)*math.cos(th_1))) >= zero_tolerance) or (abs(r_13*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_23*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_33*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance) or (abs(math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_5)**(-1)
                x1 = math.sin(th_1)
                x2 = math.sin(th_2)
                x3 = math.sin(th_0)
                x4 = math.cos(th_2)
                x5 = math.cos(th_0)
                x6 = math.cos(th_1)
                x7 = x2*x6
                x8 = math.sin(th_3)
                x9 = math.cos(th_3)
                x10 = x4*x9
                x11 = x2*x9
                x12 = x1*x8 + x10*x6
                # End of temp variables
                this_solution[4] = math.atan2(x0*(r_13*(x3*x4 - x5*x7) - r_23*(x3*x7 + x4*x5) + r_33*x1*x2), x0*(-r_13*(x11*x3 + x12*x5) - r_23*(-x11*x5 + x12*x3) - r_33*(-x1*x10 + x6*x8)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_4_processor()
    # Finish code for explicit solution node 9
    
    # Code for non-branch dispatcher node 10
    # Actually, there is no code
    
    # Code for explicit solution node 11, solved variable is th_6
    def ExplicitSolutionNode_node_11_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[10]
        this_input_valid: bool = node_input_validity[10]
        if not this_input_valid:
            return
        
        # The solution of non-root node 11
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_5 = this_solution[6]
            condition_0: bool = (abs(-r_11*((-math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + math.sin(th_0)*math.sin(th_2)*math.sin(th_3)) - r_21*((-math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - math.sin(th_2)*math.sin(th_3)*math.cos(th_0)) + r_31*(math.sin(th_1)*math.sin(th_3)*math.cos(th_2) + math.cos(th_1)*math.cos(th_3))) >= zero_tolerance) or (abs(r_12*((math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) - math.sin(th_0)*math.sin(th_2)*math.sin(th_3)) + r_22*((math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) + math.sin(th_2)*math.sin(th_3)*math.cos(th_0)) + r_32*(math.sin(th_1)*math.sin(th_3)*math.cos(th_2) + math.cos(th_1)*math.cos(th_3))) >= zero_tolerance) or (abs(math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_5)**(-1)
                x1 = math.cos(th_1)
                x2 = math.cos(th_3)
                x3 = math.sin(th_1)
                x4 = math.sin(th_3)
                x5 = x4*math.cos(th_2)
                x6 = x1*x2 + x3*x5
                x7 = math.cos(th_0)
                x8 = x4*math.sin(th_2)
                x9 = x7*x8
                x10 = math.sin(th_0)
                x11 = x2*x3
                x12 = x1*x5
                x13 = x11 - x12
                x14 = x10*x8
                x15 = -x11 + x12
                # End of temp variables
                this_solution[7] = math.atan2(x0*(r_12*(x13*x7 - x14) + r_22*(x10*x13 + x9) + r_32*x6), x0*(r_11*(x14 + x15*x7) + r_21*(x10*x15 - x9) - r_31*x6))
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_6_processor()
    # Finish code for explicit solution node 10
    
    # Collect the output
    ik_out: List[np.ndarray] = list()
    for i in range(len(solution_queue)):
        if not queue_element_validity[i]:
            continue
        ik_out_i = solution_queue[i]
        new_ik_i = np.zeros((robot_nq, 1))
        value_at_0 = th_0  # th_0
        new_ik_i[0] = value_at_0
        value_at_1 = ik_out_i[1]  # th_1
        new_ik_i[1] = value_at_1
        value_at_2 = ik_out_i[2]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[3]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[4]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[6]  # th_5
        new_ik_i[5] = value_at_5
        value_at_6 = ik_out_i[7]  # th_6
        new_ik_i[6] = value_at_6
        ik_out.append(new_ik_i)
    return ik_out


def atlas_r_hand_ik_solve(T_ee: np.ndarray, th_0):
    T_ee_raw_in = atlas_r_hand_ik_target_original_to_raw(T_ee)
    ik_output_raw = atlas_r_hand_ik_solve_raw(T_ee_raw_in, th_0 + th_0_offset_original2raw)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ik_out_i[6] -= th_6_offset_original2raw
        ee_pose_i = atlas_r_hand_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_atlas_r_hand():
    theta_in = np.random.random(size=(7, ))
    ee_pose = atlas_r_hand_fk(theta_in)
    ik_output = atlas_r_hand_ik_solve(ee_pose, th_0=theta_in[0])
    for i in range(len(ik_output)):
        ee_pose_i = atlas_r_hand_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_atlas_r_hand()
