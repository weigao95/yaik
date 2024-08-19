import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 7
n_tree_nodes: int = 52
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_1: float = 0.11
a_2: float = 0.016
a_3: float = 0.0092
a_4: float = 0.00921
d_2: float = 0.306
d_4: float = 0.29955
pre_transform_s0: float = 0.1406
pre_transform_s1: float = 0.2256
pre_transform_s2: float = 0.2326

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -1.5707963267948966
th_2_offset_original2raw: float = 3.141592653589793
th_3_offset_original2raw: float = 3.141592653589793
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 3.141592653589793
th_6_offset_original2raw: float = -1.5707963267948966


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def atlas_l_hand_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 1] = -r_23
    ee_transformed[0, 2] = -r_22
    ee_transformed[0, 3] = Py - pre_transform_s1
    ee_transformed[1, 0] = r_11
    ee_transformed[1, 1] = r_13
    ee_transformed[1, 2] = r_12
    ee_transformed[1, 3] = -Px + pre_transform_s0
    ee_transformed[2, 0] = -r_31
    ee_transformed[2, 1] = -r_33
    ee_transformed[2, 2] = -r_32
    ee_transformed[2, 3] = Pz - pre_transform_s2
    return ee_transformed


def atlas_l_hand_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 1] = r_23
    ee_transformed[0, 2] = r_22
    ee_transformed[0, 3] = -Py + pre_transform_s0
    ee_transformed[1, 0] = -r_11
    ee_transformed[1, 1] = -r_13
    ee_transformed[1, 2] = -r_12
    ee_transformed[1, 3] = Px + pre_transform_s1
    ee_transformed[2, 0] = -r_31
    ee_transformed[2, 1] = -r_33
    ee_transformed[2, 2] = -r_32
    ee_transformed[2, 3] = Pz + pre_transform_s2
    return ee_transformed


def atlas_l_hand_fk(theta_input: np.ndarray):
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
    x4 = math.cos(th_1)
    x5 = math.sin(th_0)
    x6 = math.sin(th_2)
    x7 = x5*x6
    x8 = x2*x3 - x4*x7
    x9 = math.sin(th_4)
    x10 = math.sin(th_3)
    x11 = math.sin(th_1)
    x12 = math.cos(th_3)
    x13 = x2*x6
    x14 = x3*x5
    x15 = x13 + x14*x4
    x16 = x10*x11*x5 + x12*x15
    x17 = -x1*x8 - x16*x9
    x18 = math.cos(th_6)
    x19 = math.sin(th_5)
    x20 = -x10*x15 + x11*x12*x5
    x21 = math.cos(th_5)
    x22 = x1*x16 - x8*x9
    x23 = -x19*x20 + x21*x22
    x24 = a_2*x4
    x25 = -x13*x4 - x14
    x26 = x11*x2
    x27 = x2*x3*x4 - x7
    x28 = x10*x26 + x12*x27
    x29 = -x1*x25 - x28*x9
    x30 = -x10*x27 + x11*x12*x2
    x31 = x1*x28 - x25*x9
    x32 = -x19*x30 + x21*x31
    x33 = x11*x3
    x34 = -x10*x4 + x12*x33
    x35 = x1*x11*x6 - x34*x9
    x36 = -x10*x33 - x12*x4
    x37 = x1*x34 + x11*x6*x9
    x38 = -x19*x36 + x21*x37
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -x0*x17 + x18*x23
    ee_pose[0, 1] = -x19*x22 - x20*x21
    ee_pose[0, 2] = -x0*x23 - x17*x18
    ee_pose[0, 3] = -a_1*x5 - a_3*x15 - a_4*x16 + d_2*x11*x5 - d_4*x20 + pre_transform_s0 - x24*x5
    ee_pose[1, 0] = x0*x29 - x18*x32
    ee_pose[1, 1] = x19*x31 + x21*x30
    ee_pose[1, 2] = x0*x32 + x18*x29
    ee_pose[1, 3] = a_1*x2 + a_3*x27 + a_4*x28 - d_2*x26 + d_4*x30 + pre_transform_s1 + x2*x24
    ee_pose[2, 0] = x0*x35 - x18*x38
    ee_pose[2, 1] = x19*x37 + x21*x36
    ee_pose[2, 2] = x0*x38 + x18*x35
    ee_pose[2, 3] = a_2*x11 + a_3*x33 + a_4*x34 + d_2*x4 + d_4*x36 + pre_transform_s2
    return ee_pose


def atlas_l_hand_twist_jacobian(theta_input: np.ndarray):
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
    x5 = math.cos(th_1)
    x6 = math.sin(th_2)
    x7 = x1*x6
    x8 = -x0*x4 + x5*x7
    x9 = math.cos(th_3)
    x10 = math.sin(th_3)
    x11 = x0*x6
    x12 = x1*x4
    x13 = -x11 - x12*x5
    x14 = -x10*x13 - x3*x9
    x15 = math.cos(th_4)
    x16 = math.sin(th_4)
    x17 = -x10*x3 + x13*x9
    x18 = -x15*x8 - x16*x17
    x19 = math.cos(th_5)
    x20 = math.sin(th_5)
    x21 = -x14*x19 - x20*(x15*x17 - x16*x8)
    x22 = x0*x2
    x23 = -x11*x5 - x12
    x24 = x0*x4*x5 - x7
    x25 = x0*x2*x9 - x10*x24
    x26 = x10*x22 + x24*x9
    x27 = -x15*x23 - x16*x26
    x28 = -x19*x25 - x20*(x15*x26 - x16*x23)
    x29 = x2*x6
    x30 = x2*x4
    x31 = -x10*x30 - x5*x9
    x32 = -x10*x5 + x30*x9
    x33 = x15*x2*x6 - x16*x32
    x34 = -x19*x31 - x20*(x15*x32 + x16*x29)
    x35 = a_2*x2 + d_2*x5 + pre_transform_s2
    x36 = a_2*x5
    x37 = a_1*x0 + pre_transform_s1
    x38 = -d_2*x22 + x0*x36 + x37
    x39 = a_3*x30 + x35
    x40 = a_3*x24 + x38
    x41 = a_4*x32 + d_4*x31 + x39
    x42 = a_4*x26 + d_4*x25 + x40
    x43 = -pre_transform_s0
    x44 = x1*x36
    x45 = a_1*x1
    x46 = x43 + x45
    x47 = d_2*x1*x2 - x44 - x46
    x48 = a_3*x13 + d_2*x3 + pre_transform_s0 - x44 - x45
    x49 = a_4*x17 + d_4*x14 + x48
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 7))
    jacobian_output[0, 1] = x0
    jacobian_output[0, 2] = x3
    jacobian_output[0, 3] = x8
    jacobian_output[0, 4] = x14
    jacobian_output[0, 5] = x18
    jacobian_output[0, 6] = x21
    jacobian_output[1, 1] = x1
    jacobian_output[1, 2] = -x22
    jacobian_output[1, 3] = x23
    jacobian_output[1, 4] = x25
    jacobian_output[1, 5] = x27
    jacobian_output[1, 6] = x28
    jacobian_output[2, 0] = 1
    jacobian_output[2, 2] = x5
    jacobian_output[2, 3] = -x29
    jacobian_output[2, 4] = x31
    jacobian_output[2, 5] = x33
    jacobian_output[2, 6] = x34
    jacobian_output[3, 0] = pre_transform_s1
    jacobian_output[3, 1] = -pre_transform_s2*x1
    jacobian_output[3, 2] = x22*x35 + x38*x5
    jacobian_output[3, 3] = -x23*x39 - x29*x40
    jacobian_output[3, 4] = -x25*x41 + x31*x42
    jacobian_output[3, 5] = -x27*x41 + x33*x42
    jacobian_output[3, 6] = -x28*x41 + x34*x42
    jacobian_output[4, 0] = x43
    jacobian_output[4, 1] = pre_transform_s2*x0
    jacobian_output[4, 2] = x3*x35 - x47*x5
    jacobian_output[4, 3] = x29*x48 + x39*x8
    jacobian_output[4, 4] = x14*x41 - x31*x49
    jacobian_output[4, 5] = x18*x41 - x33*x49
    jacobian_output[4, 6] = x21*x41 - x34*x49
    jacobian_output[5, 1] = -x0*x37 - x1*x46
    jacobian_output[5, 2] = -x22*x47 - x3*x38
    jacobian_output[5, 3] = x23*x48 - x40*x8
    jacobian_output[5, 4] = -x14*x42 + x25*x49
    jacobian_output[5, 5] = -x18*x42 + x27*x49
    jacobian_output[5, 6] = -x21*x42 + x28*x49
    return jacobian_output


def atlas_l_hand_angular_velocity_jacobian(theta_input: np.ndarray):
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
    x5 = math.cos(th_1)
    x6 = math.sin(th_2)
    x7 = x1*x6
    x8 = -x0*x4 + x5*x7
    x9 = math.cos(th_3)
    x10 = math.sin(th_3)
    x11 = x0*x6
    x12 = x1*x4
    x13 = -x11 - x12*x5
    x14 = -x10*x13 - x3*x9
    x15 = math.cos(th_4)
    x16 = math.sin(th_4)
    x17 = -x10*x3 + x13*x9
    x18 = math.cos(th_5)
    x19 = math.sin(th_5)
    x20 = x0*x2
    x21 = -x11*x5 - x12
    x22 = x0*x4*x5 - x7
    x23 = x0*x2*x9 - x10*x22
    x24 = x10*x20 + x22*x9
    x25 = x2*x6
    x26 = x2*x4
    x27 = -x10*x26 - x5*x9
    x28 = -x10*x5 + x26*x9
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 1] = x0
    jacobian_output[0, 2] = x3
    jacobian_output[0, 3] = x8
    jacobian_output[0, 4] = x14
    jacobian_output[0, 5] = -x15*x8 - x16*x17
    jacobian_output[0, 6] = -x14*x18 - x19*(x15*x17 - x16*x8)
    jacobian_output[1, 1] = x1
    jacobian_output[1, 2] = -x20
    jacobian_output[1, 3] = x21
    jacobian_output[1, 4] = x23
    jacobian_output[1, 5] = -x15*x21 - x16*x24
    jacobian_output[1, 6] = -x18*x23 - x19*(x15*x24 - x16*x21)
    jacobian_output[2, 0] = 1
    jacobian_output[2, 2] = x5
    jacobian_output[2, 3] = -x25
    jacobian_output[2, 4] = x27
    jacobian_output[2, 5] = x15*x2*x6 - x16*x28
    jacobian_output[2, 6] = -x18*x27 - x19*(x15*x28 + x16*x25)
    return jacobian_output


def atlas_l_hand_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x6 = a_2*x3 + d_2*x2 + pre_transform_s2
    x7 = a_2*x2
    x8 = x3*x4
    x9 = a_1*x4 + pre_transform_s1
    x10 = -d_2*x8 + x4*x7 + x9
    x11 = math.sin(th_2)
    x12 = x11*x3
    x13 = math.cos(th_2)
    x14 = x0*x13
    x15 = x11*x4
    x16 = -x14 - x15*x2
    x17 = x13*x3
    x18 = a_3*x17 + x6
    x19 = x0*x11
    x20 = x13*x2*x4 - x19
    x21 = a_3*x20 + x10
    x22 = math.cos(th_3)
    x23 = math.sin(th_3)
    x24 = -x17*x23 - x2*x22
    x25 = -x20*x23 + x22*x3*x4
    x26 = x17*x22 - x2*x23
    x27 = a_4*x26 + d_4*x24 + x18
    x28 = x20*x22 + x23*x8
    x29 = a_4*x28 + d_4*x25 + x21
    x30 = math.cos(th_4)
    x31 = math.sin(th_4)
    x32 = x11*x3*x30 - x26*x31
    x33 = -x16*x30 - x28*x31
    x34 = math.cos(th_5)
    x35 = math.sin(th_5)
    x36 = -x24*x34 - x35*(x12*x31 + x26*x30)
    x37 = -x25*x34 - x35*(-x16*x31 + x28*x30)
    x38 = -pre_transform_s0
    x39 = x0*x3
    x40 = x0*x7
    x41 = a_1*x0
    x42 = x38 + x41
    x43 = d_2*x0*x3 - x40 - x42
    x44 = -x13*x4 + x19*x2
    x45 = -x14*x2 - x15
    x46 = a_3*x45 + d_2*x39 + pre_transform_s0 - x40 - x41
    x47 = -x22*x39 - x23*x45
    x48 = x22*x45 - x23*x39
    x49 = a_4*x48 + d_4*x47 + x46
    x50 = -x30*x44 - x31*x48
    x51 = -x34*x47 - x35*(x30*x48 - x31*x44)
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 0] = -p_on_ee_y + pre_transform_s1
    jacobian_output[0, 1] = -pre_transform_s2*x0 + x1
    jacobian_output[0, 2] = -p_on_ee_y*x2 + x10*x2 + x3*x4*x6 - x3*x5
    jacobian_output[0, 3] = p_on_ee_y*x12 + p_on_ee_z*x16 - x12*x21 - x16*x18
    jacobian_output[0, 4] = -p_on_ee_y*x24 + p_on_ee_z*x25 + x24*x29 - x25*x27
    jacobian_output[0, 5] = -p_on_ee_y*x32 + p_on_ee_z*x33 - x27*x33 + x29*x32
    jacobian_output[0, 6] = -p_on_ee_y*x36 + p_on_ee_z*x37 - x27*x37 + x29*x36
    jacobian_output[1, 0] = p_on_ee_x + x38
    jacobian_output[1, 1] = pre_transform_s2*x4 - x5
    jacobian_output[1, 2] = p_on_ee_x*x2 - x1*x3 - x2*x43 + x39*x6
    jacobian_output[1, 3] = -p_on_ee_x*x12 - p_on_ee_z*x44 + x11*x3*x46 + x18*x44
    jacobian_output[1, 4] = p_on_ee_x*x24 - p_on_ee_z*x47 - x24*x49 + x27*x47
    jacobian_output[1, 5] = p_on_ee_x*x32 - p_on_ee_z*x50 + x27*x50 - x32*x49
    jacobian_output[1, 6] = p_on_ee_x*x36 - p_on_ee_z*x51 + x27*x51 - x36*x49
    jacobian_output[2, 1] = -p_on_ee_x*x0 + p_on_ee_y*x4 - x0*x42 - x4*x9
    jacobian_output[2, 2] = p_on_ee_x*x8 + p_on_ee_y*x39 - x10*x39 - x43*x8
    jacobian_output[2, 3] = -p_on_ee_x*x16 + p_on_ee_y*x44 + x16*x46 - x21*x44
    jacobian_output[2, 4] = -p_on_ee_x*x25 + p_on_ee_y*x47 + x25*x49 - x29*x47
    jacobian_output[2, 5] = -p_on_ee_x*x33 + p_on_ee_y*x50 - x29*x50 + x33*x49
    jacobian_output[2, 6] = -p_on_ee_x*x37 + p_on_ee_y*x51 - x29*x51 + x37*x49
    return jacobian_output


def atlas_l_hand_ik_solve_raw(T_ee: np.ndarray, th_0):
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
        for tmp_sol_idx in range(12):
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
    for i in range(52):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True
    
    # Code for non-branch dispatcher node 0
    # Actually, there is no code
    
    # Code for explicit solution node 1, solved variable is th_1
    def General6DoFNumericalReduceSolutionNode_node_1_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return
        
        # The general 6-dof solution of root node with semi-symbolic reduce
        R_l = np.zeros(shape=(8, 8))
        R_l[0, 1] = d_2
        R_l[0, 3] = -a_3
        R_l[0, 7] = -a_4
        R_l[1, 0] = d_2
        R_l[1, 2] = -a_3
        R_l[1, 6] = -a_4
        R_l[2, 4] = a_3
        R_l[2, 5] = d_2
        R_l[3, 1] = -1
        R_l[4, 0] = -1
        R_l[5, 5] = -1
        R_l[6, 2] = a_4
        R_l[6, 6] = a_3
        R_l[7, 3] = -a_4
        R_l[7, 7] = -a_3
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
        x0 = a_3**2
        x1 = -x0
        x2 = d_2**2
        x3 = -x2
        x4 = d_4**2
        x5 = a_4**2
        x6 = -x5
        x7 = Pz**2
        x8 = r_31**2
        x9 = x7*x8
        x10 = r_32**2
        x11 = x10*x7
        x12 = r_33**2
        x13 = x12*x7
        x14 = a_2**2
        x15 = x14*x8
        x16 = x10*x14
        x17 = x12*x14
        x18 = math.sin(th_0)
        x19 = Px*x18
        x20 = math.cos(th_0)
        x21 = Py*x20
        x22 = x19 - x21
        x23 = x22**2
        x24 = r_11*x18 - r_21*x20
        x25 = x24**2
        x26 = x23*x25
        x27 = r_12*x18 - r_22*x20
        x28 = x27**2
        x29 = x23*x28
        x30 = r_13*x18 - r_23*x20
        x31 = x23*x30**2
        x32 = r_11*x20
        x33 = r_21*x18
        x34 = x32 + x33
        x35 = x34**2
        x36 = Px*x20 + Py*x18 - a_1*x18**2 - a_1*x20**2
        x37 = x36**2
        x38 = x35*x37
        x39 = r_12*x20
        x40 = r_22*x18
        x41 = x39 + x40
        x42 = x41**2
        x43 = x37*x42
        x44 = r_13*x20
        x45 = r_23*x18
        x46 = x44 + x45
        x47 = x46**2
        x48 = x37*x47
        x49 = Pz*r_33
        x50 = 2*x49
        x51 = d_4*x50
        x52 = -x51
        x53 = R_l_inv_40*d_2
        x54 = -R_l_inv_50*a_3 + x53
        x55 = Pz*r_31
        x56 = x22*x24
        x57 = x34*x36
        x58 = x55 + x56 + x57
        x59 = a_4*x58
        x60 = 2*x59
        x61 = x54*x60
        x62 = -x61
        x63 = x22*x30
        x64 = x36*x46
        x65 = x49 + x63 + x64
        x66 = R_l_inv_42*d_2
        x67 = -R_l_inv_52*a_3 + x66
        x68 = 2*a_4
        x69 = x67*x68
        x70 = x65*x69
        x71 = -x70
        x72 = 2*x63
        x73 = d_4*x72
        x74 = -x73
        x75 = 2*x64
        x76 = d_4*x75
        x77 = -x76
        x78 = d_4*x69
        x79 = R_l_inv_41*d_2
        x80 = -R_l_inv_51*a_3 + x79
        x81 = Pz*r_32
        x82 = x22*x27
        x83 = x36*x41
        x84 = x81 + x82 + x83
        x85 = a_4*x84
        x86 = 2*x80*x85
        x87 = 2*x19 - 2*x21
        x88 = x24*x57*x87
        x89 = x27*x83*x87
        x90 = x30*x87
        x91 = x64*x90
        x92 = r_32*x46
        x93 = r_33*x41 - x92
        x94 = a_2*x93
        x95 = R_l_inv_46*d_2
        x96 = -R_l_inv_56*a_3 + x95
        x97 = x68*x96
        x98 = x94*x97
        x99 = -x98
        x100 = r_31*x46
        x101 = a_2*(r_33*x34 - x100)
        x102 = R_l_inv_47*d_2
        x103 = -R_l_inv_57*a_3 + x102
        x104 = x103*x68
        x105 = x101*x104
        x106 = -x105
        x107 = 2*x55
        x108 = x107*x56
        x109 = x107*x57
        x110 = 2*x81
        x111 = x110*x82
        x112 = x110*x83
        x113 = x50*x63
        x114 = x50*x64
        x115 = x1 + x106 + x108 + x109 + x11 + x111 + x112 + x113 + x114 + x13 + x15 + x16 + x17 + x26 + x29 + x3 + x31 + x38 + x4 + x43 + x48 + x52 + x6 + x62 + x71 + x74 + x77 + x78 + x86 + x88 + x89 + x9 + x91 + x99
        x116 = a_2*x46
        x117 = 2*x116
        x118 = d_4*x117
        x119 = -x118
        x120 = R_l_inv_43*d_2
        x121 = -R_l_inv_53*a_3 + x120
        x122 = a_4*r_31
        x123 = 2*x122
        x124 = x121*x123
        x125 = -x124
        x126 = R_l_inv_45*d_2
        x127 = -R_l_inv_55*a_3 + x126
        x128 = a_4*r_33
        x129 = 2*x128
        x130 = x127*x129
        x131 = -x130
        x132 = r_32*x63
        x133 = r_32*x64
        x134 = r_33*x22*x27 + r_33*x36*x41 - x132 - x133
        x135 = a_4*x134
        x136 = 2*x135*x96
        x137 = -x136
        x138 = a_2*x34
        x139 = x138*x54*x68
        x140 = -x139
        x141 = x116*x69
        x142 = -x141
        x143 = d_4*r_31
        x144 = x104*x143
        x145 = -x144
        x146 = x119 + x125 + x131 + x137 + x140 + x142 + x145
        x147 = 2*x36
        x148 = x147*x35
        x149 = a_2*x148
        x150 = x147*x42
        x151 = a_2*x150
        x152 = x147*x47
        x153 = a_2*x152
        x154 = x107*x138
        x155 = a_2*x41
        x156 = x110*x155
        x157 = x116*x50
        x158 = 2*x56
        x159 = x138*x158
        x160 = 2*x82
        x161 = x155*x160
        x162 = x116*x72
        x163 = r_31*x63
        x164 = r_31*x64
        x165 = r_33*x22*x24 + r_33*x34*x36 - x163 - x164
        x166 = a_4*x165
        x167 = 2*x103*x166
        x168 = R_l_inv_44*d_2
        x169 = -R_l_inv_54*a_3 + x168
        x170 = a_4*r_32
        x171 = 2*x170
        x172 = x169*x171
        x173 = d_4*r_32
        x174 = x173*x97
        x175 = x155*x68*x80
        x176 = -x167 + x172 - x174 + x175
        x177 = x149 + x151 + x153 + x154 + x156 + x157 + x159 + x161 + x162 + x176
        x178 = x54*x84
        x179 = x101*x96
        x180 = r_31*x169
        x181 = x58*x80
        x182 = x103*x134
        x183 = x138*x80
        x184 = x103*x94
        x185 = x143*x96
        x186 = -x185
        x187 = x180 + x181 + x182 + x183 + x184 + x186
        x188 = r_32*x121
        x189 = x165*x96
        x190 = x155*x54
        x191 = x103*x173
        x192 = x188 - x189 + x190 + x191
        x193 = 4*a_4
        x194 = -x172
        x195 = -x175
        x196 = x124 + x136 + x139 + x144 + x167 + x174 + x194 + x195
        x197 = -x86
        x198 = x1 + x108 + x109 + x11 + x111 + x112 + x113 + x114 + x13 + x15 + x16 + x17 + x26 + x29 + x3 + x31 + x38 + x4 + x43 + x48 + x6 + x61 + x78 + x88 + x89 + x9 + x91 + x98
        x199 = x105 + x197 + x198 + x52 + x71 + x74 + x77
        x200 = x119 + x131 + x142
        x201 = x149 + x151 + x153 + x154 + x156 + x157 + x159 + x161 + x162
        x202 = x121*x193
        x203 = r_33*x202
        x204 = r_31*x82
        x205 = r_32*x56
        x206 = r_31*x83
        x207 = r_32*x57
        x208 = x204 - x205 + x206 - x207
        x209 = -x208
        x210 = x193*x96
        x211 = x209*x210
        x212 = x193*x54
        x213 = x116*x212
        x214 = d_4*r_33
        x215 = x103*x193
        x216 = x214*x215
        x217 = -4*a_4*x54*x65
        x218 = 4*d_4
        x219 = x138*x218
        x220 = x127*x193
        x221 = r_31*x220
        x222 = r_31*x41
        x223 = r_32*x34 - x222
        x224 = -4*a_2*a_4*x223*x96
        x225 = x193*x67
        x226 = x138*x225
        x227 = x217 + x219 + x221 + x224 + x226
        x228 = x218*x55
        x229 = x225*x58
        x230 = x218*x56
        x231 = x218*x57
        x232 = x228 + x229 + x230 + x231
        x233 = 8*d_4
        x234 = x155*x233
        x235 = 8*x127*x170
        x236 = 8*a_4
        x237 = x155*x236*x67
        x238 = 8*x67
        x239 = x233*x81 + x233*x82 + x233*x83 + x238*x85
        x240 = a_2*x223
        x241 = x203 + x211 + x213 + x216
        x242 = x232 + x241
        x243 = x118 + x130 + x141
        x244 = x243 + x51 + x70 + x73 + x76
        x245 = x106 + x198 + x86
        x246 = x124 + x136 + x139 + x144
        x247 = -x178 + x179
        x248 = x125 + x137 + x140 + x145
        x249 = x167 + x174 + x194 + x195
        x250 = x1 + x105 + x108 + x109 + x11 + x111 + x112 + x113 + x114 + x13 + x15 + x16 + x17 + x197 + x26 + x29 + x3 + x31 + x38 + x4 + x43 + x48 + x6 + x62 + x78 + x88 + x89 + x9 + x91 + x99
        x251 = -x214
        x252 = R_l_inv_40*x59
        x253 = R_l_inv_42*a_4
        x254 = x253*x65
        x255 = R_l_inv_43*x122
        x256 = R_l_inv_45*x128
        x257 = R_l_inv_46*x135
        x258 = R_l_inv_40*a_4*x138
        x259 = x116*x253
        x260 = R_l_inv_46*a_4
        x261 = x260*x94
        x262 = R_l_inv_47*a_4
        x263 = x143*x262
        x264 = x251 + x252 + x254 + x255 + x256 + x257 + x258 + x259 + x261 + x263
        x265 = Pz*x8
        x266 = Pz*x10
        x267 = Pz*x12
        x268 = r_31*x56
        x269 = r_31*x57
        x270 = r_32*x82
        x271 = r_32*x83
        x272 = r_33*x63
        x273 = r_33*x64
        x274 = R_l_inv_41*x85
        x275 = x101*x262
        x276 = x265 + x266 + x267 + x268 + x269 + x270 + x271 + x272 + x273 - x274 + x275
        x277 = R_l_inv_47*x166
        x278 = R_l_inv_44*x170
        x279 = x173*x260
        x280 = R_l_inv_41*a_4*x155
        x281 = x277 - x278 + x279 - x280
        x282 = d_4*x253
        x283 = r_31*x138
        x284 = r_32*x155
        x285 = r_33*x116
        x286 = d_2 - x282 - x283 - x284 - x285
        x287 = R_l_inv_40*x84
        x288 = R_l_inv_46*x101
        x289 = R_l_inv_41*x58
        x290 = R_l_inv_44*r_31
        x291 = R_l_inv_47*x134
        x292 = R_l_inv_41*x138
        x293 = R_l_inv_47*x94
        x294 = R_l_inv_46*x143
        x295 = -x294
        x296 = x289 + x290 + x291 + x292 + x293 + x295
        x297 = R_l_inv_43*r_32
        x298 = R_l_inv_46*x165
        x299 = R_l_inv_40*x155
        x300 = R_l_inv_47*x173
        x301 = x297 - x298 + x299 + x300
        x302 = -x252
        x303 = -x261
        x304 = x254 + x302 + x303
        x305 = -x255
        x306 = -x257
        x307 = -x258
        x308 = -x263
        x309 = x251 + x256 + x259 + x305 + x306 + x307 + x308
        x310 = -x277
        x311 = -x279
        x312 = -x275
        x313 = x274 + x278 + x280 + x310 + x311 + x312
        x314 = x265 + x266 + x267 + x268 + x269 + x270 + x271 + x272 + x273 + x286 + x313
        x315 = R_l_inv_42*x60
        x316 = R_l_inv_43*x129
        x317 = R_l_inv_46*x68
        x318 = x209*x317
        x319 = R_l_inv_40*x68
        x320 = x116*x319
        x321 = R_l_inv_47*x68
        x322 = x214*x321
        x323 = -x315 + x316 + x318 + x320 + x322
        x324 = x319*x65
        x325 = x240*x317
        x326 = x324 + x325
        x327 = 2*x143
        x328 = R_l_inv_45*x123
        x329 = 2*x138
        x330 = x253*x329
        x331 = x327 - x328 - x330
        x332 = R_l_inv_42*x193
        x333 = 4*x173
        x334 = R_l_inv_45*x193
        x335 = r_32*x334 + x155*x332 - x333
        x336 = -x327 + x328 + x330
        x337 = x315 + x316 + x318 + x320 + x322
        x338 = -d_2 + x282 + x283 + x284 + x285
        x339 = -x287 + x288
        x340 = -x254
        x341 = x252 + x261 + x340
        x342 = x255 + x257 + x258 + x263
        x343 = x214 - x256 - x259
        x344 = x342 + x343
        x345 = a_2*x222
        x346 = r_32*x34
        x347 = a_2*x346
        x348 = x345 - x347
        x349 = x208 + x348
        x350 = r_33*x82
        x351 = 2*x350
        x352 = r_33*x83
        x353 = 2*x352
        x354 = 2*x132
        x355 = 2*x133
        x356 = r_33*x41
        x357 = 2*a_2
        x358 = x356*x357
        x359 = x357*x92
        x360 = -x358 + x359
        x361 = -x351 - x353 + x354 + x355 + x360
        x362 = r_33*x56
        x363 = 4*x362
        x364 = r_33*x57
        x365 = 4*x364
        x366 = 4*x163
        x367 = 4*x164
        x368 = 4*a_2
        x369 = x100*x368
        x370 = -4*a_2*r_33*x34 + x369
        x371 = -x349
        x372 = a_3*d_2
        x373 = 2*x372
        x374 = d_2*x68
        x375 = x1 + x2 + x5
        x376 = -R_l_inv_15*x375 + R_l_inv_35*x373 + R_l_inv_75*x374
        x377 = r_33*x376
        x378 = -R_l_inv_12*x375 + R_l_inv_32*x373 + R_l_inv_72*x374
        x379 = x378*x65
        x380 = -d_4*x378
        x381 = x116*x378
        x382 = x377 + x379 + x380 + x381
        x383 = R_l_inv_17*x375
        x384 = R_l_inv_37*x373 + R_l_inv_77*x374 - x383
        x385 = x165*x384
        x386 = -R_l_inv_14*x375 + R_l_inv_34*x373 + R_l_inv_74*x374
        x387 = r_32*x386
        x388 = -x387
        x389 = -R_l_inv_11*x375 + R_l_inv_31*x373 + R_l_inv_71*x374
        x390 = x389*x84
        x391 = -x390
        x392 = x101*x384
        x393 = R_l_inv_16*x375
        x394 = R_l_inv_36*x373 + R_l_inv_76*x374 - x393
        x395 = x173*x394
        x396 = x155*x389
        x397 = -x396
        x398 = d_4*x357
        x399 = x100*x398
        x400 = -x399
        x401 = 2*d_4
        x402 = x163*x401
        x403 = -x402
        x404 = x164*x401
        x405 = -x404
        x406 = r_33*x34
        x407 = x398*x406
        x408 = x158*x214
        x409 = 2*x57
        x410 = x214*x409
        x411 = x385 + x388 + x391 + x392 + x395 + x397 + x400 + x403 + x405 + x407 + x408 + x410
        x412 = r_31*x4
        x413 = -R_l_inv_13*x375 + R_l_inv_33*x373 + R_l_inv_73*x374
        x414 = r_31*x413
        x415 = x134*x394
        x416 = r_31**3
        x417 = x416*x7
        x418 = x14*x416
        x419 = -R_l_inv_10*x375 + R_l_inv_30*x373 + R_l_inv_70*x374
        x420 = x138*x419
        x421 = x143*x384
        x422 = r_31*x29
        x423 = r_31*x31
        x424 = r_31*x43
        x425 = r_31*x48
        x426 = r_31*x11
        x427 = r_31*x13
        x428 = r_31*x16
        x429 = r_31*x17
        x430 = r_31*x26
        x431 = r_31*x38
        x432 = 2*r_31
        x433 = x14*x432
        x434 = x35*x433
        x435 = x42*x433
        x436 = x433*x47
        x437 = x158*x265
        x438 = x265*x409
        x439 = x158*x266
        x440 = x266*x409
        x441 = x158*x267
        x442 = x267*x409
        x443 = 2*r_32
        x444 = x23*x24
        x445 = x27*x444
        x446 = x443*x445
        x447 = 2*x37
        x448 = x346*x41
        x449 = x447*x448
        x450 = 2*r_33
        x451 = x30*x444
        x452 = x450*x451
        x453 = 2*x46
        x454 = x37*x453
        x455 = x406*x454
        x456 = x268*x409
        x457 = x160*x207
        x458 = 2*x83
        x459 = x205*x458
        x460 = x364*x72
        x461 = x362*x75
        x462 = x204*x458
        x463 = x163*x75
        x464 = x419*x58
        x465 = x394*x94
        x466 = x266*x329
        x467 = x267*x329
        x468 = x265*x329
        x469 = x35*x36
        x470 = a_2*x432
        x471 = x469*x470
        x472 = x36*x470
        x473 = x42*x472
        x474 = x47*x472
        x475 = x160*x347
        x476 = a_2*x72
        x477 = x406*x476
        x478 = x268*x329
        x479 = x160*x345
        x480 = x100*x476
        x481 = 2*x155
        x482 = x205*x481
        x483 = x117*x362
        x484 = 4*x55
        x485 = x284*x484
        x486 = x285*x484
        x487 = x464 + x465 - x466 - x467 + x468 + x471 + x473 + x474 - x475 - x477 + x478 + x479 + x480 + x482 + x483 + x485 + x486
        x488 = x412 + x414 + x415 - x417 - x418 + x420 + x421 + x422 + x423 + x424 + x425 - x426 - x427 - x428 - x429 - x430 - x431 + x434 + x435 + x436 - x437 - x438 - x439 - x440 - x441 - x442 - x446 - x449 - x452 - x455 - x456 - x457 - x459 - x460 - x461 + x462 + x463 + x487
        x489 = 4*x372
        x490 = d_2*x193
        x491 = R_l_inv_36*x489 + R_l_inv_76*x490 - 2*x393
        x492 = x165*x491
        x493 = x107 + x158 + x409
        x494 = x389*x493
        x495 = r_32*x4
        x496 = -2*x495
        x497 = 2*x413
        x498 = -r_32*x497
        x499 = r_32**3
        x500 = x499*x7
        x501 = 2*x500
        x502 = x14*x499
        x503 = 2*x502
        x504 = 4*r_32
        x505 = x14*x504
        x506 = -x35*x505
        x507 = -x42*x505
        x508 = -x47*x505
        x509 = -x419*x481
        x510 = 2*x384
        x511 = x510*x94
        x512 = -x173*x510
        x513 = r_32*x26
        x514 = -2*x513
        x515 = r_32*x31
        x516 = -2*x515
        x517 = r_32*x38
        x518 = -2*x517
        x519 = r_32*x48
        x520 = -2*x519
        x521 = r_32*x9
        x522 = 2*x521
        x523 = r_32*x13
        x524 = 2*x523
        x525 = r_32*x15
        x526 = 2*x525
        x527 = r_32*x17
        x528 = 2*x527
        x529 = r_32*x29
        x530 = 2*x529
        x531 = r_32*x43
        x532 = 2*x531
        x533 = a_2*x218
        x534 = x356*x533
        x535 = 4*x82
        x536 = x265*x535
        x537 = 4*x83
        x538 = x265*x537
        x539 = x266*x535
        x540 = x266*x537
        x541 = x267*x535
        x542 = x267*x537
        x543 = 4*r_31
        x544 = x445*x543
        x545 = 4*x37
        x546 = x34*x545
        x547 = x222*x546
        x548 = 4*r_33
        x549 = x23*x27*x30
        x550 = x548*x549
        x551 = x46*x545
        x552 = x356*x551
        x553 = 4*x57
        x554 = -x205*x553
        x555 = 4*x64
        x556 = -x132*x555
        x557 = 4*x204
        x558 = x557*x57
        x559 = 4*x56
        x560 = x206*x559
        x561 = x270*x537
        x562 = x352*x63
        x563 = 4*x562
        x564 = x350*x555
        x565 = x492 - x494 + x496 + x498 + x501 + x503 + x506 + x507 + x508 + x509 - x511 + x512 + x514 + x516 + x518 + x520 + x522 + x524 + x526 + x528 + x530 + x532 + x533*x92 - x534 + x536 + x538 + x539 + x540 + x541 + x542 + x544 + x547 + x550 + x552 + x554 + x556 + x558 + x560 + x561 + x563 + x564
        x566 = R_l_inv_37*x489 + R_l_inv_77*x490 - 2*x383
        x567 = x134*x566
        x568 = x386*x432
        x569 = x329*x389
        x570 = x327*x394
        x571 = x214*x535
        x572 = x214*x537
        x573 = x132*x218
        x574 = x133*x218
        x575 = -x567 - x568 - x569 + x570 - x571 - x572 + x573 + x574
        x576 = x110 + x160 + x458
        x577 = x419*x576
        x578 = 2*x101
        x579 = x394*x578
        x580 = 4*x155
        x581 = x266*x580
        x582 = a_2*x504
        x583 = x469*x582
        x584 = x36*x582
        x585 = x42*x584
        x586 = x47*x584
        x587 = x265*x580
        x588 = x267*x580
        x589 = 8*x55
        x590 = x347*x589
        x591 = 8*a_2
        x592 = x49*x92
        x593 = x591*x592
        x594 = x138*x557
        x595 = x347*x559
        x596 = x270*x580
        x597 = x368*x63
        x598 = x597*x92
        x599 = 4*x116
        x600 = x350*x599
        x601 = x345*x559
        x602 = x356*x597
        x603 = -x577 + x579 - x581 - x583 - x585 - x586 + x587 + x588 - x590 - x593 - x594 - x595 - x596 - x598 - x600 + x601 + x602
        x604 = -x385
        x605 = -x392
        x606 = -x395
        x607 = -x407
        x608 = -x408
        x609 = -x410
        x610 = x387 + x390 + x396 + x399 + x402 + x404 + x604 + x605 + x606 + x607 + x608 + x609
        x611 = -x412 - x414 - x415 + x417 + x418 - x420 - x421 - x422 - x423 - x424 - x425 + x426 + x427 + x428 + x429 + x430 + x431 - x434 - x435 - x436 + x437 + x438 + x439 + x440 + x441 + x442 + x446 + x449 + x452 + x455 + x456 + x457 + x459 + x460 + x461 - x462 - x463
        x612 = -x464 - x465 + x466 + x467 - x468 - x471 - x473 - x474 + x475 + x477 - x478 - x479 - x480 - x482 - x483 - x485 - x486 + x611
        x613 = x50 + x72 + x75
        x614 = x265*x599
        x615 = x266*x599
        x616 = x369*x56
        x617 = a_2*x535
        x618 = x617*x92
        x619 = x378*x493
        x620 = x376*x432
        x621 = x329*x378
        x622 = x619 + x620 + x621
        x623 = x209*x491
        x624 = r_33*x4
        x625 = 2*x624
        x626 = r_33*x497
        x627 = r_33**3
        x628 = x627*x7
        x629 = 2*x628
        x630 = x14*x627
        x631 = 2*x630
        x632 = x14*x35
        x633 = x548*x632
        x634 = x14*x548
        x635 = x42*x634
        x636 = x47*x634
        x637 = x117*x419
        x638 = x214*x510
        x639 = r_33*x26
        x640 = 2*x639
        x641 = r_33*x29
        x642 = 2*x641
        x643 = r_33*x38
        x644 = 2*x643
        x645 = r_33*x43
        x646 = 2*x645
        x647 = r_33*x9
        x648 = 2*x647
        x649 = r_33*x11
        x650 = 2*x649
        x651 = r_33*x15
        x652 = 2*x651
        x653 = r_33*x16
        x654 = 2*x653
        x655 = r_33*x31
        x656 = 2*x655
        x657 = r_33*x48
        x658 = 2*x657
        x659 = 4*x265
        x660 = x63*x659
        x661 = x64*x659
        x662 = 4*x266
        x663 = x63*x662
        x664 = x64*x662
        x665 = 4*x267
        x666 = x63*x665
        x667 = x64*x665
        x668 = x451*x543
        x669 = x100*x546
        x670 = x504*x549
        x671 = x41*x92
        x672 = x545*x671
        x673 = x363*x57
        x674 = x350*x537
        x675 = x366*x57
        x676 = x367*x56
        x677 = x132*x537
        x678 = x133*x535
        x679 = x272*x555
        x680 = -x623 - x625 - x626 + x629 + x631 - x633 - x635 - x636 - x637 - x638 - x640 - x642 - x644 - x646 + x648 + x650 + x652 + x654 + x656 + x658 + x660 + x661 + x663 + x664 + x666 + x667 + x668 + x669 + x670 + x672 - x673 - x674 + x675 + x676 + x677 + x678 + x679
        x681 = 4*x81
        x682 = x535 + x537 + x681
        x683 = x378*x682
        x684 = x376*x504 + x378*x580
        x685 = x223*x357
        x686 = r_33*x368
        x687 = x36*x686
        x688 = x368*x406
        x689 = 8*x49
        x690 = x132*x580 + x138*x366 + x267*x599 + x272*x599 + x284*x689 + x356*x617 + x394*x685 + x406*x55*x591 + x419*x613 + x42*x687 + x469*x686 + x47*x687 + x56*x688 - x614 - x615 - x616 - x618
        x691 = d_4*x378
        x692 = x379 + x691
        x693 = x377 + x381
        x694 = x692 + x693
        x695 = x567 + x568 + x569 - x570 + x571 + x572 - x573 - x574
        x696 = -4*a_2*d_4*r_32*x46 + x492 + x494 + x496 + x498 + x501 + x503 + x506 + x507 + x508 + x509 + x511 + x512 + x514 + x516 + x518 + x520 + x522 + x524 + x526 + x528 + x530 + x532 + x534 + x536 + x538 + x539 + x540 + x541 + x542 + x544 + x547 + x550 + x552 + x554 + x556 + x558 + x560 + x561 + x563 + x564
        x697 = -R_l_inv_03*x375 + R_l_inv_23*x373 + R_l_inv_63*x374
        x698 = r_31*x697
        x699 = -R_l_inv_05*x375 + R_l_inv_25*x373 + R_l_inv_65*x374
        x700 = r_33*x699
        x701 = -R_l_inv_00*x375 + R_l_inv_20*x373 + R_l_inv_60*x374
        x702 = x58*x701
        x703 = -R_l_inv_02*x375 + R_l_inv_22*x373 + R_l_inv_62*x374
        x704 = x65*x703
        x705 = R_l_inv_06*x375
        x706 = R_l_inv_26*x373 + R_l_inv_66*x374 - x705
        x707 = x134*x706
        x708 = x138*x701
        x709 = x116*x703
        x710 = x706*x94
        x711 = R_l_inv_07*x375
        x712 = R_l_inv_27*x373 + R_l_inv_67*x374 - x711
        x713 = x143*x712
        x714 = d_4*x358
        x715 = -x714
        x716 = x160*x214
        x717 = -x716
        x718 = x214*x458
        x719 = -x718
        x720 = d_4*x359
        x721 = d_4*x354
        x722 = d_4*x355
        x723 = x698 + x700 + x702 + x704 + x707 + x708 + x709 + x710 + x713 + x715 + x717 + x719 + x720 + x721 + x722
        x724 = x165*x712
        x725 = -R_l_inv_04*x375 + R_l_inv_24*x373 + R_l_inv_64*x374
        x726 = r_32*x725
        x727 = x173*x706
        x728 = -R_l_inv_01*x375 + R_l_inv_21*x373 + R_l_inv_61*x374
        x729 = x155*x728
        x730 = x443*x632
        x731 = x14*x443
        x732 = x42*x731
        x733 = x47*x731
        x734 = x160*x265
        x735 = x265*x458
        x736 = x160*x266
        x737 = x266*x458
        x738 = x160*x267
        x739 = x267*x458
        x740 = x432*x445
        x741 = x34*x447
        x742 = x222*x741
        x743 = x450*x549
        x744 = x356*x454
        x745 = x205*x409
        x746 = x132*x75
        x747 = x204*x409
        x748 = x158*x206
        x749 = x270*x458
        x750 = x352*x72
        x751 = x350*x75
        x752 = -x495 + x500 + x502 - x513 - x515 - x517 - x519 + x521 + x523 + x525 + x527 + x529 + x531 + x724 - x726 + x727 - x729 - x730 - x732 - x733 + x734 + x735 + x736 + x737 + x738 + x739 + x740 + x742 + x743 + x744 - x745 - x746 + x747 + x748 + x749 + x750 + x751
        x753 = -d_4*x703 + x752
        x754 = x728*x84
        x755 = x101*x712
        x756 = x266*x481
        x757 = a_2*r_32
        x758 = x148*x757
        x759 = x150*x757
        x760 = x152*x757
        x761 = x265*x481
        x762 = x267*x481
        x763 = x347*x484
        x764 = x368*x592
        x765 = x204*x329
        x766 = x158*x347
        x767 = x270*x481
        x768 = x476*x92
        x769 = x117*x350
        x770 = x158*x345
        x771 = x356*x476
        x772 = -x754 + x755 - x756 - x758 - x759 - x760 + x761 + x762 - x763 - x764 - x765 - x766 - x767 - x768 - x769 + x770 + x771
        x773 = x493*x728
        x774 = R_l_inv_27*x489 + R_l_inv_67*x490 - 2*x711
        x775 = -x134*x774
        x776 = 2*x412
        x777 = -x776
        x778 = -x432*x725
        x779 = -x329*x728
        x780 = 2*x712
        x781 = x780*x94
        x782 = x327*x706
        x783 = x14*x543
        x784 = -x204*x537 + x205*x537 + x207*x535 + x268*x553 + x363*x64 + x365*x63 - x366*x64 + x406*x551 + 2*x417 + 2*x418 - x42*x783 - 2*x422 - 2*x423 - 2*x424 - 2*x425 + 2*x426 + 2*x427 + 2*x428 + 2*x429 + 2*x430 + 2*x431 + x445*x504 + x448*x545 + x451*x548 - x47*x783 - x543*x632 + x56*x659 + x56*x662 + x56*x665 + x57*x659 + x57*x662 + x57*x665
        x785 = x138*x659
        x786 = a_2*r_31
        x787 = 4*x786
        x788 = x469*x787
        x789 = x36*x787
        x790 = x42*x789
        x791 = x47*x789
        x792 = x138*x662
        x793 = x138*x665
        x794 = x284*x589
        x795 = x285*x589
        x796 = 4*x138*x268
        x797 = x345*x535
        x798 = x369*x63
        x799 = x205*x580
        x800 = x116*x363
        x801 = x347*x535
        x802 = x63*x688
        x803 = -x785 - x788 - x790 - x791 + x792 + x793 - x794 - x795 - x796 - x797 - x798 - x799 - x800 + x801 + x802
        x804 = -x773 + x775 + x777 + x778 + x779 - x781 + x782 + x784 + x803
        x805 = x576*x701
        x806 = x578*x706
        x807 = x406*x533
        x808 = x100*x533
        x809 = -x805 + x806 - x807 + x808
        x810 = R_l_inv_26*x489 + R_l_inv_66*x490 - 2*x705
        x811 = x165*x810
        x812 = 2*x697
        x813 = r_32*x812
        x814 = x481*x701
        x815 = x173*x780
        x816 = 4*x214
        x817 = x56*x816
        x818 = x57*x816
        x819 = x163*x218
        x820 = x164*x218
        x821 = x811 - x813 - x814 - x815 - x817 - x818 + x819 + x820
        x822 = d_4*x703
        x823 = x752 + x772 + x822
        x824 = -x704
        x825 = x702 + x710 + x715 + x720 + x824
        x826 = -x700 - x709
        x827 = x698 + x707 + x708 + x713 + x717 + x719 + x721 + x722 + x826
        x828 = x209*x810
        x829 = x493*x703
        x830 = r_33*x812
        x831 = x117*x701
        x832 = x214*x780
        x833 = -x333*x56
        x834 = -x333*x57
        x835 = x143*x535
        x836 = x143*x537
        x837 = x828 - x829 + x830 + x831 + x832 + x833 + x834 + x835 + x836
        x838 = x432*x699
        x839 = x329*x703
        x840 = -x838 - x839
        x841 = x218*x347
        x842 = x218*x345 + x613*x701 + x685*x706 - x841
        x843 = x682*x703
        x844 = x504*x699 + x580*x703
        x845 = x838 + x839
        x846 = x828 + x829 + x830 + x831 + x832 + x833 + x834 + x835 + x836
        x847 = x754 - x755 + x756 + x758 + x759 + x760 - x761 - x762 + x763 + x764 + x765 + x766 + x767 + x768 + x769 - x770 - x771
        x848 = x822 + x847
        x849 = x805 - x806 + x807 - x808
        x850 = -x811 + x813 + x814 + x815 + x817 + x818 - x819 - x820
        x851 = -x702
        x852 = -x710
        x853 = -x720
        x854 = x704 + x714 + x851 + x852 + x853
        x855 = x700 + x709
        x856 = -x698 - x707 - x708 - x713 + x716 + x718 - x721 - x722
        x857 = x855 + x856
        x858 = 2*a_3
        x859 = x0 + x3 + x5
        x860 = -R_l_inv_53*x859 + x120*x858
        x861 = r_31*x860
        x862 = R_l_inv_50*x859
        x863 = x53*x858 - x862
        x864 = x58*x863
        x865 = R_l_inv_56*x859
        x866 = x858*x95 - x865
        x867 = x134*x866
        x868 = R_l_inv_57*x859
        x869 = x102*x858 - x868
        x870 = x165*x869
        x871 = -R_l_inv_54*x859 + x168*x858
        x872 = r_32*x871
        x873 = -x872
        x874 = R_l_inv_51*x859
        x875 = x79*x858 - x874
        x876 = x84*x875
        x877 = -x876
        x878 = x138*x863
        x879 = x101*x869
        x880 = x866*x94
        x881 = x143*x869
        x882 = x173*x866
        x883 = x155*x875
        x884 = -x883
        x885 = x861 + x864 + x867 + x870 + x873 + x877 + x878 + x879 + x880 + x881 + x882 + x884
        x886 = -R_l_inv_55*x859 + x126*x858
        x887 = r_33*x886
        x888 = R_l_inv_52*x859
        x889 = x66*x858 - x888
        x890 = d_4*x889
        x891 = x116*x889
        x892 = -x639
        x893 = -x641
        x894 = -x643
        x895 = -x645
        x896 = x450*x632
        x897 = -x896
        x898 = x14*x450
        x899 = x42*x898
        x900 = -x899
        x901 = x47*x898
        x902 = -x901
        x903 = x265*x72
        x904 = x265*x75
        x905 = x266*x72
        x906 = x266*x75
        x907 = x267*x72
        x908 = x267*x75
        x909 = x432*x451
        x910 = x100*x741
        x911 = x443*x549
        x912 = x447*x671
        x913 = x362*x409
        x914 = -x913
        x915 = x350*x458
        x916 = -x915
        x917 = x163*x409
        x918 = x158*x164
        x919 = x132*x458
        x920 = x133*x160
        x921 = x272*x75
        x922 = a_3*x68 + x117*x214 + x138*x327 + x173*x481 + x624 + x628 + x630 + x647 + x649 + x651 + x653 + x655 + x657 + x887 - x890 + x891 + x892 + x893 + x894 + x895 + x897 + x900 + x902 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 + x914 + x916 + x917 + x918 + x919 + x920 + x921
        x923 = x265*x401
        x924 = x266*x401
        x925 = x267*x401
        x926 = x327*x56
        x927 = x327*x57
        x928 = x160*x173
        x929 = x173*x458
        x930 = x214*x72
        x931 = x214*x75
        x932 = -x923 - x924 - x925 - x926 - x927 - x928 - x929 - x930 - x931
        x933 = x65*x889
        x934 = x117*x267
        x935 = r_33*x149
        x936 = r_33*x151
        x937 = r_33*x153
        x938 = x117*x265
        x939 = x117*x266
        x940 = x55*x688
        x941 = 4*x49
        x942 = x284*x941
        x943 = x163*x329
        x944 = x155*x354
        x945 = a_2*x158
        x946 = x406*x945
        x947 = a_2*x160
        x948 = x356*x947
        x949 = x117*x272
        x950 = x100*x945
        x951 = x92*x947
        x952 = x933 - x934 - x935 - x936 - x937 + x938 + x939 - x940 - x942 - x943 - x944 - x946 - x948 - x949 + x950 + x951
        x953 = 4*a_3
        x954 = x53*x953 - 2*x862
        x955 = x84*x954
        x956 = x578*x866
        x957 = x79*x953 - 2*x874
        x958 = x58*x957
        x959 = x102*x953 - 2*x868
        x960 = x134*x959
        x961 = x432*x871
        x962 = x327*x866
        x963 = -x962
        x964 = x329*x875
        x965 = 2*x869
        x966 = x94*x965
        x967 = x958 + x960 + x961 + x963 + x964 + x966
        x968 = -2*x865 + x95*x953
        x969 = x165*x968
        x970 = 2*x860
        x971 = r_32*x970
        x972 = x481*x863
        x973 = x173*x965
        x974 = -x969 + x971 + x972 + x973
        x975 = -2*a_3*a_4
        x976 = -2*a_2*d_4*r_31*x34
        x977 = -2*a_2*d_4*r_32*x41
        x978 = -2*a_2*d_4*r_33*x46
        x979 = x890 + x923 + x924 + x925 + x926 + x927 + x928 + x929 + x930 + x931 + x975 + x976 + x977 + x978
        x980 = -x933 + x934 + x935 + x936 + x937 - x938 - x939 + x940 + x942 + x943 + x944 + x946 + x948 + x949 - x950 - x951
        x981 = x65*x954
        x982 = x432*x886
        x983 = x685*x866
        x984 = x329*x889
        x985 = x776 + x784 - x981 + x982 - x983 + x984
        x986 = x209*x968
        x987 = r_33*x970
        x988 = x117*x863
        x989 = x214*x965
        x990 = -x986 - x987 - x988 - x989
        x991 = x58*(x66*x953 - 2*x888)
        x992 = x803 + x991
        x993 = 8*a_3
        x994 = x66*x993 - 4*x888
        x995 = 8*x155
        x996 = x266*x995
        x997 = 8*x757
        x998 = x469*x997
        x999 = x36*x997
        x1000 = x42*x999
        x1001 = x47*x999
        x1002 = 16*x55
        x1003 = x1002*x347
        x1004 = 16*a_2
        x1005 = x1004*x592
        x1006 = 8*x204
        x1007 = x1006*x138
        x1008 = 8*x56
        x1009 = x1008*x347
        x1010 = x270*x995
        x1011 = x591*x63
        x1012 = x1011*x92
        x1013 = 8*x350
        x1014 = x1013*x116
        x1015 = 8*r_32
        x1016 = x1015*x14
        x1017 = 8*x265
        x1018 = 8*x266
        x1019 = 8*x267
        x1020 = 8*r_31
        x1021 = 8*x37
        x1022 = 8*r_33
        x1023 = x1006*x57 + x1008*x206 + x1013*x64 - x1015*x632 - x1016*x42 - x1016*x47 + x1017*x82 + x1017*x83 + x1018*x82 + x1018*x83 + x1019*x82 + x1019*x83 + x1020*x445 + x1021*x222*x34 + x1021*x356*x46 + x1022*x549 + x13*x504 - 8*x132*x64 + x15*x504 + x17*x504 - 8*x205*x57 - x26*x504 + 8*x270*x83 + x29*x504 - x31*x504 - x38*x504 + x43*x504 - x48*x504 + 4*x495 + 4*x500 + 4*x502 + x504*x886 + x504*x9 + 8*x562 + x580*x889
        x1024 = x986 + x987 + x988 + x989
        x1025 = x776 + x784 + x981 + x982 + x983 + x984
        x1026 = -x879
        x1027 = x1026 + x864 + x876 + x880
        x1028 = -x870
        x1029 = -x882
        x1030 = x1028 + x1029 + x861 + x867 + x872 + x878 + x881 + x883
        x1031 = x624 + x628 + x630 + x647 + x649 + x651 + x653 + x655 + x657 + x887 + x891 + x892 + x893 + x894 + x895 + x897 + x900 + x902 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 + x914 + x916 + x917 + x918 + x919 + x920 + x921 + x952 + x979
        x1032 = -x955 + x956
        x1033 = -x864
        x1034 = -x880
        x1035 = x1033 + x1034 + x877 + x879
        x1036 = x870 + x873 + x882 + x884
        x1037 = -x861 - x867 - x878 - x881
        x1038 = x1036 + x1037
        x1039 = x214*x368
        x1040 = x265*x368
        x1041 = x266*x368
        x1042 = x267*x368
        x1043 = x220*x46
        x1044 = a_2*r_33
        x1045 = x1044*x225
        x1046 = x268*x368
        x1047 = x269*x368
        x1048 = x270*x368
        x1049 = x271*x368
        x1050 = x272*x368
        x1051 = x273*x368
        x1052 = -x1039 + x1040 + x1041 + x1042 + x1043 - x1045 + x1046 + x1047 + x1048 + x1049 + x1050 + x1051
        x1053 = x169*x193*x41
        x1054 = x46*x55
        x1055 = x46*x56
        x1056 = Pz*r_33*x34 - x1054 - x1055 + x22*x30*x34
        x1057 = x1056*x215
        x1058 = x193*x757*x80
        x1059 = d_4*x41
        x1060 = x1059*x210
        x1061 = -x1053 - x1057 + x1058 + x1060
        x1062 = x46*x81
        x1063 = x46*x82
        x1064 = Pz*r_33*x41 - x1062 - x1063 + x22*x30*x41
        x1065 = x1064*x210
        x1066 = x202*x34
        x1067 = x212*x786
        x1068 = d_4*x34
        x1069 = x1068*x215
        x1070 = -x1065 + x1066 - x1067 + x1069
        x1071 = x121*x41
        x1072 = x1056*x96
        x1073 = x54*x757
        x1074 = x103*x1059
        x1075 = x103*x1064 + x1068*x96 - x169*x34 + x786*x80
        x1076 = x1065 - x1066 + x1067 - x1069
        x1077 = x1053 + x1057 - x1058 - x1060
        x1078 = x41*x55
        x1079 = x41*x56
        x1080 = Pz*r_32*x34 - x1078 - x1079 + x22*x27*x34
        x1081 = x1080*x236*x96
        x1082 = x236*x46
        x1083 = x128*x54*x591
        x1084 = a_2*x122*x238 - x127*x236*x34 + x143*x591
        x1085 = 16*x41
        x1086 = x1039 + x1040 + x1041 + x1042 - x1043 + x1045 + x1046 + x1047 + x1048 + x1049 + x1050 + x1051
        x1087 = d_4*x453
        x1088 = x24*x34*x87
        x1089 = x41*x87
        x1090 = x1089*x27
        x1091 = x46*x90
        x1092 = x107*x34
        x1093 = x110*x41
        x1094 = x46*x50
        x1095 = R_l_inv_45*x46*x68
        x1096 = r_33*x357
        x1097 = x1096*x253
        x1098 = -x1087 + x1088 + x1090 + x1091 + x1092 + x1093 + x1094 + x1095 - x1097 + x148 + x150 + x152
        x1099 = R_l_inv_44*x41*x68
        x1100 = x1056*x321
        x1101 = R_l_inv_41*a_2*x171
        x1102 = x1059*x317
        x1103 = -x1099 - x1100 + x1101 + x1102
        x1104 = x1064*x317
        x1105 = R_l_inv_43*x34*x68
        x1106 = R_l_inv_40*a_2*x123
        x1107 = x1068*x321
        x1108 = -x1104 + x1105 - x1106 + x1107
        x1109 = R_l_inv_43*x41
        x1110 = R_l_inv_46*x1056
        x1111 = R_l_inv_40*x757
        x1112 = R_l_inv_47*x1059
        x1113 = R_l_inv_41*x786 - R_l_inv_44*x34 + R_l_inv_46*x1068 + R_l_inv_47*x1064
        x1114 = x1104 - x1105 + x1106 - x1107
        x1115 = x1099 + x1100 - x1101 - x1102
        x1116 = x193*x46
        x1117 = R_l_inv_43*x1116
        x1118 = R_l_inv_47*d_4*x1116
        x1119 = 4*x1068
        x1120 = x1119 + x332*x786 - x334*x34
        x1121 = 8*x1059
        x1122 = x1087 + x1088 + x1090 + x1091 + x1092 + x1093 + x1094 - x1095 + x1097 + x148 + x150 + x152
        x1123 = -2*Pz*r_32*x34 + x107*x41 + x1089*x24 - x27*x34*x87
        x1124 = -x1123
        x1125 = x46*x681
        x1126 = 4*x19 - 4*x21
        x1127 = x1126*x27
        x1128 = -4*Pz*r_33*x41 + x1125 - x1126*x30*x41 + x1127*x46
        x1129 = 8*x1054
        x1130 = x34*x49
        x1131 = 8*x1130
        x1132 = 8*x19 - 8*x21
        x1133 = x1132*x24
        x1134 = x1132*x30
        x1135 = 2*x44 + 2*x45
        x1136 = x1135*x376
        x1137 = x1096*x378
        x1138 = 2*x32 + 2*x33
        x1139 = 2*x4
        x1140 = x34**3
        x1141 = 2*x34
        x1142 = a_2*x419
        x1143 = x1126*x24
        x1144 = x1143*x36
        x1145 = x36*x484
        x1146 = x222*x7
        x1147 = x548*x7
        x1148 = 4*x41
        x1149 = 4*x46
        x1150 = x34*x82
        x1151 = 4*x63
        x1152 = 4*x34
        x1153 = x55*x56
        x1154 = x100*x1147 + x100*x634 + x1054*x1151 + x1055*x941 + x1064*x491 - x1068*x510 + x1078*x535 + x1079*x681 - x11*x1141 - x1130*x1151 - x1138*x413 + x1138*x43 + x1138*x48 - x1139*x34 + x1140*x447 - x1141*x13 + x1141*x15 - x1141*x16 - x1141*x17 + x1141*x26 - x1141*x29 - x1141*x31 + x1141*x9 + x1142*x432 + x1143*x469 + x1144*x42 + x1144*x47 + x1145*x42 + x1145*x47 + x1146*x504 + x1148*x445 + x1149*x451 - x1150*x681 + x1152*x1153 + x222*x505 + x469*x484
        x1155 = -x1136 + x1137 + x1154
        x1156 = 2*x39 + 2*x40
        x1157 = x1156*x386
        x1158 = x1056*x566
        x1159 = 2*x757
        x1160 = x1159*x389
        x1161 = 2*x1059
        x1162 = x1161*x394
        x1163 = x1054*x218
        x1164 = x1055*x218
        x1165 = x1130*x218
        x1166 = x34*x63
        x1167 = x1166*x218
        x1168 = x1157 + x1158 - x1160 - x1162 - x1163 - x1164 + x1165 + x1167
        x1169 = 8*x372
        x1170 = d_2*x236
        x1171 = R_l_inv_37*x1169 + R_l_inv_77*x1170 - 4*x383
        x1172 = 4*x32 + 4*x33
        x1173 = x1172*x386
        x1174 = x1062*x233
        x1175 = x1063*x233
        x1176 = x41*x49
        x1177 = x41*x63
        x1178 = 4*x39 + 4*x40
        x1179 = R_l_inv_36*x1169 + R_l_inv_76*x1170 - 4*x393
        x1180 = 4*x4
        x1181 = x41**3
        x1182 = 4*x1059
        x1183 = x1132*x27
        x1184 = x1183*x36
        x1185 = 8*x81
        x1186 = x1185*x36
        x1187 = x1020*x346
        x1188 = x1022*x92
        x1189 = 8*x34
        x1190 = 8*x46
        x1191 = 8*x63
        x1192 = x34*x81
        x1193 = 8*x41
        x1194 = x81*x82
        x1195 = -x1008*x1078 + x1008*x1192 - x1056*x1179 + x1062*x1191 + x1063*x689 + x11*x1148 + x1142*x504 - x1148*x13 - x1148*x15 + x1148*x16 - x1148*x17 - x1148*x26 + x1148*x29 - x1148*x31 + x1148*x38 - x1148*x9 + x1150*x589 - x1176*x1191 - x1178*x413 + x1178*x48 - x1180*x41 + x1181*x545 - x1182*x384 + x1183*x469 + x1184*x42 + x1184*x47 + x1185*x469 + x1186*x42 + x1186*x47 + x1187*x14 + x1187*x7 + x1188*x14 + x1188*x7 + x1189*x445 + x1190*x549 + x1193*x1194
        x1196 = x1136 - x1137 + x1154
        x1197 = x1172*x376
        x1198 = x378*x787
        x1199 = 4*x44 + 4*x45
        x1200 = x46**3
        x1201 = x1134*x36
        x1202 = x36*x689
        x1203 = x1020*x406
        x1204 = x356*x7
        x1205 = 8*x82
        x1206 = x49*x63
        x1207 = -8*Pz*r_31*x22*x24*x46 - 8*Pz*r_32*x22*x27*x46 - 4*d_4*x384*x46 - 4*x10*x14*x46 - 4*x10*x46*x7 + x1015*x1204 + x1016*x356 + x1080*x1179 + x1131*x56 + x1134*x469 + x1149*x13 + x1149*x17 + x1149*x31 + x1149*x38 + x1149*x43 + x1166*x589 + x1176*x1205 + x1177*x1185 + x1189*x451 + x1190*x1206 + x1193*x549 - x1199*x413 + x1200*x545 + x1201*x42 + x1201*x47 + x1202*x42 + x1202*x47 + x1203*x14 + x1203*x7 - 4*x14*x46*x8 - 4*x23*x25*x46 - 4*x23*x28*x46 - 4*x4*x46 + x419*x686 - 4*x46*x7*x8 + x469*x689
        x1208 = 8*x39 + 8*x40
        x1209 = -x1157 - x1158 + x1160 + x1162 + x1163 + x1164 - x1165 - x1167
        x1210 = x1135*x699
        x1211 = x1096*x703
        x1212 = 2*x41
        x1213 = x1127*x36
        x1214 = x36*x681
        x1215 = x543*x7
        x1216 = x46*x535
        x1217 = -x1056*x774 - x1078*x559 + x11*x1212 + x1125*x63 + x1127*x469 - x1139*x41 + x1147*x92 + x1149*x549 + x1152*x445 + x1156*x48 - x1156*x725 + x1159*x728 + x1161*x706 - 4*x1176*x63 + x1181*x447 - x1212*x13 - x1212*x15 + x1212*x16 - x1212*x17 - x1212*x26 + x1212*x29 - x1212*x31 + x1212*x38 - x1212*x9 + x1213*x42 + x1213*x47 + x1214*x42 + x1214*x47 + x1215*x346 + x1216*x49 + x34*x535*x55 + x34*x56*x681 + x346*x783 + x41*x681*x82 + x469*x681 + x634*x92
        x1218 = x1210 - x1211 + x1217
        x1219 = x1138*x697
        x1220 = x1064*x810
        x1221 = a_2*x701
        x1222 = x1221*x432
        x1223 = x1068*x780
        x1224 = d_4*x1125
        x1225 = d_4*x1216
        x1226 = x1176*x218
        x1227 = x1177*x218
        x1228 = x1219 - x1220 - x1222 + x1223 - x1224 - x1225 + x1226 + x1227
        x1229 = x1178*x697
        x1230 = R_l_inv_26*x1169 + R_l_inv_66*x1170 - 4*x705
        x1231 = x1056*x1230
        x1232 = x1182*x712
        x1233 = x1221*x504
        x1234 = x1054*x233
        x1235 = x1055*x233
        x1236 = x1130*x233
        x1237 = x1166*x233
        x1238 = x1180*x34
        x1239 = x1133*x36
        x1240 = x36*x589
        x1241 = x100*x1022
        x1242 = x1015*x1146 + x1016*x222 + x1055*x689 + x1078*x1205 + x1079*x1185 - x11*x1152 + x1129*x63 - x1131*x63 + x1133*x469 + x1140*x545 - x1152*x13 + x1152*x15 - x1152*x16 - x1152*x17 + x1152*x26 - x1152*x29 - x1152*x31 + x1152*x9 + x1153*x1189 + x1172*x43 + x1172*x48 + x1190*x451 - x1192*x1205 + x1193*x445 + x1239*x42 + x1239*x47 + x1240*x42 + x1240*x47 + x1241*x14 + x1241*x7 + x469*x589
        x1243 = x1064*(R_l_inv_27*x1169 + R_l_inv_67*x1170 - 4*x711) + x1119*x706 - x1172*x725 - x1238 + x1242 + x728*x787
        x1244 = -x1210 + x1211 + x1217
        x1245 = x1172*x699
        x1246 = x703*x787
        x1247 = -8*Pz*d_4*r_32*x34 - 8*d_4*x22*x27*x34 - 4*d_4*x46*x712 + x1080*x1230 + x1121*x55 + x1121*x56 - x1199*x697 + x686*x701
        x1248 = -x1219 + x1220 + x1222 - x1223 + x1224 + x1225 - x1226 - x1227
        x1249 = x1138*x860
        x1250 = x1135*x886
        x1251 = x1064*x968
        x1252 = x4*x453
        x1253 = x1200*x447
        x1254 = a_2*x863
        x1255 = x1254*x432
        x1256 = -x1096*x889
        x1257 = -x453*x9
        x1258 = -x11*x453
        x1259 = -x15*x453
        x1260 = -x16*x453
        x1261 = -x26*x453
        x1262 = -x29*x453
        x1263 = x1068*x965
        x1264 = x13*x453
        x1265 = x17*x453
        x1266 = x31*x453
        x1267 = x38*x453
        x1268 = x43*x453
        x1269 = x1126*x30
        x1270 = x1269*x469
        x1271 = x1269*x36
        x1272 = x1271*x42
        x1273 = x1271*x47
        x1274 = x469*x941
        x1275 = x36*x941
        x1276 = x1275*x42
        x1277 = x1275*x47
        x1278 = x1215*x406
        x1279 = x406*x783
        x1280 = x1204*x504
        x1281 = x356*x505
        x1282 = x1152*x451
        x1283 = x1148*x549
        x1284 = -x1054*x559
        x1285 = -x1063*x681
        x1286 = x1166*x484
        x1287 = x1177*x681
        x1288 = x1130*x559
        x1289 = x1176*x535
        x1290 = x1149*x1206
        x1291 = x1249 + x1250 - x1251 + x1252 + x1253 - x1255 + x1256 + x1257 + x1258 + x1259 + x1260 + x1261 + x1262 + x1263 + x1264 + x1265 + x1266 + x1267 + x1268 + x1270 + x1272 + x1273 + x1274 + x1276 + x1277 + x1278 + x1279 + x1280 + x1281 + x1282 + x1283 + x1284 + x1285 + x1286 + x1287 + x1288 + x1289 + x1290
        x1292 = x1156*x871
        x1293 = x1056*x959
        x1294 = x1159*x875
        x1295 = x1161*x866
        x1296 = -x1292 - x1293 + x1294 + x1295
        x1297 = x218*x469
        x1298 = x218*x36
        x1299 = x1298*x42
        x1300 = x1298*x47
        x1301 = x1119*x55
        x1302 = x1059*x681
        x1303 = x218*x46
        x1304 = x1303*x49
        x1305 = x1119*x56
        x1306 = x1059*x535
        x1307 = x1303*x63
        x1308 = -x1297 - x1299 - x1300 - x1301 - x1302 - x1304 - x1305 - x1306 - x1307
        x1309 = x1178*x860
        x1310 = -4*x865 + x95*x993
        x1311 = x1056*x1310
        x1312 = x1182*x869
        x1313 = x1254*x504
        x1314 = x1064*(x102*x993 - 4*x868) + x1119*x866 - x1172*x871 + x787*x875
        x1315 = x1292 + x1293 - x1294 - x1295
        x1316 = -x1249 + x1250 + x1251 + x1252 + x1253 + x1255 + x1256 + x1257 + x1258 + x1259 + x1260 + x1261 + x1262 - x1263 + x1264 + x1265 + x1266 + x1267 + x1268 + x1270 + x1272 + x1273 + x1274 + x1276 + x1277 + x1278 + x1279 + x1280 + x1281 + x1282 + x1283 + x1284 + x1285 + x1286 + x1287 + x1288 + x1289 + x1290
        x1317 = x1199*x860
        x1318 = x1303*x869
        x1319 = x1172*x886 + x1238 + x1242 - x787*x889
        x1320 = 16*x7
        x1321 = r_31*x346
        x1322 = r_33*x92
        x1323 = 16*x14
        x1324 = 16*x56
        x1325 = 16*x63
        x1326 = 16*x81
        x1327 = x1326*x36
        x1328 = x27*(16*x19 - 16*x21)
        x1329 = x1328*x36
        x1330 = x1297 + x1299 + x1300 + x1301 + x1302 + x1304 + x1305 + x1306 + x1307
        x1331 = -x149 - x151 - x153 - x154 - x156 - x157 - x159 - x161 - x162
        x1332 = x1331 + x243
        x1333 = x192 + x247
        x1334 = x1331 + x51 + x70 + x73 + x76
        x1335 = x265 + x266 + x267 + x268 + x269 + x270 + x271 + x272 + x273 + x274 + x281 + x312 + x338
        x1336 = x301 + x339
        x1337 = x276 + x278 + x280 + x310 + x311 + x338
        x1338 = -x324 - x325
        x1339 = -x204 + x205 - x206 + x207 + x348
        x1340 = x351 + x353 - x354 - x355 + x360
        x1341 = -x1339
        x1342 = x380 + x487 + x611
        x1343 = -x377 - x381
        x1344 = x387 + x391 + x392 + x396 + x400 + x402 + x404 + x407 + x604 + x606 + x608 + x609
        x1345 = x577 - x579 + x581 + x583 + x585 + x586 - x587 - x588 + x590 + x593 + x594 + x595 + x596 + x598 + x600 - x601 - x602
        x1346 = x487 + x611
        x1347 = -x379 + x693
        x1348 = x680 + x690
        x1349 = x385 + x388 + x390 + x395 + x397 + x399 + x403 + x405 + x408 + x410 + x605 + x607
        x1350 = x752 + x848
        x1351 = x785 + x788 + x790 + x791 - x792 - x793 + x794 + x795 + x796 + x797 + x798 + x799 + x800 - x801 - x802
        x1352 = x1351 + x773 + x775 + x777 + x778 + x779 + x781 + x782 + x784
        x1353 = x753 + x847
        x1354 = -4*a_2*d_4*r_31*x41 - 2*a_2*x223*x706 - x613*x701 + x841
        x1355 = x624 + x628 + x630 + x647 + x649 + x651 + x653 + x655 + x657 + x887 + x890 + x891 + x892 + x893 + x894 + x895 + x897 + x900 + x902 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 + x914 + x916 + x917 + x918 + x919 + x920 + x921 + x932 + x975 + x976 + x977 + x978 + x980
        x1356 = x1032 + x974
        x1357 = x1351 - x991
        x1358 = x922 + x923 + x924 + x925 + x926 + x927 + x928 + x929 + x930 + x931 + x980
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = x115 + x146 + x177
        A[0, 1] = x193*(-x178 + x179 - x187 - x192)
        A[0, 2] = x196 + x199 + x200 + x201
        A[0, 3] = x203 + x211 + x213 + x216 - x227 - x232
        A[0, 4] = -x234 - x235 - x237 - x239
        A[0, 5] = x210*x240 + x212*x65 + x219 + x221 + x226 + x242
        A[0, 6] = x177 + x244 + x245 + x246
        A[0, 7] = x193*(-x187 + x188 - x189 + x190 + x191 - x247)
        A[0, 8] = x201 + x244 + x248 + x249 + x250
        A[1, 0] = x264 + x276 + x281 + x286
        A[1, 1] = x68*(x287 - x288 + x296 + x301)
        A[1, 2] = x304 + x309 + x314
        A[1, 3] = -x323 - x326 - x331
        A[1, 4] = x332*x84 + x335
        A[1, 5] = -x326 - x336 - x337
        A[1, 6] = -x264 + x265 + x266 + x267 + x268 + x269 + x270 + x271 + x272 + x273 - x313 - x338
        A[1, 7] = x68*(x296 - x297 + x298 - x299 - x300 + x339)
        A[1, 8] = x314 + x341 + x344
        A[2, 0] = x349
        A[2, 2] = x349
        A[2, 3] = x361
        A[2, 4] = x363 + x365 - x366 - x367 - x370
        A[2, 5] = -x361
        A[2, 6] = x371
        A[2, 8] = x371
        A[3, 0] = -x382 - x411 - x488
        A[3, 1] = x565 + x575 + x603
        A[3, 2] = -x382 - x610 - x612
        A[3, 3] = 8*Pz*a_2*r_31*r_33*x34 + 8*Pz*a_2*r_32*r_33*x41 + 4*Pz*a_2*x12*x46 + 4*a_2*r_31*x22*x30*x34 + 4*a_2*r_32*x22*x30*x41 + 4*a_2*r_33*x22*x24*x34 + 4*a_2*r_33*x22*x27*x41 + 4*a_2*r_33*x22*x30*x46 + 4*a_2*r_33*x35*x36 + 4*a_2*r_33*x36*x42 + 4*a_2*r_33*x36*x47 + 2*a_2*x223*x394 + x419*x613 - x614 - x615 - x616 - x618 - x622 - x680
        A[3, 4] = -x683 - x684
        A[3, 5] = x622 + x623 + x625 + x626 - x629 - x631 + x633 + x635 + x636 + x637 + x638 + x640 + x642 + x644 + x646 - x648 - x650 - x652 - x654 - x656 - x658 - x660 - x661 - x663 - x664 - x666 - x667 - x668 - x669 - x670 - x672 + x673 + x674 - x675 - x676 - x677 - x678 - x679 + x690
        A[3, 6] = x488 + x610 + x694
        A[3, 7] = -x603 - x695 - x696
        A[3, 8] = x411 + x612 + x694
        A[4, 0] = -x723 - x753 - x772
        A[4, 1] = x804 + x809 + x821
        A[4, 2] = x823 + x825 + x827
        A[4, 3] = x837 + x840 + x842
        A[4, 4] = -x843 - x844
        A[4, 5] = x842 + x845 + x846
        A[4, 6] = x495 - x500 - x502 + x513 + x515 + x517 + x519 - x521 - x523 - x525 - x527 - x529 - x531 + x723 - x724 + x726 - x727 + x729 + x730 + x732 + x733 - x734 - x735 - x736 - x737 - x738 - x739 - x740 - x742 - x743 - x744 + x745 + x746 - x747 - x748 - x749 - x750 - x751 + x848
        A[4, 7] = x804 + x849 + x850
        A[4, 8] = x823 + x854 + x857
        A[5, 0] = x885 + x922 + x932 + x952
        A[5, 1] = x955 - x956 + x967 + x974
        A[5, 2] = x624 + x628 + x630 - x639 - x641 - x643 - x645 + x647 + x649 + x651 + x653 + x655 + x657 - x885 + x887 + x891 - x896 - x899 - x901 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 - x913 - x915 + x917 + x918 + x919 + x920 + x921 - x979 - x980
        A[5, 3] = x985 + x990 + x992
        A[5, 4] = -x1000 - x1001 - x1003 - x1005 - x1007 + x1008*x345 - x1009 - x1010 + x1011*x356 - x1012 - x1014 + x1023 + x265*x995 + x267*x995 + x84*x994 - x996 - x998
        A[5, 5] = -x1024 - x1025 - x992
        A[5, 6] = -x1027 - x1030 - x1031
        A[5, 7] = x1032 + x967 + x969 - x971 - x972 - x973
        A[5, 8] = -x1031 - x1035 - x1038
        B = np.zeros(shape=(6, 9))
        B[0, 0] = -x1052 - x1061 - x1070
        B[0, 1] = x236*(-x1071 - x1072 + x1073 - x1074 + x1075)
        B[0, 2] = -x1052 - x1076 - x1077
        B[0, 3] = d_4*x103*x1082 - x1081 + x1082*x121 - x1083 + x1084
        B[0, 4] = -a_4*x1085*x127 + x1004*x170*x67 + x1004*x173
        B[0, 5] = 8*a_4*d_4*x103*x46 + 8*a_4*x121*x46 - x1081 - x1083 - x1084
        B[0, 6] = -x1061 - x1076 - x1086
        B[0, 7] = x236*(x1071 + x1072 - x1073 + x1074 + x1075)
        B[0, 8] = -x1070 - x1077 - x1086
        B[1, 0] = x1098 + x1103 + x1108
        B[1, 1] = x193*(x1109 + x1110 - x1111 + x1112 - x1113)
        B[1, 2] = x1098 + x1114 + x1115
        B[1, 3] = 4*R_l_inv_40*a_2*a_4*r_33 + 4*R_l_inv_46*a_4*x1080 - x1117 - x1118 - x1120
        B[1, 4] = 8*R_l_inv_45*a_4*x41 - x1121 - x253*x997
        B[1, 5] = R_l_inv_40*x1044*x193 + R_l_inv_46*x1080*x193 - x1117 - x1118 + x1120
        B[1, 6] = x1103 + x1114 + x1122
        B[1, 7] = x193*(-x1109 - x1110 + x1111 - x1112 - x1113)
        B[1, 8] = x1108 + x1115 + x1122
        B[2, 0] = x1124
        B[2, 2] = x1124
        B[2, 3] = -x1128
        B[2, 4] = x1129 - x1131 + x1133*x46 - x1134*x34
        B[2, 5] = x1128
        B[2, 6] = x1123
        B[2, 8] = x1123
        B[3, 0] = x1155 + x1168
        B[3, 1] = x1064*x1171 + x1119*x394 - x1173 - x1174 - x1175 + x1176*x233 + x1177*x233 + x1195 + x389*x787
        B[3, 2] = -x1168 - x1196
        B[3, 3] = -x1197 + x1198 - x1207
        B[3, 4] = -x1208*x376 + x378*x997
        B[3, 5] = x1197 - x1198 - x1207
        B[3, 6] = -x1155 - x1209
        B[3, 7] = 8*Pz*d_4*r_33*x41 + 4*a_2*r_31*x389 + 8*d_4*x22*x30*x41 + 4*d_4*x34*x394 + x1064*x1171 - x1173 - x1174 - x1175 - x1195
        B[3, 8] = x1196 + x1209
        B[4, 0] = -x1218 - x1228
        B[4, 1] = -x1229 - x1231 - x1232 + x1233 - x1234 - x1235 + x1236 + x1237 + x1243
        B[4, 2] = x1228 + x1244
        B[4, 3] = -x1245 + x1246 - x1247
        B[4, 4] = -x1208*x699 + x703*x997
        B[4, 5] = x1245 - x1246 - x1247
        B[4, 6] = -x1244 - x1248
        B[4, 7] = x1229 + x1231 + x1232 - x1233 + x1234 + x1235 - x1236 - x1237 + x1243
        B[4, 8] = x1218 + x1248
        B[5, 0] = x1291 + x1296 + x1308
        B[5, 1] = x1309 + x1311 + x1312 - x1313 - x1314
        B[5, 2] = x1308 + x1315 + x1316
        B[5, 3] = x1080*x1310 - x1317 - x1318 + x1319 + x686*x863
        B[5, 4] = x1002*x1150 + x1021*x1181 + x1062*x1325 + 16*x1063*x49 - x1078*x1324 + x1085*x1194 + x11*x1193 - x1176*x1325 + x1192*x1324 - x1193*x13 - x1193*x15 + x1193*x16 - x1193*x17 - x1193*x26 + x1193*x29 - x1193*x31 + x1193*x38 + x1193*x4 - x1193*x9 + x1208*x48 + x1208*x886 + x1320*x1321 + x1320*x1322 + x1321*x1323 + x1322*x1323 + x1326*x469 + x1327*x42 + x1327*x47 + x1328*x469 + x1329*x42 + x1329*x47 + 16*x34*x445 + 16*x46*x549 - x889*x997
        B[5, 5] = 4*a_2*r_33*x863 + x1080*x1310 - x1317 - x1318 - x1319
        B[5, 6] = -x1291 - x1315 - x1330
        B[5, 7] = -x1309 - x1311 - x1312 + x1313 - x1314
        B[5, 8] = -x1296 - x1316 - x1330
        C = np.zeros(shape=(6, 9))
        C[0, 0] = x115 + x1332 + x196
        C[0, 1] = x193*(x1333 + x180 - x181 + x182 + x183 - x184 + x186)
        C[0, 2] = x1332 + x176 + x199 + x248
        C[0, 3] = -x217 + x219 + x221 - x224 + x226 - x242
        C[0, 4] = x234 + x235 + x237 - x239
        C[0, 5] = -x227 + x228 + x229 + x230 + x231 - x241
        C[0, 6] = x1334 + x146 + x245 + x249
        C[0, 7] = x193*(-x1333 + x180 - x181 + x182 + x183 - x184 - x185)
        C[0, 8] = x1334 + x176 + x200 + x246 + x250
        C[1, 0] = -x1335 - x251 - x256 - x259 - x302 - x303 - x340 - x342
        C[1, 1] = x68*(R_l_inv_41*x58 + R_l_inv_47*a_2*x93 - x1336 - x290 - x291 - x292 - x295)
        C[1, 2] = -x1337 - x309 - x341
        C[1, 3] = x1338 + x331 + x337
        C[1, 4] = 4*R_l_inv_42*a_4*x84 - x335
        C[1, 5] = x1338 + x323 + x336
        C[1, 6] = -x1335 - x252 - x254 - x261 - x305 - x306 - x307 - x308 - x343
        C[1, 7] = x68*(x1336 + x289 - x290 - x291 - x292 + x293 + x294)
        C[1, 8] = -x1337 - x304 - x344
        C[2, 0] = x1339
        C[2, 2] = x1339
        C[2, 3] = x1340
        C[2, 4] = -x363 - x365 + x366 + x367 - x370
        C[2, 5] = -x1340
        C[2, 6] = x1341
        C[2, 8] = x1341
        C[3, 0] = -x1342 - x1343 - x1344 - x379
        C[3, 1] = -x1345 - x575 - x696
        C[3, 2] = x1344 + x1346 + x1347 + x691
        C[3, 3] = x1348 - x619 + x620 + x621
        C[3, 4] = -x683 + x684
        C[3, 5] = x1348 + x619 - x620 - x621
        C[3, 6] = x1343 + x1346 + x1349 + x692
        C[3, 7] = x1345 + x565 + x695
        C[3, 8] = -x1342 - x1347 - x1349
        C[4, 0] = x1350 + x698 + x707 + x708 + x713 + x714 + x717 + x719 + x721 + x722 + x824 + x851 + x852 + x853 + x855
        C[4, 1] = -x1352 - x821 - x849
        C[4, 2] = -x1353 - x827 - x854
        C[4, 3] = -x1354 - x840 - x846
        C[4, 4] = -x843 + x844
        C[4, 5] = -x1354 - x837 - x845
        C[4, 6] = x1350 + x702 + x704 + x710 + x715 + x720 + x826 + x856
        C[4, 7] = -x1352 - x809 - x850
        C[4, 8] = -x1353 - x825 - x857
        C[5, 0] = -x1026 - x1033 - x1034 - x1036 - x1355 - x861 - x867 - x876 - x878 - x881
        C[5, 1] = 2*a_2*x869*x93 - x1356 + x58*x957 - x960 - x961 - x963 - x964
        C[5, 2] = -x1028 - x1029 - x1037 - x1355 - x864 - x872 - x877 - x879 - x880 - x883
        C[5, 3] = -x1025 - x1357 - x990
        C[5, 4] = 8*Pz*a_2*x12*x41 + 8*Pz*a_2*x41*x8 + 8*a_2*r_31*x22*x24*x41 + 8*a_2*r_33*x22*x30*x41 - x1000 - x1001 - x1003 - x1005 - x1007 - x1009 - x1010 - x1012 - x1014 - x1023 + x84*x994 - x996 - x998
        C[5, 5] = x1024 + x1357 + x985
        C[5, 6] = x1030 + x1035 + x1358
        C[5, 7] = x1356 + x958 - x960 - x961 + x962 - x964 + x966
        C[5, 8] = x1027 + x1038 + x1358
        local_solutions = compute_solution_from_tanhalf_LME(A, B, C)
        for local_solutions_i in local_solutions:
            solution_i: IkSolution = make_ik_solution()
            solution_i[2] = local_solutions_i
            appended_idx = append_solution_to_queue(solution_i)
            add_input_index_to(2, appended_idx)
    # Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_1_processor()
    # Finish code for explicit solution node 0
    
    # Code for non-branch dispatcher node 2
    # Actually, there is no code
    
    # Code for explicit solution node 3, solved variable is th_3
    def ExplicitSolutionNode_node_3_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[2]
        this_input_valid: bool = node_input_validity[2]
        if not this_input_valid:
            return
        
        # The solution of non-root node 3
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            condition_0: bool = (abs(a_4) >= zero_tolerance) or (abs(d_4) >= zero_tolerance) or (abs(Px*math.sin(th_1)*math.cos(th_0) + Py*math.sin(th_0)*math.sin(th_1) - Pz*math.cos(th_1) - a_1*math.sin(th_1) + d_2) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.atan2(a_4, d_4)
                x1 = math.sin(th_1)
                x2 = Px*x1*math.cos(th_0) + Py*x1*math.sin(th_0) - Pz*math.cos(th_1) - a_1*x1 + d_2
                x3 = safe_sqrt(a_4**2 + d_4**2 - x2**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[7] = x0 + math.atan2(x3, x2)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs(a_4) >= zero_tolerance) or (abs(d_4) >= zero_tolerance) or (abs(Px*math.sin(th_1)*math.cos(th_0) + Py*math.sin(th_0)*math.sin(th_1) - Pz*math.cos(th_1) - a_1*math.sin(th_1) + d_2) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.atan2(a_4, d_4)
                x1 = math.sin(th_1)
                x2 = Px*x1*math.cos(th_0) + Py*x1*math.sin(th_0) - Pz*math.cos(th_1) - a_1*x1 + d_2
                x3 = safe_sqrt(a_4**2 + d_4**2 - x2**2)
                # End of temp variables
                this_solution[7] = x0 + math.atan2(-x3, x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_3_processor()
    # Finish code for explicit solution node 2
    
    # Code for solved_variable dispatcher node 4
    def SolvedVariableDispatcherNode_node_4_processor():
        this_node_input_index: List[int] = node_input_index[4]
        this_input_valid: bool = node_input_validity[4]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_3 = this_solution[7]
            degenerate_valid_0 = (abs(th_3 - math.pi + 3.0801531643337) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(12, node_input_i_idx_in_queue)
            
            th_3 = this_solution[7]
            degenerate_valid_1 = (abs(th_3 - 3.33833913195303e-5 + math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(27, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(5, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_4_processor()
    # Finish code for solved_variable dispatcher node 4
    
    # Code for explicit solution node 27, solved variable is th_2
    def ExplicitSolutionNode_node_27_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[27]
        this_input_valid: bool = node_input_validity[27]
        if not this_input_valid:
            return
        
        # The solution of non-root node 27
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            condition_0: bool = (abs(Px*math.sin(th_0) - Py*math.cos(th_0)) >= zero_tolerance) or (abs(a_3 - 0.999999999442775*a_4 + 3.33833913133296e-5*d_4) >= zero_tolerance) or (abs(1.0*a_3 - 0.999999999442775*a_4 + 3.33833913133296e-5*d_4) >= zero_tolerance) or (abs(Px*math.cos(th_0)*math.cos(th_1) + Py*math.sin(th_0)*math.cos(th_1) + Pz*math.sin(th_1) - a_1*math.cos(th_1) - a_2) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                x2 = -0.999999999442775*a_4 + 3.33833913133296e-5*d_4
                x3 = math.cos(th_1)
                # End of temp variables
                this_solution[4] = math.atan2((-Px*x0 + Py*x1)/(1.0*a_3 + x2), (Px*x1*x3 + Py*x0*x3 + Pz*math.sin(th_1) - a_1*x3 - a_2)/(a_3 + x2))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(28, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_2_processor()
    # Finish code for explicit solution node 27
    
    # Code for non-branch dispatcher node 28
    # Actually, there is no code
    
    # Code for explicit solution node 29, solved variable is th_5
    def ExplicitSolutionNode_node_29_solve_th_5_processor():
        this_node_input_index: List[int] = node_input_index[28]
        this_input_valid: bool = node_input_validity[28]
        if not this_input_valid:
            return
        
        # The solution of non-root node 29
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            condition_0: bool = (abs(r_13*((0.999999999442775*math.sin(th_1) - 3.33833913133296e-5*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + 3.33833913133296e-5*math.sin(th_0)*math.sin(th_2)) + r_23*((0.999999999442775*math.sin(th_1) - 3.33833913133296e-5*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - 3.33833913133296e-5*math.sin(th_2)*math.cos(th_0)) - r_33*(3.33833913133296e-5*math.sin(th_1)*math.cos(th_2) + 0.999999999442775*math.cos(th_1))) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.sin(th_1)
                x2 = 3.33833913133296e-5*math.cos(th_2)
                x3 = math.sin(th_0)
                x4 = 3.33833913133296e-5*math.sin(th_2)
                x5 = math.cos(th_0)
                x6 = -x0*x2 + 0.999999999442775*x1
                x7 = safe_acos(r_13*(x3*x4 + x5*x6) + r_23*(x3*x6 - x4*x5) - r_33*(0.999999999442775*x0 + x1*x2))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[10] = x7
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(30, appended_idx)
                
            condition_1: bool = (abs(r_13*((0.999999999442775*math.sin(th_1) - 3.33833913133296e-5*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + 3.33833913133296e-5*math.sin(th_0)*math.sin(th_2)) + r_23*((0.999999999442775*math.sin(th_1) - 3.33833913133296e-5*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - 3.33833913133296e-5*math.sin(th_2)*math.cos(th_0)) - r_33*(3.33833913133296e-5*math.sin(th_1)*math.cos(th_2) + 0.999999999442775*math.cos(th_1))) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.sin(th_1)
                x2 = 3.33833913133296e-5*math.cos(th_2)
                x3 = math.sin(th_0)
                x4 = 3.33833913133296e-5*math.sin(th_2)
                x5 = math.cos(th_0)
                x6 = -x0*x2 + 0.999999999442775*x1
                x7 = safe_acos(r_13*(x3*x4 + x5*x6) + r_23*(x3*x6 - x4*x5) - r_33*(0.999999999442775*x0 + x1*x2))
                # End of temp variables
                this_solution[10] = -x7
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(30, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_5_processor()
    # Finish code for explicit solution node 28
    
    # Code for solved_variable dispatcher node 30
    def SolvedVariableDispatcherNode_node_30_processor():
        this_node_input_index: List[int] = node_input_index[30]
        this_input_valid: bool = node_input_validity[30]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_5 = this_solution[10]
            degenerate_valid_0 = (abs(th_5) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
            
            th_5 = this_solution[10]
            degenerate_valid_1 = (abs(th_5 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
            
            if not taken_by_degenerate:
                add_input_index_to(31, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_30_processor()
    # Finish code for solved_variable dispatcher node 30
    
    # Code for explicit solution node 31, solved variable is th_4
    def ExplicitSolutionNode_node_31_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[31]
        this_input_valid: bool = node_input_validity[31]
        if not this_input_valid:
            return
        
        # The solution of non-root node 31
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            th_5 = this_solution[10]
            condition_0: bool = (abs(r_13*(math.sin(th_0)*math.cos(th_2) + math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_23*(-math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_33*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance) or (abs(r_13*math.sin(th_1)*math.cos(th_0) + r_23*math.sin(th_0)*math.sin(th_1) - r_33*math.cos(th_1) - 0.999999999442775*math.cos(th_5)) >= zero_tolerance) or (3.33833913133296e-5*abs(math.sin(th_5)) >= zero_tolerance) or (abs(math.sin(th_5)) >= zero_tolerance)
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
                x8 = 29955.0153731898*x1
                # End of temp variables
                this_solution[8] = math.atan2(x0*(-r_13*(x3*x4 + x5*x7) + r_23*(-x3*x7 + x4*x5) - r_33*x1*x2), x0*(r_13*x5*x8 + r_23*x3*x8 - 29955.0153731898*r_33*x6 - 29955.0153564981*math.cos(th_5)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(32, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_4_processor()
    # Finish code for explicit solution node 31
    
    # Code for non-branch dispatcher node 32
    # Actually, there is no code
    
    # Code for explicit solution node 33, solved variable is th_2th_4th_5_soa
    def ExplicitSolutionNode_node_33_solve_th_2th_4th_5_soa_processor():
        this_node_input_index: List[int] = node_input_index[32]
        this_input_valid: bool = node_input_validity[32]
        if not this_input_valid:
            return
        
        # The solution of non-root node 33
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[4]
            th_4 = this_solution[8]
            th_5 = this_solution[10]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[5] = th_2 + th_4 + th_5
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(34, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_33_solve_th_2th_4th_5_soa_processor()
    # Finish code for explicit solution node 32
    
    # Code for non-branch dispatcher node 34
    # Actually, there is no code
    
    # Code for explicit solution node 35, solved variable is th_0th_2th_4_soa
    def ExplicitSolutionNode_node_35_solve_th_0th_2th_4_soa_processor():
        this_node_input_index: List[int] = node_input_index[34]
        this_input_valid: bool = node_input_validity[34]
        if not this_input_valid:
            return
        
        # The solution of non-root node 35
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[4]
            th_4 = this_solution[8]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[1] = th_0 + th_2 + th_4
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(36, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_35_solve_th_0th_2th_4_soa_processor()
    # Finish code for explicit solution node 34
    
    # Code for non-branch dispatcher node 36
    # Actually, there is no code
    
    # Code for explicit solution node 37, solved variable is th_1th_2th_4_soa
    def ExplicitSolutionNode_node_37_solve_th_1th_2th_4_soa_processor():
        this_node_input_index: List[int] = node_input_index[36]
        this_input_valid: bool = node_input_validity[36]
        if not this_input_valid:
            return
        
        # The solution of non-root node 37
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            th_4 = this_solution[8]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[3] = th_1 + th_2 + th_4
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(38, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_37_solve_th_1th_2th_4_soa_processor()
    # Finish code for explicit solution node 36
    
    # Code for non-branch dispatcher node 38
    # Actually, there is no code
    
    # Code for explicit solution node 39, solved variable is th_6
    def ExplicitSolutionNode_node_39_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[38]
        this_input_valid: bool = node_input_validity[38]
        if not this_input_valid:
            return
        
        # The solution of non-root node 39
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            th_4 = this_solution[8]
            th_5 = this_solution[10]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*(-(((3.33833913133296e-5*math.sin(th_5) + 0.999999999442775*math.cos(th_4)*math.cos(th_5))*math.cos(th_2) - math.sin(th_2)*math.sin(th_4)*math.cos(th_5))*math.cos(th_1) - (0.999999999442775*math.sin(th_5) - 3.33833913133296e-5*math.cos(th_4)*math.cos(th_5))*math.sin(th_1))*math.cos(th_0) + ((3.33833913133296e-5*math.sin(th_5) + 0.999999999442775*math.cos(th_4)*math.cos(th_5))*math.sin(th_2) + math.sin(th_4)*math.cos(th_2)*math.cos(th_5))*math.sin(th_0)) + r_21*(-(((3.33833913133296e-5*math.sin(th_5) + 0.999999999442775*math.cos(th_4)*math.cos(th_5))*math.cos(th_2) - math.sin(th_2)*math.sin(th_4)*math.cos(th_5))*math.cos(th_1) - (0.999999999442775*math.sin(th_5) - 3.33833913133296e-5*math.cos(th_4)*math.cos(th_5))*math.sin(th_1))*math.sin(th_0) - ((3.33833913133296e-5*math.sin(th_5) + 0.999999999442775*math.cos(th_4)*math.cos(th_5))*math.sin(th_2) + math.sin(th_4)*math.cos(th_2)*math.cos(th_5))*math.cos(th_0)) - r_31*(((3.33833913133296e-5*math.sin(th_5) + 0.999999999442775*math.cos(th_4)*math.cos(th_5))*math.cos(th_2) - math.sin(th_2)*math.sin(th_4)*math.cos(th_5))*math.sin(th_1) + (0.999999999442775*math.sin(th_5) - 3.33833913133296e-5*math.cos(th_4)*math.cos(th_5))*math.cos(th_1))) >= zero_tolerance) or (abs(r_12*(-(((3.33833913133296e-5*math.sin(th_5) + 0.999999999442775*math.cos(th_4)*math.cos(th_5))*math.cos(th_2) - math.sin(th_2)*math.sin(th_4)*math.cos(th_5))*math.cos(th_1) - (0.999999999442775*math.sin(th_5) - 3.33833913133296e-5*math.cos(th_4)*math.cos(th_5))*math.sin(th_1))*math.cos(th_0) + ((3.33833913133296e-5*math.sin(th_5) + 0.999999999442775*math.cos(th_4)*math.cos(th_5))*math.sin(th_2) + math.sin(th_4)*math.cos(th_2)*math.cos(th_5))*math.sin(th_0)) + r_22*(-(((3.33833913133296e-5*math.sin(th_5) + 0.999999999442775*math.cos(th_4)*math.cos(th_5))*math.cos(th_2) - math.sin(th_2)*math.sin(th_4)*math.cos(th_5))*math.cos(th_1) - (0.999999999442775*math.sin(th_5) - 3.33833913133296e-5*math.cos(th_4)*math.cos(th_5))*math.sin(th_1))*math.sin(th_0) - ((3.33833913133296e-5*math.sin(th_5) + 0.999999999442775*math.cos(th_4)*math.cos(th_5))*math.sin(th_2) + math.sin(th_4)*math.cos(th_2)*math.cos(th_5))*math.cos(th_0)) - r_32*(((3.33833913133296e-5*math.sin(th_5) + 0.999999999442775*math.cos(th_4)*math.cos(th_5))*math.cos(th_2) - math.sin(th_2)*math.sin(th_4)*math.cos(th_5))*math.sin(th_1) + (0.999999999442775*math.sin(th_5) - 3.33833913133296e-5*math.cos(th_4)*math.cos(th_5))*math.cos(th_1))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.sin(th_5)
                x2 = math.cos(th_4)
                x3 = math.cos(th_5)
                x4 = -0.999999999442775*x1 + 3.33833913133296e-5*x2*x3
                x5 = math.sin(th_1)
                x6 = math.sin(th_2)
                x7 = x3*math.sin(th_4)
                x8 = math.cos(th_2)
                x9 = -3.33833913133296e-5*x1 - 0.999999999442775*x2*x3
                x10 = x6*x7 + x8*x9
                x11 = x0*x4 + x10*x5
                x12 = math.cos(th_0)
                x13 = x6*x9 - x7*x8
                x14 = math.sin(th_0)
                x15 = x0*x10 - x4*x5
                x16 = x12*x13 + x14*x15
                x17 = x12*x15 - x13*x14
                # End of temp variables
                this_solution[11] = math.atan2(-r_12*x17 - r_22*x16 - r_32*x11, r_11*x17 + r_21*x16 + r_31*x11)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(40, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_39_solve_th_6_processor()
    # Finish code for explicit solution node 38
    
    # Code for non-branch dispatcher node 40
    # Actually, there is no code
    
    # Code for explicit solution node 41, solved variable is th_2th_4th_6_soa
    def ExplicitSolutionNode_node_41_solve_th_2th_4th_6_soa_processor():
        this_node_input_index: List[int] = node_input_index[40]
        this_input_valid: bool = node_input_validity[40]
        if not this_input_valid:
            return
        
        # The solution of non-root node 41
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[4]
            th_4 = this_solution[8]
            th_6 = this_solution[11]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[6] = th_2 + th_4 + th_6
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_41_solve_th_2th_4th_6_soa_processor()
    # Finish code for explicit solution node 40
    
    # Code for explicit solution node 12, solved variable is th_2
    def ExplicitSolutionNode_node_12_solve_th_2_processor():
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
            th_1 = this_solution[2]
            condition_0: bool = (abs(Px*math.sin(th_0) - Py*math.cos(th_0)) >= zero_tolerance) or (abs(a_3 + 0.998113188221811*a_4 - 0.06140084280929*d_4) >= zero_tolerance) or (abs(1.0*a_3 + 0.998113188221811*a_4 - 0.06140084280929*d_4) >= zero_tolerance) or (abs(Px*math.cos(th_0)*math.cos(th_1) + Py*math.sin(th_0)*math.cos(th_1) + Pz*math.sin(th_1) - a_1*math.cos(th_1) - a_2) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                x2 = 0.998113188221811*a_4 - 0.06140084280929*d_4
                x3 = math.cos(th_1)
                # End of temp variables
                this_solution[4] = math.atan2((-Px*x0 + Py*x1)/(1.0*a_3 + x2), (Px*x1*x3 + Py*x0*x3 + Pz*math.sin(th_1) - a_1*x3 - a_2)/(a_3 + x2))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(13, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_12_solve_th_2_processor()
    # Finish code for explicit solution node 12
    
    # Code for non-branch dispatcher node 13
    # Actually, there is no code
    
    # Code for explicit solution node 14, solved variable is th_5
    def ExplicitSolutionNode_node_14_solve_th_5_processor():
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
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            condition_0: bool = (abs(r_13*((0.998113188221811*math.sin(th_1) - 0.06140084280929*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + 0.06140084280929*math.sin(th_0)*math.sin(th_2)) + r_23*((0.998113188221811*math.sin(th_1) - 0.06140084280929*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - 0.06140084280929*math.sin(th_2)*math.cos(th_0)) - r_33*(0.06140084280929*math.sin(th_1)*math.cos(th_2) + 0.998113188221811*math.cos(th_1))) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.sin(th_1)
                x2 = 0.06140084280929*math.cos(th_2)
                x3 = math.sin(th_0)
                x4 = 0.06140084280929*math.sin(th_2)
                x5 = math.cos(th_0)
                x6 = -x0*x2 + 0.998113188221811*x1
                x7 = safe_acos(-r_13*(x3*x4 + x5*x6) - r_23*(x3*x6 - x4*x5) + r_33*(0.998113188221811*x0 + x1*x2))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[10] = x7
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(15, appended_idx)
                
            condition_1: bool = (abs(r_13*((0.998113188221811*math.sin(th_1) - 0.06140084280929*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + 0.06140084280929*math.sin(th_0)*math.sin(th_2)) + r_23*((0.998113188221811*math.sin(th_1) - 0.06140084280929*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - 0.06140084280929*math.sin(th_2)*math.cos(th_0)) - r_33*(0.06140084280929*math.sin(th_1)*math.cos(th_2) + 0.998113188221811*math.cos(th_1))) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.sin(th_1)
                x2 = 0.06140084280929*math.cos(th_2)
                x3 = math.sin(th_0)
                x4 = 0.06140084280929*math.sin(th_2)
                x5 = math.cos(th_0)
                x6 = -x0*x2 + 0.998113188221811*x1
                x7 = safe_acos(-r_13*(x3*x4 + x5*x6) - r_23*(x3*x6 - x4*x5) + r_33*(0.998113188221811*x0 + x1*x2))
                # End of temp variables
                this_solution[10] = -x7
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_5_processor()
    # Finish code for explicit solution node 13
    
    # Code for solved_variable dispatcher node 15
    def SolvedVariableDispatcherNode_node_15_processor():
        this_node_input_index: List[int] = node_input_index[15]
        this_input_valid: bool = node_input_validity[15]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_5 = this_solution[10]
            degenerate_valid_0 = (abs(th_5) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
            
            th_5 = this_solution[10]
            degenerate_valid_1 = (abs(th_5 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
            
            if not taken_by_degenerate:
                add_input_index_to(16, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_15_processor()
    # Finish code for solved_variable dispatcher node 15
    
    # Code for explicit solution node 16, solved variable is th_4
    def ExplicitSolutionNode_node_16_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[16]
        this_input_valid: bool = node_input_validity[16]
        if not this_input_valid:
            return
        
        # The solution of non-root node 16
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            th_5 = this_solution[10]
            condition_0: bool = (abs(r_13*(math.sin(th_0)*math.cos(th_2) + math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_23*(-math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_33*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance) or (abs(r_13*math.sin(th_1)*math.cos(th_0) + r_23*math.sin(th_0)*math.sin(th_1) - r_33*math.cos(th_1) + 0.998113188221811*math.cos(th_5)) >= zero_tolerance) or (0.06140084280929*abs(math.sin(th_5)) >= zero_tolerance) or (abs(math.sin(th_5)) >= zero_tolerance)
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
                x8 = 16.2864213949959*x1
                # End of temp variables
                this_solution[8] = math.atan2(x0*(-r_13*(x3*x4 + x5*x7) + r_23*(-x3*x7 + x4*x5) - r_33*x1*x2), x0*(-r_13*x5*x8 - r_23*x3*x8 + 16.2864213949959*r_33*x6 - 16.2556919832833*math.cos(th_5)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(17, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_4_processor()
    # Finish code for explicit solution node 16
    
    # Code for non-branch dispatcher node 17
    # Actually, there is no code
    
    # Code for explicit solution node 18, solved variable is th_2th_4th_5_soa
    def ExplicitSolutionNode_node_18_solve_th_2th_4th_5_soa_processor():
        this_node_input_index: List[int] = node_input_index[17]
        this_input_valid: bool = node_input_validity[17]
        if not this_input_valid:
            return
        
        # The solution of non-root node 18
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[4]
            th_4 = this_solution[8]
            th_5 = this_solution[10]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[5] = th_2 + th_4 + th_5
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(19, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_2th_4th_5_soa_processor()
    # Finish code for explicit solution node 17
    
    # Code for non-branch dispatcher node 19
    # Actually, there is no code
    
    # Code for explicit solution node 20, solved variable is th_0th_2th_4_soa
    def ExplicitSolutionNode_node_20_solve_th_0th_2th_4_soa_processor():
        this_node_input_index: List[int] = node_input_index[19]
        this_input_valid: bool = node_input_validity[19]
        if not this_input_valid:
            return
        
        # The solution of non-root node 20
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[4]
            th_4 = this_solution[8]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[1] = th_0 + th_2 + th_4
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(21, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_20_solve_th_0th_2th_4_soa_processor()
    # Finish code for explicit solution node 19
    
    # Code for non-branch dispatcher node 21
    # Actually, there is no code
    
    # Code for explicit solution node 22, solved variable is th_1th_2th_4_soa
    def ExplicitSolutionNode_node_22_solve_th_1th_2th_4_soa_processor():
        this_node_input_index: List[int] = node_input_index[21]
        this_input_valid: bool = node_input_validity[21]
        if not this_input_valid:
            return
        
        # The solution of non-root node 22
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            th_4 = this_solution[8]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[3] = th_1 + th_2 + th_4
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(23, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_22_solve_th_1th_2th_4_soa_processor()
    # Finish code for explicit solution node 21
    
    # Code for non-branch dispatcher node 23
    # Actually, there is no code
    
    # Code for explicit solution node 24, solved variable is th_6
    def ExplicitSolutionNode_node_24_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[23]
        this_input_valid: bool = node_input_validity[23]
        if not this_input_valid:
            return
        
        # The solution of non-root node 24
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            th_4 = this_solution[8]
            th_5 = this_solution[10]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*((((0.06140084280929*math.sin(th_5) + 0.998113188221811*math.cos(th_4)*math.cos(th_5))*math.cos(th_2) + math.sin(th_2)*math.sin(th_4)*math.cos(th_5))*math.cos(th_1) - (0.998113188221811*math.sin(th_5) - 0.06140084280929*math.cos(th_4)*math.cos(th_5))*math.sin(th_1))*math.cos(th_0) - ((0.06140084280929*math.sin(th_5) + 0.998113188221811*math.cos(th_4)*math.cos(th_5))*math.sin(th_2) - math.sin(th_4)*math.cos(th_2)*math.cos(th_5))*math.sin(th_0)) + r_21*((((0.06140084280929*math.sin(th_5) + 0.998113188221811*math.cos(th_4)*math.cos(th_5))*math.cos(th_2) + math.sin(th_2)*math.sin(th_4)*math.cos(th_5))*math.cos(th_1) - (0.998113188221811*math.sin(th_5) - 0.06140084280929*math.cos(th_4)*math.cos(th_5))*math.sin(th_1))*math.sin(th_0) + ((0.06140084280929*math.sin(th_5) + 0.998113188221811*math.cos(th_4)*math.cos(th_5))*math.sin(th_2) - math.sin(th_4)*math.cos(th_2)*math.cos(th_5))*math.cos(th_0)) + r_31*(((0.06140084280929*math.sin(th_5) + 0.998113188221811*math.cos(th_4)*math.cos(th_5))*math.cos(th_2) + math.sin(th_2)*math.sin(th_4)*math.cos(th_5))*math.sin(th_1) + (0.998113188221811*math.sin(th_5) - 0.06140084280929*math.cos(th_4)*math.cos(th_5))*math.cos(th_1))) >= zero_tolerance) or (abs(r_12*((((0.06140084280929*math.sin(th_5) + 0.998113188221811*math.cos(th_4)*math.cos(th_5))*math.cos(th_2) + math.sin(th_2)*math.sin(th_4)*math.cos(th_5))*math.cos(th_1) - (0.998113188221811*math.sin(th_5) - 0.06140084280929*math.cos(th_4)*math.cos(th_5))*math.sin(th_1))*math.cos(th_0) - ((0.06140084280929*math.sin(th_5) + 0.998113188221811*math.cos(th_4)*math.cos(th_5))*math.sin(th_2) - math.sin(th_4)*math.cos(th_2)*math.cos(th_5))*math.sin(th_0)) + r_22*((((0.06140084280929*math.sin(th_5) + 0.998113188221811*math.cos(th_4)*math.cos(th_5))*math.cos(th_2) + math.sin(th_2)*math.sin(th_4)*math.cos(th_5))*math.cos(th_1) - (0.998113188221811*math.sin(th_5) - 0.06140084280929*math.cos(th_4)*math.cos(th_5))*math.sin(th_1))*math.sin(th_0) + ((0.06140084280929*math.sin(th_5) + 0.998113188221811*math.cos(th_4)*math.cos(th_5))*math.sin(th_2) - math.sin(th_4)*math.cos(th_2)*math.cos(th_5))*math.cos(th_0)) + r_32*(((0.06140084280929*math.sin(th_5) + 0.998113188221811*math.cos(th_4)*math.cos(th_5))*math.cos(th_2) + math.sin(th_2)*math.sin(th_4)*math.cos(th_5))*math.sin(th_1) + (0.998113188221811*math.sin(th_5) - 0.06140084280929*math.cos(th_4)*math.cos(th_5))*math.cos(th_1))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.sin(th_5)
                x2 = math.cos(th_5)
                x3 = x2*math.cos(th_4)
                x4 = 0.998113188221811*x1 - 0.06140084280929*x3
                x5 = math.sin(th_1)
                x6 = math.sin(th_2)
                x7 = x2*math.sin(th_4)
                x8 = math.cos(th_2)
                x9 = 0.06140084280929*x1 + 0.998113188221811*x3
                x10 = x6*x7 + x8*x9
                x11 = x0*x4 + x10*x5
                x12 = math.cos(th_0)
                x13 = x6*x9 - x7*x8
                x14 = math.sin(th_0)
                x15 = x0*x10 - x4*x5
                x16 = x12*x13 + x14*x15
                x17 = x12*x15 - x13*x14
                # End of temp variables
                this_solution[11] = math.atan2(-r_12*x17 - r_22*x16 - r_32*x11, r_11*x17 + r_21*x16 + r_31*x11)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(25, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_6_processor()
    # Finish code for explicit solution node 23
    
    # Code for non-branch dispatcher node 25
    # Actually, there is no code
    
    # Code for explicit solution node 26, solved variable is th_2th_4th_6_soa
    def ExplicitSolutionNode_node_26_solve_th_2th_4th_6_soa_processor():
        this_node_input_index: List[int] = node_input_index[25]
        this_input_valid: bool = node_input_validity[25]
        if not this_input_valid:
            return
        
        # The solution of non-root node 26
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[4]
            th_4 = this_solution[8]
            th_6 = this_solution[11]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[6] = th_2 + th_4 + th_6
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_26_solve_th_2th_4th_6_soa_processor()
    # Finish code for explicit solution node 25
    
    # Code for explicit solution node 5, solved variable is th_2
    def ExplicitSolutionNode_node_5_solve_th_2_processor():
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
            th_3 = this_solution[7]
            condition_0: bool = (abs((Px*math.sin(th_0) - Py*math.cos(th_0))/(a_3 + a_4*math.cos(th_3) - d_4*math.sin(th_3))) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = safe_asin((-Px*math.sin(th_0) + Py*math.cos(th_0))/(a_3 + a_4*math.cos(th_3) - d_4*math.sin(th_3)))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                
            condition_1: bool = (abs((Px*math.sin(th_0) - Py*math.cos(th_0))/(a_3 + a_4*math.cos(th_3) - d_4*math.sin(th_3))) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = safe_asin((-Px*math.sin(th_0) + Py*math.cos(th_0))/(a_3 + a_4*math.cos(th_3) - d_4*math.sin(th_3)))
                # End of temp variables
                this_solution[4] = math.pi - x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_2_processor()
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
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            th_3 = this_solution[7]
            condition_0: bool = (abs(r_13*((math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + math.sin(th_0)*math.sin(th_2)*math.sin(th_3)) + r_23*((math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - math.sin(th_2)*math.sin(th_3)*math.cos(th_0)) - r_33*(math.sin(th_1)*math.sin(th_3)*math.cos(th_2) + math.cos(th_1)*math.cos(th_3))) <= 1)
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
                x8 = -x0*x4 + x1*x2
                x9 = safe_acos(-r_13*(x5*x6 + x7*x8) - r_23*(x5*x8 - x6*x7) + r_33*(x0*x1 + x2*x4))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[10] = x9
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
            condition_1: bool = (abs(r_13*((math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + math.sin(th_0)*math.sin(th_2)*math.sin(th_3)) + r_23*((math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - math.sin(th_2)*math.sin(th_3)*math.cos(th_0)) - r_33*(math.sin(th_1)*math.sin(th_3)*math.cos(th_2) + math.cos(th_1)*math.cos(th_3))) <= 1)
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
                x8 = -x0*x4 + x1*x2
                x9 = safe_acos(-r_13*(x5*x6 + x7*x8) - r_23*(x5*x8 - x6*x7) + r_33*(x0*x1 + x2*x4))
                # End of temp variables
                this_solution[10] = -x9
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
            th_5 = this_solution[10]
            degenerate_valid_0 = (abs(th_5) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(42, node_input_i_idx_in_queue)
            
            th_5 = this_solution[10]
            degenerate_valid_1 = (abs(th_5 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(47, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(9, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor()
    # Finish code for solved_variable dispatcher node 8
    
    # Code for explicit solution node 47, solved variable is th_4th_6_soa
    def ExplicitSolutionNode_node_47_solve_th_4th_6_soa_processor():
        this_node_input_index: List[int] = node_input_index[47]
        this_input_valid: bool = node_input_validity[47]
        if not this_input_valid:
            return
        
        # The solution of non-root node 47
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*(math.sin(th_0)*math.cos(th_2) + math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_21*(-math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_31*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance) or (abs(r_12*(math.sin(th_0)*math.cos(th_2) + math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_22*(-math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_32*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_2)
                x1 = x0*math.sin(th_1)
                x2 = math.sin(th_0)
                x3 = math.cos(th_2)
                x4 = math.cos(th_0)
                x5 = x0*math.cos(th_1)
                x6 = x2*x3 + x4*x5
                x7 = -x2*x5 + x3*x4
                # End of temp variables
                this_solution[9] = math.atan2(-r_11*x6 + r_21*x7 - r_31*x1, -r_12*x6 + r_22*x7 - r_32*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(48, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_47_solve_th_4th_6_soa_processor()
    # Finish code for explicit solution node 47
    
    # Code for non-branch dispatcher node 48
    # Actually, there is no code
    
    # Code for explicit solution node 49, solved variable is th_4
    def ExplicitSolutionNode_node_49_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[48]
        this_input_valid: bool = node_input_validity[48]
        if not this_input_valid:
            return
        
        # The solution of non-root node 49
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[8] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(50, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_49_solve_th_4_processor()
    # Finish code for explicit solution node 48
    
    # Code for non-branch dispatcher node 50
    # Actually, there is no code
    
    # Code for explicit solution node 51, solved variable is th_6
    def ExplicitSolutionNode_node_51_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[50]
        this_input_valid: bool = node_input_validity[50]
        if not this_input_valid:
            return
        
        # The solution of non-root node 51
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_4 = this_solution[8]
            th_4th_6_soa = this_solution[9]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[11] = -th_4 + th_4th_6_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_51_solve_th_6_processor()
    # Finish code for explicit solution node 50
    
    # Code for explicit solution node 42, solved variable is negative_th_6_positive_th_4__soa
    def ExplicitSolutionNode_node_42_solve_negative_th_6_positive_th_4__soa_processor():
        this_node_input_index: List[int] = node_input_index[42]
        this_input_valid: bool = node_input_validity[42]
        if not this_input_valid:
            return
        
        # The solution of non-root node 42
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*(math.sin(th_0)*math.cos(th_2) + math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_21*(-math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_31*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance) or (abs(r_12*(math.sin(th_0)*math.cos(th_2) + math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_22*(-math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_32*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_2)
                x1 = x0*math.sin(th_1)
                x2 = math.sin(th_0)
                x3 = math.cos(th_2)
                x4 = math.cos(th_0)
                x5 = x0*math.cos(th_1)
                x6 = x2*x3 + x4*x5
                x7 = -x2*x5 + x3*x4
                # End of temp variables
                this_solution[0] = math.atan2(r_11*x6 - r_21*x7 + r_31*x1, -r_12*x6 + r_22*x7 - r_32*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(43, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_42_solve_negative_th_6_positive_th_4__soa_processor()
    # Finish code for explicit solution node 42
    
    # Code for non-branch dispatcher node 43
    # Actually, there is no code
    
    # Code for explicit solution node 44, solved variable is th_4
    def ExplicitSolutionNode_node_44_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[43]
        this_input_valid: bool = node_input_validity[43]
        if not this_input_valid:
            return
        
        # The solution of non-root node 44
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[8] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(45, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_44_solve_th_4_processor()
    # Finish code for explicit solution node 43
    
    # Code for non-branch dispatcher node 45
    # Actually, there is no code
    
    # Code for explicit solution node 46, solved variable is th_6
    def ExplicitSolutionNode_node_46_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[45]
        this_input_valid: bool = node_input_validity[45]
        if not this_input_valid:
            return
        
        # The solution of non-root node 46
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            negative_th_6_positive_th_4__soa = this_solution[0]
            th_4 = this_solution[8]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[11] = -negative_th_6_positive_th_4__soa + th_4
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_46_solve_th_6_processor()
    # Finish code for explicit solution node 45
    
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
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            th_3 = this_solution[7]
            th_5 = this_solution[10]
            condition_0: bool = (abs(r_13*((math.sin(th_1)*math.sin(th_3) + math.cos(th_1)*math.cos(th_2)*math.cos(th_3))*math.cos(th_0) - math.sin(th_0)*math.sin(th_2)*math.cos(th_3)) + r_23*((math.sin(th_1)*math.sin(th_3) + math.cos(th_1)*math.cos(th_2)*math.cos(th_3))*math.sin(th_0) + math.sin(th_2)*math.cos(th_0)*math.cos(th_3)) + r_33*(math.sin(th_1)*math.cos(th_2)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1))) >= zero_tolerance) or (abs(r_13*(math.sin(th_0)*math.cos(th_2) + math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_23*(-math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_33*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance) or (abs(math.sin(th_5)) >= zero_tolerance)
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
                this_solution[8] = math.atan2(x0*(-r_13*(x3*x4 + x5*x7) + r_23*(-x3*x7 + x4*x5) - r_33*x1*x2), x0*(-r_13*(-x11*x3 + x12*x5) - r_23*(x11*x5 + x12*x3) - r_33*(x1*x10 - x6*x8)))
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
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            th_3 = this_solution[7]
            th_5 = this_solution[10]
            condition_0: bool = (abs(-r_11*((-math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) - math.sin(th_0)*math.sin(th_2)*math.sin(th_3)) - r_21*((-math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) + math.sin(th_2)*math.sin(th_3)*math.cos(th_0)) - r_31*(math.sin(th_1)*math.sin(th_3)*math.cos(th_2) + math.cos(th_1)*math.cos(th_3))) >= zero_tolerance) or (abs(r_12*((math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + math.sin(th_0)*math.sin(th_2)*math.sin(th_3)) + r_22*((math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - math.sin(th_2)*math.sin(th_3)*math.cos(th_0)) - r_32*(math.sin(th_1)*math.sin(th_3)*math.cos(th_2) + math.cos(th_1)*math.cos(th_3))) >= zero_tolerance) or (abs(math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_5)**(-1)
                x1 = math.cos(th_1)
                x2 = math.cos(th_3)
                x3 = math.sin(th_1)
                x4 = math.sin(th_3)
                x5 = x4*math.cos(th_2)
                x6 = x1*x2 + x3*x5
                x7 = math.sin(th_0)
                x8 = x4*math.sin(th_2)
                x9 = x7*x8
                x10 = math.cos(th_0)
                x11 = -x1*x5 + x2*x3
                x12 = x10*x8
                x13 = -x11
                # End of temp variables
                this_solution[11] = math.atan2(x0*(r_12*(x10*x11 + x9) + r_22*(x11*x7 - x12) - r_32*x6), x0*(r_11*(x10*x13 - x9) + r_21*(x12 + x13*x7) + r_31*x6))
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
        value_at_1 = ik_out_i[2]  # th_1
        new_ik_i[1] = value_at_1
        value_at_2 = ik_out_i[4]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[7]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[8]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[10]  # th_5
        new_ik_i[5] = value_at_5
        value_at_6 = ik_out_i[11]  # th_6
        new_ik_i[6] = value_at_6
        ik_out.append(new_ik_i)
    return ik_out


def atlas_l_hand_ik_solve(T_ee: np.ndarray, th_0):
    T_ee_raw_in = atlas_l_hand_ik_target_original_to_raw(T_ee)
    ik_output_raw = atlas_l_hand_ik_solve_raw(T_ee_raw_in, th_0 + th_0_offset_original2raw)
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
        ee_pose_i = atlas_l_hand_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_atlas_l_hand():
    theta_in = np.random.random(size=(7, ))
    ee_pose = atlas_l_hand_fk(theta_in)
    ik_output = atlas_l_hand_ik_solve(ee_pose, th_0=theta_in[0])
    for i in range(len(ik_output)):
        ee_pose_i = atlas_l_hand_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_atlas_l_hand()
