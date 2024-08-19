import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 28
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 0.15
a_1: float = 0.707
a_2: float = 0.11
a_4: float = 0.08
d_3: float = 0.637
d_5: float = 0.101
pre_transform_special_symbol_23: float = 0.4

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -1.5707963267948966
th_2_offset_original2raw: float = 0.0
th_3_offset_original2raw: float = 3.141592653589793
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 0.0


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def abb_crb15000_10_1_52_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = 1.0*r_13
    ee_transformed[0, 1] = -1.0*r_12
    ee_transformed[0, 2] = 1.0*r_11
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = 1.0*r_23
    ee_transformed[1, 1] = -1.0*r_22
    ee_transformed[1, 2] = 1.0*r_21
    ee_transformed[1, 3] = 1.0*Py
    ee_transformed[2, 0] = 1.0*r_33
    ee_transformed[2, 1] = -1.0*r_32
    ee_transformed[2, 2] = 1.0*r_31
    ee_transformed[2, 3] = 1.0*Pz - 1.0*pre_transform_special_symbol_23
    return ee_transformed


def abb_crb15000_10_1_52_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = 1.0*r_13
    ee_transformed[0, 1] = -1.0*r_12
    ee_transformed[0, 2] = 1.0*r_11
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = 1.0*r_23
    ee_transformed[1, 1] = -1.0*r_22
    ee_transformed[1, 2] = 1.0*r_21
    ee_transformed[1, 3] = 1.0*Py
    ee_transformed[2, 0] = 1.0*r_33
    ee_transformed[2, 1] = -1.0*r_32
    ee_transformed[2, 2] = 1.0*r_31
    ee_transformed[2, 3] = 1.0*Pz + 1.0*pre_transform_special_symbol_23
    return ee_transformed


def abb_crb15000_10_1_52_fk(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.cos(th_4)
    x1 = math.cos(th_0)
    x2 = math.sin(th_1)
    x3 = math.cos(th_2)
    x4 = x2*x3
    x5 = math.sin(th_2)
    x6 = math.cos(th_1)
    x7 = x5*x6
    x8 = -x1*x4 - x1*x7
    x9 = x0*x8
    x10 = math.sin(th_4)
    x11 = math.sin(th_0)
    x12 = math.sin(th_3)
    x13 = math.cos(th_3)
    x14 = x2*x5
    x15 = -x1*x14 + x1*x3*x6
    x16 = x11*x12 + x13*x15
    x17 = x10*x16
    x18 = math.cos(th_5)
    x19 = 1.0*x11*x13 - 1.0*x12*x15
    x20 = math.sin(th_5)
    x21 = 1.0*x0*x16 - 1.0*x10*x8
    x22 = 1.0*a_0
    x23 = 1.0*a_1
    x24 = x23*x6
    x25 = 1.0*a_2
    x26 = 1.0*d_3
    x27 = 1.0*d_5
    x28 = -x11*x4 - x11*x7
    x29 = x0*x28
    x30 = -x11*x14 + x11*x3*x6
    x31 = -x1*x12 + x13*x30
    x32 = x10*x31
    x33 = -1.0*x1*x13 - 1.0*x12*x30
    x34 = 1.0*x0*x31 - 1.0*x10*x28
    x35 = x14 - x3*x6
    x36 = x0*x35
    x37 = -x4 - x7
    x38 = x10*x13*x37
    x39 = 1.0*x12*x37
    x40 = 1.0*x0*x13*x37 - 1.0*x10*x35
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -1.0*x17 - 1.0*x9
    ee_pose[0, 1] = x18*x19 + x20*x21
    ee_pose[0, 2] = x18*x21 - x19*x20
    ee_pose[0, 3] = a_4*x21 + x1*x22 + x1*x24 + x15*x25 + x26*x8 + x27*(-x17 - x9)
    ee_pose[1, 0] = -1.0*x29 - 1.0*x32
    ee_pose[1, 1] = x18*x33 + x20*x34
    ee_pose[1, 2] = x18*x34 - x20*x33
    ee_pose[1, 3] = a_4*x34 + x11*x22 + x11*x24 + x25*x30 + x26*x28 + x27*(-x29 - x32)
    ee_pose[2, 0] = -1.0*x36 - 1.0*x38
    ee_pose[2, 1] = -x18*x39 + x20*x40
    ee_pose[2, 2] = x18*x40 + x20*x39
    ee_pose[2, 3] = a_4*x40 + 1.0*pre_transform_special_symbol_23 - x2*x23 + x25*x37 + x26*x35 + x27*(-x36 - x38)
    return ee_pose


def abb_crb15000_10_1_52_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = 1.0*x0
    x2 = -x1
    x3 = math.cos(th_2)
    x4 = math.sin(th_1)
    x5 = math.cos(th_0)
    x6 = 1.0*x5
    x7 = x4*x6
    x8 = math.sin(th_2)
    x9 = math.cos(th_1)
    x10 = x6*x9
    x11 = -x10*x8 - x3*x7
    x12 = math.cos(th_3)
    x13 = math.sin(th_3)
    x14 = 1.0*x3*x5*x9 - x7*x8
    x15 = 1.0*x0*x12 - x13*x14
    x16 = math.cos(th_4)
    x17 = math.sin(th_4)
    x18 = x1*x13 + x12*x14
    x19 = -x11*x16 - x17*x18
    x20 = x1*x4
    x21 = x1*x9
    x22 = -x20*x3 - x21*x8
    x23 = 1.0*x0*x3*x9 - x20*x8
    x24 = -x12*x6 - x13*x23
    x25 = x12*x23 - x13*x6
    x26 = -x16*x22 - x17*x25
    x27 = 1.0*x8
    x28 = 1.0*x3
    x29 = x27*x4 - x28*x9
    x30 = -x27*x9 - x28*x4
    x31 = x13*x30
    x32 = -x12*x17*x30 - x16*x29
    x33 = 1.0*a_1*x4
    x34 = pre_transform_special_symbol_23 - x33
    x35 = a_2*x30 + d_3*x29 + pre_transform_special_symbol_23 - x33
    x36 = a_0*x1 + a_1*x21
    x37 = a_2*x23 + d_3*x22 + x36
    x38 = a_4*(x12*x16*x30 - x17*x29) + d_5*x32 + x35
    x39 = a_4*(x16*x25 - x17*x22) + d_5*x26 + x37
    x40 = a_0*x6 + a_1*x10
    x41 = a_2*x14 + d_3*x11 + x40
    x42 = a_4*(-x11*x17 + x16*x18) + d_5*x19 + x41
    x43 = 1.0*a_0
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = x2
    jacobian_output[0, 2] = x2
    jacobian_output[0, 3] = x11
    jacobian_output[0, 4] = x15
    jacobian_output[0, 5] = x19
    jacobian_output[1, 1] = x6
    jacobian_output[1, 2] = x6
    jacobian_output[1, 3] = x22
    jacobian_output[1, 4] = x24
    jacobian_output[1, 5] = x26
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x29
    jacobian_output[2, 4] = -x31
    jacobian_output[2, 5] = x32
    jacobian_output[3, 1] = -pre_transform_special_symbol_23*x6
    jacobian_output[3, 2] = -x34*x6
    jacobian_output[3, 3] = -x22*x35 + x29*x37
    jacobian_output[3, 4] = -x24*x35 - x31*x37
    jacobian_output[3, 5] = -x26*x38 + x32*x39
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = -x1*x34
    jacobian_output[4, 3] = x11*x35 - x29*x41
    jacobian_output[4, 4] = x15*x35 + x31*x41
    jacobian_output[4, 5] = x19*x38 - x32*x42
    jacobian_output[5, 1] = x0**2*x43 + x43*x5**2
    jacobian_output[5, 2] = x1*x36 + x40*x6
    jacobian_output[5, 3] = -x11*x37 + x22*x41
    jacobian_output[5, 4] = -x15*x37 + x24*x41
    jacobian_output[5, 5] = -x19*x39 + x26*x42
    return jacobian_output


def abb_crb15000_10_1_52_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = 1.0*x0
    x2 = -x1
    x3 = math.cos(th_2)
    x4 = math.sin(th_1)
    x5 = math.cos(th_0)
    x6 = 1.0*x5
    x7 = x4*x6
    x8 = math.sin(th_2)
    x9 = math.cos(th_1)
    x10 = -x3*x7 - x6*x8*x9
    x11 = math.cos(th_3)
    x12 = math.sin(th_3)
    x13 = 1.0*x3*x5*x9 - x7*x8
    x14 = math.cos(th_4)
    x15 = math.sin(th_4)
    x16 = x1*x4
    x17 = -x1*x8*x9 - x16*x3
    x18 = 1.0*x0*x3*x9 - x16*x8
    x19 = 1.0*x8
    x20 = 1.0*x3
    x21 = x19*x4 - x20*x9
    x22 = -x19*x9 - x20*x4
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 1] = x2
    jacobian_output[0, 2] = x2
    jacobian_output[0, 3] = x10
    jacobian_output[0, 4] = 1.0*x0*x11 - x12*x13
    jacobian_output[0, 5] = -x10*x14 - x15*(x1*x12 + x11*x13)
    jacobian_output[1, 1] = x6
    jacobian_output[1, 2] = x6
    jacobian_output[1, 3] = x17
    jacobian_output[1, 4] = -x11*x6 - x12*x18
    jacobian_output[1, 5] = -x14*x17 - x15*(x11*x18 - x12*x6)
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x21
    jacobian_output[2, 4] = -x12*x22
    jacobian_output[2, 5] = -x11*x15*x22 - x14*x21
    return jacobian_output


def abb_crb15000_10_1_52_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    p_on_ee_x: float = point_on_ee[0]
    p_on_ee_y: float = point_on_ee[1]
    p_on_ee_z: float = point_on_ee[2]

    # Temp variable for efficiency
    x0 = 1.0*p_on_ee_y
    x1 = math.cos(th_0)
    x2 = 1.0*x1
    x3 = p_on_ee_z*x2
    x4 = math.sin(th_1)
    x5 = 1.0*x4
    x6 = a_1*x5
    x7 = pre_transform_special_symbol_23 - x6
    x8 = math.sin(th_2)
    x9 = x5*x8
    x10 = math.cos(th_2)
    x11 = math.cos(th_1)
    x12 = 1.0*x11
    x13 = -x10*x12 + x9
    x14 = math.sin(th_0)
    x15 = x10*x5
    x16 = x12*x8
    x17 = -x14*x15 - x14*x16
    x18 = -x15 - x16
    x19 = a_2*x18 + d_3*x13 + pre_transform_special_symbol_23 - x6
    x20 = 1.0*x10*x11*x14 - x14*x9
    x21 = 1.0*x14
    x22 = a_0*x21 + a_1*x12*x14
    x23 = a_2*x20 + d_3*x17 + x22
    x24 = math.sin(th_3)
    x25 = x18*x24
    x26 = math.cos(th_3)
    x27 = -x2*x26 - x20*x24
    x28 = math.cos(th_4)
    x29 = math.sin(th_4)
    x30 = -x13*x28 - x18*x26*x29
    x31 = -x2*x24 + x20*x26
    x32 = -x17*x28 - x29*x31
    x33 = a_4*(-x13*x29 + x18*x26*x28) + d_5*x30 + x19
    x34 = a_4*(-x17*x29 + x28*x31) + d_5*x32 + x23
    x35 = 1.0*p_on_ee_x
    x36 = p_on_ee_z*x21
    x37 = x2*x4
    x38 = x11*x2
    x39 = -x10*x37 - x38*x8
    x40 = 1.0*x1*x10*x11 - x37*x8
    x41 = a_0*x2 + a_1*x38
    x42 = a_2*x40 + d_3*x39 + x41
    x43 = 1.0*x14*x26 - x24*x40
    x44 = x21*x24 + x26*x40
    x45 = -x28*x39 - x29*x44
    x46 = a_4*(x28*x44 - x29*x39) + d_5*x45 + x42
    x47 = x1*x35
    x48 = x0*x14
    x49 = 1.0*a_0
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = -x2*x7 + x3
    jacobian_output[0, 3] = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x23 - x17*x19
    jacobian_output[0, 4] = p_on_ee_y*x25 + p_on_ee_z*x27 - x19*x27 - x23*x25
    jacobian_output[0, 5] = -p_on_ee_y*x30 + p_on_ee_z*x32 + x30*x34 - x32*x33
    jacobian_output[1, 0] = x35
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x21 + x36
    jacobian_output[1, 2] = -x21*x7 + x36
    jacobian_output[1, 3] = p_on_ee_x*x13 - p_on_ee_z*x39 - x13*x42 + x19*x39
    jacobian_output[1, 4] = -p_on_ee_x*x25 - p_on_ee_z*x43 + x18*x24*x42 + x19*x43
    jacobian_output[1, 5] = p_on_ee_x*x30 - p_on_ee_z*x45 - x30*x46 + x33*x45
    jacobian_output[2, 1] = x1**2*x49 + x14**2*x49 - x47 - x48
    jacobian_output[2, 2] = 1.0*x1*x41 + 1.0*x14*x22 - x47 - x48
    jacobian_output[2, 3] = -p_on_ee_x*x17 + p_on_ee_y*x39 + x17*x42 - x23*x39
    jacobian_output[2, 4] = -p_on_ee_x*x27 + p_on_ee_y*x43 - x23*x43 + x27*x42
    jacobian_output[2, 5] = -p_on_ee_x*x32 + p_on_ee_y*x45 + x32*x46 - x34*x45
    return jacobian_output


def abb_crb15000_10_1_52_ik_solve_raw(T_ee: np.ndarray):
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
    for i in range(28):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True
    
    # Code for non-branch dispatcher node 0
    # Actually, there is no code
    
    # Code for explicit solution node 1, solved variable is th_5
    def General6DoFNumericalReduceSolutionNode_node_1_solve_th_5_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return
        
        # The general 6-dof solution of root node with semi-symbolic reduce
        R_l = np.zeros(shape=(8, 8))
        R_l[0, 3] = -a_1
        R_l[0, 7] = -a_2
        R_l[1, 2] = -a_1
        R_l[1, 6] = -a_2
        R_l[2, 4] = a_1
        R_l[3, 6] = -1
        R_l[4, 7] = 1
        R_l[5, 5] = 2*a_1*a_2
        R_l[6, 1] = -a_1
        R_l[7, 0] = -a_1
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
        x0 = -r_23
        x1 = 2*r_21
        x2 = -x1
        x3 = 2*r_13
        x4 = -x3
        x5 = 4*r_11
        x6 = d_3*r_23
        x7 = -x6
        x8 = a_4*r_21
        x9 = d_5*r_23
        x10 = Px*r_11
        x11 = r_21*x10
        x12 = Px*r_12
        x13 = r_22*x12
        x14 = Px*r_13
        x15 = r_23*x14
        x16 = Pz*r_31
        x17 = r_21*x16
        x18 = Pz*r_32
        x19 = r_22*x18
        x20 = Pz*r_33
        x21 = r_23*x20
        x22 = r_21**2
        x23 = Py*x22
        x24 = r_22**2
        x25 = Py*x24
        x26 = r_23**2
        x27 = Py*x26
        x28 = x11 + x13 + x15 + x17 + x19 + x21 + x23 + x25 + x27 - x9
        x29 = x28 + x8
        x30 = x29 + x7
        x31 = a_0*r_11
        x32 = r_21*x31
        x33 = a_0*r_12
        x34 = r_22*x33
        x35 = a_0*r_13
        x36 = r_23*x35
        x37 = -x32 - x34 - x36
        x38 = d_3*x1
        x39 = -x38
        x40 = x29 + x6
        x41 = d_3*x3
        x42 = -x41
        x43 = 2*a_4
        x44 = r_11*x43
        x45 = d_5*x3
        x46 = r_11**2
        x47 = Px*x46
        x48 = 2*x47
        x49 = r_12**2
        x50 = Px*x49
        x51 = 2*x50
        x52 = r_13**2
        x53 = Px*x52
        x54 = 2*x53
        x55 = Py*x1
        x56 = Py*r_22
        x57 = r_12*x56
        x58 = Py*r_23
        x59 = x3*x58
        x60 = r_11*x16
        x61 = r_12*x18
        x62 = x20*x3
        x63 = r_11*x55 - x45 + x48 + x51 + x54 + 2*x57 + x59 + 2*x60 + 2*x61 + x62
        x64 = x44 + x63
        x65 = d_3*x5
        x66 = x32 + x34 + x36
        x67 = -a_2
        x68 = a_4*r_22
        x69 = Py*r_21
        x70 = x10 + x16 + x69
        x71 = R_l_inv_50*a_1
        x72 = x70*x71
        x73 = a_1*r_21
        x74 = R_l_inv_53*x73
        x75 = d_5*r_22
        x76 = r_23*x12
        x77 = r_23*x18
        x78 = r_22*x14
        x79 = r_22*x20
        x80 = x75 + x76 + x77 - x78 - x79
        x81 = R_l_inv_56*a_1
        x82 = x80*x81
        x83 = -R_l_inv_52*a_1*d_3
        x84 = a_4**2
        x85 = d_3**2
        x86 = d_5**2
        x87 = a_1**2
        x88 = a_2**2
        x89 = 2*x9
        x90 = 2*d_5
        x91 = 2*x12
        x92 = 2*x10
        x93 = 2*x18
        x94 = 2*x20
        x95 = Px**2
        x96 = x46*x95
        x97 = x49*x95
        x98 = x52*x95
        x99 = Py**2
        x100 = x22*x99
        x101 = x24*x99
        x102 = x26*x99
        x103 = Pz**2
        x104 = r_31**2
        x105 = x103*x104
        x106 = r_32**2
        x107 = x103*x106
        x108 = r_33**2*x103
        x109 = a_0**2
        x110 = x109*x22
        x111 = x109*x24
        x112 = x109*x26
        x113 = -Px*x45 + Px*x59 + Px*x62 - Py*x89 + x10*x55 + x100 + x101 + x102 + x105 + x107 + x108 + x110 + x111 + x112 + x16*x55 + x16*x92 + x18*x91 - x20*x90 + x56*x91 + x56*x93 + x58*x94 + x84 + x85 + x86 - x87 - x88 + x96 + x97 + x98
        x114 = -R_l_inv_55*a_1*x113
        x115 = x31*x71
        x116 = r_12*r_23
        x117 = r_13*r_22
        x118 = x116 - x117
        x119 = a_0*x118
        x120 = x119*x81
        x121 = R_l_inv_57*a_1
        x122 = a_4*r_23
        x123 = x121*x122
        x124 = R_l_inv_57*d_3*x73
        x125 = R_l_inv_55*a_1
        x126 = d_5*r_13
        x127 = r_11*x69
        x128 = r_13*x58
        x129 = r_13*x20
        x130 = 2*a_0
        x131 = x130*(-x126 + x127 + x128 + x129 + x47 + x50 + x53 + x57 + x60 + x61)
        x132 = x125*x131
        x133 = -x132
        x134 = x114 + x115 + x120 + x123 + x124 + x133 + x67 + x68 + x72 + x74 + x82 + x83
        x135 = a_4*x71
        x136 = -d_5 + x14 + x20 + x58
        x137 = R_l_inv_52*a_1
        x138 = x136*x137
        x139 = d_5*r_21
        x140 = r_23*x10
        x141 = r_23*x16
        x142 = r_21*x14
        x143 = r_21*x20
        x144 = x139 + x140 + x141 - x142 - x143
        x145 = x121*x144
        x146 = r_11*r_22
        x147 = a_0*x146
        x148 = R_l_inv_54*a_1
        x149 = r_22*x148
        x150 = -x149
        x151 = r_12*r_21
        x152 = a_0*x151
        x153 = -x152
        x154 = d_3*r_22
        x155 = x154*x81
        x156 = R_l_inv_51*a_1
        x157 = x156*x33
        x158 = -x157
        x159 = 2*x136
        x160 = d_3*x125
        x161 = x159*x160
        x162 = x31*x43
        x163 = x125*x162
        x164 = -x163
        x165 = x135 + x138 + x145 + x147 + x150 + x153 + x155 + x158 + x161 + x164
        x166 = x137*x35
        x167 = a_0*x125
        x168 = x167*x41
        x169 = r_22*x10
        x170 = r_22*x16
        x171 = r_21*x12
        x172 = r_21*x18
        x173 = x169 + x170 - x171 - x172
        x174 = x166 + x168 + x173
        x175 = x12 + x18 + x56
        x176 = x156*x175
        x177 = r_11*r_23
        x178 = r_13*r_21
        x179 = a_0*(x177 - x178)
        x180 = x121*x179
        x181 = x43*x70
        x182 = x125*x181
        x183 = -x176 + x180 - x182
        x184 = R_l_inv_53*a_1
        x185 = 2*r_23
        x186 = x184*x185
        x187 = 2*x81
        x188 = x173*x187
        x189 = a_0*x71
        x190 = x189*x3
        x191 = 2*x121
        x192 = x191*x6
        x193 = -2*R_l_inv_50*a_1*x136
        x194 = r_22*x3
        x195 = a_0*x194
        x196 = -x195
        x197 = x137*x43
        x198 = 2*x70
        x199 = x137*x198
        x200 = a_0*x116
        x201 = 2*x200
        x202 = x146 - x151
        x203 = -2*R_l_inv_56*a_0*a_1*x202
        x204 = 4*a_4
        x205 = x160*x204
        x206 = 4*x70
        x207 = x160*x206
        x208 = x193 + x196 + x197 + x199 + x201 + x203 + x205 + x207
        x209 = 2*x75
        x210 = Px*x194
        x211 = 2*x79
        x212 = 2*x76
        x213 = 2*x77
        x214 = 2*x68
        x215 = x214*x81
        x216 = 2*x31
        x217 = x137*x216
        x218 = x167*x65
        x219 = x209 - x210 - x211 + x212 + x213 - x215 + x217 + x218
        x220 = d_3*x137
        x221 = x113*x125
        x222 = -x180
        x223 = a_2 + x120 + x176 + x182 + x220 + x221 + x222 + x72
        x224 = x115 - x123 + x124 + x132 + x68 + x74 + x82
        x225 = -x145 + x149 - x155 + x157 + x163
        x226 = x135 + x138 + x147 + x153 + x161 + x225
        x227 = r_11*x56
        x228 = 2*x227
        x229 = r_11*x18
        x230 = 2*x229
        x231 = r_12*x55
        x232 = r_12*x16
        x233 = 2*x232
        x234 = a_0*x185
        x235 = x137*x234
        x236 = 4*a_0
        x237 = x125*x236
        x238 = x237*x6
        x239 = 2*r_12
        x240 = x148*x239
        x241 = -x240
        x242 = d_5*r_11
        x243 = r_13*x69
        x244 = r_13*x16
        x245 = r_11*x58
        x246 = r_11*x20
        x247 = x242 + x243 + x244 - x245 - x246
        x248 = x191*x247
        x249 = r_22*x130
        x250 = x156*x249
        x251 = d_3*r_12
        x252 = x187*x251
        x253 = a_4*x3
        x254 = x121*x253
        x255 = x236*x28
        x256 = x125*x255
        x257 = x237*x8
        x258 = x241 + x248 + x250 + x252 + x254 + x256 + x257
        x259 = r_12*x43
        x260 = 2*r_11
        x261 = x184*x260
        x262 = d_5*r_12
        x263 = r_13*x56
        x264 = r_13*x18
        x265 = r_12*x58
        x266 = r_12*x20
        x267 = x262 + x263 + x264 - x265 - x266
        x268 = x187*x267
        x269 = x1*x189
        x270 = d_3*x260
        x271 = x121*x270
        x272 = x259 + x261 + x268 - x269 + x271
        x273 = r_12*x69
        x274 = x227 + x229 - x232 - x273
        x275 = 4*x81
        x276 = x274*x275
        x277 = 4*r_13
        x278 = 4*r_23
        x279 = x189*x278
        x280 = d_3*x277
        x281 = 4*x262
        x282 = 4*x263
        x283 = 4*x264
        x284 = 4*x265
        x285 = r_21*x236
        x286 = 4*r_12
        x287 = a_4*x286
        x288 = d_3*r_21
        x289 = 8*x288
        x290 = x137*x285 + x167*x289 + 4*x266 - x281 - x282 - x283 + x284 + x287*x81
        x291 = x228 + x230 - x231 - x233 + x235 + x238
        x292 = -x72
        x293 = -x120
        x294 = x176 + x182 + x222 + x292 + x293
        x295 = x123 + x133
        x296 = a_2 + x115 + x124 + x220 + x221 + x295 + x68 + x74 + x82
        x297 = -x135 - x138 - x147 + x152 - x161
        x298 = x145 + x150 + x155 + x158 + x164 + x297
        x299 = x186 + x188 + x190 + x192
        x300 = -x166 - x168 - x169 - x170 + x171 + x172
        x301 = x165 + x300
        x302 = a_1*a_2
        x303 = 2*x302
        x304 = x87 + x88
        x305 = R_l_inv_62*x304
        x306 = R_l_inv_22*x303 + x305
        x307 = d_3*x306
        x308 = R_l_inv_24*x303 + R_l_inv_64*x304
        x309 = r_22*x308
        x310 = R_l_inv_61*x304
        x311 = R_l_inv_21*x303 + x310
        x312 = x175*x311
        x313 = R_l_inv_25*x303 + R_l_inv_65*x304
        x314 = x113*x313
        x315 = R_l_inv_67*x304
        x316 = R_l_inv_27*x303 + x315
        x317 = x144*x316
        x318 = -x317
        x319 = x311*x33
        x320 = x179*x316
        x321 = -x320
        x322 = x122*x316
        x323 = -x322
        x324 = R_l_inv_66*x304
        x325 = R_l_inv_26*x303 + x324
        x326 = x154*x325
        x327 = -x326
        x328 = x43*x6
        x329 = -x328
        x330 = d_5*x38
        x331 = -x330
        x332 = x131*x313
        x333 = x181*x313
        x334 = 2*x6
        x335 = x10*x334
        x336 = -x335
        x337 = x16*x334
        x338 = -x337
        x339 = x216*x6
        x340 = -x339
        x341 = x14*x38
        x342 = x20*x38
        x343 = x162*x313
        x344 = x35*x38
        x345 = x307 + x309 + x312 + x314 + x318 + x319 + x321 + x323 + x327 + x329 + x331 + x332 + x333 + x336 + x338 + x340 + x341 + x342 + x343 + x344
        x346 = r_21*x84
        x347 = r_21**3
        x348 = x347*x99
        x349 = x109*x347
        x350 = R_l_inv_60*x304
        x351 = R_l_inv_20*x303 + x350
        x352 = a_4*x351
        x353 = r_21*x85
        x354 = -x353
        x355 = r_21*x86
        x356 = -x355
        x357 = R_l_inv_23*x303 + R_l_inv_63*x304
        x358 = r_21*x357
        x359 = -x358
        x360 = x136*x306
        x361 = x325*x80
        x362 = -x361
        x363 = r_21*x96
        x364 = r_21*x101
        x365 = r_21*x102
        x366 = r_21*x105
        x367 = r_21*x111
        x368 = r_21*x112
        x369 = x31*x351
        x370 = -x369
        x371 = x288*x316
        x372 = -x371
        x373 = r_21*x97
        x374 = -x373
        x375 = r_21*x98
        x376 = -x375
        x377 = r_21*x107
        x378 = -x377
        x379 = r_21*x108
        x380 = -x379
        x381 = d_3*x313
        x382 = x159*x381
        x383 = x1*x109
        x384 = x383*x46
        x385 = -x384
        x386 = x383*x49
        x387 = -x386
        x388 = x383*x52
        x389 = -x388
        x390 = x10*x89
        x391 = -x390
        x392 = x16*x89
        x393 = -x392
        x394 = a_4*x1
        x395 = x31*x394
        x396 = x214*x33
        x397 = x122*x3
        x398 = a_0*x397
        x399 = x23*x92
        x400 = x25*x92
        x401 = x27*x92
        x402 = d_5*x1
        x403 = x14*x402
        x404 = 2*x16
        x405 = x23*x404
        x406 = x25*x404
        x407 = x27*x404
        x408 = x20*x402
        x409 = x146*x239*x95
        x410 = x3*x95
        x411 = x177*x410
        x412 = 2*r_22
        x413 = r_31*x103
        x414 = r_32*x413
        x415 = x412*x414
        x416 = r_33*x413
        x417 = x185*x416
        x418 = x1*x18
        x419 = x12*x418
        x420 = -x419
        x421 = x1*x20
        x422 = x14*x421
        x423 = -x422
        x424 = x1*x16
        x425 = x10*x424
        x426 = x19*x92
        x427 = x21*x92
        x428 = x13*x404
        x429 = Px*x3
        x430 = x141*x429
        x431 = x346 + x348 + x349 - x352 + x354 + x356 + x359 - x360 + x362 + x363 + x364 + x365 + x366 + x367 + x368 + x370 + x372 + x374 + x376 + x378 + x380 - x382 + x385 + x387 + x389 + x391 + x393 - x395 - x396 - x398 + x399 + x400 + x401 + x403 + x405 + x406 + x407 + x408 + x409 + x411 + x415 + x417 + x420 + x423 + x425 + x426 + x427 + x428 + x430
        x432 = x306*x35
        x433 = x43*x9
        x434 = x23*x43
        x435 = x25*x43
        x436 = x27*x43
        x437 = a_0*x313
        x438 = x41*x437
        x439 = x10*x394
        x440 = x12*x214
        x441 = Px*x397
        x442 = x16*x394
        x443 = x18*x214
        x444 = x21*x43
        x445 = -x432 - x433 + x434 + x435 + x436 - x438 + x439 + x440 + x441 + x442 + x443 + x444
        x446 = x351*x70
        x447 = x119*x325
        x448 = a_0*x1
        x449 = x448*x47
        x450 = x448*x50
        x451 = x448*x53
        x452 = x216*x23
        x453 = x31*x89
        x454 = x216*x25
        x455 = x216*x27
        x456 = x126*x448
        x457 = 4*x56
        x458 = x152*x457
        x459 = 4*x69
        x460 = x36*x459
        x461 = x31*x424
        x462 = x33*x418
        x463 = x34*x404
        x464 = x35*x421
        x465 = a_0*x3
        x466 = x141*x465
        x467 = x147*x93
        x468 = x130*x177
        x469 = x20*x468
        x470 = -x446 - x447 - x449 - x450 - x451 - x452 - x453 + x454 + x455 + x456 - x458 - x460 - x461 - x462 - x463 - x464 - x466 + x467 + x469
        x471 = 4*x302
        x472 = R_l_inv_20*x471 + 2*x350
        x473 = x136*x472
        x474 = x70*(R_l_inv_22*x471 + 2*x305)
        x475 = -x306*x43
        x476 = -x204*x381
        x477 = x206*x381
        x478 = x130*x202
        x479 = x325*x478
        x480 = 4*x35
        x481 = -x23*x480
        x482 = -x25*x480
        x483 = 4*x8
        x484 = x35*x483
        x485 = x139*x5
        x486 = -a_0*x485
        x487 = 4*x33
        x488 = -x487*x75
        x489 = -x480*x9
        x490 = a_0*x278
        x491 = x47*x490
        x492 = x490*x50
        x493 = x490*x53
        x494 = x27*x480
        x495 = x122*x5
        x496 = a_0*x495
        x497 = x178*x236
        x498 = -x16*x497
        x499 = a_0*x117
        x500 = 4*x18
        x501 = -x499*x500
        x502 = a_0*x5
        x503 = x143*x502
        x504 = x141*x502
        x505 = 4*r_22
        x506 = x20*x505
        x507 = x33*x506
        x508 = x200*x500
        x509 = x21*x480
        x510 = 8*x177
        x511 = a_0*x510*x69
        x512 = 8*x56
        x513 = x200*x512
        x514 = x473 - x474 + x475 + x476 - x477 + x479 + x481 + x482 - x484 + x486 + x488 + x489 + x491 + x492 + x493 + x494 + x496 + x498 + x501 + x503 + x504 + x507 + x508 + x509 + x511 + x513
        x515 = x216*x306
        x516 = x214*x325
        x517 = 4*d_5
        x518 = x517*x8
        x519 = x14*x483
        x520 = x20*x483
        x521 = x437*x65
        x522 = Px*x495
        x523 = x141*x204
        x524 = -x515 + x516 + x518 - x519 - x520 - x521 + x522 + x523
        x525 = R_l_inv_26*x471 + 2*x324
        x526 = x173*x525
        x527 = r_23*x86
        x528 = 2*x527
        x529 = r_23**3
        x530 = x529*x99
        x531 = 2*x530
        x532 = x109*x529
        x533 = 2*x532
        x534 = r_23*x84
        x535 = 2*x534
        x536 = r_23*x85
        x537 = 2*x536
        x538 = 2*x357
        x539 = r_23*x538
        x540 = r_23*x98
        x541 = 2*x540
        x542 = r_23*x100
        x543 = 2*x542
        x544 = r_23*x101
        x545 = 2*x544
        x546 = r_23*x108
        x547 = 2*x546
        x548 = r_23*x110
        x549 = 2*x548
        x550 = r_23*x111
        x551 = 2*x550
        x552 = a_0*x351
        x553 = x3*x552
        x554 = x316*x334
        x555 = r_23*x96
        x556 = 2*x555
        x557 = r_23*x97
        x558 = 2*x557
        x559 = r_23*x105
        x560 = 2*x559
        x561 = r_23*x107
        x562 = 2*x561
        x563 = x23*x517
        x564 = x25*x517
        x565 = x27*x517
        x566 = x109*x46
        x567 = x278*x566
        x568 = x109*x278
        x569 = x49*x568
        x570 = x52*x568
        x571 = 4*x14
        x572 = x23*x571
        x573 = x25*x571
        x574 = x27*x571
        x575 = 4*x20
        x576 = x23*x575
        x577 = x25*x575
        x578 = x27*x575
        x579 = x5*x95
        x580 = x178*x579
        x581 = 4*x95
        x582 = r_12*x117*x581
        x583 = 4*r_21
        x584 = x416*x583
        x585 = r_32*r_33*x103
        x586 = x505*x585
        x587 = Px*x485
        x588 = 4*x75
        x589 = x12*x588
        x590 = x571*x9
        x591 = 4*x139
        x592 = x16*x591
        x593 = x18*x588
        x594 = x575*x9
        x595 = Px*x5
        x596 = x143*x595
        x597 = x12*x506
        x598 = x17*x571
        x599 = x18*x505
        x600 = x14*x599
        x601 = x15*x575
        x602 = x141*x595
        x603 = x500*x76
        x604 = x526 - x528 - x531 - x533 + x535 + x537 + x539 - x541 - x543 - x545 - x547 - x549 - x551 + x553 + x554 + x556 + x558 + x560 + x562 + x563 + x564 + x565 + x567 + x569 + x570 - x572 - x573 - x574 - x576 - x577 - x578 - x580 - x582 - x584 - x586 + x587 + x589 + x590 + x592 + x593 + x594 - x596 - x597 - x598 - x600 - x601 + x602 + x603
        x605 = -x434
        x606 = -x435
        x607 = -x436
        x608 = -x439
        x609 = -x440
        x610 = -x441
        x611 = -x442
        x612 = -x443
        x613 = -x444
        x614 = x352 + x360 + x382 + x395 + x396 + x398 + x432 + x433 + x438 + x605 + x606 + x607 + x608 + x609 + x610 + x611 + x612 + x613
        x615 = x446 + x447 + x449 + x450 + x451 + x452 + x453 - x454 - x455 - x456 + x458 + x460 + x461 + x462 + x463 + x464 + x466 - x467 - x469
        x616 = -x267*x525
        x617 = -x260*x85
        x618 = -x260*x86
        x619 = -r_11*x538
        x620 = x260*x84
        x621 = r_11**3
        x622 = 2*x95
        x623 = x621*x622
        x624 = x255*x313
        x625 = a_4*x280
        x626 = x253*x316
        x627 = -x270*x316
        x628 = -x101*x260
        x629 = -x102*x260
        x630 = -x107*x260
        x631 = -x108*x260
        x632 = -x111*x260
        x633 = -x112*x260
        x634 = x1*x552
        x635 = x260*x97
        x636 = x260*x98
        x637 = x100*x260
        x638 = x105*x260
        x639 = x110*x260
        x640 = 4*x126
        x641 = -x640*x69
        x642 = -x16*x640
        x643 = x459*x47
        x644 = x459*x50
        x645 = x459*x53
        x646 = 4*x16
        x647 = x47*x646
        x648 = x50*x646
        x649 = x53*x646
        x650 = Py*x5
        x651 = x650*x9
        x652 = d_5*x5
        x653 = x20*x652
        x654 = x505*x99
        x655 = x151*x654
        x656 = x109*x505
        x657 = x151*x656
        x658 = x286*x414
        x659 = x278*x99
        x660 = x178*x659
        x661 = x178*x568
        x662 = x277*x416
        x663 = x18*x5
        x664 = -x56*x663
        x665 = x20*x5
        x666 = -x58*x665
        x667 = x16*x5
        x668 = x667*x69
        x669 = x459*x61
        x670 = x57*x646
        x671 = x129*x459
        x672 = x128*x646
        x673 = x616 + x617 + x618 + x619 + x620 + x623 - x624 - x625 - x626 + x627 + x628 + x629 + x630 + x631 + x632 + x633 + x634 + x635 + x636 + x637 + x638 + x639 + x641 + x642 + x643 + x644 + x645 + x647 + x648 + x649 + x651 + x653 + x655 + x657 + x658 + x660 + x661 + x662 + x664 + x666 + x668 + x669 + x670 + x671 + x672
        x674 = R_l_inv_27*x471 + 2*x315
        x675 = x247*x674
        x676 = x239*x308
        x677 = d_5*x65
        x678 = x249*x311
        x679 = 2*x251
        x680 = x325*x679
        x681 = 4*d_3
        x682 = x243*x681
        x683 = x244*x681
        x684 = x236*x313
        x685 = x684*x8
        x686 = x6*x650
        x687 = x20*x65
        x688 = -x675 + x676 - x677 - x678 - x680 - x682 - x683 - x685 + x686 + x687
        x689 = x126*x204
        x690 = x234*x306
        x691 = x204*x47
        x692 = x204*x50
        x693 = x204*x53
        x694 = x650*x8
        x695 = Py*x286
        x696 = x68*x695
        x697 = x128*x204
        x698 = a_4*x5
        x699 = x16*x698
        x700 = x204*x61
        x701 = x129*x204
        x702 = x6*x684
        x703 = -x689 + x690 + x691 + x692 + x693 + x694 + x696 + x697 + x699 + x700 + x701 + x702
        x704 = 8*x242
        x705 = a_4*x704
        x706 = x285*x306
        x707 = x287*x325
        x708 = Py*x8
        x709 = 8*r_13
        x710 = x708*x709
        x711 = 8*a_4
        x712 = x244*x711
        x713 = x289*x437
        x714 = x245*x711
        x715 = x246*x711
        x716 = 8*x302
        x717 = R_l_inv_26*x716 + 4*x324
        x718 = r_13**3
        x719 = 8*x58
        x720 = 8*x20
        x721 = r_21*x99
        x722 = r_21*x109
        x723 = 8*r_11
        x724 = 8*r_22
        x725 = x724*x99
        x726 = x109*x724
        x727 = 8*r_12
        x728 = x56*x727
        x729 = x18*x727
        x730 = -8*Px*d_5*x46 - 8*Px*d_5*x49 - 8*Px*d_5*x52 - 8*Py*Pz*r_13*r_21*r_31 - 8*Py*Pz*r_13*r_22*r_32 - 8*Py*d_5*r_11*r_21 - 8*Py*d_5*r_12*r_22 - 8*Py*d_5*r_13*r_23 - 8*Pz*d_5*r_11*r_31 - 8*Pz*d_5*r_12*r_32 - 8*Pz*d_5*r_13*r_33 - 4*d_3*r_13*x316 - 4*r_13*x103*x104 - 4*r_13*x103*x106 - 4*r_13*x109*x22 - 4*r_13*x109*x24 - 4*r_13*x22*x99 - 4*r_13*x24*x99 - 4*r_13*x357 - 4*r_13*x84 - 4*r_13*x85 + x102*x277 + x108*x277 + x112*x277 + x116*x725 + x116*x726 + x127*x720 + x128*x720 + x20*x728 + x274*x717 + x277*x86 + x277*x96 + x277*x97 + x278*x552 + x416*x723 + x47*x719 + x47*x720 + x50*x719 + x50*x720 + x510*x721 + x510*x722 + x53*x719 + x53*x720 + x58*x729 + x581*x718 + x585*x727 + x60*x719
        x731 = x675 - x676 + x677 + x678 + x680 + x682 + x683 + x685 - x686 - x687
        x732 = x616 + x617 + x618 + x619 + x620 + x623 + x624 + x625 + x626 + x627 + x628 + x629 + x630 + x631 + x632 + x633 + x634 + x635 + x636 + x637 + x638 + x639 + x641 + x642 + x643 + x644 + x645 + x647 + x648 + x649 + x651 + x653 + x655 + x657 + x658 + x660 + x661 + x662 + x664 + x666 + x668 + x669 + x670 + x671 + x672
        x733 = -d_3*x306
        x734 = -x113*x313
        x735 = x323 + x329 + x332 + x615 + x733 + x734
        x736 = -x312
        x737 = -x333
        x738 = -x344
        x739 = x309 + x318 + x319 + x320 + x327 + x331 + x336 + x338 + x339 + x341 + x342 + x343 + x736 + x737 + x738
        x740 = x346 + x348 + x349 + x352 + x354 + x356 + x359 + x360 + x362 + x363 + x364 + x365 + x366 + x367 + x368 + x370 + x372 + x374 + x376 + x378 + x380 + x382 + x385 + x387 + x389 + x391 + x393 + x395 + x396 + x398 + x399 + x400 + x401 + x403 + x405 + x406 + x407 + x408 + x409 + x411 + x415 + x417 + x420 + x423 + x425 + x426 + x427 + x428 + x430 + x445
        x741 = x515 - x516 - x518 + x519 + x520 + x521 - x522 - x523
        x742 = -x526 + x528 + x531 + x533 - x535 - x537 - x539 + x541 + x543 + x545 + x547 + x549 + x551 - x553 - x554 - x556 - x558 - x560 - x562 - x563 - x564 - x565 - x567 - x569 - x570 + x572 + x573 + x574 + x576 + x577 + x578 + x580 + x582 + x584 + x586 - x587 - x589 - x590 - x592 - x593 - x594 + x596 + x597 + x598 + x600 + x601 - x602 - x603
        x743 = -x332
        x744 = x307 + x314 + x322 + x328 + x615 + x743
        x745 = x312 + x321 + x333 + x340 + x344
        x746 = -x309 + x317 - x319 + x326 + x330 + x335 + x337 - x341 - x342 - x343
        x747 = x745 + x746
        x748 = r_22*x84
        x749 = r_22*x85
        x750 = r_22*x86
        x751 = R_l_inv_77*x304
        x752 = R_l_inv_37*x303 + x751
        x753 = x144*x752
        x754 = R_l_inv_34*x303 + R_l_inv_74*x304
        x755 = r_22*x754
        x756 = r_22**3
        x757 = x756*x99
        x758 = x109*x756
        x759 = R_l_inv_76*x304
        x760 = R_l_inv_36*x303 + x759
        x761 = x154*x760
        x762 = r_22*x96
        x763 = r_22*x98
        x764 = r_22*x105
        x765 = r_22*x108
        x766 = R_l_inv_71*x304
        x767 = R_l_inv_31*x303 + x766
        x768 = x33*x767
        x769 = r_22*x97
        x770 = r_22*x100
        x771 = r_22*x102
        x772 = r_22*x107
        x773 = r_22*x110
        x774 = r_22*x112
        x775 = x412*x566
        x776 = x109*x412
        x777 = x49*x776
        x778 = x52*x776
        x779 = x23*x91
        x780 = x25*x91
        x781 = x27*x91
        x782 = x429*x75
        x783 = x23*x93
        x784 = x25*x93
        x785 = x27*x93
        x786 = x20*x209
        x787 = R_l_inv_35*x303 + R_l_inv_75*x304
        x788 = x162*x787
        x789 = r_12*x95
        x790 = r_11*x1
        x791 = x789*x790
        x792 = x116*x410
        x793 = x1*x414
        x794 = x185*x585
        x795 = x12*x89
        x796 = x18*x89
        x797 = x10*x418
        x798 = x12*x424
        x799 = x13*x93
        x800 = x21*x91
        x801 = x429*x77
        x802 = x169*x404
        x803 = x429*x79
        x804 = R_l_inv_70*x304
        x805 = R_l_inv_30*x303 + x804
        x806 = a_4*x805
        x807 = R_l_inv_72*x304
        x808 = R_l_inv_32*x303 + x807
        x809 = x136*x808
        x810 = d_3*x808
        x811 = -x810
        x812 = x113*x787
        x813 = -x812
        x814 = x35*x808
        x815 = x122*x752
        x816 = x131*x787
        x817 = -x816
        x818 = d_3*x787
        x819 = x159*x818
        x820 = x12*x394
        x821 = -x820
        x822 = x18*x394
        x823 = -x822
        x824 = x33*x394
        x825 = -x824
        x826 = x10*x214
        x827 = x16*x214
        x828 = x214*x31
        x829 = a_0*x787
        x830 = x41*x829
        x831 = x806 + x809 + x811 + x813 + x814 + x815 + x817 + x819 + x821 + x823 + x825 + x826 + x827 + x828 + x830
        x832 = R_l_inv_33*x303 + R_l_inv_73*x304
        x833 = r_21*x832
        x834 = x70*x805
        x835 = x760*x80
        x836 = x31*x805
        x837 = x119*x760
        x838 = x288*x752
        x839 = d_3*x209
        x840 = x154*x429
        x841 = -x840
        x842 = x154*x94
        x843 = -x842
        x844 = x154*x465
        x845 = -x844
        x846 = x12*x334
        x847 = x18*x334
        x848 = x33*x334
        x849 = x833 + x834 + x835 + x836 + x837 + x838 + x839 + x841 + x843 + x845 + x846 + x847 + x848
        x850 = x175*x767
        x851 = x179*x752
        x852 = x181*x787
        x853 = 2*x33
        x854 = x23*x853
        x855 = x27*x853
        x856 = x465*x75
        x857 = a_0*r_22
        x858 = x48*x857
        x859 = x51*x857
        x860 = x54*x857
        x861 = x25*x853
        x862 = x33*x89
        x863 = x33*x424
        x864 = x20*x201
        x865 = x31*x418
        x866 = x147*x404
        x867 = x19*x853
        x868 = x465*x79
        x869 = x465*x77
        x870 = r_21*x502*x56
        x871 = x36*x457
        x872 = -x850 + x851 - x852 - x854 - x855 - x856 + x858 + x859 + x860 + x861 + x862 - x863 - x864 + x865 + x866 + x867 + x868 + x869 + x870 + x871
        x873 = R_l_inv_36*x471 + 2*x759
        x874 = x173*x873
        x875 = x70*(R_l_inv_32*x471 + 2*x807)
        x876 = 2*x832
        x877 = r_23*x876
        x878 = x206*x818
        x879 = a_0*x805
        x880 = x3*x879
        x881 = x334*x752
        x882 = -x171*x681
        x883 = -x172*x681
        x884 = x154*x5
        x885 = Px*x884
        x886 = d_3*x505
        x887 = x16*x886
        x888 = x874 - x875 + x877 - x878 + x880 + x881 + x882 + x883 + x885 + x887
        x889 = x216*x808
        x890 = x214*x760
        x891 = x68*x681
        x892 = x65*x829
        x893 = -x889 + x890 + x891 - x892
        x894 = R_l_inv_30*x471 + 2*x804
        x895 = x136*x894
        x896 = x478*x760
        x897 = a_0*x884
        x898 = -4*a_0*d_3*r_12*r_21 - 4*a_4*d_3*x787 - 2*a_4*x808 + x895 + x896 + x897
        x899 = -x748
        x900 = -x749
        x901 = -x750
        x902 = -x753
        x903 = -x761
        x904 = -x762
        x905 = -x763
        x906 = -x764
        x907 = -x765
        x908 = -x775
        x909 = -x777
        x910 = -x778
        x911 = -x795
        x912 = -x796
        x913 = -2*a_0*a_4*r_11*r_22
        x914 = -x802
        x915 = -x803
        x916 = x755 + x757 + x758 + x768 + x769 + x770 + x771 + x772 + x773 + x774 + x779 + x780 + x781 + x782 + x783 + x784 + x785 + x786 + x788 + x791 + x792 + x793 + x794 + x797 + x798 + x799 + x800 + x801 + x810 + x812 + x814 + x824 + x830 + x899 + x900 + x901 + x902 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 + x913 + x914 + x915
        x917 = x806 + x809 + x819
        x918 = -x815 + x816 + x820 + x822 - x826 - x827
        x919 = x917 + x918
        x920 = x850 - x851 + x852 + x854 + x855 + x856 - x858 - x859 - x860 - x861 - x862 + x863 + x864 - x865 - x866 - x867 - x868 - x869 - x870 - x871
        x921 = x849 + x920
        x922 = R_l_inv_37*x471 + 2*x751
        x923 = -x247*x922
        x924 = -x239*x84
        x925 = -x239*x85
        x926 = -x239*x86
        x927 = x239*x754
        x928 = r_12**3
        x929 = x622*x928
        x930 = -x249*x767
        x931 = -x679*x760
        x932 = -x100*x239
        x933 = -x102*x239
        x934 = -x105*x239
        x935 = -x108*x239
        x936 = -x110*x239
        x937 = -x112*x239
        x938 = x234*x808
        x939 = x239*x96
        x940 = x239*x98
        x941 = x101*x239
        x942 = x107*x239
        x943 = x111*x239
        x944 = -x56*x640
        x945 = -x18*x640
        x946 = x236*x787
        x947 = -x8*x946
        x948 = x457*x47
        x949 = x457*x50
        x950 = x457*x53
        x951 = x47*x500
        x952 = x50*x500
        x953 = x500*x53
        x954 = x695*x9
        x955 = x20*x281
        x956 = x6*x946
        x957 = r_22*x5
        x958 = x721*x957
        x959 = x722*x957
        x960 = x414*x5
        x961 = x117*x659
        x962 = x117*x568
        x963 = x277*x585
        x964 = -x273*x646
        x965 = -x20*x284
        x966 = x663*x69
        x967 = x56*x667
        x968 = x500*x57
        x969 = x129*x457
        x970 = x128*x500
        x971 = x923 + x924 + x925 + x926 + x927 + x929 + x930 + x931 + x932 + x933 + x934 + x935 + x936 + x937 + x938 + x939 + x940 + x941 + x942 + x943 + x944 + x945 + x947 + x948 + x949 + x950 + x951 + x952 + x953 + x954 + x955 + x956 + x958 + x959 + x960 + x961 + x962 + x963 + x964 + x965 + x966 + x967 + x968 + x969 + x970
        x972 = x650*x68
        x973 = x18*x698
        x974 = -4*Py*a_4*r_12*r_21 - 4*Pz*a_4*r_12*r_31 - 4*a_0*x28*x787 - 2*a_4*r_13*x752 + x972 + x973
        x975 = x267*x873
        x976 = x260*x832
        x977 = d_3*x281
        x978 = x270*x752
        x979 = x1*x879
        x980 = d_3*x282
        x981 = d_3*x283
        x982 = x6*x695
        x983 = 4*x251
        x984 = x20*x983
        x985 = -x975 - x976 - x977 - x978 + x979 - x980 - x981 + x982 + x984
        x986 = R_l_inv_36*x716 + 4*x759
        x987 = x277*x832
        x988 = x280*x752
        x989 = d_3*x727
        x990 = x69*x989
        x991 = x16*x989
        x992 = a_4*x989 + x285*x808 + x287*x760 + x289*x829
        x993 = x975 + x976 + x977 + x978 - x979 + x980 + x981 - x982 - x984
        x994 = x923 + x924 + x925 + x926 + x927 + x929 + x930 + x931 + x932 + x933 + x934 + x935 + x936 + x937 - x938 + x939 + x940 + x941 + x942 + x943 + x944 + x945 + x947 + x948 + x949 + x950 + x951 + x952 + x953 + x954 + x955 - x956 + x958 + x959 + x960 + x961 + x962 + x963 + x964 + x965 + x966 + x967 + x968 + x969 + x970
        x995 = -x833
        x996 = -x835
        x997 = -x836
        x998 = -x838
        x999 = -x839
        x1000 = -x846
        x1001 = -x847
        x1002 = x1000 + x1001 + x834 + x837 + x840 + x842 + x845 + x848 + x872 + x995 + x996 + x997 + x998 + x999
        x1003 = x755 + x757 + x758 + x768 + x769 + x770 + x771 + x772 + x773 + x774 + x779 + x780 + x781 + x782 + x783 + x784 + x785 + x786 + x788 + x791 + x792 + x793 + x794 + x797 + x798 + x799 + x800 + x801 - x814 - x830 + x899 + x900 + x901 + x902 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 + x914 + x915
        x1004 = x811 + x813 + x825 + x828
        x1005 = x1003 + x1004
        x1006 = x152*x681 + x204*x818 + x43*x808 - x895 - x896 - x897
        x1007 = x874 + x875 + x877 + x878 + x880 + x881 + x882 + x883 + x885 + x887
        x1008 = -x806 - x809 - x819
        x1009 = -x834
        x1010 = -x837
        x1011 = -x848
        x1012 = x1009 + x1010 + x1011 + x833 + x835 + x836 + x838 + x839 + x841 + x843 + x844 + x846 + x847 + x872
        x1013 = x755 + x757 + x758 + x768 + x769 + x770 + x771 + x772 + x773 + x774 + x779 + x780 + x781 + x782 + x783 + x784 + x785 + x786 + x788 + x791 + x792 + x793 + x794 + x797 + x798 + x799 + x800 + x801 + x899 + x900 + x901 + x902 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 + x914 + x915
        x1014 = x6*x90
        x1015 = -x1014
        x1016 = 2*d_3
        x1017 = x1016*x23
        x1018 = x1016*x25
        x1019 = x1016*x27
        x1020 = a_4*x38
        x1021 = -2*a_0*d_3*r_11*r_21
        x1022 = -2*a_0*d_3*r_12*r_22
        x1023 = -2*a_0*d_3*r_13*r_23
        x1024 = x10*x38
        x1025 = x154*x91
        x1026 = x429*x6
        x1027 = x16*x38
        x1028 = x154*x93
        x1029 = x20*x334
        x1030 = x1015 + x1017 + x1018 + x1019 + x1020 + x1021 + x1022 + x1023 + x1024 + x1025 + x1026 + x1027 + x1028 + x1029
        x1031 = d_5*x394
        x1032 = x14*x394
        x1033 = -x1032
        x1034 = x20*x394
        x1035 = -x1034
        x1036 = x35*x394
        x1037 = -x1036
        x1038 = x140*x43
        x1039 = x141*x43
        x1040 = a_0*x177*x43
        x1041 = x1031 + x1033 + x1035 + x1037 + x1038 + x1039 + x1040
        x1042 = x23*x465
        x1043 = x25*x465
        x1044 = x242*x448
        x1045 = x209*x33
        x1046 = x465*x9
        x1047 = a_0*r_23
        x1048 = x1047*x48
        x1049 = x1047*x51
        x1050 = x1047*x54
        x1051 = x27*x465
        x1052 = x35*x424
        x1053 = x19*x465
        x1054 = x31*x421
        x1055 = x16*x468
        x1056 = x34*x94
        x1057 = x18*x201
        x1058 = x21*x465
        x1059 = r_23*x5
        x1060 = a_0*x1059*x69
        x1061 = x200*x457
        x1062 = -x1042 - x1043 - x1044 - x1045 - x1046 + x1048 + x1049 + x1050 + x1051 - x1052 - x1053 + x1054 + x1055 + x1056 + x1057 + x1058 + x1060 + x1061
        x1063 = x23*x90
        x1064 = x25*x90
        x1065 = x27*x90
        x1066 = x185*x566
        x1067 = x109*x185
        x1068 = x1067*x49
        x1069 = x1067*x52
        x1070 = x23*x429
        x1071 = x25*x429
        x1072 = x27*x429
        x1073 = x23*x94
        x1074 = x25*x94
        x1075 = x27*x94
        x1076 = r_13*x95
        x1077 = x1076*x790
        x1078 = x194*x789
        x1079 = x1*x416
        x1080 = x412*x585
        x1081 = x10*x402
        x1082 = x12*x209
        x1083 = x429*x9
        x1084 = x16*x402
        x1085 = x18*x209
        x1086 = x20*x89
        x1087 = x10*x421
        x1088 = x13*x94
        x1089 = x14*x424
        x1090 = x19*x429
        x1091 = x21*x429
        x1092 = x140*x404
        x1093 = x18*x212
        x1094 = x1063 + x1064 + x1065 + x1066 + x1068 + x1069 - x1070 - x1071 - x1072 - x1073 - x1074 - x1075 - x1077 - x1078 - x1079 - x1080 + x1081 + x1082 + x1083 + x1084 + x1085 + x1086 - x1087 - x1088 - x1089 - x1090 - x1091 + x1092 + x1093 - x527 - x530 - x532 + x534 - x536 - x540 - x542 - x544 - x546 - x548 - x550 + x555 + x557 + x559 + x561
        x1095 = x1062 + x1094
        x1096 = x1*x84
        x1097 = x1*x85
        x1098 = x502*x8
        x1099 = x487*x68
        x1100 = x204*x36
        x1101 = x109*x583
        x1102 = x5*x9
        x1103 = 4*x142
        x1104 = x16*x505
        x1105 = -Px*x1102 + r_12*r_22*x579 + x1*x101 + x1*x102 + x1*x105 - x1*x107 - x1*x108 + x1*x111 + x1*x112 - x1*x86 + x1*x96 - x1*x97 - x1*x98 + x1059*x1076 - x1101*x49 - x1101*x52 - x1103*x20 + x1104*x12 + x14*x591 + x15*x646 + x17*x595 - x171*x500 + x19*x595 + x20*x591 + x21*x595 + x23*x595 + x23*x646 + x25*x595 + x25*x646 + x27*x595 + x27*x646 + x278*x416 + 2*x348 + 2*x349 + x414*x505 - x566*x583 - x646*x9
        x1106 = x1096 + x1097 - x1098 - x1099 - x1100 + x1105
        x1107 = x204*x9
        x1108 = x204*x23
        x1109 = x204*x25
        x1110 = x204*x27
        x1111 = x595*x8
        x1112 = 4*x12
        x1113 = x1112*x68
        x1114 = x15*x204
        x1115 = x16*x483
        x1116 = x500*x68
        x1117 = x204*x21
        x1118 = -x1107 + x1108 + x1109 + x1110 + x1111 + x1113 + x1114 + x1115 + x1116 + x1117
        x1119 = x285*x47
        x1120 = x285*x50
        x1121 = x285*x53
        x1122 = x23*x502
        x1123 = a_0*x1102
        x1124 = x25*x502
        x1125 = x27*x502
        x1126 = x35*x591
        x1127 = x152*x512
        x1128 = 8*x69
        x1129 = x1128*x36
        x1130 = x17*x502
        x1131 = x152*x500
        x1132 = x1104*x33
        x1133 = x20*x497
        x1134 = x36*x646
        x1135 = x19*x502
        x1136 = x21*x502
        x1137 = -x1119 - x1120 - x1121 - x1122 - x1123 + x1124 + x1125 + x1126 - x1127 - x1129 - x1130 - x1131 - x1132 - x1133 - x1134 + x1135 + x1136
        x1138 = -x1031
        x1139 = -x1038
        x1140 = -x1039
        x1141 = -x1040
        x1142 = x1032 + x1034 + x1036 + x1138 + x1139 + x1140 + x1141
        x1143 = x1042 + x1043 + x1044 + x1045 + x1046 - x1048 - x1049 - x1050 - x1051 + x1052 + x1053 - x1054 - x1055 - x1056 - x1057 - x1058 - x1060 - x1061
        x1144 = -x1063 - x1064 - x1065 - x1066 - x1068 - x1069 + x1070 + x1071 + x1072 + x1073 + x1074 + x1075 + x1077 + x1078 + x1079 + x1080 - x1081 - x1082 - x1083 - x1084 - x1085 - x1086 + x1087 + x1088 + x1089 + x1090 + x1091 - x1092 - x1093 + x527 + x530 + x532 - x534 + x536 + x540 + x542 + x544 + x546 + x548 + x550 - x555 - x557 - x559 - x561
        x1145 = x1143 + x1144
        x1146 = x3*x85
        x1147 = x3*x86
        x1148 = x622*x718
        x1149 = x3*x84
        x1150 = x3*x96
        x1151 = x3*x97
        x1152 = x102*x3
        x1153 = x108*x3
        x1154 = x112*x3
        x1155 = x100*x3
        x1156 = x101*x3
        x1157 = x105*x3
        x1158 = x107*x3
        x1159 = x110*x3
        x1160 = x111*x3
        x1161 = x47*x517
        x1162 = x50*x517
        x1163 = x517*x53
        x1164 = 4*x58
        x1165 = x1164*x47
        x1166 = x1164*x50
        x1167 = x1164*x53
        x1168 = x47*x575
        x1169 = x50*x575
        x1170 = x53*x575
        x1171 = x1059*x721
        x1172 = x1059*x722
        x1173 = x416*x5
        x1174 = x116*x654
        x1175 = x116*x656
        x1176 = x286*x585
        x1177 = x652*x69
        x1178 = x281*x56
        x1179 = Py*x9
        x1180 = x1179*x277
        x1181 = x16*x652
        x1182 = x18*x281
        x1183 = x126*x575
        x1184 = x665*x69
        x1185 = x58*x667
        x1186 = x57*x575
        x1187 = x1164*x61
        x1188 = x128*x575
        x1189 = x243*x646
        x1190 = x18*x282
        x1191 = a_4*x65
        x1192 = d_5*x698
        x1193 = x58*x698
        x1194 = -x1193
        x1195 = x20*x698
        x1196 = -x1195
        x1197 = x277*x708
        x1198 = x204*x244
        x1199 = x1191 + x1192 + x1194 + x1196 + x1197 + x1198
        x1200 = x47*x681
        x1201 = x50*x681
        x1202 = x53*x681
        x1203 = x65*x69
        x1204 = x56*x983
        x1205 = Py*x6
        x1206 = x1205*x277
        x1207 = x16*x65
        x1208 = x18*x983
        x1209 = x129*x681
        x1210 = -4*d_3*d_5*r_13 + x1200 + x1201 + x1202 + x1203 + x1204 + x1206 + x1207 + x1208 + x1209
        x1211 = x126*x711
        x1212 = Py*x68
        x1213 = x5*x84
        x1214 = x5*x85
        x1215 = 8*x126
        x1216 = 8*x16
        x1217 = 8*r_23
        x1218 = x1217*x178
        x1219 = 8*x18
        x1220 = 8*x245
        x1221 = x100*x5 - x101*x5 - x102*x5 + x105*x5 - x107*x5 - x108*x5 + x109*x1218 + x110*x5 - x111*x5 - x112*x5 + x1128*x129 + x1128*x47 + x1128*x50 + x1128*x53 + x1179*x723 - x1215*x16 - x1215*x69 + x1216*x127 + x1216*x128 + x1216*x47 + x1216*x50 + x1216*x53 + x1218*x99 - x1219*x227 - x1220*x20 + x151*x725 + x151*x726 + x16*x728 + x20*x704 + x414*x727 + x416*x709 - x5*x86 + x5*x97 + x5*x98 + x581*x621 + x69*x729
        x1222 = x1213 + x1214 + x1221
        x1223 = x1146 + x1147 + x1148 - x1149 + x1150 + x1151 + x1152 + x1153 + x1154 - x1155 - x1156 - x1157 - x1158 - x1159 - x1160 - x1161 - x1162 - x1163 + x1165 + x1166 + x1167 + x1168 + x1169 + x1170 + x1171 + x1172 + x1173 + x1174 + x1175 + x1176 - x1177 - x1178 - x1180 - x1181 - x1182 - x1183 + x1184 + x1185 + x1186 + x1187 + x1188 - x1189 - x1190
        x1224 = x1210 + x1223
        x1225 = x1020 + x1031 + x1033 + x1035 + x1036 + x1038 + x1039 + x1141
        x1226 = x1015 + x1017 + x1018 + x1019 + x1024 + x1025 + x1026 + x1027 + x1028 + x1029
        x1227 = x1226 + x154*x853 + x31*x38 + x465*x6
        x1228 = x1096 + x1097 + x1098 + x1099 + x1100 + x1105
        x1229 = x1119 + x1120 + x1121 + x1122 + x1123 - x1124 - x1125 - x1126 + x1127 + x1129 + x1130 + x1131 + x1132 + x1133 + x1134 - x1135 - x1136
        x1230 = x1062 + x1144
        x1231 = x1227 + x1230
        x1232 = -x214
        x1233 = 2*x71
        x1234 = x1233*x175
        x1235 = x179*x187
        x1236 = x156*x198
        x1237 = x1*x148
        x1238 = x191*x80
        x1239 = -4*R_l_inv_55*a_1*a_4*x175
        x1240 = x38*x81
        x1241 = -x1240
        x1242 = x156*x216
        x1243 = x119*x191
        x1244 = x204*x33
        x1245 = x1244*x125
        x1246 = -x1245
        x1247 = x1236 + x1237 + x1238 + x1239 + x1241 + x1242 + x1243 + x1246
        x1248 = x144*x187
        x1249 = x184*x412
        x1250 = x1233*x33
        x1251 = x154*x191
        x1252 = -x1248 + x1249 + x1250 + x1251 - x394
        x1253 = Px*x1059
        x1254 = 4*x141
        x1255 = 4*x143
        x1256 = 4*x137
        x1257 = x1256*x33
        x1258 = x275*x8
        x1259 = x167*x989
        x1260 = 8*x175
        x1261 = -4*a_0*r_11*r_23 + x1256*x175 + x1260*x160 + x497
        x1262 = -x1234 + x1235
        x1263 = x148*x5
        x1264 = 4*x121
        x1265 = x1264*x267
        x1266 = 8*x68
        x1267 = x1266*x167
        x1268 = -x1264*x251 - x184*x286 + x189*x505 + x247*x275 + x698
        x1269 = 8*x243
        x1270 = 8*x246
        x1271 = 8*x244
        x1272 = a_0*x724
        x1273 = r_11*x711
        x1274 = 16*x154
        x1275 = x175*x204
        x1276 = x1252 + x1262
        x1277 = x144*x525
        x1278 = x70*(R_l_inv_21*x471 + 2*x310)
        x1279 = -2*x749
        x1280 = -2*x750
        x1281 = -r_22*x538
        x1282 = 2*x748
        x1283 = 2*x757
        x1284 = 2*x758
        x1285 = -x505*x566
        x1286 = -x49*x656
        x1287 = -x52*x656
        x1288 = -x351*x853
        x1289 = 2*x316
        x1290 = x119*x1289
        x1291 = -x1289*x154
        x1292 = -2*x762
        x1293 = -2*x763
        x1294 = -2*x764
        x1295 = -2*x765
        x1296 = 2*x769
        x1297 = 2*x770
        x1298 = 2*x771
        x1299 = 2*x772
        x1300 = 2*x773
        x1301 = 2*x774
        x1302 = -x1112*x9
        x1303 = -x500*x9
        x1304 = x487*x6
        x1305 = x1112*x23
        x1306 = x1112*x25
        x1307 = x1112*x27
        x1308 = x571*x75
        x1309 = x23*x500
        x1310 = x25*x500
        x1311 = x27*x500
        x1312 = x575*x75
        x1313 = x151*x579
        x1314 = x116*x277*x95
        x1315 = x414*x583
        x1316 = x278*x585
        x1317 = -x170*x595
        x1318 = -x14*x506
        x1319 = x172*x595
        x1320 = x1112*x17
        x1321 = x12*x599
        x1322 = x1112*x21
        x1323 = x15*x500
        x1324 = x1275*x313 + x1277 - x1278 + x1279 + x1280 + x1281 + x1282 + x1283 + x1284 + x1285 + x1286 + x1287 + x1288 - x1290 + x1291 + x1292 + x1293 + x1294 + x1295 + x1296 + x1297 + x1298 + x1299 + x1300 + x1301 + x1302 + x1303 - x1304 + x1305 + x1306 + x1307 + x1308 + x1309 + x1310 + x1311 + x1312 + x1313 + x1314 + x1315 + x1316 + x1317 + x1318 + x1319 + x1320 + x1321 + x1322 + x1323 + x499*x681
        x1325 = x674*x80
        x1326 = x1*x308
        x1327 = x681*x75
        x1328 = x216*x311
        x1329 = x325*x38
        x1330 = x1112*x6
        x1331 = x500*x6
        x1332 = x14*x886
        x1333 = x20*x886
        x1334 = x1244*x313
        x1335 = -x1325 - x1326 - x1327 - x1328 + x1329 - x1330 - x1331 + x1332 + x1333 + x1334
        x1336 = x175*x472
        x1337 = 2*x179
        x1338 = x1337*x325
        x1339 = a_0*x505
        x1340 = x1339*x47
        x1341 = x1339*x50
        x1342 = x1339*x53
        x1343 = x25*x487
        x1344 = x487*x9
        x1345 = x23*x487
        x1346 = x27*x487
        x1347 = x480*x75
        x1348 = x32*x512
        x1349 = x36*x512
        x1350 = x172*x502
        x1351 = x170*x502
        x1352 = x33*x599
        x1353 = x499*x575
        x1354 = x36*x500
        x1355 = x152*x646
        x1356 = x200*x575
        x1357 = -x1336 + x1338 - x1340 - x1341 - x1342 - x1343 - x1344 + x1345 + x1346 + x1347 - x1348 - x1349 - x1350 - x1351 - x1352 - x1353 - x1354 + x1355 + x1356
        x1358 = 8*d_5*x68
        x1359 = x306*x487
        x1360 = x325*x483
        x1361 = x711*x76
        x1362 = x711*x77
        x1363 = x1266*x14
        x1364 = x68*x720
        x1365 = x437*x989
        x1366 = -8*a_0*a_4*r_12*r_23 + x1260*x381 + x1266*x35 + x175*(R_l_inv_22*x716 + 4*x305)
        x1367 = x1325 + x1326 + x1327 + x1328 - x1329 + x1330 + x1331 - x1332 - x1333 - x1334
        x1368 = -4*a_0*d_3*r_13*r_22 - 4*a_4*x175*x313 + x1277 + x1278 + x1279 + x1280 + x1281 + x1282 + x1283 + x1284 + x1285 + x1286 + x1287 + x1288 + x1290 + x1291 + x1292 + x1293 + x1294 + x1295 + x1296 + x1297 + x1298 + x1299 + x1300 + x1301 + x1302 + x1303 + x1304 + x1305 + x1306 + x1307 + x1308 + x1309 + x1310 + x1311 + x1312 + x1313 + x1314 + x1315 + x1316 + x1317 + x1318 + x1319 + x1320 + x1321 + x1322 + x1323
        x1369 = x267*(R_l_inv_27*x716 + 4*x315)
        x1370 = x308*x5
        x1371 = 8*d_3
        x1372 = x1371*x262
        x1373 = x1371*x263
        x1374 = x1371*x264
        x1375 = x1266*x437
        x1376 = 8*x146
        x1377 = x117*x1217
        x1378 = -x100*x286 + x101*x286 - x102*x286 - x105*x286 + x107*x286 - x108*x286 + x109*x1377 - x110*x286 + x111*x286 - x112*x286 + x1179*x727 - x1215*x18 - x1215*x56 + x1219*x127 + x1219*x128 + x1219*x47 + x1219*x50 + x1219*x53 + x129*x512 + x1376*x721 + x1376*x722 + x1377*x99 - x16*x69*x727 - x20*x58*x727 + x247*x717 + x262*x720 - x286*x357 + x286*x84 - x286*x85 - x286*x86 + x286*x96 + x286*x98 - x316*x983 + x414*x723 + x47*x512 + x50*x512 + x505*x552 + x512*x53 + x512*x60 + x56*x729 + x581*x928 + x585*x709
        x1379 = 16*a_4
        x1380 = 16*r_13
        x1381 = x1336 - x1338 + x1340 + x1341 + x1342 + x1343 + x1344 - x1345 - x1346 - x1347 + x1348 + x1349 + x1350 + x1351 + x1352 + x1353 + x1354 - x1355 - x1356
        x1382 = R_l_inv_31*x471 + 2*x766
        x1383 = x80*x922
        x1384 = -x1096
        x1385 = -x1097
        x1386 = x1*x754
        x1387 = x1275*x787
        x1388 = -x38*x760
        x1389 = x216*x767
        x1390 = 2*x752
        x1391 = -x1244*x787
        x1392 = x1105 + x1137 + x119*x1390 + x1382*x70 + x1383 + x1384 + x1385 + x1386 - x1387 + x1388 + x1389 + x1391
        x1393 = x175*x894
        x1394 = x1337*x760
        x1395 = x502*x6
        x1396 = d_3*x497
        x1397 = x1393 - x1394 - x1395 + x1396
        x1398 = x144*x873
        x1399 = r_22*x876
        x1400 = d_3*x591
        x1401 = x805*x853
        x1402 = x1390*x154
        x1403 = x595*x6
        x1404 = x6*x646
        x1405 = x142*x681
        x1406 = x143*x681
        x1407 = -x1398 + x1399 - x1400 + x1401 + x1402 - x1403 - x1404 + x1405 + x1406
        x1408 = R_l_inv_32*x716 + 4*x807
        x1409 = x1371*x8 + x483*x760 + x487*x808 + x829*x989
        x1410 = -x1393 + x1394 + x1395 - x1396
        x1411 = x1398 - x1399 + x1400 - x1401 - x1402 + x1403 + x1404 - x1405 - x1406
        x1412 = x247*x986
        x1413 = x286*x832
        x1414 = d_3*x704
        x1415 = x505*x879
        x1416 = x752*x983
        x1417 = d_3*x1269
        x1418 = d_3*x1271
        x1419 = x1205*x723
        x1420 = d_3*x1270
        x1421 = -x1213 - x1214 + x1221 + x1266*x829 + x267*(R_l_inv_37*x716 + 4*x751) - x285*x767 + x5*x754 - x65*x760
        x1422 = -2*a_0*x118*x752 + x1105 + x1229 - x1382*x70 + x1383 + x1384 + x1385 + x1386 + x1387 + x1388 + x1389 + x1391
        x1423 = x154 + x80
        x1424 = x200 - x499
        x1425 = x1272*x47
        x1426 = x1272*x50
        x1427 = x1272*x53
        x1428 = a_0*x727
        x1429 = x1428*x25
        x1430 = x1428*x9
        x1431 = 8*x75
        x1432 = 16*x56
        x1433 = x1432*x32
        x1434 = x1432*x36
        x1435 = x1219*x32
        x1436 = x1216*x147
        x1437 = x1428*x19
        x1438 = x499*x720
        x1439 = x1219*x36
        x1440 = 8*x12
        x1441 = 8*x95
        x1442 = r_11*x1441*x151 + r_13*x116*x1441 + 8*r_21*x414 + x100*x505 + x102*x505 - x105*x505 + x107*x505 - x108*x505 + x11*x1219 + x110*x505 + x112*x505 - x1216*x169 + x1217*x585 + x1219*x13 + x1219*x15 + x1219*x23 + x1219*x25 + x1219*x27 - x1219*x9 + x14*x1431 + x1440*x17 + x1440*x21 + x1440*x23 + x1440*x25 + x1440*x27 - x1440*x9 - x49*x726 - x505*x96 + x505*x97 - x505*x98 - x52*x726 - x566*x724 + x720*x75 - x720*x78 + 4*x748 + 4*x749 - 4*x750 + 4*x757 + 4*x758
        x1443 = -x154 + x80
        x1444 = 16*x126
        x1445 = 16*x20
        x1446 = 16*x18
        x1447 = 16*x146
        x1448 = 16*r_23*x117
        x1449 = -x200 + x499
        x1450 = x28 - x8
        x1451 = x1450 + x37
        x1452 = -x44 + x63
        x1453 = x1450 + x66
        x1454 = x183 + x300
        x1455 = x219 + x299
        x1456 = x272 + x291
        x1457 = x346 + x348 + x349 + x354 + x356 + x359 + x362 + x363 + x364 + x365 + x366 + x367 + x368 + x370 + x372 + x374 + x376 + x378 + x380 + x385 + x387 + x389 + x391 + x393 + x399 + x400 + x401 + x403 + x405 + x406 + x407 + x408 + x409 + x411 + x415 + x417 + x420 + x423 + x425 + x426 + x427 + x428 + x430 + x470 + x614
        x1458 = x473 + x474 + x475 + x476 + x477 + x479 + x481 + x482 + x484 + x486 + x488 + x489 + x491 + x492 + x493 + x494 - x496 + x498 + x501 + x503 + x504 + x507 + x508 + x509 + x511 + x513
        x1459 = x689 - x690 - x691 - x692 - x693 - x694 - x696 - x697 - x699 - x700 - x701 - x702
        x1460 = x431 + x432 + x433 + x438 + x605 + x606 + x607 + x608 + x609 + x610 + x611 + x612 + x613
        x1461 = x889 - x890 - x891 + x892
        x1462 = x815 + x817 + x821 + x823 + x826 + x827
        x1463 = x1008 + x1462
        x1464 = x204*x232 + x253*x752 + x255*x787 + x695*x8 - x972 - x973
        x1465 = -x1020
        x1466 = x1021 + x1022 + x1023
        x1467 = x1226 + x1465 + x1466
        x1468 = x1107 - x1108 - x1109 - x1110 - x1111 - x1113 - x1114 - x1115 - x1116 - x1117
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = x0
        A[0, 1] = x2
        A[0, 2] = r_23
        A[0, 3] = x4
        A[0, 4] = -x5
        A[0, 5] = x3
        A[0, 6] = r_23
        A[0, 7] = x1
        A[0, 8] = x0
        A[1, 0] = x30 + x37
        A[1, 1] = x39
        A[1, 2] = x37 + x40
        A[1, 3] = x42 + x64
        A[1, 4] = -x65
        A[1, 5] = x41 + x64
        A[1, 6] = -x30 - x66
        A[1, 7] = x38
        A[1, 8] = -x40 - x66
        A[2, 0] = -x134 - x165 - x174 - x183
        A[2, 1] = x186 + x188 + x190 + x192 - x208 - x219
        A[2, 2] = x174 + x223 + x224 + x226
        A[2, 3] = x228 + x230 - x231 - x233 + x235 + x238 - x258 - x272
        A[2, 4] = x121*x280 + x184*x277 - x276 - x279 + x290
        A[2, 5] = -x258 + x259 + x261 + x268 - x269 + x271 - x291
        A[2, 6] = x174 + x294 + x296 + x298
        A[2, 7] = -x208 + x209 - x210 - x211 + x212 + x213 - x215 + x217 + x218 - x299
        A[2, 8] = -x115 - x124 + x223 + x295 + x301 - x68 - x74 - x82
        A[3, 0] = x345 + x431 + x445 + x470
        A[3, 1] = x514 + x524 + x604
        A[3, 2] = x345 - x346 - x348 - x349 + x353 + x355 + x358 + x361 - x363 - x364 - x365 - x366 - x367 - x368 + x369 + x371 + x373 + x375 + x377 + x379 + x384 + x386 + x388 + x390 + x392 - x399 - x400 - x401 - x403 - x405 - x406 - x407 - x408 - x409 - x411 - x415 - x417 + x419 + x422 - x425 - x426 - x427 - x428 - x430 + x614 + x615
        A[3, 3] = x673 + x688 + x703
        A[3, 4] = x705 + x706 + x707 + x710 + x712 + x713 - x714 - x715 - x730
        A[3, 5] = -x703 - x731 - x732
        A[3, 6] = -x735 - x739 - x740
        A[3, 7] = x514 + x741 + x742
        A[3, 8] = x740 + x744 + x747
        A[4, 0] = x748 + x749 + x750 + x753 - x755 - x757 - x758 + x761 + x762 + x763 + x764 + x765 - x768 - x769 - x770 - x771 - x772 - x773 - x774 + x775 + x777 + x778 - x779 - x780 - x781 - x782 - x783 - x784 - x785 - x786 - x788 - x791 - x792 - x793 - x794 + x795 + x796 - x797 - x798 - x799 - x800 - x801 + x802 + x803 + x831 + x849 + x872
        A[4, 1] = -x888 - x893 - x898
        A[4, 2] = -x916 - x919 - x921
        A[4, 3] = -x971 - x974 - x985
        A[4, 4] = 8*Py*d_3*r_11*r_22 + 8*Pz*d_3*r_11*r_32 + 4*a_0*r_23*x805 + x274*x986 - x987 - x988 - x990 - x991 - x992
        A[4, 5] = -x974 - x993 - x994
        A[4, 6] = x1002 + x1005 + x919
        A[4, 7] = x1006 + x1007 + x893
        A[4, 8] = x1004 + x1008 + x1012 + x1013 + x814 + x830 + x918
        A[5, 0] = -x1030 - x1041 - x1095
        A[5, 1] = x1106 + x1118 + x1137
        A[5, 2] = -x1030 - x1142 - x1145
        A[5, 3] = x1146 + x1147 + x1148 - x1149 + x1150 + x1151 + x1152 + x1153 + x1154 - x1155 - x1156 - x1157 - x1158 - x1159 - x1160 - x1161 - x1162 - x1163 + x1165 + x1166 + x1167 + x1168 + x1169 + x1170 + x1171 + x1172 + x1173 + x1174 + x1175 + x1176 - x1177 - x1178 - x1180 - x1181 - x1182 - x1183 + x1184 + x1185 + x1186 + x1187 + x1188 - x1189 - x1190 - x1199 - x1210
        A[5, 4] = a_4*x729 - x1211 + x1212*x727 + x1222 + x128*x711 + x129*x711 + x47*x711 + x50*x711 + x53*x711 + x60*x711 + x708*x723
        A[5, 5] = -x1191 + x1192 - x1193 - x1195 + x1197 + x1198 - x1224
        A[5, 6] = x1094 + x1143 + x1225 + x1227
        A[5, 7] = -x1118 - x1228 - x1229
        A[5, 8] = x1020 + x1032 + x1034 + x1037 + x1040 + x1138 + x1139 + x1140 + x1231
        B = np.zeros(shape=(6, 9))
        B[0, 1] = -x505
        B[0, 4] = -x727
        B[0, 7] = x505
        B[1, 0] = x214
        B[1, 1] = -x886
        B[1, 2] = x214
        B[1, 3] = x287
        B[1, 4] = -x989
        B[1, 5] = x287
        B[1, 6] = x1232
        B[1, 7] = x886
        B[1, 8] = x1232
        B[2, 0] = -x1234 + x1235 - x1247 - x1252
        B[2, 1] = -x1103 + x1253 + x1254 - x1255 - x1257 - x1258 - x1259 - x1261 + x591
        B[2, 2] = -x1247 - x1248 + x1249 + x1250 + x1251 - x1262 - x394
        B[2, 3] = -x1263 - x1265 - x1267 + x1268 + x156*x285 + x65*x81
        B[2, 4] = -x1220 + x1269 - x1270 + x1271 + x1272*x137 - x1273*x81 + x1274*x167 + x704
        B[2, 5] = 4*R_l_inv_51*a_0*a_1*r_21 + 4*R_l_inv_56*a_1*d_3*r_11 - x1263 - x1265 - x1267 - x1268
        B[2, 6] = -x1236 + x1237 + x1238 + x1241 + x1242 - x1243 + x1246 + x125*x1275 + x1276
        B[2, 7] = x1103 - x1253 - x1254 + x1255 + x1257 + x1258 + x1259 - x1261 - x591
        B[2, 8] = -x1236 + x1237 + x1238 - x1239 - x1240 + x1242 - x1243 - x1245 - x1276
        B[3, 0] = x1324 + x1335 + x1357
        B[3, 1] = x1358 - x1359 - x1360 + x1361 + x1362 - x1363 - x1364 - x1365 - x1366
        B[3, 2] = -x1357 - x1367 - x1368
        B[3, 3] = x1205*x727 - x1369 - x1370 - x1372 - x1373 - x1374 - x1375 + x1378 + x20*x989 + x285*x311 + x325*x65
        B[3, 4] = x1212*x1380 + x1272*x306 - x1273*x325 + x1274*x437 + x1379*x262 + x1379*x264 - x1379*x265 - x1379*x266
        B[3, 5] = 8*Py*d_3*r_12*r_23 + 8*Pz*d_3*r_12*r_33 + 4*a_0*r_21*x311 + 4*d_3*r_11*x325 - x1369 - x1370 - x1372 - x1373 - x1374 - x1375 - x1378
        B[3, 6] = -x1335 - x1368 - x1381
        B[3, 7] = -x1358 + x1359 + x1360 - x1361 - x1362 + x1363 + x1364 + x1365 - x1366
        B[3, 8] = x1324 + x1367 + x1381
        B[4, 0] = x1392 + x1397 + x1407
        B[4, 1] = x1260*x818 + x1408*x175 + x1409
        B[4, 2] = x1392 + x1410 + x1411
        B[4, 3] = -x1412 + x1413 - x1414 - x1415 + x1416 - x1417 - x1418 + x1419 + x1420 + x1421
        B[4, 4] = 16*a_4*d_3*r_11 + 8*a_4*r_11*x760 - x1272*x808 - x1274*x829
        B[4, 5] = x1412 - x1413 + x1414 + x1415 - x1416 + x1417 + x1418 - x1419 - x1420 + x1421
        B[4, 6] = -x1407 - x1410 - x1422
        B[4, 7] = 8*d_3*x175*x787 + x1408*x175 - x1409
        B[4, 8] = -x1397 - x1411 - x1422
        B[5, 0] = x204*(-x1423 - x1424)
        B[5, 1] = x1216*x152 - x1425 - x1426 - x1427 + x1428*x23 + x1428*x27 - x1429 - x1430 + x1431*x35 - x1433 - x1434 - x1435 - x1436 - x1437 - x1438 - x1439 + x1442 + x200*x720
        B[5, 2] = x204*(x1424 + x1443)
        B[5, 3] = x711*(-x251 - x267)
        B[5, 4] = 16*r_11*x414 + 16*r_12*x1179 - x100*x727 + x101*x727 - x102*x727 - x105*x727 + x107*x727 - x108*x727 + x109*x1448 - x110*x727 + x111*x727 - x112*x727 + x127*x1446 + x128*x1446 + x129*x1432 + x1380*x585 + x1432*x47 + x1432*x50 + x1432*x53 + x1432*x60 + x1441*x928 - x1444*x18 - x1444*x56 + x1445*x262 - x1445*x265 + x1446*x47 + x1446*x50 + x1446*x53 + x1446*x57 + x1447*x721 + x1447*x722 + x1448*x99 - 16*x16*x273 + x727*x84 + x727*x85 - x727*x86 + x727*x96 + x727*x98
        B[5, 5] = x711*(-x251 + x262 + x263 + x264 - x265 - x266)
        B[5, 6] = x204*(x1423 + x1449)
        B[5, 7] = 8*Py*a_0*r_12*x22 + 8*Py*a_0*r_12*x26 + 8*Pz*a_0*r_12*r_21*r_31 + 8*Pz*a_0*r_12*r_23*r_33 + 8*a_0*d_5*r_13*r_22 - x1425 - x1426 - x1427 - x1429 - x1430 - x1433 - x1434 - x1435 - x1436 - x1437 - x1438 - x1439 - x1442
        B[5, 8] = x204*(-x1443 - x1449)
        C = np.zeros(shape=(6, 9))
        C[0, 0] = x0
        C[0, 1] = x1
        C[0, 2] = r_23
        C[0, 3] = x4
        C[0, 4] = x5
        C[0, 5] = x3
        C[0, 6] = r_23
        C[0, 7] = x2
        C[0, 8] = x0
        C[1, 0] = x1451 + x7
        C[1, 1] = x38
        C[1, 2] = x1451 + x6
        C[1, 3] = x1452 + x42
        C[1, 4] = x65
        C[1, 5] = x1452 + x41
        C[1, 6] = -x1453 - x7
        C[1, 7] = x39
        C[1, 8] = -x1453 - x6
        C[2, 0] = a_2 + x120 + x1454 + x220 + x221 + x224 + x298 + x72
        C[2, 1] = a_0*x187*x202 + x1455 + x159*x71 + x196 - x197 + x199 + x201 - x205 + x207
        C[2, 2] = -x134 - x176 - x182 - x222 - x225 - x297 - x300
        C[2, 3] = x1456 + x241 + x248 + x250 + x252 - x254 - x256 + x257
        C[2, 4] = 4*R_l_inv_53*a_1*r_13 + 4*R_l_inv_57*a_1*d_3*r_13 - x276 - x279 - x290
        C[2, 5] = -x1456 - x240 + x248 + x250 + x252 - x254 - x256 + x257
        C[2, 6] = -x114 - x224 - x294 - x301 - x67 - x83
        C[2, 7] = -x1455 - x193 - x195 - x197 + x199 + x201 - x203 - x205 + x207
        C[2, 8] = x1454 + x226 + x292 + x293 + x296
        C[3, 0] = -x1457 - x309 - x318 - x319 - x322 - x327 - x328 - x331 - x336 - x338 - x341 - x342 - x343 - x733 - x734 - x743 - x745
        C[3, 1] = x1458 + x604 + x741
        C[3, 2] = x1457 + x307 + x314 + x320 + x323 + x329 + x332 + x339 + x736 + x737 + x738 + x746
        C[3, 3] = -x1459 - x688 - x732
        C[3, 4] = -x705 - x706 - x707 - x710 - x712 - x713 + x714 + x715 - x730
        C[3, 5] = x1459 + x673 + x731
        C[3, 6] = x1460 + x739 + x744
        C[3, 7] = x1458 + x524 + x742
        C[3, 8] = -x1460 - x735 - x747
        C[4, 0] = x1000 + x1001 + x1009 + x1010 + x1011 + x1013 + x831 + x840 + x842 + x844 + x920 + x995 + x996 + x997 + x998 + x999
        C[4, 1] = -x1007 - x1461 - x898
        C[4, 2] = x1005 + x1463 + x921
        C[4, 3] = x1464 + x985 + x994
        C[4, 4] = x1371*x227 + x1371*x229 + x274*x986 + x278*x879 - x987 - x988 - x990 - x991 + x992
        C[4, 5] = x1464 + x971 + x993
        C[4, 6] = -x1002 - x1463 - x916
        C[4, 7] = x1006 + x1461 + x888
        C[4, 8] = -x1003 - x1012 - x1462 - x810 - x812 - x824 - x913 - x917
        C[5, 0] = -x1095 - x1142 - x1467
        C[5, 1] = -x1137 - x1228 - x1468
        C[5, 2] = -x1041 - x1145 - x1467
        C[5, 3] = x1199 - x1200 - x1201 - x1202 - x1203 - x1204 - x1206 - x1207 - x1208 - x1209 + x1223 + x126*x681
        C[5, 4] = 8*Px*a_4*x46 + 8*Px*a_4*x49 + 8*Px*a_4*x52 + 8*Py*a_4*r_11*r_21 + 8*Py*a_4*r_12*r_22 + 8*Py*a_4*r_13*r_23 + 8*Pz*a_4*r_11*r_31 + 8*Pz*a_4*r_12*r_32 + 8*Pz*a_4*r_13*r_33 - x1211 - x1222
        C[5, 5] = x1191 - x1192 - x1194 - x1196 - x1197 - x1198 - x1224
        C[5, 6] = -x1014 + x1017 + x1018 + x1019 + x1024 + x1025 + x1026 + x1027 + x1028 + x1029 - x1225 - x1230 - x1466
        C[5, 7] = x1106 + x1229 + x1468
        C[5, 8] = x1031 + x1033 + x1035 + x1036 + x1038 + x1039 + x1141 + x1231 + x1465
        local_solutions = compute_solution_from_tanhalf_LME(A, B, C)
        for local_solutions_i in local_solutions:
            solution_i: IkSolution = make_ik_solution()
            solution_i[7] = local_solutions_i
            appended_idx = append_solution_to_queue(solution_i)
            add_input_index_to(2, appended_idx)
    # Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_5_processor()
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
            th_5 = this_solution[7]
            checked_result: bool = (abs(Px - a_4*r_11*math.cos(th_5) + a_4*r_12*math.sin(th_5) - d_5*r_13) <= 1.0e-6) and (abs(Py - a_4*r_21*math.cos(th_5) + a_4*r_22*math.sin(th_5) - d_5*r_23) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor()
    # Finish code for equation all-zero dispatcher node 2
    
    # Code for explicit solution node 3, solved variable is th_0
    def ExplicitSolutionNode_node_3_solve_th_0_processor():
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
            th_5 = this_solution[7]
            condition_0: bool = (abs(Px - a_4*r_11*math.cos(th_5) + a_4*r_12*math.sin(th_5) - d_5*r_13) >= zero_tolerance) or (abs(Py - a_4*r_21*math.cos(th_5) + a_4*r_22*math.sin(th_5) - d_5*r_23) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_4*math.sin(th_5)
                x1 = a_4*math.cos(th_5)
                x2 = math.atan2(Py - d_5*r_23 - r_21*x1 + r_22*x0, Px - d_5*r_13 - r_11*x1 + r_12*x0)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs(Px - a_4*r_11*math.cos(th_5) + a_4*r_12*math.sin(th_5) - d_5*r_13) >= zero_tolerance) or (abs(Py - a_4*r_21*math.cos(th_5) + a_4*r_22*math.sin(th_5) - d_5*r_23) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = a_4*math.sin(th_5)
                x1 = a_4*math.cos(th_5)
                x2 = math.atan2(Py - d_5*r_23 - r_21*x1 + r_22*x0, Px - d_5*r_13 - r_11*x1 + r_12*x0)
                # End of temp variables
                this_solution[0] = x2 + math.pi
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_0_processor()
    # Finish code for explicit solution node 3
    
    # Code for non-branch dispatcher node 4
    # Actually, there is no code
    
    # Code for explicit solution node 5, solved variable is th_3
    def ExplicitSolutionNode_node_5_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[4]
        this_input_valid: bool = node_input_validity[4]
        if not this_input_valid:
            return
        
        # The solution of non-root node 5
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_5 = this_solution[7]
            condition_0: bool = (abs((r_11*math.sin(th_0) - r_21*math.cos(th_0))*math.sin(th_5) + (r_12*math.sin(th_0) - r_22*math.cos(th_0))*math.cos(th_5)) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                x2 = safe_acos((-r_11*x0 + r_21*x1)*math.sin(th_5) + (-r_12*x0 + r_22*x1)*math.cos(th_5))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[5] = x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                
            condition_1: bool = (abs((r_11*math.sin(th_0) - r_21*math.cos(th_0))*math.sin(th_5) + (r_12*math.sin(th_0) - r_22*math.cos(th_0))*math.cos(th_5)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                x2 = safe_acos((-r_11*x0 + r_21*x1)*math.sin(th_5) + (-r_12*x0 + r_22*x1)*math.cos(th_5))
                # End of temp variables
                this_solution[5] = -x2
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_3_processor()
    # Finish code for explicit solution node 4
    
    # Code for non-branch dispatcher node 6
    # Actually, there is no code
    
    # Code for explicit solution node 7, solved variable is th_2
    def ExplicitSolutionNode_node_7_solve_th_2_processor():
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
            th_0 = this_solution[0]
            th_5 = this_solution[7]
            condition_0: bool = (2*abs(a_1*a_2) >= zero_tolerance) or (2*abs(a_1*d_3) >= zero_tolerance) or (abs(Px**2 - 2*Px*a_0*math.cos(th_0) - 2*Px*a_4*r_11*math.cos(th_5) + 2*Px*a_4*r_12*math.sin(th_5) - 2*Px*d_5*r_13 + Py**2 - 2*Py*a_0*math.sin(th_0) - 2*Py*a_4*r_21*math.cos(th_5) + 2*Py*a_4*r_22*math.sin(th_5) - 2*Py*d_5*r_23 + Pz**2 - 2*Pz*a_4*r_31*math.cos(th_5) + 2*Pz*a_4*r_32*math.sin(th_5) - 2*Pz*d_5*r_33 + a_0**2 + a_0*a_4*r_11*math.cos(th_0 - th_5) + a_0*a_4*r_11*math.cos(th_0 + th_5) + a_0*a_4*r_12*math.sin(th_0 - th_5) - a_0*a_4*r_12*math.sin(th_0 + th_5) + a_0*a_4*r_21*math.sin(th_0 - th_5) + a_0*a_4*r_21*math.sin(th_0 + th_5) - a_0*a_4*r_22*math.cos(th_0 - th_5) + a_0*a_4*r_22*math.cos(th_0 + th_5) + 2*a_0*d_5*r_13*math.cos(th_0) + 2*a_0*d_5*r_23*math.sin(th_0) - a_1**2 - a_2**2 + (1/2)*a_4**2*r_11**2*math.cos(2*th_5) + (1/2)*a_4**2*r_11**2 - a_4**2*r_11*r_12*math.sin(2*th_5) - 1/2*a_4**2*r_12**2*math.cos(2*th_5) + (1/2)*a_4**2*r_12**2 + (1/2)*a_4**2*r_21**2*math.cos(2*th_5) + (1/2)*a_4**2*r_21**2 - a_4**2*r_21*r_22*math.sin(2*th_5) - 1/2*a_4**2*r_22**2*math.cos(2*th_5) + (1/2)*a_4**2*r_22**2 + (1/2)*a_4**2*r_31**2*math.cos(2*th_5) + (1/2)*a_4**2*r_31**2 - a_4**2*r_31*r_32*math.sin(2*th_5) - 1/2*a_4**2*r_32**2*math.cos(2*th_5) + (1/2)*a_4**2*r_32**2 + 2*a_4*d_5*r_11*r_13*math.cos(th_5) - 2*a_4*d_5*r_12*r_13*math.sin(th_5) + 2*a_4*d_5*r_21*r_23*math.cos(th_5) - 2*a_4*d_5*r_22*r_23*math.sin(th_5) + 2*a_4*d_5*r_31*r_33*math.cos(th_5) - 2*a_4*d_5*r_32*r_33*math.sin(th_5) - d_3**2 + d_5**2*r_13**2 + d_5**2*r_23**2 + d_5**2*r_33**2) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*a_1
                x1 = math.atan2(-d_3*x0, a_2*x0)
                x2 = a_2**2
                x3 = a_1**2
                x4 = 4*x3
                x5 = d_3**2
                x6 = 2*Px
                x7 = d_5*r_13
                x8 = 2*Py
                x9 = d_5*r_23
                x10 = 2*Pz
                x11 = d_5*r_33
                x12 = a_0*math.cos(th_0)
                x13 = a_0*math.sin(th_0)
                x14 = d_5**2
                x15 = a_4*math.cos(th_5)
                x16 = r_11*x15
                x17 = a_4*math.sin(th_5)
                x18 = r_12*x17
                x19 = r_21*x15
                x20 = r_22*x17
                x21 = r_31*x15
                x22 = r_32*x17
                x23 = 2*x7
                x24 = 2*x9
                x25 = a_4**2
                x26 = (1/2)*x25
                x27 = r_11**2*x26
                x28 = r_12**2*x26
                x29 = r_21**2*x26
                x30 = r_22**2*x26
                x31 = r_31**2*x26
                x32 = r_32**2*x26
                x33 = th_0 + th_5
                x34 = math.cos(x33)
                x35 = a_0*a_4
                x36 = r_11*x35
                x37 = x35*math.sin(x33)
                x38 = r_22*x35
                x39 = 2*x11
                x40 = th_0 - th_5
                x41 = math.cos(x40)
                x42 = x35*math.sin(x40)
                x43 = 2*th_5
                x44 = x25*math.sin(x43)
                x45 = math.cos(x43)
                x46 = Px**2 + Py**2 + Pz**2 + a_0**2 - r_11*r_12*x44 - r_12*x37 + r_12*x42 + r_13**2*x14 - r_21*r_22*x44 + r_21*x37 + r_21*x42 + r_23**2*x14 - r_31*r_32*x44 + r_33**2*x14 - x10*x11 - x10*x21 + x10*x22 + x12*x23 - x12*x6 + x13*x24 - x13*x8 + x16*x23 - x16*x6 - x18*x23 + x18*x6 + x19*x24 - x19*x8 - x2 - x20*x24 + x20*x8 + x21*x39 - x22*x39 + x27*x45 + x27 - x28*x45 + x28 + x29*x45 + x29 - x3 - x30*x45 + x30 + x31*x45 + x31 - x32*x45 + x32 + x34*x36 + x34*x38 + x36*x41 - x38*x41 - x5 - x6*x7 - x8*x9
                x47 = safe_sqrt(x2*x4 + x4*x5 - x46**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x1 + math.atan2(x47, x46)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
            condition_1: bool = (2*abs(a_1*a_2) >= zero_tolerance) or (2*abs(a_1*d_3) >= zero_tolerance) or (abs(Px**2 - 2*Px*a_0*math.cos(th_0) - 2*Px*a_4*r_11*math.cos(th_5) + 2*Px*a_4*r_12*math.sin(th_5) - 2*Px*d_5*r_13 + Py**2 - 2*Py*a_0*math.sin(th_0) - 2*Py*a_4*r_21*math.cos(th_5) + 2*Py*a_4*r_22*math.sin(th_5) - 2*Py*d_5*r_23 + Pz**2 - 2*Pz*a_4*r_31*math.cos(th_5) + 2*Pz*a_4*r_32*math.sin(th_5) - 2*Pz*d_5*r_33 + a_0**2 + a_0*a_4*r_11*math.cos(th_0 - th_5) + a_0*a_4*r_11*math.cos(th_0 + th_5) + a_0*a_4*r_12*math.sin(th_0 - th_5) - a_0*a_4*r_12*math.sin(th_0 + th_5) + a_0*a_4*r_21*math.sin(th_0 - th_5) + a_0*a_4*r_21*math.sin(th_0 + th_5) - a_0*a_4*r_22*math.cos(th_0 - th_5) + a_0*a_4*r_22*math.cos(th_0 + th_5) + 2*a_0*d_5*r_13*math.cos(th_0) + 2*a_0*d_5*r_23*math.sin(th_0) - a_1**2 - a_2**2 + (1/2)*a_4**2*r_11**2*math.cos(2*th_5) + (1/2)*a_4**2*r_11**2 - a_4**2*r_11*r_12*math.sin(2*th_5) - 1/2*a_4**2*r_12**2*math.cos(2*th_5) + (1/2)*a_4**2*r_12**2 + (1/2)*a_4**2*r_21**2*math.cos(2*th_5) + (1/2)*a_4**2*r_21**2 - a_4**2*r_21*r_22*math.sin(2*th_5) - 1/2*a_4**2*r_22**2*math.cos(2*th_5) + (1/2)*a_4**2*r_22**2 + (1/2)*a_4**2*r_31**2*math.cos(2*th_5) + (1/2)*a_4**2*r_31**2 - a_4**2*r_31*r_32*math.sin(2*th_5) - 1/2*a_4**2*r_32**2*math.cos(2*th_5) + (1/2)*a_4**2*r_32**2 + 2*a_4*d_5*r_11*r_13*math.cos(th_5) - 2*a_4*d_5*r_12*r_13*math.sin(th_5) + 2*a_4*d_5*r_21*r_23*math.cos(th_5) - 2*a_4*d_5*r_22*r_23*math.sin(th_5) + 2*a_4*d_5*r_31*r_33*math.cos(th_5) - 2*a_4*d_5*r_32*r_33*math.sin(th_5) - d_3**2 + d_5**2*r_13**2 + d_5**2*r_23**2 + d_5**2*r_33**2) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*a_1
                x1 = math.atan2(-d_3*x0, a_2*x0)
                x2 = a_2**2
                x3 = a_1**2
                x4 = 4*x3
                x5 = d_3**2
                x6 = 2*Px
                x7 = d_5*r_13
                x8 = 2*Py
                x9 = d_5*r_23
                x10 = 2*Pz
                x11 = d_5*r_33
                x12 = a_0*math.cos(th_0)
                x13 = a_0*math.sin(th_0)
                x14 = d_5**2
                x15 = a_4*math.cos(th_5)
                x16 = r_11*x15
                x17 = a_4*math.sin(th_5)
                x18 = r_12*x17
                x19 = r_21*x15
                x20 = r_22*x17
                x21 = r_31*x15
                x22 = r_32*x17
                x23 = 2*x7
                x24 = 2*x9
                x25 = a_4**2
                x26 = (1/2)*x25
                x27 = r_11**2*x26
                x28 = r_12**2*x26
                x29 = r_21**2*x26
                x30 = r_22**2*x26
                x31 = r_31**2*x26
                x32 = r_32**2*x26
                x33 = th_0 + th_5
                x34 = math.cos(x33)
                x35 = a_0*a_4
                x36 = r_11*x35
                x37 = x35*math.sin(x33)
                x38 = r_22*x35
                x39 = 2*x11
                x40 = th_0 - th_5
                x41 = math.cos(x40)
                x42 = x35*math.sin(x40)
                x43 = 2*th_5
                x44 = x25*math.sin(x43)
                x45 = math.cos(x43)
                x46 = Px**2 + Py**2 + Pz**2 + a_0**2 - r_11*r_12*x44 - r_12*x37 + r_12*x42 + r_13**2*x14 - r_21*r_22*x44 + r_21*x37 + r_21*x42 + r_23**2*x14 - r_31*r_32*x44 + r_33**2*x14 - x10*x11 - x10*x21 + x10*x22 + x12*x23 - x12*x6 + x13*x24 - x13*x8 + x16*x23 - x16*x6 - x18*x23 + x18*x6 + x19*x24 - x19*x8 - x2 - x20*x24 + x20*x8 + x21*x39 - x22*x39 + x27*x45 + x27 - x28*x45 + x28 + x29*x45 + x29 - x3 - x30*x45 + x30 + x31*x45 + x31 - x32*x45 + x32 + x34*x36 + x34*x38 + x36*x41 - x38*x41 - x5 - x6*x7 - x8*x9
                x47 = safe_sqrt(x2*x4 + x4*x5 - x46**2)
                # End of temp variables
                this_solution[4] = x1 + math.atan2(-x47, x46)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_2_processor()
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
            th_3 = this_solution[5]
            degenerate_valid_0 = (abs(th_3) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(14, node_input_i_idx_in_queue)
            
            th_3 = this_solution[5]
            degenerate_valid_1 = (abs(th_3 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(17, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(9, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor()
    # Finish code for solved_variable dispatcher node 8
    
    # Code for explicit solution node 17, solved variable is th_1th_2th_4_soa
    def ExplicitSolutionNode_node_17_solve_th_1th_2th_4_soa_processor():
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
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_33) >= zero_tolerance) or (abs(r_13*math.cos(th_0) + r_23*math.sin(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[3] = math.atan2(r_13*math.cos(th_0) + r_23*math.sin(th_0), r_33)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(18, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_1th_2th_4_soa_processor()
    # Finish code for explicit solution node 17
    
    # Code for solved_variable dispatcher node 18
    def SolvedVariableDispatcherNode_node_18_processor():
        this_node_input_index: List[int] = node_input_index[18]
        this_input_valid: bool = node_input_validity[18]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_2 = this_solution[4]
            degenerate_valid_0 = (abs(th_2 - math.pi + 1.39979827560533) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(22, node_input_i_idx_in_queue)
            
            th_2 = this_solution[4]
            degenerate_valid_1 = (abs(th_2 - 2*math.pi + 1.39979827560533) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(25, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(19, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_18_processor()
    # Finish code for solved_variable dispatcher node 18
    
    # Code for explicit solution node 25, solved variable is th_1
    def ExplicitSolutionNode_node_25_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[25]
        this_input_valid: bool = node_input_validity[25]
        if not this_input_valid:
            return
        
        # The solution of non-root node 25
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_5 = this_solution[7]
            condition_0: bool = (abs(0.985415423419357*a_2 - 0.170165928690941*d_3) >= zero_tolerance) or (abs(a_1 + 0.170165928690941*a_2 + 0.985415423419357*d_3) >= zero_tolerance) or (abs(Pz - a_4*r_31*math.cos(th_5) + a_4*r_32*math.sin(th_5) - d_5*r_33) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_1 + 0.170165928690941*a_2 + 0.985415423419357*d_3
                x1 = math.atan2(x0, -0.985415423419357*a_2 + 0.170165928690941*d_3)
                x2 = -Pz + a_4*r_31*math.cos(th_5) - a_4*r_32*math.sin(th_5) + d_5*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.97104355671275*(-a_2 + 0.172684458398744*d_3)**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x1 + math.atan2(x3, x2)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(26, appended_idx)
                
            condition_1: bool = (abs(0.985415423419357*a_2 - 0.170165928690941*d_3) >= zero_tolerance) or (abs(a_1 + 0.170165928690941*a_2 + 0.985415423419357*d_3) >= zero_tolerance) or (abs(Pz - a_4*r_31*math.cos(th_5) + a_4*r_32*math.sin(th_5) - d_5*r_33) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = a_1 + 0.170165928690941*a_2 + 0.985415423419357*d_3
                x1 = math.atan2(x0, -0.985415423419357*a_2 + 0.170165928690941*d_3)
                x2 = -Pz + a_4*r_31*math.cos(th_5) - a_4*r_32*math.sin(th_5) + d_5*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.97104355671275*(-a_2 + 0.172684458398744*d_3)**2)
                # End of temp variables
                this_solution[1] = x1 + math.atan2(-x3, x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(26, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_1_processor()
    # Finish code for explicit solution node 25
    
    # Code for non-branch dispatcher node 26
    # Actually, there is no code
    
    # Code for explicit solution node 27, solved variable is th_4
    def ExplicitSolutionNode_node_27_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[26]
        this_input_valid: bool = node_input_validity[26]
        if not this_input_valid:
            return
        
        # The solution of non-root node 27
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_13*((0.985415423419357*math.sin(th_1) + 0.170165928690941*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) + r_23*((0.985415423419357*math.sin(th_1) + 0.170165928690941*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(0.170165928690941*math.sin(th_1) - 0.985415423419357*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance) or (abs(-r_13*(-0.170165928690941*math.sin(th_1) + 0.985415423419357*math.cos(th_1))*math.cos(th_0) - r_23*(-0.170165928690941*math.sin(th_1) + 0.985415423419357*math.cos(th_1))*math.sin(th_0) + r_33*(0.985415423419357*math.sin(th_1) + 0.170165928690941*math.cos(th_1))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_1)
                x3 = -0.170165928690941*x1 + 0.985415423419357*x2
                x4 = math.sin(th_0)
                x5 = math.sin(th_3)
                x6 = math.cos(th_0)
                x7 = 0.985415423419357*x1 + 0.170165928690941*x2
                x8 = x0*x7
                # End of temp variables
                this_solution[6] = math.atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*x3, -r_13*x3*x6 - r_23*x3*x4 + r_33*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_4_processor()
    # Finish code for explicit solution node 26
    
    # Code for explicit solution node 22, solved variable is th_1
    def ExplicitSolutionNode_node_22_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[22]
        this_input_valid: bool = node_input_validity[22]
        if not this_input_valid:
            return
        
        # The solution of non-root node 22
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_5 = this_solution[7]
            condition_0: bool = (abs(0.985415423419357*a_2 - 0.170165928690941*d_3) >= zero_tolerance) or (abs(-a_1 + 0.170165928690941*a_2 + 0.985415423419357*d_3) >= zero_tolerance) or (abs(Pz - a_4*r_31*math.cos(th_5) + a_4*r_32*math.sin(th_5) - d_5*r_33) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = -a_1 + 0.170165928690941*a_2 + 0.985415423419357*d_3
                x1 = math.atan2(x0, -0.985415423419357*a_2 + 0.170165928690941*d_3)
                x2 = Pz - a_4*r_31*math.cos(th_5) + a_4*r_32*math.sin(th_5) - d_5*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.97104355671275*(-a_2 + 0.172684458398744*d_3)**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x1 + math.atan2(x3, x2)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(23, appended_idx)
                
            condition_1: bool = (abs(0.985415423419357*a_2 - 0.170165928690941*d_3) >= zero_tolerance) or (abs(-a_1 + 0.170165928690941*a_2 + 0.985415423419357*d_3) >= zero_tolerance) or (abs(Pz - a_4*r_31*math.cos(th_5) + a_4*r_32*math.sin(th_5) - d_5*r_33) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = -a_1 + 0.170165928690941*a_2 + 0.985415423419357*d_3
                x1 = math.atan2(x0, -0.985415423419357*a_2 + 0.170165928690941*d_3)
                x2 = Pz - a_4*r_31*math.cos(th_5) + a_4*r_32*math.sin(th_5) - d_5*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.97104355671275*(-a_2 + 0.172684458398744*d_3)**2)
                # End of temp variables
                this_solution[1] = x1 + math.atan2(-x3, x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(23, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_22_solve_th_1_processor()
    # Finish code for explicit solution node 22
    
    # Code for non-branch dispatcher node 23
    # Actually, there is no code
    
    # Code for explicit solution node 24, solved variable is th_4
    def ExplicitSolutionNode_node_24_solve_th_4_processor():
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
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance) or (abs(-r_13*((-0.985415423419357*math.sin(th_1) - 0.170165928690941*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) - r_23*((-0.985415423419357*math.sin(th_1) - 0.170165928690941*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(0.170165928690941*math.sin(th_1) - 0.985415423419357*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance) or (abs(-r_13*(-0.170165928690941*math.sin(th_1) + 0.985415423419357*math.cos(th_1))*math.cos(th_0) - r_23*(-0.170165928690941*math.sin(th_1) + 0.985415423419357*math.cos(th_1))*math.sin(th_0) + r_33*(0.985415423419357*math.sin(th_1) + 0.170165928690941*math.cos(th_1))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_1)
                x3 = 0.170165928690941*x1 - 0.985415423419357*x2
                x4 = math.sin(th_0)
                x5 = math.sin(th_3)
                x6 = math.cos(th_0)
                x7 = 0.985415423419357*x1 + 0.170165928690941*x2
                x8 = -x0*x7
                x9 = -x3
                # End of temp variables
                this_solution[6] = math.atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*x3, r_13*x6*x9 + r_23*x4*x9 - r_33*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_4_processor()
    # Finish code for explicit solution node 23
    
    # Code for explicit solution node 19, solved variable is th_1
    def ExplicitSolutionNode_node_19_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[19]
        this_input_valid: bool = node_input_validity[19]
        if not this_input_valid:
            return
        
        # The solution of non-root node 19
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1th_2th_4_soa = this_solution[3]
            th_2 = this_solution[4]
            th_5 = this_solution[7]
            condition_0: bool = (abs(a_2*math.sin(th_2) + d_3*math.cos(th_2)) >= 1.0e-6) or (abs(a_1 + a_2*math.cos(th_2) - d_3*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_2)
                x1 = math.sin(th_2)
                x2 = a_1 + a_2*x0 - d_3*x1
                x3 = Pz - a_4*r_31*math.cos(th_5) + a_4*r_32*math.sin(th_5) - d_5*r_33
                x4 = -a_2*x1 - d_3*x0
                x5 = Px*math.cos(th_0) + Py*math.sin(th_0) - a_0 + a_4*math.cos(th_1th_2th_4_soa) - d_5*math.sin(th_1th_2th_4_soa)
                # End of temp variables
                this_solution[1] = math.atan2(-x2*x3 + x4*x5, x2*x5 + x3*x4)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(20, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_1_processor()
    # Finish code for explicit solution node 19
    
    # Code for non-branch dispatcher node 20
    # Actually, there is no code
    
    # Code for explicit solution node 21, solved variable is th_4
    def ExplicitSolutionNode_node_21_solve_th_4_processor():
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
            th_1 = this_solution[1]
            th_1th_2th_4_soa = this_solution[3]
            th_2 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[6] = -th_1 + th_1th_2th_4_soa - th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_4_processor()
    # Finish code for explicit solution node 20
    
    # Code for explicit solution node 14, solved variable is th_1
    def ExplicitSolutionNode_node_14_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[14]
        this_input_valid: bool = node_input_validity[14]
        if not this_input_valid:
            return
        
        # The solution of non-root node 14
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_2 = this_solution[4]
            th_5 = this_solution[7]
            condition_0: bool = (abs(a_2*math.sin(th_2) + d_3*math.cos(th_2)) >= 1.0e-6) or (abs(a_1 + a_2*math.cos(th_2) - d_3*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_2)
                x1 = math.sin(th_2)
                x2 = a_1 + a_2*x0 - d_3*x1
                x3 = math.sin(th_5)
                x4 = a_4*math.cos(th_5)
                x5 = Pz + a_4*r_32*x3 - d_5*r_33 - r_31*x4
                x6 = -a_2*x1 - d_3*x0
                x7 = math.cos(th_0)
                x8 = math.sin(th_0)
                x9 = Px*x7 + Py*x8 - a_0 + a_4*r_12*x3*x7 + a_4*r_22*x3*x8 - d_5*r_13*x7 - d_5*r_23*x8 - r_11*x4*x7 - r_21*x4*x8
                # End of temp variables
                this_solution[1] = math.atan2(-x2*x5 + x6*x9, x2*x9 + x5*x6)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_1_processor()
    # Finish code for explicit solution node 14
    
    # Code for non-branch dispatcher node 15
    # Actually, there is no code
    
    # Code for explicit solution node 16, solved variable is th_4
    def ExplicitSolutionNode_node_16_solve_th_4_processor():
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
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_2 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance) or (abs(-r_13*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.cos(th_0) - r_23*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_0) + r_33*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))) >= zero_tolerance) or (abs(r_13*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_23*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_0) + r_33*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_1)
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = math.cos(th_1)
                x4 = x0*x1 + x2*x3
                x5 = -x0*x2 + x1*x3
                x6 = r_13*math.cos(th_0)
                x7 = r_23*math.sin(th_0)
                # End of temp variables
                this_solution[6] = math.atan2(r_33*x4 - x5*x6 - x5*x7, r_33*x5 + x4*x6 + x4*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_4_processor()
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
            th_0 = this_solution[0]
            th_3 = this_solution[5]
            th_5 = this_solution[7]
            condition_0: bool = (abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) >= zero_tolerance) or (abs((r_11*math.sin(th_0) - r_21*math.cos(th_0))*math.cos(th_5) + (-r_12*math.sin(th_0) + r_22*math.cos(th_0))*math.sin(th_5)) >= zero_tolerance) or (abs(math.sin(th_3)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_3)**(-1)
                x1 = math.sin(th_0)
                x2 = math.cos(th_0)
                # End of temp variables
                this_solution[6] = math.atan2(x0*(-r_13*x1 + r_23*x2), x0*(-(-r_11*x1 + r_21*x2)*math.cos(th_5) + (-r_12*x1 + r_22*x2)*math.sin(th_5)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_4_processor()
    # Finish code for explicit solution node 9
    
    # Code for non-branch dispatcher node 10
    # Actually, there is no code
    
    # Code for explicit solution node 11, solved variable is th_1th_2_soa
    def ExplicitSolutionNode_node_11_solve_th_1th_2_soa_processor():
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
            th_0 = this_solution[0]
            th_3 = this_solution[5]
            th_5 = this_solution[7]
            condition_0: bool = (abs(r_31*math.sin(th_5) + r_32*math.cos(th_5)) >= zero_tolerance) or (abs((r_11*math.cos(th_0) + r_21*math.sin(th_0))*math.sin(th_5) + (r_12*math.cos(th_0) + r_22*math.sin(th_0))*math.cos(th_5)) >= zero_tolerance) or (abs(math.sin(th_3)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_3)**(-1)
                x1 = math.sin(th_5)
                x2 = math.cos(th_5)
                x3 = math.cos(th_0)
                x4 = math.sin(th_0)
                # End of temp variables
                this_solution[2] = math.atan2(x0*(-r_31*x1 - r_32*x2), x0*(-x1*(-r_11*x3 - r_21*x4) + x2*(r_12*x3 + r_22*x4)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(12, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1th_2_soa_processor()
    # Finish code for explicit solution node 10
    
    # Code for non-branch dispatcher node 12
    # Actually, there is no code
    
    # Code for explicit solution node 13, solved variable is th_1
    def ExplicitSolutionNode_node_13_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[12]
        this_input_valid: bool = node_input_validity[12]
        if not this_input_valid:
            return
        
        # The solution of non-root node 13
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1th_2_soa = this_solution[2]
            th_2 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[1] = th_1th_2_soa - th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_1_processor()
    # Finish code for explicit solution node 12
    
    # Collect the output
    ik_out: List[np.ndarray] = list()
    for i in range(len(solution_queue)):
        if not queue_element_validity[i]:
            continue
        ik_out_i = solution_queue[i]
        new_ik_i = np.zeros((robot_nq, 1))
        value_at_0 = ik_out_i[0]  # th_0
        new_ik_i[0] = value_at_0
        value_at_1 = ik_out_i[1]  # th_1
        new_ik_i[1] = value_at_1
        value_at_2 = ik_out_i[4]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[5]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[6]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[7]  # th_5
        new_ik_i[5] = value_at_5
        ik_out.append(new_ik_i)
    return ik_out


def abb_crb15000_10_1_52_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = abb_crb15000_10_1_52_ik_target_original_to_raw(T_ee)
    ik_output_raw = abb_crb15000_10_1_52_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ee_pose_i = abb_crb15000_10_1_52_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_abb_crb15000_10_1_52():
    theta_in = np.random.random(size=(6, ))
    ee_pose = abb_crb15000_10_1_52_fk(theta_in)
    ik_output = abb_crb15000_10_1_52_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = abb_crb15000_10_1_52_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_abb_crb15000_10_1_52()
