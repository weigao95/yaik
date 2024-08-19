import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 38
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 0.2942787793912432
a_1: float = 0.05
d_2: float = 0.29
d_3: float = -0.136
d_4: float = 0.1035
pre_transform_special_symbol_23: float = 0.344

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -1.4000611153196139
th_2_offset_original2raw: float = -0.17073521147528278
th_3_offset_original2raw: float = 3.141592653589793
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 0.0


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def rokae_SR3_ik_target_original_to_raw(T_ee: np.ndarray):
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


def rokae_SR3_ik_target_raw_to_original(T_ee: np.ndarray):
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


def rokae_SR3_fk(theta_input: np.ndarray):
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
    x23 = x22*x6
    x24 = 1.0*a_1
    x25 = 1.0*d_2
    x26 = 1.0*d_4
    x27 = -x11*x4 - x11*x7
    x28 = x0*x27
    x29 = -x11*x14 + x11*x3*x6
    x30 = -x1*x12 + x13*x29
    x31 = x10*x30
    x32 = -1.0*x1*x13 - 1.0*x12*x29
    x33 = 1.0*x0*x30 - 1.0*x10*x27
    x34 = x14 - x3*x6
    x35 = x0*x34
    x36 = -x4 - x7
    x37 = x10*x13*x36
    x38 = 1.0*x12*x36
    x39 = 1.0*x0*x13*x36 - 1.0*x10*x34
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -1.0*x17 - 1.0*x9
    ee_pose[0, 1] = x18*x19 + x20*x21
    ee_pose[0, 2] = x18*x21 - x19*x20
    ee_pose[0, 3] = d_3*x19 + x1*x23 + x15*x24 + x25*x8 + x26*(-x17 - x9)
    ee_pose[1, 0] = -1.0*x28 - 1.0*x31
    ee_pose[1, 1] = x18*x32 + x20*x33
    ee_pose[1, 2] = x18*x33 - x20*x32
    ee_pose[1, 3] = d_3*x32 + x11*x23 + x24*x29 + x25*x27 + x26*(-x28 - x31)
    ee_pose[2, 0] = -1.0*x35 - 1.0*x37
    ee_pose[2, 1] = -x18*x38 + x20*x39
    ee_pose[2, 2] = x18*x39 + x20*x38
    ee_pose[2, 3] = -d_3*x38 + 1.0*pre_transform_special_symbol_23 - x2*x22 + x24*x36 + x25*x34 + x26*(-x35 - x37)
    return ee_pose


def rokae_SR3_twist_jacobian(theta_input: np.ndarray):
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
    x18 = -x11*x16 - x17*(x1*x13 + x12*x14)
    x19 = x1*x4
    x20 = x1*x9
    x21 = -x19*x3 - x20*x8
    x22 = 1.0*x0*x3*x9 - x19*x8
    x23 = -x12*x6 - x13*x22
    x24 = -x16*x21 - x17*(x12*x22 - x13*x6)
    x25 = 1.0*x8
    x26 = 1.0*x3
    x27 = x25*x4 - x26*x9
    x28 = -x25*x9 - x26*x4
    x29 = x13*x28
    x30 = -x12*x17*x28 - x16*x27
    x31 = 1.0*a_0
    x32 = x31*x4
    x33 = pre_transform_special_symbol_23 - x32
    x34 = a_1*x28 + d_2*x27 + pre_transform_special_symbol_23 - x32
    x35 = a_0*x20 + a_1*x22 + d_2*x21
    x36 = -d_3*x29 + x34
    x37 = d_3*x23 + x35
    x38 = d_4*x30 + x36
    x39 = d_4*x24 + x37
    x40 = a_0*x10 + a_1*x14 + d_2*x11
    x41 = d_3*x15 + x40
    x42 = d_4*x18 + x41
    x43 = x31*x9
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = x2
    jacobian_output[0, 2] = x2
    jacobian_output[0, 3] = x11
    jacobian_output[0, 4] = x15
    jacobian_output[0, 5] = x18
    jacobian_output[1, 1] = x6
    jacobian_output[1, 2] = x6
    jacobian_output[1, 3] = x21
    jacobian_output[1, 4] = x23
    jacobian_output[1, 5] = x24
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x27
    jacobian_output[2, 4] = -x29
    jacobian_output[2, 5] = x30
    jacobian_output[3, 1] = -pre_transform_special_symbol_23*x6
    jacobian_output[3, 2] = -x33*x6
    jacobian_output[3, 3] = -x21*x34 + x27*x35
    jacobian_output[3, 4] = -x23*x36 - x29*x37
    jacobian_output[3, 5] = -x24*x38 + x30*x39
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = -x1*x33
    jacobian_output[4, 3] = x11*x34 - x27*x40
    jacobian_output[4, 4] = x15*x36 + x29*x41
    jacobian_output[4, 5] = x18*x38 - x30*x42
    jacobian_output[5, 2] = x0**2*x43 + x43*x5**2
    jacobian_output[5, 3] = -x11*x35 + x21*x40
    jacobian_output[5, 4] = -x15*x37 + x23*x41
    jacobian_output[5, 5] = -x18*x39 + x24*x42
    return jacobian_output


def rokae_SR3_angular_velocity_jacobian(theta_input: np.ndarray):
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


def rokae_SR3_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x6 = a_0*x5
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
    x19 = a_1*x18 + d_2*x13 + pre_transform_special_symbol_23 - x6
    x20 = a_0*x12
    x21 = 1.0*x10*x11*x14 - x14*x9
    x22 = a_1*x21 + d_2*x17 + x14*x20
    x23 = math.sin(th_3)
    x24 = x18*x23
    x25 = math.cos(th_3)
    x26 = -x2*x25 - x21*x23
    x27 = -d_3*x24 + x19
    x28 = d_3*x26 + x22
    x29 = math.cos(th_4)
    x30 = math.sin(th_4)
    x31 = -x13*x29 - x18*x25*x30
    x32 = -x17*x29 - x30*(-x2*x23 + x21*x25)
    x33 = d_4*x31 + x27
    x34 = d_4*x32 + x28
    x35 = 1.0*p_on_ee_x
    x36 = 1.0*x14
    x37 = p_on_ee_z*x36
    x38 = x2*x4
    x39 = x11*x2
    x40 = -x10*x38 - x39*x8
    x41 = 1.0*x1*x10*x11 - x38*x8
    x42 = a_0*x39 + a_1*x41 + d_2*x40
    x43 = 1.0*x14*x25 - x23*x41
    x44 = d_3*x43 + x42
    x45 = -x29*x40 - x30*(x23*x36 + x25*x41)
    x46 = d_4*x45 + x44
    x47 = x1*x35
    x48 = x0*x14
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = -x2*x7 + x3
    jacobian_output[0, 3] = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x22 - x17*x19
    jacobian_output[0, 4] = p_on_ee_y*x24 + p_on_ee_z*x26 - x24*x28 - x26*x27
    jacobian_output[0, 5] = -p_on_ee_y*x31 + p_on_ee_z*x32 + x31*x34 - x32*x33
    jacobian_output[1, 0] = x35
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x36 + x37
    jacobian_output[1, 2] = -x36*x7 + x37
    jacobian_output[1, 3] = p_on_ee_x*x13 - p_on_ee_z*x40 - x13*x42 + x19*x40
    jacobian_output[1, 4] = -p_on_ee_x*x24 - p_on_ee_z*x43 + x18*x23*x44 + x27*x43
    jacobian_output[1, 5] = p_on_ee_x*x31 - p_on_ee_z*x45 - x31*x46 + x33*x45
    jacobian_output[2, 1] = -x47 - x48
    jacobian_output[2, 2] = x1**2*x20 + x14**2*x20 - x47 - x48
    jacobian_output[2, 3] = -p_on_ee_x*x17 + p_on_ee_y*x40 + x17*x42 - x22*x40
    jacobian_output[2, 4] = -p_on_ee_x*x26 + p_on_ee_y*x43 + x26*x44 - x28*x43
    jacobian_output[2, 5] = -p_on_ee_x*x32 + p_on_ee_y*x45 + x32*x46 - x34*x45
    return jacobian_output


def rokae_SR3_ik_solve_raw(T_ee: np.ndarray):
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
    for i in range(38):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True
    
    # Code for non-branch dispatcher node 0
    # Actually, there is no code
    
    # Code for explicit solution node 1, solved variable is th_0
    def General6DoFNumericalReduceSolutionNode_node_1_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return
        
        # The general 6-dof solution of root node with semi-symbolic reduce
        R_l = np.zeros(shape=(8, 8))
        R_l[0, 3] = -a_0
        R_l[0, 7] = -a_1
        R_l[1, 2] = -a_0
        R_l[1, 6] = -a_1
        R_l[2, 4] = a_0
        R_l[3, 6] = -1
        R_l[4, 7] = 1
        R_l[5, 5] = 2*a_0*a_1
        R_l[6, 1] = -a_0
        R_l[7, 0] = -a_0
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
        x3 = 4*r_22
        x4 = d_3*r_22
        x5 = -x4
        x6 = d_2*r_23
        x7 = r_21**2
        x8 = Py*x7
        x9 = r_22**2
        x10 = Py*x9
        x11 = r_23**2
        x12 = Py*x11
        x13 = d_4*r_23
        x14 = Px*r_11
        x15 = r_21*x14
        x16 = Px*r_12
        x17 = r_22*x16
        x18 = Px*r_13
        x19 = r_23*x18
        x20 = Pz*r_31
        x21 = r_21*x20
        x22 = Pz*r_32
        x23 = r_22*x22
        x24 = Pz*r_33
        x25 = r_23*x24
        x26 = x10 + x12 - x13 + x15 + x17 + x19 + x21 + x23 + x25 + x8
        x27 = x26 - x6
        x28 = x27 + x5
        x29 = d_3*x1
        x30 = x27 + x4
        x31 = d_2*x1
        x32 = -x31
        x33 = d_2*x3
        x34 = x26 + x6
        x35 = x34 + x5
        x36 = x34 + x4
        x37 = Py*r_21
        x38 = x14 + x20 + x37
        x39 = R_l_inv_50*a_0
        x40 = x38*x39
        x41 = Py*r_23
        x42 = -d_4 + x18 + x24 + x41
        x43 = R_l_inv_52*a_0
        x44 = x42*x43
        x45 = d_4*r_21
        x46 = r_23*x14
        x47 = r_23*x20
        x48 = r_21*x18
        x49 = r_21*x24
        x50 = x45 + x46 + x47 - x48 - x49
        x51 = R_l_inv_57*a_0
        x52 = x50*x51
        x53 = a_0*r_22
        x54 = R_l_inv_54*x53
        x55 = -x54
        x56 = R_l_inv_56*d_2*x53
        x57 = R_l_inv_55*a_0
        x58 = 2*d_2
        x59 = x42*x58
        x60 = x57*x59
        x61 = x40 + x44 + x52 + x55 + x56 + x60
        x62 = d_3*r_21
        x63 = R_l_inv_53*a_0
        x64 = r_21*x63
        x65 = d_4*r_22
        x66 = r_23*x16
        x67 = r_23*x22
        x68 = r_22*x18
        x69 = r_22*x24
        x70 = x65 + x66 + x67 - x68 - x69
        x71 = R_l_inv_56*a_0
        x72 = x70*x71
        x73 = Py*r_22
        x74 = x16 + x22 + x73
        x75 = R_l_inv_51*a_0
        x76 = x74*x75
        x77 = d_2*r_21
        x78 = x51*x77
        x79 = 2*d_3
        x80 = x57*x74
        x81 = x79*x80
        x82 = x62 + x64 + x72 - x76 + x78 + x81
        x83 = d_3*r_23
        x84 = x71*x83
        x85 = r_22*x14
        x86 = r_22*x20
        x87 = r_21*x16
        x88 = r_21*x22
        x89 = x85 + x86 - x87 - x88
        x90 = -x84 + x89
        x91 = d_3*x75
        x92 = d_2**2
        x93 = d_3**2
        x94 = d_4**2
        x95 = a_0**2
        x96 = a_1**2
        x97 = 2*d_4
        x98 = 2*x13
        x99 = Py*x1
        x100 = 2*x16
        x101 = 2*x18
        x102 = 2*x14
        x103 = 2*x22
        x104 = 2*x24
        x105 = Px**2
        x106 = r_11**2
        x107 = x105*x106
        x108 = r_12**2
        x109 = x105*x108
        x110 = r_13**2
        x111 = x105*x110
        x112 = Py**2
        x113 = x112*x7
        x114 = x112*x9
        x115 = x11*x112
        x116 = Pz**2
        x117 = r_31**2
        x118 = x116*x117
        x119 = r_32**2
        x120 = x116*x119
        x121 = r_33**2*x116
        x122 = -Py*x98 + x100*x22 + x100*x73 + x101*x24 + x101*x41 + x102*x20 + x103*x73 + x104*x41 + x107 + x109 + x111 + x113 + x114 + x115 + x118 + x120 + x121 + x14*x99 - x18*x97 + x20*x99 - x24*x97 + x92 + x93 + x94 - x95 - x96
        x123 = -R_l_inv_52*a_0*d_2 - R_l_inv_55*a_0*x122 - a_1 + x91
        x124 = 2*x38
        x125 = x124*x75
        x126 = 2*x39
        x127 = x126*x74
        x128 = R_l_inv_54*a_0
        x129 = x1*x128
        x130 = 2*x51
        x131 = x130*x70
        x132 = x31*x71
        x133 = -x132
        x134 = x127 + x129 + x131 + x133
        x135 = 2*x4
        x136 = 2*x71
        x137 = x136*x50
        x138 = 2*R_l_inv_53*x53
        x139 = R_l_inv_57*x53*x58
        x140 = x135 - x137 + x138 + x139
        x141 = d_2*x43
        x142 = x122*x57
        x143 = -x91
        x144 = a_1 + x141 + x142 + x143 + x40
        x145 = -x85
        x146 = -x86
        x147 = -x44
        x148 = -x60
        x149 = x145 + x146 + x147 + x148 + x84 + x87 + x88
        x150 = x52 + x55 + x56
        x151 = 2*x83
        x152 = 2*r_23
        x153 = x152*x63
        x154 = x136*x89
        x155 = x130*x6
        x156 = x126*x42 + x151 + x153 + x154 + x155
        x157 = x124*x43
        x158 = 4*d_2
        x159 = x38*x57
        x160 = x158*x159
        x161 = -x157 - x160
        x162 = 2*x65
        x163 = 2*x66
        x164 = 2*x67
        x165 = 2*x68
        x166 = 2*x69
        x167 = x29*x71
        x168 = -x162 - x163 - x164 + x165 + x166 + x167
        x169 = 4*x43*x74
        x170 = 8*d_2
        x171 = x170*x80
        x172 = 4*x45
        x173 = 4*x48
        x174 = 4*x46
        x175 = d_3*x3
        x176 = x172 - x173 + x174 + x175*x71 + 4*x47 - 4*x49
        x177 = x157 + x160
        x178 = x162 + x163 + x164 - x165 - x166 - x167
        x179 = -x52
        x180 = -x56
        x181 = x179 + x180 + x44 + x54 + x60
        x182 = x76 - x81
        x183 = x182 + x62 + x64 + x72 + x78 + x90
        x184 = 4*d_3
        x185 = -x125 + x159*x184
        x186 = x140 + x185
        x187 = x123 + x182 + x62 + x64 + x72 + x78
        x188 = a_0*a_1
        x189 = 2*x188
        x190 = x95 + x96
        x191 = R_l_inv_62*x190
        x192 = R_l_inv_22*x189 + x191
        x193 = d_2*x192
        x194 = r_21**3*x112
        x195 = R_l_inv_25*x189 + R_l_inv_65*x190
        x196 = x122*x195
        x197 = R_l_inv_61*x190
        x198 = R_l_inv_21*x189 + x197
        x199 = d_3*x198
        x200 = -x199
        x201 = -r_21*x92
        x202 = -r_21*x93
        x203 = -r_21*x94
        x204 = R_l_inv_23*x189 + R_l_inv_63*x190
        x205 = -r_21*x204
        x206 = R_l_inv_60*x190
        x207 = x38*(R_l_inv_20*x189 + x206)
        x208 = -x207
        x209 = R_l_inv_66*x190
        x210 = R_l_inv_26*x189 + x209
        x211 = -x210*x70
        x212 = r_21*x107
        x213 = r_21*x114
        x214 = r_21*x115
        x215 = r_21*x118
        x216 = R_l_inv_67*x190
        x217 = R_l_inv_27*x189 + x216
        x218 = -x217*x77
        x219 = -r_21*x109
        x220 = -r_21*x111
        x221 = -r_21*x120
        x222 = -r_21*x121
        x223 = -x14*x98
        x224 = -x20*x98
        x225 = x102*x8
        x226 = x10*x102
        x227 = x102*x12
        x228 = d_4*x1
        x229 = x18*x228
        x230 = 2*x20
        x231 = x230*x8
        x232 = x10*x230
        x233 = x12*x230
        x234 = x228*x24
        x235 = 2*r_11
        x236 = r_12*x105
        x237 = r_22*x236
        x238 = x235*x237
        x239 = 2*r_13
        x240 = r_23*x105
        x241 = r_11*x239*x240
        x242 = r_31*x116
        x243 = r_32*x242
        x244 = 2*r_22
        x245 = x243*x244
        x246 = r_33*x242
        x247 = x152*x246
        x248 = x16*x22
        x249 = -x1*x248
        x250 = x18*x24
        x251 = -x1*x250
        x252 = x14*x20
        x253 = x1*x252
        x254 = x102*x23
        x255 = x102*x25
        x256 = x17*x230
        x257 = x19*x230
        x258 = x193 + x194 + x196 + x200 + x201 + x202 + x203 + x205 + x208 + x211 + x212 + x213 + x214 + x215 + x218 + x219 + x220 + x221 + x222 + x223 + x224 + x225 + x226 + x227 + x229 + x231 + x232 + x233 + x234 + x238 + x241 + x245 + x247 + x249 + x251 + x253 + x254 + x255 + x256 + x257
        x259 = x198*x74
        x260 = x192*x42
        x261 = -x260
        x262 = x195*x59
        x263 = -x262
        x264 = x195*x74
        x265 = x264*x79
        x266 = -x265
        x267 = x259 + x261 + x263 + x266
        x268 = x210*x83
        x269 = x135*x14
        x270 = x135*x20
        x271 = x16*x29
        x272 = x22*x29
        x273 = x268 - x269 - x270 + x271 + x272
        x274 = R_l_inv_24*x189 + R_l_inv_64*x190
        x275 = r_22*x274
        x276 = x217*x50
        x277 = d_2*r_22
        x278 = x210*x277
        x279 = d_4*x31
        x280 = 2*x6
        x281 = x14*x280
        x282 = x20*x280
        x283 = x18*x31
        x284 = x24*x31
        x285 = x275 - x276 - x278 - x279 - x281 - x282 + x283 + x284
        x286 = x273 + x285
        x287 = 4*x188
        x288 = R_l_inv_26*x287 + 2*x209
        x289 = x288*x50
        x290 = R_l_inv_20*x287 + 2*x206
        x291 = x290*x74
        x292 = r_22*x92
        x293 = -2*x292
        x294 = r_22*x93
        x295 = -2*x294
        x296 = r_22*x94
        x297 = -2*x296
        x298 = 2*x204
        x299 = -r_22*x298
        x300 = r_22**3*x112
        x301 = 2*x300
        x302 = r_22*x58
        x303 = -x217*x302
        x304 = r_22*x107
        x305 = -2*x304
        x306 = r_22*x111
        x307 = -2*x306
        x308 = r_22*x118
        x309 = -2*x308
        x310 = r_22*x121
        x311 = -2*x310
        x312 = r_22*x109
        x313 = 2*x312
        x314 = r_22*x113
        x315 = 2*x314
        x316 = r_22*x115
        x317 = 2*x316
        x318 = r_22*x120
        x319 = 2*x318
        x320 = 4*x16
        x321 = -x13*x320
        x322 = 4*x22
        x323 = -x13*x322
        x324 = x320*x8
        x325 = x10*x320
        x326 = x12*x320
        x327 = d_4*x3
        x328 = x18*x327
        x329 = x322*x8
        x330 = x10*x322
        x331 = x12*x322
        x332 = x24*x327
        x333 = 4*r_11
        x334 = r_21*x333
        x335 = x236*x334
        x336 = 4*r_12
        x337 = r_13*x240
        x338 = x336*x337
        x339 = 4*r_21
        x340 = x243*x339
        x341 = 4*r_23
        x342 = r_32*r_33*x116
        x343 = x341*x342
        x344 = -x252*x3
        x345 = -x250*x3
        x346 = x15*x322
        x347 = x21*x320
        x348 = x248*x3
        x349 = x25*x320
        x350 = x19*x322
        x351 = x289 - x291 + x293 + x295 + x297 + x299 + x301 + x303 + x305 + x307 + x309 + x311 + x313 + x315 + x317 + x319 + x321 + x323 + x324 + x325 + x326 + x328 + x329 + x330 + x331 + x332 + x335 + x338 + x340 + x343 + x344 + x345 + x346 + x347 + x348 + x349 + x350
        x352 = x38*(R_l_inv_21*x287 + 2*x197)
        x353 = x195*x38
        x354 = x184*x353 - x352
        x355 = R_l_inv_27*x287 + 2*x216
        x356 = x355*x70
        x357 = x1*x274
        x358 = d_4*x33
        x359 = x210*x31
        x360 = x320*x6
        x361 = x322*x6
        x362 = x18*x33
        x363 = x24*x33
        x364 = -x356 - x357 - x358 + x359 - x360 - x361 + x362 + x363
        x365 = -x268
        x366 = -x271
        x367 = -x272
        x368 = x260 + x262 + x269 + x270 + x365 + x366 + x367
        x369 = -d_2*x192
        x370 = -x122*x195
        x371 = x194 + x199 + x201 + x202 + x203 + x205 + x208 + x211 + x212 + x213 + x214 + x215 + x218 + x219 + x220 + x221 + x222 + x223 + x224 + x225 + x226 + x227 + x229 + x231 + x232 + x233 + x234 + x238 + x241 + x245 + x247 + x249 + x251 + x253 + x254 + x255 + x256 + x257 + x369 + x370
        x372 = x259 + x266
        x373 = x38*(R_l_inv_22*x287 + 2*x191)
        x374 = x158*x353
        x375 = -x290*x42 + x373 + x374
        x376 = x210*x29
        x377 = d_4*x175
        x378 = x175*x18
        x379 = x175*x24
        x380 = x184*x66
        x381 = x184*x67
        x382 = -x376 + x377 - x378 - x379 + x380 + x381
        x383 = x288*x89
        x384 = r_23*x92
        x385 = 2*x384
        x386 = r_23*x93
        x387 = 2*x386
        x388 = r_23*x298
        x389 = r_23*x94
        x390 = 2*x389
        x391 = r_23**3*x112
        x392 = 2*x391
        x393 = 4*d_4
        x394 = x393*x8
        x395 = x10*x393
        x396 = x12*x393
        x397 = x217*x280
        x398 = r_23*x107
        x399 = 2*x398
        x400 = r_23*x109
        x401 = 2*x400
        x402 = r_23*x118
        x403 = 2*x402
        x404 = r_23*x120
        x405 = 2*x404
        x406 = r_23*x111
        x407 = 2*x406
        x408 = r_23*x113
        x409 = 2*x408
        x410 = r_23*x114
        x411 = 2*x410
        x412 = r_23*x121
        x413 = 2*x412
        x414 = x14*x172
        x415 = x16*x327
        x416 = 4*x18
        x417 = x13*x416
        x418 = x172*x20
        x419 = x22*x327
        x420 = 4*x24
        x421 = x13*x420
        x422 = x416*x8
        x423 = x10*x416
        x424 = x12*x416
        x425 = x420*x8
        x426 = x10*x420
        x427 = x12*x420
        x428 = r_13*x105
        x429 = x334*x428
        x430 = r_13*x3
        x431 = x236*x430
        x432 = x246*x339
        x433 = x3*x342
        x434 = x174*x20
        x435 = x322*x66
        x436 = x15*x420
        x437 = x16*x3
        x438 = x24*x437
        x439 = x21*x416
        x440 = x22*x3
        x441 = x18*x440
        x442 = x19*x420
        x443 = -x383 - x385 - x387 - x388 + x390 + x392 - x394 - x395 - x396 - x397 - x399 - x401 - x403 - x405 + x407 + x409 + x411 + x413 - x414 - x415 - x417 - x418 - x419 - x421 + x422 + x423 + x424 + x425 + x426 + x427 + x429 + x431 + x432 + x433 - x434 - x435 + x436 + x438 + x439 + x441 + x442
        x444 = 8*x188
        x445 = x74*(R_l_inv_22*x444 + 4*x191)
        x446 = x170*x264
        x447 = 8*d_4
        x448 = 8*x62
        x449 = 8*d_3
        x450 = x175*x210 - x18*x448 - x24*x448 + x447*x62 + x449*x46 + x449*x47
        x451 = x290*x42 + x373 + x374
        x452 = x383 + x385 + x387 + x388 - x390 - x392 + x394 + x395 + x396 + x397 + x399 + x401 + x403 + x405 - x407 - x409 - x411 - x413 + x414 + x415 + x417 + x418 + x419 + x421 - x422 - x423 - x424 - x425 - x426 - x427 - x429 - x431 - x432 - x433 + x434 + x435 - x436 - x438 - x439 - x441 - x442
        x453 = -x259 + x265
        x454 = x261 + x263 + x453
        x455 = -x275 + x276 + x278 + x279 + x281 + x282 - x283 - x284
        x456 = x273 + x455
        x457 = -4*d_3*x195*x38 + x352
        x458 = x356 + x357 + x358 - x359 + x360 + x361 - x362 - x363
        x459 = R_l_inv_77*x190
        x460 = R_l_inv_37*x189 + x459
        x461 = x460*x50
        x462 = R_l_inv_34*x189 + R_l_inv_74*x190
        x463 = r_22*x462
        x464 = R_l_inv_76*x190
        x465 = R_l_inv_36*x189 + x464
        x466 = x277*x465
        x467 = x100*x8
        x468 = x10*x100
        x469 = x100*x12
        x470 = x162*x18
        x471 = x103*x8
        x472 = x10*x103
        x473 = x103*x12
        x474 = x162*x24
        x475 = r_11*x236
        x476 = x1*x475
        x477 = r_23*x236*x239
        x478 = x1*x243
        x479 = x152*x342
        x480 = x16*x98
        x481 = x22*x98
        x482 = x1*x14
        x483 = x22*x482
        x484 = x1*x20
        x485 = x16*x484
        x486 = x103*x17
        x487 = x100*x25
        x488 = x103*x19
        x489 = x230*x85
        x490 = x165*x24
        x491 = R_l_inv_71*x190
        x492 = R_l_inv_31*x189 + x491
        x493 = d_3*x492
        x494 = R_l_inv_70*x190
        x495 = x38*(R_l_inv_30*x189 + x494)
        x496 = R_l_inv_72*x190
        x497 = R_l_inv_32*x189 + x496
        x498 = x42*x497
        x499 = d_2*x497
        x500 = -x499
        x501 = x492*x74
        x502 = -x501
        x503 = R_l_inv_35*x189 + R_l_inv_75*x190
        x504 = x122*x503
        x505 = -x504
        x506 = x503*x59
        x507 = x503*x74
        x508 = x507*x79
        x509 = x493 + x495 + x498 + x500 + x502 + x505 + x506 + x508
        x510 = R_l_inv_33*x189 + R_l_inv_73*x190
        x511 = r_21*x510
        x512 = x465*x70
        x513 = x460*x77
        x514 = x465*x83
        x515 = -x514
        x516 = x6*x79
        x517 = -x516
        x518 = x58*x65
        x519 = x58*x68
        x520 = -x519
        x521 = x58*x69
        x522 = -x521
        x523 = x16*x280
        x524 = x22*x280
        x525 = x511 + x512 + x513 + x515 + x517 + x518 + x520 + x522 + x523 + x524
        x526 = x13*x79
        x527 = x79*x8
        x528 = x10*x79
        x529 = x12*x79
        x530 = x14*x29
        x531 = x135*x16
        x532 = x19*x79
        x533 = x20*x29
        x534 = x135*x22
        x535 = x25*x79
        x536 = -x526 + x527 + x528 + x529 + x530 + x531 + x532 + x533 + x534 + x535
        x537 = R_l_inv_30*x287 + 2*x494
        x538 = x537*x74
        x539 = R_l_inv_37*x287 + 2*x459
        x540 = x539*x70
        x541 = x1*x92
        x542 = -x541
        x543 = x1*x93
        x544 = x1*x462
        x545 = -x31*x465
        x546 = 4*x14
        x547 = 4*x20
        x548 = x1*x107 - x1*x109 - x1*x111 + x1*x114 + x1*x115 + x1*x118 - x1*x120 - x1*x121 - x1*x94 + x10*x546 + x10*x547 + x12*x546 + x12*x547 - x13*x546 - x13*x547 + x14*x440 + x15*x547 + x172*x18 + x172*x24 - x173*x24 + x19*x547 + 2*x194 + x20*x437 + x243*x3 + x246*x341 + x25*x546 + x3*x475 - x322*x87 + x333*x337 + x546*x8 + x547*x8
        x549 = x538 + x540 + x542 + x543 + x544 + x545 + x548
        x550 = R_l_inv_31*x287 + 2*x491
        x551 = x38*x503
        x552 = x184*x551
        x553 = x38*x550 - x552
        x554 = R_l_inv_36*x287 + 2*x464
        x555 = x50*x554
        x556 = 2*x510
        x557 = r_22*x556
        x558 = d_2*x172
        x559 = x302*x460
        x560 = x546*x6
        x561 = x547*x6
        x562 = x416*x77
        x563 = x420*x77
        x564 = -x555 + x557 - x558 + x559 - x560 - x561 + x562 + x563
        x565 = -x292
        x566 = -x296
        x567 = -x461
        x568 = -x466
        x569 = -x304
        x570 = -x306
        x571 = -x308
        x572 = -x310
        x573 = -x508
        x574 = -x480
        x575 = -x481
        x576 = -x489
        x577 = -x490
        x578 = x294 + x300 + x312 + x314 + x316 + x318 + x463 + x467 + x468 + x469 + x470 + x471 + x472 + x473 + x474 + x476 + x477 + x478 + x479 + x483 + x485 + x486 + x487 + x488 + x498 + x501 + x506 + x565 + x566 + x567 + x568 + x569 + x570 + x571 + x572 + x573 + x574 + x575 + x576 + x577
        x579 = -x495
        x580 = x493 + x500 + x505 + x579
        x581 = -x511 - x512 - x513 - x518 + x519 + x521 - x523 - x524
        x582 = x515 + x517 + x536 + x581
        x583 = x38*(R_l_inv_32*x287 + 2*x496)
        x584 = x158*x551
        x585 = x42*x537
        x586 = x29*x465
        x587 = x158*x62
        x588 = x585 + x586 + x587
        x589 = x554*x89
        x590 = r_23*x556
        x591 = x280*x460
        x592 = x320*x77
        x593 = x322*x77
        x594 = x14*x33
        x595 = x20*x33
        x596 = x589 + x590 + x591 - x592 - x593 + x594 + x595
        x597 = R_l_inv_32*x444 + 4*x496
        x598 = x170*x4 + x175*x465
        x599 = x583 + x584
        x600 = x596 + x599
        x601 = -d_3*x492 + x499 + x504
        x602 = x495 + x601
        x603 = x526 - x527 - x528 - x529 - x530 - x531 - x532 - x533 - x534 - x535
        x604 = x525 + x603
        x605 = -x538 + x540 + x542 + x543 + x544 + x545 + x548
        x606 = x555 - x557 + x558 - x559 + x560 + x561 - x562 - x563
        x607 = -x498 - x506
        x608 = x294 + x300 + x312 + x314 + x316 + x318 + x463 + x467 + x468 + x469 + x470 + x471 + x472 + x473 + x474 + x476 + x477 + x478 + x479 + x483 + x485 + x486 + x487 + x488 + x514 + x516 + x565 + x566 + x567 + x568 + x569 + x570 + x571 + x572 + x574 + x575 + x576 + x577
        x609 = x511 + x512 + x513 + x518 + x520 + x522 + x523 + x524 + x536 + x608
        x610 = x4*x58
        x611 = -2*d_2*d_4*r_23
        x612 = x58*x8
        x613 = x10*x58
        x614 = x12*x58
        x615 = x14*x31
        x616 = x17*x58
        x617 = x18*x280
        x618 = x20*x31
        x619 = x23*x58
        x620 = x24*x280
        x621 = -x610 + x611 + x612 + x613 + x614 + x615 + x616 + x617 + x618 + x619 + x620
        x622 = d_4*x135
        x623 = x66*x79
        x624 = x67*x79
        x625 = x135*x18
        x626 = x135*x24
        x627 = -x622 - x623 - x624 + x625 + x626
        x628 = x8*x97
        x629 = x10*x97
        x630 = x12*x97
        x631 = x101*x8
        x632 = x10*x101
        x633 = x101*x12
        x634 = x104*x8
        x635 = x10*x104
        x636 = x104*x12
        x637 = r_11*x1*x428
        x638 = x237*x239
        x639 = x1*x246
        x640 = x244*x342
        x641 = x14*x228
        x642 = x16*x162
        x643 = x18*x98
        x644 = x20*x228
        x645 = x162*x22
        x646 = x24*x98
        x647 = x24*x482
        x648 = x104*x17
        x649 = x18*x484
        x650 = x101*x23
        x651 = x104*x19
        x652 = x230*x46
        x653 = x163*x22
        x654 = -x384 + x386 - x389 - x391 + x398 + x400 + x402 + x404 - x406 - x408 - x410 - x412 + x628 + x629 + x630 - x631 - x632 - x633 - x634 - x635 - x636 - x637 - x638 - x639 - x640 + x641 + x642 + x643 + x644 + x645 + x646 - x647 - x648 - x649 - x650 - x651 + x652 + x653
        x655 = x621 + x627 + x654
        x656 = x50 + x77
        x657 = x622 + x623 + x624 - x625 - x626
        x658 = x610 + x611 + x612 + x613 + x614 + x615 + x616 + x617 + x618 + x619 + x620
        x659 = x654 + x657 + x658
        x660 = x14*x175
        x661 = x175*x20
        x662 = x320*x62
        x663 = x322*x62
        x664 = x541 - x543 + x548
        x665 = -x660 - x661 + x662 + x663 + x664
        x666 = 8*x13
        x667 = 8*x65
        x668 = 8*x22
        x669 = 8*x20
        x670 = 8*x16
        x671 = 8*x24
        x672 = 8*r_12
        x673 = r_11*r_21
        x674 = 8*x243
        x675 = 8*r_23
        x676 = r_21*x674 + x10*x668 + x10*x670 + x105*x672*x673 - x107*x3 + x109*x3 - x111*x3 + x113*x3 + x115*x3 - x118*x3 + x12*x668 + x12*x670 + x120*x3 - x121*x3 + x15*x668 - x16*x666 + x17*x668 + x18*x667 + x19*x668 + x21*x670 - x22*x666 + x24*x667 + x25*x670 + x3*x92 - x3*x93 - x3*x94 + 4*x300 + x337*x672 + x342*x675 + x668*x8 - x669*x85 + x670*x8 - x671*x68
        x677 = x660 + x661 - x662 - x663 + x664
        x678 = x384 - x386 + x389 + x391 - x398 - x400 - x402 - x404 + x406 + x408 + x410 + x412 - x628 - x629 - x630 + x631 + x632 + x633 + x634 + x635 + x636 + x637 + x638 + x639 + x640 - x641 - x642 - x643 - x644 - x645 - x646 + x647 + x648 + x649 + x650 + x651 - x652 - x653
        x679 = x621 + x657 + x678
        x680 = x50 - x77
        x681 = x627 + x658 + x678
        x682 = -x239
        x683 = r_12*x79
        x684 = -x683
        x685 = d_2*x239
        x686 = 2*Px
        x687 = 2*r_12
        x688 = -d_4*x239 + r_11*x99 + x106*x686 + x108*x686 + x110*x686 + x20*x235 + x22*x687 + x239*x24 + x239*x41 + x687*x73
        x689 = -x685 + x688
        x690 = d_3*x333
        x691 = d_2*x333
        x692 = d_2*x672
        x693 = x685 + x688
        x694 = x128*x687
        x695 = d_4*r_11
        x696 = r_13*x37
        x697 = r_13*x20
        x698 = r_11*x41
        x699 = r_11*x24
        x700 = x695 + x696 + x697 - x698 - x699
        x701 = x130*x700
        x702 = r_12*x58
        x703 = x702*x71
        x704 = d_4*r_12
        x705 = r_13*x73
        x706 = r_13*x22
        x707 = r_12*x41
        x708 = r_12*x24
        x709 = x704 + x705 + x706 - x707 - x708
        x710 = r_11*x58
        x711 = r_11*x79 + x136*x709 + x235*x63 + x51*x710
        x712 = -x694 + x701 + x703 + x711
        x713 = r_11*x73
        x714 = 2*x713
        x715 = r_11*x22
        x716 = 2*x715
        x717 = r_12*x99
        x718 = r_12*x20
        x719 = 2*x718
        x720 = d_3*x239
        x721 = x71*x720
        x722 = -x714 - x716 + x717 + x719 - x721
        x723 = x128*x333
        x724 = 4*x51*x709
        x725 = r_12*x184
        x726 = 4*x71
        x727 = d_2*x336
        x728 = x336*x63 + x51*x727 - x700*x726 + x725
        x729 = x714 + x716 - x717 - x719 + x721
        x730 = 4*x704
        x731 = Py*x430
        x732 = 4*x706
        x733 = 4*x707
        x734 = 4*x708
        x735 = x690*x71
        x736 = r_13*x184
        x737 = r_12*x37
        x738 = x713 + x715 - x718 - x737
        x739 = 4*r_13
        x740 = d_2*x739
        x741 = x51*x740 + x63*x739 - x726*x738 + x736
        x742 = 8*x695
        x743 = 8*x698
        x744 = 8*x696
        x745 = 8*x699
        x746 = 8*x697
        x747 = d_3*x672
        x748 = x694 - x701 - x703 + x711
        x749 = x210*x720
        x750 = Py*x336
        x751 = x62*x750
        x752 = x20*x725
        x753 = Py*x175
        x754 = r_11*x753
        x755 = x22*x690
        x756 = r_11**3
        x757 = 2*x105
        x758 = Px*x106
        x759 = 4*x37
        x760 = Px*x759
        x761 = Px*x547
        x762 = Py*x333
        x763 = d_4*x24
        x764 = r_21*x112
        x765 = r_12*x3
        x766 = Py*x3
        x767 = x24*x41
        x768 = x20*x37
        x769 = 4*x41
        x770 = -r_11*x298 + r_23*x739*x764 + x108*x760 + x108*x761 + x109*x235 + x110*x760 + x110*x761 + x111*x235 + x113*x235 - x114*x235 - x115*x235 + x118*x235 - x120*x235 - x121*x235 + x13*x762 - x217*x710 - x235*x92 - x235*x93 - x235*x94 + x243*x336 + x246*x739 - x288*x709 + x322*x737 + x333*x763 - x333*x767 + x333*x768 - x393*x696 - x393*x697 + x420*x696 + x547*x758 + x697*x769 - x715*x766 + x718*x766 + x756*x757 + x758*x759 + x764*x765
        x771 = x749 - x751 - x752 + x754 + x755 + x770
        x772 = x355*x700
        x773 = x274*x687
        x774 = d_4*x691
        x775 = r_12*x210
        x776 = x58*x775
        x777 = x158*x696
        x778 = x158*x697
        x779 = x6*x762
        x780 = x24*x691
        x781 = -x772 + x773 - x774 - x776 - x777 - x778 + x779 + x780
        x782 = x709*(R_l_inv_27*x444 + 4*x216)
        x783 = x274*x333
        x784 = d_4*x692
        x785 = x170*x705
        x786 = x170*x706
        x787 = Py*x672
        x788 = R_l_inv_26*x444 + 4*x209
        x789 = r_12**3
        x790 = 4*x105
        x791 = 8*x73
        x792 = Px*x791
        x793 = Px*x668
        x794 = r_22*x112
        x795 = x673*x794
        x796 = r_13*x675
        x797 = 8*r_13
        x798 = 8*x37
        x799 = x672*x73
        x800 = 8*x41
        x801 = -d_2*x217*x336 + r_11*x674 + x107*x336 + x108*x792 + x108*x793 + x110*x792 + x110*x793 + x111*x336 - x113*x336 + x114*x336 - x115*x336 - x118*x336 + x120*x336 - x121*x336 + x13*x787 - x204*x336 + x22*x799 - x336*x92 - x336*x93 - x336*x94 + x342*x797 - x447*x705 - x447*x706 + x668*x758 + x669*x713 + x671*x705 + x672*x763 - x672*x767 - x672*x768 + x700*x788 + x706*x800 + x715*x798 + x758*x791 + x789*x790 + x794*x796 + 8*x795
        x802 = -x749 + x751 + x752 - x754 - x755 + x770
        x803 = x210*x690
        x804 = d_4*x747
        x805 = x41*x747
        x806 = x24*x747
        x807 = Py*x4
        x808 = x797*x807
        x809 = x449*x706
        x810 = r_13**3
        x811 = Px*x800
        x812 = Px*x671
        x813 = x112*x673
        x814 = 8*r_11
        x815 = r_23*x794
        x816 = x22*x672
        x817 = -8*Px*d_4*x106 - 8*Px*d_4*x108 - 8*Px*d_4*x110 - 8*Py*Pz*r_13*r_21*r_31 - 8*Py*Pz*r_13*r_22*r_32 - 8*Py*d_4*r_11*r_21 - 8*Py*d_4*r_12*r_22 - 8*Py*d_4*r_13*r_23 - 8*Pz*d_4*r_11*r_31 - 8*Pz*d_4*r_12*r_32 - 8*Pz*d_4*r_13*r_33 - 4*d_2*r_13*x217 - 4*r_13*x112*x7 - 4*r_13*x112*x9 - 4*r_13*x116*x117 - 4*r_13*x116*x119 - 4*r_13*x204 - 4*r_13*x92 - 4*r_13*x93 + x107*x739 + x108*x811 + x108*x812 + x109*x739 + x110*x811 + x110*x812 + x115*x739 + x121*x739 + x20*x743 + x24*x799 + x246*x814 + x342*x672 + x37*x745 + x41*x816 + x671*x758 + x672*x815 + x675*x813 + x738*x788 + x739*x94 + x758*x800 + x767*x797 + x790*x810
        x818 = x772 - x773 + x774 + x776 + x777 + x778 - x779 - x780
        x819 = -x539*x700
        x820 = -x687*x92
        x821 = -x687*x94
        x822 = x687*x93
        x823 = x462*x687
        x824 = x757*x789
        x825 = -x465*x702
        x826 = -x113*x687
        x827 = -x115*x687
        x828 = -x118*x687
        x829 = -x121*x687
        x830 = x465*x720
        x831 = x107*x687
        x832 = x111*x687
        x833 = x114*x687
        x834 = x120*x687
        x835 = d_2*x736
        x836 = -d_4*x731
        x837 = -x393*x706
        x838 = x758*x766
        x839 = Px*x766
        x840 = x108*x839
        x841 = x110*x839
        x842 = x322*x758
        x843 = Px*x322
        x844 = x108*x843
        x845 = x110*x843
        x846 = x13*x750
        x847 = x24*x730
        x848 = x3*x813
        x849 = x243*x333
        x850 = r_23*x112
        x851 = x430*x850
        x852 = x342*x739
        x853 = -x547*x737
        x854 = -x24*x733
        x855 = x333*x37
        x856 = x22*x855
        x857 = r_11*x20*x766
        x858 = Py*x22*x765
        x859 = x24*x731
        x860 = x41*x732
        x861 = x819 + x820 + x821 + x822 + x823 + x824 + x825 + x826 + x827 + x828 + x829 + x830 + x831 + x832 + x833 + x834 + x835 + x836 + x837 + x838 + x840 + x841 + x842 + x844 + x845 + x846 + x847 + x848 + x849 + x851 + x852 + x853 + x854 + x856 + x857 + x858 + x859 + x860
        x862 = x554*x709
        x863 = r_11*x556
        x864 = d_2*x730
        x865 = x460*x710
        x866 = Py*r_13*x33
        x867 = x158*x706
        x868 = x6*x750
        x869 = x158*x708
        x870 = -x862 - x863 - x864 - x865 - x866 - x867 + x868 + x869
        x871 = d_4*x736
        x872 = -4*Px*d_3*x106 - 4*Px*d_3*x108 - 4*Px*d_3*x110 - 4*Py*d_3*r_11*r_21 - 4*Py*d_3*r_12*r_22 - 4*Py*d_3*r_13*r_23 - 4*Pz*d_3*r_11*r_31 - 4*Pz*d_3*r_12*r_32 - 4*Pz*d_3*r_13*r_33 + x871
        x873 = R_l_inv_36*x444 + 4*x464
        x874 = x700*x873
        x875 = x336*x510
        x876 = d_2*x742
        x877 = d_2*x460
        x878 = x336*x877
        x879 = x170*x696
        x880 = x170*x697
        x881 = Py*x6
        x882 = x814*x881
        x883 = d_2*r_11
        x884 = x671*x883
        x885 = x333*x92
        x886 = x333*x93
        x887 = Px*x798
        x888 = Px*x669
        x889 = Py*r_11*x666 + r_22*x672*x764 + x108*x887 + x108*x888 + x109*x333 + x110*x887 + x110*x888 + x111*x333 + x113*x333 - x114*x333 - x115*x333 + x118*x333 - x120*x333 - x121*x333 + x20*x799 + x24*x742 - x24*x743 + x24*x744 + x243*x672 + x246*x797 - x333*x94 + x37*x816 + x41*x746 - x447*x696 - x447*x697 - x668*x713 + x669*x758 + x756*x790 + x758*x798 + x764*x796 + x768*x814
        x890 = x333*x462 - x465*x691 + x709*(R_l_inv_37*x444 + 4*x459) - x885 + x886 + x889
        x891 = Px*x184
        x892 = r_12*x753 + x106*x891 + x108*x891 + x110*x891 + x20*x690 + x22*x725 + x24*x736 + x41*x736 + x62*x762 - x871
        x893 = x819 + x820 + x821 + x822 + x823 + x824 + x825 + x826 + x827 + x828 + x829 - x830 + x831 + x832 + x833 + x834 - x835 + x836 + x837 + x838 + x840 + x841 + x842 + x844 + x845 + x846 + x847 + x848 + x849 + x851 + x852 + x853 + x854 + x856 + x857 + x858 + x859 + x860
        x894 = x510*x739
        x895 = x739*x877
        x896 = x37*x692
        x897 = x20*x692
        x898 = x449*x883 + x465*x690
        x899 = x862 + x863 + x864 + x865 + x866 + x867 - x868 - x869
        x900 = Px*x158
        x901 = x106*x900
        x902 = x108*x900
        x903 = x110*x900
        x904 = x37*x691
        x905 = Py*r_12
        x906 = x33*x905
        x907 = x739*x881
        x908 = x20*x691
        x909 = x22*x727
        x910 = x24*x740
        x911 = d_2*x725
        x912 = x184*x704
        x913 = x184*x707
        x914 = -x913
        x915 = x184*x708
        x916 = -x915
        x917 = r_13*x753
        x918 = x184*x706
        x919 = x911 + x912 + x914 + x916 + x917 + x918
        x920 = x239*x93
        x921 = x239*x92
        x922 = x239*x94
        x923 = x757*x810
        x924 = Px*x393
        x925 = x106*x924
        x926 = x108*x924
        x927 = x110*x924
        x928 = x113*x239
        x929 = x114*x239
        x930 = x118*x239
        x931 = x120*x239
        x932 = x107*x239
        x933 = x109*x239
        x934 = x115*x239
        x935 = x121*x239
        x936 = d_4*x333
        x937 = x37*x936
        x938 = x704*x766
        x939 = Py*x13*x739
        x940 = x20*x936
        x941 = x22*x730
        x942 = x739*x763
        x943 = x758*x769
        x944 = Px*x769
        x945 = x108*x944
        x946 = x110*x944
        x947 = x420*x758
        x948 = Px*x420
        x949 = x108*x948
        x950 = x110*x948
        x951 = x334*x850
        x952 = x246*x333
        x953 = x765*x850
        x954 = x336*x342
        x955 = x547*x696
        x956 = x706*x766
        x957 = x24*x855
        x958 = x20*x333*x41
        x959 = x708*x766
        x960 = x22*x733
        x961 = x739*x767
        x962 = -x920 + x921 + x922 + x923 - x925 - x926 - x927 - x928 - x929 - x930 - x931 + x932 + x933 + x934 + x935 - x937 - x938 - x939 - x940 - x941 - x942 + x943 + x945 + x946 + x947 + x949 + x950 + x951 + x952 + x953 + x954 - x955 - x956 + x957 + x958 + x959 + x960 + x961
        x963 = -4*d_2*d_4*r_13 + x901 + x902 + x903 + x904 + x906 + x907 + x908 + x909 + x910
        x964 = x62*x787
        x965 = x20*x747
        x966 = x885 - x886 + x889
        x967 = 16*d_4
        x968 = 16*x24
        x969 = 16*x20
        x970 = 16*x73
        x971 = Px*x970
        x972 = 16*x22
        x973 = Px*x972
        x974 = 16*r_13
        x975 = x962 + x963
        x976 = -x29
        x977 = -x40
        x978 = x150 + x977
        x979 = a_1 + x141 + x142 + x143
        x980 = x145 + x146 + x84 + x87 + x88
        x981 = -2*R_l_inv_50*a_0*x42 + x151 + x153 + x154 + x155
        x982 = x979 + x980
        x983 = x194 + x201 + x202 + x203 + x205 + x207 + x211 + x212 + x213 + x214 + x215 + x218 + x219 + x220 + x221 + x222 + x223 + x224 + x225 + x226 + x227 + x229 + x231 + x232 + x233 + x234 + x238 + x241 + x245 + x247 + x249 + x251 + x253 + x254 + x255 + x256 + x257
        x984 = x260 + x262 + x983
        x985 = x199 + x369 + x370
        x986 = x289 + x291 + x293 + x295 + x297 + x299 + x301 + x303 + x305 + x307 + x309 + x311 + x313 + x315 + x317 + x319 + x321 + x323 + x324 + x325 + x326 + x328 + x329 + x330 + x331 + x332 + x335 + x338 + x340 + x343 + x344 + x345 + x346 + x347 + x348 + x349 + x350
        x987 = x193 + x196 + x200
        x988 = x269 + x270 + x365 + x366 + x367 + x983
        x989 = x376 - x377 + x378 + x379 - x380 - x381
        x990 = -x38*x550 + x552
        x991 = x502 + x508
        x992 = x294 + x300 + x312 + x314 + x316 + x318 + x463 + x467 + x468 + x469 + x470 + x471 + x472 + x473 + x474 + x476 + x477 + x478 + x479 + x483 + x485 + x486 + x487 + x488 + x565 + x566 + x567 + x568 + x569 + x570 + x571 + x572 + x574 + x575 + x576 + x577 + x607 + x991
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = x0
        A[0, 2] = x0
        A[0, 3] = x2
        A[0, 4] = -x3
        A[0, 5] = x1
        A[0, 6] = r_23
        A[0, 8] = r_23
        A[1, 0] = x28
        A[1, 1] = x29
        A[1, 2] = x30
        A[1, 3] = x32
        A[1, 4] = -x33
        A[1, 5] = x31
        A[1, 6] = x35
        A[1, 7] = x29
        A[1, 8] = x36
        A[2, 0] = -x123 - x61 - x82 - x90
        A[2, 1] = 4*R_l_inv_55*a_0*d_3*x38 - x125 - x134 - x140
        A[2, 2] = x144 + x149 + x150 + x82
        A[2, 3] = x156 + x161 + x168
        A[2, 4] = -x169 - x171 + x176
        A[2, 5] = x156 + x177 + x178
        A[2, 6] = x144 + x181 + x183
        A[2, 7] = x127 - x129 - x131 + x132 + x186
        A[2, 8] = -x149 - x179 - x180 - x187 - x40 - x54
        A[3, 0] = x258 + x267 + x286
        A[3, 1] = x351 + x354 + x364
        A[3, 2] = -x285 - x368 - x371 - x372
        A[3, 3] = -x375 - x382 - x443
        A[3, 4] = -x445 - x446 + x450
        A[3, 5] = x382 + x451 + x452
        A[3, 6] = -x371 - x454 - x456
        A[3, 7] = -x351 - x457 - x458
        A[3, 8] = x258 + x368 + x453 + x455
        A[4, 0] = x292 - x294 + x296 - x300 + x304 + x306 + x308 + x310 - x312 - x314 - x316 - x318 + x461 - x463 + x466 - x467 - x468 - x469 - x470 - x471 - x472 - x473 - x474 - x476 - x477 - x478 - x479 + x480 + x481 - x483 - x485 - x486 - x487 - x488 + x489 + x490 + x509 + x525 + x536
        A[4, 1] = x549 + x553 + x564
        A[4, 2] = x578 + x580 + x582
        A[4, 3] = x583 + x584 - x588 - x596
        A[4, 4] = 8*d_2*x503*x74 + x597*x74 - x598
        A[4, 5] = -x585 + x586 + x587 - x600
        A[4, 6] = -x578 - x602 - x604
        A[4, 7] = x553 + x605 + x606
        A[4, 8] = x493 + x495 + x500 + x501 + x505 + x573 + x607 + x609
        A[5, 0] = -x655
        A[5, 1] = -x184*x656
        A[5, 2] = -x659
        A[5, 3] = x665
        A[5, 4] = x676
        A[5, 5] = -x677
        A[5, 6] = -x679
        A[5, 7] = x184*x680
        A[5, 8] = -x681
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x682
        B[0, 2] = x682
        B[0, 3] = -x333
        B[0, 4] = -x672
        B[0, 5] = x333
        B[0, 6] = x239
        B[0, 8] = x239
        B[1, 0] = x684 + x689
        B[1, 1] = x690
        B[1, 2] = x683 + x689
        B[1, 3] = -x691
        B[1, 4] = -x692
        B[1, 5] = x691
        B[1, 6] = x684 + x693
        B[1, 7] = x690
        B[1, 8] = x683 + x693
        B[2, 0] = -x712 - x722
        B[2, 1] = 4*R_l_inv_56*a_0*d_2*r_11 - x723 - x724 - x728
        B[2, 2] = x712 + x729
        B[2, 3] = -x730 - x731 - x732 + x733 + x734 + x735 + x741
        B[2, 4] = x71*x747 + x742 - x743 + x744 - x745 + x746
        B[2, 5] = x730 + x731 + x732 - x733 - x734 - x735 + x741
        B[2, 6] = x722 + x748
        B[2, 7] = x691*x71 - x723 - x724 + x728
        B[2, 8] = -x729 - x748
        B[3, 0] = x771 + x781
        B[3, 1] = x210*x691 + x24*x692 + x6*x787 - x782 - x783 - x784 - x785 - x786 + x801
        B[3, 2] = -x781 - x802
        B[3, 3] = x803 - x804 + x805 + x806 - x808 - x809 - x817
        B[3, 4] = x449*(r_13*x99 + x20*x239 + 2*x695 - 2*x698 - 2*x699 + x775)
        B[3, 5] = -x803 + x804 - x805 - x806 + x808 + x809 - x817
        B[3, 6] = -x771 - x818
        B[3, 7] = 8*Py*d_2*r_12*r_23 + 8*Pz*d_2*r_12*r_33 + 4*d_2*r_11*x210 - x782 - x783 - x784 - x785 - x786 - x801
        B[3, 8] = x802 + x818
        B[4, 0] = -x861 - x870 - x872
        B[4, 1] = -x874 + x875 - x876 + x878 - x879 - x880 + x882 + x884 + x890
        B[4, 2] = x870 + x892 + x893
        B[4, 3] = 8*Py*d_2*r_11*r_22 + 8*Pz*d_2*r_11*r_32 + x738*x873 - x894 - x895 - x896 - x897 - x898
        B[4, 4] = x747*(-x465 - x58)
        B[4, 5] = x668*x883 + x738*x873 + x791*x883 - x894 - x895 - x896 - x897 + x898
        B[4, 6] = -x872 - x893 - x899
        B[4, 7] = x874 - x875 + x876 - x878 + x879 + x880 - x882 - x884 + x890
        B[4, 8] = x861 + x892 + x899
        B[5, 0] = d_4*x740 - x901 - x902 - x903 - x904 - x906 - x907 - x908 - x909 - x910 + x919 + x962
        B[5, 1] = x449*(-x700 - x883)
        B[5, 2] = -x919 - x920 + x921 + x922 + x923 - x925 - x926 - x927 - x928 - x929 - x930 - x931 + x932 + x933 + x934 + x935 - x937 - x938 - x939 - x940 - x941 - x942 + x943 + x945 + x946 + x947 + x949 + x950 + x951 + x952 + x953 + x954 - x955 - x956 + x957 + x958 + x959 + x960 + x961 - x963
        B[5, 3] = x449*x715 + x807*x814 - x964 - x965 + x966
        B[5, 4] = 16*r_11*x243 + r_12*x22*x970 + 8*x105*x789 + x107*x672 + x108*x971 + x108*x973 + x110*x971 + x110*x973 + x111*x672 - x113*x672 + x114*x672 - x115*x672 - x118*x672 + x120*x672 - x121*x672 + 16*x13*x905 + x342*x974 + 16*x37*x715 + 16*x41*x706 + x672*x92 - x672*x93 - x672*x94 + x704*x968 - x705*x967 + x705*x968 - x706*x967 - x707*x968 + x713*x969 - x737*x969 + x758*x970 + x758*x972 + 16*x795 + x815*x974
        B[5, 5] = 8*Py*d_3*r_11*r_22 + 8*Pz*d_3*r_11*r_32 - x964 - x965 - x966
        B[5, 6] = x911 - x912 - x914 - x916 - x917 - x918 - x975
        B[5, 7] = x449*(x695 + x696 + x697 - x698 - x699 - x883)
        B[5, 8] = -x911 + x912 - x913 - x915 + x917 + x918 - x975
        C = np.zeros(shape=(6, 9))
        C[0, 0] = r_23
        C[0, 2] = r_23
        C[0, 3] = x1
        C[0, 4] = x3
        C[0, 5] = x2
        C[0, 6] = x0
        C[0, 8] = x0
        C[1, 0] = -x28
        C[1, 1] = x976
        C[1, 2] = -x30
        C[1, 3] = x31
        C[1, 4] = x33
        C[1, 5] = x32
        C[1, 6] = -x35
        C[1, 7] = x976
        C[1, 8] = -x36
        C[2, 0] = x147 + x148 + x183 + x978 + x979
        C[2, 1] = -x127 + x129 + x131 + x133 + x186
        C[2, 2] = -x187 - x44 - x60 - x978 - x980
        C[2, 3] = -x168 - x177 - x981
        C[2, 4] = -x169 - x171 - x176
        C[2, 5] = -x161 - x178 - x981
        C[2, 6] = x182 + x61 - x62 - x64 - x72 - x78 + x982
        C[2, 7] = x134 - x135 + x137 - x138 - x139 + x185
        C[2, 8] = x181 + x82 + x977 + x982
        C[3, 0] = -x286 - x453 - x984 - x985
        C[3, 1] = -x364 - x457 - x986
        C[3, 2] = x285 + x454 + x987 + x988
        C[3, 3] = -x375 - x452 - x989
        C[3, 4] = -x445 - x446 - x450
        C[3, 5] = x443 + x451 + x989
        C[3, 6] = x372 + x456 + x984 + x987
        C[3, 7] = x354 + x458 + x986
        C[3, 8] = -x267 - x455 - x985 - x988
        C[4, 0] = x509 + x581 + x603 + x608
        C[4, 1] = -x564 - x605 - x990
        C[4, 2] = -x582 - x602 - x992
        C[4, 3] = -x585 + x586 + x587 + x600
        C[4, 4] = x170*x507 + x597*x74 + x598
        C[4, 5] = -x588 + x589 + x590 + x591 - x592 - x593 + x594 + x595 - x599
        C[4, 6] = x580 + x604 + x992
        C[4, 7] = -x549 - x606 - x990
        C[4, 8] = -x498 - x506 - x579 - x601 - x609 - x991
        C[5, 0] = x655
        C[5, 1] = x184*x656
        C[5, 2] = x659
        C[5, 3] = -x665
        C[5, 4] = -x676
        C[5, 5] = x677
        C[5, 6] = x679
        C[5, 7] = -x184*x680
        C[5, 8] = x681
        local_solutions = compute_solution_from_tanhalf_LME(A, B, C)
        for local_solutions_i in local_solutions:
            solution_i: IkSolution = make_ik_solution()
            solution_i[0] = local_solutions_i
            appended_idx = append_solution_to_queue(solution_i)
            add_input_index_to(2, appended_idx)
    # Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_0_processor()
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
            th_0 = this_solution[0]
            condition_0: bool = (abs((-Px*math.sin(th_0) + Py*math.cos(th_0) + d_4*(r_13*math.sin(th_0) - r_23*math.cos(th_0)))/d_3) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = safe_acos((Px*x1 - Py*x0 - d_4*(r_13*x1 - r_23*x0))/d_3)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[5] = x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs((-Px*math.sin(th_0) + Py*math.cos(th_0) + d_4*(r_13*math.sin(th_0) - r_23*math.cos(th_0)))/d_3) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = safe_acos((Px*x1 - Py*x0 - d_4*(r_13*x1 - r_23*x0))/d_3)
                # End of temp variables
                this_solution[5] = -x2
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_3_processor()
    # Finish code for explicit solution node 2
    
    # Code for non-branch dispatcher node 4
    # Actually, there is no code
    
    # Code for explicit solution node 5, solved variable is th_2
    def ExplicitSolutionNode_node_5_solve_th_2_processor():
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
            th_3 = this_solution[5]
            condition_0: bool = (2*abs(a_0*d_2) >= zero_tolerance) or (abs(2*a_0*a_1 - 2*a_0*d_3*math.sin(th_3)) >= zero_tolerance) or (abs(-a_0**2 - a_1**2 + 2*a_1*d_3*math.sin(th_3) - d_2**2 - d_3**2 + d_4**2 + 2*d_4*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*a_0
                x1 = d_3*math.sin(th_3)
                x2 = a_1*x0 - x0*x1
                x3 = math.atan2(-d_2*x0, x2)
                x4 = a_0**2
                x5 = d_2**2
                x6 = -a_1**2 + 2*a_1*x1 - d_3**2 + d_4**2 + 2*d_4*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2 - x4 - x5
                x7 = safe_sqrt(x2**2 + 4*x4*x5 - x6**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x3 + math.atan2(x7, x6)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                
            condition_1: bool = (2*abs(a_0*d_2) >= zero_tolerance) or (abs(2*a_0*a_1 - 2*a_0*d_3*math.sin(th_3)) >= zero_tolerance) or (abs(-a_0**2 - a_1**2 + 2*a_1*d_3*math.sin(th_3) - d_2**2 - d_3**2 + d_4**2 + 2*d_4*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*a_0
                x1 = d_3*math.sin(th_3)
                x2 = a_1*x0 - x0*x1
                x3 = math.atan2(-d_2*x0, x2)
                x4 = a_0**2
                x5 = d_2**2
                x6 = -a_1**2 + 2*a_1*x1 - d_3**2 + d_4**2 + 2*d_4*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2 - x4 - x5
                x7 = safe_sqrt(x2**2 + 4*x4*x5 - x6**2)
                # End of temp variables
                this_solution[4] = x3 + math.atan2(-x7, x6)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_2_processor()
    # Finish code for explicit solution node 4
    
    # Code for solved_variable dispatcher node 6
    def SolvedVariableDispatcherNode_node_6_processor():
        this_node_input_index: List[int] = node_input_index[6]
        this_input_valid: bool = node_input_validity[6]
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
                add_input_index_to(19, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(7, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_6_processor()
    # Finish code for solved_variable dispatcher node 6
    
    # Code for explicit solution node 19, solved variable is th_1th_2th_4_soa
    def ExplicitSolutionNode_node_19_solve_th_1th_2th_4_soa_processor():
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
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_33) >= zero_tolerance) or (abs(r_13*math.cos(th_0) + r_23*math.sin(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[3] = math.atan2(r_13*math.cos(th_0) + r_23*math.sin(th_0), r_33)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(20, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_1th_2th_4_soa_processor()
    # Finish code for explicit solution node 19
    
    # Code for non-branch dispatcher node 20
    # Actually, there is no code
    
    # Code for explicit solution node 21, solved variable is th_5
    def ExplicitSolutionNode_node_21_solve_th_5_processor():
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
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                # End of temp variables
                this_solution[7] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(22, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_5_processor()
    # Finish code for explicit solution node 20
    
    # Code for solved_variable dispatcher node 22
    def SolvedVariableDispatcherNode_node_22_processor():
        this_node_input_index: List[int] = node_input_index[22]
        this_input_valid: bool = node_input_validity[22]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_2 = this_solution[4]
            degenerate_valid_0 = (abs(th_2 - math.pi + 1.40006111531961) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(32, node_input_i_idx_in_queue)
            
            th_2 = this_solution[4]
            degenerate_valid_1 = (abs(th_2 - 2*math.pi + 1.40006111531961) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(35, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(23, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_22_processor()
    # Finish code for solved_variable dispatcher node 22
    
    # Code for explicit solution node 35, solved variable is th_4
    def ExplicitSolutionNode_node_35_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[35]
        this_input_valid: bool = node_input_validity[35]
        if not this_input_valid:
            return
        
        # The solution of non-root node 35
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_5 = this_solution[7]
            condition_0: bool = (abs(0.985460115744348*a_0 + d_2) >= zero_tolerance) or (abs(d_4 + inv_Pz) >= zero_tolerance) or (abs(inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)
                x1 = d_4 + inv_Pz
                x2 = math.atan2(x0, x1)
                x3 = 0.985460115744348*a_0 + d_2
                x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[6] = x2 + math.atan2(x4, x3)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(36, appended_idx)
                
            condition_1: bool = (abs(0.985460115744348*a_0 + d_2) >= zero_tolerance) or (abs(d_4 + inv_Pz) >= zero_tolerance) or (abs(inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)
                x1 = d_4 + inv_Pz
                x2 = math.atan2(x0, x1)
                x3 = 0.985460115744348*a_0 + d_2
                x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                this_solution[6] = x2 + math.atan2(-x4, x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(36, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_35_solve_th_4_processor()
    # Finish code for explicit solution node 35
    
    # Code for non-branch dispatcher node 36
    # Actually, there is no code
    
    # Code for explicit solution node 37, solved variable is th_1
    def ExplicitSolutionNode_node_37_solve_th_1_processor():
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
            th_5 = this_solution[7]
            condition_0: bool = (abs(0.985460115744348*a_1 - 0.169906916507646*d_2) >= zero_tolerance) or (abs(a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2) >= zero_tolerance) or (abs(Pz + d_3*r_31*math.sin(th_5) + d_3*r_32*math.cos(th_5) - d_4*r_33) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2
                x1 = math.atan2(x0, -0.985460115744348*a_1 + 0.169906916507646*d_2)
                x2 = -Pz - d_3*r_31*math.sin(th_5) - d_3*r_32*math.cos(th_5) + d_4*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.971131639722864*(-a_1 + 0.172413793103448*d_2)**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x1 + math.atan2(x3, x2)
                appended_idx = append_solution_to_queue(solution_0)
                
            condition_1: bool = (abs(0.985460115744348*a_1 - 0.169906916507646*d_2) >= zero_tolerance) or (abs(a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2) >= zero_tolerance) or (abs(Pz + d_3*r_31*math.sin(th_5) + d_3*r_32*math.cos(th_5) - d_4*r_33) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2
                x1 = math.atan2(x0, -0.985460115744348*a_1 + 0.169906916507646*d_2)
                x2 = -Pz - d_3*r_31*math.sin(th_5) - d_3*r_32*math.cos(th_5) + d_4*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.971131639722864*(-a_1 + 0.172413793103448*d_2)**2)
                # End of temp variables
                this_solution[1] = x1 + math.atan2(-x3, x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_37_solve_th_1_processor()
    # Finish code for explicit solution node 36
    
    # Code for explicit solution node 32, solved variable is th_1
    def ExplicitSolutionNode_node_32_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[32]
        this_input_valid: bool = node_input_validity[32]
        if not this_input_valid:
            return
        
        # The solution of non-root node 32
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_5 = this_solution[7]
            condition_0: bool = (abs(0.985460115744348*a_1 - 0.169906916507646*d_2) >= zero_tolerance) or (abs(-a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2) >= zero_tolerance) or (abs(Pz + d_3*r_31*math.sin(th_5) + d_3*r_32*math.cos(th_5) - d_4*r_33) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = -a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2
                x1 = math.atan2(x0, -0.985460115744348*a_1 + 0.169906916507646*d_2)
                x2 = Pz + d_3*r_31*math.sin(th_5) + d_3*r_32*math.cos(th_5) - d_4*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.971131639722864*(-a_1 + 0.172413793103448*d_2)**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x1 + math.atan2(x3, x2)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(33, appended_idx)
                
            condition_1: bool = (abs(0.985460115744348*a_1 - 0.169906916507646*d_2) >= zero_tolerance) or (abs(-a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2) >= zero_tolerance) or (abs(Pz + d_3*r_31*math.sin(th_5) + d_3*r_32*math.cos(th_5) - d_4*r_33) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = -a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2
                x1 = math.atan2(x0, -0.985460115744348*a_1 + 0.169906916507646*d_2)
                x2 = Pz + d_3*r_31*math.sin(th_5) + d_3*r_32*math.cos(th_5) - d_4*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.971131639722864*(-a_1 + 0.172413793103448*d_2)**2)
                # End of temp variables
                this_solution[1] = x1 + math.atan2(-x3, x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(33, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_32_solve_th_1_processor()
    # Finish code for explicit solution node 32
    
    # Code for non-branch dispatcher node 33
    # Actually, there is no code
    
    # Code for explicit solution node 34, solved variable is th_4
    def ExplicitSolutionNode_node_34_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[33]
        this_input_valid: bool = node_input_validity[33]
        if not this_input_valid:
            return
        
        # The solution of non-root node 34
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance) or (abs(-r_13*((-0.985460115744348*math.sin(th_1) - 0.169906916507646*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) - r_23*((-0.985460115744348*math.sin(th_1) - 0.169906916507646*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(0.169906916507646*math.sin(th_1) - 0.985460115744348*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance) or (abs(-r_13*(-0.169906916507646*math.sin(th_1) + 0.985460115744348*math.cos(th_1))*math.cos(th_0) - r_23*(-0.169906916507646*math.sin(th_1) + 0.985460115744348*math.cos(th_1))*math.sin(th_0) + r_33*(0.985460115744348*math.sin(th_1) + 0.169906916507646*math.cos(th_1))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_1)
                x3 = 0.169906916507646*x1 - 0.985460115744348*x2
                x4 = math.sin(th_0)
                x5 = math.sin(th_3)
                x6 = math.cos(th_0)
                x7 = 0.985460115744348*x1 + 0.169906916507646*x2
                x8 = -x0*x7
                x9 = -x3
                # End of temp variables
                this_solution[6] = math.atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*x3, r_13*x6*x9 + r_23*x4*x9 - r_33*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_34_solve_th_4_processor()
    # Finish code for explicit solution node 33
    
    # Code for explicit solution node 23, solved variable is th_1
    def ExplicitSolutionNode_node_23_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[23]
        this_input_valid: bool = node_input_validity[23]
        if not this_input_valid:
            return
        
        # The solution of non-root node 23
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1th_2th_4_soa = this_solution[3]
            th_2 = this_solution[4]
            condition_0: bool = (abs(a_1*math.sin(th_2) + d_2*math.cos(th_2)) >= 1.0e-6) or (abs(a_0 + a_1*math.cos(th_2) - d_2*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = Pz - d_4*r_33
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = a_0 + a_1*x1 - d_2*x2
                x4 = -a_1*x2 - d_2*x1
                x5 = Px*math.cos(th_0) + Py*math.sin(th_0) - d_4*math.sin(th_1th_2th_4_soa)
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x3 + x4*x5, x0*x4 + x3*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(24, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_1_processor()
    # Finish code for explicit solution node 23
    
    # Code for non-branch dispatcher node 24
    # Actually, there is no code
    
    # Code for explicit solution node 25, solved variable is th_4
    def ExplicitSolutionNode_node_25_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[24]
        this_input_valid: bool = node_input_validity[24]
        if not this_input_valid:
            return
        
        # The solution of non-root node 25
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
    ExplicitSolutionNode_node_25_solve_th_4_processor()
    # Finish code for explicit solution node 24
    
    # Code for explicit solution node 14, solved variable is th_5
    def ExplicitSolutionNode_node_14_solve_th_5_processor():
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
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                # End of temp variables
                this_solution[7] = math.atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_5_processor()
    # Finish code for explicit solution node 14
    
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
            th_2 = this_solution[4]
            degenerate_valid_0 = (abs(th_2 - math.pi + 1.40006111531961) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(26, node_input_i_idx_in_queue)
            
            th_2 = this_solution[4]
            degenerate_valid_1 = (abs(th_2 - 2*math.pi + 1.40006111531961) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(29, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(16, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_15_processor()
    # Finish code for solved_variable dispatcher node 15
    
    # Code for explicit solution node 29, solved variable is th_4
    def ExplicitSolutionNode_node_29_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[29]
        this_input_valid: bool = node_input_validity[29]
        if not this_input_valid:
            return
        
        # The solution of non-root node 29
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_5 = this_solution[7]
            condition_0: bool = (abs(0.985460115744348*a_0 + d_2) >= zero_tolerance) or (abs(d_4 + inv_Pz) >= zero_tolerance) or (abs(inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)
                x1 = d_4 + inv_Pz
                x2 = math.atan2(x0, x1)
                x3 = 0.985460115744348*a_0 + d_2
                x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[6] = x2 + math.atan2(x4, x3)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(30, appended_idx)
                
            condition_1: bool = (abs(0.985460115744348*a_0 + d_2) >= zero_tolerance) or (abs(d_4 + inv_Pz) >= zero_tolerance) or (abs(inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)
                x1 = d_4 + inv_Pz
                x2 = math.atan2(x0, x1)
                x3 = 0.985460115744348*a_0 + d_2
                x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                this_solution[6] = x2 + math.atan2(-x4, x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(30, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_4_processor()
    # Finish code for explicit solution node 29
    
    # Code for non-branch dispatcher node 30
    # Actually, there is no code
    
    # Code for explicit solution node 31, solved variable is th_1
    def ExplicitSolutionNode_node_31_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[30]
        this_input_valid: bool = node_input_validity[30]
        if not this_input_valid:
            return
        
        # The solution of non-root node 31
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_5 = this_solution[7]
            condition_0: bool = (abs(0.985460115744348*a_1 - 0.169906916507646*d_2) >= zero_tolerance) or (abs(a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2) >= zero_tolerance) or (abs(Pz + d_3*r_31*math.sin(th_5) + d_3*r_32*math.cos(th_5) - d_4*r_33) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2
                x1 = math.atan2(x0, -0.985460115744348*a_1 + 0.169906916507646*d_2)
                x2 = -Pz - d_3*r_31*math.sin(th_5) - d_3*r_32*math.cos(th_5) + d_4*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.971131639722864*(-a_1 + 0.172413793103448*d_2)**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x1 + math.atan2(x3, x2)
                appended_idx = append_solution_to_queue(solution_0)
                
            condition_1: bool = (abs(0.985460115744348*a_1 - 0.169906916507646*d_2) >= zero_tolerance) or (abs(a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2) >= zero_tolerance) or (abs(Pz + d_3*r_31*math.sin(th_5) + d_3*r_32*math.cos(th_5) - d_4*r_33) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2
                x1 = math.atan2(x0, -0.985460115744348*a_1 + 0.169906916507646*d_2)
                x2 = -Pz - d_3*r_31*math.sin(th_5) - d_3*r_32*math.cos(th_5) + d_4*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.971131639722864*(-a_1 + 0.172413793103448*d_2)**2)
                # End of temp variables
                this_solution[1] = x1 + math.atan2(-x3, x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_1_processor()
    # Finish code for explicit solution node 30
    
    # Code for explicit solution node 26, solved variable is th_1
    def ExplicitSolutionNode_node_26_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[26]
        this_input_valid: bool = node_input_validity[26]
        if not this_input_valid:
            return
        
        # The solution of non-root node 26
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_5 = this_solution[7]
            condition_0: bool = (abs(0.985460115744348*a_1 - 0.169906916507646*d_2) >= zero_tolerance) or (abs(-a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2) >= zero_tolerance) or (abs(Pz + d_3*r_31*math.sin(th_5) + d_3*r_32*math.cos(th_5) - d_4*r_33) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = -a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2
                x1 = math.atan2(x0, -0.985460115744348*a_1 + 0.169906916507646*d_2)
                x2 = Pz + d_3*r_31*math.sin(th_5) + d_3*r_32*math.cos(th_5) - d_4*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.971131639722864*(-a_1 + 0.172413793103448*d_2)**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x1 + math.atan2(x3, x2)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(27, appended_idx)
                
            condition_1: bool = (abs(0.985460115744348*a_1 - 0.169906916507646*d_2) >= zero_tolerance) or (abs(-a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2) >= zero_tolerance) or (abs(Pz + d_3*r_31*math.sin(th_5) + d_3*r_32*math.cos(th_5) - d_4*r_33) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = -a_0 + 0.169906916507646*a_1 + 0.985460115744348*d_2
                x1 = math.atan2(x0, -0.985460115744348*a_1 + 0.169906916507646*d_2)
                x2 = Pz + d_3*r_31*math.sin(th_5) + d_3*r_32*math.cos(th_5) - d_4*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.971131639722864*(-a_1 + 0.172413793103448*d_2)**2)
                # End of temp variables
                this_solution[1] = x1 + math.atan2(-x3, x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(27, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_26_solve_th_1_processor()
    # Finish code for explicit solution node 26
    
    # Code for non-branch dispatcher node 27
    # Actually, there is no code
    
    # Code for explicit solution node 28, solved variable is th_4
    def ExplicitSolutionNode_node_28_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[27]
        this_input_valid: bool = node_input_validity[27]
        if not this_input_valid:
            return
        
        # The solution of non-root node 28
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance) or (abs(-r_13*((-0.985460115744348*math.sin(th_1) - 0.169906916507646*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) - r_23*((-0.985460115744348*math.sin(th_1) - 0.169906916507646*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(0.169906916507646*math.sin(th_1) - 0.985460115744348*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance) or (abs(-r_13*(-0.169906916507646*math.sin(th_1) + 0.985460115744348*math.cos(th_1))*math.cos(th_0) - r_23*(-0.169906916507646*math.sin(th_1) + 0.985460115744348*math.cos(th_1))*math.sin(th_0) + r_33*(0.985460115744348*math.sin(th_1) + 0.169906916507646*math.cos(th_1))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_1)
                x3 = 0.169906916507646*x1 - 0.985460115744348*x2
                x4 = math.sin(th_0)
                x5 = math.sin(th_3)
                x6 = math.cos(th_0)
                x7 = 0.985460115744348*x1 + 0.169906916507646*x2
                x8 = -x0*x7
                x9 = -x3
                # End of temp variables
                this_solution[6] = math.atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*x3, r_13*x6*x9 + r_23*x4*x9 - r_33*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_28_solve_th_4_processor()
    # Finish code for explicit solution node 27
    
    # Code for explicit solution node 16, solved variable is th_1
    def ExplicitSolutionNode_node_16_solve_th_1_processor():
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
            th_0 = this_solution[0]
            th_2 = this_solution[4]
            condition_0: bool = (abs(a_1*math.sin(th_2) + d_2*math.cos(th_2)) >= 1.0e-6) or (abs(a_0 + a_1*math.cos(th_2) - d_2*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = Pz - d_4*r_33
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = a_0 + a_1*x1 - d_2*x2
                x4 = -a_1*x2 - d_2*x1
                x5 = math.cos(th_0)
                x6 = math.sin(th_0)
                x7 = Px*x5 + Py*x6 - d_4*r_13*x5 - d_4*r_23*x6
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x3 + x4*x7, x0*x4 + x3*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(17, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_1_processor()
    # Finish code for explicit solution node 16
    
    # Code for non-branch dispatcher node 17
    # Actually, there is no code
    
    # Code for explicit solution node 18, solved variable is th_4
    def ExplicitSolutionNode_node_18_solve_th_4_processor():
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
    ExplicitSolutionNode_node_18_solve_th_4_processor()
    # Finish code for explicit solution node 17
    
    # Code for explicit solution node 7, solved variable is th_4
    def ExplicitSolutionNode_node_7_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[7]
        this_input_valid: bool = node_input_validity[7]
        if not this_input_valid:
            return
        
        # The solution of non-root node 7
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_3 = this_solution[5]
            condition_0: bool = (abs((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3)) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = safe_asin((-r_13*math.sin(th_0) + r_23*math.cos(th_0))/math.sin(th_3))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[6] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
            condition_1: bool = (abs((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = safe_asin((-r_13*math.sin(th_0) + r_23*math.cos(th_0))/math.sin(th_3))
                # End of temp variables
                this_solution[6] = math.pi - x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_4_processor()
    # Finish code for explicit solution node 7
    
    # Code for equation all-zero dispatcher node 8
    def EquationAllZeroDispatcherNode_node_8_processor():
        this_node_input_index: List[int] = node_input_index[8]
        this_input_valid: bool = node_input_validity[8]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            checked_result: bool = (abs(Pz) <= 1.0e-6) and (abs(Px*math.cos(th_0) + Py*math.sin(th_0)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor()
    # Finish code for equation all-zero dispatcher node 8
    
    # Code for explicit solution node 9, solved variable is th_1th_2_soa
    def ExplicitSolutionNode_node_9_solve_th_1th_2_soa_processor():
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
            th_2 = this_solution[4]
            th_3 = this_solution[5]
            th_4 = this_solution[6]
            condition_0: bool = (abs(Pz) >= 1.0e-6) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = -a_0*math.cos(th_2) - a_1 + d_3*math.sin(th_3) + d_4*math.sin(th_4)*math.cos(th_3)
                x1 = -Px*math.cos(th_0) - Py*math.sin(th_0)
                x2 = a_0*math.sin(th_2) - d_2 + d_4*math.cos(th_4)
                # End of temp variables
                this_solution[2] = math.atan2(Pz*x0 - x1*x2, Pz*x2 + x0*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_1th_2_soa_processor()
    # Finish code for explicit solution node 9
    
    # Code for non-branch dispatcher node 10
    # Actually, there is no code
    
    # Code for explicit solution node 11, solved variable is th_1
    def ExplicitSolutionNode_node_11_solve_th_1_processor():
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
            th_1th_2_soa = this_solution[2]
            th_2 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[1] = th_1th_2_soa - th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(12, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1_processor()
    # Finish code for explicit solution node 10
    
    # Code for non-branch dispatcher node 12
    # Actually, there is no code
    
    # Code for explicit solution node 13, solved variable is th_5
    def ExplicitSolutionNode_node_13_solve_th_5_processor():
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
            th_0 = this_solution[0]
            th_1th_2_soa = this_solution[2]
            th_3 = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*(math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)*math.cos(th_1th_2_soa)) - r_21*(math.sin(th_0)*math.sin(th_3)*math.cos(th_1th_2_soa) + math.cos(th_0)*math.cos(th_3)) + r_31*math.sin(th_1th_2_soa)*math.sin(th_3)) >= zero_tolerance) or (abs(r_12*(math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)*math.cos(th_1th_2_soa)) - r_22*(math.sin(th_0)*math.sin(th_3)*math.cos(th_1th_2_soa) + math.cos(th_0)*math.cos(th_3)) + r_32*math.sin(th_1th_2_soa)*math.sin(th_3)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_3)
                x1 = x0*math.sin(th_1th_2_soa)
                x2 = math.sin(th_0)
                x3 = math.cos(th_3)
                x4 = math.cos(th_0)
                x5 = x0*math.cos(th_1th_2_soa)
                x6 = x2*x3 - x4*x5
                x7 = x2*x5 + x3*x4
                # End of temp variables
                this_solution[7] = math.atan2(-r_11*x6 + r_21*x7 - r_31*x1, -r_12*x6 + r_22*x7 - r_32*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_5_processor()
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


def rokae_SR3_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = rokae_SR3_ik_target_original_to_raw(T_ee)
    ik_output_raw = rokae_SR3_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ee_pose_i = rokae_SR3_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_rokae_SR3():
    theta_in = np.random.random(size=(6, ))
    ee_pose = rokae_SR3_fk(theta_in)
    ik_output = rokae_SR3_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = rokae_SR3_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_rokae_SR3()
