import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 34
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_2: float = 0.403113
a_3: float = 0.05
d_3: float = 0.4
d_4: float = -0.136
d_5: float = 0.1035
pre_transform_s0: float = 0.328

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -1.44644
th_2_offset_original2raw: float = 1.69515
th_3_offset_original2raw: float = 0.0
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 0.0


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def rokae_SR5_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = r_11
    ee_transformed[0, 1] = r_12
    ee_transformed[0, 2] = r_13
    ee_transformed[0, 3] = Px
    ee_transformed[1, 0] = r_21
    ee_transformed[1, 1] = r_22
    ee_transformed[1, 2] = r_23
    ee_transformed[1, 3] = Py
    ee_transformed[2, 0] = r_31
    ee_transformed[2, 1] = r_32
    ee_transformed[2, 2] = r_33
    ee_transformed[2, 3] = Pz - pre_transform_s0
    return ee_transformed


def rokae_SR5_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = r_11
    ee_transformed[0, 1] = r_12
    ee_transformed[0, 2] = r_13
    ee_transformed[0, 3] = Px
    ee_transformed[1, 0] = r_21
    ee_transformed[1, 1] = r_22
    ee_transformed[1, 2] = r_23
    ee_transformed[1, 3] = Py
    ee_transformed[2, 0] = r_31
    ee_transformed[2, 1] = r_32
    ee_transformed[2, 2] = r_33
    ee_transformed[2, 3] = Pz + pre_transform_s0
    return ee_transformed


def rokae_SR5_fk(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_5)
    x1 = math.sin(th_0)
    x2 = math.cos(th_3)
    x3 = math.sin(th_3)
    x4 = math.cos(th_0)
    x5 = math.sin(th_1)
    x6 = math.sin(th_2)
    x7 = x5*x6
    x8 = math.cos(th_1)
    x9 = math.cos(th_2)
    x10 = x8*x9
    x11 = x10*x4 + x4*x7
    x12 = x1*x2 - x11*x3
    x13 = math.cos(th_5)
    x14 = math.sin(th_4)
    x15 = x6*x8
    x16 = x15*x4
    x17 = x5*x9
    x18 = x17*x4
    x19 = x16 - x18
    x20 = math.cos(th_4)
    x21 = x1*x3 + x11*x2
    x22 = -x14*x19 + x20*x21
    x23 = -x14*x21 - x19*x20
    x24 = a_2*x8
    x25 = x1*x10 + x1*x7
    x26 = -x2*x4 - x25*x3
    x27 = x1*x15
    x28 = x1*x17
    x29 = x27 - x28
    x30 = x2*x25 - x3*x4
    x31 = -x14*x29 + x20*x30
    x32 = -x14*x30 - x20*x29
    x33 = x15 - x17
    x34 = x3*x33
    x35 = -x10 - x7
    x36 = x2*x33
    x37 = -x14*x35 + x20*x36
    x38 = -x14*x36 - x20*x35
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -x0*x12 + x13*x22
    ee_pose[0, 1] = -x0*x22 - x12*x13
    ee_pose[0, 2] = x23
    ee_pose[0, 3] = a_3*x11 - d_3*(-x16 + x18) + d_4*x12 + d_5*x23 + x24*x4
    ee_pose[1, 0] = -x0*x26 + x13*x31
    ee_pose[1, 1] = -x0*x31 - x13*x26
    ee_pose[1, 2] = x32
    ee_pose[1, 3] = a_3*x25 - d_3*(-x27 + x28) + d_4*x26 + d_5*x32 + x1*x24
    ee_pose[2, 0] = x0*x34 + x13*x37
    ee_pose[2, 1] = -x0*x37 + x13*x34
    ee_pose[2, 2] = x38
    ee_pose[2, 3] = -a_2*x5 + a_3*x33 - d_3*(x10 + x7) - d_4*x34 + d_5*x38 + pre_transform_s0
    return ee_pose


def rokae_SR5_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.cos(th_0)
    x2 = math.sin(th_2)
    x3 = math.cos(th_1)
    x4 = x2*x3
    x5 = x1*x4
    x6 = math.sin(th_1)
    x7 = math.cos(th_2)
    x8 = x6*x7
    x9 = x1*x8
    x10 = x5 - x9
    x11 = math.cos(th_3)
    x12 = math.sin(th_3)
    x13 = x2*x6
    x14 = x3*x7
    x15 = x1*x13 + x1*x14
    x16 = x0*x11 - x12*x15
    x17 = math.cos(th_4)
    x18 = math.sin(th_4)
    x19 = -x10*x17 - x18*(x0*x12 + x11*x15)
    x20 = x0*x4
    x21 = x0*x8
    x22 = x20 - x21
    x23 = x0*x13 + x0*x14
    x24 = -x1*x11 - x12*x23
    x25 = -x17*x22 - x18*(-x1*x12 + x11*x23)
    x26 = -x13 - x14
    x27 = x4 - x8
    x28 = x12*x27
    x29 = -x11*x18*x27 - x17*x26
    x30 = -a_2*x6 + pre_transform_s0
    x31 = a_3*x27 - d_3*(x13 + x14) + x30
    x32 = a_2*x3
    x33 = a_3*x23 - d_3*(-x20 + x21) + x0*x32
    x34 = -d_4*x28 + x31
    x35 = d_4*x24 + x33
    x36 = d_5*x29 + x34
    x37 = d_5*x25 + x35
    x38 = a_3*x15 - d_3*(-x5 + x9) + x1*x32
    x39 = d_4*x16 + x38
    x40 = d_5*x19 + x39
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = -x0
    jacobian_output[0, 2] = x0
    jacobian_output[0, 3] = x10
    jacobian_output[0, 4] = x16
    jacobian_output[0, 5] = x19
    jacobian_output[1, 1] = x1
    jacobian_output[1, 2] = -x1
    jacobian_output[1, 3] = x22
    jacobian_output[1, 4] = x24
    jacobian_output[1, 5] = x25
    jacobian_output[2, 0] = 1
    jacobian_output[2, 3] = x26
    jacobian_output[2, 4] = -x28
    jacobian_output[2, 5] = x29
    jacobian_output[3, 1] = -pre_transform_s0*x1
    jacobian_output[3, 2] = x1*x30
    jacobian_output[3, 3] = -x22*x31 + x26*x33
    jacobian_output[3, 4] = -x24*x34 - x28*x35
    jacobian_output[3, 5] = -x25*x36 + x29*x37
    jacobian_output[4, 1] = -pre_transform_s0*x0
    jacobian_output[4, 2] = x0*x30
    jacobian_output[4, 3] = x10*x31 - x26*x38
    jacobian_output[4, 4] = x16*x34 + x28*x39
    jacobian_output[4, 5] = x19*x36 - x29*x40
    jacobian_output[5, 2] = -x0**2*x32 - x1**2*x32
    jacobian_output[5, 3] = -x10*x33 + x22*x38
    jacobian_output[5, 4] = -x16*x35 + x24*x39
    jacobian_output[5, 5] = -x19*x37 + x25*x40
    return jacobian_output


def rokae_SR5_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.cos(th_0)
    x2 = math.sin(th_2)
    x3 = math.cos(th_1)
    x4 = x2*x3
    x5 = math.sin(th_1)
    x6 = math.cos(th_2)
    x7 = x5*x6
    x8 = x1*x4 - x1*x7
    x9 = math.cos(th_3)
    x10 = math.sin(th_3)
    x11 = x2*x5
    x12 = x3*x6
    x13 = x1*x11 + x1*x12
    x14 = math.cos(th_4)
    x15 = math.sin(th_4)
    x16 = x0*x4 - x0*x7
    x17 = x0*x11 + x0*x12
    x18 = -x11 - x12
    x19 = x4 - x7
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 1] = -x0
    jacobian_output[0, 2] = x0
    jacobian_output[0, 3] = x8
    jacobian_output[0, 4] = x0*x9 - x10*x13
    jacobian_output[0, 5] = -x14*x8 - x15*(x0*x10 + x13*x9)
    jacobian_output[1, 1] = x1
    jacobian_output[1, 2] = -x1
    jacobian_output[1, 3] = x16
    jacobian_output[1, 4] = -x1*x9 - x10*x17
    jacobian_output[1, 5] = -x14*x16 - x15*(-x1*x10 + x17*x9)
    jacobian_output[2, 0] = 1
    jacobian_output[2, 3] = x18
    jacobian_output[2, 4] = -x10*x19
    jacobian_output[2, 5] = -x14*x18 - x15*x19*x9
    return jacobian_output


def rokae_SR5_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x0 = math.cos(th_0)
    x1 = p_on_ee_z*x0
    x2 = math.sin(th_1)
    x3 = -a_2*x2 + pre_transform_s0
    x4 = math.sin(th_2)
    x5 = x2*x4
    x6 = math.cos(th_1)
    x7 = math.cos(th_2)
    x8 = x6*x7
    x9 = -x5 - x8
    x10 = math.sin(th_0)
    x11 = x4*x6
    x12 = x10*x11
    x13 = x2*x7
    x14 = x10*x13
    x15 = x12 - x14
    x16 = x11 - x13
    x17 = a_3*x16 - d_3*(x5 + x8) + x3
    x18 = a_2*x6
    x19 = x10*x5 + x10*x8
    x20 = a_3*x19 - d_3*(-x12 + x14) + x10*x18
    x21 = math.sin(th_3)
    x22 = x16*x21
    x23 = math.cos(th_3)
    x24 = -x0*x23 - x19*x21
    x25 = -d_4*x22 + x17
    x26 = d_4*x24 + x20
    x27 = math.cos(th_4)
    x28 = math.sin(th_4)
    x29 = -x16*x23*x28 - x27*x9
    x30 = -x15*x27 - x28*(-x0*x21 + x19*x23)
    x31 = d_5*x29 + x25
    x32 = d_5*x30 + x26
    x33 = p_on_ee_z*x10
    x34 = x0*x11
    x35 = x0*x13
    x36 = x34 - x35
    x37 = x0*x5 + x0*x8
    x38 = a_3*x37 - d_3*(-x34 + x35) + x0*x18
    x39 = x10*x23 - x21*x37
    x40 = d_4*x39 + x38
    x41 = -x27*x36 - x28*(x10*x21 + x23*x37)
    x42 = d_5*x41 + x40
    x43 = p_on_ee_x*x0
    x44 = p_on_ee_y*x10
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -p_on_ee_y
    jacobian_output[0, 1] = -pre_transform_s0*x0 + x1
    jacobian_output[0, 2] = x0*x3 - x1
    jacobian_output[0, 3] = -p_on_ee_y*x9 + p_on_ee_z*x15 - x15*x17 + x20*x9
    jacobian_output[0, 4] = p_on_ee_y*x22 + p_on_ee_z*x24 - x22*x26 - x24*x25
    jacobian_output[0, 5] = -p_on_ee_y*x29 + p_on_ee_z*x30 + x29*x32 - x30*x31
    jacobian_output[1, 0] = p_on_ee_x
    jacobian_output[1, 1] = -pre_transform_s0*x10 + x33
    jacobian_output[1, 2] = x10*x3 - x33
    jacobian_output[1, 3] = p_on_ee_x*x9 - p_on_ee_z*x36 + x17*x36 - x38*x9
    jacobian_output[1, 4] = -p_on_ee_x*x22 - p_on_ee_z*x39 + x22*x40 + x25*x39
    jacobian_output[1, 5] = p_on_ee_x*x29 - p_on_ee_z*x41 - x29*x42 + x31*x41
    jacobian_output[2, 1] = -x43 - x44
    jacobian_output[2, 2] = -x0**2*x18 - x10**2*x18 + x43 + x44
    jacobian_output[2, 3] = -p_on_ee_x*x15 + p_on_ee_y*x36 + x15*x38 - x20*x36
    jacobian_output[2, 4] = -p_on_ee_x*x24 + p_on_ee_y*x39 + x24*x40 - x26*x39
    jacobian_output[2, 5] = -p_on_ee_x*x30 + p_on_ee_y*x41 + x30*x42 - x32*x41
    return jacobian_output


def rokae_SR5_ik_solve_raw(T_ee: np.ndarray):
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
        for tmp_sol_idx in range(6):
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
    for i in range(34):
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
        R_l[0, 3] = -a_2
        R_l[0, 7] = -a_3
        R_l[1, 2] = -a_2
        R_l[1, 6] = -a_3
        R_l[2, 4] = -a_2
        R_l[3, 6] = -1
        R_l[4, 7] = 1
        R_l[5, 5] = 2*a_2*a_3
        R_l[6, 1] = a_2
        R_l[7, 0] = a_2
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
        x4 = d_3*r_23
        x5 = -x4
        x6 = d_4*r_22
        x7 = -x6
        x8 = x5 + x7
        x9 = r_21**2
        x10 = Py*x9
        x11 = r_22**2
        x12 = Py*x11
        x13 = r_23**2
        x14 = Py*x13
        x15 = d_5*r_23
        x16 = Px*r_11
        x17 = r_21*x16
        x18 = Px*r_12
        x19 = r_22*x18
        x20 = Px*r_13
        x21 = r_23*x20
        x22 = Pz*r_31
        x23 = r_21*x22
        x24 = Pz*r_32
        x25 = r_22*x24
        x26 = Pz*r_33
        x27 = r_23*x26
        x28 = x10 + x12 + x14 - x15 + x17 + x19 + x21 + x23 + x25 + x27
        x29 = d_4*x1
        x30 = x28 + x6
        x31 = d_3*x1
        x32 = -x31
        x33 = d_3*x3
        x34 = x4 + x7
        x35 = Py*r_22
        x36 = x18 + x24 + x35
        x37 = R_l_inv_51*a_2
        x38 = x36*x37
        x39 = R_l_inv_52*a_2
        x40 = d_3*x39
        x41 = a_2*r_22
        x42 = R_l_inv_54*x41
        x43 = d_3**2
        x44 = d_4**2
        x45 = d_5**2
        x46 = a_2**2
        x47 = a_3**2
        x48 = 2*d_5
        x49 = 2*x15
        x50 = Py*x1
        x51 = 2*x18
        x52 = Py*r_23
        x53 = 2*x20
        x54 = 2*x16
        x55 = 2*x24
        x56 = 2*x26
        x57 = Px**2
        x58 = r_11**2
        x59 = x57*x58
        x60 = r_12**2
        x61 = x57*x60
        x62 = r_13**2
        x63 = x57*x62
        x64 = Py**2
        x65 = x64*x9
        x66 = x11*x64
        x67 = x13*x64
        x68 = Pz**2
        x69 = r_31**2*x68
        x70 = r_32**2*x68
        x71 = r_33**2*x68
        x72 = -Py*x49 + x16*x50 - x20*x48 + x22*x50 + x22*x54 + x24*x51 - x26*x48 + x26*x53 + x35*x51 + x35*x55 + x43 + x44 + x45 - x46 - x47 + x52*x53 + x52*x56 + x59 + x61 + x63 + x65 + x66 + x67 + x69 + x70 + x71
        x73 = R_l_inv_55*a_2
        x74 = x72*x73
        x75 = -d_4*x37
        x76 = d_5*r_21
        x77 = r_23*x16
        x78 = r_23*x22
        x79 = r_21*x20
        x80 = r_21*x26
        x81 = x76 + x77 + x78 - x79 - x80
        x82 = R_l_inv_57*a_2
        x83 = x81*x82
        x84 = -x83
        x85 = R_l_inv_56*d_3*x41
        x86 = -x85
        x87 = 2*d_4
        x88 = x36*x73
        x89 = x87*x88
        x90 = -x89
        x91 = a_3 + x38 + x40 + x42 + x74 + x75 + x84 + x86 + x90
        x92 = d_4*r_21
        x93 = -x92
        x94 = Py*r_21
        x95 = x16 + x22 + x94
        x96 = R_l_inv_50*a_2
        x97 = x95*x96
        x98 = -x97
        x99 = R_l_inv_53*a_2
        x100 = r_21*x99
        x101 = -x100
        x102 = d_5*r_22
        x103 = r_23*x18
        x104 = r_23*x24
        x105 = r_22*x20
        x106 = r_22*x26
        x107 = x102 + x103 + x104 - x105 - x106
        x108 = R_l_inv_56*a_2
        x109 = x107*x108
        x110 = -x109
        x111 = d_3*r_21
        x112 = x111*x82
        x113 = -x112
        x114 = x101 + x110 + x113 + x93 + x98
        x115 = r_21*x18
        x116 = r_21*x24
        x117 = r_22*x16
        x118 = -x117
        x119 = r_22*x22
        x120 = -x119
        x121 = d_4*r_23
        x122 = x108*x121
        x123 = -d_5 + x20 + x26 + x52
        x124 = x123*x39
        x125 = 2*d_3
        x126 = x123*x125
        x127 = x126*x73
        x128 = -x124 - x127
        x129 = x115 + x116 + x118 + x120 + x122 + x128
        x130 = 2*x96
        x131 = x130*x36
        x132 = 2*x95
        x133 = -x132*x37
        x134 = 4*d_4
        x135 = x73*x95
        x136 = x134*x135
        x137 = -x131 + x133 + x136
        x138 = R_l_inv_54*a_2
        x139 = x1*x138
        x140 = 2*x82
        x141 = x107*x140
        x142 = x108*x31
        x143 = -x139 - x141 + x142
        x144 = 2*x6
        x145 = 2*R_l_inv_53*x41
        x146 = 2*x108
        x147 = x146*x81
        x148 = R_l_inv_57*x125*x41
        x149 = -x144 - x145 + x147 - x148
        x150 = x100 + x109 + x112 + x92 + x97
        x151 = a_3 - x38 + x40 + x74 + x75 + x89
        x152 = -x42 + x83 + x85
        x153 = x151 + x152
        x154 = x132*x39
        x155 = x123*x130
        x156 = 4*d_3
        x157 = x135*x156
        x158 = -x154 + x155 - x157
        x159 = 2*x121
        x160 = 2*r_23
        x161 = x160*x99
        x162 = -x115 - x116 + x117 + x119
        x163 = x146*x162
        x164 = x140*x4
        x165 = x159 + x161 + x163 + x164
        x166 = 2*x102
        x167 = 2*x103
        x168 = 2*x104
        x169 = 2*x105
        x170 = 2*x106
        x171 = x108*x29
        x172 = -x166 - x167 - x168 + x169 + x170 + x171
        x173 = 4*x76
        x174 = 4*x79
        x175 = 4*x80
        x176 = 4*x77
        x177 = 4*x78
        x178 = d_4*x3
        x179 = x108*x178
        x180 = 8*d_3
        x181 = -x180*x88 - 4*x36*x39
        x182 = x154 + x155 + x157
        x183 = x166 + x167 + x168 - x169 - x170 - x171
        x184 = -x122 + x162
        x185 = x124 + x127
        x186 = x184 + x185
        x187 = x131 + x133 + x136
        x188 = x144 + x145 - x147 + x148
        x189 = a_2*a_3
        x190 = 2*x189
        x191 = x46 + x47
        x192 = R_l_inv_62*x191
        x193 = R_l_inv_22*x190 + x192
        x194 = d_3*x193
        x195 = R_l_inv_61*x191
        x196 = R_l_inv_21*x190 + x195
        x197 = x196*x36
        x198 = R_l_inv_25*x190 + R_l_inv_65*x191
        x199 = x198*x72
        x200 = -d_4*x196
        x201 = R_l_inv_60*x191
        x202 = x95*(R_l_inv_20*x190 + x201)
        x203 = -x202
        x204 = x123*x193
        x205 = -x204
        x206 = x126*x198
        x207 = -x206
        x208 = x198*x36
        x209 = x208*x87
        x210 = -x209
        x211 = x194 + x197 + x199 + x200 + x203 + x205 + x207 + x210
        x212 = R_l_inv_66*x191
        x213 = R_l_inv_26*x190 + x212
        x214 = x121*x213
        x215 = x144*x16
        x216 = x144*x22
        x217 = x18*x29
        x218 = x24*x29
        x219 = x214 - x215 - x216 + x217 + x218
        x220 = R_l_inv_24*x190 + R_l_inv_64*x191
        x221 = r_22*x220
        x222 = R_l_inv_67*x191
        x223 = R_l_inv_27*x190 + x222
        x224 = x223*x81
        x225 = d_3*r_22
        x226 = x213*x225
        x227 = d_5*x31
        x228 = 2*x4
        x229 = x16*x228
        x230 = x22*x228
        x231 = x20*x31
        x232 = x26*x31
        x233 = x221 - x224 - x226 - x227 - x229 - x230 + x231 + x232
        x234 = x219 + x233
        x235 = r_21**3*x64
        x236 = r_21*x43
        x237 = r_21*x44
        x238 = r_21*x45
        x239 = R_l_inv_23*x190 + R_l_inv_63*x191
        x240 = r_21*x239
        x241 = x107*x213
        x242 = r_21*x59
        x243 = r_21*x66
        x244 = r_21*x67
        x245 = r_21*x69
        x246 = x111*x223
        x247 = r_21*x61
        x248 = r_21*x63
        x249 = r_21*x70
        x250 = r_21*x71
        x251 = x16*x49
        x252 = x22*x49
        x253 = x10*x54
        x254 = x12*x54
        x255 = x14*x54
        x256 = d_5*x1
        x257 = x20*x256
        x258 = 2*x22
        x259 = x10*x258
        x260 = x12*x258
        x261 = x14*x258
        x262 = x256*x26
        x263 = 2*r_11
        x264 = r_12*x57
        x265 = r_22*x264
        x266 = x263*x265
        x267 = 2*r_13
        x268 = r_23*x57
        x269 = r_11*x267*x268
        x270 = r_31*x68
        x271 = r_32*x270
        x272 = 2*r_22
        x273 = x271*x272
        x274 = r_33*x270
        x275 = x160*x274
        x276 = x18*x24
        x277 = x1*x276
        x278 = x20*x26
        x279 = x1*x278
        x280 = x16*x22
        x281 = x1*x280
        x282 = x25*x54
        x283 = x27*x54
        x284 = x19*x258
        x285 = x21*x258
        x286 = x235 - x236 - x237 - x238 - x240 - x241 + x242 + x243 + x244 + x245 - x246 - x247 - x248 - x249 - x250 - x251 - x252 + x253 + x254 + x255 + x257 + x259 + x260 + x261 + x262 + x266 + x269 + x273 + x275 - x277 - x279 + x281 + x282 + x283 + x284 + x285
        x287 = 4*x189
        x288 = R_l_inv_20*x287 + 2*x201
        x289 = x288*x36
        x290 = -x95*(R_l_inv_21*x287 + 2*x195)
        x291 = x198*x95
        x292 = x134*x291
        x293 = -x289 + x290 + x292
        x294 = R_l_inv_27*x287 + 2*x222
        x295 = x107*x294
        x296 = x1*x220
        x297 = d_5*x33
        x298 = x213*x31
        x299 = 4*x18
        x300 = x299*x4
        x301 = 4*x24
        x302 = x301*x4
        x303 = x20*x33
        x304 = x26*x33
        x305 = -x295 - x296 - x297 + x298 - x300 - x302 + x303 + x304
        x306 = R_l_inv_26*x287 + 2*x212
        x307 = x306*x81
        x308 = r_22*x43
        x309 = 2*x308
        x310 = r_22*x44
        x311 = 2*x310
        x312 = r_22*x45
        x313 = 2*x312
        x314 = 2*x239
        x315 = r_22*x314
        x316 = r_22**3*x64
        x317 = 2*x316
        x318 = r_22*x125
        x319 = x223*x318
        x320 = r_22*x59
        x321 = 2*x320
        x322 = r_22*x63
        x323 = 2*x322
        x324 = r_22*x69
        x325 = 2*x324
        x326 = r_22*x71
        x327 = 2*x326
        x328 = r_22*x61
        x329 = 2*x328
        x330 = r_22*x65
        x331 = 2*x330
        x332 = r_22*x67
        x333 = 2*x332
        x334 = r_22*x70
        x335 = 2*x334
        x336 = x15*x299
        x337 = x15*x301
        x338 = x10*x299
        x339 = x12*x299
        x340 = x14*x299
        x341 = d_5*x3
        x342 = x20*x341
        x343 = x10*x301
        x344 = x12*x301
        x345 = x14*x301
        x346 = x26*x341
        x347 = 4*r_11
        x348 = r_21*x347
        x349 = x264*x348
        x350 = 4*r_12
        x351 = r_13*x268
        x352 = x350*x351
        x353 = 4*r_21
        x354 = x271*x353
        x355 = 4*r_23
        x356 = r_32*r_33*x68
        x357 = x355*x356
        x358 = x280*x3
        x359 = x278*x3
        x360 = x17*x301
        x361 = x23*x299
        x362 = x276*x3
        x363 = x27*x299
        x364 = x21*x301
        x365 = x307 - x309 - x311 - x313 - x315 + x317 - x319 - x321 - x323 - x325 - x327 + x329 + x331 + x333 + x335 - x336 - x337 + x338 + x339 + x340 + x342 + x343 + x344 + x345 + x346 + x349 + x352 + x354 + x357 - x358 - x359 + x360 + x361 + x362 + x363 + x364
        x366 = -x197
        x367 = x194 + x199 + x200 + x202 + x205 + x207 + x209 + x366
        x368 = -x221 + x224 + x226 + x227 + x229 + x230 - x231 - x232
        x369 = x219 + x368
        x370 = -x235 + x236 + x237 + x238 + x240 + x241 - x242 - x243 - x244 - x245 + x246 + x247 + x248 + x249 + x250 + x251 + x252 - x253 - x254 - x255 - x257 - x259 - x260 - x261 - x262 - x266 - x269 - x273 - x275 + x277 + x279 - x281 - x282 - x283 - x284 - x285
        x371 = x123*x288
        x372 = x95*(R_l_inv_22*x287 + 2*x192)
        x373 = x156*x291
        x374 = x371 - x372 - x373
        x375 = d_5*x178
        x376 = x213*x29
        x377 = x103*x134
        x378 = x104*x134
        x379 = x178*x20
        x380 = x178*x26
        x381 = -x375 + x376 - x377 - x378 + x379 + x380
        x382 = x162*x306
        x383 = r_23*x45
        x384 = 2*x383
        x385 = r_23**3*x64
        x386 = 2*x385
        x387 = r_23*x43
        x388 = 2*x387
        x389 = r_23*x44
        x390 = 2*x389
        x391 = r_23*x314
        x392 = r_23*x63
        x393 = 2*x392
        x394 = r_23*x65
        x395 = 2*x394
        x396 = r_23*x66
        x397 = 2*x396
        x398 = r_23*x71
        x399 = 2*x398
        x400 = x223*x228
        x401 = r_23*x59
        x402 = 2*x401
        x403 = r_23*x61
        x404 = 2*x403
        x405 = r_23*x69
        x406 = 2*x405
        x407 = r_23*x70
        x408 = 2*x407
        x409 = 4*d_5
        x410 = x10*x409
        x411 = x12*x409
        x412 = x14*x409
        x413 = 4*x20
        x414 = x10*x413
        x415 = x12*x413
        x416 = x14*x413
        x417 = 4*x26
        x418 = x10*x417
        x419 = x12*x417
        x420 = x14*x417
        x421 = r_13*x57
        x422 = x348*x421
        x423 = r_13*x3
        x424 = x264*x423
        x425 = x274*x353
        x426 = x3*x356
        x427 = x16*x173
        x428 = x18*x341
        x429 = x15*x413
        x430 = x173*x22
        x431 = x24*x341
        x432 = x15*x417
        x433 = x17*x417
        x434 = x18*x3
        x435 = x26*x434
        x436 = x23*x413
        x437 = x24*x3
        x438 = x20*x437
        x439 = x21*x417
        x440 = x176*x22
        x441 = x103*x301
        x442 = x382 - x384 - x386 + x388 + x390 + x391 - x393 - x395 - x397 - x399 + x400 + x402 + x404 + x406 + x408 + x410 + x411 + x412 - x414 - x415 - x416 - x418 - x419 - x420 - x422 - x424 - x425 - x426 + x427 + x428 + x429 + x430 + x431 + x432 - x433 - x435 - x436 - x438 - x439 + x440 + x441
        x443 = x178*x213
        x444 = 8*d_5
        x445 = x444*x92
        x446 = 8*x92
        x447 = x20*x446
        x448 = x26*x446
        x449 = 8*d_4
        x450 = x449*x77
        x451 = x449*x78
        x452 = 8*x189
        x453 = -x180*x208 - x36*(R_l_inv_22*x452 + 4*x192)
        x454 = x371 + x372 + x373
        x455 = x375 - x376 + x377 + x378 - x379 - x380
        x456 = -x214
        x457 = -x217
        x458 = -x218
        x459 = x215 + x216 + x370 + x456 + x457 + x458
        x460 = x194 + x199 + x200 + x204 + x206
        x461 = x197 + x202 + x210 + x460
        x462 = x289 + x290 + x292
        x463 = -x307 + x309 + x311 + x313 + x315 - x317 + x319 + x321 + x323 + x325 + x327 - x329 - x331 - x333 - x335 + x336 + x337 - x338 - x339 - x340 - x342 - x343 - x344 - x345 - x346 - x349 - x352 - x354 - x357 + x358 + x359 - x360 - x361 - x362 - x363 - x364
        x464 = x203 + x209 + x366 + x460
        x465 = x215 + x216 + x286 + x456 + x457 + x458
        x466 = R_l_inv_71*x191
        x467 = R_l_inv_31*x190 + x466
        x468 = d_4*x467
        x469 = R_l_inv_70*x191
        x470 = x95*(R_l_inv_30*x190 + x469)
        x471 = R_l_inv_72*x191
        x472 = R_l_inv_32*x190 + x471
        x473 = x123*x472
        x474 = -d_3*x472
        x475 = x36*x467
        x476 = -x475
        x477 = R_l_inv_35*x190 + R_l_inv_75*x191
        x478 = -x477*x72
        x479 = x126*x477
        x480 = x36*x477
        x481 = x480*x87
        x482 = x468 + x470 + x473 + x474 + x476 + x478 + x479 + x481
        x483 = R_l_inv_33*x190 + R_l_inv_73*x191
        x484 = r_21*x483
        x485 = R_l_inv_76*x191
        x486 = R_l_inv_36*x190 + x485
        x487 = x107*x486
        x488 = R_l_inv_77*x191
        x489 = R_l_inv_37*x190 + x488
        x490 = x111*x489
        x491 = x121*x486
        x492 = -x491
        x493 = x4*x87
        x494 = -x493
        x495 = x102*x125
        x496 = x105*x125
        x497 = -x496
        x498 = x106*x125
        x499 = -x498
        x500 = x18*x228
        x501 = x228*x24
        x502 = x484 + x487 + x490 + x492 + x494 + x495 + x497 + x499 + x500 + x501
        x503 = x15*x87
        x504 = x10*x87
        x505 = x12*x87
        x506 = x14*x87
        x507 = x16*x29
        x508 = x144*x18
        x509 = x21*x87
        x510 = x22*x29
        x511 = x144*x24
        x512 = x27*x87
        x513 = -x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512
        x514 = x489*x81
        x515 = R_l_inv_34*x190 + R_l_inv_74*x191
        x516 = r_22*x515
        x517 = x225*x486
        x518 = x10*x51
        x519 = x12*x51
        x520 = x14*x51
        x521 = x166*x20
        x522 = x10*x55
        x523 = x12*x55
        x524 = x14*x55
        x525 = x166*x26
        x526 = r_11*x264
        x527 = x1*x526
        x528 = r_23*x264*x267
        x529 = x1*x271
        x530 = x160*x356
        x531 = x18*x49
        x532 = x24*x49
        x533 = x1*x16
        x534 = x24*x533
        x535 = x1*x22
        x536 = x18*x535
        x537 = x19*x55
        x538 = x27*x51
        x539 = x21*x55
        x540 = x117*x258
        x541 = x169*x26
        x542 = x308 - x310 + x312 - x316 + x320 + x322 + x324 + x326 - x328 - x330 - x332 - x334 + x514 - x516 + x517 - x518 - x519 - x520 - x521 - x522 - x523 - x524 - x525 - x527 - x528 - x529 - x530 + x531 + x532 - x534 - x536 - x537 - x538 - x539 + x540 + x541
        x543 = x513 + x542
        x544 = R_l_inv_30*x287 + 2*x469
        x545 = x36*x544
        x546 = x95*(R_l_inv_31*x287 + 2*x466)
        x547 = x477*x95
        x548 = -x134*x547
        x549 = x545 + x546 + x548
        x550 = R_l_inv_37*x287 + 2*x488
        x551 = x107*x550
        x552 = x1*x515
        x553 = x31*x486
        x554 = x1*x45
        x555 = 2*x235
        x556 = x1*x61
        x557 = x1*x63
        x558 = x1*x70
        x559 = x1*x71
        x560 = x1*x59
        x561 = x1*x66
        x562 = x1*x67
        x563 = x1*x69
        x564 = 4*x16
        x565 = x15*x564
        x566 = 4*x22
        x567 = x15*x566
        x568 = x10*x564
        x569 = x12*x564
        x570 = x14*x564
        x571 = x173*x20
        x572 = x10*x566
        x573 = x12*x566
        x574 = x14*x566
        x575 = x173*x26
        x576 = x3*x526
        x577 = x347*x351
        x578 = x271*x3
        x579 = x274*x355
        x580 = x115*x301
        x581 = x174*x26
        x582 = x17*x566
        x583 = x16*x437
        x584 = x27*x564
        x585 = x22*x434
        x586 = x21*x566
        x587 = -x554 + x555 - x556 - x557 - x558 - x559 + x560 + x561 + x562 + x563 - x565 - x567 + x568 + x569 + x570 + x571 + x572 + x573 + x574 + x575 + x576 + x577 + x578 + x579 - x580 - x581 + x582 + x583 + x584 + x585 + x586
        x588 = x1*x43
        x589 = x1*x44
        x590 = -x588 + x589
        x591 = x551 + x552 - x553 + x587 + x590
        x592 = R_l_inv_36*x287 + 2*x485
        x593 = x592*x81
        x594 = 2*x483
        x595 = r_22*x594
        x596 = d_3*x173
        x597 = x318*x489
        x598 = x4*x564
        x599 = x4*x566
        x600 = x111*x413
        x601 = x111*x417
        x602 = -x593 + x595 - x596 + x597 - x598 - x599 + x600 + x601
        x603 = -x470
        x604 = -x481
        x605 = x468 + x473 + x474 + x475 + x478 + x479 + x603 + x604
        x606 = -x484 - x487 - x490 - x495 + x496 + x498 - x500 - x501
        x607 = x492 + x494 + x606
        x608 = -x308 + x310 - x312 + x316 - x320 - x322 - x324 - x326 + x328 + x330 + x332 + x334 - x514 + x516 - x517 + x518 + x519 + x520 + x521 + x522 + x523 + x524 + x525 + x527 + x528 + x529 + x530 - x531 - x532 + x534 + x536 + x537 + x538 + x539 - x540 - x541
        x609 = x513 + x608
        x610 = x95*(R_l_inv_32*x287 + 2*x471)
        x611 = -x123*x544
        x612 = x156*x547
        x613 = x610 + x611 + x612
        x614 = x156*x92
        x615 = x29*x486
        x616 = -x614 - x615
        x617 = x162*x592
        x618 = r_23*x594
        x619 = x228*x489
        x620 = x16*x33
        x621 = x22*x33
        x622 = x111*x299
        x623 = x111*x301
        x624 = -x617 - x618 - x619 - x620 - x621 + x622 + x623
        x625 = x180*x6
        x626 = x178*x486
        x627 = x180*x480 + x36*(R_l_inv_32*x452 + 4*x471)
        x628 = x614 + x615
        x629 = -x610 + x611 - x612
        x630 = x491 + x493
        x631 = x606 + x630
        x632 = x468 - x473 + x474 + x478 - x479
        x633 = x476 + x481 + x603 + x632
        x634 = -x545 + x546 + x548
        x635 = x593 - x595 + x596 - x597 + x598 + x599 - x600 - x601
        x636 = x470 + x475 + x604 + x632
        x637 = x484 + x487 + x490 + x495 + x497 + x499 + x500 + x501 + x630
        x638 = x125*x6
        x639 = d_5*x144
        x640 = x144*x20
        x641 = -x640
        x642 = x144*x26
        x643 = -x642
        x644 = x103*x87
        x645 = x104*x87
        x646 = x638 + x639 + x641 + x643 + x644 + x645
        x647 = x10*x125
        x648 = x12*x125
        x649 = x125*x14
        x650 = x4*x48
        x651 = x16*x31
        x652 = x125*x19
        x653 = x20*x228
        x654 = x22*x31
        x655 = x125*x25
        x656 = x228*x26
        x657 = -x647 - x648 - x649 + x650 - x651 - x652 - x653 - x654 - x655 - x656
        x658 = x10*x48
        x659 = x12*x48
        x660 = x14*x48
        x661 = x16*x256
        x662 = x166*x18
        x663 = x20*x49
        x664 = x22*x256
        x665 = x166*x24
        x666 = x26*x49
        x667 = x10*x53
        x668 = x12*x53
        x669 = x14*x53
        x670 = x10*x56
        x671 = x12*x56
        x672 = x14*x56
        x673 = r_11*x1*x421
        x674 = x265*x267
        x675 = x1*x274
        x676 = x272*x356
        x677 = x258*x77
        x678 = x167*x24
        x679 = x26*x533
        x680 = x19*x56
        x681 = x20*x535
        x682 = x25*x53
        x683 = x21*x56
        x684 = x383 + x385 + x387 - x389 + x392 + x394 + x396 + x398 - x401 - x403 - x405 - x407 - x658 - x659 - x660 - x661 - x662 - x663 - x664 - x665 - x666 + x667 + x668 + x669 + x670 + x671 + x672 + x673 + x674 + x675 + x676 - x677 - x678 + x679 + x680 + x681 + x682 + x683
        x685 = x657 + x684
        x686 = -x111
        x687 = -x76 - x77 - x78 + x79 + x80
        x688 = -x638
        x689 = -x639 + x640 + x642 - x644 - x645
        x690 = x688 + x689
        x691 = x16*x178
        x692 = x178*x22
        x693 = x299*x92
        x694 = x301*x92
        x695 = -x691 - x692 + x693 + x694
        x696 = x588 - x589
        x697 = x587 + x696
        x698 = x3*x43
        x699 = x3*x44
        x700 = x3*x45
        x701 = 8*x15
        x702 = x18*x701
        x703 = 8*x102
        x704 = x20*x703
        x705 = x26*x703
        x706 = x24*x701
        x707 = 8*x24
        x708 = x17*x707
        x709 = 8*x22
        x710 = x117*x709
        x711 = 8*x18
        x712 = x23*x711
        x713 = x19*x707
        x714 = x27*x711
        x715 = 8*x26
        x716 = x105*x715
        x717 = x21*x707
        x718 = 4*x316
        x719 = x10*x711
        x720 = x12*x711
        x721 = x14*x711
        x722 = 8*r_12
        x723 = r_11*r_21
        x724 = x57*x722*x723
        x725 = x351*x722
        x726 = x10*x707
        x727 = x12*x707
        x728 = x14*x707
        x729 = 8*x271
        x730 = r_21*x729
        x731 = 8*r_23
        x732 = x356*x731
        x733 = x3*x59
        x734 = x3*x61
        x735 = x3*x63
        x736 = x3*x65
        x737 = x3*x67
        x738 = x3*x69
        x739 = x3*x70
        x740 = x3*x71
        x741 = x554 - x555 + x556 + x557 + x558 + x559 - x560 - x561 - x562 - x563 + x565 + x567 - x568 - x569 - x570 - x571 - x572 - x573 - x574 - x575 - x576 - x577 - x578 - x579 + x580 + x581 - x582 - x583 - x584 - x585 - x586
        x742 = x590 + x741
        x743 = x638 + x689
        x744 = -x383 - x385 - x387 + x389 - x392 - x394 - x396 - x398 + x401 + x403 + x405 + x407 + x658 + x659 + x660 + x661 + x662 + x663 + x664 + x665 + x666 - x667 - x668 - x669 - x670 - x671 - x672 - x673 - x674 - x675 - x676 + x677 + x678 - x679 - x680 - x681 - x682 - x683
        x745 = x657 + x744
        x746 = x639 + x641 + x643 + x644 + x645 + x688
        x747 = -x267
        x748 = r_12*x87
        x749 = -x748
        x750 = d_3*x267
        x751 = 2*Px
        x752 = 2*r_12
        x753 = -d_5*x267 + r_11*x50 + x22*x263 + x24*x752 + x26*x267 + x267*x52 + x35*x752 + x58*x751 + x60*x751 + x62*x751
        x754 = -x750 + x753
        x755 = d_4*x347
        x756 = d_3*x347
        x757 = d_3*x722
        x758 = x750 + x753
        x759 = d_5*r_11
        x760 = r_13*x94
        x761 = r_13*x22
        x762 = r_11*x52
        x763 = r_11*x26
        x764 = x759 + x760 + x761 - x762 - x763
        x765 = x140*x764
        x766 = x138*x752
        x767 = r_12*x125
        x768 = x108*x767
        x769 = -x765 + x766 - x768
        x770 = r_11*x87
        x771 = x263*x99
        x772 = d_5*r_12
        x773 = r_13*x35
        x774 = r_13*x24
        x775 = r_12*x52
        x776 = r_12*x26
        x777 = x772 + x773 + x774 - x775 - x776
        x778 = x146*x777
        x779 = r_11*x125
        x780 = x779*x82
        x781 = -x770 - x771 - x778 - x780
        x782 = r_12*x50
        x783 = r_12*x22
        x784 = 2*x783
        x785 = r_11*x35
        x786 = 2*x785
        x787 = r_11*x24
        x788 = 2*x787
        x789 = d_4*x267
        x790 = x108*x789
        x791 = -x782 - x784 + x786 + x788 + x790
        x792 = r_12*x134
        x793 = x350*x99
        x794 = 4*x108
        x795 = x764*x794
        x796 = d_3*x350
        x797 = x796*x82
        x798 = x108*x756 - x138*x347 - 4*x777*x82
        x799 = x765 - x766 + x768
        x800 = x770 + x771 + x778 + x780
        x801 = 4*x772
        x802 = Py*x423
        x803 = 4*x774
        x804 = 4*x775
        x805 = 4*x776
        x806 = x108*x755
        x807 = r_13*x134
        x808 = r_12*x94
        x809 = -x783 + x785 + x787 - x808
        x810 = 4*r_13
        x811 = d_3*x810
        x812 = -x794*x809 + x807 + x810*x99 + x811*x82
        x813 = 8*x759
        x814 = 8*x762
        x815 = 8*x760
        x816 = 8*x763
        x817 = 8*x761
        x818 = d_4*x722
        x819 = x782 + x784 - x786 - x788 - x790
        x820 = x213*x789
        x821 = Py*x350
        x822 = x821*x92
        x823 = x22*x792
        x824 = Py*x178
        x825 = r_11*x824
        x826 = x24*x755
        x827 = x820 - x822 - x823 + x825 + x826
        x828 = x294*x764
        x829 = x220*x752
        x830 = d_5*x756
        x831 = r_12*x213
        x832 = x125*x831
        x833 = x156*x760
        x834 = x156*x761
        x835 = Py*x347
        x836 = x4*x835
        x837 = x26*x756
        x838 = -x828 + x829 - x830 - x832 - x833 - x834 + x836 + x837
        x839 = x306*x777
        x840 = x263*x43
        x841 = x263*x44
        x842 = x263*x45
        x843 = r_11*x314
        x844 = r_11**3
        x845 = 2*x57
        x846 = x844*x845
        x847 = x223*x779
        x848 = x263*x66
        x849 = x263*x67
        x850 = x263*x70
        x851 = x263*x71
        x852 = x263*x61
        x853 = x263*x63
        x854 = x263*x65
        x855 = x263*x69
        x856 = x409*x760
        x857 = x409*x761
        x858 = Px*x58
        x859 = 4*x94
        x860 = x858*x859
        x861 = Px*x859
        x862 = x60*x861
        x863 = x62*x861
        x864 = x566*x858
        x865 = Px*x566
        x866 = x60*x865
        x867 = x62*x865
        x868 = x15*x835
        x869 = d_5*x26
        x870 = x347*x869
        x871 = r_21*x64
        x872 = r_12*x3
        x873 = x871*x872
        x874 = x271*x350
        x875 = r_23*x810*x871
        x876 = x274*x810
        x877 = Py*x3
        x878 = x787*x877
        x879 = x26*x52
        x880 = x347*x879
        x881 = x22*x94
        x882 = x347*x881
        x883 = x301*x808
        x884 = x783*x877
        x885 = x417*x760
        x886 = 4*x52
        x887 = x761*x886
        x888 = -x839 - x840 - x841 - x842 - x843 + x846 - x847 - x848 - x849 - x850 - x851 + x852 + x853 + x854 + x855 - x856 - x857 + x860 + x862 + x863 + x864 + x866 + x867 + x868 + x870 + x873 + x874 + x875 + x876 - x878 - x880 + x882 + x883 + x884 + x885 + x887
        x889 = R_l_inv_26*x452 + 4*x212
        x890 = x764*x889
        x891 = x350*x43
        x892 = x350*x44
        x893 = x350*x45
        x894 = x239*x350
        x895 = r_12**3
        x896 = 4*x57
        x897 = x895*x896
        x898 = d_3*x223
        x899 = x350*x898
        x900 = x350*x65
        x901 = x350*x67
        x902 = x350*x69
        x903 = x350*x71
        x904 = x350*x59
        x905 = x350*x63
        x906 = x350*x66
        x907 = x350*x70
        x908 = x444*x773
        x909 = x444*x774
        x910 = 8*x35
        x911 = x858*x910
        x912 = Px*x910
        x913 = x60*x912
        x914 = x62*x912
        x915 = x707*x858
        x916 = Px*x707
        x917 = x60*x916
        x918 = x62*x916
        x919 = Py*x722
        x920 = x15*x919
        x921 = x722*x869
        x922 = r_22*x64
        x923 = x723*x922
        x924 = 8*x923
        x925 = r_11*x729
        x926 = r_13*x731
        x927 = x922*x926
        x928 = 8*r_13
        x929 = x356*x928
        x930 = x722*x881
        x931 = x722*x879
        x932 = 8*x94
        x933 = x787*x932
        x934 = x709*x785
        x935 = x35*x722
        x936 = x24*x935
        x937 = x715*x773
        x938 = 8*x52
        x939 = x774*x938
        x940 = -d_5*x757 - x180*x773 - x180*x774 + x213*x756 - x220*x347 + x26*x757 + x4*x919 - x777*(R_l_inv_27*x452 + 4*x222)
        x941 = x828 - x829 + x830 + x832 + x833 + x834 - x836 - x837
        x942 = x839 + x840 + x841 + x842 + x843 - x846 + x847 + x848 + x849 + x850 + x851 - x852 - x853 - x854 - x855 + x856 + x857 - x860 - x862 - x863 - x864 - x866 - x867 - x868 - x870 - x873 - x874 - x875 - x876 + x878 + x880 - x882 - x883 - x884 - x885 - x887
        x943 = d_5*x818
        x944 = x213*x755
        x945 = Py*x6
        x946 = x928*x945
        x947 = x449*x774
        x948 = x52*x818
        x949 = x26*x818
        x950 = r_13**3
        x951 = Px*x444
        x952 = Px*x938
        x953 = Px*x715
        x954 = x64*x723
        x955 = 8*r_11
        x956 = r_23*x922
        x957 = d_5*x722
        x958 = Py*r_13
        x959 = x24*x722
        x960 = r_13*x26*x444 + x22*x813 - x22*x814 + x22*x815 + x239*x810 + x24*x957 - x26*x935 - x274*x955 + x35*x957 - x356*x722 + x43*x810 + x44*x810 - x45*x810 - x52*x959 + x58*x951 - x59*x810 + x60*x951 - x60*x952 - x60*x953 - x61*x810 + x62*x951 - x62*x952 - x62*x953 + x65*x810 + x66*x810 - x67*x810 + x69*x810 + x70*x810 + x701*x958 + x707*x773 - x71*x810 - x715*x858 - x722*x956 - x731*x954 - x809*x889 + x810*x898 + x813*x94 - x816*x94 - x858*x938 - x879*x928 - x896*x950
        x961 = -x820 + x822 + x823 - x825 - x826
        x962 = d_3*x807
        x963 = -d_5*x807
        x964 = x486*x789
        x965 = Px*x134
        x966 = x58*x965
        x967 = x60*x965
        x968 = x62*x965
        x969 = x835*x92
        x970 = r_12*x824
        x971 = x52*x807
        x972 = x22*x755
        x973 = x24*x792
        x974 = x26*x807
        x975 = -x962 + x963 - x964 + x966 + x967 + x968 + x969 + x970 + x971 + x972 + x973 + x974
        x976 = x592*x777
        x977 = r_11*x594
        x978 = x489*x779
        x979 = d_3*x801
        x980 = x4*x821
        x981 = x156*x776
        x982 = x33*x958
        x983 = x156*x774
        x984 = x976 + x977 + x978 + x979 - x980 - x981 + x982 + x983
        x985 = x550*x764
        x986 = x44*x752
        x987 = x515*x752
        x988 = x845*x895
        x989 = x43*x752
        x990 = x45*x752
        x991 = x59*x752
        x992 = x63*x752
        x993 = x66*x752
        x994 = x70*x752
        x995 = x486*x767
        x996 = x65*x752
        x997 = x67*x752
        x998 = x69*x752
        x999 = x71*x752
        x1000 = x858*x877
        x1001 = Px*x877
        x1002 = x1001*x60
        x1003 = x1001*x62
        x1004 = x301*x858
        x1005 = Px*x301
        x1006 = x1005*x60
        x1007 = x1005*x62
        x1008 = x15*x821
        x1009 = x26*x801
        x1010 = x3*x954
        x1011 = x271*x347
        x1012 = r_23*x64
        x1013 = x1012*x423
        x1014 = x356*x810
        x1015 = d_5*x802
        x1016 = x409*x774
        x1017 = x347*x94
        x1018 = x1017*x24
        x1019 = r_11*x22*x877
        x1020 = Py*x24*x872
        x1021 = x26*x802
        x1022 = x52*x803
        x1023 = x566*x808
        x1024 = x26*x804
        x1025 = -x1000 - x1002 - x1003 - x1004 - x1006 - x1007 - x1008 - x1009 - x1010 - x1011 - x1013 - x1014 + x1015 + x1016 - x1018 - x1019 - x1020 - x1021 - x1022 + x1023 + x1024 + x985 - x986 - x987 - x988 + x989 + x990 - x991 - x992 - x993 - x994 + x995 + x996 + x997 + x998 + x999
        x1026 = R_l_inv_36*x452 + 4*x485
        x1027 = x1026*x764
        x1028 = x350*x483
        x1029 = d_3*x813
        x1030 = d_3*x489
        x1031 = x1030*x350
        x1032 = x180*x760
        x1033 = x180*x761
        x1034 = Py*x4
        x1035 = x1034*x955
        x1036 = d_3*r_11
        x1037 = x1036*x715
        x1038 = x347*x45
        x1039 = x844*x896
        x1040 = x347*x66
        x1041 = x347*x67
        x1042 = x347*x70
        x1043 = x347*x71
        x1044 = x347*x61
        x1045 = x347*x63
        x1046 = x347*x65
        x1047 = x347*x69
        x1048 = x444*x760
        x1049 = x444*x761
        x1050 = x858*x932
        x1051 = Px*x932
        x1052 = x1051*x60
        x1053 = x1051*x62
        x1054 = x709*x858
        x1055 = Px*x709
        x1056 = x1055*x60
        x1057 = x1055*x62
        x1058 = Py*r_11*x701
        x1059 = x26*x813
        x1060 = r_22*x722*x871
        x1061 = x271*x722
        x1062 = x871*x926
        x1063 = x274*x928
        x1064 = x707*x785
        x1065 = x26*x814
        x1066 = x881*x955
        x1067 = x94*x959
        x1068 = x22*x935
        x1069 = x26*x815
        x1070 = x52*x817
        x1071 = -x1038 + x1039 - x1040 - x1041 - x1042 - x1043 + x1044 + x1045 + x1046 + x1047 - x1048 - x1049 + x1050 + x1052 + x1053 + x1054 + x1056 + x1057 + x1058 + x1059 + x1060 + x1061 + x1062 + x1063 - x1064 - x1065 + x1066 + x1067 + x1068 + x1069 + x1070
        x1072 = x347*x43
        x1073 = x347*x44
        x1074 = -x1072 + x1073
        x1075 = x1071 + x1074 + x347*x515 - x486*x756 + x777*(R_l_inv_37*x452 + 4*x488)
        x1076 = -x976 - x977 - x978 - x979 + x980 + x981 - x982 - x983
        x1077 = x1000 + x1002 + x1003 + x1004 + x1006 + x1007 + x1008 + x1009 + x1010 + x1011 + x1013 + x1014 - x1015 - x1016 + x1018 + x1019 + x1020 + x1021 + x1022 - x1023 - x1024 - x985 + x986 + x987 + x988 - x989 - x990 + x991 + x992 + x993 + x994 - x995 - x996 - x997 - x998 - x999
        x1078 = x1036*x449
        x1079 = x486*x755
        x1080 = x1026*x809 - x1030*x810 + x1036*x707 + x1036*x910 - x22*x757 - x483*x810 - x757*x94
        x1081 = x962 + x963 + x964 + x966 + x967 + x968 + x969 + x970 + x971 + x972 + x973 + x974
        x1082 = Px*x156
        x1083 = -x1082*x58
        x1084 = -x1082*x60
        x1085 = -x1082*x62
        x1086 = d_3*x792
        x1087 = d_5*x811
        x1088 = -x756*x94
        x1089 = Py*r_12
        x1090 = -x1089*x33
        x1091 = -x1034*x810
        x1092 = -x22*x756
        x1093 = -x24*x796
        x1094 = -x26*x811
        x1095 = x1083 + x1084 + x1085 + x1086 + x1087 + x1088 + x1090 + x1091 + x1092 + x1093 + x1094
        x1096 = x134*x772
        x1097 = x134*x775
        x1098 = x134*x776
        x1099 = r_13*x824
        x1100 = x134*x774
        x1101 = x1096 - x1097 - x1098 + x1099 + x1100
        x1102 = x267*x44
        x1103 = x267*x43
        x1104 = x267*x45
        x1105 = x845*x950
        x1106 = Px*x409
        x1107 = x1106*x58
        x1108 = x1106*x60
        x1109 = x1106*x62
        x1110 = x267*x65
        x1111 = x267*x66
        x1112 = x267*x69
        x1113 = x267*x70
        x1114 = x267*x59
        x1115 = x267*x61
        x1116 = x267*x67
        x1117 = x267*x71
        x1118 = d_5*x347
        x1119 = x1118*x94
        x1120 = x772*x877
        x1121 = Py*x15*x810
        x1122 = x1118*x22
        x1123 = x24*x801
        x1124 = x810*x869
        x1125 = x858*x886
        x1126 = Px*x886
        x1127 = x1126*x60
        x1128 = x1126*x62
        x1129 = x417*x858
        x1130 = Px*x417
        x1131 = x1130*x60
        x1132 = x1130*x62
        x1133 = x1012*x348
        x1134 = x274*x347
        x1135 = x1012*x872
        x1136 = x350*x356
        x1137 = x566*x760
        x1138 = x774*x877
        x1139 = x1017*x26
        x1140 = x22*x347*x52
        x1141 = x776*x877
        x1142 = x24*x804
        x1143 = x810*x879
        x1144 = -x1102 + x1103 + x1104 + x1105 - x1107 - x1108 - x1109 - x1110 - x1111 - x1112 - x1113 + x1114 + x1115 + x1116 + x1117 - x1119 - x1120 - x1121 - x1122 - x1123 - x1124 + x1125 + x1127 + x1128 + x1129 + x1131 + x1132 + x1133 + x1134 + x1135 + x1136 - x1137 - x1138 + x1139 + x1140 + x1141 + x1142 + x1143
        x1145 = -x1036
        x1146 = -x1096 + x1097 + x1098 - x1099 - x1100
        x1147 = x1083 + x1084 + x1085 - x1086 + x1087 + x1088 + x1090 + x1091 + x1092 + x1093 + x1094
        x1148 = -x22*x818 + x449*x787 - x919*x92 + x945*x955
        x1149 = 16*d_5
        x1150 = 16*x26
        x1151 = 16*x22
        x1152 = 16*x35
        x1153 = Px*x1152
        x1154 = 16*x24
        x1155 = Px*x1154
        x1156 = 16*r_13
        x1157 = x1102 - x1103 - x1104 - x1105 + x1107 + x1108 + x1109 + x1110 + x1111 + x1112 + x1113 - x1114 - x1115 - x1116 - x1117 + x1119 + x1120 + x1121 + x1122 + x1123 + x1124 - x1125 - x1127 - x1128 - x1129 - x1131 - x1132 - x1133 - x1134 - x1135 - x1136 + x1137 + x1138 - x1139 - x1140 - x1141 - x1142 - x1143
        x1158 = -x10 - x12 - x14 + x15 - x17 - x19 - x21 - x23 - x25 - x27
        x1159 = x1158 + x6
        x1160 = -x29
        x1161 = x128 + x184
        x1162 = x100 + x109 + x112 + x92 + x98
        x1163 = a_3 + x152 + x38 + x40 + x74 + x75 + x90
        x1164 = x139 + x141 - x142
        x1165 = x151 + x42 + x84 + x86
        x1166 = x101 + x110 + x113 + x93 + x97
        x1167 = -x159 - x161 - x163 - x164
        x1168 = x115 + x116 + x118 + x120 + x122 + x185
        x1169 = x295 + x296 + x297 - x298 + x300 + x302 - x303 - x304
        x1170 = -x382 + x384 + x386 - x388 - x390 - x391 + x393 + x395 + x397 + x399 - x400 - x402 - x404 - x406 - x408 - x410 - x411 - x412 + x414 + x415 + x416 + x418 + x419 + x420 + x422 + x424 + x425 + x426 - x427 - x428 - x429 - x430 - x431 - x432 + x433 + x435 + x436 + x438 + x439 - x440 - x441
        x1171 = x503 - x504 - x505 - x506 - x507 - x508 - x509 - x510 - x511 - x512
        x1172 = x1171 + x608
        x1173 = -x551 - x552 + x553 + x696 + x741
        x1174 = x1171 + x542
        x1175 = x617 + x618 + x619 + x620 + x621 - x622 - x623
        x1176 = x647 + x648 + x649 - x650 + x651 + x652 + x653 + x654 + x655 + x656
        x1177 = x1176 + x744
        x1178 = x691 + x692 - x693 - x694
        x1179 = x1176 + x684
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = x0
        A[0, 2] = x0
        A[0, 3] = x2
        A[0, 4] = -x3
        A[0, 5] = x1
        A[0, 6] = r_23
        A[0, 8] = r_23
        A[1, 0] = x28 + x8
        A[1, 1] = x29
        A[1, 2] = x30 + x5
        A[1, 3] = x32
        A[1, 4] = -x33
        A[1, 5] = x31
        A[1, 6] = x28 + x34
        A[1, 7] = x29
        A[1, 8] = x30 + x4
        A[2, 0] = x114 + x129 + x91
        A[2, 1] = x137 + x143 + x149
        A[2, 2] = x129 + x150 + x153
        A[2, 3] = x158 + x165 + x172
        A[2, 4] = x173 - x174 - x175 + x176 + x177 + x179 + x181
        A[2, 5] = x165 + x182 + x183
        A[2, 6] = x150 + x186 + x91
        A[2, 7] = x143 + x187 + x188
        A[2, 8] = x114 + x153 + x186
        A[3, 0] = x211 + x234 + x286
        A[3, 1] = x293 + x305 + x365
        A[3, 2] = x367 + x369 + x370
        A[3, 3] = x374 + x381 + x442
        A[3, 4] = x443 + x445 - x447 - x448 + x450 + x451 + x453
        A[3, 5] = x442 + x454 + x455
        A[3, 6] = x233 + x459 + x461
        A[3, 7] = x305 + x462 + x463
        A[3, 8] = x368 + x464 + x465
        A[4, 0] = x482 + x502 + x543
        A[4, 1] = x549 + x591 + x602
        A[4, 2] = x605 + x607 + x609
        A[4, 3] = x613 + x616 + x624
        A[4, 4] = -x625 - x626 + x627
        A[4, 5] = x624 + x628 + x629
        A[4, 6] = x543 + x631 + x633
        A[4, 7] = x591 + x634 + x635
        A[4, 8] = x609 + x636 + x637
        A[5, 0] = x646 + x685
        A[5, 1] = x134*(x686 + x687)
        A[5, 2] = x685 + x690
        A[5, 3] = x695 + x697
        A[5, 4] = x698 - x699 - x700 - x702 + x704 + x705 - x706 + x708 - x710 + x712 + x713 + x714 - x716 + x717 + x718 + x719 + x720 + x721 + x724 + x725 + x726 + x727 + x728 + x730 + x732 - x733 + x734 - x735 + x736 + x737 - x738 + x739 - x740
        A[5, 5] = x695 + x742
        A[5, 6] = x743 + x745
        A[5, 7] = x134*(x686 + x81)
        A[5, 8] = x745 + x746
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x747
        B[0, 2] = x747
        B[0, 3] = -x347
        B[0, 4] = -x722
        B[0, 5] = x347
        B[0, 6] = x267
        B[0, 8] = x267
        B[1, 0] = x749 + x754
        B[1, 1] = x755
        B[1, 2] = x748 + x754
        B[1, 3] = -x756
        B[1, 4] = -x757
        B[1, 5] = x756
        B[1, 6] = x749 + x758
        B[1, 7] = x755
        B[1, 8] = x748 + x758
        B[2, 0] = x769 + x781 + x791
        B[2, 1] = -x792 - x793 + x795 - x797 + x798
        B[2, 2] = x791 + x799 + x800
        B[2, 3] = -x801 - x802 - x803 + x804 + x805 + x806 + x812
        B[2, 4] = x108*x818 + x813 - x814 + x815 - x816 + x817
        B[2, 5] = x801 + x802 + x803 - x804 - x805 - x806 + x812
        B[2, 6] = x769 + x800 + x819
        B[2, 7] = x792 + x793 - x795 + x797 + x798
        B[2, 8] = x781 + x799 + x819
        B[3, 0] = x827 + x838 + x888
        B[3, 1] = x890 - x891 - x892 - x893 - x894 + x897 - x899 - x900 - x901 - x902 - x903 + x904 + x905 + x906 + x907 - x908 - x909 + x911 + x913 + x914 + x915 + x917 + x918 + x920 + x921 + x924 + x925 + x927 + x929 - x930 - x931 + x933 + x934 + x936 + x937 + x939 + x940
        B[3, 2] = x827 + x941 + x942
        B[3, 3] = -x943 + x944 - x946 - x947 + x948 + x949 + x960
        B[3, 4] = x449*(r_13*x50 + x22*x267 + 2*x759 - 2*x762 - 2*x763 + x831)
        B[3, 5] = x943 - x944 + x946 + x947 - x948 - x949 + x960
        B[3, 6] = x838 + x942 + x961
        B[3, 7] = -x890 + x891 + x892 + x893 + x894 - x897 + x899 + x900 + x901 + x902 + x903 - x904 - x905 - x906 - x907 + x908 + x909 - x911 - x913 - x914 - x915 - x917 - x918 - x920 - x921 - x924 - x925 - x927 - x929 + x930 + x931 - x933 - x934 - x936 - x937 - x939 + x940
        B[3, 8] = x888 + x941 + x961
        B[4, 0] = x1025 + x975 + x984
        B[4, 1] = -x1027 + x1028 - x1029 + x1031 - x1032 - x1033 + x1035 + x1037 + x1075
        B[4, 2] = x1076 + x1077 + x975
        B[4, 3] = -x1078 - x1079 + x1080
        B[4, 4] = -x818*(x125 + x486)
        B[4, 5] = x1078 + x1079 + x1080
        B[4, 6] = x1025 + x1076 + x1081
        B[4, 7] = x1027 - x1028 + x1029 - x1031 + x1032 + x1033 - x1035 - x1037 + x1075
        B[4, 8] = x1077 + x1081 + x984
        B[5, 0] = x1095 + x1101 + x1144
        B[5, 1] = x449*(x1145 - x759 - x760 - x761 + x762 + x763)
        B[5, 2] = x1144 + x1146 + x1147
        B[5, 3] = x1071 + x1072 - x1073 + x1148
        B[5, 4] = 16*r_11*x271 + r_12*x1152*x24 + 16*x1089*x15 - x1149*x773 - x1149*x774 + x1150*x772 + x1150*x773 - x1150*x775 + x1151*x785 - x1151*x808 + x1152*x858 + x1153*x60 + x1153*x62 + x1154*x858 + x1155*x60 + x1155*x62 + x1156*x356 + x1156*x956 + x43*x722 - x44*x722 - x45*x722 + 16*x52*x774 + 8*x57*x895 + x59*x722 + x63*x722 - x65*x722 + x66*x722 - x67*x722 - x69*x722 + x70*x722 - x71*x722 + 16*x787*x94 + 16*x923
        B[5, 5] = x1038 - x1039 + x1040 + x1041 + x1042 + x1043 - x1044 - x1045 - x1046 - x1047 + x1048 + x1049 - x1050 - x1052 - x1053 - x1054 - x1056 - x1057 - x1058 - x1059 - x1060 - x1061 - x1062 - x1063 + x1064 + x1065 - x1066 - x1067 - x1068 - x1069 - x1070 + x1074 + x1148
        B[5, 6] = x1095 + x1146 + x1157
        B[5, 7] = x449*(x1145 + x764)
        B[5, 8] = x1101 + x1147 + x1157
        C = np.zeros(shape=(6, 9))
        C[0, 0] = r_23
        C[0, 2] = r_23
        C[0, 3] = x1
        C[0, 4] = x3
        C[0, 5] = x2
        C[0, 6] = x0
        C[0, 8] = x0
        C[1, 0] = x1159 + x4
        C[1, 1] = x1160
        C[1, 2] = x1158 + x34
        C[1, 3] = x31
        C[1, 4] = x33
        C[1, 5] = x32
        C[1, 6] = x1159 + x5
        C[1, 7] = x1160
        C[1, 8] = x1158 + x8
        C[2, 0] = x1161 + x1162 + x1163
        C[2, 1] = x1164 + x137 + x188
        C[2, 2] = x1161 + x1165 + x1166
        C[2, 3] = x1167 + x158 + x183
        C[2, 4] = -x173 + x174 + x175 - x176 - x177 - x179 + x181
        C[2, 5] = x1167 + x172 + x182
        C[2, 6] = x1163 + x1166 + x1168
        C[2, 7] = x1164 + x149 + x187
        C[2, 8] = x1162 + x1165 + x1168
        C[3, 0] = x211 + x368 + x459
        C[3, 1] = x1169 + x293 + x463
        C[3, 2] = x233 + x367 + x465
        C[3, 3] = x1170 + x374 + x455
        C[3, 4] = -x443 - x445 + x447 + x448 - x450 - x451 + x453
        C[3, 5] = x1170 + x381 + x454
        C[3, 6] = x286 + x369 + x461
        C[3, 7] = x1169 + x365 + x462
        C[3, 8] = x234 + x370 + x464
        C[4, 0] = x1172 + x482 + x631
        C[4, 1] = x1173 + x549 + x635
        C[4, 2] = x1174 + x605 + x637
        C[4, 3] = x1175 + x613 + x628
        C[4, 4] = x625 + x626 + x627
        C[4, 5] = x1175 + x616 + x629
        C[4, 6] = x1172 + x502 + x633
        C[4, 7] = x1173 + x602 + x634
        C[4, 8] = x1174 + x607 + x636
        C[5, 0] = x1177 + x690
        C[5, 1] = x134*(x111 + x81)
        C[5, 2] = x1177 + x646
        C[5, 3] = x1178 + x742
        C[5, 4] = -x698 + x699 + x700 + x702 - x704 - x705 + x706 - x708 + x710 - x712 - x713 - x714 + x716 - x717 - x718 - x719 - x720 - x721 - x724 - x725 - x726 - x727 - x728 - x730 - x732 + x733 - x734 + x735 - x736 - x737 + x738 - x739 + x740
        C[5, 5] = x1178 + x697
        C[5, 6] = x1179 + x746
        C[5, 7] = x134*(x111 + x687)
        C[5, 8] = x1179 + x743
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
            condition_0: bool = (abs((-Px*math.sin(th_0) + Py*math.cos(th_0) + d_5*(r_13*math.sin(th_0) - r_23*math.cos(th_0)))/d_4) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                x2 = safe_acos((Px*x0 - Py*x1 - d_5*(r_13*x0 - r_23*x1))/d_4)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs((-Px*math.sin(th_0) + Py*math.cos(th_0) + d_5*(r_13*math.sin(th_0) - r_23*math.cos(th_0)))/d_4) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                x2 = safe_acos((Px*x0 - Py*x1 - d_5*(r_13*x0 - r_23*x1))/d_4)
                # End of temp variables
                this_solution[3] = -x2
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
            th_3 = this_solution[3]
            condition_0: bool = (2*abs(a_2*d_3) >= zero_tolerance) or (abs(2*a_2*a_3 - 2*a_2*d_4*math.sin(th_3)) >= zero_tolerance) or (abs(-a_2**2 - a_3**2 + 2*a_3*d_4*math.sin(th_3) - d_3**2 - d_4**2 + d_5**2 + 2*d_5*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*a_2
                x1 = d_4*math.sin(th_3)
                x2 = a_3*x0 - x0*x1
                x3 = math.atan2(d_3*x0, x2)
                x4 = a_2**2
                x5 = d_3**2
                x6 = -a_3**2 + 2*a_3*x1 - d_4**2 + d_5**2 + 2*d_5*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2 - x4 - x5
                x7 = safe_sqrt(x2**2 + 4*x4*x5 - x6**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = x3 + math.atan2(x7, x6)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                
            condition_1: bool = (2*abs(a_2*d_3) >= zero_tolerance) or (abs(2*a_2*a_3 - 2*a_2*d_4*math.sin(th_3)) >= zero_tolerance) or (abs(-a_2**2 - a_3**2 + 2*a_3*d_4*math.sin(th_3) - d_3**2 - d_4**2 + d_5**2 + 2*d_5*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*a_2
                x1 = d_4*math.sin(th_3)
                x2 = a_3*x0 - x0*x1
                x3 = math.atan2(d_3*x0, x2)
                x4 = a_2**2
                x5 = d_3**2
                x6 = -a_3**2 + 2*a_3*x1 - d_4**2 + d_5**2 + 2*d_5*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2 - x4 - x5
                x7 = safe_sqrt(x2**2 + 4*x4*x5 - x6**2)
                # End of temp variables
                this_solution[2] = x3 + math.atan2(-x7, x6)
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
            th_3 = this_solution[3]
            degenerate_valid_0 = (abs(th_3) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(12, node_input_i_idx_in_queue)
            
            th_3 = this_solution[3]
            degenerate_valid_1 = (abs(th_3 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(17, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(7, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_6_processor()
    # Finish code for solved_variable dispatcher node 6
    
    # Code for explicit solution node 17, solved variable is th_5
    def ExplicitSolutionNode_node_17_solve_th_5_processor():
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
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                # End of temp variables
                this_solution[5] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(18, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_5_processor()
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
            th_2 = this_solution[2]
            degenerate_valid_0 = (abs(th_2 - 1.44644133224814) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(28, node_input_i_idx_in_queue)
            
            th_2 = this_solution[2]
            degenerate_valid_1 = (abs(-th_2 + 1.44644133224814 + math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(31, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(19, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_18_processor()
    # Finish code for solved_variable dispatcher node 18
    
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
            th_5 = this_solution[5]
            condition_0: bool = (abs(0.992277876713668*a_2 - d_3) >= zero_tolerance) or (abs(d_5 + inv_Pz) >= zero_tolerance) or (abs(inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)
                x1 = d_5 + inv_Pz
                x2 = math.atan2(x0, x1)
                x3 = -0.992277876713668*a_2 + d_3
                x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x2 + math.atan2(x4, x3)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(32, appended_idx)
                
            condition_1: bool = (abs(0.992277876713668*a_2 - d_3) >= zero_tolerance) or (abs(d_5 + inv_Pz) >= zero_tolerance) or (abs(inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)
                x1 = d_5 + inv_Pz
                x2 = math.atan2(x0, x1)
                x3 = -0.992277876713668*a_2 + d_3
                x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                this_solution[4] = x2 + math.atan2(-x4, x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(32, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_4_processor()
    # Finish code for explicit solution node 31
    
    # Code for non-branch dispatcher node 32
    # Actually, there is no code
    
    # Code for explicit solution node 33, solved variable is th_1
    def ExplicitSolutionNode_node_33_solve_th_1_processor():
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
            th_5 = this_solution[5]
            condition_0: bool = (abs(0.992277876713668*a_3 - 0.124034734589209*d_3) >= zero_tolerance) or (abs(-a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = -a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3
                x1 = math.atan2(x0, -0.992277876713668*a_3 + 0.124034734589209*d_3)
                x2 = Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.984615384615385*(-a_3 + 0.125*d_3)**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x1 + math.atan2(x3, x2)
                appended_idx = append_solution_to_queue(solution_0)
                
            condition_1: bool = (abs(0.992277876713668*a_3 - 0.124034734589209*d_3) >= zero_tolerance) or (abs(-a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = -a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3
                x1 = math.atan2(x0, -0.992277876713668*a_3 + 0.124034734589209*d_3)
                x2 = Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.984615384615385*(-a_3 + 0.125*d_3)**2)
                # End of temp variables
                this_solution[1] = x1 + math.atan2(-x3, x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_33_solve_th_1_processor()
    # Finish code for explicit solution node 32
    
    # Code for explicit solution node 28, solved variable is th_4
    def ExplicitSolutionNode_node_28_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[28]
        this_input_valid: bool = node_input_validity[28]
        if not this_input_valid:
            return
        
        # The solution of non-root node 28
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_5 = this_solution[5]
            condition_0: bool = (abs(0.992277876713668*a_2 + d_3) >= zero_tolerance) or (abs(d_5 + inv_Pz) >= zero_tolerance) or (abs(inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)
                x1 = d_5 + inv_Pz
                x2 = math.atan2(x0, x1)
                x3 = 0.992277876713668*a_2 + d_3
                x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x2 + math.atan2(x4, x3)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(29, appended_idx)
                
            condition_1: bool = (abs(0.992277876713668*a_2 + d_3) >= zero_tolerance) or (abs(d_5 + inv_Pz) >= zero_tolerance) or (abs(inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)
                x1 = d_5 + inv_Pz
                x2 = math.atan2(x0, x1)
                x3 = 0.992277876713668*a_2 + d_3
                x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                this_solution[4] = x2 + math.atan2(-x4, x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(29, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_28_solve_th_4_processor()
    # Finish code for explicit solution node 28
    
    # Code for non-branch dispatcher node 29
    # Actually, there is no code
    
    # Code for explicit solution node 30, solved variable is th_1
    def ExplicitSolutionNode_node_30_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[29]
        this_input_valid: bool = node_input_validity[29]
        if not this_input_valid:
            return
        
        # The solution of non-root node 30
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_5 = this_solution[5]
            condition_0: bool = (abs(0.992277876713668*a_3 - 0.124034734589209*d_3) >= zero_tolerance) or (abs(a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3
                x1 = math.atan2(x0, -0.992277876713668*a_3 + 0.124034734589209*d_3)
                x2 = -Pz - d_4*r_31*math.sin(th_5) - d_4*r_32*math.cos(th_5) + d_5*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.984615384615385*(-a_3 + 0.125*d_3)**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x1 + math.atan2(x3, x2)
                appended_idx = append_solution_to_queue(solution_0)
                
            condition_1: bool = (abs(0.992277876713668*a_3 - 0.124034734589209*d_3) >= zero_tolerance) or (abs(a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3
                x1 = math.atan2(x0, -0.992277876713668*a_3 + 0.124034734589209*d_3)
                x2 = -Pz - d_4*r_31*math.sin(th_5) - d_4*r_32*math.cos(th_5) + d_5*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.984615384615385*(-a_3 + 0.125*d_3)**2)
                # End of temp variables
                this_solution[1] = x1 + math.atan2(-x3, x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_30_solve_th_1_processor()
    # Finish code for explicit solution node 29
    
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
            th_2 = this_solution[2]
            condition_0: bool = (abs(a_3*math.sin(th_2) - d_3*math.cos(th_2)) >= 1.0e-6) or (abs(a_2 + a_3*math.cos(th_2) + d_3*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = d_5*r_33
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = a_2 + a_3*x1 + d_3*x2
                x4 = a_3*x2 - d_3*x1
                x5 = math.cos(th_0)
                x6 = math.sin(th_0)
                x7 = Px*x5 + Py*x6 - d_5*r_13*x5 - d_5*r_23*x6
                # End of temp variables
                this_solution[1] = math.atan2(x3*(-Pz + x0) + x4*x7, x3*x7 + x4*(Pz - x0))
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
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_13*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + r_23*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - r_33*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))) >= zero_tolerance) or (abs(r_13*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_23*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.sin(th_0) + r_33*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_1)
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = math.cos(th_1)
                x4 = x0*x1 - x2*x3
                x5 = x0*x2 + x1*x3
                x6 = r_13*math.cos(th_0)
                x7 = r_23*math.sin(th_0)
                # End of temp variables
                this_solution[4] = math.atan2(-r_33*x4 + x5*x6 + x5*x7, r_33*x5 + x4*x6 + x4*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_4_processor()
    # Finish code for explicit solution node 20
    
    # Code for explicit solution node 12, solved variable is th_5
    def ExplicitSolutionNode_node_12_solve_th_5_processor():
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
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                # End of temp variables
                this_solution[5] = math.atan2(-r_11*x1 + r_21*x0, -r_12*x1 + r_22*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(13, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_12_solve_th_5_processor()
    # Finish code for explicit solution node 12
    
    # Code for solved_variable dispatcher node 13
    def SolvedVariableDispatcherNode_node_13_processor():
        this_node_input_index: List[int] = node_input_index[13]
        this_input_valid: bool = node_input_validity[13]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_2 = this_solution[2]
            degenerate_valid_0 = (abs(th_2 - 1.44644133224814) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(22, node_input_i_idx_in_queue)
            
            th_2 = this_solution[2]
            degenerate_valid_1 = (abs(-th_2 + 1.44644133224814 + math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(25, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(14, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_13_processor()
    # Finish code for solved_variable dispatcher node 13
    
    # Code for explicit solution node 25, solved variable is th_4
    def ExplicitSolutionNode_node_25_solve_th_4_processor():
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
            th_5 = this_solution[5]
            condition_0: bool = (abs(0.992277876713668*a_2 - d_3) >= zero_tolerance) or (abs(d_5 + inv_Pz) >= zero_tolerance) or (abs(inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)
                x1 = d_5 + inv_Pz
                x2 = math.atan2(x0, x1)
                x3 = -0.992277876713668*a_2 + d_3
                x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x2 + math.atan2(x4, x3)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(26, appended_idx)
                
            condition_1: bool = (abs(0.992277876713668*a_2 - d_3) >= zero_tolerance) or (abs(d_5 + inv_Pz) >= zero_tolerance) or (abs(inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)
                x1 = d_5 + inv_Pz
                x2 = math.atan2(x0, x1)
                x3 = -0.992277876713668*a_2 + d_3
                x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                this_solution[4] = x2 + math.atan2(-x4, x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(26, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_4_processor()
    # Finish code for explicit solution node 25
    
    # Code for non-branch dispatcher node 26
    # Actually, there is no code
    
    # Code for explicit solution node 27, solved variable is th_1
    def ExplicitSolutionNode_node_27_solve_th_1_processor():
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
            th_5 = this_solution[5]
            condition_0: bool = (abs(0.992277876713668*a_3 - 0.124034734589209*d_3) >= zero_tolerance) or (abs(-a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = -a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3
                x1 = math.atan2(x0, -0.992277876713668*a_3 + 0.124034734589209*d_3)
                x2 = Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.984615384615385*(-a_3 + 0.125*d_3)**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x1 + math.atan2(x3, x2)
                appended_idx = append_solution_to_queue(solution_0)
                
            condition_1: bool = (abs(0.992277876713668*a_3 - 0.124034734589209*d_3) >= zero_tolerance) or (abs(-a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = -a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3
                x1 = math.atan2(x0, -0.992277876713668*a_3 + 0.124034734589209*d_3)
                x2 = Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.984615384615385*(-a_3 + 0.125*d_3)**2)
                # End of temp variables
                this_solution[1] = x1 + math.atan2(-x3, x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_1_processor()
    # Finish code for explicit solution node 26
    
    # Code for explicit solution node 22, solved variable is th_4
    def ExplicitSolutionNode_node_22_solve_th_4_processor():
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
            th_5 = this_solution[5]
            condition_0: bool = (abs(0.992277876713668*a_2 + d_3) >= zero_tolerance) or (abs(d_5 + inv_Pz) >= zero_tolerance) or (abs(inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)
                x1 = d_5 + inv_Pz
                x2 = math.atan2(x0, x1)
                x3 = 0.992277876713668*a_2 + d_3
                x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x2 + math.atan2(x4, x3)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(23, appended_idx)
                
            condition_1: bool = (abs(0.992277876713668*a_2 + d_3) >= zero_tolerance) or (abs(d_5 + inv_Pz) >= zero_tolerance) or (abs(inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = inv_Px*math.cos(th_5) - inv_Py*math.sin(th_5)
                x1 = d_5 + inv_Pz
                x2 = math.atan2(x0, x1)
                x3 = 0.992277876713668*a_2 + d_3
                x4 = safe_sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                this_solution[4] = x2 + math.atan2(-x4, x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(23, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_22_solve_th_4_processor()
    # Finish code for explicit solution node 22
    
    # Code for non-branch dispatcher node 23
    # Actually, there is no code
    
    # Code for explicit solution node 24, solved variable is th_1
    def ExplicitSolutionNode_node_24_solve_th_1_processor():
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
            th_5 = this_solution[5]
            condition_0: bool = (abs(0.992277876713668*a_3 - 0.124034734589209*d_3) >= zero_tolerance) or (abs(a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3
                x1 = math.atan2(x0, -0.992277876713668*a_3 + 0.124034734589209*d_3)
                x2 = -Pz - d_4*r_31*math.sin(th_5) - d_4*r_32*math.cos(th_5) + d_5*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.984615384615385*(-a_3 + 0.125*d_3)**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x1 + math.atan2(x3, x2)
                appended_idx = append_solution_to_queue(solution_0)
                
            condition_1: bool = (abs(0.992277876713668*a_3 - 0.124034734589209*d_3) >= zero_tolerance) or (abs(a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = a_2 + 0.124034734589209*a_3 + 0.992277876713668*d_3
                x1 = math.atan2(x0, -0.992277876713668*a_3 + 0.124034734589209*d_3)
                x2 = -Pz - d_4*r_31*math.sin(th_5) - d_4*r_32*math.cos(th_5) + d_5*r_33
                x3 = safe_sqrt(x0**2 - x2**2 + 0.984615384615385*(-a_3 + 0.125*d_3)**2)
                # End of temp variables
                this_solution[1] = x1 + math.atan2(-x3, x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_1_processor()
    # Finish code for explicit solution node 23
    
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
            th_2 = this_solution[2]
            condition_0: bool = (abs(a_3*math.sin(th_2) - d_3*math.cos(th_2)) >= 1.0e-6) or (abs(a_2 + a_3*math.cos(th_2) + d_3*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = d_5*r_33
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = a_2 + a_3*x1 + d_3*x2
                x4 = a_3*x2 - d_3*x1
                x5 = math.cos(th_0)
                x6 = math.sin(th_0)
                x7 = Px*x5 + Py*x6 - d_5*r_13*x5 - d_5*r_23*x6
                # End of temp variables
                this_solution[1] = math.atan2(x3*(-Pz + x0) + x4*x7, x3*x7 + x4*(Pz - x0))
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
            th_2 = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_13*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + r_23*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_0) - r_33*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))) >= zero_tolerance) or (abs(r_13*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_23*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.sin(th_0) + r_33*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_1)
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = math.cos(th_1)
                x4 = x0*x1 - x2*x3
                x5 = x0*x2 + x1*x3
                x6 = r_13*math.cos(th_0)
                x7 = r_23*math.sin(th_0)
                # End of temp variables
                this_solution[4] = math.atan2(r_33*x4 - x5*x6 - x5*x7, r_33*x5 + x4*x6 + x4*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_4_processor()
    # Finish code for explicit solution node 15
    
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
            th_3 = this_solution[3]
            condition_0: bool = (abs((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3)) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = safe_asin((-r_13*math.sin(th_0) + r_23*math.cos(th_0))/math.sin(th_3))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
            condition_1: bool = (abs((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = safe_asin((-r_13*math.sin(th_0) + r_23*math.cos(th_0))/math.sin(th_3))
                # End of temp variables
                this_solution[4] = math.pi - x0
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
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_4 = this_solution[4]
            checked_result: bool = (abs(a_2*math.sin(th_3)*math.cos(th_2) + a_3*math.sin(th_3) - d_4) <= 1.0e-6) and (abs(a_2*math.sin(th_2)*math.sin(th_4) - a_2*math.cos(th_2)*math.cos(th_3)*math.cos(th_4) - a_3*math.cos(th_3)*math.cos(th_4) + d_3*math.sin(th_4)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor()
    # Finish code for equation all-zero dispatcher node 8
    
    # Code for explicit solution node 9, solved variable is th_5
    def ExplicitSolutionNode_node_9_solve_th_5_processor():
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
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_4 = this_solution[4]
            condition_0: bool = (abs(a_2*math.sin(th_3)*math.cos(th_2) + a_3*math.sin(th_3) - d_4) >= 1.0e-6) or (abs(a_2*math.sin(th_2)*math.sin(th_4) - a_2*math.cos(th_2)*math.cos(th_3)*math.cos(th_4) - a_3*math.cos(th_3)*math.cos(th_4) + d_3*math.sin(th_4)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_3)
                x1 = a_2*math.cos(th_2)
                x2 = -a_3*x0 + d_4 - x0*x1
                x3 = math.sin(th_4)
                x4 = math.cos(th_3)*math.cos(th_4)
                x5 = a_2*x3*math.sin(th_2) - a_3*x4 + d_3*x3 - x1*x4
                # End of temp variables
                this_solution[5] = math.atan2(inv_Px*x2 - inv_Py*x5, inv_Px*x5 + inv_Py*x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_5_processor()
    # Finish code for explicit solution node 9
    
    # Code for equation all-zero dispatcher node 10
    def EquationAllZeroDispatcherNode_node_10_processor():
        this_node_input_index: List[int] = node_input_index[10]
        this_input_valid: bool = node_input_validity[10]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            checked_result: bool = (abs(Pz - d_5*r_33) <= 1.0e-6) and (abs(Px*math.cos(th_0) + Py*math.sin(th_0) - d_5*r_13*math.cos(th_0) - d_5*r_23*math.sin(th_0)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(11, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_10_processor()
    # Finish code for equation all-zero dispatcher node 10
    
    # Code for explicit solution node 11, solved variable is th_1
    def ExplicitSolutionNode_node_11_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[11]
        this_input_valid: bool = node_input_validity[11]
        if not this_input_valid:
            return
        
        # The solution of non-root node 11
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            condition_0: bool = (abs(Pz - d_5*r_33) >= 1.0e-6) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) - d_5*r_13*math.cos(th_0) - d_5*r_23*math.sin(th_0)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = Pz - d_5*r_33
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = d_4*math.sin(th_3)
                x4 = -a_2 - a_3*x1 - d_3*x2 + x1*x3
                x5 = a_3*x2 - d_3*x1 - x2*x3
                x6 = math.cos(th_0)
                x7 = math.sin(th_0)
                x8 = -Px*x6 - Py*x7 + d_5*r_13*x6 + d_5*r_23*x7
                # End of temp variables
                this_solution[1] = math.atan2(x0*x4 - x5*x8, x0*x5 + x4*x8)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1_processor()
    # Finish code for explicit solution node 11
    
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
        value_at_2 = ik_out_i[2]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[3]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[4]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[5]  # th_5
        new_ik_i[5] = value_at_5
        ik_out.append(new_ik_i)
    return ik_out


def rokae_SR5_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = rokae_SR5_ik_target_original_to_raw(T_ee)
    ik_output_raw = rokae_SR5_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ee_pose_i = rokae_SR5_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_rokae_SR5():
    theta_in = np.random.random(size=(6, ))
    ee_pose = rokae_SR5_fk(theta_in)
    ik_output = rokae_SR5_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = rokae_SR5_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_rokae_SR5()
