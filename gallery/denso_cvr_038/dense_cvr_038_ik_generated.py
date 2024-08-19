import numpy as np
import copy
import math
from typing import List, NewType

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 35
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 0.165
a_2: float = 0.012
d_1: float = 0.02
d_3: float = 0.1775
d_4: float = -0.0445
d_5: float = 0.045
pre_transform_special_symbol_23: float = 0.18

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
def dense_cvr_038_ik_target_original_to_raw(T_ee: np.ndarray):
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


def dense_cvr_038_ik_target_raw_to_original(T_ee: np.ndarray):
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


def dense_cvr_038_fk(theta_input: np.ndarray):
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
    x14 = x3*x6
    x15 = x2*x5
    x16 = x1*x14 - x1*x15
    x17 = x11*x12 + x13*x16
    x18 = x10*x17
    x19 = math.cos(th_5)
    x20 = 1.0*x11*x13 - 1.0*x12*x16
    x21 = math.sin(th_5)
    x22 = 1.0*x0*x17 - 1.0*x10*x8
    x23 = 1.0*d_1
    x24 = 1.0*a_0
    x25 = x24*x6
    x26 = 1.0*a_2
    x27 = 1.0*d_3
    x28 = 1.0*d_5
    x29 = -x11*x4 - x11*x7
    x30 = x0*x29
    x31 = x11*x14 - x11*x15
    x32 = -x1*x12 + x13*x31
    x33 = x10*x32
    x34 = -1.0*x1*x13 - 1.0*x12*x31
    x35 = 1.0*x0*x32 - 1.0*x10*x29
    x36 = -x14 + x15
    x37 = x0*x36
    x38 = -x4 - x7
    x39 = x13*x38
    x40 = x10*x39
    x41 = 1.0*x12*x38
    x42 = 1.0*x0*x39 - 1.0*x10*x36
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -1.0*x18 - 1.0*x9
    ee_pose[0, 1] = x19*x20 + x21*x22
    ee_pose[0, 2] = x19*x22 - x20*x21
    ee_pose[0, 3] = d_4*x20 + x1*x25 - x11*x23 + x16*x26 + x27*x8 + x28*(-x18 - x9)
    ee_pose[1, 0] = -1.0*x30 - 1.0*x33
    ee_pose[1, 1] = x19*x34 + x21*x35
    ee_pose[1, 2] = x19*x35 - x21*x34
    ee_pose[1, 3] = d_4*x34 + x1*x23 + x11*x25 + x26*x31 + x27*x29 + x28*(-x30 - x33)
    ee_pose[2, 0] = -1.0*x37 - 1.0*x40
    ee_pose[2, 1] = -x19*x41 + x21*x42
    ee_pose[2, 2] = x19*x42 + x21*x41
    ee_pose[2, 3] = -d_4*x41 + 1.0*pre_transform_special_symbol_23 - x2*x24 + x26*x38 + x27*x36 + x28*(-x37 - x40)
    return ee_pose


def dense_cvr_038_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = 1.0*math.sin(th_0)
    x1 = -x0
    x2 = math.cos(th_2)
    x3 = math.sin(th_1)
    x4 = 1.0*math.cos(th_0)
    x5 = x3*x4
    x6 = math.sin(th_2)
    x7 = math.cos(th_1)
    x8 = x4*x7
    x9 = -x2*x5 - x6*x8
    x10 = math.cos(th_3)
    x11 = math.sin(th_3)
    x12 = x2*x8 - x5*x6
    x13 = x0*x10 - x11*x12
    x14 = math.cos(th_4)
    x15 = math.sin(th_4)
    x16 = -x14*x9 - x15*(x0*x11 + x10*x12)
    x17 = x0*x3
    x18 = x0*x7
    x19 = -x17*x2 - x18*x6
    x20 = -x17*x6 + x18*x2
    x21 = -x10*x4 - x11*x20
    x22 = -x14*x19 - x15*(x10*x20 - x11*x4)
    x23 = 1.0*x6
    x24 = 1.0*x2
    x25 = x23*x3 - x24*x7
    x26 = -x23*x7 - x24*x3
    x27 = x11*x26
    x28 = -x10*x15*x26 - x14*x25
    x29 = -1.0*a_0*x3 + pre_transform_special_symbol_23
    x30 = a_2*x26 + d_3*x25 + x29
    x31 = a_0*x18 + d_1*x4
    x32 = a_2*x20 + d_3*x19 + x31
    x33 = -d_4*x27 + x30
    x34 = d_4*x21 + x32
    x35 = d_5*x28 + x33
    x36 = d_5*x22 + x34
    x37 = a_0*x8 - d_1*x0
    x38 = a_2*x12 + d_3*x9 + x37
    x39 = d_4*x13 + x38
    x40 = d_5*x16 + x39
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = x1
    jacobian_output[0, 2] = x1
    jacobian_output[0, 3] = x9
    jacobian_output[0, 4] = x13
    jacobian_output[0, 5] = x16
    jacobian_output[1, 1] = x4
    jacobian_output[1, 2] = x4
    jacobian_output[1, 3] = x19
    jacobian_output[1, 4] = x21
    jacobian_output[1, 5] = x22
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x25
    jacobian_output[2, 4] = -x27
    jacobian_output[2, 5] = x28
    jacobian_output[3, 1] = -pre_transform_special_symbol_23*x4
    jacobian_output[3, 2] = -x29*x4
    jacobian_output[3, 3] = -x19*x30 + x25*x32
    jacobian_output[3, 4] = -x21*x33 - x27*x34
    jacobian_output[3, 5] = -x22*x35 + x28*x36
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x0
    jacobian_output[4, 2] = -x0*x29
    jacobian_output[4, 3] = -x25*x38 + x30*x9
    jacobian_output[4, 4] = x13*x33 + x27*x39
    jacobian_output[4, 5] = x16*x35 - x28*x40
    jacobian_output[5, 2] = x0*x31 + x37*x4
    jacobian_output[5, 3] = x19*x38 - x32*x9
    jacobian_output[5, 4] = -x13*x34 + x21*x39
    jacobian_output[5, 5] = -x16*x36 + x22*x40
    return jacobian_output


def dense_cvr_038_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = 1.0*math.sin(th_0)
    x1 = -x0
    x2 = math.cos(th_2)
    x3 = math.sin(th_1)
    x4 = 1.0*math.cos(th_0)
    x5 = x3*x4
    x6 = math.sin(th_2)
    x7 = math.cos(th_1)
    x8 = x4*x7
    x9 = -x2*x5 - x6*x8
    x10 = math.cos(th_3)
    x11 = math.sin(th_3)
    x12 = x2*x8 - x5*x6
    x13 = math.cos(th_4)
    x14 = math.sin(th_4)
    x15 = x0*x3
    x16 = x0*x7
    x17 = -x15*x2 - x16*x6
    x18 = -x15*x6 + x16*x2
    x19 = 1.0*x6
    x20 = 1.0*x2
    x21 = x19*x3 - x20*x7
    x22 = -x19*x7 - x20*x3
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 1] = x1
    jacobian_output[0, 2] = x1
    jacobian_output[0, 3] = x9
    jacobian_output[0, 4] = x0*x10 - x11*x12
    jacobian_output[0, 5] = -x13*x9 - x14*(x0*x11 + x10*x12)
    jacobian_output[1, 1] = x4
    jacobian_output[1, 2] = x4
    jacobian_output[1, 3] = x17
    jacobian_output[1, 4] = -x10*x4 - x11*x18
    jacobian_output[1, 5] = -x13*x17 - x14*(x10*x18 - x11*x4)
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x21
    jacobian_output[2, 4] = -x11*x22
    jacobian_output[2, 5] = -x10*x14*x22 - x13*x21
    return jacobian_output


def dense_cvr_038_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x6 = -a_0*x5 + pre_transform_special_symbol_23
    x7 = math.sin(th_2)
    x8 = x5*x7
    x9 = math.cos(th_2)
    x10 = math.cos(th_1)
    x11 = 1.0*x10
    x12 = x11*x9
    x13 = -x12 + x8
    x14 = math.sin(th_0)
    x15 = x5*x9
    x16 = x11*x7
    x17 = -x14*x15 - x14*x16
    x18 = -x15 - x16
    x19 = a_2*x18 + d_3*x13 + x6
    x20 = x12*x14 - x14*x8
    x21 = a_0*x11*x14 + d_1*x2
    x22 = a_2*x20 + d_3*x17 + x21
    x23 = math.sin(th_3)
    x24 = x18*x23
    x25 = math.cos(th_3)
    x26 = -x2*x25 - x20*x23
    x27 = -d_4*x24 + x19
    x28 = d_4*x26 + x22
    x29 = math.cos(th_4)
    x30 = math.sin(th_4)
    x31 = -x13*x29 - x18*x25*x30
    x32 = -x17*x29 - x30*(-x2*x23 + x20*x25)
    x33 = d_5*x31 + x27
    x34 = d_5*x32 + x28
    x35 = 1.0*p_on_ee_x
    x36 = 1.0*x14
    x37 = p_on_ee_z*x36
    x38 = x2*x4
    x39 = x10*x2
    x40 = -x38*x9 - x39*x7
    x41 = -x38*x7 + x39*x9
    x42 = a_0*x39 - d_1*x36
    x43 = a_2*x41 + d_3*x40 + x42
    x44 = -x23*x41 + x25*x36
    x45 = d_4*x44 + x43
    x46 = -x29*x40 - x30*(x23*x36 + x25*x41)
    x47 = d_5*x46 + x45
    x48 = -x0*x14 - x1*x35
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = -x2*x6 + x3
    jacobian_output[0, 3] = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x22 - x17*x19
    jacobian_output[0, 4] = p_on_ee_y*x24 + p_on_ee_z*x26 - x24*x28 - x26*x27
    jacobian_output[0, 5] = -p_on_ee_y*x31 + p_on_ee_z*x32 + x31*x34 - x32*x33
    jacobian_output[1, 0] = x35
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x36 + x37
    jacobian_output[1, 2] = -x36*x6 + x37
    jacobian_output[1, 3] = p_on_ee_x*x13 - p_on_ee_z*x40 - x13*x43 + x19*x40
    jacobian_output[1, 4] = -p_on_ee_x*x24 - p_on_ee_z*x44 + x24*x45 + x27*x44
    jacobian_output[1, 5] = p_on_ee_x*x31 - p_on_ee_z*x46 - x31*x47 + x33*x46
    jacobian_output[2, 1] = x48
    jacobian_output[2, 2] = x2*x42 + x21*x36 + x48
    jacobian_output[2, 3] = -p_on_ee_x*x17 + p_on_ee_y*x40 + x17*x43 - x22*x40
    jacobian_output[2, 4] = -p_on_ee_x*x26 + p_on_ee_y*x44 + x26*x45 - x28*x44
    jacobian_output[2, 5] = -p_on_ee_x*x32 + p_on_ee_y*x46 + x32*x47 - x34*x46
    return jacobian_output


def dense_cvr_038_ik_solve_raw(T_ee: np.ndarray):
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
    for i in range(35):
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
        R_l[0, 6] = d_1
        R_l[0, 7] = -a_2
        R_l[1, 2] = -a_0
        R_l[1, 6] = -a_2
        R_l[1, 7] = -d_1
        R_l[2, 4] = a_0
        R_l[3, 6] = -1
        R_l[4, 7] = 1
        R_l[5, 5] = 2*a_0*a_2
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
        x4 = d_3*r_23
        x5 = -x4
        x6 = d_4*r_22
        x7 = d_1 - x6
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
        x30 = d_1 + x6
        x31 = x28 + x30
        x32 = d_3*x1
        x33 = -x32
        x34 = d_3*x3
        x35 = x4 + x7
        x36 = Py*r_22
        x37 = x18 + x24 + x36
        x38 = R_l_inv_51*a_0
        x39 = x37*x38
        x40 = R_l_inv_52*a_0
        x41 = d_3*x40
        x42 = a_0*r_22
        x43 = R_l_inv_54*x42
        x44 = d_3**2
        x45 = d_4**2
        x46 = d_5**2
        x47 = a_0**2
        x48 = a_2**2
        x49 = -d_1**2
        x50 = 2*d_5
        x51 = Py*x15
        x52 = Py*x1
        x53 = 2*x18
        x54 = Py*r_23
        x55 = 2*x20
        x56 = 2*x16
        x57 = 2*x36
        x58 = 2*x54
        x59 = Px**2
        x60 = r_11**2
        x61 = x59*x60
        x62 = r_12**2
        x63 = x59*x62
        x64 = r_13**2
        x65 = x59*x64
        x66 = Py**2
        x67 = x66*x9
        x68 = x11*x66
        x69 = x13*x66
        x70 = Pz**2
        x71 = r_31**2*x70
        x72 = r_32**2*x70
        x73 = r_33**2*x70
        x74 = x16*x52 - x20*x50 + x22*x52 + x22*x56 + x24*x53 + x24*x57 - x26*x50 + x26*x55 + x26*x58 + x36*x53 + x44 + x45 + x46 - x47 - x48 + x49 - 2*x51 + x54*x55 + x61 + x63 + x65 + x67 + x68 + x69 + x71 + x72 + x73
        x75 = R_l_inv_55*a_0
        x76 = x74*x75
        x77 = -d_4*x38
        x78 = d_5*r_21
        x79 = r_23*x16
        x80 = r_23*x22
        x81 = r_21*x20
        x82 = r_21*x26
        x83 = x78 + x79 + x80 - x81 - x82
        x84 = R_l_inv_57*a_0
        x85 = x83*x84
        x86 = -x85
        x87 = R_l_inv_56*d_3*x42
        x88 = -x87
        x89 = 2*d_4
        x90 = x37*x75
        x91 = x89*x90
        x92 = -x91
        x93 = a_2 + x39 + x41 + x43 + x76 + x77 + x86 + x88 + x92
        x94 = d_4*r_21
        x95 = -x94
        x96 = Py*r_21
        x97 = x16 + x22 + x96
        x98 = R_l_inv_50*a_0
        x99 = x97*x98
        x100 = -x99
        x101 = a_0*r_21
        x102 = R_l_inv_53*x101
        x103 = -x102
        x104 = d_5*r_22
        x105 = r_23*x18
        x106 = r_23*x24
        x107 = r_22*x20
        x108 = r_22*x26
        x109 = x104 + x105 + x106 - x107 - x108
        x110 = R_l_inv_56*a_0
        x111 = x109*x110
        x112 = -x111
        x113 = d_3*x101
        x114 = R_l_inv_57*x113
        x115 = -x114
        x116 = x100 + x103 + x112 + x115 + x95
        x117 = r_21*x18
        x118 = r_21*x24
        x119 = r_22*x16
        x120 = -x119
        x121 = r_22*x22
        x122 = -x121
        x123 = d_4*r_23
        x124 = x110*x123
        x125 = -d_5 + x20 + x26 + x54
        x126 = x125*x40
        x127 = 2*d_3
        x128 = x125*x127
        x129 = x128*x75
        x130 = -x126 - x129
        x131 = x117 + x118 + x120 + x122 + x124 + x130
        x132 = 2*x98
        x133 = x132*x37
        x134 = 2*x97
        x135 = -x134*x38
        x136 = 4*d_4
        x137 = x75*x97
        x138 = x136*x137
        x139 = -x133 + x135 + x138
        x140 = R_l_inv_54*a_0
        x141 = x1*x140
        x142 = 2*x84
        x143 = x109*x142
        x144 = x110*x32
        x145 = -x141 - x143 + x144
        x146 = 2*x6
        x147 = 2*R_l_inv_53
        x148 = x147*x42
        x149 = 2*x110
        x150 = x149*x83
        x151 = x127*x42
        x152 = R_l_inv_57*x151
        x153 = -x146 - x148 + x150 - x152
        x154 = x102 + x111 + x114 + x94 + x99
        x155 = a_2 - x39 + x41 + x76 + x77 + x91
        x156 = -x43 + x85 + x87
        x157 = x155 + x156
        x158 = x134*x40
        x159 = x125*x132
        x160 = 4*d_3
        x161 = x137*x160
        x162 = -x158 + x159 - x161
        x163 = 2*x123
        x164 = a_0*r_23
        x165 = x147*x164
        x166 = -x117 - x118 + x119 + x121
        x167 = x149*x166
        x168 = x142*x4
        x169 = x163 + x165 + x167 + x168
        x170 = 2*x104
        x171 = 2*x105
        x172 = 2*x106
        x173 = 2*x107
        x174 = 2*x108
        x175 = x110*x29
        x176 = -x170 - x171 - x172 + x173 + x174 + x175
        x177 = 4*x78
        x178 = 4*x81
        x179 = 4*x82
        x180 = 4*x79
        x181 = 4*x80
        x182 = d_4*x3
        x183 = x110*x182
        x184 = 4*x37
        x185 = 8*d_3
        x186 = -x184*x40 - x185*x90
        x187 = x158 + x159 + x161
        x188 = x170 + x171 + x172 - x173 - x174 - x175
        x189 = -x124 + x166
        x190 = x126 + x129
        x191 = x189 + x190
        x192 = x133 + x135 + x138
        x193 = x146 + x148 - x150 + x152
        x194 = 2*a_0
        x195 = a_2*x194
        x196 = d_1*x194
        x197 = a_2*d_1
        x198 = 2*x197
        x199 = x47 + x48 + x49
        x200 = R_l_inv_22*x195 + R_l_inv_32*x196 + R_l_inv_62*x199 + R_l_inv_72*x198
        x201 = d_3*x200
        x202 = R_l_inv_21*x195 + R_l_inv_31*x196 + R_l_inv_61*x199 + R_l_inv_71*x198
        x203 = x202*x37
        x204 = R_l_inv_25*x195 + R_l_inv_35*x196 + R_l_inv_65*x199 + R_l_inv_75*x198
        x205 = x204*x74
        x206 = -d_4*x202
        x207 = R_l_inv_20*x195 + R_l_inv_30*x196 + R_l_inv_60*x199 + R_l_inv_70*x198
        x208 = x207*x97
        x209 = -x208
        x210 = x125*x200
        x211 = -x210
        x212 = x128*x204
        x213 = -x212
        x214 = x204*x37
        x215 = x214*x89
        x216 = -x215
        x217 = x201 + x203 + x205 + x206 + x209 + x211 + x213 + x216
        x218 = R_l_inv_66*x199
        x219 = R_l_inv_26*x195 + R_l_inv_36*x196 + R_l_inv_76*x198 + x218
        x220 = x123*x219
        x221 = x146*x16
        x222 = x146*x22
        x223 = x18*x29
        x224 = x24*x29
        x225 = x220 - x221 - x222 + x223 + x224
        x226 = R_l_inv_24*x195 + R_l_inv_34*x196 + R_l_inv_64*x199 + R_l_inv_74*x198
        x227 = r_22*x226
        x228 = R_l_inv_67*x199
        x229 = R_l_inv_27*x195 + R_l_inv_37*x196 + R_l_inv_77*x198 + x228
        x230 = x229*x83
        x231 = d_3*r_22
        x232 = x219*x231
        x233 = d_5*x32
        x234 = x4*x56
        x235 = 2*x22
        x236 = x235*x4
        x237 = x20*x32
        x238 = x26*x32
        x239 = x227 - x230 - x232 - x233 - x234 - x236 + x237 + x238
        x240 = x225 + x239
        x241 = r_21**3*x66
        x242 = r_21*x44
        x243 = r_21*x45
        x244 = r_21*x46
        x245 = R_l_inv_23*x195 + R_l_inv_33*x196 + R_l_inv_63*x199 + R_l_inv_73*x198
        x246 = r_21*x245
        x247 = x109*x219
        x248 = r_21*x61
        x249 = r_21*x68
        x250 = r_21*x69
        x251 = r_21*x71
        x252 = d_3*r_21
        x253 = x229*x252
        x254 = r_21*x63
        x255 = r_21*x65
        x256 = r_21*x72
        x257 = r_21*x73
        x258 = x15*x56
        x259 = x15*x235
        x260 = x10*x56
        x261 = x12*x56
        x262 = x14*x56
        x263 = d_5*x1
        x264 = x20*x263
        x265 = x10*x235
        x266 = x12*x235
        x267 = x14*x235
        x268 = x26*x263
        x269 = 2*r_11
        x270 = r_12*x59
        x271 = r_22*x270
        x272 = x269*x271
        x273 = 2*r_13
        x274 = r_23*x59
        x275 = r_11*x273*x274
        x276 = 2*r_31
        x277 = r_32*x70
        x278 = r_22*x277
        x279 = x276*x278
        x280 = r_23*r_33
        x281 = x280*x70
        x282 = x276*x281
        x283 = x1*x18
        x284 = x24*x283
        x285 = x20*x26
        x286 = x1*x285
        x287 = x16*x22
        x288 = x1*x287
        x289 = x25*x56
        x290 = x27*x56
        x291 = x19*x235
        x292 = x21*x235
        x293 = x241 - x242 - x243 - x244 - x246 - x247 + x248 + x249 + x250 + x251 - x253 - x254 - x255 - x256 - x257 - x258 - x259 + x260 + x261 + x262 + x264 + x265 + x266 + x267 + x268 + x272 + x275 + x279 + x282 - x284 - x286 + x288 + x289 + x290 + x291 + x292
        x294 = x235 + x52 + x56
        x295 = -x202*x294
        x296 = 2*x24
        x297 = x296 + x53 + x57
        x298 = x207*x297
        x299 = x204*x97
        x300 = x136*x299
        x301 = x295 - x298 + x300
        x302 = 4*a_0
        x303 = a_2*x302
        x304 = d_1*x302
        x305 = 4*x197
        x306 = R_l_inv_27*x303 + R_l_inv_37*x304 + R_l_inv_77*x305 + 2*x228
        x307 = x109*x306
        x308 = x1*x226
        x309 = d_5*x34
        x310 = x219*x32
        x311 = 4*x18
        x312 = x311*x4
        x313 = 4*x24
        x314 = x313*x4
        x315 = x20*x34
        x316 = x26*x34
        x317 = -x307 - x308 - x309 + x310 - x312 - x314 + x315 + x316
        x318 = R_l_inv_26*x303 + R_l_inv_36*x304 + R_l_inv_76*x305 + 2*x218
        x319 = x318*x83
        x320 = r_22*x44
        x321 = 2*x320
        x322 = r_22*x45
        x323 = 2*x322
        x324 = r_22*x46
        x325 = 2*x324
        x326 = 2*x245
        x327 = r_22*x326
        x328 = r_22**3*x66
        x329 = 2*x328
        x330 = r_22*x127
        x331 = x229*x330
        x332 = r_22*x61
        x333 = 2*x332
        x334 = r_22*x65
        x335 = 2*x334
        x336 = r_22*x71
        x337 = 2*x336
        x338 = r_22*x73
        x339 = 2*x338
        x340 = r_22*x63
        x341 = 2*x340
        x342 = r_22*x67
        x343 = 2*x342
        x344 = r_22*x69
        x345 = 2*x344
        x346 = r_22*x72
        x347 = 2*x346
        x348 = x15*x311
        x349 = x15*x313
        x350 = x10*x311
        x351 = x12*x311
        x352 = x14*x311
        x353 = d_5*x3
        x354 = x20*x353
        x355 = x10*x313
        x356 = x12*x313
        x357 = x14*x313
        x358 = x26*x353
        x359 = 4*r_11
        x360 = r_21*x359
        x361 = x270*x360
        x362 = 4*r_12
        x363 = r_13*x274
        x364 = x362*x363
        x365 = 4*x277
        x366 = r_21*r_31
        x367 = x365*x366
        x368 = x280*x365
        x369 = x287*x3
        x370 = x285*x3
        x371 = x17*x313
        x372 = x23*x311
        x373 = x18*x3
        x374 = x24*x373
        x375 = x27*x311
        x376 = x21*x313
        x377 = x319 - x321 - x323 - x325 - x327 + x329 - x331 - x333 - x335 - x337 - x339 + x341 + x343 + x345 + x347 - x348 - x349 + x350 + x351 + x352 + x354 + x355 + x356 + x357 + x358 + x361 + x364 + x367 + x368 - x369 - x370 + x371 + x372 + x374 + x375 + x376
        x378 = -x203
        x379 = x201 + x205 + x206 + x208 + x211 + x213 + x215 + x378
        x380 = -x227 + x230 + x232 + x233 + x234 + x236 - x237 - x238
        x381 = x225 + x380
        x382 = -x241 + x242 + x243 + x244 + x246 + x247 - x248 - x249 - x250 - x251 + x253 + x254 + x255 + x256 + x257 + x258 + x259 - x260 - x261 - x262 - x264 - x265 - x266 - x267 - x268 - x272 - x275 - x279 - x282 + x284 + x286 - x288 - x289 - x290 - x291 - x292
        x383 = 2*x26
        x384 = x383 - x50 + x55 + x58
        x385 = x207*x384
        x386 = x200*x294
        x387 = x160*x299
        x388 = x385 - x386 - x387
        x389 = d_5*x182
        x390 = x219*x29
        x391 = x105*x136
        x392 = x106*x136
        x393 = x182*x20
        x394 = x182*x26
        x395 = -x389 + x390 - x391 - x392 + x393 + x394
        x396 = x1*x24
        x397 = 2*x119 + 2*x121 - x283 - x396
        x398 = x219*x397
        x399 = r_23*x46
        x400 = 2*x399
        x401 = r_23**3*x66
        x402 = 2*x401
        x403 = r_23*x44
        x404 = 2*x403
        x405 = r_23*x45
        x406 = 2*x405
        x407 = r_23*x326
        x408 = r_23*x65
        x409 = 2*x408
        x410 = r_23*x67
        x411 = 2*x410
        x412 = r_23*x68
        x413 = 2*x412
        x414 = r_23*x73
        x415 = 2*x414
        x416 = 2*x4
        x417 = x229*x416
        x418 = r_23*x61
        x419 = 2*x418
        x420 = r_23*x63
        x421 = 2*x420
        x422 = r_23*x71
        x423 = 2*x422
        x424 = r_23*x72
        x425 = 2*x424
        x426 = 4*d_5
        x427 = x10*x426
        x428 = x12*x426
        x429 = x14*x426
        x430 = 4*x20
        x431 = x10*x430
        x432 = x12*x430
        x433 = x14*x430
        x434 = 4*x26
        x435 = x10*x434
        x436 = x12*x434
        x437 = x14*x434
        x438 = r_13*x59
        x439 = x360*x438
        x440 = x270*x3
        x441 = r_13*x440
        x442 = r_33*x70
        x443 = 4*x366*x442
        x444 = x277*x3
        x445 = r_33*x444
        x446 = x16*x177
        x447 = x18*x353
        x448 = x15*x430
        x449 = x177*x22
        x450 = x24*x353
        x451 = x15*x434
        x452 = x17*x434
        x453 = x26*x373
        x454 = x23*x430
        x455 = x24*x3
        x456 = x20*x455
        x457 = x21*x434
        x458 = x180*x22
        x459 = x105*x313
        x460 = x398 - x400 - x402 + x404 + x406 + x407 - x409 - x411 - x413 - x415 + x417 + x419 + x421 + x423 + x425 + x427 + x428 + x429 - x431 - x432 - x433 - x435 - x436 - x437 - x439 - x441 - x443 - x445 + x446 + x447 + x448 + x449 + x450 + x451 - x452 - x453 - x454 - x456 - x457 + x458 + x459
        x461 = x182*x219
        x462 = 8*d_5
        x463 = x462*x94
        x464 = 8*x94
        x465 = x20*x464
        x466 = x26*x464
        x467 = 8*d_4
        x468 = x467*x79
        x469 = x467*x80
        x470 = Py*x3
        x471 = x311 + x313 + x470
        x472 = -x185*x214 - x200*x471
        x473 = x385 + x386 + x387
        x474 = x389 - x390 + x391 + x392 - x393 - x394
        x475 = -x220
        x476 = -x223
        x477 = -x224
        x478 = x221 + x222 + x382 + x475 + x476 + x477
        x479 = x201 + x205 + x206 + x210 + x212
        x480 = x203 + x208 + x216 + x479
        x481 = x295 + x298 + x300
        x482 = -x319 + x321 + x323 + x325 + x327 - x329 + x331 + x333 + x335 + x337 + x339 - x341 - x343 - x345 - x347 + x348 + x349 - x350 - x351 - x352 - x354 - x355 - x356 - x357 - x358 - x361 - x364 - x367 - x368 + x369 + x370 - x371 - x372 - x374 - x375 - x376
        x483 = x209 + x215 + x378 + x479
        x484 = x221 + x222 + x293 + x475 + x476 + x477
        x485 = R_l_inv_22*x196 - R_l_inv_32*x195 + R_l_inv_62*x198 - R_l_inv_72*x199
        x486 = d_3*x485
        x487 = R_l_inv_21*x196 - R_l_inv_31*x195 + R_l_inv_61*x198 - R_l_inv_71*x199
        x488 = x37*x487
        x489 = R_l_inv_25*x196 - R_l_inv_35*x195 + R_l_inv_65*x198 - R_l_inv_75*x199
        x490 = x489*x74
        x491 = -d_4*x487
        x492 = R_l_inv_20*x196 - R_l_inv_30*x195 + R_l_inv_60*x198 - R_l_inv_70*x199
        x493 = x492*x97
        x494 = -x493
        x495 = x125*x485
        x496 = -x495
        x497 = x128*x489
        x498 = -x497
        x499 = x37*x489
        x500 = x499*x89
        x501 = -x500
        x502 = x486 + x488 + x490 + x491 + x494 + x496 + x498 + x501
        x503 = R_l_inv_23*x196 - R_l_inv_33*x195 + R_l_inv_63*x198 - R_l_inv_73*x199
        x504 = r_21*x503
        x505 = -x504
        x506 = R_l_inv_76*x199
        x507 = R_l_inv_26*x196 - R_l_inv_36*x195 + R_l_inv_66*x198 - x506
        x508 = x109*x507
        x509 = -x508
        x510 = x123*x507
        x511 = R_l_inv_77*x199
        x512 = R_l_inv_27*x196 - R_l_inv_37*x195 + R_l_inv_67*x198 - x511
        x513 = x252*x512
        x514 = -x513
        x515 = x4*x89
        x516 = -x515
        x517 = x104*x127
        x518 = x107*x127
        x519 = -x518
        x520 = x108*x127
        x521 = -x520
        x522 = x4*x53
        x523 = x296*x4
        x524 = x505 + x509 + x510 + x514 + x516 + x517 + x519 + x521 + x522 + x523
        x525 = x15*x89
        x526 = x10*x89
        x527 = x12*x89
        x528 = x14*x89
        x529 = x16*x29
        x530 = x146*x18
        x531 = x21*x89
        x532 = x22*x29
        x533 = x146*x24
        x534 = x27*x89
        x535 = -x525 + x526 + x527 + x528 + x529 + x530 + x531 + x532 + x533 + x534
        x536 = R_l_inv_24*x196 - R_l_inv_34*x195 + R_l_inv_64*x198 - R_l_inv_74*x199
        x537 = r_22*x536
        x538 = x512*x83
        x539 = x231*x507
        x540 = x10*x53
        x541 = x12*x53
        x542 = x14*x53
        x543 = x170*x20
        x544 = x10*x296
        x545 = x12*x296
        x546 = x14*x296
        x547 = x170*x26
        x548 = r_11*x1
        x549 = x270*x548
        x550 = r_23*x270*x273
        x551 = r_31*x1
        x552 = x277*x551
        x553 = x277*x280
        x554 = 2*x553
        x555 = x15*x53
        x556 = x15*x296
        x557 = x16*x396
        x558 = x22*x283
        x559 = x19*x296
        x560 = x27*x53
        x561 = x21*x296
        x562 = x119*x235
        x563 = x107*x383
        x564 = x320 - x322 + x324 - x328 + x332 + x334 + x336 + x338 - x340 - x342 - x344 - x346 + x537 - x538 - x539 - x540 - x541 - x542 - x543 - x544 - x545 - x546 - x547 - x549 - x550 - x552 - x554 + x555 + x556 - x557 - x558 - x559 - x560 - x561 + x562 + x563
        x565 = x535 + x564
        x566 = -x294*x487
        x567 = x297*x492
        x568 = x489*x97
        x569 = x136*x568
        x570 = x566 - x567 + x569
        x571 = R_l_inv_27*x304 - R_l_inv_37*x303 + R_l_inv_67*x305 - 2*x511
        x572 = x109*x571
        x573 = x1*x536
        x574 = x32*x507
        x575 = x1*x46
        x576 = 2*x241
        x577 = x1*x63
        x578 = x1*x65
        x579 = x1*x72
        x580 = x1*x73
        x581 = x1*x61
        x582 = x1*x68
        x583 = x1*x69
        x584 = x1*x71
        x585 = 4*x16
        x586 = x15*x585
        x587 = 4*x22
        x588 = x15*x587
        x589 = x10*x585
        x590 = x12*x585
        x591 = x14*x585
        x592 = x177*x20
        x593 = x10*x587
        x594 = x12*x587
        x595 = x14*x587
        x596 = x177*x26
        x597 = r_11*x440
        x598 = x359*x363
        x599 = r_31*x444
        x600 = 4*r_31*x281
        x601 = x117*x313
        x602 = x178*x26
        x603 = x17*x587
        x604 = x16*x455
        x605 = x27*x585
        x606 = x22*x373
        x607 = x21*x587
        x608 = -x575 + x576 - x577 - x578 - x579 - x580 + x581 + x582 + x583 + x584 - x586 - x588 + x589 + x590 + x591 + x592 + x593 + x594 + x595 + x596 + x597 + x598 + x599 + x600 - x601 - x602 + x603 + x604 + x605 + x606 + x607
        x609 = x1*x44
        x610 = x1*x45
        x611 = -x609 + x610
        x612 = -x572 - x573 + x574 + x608 + x611
        x613 = R_l_inv_26*x304 - R_l_inv_36*x303 + R_l_inv_66*x305 - 2*x506
        x614 = x613*x83
        x615 = 2*x503
        x616 = r_22*x615
        x617 = d_3*x177
        x618 = x330*x512
        x619 = x4*x585
        x620 = x4*x587
        x621 = d_3*x178
        x622 = d_3*x179
        x623 = x614 - x616 - x617 - x618 - x619 - x620 + x621 + x622
        x624 = -x488
        x625 = x486 + x490 + x491 + x493 + x496 + x498 + x500 + x624
        x626 = x504 + x508 + x513 - x517 + x518 + x520 - x522 - x523
        x627 = x510 + x516 + x626
        x628 = -x320 + x322 - x324 + x328 - x332 - x334 - x336 - x338 + x340 + x342 + x344 + x346 - x537 + x538 + x539 + x540 + x541 + x542 + x543 + x544 + x545 + x546 + x547 + x549 + x550 + x552 + x554 - x555 - x556 + x557 + x558 + x559 + x560 + x561 - x562 - x563
        x629 = x535 + x628
        x630 = x384*x492
        x631 = x294*x485
        x632 = x160*x568
        x633 = x630 - x631 - x632
        x634 = x160*x94
        x635 = -x634
        x636 = x29*x507
        x637 = x635 + x636
        x638 = x397*x507
        x639 = r_23*x615
        x640 = x416*x512
        x641 = x16*x34
        x642 = x22*x34
        x643 = x117*x160
        x644 = x118*x160
        x645 = x638 + x639 + x640 - x641 - x642 + x643 + x644
        x646 = x185*x6
        x647 = x182*x507
        x648 = -x185*x499 - x471*x485
        x649 = x634 - x636
        x650 = x630 + x631 + x632
        x651 = -x510 + x515
        x652 = x626 + x651
        x653 = x486 + x490 + x491 + x495 + x497
        x654 = x488 + x493 + x501 + x653
        x655 = x566 + x567 + x569
        x656 = -x614 + x616 + x617 + x618 + x619 + x620 - x621 - x622
        x657 = x494 + x500 + x624 + x653
        x658 = x505 + x509 + x514 + x517 + x519 + x521 + x522 + x523 + x651
        x659 = R_l_inv_41*x196*x37
        x660 = -x659
        x661 = a_0*d_1
        x662 = -R_l_inv_42*x127*x661
        x663 = -R_l_inv_45*x196*x74
        x664 = R_l_inv_40*x661
        x665 = x134*x664
        x666 = R_l_inv_41*x661*x89
        x667 = R_l_inv_42*x125*x196
        x668 = x125*x304
        x669 = R_l_inv_45*d_3
        x670 = x668*x669
        x671 = R_l_inv_45*x661
        x672 = x136*x37*x671
        x673 = x660 + x662 + x663 + x665 + x666 + x667 + x670 + x672
        x674 = x127*x6
        x675 = x50*x6
        x676 = x146*x20
        x677 = -x676
        x678 = x146*x26
        x679 = -x678
        x680 = R_l_inv_44*d_1
        x681 = 2*x42*x680
        x682 = -x681
        x683 = x105*x89
        x684 = x106*x89
        x685 = R_l_inv_43*x661
        x686 = x1*x685
        x687 = R_l_inv_46*x109*x196
        x688 = R_l_inv_47*x196*x83
        x689 = R_l_inv_46*d_1*x151
        x690 = R_l_inv_47*x661
        x691 = x32*x690
        x692 = x674 + x675 + x677 + x679 + x682 + x683 + x684 + x686 + x687 + x688 + x689 + x691
        x693 = x10*x127
        x694 = x12*x127
        x695 = x127*x14
        x696 = x4*x50
        x697 = x16*x32
        x698 = x127*x19
        x699 = x4*x55
        x700 = x22*x32
        x701 = x127*x25
        x702 = x383*x4
        x703 = -x693 - x694 - x695 + x696 - x697 - x698 - x699 - x700 - x701 - x702
        x704 = x10*x50
        x705 = x12*x50
        x706 = x14*x50
        x707 = x16*x263
        x708 = x170*x18
        x709 = x15*x55
        x710 = x22*x263
        x711 = x170*x24
        x712 = x15*x383
        x713 = x10*x55
        x714 = x12*x55
        x715 = x14*x55
        x716 = x10*x383
        x717 = x12*x383
        x718 = x14*x383
        x719 = x438*x548
        x720 = x271*x273
        x721 = x442*x551
        x722 = 2*r_33*x278
        x723 = x235*x79
        x724 = x105*x296
        x725 = R_l_inv_46*x661
        x726 = x163*x725
        x727 = x1*x16*x26
        x728 = x19*x383
        x729 = x1*x20*x22
        x730 = x25*x55
        x731 = x21*x383
        x732 = x399 + x401 + x403 - x405 + x408 + x410 + x412 + x414 - x418 - x420 - x422 - x424 - x704 - x705 - x706 - x707 - x708 - x709 - x710 - x711 - x712 + x713 + x714 + x715 + x716 + x717 + x718 + x719 + x720 + x721 + x722 - x723 - x724 - x726 + x727 + x728 + x729 + x730 + x731
        x733 = x703 + x732
        x734 = x184*x664
        x735 = x304*x97
        x736 = R_l_inv_41*x735
        x737 = x671*x97
        x738 = -x467*x737
        x739 = x734 + x736 + x738
        x740 = 4*x101*x680
        x741 = R_l_inv_47*x304
        x742 = x109*x741
        x743 = 4*d_1
        x744 = R_l_inv_46*x113*x743
        x745 = x635 + x740 + x742 - x744
        x746 = x426*x94
        x747 = x136*x79
        x748 = x136*x80
        x749 = R_l_inv_46*x304
        x750 = x749*x83
        x751 = x430*x94
        x752 = x434*x94
        x753 = x3*x685
        x754 = x34*x690
        x755 = -x746 - x747 - x748 - x750 + x751 + x752 + x753 + x754
        x756 = -x665
        x757 = -x672
        x758 = x659 + x662 + x663 + x666 + x667 + x670 + x756 + x757
        x759 = -x674 + x681 - x688 - x689
        x760 = -x675 + x676 + x678 - x683 - x684 - x686 - x687 - x691
        x761 = x759 + x760
        x762 = -R_l_inv_40*x668
        x763 = R_l_inv_42*x735
        x764 = x185*x737
        x765 = x762 + x763 + x764
        x766 = x16*x182
        x767 = x182*x22
        x768 = R_l_inv_43*x164*x743
        x769 = x166*x749
        x770 = x311*x94
        x771 = x313*x94
        x772 = x4*x741
        x773 = -x766 - x767 - x768 - x769 + x770 + x771 - x772
        x774 = x749*x94
        x775 = x609 - x610
        x776 = x608 - x774 + x775
        x777 = x3*x45
        x778 = x3*x46
        x779 = x3*x44
        x780 = 4*x328
        x781 = x3*x61
        x782 = x3*x65
        x783 = x3*x71
        x784 = x3*x73
        x785 = x3*x63
        x786 = x3*x67
        x787 = x3*x69
        x788 = x3*x72
        x789 = 8*x15
        x790 = x18*x789
        x791 = x24*x789
        x792 = 8*x18
        x793 = x10*x792
        x794 = x12*x792
        x795 = x14*x792
        x796 = 8*x104
        x797 = x20*x796
        x798 = 8*x24
        x799 = x10*x798
        x800 = x12*x798
        x801 = x14*x798
        x802 = x26*x796
        x803 = 8*r_12
        x804 = r_11*r_21
        x805 = x59*x803*x804
        x806 = x363*x803
        x807 = 8*x277
        x808 = x366*x807
        x809 = 8*x553
        x810 = 8*x22
        x811 = x119*x810
        x812 = 8*x26
        x813 = x107*x812
        x814 = 8*a_0
        x815 = d_1*x814
        x816 = R_l_inv_46*x815
        x817 = x6*x816
        x818 = x17*x798
        x819 = x23*x792
        x820 = x19*x798
        x821 = x27*x792
        x822 = x21*x798
        x823 = R_l_inv_42*x37*x815 + 16*x37*x661*x669
        x824 = x762 - x763 - x764
        x825 = x575 - x576 + x577 + x578 + x579 + x580 - x581 - x582 - x583 - x584 + x586 + x588 - x589 - x590 - x591 - x592 - x593 - x594 - x595 - x596 - x597 - x598 - x599 - x600 + x601 + x602 - x603 - x604 - x605 - x606 - x607
        x826 = x611 + x774 + x825
        x827 = -x667
        x828 = -x670
        x829 = x660 + x662 + x663 + x666 + x672 + x756 + x827 + x828
        x830 = x674 + x682 + x688 + x689 + x760
        x831 = -x399 - x401 - x403 + x405 - x408 - x410 - x412 - x414 + x418 + x420 + x422 + x424 + x704 + x705 + x706 + x707 + x708 + x709 + x710 + x711 + x712 - x713 - x714 - x715 - x716 - x717 - x718 - x719 - x720 - x721 - x722 + x723 + x724 + x726 - x727 - x728 - x729 - x730 - x731
        x832 = x703 + x831
        x833 = -x734 + x736 + x738
        x834 = x746 + x747 + x748 + x750 - x751 - x752 - x753 - x754
        x835 = x659 + x662 + x663 + x665 + x666 + x757 + x827 + x828
        x836 = x675 + x677 + x679 + x683 + x684 + x686 + x687 + x691 + x759
        x837 = -x273
        x838 = r_12*x89
        x839 = -x838
        x840 = d_3*x273
        x841 = 2*Px
        x842 = r_11*x235 + r_11*x52 + r_12*x296 + r_12*x57 - r_13*x50 + x26*x273 + x273*x54 + x60*x841 + x62*x841 + x64*x841
        x843 = -x840 + x842
        x844 = d_4*x359
        x845 = d_3*x359
        x846 = d_3*x803
        x847 = x840 + x842
        x848 = d_5*r_11
        x849 = r_13*x96
        x850 = r_13*x22
        x851 = r_11*x54
        x852 = r_11*x26
        x853 = x848 + x849 + x850 - x851 - x852
        x854 = x142*x853
        x855 = 2*r_12
        x856 = x140*x855
        x857 = r_12*x127
        x858 = x110*x857
        x859 = -x854 + x856 - x858
        x860 = r_11*x89
        x861 = R_l_inv_53*a_0
        x862 = x269*x861
        x863 = d_5*r_12
        x864 = r_13*x36
        x865 = r_13*x24
        x866 = r_12*x54
        x867 = r_12*x26
        x868 = x863 + x864 + x865 - x866 - x867
        x869 = x149*x868
        x870 = r_11*x127
        x871 = x84*x870
        x872 = -x860 - x862 - x869 - x871
        x873 = r_12*x52
        x874 = r_12*x235
        x875 = r_11*x57
        x876 = r_11*x296
        x877 = d_4*x273
        x878 = x110*x877
        x879 = -x873 - x874 + x875 + x876 + x878
        x880 = r_12*x136
        x881 = x362*x861
        x882 = 4*x110
        x883 = x853*x882
        x884 = d_3*x362
        x885 = x84*x884
        x886 = x110*x845 - x140*x359 - 4*x84*x868
        x887 = x854 - x856 + x858
        x888 = x860 + x862 + x869 + x871
        x889 = 4*x863
        x890 = r_13*x470
        x891 = r_13*x313
        x892 = 4*x866
        x893 = 4*x867
        x894 = x110*x844
        x895 = r_13*x136
        x896 = r_11*x36
        x897 = r_11*x24
        x898 = r_12*x96
        x899 = r_12*x22
        x900 = x896 + x897 - x898 - x899
        x901 = 4*r_13
        x902 = d_3*x901
        x903 = x84*x902 + x861*x901 - x882*x900 + x895
        x904 = 8*x848
        x905 = 8*x851
        x906 = 8*x849
        x907 = 8*x852
        x908 = 8*x850
        x909 = d_4*x803
        x910 = x873 + x874 - x875 - x876 - x878
        x911 = x219*x877
        x912 = Py*x362
        x913 = x912*x94
        x914 = x22*x880
        x915 = r_11*x470
        x916 = d_4*x915
        x917 = x24*x844
        x918 = x911 - x913 - x914 + x916 + x917
        x919 = x306*x853
        x920 = x226*x855
        x921 = d_5*x845
        x922 = r_12*x219
        x923 = x127*x922
        x924 = x160*x849
        x925 = x160*x850
        x926 = Py*x4
        x927 = x359*x926
        x928 = x26*x845
        x929 = -x919 + x920 - x921 - x923 - x924 - x925 + x927 + x928
        x930 = x318*x868
        x931 = x269*x44
        x932 = x269*x45
        x933 = x269*x46
        x934 = r_11*x326
        x935 = r_11**3
        x936 = 2*x59
        x937 = x935*x936
        x938 = x229*x870
        x939 = x269*x68
        x940 = x269*x69
        x941 = x269*x72
        x942 = x269*x73
        x943 = x269*x63
        x944 = x269*x65
        x945 = x269*x67
        x946 = x269*x71
        x947 = x426*x849
        x948 = x426*x850
        x949 = Px*x60
        x950 = 4*x96
        x951 = x949*x950
        x952 = Px*x950
        x953 = x62*x952
        x954 = x64*x952
        x955 = x587*x949
        x956 = Px*x587
        x957 = x62*x956
        x958 = x64*x956
        x959 = x359*x51
        x960 = d_5*x26
        x961 = x359*x960
        x962 = r_21*x66
        x963 = r_12*x3
        x964 = x962*x963
        x965 = r_31*x277
        x966 = x362*x965
        x967 = r_23*x962
        x968 = x901*x967
        x969 = r_31*x442
        x970 = x901*x969
        x971 = x470*x897
        x972 = x26*x54
        x973 = x359*x972
        x974 = x22*x96
        x975 = x359*x974
        x976 = x313*x898
        x977 = x470*x899
        x978 = x434*x849
        x979 = 4*x54
        x980 = x850*x979
        x981 = -x930 - x931 - x932 - x933 - x934 + x937 - x938 - x939 - x940 - x941 - x942 + x943 + x944 + x945 + x946 - x947 - x948 + x951 + x953 + x954 + x955 + x957 + x958 + x959 + x961 + x964 + x966 + x968 + x970 - x971 - x973 + x975 + x976 + x977 + x978 + x980
        x982 = a_2*x814
        x983 = 8*x197
        x984 = x853*(R_l_inv_26*x982 + R_l_inv_36*x815 + R_l_inv_76*x983 + 4*x218)
        x985 = x362*x44
        x986 = x362*x45
        x987 = x362*x46
        x988 = x245*x362
        x989 = r_12**3
        x990 = 4*x59
        x991 = x989*x990
        x992 = d_3*x229
        x993 = x362*x992
        x994 = x362*x67
        x995 = x362*x69
        x996 = x362*x71
        x997 = x362*x73
        x998 = x362*x61
        x999 = x362*x65
        x1000 = x362*x68
        x1001 = x362*x72
        x1002 = x462*x864
        x1003 = x462*x865
        x1004 = 8*x36
        x1005 = x1004*x949
        x1006 = Px*x1004
        x1007 = x1006*x62
        x1008 = x1006*x64
        x1009 = x798*x949
        x1010 = Px*x798
        x1011 = x1010*x62
        x1012 = x1010*x64
        x1013 = x51*x803
        x1014 = x803*x960
        x1015 = r_22*x66
        x1016 = 8*x1015
        x1017 = x1016*x804
        x1018 = r_11*r_31*x807
        x1019 = r_13*r_23
        x1020 = x1016*x1019
        x1021 = r_13*r_33
        x1022 = x1021*x807
        x1023 = x803*x974
        x1024 = x803*x972
        x1025 = 8*x96
        x1026 = x1025*x897
        x1027 = x810*x896
        x1028 = x36*x803
        x1029 = x1028*x24
        x1030 = x812*x864
        x1031 = 8*x54
        x1032 = x1031*x865
        x1033 = -d_5*x846 - x185*x864 - x185*x865 + x219*x845 - x226*x359 + x26*x846 + x803*x926 - x868*(R_l_inv_27*x982 + R_l_inv_37*x815 + R_l_inv_77*x983 + 4*x228)
        x1034 = x919 - x920 + x921 + x923 + x924 + x925 - x927 - x928
        x1035 = x930 + x931 + x932 + x933 + x934 - x937 + x938 + x939 + x940 + x941 + x942 - x943 - x944 - x945 - x946 + x947 + x948 - x951 - x953 - x954 - x955 - x957 - x958 - x959 - x961 - x964 - x966 - x968 - x970 + x971 + x973 - x975 - x976 - x977 - x978 - x980
        x1036 = d_5*x909
        x1037 = x219*x844
        x1038 = 8*r_13
        x1039 = Py*x6
        x1040 = x1038*x1039
        x1041 = x467*x865
        x1042 = x54*x909
        x1043 = x26*x909
        x1044 = 4*x898
        x1045 = x24*x359
        x1046 = -x1044 + x1045 - 4*x899 + x915
        x1047 = r_13**3
        x1048 = Px*x462
        x1049 = Px*x1031
        x1050 = Px*x812
        x1051 = x66*x804
        x1052 = 8*r_11
        x1053 = r_33*x277
        x1054 = d_5*x803
        x1055 = r_13*x26
        x1056 = x24*x803
        x1057 = -r_23*x1015*x803 - 8*r_23*x1051 - x1028*x26 - x1031*x1055 - x1031*x949 + x1038*x51 - x1046*x219 - x1047*x990 + x1048*x60 + x1048*x62 + x1048*x64 - x1049*x62 - x1049*x64 - x1050*x62 - x1050*x64 - x1052*x969 - x1053*x803 + x1054*x24 + x1054*x36 + x1055*x462 - x1056*x54 + x22*x904 - x22*x905 + x22*x906 + x245*x901 + x44*x901 + x45*x901 - x46*x901 - x61*x901 - x63*x901 + x67*x901 + x68*x901 - x69*x901 + x71*x901 + x72*x901 - x73*x901 + x798*x864 - x812*x949 + x901*x992 + x904*x96 - x907*x96
        x1058 = -x911 + x913 + x914 - x916 - x917
        x1059 = d_3*x895
        x1060 = -d_5*x895
        x1061 = x507*x877
        x1062 = Px*x136
        x1063 = x1062*x60
        x1064 = x1062*x62
        x1065 = x1062*x64
        x1066 = Py*x94
        x1067 = x1066*x359
        x1068 = d_4*r_12
        x1069 = x1068*x470
        x1070 = x54*x895
        x1071 = x22*x844
        x1072 = x1068*x313
        x1073 = x26*x895
        x1074 = -x1059 + x1060 + x1061 + x1063 + x1064 + x1065 + x1067 + x1069 + x1070 + x1071 + x1072 + x1073
        x1075 = x613*x868
        x1076 = r_11*x615
        x1077 = x512*x870
        x1078 = d_3*x889
        x1079 = x4*x912
        x1080 = x160*x867
        x1081 = Py*x34
        x1082 = r_13*x1081
        x1083 = d_3*x891
        x1084 = -x1075 - x1076 - x1077 + x1078 - x1079 - x1080 + x1082 + x1083
        x1085 = x571*x853
        x1086 = x45*x855
        x1087 = x936*x989
        x1088 = x44*x855
        x1089 = x46*x855
        x1090 = x536*x855
        x1091 = x507*x857
        x1092 = x61*x855
        x1093 = x65*x855
        x1094 = x68*x855
        x1095 = x72*x855
        x1096 = x67*x855
        x1097 = x69*x855
        x1098 = x71*x855
        x1099 = x73*x855
        x1100 = x470*x949
        x1101 = Px*x470
        x1102 = x1101*x62
        x1103 = x1101*x64
        x1104 = x313*x949
        x1105 = Px*x313
        x1106 = x1105*x62
        x1107 = x1105*x64
        x1108 = x362*x51
        x1109 = x26*x889
        x1110 = x1051*x3
        x1111 = x359*x965
        x1112 = x1019*x3*x66
        x1113 = x1053*x901
        x1114 = d_5*x890
        x1115 = d_5*x891
        x1116 = x1045*x96
        x1117 = x22*x915
        x1118 = r_12*x24
        x1119 = x1118*x470
        x1120 = x26*x890
        x1121 = x54*x891
        x1122 = x1044*x22
        x1123 = x26*x892
        x1124 = -x1085 - x1086 - x1087 + x1088 + x1089 + x1090 - x1091 - x1092 - x1093 - x1094 - x1095 + x1096 + x1097 + x1098 + x1099 - x1100 - x1102 - x1103 - x1104 - x1106 - x1107 - x1108 - x1109 - x1110 - x1111 - x1112 - x1113 + x1114 + x1115 - x1116 - x1117 - x1119 - x1120 - x1121 + x1122 + x1123
        x1125 = x853*(R_l_inv_26*x815 - R_l_inv_36*x982 + R_l_inv_66*x983 - 4*x506)
        x1126 = x362*x503
        x1127 = d_3*x904
        x1128 = d_3*x512
        x1129 = x1128*x362
        x1130 = x185*x849
        x1131 = x185*x850
        x1132 = x1052*x926
        x1133 = x185*x852
        x1134 = x359*x46
        x1135 = x935*x990
        x1136 = x359*x68
        x1137 = x359*x69
        x1138 = x359*x72
        x1139 = x359*x73
        x1140 = x359*x63
        x1141 = x359*x65
        x1142 = x359*x67
        x1143 = x359*x71
        x1144 = x462*x849
        x1145 = x462*x850
        x1146 = x1025*x949
        x1147 = Px*x1025
        x1148 = x1147*x62
        x1149 = x1147*x64
        x1150 = x810*x949
        x1151 = Px*x810
        x1152 = x1151*x62
        x1153 = x1151*x64
        x1154 = x1052*x51
        x1155 = x26*x904
        x1156 = r_22*x803*x962
        x1157 = x803*x965
        x1158 = x1038*x967
        x1159 = x1038*x969
        x1160 = x798*x896
        x1161 = x26*x905
        x1162 = x1052*x974
        x1163 = x1056*x96
        x1164 = x1028*x22
        x1165 = x26*x906
        x1166 = x54*x908
        x1167 = -x1134 + x1135 - x1136 - x1137 - x1138 - x1139 + x1140 + x1141 + x1142 + x1143 - x1144 - x1145 + x1146 + x1148 + x1149 + x1150 + x1152 + x1153 + x1154 + x1155 + x1156 + x1157 + x1158 + x1159 - x1160 - x1161 + x1162 + x1163 + x1164 + x1165 + x1166
        x1168 = x359*x44
        x1169 = x359*x45
        x1170 = -x1168 + x1169
        x1171 = x1167 + x1170 - x359*x536 + x507*x845 - x868*(R_l_inv_27*x815 - R_l_inv_37*x982 + R_l_inv_67*x983 - 4*x511)
        x1172 = x1075 + x1076 + x1077 - x1078 + x1079 + x1080 - x1082 - x1083
        x1173 = x1085 + x1086 + x1087 - x1088 - x1089 - x1090 + x1091 + x1092 + x1093 + x1094 + x1095 - x1096 - x1097 - x1098 - x1099 + x1100 + x1102 + x1103 + x1104 + x1106 + x1107 + x1108 + x1109 + x1110 + x1111 + x1112 + x1113 - x1114 - x1115 + x1116 + x1117 + x1119 + x1120 + x1121 - x1122 - x1123
        x1174 = r_11*x467
        x1175 = d_3*x1174
        x1176 = -x1175
        x1177 = x507*x844
        x1178 = -x1046*x507 + x1128*x901 + x185*x896 + x185*x897 - x22*x846 + x503*x901 - x846*x96
        x1179 = x1059 + x1060 - x1061 + x1063 + x1064 + x1065 + x1067 + x1069 + x1070 + x1071 + x1072 + x1073
        x1180 = Px*x160
        x1181 = -x1180*x60
        x1182 = -x1180*x62
        x1183 = -x1180*x64
        x1184 = d_3*x880
        x1185 = d_5*x902
        x1186 = -x845*x96
        x1187 = -r_12*x1081
        x1188 = -x901*x926
        x1189 = -x22*x845
        x1190 = -d_3*r_12*x313
        x1191 = -x26*x902
        x1192 = R_l_inv_44*x362*x661
        x1193 = x741*x853
        x1194 = x725*x884
        x1195 = x1181 + x1182 + x1183 + x1184 + x1185 + x1186 + x1187 + x1188 + x1189 + x1190 + x1191 - x1192 + x1193 + x1194
        x1196 = x136*x863
        x1197 = x136*x866
        x1198 = x136*x867
        x1199 = d_4*x890
        x1200 = d_4*x891
        x1201 = x359*x685
        x1202 = x749*x868
        x1203 = x690*x845
        x1204 = x1196 - x1197 - x1198 + x1199 + x1200 + x1201 + x1202 + x1203
        x1205 = x273*x45
        x1206 = x273*x44
        x1207 = x273*x46
        x1208 = x1047*x936
        x1209 = Px*x426
        x1210 = x1209*x60
        x1211 = x1209*x62
        x1212 = x1209*x64
        x1213 = x273*x67
        x1214 = x273*x68
        x1215 = x273*x71
        x1216 = x273*x72
        x1217 = x273*x61
        x1218 = x273*x63
        x1219 = x273*x69
        x1220 = x273*x73
        x1221 = d_5*x359
        x1222 = x1221*x96
        x1223 = x470*x863
        x1224 = x51*x901
        x1225 = x1221*x22
        x1226 = x313*x863
        x1227 = x901*x960
        x1228 = x949*x979
        x1229 = Px*x979
        x1230 = x1229*x62
        x1231 = x1229*x64
        x1232 = x434*x949
        x1233 = Px*x434
        x1234 = x1233*x62
        x1235 = x1233*x64
        x1236 = r_23*x66
        x1237 = x1236*x360
        x1238 = x359*x969
        x1239 = x1236*x963
        x1240 = x1053*x362
        x1241 = x587*x849
        x1242 = x470*x865
        x1243 = x725*x895
        x1244 = x26*x359*x96
        x1245 = x22*x359*x54
        x1246 = x470*x867
        x1247 = x313*x866
        x1248 = x901*x972
        x1249 = -x1205 + x1206 + x1207 + x1208 - x1210 - x1211 - x1212 - x1213 - x1214 - x1215 - x1216 + x1217 + x1218 + x1219 + x1220 - x1222 - x1223 - x1224 - x1225 - x1226 - x1227 + x1228 + x1230 + x1231 + x1232 + x1234 + x1235 + x1237 + x1238 + x1239 + x1240 - x1241 - x1242 - x1243 + x1244 + x1245 + x1246 + x1247 + x1248
        x1250 = d_4*x904
        x1251 = Py*r_13*x464
        x1252 = x467*x850
        x1253 = x816*x853
        x1254 = x467*x851
        x1255 = x467*x852
        x1256 = x685*x803
        x1257 = x690*x846
        x1258 = R_l_inv_44*r_11*x815 + R_l_inv_47*x815*x868 - r_11*x185*x725 + x1176
        x1259 = -x1196 + x1197 + x1198 - x1199 - x1200 - x1201 - x1202 - x1203
        x1260 = x1181 + x1182 + x1183 - x1184 + x1185 + x1186 + x1187 + x1188 + x1189 + x1190 + x1191 + x1192 - x1193 - x1194
        x1261 = x1174*x725
        x1262 = -R_l_inv_43*r_13*x815 - r_13*x185*x690 + x1039*x1052 - x1066*x803 - x22*x909 + x467*x897 + x816*x900
        x1263 = 16*d_5
        x1264 = 16*x26
        x1265 = 16*x22
        x1266 = 16*x36
        x1267 = Px*x1266
        x1268 = 16*x24
        x1269 = Px*x1268
        x1270 = 16*x1015
        x1271 = x1205 - x1206 - x1207 - x1208 + x1210 + x1211 + x1212 + x1213 + x1214 + x1215 + x1216 - x1217 - x1218 - x1219 - x1220 + x1222 + x1223 + x1224 + x1225 + x1226 + x1227 - x1228 - x1230 - x1231 - x1232 - x1234 - x1235 - x1237 - x1238 - x1239 - x1240 + x1241 + x1242 + x1243 - x1244 - x1245 - x1246 - x1247 - x1248
        x1272 = -x10 - x12 - x14 + x15 - x17 - x19 - x21 - x23 - x25 - x27
        x1273 = x1272 + x30
        x1274 = -x29
        x1275 = x130 + x189
        x1276 = x100 + x102 + x111 + x114 + x94
        x1277 = a_2 + x156 + x39 + x41 + x76 + x77 + x92
        x1278 = x141 + x143 - x144
        x1279 = x155 + x43 + x86 + x88
        x1280 = x103 + x112 + x115 + x95 + x99
        x1281 = -x163 - x165 - x167 - x168
        x1282 = x117 + x118 + x120 + x122 + x124 + x190
        x1283 = x307 + x308 + x309 - x310 + x312 + x314 - x315 - x316
        x1284 = -x398 + x400 + x402 - x404 - x406 - x407 + x409 + x411 + x413 + x415 - x417 - x419 - x421 - x423 - x425 - x427 - x428 - x429 + x431 + x432 + x433 + x435 + x436 + x437 + x439 + x441 + x443 + x445 - x446 - x447 - x448 - x449 - x450 - x451 + x452 + x453 + x454 + x456 + x457 - x458 - x459
        x1285 = x525 - x526 - x527 - x528 - x529 - x530 - x531 - x532 - x533 - x534
        x1286 = x1285 + x628
        x1287 = x572 + x573 - x574 + x775 + x825
        x1288 = x1285 + x564
        x1289 = -x638 - x639 - x640 + x641 + x642 - x643 - x644
        x1290 = x693 + x694 + x695 - x696 + x697 + x698 + x699 + x700 + x701 + x702
        x1291 = x1290 + x831
        x1292 = x634 - x740 - x742 + x744
        x1293 = x766 + x767 + x768 + x769 - x770 - x771 + x772
        x1294 = x1290 + x732
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
        A[1, 2] = x31 + x5
        A[1, 3] = x33
        A[1, 4] = -x34
        A[1, 5] = x32
        A[1, 6] = x28 + x35
        A[1, 7] = x29
        A[1, 8] = x31 + x4
        A[2, 0] = x116 + x131 + x93
        A[2, 1] = x139 + x145 + x153
        A[2, 2] = x131 + x154 + x157
        A[2, 3] = x162 + x169 + x176
        A[2, 4] = x177 - x178 - x179 + x180 + x181 + x183 + x186
        A[2, 5] = x169 + x187 + x188
        A[2, 6] = x154 + x191 + x93
        A[2, 7] = x145 + x192 + x193
        A[2, 8] = x116 + x157 + x191
        A[3, 0] = x217 + x240 + x293
        A[3, 1] = x301 + x317 + x377
        A[3, 2] = x379 + x381 + x382
        A[3, 3] = x388 + x395 + x460
        A[3, 4] = x461 + x463 - x465 - x466 + x468 + x469 + x472
        A[3, 5] = x460 + x473 + x474
        A[3, 6] = x239 + x478 + x480
        A[3, 7] = x317 + x481 + x482
        A[3, 8] = x380 + x483 + x484
        A[4, 0] = x502 + x524 + x565
        A[4, 1] = x570 + x612 + x623
        A[4, 2] = x625 + x627 + x629
        A[4, 3] = x633 + x637 + x645
        A[4, 4] = -x646 + x647 + x648
        A[4, 5] = x645 + x649 + x650
        A[4, 6] = x565 + x652 + x654
        A[4, 7] = x612 + x655 + x656
        A[4, 8] = x629 + x657 + x658
        A[5, 0] = x673 + x692 + x733
        A[5, 1] = x739 + x745 + x755
        A[5, 2] = x733 + x758 + x761
        A[5, 3] = x765 + x773 + x776
        A[5, 4] = -x777 - x778 + x779 + x780 - x781 - x782 - x783 - x784 + x785 + x786 + x787 + x788 - x790 - x791 + x793 + x794 + x795 + x797 + x799 + x800 + x801 + x802 + x805 + x806 + x808 + x809 - x811 - x813 - x817 + x818 + x819 + x820 + x821 + x822 + x823
        A[5, 5] = x773 + x824 + x826
        A[5, 6] = x829 + x830 + x832
        A[5, 7] = x745 + x833 + x834
        A[5, 8] = x832 + x835 + x836
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x837
        B[0, 2] = x837
        B[0, 3] = -x359
        B[0, 4] = -x803
        B[0, 5] = x359
        B[0, 6] = x273
        B[0, 8] = x273
        B[1, 0] = x839 + x843
        B[1, 1] = x844
        B[1, 2] = x838 + x843
        B[1, 3] = -x845
        B[1, 4] = -x846
        B[1, 5] = x845
        B[1, 6] = x839 + x847
        B[1, 7] = x844
        B[1, 8] = x838 + x847
        B[2, 0] = x859 + x872 + x879
        B[2, 1] = -x880 - x881 + x883 - x885 + x886
        B[2, 2] = x879 + x887 + x888
        B[2, 3] = -x889 - x890 - x891 + x892 + x893 + x894 + x903
        B[2, 4] = x110*x909 + x904 - x905 + x906 - x907 + x908
        B[2, 5] = x889 + x890 + x891 - x892 - x893 - x894 + x903
        B[2, 6] = x859 + x888 + x910
        B[2, 7] = x880 + x881 - x883 + x885 + x886
        B[2, 8] = x872 + x887 + x910
        B[3, 0] = x918 + x929 + x981
        B[3, 1] = x1000 + x1001 - x1002 - x1003 + x1005 + x1007 + x1008 + x1009 + x1011 + x1012 + x1013 + x1014 + x1017 + x1018 + x1020 + x1022 - x1023 - x1024 + x1026 + x1027 + x1029 + x1030 + x1032 + x1033 + x984 - x985 - x986 - x987 - x988 + x991 - x993 - x994 - x995 - x996 - x997 + x998 + x999
        B[3, 2] = x1034 + x1035 + x918
        B[3, 3] = -x1036 + x1037 - x1040 - x1041 + x1042 + x1043 + x1057
        B[3, 4] = x467*(-r_11*x383 + r_11*x50 - r_11*x58 + r_13*x52 + x22*x273 + x922)
        B[3, 5] = x1036 - x1037 + x1040 + x1041 - x1042 - x1043 + x1057
        B[3, 6] = x1035 + x1058 + x929
        B[3, 7] = -x1000 - x1001 + x1002 + x1003 - x1005 - x1007 - x1008 - x1009 - x1011 - x1012 - x1013 - x1014 - x1017 - x1018 - x1020 - x1022 + x1023 + x1024 - x1026 - x1027 - x1029 - x1030 - x1032 + x1033 - x984 + x985 + x986 + x987 + x988 - x991 + x993 + x994 + x995 + x996 + x997 - x998 - x999
        B[3, 8] = x1034 + x1058 + x981
        B[4, 0] = x1074 + x1084 + x1124
        B[4, 1] = x1125 - x1126 - x1127 - x1129 - x1130 - x1131 + x1132 + x1133 + x1171
        B[4, 2] = x1074 + x1172 + x1173
        B[4, 3] = x1176 + x1177 + x1178
        B[4, 4] = x909*(-x127 + x507)
        B[4, 5] = x1175 - x1177 + x1178
        B[4, 6] = x1124 + x1172 + x1179
        B[4, 7] = -x1125 + x1126 + x1127 + x1129 + x1130 + x1131 - x1132 - x1133 + x1171
        B[4, 8] = x1084 + x1173 + x1179
        B[5, 0] = x1195 + x1204 + x1249
        B[5, 1] = -x1250 - x1251 - x1252 - x1253 + x1254 + x1255 + x1256 + x1257 + x1258
        B[5, 2] = x1249 + x1259 + x1260
        B[5, 3] = x1167 + x1168 - x1169 - x1261 + x1262
        B[5, 4] = 16*r_11*x965 + 16*r_12*x51 + x1019*x1270 + 16*x1021*x277 - 16*x1068*x725 + x1118*x1266 - x1263*x864 - x1263*x865 + x1264*x863 + x1264*x864 - x1264*x866 + x1265*x896 - x1265*x898 + x1266*x949 + x1267*x62 + x1267*x64 + x1268*x949 + x1269*x62 + x1269*x64 + x1270*x804 + x44*x803 - x45*x803 - x46*x803 + 16*x54*x865 + 8*x59*x989 + x61*x803 + x65*x803 - x67*x803 + x68*x803 - x69*x803 - x71*x803 + x72*x803 - x73*x803 + 16*x897*x96
        B[5, 5] = x1134 - x1135 + x1136 + x1137 + x1138 + x1139 - x1140 - x1141 - x1142 - x1143 + x1144 + x1145 - x1146 - x1148 - x1149 - x1150 - x1152 - x1153 - x1154 - x1155 - x1156 - x1157 - x1158 - x1159 + x1160 + x1161 - x1162 - x1163 - x1164 - x1165 - x1166 + x1170 + x1261 + x1262
        B[5, 6] = x1195 + x1259 + x1271
        B[5, 7] = x1250 + x1251 + x1252 + x1253 - x1254 - x1255 - x1256 - x1257 + x1258
        B[5, 8] = x1204 + x1260 + x1271
        C = np.zeros(shape=(6, 9))
        C[0, 0] = r_23
        C[0, 2] = r_23
        C[0, 3] = x1
        C[0, 4] = x3
        C[0, 5] = x2
        C[0, 6] = x0
        C[0, 8] = x0
        C[1, 0] = x1273 + x4
        C[1, 1] = x1274
        C[1, 2] = x1272 + x35
        C[1, 3] = x32
        C[1, 4] = x34
        C[1, 5] = x33
        C[1, 6] = x1273 + x5
        C[1, 7] = x1274
        C[1, 8] = x1272 + x8
        C[2, 0] = x1275 + x1276 + x1277
        C[2, 1] = x1278 + x139 + x193
        C[2, 2] = x1275 + x1279 + x1280
        C[2, 3] = x1281 + x162 + x188
        C[2, 4] = -x177 + x178 + x179 - x180 - x181 - x183 + x186
        C[2, 5] = x1281 + x176 + x187
        C[2, 6] = x1277 + x1280 + x1282
        C[2, 7] = x1278 + x153 + x192
        C[2, 8] = x1276 + x1279 + x1282
        C[3, 0] = x217 + x380 + x478
        C[3, 1] = x1283 + x301 + x482
        C[3, 2] = x239 + x379 + x484
        C[3, 3] = x1284 + x388 + x474
        C[3, 4] = -x461 - x463 + x465 + x466 - x468 - x469 + x472
        C[3, 5] = x1284 + x395 + x473
        C[3, 6] = x293 + x381 + x480
        C[3, 7] = x1283 + x377 + x481
        C[3, 8] = x240 + x382 + x483
        C[4, 0] = x1286 + x502 + x652
        C[4, 1] = x1287 + x570 + x656
        C[4, 2] = x1288 + x625 + x658
        C[4, 3] = x1289 + x633 + x649
        C[4, 4] = x646 - x647 + x648
        C[4, 5] = x1289 + x637 + x650
        C[4, 6] = x1286 + x524 + x654
        C[4, 7] = x1287 + x623 + x655
        C[4, 8] = x1288 + x627 + x657
        C[5, 0] = x1291 + x673 + x761
        C[5, 1] = x1292 + x739 + x834
        C[5, 2] = x1291 + x692 + x758
        C[5, 3] = x1293 + x765 + x826
        C[5, 4] = x777 + x778 - x779 - x780 + x781 + x782 + x783 + x784 - x785 - x786 - x787 - x788 + x790 + x791 - x793 - x794 - x795 - x797 - x799 - x800 - x801 - x802 - x805 - x806 - x808 - x809 + x811 + x813 + x817 - x818 - x819 - x820 - x821 - x822 + x823
        C[5, 5] = x1293 + x776 + x824
        C[5, 6] = x1294 + x829 + x836
        C[5, 7] = x1292 + x755 + x833
        C[5, 8] = x1294 + x830 + x835
        from solver.general_6dof.numerical_reduce_closure_equation import compute_solution_from_tanhalf_LME
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
            condition_0: bool = (abs((Px*math.sin(th_0) - Py*math.cos(th_0) + d_1 - d_5*(r_13*math.sin(th_0) - r_23*math.cos(th_0)))/d_4) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                x2 = math.acos((Px*x0 - Py*x1 + d_1 + d_5*(-r_13*x0 + r_23*x1))/d_4)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[5] = x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs((Px*math.sin(th_0) - Py*math.cos(th_0) + d_1 - d_5*(r_13*math.sin(th_0) - r_23*math.cos(th_0)))/d_4) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                x2 = math.acos((Px*x0 - Py*x1 + d_1 + d_5*(-r_13*x0 + r_23*x1))/d_4)
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
            condition_0: bool = (2*abs(a_0*d_3) >= zero_tolerance) or (abs(2*a_0*a_2 - 2*a_0*d_4*math.sin(th_3)) >= zero_tolerance) or (abs(Px**2 - 2*Px*d_5*r_13 + Py**2 - 2*Py*d_5*r_23 + Pz**2 - 2*Pz*d_5*r_33 - a_0**2 - a_2**2 + 2*a_2*d_4*math.sin(th_3) - d_1**2 + 2*d_1*d_4*math.cos(th_3) - d_3**2 - d_4**2 + d_5**2*r_13**2 + d_5**2*r_23**2 + d_5**2*r_33**2) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*a_0
                x1 = d_4*math.sin(th_3)
                x2 = a_2*x0 - x0*x1
                x3 = math.atan2(-d_3*x0, x2)
                x4 = a_0**2
                x5 = d_3**2
                x6 = 2*d_5
                x7 = d_5**2
                x8 = Px**2 - Px*r_13*x6 + Py**2 - Py*r_23*x6 + Pz**2 - Pz*r_33*x6 - a_2**2 + 2*a_2*x1 - d_1**2 + 2*d_1*d_4*math.cos(th_3) - d_4**2 + r_13**2*x7 + r_23**2*x7 + r_33**2*x7 - x4 - x5
                x9 = math.sqrt(x2**2 + 4*x4*x5 - x8**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x3 + math.atan2(x9, x8)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                
            condition_1: bool = (2*abs(a_0*d_3) >= zero_tolerance) or (abs(2*a_0*a_2 - 2*a_0*d_4*math.sin(th_3)) >= zero_tolerance) or (abs(Px**2 - 2*Px*d_5*r_13 + Py**2 - 2*Py*d_5*r_23 + Pz**2 - 2*Pz*d_5*r_33 - a_0**2 - a_2**2 + 2*a_2*d_4*math.sin(th_3) - d_1**2 + 2*d_1*d_4*math.cos(th_3) - d_3**2 - d_4**2 + d_5**2*r_13**2 + d_5**2*r_23**2 + d_5**2*r_33**2) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*a_0
                x1 = d_4*math.sin(th_3)
                x2 = a_2*x0 - x0*x1
                x3 = math.atan2(-d_3*x0, x2)
                x4 = a_0**2
                x5 = d_3**2
                x6 = 2*d_5
                x7 = d_5**2
                x8 = Px**2 - Px*r_13*x6 + Py**2 - Py*r_23*x6 + Pz**2 - Pz*r_33*x6 - a_2**2 + 2*a_2*x1 - d_1**2 + 2*d_1*d_4*math.cos(th_3) - d_4**2 + r_13**2*x7 + r_23**2*x7 + r_33**2*x7 - x4 - x5
                x9 = math.sqrt(x2**2 + 4*x4*x5 - x8**2)
                # End of temp variables
                this_solution[4] = x3 + math.atan2(-x9, x8)
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
            degenerate_valid_0 = (abs(th_2 - math.pi + 1.50329340913167) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(32, node_input_i_idx_in_queue)
            
            th_2 = this_solution[4]
            degenerate_valid_1 = (abs(th_2 - 2*math.pi + 1.50329340913167) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
            
            if not taken_by_degenerate:
                add_input_index_to(23, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_22_processor()
    # Finish code for solved_variable dispatcher node 22
    
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
            th_0 = this_solution[0]
            th_5 = this_solution[7]
            condition_0: bool = (abs(0.997722543047233*a_2 - 0.0674516648820665*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_4*r_11*math.sin(th_5)*math.cos(th_0) + d_4*r_12*math.cos(th_0)*math.cos(th_5) + d_4*r_21*math.sin(th_0)*math.sin(th_5) + d_4*r_22*math.sin(th_0)*math.cos(th_5) - d_5*r_13*math.cos(th_0) - d_5*r_23*math.sin(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = d_4*math.sin(th_5)
                x3 = d_4*math.cos(th_5)
                x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3
                x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3
                x6 = math.atan2(x4, x5)
                x7 = math.sqrt(x4**2 + x5**2 - 0.995450272904637*(-a_2 + 0.067605633802817*d_3)**2)
                x8 = -0.997722543047233*a_2 + 0.0674516648820665*d_3
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x6 + math.atan2(x7, x8)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(33, appended_idx)
                
            condition_1: bool = (abs(0.997722543047233*a_2 - 0.0674516648820665*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_4*r_11*math.sin(th_5)*math.cos(th_0) + d_4*r_12*math.cos(th_0)*math.cos(th_5) + d_4*r_21*math.sin(th_0)*math.sin(th_5) + d_4*r_22*math.sin(th_0)*math.cos(th_5) - d_5*r_13*math.cos(th_0) - d_5*r_23*math.sin(th_0)) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = d_4*math.sin(th_5)
                x3 = d_4*math.cos(th_5)
                x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3
                x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3
                x6 = math.atan2(x4, x5)
                x7 = math.sqrt(x4**2 + x5**2 - 0.995450272904637*(-a_2 + 0.067605633802817*d_3)**2)
                x8 = -0.997722543047233*a_2 + 0.0674516648820665*d_3
                # End of temp variables
                this_solution[1] = x6 + math.atan2(-x7, x8)
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
            condition_0: bool = (1 >= zero_tolerance) or (abs(-r_13*((-0.997722543047233*math.sin(th_1) - 0.0674516648820665*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) - r_23*((-0.997722543047233*math.sin(th_1) - 0.0674516648820665*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(0.0674516648820665*math.sin(th_1) - 0.997722543047233*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance) or (abs(-r_13*(-0.0674516648820665*math.sin(th_1) + 0.997722543047233*math.cos(th_1))*math.cos(th_0) - r_23*(-0.0674516648820665*math.sin(th_1) + 0.997722543047233*math.cos(th_1))*math.sin(th_0) + r_33*(0.997722543047233*math.sin(th_1) + 0.0674516648820665*math.cos(th_1))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = 0.0674516648820665*x1
                x3 = math.cos(th_1)
                x4 = 0.997722543047233*x3
                x5 = math.sin(th_0)
                x6 = math.sin(th_3)
                x7 = math.cos(th_0)
                x8 = 0.997722543047233*x1
                x9 = 0.0674516648820665*x3
                x10 = x0*(-x8 - x9)
                x11 = -x2 + x4
                # End of temp variables
                this_solution[6] = math.atan2(-r_13*(x10*x7 + x5*x6) - r_23*(x10*x5 - x6*x7) - r_33*x0*(x2 - x4), r_13*x11*x7 + r_23*x11*x5 - r_33*(x8 + x9))
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
            condition_0: bool = (abs(a_2*math.sin(th_2) + d_3*math.cos(th_2)) >= 1.0e-6) or (abs(a_0 + a_2*math.cos(th_2) - d_3*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = d_5*r_33
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = a_0 + a_2*x1 - d_3*x2
                x4 = -a_2*x2 - d_3*x1
                x5 = Px*math.cos(th_0) + Py*math.sin(th_0) - d_5*math.sin(th_1th_2th_4_soa)
                # End of temp variables
                this_solution[1] = math.atan2(x3*(-Pz + x0) + x4*x5, x3*x5 + x4*(Pz - x0))
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
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                # End of temp variables
                this_solution[7] = math.atan2(-r_11*x1 + r_21*x0, -r_12*x1 + r_22*x0)
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
            degenerate_valid_0 = (abs(th_2 - math.pi + 1.50329340913167) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(26, node_input_i_idx_in_queue)
            
            th_2 = this_solution[4]
            degenerate_valid_1 = (abs(th_2 - 2*math.pi + 1.50329340913167) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(29, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(16, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_15_processor()
    # Finish code for solved_variable dispatcher node 15
    
    # Code for explicit solution node 29, solved variable is th_1
    def ExplicitSolutionNode_node_29_solve_th_1_processor():
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
            th_0 = this_solution[0]
            th_5 = this_solution[7]
            condition_0: bool = (abs(0.997722543047233*a_2 - 0.0674516648820665*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_4*r_11*math.sin(th_5)*math.cos(th_0) + d_4*r_12*math.cos(th_0)*math.cos(th_5) + d_4*r_21*math.sin(th_0)*math.sin(th_5) + d_4*r_22*math.sin(th_0)*math.cos(th_5) - d_5*r_13*math.cos(th_0) - d_5*r_23*math.sin(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = d_4*math.sin(th_5)
                x3 = d_4*math.cos(th_5)
                x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3
                x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3
                x6 = math.atan2(x4, x5)
                x7 = math.sqrt(x4**2 + x5**2 - 0.995450272904637*(a_2 - 0.067605633802817*d_3)**2)
                x8 = 0.997722543047233*a_2 - 0.0674516648820665*d_3
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x6 + math.atan2(x7, x8)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(30, appended_idx)
                
            condition_1: bool = (abs(0.997722543047233*a_2 - 0.0674516648820665*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_4*r_11*math.sin(th_5)*math.cos(th_0) + d_4*r_12*math.cos(th_0)*math.cos(th_5) + d_4*r_21*math.sin(th_0)*math.sin(th_5) + d_4*r_22*math.sin(th_0)*math.cos(th_5) - d_5*r_13*math.cos(th_0) - d_5*r_23*math.sin(th_0)) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = d_4*math.sin(th_5)
                x3 = d_4*math.cos(th_5)
                x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3
                x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3
                x6 = math.atan2(x4, x5)
                x7 = math.sqrt(x4**2 + x5**2 - 0.995450272904637*(a_2 - 0.067605633802817*d_3)**2)
                x8 = 0.997722543047233*a_2 - 0.0674516648820665*d_3
                # End of temp variables
                this_solution[1] = x6 + math.atan2(-x7, x8)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(30, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_1_processor()
    # Finish code for explicit solution node 29
    
    # Code for non-branch dispatcher node 30
    # Actually, there is no code
    
    # Code for explicit solution node 31, solved variable is th_4
    def ExplicitSolutionNode_node_31_solve_th_4_processor():
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
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance) or (abs(-r_13*((0.997722543047233*math.sin(th_1) + 0.0674516648820665*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) - r_23*((0.997722543047233*math.sin(th_1) + 0.0674516648820665*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(-0.0674516648820665*math.sin(th_1) + 0.997722543047233*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance) or (abs(-r_13*(-0.0674516648820665*math.sin(th_1) + 0.997722543047233*math.cos(th_1))*math.cos(th_0) - r_23*(-0.0674516648820665*math.sin(th_1) + 0.997722543047233*math.cos(th_1))*math.sin(th_0) + r_33*(0.997722543047233*math.sin(th_1) + 0.0674516648820665*math.cos(th_1))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_1)
                x3 = -0.0674516648820665*x1 + 0.997722543047233*x2
                x4 = math.sin(th_0)
                x5 = math.sin(th_3)
                x6 = math.cos(th_0)
                x7 = 0.997722543047233*x1 + 0.0674516648820665*x2
                x8 = x0*x7
                # End of temp variables
                this_solution[6] = math.atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*x3, -r_13*x3*x6 - r_23*x3*x4 + r_33*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_4_processor()
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
            th_0 = this_solution[0]
            th_5 = this_solution[7]
            condition_0: bool = (abs(0.997722543047233*a_2 - 0.0674516648820665*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_4*r_11*math.sin(th_5)*math.cos(th_0) + d_4*r_12*math.cos(th_0)*math.cos(th_5) + d_4*r_21*math.sin(th_0)*math.sin(th_5) + d_4*r_22*math.sin(th_0)*math.cos(th_5) - d_5*r_13*math.cos(th_0) - d_5*r_23*math.sin(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = d_4*math.sin(th_5)
                x3 = d_4*math.cos(th_5)
                x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3
                x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3
                x6 = math.atan2(x4, x5)
                x7 = math.sqrt(x4**2 + x5**2 - 0.995450272904637*(-a_2 + 0.067605633802817*d_3)**2)
                x8 = -0.997722543047233*a_2 + 0.0674516648820665*d_3
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x6 + math.atan2(x7, x8)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(27, appended_idx)
                
            condition_1: bool = (abs(0.997722543047233*a_2 - 0.0674516648820665*d_3) >= zero_tolerance) or (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5) - d_5*r_33) >= zero_tolerance) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_4*r_11*math.sin(th_5)*math.cos(th_0) + d_4*r_12*math.cos(th_0)*math.cos(th_5) + d_4*r_21*math.sin(th_0)*math.sin(th_5) + d_4*r_22*math.sin(th_0)*math.cos(th_5) - d_5*r_13*math.cos(th_0) - d_5*r_23*math.sin(th_0)) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = d_4*math.sin(th_5)
                x3 = d_4*math.cos(th_5)
                x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3
                x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3
                x6 = math.atan2(x4, x5)
                x7 = math.sqrt(x4**2 + x5**2 - 0.995450272904637*(-a_2 + 0.067605633802817*d_3)**2)
                x8 = -0.997722543047233*a_2 + 0.0674516648820665*d_3
                # End of temp variables
                this_solution[1] = x6 + math.atan2(-x7, x8)
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
            condition_0: bool = (1 >= zero_tolerance) or (abs(-r_13*((-0.997722543047233*math.sin(th_1) - 0.0674516648820665*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) - r_23*((-0.997722543047233*math.sin(th_1) - 0.0674516648820665*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(0.0674516648820665*math.sin(th_1) - 0.997722543047233*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance) or (abs(-r_13*(-0.0674516648820665*math.sin(th_1) + 0.997722543047233*math.cos(th_1))*math.cos(th_0) - r_23*(-0.0674516648820665*math.sin(th_1) + 0.997722543047233*math.cos(th_1))*math.sin(th_0) + r_33*(0.997722543047233*math.sin(th_1) + 0.0674516648820665*math.cos(th_1))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = 0.0674516648820665*x1
                x3 = math.cos(th_1)
                x4 = 0.997722543047233*x3
                x5 = math.sin(th_0)
                x6 = math.sin(th_3)
                x7 = math.cos(th_0)
                x8 = 0.997722543047233*x1
                x9 = 0.0674516648820665*x3
                x10 = x0*(-x8 - x9)
                x11 = -x2 + x4
                # End of temp variables
                this_solution[6] = math.atan2(-r_13*(x10*x7 + x5*x6) - r_23*(x10*x5 - x6*x7) - r_33*x0*(x2 - x4), r_13*x11*x7 + r_23*x11*x5 - r_33*(x8 + x9))
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
            condition_0: bool = (abs(a_2*math.sin(th_2) + d_3*math.cos(th_2)) >= 1.0e-6) or (abs(a_0 + a_2*math.cos(th_2) - d_3*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = d_5*r_33
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = a_0 + a_2*x1 - d_3*x2
                x4 = -a_2*x2 - d_3*x1
                x5 = math.cos(th_0)
                x6 = math.sin(th_0)
                x7 = Px*x5 + Py*x6 - d_5*r_13*x5 - d_5*r_23*x6
                # End of temp variables
                this_solution[1] = math.atan2(x3*(-Pz + x0) + x4*x7, x3*x7 + x4*(Pz - x0))
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
                x0 = math.asin((-r_13*math.sin(th_0) + r_23*math.cos(th_0))/math.sin(th_3))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[6] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
            condition_1: bool = (abs((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.asin((-r_13*math.sin(th_0) + r_23*math.cos(th_0))/math.sin(th_3))
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
                x0 = -a_0*math.cos(th_2) - a_2 + d_4*math.sin(th_3) + d_5*math.sin(th_4)*math.cos(th_3)
                x1 = -Px*math.cos(th_0) - Py*math.sin(th_0)
                x2 = a_0*math.sin(th_2) - d_3 + d_5*math.cos(th_4)
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
                x2 = math.cos(th_0)
                x3 = math.cos(th_3)
                x4 = math.sin(th_0)
                x5 = x0*math.cos(th_1th_2_soa)
                x6 = x2*x3 + x4*x5
                x7 = -x2*x5 + x3*x4
                # End of temp variables
                this_solution[7] = math.atan2(-r_11*x7 + r_21*x6 - r_31*x1, -r_12*x7 + r_22*x6 - r_32*x1)
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


def dense_cvr_038_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = dense_cvr_038_ik_target_original_to_raw(T_ee)
    ik_output_raw = dense_cvr_038_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ee_pose_i = dense_cvr_038_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_dense_cvr_038():
    theta_in = np.random.random(size=(6, ))
    ee_pose = dense_cvr_038_fk(theta_in)
    ik_output = dense_cvr_038_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = dense_cvr_038_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_dense_cvr_038()
