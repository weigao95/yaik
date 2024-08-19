import numpy as np
import copy
import math
from typing import List, NewType

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 28
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 0.444
a_1: float = 0.11
a_3: float = 0.08
d_2: float = 0.47
d_4: float = 0.101
pre_transform_special_symbol_23: float = 0.265

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
def abb_crb_15000_ik_target_original_to_raw(T_ee: np.ndarray):
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


def abb_crb_15000_ik_target_raw_to_original(T_ee: np.ndarray):
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


def abb_crb_15000_fk(theta_input: np.ndarray):
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
    x23 = 1.0*a_0
    x24 = x23*x6
    x25 = 1.0*a_1
    x26 = 1.0*d_2
    x27 = 1.0*d_4
    x28 = -x11*x4 - x11*x7
    x29 = x0*x28
    x30 = x11*x14 - x11*x15
    x31 = -x1*x12 + x13*x30
    x32 = x10*x31
    x33 = -1.0*x1*x13 - 1.0*x12*x30
    x34 = 1.0*x0*x31 - 1.0*x10*x28
    x35 = -x14 + x15
    x36 = x0*x35
    x37 = -x4 - x7
    x38 = x13*x37
    x39 = x10*x38
    x40 = 1.0*x12*x37
    x41 = 1.0*x0*x38 - 1.0*x10*x35
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -1.0*x18 - 1.0*x9
    ee_pose[0, 1] = x19*x20 + x21*x22
    ee_pose[0, 2] = x19*x22 - x20*x21
    ee_pose[0, 3] = a_3*x22 + x1*x24 + x16*x25 + x26*x8 + x27*(-x18 - x9)
    ee_pose[1, 0] = -1.0*x29 - 1.0*x32
    ee_pose[1, 1] = x19*x33 + x21*x34
    ee_pose[1, 2] = x19*x34 - x21*x33
    ee_pose[1, 3] = a_3*x34 + x11*x24 + x25*x30 + x26*x28 + x27*(-x29 - x32)
    ee_pose[2, 0] = -1.0*x36 - 1.0*x39
    ee_pose[2, 1] = -x19*x40 + x21*x41
    ee_pose[2, 2] = x19*x41 + x21*x40
    ee_pose[2, 3] = a_3*x41 + 1.0*pre_transform_special_symbol_23 - x2*x23 + x25*x37 + x26*x35 + x27*(-x36 - x39)
    return ee_pose


def abb_crb_15000_twist_jacobian(theta_input: np.ndarray):
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
    x14 = x10*x3 - x7*x8
    x15 = x1*x12 - x13*x14
    x16 = math.cos(th_4)
    x17 = math.sin(th_4)
    x18 = x1*x13 + x12*x14
    x19 = -x11*x16 - x17*x18
    x20 = x1*x4
    x21 = x1*x9
    x22 = -x20*x3 - x21*x8
    x23 = -x20*x8 + x21*x3
    x24 = -x12*x6 - x13*x23
    x25 = x12*x23 - x13*x6
    x26 = -x16*x22 - x17*x25
    x27 = 1.0*x8
    x28 = 1.0*x3
    x29 = x27*x4 - x28*x9
    x30 = -x27*x9 - x28*x4
    x31 = x13*x30
    x32 = x12*x30
    x33 = -x16*x29 - x17*x32
    x34 = 1.0*a_0
    x35 = pre_transform_special_symbol_23 - x34*x4
    x36 = a_1*x30 + d_2*x29 + x35
    x37 = a_0*x21 + a_1*x23 + d_2*x22
    x38 = a_3*(x16*x32 - x17*x29) + d_4*x33 + x36
    x39 = a_3*(x16*x25 - x17*x22) + d_4*x26 + x37
    x40 = a_0*x10 + a_1*x14 + d_2*x11
    x41 = a_3*(-x11*x17 + x16*x18) + d_4*x19 + x40
    x42 = x34*x9
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
    jacobian_output[2, 5] = x33
    jacobian_output[3, 1] = -pre_transform_special_symbol_23*x6
    jacobian_output[3, 2] = -x35*x6
    jacobian_output[3, 3] = -x22*x36 + x29*x37
    jacobian_output[3, 4] = -x24*x36 - x31*x37
    jacobian_output[3, 5] = -x26*x38 + x33*x39
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = -x1*x35
    jacobian_output[4, 3] = x11*x36 - x29*x40
    jacobian_output[4, 4] = x15*x36 + x31*x40
    jacobian_output[4, 5] = x19*x38 - x33*x41
    jacobian_output[5, 2] = x0**2*x42 + x42*x5**2
    jacobian_output[5, 3] = -x11*x37 + x22*x40
    jacobian_output[5, 4] = -x15*x37 + x24*x40
    jacobian_output[5, 5] = -x19*x39 + x26*x41
    return jacobian_output


def abb_crb_15000_angular_velocity_jacobian(theta_input: np.ndarray):
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


def abb_crb_15000_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x19 = a_1*x18 + d_2*x13 + x6
    x20 = a_0*x11
    x21 = x12*x14 - x14*x8
    x22 = a_1*x21 + d_2*x17 + x14*x20
    x23 = math.sin(th_3)
    x24 = x18*x23
    x25 = math.cos(th_3)
    x26 = -x2*x25 - x21*x23
    x27 = math.cos(th_4)
    x28 = math.sin(th_4)
    x29 = x18*x25
    x30 = -x13*x27 - x28*x29
    x31 = -x2*x23 + x21*x25
    x32 = -x17*x27 - x28*x31
    x33 = a_3*(-x13*x28 + x27*x29) + d_4*x30 + x19
    x34 = a_3*(-x17*x28 + x27*x31) + d_4*x32 + x22
    x35 = 1.0*p_on_ee_x
    x36 = 1.0*x14
    x37 = p_on_ee_z*x36
    x38 = x2*x4
    x39 = x10*x2
    x40 = -x38*x9 - x39*x7
    x41 = -x38*x7 + x39*x9
    x42 = a_0*x39 + a_1*x41 + d_2*x40
    x43 = -x23*x41 + x25*x36
    x44 = x23*x36 + x25*x41
    x45 = -x27*x40 - x28*x44
    x46 = a_3*(x27*x44 - x28*x40) + d_4*x45 + x42
    x47 = -x0*x14 - x1*x35
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = -x2*x6 + x3
    jacobian_output[0, 3] = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x22 - x17*x19
    jacobian_output[0, 4] = p_on_ee_y*x24 + p_on_ee_z*x26 - x19*x26 - x22*x24
    jacobian_output[0, 5] = -p_on_ee_y*x30 + p_on_ee_z*x32 + x30*x34 - x32*x33
    jacobian_output[1, 0] = x35
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x36 + x37
    jacobian_output[1, 2] = -x36*x6 + x37
    jacobian_output[1, 3] = p_on_ee_x*x13 - p_on_ee_z*x40 - x13*x42 + x19*x40
    jacobian_output[1, 4] = -p_on_ee_x*x24 - p_on_ee_z*x43 + x19*x43 + x24*x42
    jacobian_output[1, 5] = p_on_ee_x*x30 - p_on_ee_z*x45 - x30*x46 + x33*x45
    jacobian_output[2, 1] = x47
    jacobian_output[2, 2] = x1**2*x20 + x14**2*x20 + x47
    jacobian_output[2, 3] = -p_on_ee_x*x17 + p_on_ee_y*x40 + x17*x42 - x22*x40
    jacobian_output[2, 4] = -p_on_ee_x*x26 + p_on_ee_y*x43 - x22*x43 + x26*x42
    jacobian_output[2, 5] = -p_on_ee_x*x32 + p_on_ee_y*x45 + x32*x46 - x34*x45
    return jacobian_output


def abb_crb_15000_ik_solve_raw(T_ee: np.ndarray):
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
        x4 = a_3*r_21
        x5 = d_2*r_23
        x6 = -x5
        x7 = x4 + x6
        x8 = r_21**2
        x9 = Py*x8
        x10 = r_22**2
        x11 = Py*x10
        x12 = r_23**2
        x13 = Py*x12
        x14 = d_4*r_23
        x15 = Px*r_11
        x16 = r_21*x15
        x17 = Px*r_12
        x18 = r_22*x17
        x19 = Px*r_13
        x20 = r_23*x19
        x21 = Pz*r_31
        x22 = r_21*x21
        x23 = Pz*r_32
        x24 = r_22*x23
        x25 = Pz*r_33
        x26 = r_23*x25
        x27 = x11 + x13 - x14 + x16 + x18 + x20 + x22 + x24 + x26 + x9
        x28 = a_3*r_22
        x29 = 2*x28
        x30 = -x4
        x31 = x27 + x30
        x32 = d_2*x1
        x33 = -x32
        x34 = d_2*x3
        x35 = x4 + x5
        x36 = Py*r_22
        x37 = x17 + x23 + x36
        x38 = a_0*x37
        x39 = R_l_inv_51*x38
        x40 = R_l_inv_52*a_0
        x41 = d_2*x40
        x42 = a_0*r_22
        x43 = R_l_inv_54*x42
        x44 = a_3**2
        x45 = d_2**2
        x46 = d_4**2
        x47 = a_0**2
        x48 = a_1**2
        x49 = 2*d_4
        x50 = 2*x14
        x51 = Py*x1
        x52 = 2*x17
        x53 = Py*r_23
        x54 = 2*x19
        x55 = 2*x15
        x56 = 2*x23
        x57 = 2*x25
        x58 = Px**2
        x59 = r_11**2
        x60 = x58*x59
        x61 = r_12**2
        x62 = x58*x61
        x63 = r_13**2
        x64 = x58*x63
        x65 = Py**2
        x66 = x65*x8
        x67 = x10*x65
        x68 = x12*x65
        x69 = Pz**2
        x70 = r_31**2*x69
        x71 = r_32**2*x69
        x72 = r_33**2*x69
        x73 = -Py*x50 + x15*x51 - x19*x49 + x21*x51 + x21*x55 + x23*x52 - x25*x49 + x25*x54 + x36*x52 + x36*x56 + x44 + x45 + x46 - x47 - x48 + x53*x54 + x53*x57 + x60 + x62 + x64 + x66 + x67 + x68 + x70 + x71 + x72
        x74 = R_l_inv_55*a_0
        x75 = x73*x74
        x76 = d_4*r_21
        x77 = r_23*x15
        x78 = r_23*x21
        x79 = r_21*x19
        x80 = r_21*x25
        x81 = x76 + x77 + x78 - x79 - x80
        x82 = R_l_inv_57*a_0
        x83 = x81*x82
        x84 = -x83
        x85 = d_2*r_22
        x86 = R_l_inv_56*a_0
        x87 = x85*x86
        x88 = -x87
        x89 = a_3*r_23
        x90 = x82*x89
        x91 = -x90
        x92 = 2*a_3
        x93 = Py*r_21
        x94 = x15 + x21 + x93
        x95 = x74*x94
        x96 = x92*x95
        x97 = a_1 + x39 + x41 + x43 + x75 + x84 + x88 + x91 + x96
        x98 = -x28
        x99 = R_l_inv_50*a_0
        x100 = x94*x99
        x101 = -x100
        x102 = a_0*r_21
        x103 = R_l_inv_53*x102
        x104 = -x103
        x105 = d_4*r_22
        x106 = r_23*x17
        x107 = r_23*x23
        x108 = r_22*x19
        x109 = r_22*x25
        x110 = x105 + x106 + x107 - x108 - x109
        x111 = x110*x86
        x112 = -x111
        x113 = R_l_inv_57*d_2*x102
        x114 = -x113
        x115 = x101 + x104 + x112 + x114 + x98
        x116 = r_21*x17
        x117 = r_21*x23
        x118 = r_22*x15
        x119 = -x118
        x120 = r_22*x21
        x121 = -x120
        x122 = a_3*x99
        x123 = -d_4 + x19 + x25 + x53
        x124 = x123*x40
        x125 = 2*d_2
        x126 = x123*x125
        x127 = x126*x74
        x128 = -x122 - x124 - x127
        x129 = x116 + x117 + x119 + x121 + x128
        x130 = 2*R_l_inv_50*x38
        x131 = 2*x94
        x132 = -R_l_inv_51*a_0*x131
        x133 = 4*a_3
        x134 = R_l_inv_55*x38
        x135 = x133*x134
        x136 = -x130 + x132 + x135
        x137 = R_l_inv_54*a_0
        x138 = x1*x137
        x139 = 2*x82
        x140 = x110*x139
        x141 = x32*x86
        x142 = -x138 - x140 + x141
        x143 = a_3*x1
        x144 = 2*R_l_inv_53
        x145 = x144*x42
        x146 = 2*x86
        x147 = x146*x81
        x148 = x139*x85
        x149 = x143 - x145 + x147 - x148
        x150 = x100 + x103 + x111 + x113 + x28
        x151 = a_1 - x39 + x41 + x75 - x96
        x152 = -x43 + x83 + x87
        x153 = x151 + x152 + x91
        x154 = -x40*x92
        x155 = x131*x40
        x156 = 2*x123*x99
        x157 = -d_2*x133*x74
        x158 = 4*d_2
        x159 = x158*x95
        x160 = x154 - x155 + x156 + x157 - x159
        x161 = a_0*r_23*x144
        x162 = -x116 - x117 + x118 + x120
        x163 = x146*x162
        x164 = x139*x5
        x165 = x161 + x163 + x164
        x166 = 2*x105
        x167 = 2*x106
        x168 = 2*x107
        x169 = 2*x108
        x170 = 2*x109
        x171 = x29*x86
        x172 = -x166 - x167 - x168 + x169 + x170 + x171
        x173 = 4*x76
        x174 = 4*x79
        x175 = 4*x80
        x176 = 4*x77
        x177 = 4*x78
        x178 = 4*x4
        x179 = x178*x86
        x180 = 8*d_2
        x181 = -4*R_l_inv_52*x38 - x134*x180
        x182 = x154 + x155 + x156 + x157 + x159
        x183 = x166 + x167 + x168 - x169 - x170 - x171
        x184 = x122 + x124 + x127
        x185 = x162 + x184
        x186 = x130 + x132 + x135
        x187 = -x143 + x145 - x147 + x148
        x188 = a_0*a_1
        x189 = 2*x188
        x190 = x47 + x48
        x191 = R_l_inv_62*x190
        x192 = R_l_inv_22*x189 + x191
        x193 = d_2*x192
        x194 = R_l_inv_61*x190
        x195 = x37*(R_l_inv_21*x189 + x194)
        x196 = R_l_inv_25*x189 + R_l_inv_65*x190
        x197 = x196*x73
        x198 = R_l_inv_60*x190
        x199 = R_l_inv_20*x189 + x198
        x200 = a_3*x199
        x201 = -x200
        x202 = x199*x94
        x203 = -x202
        x204 = x123*x192
        x205 = -x204
        x206 = x126*x196
        x207 = -x206
        x208 = x196*x94
        x209 = x208*x92
        x210 = x193 + x195 + x197 + x201 + x203 + x205 + x207 + x209
        x211 = R_l_inv_24*x189 + R_l_inv_64*x190
        x212 = r_22*x211
        x213 = R_l_inv_67*x190
        x214 = R_l_inv_27*x189 + x213
        x215 = x214*x81
        x216 = -x215
        x217 = x214*x89
        x218 = -x217
        x219 = R_l_inv_66*x190
        x220 = R_l_inv_26*x189 + x219
        x221 = x220*x85
        x222 = -x221
        x223 = x5*x92
        x224 = -x223
        x225 = d_4*x32
        x226 = -x225
        x227 = x5*x55
        x228 = -x227
        x229 = 2*x21
        x230 = x229*x5
        x231 = -x230
        x232 = x19*x32
        x233 = x25*x32
        x234 = x212 + x216 + x218 + x222 + x224 + x226 + x228 + x231 + x232 + x233
        x235 = x14*x92
        x236 = x9*x92
        x237 = x11*x92
        x238 = x13*x92
        x239 = x143*x15
        x240 = x17*x29
        x241 = x20*x92
        x242 = x143*x21
        x243 = x23*x29
        x244 = x26*x92
        x245 = -x235 + x236 + x237 + x238 + x239 + x240 + x241 + x242 + x243 + x244
        x246 = r_21*x44
        x247 = r_21**3*x65
        x248 = r_21*x45
        x249 = r_21*x46
        x250 = R_l_inv_23*x189 + R_l_inv_63*x190
        x251 = r_21*x250
        x252 = x110*x220
        x253 = r_21*x60
        x254 = r_21*x67
        x255 = r_21*x68
        x256 = r_21*x70
        x257 = d_2*r_21
        x258 = x214*x257
        x259 = r_21*x62
        x260 = r_21*x64
        x261 = r_21*x71
        x262 = r_21*x72
        x263 = x15*x50
        x264 = x21*x50
        x265 = x55*x9
        x266 = x11*x55
        x267 = x13*x55
        x268 = d_4*x1
        x269 = x19*x268
        x270 = x229*x9
        x271 = x11*x229
        x272 = x13*x229
        x273 = x25*x268
        x274 = 2*r_11
        x275 = r_12*x58
        x276 = r_22*x275
        x277 = x274*x276
        x278 = 2*r_13
        x279 = r_23*x58
        x280 = r_11*x278*x279
        x281 = 2*r_31
        x282 = r_32*x69
        x283 = r_22*x282
        x284 = x281*x283
        x285 = r_23*r_33
        x286 = x285*x69
        x287 = x281*x286
        x288 = x17*x23
        x289 = x1*x288
        x290 = x19*x25
        x291 = x1*x290
        x292 = x15*x21
        x293 = x1*x292
        x294 = x24*x55
        x295 = x26*x55
        x296 = x18*x229
        x297 = x20*x229
        x298 = x246 + x247 - x248 - x249 - x251 - x252 + x253 + x254 + x255 + x256 - x258 - x259 - x260 - x261 - x262 - x263 - x264 + x265 + x266 + x267 + x269 + x270 + x271 + x272 + x273 + x277 + x280 + x284 + x287 - x289 - x291 + x293 + x294 + x295 + x296 + x297
        x299 = x245 + x298
        x300 = 4*x188
        x301 = R_l_inv_20*x300 + 2*x198
        x302 = x301*x37
        x303 = -x94*(R_l_inv_21*x300 + 2*x194)
        x304 = x133*x196
        x305 = x304*x37
        x306 = -x302 + x303 + x305
        x307 = R_l_inv_27*x300 + 2*x213
        x308 = x110*x307
        x309 = x1*x211
        x310 = d_4*x34
        x311 = x220*x32
        x312 = 4*x17
        x313 = x312*x5
        x314 = 4*x23
        x315 = x314*x5
        x316 = x19*x34
        x317 = x25*x34
        x318 = -x308 - x309 - x310 + x311 - x313 - x315 + x316 + x317
        x319 = R_l_inv_26*x300 + 2*x219
        x320 = x319*x81
        x321 = r_22*x45
        x322 = 2*x321
        x323 = r_22*x46
        x324 = 2*x323
        x325 = 2*x250
        x326 = r_22*x325
        x327 = r_22*x44
        x328 = 2*x327
        x329 = r_22**3*x65
        x330 = 2*x329
        x331 = 2*x214
        x332 = x331*x85
        x333 = r_22*x60
        x334 = 2*x333
        x335 = r_22*x64
        x336 = 2*x335
        x337 = r_22*x70
        x338 = 2*x337
        x339 = r_22*x72
        x340 = 2*x339
        x341 = r_22*x62
        x342 = 2*x341
        x343 = r_22*x66
        x344 = 2*x343
        x345 = r_22*x68
        x346 = 2*x345
        x347 = r_22*x71
        x348 = 2*x347
        x349 = x14*x312
        x350 = x14*x314
        x351 = x312*x9
        x352 = x11*x312
        x353 = x13*x312
        x354 = d_4*x3
        x355 = x19*x354
        x356 = x314*x9
        x357 = x11*x314
        x358 = x13*x314
        x359 = x25*x354
        x360 = 4*r_11
        x361 = r_21*x360
        x362 = x275*x361
        x363 = 4*r_12
        x364 = r_13*x279
        x365 = x363*x364
        x366 = 4*x282
        x367 = r_21*r_31
        x368 = x366*x367
        x369 = x285*x366
        x370 = x292*x3
        x371 = x290*x3
        x372 = x16*x314
        x373 = x22*x312
        x374 = x288*x3
        x375 = x26*x312
        x376 = x20*x314
        x377 = x320 - x322 - x324 - x326 + x328 + x330 - x332 - x334 - x336 - x338 - x340 + x342 + x344 + x346 + x348 - x349 - x350 + x351 + x352 + x353 + x355 + x356 + x357 + x358 + x359 + x362 + x365 + x368 + x369 - x370 - x371 + x372 + x373 + x374 + x375 + x376
        x378 = -x195
        x379 = -x209
        x380 = x193 + x197 + x201 + x202 + x205 + x207 + x378 + x379
        x381 = -x212 + x215 + x221 + x225 + x227 + x230 - x232 - x233
        x382 = x218 + x224 + x381
        x383 = -x246 - x247 + x248 + x249 + x251 + x252 - x253 - x254 - x255 - x256 + x258 + x259 + x260 + x261 + x262 + x263 + x264 - x265 - x266 - x267 - x269 - x270 - x271 - x272 - x273 - x277 - x280 - x284 - x287 + x289 + x291 - x293 - x294 - x295 - x296 - x297
        x384 = x245 + x383
        x385 = x123*x301
        x386 = x94*(R_l_inv_22*x300 + 2*x191)
        x387 = -x192*x92
        x388 = -d_2*x304
        x389 = x158*x208
        x390 = x385 - x386 + x387 + x388 - x389
        x391 = x220*x29
        x392 = 4*d_4
        x393 = x392*x4
        x394 = x178*x19
        x395 = x178*x25
        x396 = x133*x77
        x397 = x133*x78
        x398 = x391 + x393 - x394 - x395 + x396 + x397
        x399 = x162*x319
        x400 = r_23*x46
        x401 = 2*x400
        x402 = r_23**3*x65
        x403 = 2*x402
        x404 = r_23*x44
        x405 = 2*x404
        x406 = r_23*x45
        x407 = 2*x406
        x408 = r_23*x325
        x409 = r_23*x64
        x410 = 2*x409
        x411 = r_23*x66
        x412 = 2*x411
        x413 = r_23*x67
        x414 = 2*x413
        x415 = r_23*x72
        x416 = 2*x415
        x417 = x331*x5
        x418 = r_23*x60
        x419 = 2*x418
        x420 = r_23*x62
        x421 = 2*x420
        x422 = r_23*x70
        x423 = 2*x422
        x424 = r_23*x71
        x425 = 2*x424
        x426 = x392*x9
        x427 = x11*x392
        x428 = x13*x392
        x429 = 4*x19
        x430 = x429*x9
        x431 = x11*x429
        x432 = x13*x429
        x433 = 4*x25
        x434 = x433*x9
        x435 = x11*x433
        x436 = x13*x433
        x437 = r_13*x58
        x438 = x361*x437
        x439 = r_13*x3
        x440 = x275*x439
        x441 = r_33*x69
        x442 = 4*x367*x441
        x443 = x282*x3
        x444 = r_33*x443
        x445 = x15*x173
        x446 = x17*x354
        x447 = x14*x429
        x448 = x173*x21
        x449 = x23*x354
        x450 = x14*x433
        x451 = x16*x433
        x452 = x17*x3
        x453 = x25*x452
        x454 = x22*x429
        x455 = x23*x3
        x456 = x19*x455
        x457 = x20*x433
        x458 = x176*x21
        x459 = x106*x314
        x460 = x399 - x401 - x403 + x405 + x407 + x408 - x410 - x412 - x414 - x416 + x417 + x419 + x421 + x423 + x425 + x426 + x427 + x428 - x430 - x431 - x432 - x434 - x435 - x436 - x438 - x440 - x442 - x444 + x445 + x446 + x447 + x448 + x449 + x450 - x451 - x453 - x454 - x456 - x457 + x458 + x459
        x461 = x178*x220
        x462 = 8*d_4
        x463 = x28*x462
        x464 = 8*x28
        x465 = x19*x464
        x466 = x25*x464
        x467 = 8*a_3
        x468 = x106*x467
        x469 = x107*x467
        x470 = 8*x188
        x471 = x180*x37
        x472 = -x196*x471 - x37*(R_l_inv_22*x470 + 4*x191)
        x473 = x385 + x386 + x387 + x388 + x389
        x474 = -x391 - x393 + x394 + x395 - x396 - x397
        x475 = x193 + x197 + x200 + x204 + x206
        x476 = x195 + x202 + x209 + x475
        x477 = x235 - x236 - x237 - x238 - x239 - x240 - x241 - x242 - x243 - x244
        x478 = x383 + x477
        x479 = x302 + x303 + x305
        x480 = -x320 + x322 + x324 + x326 - x328 - x330 + x332 + x334 + x336 + x338 + x340 - x342 - x344 - x346 - x348 + x349 + x350 - x351 - x352 - x353 - x355 - x356 - x357 - x358 - x359 - x362 - x365 - x368 - x369 + x370 + x371 - x372 - x373 - x374 - x375 - x376
        x481 = x298 + x477
        x482 = x203 + x378 + x379 + x475
        x483 = R_l_inv_70*x190
        x484 = R_l_inv_30*x189 + x483
        x485 = a_3*x484
        x486 = x484*x94
        x487 = R_l_inv_72*x190
        x488 = R_l_inv_32*x189 + x487
        x489 = x123*x488
        x490 = -d_2*x488
        x491 = R_l_inv_71*x190
        x492 = x37*(R_l_inv_31*x189 + x491)
        x493 = -x492
        x494 = R_l_inv_35*x189 + R_l_inv_75*x190
        x495 = -x494*x73
        x496 = x494*x94
        x497 = x496*x92
        x498 = -x497
        x499 = x126*x494
        x500 = x485 + x486 + x489 + x490 + x493 + x495 + x498 + x499
        x501 = R_l_inv_77*x190
        x502 = R_l_inv_37*x189 + x501
        x503 = x502*x89
        x504 = x143*x17
        x505 = x143*x23
        x506 = x15*x29
        x507 = x21*x29
        x508 = x503 - x504 - x505 + x506 + x507
        x509 = R_l_inv_33*x189 + R_l_inv_73*x190
        x510 = r_21*x509
        x511 = R_l_inv_76*x190
        x512 = R_l_inv_36*x189 + x511
        x513 = x110*x512
        x514 = x257*x502
        x515 = x105*x125
        x516 = x54*x85
        x517 = x57*x85
        x518 = x5*x52
        x519 = x5*x56
        x520 = x510 + x513 + x514 + x515 - x516 - x517 + x518 + x519
        x521 = x508 + x520
        x522 = x502*x81
        x523 = R_l_inv_34*x189 + R_l_inv_74*x190
        x524 = r_22*x523
        x525 = x512*x85
        x526 = x52*x9
        x527 = x11*x52
        x528 = x13*x52
        x529 = x166*x19
        x530 = x56*x9
        x531 = x11*x56
        x532 = x13*x56
        x533 = x166*x25
        x534 = r_11*x275
        x535 = x1*x534
        x536 = r_23*x275*x278
        x537 = r_31*x1
        x538 = x282*x537
        x539 = x282*x285
        x540 = 2*x539
        x541 = x17*x50
        x542 = x23*x50
        x543 = x1*x15
        x544 = x23*x543
        x545 = x1*x21
        x546 = x17*x545
        x547 = x18*x56
        x548 = x26*x52
        x549 = x20*x56
        x550 = x118*x229
        x551 = x169*x25
        x552 = x321 + x323 + x327 - x329 + x333 + x335 + x337 + x339 - x341 - x343 - x345 - x347 + x522 - x524 + x525 - x526 - x527 - x528 - x529 - x530 - x531 - x532 - x533 - x535 - x536 - x538 - x540 + x541 + x542 - x544 - x546 - x547 - x548 - x549 + x550 + x551
        x553 = R_l_inv_30*x300 + 2*x483
        x554 = x37*x553
        x555 = x94*(R_l_inv_31*x300 + 2*x491)
        x556 = x133*x494
        x557 = -x37*x556
        x558 = x554 + x555 + x557
        x559 = R_l_inv_37*x300 + 2*x501
        x560 = x110*x559
        x561 = x1*x523
        x562 = x32*x512
        x563 = x1*x46
        x564 = 2*x247
        x565 = x1*x62
        x566 = x1*x64
        x567 = x1*x71
        x568 = x1*x72
        x569 = x1*x60
        x570 = x1*x67
        x571 = x1*x68
        x572 = x1*x70
        x573 = 4*x15
        x574 = x14*x573
        x575 = 4*x21
        x576 = x14*x575
        x577 = x573*x9
        x578 = x11*x573
        x579 = x13*x573
        x580 = x173*x19
        x581 = x575*x9
        x582 = x11*x575
        x583 = x13*x575
        x584 = x173*x25
        x585 = x3*x534
        x586 = x360*x364
        x587 = r_31*x443
        x588 = 4*r_31*x286
        x589 = x116*x314
        x590 = x174*x25
        x591 = x16*x575
        x592 = x15*x455
        x593 = x26*x573
        x594 = x21*x452
        x595 = x20*x575
        x596 = -x563 + x564 - x565 - x566 - x567 - x568 + x569 + x570 + x571 + x572 - x574 - x576 + x577 + x578 + x579 + x580 + x581 + x582 + x583 + x584 + x585 + x586 + x587 + x588 - x589 - x590 + x591 + x592 + x593 + x594 + x595
        x597 = x1*x44
        x598 = x1*x45
        x599 = -x597 - x598
        x600 = x560 + x561 - x562 + x596 + x599
        x601 = R_l_inv_36*x300 + 2*x511
        x602 = x601*x81
        x603 = 2*x509
        x604 = r_22*x603
        x605 = d_2*x173
        x606 = 2*x502
        x607 = x606*x85
        x608 = x5*x573
        x609 = x5*x575
        x610 = d_2*x174
        x611 = d_2*x175
        x612 = -x602 + x604 - x605 + x607 - x608 - x609 + x610 + x611
        x613 = -x486
        x614 = x485 + x489 + x490 + x492 + x495 + x497 + x499 + x613
        x615 = -x510
        x616 = -x513
        x617 = -x514
        x618 = -x515
        x619 = -x518
        x620 = -x519
        x621 = x508 + x516 + x517 + x615 + x616 + x617 + x618 + x619 + x620
        x622 = -x321 - x323 - x327 + x329 - x333 - x335 - x337 - x339 + x341 + x343 + x345 + x347 - x522 + x524 - x525 + x526 + x527 + x528 + x529 + x530 + x531 + x532 + x533 + x535 + x536 + x538 + x540 - x541 - x542 + x544 + x546 + x547 + x548 + x549 - x550 - x551
        x623 = x94*(R_l_inv_32*x300 + 2*x487)
        x624 = -x123*x553
        x625 = x488*x92
        x626 = d_2*x556
        x627 = x158*x496
        x628 = x623 + x624 + x625 + x626 + x627
        x629 = a_3*x34
        x630 = x29*x512
        x631 = -x629 - x630
        x632 = x162*x601
        x633 = r_23*x603
        x634 = x5*x606
        x635 = x15*x34
        x636 = x21*x34
        x637 = x116*x158
        x638 = x117*x158
        x639 = -x632 - x633 - x634 - x635 - x636 + x637 + x638
        x640 = x178*x512
        x641 = x180*x4
        x642 = x37*(R_l_inv_32*x470 + 4*x487) + x471*x494
        x643 = x629 + x630
        x644 = -x623 + x624 + x625 + x626 - x627
        x645 = -x485 - x489 + x490 + x495 - x499
        x646 = x552 + x645
        x647 = x493 + x498 + x613
        x648 = -x554 + x555 + x557
        x649 = x602 - x604 + x605 - x607 + x608 + x609 - x610 - x611
        x650 = x622 + x645
        x651 = x486 + x492 + x497
        x652 = a_3*x32
        x653 = -x652
        x654 = d_4*x143
        x655 = -x654
        x656 = x77*x92
        x657 = -x656
        x658 = x78*x92
        x659 = -x658
        x660 = x143*x19
        x661 = x143*x25
        x662 = x653 + x655 + x657 + x659 + x660 + x661
        x663 = x125*x9
        x664 = x11*x125
        x665 = x125*x13
        x666 = x49*x5
        x667 = x15*x32
        x668 = x52*x85
        x669 = x5*x54
        x670 = x21*x32
        x671 = x56*x85
        x672 = x5*x57
        x673 = -x663 - x664 - x665 + x666 - x667 - x668 - x669 - x670 - x671 - x672
        x674 = x49*x9
        x675 = x11*x49
        x676 = x13*x49
        x677 = x15*x268
        x678 = x166*x17
        x679 = x19*x50
        x680 = x21*x268
        x681 = x166*x23
        x682 = x25*x50
        x683 = x54*x9
        x684 = x11*x54
        x685 = x13*x54
        x686 = x57*x9
        x687 = x11*x57
        x688 = x13*x57
        x689 = r_11*x1*x437
        x690 = x276*x278
        x691 = x441*x537
        x692 = 2*r_33*x283
        x693 = x229*x77
        x694 = x167*x23
        x695 = x25*x543
        x696 = x18*x57
        x697 = x19*x545
        x698 = x24*x54
        x699 = x20*x57
        x700 = x400 + x402 - x404 + x406 + x409 + x411 + x413 + x415 - x418 - x420 - x422 - x424 - x674 - x675 - x676 - x677 - x678 - x679 - x680 - x681 - x682 + x683 + x684 + x685 + x686 + x687 + x688 + x689 + x690 + x691 + x692 - x693 - x694 + x695 + x696 + x697 + x698 + x699
        x701 = x673 + x700
        x702 = -x85
        x703 = -x105 - x106 - x107 + x108 + x109
        x704 = x654 + x656 + x658 - x660 - x661
        x705 = x652 + x704
        x706 = x133*x14
        x707 = x133*x9
        x708 = x11*x133
        x709 = x13*x133
        x710 = x15*x178
        x711 = a_3*x3
        x712 = x17*x711
        x713 = x133*x20
        x714 = x178*x21
        x715 = x23*x711
        x716 = x133*x26
        x717 = -x706 + x707 + x708 + x709 + x710 + x712 + x713 + x714 + x715 + x716
        x718 = x597 + x598
        x719 = x596 + x718
        x720 = x3*x44
        x721 = x3*x45
        x722 = x3*x46
        x723 = 8*x14
        x724 = x17*x723
        x725 = 8*x105
        x726 = x19*x725
        x727 = x25*x725
        x728 = x23*x723
        x729 = 8*x23
        x730 = x16*x729
        x731 = 8*x21
        x732 = x118*x731
        x733 = 8*x17
        x734 = x22*x733
        x735 = x18*x729
        x736 = x26*x733
        x737 = 8*x25
        x738 = x108*x737
        x739 = x20*x729
        x740 = 4*x329
        x741 = x733*x9
        x742 = x11*x733
        x743 = x13*x733
        x744 = 8*r_12
        x745 = r_11*r_21
        x746 = x58*x744*x745
        x747 = x364*x744
        x748 = x729*x9
        x749 = x11*x729
        x750 = x13*x729
        x751 = 8*x282
        x752 = x367*x751
        x753 = 8*x539
        x754 = x3*x60
        x755 = x3*x62
        x756 = x3*x64
        x757 = x3*x66
        x758 = x3*x68
        x759 = x3*x70
        x760 = x3*x71
        x761 = x3*x72
        x762 = x563 - x564 + x565 + x566 + x567 + x568 - x569 - x570 - x571 - x572 + x574 + x576 - x577 - x578 - x579 - x580 - x581 - x582 - x583 - x584 - x585 - x586 - x587 - x588 + x589 + x590 - x591 - x592 - x593 - x594 - x595
        x763 = x599 + x762
        x764 = x653 + x704
        x765 = -x400 - x402 + x404 - x406 - x409 - x411 - x413 - x415 + x418 + x420 + x422 + x424 + x674 + x675 + x676 + x677 + x678 + x679 + x680 + x681 + x682 - x683 - x684 - x685 - x686 - x687 - x688 - x689 - x690 - x691 - x692 + x693 + x694 - x695 - x696 - x697 - x698 - x699
        x766 = x673 + x765
        x767 = x652 + x655 + x657 + x659 + x660 + x661
        x768 = -x278
        x769 = r_11*x92
        x770 = d_2*x278
        x771 = 2*Px
        x772 = 2*r_12
        x773 = -d_4*x278 + r_11*x51 + x21*x274 + x23*x772 + x25*x278 + x278*x53 + x36*x772 + x59*x771 + x61*x771 + x63*x771
        x774 = -x770 + x773
        x775 = r_12*x133
        x776 = -x769
        x777 = d_2*x360
        x778 = d_2*x744
        x779 = x770 + x773
        x780 = d_4*r_11
        x781 = r_13*x93
        x782 = r_13*x21
        x783 = r_11*x53
        x784 = r_11*x25
        x785 = x780 + x781 + x782 - x783 - x784
        x786 = x139*x785
        x787 = x137*x772
        x788 = r_12*x125
        x789 = x788*x86
        x790 = a_3*x278
        x791 = -x790*x82
        x792 = -x786 + x787 - x789 + x791
        x793 = r_12*x51
        x794 = r_12*x21
        x795 = 2*x794
        x796 = r_11*x36
        x797 = 2*x796
        x798 = r_11*x23
        x799 = 2*x798
        x800 = -x793 - x795 + x797 + x799
        x801 = r_12*x92
        x802 = R_l_inv_53*a_0
        x803 = x274*x802
        x804 = d_4*r_12
        x805 = r_13*x36
        x806 = r_13*x23
        x807 = r_12*x53
        x808 = r_12*x25
        x809 = x804 + x805 + x806 - x807 - x808
        x810 = x146*x809
        x811 = r_11*x125
        x812 = x811*x82
        x813 = -x801 - x803 - x810 - x812
        x814 = a_3*x360
        x815 = x363*x802
        x816 = 4*x86
        x817 = x785*x816
        x818 = d_2*r_12
        x819 = 4*x82
        x820 = x818*x819
        x821 = -x137*x360 + x777*x86 - x809*x819
        x822 = x801 + x803 + x810 + x812
        x823 = x786 - x787 + x789 + x791
        x824 = 4*x804
        x825 = Py*x439
        x826 = 4*x806
        x827 = 4*x807
        x828 = 4*x808
        x829 = x775*x86
        x830 = r_12*x93
        x831 = -x794 + x796 + x798 - x830
        x832 = 4*r_13
        x833 = d_2*x832
        x834 = x802*x832 - x816*x831 + x82*x833
        x835 = 8*x780
        x836 = 8*x783
        x837 = 8*x781
        x838 = 8*x784
        x839 = 8*x782
        x840 = r_11*x467
        x841 = x793 + x795 - x797 - x799
        x842 = x307*x785
        x843 = x211*x772
        x844 = r_13*x133
        x845 = -d_2*x844
        x846 = d_4*x777
        x847 = -x214*x790
        x848 = x220*x788
        x849 = x158*x781
        x850 = x158*x782
        x851 = Py*x360
        x852 = x5*x851
        x853 = x25*x777
        x854 = -x842 + x843 + x845 - x846 + x847 - x848 - x849 - x850 + x852 + x853
        x855 = d_4*x844
        x856 = Px*x133
        x857 = x59*x856
        x858 = x61*x856
        x859 = x63*x856
        x860 = x4*x851
        x861 = Py*x711
        x862 = r_12*x861
        x863 = x53*x844
        x864 = x21*x814
        x865 = x23*x775
        x866 = x25*x844
        x867 = -x855 + x857 + x858 + x859 + x860 + x862 + x863 + x864 + x865 + x866
        x868 = x319*x809
        x869 = x274*x45
        x870 = x274*x46
        x871 = r_11*x325
        x872 = x274*x44
        x873 = r_11**3
        x874 = 2*x58
        x875 = x873*x874
        x876 = x214*x811
        x877 = x274*x67
        x878 = x274*x68
        x879 = x274*x71
        x880 = x274*x72
        x881 = x274*x62
        x882 = x274*x64
        x883 = x274*x66
        x884 = x274*x70
        x885 = x392*x781
        x886 = x392*x782
        x887 = Px*x59
        x888 = 4*x93
        x889 = x887*x888
        x890 = Px*x888
        x891 = x61*x890
        x892 = x63*x890
        x893 = x575*x887
        x894 = Px*x575
        x895 = x61*x894
        x896 = x63*x894
        x897 = x14*x851
        x898 = d_4*x25
        x899 = x360*x898
        x900 = r_21*x65
        x901 = r_12*x3
        x902 = x900*x901
        x903 = r_31*x282
        x904 = x363*x903
        x905 = r_23*x900
        x906 = x832*x905
        x907 = r_31*x441
        x908 = x832*x907
        x909 = Py*x3
        x910 = x798*x909
        x911 = x25*x53
        x912 = x360*x911
        x913 = x21*x93
        x914 = x360*x913
        x915 = x314*x830
        x916 = x794*x909
        x917 = x433*x781
        x918 = 4*x53
        x919 = x782*x918
        x920 = -x868 - x869 - x870 - x871 + x872 + x875 - x876 - x877 - x878 - x879 - x880 + x881 + x882 + x883 + x884 - x885 - x886 + x889 + x891 + x892 + x893 + x895 + x896 + x897 + x899 + x902 + x904 + x906 + x908 - x910 - x912 + x914 + x915 + x916 + x917 + x919
        x921 = R_l_inv_26*x470 + 4*x219
        x922 = x785*x921
        x923 = x363*x45
        x924 = x363*x46
        x925 = x250*x363
        x926 = x363*x44
        x927 = r_12**3
        x928 = 4*x58
        x929 = x927*x928
        x930 = 4*x818
        x931 = x214*x930
        x932 = x363*x66
        x933 = x363*x68
        x934 = x363*x70
        x935 = x363*x72
        x936 = x363*x60
        x937 = x363*x64
        x938 = x363*x67
        x939 = x363*x71
        x940 = x462*x805
        x941 = x462*x806
        x942 = 8*x36
        x943 = x887*x942
        x944 = Px*x942
        x945 = x61*x944
        x946 = x63*x944
        x947 = x729*x887
        x948 = Px*x729
        x949 = x61*x948
        x950 = x63*x948
        x951 = Py*x744
        x952 = x14*x951
        x953 = x744*x898
        x954 = r_22*x65
        x955 = 8*x954
        x956 = x745*x955
        x957 = r_11*r_31*x751
        x958 = r_13*r_23
        x959 = x955*x958
        x960 = r_13*r_33
        x961 = x751*x960
        x962 = x744*x913
        x963 = x744*x911
        x964 = 8*x93
        x965 = x798*x964
        x966 = x731*x796
        x967 = x23*x744
        x968 = x36*x967
        x969 = x737*x805
        x970 = 8*x53
        x971 = x806*x970
        x972 = -d_4*x778 - x180*x805 - x180*x806 - x211*x360 + x220*x777 + x25*x778 + x5*x951 - x809*(R_l_inv_27*x470 + 4*x213)
        x973 = x842 - x843 + x845 + x846 + x847 + x848 + x849 + x850 - x852 - x853
        x974 = x868 + x869 + x870 + x871 - x872 - x875 + x876 + x877 + x878 + x879 + x880 - x881 - x882 - x883 - x884 + x885 + x886 - x889 - x891 - x892 - x893 - x895 - x896 - x897 - x899 - x902 - x904 - x906 - x908 + x910 + x912 - x914 - x915 - x916 - x917 - x919
        x975 = x220*x775
        x976 = a_3*x835
        x977 = x467*x783
        x978 = x467*x784
        x979 = Py*r_13
        x980 = 8*x4*x979
        x981 = x467*x782
        x982 = r_13**3
        x983 = Px*x462
        x984 = Px*x970
        x985 = Px*x737
        x986 = x65*x745
        x987 = 8*r_11
        x988 = r_33*x282
        x989 = d_4*x744
        x990 = r_13*x25
        x991 = x36*x744
        x992 = -r_23*x744*x954 - 8*r_23*x986 + x21*x835 - x21*x836 + x21*x837 + x214*x833 + x23*x989 - x25*x991 + x250*x832 + x36*x989 + x44*x832 + x45*x832 - x46*x832 + x462*x990 - x53*x967 + x59*x983 - x60*x832 + x61*x983 - x61*x984 - x61*x985 - x62*x832 + x63*x983 - x63*x984 - x63*x985 + x66*x832 + x67*x832 - x68*x832 + x70*x832 + x71*x832 - x72*x832 + x723*x979 + x729*x805 - x737*x887 - x744*x988 - x831*x921 + x835*x93 - x838*x93 - x887*x970 - x907*x987 - x928*x982 - x970*x990
        x993 = x855 - x857 - x858 - x859 - x860 - x862 - x863 - x864 - x865 - x866
        x994 = x601*x809
        x995 = r_11*x603
        x996 = x502*x811
        x997 = d_2*x824
        x998 = Py*x363
        x999 = x5*x998
        x1000 = x25*x930
        x1001 = x34*x979
        x1002 = x158*x806
        x1003 = -r_11*x861 + x21*x775 - x23*x814 + x4*x998 + x502*x790
        x1004 = -x1000 + x1001 + x1002 + x1003 + x994 + x995 + x996 + x997 - x999
        x1005 = x559*x785
        x1006 = x523*x772
        x1007 = x874*x927
        x1008 = x44*x772
        x1009 = x45*x772
        x1010 = x46*x772
        x1011 = x60*x772
        x1012 = x64*x772
        x1013 = x67*x772
        x1014 = x71*x772
        x1015 = x512*x788
        x1016 = x66*x772
        x1017 = x68*x772
        x1018 = x70*x772
        x1019 = x72*x772
        x1020 = x887*x909
        x1021 = Px*x909
        x1022 = x1021*x61
        x1023 = x1021*x63
        x1024 = x314*x887
        x1025 = Px*x314
        x1026 = x1025*x61
        x1027 = x1025*x63
        x1028 = x14*x998
        x1029 = x25*x824
        x1030 = x3*x986
        x1031 = x360*x903
        x1032 = r_23*x65
        x1033 = x1032*x439
        x1034 = x832*x988
        x1035 = d_4*x825
        x1036 = x392*x806
        x1037 = x360*x93
        x1038 = x1037*x23
        x1039 = r_11*x21*x909
        x1040 = Py*x23*x901
        x1041 = x25*x825
        x1042 = x53*x826
        x1043 = x575*x830
        x1044 = x25*x827
        x1045 = x1005 - x1006 - x1007 + x1008 + x1009 + x1010 - x1011 - x1012 - x1013 - x1014 + x1015 + x1016 + x1017 + x1018 + x1019 - x1020 - x1022 - x1023 - x1024 - x1026 - x1027 - x1028 - x1029 - x1030 - x1031 - x1033 - x1034 + x1035 + x1036 - x1038 - x1039 - x1040 - x1041 - x1042 + x1043 + x1044
        x1046 = R_l_inv_36*x470 + 4*x511
        x1047 = x1046*x785
        x1048 = x363*x509
        x1049 = d_2*x835
        x1050 = x502*x930
        x1051 = x180*x781
        x1052 = x180*x782
        x1053 = Py*x5
        x1054 = x1053*x987
        x1055 = x180*x784
        x1056 = x360*x46
        x1057 = x873*x928
        x1058 = x360*x67
        x1059 = x360*x68
        x1060 = x360*x71
        x1061 = x360*x72
        x1062 = x360*x62
        x1063 = x360*x64
        x1064 = x360*x66
        x1065 = x360*x70
        x1066 = x462*x781
        x1067 = x462*x782
        x1068 = x887*x964
        x1069 = Px*x964
        x1070 = x1069*x61
        x1071 = x1069*x63
        x1072 = x731*x887
        x1073 = Px*x731
        x1074 = x1073*x61
        x1075 = x1073*x63
        x1076 = Py*r_11*x723
        x1077 = x25*x835
        x1078 = r_22*x744*x900
        x1079 = x744*x903
        x1080 = 8*r_13
        x1081 = x1080*x905
        x1082 = x1080*x907
        x1083 = x729*x796
        x1084 = x25*x836
        x1085 = x913*x987
        x1086 = x93*x967
        x1087 = x21*x991
        x1088 = x25*x837
        x1089 = x53*x839
        x1090 = -x1056 + x1057 - x1058 - x1059 - x1060 - x1061 + x1062 + x1063 + x1064 + x1065 - x1066 - x1067 + x1068 + x1070 + x1071 + x1072 + x1074 + x1075 + x1076 + x1077 + x1078 + x1079 + x1081 + x1082 - x1083 - x1084 + x1085 + x1086 + x1087 + x1088 + x1089
        x1091 = x360*x44
        x1092 = x360*x45
        x1093 = -x1091 - x1092
        x1094 = x1090 + x1093 + x360*x523 - x512*x777 + x809*(R_l_inv_37*x470 + 4*x501)
        x1095 = x1000 - x1001 - x1002 + x1003 - x994 - x995 - x996 - x997 + x999
        x1096 = -x1005 + x1006 + x1007 - x1008 - x1009 - x1010 + x1011 + x1012 + x1013 + x1014 - x1015 - x1016 - x1017 - x1018 - x1019 + x1020 + x1022 + x1023 + x1024 + x1026 + x1027 + x1028 + x1029 + x1030 + x1031 + x1033 + x1034 - x1035 - x1036 + x1038 + x1039 + x1040 + x1041 + x1042 - x1043 - x1044
        x1097 = a_3*x778
        x1098 = x512*x775
        x1099 = x1046*x831 + x180*x796 + x180*x798 - x21*x778 - x502*x833 - x509*x832 - x778*x93
        x1100 = Px*x158
        x1101 = -x1100*x59
        x1102 = -x1100*x61
        x1103 = -x1100*x63
        x1104 = a_3*x777
        x1105 = d_4*x833
        x1106 = -x777*x93
        x1107 = -x818*x909
        x1108 = -x1053*x832
        x1109 = -x21*x777
        x1110 = -x23*x930
        x1111 = -x25*x833
        x1112 = x1101 + x1102 + x1103 - x1104 + x1105 + x1106 + x1107 + x1108 + x1109 + x1110 + x1111
        x1113 = d_4*x814
        x1114 = Py*x832
        x1115 = x1114*x4
        x1116 = x133*x782
        x1117 = x53*x814
        x1118 = x25*x814
        x1119 = -x1113 - x1115 - x1116 + x1117 + x1118
        x1120 = x278*x44
        x1121 = x278*x45
        x1122 = x278*x46
        x1123 = x874*x982
        x1124 = Px*x392
        x1125 = x1124*x59
        x1126 = x1124*x61
        x1127 = x1124*x63
        x1128 = x278*x66
        x1129 = x278*x67
        x1130 = x278*x70
        x1131 = x278*x71
        x1132 = x278*x60
        x1133 = x278*x62
        x1134 = x278*x68
        x1135 = x278*x72
        x1136 = d_4*x360
        x1137 = x1136*x93
        x1138 = x804*x909
        x1139 = x1114*x14
        x1140 = x1136*x21
        x1141 = x23*x824
        x1142 = x832*x898
        x1143 = x887*x918
        x1144 = Px*x918
        x1145 = x1144*x61
        x1146 = x1144*x63
        x1147 = x433*x887
        x1148 = Px*x433
        x1149 = x1148*x61
        x1150 = x1148*x63
        x1151 = x1032*x361
        x1152 = x360*x907
        x1153 = x1032*x901
        x1154 = x363*x988
        x1155 = x575*x781
        x1156 = x806*x909
        x1157 = x1037*x25
        x1158 = x21*x360*x53
        x1159 = x808*x909
        x1160 = x23*x827
        x1161 = x832*x911
        x1162 = -x1120 + x1121 + x1122 + x1123 - x1125 - x1126 - x1127 - x1128 - x1129 - x1130 - x1131 + x1132 + x1133 + x1134 + x1135 - x1137 - x1138 - x1139 - x1140 - x1141 - x1142 + x1143 + x1145 + x1146 + x1147 + x1149 + x1150 + x1151 + x1152 + x1153 + x1154 - x1155 - x1156 + x1157 + x1158 + x1159 + x1160 + x1161
        x1163 = -x818
        x1164 = x1113 + x1115 + x1116 - x1117 - x1118
        x1165 = x1101 + x1102 + x1103 + x1104 + x1105 + x1106 + x1107 + x1108 + x1109 + x1110 + x1111
        x1166 = r_13*x467
        x1167 = Px*x467
        x1168 = Py*x4*x987 + a_3*x967 - d_4*x1166 + x1166*x25 + x1166*x53 + x1167*x59 + x1167*x61 + x1167*x63 + x21*x840 + x28*x951
        x1169 = 16*d_4
        x1170 = 16*x25
        x1171 = 16*x21
        x1172 = 16*x36
        x1173 = Px*x1172
        x1174 = 16*x23
        x1175 = Px*x1174
        x1176 = 16*x954
        x1177 = x1120 - x1121 - x1122 - x1123 + x1125 + x1126 + x1127 + x1128 + x1129 + x1130 + x1131 - x1132 - x1133 - x1134 - x1135 + x1137 + x1138 + x1139 + x1140 + x1141 + x1142 - x1143 - x1145 - x1146 - x1147 - x1149 - x1150 - x1151 - x1152 - x1153 - x1154 + x1155 + x1156 - x1157 - x1158 - x1159 - x1160 - x1161
        x1178 = -x11 - x13 + x14 - x16 - x18 - x20 - x22 - x24 - x26 - x9
        x1179 = x1178 + x30
        x1180 = -x29
        x1181 = x128 + x162 + x90
        x1182 = x101 + x103 + x111 + x113 + x28
        x1183 = a_1 + x152 + x39 + x41 + x75 + x96
        x1184 = x138 + x140 - x141
        x1185 = x151 + x43 + x84 + x88
        x1186 = x100 + x104 + x112 + x114 + x98
        x1187 = -x161 - x163 - x164
        x1188 = x116 + x117 + x119 + x121 + x184 + x90
        x1189 = x217 + x223
        x1190 = x1189 + x381
        x1191 = x308 + x309 + x310 - x311 + x313 + x315 - x316 - x317
        x1192 = x1189 + x212 + x216 + x222 + x226 + x228 + x231 + x232 + x233
        x1193 = -x399 + x401 + x403 - x405 - x407 - x408 + x410 + x412 + x414 + x416 - x417 - x419 - x421 - x423 - x425 - x426 - x427 - x428 + x430 + x431 + x432 + x434 + x435 + x436 + x438 + x440 + x442 + x444 - x445 - x446 - x447 - x448 - x449 - x450 + x451 + x453 + x454 + x456 + x457 - x458 - x459
        x1194 = -x503 + x504 + x505 - x506 - x507
        x1195 = x1194 + x516 + x517 + x615 + x616 + x617 + x618 + x619 + x620
        x1196 = -x560 - x561 + x562 + x718 + x762
        x1197 = x1194 + x520
        x1198 = x632 + x633 + x634 + x635 + x636 - x637 - x638
        x1199 = x663 + x664 + x665 - x666 + x667 + x668 + x669 + x670 + x671 + x672
        x1200 = x1199 + x765
        x1201 = x706 - x707 - x708 - x709 - x710 - x712 - x713 - x714 - x715 - x716
        x1202 = x1199 + x700
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = x0
        A[0, 2] = x0
        A[0, 3] = x2
        A[0, 4] = -x3
        A[0, 5] = x1
        A[0, 6] = r_23
        A[0, 8] = r_23
        A[1, 0] = x27 + x7
        A[1, 1] = x29
        A[1, 2] = x31 + x6
        A[1, 3] = x33
        A[1, 4] = -x34
        A[1, 5] = x32
        A[1, 6] = x27 + x35
        A[1, 7] = x29
        A[1, 8] = x31 + x5
        A[2, 0] = x115 + x129 + x97
        A[2, 1] = x136 + x142 + x149
        A[2, 2] = x129 + x150 + x153
        A[2, 3] = x160 + x165 + x172
        A[2, 4] = x173 - x174 - x175 + x176 + x177 - x179 + x181
        A[2, 5] = x165 + x182 + x183
        A[2, 6] = x150 + x185 + x97
        A[2, 7] = x142 + x186 + x187
        A[2, 8] = x115 + x153 + x185
        A[3, 0] = x210 + x234 + x299
        A[3, 1] = x306 + x318 + x377
        A[3, 2] = x380 + x382 + x384
        A[3, 3] = x390 + x398 + x460
        A[3, 4] = -x461 + x463 - x465 - x466 + x468 + x469 + x472
        A[3, 5] = x460 + x473 + x474
        A[3, 6] = x234 + x476 + x478
        A[3, 7] = x318 + x479 + x480
        A[3, 8] = x382 + x481 + x482
        A[4, 0] = x500 + x521 + x552
        A[4, 1] = x558 + x600 + x612
        A[4, 2] = x614 + x621 + x622
        A[4, 3] = x628 + x631 + x639
        A[4, 4] = x640 + x641 + x642
        A[4, 5] = x639 + x643 + x644
        A[4, 6] = x621 + x646 + x647
        A[4, 7] = x600 + x648 + x649
        A[4, 8] = x521 + x650 + x651
        A[5, 0] = x662 + x701
        A[5, 1] = x133*(x702 + x703)
        A[5, 2] = x701 + x705
        A[5, 3] = x717 + x719
        A[5, 4] = x720 + x721 - x722 - x724 + x726 + x727 - x728 + x730 - x732 + x734 + x735 + x736 - x738 + x739 + x740 + x741 + x742 + x743 + x746 + x747 + x748 + x749 + x750 + x752 + x753 - x754 + x755 - x756 + x757 + x758 - x759 + x760 - x761
        A[5, 5] = x717 + x763
        A[5, 6] = x764 + x766
        A[5, 7] = x133*(x110 + x702)
        A[5, 8] = x766 + x767
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x768
        B[0, 2] = x768
        B[0, 3] = -x360
        B[0, 4] = -x744
        B[0, 5] = x360
        B[0, 6] = x278
        B[0, 8] = x278
        B[1, 0] = x769 + x774
        B[1, 1] = x775
        B[1, 2] = x774 + x776
        B[1, 3] = -x777
        B[1, 4] = -x778
        B[1, 5] = x777
        B[1, 6] = x769 + x779
        B[1, 7] = x775
        B[1, 8] = x776 + x779
        B[2, 0] = x792 + x800 + x813
        B[2, 1] = x814 - x815 + x817 - x820 + x821
        B[2, 2] = x800 + x822 + x823
        B[2, 3] = -x824 - x825 - x826 + x827 + x828 + x829 + x834
        B[2, 4] = x835 - x836 + x837 - x838 + x839 - x840*x86
        B[2, 5] = x824 + x825 + x826 - x827 - x828 - x829 + x834
        B[2, 6] = x792 + x822 + x841
        B[2, 7] = -x814 + x815 - x817 + x820 + x821
        B[2, 8] = x813 + x823 + x841
        B[3, 0] = x854 + x867 + x920
        B[3, 1] = x922 - x923 - x924 - x925 + x926 + x929 - x931 - x932 - x933 - x934 - x935 + x936 + x937 + x938 + x939 - x940 - x941 + x943 + x945 + x946 + x947 + x949 + x950 + x952 + x953 + x956 + x957 + x959 + x961 - x962 - x963 + x965 + x966 + x968 + x969 + x971 + x972
        B[3, 2] = x867 + x973 + x974
        B[3, 3] = x975 + x976 - x977 - x978 + x980 + x981 + x992
        B[3, 4] = x467*(-r_11*x220 + x23*x278 + x278*x36 + 2*x804 - 2*x807 - 2*x808)
        B[3, 5] = -x975 - x976 + x977 + x978 - x980 - x981 + x992
        B[3, 6] = x854 + x974 + x993
        B[3, 7] = -x922 + x923 + x924 + x925 - x926 - x929 + x931 + x932 + x933 + x934 + x935 - x936 - x937 - x938 - x939 + x940 + x941 - x943 - x945 - x946 - x947 - x949 - x950 - x952 - x953 - x956 - x957 - x959 - x961 + x962 + x963 - x965 - x966 - x968 - x969 - x971 + x972
        B[3, 8] = x920 + x973 + x993
        B[4, 0] = x1004 + x1045
        B[4, 1] = -x1047 + x1048 - x1049 + x1050 - x1051 - x1052 + x1054 + x1055 + x1094
        B[4, 2] = x1095 + x1096
        B[4, 3] = -x1097 - x1098 + x1099
        B[4, 4] = x840*(x125 + x512)
        B[4, 5] = x1097 + x1098 + x1099
        B[4, 6] = x1045 + x1095
        B[4, 7] = x1047 - x1048 + x1049 - x1050 + x1051 + x1052 - x1054 - x1055 + x1094
        B[4, 8] = x1004 + x1096
        B[5, 0] = x1112 + x1119 + x1162
        B[5, 1] = x467*(x1163 - x804 - x805 - x806 + x807 + x808)
        B[5, 2] = x1162 + x1164 + x1165
        B[5, 3] = x1090 + x1091 + x1092 + x1168
        B[5, 4] = 16*Py*r_12*x14 + 16*r_11*x903 + r_12*x1172*x23 - x1169*x805 - x1169*x806 + x1170*x804 + x1170*x805 - x1170*x807 + x1171*x796 - x1171*x830 + x1172*x887 + x1173*x61 + x1173*x63 + x1174*x887 + x1175*x61 + x1175*x63 + x1176*x745 + x1176*x958 + 16*x282*x960 + x44*x744 + x45*x744 - x46*x744 + 16*x53*x806 + 8*x58*x927 + x60*x744 + x64*x744 - x66*x744 + x67*x744 - x68*x744 - x70*x744 + x71*x744 - x72*x744 + 16*x798*x93
        B[5, 5] = x1056 - x1057 + x1058 + x1059 + x1060 + x1061 - x1062 - x1063 - x1064 - x1065 + x1066 + x1067 - x1068 - x1070 - x1071 - x1072 - x1074 - x1075 - x1076 - x1077 - x1078 - x1079 - x1081 - x1082 + x1083 + x1084 - x1085 - x1086 - x1087 - x1088 - x1089 + x1093 + x1168
        B[5, 6] = x1112 + x1164 + x1177
        B[5, 7] = x467*(x1163 + x809)
        B[5, 8] = x1119 + x1165 + x1177
        C = np.zeros(shape=(6, 9))
        C[0, 0] = r_23
        C[0, 2] = r_23
        C[0, 3] = x1
        C[0, 4] = x3
        C[0, 5] = x2
        C[0, 6] = x0
        C[0, 8] = x0
        C[1, 0] = x1179 + x5
        C[1, 1] = x1180
        C[1, 2] = x1178 + x35
        C[1, 3] = x32
        C[1, 4] = x34
        C[1, 5] = x33
        C[1, 6] = x1179 + x6
        C[1, 7] = x1180
        C[1, 8] = x1178 + x7
        C[2, 0] = x1181 + x1182 + x1183
        C[2, 1] = x1184 + x136 + x187
        C[2, 2] = x1181 + x1185 + x1186
        C[2, 3] = x1187 + x160 + x183
        C[2, 4] = -x173 + x174 + x175 - x176 - x177 + x179 + x181
        C[2, 5] = x1187 + x172 + x182
        C[2, 6] = x1183 + x1186 + x1188
        C[2, 7] = x1184 + x149 + x186
        C[2, 8] = x1182 + x1185 + x1188
        C[3, 0] = x1190 + x210 + x478
        C[3, 1] = x1191 + x306 + x480
        C[3, 2] = x1192 + x380 + x481
        C[3, 3] = x1193 + x390 + x474
        C[3, 4] = x461 - x463 + x465 + x466 - x468 - x469 + x472
        C[3, 5] = x1193 + x398 + x473
        C[3, 6] = x1190 + x299 + x476
        C[3, 7] = x1191 + x377 + x479
        C[3, 8] = x1192 + x384 + x482
        C[4, 0] = x1195 + x500 + x622
        C[4, 1] = x1196 + x558 + x649
        C[4, 2] = x1197 + x552 + x614
        C[4, 3] = x1198 + x628 + x643
        C[4, 4] = -x640 - x641 + x642
        C[4, 5] = x1198 + x631 + x644
        C[4, 6] = x1197 + x647 + x650
        C[4, 7] = x1196 + x612 + x648
        C[4, 8] = x1195 + x646 + x651
        C[5, 0] = x1200 + x705
        C[5, 1] = x133*(x110 + x85)
        C[5, 2] = x1200 + x662
        C[5, 3] = x1201 + x763
        C[5, 4] = -x720 - x721 + x722 + x724 - x726 - x727 + x728 - x730 + x732 - x734 - x735 - x736 + x738 - x739 - x740 - x741 - x742 - x743 - x746 - x747 - x748 - x749 - x750 - x752 - x753 + x754 - x755 + x756 - x757 - x758 + x759 - x760 + x761
        C[5, 5] = x1201 + x719
        C[5, 6] = x1202 + x767
        C[5, 7] = x133*(x703 + x85)
        C[5, 8] = x1202 + x764
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
            th_0 = this_solution[0]
            checked_result: bool = (abs(a_3*r_11*math.sin(th_0) - a_3*r_21*math.cos(th_0)) <= 1.0e-6) and (abs(a_3*r_12*math.sin(th_0) - a_3*r_22*math.cos(th_0)) <= 1.0e-6) and (abs(Px*math.sin(th_0) - Py*math.cos(th_0) - d_4*r_13*math.sin(th_0) + d_4*r_23*math.cos(th_0)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor()
    # Finish code for equation all-zero dispatcher node 2
    
    # Code for explicit solution node 3, solved variable is th_5
    def ExplicitSolutionNode_node_3_solve_th_5_processor():
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
            th_0 = this_solution[0]
            condition_0: bool = (abs(a_3*r_11*math.sin(th_0) - a_3*r_21*math.cos(th_0)) >= zero_tolerance) or (abs(a_3*r_12*math.sin(th_0) - a_3*r_22*math.cos(th_0)) >= zero_tolerance) or (abs(Px*math.sin(th_0) - Py*math.cos(th_0) - d_4*r_13*math.sin(th_0) + d_4*r_23*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = a_3*x0
                x2 = math.cos(th_0)
                x3 = a_3*x2
                x4 = r_12*x1 - r_22*x3
                x5 = -r_11*x1 + r_21*x3
                x6 = math.atan2(x4, x5)
                x7 = -Px*x0 + Py*x2 + d_4*r_13*x0 - d_4*r_23*x2
                x8 = math.sqrt(x4**2 + x5**2 - x7**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[7] = x6 + math.atan2(x8, x7)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs(a_3*r_11*math.sin(th_0) - a_3*r_21*math.cos(th_0)) >= zero_tolerance) or (abs(a_3*r_12*math.sin(th_0) - a_3*r_22*math.cos(th_0)) >= zero_tolerance) or (abs(Px*math.sin(th_0) - Py*math.cos(th_0) - d_4*r_13*math.sin(th_0) + d_4*r_23*math.cos(th_0)) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = a_3*x0
                x2 = math.cos(th_0)
                x3 = a_3*x2
                x4 = r_12*x1 - r_22*x3
                x5 = -r_11*x1 + r_21*x3
                x6 = math.atan2(x4, x5)
                x7 = -Px*x0 + Py*x2 + d_4*r_13*x0 - d_4*r_23*x2
                x8 = math.sqrt(x4**2 + x5**2 - x7**2)
                # End of temp variables
                this_solution[7] = x6 + math.atan2(-x8, x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_5_processor()
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
            condition_0: bool = (abs((-r_11*math.sin(th_0) + r_21*math.cos(th_0))*math.sin(th_5) + (-r_12*math.sin(th_0) + r_22*math.cos(th_0))*math.cos(th_5)) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = math.acos((-r_11*x1 + r_21*x0)*math.sin(th_5) + (-r_12*x1 + r_22*x0)*math.cos(th_5))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[5] = x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                
            condition_1: bool = (abs((-r_11*math.sin(th_0) + r_21*math.cos(th_0))*math.sin(th_5) + (-r_12*math.sin(th_0) + r_22*math.cos(th_0))*math.cos(th_5)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = math.acos((-r_11*x1 + r_21*x0)*math.sin(th_5) + (-r_12*x1 + r_22*x0)*math.cos(th_5))
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
            th_5 = this_solution[7]
            condition_0: bool = (2*abs(a_0*a_1) >= zero_tolerance) or (2*abs(a_0*d_2) >= zero_tolerance) or (abs(Px**2 - 2*Px*a_3*r_11*math.cos(th_5) + 2*Px*a_3*r_12*math.sin(th_5) - 2*Px*d_4*r_13 + Py**2 - 2*Py*a_3*r_21*math.cos(th_5) + 2*Py*a_3*r_22*math.sin(th_5) - 2*Py*d_4*r_23 + Pz**2 - 2*Pz*a_3*r_31*math.cos(th_5) + 2*Pz*a_3*r_32*math.sin(th_5) - 2*Pz*d_4*r_33 - a_0**2 - a_1**2 + a_3**2*r_11**2*math.cos(th_5)**2 - a_3**2*r_11*r_12*math.sin(2*th_5) - a_3**2*r_12**2*math.cos(th_5)**2 + a_3**2*r_12**2 + a_3**2*r_21**2*math.cos(th_5)**2 - a_3**2*r_21*r_22*math.sin(2*th_5) - a_3**2*r_22**2*math.cos(th_5)**2 + a_3**2*r_22**2 + a_3**2*r_31**2*math.cos(th_5)**2 - a_3**2*r_31*r_32*math.sin(2*th_5) - a_3**2*r_32**2*math.cos(th_5)**2 + a_3**2*r_32**2 + 2*a_3*d_4*r_11*r_13*math.cos(th_5) - 2*a_3*d_4*r_12*r_13*math.sin(th_5) + 2*a_3*d_4*r_21*r_23*math.cos(th_5) - 2*a_3*d_4*r_22*r_23*math.sin(th_5) + 2*a_3*d_4*r_31*r_33*math.cos(th_5) - 2*a_3*d_4*r_32*r_33*math.sin(th_5) - d_2**2 + d_4**2*r_13**2 + d_4**2*r_23**2 + d_4**2*r_33**2) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*a_0
                x1 = math.atan2(-d_2*x0, a_1*x0)
                x2 = a_1**2
                x3 = a_0**2
                x4 = 4*x3
                x5 = d_2**2
                x6 = 2*d_4
                x7 = r_13*x6
                x8 = r_23*x6
                x9 = r_33*x6
                x10 = a_3**2
                x11 = r_12**2*x10
                x12 = r_22**2*x10
                x13 = r_32**2*x10
                x14 = d_4**2
                x15 = math.cos(th_5)
                x16 = 2*a_3
                x17 = x15*x16
                x18 = math.sin(th_5)
                x19 = x16*x18
                x20 = a_3*x15
                x21 = a_3*x18
                x22 = x15**2
                x23 = x10*x22
                x24 = x10*math.sin(2*th_5)
                x25 = Px**2 - Px*r_11*x17 + Px*r_12*x19 - Px*x7 + Py**2 - Py*r_21*x17 + Py*r_22*x19 - Py*x8 + Pz**2 - Pz*r_31*x17 + Pz*r_32*x19 - Pz*x9 + r_11**2*x23 - r_11*r_12*x24 + r_11*x20*x7 - r_12*x21*x7 + r_13**2*x14 + r_21**2*x23 - r_21*r_22*x24 + r_21*x20*x8 - r_22*x21*x8 + r_23**2*x14 + r_31**2*x23 - r_31*r_32*x24 + r_31*x20*x9 - r_32*x21*x9 + r_33**2*x14 - x11*x22 + x11 - x12*x22 + x12 - x13*x22 + x13 - x2 - x3 - x5
                x26 = math.sqrt(x2*x4 - x25**2 + x4*x5)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x1 + math.atan2(x26, x25)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
            condition_1: bool = (2*abs(a_0*a_1) >= zero_tolerance) or (2*abs(a_0*d_2) >= zero_tolerance) or (abs(Px**2 - 2*Px*a_3*r_11*math.cos(th_5) + 2*Px*a_3*r_12*math.sin(th_5) - 2*Px*d_4*r_13 + Py**2 - 2*Py*a_3*r_21*math.cos(th_5) + 2*Py*a_3*r_22*math.sin(th_5) - 2*Py*d_4*r_23 + Pz**2 - 2*Pz*a_3*r_31*math.cos(th_5) + 2*Pz*a_3*r_32*math.sin(th_5) - 2*Pz*d_4*r_33 - a_0**2 - a_1**2 + a_3**2*r_11**2*math.cos(th_5)**2 - a_3**2*r_11*r_12*math.sin(2*th_5) - a_3**2*r_12**2*math.cos(th_5)**2 + a_3**2*r_12**2 + a_3**2*r_21**2*math.cos(th_5)**2 - a_3**2*r_21*r_22*math.sin(2*th_5) - a_3**2*r_22**2*math.cos(th_5)**2 + a_3**2*r_22**2 + a_3**2*r_31**2*math.cos(th_5)**2 - a_3**2*r_31*r_32*math.sin(2*th_5) - a_3**2*r_32**2*math.cos(th_5)**2 + a_3**2*r_32**2 + 2*a_3*d_4*r_11*r_13*math.cos(th_5) - 2*a_3*d_4*r_12*r_13*math.sin(th_5) + 2*a_3*d_4*r_21*r_23*math.cos(th_5) - 2*a_3*d_4*r_22*r_23*math.sin(th_5) + 2*a_3*d_4*r_31*r_33*math.cos(th_5) - 2*a_3*d_4*r_32*r_33*math.sin(th_5) - d_2**2 + d_4**2*r_13**2 + d_4**2*r_23**2 + d_4**2*r_33**2) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*a_0
                x1 = math.atan2(-d_2*x0, a_1*x0)
                x2 = a_1**2
                x3 = a_0**2
                x4 = 4*x3
                x5 = d_2**2
                x6 = 2*d_4
                x7 = r_13*x6
                x8 = r_23*x6
                x9 = r_33*x6
                x10 = a_3**2
                x11 = r_12**2*x10
                x12 = r_22**2*x10
                x13 = r_32**2*x10
                x14 = d_4**2
                x15 = math.cos(th_5)
                x16 = 2*a_3
                x17 = x15*x16
                x18 = math.sin(th_5)
                x19 = x16*x18
                x20 = a_3*x15
                x21 = a_3*x18
                x22 = x15**2
                x23 = x10*x22
                x24 = x10*math.sin(2*th_5)
                x25 = Px**2 - Px*r_11*x17 + Px*r_12*x19 - Px*x7 + Py**2 - Py*r_21*x17 + Py*r_22*x19 - Py*x8 + Pz**2 - Pz*r_31*x17 + Pz*r_32*x19 - Pz*x9 + r_11**2*x23 - r_11*r_12*x24 + r_11*x20*x7 - r_12*x21*x7 + r_13**2*x14 + r_21**2*x23 - r_21*r_22*x24 + r_21*x20*x8 - r_22*x21*x8 + r_23**2*x14 + r_31**2*x23 - r_31*r_32*x24 + r_31*x20*x9 - r_32*x21*x9 + r_33**2*x14 - x11*x22 + x11 - x12*x22 + x12 - x13*x22 + x13 - x2 - x3 - x5
                x26 = math.sqrt(x2*x4 - x25**2 + x4*x5)
                # End of temp variables
                this_solution[4] = x1 + math.atan2(-x26, x25)
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
            degenerate_valid_0 = (abs(th_2 - math.pi + 1.34089189866299) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(22, node_input_i_idx_in_queue)
            
            th_2 = this_solution[4]
            degenerate_valid_1 = (abs(th_2 - 2*math.pi + 1.34089189866299) <= 1.0e-6)
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
            th_0 = this_solution[0]
            th_5 = this_solution[7]
            condition_0: bool = (abs(0.973688178796424*a_1 - 0.227884467377887*d_2) >= 1.0e-6) or (abs(a_0 + 0.227884467377887*a_1 + 0.973688178796424*d_2) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_0 + 0.227884467377887*a_1 + 0.973688178796424*d_2
                x1 = a_3*math.sin(th_5)
                x2 = a_3*math.cos(th_5)
                x3 = Pz - d_4*r_33 - r_31*x2 + r_32*x1
                x4 = 0.973688178796424*a_1 - 0.227884467377887*d_2
                x5 = math.cos(th_0)
                x6 = math.sin(th_0)
                x7 = Px*x5 + Py*x6 - d_4*r_13*x5 - d_4*r_23*x6 - r_11*x2*x5 + r_12*x1*x5 - r_21*x2*x6 + r_22*x1*x6
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x3 + x4*x7, x0*x7 + x3*x4)
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
            condition_0: bool = (1 >= zero_tolerance) or (abs(-r_13*((0.973688178796424*math.sin(th_1) + 0.227884467377886*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) - r_23*((0.973688178796424*math.sin(th_1) + 0.227884467377886*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(-0.227884467377886*math.sin(th_1) + 0.973688178796424*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance) or (abs(-r_13*(-0.227884467377887*math.sin(th_1) + 0.973688178796424*math.cos(th_1))*math.cos(th_0) - r_23*(-0.227884467377887*math.sin(th_1) + 0.973688178796424*math.cos(th_1))*math.sin(th_0) + r_33*(0.973688178796424*math.sin(th_1) + 0.227884467377887*math.cos(th_1))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_1)
                x3 = 0.973688178796424*x2
                x4 = math.sin(th_0)
                x5 = math.sin(th_3)
                x6 = math.cos(th_0)
                x7 = 0.973688178796424*x1
                x8 = x0*(0.227884467377886*x2 + x7)
                x9 = -0.227884467377887*x1 + x3
                # End of temp variables
                this_solution[6] = math.atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*(-0.227884467377886*x1 + x3), -r_13*x6*x9 - r_23*x4*x9 + r_33*(0.227884467377887*x2 + x7))
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
            th_0 = this_solution[0]
            th_5 = this_solution[7]
            condition_0: bool = (abs(0.973688178796424*a_1 - 0.227884467377887*d_2) >= 1.0e-6) or (abs(-a_0 + 0.227884467377887*a_1 + 0.973688178796424*d_2) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_0 - 0.227884467377887*a_1 - 0.973688178796424*d_2
                x1 = a_3*math.sin(th_5)
                x2 = a_3*math.cos(th_5)
                x3 = Pz - d_4*r_33 - r_31*x2 + r_32*x1
                x4 = -0.973688178796424*a_1 + 0.227884467377887*d_2
                x5 = math.cos(th_0)
                x6 = math.sin(th_0)
                x7 = Px*x5 + Py*x6 - d_4*r_13*x5 - d_4*r_23*x6 - r_11*x2*x5 + r_12*x1*x5 - r_21*x2*x6 + r_22*x1*x6
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x3 + x4*x7, x0*x7 + x3*x4)
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
            condition_0: bool = (1 >= zero_tolerance) or (abs(-r_13*((-0.973688178796424*math.sin(th_1) - 0.227884467377886*math.cos(th_1))*math.cos(th_0)*math.cos(th_3) + math.sin(th_0)*math.sin(th_3)) - r_23*((-0.973688178796424*math.sin(th_1) - 0.227884467377886*math.cos(th_1))*math.sin(th_0)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) - r_33*(0.227884467377886*math.sin(th_1) - 0.973688178796424*math.cos(th_1))*math.cos(th_3)) >= zero_tolerance) or (abs(-r_13*(-0.227884467377887*math.sin(th_1) + 0.973688178796424*math.cos(th_1))*math.cos(th_0) - r_23*(-0.227884467377887*math.sin(th_1) + 0.973688178796424*math.cos(th_1))*math.sin(th_0) + r_33*(0.973688178796424*math.sin(th_1) + 0.227884467377887*math.cos(th_1))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_1)
                x3 = 0.973688178796424*x2
                x4 = math.sin(th_0)
                x5 = math.sin(th_3)
                x6 = math.cos(th_0)
                x7 = 0.973688178796424*x1
                x8 = x0*(-0.227884467377886*x2 - x7)
                x9 = -0.227884467377887*x1 + x3
                # End of temp variables
                this_solution[6] = math.atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*(0.227884467377886*x1 - x3), r_13*x6*x9 + r_23*x4*x9 - r_33*(0.227884467377887*x2 + x7))
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
            condition_0: bool = (abs(a_1*math.sin(th_2) + d_2*math.cos(th_2)) >= 1.0e-6) or (abs(a_0 + a_1*math.cos(th_2) - d_2*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_2)
                x1 = math.cos(th_2)
                x2 = -a_1*x0 - d_2*x1
                x3 = Px*math.cos(th_0) + Py*math.sin(th_0) + a_3*math.cos(th_1th_2th_4_soa) - d_4*math.sin(th_1th_2th_4_soa)
                x4 = a_0 + a_1*x1 - d_2*x0
                x5 = Pz - a_3*r_31*math.cos(th_5) + a_3*r_32*math.sin(th_5) - d_4*r_33
                # End of temp variables
                this_solution[1] = math.atan2(x2*x3 - x4*x5, x2*x5 + x3*x4)
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
            condition_0: bool = (abs(a_1*math.sin(th_2) + d_2*math.cos(th_2)) >= 1.0e-6) or (abs(a_0 + a_1*math.cos(th_2) - d_2*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_2)
                x1 = math.sin(th_2)
                x2 = a_0 + a_1*x0 - d_2*x1
                x3 = a_3*math.sin(th_5)
                x4 = a_3*math.cos(th_5)
                x5 = Pz - d_4*r_33 - r_31*x4 + r_32*x3
                x6 = -a_1*x1 - d_2*x0
                x7 = math.cos(th_0)
                x8 = math.sin(th_0)
                x9 = Px*x7 + Py*x8 - d_4*r_13*x7 - d_4*r_23*x8 - r_11*x4*x7 + r_12*x3*x7 - r_21*x4*x8 + r_22*x3*x8
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
                x1 = math.cos(th_0)
                x2 = math.sin(th_0)
                # End of temp variables
                this_solution[6] = math.atan2(x0*(-r_13*x2 + r_23*x1), x0*(-(-r_11*x2 + r_21*x1)*math.cos(th_5) + (-r_12*x2 + r_22*x1)*math.sin(th_5)))
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


def abb_crb_15000_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = abb_crb_15000_ik_target_original_to_raw(T_ee)
    ik_output_raw = abb_crb_15000_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ee_pose_i = abb_crb_15000_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_abb_crb_15000():
    theta_in = np.random.random(size=(6, ))
    ee_pose = abb_crb_15000_fk(theta_in)
    ik_output = abb_crb_15000_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = abb_crb_15000_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_abb_crb_15000()
