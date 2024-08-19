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
a_0: float = 0.7
d_1: float = -0.5
d_2: float = -0.162
d_3: float = -0.13
pre_transform_special_symbol_23: float = 0.275

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -1.5707963267948966
th_2_offset_original2raw: float = 0.0
th_3_offset_original2raw: float = 3.141592653589793
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 3.141592653589793


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def yaskawa_HC10_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 1] = 1.0*r_12
    ee_transformed[0, 2] = -1.0*r_11
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = 1.0*r_23
    ee_transformed[1, 1] = 1.0*r_22
    ee_transformed[1, 2] = -1.0*r_21
    ee_transformed[1, 3] = 1.0*Py
    ee_transformed[2, 0] = 1.0*r_33
    ee_transformed[2, 1] = 1.0*r_32
    ee_transformed[2, 2] = -1.0*r_31
    ee_transformed[2, 3] = 1.0*Pz - 1.0*pre_transform_special_symbol_23
    return ee_transformed


def yaskawa_HC10_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = -1.0*r_13
    ee_transformed[0, 1] = 1.0*r_12
    ee_transformed[0, 2] = 1.0*r_11
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = -1.0*r_23
    ee_transformed[1, 1] = 1.0*r_22
    ee_transformed[1, 2] = 1.0*r_21
    ee_transformed[1, 3] = 1.0*Py
    ee_transformed[2, 0] = -1.0*r_33
    ee_transformed[2, 1] = 1.0*r_32
    ee_transformed[2, 2] = 1.0*r_31
    ee_transformed[2, 3] = 1.0*Pz + 1.0*pre_transform_special_symbol_23
    return ee_transformed


def yaskawa_HC10_fk(theta_input: np.ndarray):
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
    x8 = x1*x4 - x1*x7
    x9 = x0*x8
    x10 = math.sin(th_4)
    x11 = math.sin(th_0)
    x12 = math.sin(th_3)
    x13 = math.cos(th_3)
    x14 = x2*x5
    x15 = x3*x6
    x16 = x1*x14 + x1*x15
    x17 = -x11*x12 + x13*x16
    x18 = x10*x17
    x19 = math.cos(th_5)
    x20 = -1.0*x11*x13 - 1.0*x12*x16
    x21 = math.sin(th_5)
    x22 = 1.0*x0*x17 - 1.0*x10*x8
    x23 = 1.0*a_0
    x24 = x23*x6
    x25 = 1.0*d_1
    x26 = 1.0*d_3
    x27 = x11*x4 - x11*x7
    x28 = x0*x27
    x29 = x11*x14 + x11*x15
    x30 = x1*x12 + x13*x29
    x31 = x10*x30
    x32 = 1.0*x1*x13 - 1.0*x12*x29
    x33 = 1.0*x0*x30 - 1.0*x10*x27
    x34 = x14 + x15
    x35 = x0*x34
    x36 = -x4 + x7
    x37 = x13*x36
    x38 = x10*x37
    x39 = 1.0*x12*x36
    x40 = 1.0*x0*x37 - 1.0*x10*x34
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = 1.0*x18 + 1.0*x9
    ee_pose[0, 1] = -x19*x20 - x21*x22
    ee_pose[0, 2] = x19*x22 - x20*x21
    ee_pose[0, 3] = d_2*x20 + x1*x24 + x25*x8 + x26*(-x18 - x9)
    ee_pose[1, 0] = 1.0*x28 + 1.0*x31
    ee_pose[1, 1] = -x19*x32 - x21*x33
    ee_pose[1, 2] = x19*x33 - x21*x32
    ee_pose[1, 3] = d_2*x32 + x11*x24 + x25*x27 + x26*(-x28 - x31)
    ee_pose[2, 0] = 1.0*x35 + 1.0*x38
    ee_pose[2, 1] = x19*x39 - x21*x40
    ee_pose[2, 2] = x19*x40 + x21*x39
    ee_pose[2, 3] = -d_2*x39 + 1.0*pre_transform_special_symbol_23 - x2*x23 + x25*x34 + x26*(-x35 - x38)
    return ee_pose


def yaskawa_HC10_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = 1.0*x0
    x2 = math.sin(th_1)
    x3 = math.cos(th_2)
    x4 = math.cos(th_0)
    x5 = 1.0*x4
    x6 = x3*x5
    x7 = math.cos(th_1)
    x8 = math.sin(th_2)
    x9 = x5*x8
    x10 = x2*x6 - x7*x9
    x11 = math.cos(th_3)
    x12 = math.sin(th_3)
    x13 = x2*x9 + x6*x7
    x14 = -x1*x11 - x12*x13
    x15 = math.cos(th_4)
    x16 = math.sin(th_4)
    x17 = -x10*x15 - x16*(-x1*x12 + x11*x13)
    x18 = x1*x3
    x19 = x1*x8
    x20 = x18*x2 - x19*x7
    x21 = x18*x7 + x19*x2
    x22 = x11*x5 - x12*x21
    x23 = -x15*x20 - x16*(x11*x21 + x12*x5)
    x24 = 1.0*x2
    x25 = 1.0*x7
    x26 = x24*x8 + x25*x3
    x27 = -x24*x3 + x25*x8
    x28 = x12*x27
    x29 = -x11*x16*x27 - x15*x26
    x30 = -a_0*x24 + pre_transform_special_symbol_23
    x31 = a_0*x7
    x32 = d_1*x20 + x1*x31
    x33 = d_1*x26 + x30
    x34 = -d_2*x28 + x33
    x35 = d_2*x22 + x32
    x36 = d_3*x29 + x34
    x37 = d_3*x23 + x35
    x38 = d_1*x10 + x31*x5
    x39 = d_2*x14 + x38
    x40 = d_3*x17 + x39
    x41 = a_0*x25
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = -x1
    jacobian_output[0, 2] = x1
    jacobian_output[0, 3] = x10
    jacobian_output[0, 4] = x14
    jacobian_output[0, 5] = x17
    jacobian_output[1, 1] = x5
    jacobian_output[1, 2] = -x5
    jacobian_output[1, 3] = x20
    jacobian_output[1, 4] = x22
    jacobian_output[1, 5] = x23
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x26
    jacobian_output[2, 4] = -x28
    jacobian_output[2, 5] = x29
    jacobian_output[3, 1] = -pre_transform_special_symbol_23*x5
    jacobian_output[3, 2] = x30*x5
    jacobian_output[3, 3] = -x20*x33 + x26*x32
    jacobian_output[3, 4] = -x22*x34 - x28*x35
    jacobian_output[3, 5] = -x23*x36 + x29*x37
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = x1*x30
    jacobian_output[4, 3] = x10*x33 - x26*x38
    jacobian_output[4, 4] = x14*x34 + x28*x39
    jacobian_output[4, 5] = x17*x36 - x29*x40
    jacobian_output[5, 2] = -x0**2*x41 - x4**2*x41
    jacobian_output[5, 3] = -x10*x32 + x20*x38
    jacobian_output[5, 4] = -x14*x35 + x22*x39
    jacobian_output[5, 5] = -x17*x37 + x23*x40
    return jacobian_output


def yaskawa_HC10_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = 1.0*math.sin(th_0)
    x1 = math.sin(th_1)
    x2 = math.cos(th_2)
    x3 = 1.0*math.cos(th_0)
    x4 = x2*x3
    x5 = math.cos(th_1)
    x6 = math.sin(th_2)
    x7 = x3*x6
    x8 = x1*x4 - x5*x7
    x9 = math.cos(th_3)
    x10 = math.sin(th_3)
    x11 = x1*x7 + x4*x5
    x12 = math.cos(th_4)
    x13 = math.sin(th_4)
    x14 = x0*x2
    x15 = x0*x6
    x16 = x1*x14 - x15*x5
    x17 = x1*x15 + x14*x5
    x18 = 1.0*x1
    x19 = 1.0*x5
    x20 = x18*x6 + x19*x2
    x21 = -x18*x2 + x19*x6
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 1] = -x0
    jacobian_output[0, 2] = x0
    jacobian_output[0, 3] = x8
    jacobian_output[0, 4] = -x0*x9 - x10*x11
    jacobian_output[0, 5] = -x12*x8 - x13*(-x0*x10 + x11*x9)
    jacobian_output[1, 1] = x3
    jacobian_output[1, 2] = -x3
    jacobian_output[1, 3] = x16
    jacobian_output[1, 4] = -x10*x17 + x3*x9
    jacobian_output[1, 5] = -x12*x16 - x13*(x10*x3 + x17*x9)
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x20
    jacobian_output[2, 4] = -x10*x21
    jacobian_output[2, 5] = -x12*x20 - x13*x21*x9
    return jacobian_output


def yaskawa_HC10_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x13 = x12 + x8
    x14 = math.sin(th_0)
    x15 = x5*x9
    x16 = x11*x7
    x17 = x14*x15 - x14*x16
    x18 = a_0*x11
    x19 = d_1*x17 + x14*x18
    x20 = d_1*x13 + x6
    x21 = math.sin(th_3)
    x22 = -x15 + x16
    x23 = x21*x22
    x24 = math.cos(th_3)
    x25 = x12*x14 + x14*x8
    x26 = x2*x24 - x21*x25
    x27 = -d_2*x23 + x20
    x28 = d_2*x26 + x19
    x29 = math.cos(th_4)
    x30 = math.sin(th_4)
    x31 = -x13*x29 - x22*x24*x30
    x32 = -x17*x29 - x30*(x2*x21 + x24*x25)
    x33 = d_3*x31 + x27
    x34 = d_3*x32 + x28
    x35 = 1.0*p_on_ee_x
    x36 = 1.0*x14
    x37 = p_on_ee_z*x36
    x38 = x2*x9
    x39 = x10*x2
    x40 = x38*x4 - x39*x7
    x41 = a_0*x39 + d_1*x40
    x42 = x10*x38 + x2*x4*x7
    x43 = -x21*x42 - x24*x36
    x44 = d_2*x43 + x41
    x45 = -x29*x40 - x30*(-x21*x36 + x24*x42)
    x46 = d_3*x45 + x44
    x47 = x1*x35
    x48 = x0*x14
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = x2*x6 - x3
    jacobian_output[0, 3] = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x19 - x17*x20
    jacobian_output[0, 4] = p_on_ee_y*x23 + p_on_ee_z*x26 - x23*x28 - x26*x27
    jacobian_output[0, 5] = -p_on_ee_y*x31 + p_on_ee_z*x32 + x31*x34 - x32*x33
    jacobian_output[1, 0] = x35
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x36 + x37
    jacobian_output[1, 2] = x36*x6 - x37
    jacobian_output[1, 3] = p_on_ee_x*x13 - p_on_ee_z*x40 - x13*x41 + x20*x40
    jacobian_output[1, 4] = -p_on_ee_x*x23 - p_on_ee_z*x43 + x23*x44 + x27*x43
    jacobian_output[1, 5] = p_on_ee_x*x31 - p_on_ee_z*x45 - x31*x46 + x33*x45
    jacobian_output[2, 1] = -x47 - x48
    jacobian_output[2, 2] = -x1**2*x18 - x14**2*x18 + x47 + x48
    jacobian_output[2, 3] = -p_on_ee_x*x17 + p_on_ee_y*x40 + x17*x41 - x19*x40
    jacobian_output[2, 4] = -p_on_ee_x*x26 + p_on_ee_y*x43 + x26*x44 - x28*x43
    jacobian_output[2, 5] = -p_on_ee_x*x32 + p_on_ee_y*x45 + x32*x46 - x34*x45
    return jacobian_output


def yaskawa_HC10_ik_solve_raw(T_ee: np.ndarray):
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
    for i in range(28):
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
        R_l[0, 0] = d_2*r_21
        R_l[0, 1] = d_2*r_22
        R_l[0, 2] = d_2*r_11
        R_l[0, 3] = d_2*r_12
        R_l[0, 4] = Py - d_3*r_23
        R_l[0, 5] = Px - d_3*r_13
        R_l[1, 0] = d_2*r_11
        R_l[1, 1] = d_2*r_12
        R_l[1, 2] = -d_2*r_21
        R_l[1, 3] = -d_2*r_22
        R_l[1, 4] = Px - d_3*r_13
        R_l[1, 5] = -Py + d_3*r_23
        R_l[2, 6] = d_2*r_31
        R_l[2, 7] = d_2*r_32
        R_l[3, 0] = -r_21
        R_l[3, 1] = -r_22
        R_l[3, 2] = -r_11
        R_l[3, 3] = -r_12
        R_l[4, 0] = -r_11
        R_l[4, 1] = -r_12
        R_l[4, 2] = r_21
        R_l[4, 3] = r_22
        R_l[5, 6] = 2*Px*d_2*r_11 + 2*Py*d_2*r_21 + 2*Pz*d_2*r_31 - 2*d_2*d_3*r_11*r_13 - 2*d_2*d_3*r_21*r_23 - 2*d_2*d_3*r_31*r_33
        R_l[5, 7] = 2*Px*d_2*r_12 + 2*Py*d_2*r_22 + 2*Pz*d_2*r_32 - 2*d_2*d_3*r_12*r_13 - 2*d_2*d_3*r_22*r_23 - 2*d_2*d_3*r_32*r_33
        R_l[6, 0] = -Px*r_31 + Pz*r_11 - d_3*r_11*r_33 + d_3*r_13*r_31
        R_l[6, 1] = -Px*r_32 + Pz*r_12 - d_3*r_12*r_33 + d_3*r_13*r_32
        R_l[6, 2] = Py*r_31 - Pz*r_21 + d_3*r_21*r_33 - d_3*r_23*r_31
        R_l[6, 3] = Py*r_32 - Pz*r_22 + d_3*r_22*r_33 - d_3*r_23*r_32
        R_l[7, 0] = Py*r_31 - Pz*r_21 + d_3*r_21*r_33 - d_3*r_23*r_31
        R_l[7, 1] = Py*r_32 - Pz*r_22 + d_3*r_22*r_33 - d_3*r_23*r_32
        R_l[7, 2] = Px*r_31 - Pz*r_11 + d_3*r_11*r_33 - d_3*r_13*r_31
        R_l[7, 3] = Px*r_32 - Pz*r_12 + d_3*r_12*r_33 - d_3*r_13*r_32
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
        x0 = R_l_inv_62*r_31 + R_l_inv_72*r_32
        x1 = d_1*x0
        x2 = R_l_inv_60*r_31 + R_l_inv_70*r_32
        x3 = a_0*x2
        x4 = -x3
        x5 = x1 + x4
        x6 = R_l_inv_66*r_31
        x7 = R_l_inv_76*r_32
        x8 = x6 + x7
        x9 = d_1*x8
        x10 = d_3*r_33
        x11 = Pz - x10
        x12 = -x0*x11
        x13 = R_l_inv_65*r_31 + R_l_inv_75*r_32
        x14 = Px**2
        x15 = Py**2
        x16 = Pz**2
        x17 = d_2**2
        x18 = r_11**2
        x19 = x17*x18
        x20 = r_21**2
        x21 = x17*x20
        x22 = r_31**2
        x23 = x17*x22
        x24 = d_3**2
        x25 = r_13**2*x24
        x26 = r_23**2*x24
        x27 = r_33**2*x24
        x28 = d_3*r_13
        x29 = 2*Px
        x30 = x28*x29
        x31 = d_3*r_23
        x32 = 2*Py
        x33 = x31*x32
        x34 = 2*Pz
        x35 = x10*x34
        x36 = a_0**2
        x37 = d_1**2
        x38 = -x36 - x37
        x39 = x14 + x15 + x16 + x19 + x21 + x23 + x25 + x26 + x27 - x30 - x33 - x35 + x38
        x40 = -x13*x39
        x41 = x12 + x40 - x9
        x42 = R_l_inv_64*r_31
        x43 = R_l_inv_74*r_32
        x44 = x42 + x43
        x45 = x41 + x44
        x46 = 2*a_0
        x47 = x46*x8
        x48 = -x0*x46
        x49 = 2*d_1
        x50 = x2*x49
        x51 = x48 - x50
        x52 = -x1
        x53 = x3 + x52
        x54 = x12 + x40 + x9
        x55 = x44 + x54
        x56 = 2*R_l_inv_63
        x57 = r_31*x56
        x58 = 2*R_l_inv_73
        x59 = r_32*x58
        x60 = R_l_inv_67*r_31 + R_l_inv_77*r_32
        x61 = -x49*x60
        x62 = -x57 - x59 + x61
        x63 = x57 + x59 + x61
        x64 = -x42 - x43
        x65 = x54 + x64
        x66 = -x47
        x67 = x41 + x64
        x68 = Px*r_11
        x69 = Py*r_21
        x70 = Pz*r_31
        x71 = r_11*x28
        x72 = r_21*x31
        x73 = r_31*x10
        x74 = x68 + x69 + x70 - x71 - x72 - x73
        x75 = Px*r_12
        x76 = Py*r_22
        x77 = Pz*r_32
        x78 = r_12*x28
        x79 = r_22*x31
        x80 = r_32*x10
        x81 = x75 + x76 + x77 - x78 - x79 - x80
        x82 = R_l_inv_62*x74 + R_l_inv_72*x81
        x83 = d_1*x82
        x84 = R_l_inv_60*x74 + R_l_inv_70*x81
        x85 = a_0*x84
        x86 = -x85
        x87 = x83 + x86
        x88 = d_2*x18
        x89 = d_2*x20
        x90 = d_2*x22
        x91 = R_l_inv_66*x74
        x92 = R_l_inv_76*x81
        x93 = x91 + x92
        x94 = d_1*x93
        x95 = -x11*x82
        x96 = R_l_inv_65*x74 + R_l_inv_75*x81
        x97 = -x39*x96
        x98 = x88 + x89 + x90 - x94 + x95 + x97
        x99 = R_l_inv_64*x74
        x100 = R_l_inv_74*x81
        x101 = x100 + x99
        x102 = x101 + x98
        x103 = x46*x93
        x104 = -x46*x82
        x105 = x49*x84
        x106 = x104 - x105
        x107 = -x83
        x108 = x107 + x85
        x109 = x88 + x89 + x90 + x94 + x95 + x97
        x110 = x101 + x109
        x111 = R_l_inv_67*x74 + R_l_inv_77*x81
        x112 = -x111*x49
        x113 = x112 + x46
        x114 = x56*x74
        x115 = x58*x81
        x116 = -x114 - x115
        x117 = x114 + x115
        x118 = -x100 - x99
        x119 = x109 + x118
        x120 = -x103
        x121 = x118 + x98
        x122 = Px*r_21
        x123 = Py*r_11
        x124 = r_11*x31 - r_21*x28 + x122 - x123
        x125 = R_l_inv_62*x124
        x126 = Px*r_22
        x127 = Py*r_12
        x128 = r_12*x31 - r_22*x28 + x126 - x127
        x129 = R_l_inv_72*x128
        x130 = x125 + x129
        x131 = x11*x130
        x132 = R_l_inv_65*x124 + R_l_inv_75*x128
        x133 = x132*x39
        x134 = R_l_inv_64*x124
        x135 = R_l_inv_74*x128
        x136 = x131 + x133 - x134 - x135
        x137 = R_l_inv_60*x124
        x138 = R_l_inv_70*x128
        x139 = x137 + x138
        x140 = a_0*x139
        x141 = d_1*x130
        x142 = -x141
        x143 = x140 + x142
        x144 = -a_0
        x145 = R_l_inv_66*x124 + R_l_inv_76*x128
        x146 = d_1*x145
        x147 = x144 + x146
        x148 = -x49
        x149 = x145*x46
        x150 = -x149
        x151 = x130*x46
        x152 = x139*x49
        x153 = x151 + x152
        x154 = -x146
        x155 = a_0 + x154
        x156 = -x140
        x157 = x141 + x156
        x158 = x124*x56
        x159 = x128*x58
        x160 = R_l_inv_67*x124 + R_l_inv_77*x128
        x161 = x160*x49
        x162 = x158 + x159 + x161
        x163 = -x158 - x159 + x161
        x164 = x131 + x133 + x134 + x135
        x165 = Py*x18 + Py*x20 + Py*x22 - x18*x31 - x20*x31 - x22*x31
        x166 = 2*d_2
        x167 = x165*x166
        x168 = R_l_inv_40*x167
        x169 = Px*x18 + Px*x20 + Px*x22 - x18*x28 - x20*x28 - x22*x28
        x170 = x166*x169
        x171 = R_l_inv_50*x170
        x172 = 2*x31
        x173 = 2*x28
        x174 = 2*r_11
        x175 = r_23*x24
        x176 = r_13*x175
        x177 = 2*r_31
        x178 = r_33*x175
        x179 = r_21**3*x17 - r_21*x14 + r_21*x15 - r_21*x16 + r_21*x19 + r_21*x23 - r_21*x25 + r_21*x26 - r_21*x27 + r_21*x35 + x122*x173 - x123*x173 - x172*x68 - x172*x69 - x172*x70 + x174*x176 + x177*x178 + x32*x68 + x32*x70 - x32*x73
        x180 = R_l_inv_00*x179
        x181 = 2*x176
        x182 = r_13*r_33*x24
        x183 = r_11**3*x17 + r_11*x14 - r_11*x15 - r_11*x16 + r_11*x21 + r_11*x23 + r_11*x25 - r_11*x26 - r_11*x27 + r_11*x35 + r_21*x181 - x122*x172 + x123*x172 - x173*x68 - x173*x69 - x173*x70 + x177*x182 + x29*x69 + x29*x70 - x29*x73
        x184 = R_l_inv_20*x183
        x185 = r_21*x17
        x186 = x174*x185
        x187 = x177*x185
        x188 = 2*r_32
        x189 = r_12*x181 + r_12*x186 - r_22*x14 + r_22*x15 - r_22*x16 + r_22*x19 + 3*r_22*x21 + r_22*x23 - r_22*x25 + r_22*x26 - r_22*x27 + r_22*x35 + r_32*x187 + x126*x173 - x127*x173 - x172*x75 - x172*x76 - x172*x77 + x178*x188 + x32*x75 + x32*x77 - x32*x80
        x190 = R_l_inv_10*x189
        x191 = r_31*x17*x174
        x192 = r_12*x14 - r_12*x15 - r_12*x16 + 3*r_12*x19 + r_12*x21 + r_12*x23 + r_12*x25 - r_12*x26 - r_12*x27 + r_12*x35 + r_22*x181 + r_22*x186 + r_32*x191 - x126*x172 + x127*x172 - x173*x75 - x173*x76 - x173*x77 + x182*x188 + x29*x76 + x29*x77 - x29*x80
        x193 = R_l_inv_30*x192
        x194 = x168 + x171 + x180 + x184 + x190 + x193
        x195 = a_0*x194
        x196 = R_l_inv_42*x167
        x197 = R_l_inv_52*x170
        x198 = R_l_inv_02*x179
        x199 = R_l_inv_22*x183
        x200 = R_l_inv_12*x189
        x201 = R_l_inv_32*x192
        x202 = x196 + x197 + x198 + x199 + x200 + x201
        x203 = d_1*x202
        x204 = -x203
        x205 = x195 + x204
        x206 = R_l_inv_06*x179 + R_l_inv_16*x189 + R_l_inv_26*x183 + R_l_inv_36*x192 + R_l_inv_46*x167 + R_l_inv_56*x170
        x207 = d_1*x206
        x208 = x11*x202
        x209 = R_l_inv_05*x179 + R_l_inv_15*x189 + R_l_inv_25*x183 + R_l_inv_35*x192 + R_l_inv_45*x167 + R_l_inv_55*x170
        x210 = x209*x39
        x211 = x207 + x208 + x210
        x212 = R_l_inv_04*x179
        x213 = R_l_inv_14*x189
        x214 = R_l_inv_24*x183
        x215 = R_l_inv_34*x192
        x216 = R_l_inv_44*x167
        x217 = R_l_inv_54*x170
        x218 = -x212 - x213 - x214 - x215 - x216 - x217
        x219 = x211 + x218
        x220 = x206*x46
        x221 = -x220
        x222 = x202*x46
        x223 = x194*x49
        x224 = x222 + x223
        x225 = -x195
        x226 = x203 + x225
        x227 = -x207 + x208 + x210
        x228 = x218 + x227
        x229 = 2*x36
        x230 = 2*x37
        x231 = 4*d_2
        x232 = x165*x231
        x233 = R_l_inv_43*x232
        x234 = x169*x231
        x235 = R_l_inv_53*x234
        x236 = 2*x179
        x237 = R_l_inv_03*x236
        x238 = 2*x183
        x239 = R_l_inv_23*x238
        x240 = 2*x189
        x241 = R_l_inv_13*x240
        x242 = 2*x192
        x243 = R_l_inv_33*x242
        x244 = R_l_inv_07*x179 + R_l_inv_17*x189 + R_l_inv_27*x183 + R_l_inv_37*x192 + R_l_inv_47*x167 + R_l_inv_57*x170
        x245 = x244*x49
        x246 = x229 - x230 + x233 + x235 + x237 + x239 + x241 + x243 + x245
        x247 = a_0*d_1
        x248 = 8*x247
        x249 = -x229 + x230 - x233 - x235 - x237 - x239 - x241 - x243 + x245
        x250 = x212 + x213 + x214 + x215 + x216 + x217
        x251 = x227 + x250
        x252 = x211 + x250
        x253 = R_l_inv_40*x170
        x254 = R_l_inv_50*x167
        x255 = R_l_inv_00*x183
        x256 = R_l_inv_20*x179
        x257 = R_l_inv_10*x192
        x258 = R_l_inv_30*x189
        x259 = x253 - x254 + x255 - x256 + x257 - x258
        x260 = a_0*x259
        x261 = R_l_inv_42*x170
        x262 = R_l_inv_52*x167
        x263 = R_l_inv_02*x183
        x264 = R_l_inv_22*x179
        x265 = R_l_inv_12*x192
        x266 = R_l_inv_32*x189
        x267 = x261 - x262 + x263 - x264 + x265 - x266
        x268 = d_1*x267
        x269 = -x268
        x270 = x260 + x269
        x271 = R_l_inv_46*x170
        x272 = R_l_inv_56*x167
        x273 = R_l_inv_06*x183
        x274 = R_l_inv_26*x179
        x275 = R_l_inv_16*x192
        x276 = R_l_inv_36*x189
        x277 = x271 - x272 + x273 - x274 + x275 - x276
        x278 = d_1*x277
        x279 = x11*x267
        x280 = R_l_inv_05*x183 + R_l_inv_15*x192 - R_l_inv_25*x179 - R_l_inv_35*x189 + R_l_inv_45*x170 - R_l_inv_55*x167
        x281 = x280*x39
        x282 = x278 + x279 + x281
        x283 = R_l_inv_24*x179
        x284 = R_l_inv_34*x189
        x285 = R_l_inv_04*x183
        x286 = R_l_inv_14*x192
        x287 = R_l_inv_44*x170
        x288 = R_l_inv_54*x167
        x289 = x283 + x284 - x285 - x286 - x287 + x288 + x36 + x37
        x290 = x282 + x289
        x291 = x277*x46
        x292 = -x291
        x293 = x267*x46
        x294 = x259*x49
        x295 = x293 + x294
        x296 = -x260
        x297 = x268 + x296
        x298 = -x278 + x279 + x281
        x299 = x289 + x298
        x300 = R_l_inv_43*x234
        x301 = R_l_inv_53*x232
        x302 = R_l_inv_03*x238
        x303 = R_l_inv_23*x236
        x304 = R_l_inv_13*x242
        x305 = R_l_inv_33*x240
        x306 = R_l_inv_07*x183 + R_l_inv_17*x192 - R_l_inv_27*x179 - R_l_inv_37*x189 + R_l_inv_47*x170 - R_l_inv_57*x167
        x307 = x306*x49
        x308 = x300 - x301 + x302 - x303 + x304 - x305 + x307
        x309 = -x300 + x301 - x302 + x303 - x304 + x305 + x307
        x310 = -x283 - x284 + x285 + x286 + x287 - x288 + x38
        x311 = x298 + x310
        x312 = x282 + x310
        x313 = 2*x10
        x314 = 2*x178
        x315 = r_21*x314 + r_31**3*x17 - r_31*x14 - r_31*x15 + r_31*x16 + r_31*x19 + r_31*x21 - r_31*x25 - r_31*x26 + r_31*x27 + r_31*x30 + r_31*x33 + x174*x182 - x313*x68 - x313*x69 - x313*x70 + x34*x68 + x34*x69 - x34*x71 - x34*x72
        x316 = R_l_inv_60*x315
        x317 = 2*r_12*x182 + r_12*x191 + r_22*x187 + r_22*x314 - r_32*x14 - r_32*x15 + r_32*x16 + r_32*x19 + r_32*x21 + 3*r_32*x23 - r_32*x25 - r_32*x26 + r_32*x27 + r_32*x30 + r_32*x33 - x313*x75 - x313*x76 - x313*x77 + x34*x75 + x34*x76 - x34*x78 - x34*x79
        x318 = R_l_inv_70*x317
        x319 = x316 + x318
        x320 = a_0*x319
        x321 = R_l_inv_62*x315
        x322 = R_l_inv_72*x317
        x323 = x321 + x322
        x324 = d_1*x323
        x325 = -x324
        x326 = x320 + x325
        x327 = R_l_inv_66*x315 + R_l_inv_76*x317
        x328 = d_1*x327
        x329 = x11*x323
        x330 = R_l_inv_65*x315 + R_l_inv_75*x317
        x331 = x330*x39
        x332 = -x34*x88
        x333 = -x34*x89
        x334 = -x34*x90
        x335 = x313*x88
        x336 = x313*x89
        x337 = x313*x90
        x338 = x328 + x329 + x331 + x332 + x333 + x334 + x335 + x336 + x337
        x339 = R_l_inv_64*x315
        x340 = R_l_inv_74*x317
        x341 = -x339 - x340
        x342 = x338 + x341
        x343 = x327*x46
        x344 = -x343
        x345 = x323*x46
        x346 = x319*x49
        x347 = x345 + x346
        x348 = -x320
        x349 = x324 + x348
        x350 = -x328 + x329 + x331 + x332 + x333 + x334 + x335 + x336 + x337
        x351 = x341 + x350
        x352 = 4*x247
        x353 = R_l_inv_67*x315 + R_l_inv_77*x317
        x354 = x353*x49
        x355 = -x352 + x354
        x356 = x315*x56
        x357 = x317*x58
        x358 = x356 + x357
        x359 = 4*x36
        x360 = 4*x37
        x361 = -x360
        x362 = -x356 - x357
        x363 = x352 + x354
        x364 = x339 + x340
        x365 = x350 + x364
        x366 = x338 + x364
        x367 = x13*x46
        x368 = x49*(x2 - x367)
        x369 = 4*d_1
        x370 = -x49*(x2 + x367)
        x371 = 4*a_0
        x372 = x371*x60
        x373 = 8*R_l_inv_63
        x374 = 8*R_l_inv_73
        x375 = x46*x96
        x376 = x49*(-x375 + x84)
        x377 = -x49*(x375 + x84)
        x378 = x111*x371
        x379 = x132*x46
        x380 = x379 + 1
        x381 = -x137 - x138
        x382 = x379 - 1
        x383 = -x160*x371
        x384 = x209*x46
        x385 = x49*(-x168 - x171 - x180 - x184 - x190 - x193 + x384)
        x386 = x49*(x194 + x384)
        x387 = -x244*x371
        x388 = 16*d_2
        x389 = x165*x388
        x390 = x169*x388
        x391 = 8*x179
        x392 = 8*x183
        x393 = 8*x189
        x394 = 8*x192
        x395 = -x46
        x396 = x280*x46
        x397 = x395 + x396
        x398 = -x253 + x254 - x255 + x256 - x257 + x258
        x399 = -x261 + x262 - x263 + x264 - x265 + x266
        x400 = -x306*x371
        x401 = x396 + x46
        x402 = x330*x46
        x403 = x49*(-x316 - x318 + x402)
        x404 = x49*(x319 + x402)
        x405 = -x353*x371
        x406 = -x359
        x407 = x4 + x52
        x408 = x48 + x50
        x409 = x1 + x3
        x410 = x107 + x86
        x411 = x104 + x105
        x412 = x83 + x85
        x413 = x112 + x395
        x414 = x140 + x141
        x415 = x144 + x154
        x416 = x151 - x152
        x417 = a_0 + x146
        x418 = x142 + x156
        x419 = x195 + x203
        x420 = x222 - x223
        x421 = x204 + x225
        x422 = x260 + x268
        x423 = x293 - x294
        x424 = x269 + x296
        x425 = x320 + x324
        x426 = x345 - x346
        x427 = x325 + x348
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = x45 + x5
        A[0, 1] = x47 + x51
        A[0, 2] = x53 + x55
        A[0, 3] = x62
        A[0, 4] = -4
        A[0, 5] = x63
        A[0, 6] = x5 + x65
        A[0, 7] = x51 + x66
        A[0, 8] = x53 + x67
        A[1, 0] = x102 + x87
        A[1, 1] = x103 + x106
        A[1, 2] = x108 + x110
        A[1, 3] = x113 + x116
        A[1, 5] = x113 + x117
        A[1, 6] = x119 + x87
        A[1, 7] = x106 + x120
        A[1, 8] = x108 + x121
        A[2, 0] = x136 + x143 + x147
        A[2, 1] = x148 + x150 + x153
        A[2, 2] = x136 + x155 + x157
        A[2, 3] = x162
        A[2, 5] = x163
        A[2, 6] = x143 + x155 + x164
        A[2, 7] = x149 + x153 + x49
        A[2, 8] = x147 + x157 + x164
        A[3, 0] = x205 + x219
        A[3, 1] = x221 + x224
        A[3, 2] = x226 + x228
        A[3, 3] = x246
        A[3, 4] = x248
        A[3, 5] = x249
        A[3, 6] = x205 + x251
        A[3, 7] = x220 + x224
        A[3, 8] = x226 + x252
        A[4, 0] = x270 + x290
        A[4, 1] = x292 + x295
        A[4, 2] = x297 + x299
        A[4, 3] = x308
        A[4, 5] = x309
        A[4, 6] = x270 + x311
        A[4, 7] = x291 + x295
        A[4, 8] = x297 + x312
        A[5, 0] = x326 + x342
        A[5, 1] = x344 + x347
        A[5, 2] = x349 + x351
        A[5, 3] = x355 + x358
        A[5, 4] = x359 + x361
        A[5, 5] = x362 + x363
        A[5, 6] = x326 + x365
        A[5, 7] = x343 + x347
        A[5, 8] = x349 + x366
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x368
        B[0, 1] = x369*(x0 - x6 - x7)
        B[0, 2] = x370
        B[0, 3] = x372 + 4
        B[0, 4] = -r_31*x373 - r_32*x374
        B[0, 5] = x372 - 4
        B[0, 6] = x368
        B[0, 7] = x369*(x0 + x8)
        B[0, 8] = x370
        B[1, 0] = x376
        B[1, 1] = x369*(x82 - x91 - x92)
        B[1, 2] = x377
        B[1, 3] = x378
        B[1, 4] = -x373*x74 - x374*x81
        B[1, 5] = x378
        B[1, 6] = x376
        B[1, 7] = x369*(x82 + x93)
        B[1, 8] = x377
        B[2, 0] = x49*(x380 + x381)
        B[2, 1] = x369*(-x125 - x129 + x145)
        B[2, 2] = x49*(x139 + x382)
        B[2, 3] = x383
        B[2, 4] = x124*x373 + x128*x374
        B[2, 5] = x383
        B[2, 6] = x49*(x381 + x382)
        B[2, 7] = -x369*(x130 + x145)
        B[2, 8] = x49*(x139 + x380)
        B[3, 0] = x385
        B[3, 1] = x369*(-x196 - x197 - x198 - x199 - x200 - x201 + x206)
        B[3, 2] = x386
        B[3, 3] = x387
        B[3, 4] = R_l_inv_03*x391 + R_l_inv_13*x393 + R_l_inv_23*x392 + R_l_inv_33*x394 + R_l_inv_43*x389 + R_l_inv_53*x390 - 8*x36 - 8*x37
        B[3, 5] = x387
        B[3, 6] = x385
        B[3, 7] = -x369*(x202 + x206)
        B[3, 8] = x386
        B[4, 0] = x49*(x397 + x398)
        B[4, 1] = x369*(x277 + x399)
        B[4, 2] = x49*(x259 + x397)
        B[4, 3] = x400
        B[4, 4] = R_l_inv_03*x392 + R_l_inv_13*x394 - R_l_inv_23*x391 - R_l_inv_33*x393 + R_l_inv_43*x390 - R_l_inv_53*x389
        B[4, 5] = x400
        B[4, 6] = x49*(x398 + x401)
        B[4, 7] = x369*(-x271 + x272 - x273 + x274 - x275 + x276 + x399)
        B[4, 8] = x49*(x259 + x401)
        B[5, 0] = x403
        B[5, 1] = x369*(-x321 - x322 + x327)
        B[5, 2] = x404
        B[5, 3] = x359 + x360 + x405
        B[5, 4] = x315*x373 + x317*x374
        B[5, 5] = x361 + x405 + x406
        B[5, 6] = x403
        B[5, 7] = -x369*(x323 + x327)
        B[5, 8] = x404
        C = np.zeros(shape=(6, 9))
        C[0, 0] = x407 + x55
        C[0, 1] = x408 + x47
        C[0, 2] = x409 + x45
        C[0, 3] = x63
        C[0, 4] = 4
        C[0, 5] = x62
        C[0, 6] = x407 + x67
        C[0, 7] = x408 + x66
        C[0, 8] = x409 + x65
        C[1, 0] = x110 + x410
        C[1, 1] = x103 + x411
        C[1, 2] = x102 + x412
        C[1, 3] = x117 + x413
        C[1, 5] = x116 + x413
        C[1, 6] = x121 + x410
        C[1, 7] = x120 + x411
        C[1, 8] = x119 + x412
        C[2, 0] = x136 + x414 + x415
        C[2, 1] = x150 + x416 + x49
        C[2, 2] = x136 + x417 + x418
        C[2, 3] = x163
        C[2, 5] = x162
        C[2, 6] = x164 + x414 + x417
        C[2, 7] = x148 + x149 + x416
        C[2, 8] = x164 + x415 + x418
        C[3, 0] = x228 + x419
        C[3, 1] = x221 + x420
        C[3, 2] = x219 + x421
        C[3, 3] = x249
        C[3, 4] = x248
        C[3, 5] = x246
        C[3, 6] = x252 + x419
        C[3, 7] = x220 + x420
        C[3, 8] = x251 + x421
        C[4, 0] = x299 + x422
        C[4, 1] = x292 + x423
        C[4, 2] = x290 + x424
        C[4, 3] = x309
        C[4, 5] = x308
        C[4, 6] = x312 + x422
        C[4, 7] = x291 + x423
        C[4, 8] = x311 + x424
        C[5, 0] = x351 + x425
        C[5, 1] = x344 + x426
        C[5, 2] = x342 + x427
        C[5, 3] = x355 + x362
        C[5, 4] = x360 + x406
        C[5, 5] = x358 + x363
        C[5, 6] = x366 + x425
        C[5, 7] = x343 + x426
        C[5, 8] = x365 + x427
        from solver.general_6dof.numerical_reduce_closure_equation import compute_solution_from_tanhalf_LME
        local_solutions = compute_solution_from_tanhalf_LME(A, B, C)
        for local_solutions_i in local_solutions:
            solution_i: IkSolution = make_ik_solution()
            solution_i[2] = local_solutions_i
            appended_idx = append_solution_to_queue(solution_i)
            add_input_index_to(2, appended_idx)
    # Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor()
    # Finish code for explicit solution node 0
    
    # Code for solved_variable dispatcher node 2
    def SolvedVariableDispatcherNode_node_2_processor():
        this_node_input_index: List[int] = node_input_index[2]
        this_input_valid: bool = node_input_validity[2]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_2 = this_solution[2]
            degenerate_valid_0 = (abs(th_2 - 1/2*math.pi) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
            
            th_2 = this_solution[2]
            degenerate_valid_1 = (abs(th_2 + (1/2)*math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
            
            if not taken_by_degenerate:
                add_input_index_to(3, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_2_processor()
    # Finish code for solved_variable dispatcher node 2
    
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
            condition_0: bool = ((1/2)*abs((Px**2 - 2*Px*d_3*r_13 + Py**2 - 2*Py*d_3*r_23 + Pz**2 - 2*Pz*d_3*r_33 - a_0**2 + 2*a_0*d_1*math.sin(th_2) - d_1**2 - d_2**2 + d_3**2*r_13**2 + d_3**2*r_23**2 + d_3**2*r_33**2)/(a_0*d_2*math.cos(th_2))) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*d_3
                x1 = d_3**2
                x2 = math.asin((1/2)*(Px**2 - Px*r_13*x0 + Py**2 - Py*r_23*x0 + Pz**2 - Pz*r_33*x0 - a_0**2 + 2*a_0*d_1*math.sin(th_2) - d_1**2 - d_2**2 + r_13**2*x1 + r_23**2*x1 + r_33**2*x1)/(a_0*d_2*math.cos(th_2)))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = -x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = ((1/2)*abs((Px**2 - 2*Px*d_3*r_13 + Py**2 - 2*Py*d_3*r_23 + Pz**2 - 2*Pz*d_3*r_33 - a_0**2 + 2*a_0*d_1*math.sin(th_2) - d_1**2 - d_2**2 + d_3**2*r_13**2 + d_3**2*r_23**2 + d_3**2*r_33**2)/(a_0*d_2*math.cos(th_2))) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*d_3
                x1 = d_3**2
                x2 = math.asin((1/2)*(Px**2 - Px*r_13*x0 + Py**2 - Py*r_23*x0 + Pz**2 - Pz*r_33*x0 - a_0**2 + 2*a_0*d_1*math.sin(th_2) - d_1**2 - d_2**2 + r_13**2*x1 + r_23**2*x1 + r_33**2*x1)/(a_0*d_2*math.cos(th_2)))
                # End of temp variables
                this_solution[3] = x2 + math.pi
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
            th_3 = this_solution[3]
            checked_result: bool = (abs(d_2*math.cos(th_3)) <= 1.0e-6) and (abs(Px - d_3*r_13) <= 1.0e-6) and (abs(Py - d_3*r_23) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor()
    # Finish code for equation all-zero dispatcher node 4
    
    # Code for explicit solution node 5, solved variable is th_0
    def ExplicitSolutionNode_node_5_solve_th_0_processor():
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
            th_3 = this_solution[3]
            condition_0: bool = (abs(d_2*math.cos(th_3)) >= zero_tolerance) or (abs(Px - d_3*r_13) >= zero_tolerance) or (abs(Py - d_3*r_23) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = -Px + d_3*r_13
                x1 = Py - d_3*r_23
                x2 = math.atan2(x0, x1)
                x3 = math.cos(th_3)
                x4 = math.sqrt(-d_2**2*x3**2 + x0**2 + x1**2)
                x5 = d_2*x3
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x2 + math.atan2(x4, x5)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                
            condition_1: bool = (abs(d_2*math.cos(th_3)) >= zero_tolerance) or (abs(Px - d_3*r_13) >= zero_tolerance) or (abs(Py - d_3*r_23) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = -Px + d_3*r_13
                x1 = Py - d_3*r_23
                x2 = math.atan2(x0, x1)
                x3 = math.cos(th_3)
                x4 = math.sqrt(-d_2**2*x3**2 + x0**2 + x1**2)
                x5 = d_2*x3
                # End of temp variables
                this_solution[0] = x2 + math.atan2(-x4, x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_0_processor()
    # Finish code for explicit solution node 5
    
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
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                # End of temp variables
                this_solution[5] = math.atan2(-r_11*x1 + r_21*x0, -r_12*x1 + r_22*x0)
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
            degenerate_valid_0 = (abs(th_2 + (1/2)*math.pi) <= 1.0e-6)
            if degenerate_valid_0:
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
            th_3 = this_solution[3]
            condition_0: bool = (abs(d_2*math.sin(th_3)) >= 1.0e-6) or (abs(a_0 + d_1) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_0 + d_1
                x1 = Pz - d_3*r_33
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3
                x5 = d_2*math.sin(th_3)
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5)
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
            th_3 = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_13*(math.sin(th_0)*math.sin(th_3) + math.sin(th_1)*math.cos(th_0)*math.cos(th_3)) - r_23*(-math.sin(th_0)*math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_0)) + r_33*math.cos(th_1)*math.cos(th_3)) >= zero_tolerance) or (abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.cos(th_3)
                x2 = math.sin(th_0)
                x3 = math.sin(th_3)
                x4 = math.cos(th_0)
                x5 = math.sin(th_1)
                x6 = x1*x5
                # End of temp variables
                this_solution[4] = math.atan2(r_13*(x2*x3 + x4*x6) - r_23*(-x2*x6 + x3*x4) + r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_4_processor()
    # Finish code for explicit solution node 26
    
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
            condition_0: bool = (abs(d_1*math.cos(th_2)) >= 1.0e-6) or (abs(a_0 - d_1*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = Pz - d_3*r_33
                x1 = a_0 - d_1*math.sin(th_2)
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3
                x5 = d_1*math.cos(th_2)
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x1 + x4*x5, x0*x5 + x1*x4)
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
                this_solution[4] = math.atan2(-r_33*x4 + x5*x6 + x5*x7, -r_33*x5 - x4*x6 - x4*x7)
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
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                # End of temp variables
                this_solution[5] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
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
            degenerate_valid_0 = (abs(th_2 + (1/2)*math.pi) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(22, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(14, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_13_processor()
    # Finish code for solved_variable dispatcher node 13
    
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
            th_3 = this_solution[3]
            condition_0: bool = (abs(d_2*math.sin(th_3)) >= 1.0e-6) or (abs(a_0 + d_1) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_0 + d_1
                x1 = Pz - d_3*r_33
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3
                x5 = d_2*math.sin(th_3)
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5)
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
            th_3 = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_13*(math.sin(th_0)*math.sin(th_3) + math.sin(th_1)*math.cos(th_0)*math.cos(th_3)) - r_23*(-math.sin(th_0)*math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_0)) + r_33*math.cos(th_1)*math.cos(th_3)) >= zero_tolerance) or (abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.cos(th_3)
                x2 = math.sin(th_0)
                x3 = math.sin(th_3)
                x4 = math.cos(th_0)
                x5 = math.sin(th_1)
                x6 = x1*x5
                # End of temp variables
                this_solution[4] = math.atan2(r_13*(x2*x3 + x4*x6) - r_23*(-x2*x6 + x3*x4) + r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_4_processor()
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
            condition_0: bool = (abs(d_1*math.cos(th_2)) >= 1.0e-6) or (abs(a_0 - d_1*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = Pz - d_3*r_33
                x1 = a_0 - d_1*math.sin(th_2)
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3
                x5 = d_1*math.cos(th_2)
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x1 + x4*x5, x0*x5 + x1*x4)
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
                this_solution[4] = math.atan2(r_33*x4 - x5*x6 - x5*x7, -r_33*x5 - x4*x6 - x4*x7)
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
                x0 = math.asin((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
            condition_1: bool = (abs((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.asin((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3))
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
            checked_result: bool = (abs(d_1*math.cos(th_2) - d_2*math.sin(th_2)*math.sin(th_3)) <= 1.0e-6) and (abs(-a_0 + d_1*math.sin(th_2) + d_2*math.sin(th_3)*math.cos(th_2)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor()
    # Finish code for equation all-zero dispatcher node 8
    
    # Code for explicit solution node 9, solved variable is th_1
    def ExplicitSolutionNode_node_9_solve_th_1_processor():
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
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            condition_0: bool = (abs(d_1*math.cos(th_2) - d_2*math.sin(th_2)*math.sin(th_3)) >= 1.0e-6) or (abs(-a_0 + d_1*math.sin(th_2) + d_2*math.sin(th_3)*math.cos(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = d_3*r_33
                x1 = math.sin(th_2)
                x2 = math.cos(th_2)
                x3 = d_2*math.sin(th_3)
                x4 = a_0 - d_1*x1 - x2*x3
                x5 = d_1*x2 - x1*x3
                x6 = math.cos(th_0)
                x7 = math.sin(th_0)
                x8 = Px*x6 + Py*x7 - d_3*r_13*x6 - d_3*r_23*x7
                # End of temp variables
                this_solution[1] = math.atan2(x4*(-Pz + x0) + x5*x8, x4*x8 + x5*(Pz - x0))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_1_processor()
    # Finish code for explicit solution node 9
    
    # Code for non-branch dispatcher node 10
    # Actually, there is no code
    
    # Code for explicit solution node 11, solved variable is th_5
    def ExplicitSolutionNode_node_11_solve_th_5_processor():
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*((math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_3)*math.cos(th_0) + math.sin(th_0)*math.cos(th_3)) - r_21*((-math.sin(th_1)*math.sin(th_2) - math.cos(th_1)*math.cos(th_2))*math.sin(th_0)*math.sin(th_3) + math.cos(th_0)*math.cos(th_3)) - r_31*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.sin(th_3)) >= zero_tolerance) or (abs(r_12*((math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_3)*math.cos(th_0) + math.sin(th_0)*math.cos(th_3)) - r_22*((-math.sin(th_1)*math.sin(th_2) - math.cos(th_1)*math.cos(th_2))*math.sin(th_0)*math.sin(th_3) + math.cos(th_0)*math.cos(th_3)) - r_32*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.sin(th_3)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_2)
                x3 = math.sin(th_2)
                x4 = math.cos(th_1)
                x5 = x0*(x1*x2 - x3*x4)
                x6 = math.sin(th_0)
                x7 = math.cos(th_3)
                x8 = math.cos(th_0)
                x9 = x1*x3
                x10 = x2*x4
                x11 = x0*x8*(x10 + x9) + x6*x7
                x12 = x0*x6*(-x10 - x9) + x7*x8
                # End of temp variables
                this_solution[5] = math.atan2(r_11*x11 - r_21*x12 - r_31*x5, r_12*x11 - r_22*x12 - r_32*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_5_processor()
    # Finish code for explicit solution node 10
    
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


def yaskawa_HC10_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = yaskawa_HC10_ik_target_original_to_raw(T_ee)
    ik_output_raw = yaskawa_HC10_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ee_pose_i = yaskawa_HC10_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_yaskawa_HC10():
    theta_in = np.random.random(size=(6, ))
    ee_pose = yaskawa_HC10_fk(theta_in)
    ik_output = yaskawa_HC10_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = yaskawa_HC10_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_yaskawa_HC10()
