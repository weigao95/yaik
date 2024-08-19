import numpy as np
import copy
import math
from typing import List, NewType

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 36
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 0.71
d_1: float = -0.05
d_2: float = 0.59
d_3: float = 0.1
d_4: float = 0.16
pre_transform_special_symbol_23: float = 0.22

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
def denso_cvrb_ik_target_original_to_raw(T_ee: np.ndarray):
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


def denso_cvrb_ik_target_raw_to_original(T_ee: np.ndarray):
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


def denso_cvrb_fk(theta_input: np.ndarray):
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
    ee_pose[0, 3] = d_3*x20 + x1*x25 - x11*x23 + x26*x8 + x27*(-x18 - x9)
    ee_pose[1, 0] = -1.0*x29 - 1.0*x32
    ee_pose[1, 1] = x19*x33 + x21*x34
    ee_pose[1, 2] = x19*x34 - x21*x33
    ee_pose[1, 3] = d_3*x33 + x1*x23 + x11*x25 + x26*x28 + x27*(-x29 - x32)
    ee_pose[2, 0] = -1.0*x36 - 1.0*x39
    ee_pose[2, 1] = -x19*x40 + x21*x41
    ee_pose[2, 2] = x19*x41 + x21*x40
    ee_pose[2, 3] = -d_3*x40 + 1.0*pre_transform_special_symbol_23 - x2*x24 + x26*x35 + x27*(-x36 - x39)
    return ee_pose


def denso_cvrb_twist_jacobian(theta_input: np.ndarray):
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
    x30 = d_2*x25 + x29
    x31 = a_0*x18 + d_1*x4
    x32 = d_2*x19 + x31
    x33 = -d_3*x27 + x30
    x34 = d_3*x21 + x32
    x35 = d_4*x28 + x33
    x36 = d_4*x22 + x34
    x37 = a_0*x8 - d_1*x0
    x38 = d_2*x9 + x37
    x39 = d_3*x13 + x38
    x40 = d_4*x16 + x39
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


def denso_cvrb_angular_velocity_jacobian(theta_input: np.ndarray):
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


def denso_cvrb_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x18 = d_2*x13 + x6
    x19 = a_0*x11*x14 + d_1*x2
    x20 = d_2*x17 + x19
    x21 = math.sin(th_3)
    x22 = -x15 - x16
    x23 = x21*x22
    x24 = math.cos(th_3)
    x25 = x12*x14 - x14*x8
    x26 = -x2*x24 - x21*x25
    x27 = -d_3*x23 + x18
    x28 = d_3*x26 + x20
    x29 = math.cos(th_4)
    x30 = math.sin(th_4)
    x31 = -x13*x29 - x22*x24*x30
    x32 = -x17*x29 - x30*(-x2*x21 + x24*x25)
    x33 = d_4*x31 + x27
    x34 = d_4*x32 + x28
    x35 = 1.0*p_on_ee_x
    x36 = 1.0*x14
    x37 = p_on_ee_z*x36
    x38 = x2*x4
    x39 = x10*x2
    x40 = -x38*x9 - x39*x7
    x41 = a_0*x39 - d_1*x36
    x42 = d_2*x40 + x41
    x43 = -x38*x7 + x39*x9
    x44 = -x21*x43 + x24*x36
    x45 = d_3*x44 + x42
    x46 = -x29*x40 - x30*(x21*x36 + x24*x43)
    x47 = d_4*x46 + x45
    x48 = -x0*x14 - x1*x35
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = -x2*x6 + x3
    jacobian_output[0, 3] = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x20 - x17*x18
    jacobian_output[0, 4] = p_on_ee_y*x23 + p_on_ee_z*x26 - x23*x28 - x26*x27
    jacobian_output[0, 5] = -p_on_ee_y*x31 + p_on_ee_z*x32 + x31*x34 - x32*x33
    jacobian_output[1, 0] = x35
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x36 + x37
    jacobian_output[1, 2] = -x36*x6 + x37
    jacobian_output[1, 3] = p_on_ee_x*x13 - p_on_ee_z*x40 - x13*x42 + x18*x40
    jacobian_output[1, 4] = -p_on_ee_x*x23 - p_on_ee_z*x44 + x23*x45 + x27*x44
    jacobian_output[1, 5] = p_on_ee_x*x31 - p_on_ee_z*x46 - x31*x47 + x33*x46
    jacobian_output[2, 1] = x48
    jacobian_output[2, 2] = x19*x36 + x2*x41 + x48
    jacobian_output[2, 3] = -p_on_ee_x*x17 + p_on_ee_y*x40 + x17*x42 - x20*x40
    jacobian_output[2, 4] = -p_on_ee_x*x26 + p_on_ee_y*x44 + x26*x45 - x28*x44
    jacobian_output[2, 5] = -p_on_ee_x*x32 + p_on_ee_y*x46 + x32*x47 - x34*x46
    return jacobian_output


def denso_cvrb_ik_solve_raw(T_ee: np.ndarray):
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
        for tmp_sol_idx in range(7):
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
    for i in range(36):
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
        R_l[0, 0] = d_3*r_21
        R_l[0, 1] = d_3*r_22
        R_l[0, 2] = d_3*r_11
        R_l[0, 3] = d_3*r_12
        R_l[0, 4] = Py - d_4*r_23
        R_l[0, 5] = Px - d_4*r_13
        R_l[1, 0] = d_3*r_11
        R_l[1, 1] = d_3*r_12
        R_l[1, 2] = -d_3*r_21
        R_l[1, 3] = -d_3*r_22
        R_l[1, 4] = Px - d_4*r_13
        R_l[1, 5] = -Py + d_4*r_23
        R_l[2, 6] = d_3*r_31
        R_l[2, 7] = d_3*r_32
        R_l[3, 0] = -r_21
        R_l[3, 1] = -r_22
        R_l[3, 2] = -r_11
        R_l[3, 3] = -r_12
        R_l[4, 0] = -r_11
        R_l[4, 1] = -r_12
        R_l[4, 2] = r_21
        R_l[4, 3] = r_22
        R_l[5, 6] = 2*Px*d_3*r_11 + 2*Py*d_3*r_21 + 2*Pz*d_3*r_31 - 2*d_3*d_4*r_11*r_13 - 2*d_3*d_4*r_21*r_23 - 2*d_3*d_4*r_31*r_33
        R_l[5, 7] = 2*Px*d_3*r_12 + 2*Py*d_3*r_22 + 2*Pz*d_3*r_32 - 2*d_3*d_4*r_12*r_13 - 2*d_3*d_4*r_22*r_23 - 2*d_3*d_4*r_32*r_33
        R_l[6, 0] = -Px*r_31 + Pz*r_11 - d_4*r_11*r_33 + d_4*r_13*r_31
        R_l[6, 1] = -Px*r_32 + Pz*r_12 - d_4*r_12*r_33 + d_4*r_13*r_32
        R_l[6, 2] = Py*r_31 - Pz*r_21 + d_4*r_21*r_33 - d_4*r_23*r_31
        R_l[6, 3] = Py*r_32 - Pz*r_22 + d_4*r_22*r_33 - d_4*r_23*r_32
        R_l[7, 0] = Py*r_31 - Pz*r_21 + d_4*r_21*r_33 - d_4*r_23*r_31
        R_l[7, 1] = Py*r_32 - Pz*r_22 + d_4*r_22*r_33 - d_4*r_23*r_32
        R_l[7, 2] = Px*r_31 - Pz*r_11 + d_4*r_11*r_33 - d_4*r_13*r_31
        R_l[7, 3] = Px*r_32 - Pz*r_12 + d_4*r_12*r_33 - d_4*r_13*r_32
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
        x0 = R_l_inv_60*r_31 + R_l_inv_70*r_32
        x1 = a_0*x0
        x2 = -x1
        x3 = R_l_inv_62*r_31 + R_l_inv_72*r_32
        x4 = d_2*x3
        x5 = -x4
        x6 = x2 + x5
        x7 = -d_1*(R_l_inv_61*r_31 + R_l_inv_71*r_32)
        x8 = R_l_inv_66*r_31
        x9 = R_l_inv_76*r_32
        x10 = x8 + x9
        x11 = d_2*x10
        x12 = d_4*r_33
        x13 = Pz - x12
        x14 = -x13*x3
        x15 = R_l_inv_65*r_31 + R_l_inv_75*r_32
        x16 = Px**2
        x17 = Py**2
        x18 = Pz**2
        x19 = d_1**2
        x20 = -x19
        x21 = d_3**2
        x22 = r_11**2
        x23 = x21*x22
        x24 = r_21**2
        x25 = x21*x24
        x26 = r_31**2
        x27 = x21*x26
        x28 = d_4**2
        x29 = r_13**2*x28
        x30 = r_23**2*x28
        x31 = r_33**2*x28
        x32 = d_4*r_13
        x33 = 2*Px
        x34 = x32*x33
        x35 = d_4*r_23
        x36 = 2*Py
        x37 = x35*x36
        x38 = 2*Pz
        x39 = x12*x38
        x40 = a_0**2
        x41 = d_2**2
        x42 = -x40 - x41
        x43 = x16 + x17 + x18 + x20 + x23 + x25 + x27 + x29 + x30 + x31 - x34 - x37 - x39 + x42
        x44 = -x15*x43
        x45 = -x11 + x14 + x44 + x7
        x46 = R_l_inv_64*r_31
        x47 = R_l_inv_74*r_32
        x48 = -x46 - x47
        x49 = x45 + x48
        x50 = 2*a_0
        x51 = x10*x50
        x52 = -x51
        x53 = -x3*x50
        x54 = 2*d_2
        x55 = x0*x54
        x56 = x53 + x55
        x57 = x1 + x4
        x58 = x11 + x14 + x44 + x7
        x59 = x48 + x58
        x60 = 2*R_l_inv_63
        x61 = r_31*x60
        x62 = 2*R_l_inv_73
        x63 = r_32*x62
        x64 = R_l_inv_67*r_31 + R_l_inv_77*r_32
        x65 = x54*x64
        x66 = -x61 - x63 + x65
        x67 = 4*d_1
        x68 = x10*x67
        x69 = x68 - 4
        x70 = x61 + x63 + x65
        x71 = x46 + x47
        x72 = x58 + x71
        x73 = x45 + x71
        x74 = Px*r_11
        x75 = Py*r_21
        x76 = Pz*r_31
        x77 = r_11*x32
        x78 = r_21*x35
        x79 = r_31*x12
        x80 = x74 + x75 + x76 - x77 - x78 - x79
        x81 = Px*r_12
        x82 = Py*r_22
        x83 = Pz*r_32
        x84 = r_12*x32
        x85 = r_22*x35
        x86 = r_32*x12
        x87 = x81 + x82 + x83 - x84 - x85 - x86
        x88 = R_l_inv_60*x80 + R_l_inv_70*x87
        x89 = a_0*x88
        x90 = -x89
        x91 = R_l_inv_62*x80 + R_l_inv_72*x87
        x92 = d_2*x91
        x93 = -x92
        x94 = x90 + x93
        x95 = d_3*x22
        x96 = d_3*x24
        x97 = d_3*x26
        x98 = -d_1*(R_l_inv_61*x80 + R_l_inv_71*x87)
        x99 = R_l_inv_66*x80
        x100 = R_l_inv_76*x87
        x101 = x100 + x99
        x102 = d_2*x101
        x103 = -x13*x91
        x104 = R_l_inv_65*x80 + R_l_inv_75*x87
        x105 = -x104*x43
        x106 = -x102 + x103 + x105 + x95 + x96 + x97 + x98
        x107 = R_l_inv_64*x80
        x108 = R_l_inv_74*x87
        x109 = d_1 - x107 - x108
        x110 = x106 + x109
        x111 = x101*x50
        x112 = -x111
        x113 = -x50*x91
        x114 = x54*x88
        x115 = x113 + x114
        x116 = x89 + x92
        x117 = x102 + x103 + x105 + x95 + x96 + x97 + x98
        x118 = x109 + x117
        x119 = R_l_inv_67*x80 + R_l_inv_77*x87
        x120 = x119*x54
        x121 = x120 + x50
        x122 = x60*x80
        x123 = x62*x87
        x124 = -x122 - x123
        x125 = x101*x67
        x126 = x122 + x123
        x127 = -d_1 + x107 + x108
        x128 = x117 + x127
        x129 = x106 + x127
        x130 = Px*r_21
        x131 = Py*r_11
        x132 = r_11*x35 - r_21*x32 + x130 - x131
        x133 = R_l_inv_64*x132
        x134 = Px*r_22
        x135 = Py*r_12
        x136 = r_12*x35 - r_22*x32 + x134 - x135
        x137 = R_l_inv_74*x136
        x138 = d_1*(R_l_inv_61*x132 + R_l_inv_71*x136)
        x139 = R_l_inv_62*x132
        x140 = R_l_inv_72*x136
        x141 = x139 + x140
        x142 = x13*x141
        x143 = R_l_inv_65*x132 + R_l_inv_75*x136
        x144 = x143*x43
        x145 = x133 + x137 + x138 + x142 + x144
        x146 = R_l_inv_60*x132
        x147 = R_l_inv_70*x136
        x148 = x146 + x147
        x149 = a_0*x148
        x150 = d_2*x141
        x151 = x149 + x150
        x152 = R_l_inv_66*x132 + R_l_inv_76*x136
        x153 = d_2*x152
        x154 = a_0 + x153
        x155 = -x54
        x156 = x152*x50
        x157 = x148*x54
        x158 = x141*x50
        x159 = -x157 + x158
        x160 = -a_0
        x161 = -x153
        x162 = x160 + x161
        x163 = -x149
        x164 = -x150
        x165 = x163 + x164
        x166 = 2*d_1
        x167 = -x166
        x168 = x132*x60
        x169 = x136*x62
        x170 = R_l_inv_67*x132 + R_l_inv_77*x136
        x171 = -x170*x54
        x172 = x167 + x168 + x169 + x171
        x173 = x152*x67
        x174 = -x173
        x175 = x166 - x168 - x169 + x171
        x176 = -x133 - x137 + x138 + x142 + x144
        x177 = -x156
        x178 = Py*x22 + Py*x24 + Py*x26 - x22*x35 - x24*x35 - x26*x35
        x179 = 2*d_3
        x180 = x178*x179
        x181 = R_l_inv_40*x180
        x182 = Px*x22 + Px*x24 + Px*x26 - x22*x32 - x24*x32 - x26*x32
        x183 = x179*x182
        x184 = R_l_inv_50*x183
        x185 = 2*x35
        x186 = 2*x32
        x187 = 2*r_11
        x188 = r_23*x28
        x189 = r_13*x188
        x190 = 2*r_31
        x191 = r_33*x188
        x192 = r_21**3*x21 - r_21*x16 + r_21*x17 - r_21*x18 + r_21*x23 + r_21*x27 - r_21*x29 + r_21*x30 - r_21*x31 + r_21*x39 + x130*x186 - x131*x186 - x185*x74 - x185*x75 - x185*x76 + x187*x189 + x190*x191 + x36*x74 + x36*x76 - x36*x79
        x193 = R_l_inv_00*x192
        x194 = 2*x189
        x195 = r_13*r_33*x28
        x196 = r_11**3*x21 + r_11*x16 - r_11*x17 - r_11*x18 + r_11*x25 + r_11*x27 + r_11*x29 - r_11*x30 - r_11*x31 + r_11*x39 + r_21*x194 - x130*x185 + x131*x185 - x186*x74 - x186*x75 - x186*x76 + x190*x195 + x33*x75 + x33*x76 - x33*x79
        x197 = R_l_inv_20*x196
        x198 = r_21*x21
        x199 = x187*x198
        x200 = x190*x198
        x201 = 2*r_32
        x202 = r_12*x194 + r_12*x199 - r_22*x16 + r_22*x17 - r_22*x18 + r_22*x23 + 3*r_22*x25 + r_22*x27 - r_22*x29 + r_22*x30 - r_22*x31 + r_22*x39 + r_32*x200 + x134*x186 - x135*x186 - x185*x81 - x185*x82 - x185*x83 + x191*x201 + x36*x81 + x36*x83 - x36*x86
        x203 = R_l_inv_10*x202
        x204 = r_31*x187*x21
        x205 = r_12*x16 - r_12*x17 - r_12*x18 + 3*r_12*x23 + r_12*x25 + r_12*x27 + r_12*x29 - r_12*x30 - r_12*x31 + r_12*x39 + r_22*x194 + r_22*x199 + r_32*x204 - x134*x185 + x135*x185 - x186*x81 - x186*x82 - x186*x83 + x195*x201 + x33*x82 + x33*x83 - x33*x86
        x206 = R_l_inv_30*x205
        x207 = x181 + x184 + x193 + x197 + x203 + x206
        x208 = a_0*x207
        x209 = d_1*(R_l_inv_01*x192 + R_l_inv_11*x202 + R_l_inv_21*x196 + R_l_inv_31*x205 + R_l_inv_41*x180 + R_l_inv_51*x183)
        x210 = R_l_inv_42*x180
        x211 = R_l_inv_52*x183
        x212 = R_l_inv_02*x192
        x213 = R_l_inv_22*x196
        x214 = R_l_inv_12*x202
        x215 = R_l_inv_32*x205
        x216 = x210 + x211 + x212 + x213 + x214 + x215
        x217 = d_2*x216
        x218 = x13*x216
        x219 = R_l_inv_05*x192 + R_l_inv_15*x202 + R_l_inv_25*x196 + R_l_inv_35*x205 + R_l_inv_45*x180 + R_l_inv_55*x183
        x220 = x219*x43
        x221 = x208 + x209 + x217 + x218 + x220
        x222 = R_l_inv_06*x192 + R_l_inv_16*x202 + R_l_inv_26*x196 + R_l_inv_36*x205 + R_l_inv_46*x180 + R_l_inv_56*x183
        x223 = d_2*x222
        x224 = d_1*x50
        x225 = x223 + x224
        x226 = R_l_inv_04*x192
        x227 = R_l_inv_14*x202
        x228 = R_l_inv_24*x196
        x229 = R_l_inv_34*x205
        x230 = R_l_inv_44*x180
        x231 = R_l_inv_54*x183
        x232 = x226 + x227 + x228 + x229 + x230 + x231
        x233 = d_2*x67
        x234 = -x233
        x235 = x222*x50
        x236 = x207*x54
        x237 = x216*x50
        x238 = -x236 + x237
        x239 = -x223
        x240 = -x224
        x241 = x239 + x240
        x242 = -x208
        x243 = -x217
        x244 = x209 + x218 + x220 + x242 + x243
        x245 = 2*x40
        x246 = 2*x19
        x247 = 2*x41
        x248 = 4*d_3
        x249 = x178*x248
        x250 = R_l_inv_43*x249
        x251 = x182*x248
        x252 = R_l_inv_53*x251
        x253 = 2*x192
        x254 = R_l_inv_03*x253
        x255 = 2*x196
        x256 = R_l_inv_23*x255
        x257 = 2*x202
        x258 = R_l_inv_13*x257
        x259 = 2*x205
        x260 = R_l_inv_33*x259
        x261 = R_l_inv_07*x192 + R_l_inv_17*x202 + R_l_inv_27*x196 + R_l_inv_37*x205 + R_l_inv_47*x180 + R_l_inv_57*x183
        x262 = -x261*x54
        x263 = x245 - x246 - x247 + x250 + x252 + x254 + x256 + x258 + x260 + x262
        x264 = a_0*d_2
        x265 = -8*x264
        x266 = x222*x67
        x267 = -x266
        x268 = -x245 + x246 + x247 - x250 - x252 - x254 - x256 - x258 - x260 + x262
        x269 = -x226 - x227 - x228 - x229 - x230 - x231
        x270 = -x235
        x271 = R_l_inv_40*x183
        x272 = R_l_inv_50*x180
        x273 = R_l_inv_00*x196
        x274 = R_l_inv_20*x192
        x275 = R_l_inv_10*x205
        x276 = R_l_inv_30*x202
        x277 = x271 - x272 + x273 - x274 + x275 - x276
        x278 = a_0*x277
        x279 = R_l_inv_42*x183
        x280 = R_l_inv_52*x180
        x281 = R_l_inv_02*x196
        x282 = R_l_inv_22*x192
        x283 = R_l_inv_12*x205
        x284 = R_l_inv_32*x202
        x285 = x279 - x280 + x281 - x282 + x283 - x284
        x286 = d_2*x285
        x287 = x278 + x286
        x288 = d_1*(R_l_inv_01*x196 + R_l_inv_11*x205 - R_l_inv_21*x192 - R_l_inv_31*x202 + R_l_inv_41*x183 - R_l_inv_51*x180)
        x289 = R_l_inv_46*x183
        x290 = R_l_inv_56*x180
        x291 = R_l_inv_06*x196
        x292 = R_l_inv_26*x192
        x293 = R_l_inv_16*x205
        x294 = R_l_inv_36*x202
        x295 = x289 - x290 + x291 - x292 + x293 - x294
        x296 = d_2*x295
        x297 = x13*x285
        x298 = R_l_inv_05*x196 + R_l_inv_15*x205 - R_l_inv_25*x192 - R_l_inv_35*x202 + R_l_inv_45*x183 - R_l_inv_55*x180
        x299 = x298*x43
        x300 = x288 + x296 + x297 + x299
        x301 = R_l_inv_04*x196
        x302 = R_l_inv_14*x205
        x303 = R_l_inv_24*x192
        x304 = R_l_inv_34*x202
        x305 = R_l_inv_54*x180
        x306 = R_l_inv_44*x183
        x307 = x19 + x301 + x302 - x303 - x304 - x305 + x306 + x42
        x308 = x300 + x307
        x309 = x295*x50
        x310 = x277*x54
        x311 = x285*x50
        x312 = -x310 + x311
        x313 = -x278
        x314 = -x286
        x315 = x313 + x314
        x316 = x288 - x296 + x297 + x299
        x317 = x307 + x316
        x318 = a_0*x67
        x319 = R_l_inv_07*x196 + R_l_inv_17*x205 - R_l_inv_27*x192 - R_l_inv_37*x202 + R_l_inv_47*x183 - R_l_inv_57*x180
        x320 = -x319*x54
        x321 = x318 + x320
        x322 = R_l_inv_23*x253
        x323 = R_l_inv_33*x257
        x324 = R_l_inv_03*x255
        x325 = R_l_inv_13*x259
        x326 = R_l_inv_53*x249
        x327 = R_l_inv_43*x251
        x328 = -x322 - x323 + x324 + x325 - x326 + x327
        x329 = x295*x67
        x330 = -x329
        x331 = x322 + x323 - x324 - x325 + x326 - x327
        x332 = x20 - x301 - x302 + x303 + x304 + x305 - x306 + x40 + x41
        x333 = x316 + x332
        x334 = -x309
        x335 = x300 + x332
        x336 = 2*x12
        x337 = 2*x191
        x338 = r_21*x337 + r_31**3*x21 - r_31*x16 - r_31*x17 + r_31*x18 + r_31*x23 + r_31*x25 - r_31*x29 - r_31*x30 + r_31*x31 + r_31*x34 + r_31*x37 + x187*x195 - x336*x74 - x336*x75 - x336*x76 + x38*x74 + x38*x75 - x38*x77 - x38*x78
        x339 = R_l_inv_60*x338
        x340 = 2*r_12*x195 + r_12*x204 + r_22*x200 + r_22*x337 - r_32*x16 - r_32*x17 + r_32*x18 + r_32*x23 + r_32*x25 + 3*r_32*x27 - r_32*x29 - r_32*x30 + r_32*x31 + r_32*x34 + r_32*x37 - x336*x81 - x336*x82 - x336*x83 + x38*x81 + x38*x82 - x38*x84 - x38*x85
        x341 = R_l_inv_70*x340
        x342 = x339 + x341
        x343 = a_0*x342
        x344 = d_1*(R_l_inv_61*x338 + R_l_inv_71*x340)
        x345 = R_l_inv_62*x338
        x346 = R_l_inv_72*x340
        x347 = x345 + x346
        x348 = d_2*x347
        x349 = x13*x347
        x350 = R_l_inv_65*x338 + R_l_inv_75*x340
        x351 = x350*x43
        x352 = -x38*x95
        x353 = -x38*x96
        x354 = -x38*x97
        x355 = x336*x95
        x356 = x336*x96
        x357 = x336*x97
        x358 = x344 + x348 + x349 + x351 + x352 + x353 + x354 + x355 + x356 + x357
        x359 = x343 + x358
        x360 = R_l_inv_64*x338
        x361 = R_l_inv_74*x340
        x362 = x360 + x361
        x363 = R_l_inv_66*x338 + R_l_inv_76*x340
        x364 = d_2*x363
        x365 = d_1*x54
        x366 = x364 + x365
        x367 = x362 + x366
        x368 = x342*x54
        x369 = x347*x50
        x370 = -x368 + x369
        x371 = x363*x50
        x372 = x318 + x371
        x373 = -x343
        x374 = -x364 - x365
        x375 = x373 + x374
        x376 = x344 - x348 + x349 + x351 + x352 + x353 + x354 + x355 + x356 + x357
        x377 = x362 + x376
        x378 = R_l_inv_67*x338 + R_l_inv_77*x340
        x379 = -x378*x54
        x380 = 4*x264
        x381 = x379 + x380
        x382 = x338*x60
        x383 = x340*x62
        x384 = x382 + x383
        x385 = 4*x40
        x386 = 4*x19
        x387 = 4*x41
        x388 = x363*x67
        x389 = -x386 - x387 - x388
        x390 = -x382 - x383
        x391 = x379 - x380
        x392 = -x360 - x361
        x393 = -x318
        x394 = -x371 + x393
        x395 = x366 + x376 + x392
        x396 = x15*x50
        x397 = x54*(x0 - x396)
        x398 = 4*d_2
        x399 = -x54*(x0 + x396)
        x400 = 4*a_0
        x401 = -x400*x64
        x402 = 8*R_l_inv_63
        x403 = 8*R_l_inv_73
        x404 = 4 - x68
        x405 = x104*x50
        x406 = x54*(-x405 + x88)
        x407 = -x54*(x405 + x88)
        x408 = -x119*x400
        x409 = -x125
        x410 = x143*x50
        x411 = x410 - 1
        x412 = -x146 - x147
        x413 = x410 + 1
        x414 = x170*x400
        x415 = x219*x50
        x416 = x167 + x415
        x417 = -x181 - x184 - x193 - x197 - x203 - x206
        x418 = x166 + x415
        x419 = x261*x400
        x420 = 16*d_3
        x421 = R_l_inv_43*x420
        x422 = R_l_inv_53*x420
        x423 = 8*R_l_inv_03
        x424 = 8*R_l_inv_23
        x425 = 8*R_l_inv_13
        x426 = 8*R_l_inv_33
        x427 = x298*x50
        x428 = x427 + x50
        x429 = -x271 + x272 - x273 + x274 - x275 + x276
        x430 = -x279 + x280 - x281 + x282 - x283 + x284
        x431 = x319*x400
        x432 = -x50
        x433 = x427 + x432
        x434 = x350*x50
        x435 = x54*(-x339 - x341 + x434)
        x436 = x166 + x363
        x437 = x54*(x342 + x434)
        x438 = -x385
        x439 = x378*x400
        x440 = x386 + x387 + x388
        x441 = x2 + x4
        x442 = x53 - x55
        x443 = x1 + x5
        x444 = x90 + x92
        x445 = x113 - x114
        x446 = x89 + x93
        x447 = x120 + x432
        x448 = x149 + x164
        x449 = a_0 + x161
        x450 = x157 + x158
        x451 = x153 + x160
        x452 = x150 + x163
        x453 = x209 + x218 + x220 + x232
        x454 = x208 + x243
        x455 = x224 + x239
        x456 = x236 + x237
        x457 = x223 + x240
        x458 = x217 + x242
        x459 = x209 + x218 + x220 + x269
        x460 = x278 + x314
        x461 = x310 + x311
        x462 = x286 + x313
        x463 = x320 + x393
        x464 = x368 + x369
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = x49 + x6
        A[0, 1] = x52 + x56
        A[0, 2] = x57 + x59
        A[0, 3] = x66
        A[0, 4] = x69
        A[0, 5] = x70
        A[0, 6] = x6 + x72
        A[0, 7] = x51 + x56
        A[0, 8] = x57 + x73
        A[1, 0] = x110 + x94
        A[1, 1] = x112 + x115
        A[1, 2] = x116 + x118
        A[1, 3] = x121 + x124
        A[1, 4] = x125
        A[1, 5] = x121 + x126
        A[1, 6] = x128 + x94
        A[1, 7] = x111 + x115
        A[1, 8] = x116 + x129
        A[2, 0] = x145 + x151 + x154
        A[2, 1] = x155 + x156 + x159
        A[2, 2] = x145 + x162 + x165
        A[2, 3] = x172
        A[2, 4] = x174
        A[2, 5] = x175
        A[2, 6] = x151 + x162 + x176
        A[2, 7] = x159 + x177 + x54
        A[2, 8] = x154 + x165 + x176
        A[3, 0] = x221 + x225 + x232
        A[3, 1] = x234 + x235 + x238
        A[3, 2] = x232 + x241 + x244
        A[3, 3] = x263
        A[3, 4] = x265 + x267
        A[3, 5] = x268
        A[3, 6] = x221 + x241 + x269
        A[3, 7] = x233 + x238 + x270
        A[3, 8] = x225 + x244 + x269
        A[4, 0] = x287 + x308
        A[4, 1] = x309 + x312
        A[4, 2] = x315 + x317
        A[4, 3] = x321 + x328
        A[4, 4] = x330
        A[4, 5] = x321 + x331
        A[4, 6] = x287 + x333
        A[4, 7] = x312 + x334
        A[4, 8] = x315 + x335
        A[5, 0] = x359 + x367
        A[5, 1] = x370 + x372
        A[5, 2] = x375 + x377
        A[5, 3] = x381 + x384
        A[5, 4] = x385 + x389
        A[5, 5] = x390 + x391
        A[5, 6] = x359 + x374 + x392
        A[5, 7] = x370 + x394
        A[5, 8] = x373 + x395
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x397
        B[0, 1] = x398*(x10 + x3)
        B[0, 2] = x399
        B[0, 3] = x401 + x69
        B[0, 4] = r_31*x402 + r_32*x403
        B[0, 5] = x401 + x404
        B[0, 6] = x397
        B[0, 7] = x398*(x3 - x8 - x9)
        B[0, 8] = x399
        B[1, 0] = x406
        B[1, 1] = x398*(x101 + x91)
        B[1, 2] = x407
        B[1, 3] = x125 + x408
        B[1, 4] = x402*x80 + x403*x87
        B[1, 5] = x408 + x409
        B[1, 6] = x406
        B[1, 7] = x398*(-x100 + x91 - x99)
        B[1, 8] = x407
        B[2, 0] = x54*(x411 + x412)
        B[2, 1] = -x398*(x141 + x152)
        B[2, 2] = x54*(x148 + x413)
        B[2, 3] = x174 + x414
        B[2, 4] = 8*d_1 - x132*x402 - x136*x403
        B[2, 5] = x173 + x414
        B[2, 6] = x54*(x412 + x413)
        B[2, 7] = x398*(-x139 - x140 + x152)
        B[2, 8] = x54*(x148 + x411)
        B[3, 0] = x54*(x416 + x417)
        B[3, 1] = -x398*(x216 + x222)
        B[3, 2] = x54*(x207 + x418)
        B[3, 3] = x267 + x419
        B[3, 4] = -x178*x421 - x182*x422 + 8*x19 - x192*x423 - x196*x424 - x202*x425 - x205*x426 + 8*x40 + 8*x41
        B[3, 5] = x266 + x419
        B[3, 6] = x54*(x417 + x418)
        B[3, 7] = x398*(-x210 - x211 - x212 - x213 - x214 - x215 + x222)
        B[3, 8] = x54*(x207 + x416)
        B[4, 0] = x54*(x428 + x429)
        B[4, 1] = x398*(-x289 + x290 - x291 + x292 - x293 + x294 + x430)
        B[4, 2] = x54*(x277 + x428)
        B[4, 3] = x330 + x431
        B[4, 4] = x178*x422 - x182*x421 + x192*x424 - x196*x423 + x202*x426 - x205*x425
        B[4, 5] = x329 + x431
        B[4, 6] = x54*(x429 + x433)
        B[4, 7] = x398*(x295 + x430)
        B[4, 8] = x54*(x277 + x433)
        B[5, 0] = x435
        B[5, 1] = -x398*(x347 + x436)
        B[5, 2] = x437
        B[5, 3] = x389 + x438 + x439
        B[5, 4] = -x338*x402 - x340*x403
        B[5, 5] = x385 + x439 + x440
        B[5, 6] = x435
        B[5, 7] = x398*(-x345 - x346 + x436)
        B[5, 8] = x437
        C = np.zeros(shape=(6, 9))
        C[0, 0] = x441 + x59
        C[0, 1] = x442 + x52
        C[0, 2] = x443 + x49
        C[0, 3] = x70
        C[0, 4] = x404
        C[0, 5] = x66
        C[0, 6] = x441 + x73
        C[0, 7] = x442 + x51
        C[0, 8] = x443 + x72
        C[1, 0] = x118 + x444
        C[1, 1] = x112 + x445
        C[1, 2] = x110 + x446
        C[1, 3] = x126 + x447
        C[1, 4] = x409
        C[1, 5] = x124 + x447
        C[1, 6] = x129 + x444
        C[1, 7] = x111 + x445
        C[1, 8] = x128 + x446
        C[2, 0] = x145 + x448 + x449
        C[2, 1] = x156 + x450 + x54
        C[2, 2] = x145 + x451 + x452
        C[2, 3] = x175
        C[2, 4] = x173
        C[2, 5] = x172
        C[2, 6] = x176 + x448 + x451
        C[2, 7] = x155 + x177 + x450
        C[2, 8] = x176 + x449 + x452
        C[3, 0] = x453 + x454 + x455
        C[3, 1] = x233 + x235 + x456
        C[3, 2] = x453 + x457 + x458
        C[3, 3] = x268
        C[3, 4] = x265 + x266
        C[3, 5] = x263
        C[3, 6] = x454 + x457 + x459
        C[3, 7] = x234 + x270 + x456
        C[3, 8] = x455 + x458 + x459
        C[4, 0] = x317 + x460
        C[4, 1] = x309 + x461
        C[4, 2] = x308 + x462
        C[4, 3] = x331 + x463
        C[4, 4] = x329
        C[4, 5] = x328 + x463
        C[4, 6] = x335 + x460
        C[4, 7] = x334 + x461
        C[4, 8] = x333 + x462
        C[5, 0] = x343 + x374 + x377
        C[5, 1] = x372 + x464
        C[5, 2] = x358 + x367 + x373
        C[5, 3] = x381 + x390
        C[5, 4] = x438 + x440
        C[5, 5] = x384 + x391
        C[5, 6] = x343 + x395
        C[5, 7] = x394 + x464
        C[5, 8] = x358 + x375 + x392
        from solver.general_6dof.numerical_reduce_closure_equation import compute_solution_from_tanhalf_LME
        local_solutions = compute_solution_from_tanhalf_LME(A, B, C)
        for local_solutions_i in local_solutions:
            solution_i: IkSolution = make_ik_solution()
            solution_i[3] = local_solutions_i
            appended_idx = append_solution_to_queue(solution_i)
            add_input_index_to(2, appended_idx)
    # Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor()
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
            th_2 = this_solution[3]
            condition_0: bool = (2*abs(d_1*d_3) >= zero_tolerance) or (2*abs(a_0*d_3*math.cos(th_2)) >= zero_tolerance) or (abs(Px**2 - 2*Px*d_4*r_13 + Py**2 - 2*Py*d_4*r_23 + Pz**2 - 2*Pz*d_4*r_33 - a_0**2 + 2*a_0*d_2*math.sin(th_2) - d_1**2 - d_2**2 - d_3**2 + d_4**2*r_13**2 + d_4**2*r_23**2 + d_4**2*r_33**2) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_2)
                x1 = 2*d_3
                x2 = math.atan2(-a_0*x0*x1, -d_1*x1)
                x3 = d_1**2
                x4 = d_3**2
                x5 = 4*x4
                x6 = a_0**2
                x7 = 2*d_4
                x8 = d_4**2
                x9 = Px**2 - Px*r_13*x7 + Py**2 - Py*r_23*x7 + Pz**2 - Pz*r_33*x7 + 2*a_0*d_2*math.sin(th_2) - d_2**2 + r_13**2*x8 + r_23**2*x8 + r_33**2*x8 - x3 - x4 - x6
                x10 = math.sqrt(x0**2*x5*x6 + x3*x5 - x9**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x2 + math.atan2(x10, x9)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (2*abs(d_1*d_3) >= zero_tolerance) or (2*abs(a_0*d_3*math.cos(th_2)) >= zero_tolerance) or (abs(Px**2 - 2*Px*d_4*r_13 + Py**2 - 2*Py*d_4*r_23 + Pz**2 - 2*Pz*d_4*r_33 - a_0**2 + 2*a_0*d_2*math.sin(th_2) - d_1**2 - d_2**2 - d_3**2 + d_4**2*r_13**2 + d_4**2*r_23**2 + d_4**2*r_33**2) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.cos(th_2)
                x1 = 2*d_3
                x2 = math.atan2(-a_0*x0*x1, -d_1*x1)
                x3 = d_1**2
                x4 = d_3**2
                x5 = 4*x4
                x6 = a_0**2
                x7 = 2*d_4
                x8 = d_4**2
                x9 = Px**2 - Px*r_13*x7 + Py**2 - Py*r_23*x7 + Pz**2 - Pz*r_33*x7 + 2*a_0*d_2*math.sin(th_2) - d_2**2 + r_13**2*x8 + r_23**2*x8 + r_33**2*x8 - x3 - x4 - x6
                x10 = math.sqrt(x0**2*x5*x6 + x3*x5 - x9**2)
                # End of temp variables
                this_solution[4] = x2 + math.atan2(-x10, x9)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_3_processor()
    # Finish code for explicit solution node 2
    
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
            th_3 = this_solution[4]
            checked_result: bool = (abs(Px - d_4*r_13) <= 1.0e-6) and (abs(Py - d_4*r_23) <= 1.0e-6) and (abs(d_1 - d_3*math.cos(th_3)) <= 1.0e-6)
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
            th_3 = this_solution[4]
            condition_0: bool = (abs(Px - d_4*r_13) >= zero_tolerance) or (abs(Py - d_4*r_23) >= zero_tolerance) or (abs(d_1 - d_3*math.cos(th_3)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = Px - d_4*r_13
                x1 = -Py + d_4*r_23
                x2 = math.atan2(x0, x1)
                x3 = -d_1 + d_3*math.cos(th_3)
                x4 = math.sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x2 + math.atan2(x4, x3)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                
            condition_1: bool = (abs(Px - d_4*r_13) >= zero_tolerance) or (abs(Py - d_4*r_23) >= zero_tolerance) or (abs(d_1 - d_3*math.cos(th_3)) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = Px - d_4*r_13
                x1 = -Py + d_4*r_23
                x2 = math.atan2(x0, x1)
                x3 = -d_1 + d_3*math.cos(th_3)
                x4 = math.sqrt(x0**2 + x1**2 - x3**2)
                # End of temp variables
                this_solution[0] = x2 + math.atan2(-x4, x3)
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
            th_3 = this_solution[4]
            degenerate_valid_0 = (abs(th_3) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(12, node_input_i_idx_in_queue)
            
            th_3 = this_solution[4]
            degenerate_valid_1 = (abs(th_3 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(17, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(7, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_6_processor()
    # Finish code for solved_variable dispatcher node 6
    
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
                this_solution[2] = math.atan2(r_13*math.cos(th_0) + r_23*math.sin(th_0), r_33)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(18, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_1th_2th_4_soa_processor()
    # Finish code for explicit solution node 17
    
    # Code for non-branch dispatcher node 18
    # Actually, there is no code
    
    # Code for explicit solution node 19, solved variable is th_5
    def ExplicitSolutionNode_node_19_solve_th_5_processor():
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
            th_0 = this_solution[0]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                # End of temp variables
                this_solution[6] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(20, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_5_processor()
    # Finish code for explicit solution node 18
    
    # Code for solved_variable dispatcher node 20
    def SolvedVariableDispatcherNode_node_20_processor():
        this_node_input_index: List[int] = node_input_index[20]
        this_input_valid: bool = node_input_validity[20]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_2 = this_solution[3]
            degenerate_valid_0 = (abs(th_2 - 1/2*math.pi) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(30, node_input_i_idx_in_queue)
            
            th_2 = this_solution[3]
            degenerate_valid_1 = (abs(th_2 + (1/2)*math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(33, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(21, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_20_processor()
    # Finish code for solved_variable dispatcher node 20
    
    # Code for explicit solution node 33, solved variable is th_1
    def ExplicitSolutionNode_node_33_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[33]
        this_input_valid: bool = node_input_validity[33]
        if not this_input_valid:
            return
        
        # The solution of non-root node 33
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_3 = this_solution[4]
            condition_0: bool = (abs(d_3*math.sin(th_3)) >= 1.0e-6) or (abs(a_0 + d_2) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_0 + d_2
                x1 = Pz - d_4*r_33
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3
                x5 = d_3*math.sin(th_3)
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x1 - x4*x5, x0*x4 - x1*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(34, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_33_solve_th_1_processor()
    # Finish code for explicit solution node 33
    
    # Code for non-branch dispatcher node 34
    # Actually, there is no code
    
    # Code for explicit solution node 35, solved variable is th_4
    def ExplicitSolutionNode_node_35_solve_th_4_processor():
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
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_13*(math.sin(th_0)*math.sin(th_3) + math.sin(th_1)*math.cos(th_0)*math.cos(th_3)) + r_23*(math.sin(th_0)*math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) + r_33*math.cos(th_1)*math.cos(th_3)) >= zero_tolerance) or (abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance)
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
                this_solution[5] = math.atan2(-r_13*(x2*x3 + x4*x6) - r_23*(x2*x6 - x3*x4) - r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_35_solve_th_4_processor()
    # Finish code for explicit solution node 34
    
    # Code for explicit solution node 30, solved variable is th_1
    def ExplicitSolutionNode_node_30_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[30]
        this_input_valid: bool = node_input_validity[30]
        if not this_input_valid:
            return
        
        # The solution of non-root node 30
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_3 = this_solution[4]
            condition_0: bool = (abs(d_3*math.sin(th_3)) >= 1.0e-6) or (abs(a_0 - d_2) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_0 - d_2
                x1 = Pz - d_4*r_33
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3
                x5 = d_3*math.sin(th_3)
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(31, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_30_solve_th_1_processor()
    # Finish code for explicit solution node 30
    
    # Code for non-branch dispatcher node 31
    # Actually, there is no code
    
    # Code for explicit solution node 32, solved variable is th_4
    def ExplicitSolutionNode_node_32_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[31]
        this_input_valid: bool = node_input_validity[31]
        if not this_input_valid:
            return
        
        # The solution of non-root node 32
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance) or (abs(-r_13*(math.sin(th_0)*math.sin(th_3) - math.sin(th_1)*math.cos(th_0)*math.cos(th_3)) + r_23*(math.sin(th_0)*math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_0)) + r_33*math.cos(th_1)*math.cos(th_3)) >= zero_tolerance) or (abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.cos(th_3)
                x2 = math.sin(th_3)
                x3 = math.cos(th_0)
                x4 = math.sin(th_0)
                x5 = math.sin(th_1)
                x6 = x1*x5
                # End of temp variables
                this_solution[5] = math.atan2(-r_13*(x2*x4 - x3*x6) + r_23*(x2*x3 + x4*x6) + r_33*x0*x1, r_13*x0*x3 + r_23*x0*x4 - r_33*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_32_solve_th_4_processor()
    # Finish code for explicit solution node 31
    
    # Code for explicit solution node 21, solved variable is th_1
    def ExplicitSolutionNode_node_21_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[21]
        this_input_valid: bool = node_input_validity[21]
        if not this_input_valid:
            return
        
        # The solution of non-root node 21
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_1th_2th_4_soa = this_solution[2]
            th_2 = this_solution[3]
            condition_0: bool = (abs(d_2*math.cos(th_2)) >= 1.0e-6) or (abs(a_0 - d_2*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = Pz - d_4*r_33
                x1 = a_0 - d_2*math.sin(th_2)
                x2 = Px*math.cos(th_0) + Py*math.sin(th_0) - d_4*math.sin(th_1th_2th_4_soa)
                x3 = d_2*math.cos(th_2)
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x1 - x2*x3, -x0*x3 + x1*x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(22, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_1_processor()
    # Finish code for explicit solution node 21
    
    # Code for non-branch dispatcher node 22
    # Actually, there is no code
    
    # Code for explicit solution node 23, solved variable is th_4
    def ExplicitSolutionNode_node_23_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[22]
        this_input_valid: bool = node_input_validity[22]
        if not this_input_valid:
            return
        
        # The solution of non-root node 23
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[1]
            th_1th_2th_4_soa = this_solution[2]
            th_2 = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[5] = -th_1 + th_1th_2th_4_soa - th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_4_processor()
    # Finish code for explicit solution node 22
    
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
                this_solution[6] = math.atan2(-r_11*x1 + r_21*x0, -r_12*x1 + r_22*x0)
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
            th_2 = this_solution[3]
            degenerate_valid_0 = (abs(th_2 - 1/2*math.pi) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(24, node_input_i_idx_in_queue)
            
            th_2 = this_solution[3]
            degenerate_valid_1 = (abs(th_2 + (1/2)*math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(27, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(14, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_13_processor()
    # Finish code for solved_variable dispatcher node 13
    
    # Code for explicit solution node 27, solved variable is th_1
    def ExplicitSolutionNode_node_27_solve_th_1_processor():
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
            th_0 = this_solution[0]
            th_3 = this_solution[4]
            condition_0: bool = (abs(d_3*math.sin(th_3)) >= 1.0e-6) or (abs(a_0 + d_2) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_0 + d_2
                x1 = Pz - d_4*r_33
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3
                x5 = d_3*math.sin(th_3)
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x1 - x4*x5, x0*x4 - x1*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(28, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_1_processor()
    # Finish code for explicit solution node 27
    
    # Code for non-branch dispatcher node 28
    # Actually, there is no code
    
    # Code for explicit solution node 29, solved variable is th_4
    def ExplicitSolutionNode_node_29_solve_th_4_processor():
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
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_13*(math.sin(th_0)*math.sin(th_3) + math.sin(th_1)*math.cos(th_0)*math.cos(th_3)) + r_23*(math.sin(th_0)*math.sin(th_1)*math.cos(th_3) - math.sin(th_3)*math.cos(th_0)) + r_33*math.cos(th_1)*math.cos(th_3)) >= zero_tolerance) or (abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance)
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
                this_solution[5] = math.atan2(-r_13*(x2*x3 + x4*x6) - r_23*(x2*x6 - x3*x4) - r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_4_processor()
    # Finish code for explicit solution node 28
    
    # Code for explicit solution node 24, solved variable is th_1
    def ExplicitSolutionNode_node_24_solve_th_1_processor():
        this_node_input_index: List[int] = node_input_index[24]
        this_input_valid: bool = node_input_validity[24]
        if not this_input_valid:
            return
        
        # The solution of non-root node 24
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[0]
            th_3 = this_solution[4]
            condition_0: bool = (abs(d_3*math.sin(th_3)) >= 1.0e-6) or (abs(a_0 - d_2) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = a_0 - d_2
                x1 = Pz - d_4*r_33
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3
                x5 = d_3*math.sin(th_3)
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(25, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_1_processor()
    # Finish code for explicit solution node 24
    
    # Code for non-branch dispatcher node 25
    # Actually, there is no code
    
    # Code for explicit solution node 26, solved variable is th_4
    def ExplicitSolutionNode_node_26_solve_th_4_processor():
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
            th_0 = this_solution[0]
            th_1 = this_solution[1]
            th_3 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance) or (abs(-r_13*(math.sin(th_0)*math.sin(th_3) - math.sin(th_1)*math.cos(th_0)*math.cos(th_3)) + r_23*(math.sin(th_0)*math.sin(th_1)*math.cos(th_3) + math.sin(th_3)*math.cos(th_0)) + r_33*math.cos(th_1)*math.cos(th_3)) >= zero_tolerance) or (abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.cos(th_3)
                x2 = math.sin(th_3)
                x3 = math.cos(th_0)
                x4 = math.sin(th_0)
                x5 = math.sin(th_1)
                x6 = x1*x5
                # End of temp variables
                this_solution[5] = math.atan2(-r_13*(x2*x4 - x3*x6) + r_23*(x2*x3 + x4*x6) + r_33*x0*x1, r_13*x0*x3 + r_23*x0*x4 - r_33*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_26_solve_th_4_processor()
    # Finish code for explicit solution node 25
    
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
            th_2 = this_solution[3]
            condition_0: bool = (abs(d_2*math.cos(th_2)) >= 1.0e-6) or (abs(a_0 - d_2*math.sin(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = Pz - d_4*r_33
                x1 = a_0 - d_2*math.sin(th_2)
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3
                x5 = d_2*math.cos(th_2)
                # End of temp variables
                this_solution[1] = math.atan2(-x0*x1 - x4*x5, -x0*x5 + x1*x4)
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
            th_2 = this_solution[3]
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
                this_solution[5] = math.atan2(r_33*x4 - x5*x6 - x5*x7, r_33*x5 + x4*x6 + x4*x7)
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
            th_3 = this_solution[4]
            condition_0: bool = (abs((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3)) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.asin((-r_13*math.sin(th_0) + r_23*math.cos(th_0))/math.sin(th_3))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[5] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
            condition_1: bool = (abs((r_13*math.sin(th_0) - r_23*math.cos(th_0))/math.sin(th_3)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.asin((-r_13*math.sin(th_0) + r_23*math.cos(th_0))/math.sin(th_3))
                # End of temp variables
                this_solution[5] = math.pi - x0
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
            th_2 = this_solution[3]
            th_3 = this_solution[4]
            checked_result: bool = (abs(d_2*math.cos(th_2) - d_3*math.sin(th_2)*math.sin(th_3)) <= 1.0e-6) and (abs(-a_0 + d_2*math.sin(th_2) + d_3*math.sin(th_3)*math.cos(th_2)) <= 1.0e-6)
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
            th_2 = this_solution[3]
            th_3 = this_solution[4]
            condition_0: bool = (abs(d_2*math.cos(th_2) - d_3*math.sin(th_2)*math.sin(th_3)) >= 1.0e-6) or (abs(-a_0 + d_2*math.sin(th_2) + d_3*math.sin(th_3)*math.cos(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = d_4*r_33
                x1 = math.sin(th_2)
                x2 = math.cos(th_2)
                x3 = d_3*math.sin(th_3)
                x4 = a_0 - d_2*x1 - x2*x3
                x5 = -d_2*x2 + x1*x3
                x6 = math.cos(th_0)
                x7 = math.sin(th_0)
                x8 = Px*x6 + Py*x7 - d_4*r_13*x6 - d_4*r_23*x7
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
            th_2 = this_solution[3]
            th_3 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance) or (abs(-r_11*((math.sin(th_1)*math.sin(th_2) - math.cos(th_1)*math.cos(th_2))*math.sin(th_3)*math.cos(th_0) + math.sin(th_0)*math.cos(th_3)) + r_21*((-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_0)*math.sin(th_3) + math.cos(th_0)*math.cos(th_3)) - r_31*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_3)) >= zero_tolerance) or (abs(-r_12*((math.sin(th_1)*math.sin(th_2) - math.cos(th_1)*math.cos(th_2))*math.sin(th_3)*math.cos(th_0) + math.sin(th_0)*math.cos(th_3)) + r_22*((-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_0)*math.sin(th_3) + math.cos(th_0)*math.cos(th_3)) - r_32*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_3)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_3)
                x1 = math.sin(th_1)
                x2 = math.cos(th_2)
                x3 = math.sin(th_2)
                x4 = math.cos(th_1)
                x5 = x0*(x1*x2 + x3*x4)
                x6 = math.cos(th_0)
                x7 = math.cos(th_3)
                x8 = math.sin(th_0)
                x9 = x2*x4
                x10 = x1*x3
                x11 = x0*x8*(-x10 + x9) + x6*x7
                x12 = x0*x6*(x10 - x9) + x7*x8
                # End of temp variables
                this_solution[6] = math.atan2(-r_11*x12 + r_21*x11 - r_31*x5, -r_12*x12 + r_22*x11 - r_32*x5)
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
        value_at_2 = ik_out_i[3]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[4]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[5]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[6]  # th_5
        new_ik_i[5] = value_at_5
        ik_out.append(new_ik_i)
    return ik_out


def denso_cvrb_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = denso_cvrb_ik_target_original_to_raw(T_ee)
    ik_output_raw = denso_cvrb_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ee_pose_i = denso_cvrb_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_denso_cvrb():
    theta_in = np.random.random(size=(6, ))
    ee_pose = denso_cvrb_fk(theta_in)
    ik_output = denso_cvrb_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = denso_cvrb_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_denso_cvrb()
