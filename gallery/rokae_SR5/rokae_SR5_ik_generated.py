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
    
    # Code for explicit solution node 1, solved variable is th_2
    def General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return
        
        # The general 6-dof solution of root node with semi-symbolic reduce
        R_l = np.zeros(shape=(8, 8))
        R_l[0, 0] = d_4*r_21
        R_l[0, 1] = d_4*r_22
        R_l[0, 2] = d_4*r_11
        R_l[0, 3] = d_4*r_12
        R_l[0, 4] = Py - d_5*r_23
        R_l[0, 5] = Px - d_5*r_13
        R_l[1, 0] = d_4*r_11
        R_l[1, 1] = d_4*r_12
        R_l[1, 2] = -d_4*r_21
        R_l[1, 3] = -d_4*r_22
        R_l[1, 4] = Px - d_5*r_13
        R_l[1, 5] = -Py + d_5*r_23
        R_l[2, 6] = d_4*r_31
        R_l[2, 7] = d_4*r_32
        R_l[3, 0] = -r_21
        R_l[3, 1] = -r_22
        R_l[3, 2] = -r_11
        R_l[3, 3] = -r_12
        R_l[4, 0] = -r_11
        R_l[4, 1] = -r_12
        R_l[4, 2] = r_21
        R_l[4, 3] = r_22
        R_l[5, 6] = 2*Px*d_4*r_11 + 2*Py*d_4*r_21 + 2*Pz*d_4*r_31 - 2*d_4*d_5*r_11*r_13 - 2*d_4*d_5*r_21*r_23 - 2*d_4*d_5*r_31*r_33
        R_l[5, 7] = 2*Px*d_4*r_12 + 2*Py*d_4*r_22 + 2*Pz*d_4*r_32 - 2*d_4*d_5*r_12*r_13 - 2*d_4*d_5*r_22*r_23 - 2*d_4*d_5*r_32*r_33
        R_l[6, 0] = -Px*r_31 + Pz*r_11 - d_5*r_11*r_33 + d_5*r_13*r_31
        R_l[6, 1] = -Px*r_32 + Pz*r_12 - d_5*r_12*r_33 + d_5*r_13*r_32
        R_l[6, 2] = Py*r_31 - Pz*r_21 + d_5*r_21*r_33 - d_5*r_23*r_31
        R_l[6, 3] = Py*r_32 - Pz*r_22 + d_5*r_22*r_33 - d_5*r_23*r_32
        R_l[7, 0] = Py*r_31 - Pz*r_21 + d_5*r_21*r_33 - d_5*r_23*r_31
        R_l[7, 1] = Py*r_32 - Pz*r_22 + d_5*r_22*r_33 - d_5*r_23*r_32
        R_l[7, 2] = Px*r_31 - Pz*r_11 + d_5*r_11*r_33 - d_5*r_13*r_31
        R_l[7, 3] = Px*r_32 - Pz*r_12 + d_5*r_12*r_33 - d_5*r_13*r_32
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
        x0 = R_l_inv_66*r_31 + R_l_inv_76*r_32
        x1 = d_3*x0
        x2 = -x1
        x3 = d_5*r_33
        x4 = Pz - x3
        x5 = R_l_inv_62*r_31 + R_l_inv_72*r_32
        x6 = -x4*x5
        x7 = R_l_inv_65*r_31 + R_l_inv_75*r_32
        x8 = Px**2
        x9 = Py**2
        x10 = Pz**2
        x11 = d_4**2
        x12 = r_11**2
        x13 = x11*x12
        x14 = r_21**2
        x15 = x11*x14
        x16 = r_31**2
        x17 = x11*x16
        x18 = d_5**2
        x19 = r_13**2*x18
        x20 = r_23**2*x18
        x21 = r_33**2*x18
        x22 = d_5*r_13
        x23 = 2*Px
        x24 = x22*x23
        x25 = d_5*r_23
        x26 = 2*Py
        x27 = x25*x26
        x28 = 2*Pz
        x29 = x28*x3
        x30 = a_2**2
        x31 = a_3**2
        x32 = d_3**2
        x33 = -x30 - x31 - x32
        x34 = x10 + x13 + x15 + x17 + x19 + x20 + x21 - x24 - x27 - x29 + x33 + x8 + x9
        x35 = -x34*x7
        x36 = 2*a_3
        x37 = a_2*x36
        x38 = x37*x7
        x39 = -x38
        x40 = x2 + x35 + x39 + x6
        x41 = R_l_inv_60*r_31 + R_l_inv_70*r_32
        x42 = a_3*x41
        x43 = d_3*x5
        x44 = x42 - x43
        x45 = a_2*x41
        x46 = -x45
        x47 = R_l_inv_64*r_31
        x48 = R_l_inv_74*r_32
        x49 = -x47 - x48
        x50 = x46 + x49
        x51 = x0*x36
        x52 = 2*a_2
        x53 = x0*x52
        x54 = -x53
        x55 = -x5*x52
        x56 = x36*x5
        x57 = 2*d_3
        x58 = x41*x57
        x59 = x56 + x58
        x60 = x55 + x59
        x61 = -x42
        x62 = x43 + x45 + x61
        x63 = x1 + x35 + x39 + x6
        x64 = 2*R_l_inv_63
        x65 = r_31*x64
        x66 = 2*R_l_inv_73
        x67 = r_32*x66
        x68 = R_l_inv_67*r_31 + R_l_inv_77*r_32
        x69 = x57*x68
        x70 = -x65 - x67 + x69
        x71 = x65 + x67 + x69
        x72 = x47 + x48
        x73 = x46 + x72
        x74 = -x51
        x75 = d_4*x12
        x76 = d_4*x14
        x77 = d_4*x16
        x78 = Px*r_11
        x79 = Py*r_21
        x80 = Pz*r_31
        x81 = r_11*x22
        x82 = r_21*x25
        x83 = r_31*x3
        x84 = x78 + x79 + x80 - x81 - x82 - x83
        x85 = Px*r_12
        x86 = Py*r_22
        x87 = Pz*r_32
        x88 = r_12*x22
        x89 = r_22*x25
        x90 = r_32*x3
        x91 = x85 + x86 + x87 - x88 - x89 - x90
        x92 = R_l_inv_66*x84 + R_l_inv_76*x91
        x93 = d_3*x92
        x94 = -x93
        x95 = R_l_inv_62*x84 + R_l_inv_72*x91
        x96 = -x4*x95
        x97 = R_l_inv_65*x84 + R_l_inv_75*x91
        x98 = -x34*x97
        x99 = x37*x97
        x100 = -x99
        x101 = x100 + x75 + x76 + x77 + x94 + x96 + x98
        x102 = R_l_inv_60*x84 + R_l_inv_70*x91
        x103 = a_3*x102
        x104 = d_3*x95
        x105 = x103 - x104
        x106 = a_2*x102
        x107 = -x106
        x108 = R_l_inv_64*x84
        x109 = R_l_inv_74*x91
        x110 = -x108 - x109
        x111 = x107 + x110
        x112 = x36*x92
        x113 = x52*x92
        x114 = -x113
        x115 = -x52*x95
        x116 = x36*x95
        x117 = x102*x57
        x118 = x116 + x117
        x119 = x115 + x118
        x120 = -x103
        x121 = x104 + x106 + x120
        x122 = x100 + x75 + x76 + x77 + x93 + x96 + x98
        x123 = R_l_inv_67*x84 + R_l_inv_77*x91
        x124 = x123*x57 - x36
        x125 = x124 + x52
        x126 = x64*x84
        x127 = x66*x91
        x128 = -x126 - x127
        x129 = x126 + x127
        x130 = x108 + x109
        x131 = x107 + x130
        x132 = -x112
        x133 = Px*r_21
        x134 = Py*r_11
        x135 = r_11*x25 - r_21*x22 + x133 - x134
        x136 = Px*r_22
        x137 = Py*r_12
        x138 = r_12*x25 - r_22*x22 + x136 - x137
        x139 = R_l_inv_62*x135 + R_l_inv_72*x138
        x140 = d_3*x139
        x141 = x139*x4
        x142 = R_l_inv_65*x135 + R_l_inv_75*x138
        x143 = x142*x34
        x144 = R_l_inv_60*x135 + R_l_inv_70*x138
        x145 = a_3*x144
        x146 = x140 + x141 + x143 - x145
        x147 = R_l_inv_64*x135
        x148 = R_l_inv_74*x138
        x149 = x142*x37
        x150 = x147 + x148 + x149
        x151 = -a_3
        x152 = a_2*x144
        x153 = R_l_inv_66*x135 + R_l_inv_76*x138
        x154 = d_3*x153
        x155 = x151 + x152 + x154
        x156 = x153*x52
        x157 = x153*x36
        x158 = -x157 - x57
        x159 = x139*x52
        x160 = x139*x36
        x161 = x144*x57
        x162 = -x160 - x161
        x163 = x159 + x162
        x164 = -x154
        x165 = -a_2
        x166 = a_3 + x165
        x167 = x164 + x166
        x168 = -x152
        x169 = -x140 + x141 + x143 + x145
        x170 = x168 + x169
        x171 = x135*x64
        x172 = x138*x66
        x173 = R_l_inv_67*x135 + R_l_inv_77*x138
        x174 = -x173*x57
        x175 = x171 + x172 + x174
        x176 = -x171 - x172 + x174
        x177 = -x147 - x148
        x178 = x149 + x177
        x179 = -x156
        x180 = x157 + x57
        x181 = x151 + x154
        x182 = Py*x12 + Py*x14 + Py*x16 - x12*x25 - x14*x25 - x16*x25
        x183 = 2*d_4
        x184 = x182*x183
        x185 = Px*x12 + Px*x14 + Px*x16 - x12*x22 - x14*x22 - x16*x22
        x186 = x183*x185
        x187 = 2*x25
        x188 = 2*x22
        x189 = 2*r_11
        x190 = r_23*x18
        x191 = r_13*x190
        x192 = 2*r_31
        x193 = r_33*x190
        x194 = r_21**3*x11 - r_21*x10 + r_21*x13 + r_21*x17 - r_21*x19 + r_21*x20 - r_21*x21 + r_21*x29 - r_21*x8 + r_21*x9 + x133*x188 - x134*x188 - x187*x78 - x187*x79 - x187*x80 + x189*x191 + x192*x193 + x26*x78 + x26*x80 - x26*x83
        x195 = 2*x191
        x196 = r_13*r_33*x18
        x197 = r_11**3*x11 - r_11*x10 + r_11*x15 + r_11*x17 + r_11*x19 - r_11*x20 - r_11*x21 + r_11*x29 + r_11*x8 - r_11*x9 + r_21*x195 - x133*x187 + x134*x187 - x188*x78 - x188*x79 - x188*x80 + x192*x196 + x23*x79 + x23*x80 - x23*x83
        x198 = r_21*x11
        x199 = x189*x198
        x200 = x192*x198
        x201 = 2*r_32
        x202 = r_12*x195 + r_12*x199 - r_22*x10 + r_22*x13 + 3*r_22*x15 + r_22*x17 - r_22*x19 + r_22*x20 - r_22*x21 + r_22*x29 - r_22*x8 + r_22*x9 + r_32*x200 + x136*x188 - x137*x188 - x187*x85 - x187*x86 - x187*x87 + x193*x201 + x26*x85 + x26*x87 - x26*x90
        x203 = r_31*x11*x189
        x204 = -r_12*x10 + 3*r_12*x13 + r_12*x15 + r_12*x17 + r_12*x19 - r_12*x20 - r_12*x21 + r_12*x29 + r_12*x8 - r_12*x9 + r_22*x195 + r_22*x199 + r_32*x203 - x136*x187 + x137*x187 - x188*x85 - x188*x86 - x188*x87 + x196*x201 + x23*x86 + x23*x87 - x23*x90
        x205 = R_l_inv_06*x194 + R_l_inv_16*x202 + R_l_inv_26*x197 + R_l_inv_36*x204 + R_l_inv_46*x184 + R_l_inv_56*x186
        x206 = d_3*x205
        x207 = R_l_inv_02*x194 + R_l_inv_12*x202 + R_l_inv_22*x197 + R_l_inv_32*x204 + R_l_inv_42*x184 + R_l_inv_52*x186
        x208 = x207*x4
        x209 = R_l_inv_05*x194 + R_l_inv_15*x202 + R_l_inv_25*x197 + R_l_inv_35*x204 + R_l_inv_45*x184 + R_l_inv_55*x186
        x210 = x209*x34
        x211 = x209*x37
        x212 = x206 + x208 + x210 + x211
        x213 = R_l_inv_00*x194 + R_l_inv_10*x202 + R_l_inv_20*x197 + R_l_inv_30*x204 + R_l_inv_40*x184 + R_l_inv_50*x186
        x214 = a_2*x213
        x215 = d_3*x207
        x216 = a_3*x213
        x217 = x215 - x216
        x218 = x214 + x217
        x219 = R_l_inv_04*x194
        x220 = R_l_inv_14*x202
        x221 = R_l_inv_24*x197
        x222 = R_l_inv_34*x204
        x223 = R_l_inv_44*x184
        x224 = R_l_inv_54*x186
        x225 = x219 + x220 + x221 + x222 + x223 + x224
        x226 = x205*x36
        x227 = -x226
        x228 = x205*x52
        x229 = x207*x52
        x230 = x207*x36
        x231 = x213*x57
        x232 = -x230 - x231
        x233 = x229 + x232
        x234 = -x214
        x235 = -x215
        x236 = x216 + x234 + x235
        x237 = -x206
        x238 = x208 + x210 + x211 + x237
        x239 = 4*a_3
        x240 = a_2*x239
        x241 = R_l_inv_07*x194 + R_l_inv_17*x202 + R_l_inv_27*x197 + R_l_inv_37*x204 + R_l_inv_47*x184 + R_l_inv_57*x186
        x242 = -x241*x57
        x243 = -x240 + x242
        x244 = 2*x32
        x245 = 2*x30
        x246 = 2*x31
        x247 = 2*x194
        x248 = R_l_inv_03*x247
        x249 = 2*x202
        x250 = R_l_inv_13*x249
        x251 = 2*x197
        x252 = R_l_inv_23*x251
        x253 = 2*x204
        x254 = R_l_inv_33*x253
        x255 = 4*d_4
        x256 = x182*x255
        x257 = R_l_inv_43*x256
        x258 = x185*x255
        x259 = R_l_inv_53*x258
        x260 = -x244 + x245 + x246 + x248 + x250 + x252 + x254 + x257 + x259
        x261 = 8*d_3
        x262 = x240 + x242
        x263 = x244 - x245 - x246 - x248 - x250 - x252 - x254 - x257 - x259
        x264 = -x219 - x220 - x221 - x222 - x223 - x224
        x265 = -x228
        x266 = R_l_inv_06*x197 + R_l_inv_16*x204 - R_l_inv_26*x194 - R_l_inv_36*x202 + R_l_inv_46*x186 - R_l_inv_56*x184
        x267 = d_3*x266
        x268 = R_l_inv_02*x197 + R_l_inv_12*x204 - R_l_inv_22*x194 - R_l_inv_32*x202 + R_l_inv_42*x186 - R_l_inv_52*x184
        x269 = x268*x4
        x270 = R_l_inv_05*x197 + R_l_inv_15*x204 - R_l_inv_25*x194 - R_l_inv_35*x202 + R_l_inv_45*x186 - R_l_inv_55*x184
        x271 = x270*x34
        x272 = x270*x37
        x273 = x267 + x269 + x271 + x272
        x274 = R_l_inv_00*x197 + R_l_inv_10*x204 - R_l_inv_20*x194 - R_l_inv_30*x202 + R_l_inv_40*x186 - R_l_inv_50*x184
        x275 = a_2*x274
        x276 = x275 + x37
        x277 = d_3*x268
        x278 = a_3*x274
        x279 = x277 - x278
        x280 = R_l_inv_04*x197
        x281 = R_l_inv_14*x204
        x282 = R_l_inv_24*x194
        x283 = R_l_inv_34*x202
        x284 = R_l_inv_54*x184
        x285 = R_l_inv_44*x186
        x286 = x280 + x281 - x282 - x283 - x284 + x285 + x33
        x287 = x279 + x286
        x288 = x268*x52
        x289 = x266*x52
        x290 = x288 + x289
        x291 = x266*x36
        x292 = -x291
        x293 = x268*x36
        x294 = x274*x57
        x295 = -x293 - x294
        x296 = x292 + x295
        x297 = -x275
        x298 = -x277
        x299 = x278 + x297 + x298
        x300 = -x267
        x301 = x269 + x271
        x302 = x272 + x300 + x301
        x303 = R_l_inv_43*x258
        x304 = R_l_inv_53*x256
        x305 = R_l_inv_03*x251
        x306 = R_l_inv_23*x247
        x307 = R_l_inv_13*x253
        x308 = R_l_inv_33*x249
        x309 = R_l_inv_07*x197 + R_l_inv_17*x204 - R_l_inv_27*x194 - R_l_inv_37*x202 + R_l_inv_47*x186 - R_l_inv_57*x184
        x310 = -x309*x57
        x311 = x303 - x304 + x305 - x306 + x307 - x308 + x310
        x312 = -x303 + x304 - x305 + x306 - x307 + x308 + x310
        x313 = -x37
        x314 = x275 + x313
        x315 = -x280 - x281 + x282 + x283 + x284 - x285 + x30 + x31 + x32
        x316 = x279 + x315
        x317 = x291 + x295
        x318 = x288 - x289
        x319 = 2*x3
        x320 = 2*x193
        x321 = r_21*x320 + r_31**3*x11 + r_31*x10 + r_31*x13 + r_31*x15 - r_31*x19 - r_31*x20 + r_31*x21 + r_31*x24 + r_31*x27 - r_31*x8 - r_31*x9 + x189*x196 + x28*x78 + x28*x79 - x28*x81 - x28*x82 - x319*x78 - x319*x79 - x319*x80
        x322 = 2*r_12*x196 + r_12*x203 + r_22*x200 + r_22*x320 + r_32*x10 + r_32*x13 + r_32*x15 + 3*r_32*x17 - r_32*x19 - r_32*x20 + r_32*x21 + r_32*x24 + r_32*x27 - r_32*x8 - r_32*x9 + x28*x85 + x28*x86 - x28*x88 - x28*x89 - x319*x85 - x319*x86 - x319*x87
        x323 = R_l_inv_66*x321 + R_l_inv_76*x322
        x324 = d_3*x323
        x325 = R_l_inv_62*x321 + R_l_inv_72*x322
        x326 = x325*x4
        x327 = R_l_inv_65*x321 + R_l_inv_75*x322
        x328 = x327*x34
        x329 = -x28*x75
        x330 = -x28*x76
        x331 = -x28*x77
        x332 = x327*x37
        x333 = x319*x75
        x334 = x319*x76
        x335 = x319*x77
        x336 = x324 + x326 + x328 + x329 + x330 + x331 + x332 + x333 + x334 + x335
        x337 = d_3*x325
        x338 = R_l_inv_60*x321 + R_l_inv_70*x322
        x339 = a_3*x338
        x340 = x337 - x339
        x341 = a_2*x338
        x342 = R_l_inv_64*x321
        x343 = R_l_inv_74*x322
        x344 = x342 + x343
        x345 = x341 + x344
        x346 = x323*x36
        x347 = -x346
        x348 = x323*x52
        x349 = x325*x52
        x350 = x325*x36
        x351 = x338*x57
        x352 = -x350 - x351
        x353 = x349 + x352
        x354 = -x341
        x355 = -x337
        x356 = x339 + x354 + x355
        x357 = -x324
        x358 = x326 + x328 + x329 + x330 + x331 + x332 + x333 + x334 + x335 + x357
        x359 = R_l_inv_67*x321 + R_l_inv_77*x322
        x360 = -x359*x57
        x361 = 4*a_2
        x362 = d_3*x361
        x363 = x360 + x362
        x364 = d_3*x239
        x365 = x321*x64
        x366 = x322*x66
        x367 = -x364 + x365 + x366
        x368 = 4*x30
        x369 = -8*a_2*a_3
        x370 = 4*x32
        x371 = 4*x31
        x372 = -x370 + x371
        x373 = -x362
        x374 = x360 + x373
        x375 = x364 - x365 - x366
        x376 = -x342 - x343
        x377 = x341 + x376
        x378 = -x348
        x379 = x362*x7
        x380 = x379 + x74
        x381 = -x56 - x58
        x382 = 4*x1
        x383 = 4*x42 - 4*x43
        x384 = x379 + x51
        x385 = x361*x68
        x386 = 8*R_l_inv_63
        x387 = 8*R_l_inv_73
        x388 = x362*x97
        x389 = x132 + x388
        x390 = -x116 - x117
        x391 = 4*x93
        x392 = 4*x103 - 4*x104
        x393 = x112 + x388
        x394 = x123*x361
        x395 = -x142*x362
        x396 = x160 + x161
        x397 = x395 + x396
        x398 = 4*x154
        x399 = 4*x140 - 4*x145
        x400 = x162 + x395
        x401 = -x173*x361
        x402 = -x209*x362
        x403 = x226 + x402
        x404 = x230 + x231
        x405 = 4*x206
        x406 = 4*x215 - 4*x216
        x407 = x227 + x402
        x408 = a_3*x261
        x409 = -x241*x361
        x410 = 16*d_4
        x411 = x182*x410
        x412 = x185*x410
        x413 = 8*x194
        x414 = 8*x197
        x415 = 8*x202
        x416 = 8*x204
        x417 = -x270*x362
        x418 = x373 + x417
        x419 = x293 + x294
        x420 = x291 + x419
        x421 = 4*x267
        x422 = 4*x277 - 4*x278
        x423 = -x309*x361
        x424 = x362 + x417
        x425 = x292 + x419
        x426 = -x327*x362
        x427 = x346 + x426
        x428 = x350 + x351
        x429 = 4*x324
        x430 = 4*x337 - 4*x339
        x431 = x347 + x426
        x432 = -x359*x361
        x433 = x370 - x371
        x434 = -x368
        x435 = x35 + x38 + x6
        x436 = x1 + x435
        x437 = x43 + x61
        x438 = x381 + x55
        x439 = x2 + x435
        x440 = x44 + x45
        x441 = x75 + x76 + x77 + x96 + x98 + x99
        x442 = x441 + x93
        x443 = x104 + x120
        x444 = x115 + x390
        x445 = x441 + x94
        x446 = x105 + x106
        x447 = x124 - x52
        x448 = -x149
        x449 = x169 + x448
        x450 = x147 + x148
        x451 = a_2 + a_3
        x452 = x164 + x451
        x453 = x159 + x396
        x454 = x146 + x168 + x448
        x455 = -x211
        x456 = x208 + x210 + x225 + x455
        x457 = x214 + x216 + x235
        x458 = x229 + x404
        x459 = x217 + x234
        x460 = x208 + x210 + x264 + x455
        x461 = -x272 + x301
        x462 = x300 + x461
        x463 = x278 + x298
        x464 = x267 + x461
        x465 = x326 + x328 + x329 + x330 + x331 - x332 + x333 + x334 + x335
        x466 = x357 + x465
        x467 = x339 + x355
        x468 = x349 + x428
        x469 = x324 + x465
        x470 = x340 + x354
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = x40 + x44 + x50
        A[0, 1] = x51 + x54 + x60
        A[0, 2] = x49 + x62 + x63
        A[0, 3] = x70
        A[0, 4] = -4
        A[0, 5] = x71
        A[0, 6] = x44 + x63 + x73
        A[0, 7] = x53 + x60 + x74
        A[0, 8] = x40 + x62 + x72
        A[1, 0] = x101 + x105 + x111
        A[1, 1] = x112 + x114 + x119
        A[1, 2] = x110 + x121 + x122
        A[1, 3] = x125 + x128
        A[1, 5] = x125 + x129
        A[1, 6] = x105 + x122 + x131
        A[1, 7] = x113 + x119 + x132
        A[1, 8] = x101 + x121 + x130
        A[2, 0] = a_2 + x146 + x150 + x155
        A[2, 1] = x156 + x158 + x163
        A[2, 2] = x150 + x167 + x170
        A[2, 3] = x175
        A[2, 5] = x176
        A[2, 6] = x146 + x152 + x167 + x178
        A[2, 7] = x163 + x179 + x180
        A[2, 8] = a_2 + x170 + x178 + x181
        A[3, 0] = x212 + x218 + x225
        A[3, 1] = x227 + x228 + x233
        A[3, 2] = x225 + x236 + x238
        A[3, 3] = x243 + x260
        A[3, 4] = x166*x261
        A[3, 5] = x262 + x263
        A[3, 6] = x218 + x238 + x264
        A[3, 7] = x226 + x233 + x265
        A[3, 8] = x212 + x236 + x264
        A[4, 0] = x273 + x276 + x287
        A[4, 1] = x290 + x296
        A[4, 2] = x286 + x299 + x302 + x37
        A[4, 3] = x311
        A[4, 5] = x312
        A[4, 6] = x302 + x314 + x316
        A[4, 7] = x317 + x318
        A[4, 8] = x273 + x299 + x313 + x315
        A[5, 0] = x336 + x340 + x345
        A[5, 1] = x347 + x348 + x353
        A[5, 2] = x344 + x356 + x358
        A[5, 3] = x363 + x367
        A[5, 4] = x368 + x369 + x372
        A[5, 5] = x374 + x375
        A[5, 6] = x340 + x358 + x377
        A[5, 7] = x346 + x353 + x378
        A[5, 8] = x336 + x356 + x376
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x380 + x381
        B[0, 1] = -x382 + x383
        B[0, 2] = x384 + x59
        B[0, 3] = x385 + 4
        B[0, 4] = -r_31*x386 - r_32*x387
        B[0, 5] = x385 - 4
        B[0, 6] = x381 + x384
        B[0, 7] = x382 + x383
        B[0, 8] = x380 + x59
        B[1, 0] = x389 + x390
        B[1, 1] = -x391 + x392
        B[1, 2] = x118 + x393
        B[1, 3] = x394
        B[1, 4] = -x386*x84 - x387*x91
        B[1, 5] = x394
        B[1, 6] = x390 + x393
        B[1, 7] = x391 + x392
        B[1, 8] = x118 + x389
        B[2, 0] = x180 + x397
        B[2, 1] = -x239 + x398 + x399
        B[2, 2] = x158 + x400
        B[2, 3] = x401
        B[2, 4] = x135*x386 + x138*x387
        B[2, 5] = x401
        B[2, 6] = x158 + x397
        B[2, 7] = x239 - x398 + x399
        B[2, 8] = x180 + x400
        B[3, 0] = x403 + x404
        B[3, 1] = x405 + x406
        B[3, 2] = x232 + x407
        B[3, 3] = -x408 + x409
        B[3, 4] = R_l_inv_03*x413 + R_l_inv_13*x415 + R_l_inv_23*x414 + R_l_inv_33*x416 + R_l_inv_43*x411 + R_l_inv_53*x412 - 8*x30 + 8*x31 - 8*x32
        B[3, 5] = x408 + x409
        B[3, 6] = x404 + x407
        B[3, 7] = -x405 + x406
        B[3, 8] = x232 + x403
        B[4, 0] = x418 + x420
        B[4, 1] = x421 + x422
        B[4, 2] = x296 + x418
        B[4, 3] = x423
        B[4, 4] = R_l_inv_03*x414 + R_l_inv_13*x416 - R_l_inv_23*x413 - R_l_inv_33*x415 + R_l_inv_43*x412 - R_l_inv_53*x411
        B[4, 5] = x423
        B[4, 6] = x424 + x425
        B[4, 7] = -x421 + x422
        B[4, 8] = x317 + x424
        B[5, 0] = x427 + x428
        B[5, 1] = x429 + x430
        B[5, 2] = x352 + x431
        B[5, 3] = x368 + x432 + x433
        B[5, 4] = -16*a_3*d_3 + x321*x386 + x322*x387
        B[5, 5] = x372 + x432 + x434
        B[5, 6] = x428 + x431
        B[5, 7] = -x429 + x430
        B[5, 8] = x352 + x427
        C = np.zeros(shape=(6, 9))
        C[0, 0] = x436 + x437 + x50
        C[0, 1] = x438 + x54 + x74
        C[0, 2] = x439 + x440 + x49
        C[0, 3] = x71
        C[0, 4] = 4
        C[0, 5] = x70
        C[0, 6] = x437 + x439 + x73
        C[0, 7] = x438 + x51 + x53
        C[0, 8] = x436 + x440 + x72
        C[1, 0] = x111 + x442 + x443
        C[1, 1] = x114 + x132 + x444
        C[1, 2] = x110 + x445 + x446
        C[1, 3] = x129 + x447
        C[1, 5] = x128 + x447
        C[1, 6] = x131 + x443 + x445
        C[1, 7] = x112 + x113 + x444
        C[1, 8] = x130 + x442 + x446
        C[2, 0] = x152 + x449 + x450 + x452
        C[2, 1] = x156 + x180 + x453
        C[2, 2] = x165 + x181 + x450 + x454
        C[2, 3] = x176
        C[2, 5] = x175
        C[2, 6] = x155 + x165 + x177 + x449
        C[2, 7] = x158 + x179 + x453
        C[2, 8] = x177 + x452 + x454
        C[3, 0] = x237 + x456 + x457
        C[3, 1] = x226 + x228 + x458
        C[3, 2] = x206 + x456 + x459
        C[3, 3] = x243 + x263
        C[3, 4] = -x261*x451
        C[3, 5] = x260 + x262
        C[3, 6] = x206 + x457 + x460
        C[3, 7] = x227 + x265 + x458
        C[3, 8] = x237 + x459 + x460
        C[4, 0] = x286 + x314 + x462 + x463
        C[4, 1] = x290 + x420
        C[4, 2] = x287 + x297 + x313 + x464
        C[4, 3] = x312
        C[4, 5] = x311
        C[4, 6] = x276 + x315 + x463 + x464
        C[4, 7] = x318 + x425
        C[4, 8] = x297 + x316 + x37 + x462
        C[5, 0] = x345 + x466 + x467
        C[5, 1] = x346 + x348 + x468
        C[5, 2] = x344 + x469 + x470
        C[5, 3] = x363 + x375
        C[5, 4] = x369 + x433 + x434
        C[5, 5] = x367 + x374
        C[5, 6] = x377 + x467 + x469
        C[5, 7] = x347 + x378 + x468
        C[5, 8] = x376 + x466 + x470
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
            degenerate_valid_0 = (abs(th_2 - 1.69515128643052) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
            
            th_2 = this_solution[2]
            degenerate_valid_1 = (abs(th_2 + 1.69515128643052) <= 1.0e-6)
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
            condition_0: bool = (abs((a_2**2 + 2*a_2*a_3*math.cos(th_2) + 2*a_2*d_3*math.sin(th_2) + a_3**2 + d_3**2 + d_4**2 - inv_Px**2 - inv_Py**2 - (d_5 + inv_Pz)**2)/(2*a_2*d_4*math.cos(th_2) + 2*a_3*d_4)) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*d_4
                x1 = a_2*math.cos(th_2)
                x2 = safe_asin((-a_2**2 - 2*a_2*d_3*math.sin(th_2) - a_3**2 - 2*a_3*x1 - d_3**2 - d_4**2 + inv_Px**2 + inv_Py**2 + (d_5 + inv_Pz)**2)/(-a_3*x0 - x0*x1))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs((a_2**2 + 2*a_2*a_3*math.cos(th_2) + 2*a_2*d_3*math.sin(th_2) + a_3**2 + d_3**2 + d_4**2 - inv_Px**2 - inv_Py**2 - (d_5 + inv_Pz)**2)/(2*a_2*d_4*math.cos(th_2) + 2*a_3*d_4)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*d_4
                x1 = a_2*math.cos(th_2)
                x2 = safe_asin((-a_2**2 - 2*a_2*d_3*math.sin(th_2) - a_3**2 - 2*a_3*x1 - d_3**2 - d_4**2 + inv_Px**2 + inv_Py**2 + (d_5 + inv_Pz)**2)/(-a_3*x0 - x0*x1))
                # End of temp variables
                this_solution[3] = math.pi - x2
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
            checked_result: bool = (abs(d_4*math.cos(th_3)) <= 1.0e-6) and (abs(Px - d_5*r_13) <= 1.0e-6) and (abs(Py - d_5*r_23) <= 1.0e-6)
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
            condition_0: bool = (abs(d_4*math.cos(th_3)) >= zero_tolerance) or (abs(Px - d_5*r_13) >= zero_tolerance) or (abs(Py - d_5*r_23) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = Px - d_5*r_13
                x1 = -Py + d_5*r_23
                x2 = math.atan2(x0, x1)
                x3 = math.cos(th_3)
                x4 = safe_sqrt(-d_4**2*x3**2 + x0**2 + x1**2)
                x5 = d_4*x3
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[0] = x2 + math.atan2(x4, x5)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                
            condition_1: bool = (abs(d_4*math.cos(th_3)) >= zero_tolerance) or (abs(Px - d_5*r_13) >= zero_tolerance) or (abs(Py - d_5*r_23) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = Px - d_5*r_13
                x1 = -Py + d_5*r_23
                x2 = math.atan2(x0, x1)
                x3 = math.cos(th_3)
                x4 = safe_sqrt(-d_4**2*x3**2 + x0**2 + x1**2)
                x5 = d_4*x3
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
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            checked_result: bool = (abs(-a_3*math.sin(th_2) + d_3*math.cos(th_2) + d_4*math.sin(th_2)*math.sin(th_3)) <= 1.0e-6) and (abs(a_2 + a_3*math.cos(th_2) + d_3*math.sin(th_2) - d_4*math.sin(th_3)*math.cos(th_2)) <= 1.0e-6)
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
            condition_0: bool = (abs(-a_3*math.sin(th_2) + d_3*math.cos(th_2) + d_4*math.sin(th_2)*math.sin(th_3)) >= 1.0e-6) or (abs(a_2 + a_3*math.cos(th_2) + d_3*math.sin(th_2) - d_4*math.sin(th_3)*math.cos(th_2)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = d_5*r_33
                x1 = math.cos(th_2)
                x2 = math.sin(th_2)
                x3 = d_4*math.sin(th_3)
                x4 = a_2 + a_3*x1 + d_3*x2 - x1*x3
                x5 = a_3*x2 - d_3*x1 - x2*x3
                x6 = math.cos(th_0)
                x7 = math.sin(th_0)
                x8 = Px*x6 + Py*x7 - d_5*r_13*x6 - d_5*r_23*x7
                # End of temp variables
                this_solution[1] = math.atan2(x4*(-Pz + x0) + x5*x8, x4*x8 + x5*(Pz - x0))
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
