import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 7
n_tree_nodes: int = 24
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
d_2: float = 0.26
d_4: float = 0.264
post_transform_s4: float = -0.066
pre_transform_s0: float = 0.1025

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = 3.141592653589793
th_2_offset_original2raw: float = 1.5707963267948966
th_3_offset_original2raw: float = 3.141592653589793
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 1.5707963267948966
th_6_offset_original2raw: float = 0.0


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def w1_left_arm_anthropomorphic_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = -r_13
    ee_transformed[0, 1] = r_12
    ee_transformed[0, 2] = r_11
    ee_transformed[0, 3] = Px + post_transform_s4*r_13
    ee_transformed[1, 0] = -r_23
    ee_transformed[1, 1] = r_22
    ee_transformed[1, 2] = r_21
    ee_transformed[1, 3] = Py + post_transform_s4*r_23
    ee_transformed[2, 0] = -r_33
    ee_transformed[2, 1] = r_32
    ee_transformed[2, 2] = r_31
    ee_transformed[2, 3] = Pz + post_transform_s4*r_33 - pre_transform_s0
    return ee_transformed


def w1_left_arm_anthropomorphic_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = r_13
    ee_transformed[0, 1] = r_12
    ee_transformed[0, 2] = -r_11
    ee_transformed[0, 3] = Px + post_transform_s4*r_11
    ee_transformed[1, 0] = r_23
    ee_transformed[1, 1] = r_22
    ee_transformed[1, 2] = -r_21
    ee_transformed[1, 3] = Py + post_transform_s4*r_21
    ee_transformed[2, 0] = r_33
    ee_transformed[2, 1] = r_32
    ee_transformed[2, 2] = -r_31
    ee_transformed[2, 3] = Pz + post_transform_s4*r_31 + pre_transform_s0
    return ee_transformed


def w1_left_arm_anthropomorphic_fk(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.cos(th_5)
    x1 = math.cos(th_3)
    x2 = math.cos(th_0)
    x3 = math.sin(th_1)
    x4 = x2*x3
    x5 = math.sin(th_3)
    x6 = math.sin(th_0)
    x7 = math.sin(th_2)
    x8 = x6*x7
    x9 = math.cos(th_1)
    x10 = math.cos(th_2)
    x11 = x10*x2
    x12 = x11*x9 + x8
    x13 = x1*x4 - x12*x5
    x14 = math.sin(th_5)
    x15 = math.sin(th_4)
    x16 = x10*x6
    x17 = x2*x7
    x18 = x16 - x17*x9
    x19 = math.cos(th_4)
    x20 = x1*x12 + x4*x5
    x21 = -x15*x18 + x19*x20
    x22 = math.cos(th_6)
    x23 = -x15*x20 - x18*x19
    x24 = math.sin(th_6)
    x25 = x0*x21 - x13*x14
    x26 = x23*x24
    x27 = x22*x25
    x28 = x3*x6
    x29 = x16*x9 - x17
    x30 = x1*x28 - x29*x5
    x31 = -x11 - x8*x9
    x32 = x1*x29 + x28*x5
    x33 = -x15*x31 + x19*x32
    x34 = -x15*x32 - x19*x31
    x35 = x0*x33 - x14*x30
    x36 = x24*x34
    x37 = x22*x35
    x38 = x10*x3
    x39 = x1*x9 + x38*x5
    x40 = x3*x7
    x41 = -x1*x38 + x5*x9
    x42 = -x15*x40 + x19*x41
    x43 = -x15*x41 - x19*x40
    x44 = x0*x42 - x14*x39
    x45 = x24*x43
    x46 = x22*x44
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -x0*x13 - x14*x21
    ee_pose[0, 1] = -x22*x23 - x24*x25
    ee_pose[0, 2] = x26 - x27
    ee_pose[0, 3] = -d_2*x4 + d_4*x13 + post_transform_s4*(-x26 + x27)
    ee_pose[1, 0] = -x0*x30 - x14*x33
    ee_pose[1, 1] = -x22*x34 - x24*x35
    ee_pose[1, 2] = x36 - x37
    ee_pose[1, 3] = -d_2*x28 + d_4*x30 + post_transform_s4*(-x36 + x37)
    ee_pose[2, 0] = -x0*x39 - x14*x42
    ee_pose[2, 1] = -x22*x43 - x24*x44
    ee_pose[2, 2] = x45 - x46
    ee_pose[2, 3] = -d_2*x9 + d_4*x39 + post_transform_s4*(-x45 + x46) + pre_transform_s0
    return ee_pose


def w1_left_arm_anthropomorphic_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.sin(th_1)
    x2 = math.cos(th_0)
    x3 = x1*x2
    x4 = math.cos(th_2)
    x5 = x0*x4
    x6 = math.cos(th_1)
    x7 = math.sin(th_2)
    x8 = x2*x7
    x9 = x5 - x6*x8
    x10 = math.cos(th_3)
    x11 = math.sin(th_3)
    x12 = x0*x7
    x13 = x2*x4
    x14 = x12 + x13*x6
    x15 = x10*x3 - x11*x14
    x16 = math.cos(th_4)
    x17 = math.sin(th_4)
    x18 = x10*x14 + x11*x3
    x19 = -x16*x9 - x17*x18
    x20 = math.cos(th_5)
    x21 = math.sin(th_5)
    x22 = -x15*x20 - x21*(x16*x18 - x17*x9)
    x23 = x0*x1
    x24 = -x12*x6 - x13
    x25 = x5*x6 - x8
    x26 = x10*x23 - x11*x25
    x27 = x10*x25 + x11*x23
    x28 = -x16*x24 - x17*x27
    x29 = -x20*x26 - x21*(x16*x27 - x17*x24)
    x30 = x1*x7
    x31 = x1*x4
    x32 = x10*x6 + x11*x31
    x33 = -x10*x31 + x11*x6
    x34 = -x16*x30 - x17*x33
    x35 = -x20*x32 - x21*(x16*x33 - x17*x30)
    x36 = d_2*x6
    x37 = pre_transform_s0 - x36
    x38 = d_2*x1**2
    x39 = d_4*x32 + x37
    x40 = d_2*x23
    x41 = d_4*x26 - x40
    x42 = d_2*x3
    x43 = d_4*x15 - x42
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 7))
    jacobian_output[0, 1] = -x0
    jacobian_output[0, 2] = -x3
    jacobian_output[0, 3] = x9
    jacobian_output[0, 4] = x15
    jacobian_output[0, 5] = x19
    jacobian_output[0, 6] = x22
    jacobian_output[1, 1] = x2
    jacobian_output[1, 2] = -x23
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
    jacobian_output[3, 1] = -pre_transform_s0*x2
    jacobian_output[3, 2] = x23*x36 + x23*x37
    jacobian_output[3, 3] = -x12*x38 - x24*x37
    jacobian_output[3, 4] = -x26*x39 + x32*x41
    jacobian_output[3, 5] = -x28*x39 + x34*x41
    jacobian_output[3, 6] = -x29*x39 + x35*x41
    jacobian_output[4, 1] = -pre_transform_s0*x0
    jacobian_output[4, 2] = -x3*x36 - x3*x37
    jacobian_output[4, 3] = x37*x9 + x38*x8
    jacobian_output[4, 4] = x15*x39 - x32*x43
    jacobian_output[4, 5] = x19*x39 - x34*x43
    jacobian_output[4, 6] = x22*x39 - x35*x43
    jacobian_output[5, 3] = -x24*x42 + x40*x9
    jacobian_output[5, 4] = -x15*x41 + x26*x43
    jacobian_output[5, 5] = -x19*x41 + x28*x43
    jacobian_output[5, 6] = -x22*x41 + x29*x43
    return jacobian_output


def w1_left_arm_anthropomorphic_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.sin(th_1)
    x2 = math.cos(th_0)
    x3 = x1*x2
    x4 = math.cos(th_2)
    x5 = x0*x4
    x6 = math.cos(th_1)
    x7 = math.sin(th_2)
    x8 = x2*x7
    x9 = x5 - x6*x8
    x10 = math.cos(th_3)
    x11 = math.sin(th_3)
    x12 = x0*x7
    x13 = x2*x4
    x14 = x12 + x13*x6
    x15 = x10*x3 - x11*x14
    x16 = math.cos(th_4)
    x17 = math.sin(th_4)
    x18 = x10*x14 + x11*x3
    x19 = math.cos(th_5)
    x20 = math.sin(th_5)
    x21 = x0*x1
    x22 = -x12*x6 - x13
    x23 = x5*x6 - x8
    x24 = x10*x21 - x11*x23
    x25 = x10*x23 + x11*x21
    x26 = x1*x7
    x27 = x1*x4
    x28 = x10*x6 + x11*x27
    x29 = -x10*x27 + x11*x6
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 1] = -x0
    jacobian_output[0, 2] = -x3
    jacobian_output[0, 3] = x9
    jacobian_output[0, 4] = x15
    jacobian_output[0, 5] = -x16*x9 - x17*x18
    jacobian_output[0, 6] = -x15*x19 - x20*(x16*x18 - x17*x9)
    jacobian_output[1, 1] = x2
    jacobian_output[1, 2] = -x21
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


def w1_left_arm_anthropomorphic_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x0 = math.cos(th_0)
    x1 = p_on_ee_z*x0
    x2 = math.cos(th_1)
    x3 = math.sin(th_1)
    x4 = math.sin(th_0)
    x5 = p_on_ee_z*x4
    x6 = d_2*x2
    x7 = x3*x4
    x8 = pre_transform_s0 - x6
    x9 = math.sin(th_2)
    x10 = x3*x9
    x11 = x4*x9
    x12 = d_2*x3**2
    x13 = math.cos(th_2)
    x14 = x0*x13
    x15 = -x11*x2 - x14
    x16 = math.cos(th_3)
    x17 = math.sin(th_3)
    x18 = x13*x3
    x19 = x16*x2 + x17*x18
    x20 = x0*x9
    x21 = x13*x4
    x22 = x2*x21 - x20
    x23 = x16*x7 - x17*x22
    x24 = d_4*x19 + x8
    x25 = d_2*x7
    x26 = d_4*x23 - x25
    x27 = math.cos(th_4)
    x28 = math.sin(th_4)
    x29 = -x16*x18 + x17*x2
    x30 = -x10*x27 - x28*x29
    x31 = x16*x22 + x17*x7
    x32 = -x15*x27 - x28*x31
    x33 = math.cos(th_5)
    x34 = math.sin(th_5)
    x35 = -x19*x33 - x34*(-x10*x28 + x27*x29)
    x36 = -x23*x33 - x34*(-x15*x28 + x27*x31)
    x37 = x0*x3
    x38 = -x2*x20 + x21
    x39 = x11 + x14*x2
    x40 = x16*x37 - x17*x39
    x41 = d_2*x37
    x42 = d_4*x40 - x41
    x43 = x16*x39 + x17*x37
    x44 = -x27*x38 - x28*x43
    x45 = -x33*x40 - x34*(x27*x43 - x28*x38)
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 0] = -p_on_ee_y
    jacobian_output[0, 1] = -pre_transform_s0*x0 + x1
    jacobian_output[0, 2] = p_on_ee_y*x2 - x3*x5 + x6*x7 + x7*x8
    jacobian_output[0, 3] = -p_on_ee_y*x10 + p_on_ee_z*x15 - x11*x12 - x15*x8
    jacobian_output[0, 4] = -p_on_ee_y*x19 + p_on_ee_z*x23 + x19*x26 - x23*x24
    jacobian_output[0, 5] = -p_on_ee_y*x30 + p_on_ee_z*x32 - x24*x32 + x26*x30
    jacobian_output[0, 6] = -p_on_ee_y*x35 + p_on_ee_z*x36 - x24*x36 + x26*x35
    jacobian_output[1, 0] = p_on_ee_x
    jacobian_output[1, 1] = -pre_transform_s0*x4 + x5
    jacobian_output[1, 2] = -p_on_ee_x*x2 + x1*x3 - x37*x6 - x37*x8
    jacobian_output[1, 3] = p_on_ee_x*x10 - p_on_ee_z*x38 + x12*x20 + x38*x8
    jacobian_output[1, 4] = p_on_ee_x*x19 - p_on_ee_z*x40 - x19*x42 + x24*x40
    jacobian_output[1, 5] = p_on_ee_x*x30 - p_on_ee_z*x44 + x24*x44 - x30*x42
    jacobian_output[1, 6] = p_on_ee_x*x35 - p_on_ee_z*x45 + x24*x45 - x35*x42
    jacobian_output[2, 1] = -p_on_ee_x*x0 - p_on_ee_y*x4
    jacobian_output[2, 2] = p_on_ee_x*x7 - p_on_ee_y*x37
    jacobian_output[2, 3] = -p_on_ee_x*x15 + p_on_ee_y*x38 - x15*x41 + x25*x38
    jacobian_output[2, 4] = -p_on_ee_x*x23 + p_on_ee_y*x40 + x23*x42 - x26*x40
    jacobian_output[2, 5] = -p_on_ee_x*x32 + p_on_ee_y*x44 - x26*x44 + x32*x42
    jacobian_output[2, 6] = -p_on_ee_x*x36 + p_on_ee_y*x45 - x26*x45 + x36*x42
    return jacobian_output


def w1_left_arm_anthropomorphic_ik_solve_raw(T_ee: np.ndarray, th_4):
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
    for i in range(24):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True
    
    # Code for non-branch dispatcher node 0
    # Actually, there is no code
    
    # Code for explicit solution node 1, solved variable is th_3
    def ExplicitSolutionNode_node_1_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return
        
        # The explicit solution of root node
        condition_0: bool = ((1/2)*abs((-d_2**2 - d_4**2 + inv_Px**2 + inv_Py**2 + inv_Pz**2)/(d_2*d_4)) <= 1)
        if condition_0:
            # Temp variable for efficiency
            x0 = safe_acos((1/2)*(d_2**2 + d_4**2 - inv_Px**2 - inv_Py**2 - inv_Pz**2)/(d_2*d_4))
            # End of temp variables
            solution_0: IkSolution = make_ik_solution()
            solution_0[5] = x0
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)
            
        condition_1: bool = ((1/2)*abs((-d_2**2 - d_4**2 + inv_Px**2 + inv_Py**2 + inv_Pz**2)/(d_2*d_4)) <= 1)
        if condition_1:
            # Temp variable for efficiency
            x0 = safe_acos((1/2)*(d_2**2 + d_4**2 - inv_Px**2 - inv_Py**2 - inv_Pz**2)/(d_2*d_4))
            # End of temp variables
            solution_1: IkSolution = make_ik_solution()
            solution_1[5] = -x0
            appended_idx = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx)
            
    # Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_3_processor()
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
            th_3 = this_solution[5]
            checked_result: bool = (abs(inv_Pz) <= 1.0e-6) and (abs(d_2*math.sin(th_3)*math.cos(th_4)) <= 1.0e-6) and (abs(d_2*math.cos(th_3) - d_4) <= 1.0e-6)
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
            th_3 = this_solution[5]
            condition_0: bool = (abs(inv_Pz) >= zero_tolerance) or (abs(d_2*math.sin(th_3)*math.cos(th_4)) >= zero_tolerance) or (abs(d_2*math.cos(th_3) - d_4) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_3)
                x1 = math.cos(th_4)
                x2 = -d_2*math.cos(th_3) + d_4
                x3 = math.atan2(-d_2*x0*x1, x2)
                x4 = safe_sqrt(d_2**2*x0**2*x1**2 - inv_Pz**2 + x2**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[6] = x3 + math.atan2(x4, inv_Pz)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs(inv_Pz) >= zero_tolerance) or (abs(d_2*math.sin(th_3)*math.cos(th_4)) >= zero_tolerance) or (abs(d_2*math.cos(th_3) - d_4) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.sin(th_3)
                x1 = math.cos(th_4)
                x2 = -d_2*math.cos(th_3) + d_4
                x3 = math.atan2(-d_2*x0*x1, x2)
                x4 = safe_sqrt(d_2**2*x0**2*x1**2 - inv_Pz**2 + x2**2)
                # End of temp variables
                this_solution[6] = x3 + math.atan2(-x4, inv_Pz)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_5_processor()
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
            th_3 = this_solution[5]
            th_5 = this_solution[6]
            checked_result: bool = (abs(d_2*math.sin(th_3)*math.sin(th_4)) <= 1.0e-6) and (abs(d_2*math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - d_2*math.sin(th_5)*math.cos(th_3) + d_4*math.sin(th_5)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor()
    # Finish code for equation all-zero dispatcher node 4
    
    # Code for explicit solution node 5, solved variable is th_6
    def ExplicitSolutionNode_node_5_solve_th_6_processor():
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
            th_3 = this_solution[5]
            th_5 = this_solution[6]
            condition_0: bool = (abs(d_2*math.sin(th_3)*math.sin(th_4)) >= 1.0e-6) or (abs(d_2*math.sin(th_3)*math.cos(th_4)*math.cos(th_5) - d_2*math.sin(th_5)*math.cos(th_3) + d_4*math.sin(th_5)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = d_2*math.sin(th_3)
                x1 = x0*math.sin(th_4)
                x2 = math.sin(th_5)
                x3 = -d_2*x2*math.cos(th_3) + d_4*x2 + x0*math.cos(th_4)*math.cos(th_5)
                # End of temp variables
                this_solution[7] = math.atan2(inv_Px*x1 - inv_Py*x3, inv_Px*x3 + inv_Py*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_6_processor()
    # Finish code for explicit solution node 5
    
    # Code for non-branch dispatcher node 6
    # Actually, there is no code
    
    # Code for explicit solution node 7, solved variable is th_1
    def ExplicitSolutionNode_node_7_solve_th_1_processor():
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
            th_5 = this_solution[6]
            th_6 = this_solution[7]
            condition_0: bool = (abs((Pz + d_4*(r_33*math.cos(th_5) + (r_31*math.cos(th_6) - r_32*math.sin(th_6))*math.sin(th_5)))/d_2) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = safe_acos((-Pz - d_4*(r_33*math.cos(th_5) + (r_31*math.cos(th_6) - r_32*math.sin(th_6))*math.sin(th_5)))/d_2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
            condition_1: bool = (abs((Pz + d_4*(r_33*math.cos(th_5) + (r_31*math.cos(th_6) - r_32*math.sin(th_6))*math.sin(th_5)))/d_2) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = safe_acos((-Pz - d_4*(r_33*math.cos(th_5) + (r_31*math.cos(th_6) - r_32*math.sin(th_6))*math.sin(th_5)))/d_2)
                # End of temp variables
                this_solution[3] = -x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_1_processor()
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
            th_1 = this_solution[3]
            degenerate_valid_0 = (abs(th_1) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(12, node_input_i_idx_in_queue)
            
            th_1 = this_solution[3]
            degenerate_valid_1 = (abs(th_1 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(17, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(9, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor()
    # Finish code for solved_variable dispatcher node 8
    
    # Code for explicit solution node 17, solved variable is th_0th_2_soa
    def ExplicitSolutionNode_node_17_solve_th_0th_2_soa_processor():
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
            th_3 = this_solution[5]
            condition_0: bool = (abs(Px) >= zero_tolerance) or (abs(Py) >= zero_tolerance) or (abs(d_4*math.sin(th_3)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 1/(d_4*math.sin(th_3))
                # End of temp variables
                this_solution[2] = math.atan2(Py*x0, Px*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(18, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_0th_2_soa_processor()
    # Finish code for explicit solution node 17
    
    # Code for non-branch dispatcher node 18
    # Actually, there is no code
    
    # Code for explicit solution node 19, solved variable is th_0
    def ExplicitSolutionNode_node_19_solve_th_0_processor():
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
                this_solution[1] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(20, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_0_processor()
    # Finish code for explicit solution node 18
    
    # Code for non-branch dispatcher node 20
    # Actually, there is no code
    
    # Code for explicit solution node 21, solved variable is th_2
    def ExplicitSolutionNode_node_21_solve_th_2_processor():
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
            th_0 = this_solution[1]
            th_0th_2_soa = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[4] = -th_0 + th_0th_2_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_2_processor()
    # Finish code for explicit solution node 20
    
    # Code for explicit solution node 12, solved variable is negative_th_2_positive_th_0__soa
    def ExplicitSolutionNode_node_12_solve_negative_th_2_positive_th_0__soa_processor():
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
            th_3 = this_solution[5]
            condition_0: bool = (abs(Px) >= zero_tolerance) or (abs(Py) >= zero_tolerance) or (abs(d_4*math.sin(th_3)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 1/(d_4*math.sin(th_3))
                # End of temp variables
                this_solution[0] = math.atan2(-Py*x0, -Px*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(13, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_12_solve_negative_th_2_positive_th_0__soa_processor()
    # Finish code for explicit solution node 12
    
    # Code for non-branch dispatcher node 13
    # Actually, there is no code
    
    # Code for explicit solution node 14, solved variable is th_0
    def ExplicitSolutionNode_node_14_solve_th_0_processor():
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
                this_solution[1] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_0_processor()
    # Finish code for explicit solution node 13
    
    # Code for non-branch dispatcher node 15
    # Actually, there is no code
    
    # Code for explicit solution node 16, solved variable is th_2
    def ExplicitSolutionNode_node_16_solve_th_2_processor():
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
            negative_th_2_positive_th_0__soa = this_solution[0]
            th_0 = this_solution[1]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[4] = -negative_th_2_positive_th_0__soa + th_0
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_2_processor()
    # Finish code for explicit solution node 15
    
    # Code for explicit solution node 9, solved variable is th_0
    def ExplicitSolutionNode_node_9_solve_th_0_processor():
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
            th_1 = this_solution[3]
            th_5 = this_solution[6]
            th_6 = this_solution[7]
            condition_0: bool = (abs(d_2*math.sin(th_1)) >= zero_tolerance) or (abs(-Px + d_4*(-r_13*math.cos(th_5) - (r_11*math.cos(th_6) - r_12*math.sin(th_6))*math.sin(th_5))) >= zero_tolerance) or (abs(-Py + d_4*(-r_23*math.cos(th_5) - (r_21*math.cos(th_6) - r_22*math.sin(th_6))*math.sin(th_5))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_5)
                x1 = math.sin(th_5)
                x2 = math.cos(th_6)
                x3 = math.sin(th_6)
                x4 = 1/(d_2*math.sin(th_1))
                # End of temp variables
                this_solution[1] = math.atan2(x4*(-Py + d_4*(-r_23*x0 - x1*(r_21*x2 - r_22*x3))), x4*(-Px + d_4*(-r_13*x0 - x1*(r_11*x2 - r_12*x3))))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_0_processor()
    # Finish code for explicit solution node 9
    
    # Code for solved_variable dispatcher node 10
    def SolvedVariableDispatcherNode_node_10_processor():
        this_node_input_index: List[int] = node_input_index[10]
        this_input_valid: bool = node_input_validity[10]
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
                add_input_index_to(22, node_input_i_idx_in_queue)
            
            th_3 = this_solution[5]
            degenerate_valid_1 = (abs(th_3 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(23, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(11, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_10_processor()
    # Finish code for solved_variable dispatcher node 10
    
    # Code for explicit solution node 23, solved variable is th_2
    def ExplicitSolutionNode_node_23_solve_th_2_processor():
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
            th_0 = this_solution[1]
            th_1 = this_solution[3]
            th_5 = this_solution[6]
            condition_0: bool = (abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) >= 1.0e-6) or (abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_4)
                x1 = math.sin(th_0)
                x2 = math.cos(th_0)
                x3 = math.sin(th_5)
                x4 = x3*(r_13*x1 - r_23*x2)
                x5 = math.sin(th_4)
                x6 = math.cos(th_1)
                x7 = x3*(r_13*x2*x6 + r_23*x1*x6 - r_33*math.sin(th_1))
                # End of temp variables
                this_solution[4] = math.atan2(x0*x4 - x5*x7, x0*x7 + x4*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_2_processor()
    # Finish code for explicit solution node 23
    
    # Code for explicit solution node 22, solved variable is th_2
    def ExplicitSolutionNode_node_22_solve_th_2_processor():
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
            th_0 = this_solution[1]
            th_1 = this_solution[3]
            th_5 = this_solution[6]
            condition_0: bool = (abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) >= 1.0e-6) or (abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = r_23*x0
                x2 = math.sin(th_0)
                x3 = r_13*x2
                x4 = math.sin(th_5)
                x5 = x4*math.cos(th_4)
                x6 = math.cos(th_1)
                x7 = -r_13*x0*x6 - r_23*x2*x6 + r_33*math.sin(th_1)
                x8 = x4*math.sin(th_4)
                # End of temp variables
                this_solution[4] = math.atan2(x5*(x1 - x3) + x7*x8, x5*x7 + x8*(-x1 + x3))
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_22_solve_th_2_processor()
    # Finish code for explicit solution node 22
    
    # Code for explicit solution node 11, solved variable is th_2
    def ExplicitSolutionNode_node_11_solve_th_2_processor():
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
            th_0 = this_solution[1]
            th_1 = this_solution[3]
            th_3 = this_solution[5]
            condition_0: bool = (abs(d_4*math.sin(th_3)) >= zero_tolerance) or (abs(Px*math.sin(th_0) - Py*math.cos(th_0)) >= zero_tolerance) or (abs(Px*math.cos(th_0)*math.cos(th_1) + Py*math.sin(th_0)*math.cos(th_1) - Pz*math.sin(th_1)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = 1/(d_4*math.sin(th_3))
                x3 = math.cos(th_1)
                # End of temp variables
                this_solution[4] = math.atan2(x2*(-Px*x1 + Py*x0), x2*(-Px*x0*x3 - Py*x1*x3 + Pz*math.sin(th_1)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_2_processor()
    # Finish code for explicit solution node 11
    
    # Collect the output
    ik_out: List[np.ndarray] = list()
    for i in range(len(solution_queue)):
        if not queue_element_validity[i]:
            continue
        ik_out_i = solution_queue[i]
        new_ik_i = np.zeros((robot_nq, 1))
        value_at_0 = ik_out_i[1]  # th_0
        new_ik_i[0] = value_at_0
        value_at_1 = ik_out_i[3]  # th_1
        new_ik_i[1] = value_at_1
        value_at_2 = ik_out_i[4]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[5]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = th_4  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[6]  # th_5
        new_ik_i[5] = value_at_5
        value_at_6 = ik_out_i[7]  # th_6
        new_ik_i[6] = value_at_6
        ik_out.append(new_ik_i)
    return ik_out


def w1_left_arm_anthropomorphic_ik_solve(T_ee: np.ndarray, th_4):
    T_ee_raw_in = w1_left_arm_anthropomorphic_ik_target_original_to_raw(T_ee)
    ik_output_raw = w1_left_arm_anthropomorphic_ik_solve_raw(T_ee_raw_in, th_4 + th_4_offset_original2raw)
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
        ee_pose_i = w1_left_arm_anthropomorphic_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_w1_left_arm_anthropomorphic():
    theta_in = np.random.random(size=(7, ))
    ee_pose = w1_left_arm_anthropomorphic_fk(theta_in)
    ik_output = w1_left_arm_anthropomorphic_ik_solve(ee_pose, th_4=theta_in[4])
    for i in range(len(ik_output)):
        ee_pose_i = w1_left_arm_anthropomorphic_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_w1_left_arm_anthropomorphic()
