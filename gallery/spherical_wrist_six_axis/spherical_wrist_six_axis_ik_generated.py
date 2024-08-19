import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 24
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 0.32
a_1: float = 1.28
a_2: float = 0.2
d_3: float = 1.1825
d_4: float = 0.2
pre_transform_special_symbol_23: float = 0.78

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
def spherical_wrist_six_axis_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = -1.0*r_11 + 6.1232339957367697e-17*r_13
    ee_transformed[0, 1] = -1.0*r_12
    ee_transformed[0, 2] = 6.1232339957367697e-17*r_11 + 1.0*r_13
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = -1.0*r_21 + 6.1232339957367697e-17*r_23
    ee_transformed[1, 1] = -1.0*r_22
    ee_transformed[1, 2] = 6.1232339957367697e-17*r_21 + 1.0*r_23
    ee_transformed[1, 3] = 1.0*Py
    ee_transformed[2, 0] = -1.0*r_31 + 6.1232339957367697e-17*r_33
    ee_transformed[2, 1] = -1.0*r_32
    ee_transformed[2, 2] = 6.1232339957367697e-17*r_31 + 1.0*r_33
    ee_transformed[2, 3] = 1.0*Pz - 1.0*pre_transform_special_symbol_23
    return ee_transformed


def spherical_wrist_six_axis_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = -1.0*r_11 + 6.1232339957367697e-17*r_13
    ee_transformed[0, 1] = -1.0*r_12
    ee_transformed[0, 2] = 6.1232339957367697e-17*r_11 + 1.0*r_13
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = -1.0*r_21 + 6.1232339957367697e-17*r_23
    ee_transformed[1, 1] = -1.0*r_22
    ee_transformed[1, 2] = 6.1232339957367697e-17*r_21 + 1.0*r_23
    ee_transformed[1, 3] = 1.0*Py
    ee_transformed[2, 0] = -1.0*r_31 + 6.1232339957367697e-17*r_33
    ee_transformed[2, 1] = -1.0*r_32
    ee_transformed[2, 2] = 6.1232339957367697e-17*r_31 + 1.0*r_33
    ee_transformed[2, 3] = 1.0*Pz + 1.0*pre_transform_special_symbol_23
    return ee_transformed


def spherical_wrist_six_axis_fk(theta_input: np.ndarray):
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
    x18 = math.sin(th_5)
    x19 = x11*x13 - x12*x15
    x20 = x0*x16 - x10*x8
    x21 = math.cos(th_5)
    x22 = 1.0*x21
    x23 = 1.0*x18
    x24 = 6.12323399573677e-17*x18
    x25 = 1.0*a_0
    x26 = 1.0*a_1
    x27 = x26*x6
    x28 = 1.0*a_2
    x29 = 1.0*d_3
    x30 = 1.0*d_4
    x31 = -x11*x4 - x11*x7
    x32 = x0*x31
    x33 = -x11*x14 + x11*x3*x6
    x34 = -x1*x12 + x13*x33
    x35 = x10*x34
    x36 = -x1*x13 - x12*x33
    x37 = x0*x34 - x10*x31
    x38 = x14 - x3*x6
    x39 = x0*x38
    x40 = -x4 - x7
    x41 = x12*x40
    x42 = x10*x13*x40
    x43 = x0*x13*x40 - x10*x38
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -6.12323399573677e-17*x17 + 1.0*x18*x19 - x20*x22 - 6.12323399573677e-17*x9
    ee_pose[0, 1] = x19*x22 + x20*x23
    ee_pose[0, 2] = -1.0*x17 - x19*x24 + 6.12323399573677e-17*x20*x21 - 1.0*x9
    ee_pose[0, 3] = x1*x25 + x1*x27 + x15*x28 + x29*x8 + x30*(-x17 - x9)
    ee_pose[1, 0] = 1.0*x18*x36 - x22*x37 - 6.12323399573677e-17*x32 - 6.12323399573677e-17*x35
    ee_pose[1, 1] = x22*x36 + x23*x37
    ee_pose[1, 2] = 6.12323399573677e-17*x21*x37 - x24*x36 - 1.0*x32 - 1.0*x35
    ee_pose[1, 3] = x11*x25 + x11*x27 + x28*x33 + x29*x31 + x30*(-x32 - x35)
    ee_pose[2, 0] = -x22*x43 - x23*x41 - 6.12323399573677e-17*x39 - 6.12323399573677e-17*x42
    ee_pose[2, 1] = -x22*x41 + x23*x43
    ee_pose[2, 2] = 6.12323399573677e-17*x21*x43 + x24*x41 - 1.0*x39 - 1.0*x42
    ee_pose[2, 3] = 1.0*pre_transform_special_symbol_23 - x2*x26 + x28*x40 + x29*x38 + x30*(-x39 - x42)
    return ee_pose


def spherical_wrist_six_axis_twist_jacobian(theta_input: np.ndarray):
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
    x31 = 1.0*a_1*x4
    x32 = pre_transform_special_symbol_23 - x31
    x33 = a_2*x28 + d_3*x27 + pre_transform_special_symbol_23 - x31
    x34 = a_0*x1 + a_1*x20
    x35 = a_2*x22 + d_3*x21 + x34
    x36 = d_4*x30 + x33
    x37 = d_4*x24 + x35
    x38 = a_0*x6 + a_1*x10
    x39 = a_2*x14 + d_3*x11 + x38
    x40 = d_4*x18 + x39
    x41 = 1.0*a_0
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
    jacobian_output[3, 2] = -x32*x6
    jacobian_output[3, 3] = -x21*x33 + x27*x35
    jacobian_output[3, 4] = -x23*x33 - x29*x35
    jacobian_output[3, 5] = -x24*x36 + x30*x37
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = -x1*x32
    jacobian_output[4, 3] = x11*x33 - x27*x39
    jacobian_output[4, 4] = x15*x33 + x29*x39
    jacobian_output[4, 5] = x18*x36 - x30*x40
    jacobian_output[5, 1] = x0**2*x41 + x41*x5**2
    jacobian_output[5, 2] = x1*x34 + x38*x6
    jacobian_output[5, 3] = -x11*x35 + x21*x39
    jacobian_output[5, 4] = -x15*x35 + x23*x39
    jacobian_output[5, 5] = -x18*x37 + x24*x40
    return jacobian_output


def spherical_wrist_six_axis_angular_velocity_jacobian(theta_input: np.ndarray):
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


def spherical_wrist_six_axis_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x31 = -x17*x28 - x29*(-x2*x24 + x20*x26)
    x32 = d_4*x30 + x19
    x33 = d_4*x31 + x23
    x34 = 1.0*p_on_ee_x
    x35 = p_on_ee_z*x21
    x36 = x2*x4
    x37 = x11*x2
    x38 = -x10*x36 - x37*x8
    x39 = 1.0*x1*x10*x11 - x36*x8
    x40 = a_0*x2 + a_1*x37
    x41 = a_2*x39 + d_3*x38 + x40
    x42 = 1.0*x14*x26 - x24*x39
    x43 = -x28*x38 - x29*(x21*x24 + x26*x39)
    x44 = d_4*x43 + x41
    x45 = x1*x34
    x46 = x0*x14
    x47 = 1.0*a_0
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = -x2*x7 + x3
    jacobian_output[0, 3] = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x23 - x17*x19
    jacobian_output[0, 4] = p_on_ee_y*x25 + p_on_ee_z*x27 - x19*x27 - x23*x25
    jacobian_output[0, 5] = -p_on_ee_y*x30 + p_on_ee_z*x31 + x30*x33 - x31*x32
    jacobian_output[1, 0] = x34
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x21 + x35
    jacobian_output[1, 2] = -x21*x7 + x35
    jacobian_output[1, 3] = p_on_ee_x*x13 - p_on_ee_z*x38 - x13*x41 + x19*x38
    jacobian_output[1, 4] = -p_on_ee_x*x25 - p_on_ee_z*x42 + x18*x24*x41 + x19*x42
    jacobian_output[1, 5] = p_on_ee_x*x30 - p_on_ee_z*x43 - x30*x44 + x32*x43
    jacobian_output[2, 1] = x1**2*x47 + x14**2*x47 - x45 - x46
    jacobian_output[2, 2] = 1.0*x1*x40 + 1.0*x14*x22 - x45 - x46
    jacobian_output[2, 3] = -p_on_ee_x*x17 + p_on_ee_y*x38 + x17*x41 - x23*x38
    jacobian_output[2, 4] = -p_on_ee_x*x27 + p_on_ee_y*x42 - x23*x42 + x27*x41
    jacobian_output[2, 5] = -p_on_ee_x*x31 + p_on_ee_y*x43 + x31*x44 - x33*x43
    return jacobian_output


def spherical_wrist_six_axis_ik_solve_raw(T_ee: np.ndarray):
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
        for tmp_sol_idx in range(9):
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
    
    # Code for equation all-zero dispatcher node 0
    def EquationAllZeroDispatcherNode_node_0_processor():
        checked_result: bool = (abs(Px - d_4*r_13) <= 1.0e-6) and (abs(Py - d_4*r_23) <= 1.0e-6)
        if not checked_result:  # To non-degenerate node
            node_input_validity[1] = True
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_0_processor()
    # Finish code for equation all-zero dispatcher node 0
    
    # Code for explicit solution node 1, solved variable is th_0
    def ExplicitSolutionNode_node_1_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[1]
        this_input_valid: bool = node_input_validity[1]
        if not this_input_valid:
            return
        
        # The explicit solution of root node
        condition_0: bool = (abs(Px - d_4*r_13) >= zero_tolerance) or (abs(Py - d_4*r_23) >= zero_tolerance)
        if condition_0:
            # Temp variable for efficiency
            x0 = math.atan2(Py - d_4*r_23, Px - d_4*r_13)
            # End of temp variables
            solution_0: IkSolution = make_ik_solution()
            solution_0[1] = x0
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)
            
        condition_1: bool = (abs(Px - d_4*r_13) >= zero_tolerance) or (abs(Py - d_4*r_23) >= zero_tolerance)
        if condition_1:
            # Temp variable for efficiency
            x0 = math.atan2(Py - d_4*r_23, Px - d_4*r_13)
            # End of temp variables
            solution_1: IkSolution = make_ik_solution()
            solution_1[1] = x0 + math.pi
            appended_idx = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx)
            
    # Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_0_processor()
    # Finish code for explicit solution node 1
    
    # Code for non-branch dispatcher node 2
    # Actually, there is no code
    
    # Code for explicit solution node 3, solved variable is th_2
    def ExplicitSolutionNode_node_3_solve_th_2_processor():
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
            th_0 = this_solution[1]
            condition_0: bool = (2*abs(a_1*a_2) >= zero_tolerance) or (2*abs(a_1*d_3) >= zero_tolerance) or (abs(Px**2 - 2*Px*a_0*math.cos(th_0) - 2*Px*d_4*r_13 + Py**2 - 2*Py*a_0*math.sin(th_0) - 2*Py*d_4*r_23 + Pz**2 - 2*Pz*d_4*r_33 + a_0**2 + 2*a_0*d_4*r_13*math.cos(th_0) + 2*a_0*d_4*r_23*math.sin(th_0) - a_1**2 - a_2**2 - d_3**2 + d_4**2*r_13**2 + d_4**2*r_23**2 + d_4**2*r_33**2) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*a_1
                x1 = math.atan2(-d_3*x0, a_2*x0)
                x2 = a_2**2
                x3 = a_1**2
                x4 = 4*x3
                x5 = d_3**2
                x6 = 2*Px
                x7 = d_4*r_13
                x8 = 2*Py
                x9 = d_4*r_23
                x10 = a_0*math.cos(th_0)
                x11 = a_0*math.sin(th_0)
                x12 = d_4**2
                x13 = Px**2 + Py**2 + Pz**2 - 2*Pz*d_4*r_33 + a_0**2 + r_13**2*x12 + r_23**2*x12 + r_33**2*x12 - x10*x6 + 2*x10*x7 - x11*x8 + 2*x11*x9 - x2 - x3 - x5 - x6*x7 - x8*x9
                x14 = safe_sqrt(-x13**2 + x2*x4 + x4*x5)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x1 + math.atan2(x14, x13)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (2*abs(a_1*a_2) >= zero_tolerance) or (2*abs(a_1*d_3) >= zero_tolerance) or (abs(Px**2 - 2*Px*a_0*math.cos(th_0) - 2*Px*d_4*r_13 + Py**2 - 2*Py*a_0*math.sin(th_0) - 2*Py*d_4*r_23 + Pz**2 - 2*Pz*d_4*r_33 + a_0**2 + 2*a_0*d_4*r_13*math.cos(th_0) + 2*a_0*d_4*r_23*math.sin(th_0) - a_1**2 - a_2**2 - d_3**2 + d_4**2*r_13**2 + d_4**2*r_23**2 + d_4**2*r_33**2) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*a_1
                x1 = math.atan2(-d_3*x0, a_2*x0)
                x2 = a_2**2
                x3 = a_1**2
                x4 = 4*x3
                x5 = d_3**2
                x6 = 2*Px
                x7 = d_4*r_13
                x8 = 2*Py
                x9 = d_4*r_23
                x10 = a_0*math.cos(th_0)
                x11 = a_0*math.sin(th_0)
                x12 = d_4**2
                x13 = Px**2 + Py**2 + Pz**2 - 2*Pz*d_4*r_33 + a_0**2 + r_13**2*x12 + r_23**2*x12 + r_33**2*x12 - x10*x6 + 2*x10*x7 - x11*x8 + 2*x11*x9 - x2 - x3 - x5 - x6*x7 - x8*x9
                x14 = safe_sqrt(-x13**2 + x2*x4 + x4*x5)
                # End of temp variables
                this_solution[4] = x1 + math.atan2(-x14, x13)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_2_processor()
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
            th_0 = this_solution[1]
            checked_result: bool = (abs(Pz - d_4*r_33) <= 1.0e-6) and (abs(-Px*math.cos(th_0) - Py*math.sin(th_0) + a_0 + d_4*r_13*math.cos(th_0) + d_4*r_23*math.sin(th_0)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor()
    # Finish code for equation all-zero dispatcher node 4
    
    # Code for explicit solution node 5, solved variable is th_1th_2_soa
    def ExplicitSolutionNode_node_5_solve_th_1th_2_soa_processor():
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
            th_0 = this_solution[1]
            th_2 = this_solution[4]
            condition_0: bool = (abs(Pz - d_4*r_33) >= 1.0e-6) or (abs(-Px*math.cos(th_0) - Py*math.sin(th_0) + a_0 + d_4*r_13*math.cos(th_0) + d_4*r_23*math.sin(th_0)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = Pz - d_4*r_33
                x1 = -a_1*math.cos(th_2) - a_2
                x2 = a_1*math.sin(th_2) - d_3
                x3 = math.cos(th_0)
                x4 = math.sin(th_0)
                x5 = -Px*x3 - Py*x4 + a_0 + d_4*r_13*x3 + d_4*r_23*x4
                # End of temp variables
                this_solution[3] = math.atan2(x0*x1 - x2*x5, x0*x2 + x1*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_1th_2_soa_processor()
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
            th_1th_2_soa = this_solution[3]
            th_2 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[2] = th_1th_2_soa - th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_1_processor()
    # Finish code for explicit solution node 6
    
    # Code for non-branch dispatcher node 8
    # Actually, there is no code
    
    # Code for explicit solution node 9, solved variable is th_4
    def ExplicitSolutionNode_node_9_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[8]
        this_input_valid: bool = node_input_validity[8]
        if not this_input_valid:
            return
        
        # The solution of non-root node 9
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[1]
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            condition_0: bool = (abs(r_13*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_23*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_0) + r_33*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_1)
                x1 = math.sin(th_2)
                x2 = math.cos(th_1)
                x3 = math.cos(th_2)
                x4 = x0*x3 + x1*x2
                x5 = safe_acos(r_13*x4*math.cos(th_0) + r_23*x4*math.sin(th_0) + r_33*(-x0*x1 + x2*x3))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[7] = x5
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(10, appended_idx)
                
            condition_1: bool = (abs(r_13*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_23*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_0) + r_33*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.sin(th_1)
                x1 = math.sin(th_2)
                x2 = math.cos(th_1)
                x3 = math.cos(th_2)
                x4 = x0*x3 + x1*x2
                x5 = safe_acos(r_13*x4*math.cos(th_0) + r_23*x4*math.sin(th_0) + r_33*(-x0*x1 + x2*x3))
                # End of temp variables
                this_solution[7] = -x5
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_4_processor()
    # Finish code for explicit solution node 8
    
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
            th_4 = this_solution[7]
            degenerate_valid_0 = (abs(th_4) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(14, node_input_i_idx_in_queue)
            
            th_4 = this_solution[7]
            degenerate_valid_1 = (abs(th_4 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(19, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(11, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_10_processor()
    # Finish code for solved_variable dispatcher node 10
    
    # Code for explicit solution node 19, solved variable is th_3th_5_soa
    def ExplicitSolutionNode_node_19_solve_th_3th_5_soa_processor():
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
            th_0 = this_solution[1]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                # End of temp variables
                this_solution[6] = math.atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(20, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_3th_5_soa_processor()
    # Finish code for explicit solution node 19
    
    # Code for non-branch dispatcher node 20
    # Actually, there is no code
    
    # Code for explicit solution node 21, solved variable is th_3
    def ExplicitSolutionNode_node_21_solve_th_3_processor():
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
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[5] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(22, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_3_processor()
    # Finish code for explicit solution node 20
    
    # Code for non-branch dispatcher node 22
    # Actually, there is no code
    
    # Code for explicit solution node 23, solved variable is th_5
    def ExplicitSolutionNode_node_23_solve_th_5_processor():
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
            th_3 = this_solution[5]
            th_3th_5_soa = this_solution[6]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[8] = -th_3 + th_3th_5_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_5_processor()
    # Finish code for explicit solution node 22
    
    # Code for explicit solution node 14, solved variable is negative_th_5_positive_th_3__soa
    def ExplicitSolutionNode_node_14_solve_negative_th_5_positive_th_3__soa_processor():
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
            th_0 = this_solution[1]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                # End of temp variables
                this_solution[0] = math.atan2(r_11*x0 - r_21*x1, -r_12*x0 + r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_negative_th_5_positive_th_3__soa_processor()
    # Finish code for explicit solution node 14
    
    # Code for non-branch dispatcher node 15
    # Actually, there is no code
    
    # Code for explicit solution node 16, solved variable is th_3
    def ExplicitSolutionNode_node_16_solve_th_3_processor():
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
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[5] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(17, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_3_processor()
    # Finish code for explicit solution node 15
    
    # Code for non-branch dispatcher node 17
    # Actually, there is no code
    
    # Code for explicit solution node 18, solved variable is th_5
    def ExplicitSolutionNode_node_18_solve_th_5_processor():
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
            negative_th_5_positive_th_3__soa = this_solution[0]
            th_3 = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[8] = -negative_th_5_positive_th_3__soa + th_3
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_5_processor()
    # Finish code for explicit solution node 17
    
    # Code for explicit solution node 11, solved variable is th_3
    def ExplicitSolutionNode_node_11_solve_th_3_processor():
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
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            th_4 = this_solution[7]
            condition_0: bool = (abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) >= zero_tolerance) or (abs(-r_13*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.cos(th_0) - r_23*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_0) + r_33*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))) >= zero_tolerance) or (abs(math.sin(th_4)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_4)**(-1)
                x1 = math.sin(th_0)
                x2 = math.cos(th_0)
                x3 = math.sin(th_1)
                x4 = math.cos(th_2)
                x5 = math.sin(th_2)
                x6 = math.cos(th_1)
                x7 = -x3*x5 + x4*x6
                # End of temp variables
                this_solution[5] = math.atan2(x0*(-r_13*x1 + r_23*x2), x0*(-r_13*x2*x7 - r_23*x1*x7 + r_33*(x3*x4 + x5*x6)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(12, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_3_processor()
    # Finish code for explicit solution node 11
    
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
            th_0 = this_solution[1]
            th_1 = this_solution[2]
            th_2 = this_solution[4]
            th_4 = this_solution[7]
            condition_0: bool = (abs(r_11*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_21*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_0) - r_31*(math.sin(th_1)*math.sin(th_2) - math.cos(th_1)*math.cos(th_2))) >= zero_tolerance) or (abs(r_12*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_22*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_0) + r_32*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) >= zero_tolerance) or (abs(math.sin(th_4)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_4)**(-1)
                x1 = math.sin(th_1)
                x2 = math.sin(th_2)
                x3 = math.cos(th_1)
                x4 = math.cos(th_2)
                x5 = -x1*x2 + x3*x4
                x6 = x1*x4 + x2*x3
                x7 = x6*math.cos(th_0)
                x8 = x6*math.sin(th_0)
                # End of temp variables
                this_solution[8] = math.atan2(x0*(-r_12*x7 - r_22*x8 - r_32*x5), x0*(r_11*x7 + r_21*x8 + r_31*x5))
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
        value_at_0 = ik_out_i[1]  # th_0
        new_ik_i[0] = value_at_0
        value_at_1 = ik_out_i[2]  # th_1
        new_ik_i[1] = value_at_1
        value_at_2 = ik_out_i[4]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[5]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[7]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[8]  # th_5
        new_ik_i[5] = value_at_5
        ik_out.append(new_ik_i)
    return ik_out


def spherical_wrist_six_axis_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = spherical_wrist_six_axis_ik_target_original_to_raw(T_ee)
    ik_output_raw = spherical_wrist_six_axis_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ee_pose_i = spherical_wrist_six_axis_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_spherical_wrist_six_axis():
    theta_in = np.random.random(size=(6, ))
    ee_pose = spherical_wrist_six_axis_fk(theta_in)
    ik_output = spherical_wrist_six_axis_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = spherical_wrist_six_axis_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_spherical_wrist_six_axis()
