import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 18
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_2: float = 0.612
a_3: float = 0.5723
d_1: float = 0.220941
d_2: float = -0.1719
d_3: float = 0.1149
d_4: float = 0.1157
post_transform_d5: float = 0.0922
pre_transform_d4: float = 0.1273

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = 0.0
th_2_offset_original2raw: float = 0.0
th_3_offset_original2raw: float = 0.0
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 3.141592653589793


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def ur10_urdf_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = r_12
    ee_transformed[0, 1] = r_13
    ee_transformed[0, 2] = r_11
    ee_transformed[0, 3] = Px - post_transform_d5*r_11
    ee_transformed[1, 0] = r_22
    ee_transformed[1, 1] = r_23
    ee_transformed[1, 2] = r_21
    ee_transformed[1, 3] = Py - post_transform_d5*r_21
    ee_transformed[2, 0] = r_32
    ee_transformed[2, 1] = r_33
    ee_transformed[2, 2] = r_31
    ee_transformed[2, 3] = Pz - post_transform_d5*r_31 - pre_transform_d4
    return ee_transformed


def ur10_urdf_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 1] = r_11
    ee_transformed[0, 2] = r_12
    ee_transformed[0, 3] = Px + post_transform_d5*r_13
    ee_transformed[1, 0] = r_23
    ee_transformed[1, 1] = r_21
    ee_transformed[1, 2] = r_22
    ee_transformed[1, 3] = Py + post_transform_d5*r_23
    ee_transformed[2, 0] = r_33
    ee_transformed[2, 1] = r_31
    ee_transformed[2, 2] = r_32
    ee_transformed[2, 3] = Pz + post_transform_d5*r_33 + pre_transform_d4
    return ee_transformed


def ur10_urdf_fk(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.cos(th_4)
    x2 = math.sin(th_4)
    x3 = math.cos(th_3)
    x4 = math.cos(th_0)
    x5 = math.cos(th_1)
    x6 = math.cos(th_2)
    x7 = x5*x6
    x8 = math.sin(th_1)
    x9 = math.sin(th_2)
    x10 = x8*x9
    x11 = -x10*x4 + x4*x7
    x12 = math.sin(th_3)
    x13 = x6*x8
    x14 = x5*x9
    x15 = -x13*x4 - x14*x4
    x16 = x11*x3 + x12*x15
    x17 = x0*x1 - x16*x2
    x18 = math.sin(th_5)
    x19 = -x11*x12 + x15*x3
    x20 = math.cos(th_5)
    x21 = x0*x2 + x1*x16
    x22 = a_2*x5
    x23 = -x0*x10 + x0*x7
    x24 = -x0*x13 - x0*x14
    x25 = x12*x24 + x23*x3
    x26 = -x1*x4 - x2*x25
    x27 = -x12*x23 + x24*x3
    x28 = x1*x25 - x2*x4
    x29 = x10 - x7
    x30 = -x13 - x14
    x31 = x12*x29 + x3*x30
    x32 = x2*x31
    x33 = -x12*x30 + x29*x3
    x34 = x1*x31
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = x17
    ee_pose[0, 1] = -x18*x19 + x20*x21
    ee_pose[0, 2] = -x18*x21 - x19*x20
    ee_pose[0, 3] = a_3*x11 - d_1*x0 - d_2*x0 - d_3*x0 + d_4*x19 + post_transform_d5*x17 + x22*x4
    ee_pose[1, 0] = x26
    ee_pose[1, 1] = -x18*x27 + x20*x28
    ee_pose[1, 2] = -x18*x28 - x20*x27
    ee_pose[1, 3] = a_3*x23 + d_1*x4 + d_2*x4 + d_3*x4 + d_4*x27 + post_transform_d5*x26 + x0*x22
    ee_pose[2, 0] = -x32
    ee_pose[2, 1] = -x18*x33 + x20*x34
    ee_pose[2, 2] = -x18*x34 - x20*x33
    ee_pose[2, 3] = -a_2*x8 + a_3*x30 + d_4*x33 - post_transform_d5*x32 + pre_transform_d4
    return ee_pose


def ur10_urdf_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = -x0
    x2 = math.sin(th_3)
    x3 = math.cos(th_0)
    x4 = math.cos(th_1)
    x5 = math.cos(th_2)
    x6 = x4*x5
    x7 = math.sin(th_1)
    x8 = math.sin(th_2)
    x9 = x7*x8
    x10 = x3*x6 - x3*x9
    x11 = math.cos(th_3)
    x12 = x5*x7
    x13 = x4*x8
    x14 = -x12*x3 - x13*x3
    x15 = -x10*x2 + x11*x14
    x16 = math.cos(th_4)
    x17 = math.sin(th_4)
    x18 = x0*x16 - x17*(x10*x11 + x14*x2)
    x19 = x0*x6 - x0*x9
    x20 = -x0*x12 - x0*x13
    x21 = x11*x20 - x19*x2
    x22 = -x16*x3 - x17*(x11*x19 + x2*x20)
    x23 = -x6 + x9
    x24 = -x12 - x13
    x25 = x11*x23 - x2*x24
    x26 = x17*(x11*x24 + x2*x23)
    x27 = -a_2*x7 + pre_transform_d4
    x28 = a_3*x24 + x27
    x29 = d_4*x25 + x28
    x30 = a_2*x4
    x31 = d_1*x3 + d_2*x3 + x0*x30
    x32 = a_3*x19 + d_3*x3 + x31
    x33 = d_4*x21 + x32
    x34 = -d_1*x0 - d_2*x0 + x3*x30
    x35 = a_3*x10 - d_3*x0 + x34
    x36 = d_4*x15 + x35
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = x1
    jacobian_output[0, 2] = x1
    jacobian_output[0, 3] = x1
    jacobian_output[0, 4] = x15
    jacobian_output[0, 5] = x18
    jacobian_output[1, 1] = x3
    jacobian_output[1, 2] = x3
    jacobian_output[1, 3] = x3
    jacobian_output[1, 4] = x21
    jacobian_output[1, 5] = x22
    jacobian_output[2, 0] = 1
    jacobian_output[2, 4] = x25
    jacobian_output[2, 5] = -x26
    jacobian_output[3, 1] = -pre_transform_d4*x3
    jacobian_output[3, 2] = -x27*x3
    jacobian_output[3, 3] = -x28*x3
    jacobian_output[3, 4] = -x21*x29 + x25*x33
    jacobian_output[3, 5] = -x22*x29 - x26*x33
    jacobian_output[4, 1] = -pre_transform_d4*x0
    jacobian_output[4, 2] = -x0*x27
    jacobian_output[4, 3] = -x0*x28
    jacobian_output[4, 4] = x15*x29 - x25*x36
    jacobian_output[4, 5] = x18*x29 + x26*x36
    jacobian_output[5, 2] = x0*x31 + x3*x34
    jacobian_output[5, 3] = x0*x32 + x3*x35
    jacobian_output[5, 4] = -x15*x33 + x21*x36
    jacobian_output[5, 5] = -x18*x33 + x22*x36
    return jacobian_output


def ur10_urdf_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = -x0
    x2 = math.sin(th_3)
    x3 = math.cos(th_0)
    x4 = math.cos(th_1)
    x5 = math.cos(th_2)
    x6 = x4*x5
    x7 = math.sin(th_1)
    x8 = math.sin(th_2)
    x9 = x7*x8
    x10 = x3*x6 - x3*x9
    x11 = math.cos(th_3)
    x12 = x5*x7
    x13 = x4*x8
    x14 = -x12*x3 - x13*x3
    x15 = math.cos(th_4)
    x16 = math.sin(th_4)
    x17 = x0*x6 - x0*x9
    x18 = -x0*x12 - x0*x13
    x19 = -x6 + x9
    x20 = -x12 - x13
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 1] = x1
    jacobian_output[0, 2] = x1
    jacobian_output[0, 3] = x1
    jacobian_output[0, 4] = -x10*x2 + x11*x14
    jacobian_output[0, 5] = x0*x15 - x16*(x10*x11 + x14*x2)
    jacobian_output[1, 1] = x3
    jacobian_output[1, 2] = x3
    jacobian_output[1, 3] = x3
    jacobian_output[1, 4] = x11*x18 - x17*x2
    jacobian_output[1, 5] = -x15*x3 - x16*(x11*x17 + x18*x2)
    jacobian_output[2, 0] = 1
    jacobian_output[2, 4] = x11*x19 - x2*x20
    jacobian_output[2, 5] = -x16*(x11*x20 + x19*x2)
    return jacobian_output


def ur10_urdf_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x3 = -a_2*x2 + pre_transform_d4
    x4 = math.cos(th_2)
    x5 = x2*x4
    x6 = math.sin(th_2)
    x7 = math.cos(th_1)
    x8 = x6*x7
    x9 = -x5 - x8
    x10 = a_3*x9 + x3
    x11 = math.cos(th_3)
    x12 = x2*x6
    x13 = x4*x7
    x14 = x12 - x13
    x15 = math.sin(th_3)
    x16 = x11*x14 - x15*x9
    x17 = math.sin(th_0)
    x18 = -x12*x17 + x13*x17
    x19 = -x17*x5 - x17*x8
    x20 = x11*x19 - x15*x18
    x21 = d_4*x16 + x10
    x22 = a_2*x7
    x23 = d_1*x0 + d_2*x0 + x17*x22
    x24 = a_3*x18 + d_3*x0 + x23
    x25 = d_4*x20 + x24
    x26 = math.sin(th_4)
    x27 = x26*(x11*x9 + x14*x15)
    x28 = math.cos(th_4)
    x29 = -x0*x28 - x26*(x11*x18 + x15*x19)
    x30 = p_on_ee_z*x17
    x31 = -x0*x12 + x0*x13
    x32 = -x0*x5 - x0*x8
    x33 = x11*x32 - x15*x31
    x34 = -d_1*x17 - d_2*x17 + x0*x22
    x35 = a_3*x31 - d_3*x17 + x34
    x36 = d_4*x33 + x35
    x37 = x17*x28 - x26*(x11*x31 + x15*x32)
    x38 = -p_on_ee_x*x0 - p_on_ee_y*x17
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -p_on_ee_y
    jacobian_output[0, 1] = -pre_transform_d4*x0 + x1
    jacobian_output[0, 2] = -x0*x3 + x1
    jacobian_output[0, 3] = -x0*x10 + x1
    jacobian_output[0, 4] = -p_on_ee_y*x16 + p_on_ee_z*x20 + x16*x25 - x20*x21
    jacobian_output[0, 5] = p_on_ee_y*x27 + p_on_ee_z*x29 - x21*x29 - x25*x27
    jacobian_output[1, 0] = p_on_ee_x
    jacobian_output[1, 1] = -pre_transform_d4*x17 + x30
    jacobian_output[1, 2] = -x17*x3 + x30
    jacobian_output[1, 3] = -x10*x17 + x30
    jacobian_output[1, 4] = p_on_ee_x*x16 - p_on_ee_z*x33 - x16*x36 + x21*x33
    jacobian_output[1, 5] = -p_on_ee_x*x27 - p_on_ee_z*x37 + x21*x37 + x27*x36
    jacobian_output[2, 1] = x38
    jacobian_output[2, 2] = x0*x34 + x17*x23 + x38
    jacobian_output[2, 3] = x0*x35 + x17*x24 + x38
    jacobian_output[2, 4] = -p_on_ee_x*x20 + p_on_ee_y*x33 + x20*x36 - x25*x33
    jacobian_output[2, 5] = -p_on_ee_x*x29 + p_on_ee_y*x37 - x25*x37 + x29*x36
    return jacobian_output


def ur10_urdf_ik_solve_raw(T_ee: np.ndarray):
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
    for i in range(18):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True
    
    # Code for non-branch dispatcher node 0
    # Actually, there is no code
    
    # Code for explicit solution node 1, solved variable is th_0
    def ExplicitSolutionNode_node_1_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return
        
        # The explicit solution of root node
        condition_0: bool = (abs(Px) >= zero_tolerance) or (abs(Py) >= zero_tolerance) or (abs(d_1 + d_2 + d_3) >= zero_tolerance)
        if condition_0:
            # Temp variable for efficiency
            x0 = math.atan2(Px, -Py)
            x1 = -d_1 - d_2 - d_3
            x2 = safe_sqrt(Px**2 + Py**2 - x1**2)
            # End of temp variables
            solution_0: IkSolution = make_ik_solution()
            solution_0[0] = x0 + math.atan2(x2, x1)
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)
            
        condition_1: bool = (abs(Px) >= zero_tolerance) or (abs(Py) >= zero_tolerance) or (abs(d_1 + d_2 + d_3) >= zero_tolerance)
        if condition_1:
            # Temp variable for efficiency
            x0 = math.atan2(Px, -Py)
            x1 = -d_1 - d_2 - d_3
            x2 = safe_sqrt(Px**2 + Py**2 - x1**2)
            # End of temp variables
            solution_1: IkSolution = make_ik_solution()
            solution_1[0] = x0 + math.atan2(-x2, x1)
            appended_idx = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx)
            
    # Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_0_processor()
    # Finish code for explicit solution node 0
    
    # Code for non-branch dispatcher node 2
    # Actually, there is no code
    
    # Code for explicit solution node 3, solved variable is th_4
    def ExplicitSolutionNode_node_3_solve_th_4_processor():
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
            condition_0: bool = (abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = safe_acos(r_13*math.sin(th_0) - r_23*math.cos(th_0))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[7] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = safe_acos(r_13*math.sin(th_0) - r_23*math.cos(th_0))
                # End of temp variables
                this_solution[7] = -x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_4_processor()
    # Finish code for explicit solution node 2
    
    # Code for solved_variable dispatcher node 4
    def SolvedVariableDispatcherNode_node_4_processor():
        this_node_input_index: List[int] = node_input_index[4]
        this_input_valid: bool = node_input_validity[4]
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
            
            th_4 = this_solution[7]
            degenerate_valid_1 = (abs(th_4 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
            
            if not taken_by_degenerate:
                add_input_index_to(5, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_4_processor()
    # Finish code for solved_variable dispatcher node 4
    
    # Code for explicit solution node 5, solved variable is th_1th_2th_3_soa
    def ExplicitSolutionNode_node_5_solve_th_1th_2th_3_soa_processor():
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
            th_0 = this_solution[0]
            th_4 = this_solution[7]
            condition_0: bool = (abs(r_33) >= zero_tolerance) or (abs(r_13*math.cos(th_0) + r_23*math.sin(th_0)) >= zero_tolerance) or (abs(math.sin(th_4)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_4)**(-1)
                # End of temp variables
                this_solution[3] = math.atan2(r_33*x0, x0*(-r_13*math.cos(th_0) - r_23*math.sin(th_0)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_1th_2th_3_soa_processor()
    # Finish code for explicit solution node 5
    
    # Code for non-branch dispatcher node 6
    # Actually, there is no code
    
    # Code for explicit solution node 7, solved variable is th_5
    def ExplicitSolutionNode_node_7_solve_th_5_processor():
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
            th_4 = this_solution[7]
            condition_0: bool = (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance) or (abs(math.sin(th_4)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_4)**(-1)
                x1 = math.cos(th_0)
                x2 = math.sin(th_0)
                # End of temp variables
                this_solution[8] = math.atan2(x0*(-r_12*x2 + r_22*x1), x0*(r_11*x2 - r_21*x1))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_5_processor()
    # Finish code for explicit solution node 6
    
    # Code for non-branch dispatcher node 8
    # Actually, there is no code
    
    # Code for explicit solution node 9, solved variable is th_2
    def ExplicitSolutionNode_node_9_solve_th_2_processor():
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
            th_5 = this_solution[8]
            condition_0: bool = ((1/2)*abs((a_2**2 + a_3**2 + d_1**2 + 2*d_1*d_2 + 2*d_1*d_3 + d_2**2 + 2*d_2*d_3 + d_3**2 - d_4**2 + 2*d_4*inv_Px*math.sin(th_5) + 2*d_4*inv_Py*math.cos(th_5) - inv_Px**2 - inv_Py**2 - inv_Pz**2)/(a_2*a_3)) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*d_1
                x1 = 2*d_4
                x2 = safe_acos((1/2)*(-a_2**2 - a_3**2 - d_1**2 - d_2**2 - 2*d_2*d_3 - d_2*x0 - d_3**2 - d_3*x0 + d_4**2 + inv_Px**2 - inv_Px*x1*math.sin(th_5) + inv_Py**2 - inv_Py*x1*math.cos(th_5) + inv_Pz**2)/(a_2*a_3))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(10, appended_idx)
                
            condition_1: bool = ((1/2)*abs((a_2**2 + a_3**2 + d_1**2 + 2*d_1*d_2 + 2*d_1*d_3 + d_2**2 + 2*d_2*d_3 + d_3**2 - d_4**2 + 2*d_4*inv_Px*math.sin(th_5) + 2*d_4*inv_Py*math.cos(th_5) - inv_Px**2 - inv_Py**2 - inv_Pz**2)/(a_2*a_3)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*d_1
                x1 = 2*d_4
                x2 = safe_acos((1/2)*(-a_2**2 - a_3**2 - d_1**2 - d_2**2 - 2*d_2*d_3 - d_2*x0 - d_3**2 - d_3*x0 + d_4**2 + inv_Px**2 - inv_Px*x1*math.sin(th_5) + inv_Py**2 - inv_Py*x1*math.cos(th_5) + inv_Pz**2)/(a_2*a_3))
                # End of temp variables
                this_solution[4] = -x2
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_2_processor()
    # Finish code for explicit solution node 8
    
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
            th_5 = this_solution[8]
            checked_result: bool = (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5)) <= 1.0e-6) and (abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_4*r_11*math.sin(th_5)*math.cos(th_0) + d_4*r_12*math.cos(th_0)*math.cos(th_5) + d_4*r_21*math.sin(th_0)*math.sin(th_5) + d_4*r_22*math.sin(th_0)*math.cos(th_5)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(11, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_10_processor()
    # Finish code for equation all-zero dispatcher node 10
    
    # Code for explicit solution node 11, solved variable is th_1th_2_soa
    def ExplicitSolutionNode_node_11_solve_th_1th_2_soa_processor():
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
            th_2 = this_solution[4]
            th_5 = this_solution[8]
            condition_0: bool = (abs(Pz + d_4*r_31*math.sin(th_5) + d_4*r_32*math.cos(th_5)) >= 1.0e-6) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_4*r_11*math.sin(th_5)*math.cos(th_0) + d_4*r_12*math.cos(th_0)*math.cos(th_5) + d_4*r_21*math.sin(th_0)*math.sin(th_5) + d_4*r_22*math.sin(th_0)*math.cos(th_5)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = -a_2*math.cos(th_2) - a_3
                x1 = d_4*math.sin(th_5)
                x2 = d_4*math.cos(th_5)
                x3 = Pz + r_31*x1 + r_32*x2
                x4 = math.cos(th_0)
                x5 = math.sin(th_0)
                x6 = -Px*x4 - Py*x5 - r_11*x1*x4 - r_12*x2*x4 - r_21*x1*x5 - r_22*x2*x5
                x7 = a_2*math.sin(th_2)
                # End of temp variables
                this_solution[2] = math.atan2(x0*x3 - x6*x7, x0*x6 + x3*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(12, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1th_2_soa_processor()
    # Finish code for explicit solution node 11
    
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
                add_input_index_to(14, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_1_processor()
    # Finish code for explicit solution node 12
    
    # Code for non-branch dispatcher node 14
    # Actually, there is no code
    
    # Code for explicit solution node 15, solved variable is th_3
    def ExplicitSolutionNode_node_15_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[14]
        this_input_valid: bool = node_input_validity[14]
        if not this_input_valid:
            return
        
        # The solution of non-root node 15
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1th_2_soa = this_solution[2]
            th_1th_2th_3_soa = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[6] = -th_1th_2_soa + th_1th_2th_3_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(16, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_15_solve_th_3_processor()
    # Finish code for explicit solution node 14
    
    # Code for non-branch dispatcher node 16
    # Actually, there is no code
    
    # Code for explicit solution node 17, solved variable is th_2th_3_soa
    def ExplicitSolutionNode_node_17_solve_th_2th_3_soa_processor():
        this_node_input_index: List[int] = node_input_index[16]
        this_input_valid: bool = node_input_validity[16]
        if not this_input_valid:
            return
        
        # The solution of non-root node 17
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[4]
            th_3 = this_solution[6]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[5] = th_2 + th_3
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_2th_3_soa_processor()
    # Finish code for explicit solution node 16
    
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
        value_at_3 = ik_out_i[6]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[7]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[8]  # th_5
        new_ik_i[5] = value_at_5
        ik_out.append(new_ik_i)
    return ik_out


def ur10_urdf_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = ur10_urdf_ik_target_original_to_raw(T_ee)
    ik_output_raw = ur10_urdf_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ee_pose_i = ur10_urdf_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_ur10_urdf():
    theta_in = np.random.random(size=(6, ))
    ee_pose = ur10_urdf_fk(theta_in)
    ik_output = ur10_urdf_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = ur10_urdf_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_ur10_urdf()
