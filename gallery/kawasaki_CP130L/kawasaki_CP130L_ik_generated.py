import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 4
n_tree_nodes: int = 16
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 0.255
a_1: float = 0.26
a_2: float = 0.25
d_3: float = 0.24
pre_transform_special_symbol_23: float = 0.75

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -1.5707963267948966
th_2_offset_original2raw: float = -1.5707963267948966
th_3_offset_original2raw: float = 1.5707963267948968


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def kawasaki_CP130L_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = -1.0*r_11
    ee_transformed[0, 1] = 1.0*r_12
    ee_transformed[0, 2] = -1.0*r_13
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = 1.0*r_21
    ee_transformed[1, 1] = -1.0*r_22
    ee_transformed[1, 2] = 1.0*r_23
    ee_transformed[1, 3] = -1.0*Py
    ee_transformed[2, 0] = 1.0*r_31
    ee_transformed[2, 1] = -1.0*r_32
    ee_transformed[2, 2] = 1.0*r_33
    ee_transformed[2, 3] = -1.0*Pz + 1.0*pre_transform_special_symbol_23
    return ee_transformed


def kawasaki_CP130L_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = -1.0*r_11
    ee_transformed[0, 1] = 1.0*r_12
    ee_transformed[0, 2] = -1.0*r_13
    ee_transformed[0, 3] = 1.0*Px
    ee_transformed[1, 0] = 1.0*r_21
    ee_transformed[1, 1] = -1.0*r_22
    ee_transformed[1, 2] = 1.0*r_23
    ee_transformed[1, 3] = -1.0*Py
    ee_transformed[2, 0] = 1.0*r_31
    ee_transformed[2, 1] = -1.0*r_32
    ee_transformed[2, 2] = 1.0*r_33
    ee_transformed[2, 3] = -1.0*Pz + 1.0*pre_transform_special_symbol_23
    return ee_transformed


def kawasaki_CP130L_fk(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.sin(th_3)
    x2 = math.cos(th_0)
    x3 = math.sin(th_1)
    x4 = math.sin(th_2)
    x5 = x3*x4
    x6 = math.cos(th_1)
    x7 = math.cos(th_2)
    x8 = x6*x7
    x9 = x2*x5 + x2*x8
    x10 = 1.0*math.cos(th_3)
    x11 = 1.0*x1
    x12 = x3*x7
    x13 = x12*x2
    x14 = x4*x6
    x15 = x14*x2
    x16 = 1.0*a_0
    x17 = 1.0*a_1
    x18 = x17*x6
    x19 = 1.0*a_2
    x20 = x0*x5 + x0*x8
    x21 = x0*x12
    x22 = x12 - x14
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = 1.0*x0*x1 - x10*x9
    ee_pose[0, 1] = -x0*x10 - x11*x9
    ee_pose[0, 2] = 1.0*x13 - 1.0*x15
    ee_pose[0, 3] = -1.0*d_3*(x13 - x15) + x16*x2 + x18*x2 + x19*x9
    ee_pose[1, 0] = x10*x20 + x11*x2
    ee_pose[1, 1] = -x10*x2 + x11*x20
    ee_pose[1, 2] = 1.0*x0*x4*x6 - 1.0*x21
    ee_pose[1, 3] = 1.0*d_3*(-x0*x14 + x21) - x0*x16 - x0*x18 - x19*x20
    ee_pose[2, 0] = x10*x22
    ee_pose[2, 1] = x11*x22
    ee_pose[2, 2] = 1.0*x5 + 1.0*x8
    ee_pose[2, 3] = 1.0*d_3*(-x5 - x8) + 1.0*pre_transform_special_symbol_23 - x17*x3 - x19*x22
    return ee_pose


def kawasaki_CP130L_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = 1.0*x0
    x2 = math.cos(th_0)
    x3 = 1.0*x2
    x4 = math.cos(th_2)
    x5 = math.sin(th_1)
    x6 = x4*x5
    x7 = math.sin(th_2)
    x8 = math.cos(th_1)
    x9 = -1.0*x2*x7*x8 + x3*x6
    x10 = -x9
    x11 = x1*x6 - x1*x7*x8
    x12 = 1.0*x5
    x13 = x12*x7 + 1.0*x4*x8
    x14 = -x13
    x15 = a_1*x12 - pre_transform_special_symbol_23
    x16 = -x15
    x17 = a_2*(-x12*x4 + 1.0*x7*x8) - d_3*x13 - x15
    x18 = x5*x7
    x19 = x1*x8
    x20 = a_0*x1 + a_1*x19
    x21 = a_2*(-x1*x18 - x19*x4) + d_3*x11 - x20
    x22 = x3*x8
    x23 = a_0*x3 + a_1*x22
    x24 = a_2*(x18*x3 + x22*x4) - d_3*x9 + x23
    x25 = 1.0*a_0
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 4))
    jacobian_output[0, 1] = x1
    jacobian_output[0, 2] = -x1
    jacobian_output[0, 3] = x10
    jacobian_output[1, 1] = x3
    jacobian_output[1, 2] = -x3
    jacobian_output[1, 3] = x11
    jacobian_output[2, 0] = -1.00000000000000
    jacobian_output[2, 3] = x14
    jacobian_output[3, 1] = -pre_transform_special_symbol_23*x3
    jacobian_output[3, 2] = x16*x3
    jacobian_output[3, 3] = -x11*x17 + x14*x21
    jacobian_output[4, 1] = pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = -x1*x16
    jacobian_output[4, 3] = x10*x17 - x14*x24
    jacobian_output[5, 1] = x0**2*x25 + x2**2*x25
    jacobian_output[5, 2] = -x1*x20 - x23*x3
    jacobian_output[5, 3] = -x10*x21 + x11*x24
    return jacobian_output


def kawasaki_CP130L_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw

    # Temp variable for efficiency
    x0 = 1.0*math.sin(th_0)
    x1 = math.cos(th_0)
    x2 = 1.0*x1
    x3 = math.cos(th_2)
    x4 = math.sin(th_1)
    x5 = x3*x4
    x6 = math.sin(th_2)
    x7 = math.cos(th_1)
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 4))
    jacobian_output[0, 1] = x0
    jacobian_output[0, 2] = -x0
    jacobian_output[0, 3] = 1.0*x1*x6*x7 - x2*x5
    jacobian_output[1, 1] = x2
    jacobian_output[1, 2] = -x2
    jacobian_output[1, 3] = x0*x5 - x0*x6*x7
    jacobian_output[2, 0] = -1.00000000000000
    jacobian_output[2, 3] = -1.0*x3*x7 - 1.0*x4*x6
    return jacobian_output


def kawasaki_CP130L_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
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
    x6 = a_1*x5 - pre_transform_special_symbol_23
    x7 = -x6
    x8 = math.sin(th_2)
    x9 = x5*x8
    x10 = math.cos(th_2)
    x11 = math.cos(th_1)
    x12 = 1.0*x11
    x13 = x10*x12
    x14 = x13 + x9
    x15 = -x14
    x16 = math.sin(th_0)
    x17 = x10*x5
    x18 = -x12*x16*x8 + x16*x17
    x19 = a_2*(1.0*x11*x8 - x17) - d_3*x14 - x6
    x20 = 1.0*x16
    x21 = a_0*x20 + a_1*x12*x16
    x22 = a_2*(-x13*x16 - x16*x9) + d_3*x18 - x21
    x23 = 1.0*p_on_ee_x
    x24 = p_on_ee_z*x20
    x25 = x2*x4
    x26 = -1.0*x1*x11*x8 + x10*x25
    x27 = -x26
    x28 = x11*x2
    x29 = a_0*x2 + a_1*x28
    x30 = a_2*(x10*x28 + x25*x8) - d_3*x26 + x29
    x31 = x1*x23
    x32 = x0*x16
    x33 = 1.0*a_0
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 4))
    jacobian_output[0, 0] = x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = 1.0*x1*x7 - x3
    jacobian_output[0, 3] = -p_on_ee_y*x15 + p_on_ee_z*x18 + x15*x22 - x18*x19
    jacobian_output[1, 0] = -x23
    jacobian_output[1, 1] = 1.0*pre_transform_special_symbol_23*x16 - x24
    jacobian_output[1, 2] = -x20*x7 + x24
    jacobian_output[1, 3] = p_on_ee_x*x15 - p_on_ee_z*x27 - x15*x30 + x19*x27
    jacobian_output[2, 1] = x1**2*x33 + x16**2*x33 - x31 + x32
    jacobian_output[2, 2] = -x2*x29 - x20*x21 + x31 - x32
    jacobian_output[2, 3] = -p_on_ee_x*x18 + p_on_ee_y*x27 + x18*x30 - x22*x27
    return jacobian_output


def kawasaki_CP130L_ik_solve_raw(T_ee: np.ndarray):
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
        for tmp_sol_idx in range(5):
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
    for i in range(16):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True
    
    # Code for non-branch dispatcher node 0
    # Actually, there is no code
    
    # Code for explicit solution node 1, solved variable is negative_th_2_positive_th_1__soa
    def ExplicitSolutionNode_node_1_solve_negative_th_2_positive_th_1__soa_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return
        
        # The explicit solution of root node
        condition_0: bool = (abs(r_33) <= 1)
        if condition_0:
            # Temp variable for efficiency
            x0 = safe_acos(r_33)
            # End of temp variables
            solution_0: IkSolution = make_ik_solution()
            solution_0[0] = x0
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)
            
        condition_1: bool = (abs(r_33) <= 1)
        if condition_1:
            # Temp variable for efficiency
            x0 = safe_acos(r_33)
            # End of temp variables
            solution_1: IkSolution = make_ik_solution()
            solution_1[0] = -x0
            appended_idx = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx)
            
    # Invoke the processor
    ExplicitSolutionNode_node_1_solve_negative_th_2_positive_th_1__soa_processor()
    # Finish code for explicit solution node 0
    
    # Code for non-branch dispatcher node 2
    # Actually, there is no code
    
    # Code for explicit solution node 3, solved variable is th_1
    def ExplicitSolutionNode_node_3_solve_th_1_processor():
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
            negative_th_2_positive_th_1__soa = this_solution[0]
            condition_0: bool = (abs((-Pz + a_2*math.sin(negative_th_2_positive_th_1__soa) + d_3*math.cos(negative_th_2_positive_th_1__soa))/a_1) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = safe_asin((Pz - a_2*math.sin(negative_th_2_positive_th_1__soa) - d_3*math.cos(negative_th_2_positive_th_1__soa))/a_1)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs((-Pz + a_2*math.sin(negative_th_2_positive_th_1__soa) + d_3*math.cos(negative_th_2_positive_th_1__soa))/a_1) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = safe_asin((Pz - a_2*math.sin(negative_th_2_positive_th_1__soa) - d_3*math.cos(negative_th_2_positive_th_1__soa))/a_1)
                # End of temp variables
                this_solution[2] = math.pi - x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_1_processor()
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
            negative_th_2_positive_th_1__soa = this_solution[0]
            th_1 = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[3] = -negative_th_2_positive_th_1__soa + th_1
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
            negative_th_2_positive_th_1__soa = this_solution[0]
            degenerate_valid_0 = (abs(negative_th_2_positive_th_1__soa) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(10, node_input_i_idx_in_queue)
            
            negative_th_2_positive_th_1__soa = this_solution[0]
            degenerate_valid_1 = (abs(negative_th_2_positive_th_1__soa - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(13, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(7, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_6_processor()
    # Finish code for solved_variable dispatcher node 6
    
    # Code for explicit solution node 13, solved variable is th_0
    def ExplicitSolutionNode_node_13_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[13]
        this_input_valid: bool = node_input_validity[13]
        if not this_input_valid:
            return
        
        # The solution of non-root node 13
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_2 = this_solution[3]
            condition_0: bool = (abs(r_13) >= 1.0e-6) or (abs(r_23) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_1 - th_2)
                # End of temp variables
                this_solution[1] = math.atan2(-r_23*x0, -r_13*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(14, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_0_processor()
    # Finish code for explicit solution node 13
    
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
            th_0 = this_solution[1]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                # End of temp variables
                this_solution[4] = math.atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_15_solve_th_3_processor()
    # Finish code for explicit solution node 14
    
    # Code for explicit solution node 10, solved variable is th_0
    def ExplicitSolutionNode_node_10_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[10]
        this_input_valid: bool = node_input_validity[10]
        if not this_input_valid:
            return
        
        # The solution of non-root node 10
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_2 = this_solution[3]
            condition_0: bool = (abs(r_13) >= 1.0e-6) or (abs(r_23) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_1 - th_2)
                # End of temp variables
                this_solution[1] = math.atan2(-r_23*x0, -r_13*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(11, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_10_solve_th_0_processor()
    # Finish code for explicit solution node 10
    
    # Code for non-branch dispatcher node 11
    # Actually, there is no code
    
    # Code for explicit solution node 12, solved variable is th_3
    def ExplicitSolutionNode_node_12_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[11]
        this_input_valid: bool = node_input_validity[11]
        if not this_input_valid:
            return
        
        # The solution of non-root node 12
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
                this_solution[4] = math.atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_12_solve_th_3_processor()
    # Finish code for explicit solution node 11
    
    # Code for explicit solution node 7, solved variable is th_3
    def ExplicitSolutionNode_node_7_solve_th_3_processor():
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
            negative_th_2_positive_th_1__soa = this_solution[0]
            condition_0: bool = (abs(r_31) >= zero_tolerance) or (abs(r_32) >= zero_tolerance) or (abs(math.sin(negative_th_2_positive_th_1__soa)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(negative_th_2_positive_th_1__soa)**(-1)
                # End of temp variables
                this_solution[4] = math.atan2(-r_32*x0, r_31*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_3_processor()
    # Finish code for explicit solution node 7
    
    # Code for non-branch dispatcher node 8
    # Actually, there is no code
    
    # Code for explicit solution node 9, solved variable is th_0
    def ExplicitSolutionNode_node_9_solve_th_0_processor():
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
            negative_th_2_positive_th_1__soa = this_solution[0]
            condition_0: bool = (abs(r_13) >= zero_tolerance) or (abs(r_23) >= zero_tolerance) or (abs(math.sin(negative_th_2_positive_th_1__soa)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(negative_th_2_positive_th_1__soa)**(-1)
                # End of temp variables
                this_solution[1] = math.atan2(-r_23*x0, -r_13*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_0_processor()
    # Finish code for explicit solution node 8
    
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
        value_at_2 = ik_out_i[3]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[4]  # th_3
        new_ik_i[3] = value_at_3
        ik_out.append(new_ik_i)
    return ik_out


def kawasaki_CP130L_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = kawasaki_CP130L_ik_target_original_to_raw(T_ee)
    ik_output_raw = kawasaki_CP130L_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ee_pose_i = kawasaki_CP130L_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_kawasaki_CP130L():
    theta_in = np.random.random(size=(4, ))
    ee_pose = kawasaki_CP130L_fk(theta_in)
    ik_output = kawasaki_CP130L_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = kawasaki_CP130L_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_kawasaki_CP130L()
