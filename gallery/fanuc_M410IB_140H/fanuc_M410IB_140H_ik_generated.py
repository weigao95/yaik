import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 5
n_tree_nodes: int = 12
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 0.24
a_1: float = 1.16
a_2: float = 1.7364907140552177
d_3: float = -0.215
pre_transform_special_symbol_23: float = 0.7200000000000002

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -1.5707963267948966
th_2_offset_original2raw: float = -1.484307426876522
th_3_offset_original2raw: float = -0.08648889991837498
th_4_offset_original2raw: float = 3.141592653589793


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def fanuc_M410IB_140H_ik_target_original_to_raw(T_ee: np.ndarray):
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


def fanuc_M410IB_140H_ik_target_raw_to_original(T_ee: np.ndarray):
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


def fanuc_M410IB_140H_fk(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_3)
    x1 = math.cos(th_0)
    x2 = math.sin(th_1)
    x3 = math.sin(th_2)
    x4 = x2*x3
    x5 = math.cos(th_1)
    x6 = math.cos(th_2)
    x7 = x5*x6
    x8 = x1*x4 + x1*x7
    x9 = x0*x8
    x10 = math.cos(th_3)
    x11 = x2*x6
    x12 = x3*x5
    x13 = x1*x11 - x1*x12
    x14 = math.cos(th_4)
    x15 = math.sin(th_0)
    x16 = 1.0*x15
    x17 = math.sin(th_4)
    x18 = 1.0*x0*x13 + 1.0*x10*x8
    x19 = 1.0*x1
    x20 = a_1*x5
    x21 = 1.0*a_2
    x22 = 1.0*d_3
    x23 = x15*x4 + x15*x7
    x24 = x0*x23
    x25 = x11*x15 - x12*x15
    x26 = 1.0*x0*x25 + 1.0*x10*x23
    x27 = x4 + x7
    x28 = x10*x27
    x29 = -x11 + x3*x5
    x30 = 1.0*x0*x27 + 1.0*x10*x29
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -1.0*x10*x13 + 1.0*x9
    ee_pose[0, 1] = -x14*x16 - x17*x18
    ee_pose[0, 2] = x14*x18 - x16*x17
    ee_pose[0, 3] = a_0*x19 + x19*x20 + x21*x8 + x22*(x10*x13 - x9)
    ee_pose[1, 0] = -1.0*x10*x25 + 1.0*x24
    ee_pose[1, 1] = 1.0*x1*x14 - x17*x26
    ee_pose[1, 2] = x14*x26 + x17*x19
    ee_pose[1, 3] = a_0*x16 + x16*x20 + x21*x23 + x22*(x10*x25 - x24)
    ee_pose[2, 0] = 1.0*x0*x29 - 1.0*x28
    ee_pose[2, 1] = -x17*x30
    ee_pose[2, 2] = x14*x30
    ee_pose[2, 3] = -1.0*a_1*x2 + 1.0*pre_transform_special_symbol_23 + x21*x29 + x22*(-x0*x29 + x28)
    return ee_pose


def fanuc_M410IB_140H_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = 1.0*x0
    x2 = math.sin(th_3)
    x3 = math.sin(th_1)
    x4 = math.sin(th_2)
    x5 = math.cos(th_0)
    x6 = 1.0*x5
    x7 = x4*x6
    x8 = math.cos(th_1)
    x9 = math.cos(th_2)
    x10 = x6*x9
    x11 = x10*x8 + x3*x7
    x12 = math.cos(th_3)
    x13 = -x11*x2 + x12*(x10*x3 - x7*x8)
    x14 = -x6
    x15 = x1*x4
    x16 = x1*x9
    x17 = x15*x3 + x16*x8
    x18 = x12*(-x15*x8 + x16*x3) - x17*x2
    x19 = 1.0*x3
    x20 = -x19*x9 + 1.0*x4*x8
    x21 = x12*(x19*x4 + 1.0*x8*x9) - x2*x20
    x22 = a_1*x19
    x23 = pre_transform_special_symbol_23 - x22
    x24 = a_2*x20 + pre_transform_special_symbol_23 - x22
    x25 = d_3*x21 + x24
    x26 = a_1*x8
    x27 = a_0*x1 + x1*x26
    x28 = a_2*x17 + x27
    x29 = d_3*x18 + x28
    x30 = a_0*x6 + x26*x6
    x31 = a_2*x11 + x30
    x32 = d_3*x13 + x31
    x33 = 1.0*a_0
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 5))
    jacobian_output[0, 1] = -x1
    jacobian_output[0, 2] = x1
    jacobian_output[0, 3] = x1
    jacobian_output[0, 4] = x13
    jacobian_output[1, 1] = x6
    jacobian_output[1, 2] = x14
    jacobian_output[1, 3] = x14
    jacobian_output[1, 4] = x18
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 4] = x21
    jacobian_output[3, 1] = -pre_transform_special_symbol_23*x6
    jacobian_output[3, 2] = x23*x6
    jacobian_output[3, 3] = x24*x6
    jacobian_output[3, 4] = -x18*x25 + x21*x29
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = x1*x23
    jacobian_output[4, 3] = x1*x24
    jacobian_output[4, 4] = x13*x25 - x21*x32
    jacobian_output[5, 1] = x0**2*x33 + x33*x5**2
    jacobian_output[5, 2] = -x1*x27 - x30*x6
    jacobian_output[5, 3] = -x1*x28 - x31*x6
    jacobian_output[5, 4] = -x13*x29 + x18*x32
    return jacobian_output


def fanuc_M410IB_140H_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw

    # Temp variable for efficiency
    x0 = 1.0*math.sin(th_0)
    x1 = math.sin(th_3)
    x2 = math.sin(th_1)
    x3 = math.sin(th_2)
    x4 = 1.0*math.cos(th_0)
    x5 = x3*x4
    x6 = math.cos(th_1)
    x7 = math.cos(th_2)
    x8 = x4*x7
    x9 = math.cos(th_3)
    x10 = -x4
    x11 = x0*x3
    x12 = x0*x7
    x13 = 1.0*x2
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 5))
    jacobian_output[0, 1] = -x0
    jacobian_output[0, 2] = x0
    jacobian_output[0, 3] = x0
    jacobian_output[0, 4] = -x1*(x2*x5 + x6*x8) + x9*(x2*x8 - x5*x6)
    jacobian_output[1, 1] = x4
    jacobian_output[1, 2] = x10
    jacobian_output[1, 3] = x10
    jacobian_output[1, 4] = -x1*(x11*x2 + x12*x6) + x9*(-x11*x6 + x12*x2)
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 4] = -x1*(-x13*x7 + 1.0*x3*x6) + x9*(x13*x3 + 1.0*x6*x7)
    return jacobian_output


def fanuc_M410IB_140H_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
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
    x8 = math.cos(th_2)
    x9 = x5*x8
    x10 = math.sin(th_2)
    x11 = math.cos(th_1)
    x12 = 1.0*x10*x11 - x9
    x13 = a_2*x12 + pre_transform_special_symbol_23 - x6
    x14 = math.cos(th_3)
    x15 = x10*x5
    x16 = 1.0*x11
    x17 = x16*x8
    x18 = math.sin(th_3)
    x19 = -x12*x18 + x14*(x15 + x17)
    x20 = math.sin(th_0)
    x21 = x15*x20 + x17*x20
    x22 = x14*(-x10*x16*x20 + x20*x9) - x18*x21
    x23 = d_3*x19 + x13
    x24 = 1.0*x20
    x25 = a_0*x24 + a_1*x16*x20
    x26 = a_2*x21 + x25
    x27 = d_3*x22 + x26
    x28 = 1.0*p_on_ee_x
    x29 = p_on_ee_z*x24
    x30 = x10*x2
    x31 = x2*x8
    x32 = x11*x31 + x30*x4
    x33 = x14*(-x11*x30 + x31*x4) - x18*x32
    x34 = a_0*x2 + a_1*x11*x2
    x35 = a_2*x32 + x34
    x36 = d_3*x33 + x35
    x37 = x1*x28
    x38 = x0*x20
    x39 = 1.0*a_0
    x40 = x37 + x38
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 5))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = 1.0*x1*x7 - x3
    jacobian_output[0, 3] = 1.0*x1*x13 - x3
    jacobian_output[0, 4] = -p_on_ee_y*x19 + p_on_ee_z*x22 + x19*x27 - x22*x23
    jacobian_output[1, 0] = x28
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x24 + x29
    jacobian_output[1, 2] = 1.0*x20*x7 - x29
    jacobian_output[1, 3] = 1.0*x13*x20 - x29
    jacobian_output[1, 4] = p_on_ee_x*x19 - p_on_ee_z*x33 - x19*x36 + x23*x33
    jacobian_output[2, 1] = x1**2*x39 + x20**2*x39 - x37 - x38
    jacobian_output[2, 2] = -x2*x34 - x24*x25 + x40
    jacobian_output[2, 3] = -x2*x35 - x24*x26 + x40
    jacobian_output[2, 4] = -p_on_ee_x*x22 + p_on_ee_y*x33 + x22*x36 - x27*x33
    return jacobian_output


def fanuc_M410IB_140H_ik_solve_raw(T_ee: np.ndarray):
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
    for i in range(12):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True
    
    # Code for equation all-zero dispatcher node 0
    def EquationAllZeroDispatcherNode_node_0_processor():
        checked_result: bool = (abs(Px - d_3*r_13) <= 1.0e-6) and (abs(Py - d_3*r_23) <= 1.0e-6)
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
        condition_0: bool = (abs(Px - d_3*r_13) >= zero_tolerance) or (abs(Py - d_3*r_23) >= zero_tolerance)
        if condition_0:
            # Temp variable for efficiency
            x0 = math.atan2(Py - d_3*r_23, Px - d_3*r_13)
            # End of temp variables
            solution_0: IkSolution = make_ik_solution()
            solution_0[1] = x0
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)
            
        condition_1: bool = (abs(Px - d_3*r_13) >= zero_tolerance) or (abs(Py - d_3*r_23) >= zero_tolerance)
        if condition_1:
            # Temp variable for efficiency
            x0 = math.atan2(Py - d_3*r_23, Px - d_3*r_13)
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
            condition_0: bool = ((1/2)*abs((Px**2 - 2*Px*a_0*math.cos(th_0) - 2*Px*d_3*r_13 + Py**2 - 2*Py*a_0*math.sin(th_0) - 2*Py*d_3*r_23 + Pz**2 - 2*Pz*d_3*r_33 + a_0**2 + 2*a_0*d_3*r_13*math.cos(th_0) + 2*a_0*d_3*r_23*math.sin(th_0) - a_1**2 - a_2**2 + d_3**2*r_13**2 + d_3**2*r_23**2 + d_3**2*r_33**2)/(a_1*a_2)) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*Px
                x1 = d_3*r_13
                x2 = 2*Py
                x3 = d_3*r_23
                x4 = a_0*math.cos(th_0)
                x5 = a_0*math.sin(th_0)
                x6 = d_3**2
                x7 = safe_acos((1/2)*(Px**2 + Py**2 + Pz**2 - 2*Pz*d_3*r_33 + a_0**2 - a_1**2 - a_2**2 + r_13**2*x6 + r_23**2*x6 + r_33**2*x6 - x0*x1 - x0*x4 + 2*x1*x4 - x2*x3 - x2*x5 + 2*x3*x5)/(a_1*a_2))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = x7
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = ((1/2)*abs((Px**2 - 2*Px*a_0*math.cos(th_0) - 2*Px*d_3*r_13 + Py**2 - 2*Py*a_0*math.sin(th_0) - 2*Py*d_3*r_23 + Pz**2 - 2*Pz*d_3*r_33 + a_0**2 + 2*a_0*d_3*r_13*math.cos(th_0) + 2*a_0*d_3*r_23*math.sin(th_0) - a_1**2 - a_2**2 + d_3**2*r_13**2 + d_3**2*r_23**2 + d_3**2*r_33**2)/(a_1*a_2)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*Px
                x1 = d_3*r_13
                x2 = 2*Py
                x3 = d_3*r_23
                x4 = a_0*math.cos(th_0)
                x5 = a_0*math.sin(th_0)
                x6 = d_3**2
                x7 = safe_acos((1/2)*(Px**2 + Py**2 + Pz**2 - 2*Pz*d_3*r_33 + a_0**2 - a_1**2 - a_2**2 + r_13**2*x6 + r_23**2*x6 + r_33**2*x6 - x0*x1 - x0*x4 + 2*x1*x4 - x2*x3 - x2*x5 + 2*x3*x5)/(a_1*a_2))
                # End of temp variables
                this_solution[3] = -x7
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
            checked_result: bool = (abs(Pz - d_3*r_33) <= 1.0e-6) and (abs(-Px*math.cos(th_0) - Py*math.sin(th_0) + a_0 + d_3*r_13*math.cos(th_0) + d_3*r_23*math.sin(th_0)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor()
    # Finish code for equation all-zero dispatcher node 4
    
    # Code for explicit solution node 5, solved variable is th_1
    def ExplicitSolutionNode_node_5_solve_th_1_processor():
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
            th_2 = this_solution[3]
            condition_0: bool = (abs(Pz - d_3*r_33) >= 1.0e-6) or (abs(-Px*math.cos(th_0) - Py*math.sin(th_0) + a_0 + d_3*r_13*math.cos(th_0) + d_3*r_23*math.sin(th_0)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = Pz - d_3*r_33
                x1 = -a_1 - a_2*math.cos(th_2)
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = -Px*x2 - Py*x3 + a_0 + d_3*r_13*x2 + d_3*r_23*x3
                x5 = a_2*math.sin(th_2)
                # End of temp variables
                this_solution[2] = math.atan2(x0*x1 - x4*x5, x0*x5 + x1*x4)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_1_processor()
    # Finish code for explicit solution node 5
    
    # Code for non-branch dispatcher node 6
    # Actually, there is no code
    
    # Code for explicit solution node 7, solved variable is negative_th_2_positive_th_1__soa
    def ExplicitSolutionNode_node_7_solve_negative_th_2_positive_th_1__soa_processor():
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
            th_1 = this_solution[2]
            th_2 = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[0] = th_1 - th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_negative_th_2_positive_th_1__soa_processor()
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
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                # End of temp variables
                this_solution[5] = math.atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_4_processor()
    # Finish code for explicit solution node 8
    
    # Code for non-branch dispatcher node 10
    # Actually, there is no code
    
    # Code for explicit solution node 11, solved variable is th_3
    def ExplicitSolutionNode_node_11_solve_th_3_processor():
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
            th_0 = this_solution[1]
            th_1 = this_solution[2]
            th_2 = this_solution[3]
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
    ExplicitSolutionNode_node_11_solve_th_3_processor()
    # Finish code for explicit solution node 10
    
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
        value_at_4 = ik_out_i[5]  # th_4
        new_ik_i[4] = value_at_4
        ik_out.append(new_ik_i)
    return ik_out


def fanuc_M410IB_140H_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = fanuc_M410IB_140H_ik_target_original_to_raw(T_ee)
    ik_output_raw = fanuc_M410IB_140H_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ee_pose_i = fanuc_M410IB_140H_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_fanuc_M410IB_140H():
    theta_in = np.random.random(size=(5, ))
    ee_pose = fanuc_M410IB_140H_fk(theta_in)
    ik_output = fanuc_M410IB_140H_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = fanuc_M410IB_140H_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_fanuc_M410IB_140H()
