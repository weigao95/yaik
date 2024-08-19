import numpy as np
import copy
import math
from typing import List, NewType

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 24
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_2: float = 0.432
a_3: float = 0.0203
d_1: float = 0.6
d_3: float = 0.1245
d_4: float = 0.432

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def puma_ik_target_original_to_raw(T_ee: np.ndarray):
    return T_ee


def puma_ik_target_raw_to_original(T_ee: np.ndarray):
    return T_ee


def puma_fk(theta_input: np.ndarray):
    th_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]
    th_5 = theta_input[5]

    # Temp variable for efficiency
    x0 = math.sin(th_5)
    x1 = math.sin(th_0)
    x2 = math.cos(th_3)
    x3 = math.sin(th_3)
    x4 = math.cos(th_0)
    x5 = math.cos(th_1)
    x6 = math.cos(th_2)
    x7 = x5*x6
    x8 = math.sin(th_1)
    x9 = math.sin(th_2)
    x10 = x8*x9
    x11 = -x10*x4 + x4*x7
    x12 = x1*x2 - x11*x3
    x13 = math.cos(th_5)
    x14 = math.sin(th_4)
    x15 = x6*x8
    x16 = x5*x9
    x17 = -x15*x4 - x16*x4
    x18 = math.cos(th_4)
    x19 = x1*x3 + x11*x2
    x20 = -x14*x17 + x18*x19
    x21 = a_2*x5
    x22 = -x1*x10 + x1*x7
    x23 = -x2*x4 - x22*x3
    x24 = -x1*x15 - x1*x16
    x25 = x2*x22 - x3*x4
    x26 = -x14*x24 + x18*x25
    x27 = -x15 - x16
    x28 = x27*x3
    x29 = x10 - x7
    x30 = x2*x27
    x31 = -x14*x29 + x18*x30
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = x0*x12 + x13*x20
    ee_pose[0, 1] = -x0*x20 + x12*x13
    ee_pose[0, 2] = x14*x19 + x17*x18
    ee_pose[0, 3] = a_3*x11 - d_3*x1 + d_4*x17 + x21*x4
    ee_pose[1, 0] = x0*x23 + x13*x26
    ee_pose[1, 1] = -x0*x26 + x13*x23
    ee_pose[1, 2] = x14*x25 + x18*x24
    ee_pose[1, 3] = a_3*x22 + d_3*x4 + d_4*x24 + x1*x21
    ee_pose[2, 0] = -x0*x28 + x13*x31
    ee_pose[2, 1] = -x0*x31 - x13*x28
    ee_pose[2, 2] = x14*x30 + x18*x29
    ee_pose[2, 3] = -a_2*x8 + a_3*x27 + d_1 + d_4*x29
    return ee_pose


def puma_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]
    th_5 = theta_input[5]

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = -x0
    x2 = math.cos(th_0)
    x3 = math.sin(th_1)
    x4 = math.cos(th_2)
    x5 = x3*x4
    x6 = math.sin(th_2)
    x7 = math.cos(th_1)
    x8 = x6*x7
    x9 = -x2*x5 - x2*x8
    x10 = math.cos(th_3)
    x11 = math.sin(th_3)
    x12 = x4*x7
    x13 = x3*x6
    x14 = x12*x2 - x13*x2
    x15 = x0*x10 - x11*x14
    x16 = math.cos(th_4)
    x17 = math.sin(th_4)
    x18 = x16*x9 + x17*(x0*x11 + x10*x14)
    x19 = -x0*x5 - x0*x8
    x20 = x0*x12 - x0*x13
    x21 = -x10*x2 - x11*x20
    x22 = x16*x19 + x17*(x10*x20 - x11*x2)
    x23 = -x12 + x13
    x24 = -x5 - x8
    x25 = x11*x24
    x26 = x10*x17*x24 + x16*x23
    x27 = -a_2*x3 + d_1
    x28 = a_3*x24 + d_4*x23 + x27
    x29 = a_2*x7
    x30 = d_3*x2 + x0*x29
    x31 = a_3*x20 + d_4*x19 + x30
    x32 = -d_3*x0 + x2*x29
    x33 = a_3*x14 + d_4*x9 + x32
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = x1
    jacobian_output[0, 2] = x1
    jacobian_output[0, 3] = x9
    jacobian_output[0, 4] = x15
    jacobian_output[0, 5] = x18
    jacobian_output[1, 1] = x2
    jacobian_output[1, 2] = x2
    jacobian_output[1, 3] = x19
    jacobian_output[1, 4] = x21
    jacobian_output[1, 5] = x22
    jacobian_output[2, 0] = 1
    jacobian_output[2, 3] = x23
    jacobian_output[2, 4] = -x25
    jacobian_output[2, 5] = x26
    jacobian_output[3, 1] = -d_1*x2
    jacobian_output[3, 2] = -x2*x27
    jacobian_output[3, 3] = -x19*x28 + x23*x31
    jacobian_output[3, 4] = -x21*x28 - x25*x31
    jacobian_output[3, 5] = -x22*x28 + x26*x31
    jacobian_output[4, 1] = -d_1*x0
    jacobian_output[4, 2] = -x0*x27
    jacobian_output[4, 3] = -x23*x33 + x28*x9
    jacobian_output[4, 4] = x15*x28 + x25*x33
    jacobian_output[4, 5] = x18*x28 - x26*x33
    jacobian_output[5, 2] = x0*x30 + x2*x32
    jacobian_output[5, 3] = x19*x33 - x31*x9
    jacobian_output[5, 4] = -x15*x31 + x21*x33
    jacobian_output[5, 5] = -x18*x31 + x22*x33
    return jacobian_output


def puma_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]
    th_5 = theta_input[5]

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = -x0
    x2 = math.cos(th_0)
    x3 = math.sin(th_1)
    x4 = math.cos(th_2)
    x5 = x3*x4
    x6 = math.sin(th_2)
    x7 = math.cos(th_1)
    x8 = x6*x7
    x9 = -x2*x5 - x2*x8
    x10 = math.cos(th_3)
    x11 = math.sin(th_3)
    x12 = x4*x7
    x13 = x3*x6
    x14 = x12*x2 - x13*x2
    x15 = math.cos(th_4)
    x16 = math.sin(th_4)
    x17 = -x0*x5 - x0*x8
    x18 = x0*x12 - x0*x13
    x19 = -x12 + x13
    x20 = -x5 - x8
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 1] = x1
    jacobian_output[0, 2] = x1
    jacobian_output[0, 3] = x9
    jacobian_output[0, 4] = x0*x10 - x11*x14
    jacobian_output[0, 5] = x15*x9 + x16*(x0*x11 + x10*x14)
    jacobian_output[1, 1] = x2
    jacobian_output[1, 2] = x2
    jacobian_output[1, 3] = x17
    jacobian_output[1, 4] = -x10*x2 - x11*x18
    jacobian_output[1, 5] = x15*x17 + x16*(x10*x18 - x11*x2)
    jacobian_output[2, 0] = 1
    jacobian_output[2, 3] = x19
    jacobian_output[2, 4] = -x11*x20
    jacobian_output[2, 5] = x10*x16*x20 + x15*x19
    return jacobian_output


def puma_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    th_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]
    th_5 = theta_input[5]
    p_on_ee_x: float = point_on_ee[0]
    p_on_ee_y: float = point_on_ee[1]
    p_on_ee_z: float = point_on_ee[2]

    # Temp variable for efficiency
    x0 = math.cos(th_0)
    x1 = p_on_ee_z*x0
    x2 = math.sin(th_1)
    x3 = -a_2*x2 + d_1
    x4 = math.sin(th_2)
    x5 = x2*x4
    x6 = math.cos(th_1)
    x7 = math.cos(th_2)
    x8 = x6*x7
    x9 = x5 - x8
    x10 = math.sin(th_0)
    x11 = x2*x7
    x12 = x4*x6
    x13 = -x10*x11 - x10*x12
    x14 = -x11 - x12
    x15 = a_3*x14 + d_4*x9 + x3
    x16 = -x10*x5 + x10*x8
    x17 = a_2*x6
    x18 = d_3*x0 + x10*x17
    x19 = a_3*x16 + d_4*x13 + x18
    x20 = math.sin(th_3)
    x21 = x14*x20
    x22 = math.cos(th_3)
    x23 = -x0*x22 - x16*x20
    x24 = math.cos(th_4)
    x25 = math.sin(th_4)
    x26 = x14*x22*x25 + x24*x9
    x27 = x13*x24 + x25*(-x0*x20 + x16*x22)
    x28 = p_on_ee_z*x10
    x29 = -x0*x11 - x0*x12
    x30 = -x0*x5 + x0*x8
    x31 = -d_3*x10 + x0*x17
    x32 = a_3*x30 + d_4*x29 + x31
    x33 = x10*x22 - x20*x30
    x34 = x24*x29 + x25*(x10*x20 + x22*x30)
    x35 = -p_on_ee_x*x0 - p_on_ee_y*x10
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -p_on_ee_y
    jacobian_output[0, 1] = -d_1*x0 + x1
    jacobian_output[0, 2] = -x0*x3 + x1
    jacobian_output[0, 3] = -p_on_ee_y*x9 + p_on_ee_z*x13 - x13*x15 + x19*x9
    jacobian_output[0, 4] = p_on_ee_y*x21 + p_on_ee_z*x23 - x15*x23 - x19*x21
    jacobian_output[0, 5] = -p_on_ee_y*x26 + p_on_ee_z*x27 - x15*x27 + x19*x26
    jacobian_output[1, 0] = p_on_ee_x
    jacobian_output[1, 1] = -d_1*x10 + x28
    jacobian_output[1, 2] = -x10*x3 + x28
    jacobian_output[1, 3] = p_on_ee_x*x9 - p_on_ee_z*x29 + x15*x29 - x32*x9
    jacobian_output[1, 4] = -p_on_ee_x*x21 - p_on_ee_z*x33 + x15*x33 + x21*x32
    jacobian_output[1, 5] = p_on_ee_x*x26 - p_on_ee_z*x34 + x15*x34 - x26*x32
    jacobian_output[2, 1] = x35
    jacobian_output[2, 2] = x0*x31 + x10*x18 + x35
    jacobian_output[2, 3] = -p_on_ee_x*x13 + p_on_ee_y*x29 + x13*x32 - x19*x29
    jacobian_output[2, 4] = -p_on_ee_x*x23 + p_on_ee_y*x33 - x19*x33 + x23*x32
    jacobian_output[2, 5] = -p_on_ee_x*x27 + p_on_ee_y*x34 - x19*x34 + x27*x32
    return jacobian_output


def puma_ik_solve_raw(T_ee: np.ndarray):
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
    
    # Code for non-branch dispatcher node 0
    # Actually, there is no code
    
    # Code for explicit solution node 1, solved variable is th_0
    def ExplicitSolutionNode_node_1_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return
        
        # The explicit solution of root node
        condition_0: bool = (abs(Px) >= zero_tolerance) or (abs(Py) >= zero_tolerance) or (abs(d_3) >= zero_tolerance)
        if condition_0:
            # Temp variable for efficiency
            x0 = math.atan2(Px, -Py)
            x1 = math.sqrt(Px**2 + Py**2 - d_3**2)
            x2 = -d_3
            # End of temp variables
            solution_0: IkSolution = make_ik_solution()
            solution_0[1] = x0 + math.atan2(x1, x2)
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)
            
        condition_1: bool = (abs(Px) >= zero_tolerance) or (abs(Py) >= zero_tolerance) or (abs(d_3) >= zero_tolerance)
        if condition_1:
            # Temp variable for efficiency
            x0 = math.atan2(Px, -Py)
            x1 = math.sqrt(Px**2 + Py**2 - d_3**2)
            x2 = -d_3
            # End of temp variables
            solution_1: IkSolution = make_ik_solution()
            solution_1[1] = x0 + math.atan2(-x1, x2)
            appended_idx = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx)
            
    # Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_0_processor()
    # Finish code for explicit solution node 0
    
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
            condition_0: bool = (2*abs(a_2*a_3) >= zero_tolerance) or (2*abs(a_2*d_4) >= zero_tolerance) or (abs(-Px**2 - Py**2 - Pz**2 + 2*Pz*d_1 + a_2**2 + a_3**2 - d_1**2 + d_3**2 + d_4**2) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*a_2
                x1 = math.atan2(-d_4*x0, a_3*x0)
                x2 = a_3**2
                x3 = a_2**2
                x4 = 4*x3
                x5 = d_4**2
                x6 = Px**2 + Py**2 + Pz**2 - 2*Pz*d_1 + d_1**2 - d_3**2 - x2 - x3 - x5
                x7 = math.sqrt(x2*x4 + x4*x5 - x6**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x1 + math.atan2(x7, x6)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (2*abs(a_2*a_3) >= zero_tolerance) or (2*abs(a_2*d_4) >= zero_tolerance) or (abs(-Px**2 - Py**2 - Pz**2 + 2*Pz*d_1 + a_2**2 + a_3**2 - d_1**2 + d_3**2 + d_4**2) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*a_2
                x1 = math.atan2(-d_4*x0, a_3*x0)
                x2 = a_3**2
                x3 = a_2**2
                x4 = 4*x3
                x5 = d_4**2
                x6 = Px**2 + Py**2 + Pz**2 - 2*Pz*d_1 + d_1**2 - d_3**2 - x2 - x3 - x5
                x7 = math.sqrt(x2*x4 + x4*x5 - x6**2)
                # End of temp variables
                this_solution[4] = x1 + math.atan2(-x7, x6)
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
            checked_result: bool = (abs(Pz - d_1) <= 1.0e-6) and (abs(Px*math.cos(th_0) + Py*math.sin(th_0)) <= 1.0e-6)
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
            condition_0: bool = (abs(Pz - d_1) >= 1.0e-6) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = Pz - d_1
                x1 = -a_2*math.cos(th_2) - a_3
                x2 = a_2*math.sin(th_2) - d_4
                x3 = -Px*math.cos(th_0) - Py*math.sin(th_0)
                # End of temp variables
                this_solution[3] = math.atan2(x0*x1 - x2*x3, x0*x2 + x1*x3)
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
            condition_0: bool = (abs(-r_13*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.cos(th_0) - r_23*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_0) - r_33*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.cos(th_2)
                x2 = math.sin(th_1)
                x3 = math.sin(th_2)
                x4 = x0*x3 + x1*x2
                x5 = math.acos(-r_13*x4*math.cos(th_0) - r_23*x4*math.sin(th_0) - r_33*(x0*x1 - x2*x3))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[7] = x5
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(10, appended_idx)
                
            condition_1: bool = (abs(-r_13*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.cos(th_0) - r_23*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_0) - r_33*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.cos(th_2)
                x2 = math.sin(th_1)
                x3 = math.sin(th_2)
                x4 = x0*x3 + x1*x2
                x5 = math.acos(-r_13*x4*math.cos(th_0) - r_23*x4*math.sin(th_0) - r_33*(x0*x1 - x2*x3))
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
    
    # Code for explicit solution node 19, solved variable is negative_th_5_positive_th_3__soa
    def ExplicitSolutionNode_node_19_solve_negative_th_5_positive_th_3__soa_processor():
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
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                # End of temp variables
                this_solution[0] = math.atan2(-r_11*x1 + r_21*x0, r_12*x1 - r_22*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(20, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_19_solve_negative_th_5_positive_th_3__soa_processor()
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
    ExplicitSolutionNode_node_23_solve_th_5_processor()
    # Finish code for explicit solution node 22
    
    # Code for explicit solution node 14, solved variable is th_3th_5_soa
    def ExplicitSolutionNode_node_14_solve_th_3th_5_soa_processor():
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
                this_solution[6] = math.atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_3th_5_soa_processor()
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
                this_solution[5] = math.atan2(x0*(r_13*x1 - r_23*x2), x0*(r_13*x2*x7 + r_23*x1*x7 - r_33*(x3*x4 + x5*x6)))
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
            condition_0: bool = (abs(-r_11*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.cos(th_0) - r_21*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_0) - r_31*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) >= zero_tolerance) or (abs(r_12*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_22*(math.sin(th_1)*math.cos(th_2) + math.sin(th_2)*math.cos(th_1))*math.sin(th_0) + r_32*(-math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) >= zero_tolerance) or (abs(math.sin(th_4)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_4)**(-1)
                x1 = math.cos(th_1)
                x2 = math.cos(th_2)
                x3 = math.sin(th_1)
                x4 = math.sin(th_2)
                x5 = x1*x2 - x3*x4
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


def puma_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = T_ee
    ik_output_raw = puma_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ee_pose_i = puma_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_puma():
    theta_in = np.random.random(size=(6, ))
    ee_pose = puma_fk(theta_in)
    ik_output = puma_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = puma_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_puma()
