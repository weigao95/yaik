import numpy as np
import copy
import math
from typing import List, NewType

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 26
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
l_1: float = 0.19681
l_2: float = 0.251
l_3: float = 0.145423

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def arm_robo_ik_target_original_to_raw(T_ee: np.ndarray):
    return T_ee


def arm_robo_ik_target_raw_to_original(T_ee: np.ndarray):
    return T_ee


def arm_robo_fk(theta_input: np.ndarray):
    th_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]
    d_6 = theta_input[5]

    # Temp variable for efficiency
    x0 = math.sin(th_4)
    x1 = math.sin(th_2)
    x2 = math.cos(th_0)
    x3 = math.cos(th_1)
    x4 = x2*x3
    x5 = math.sin(th_0)
    x6 = math.sin(th_1)
    x7 = x5*x6
    x8 = x4 - x7
    x9 = x1*x8
    x10 = math.cos(th_4)
    x11 = math.sin(th_3)
    x12 = x2*x6 + x3*x5
    x13 = x11*x12
    x14 = math.cos(th_3)
    x15 = math.cos(th_2)
    x16 = x15*x8
    x17 = -x13 + x14*x16
    x18 = x12*x14
    x19 = x11*x16 + x18
    x20 = x1*x14
    x21 = x1*x11
    x22 = x1*x12
    x23 = -x4 + x7
    x24 = -x11*x23 + x15*x18
    x25 = x13*x15 + x14*x23
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -x0*x9 + x10*x17
    ee_pose[0, 1] = -x0*x17 - x10*x9
    ee_pose[0, 2] = x19
    ee_pose[0, 3] = d_6*x19 + l_1*x2 - l_3*x9
    ee_pose[1, 0] = -x0*x15 - x10*x20
    ee_pose[1, 1] = x0*x20 - x10*x15
    ee_pose[1, 2] = -x21
    ee_pose[1, 3] = -d_6*x21 - l_2 - l_3*x15
    ee_pose[2, 0] = -x0*x22 + x10*x24
    ee_pose[2, 1] = -x0*x24 - x10*x22
    ee_pose[2, 2] = x25
    ee_pose[2, 3] = d_6*x25 + l_1*x5 - l_3*x22
    return ee_pose


def arm_robo_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]
    d_6 = theta_input[5]

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.cos(th_1)
    x2 = math.sin(th_1)
    x3 = math.cos(th_0)
    x4 = x0*x1 + x2*x3
    x5 = math.sin(th_2)
    x6 = x1*x3
    x7 = x0*x2
    x8 = x6 - x7
    x9 = x5*x8
    x10 = math.cos(th_3)
    x11 = math.sin(th_3)
    x12 = math.cos(th_2)
    x13 = x11*x12
    x14 = x10*x4 + x13*x8
    x15 = x11*x5
    x16 = -x15
    x17 = -x6 + x7
    x18 = x4*x5
    x19 = x10*x17 + x13*x4
    x20 = l_1*x0
    x21 = -l_3*x18 + x20
    x22 = -l_2 - l_3*x12
    x23 = l_1*x3
    x24 = -l_3*x9 + x23
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 2] = x4
    jacobian_output[0, 3] = -x9
    jacobian_output[0, 4] = x14
    jacobian_output[1, 0] = -1
    jacobian_output[1, 1] = -1
    jacobian_output[1, 3] = -x12
    jacobian_output[1, 4] = x16
    jacobian_output[2, 2] = x17
    jacobian_output[2, 3] = -x18
    jacobian_output[2, 4] = x19
    jacobian_output[3, 1] = x20
    jacobian_output[3, 2] = -l_2*x17
    jacobian_output[3, 3] = x12*x21 - x18*x22
    jacobian_output[3, 4] = x15*x21 + x19*x22
    jacobian_output[3, 5] = x14
    jacobian_output[4, 2] = -x17*x23 + x20*x4
    jacobian_output[4, 3] = x18*x24 - x21*x9
    jacobian_output[4, 4] = x14*x21 - x19*x24
    jacobian_output[4, 5] = x16
    jacobian_output[5, 1] = -x23
    jacobian_output[5, 2] = l_2*x4
    jacobian_output[5, 3] = -x12*x24 + x22*x9
    jacobian_output[5, 4] = -x14*x22 - x15*x24
    jacobian_output[5, 5] = x19
    return jacobian_output


def arm_robo_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]
    d_6 = theta_input[5]

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.cos(th_1)
    x2 = math.sin(th_1)
    x3 = math.cos(th_0)
    x4 = x0*x1 + x2*x3
    x5 = math.sin(th_2)
    x6 = x1*x3
    x7 = x0*x2
    x8 = x6 - x7
    x9 = math.cos(th_3)
    x10 = math.sin(th_3)
    x11 = math.cos(th_2)
    x12 = x10*x11
    x13 = -x6 + x7
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 2] = x4
    jacobian_output[0, 3] = -x5*x8
    jacobian_output[0, 4] = x12*x8 + x4*x9
    jacobian_output[1, 0] = -1
    jacobian_output[1, 1] = -1
    jacobian_output[1, 3] = -x11
    jacobian_output[1, 4] = -x10*x5
    jacobian_output[2, 2] = x13
    jacobian_output[2, 3] = -x4*x5
    jacobian_output[2, 4] = x12*x4 + x13*x9
    return jacobian_output


def arm_robo_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    th_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]
    d_6 = theta_input[5]
    p_on_ee_x: float = point_on_ee[0]
    p_on_ee_y: float = point_on_ee[1]
    p_on_ee_z: float = point_on_ee[2]

    # Temp variable for efficiency
    x0 = -p_on_ee_z
    x1 = math.sin(th_0)
    x2 = l_1*x1
    x3 = math.sin(th_1)
    x4 = x1*x3
    x5 = math.cos(th_0)
    x6 = math.cos(th_1)
    x7 = x5*x6
    x8 = x4 - x7
    x9 = math.cos(th_2)
    x10 = math.sin(th_2)
    x11 = x1*x6 + x3*x5
    x12 = p_on_ee_y*x11
    x13 = x10*x11
    x14 = -l_3*x13 + x2
    x15 = -l_2 - l_3*x9
    x16 = math.sin(th_3)
    x17 = x10*x16
    x18 = math.cos(th_3)
    x19 = x16*x9
    x20 = x11*x19 + x18*x8
    x21 = -x4 + x7
    x22 = x11*x18 + x19*x21
    x23 = l_1*x5
    x24 = x10*x21
    x25 = -l_3*x24 + x23
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = x0
    jacobian_output[0, 1] = x0 + x2
    jacobian_output[0, 2] = -l_2*x8 - p_on_ee_y*x8
    jacobian_output[0, 3] = -p_on_ee_z*x9 + x10*x12 - x13*x15 + x14*x9
    jacobian_output[0, 4] = -p_on_ee_y*x20 - p_on_ee_z*x17 + x14*x17 + x15*x20
    jacobian_output[0, 5] = x22
    jacobian_output[1, 2] = p_on_ee_x*x8 - p_on_ee_z*x11 + x11*x2 - x23*x8
    jacobian_output[1, 3] = -p_on_ee_x*x13 + p_on_ee_z*x24 + x13*x25 - x14*x24
    jacobian_output[1, 4] = p_on_ee_x*x20 - p_on_ee_z*x22 + x14*x22 - x20*x25
    jacobian_output[1, 5] = -x17
    jacobian_output[2, 0] = p_on_ee_x
    jacobian_output[2, 1] = p_on_ee_x - x23
    jacobian_output[2, 2] = l_2*x11 + x12
    jacobian_output[2, 3] = p_on_ee_x*x9 - p_on_ee_y*x24 + x15*x24 - x25*x9
    jacobian_output[2, 4] = p_on_ee_x*x17 + p_on_ee_y*x22 - x15*x22 - x17*x25
    jacobian_output[2, 5] = x20
    return jacobian_output


def arm_robo_ik_solve_raw(T_ee: np.ndarray):
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
        for tmp_sol_idx in range(10):
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
    for i in range(26):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True
    
    # Code for non-branch dispatcher node 0
    # Actually, there is no code
    
    # Code for polynomial solution node 0, solved variable is th_0
    def PolynomialSolutionNode_node_1_solve_th_0_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return
        
        # The polynomial solution of root node
        poly_coefficient_0_num = (inv_Px**2 - 2*inv_Px*l_1*r_11 - 2*inv_Px*l_2*r_21 + inv_Py**2 - 2*inv_Py*l_1*r_12 - 2*inv_Py*l_2*r_22 + l_1**2*r_11**2 + l_1**2*r_12**2 + 2*l_1*l_2*r_11*r_21 + 2*l_1*l_2*r_12*r_22 + l_2**2*r_21**2 + l_2**2*r_22**2 - l_3**2)*(inv_Px**2 + 2*inv_Px*l_1*r_11 - 2*inv_Px*l_2*r_21 + inv_Py**2 + 2*inv_Py*l_1*r_12 - 2*inv_Py*l_2*r_22 + l_1**2*r_11**2 + l_1**2*r_12**2 - 2*l_1*l_2*r_11*r_21 - 2*l_1*l_2*r_12*r_22 + l_2**2*r_21**2 + l_2**2*r_22**2 - l_3**2)
        poly_coefficient_0_denom = 1
        poly_coefficient_0 = poly_coefficient_0_num / poly_coefficient_0_denom
        poly_coefficient_1_num = 4*l_1*(inv_Px**3*r_31 + inv_Px**2*inv_Py*r_32 - 3*inv_Px**2*l_2*r_21*r_31 - inv_Px**2*l_2*r_22*r_32 + inv_Px*inv_Py**2*r_31 - 2*inv_Px*inv_Py*l_2*r_21*r_32 - 2*inv_Px*inv_Py*l_2*r_22*r_31 - inv_Px*l_1**2*r_11**2*r_31 - 2*inv_Px*l_1**2*r_11*r_12*r_32 + inv_Px*l_1**2*r_12**2*r_31 + 3*inv_Px*l_2**2*r_21**2*r_31 + 2*inv_Px*l_2**2*r_21*r_22*r_32 + inv_Px*l_2**2*r_22**2*r_31 - inv_Px*l_3**2*r_31 + inv_Py**3*r_32 - inv_Py**2*l_2*r_21*r_31 - 3*inv_Py**2*l_2*r_22*r_32 + inv_Py*l_1**2*r_11**2*r_32 - 2*inv_Py*l_1**2*r_11*r_12*r_31 - inv_Py*l_1**2*r_12**2*r_32 + inv_Py*l_2**2*r_21**2*r_32 + 2*inv_Py*l_2**2*r_21*r_22*r_31 + 3*inv_Py*l_2**2*r_22**2*r_32 - inv_Py*l_3**2*r_32 + l_1**2*l_2*r_11**2*r_21*r_31 - l_1**2*l_2*r_11**2*r_22*r_32 + 2*l_1**2*l_2*r_11*r_12*r_21*r_32 + 2*l_1**2*l_2*r_11*r_12*r_22*r_31 - l_1**2*l_2*r_12**2*r_21*r_31 + l_1**2*l_2*r_12**2*r_22*r_32 - l_2**3*r_21**3*r_31 - l_2**3*r_21**2*r_22*r_32 - l_2**3*r_21*r_22**2*r_31 - l_2**3*r_22**3*r_32 + l_2*l_3**2*r_21*r_31 + l_2*l_3**2*r_22*r_32)
        poly_coefficient_1_denom = 1
        poly_coefficient_1 = poly_coefficient_1_num / poly_coefficient_1_denom
        poly_coefficient_2_num = 2*l_1**2*(inv_Px**2*r_11**2 - inv_Px**2*r_12**2 + 3*inv_Px**2*r_31**2 + inv_Px**2*r_32**2 + 4*inv_Px*inv_Py*r_11*r_12 + 4*inv_Px*inv_Py*r_31*r_32 - 2*inv_Px*l_2*r_11**2*r_21 - 4*inv_Px*l_2*r_11*r_12*r_22 + 2*inv_Px*l_2*r_12**2*r_21 - 6*inv_Px*l_2*r_21*r_31**2 - 2*inv_Px*l_2*r_21*r_32**2 - 4*inv_Px*l_2*r_22*r_31*r_32 - inv_Py**2*r_11**2 + inv_Py**2*r_12**2 + inv_Py**2*r_31**2 + 3*inv_Py**2*r_32**2 + 2*inv_Py*l_2*r_11**2*r_22 - 4*inv_Py*l_2*r_11*r_12*r_21 - 2*inv_Py*l_2*r_12**2*r_22 - 4*inv_Py*l_2*r_21*r_31*r_32 - 2*inv_Py*l_2*r_22*r_31**2 - 6*inv_Py*l_2*r_22*r_32**2 - l_1**2*r_11**4 - 2*l_1**2*r_11**2*r_12**2 - l_1**2*r_11**2*r_31**2 + l_1**2*r_11**2*r_32**2 - 4*l_1**2*r_11*r_12*r_31*r_32 - l_1**2*r_12**4 + l_1**2*r_12**2*r_31**2 - l_1**2*r_12**2*r_32**2 + l_2**2*r_11**2*r_21**2 - l_2**2*r_11**2*r_22**2 + 4*l_2**2*r_11*r_12*r_21*r_22 - l_2**2*r_12**2*r_21**2 + l_2**2*r_12**2*r_22**2 + 3*l_2**2*r_21**2*r_31**2 + l_2**2*r_21**2*r_32**2 + 4*l_2**2*r_21*r_22*r_31*r_32 + l_2**2*r_22**2*r_31**2 + 3*l_2**2*r_22**2*r_32**2 + l_3**2*r_11**2 + l_3**2*r_12**2 - l_3**2*r_31**2 - l_3**2*r_32**2)
        poly_coefficient_2_denom = 1
        poly_coefficient_2 = poly_coefficient_2_num / poly_coefficient_2_denom
        poly_coefficient_3_num = 4*l_1**3*(inv_Px*r_11**2*r_31 + 2*inv_Px*r_11*r_12*r_32 - inv_Px*r_12**2*r_31 + inv_Px*r_31**3 + inv_Px*r_31*r_32**2 - inv_Py*r_11**2*r_32 + 2*inv_Py*r_11*r_12*r_31 + inv_Py*r_12**2*r_32 + inv_Py*r_31**2*r_32 + inv_Py*r_32**3 - l_2*r_11**2*r_21*r_31 + l_2*r_11**2*r_22*r_32 - 2*l_2*r_11*r_12*r_21*r_32 - 2*l_2*r_11*r_12*r_22*r_31 + l_2*r_12**2*r_21*r_31 - l_2*r_12**2*r_22*r_32 - l_2*r_21*r_31**3 - l_2*r_21*r_31*r_32**2 - l_2*r_22*r_31**2*r_32 - l_2*r_22*r_32**3)
        poly_coefficient_3_denom = 1
        poly_coefficient_3 = poly_coefficient_3_num / poly_coefficient_3_denom
        poly_coefficient_4_num = l_1**4*(r_11**2 - 2*r_11*r_32 + r_12**2 + 2*r_12*r_31 + r_31**2 + r_32**2)*(r_11**2 + 2*r_11*r_32 + r_12**2 - 2*r_12*r_31 + r_31**2 + r_32**2)
        poly_coefficient_4_denom = 1
        poly_coefficient_4 = poly_coefficient_4_num / poly_coefficient_4_denom
        p_coefficients = np.zeros(shape=(4 + 1,))
        p_coefficients[4] = poly_coefficient_0
        p_coefficients[3] = poly_coefficient_1
        p_coefficients[2] = poly_coefficient_2
        p_coefficients[1] = poly_coefficient_3
        p_coefficients[0] = poly_coefficient_4
        
        # Note that in np.roots, p_coefficient[0] is the highest order
        poly_roots = np.roots(p_coefficients)
        
        # Result collection
        for root_idx in range(len(poly_roots)):
            this_root = poly_roots[root_idx]
            if not np.isreal(this_root):
                continue
            if abs(this_root) > 1:
                continue
            first_angle = np.arcsin(this_root)
            second_angle = np.pi - np.arcsin(this_root)
            solution_0: IkSolution = make_ik_solution()
            solution_0[2] = first_angle
            solution_1: IkSolution = make_ik_solution()
            solution_1[2] = second_angle
            appended_idx_0 = append_solution_to_queue(solution_0)
            appended_idx_1 = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx_0)
            add_input_index_to(2, appended_idx_1)
    
    # Invoke the processor
    PolynomialSolutionNode_node_1_solve_th_0_processor()
    # Finish code for polynomial solution node 0
    
    # Code for non-branch dispatcher node 2
    # Actually, there is no code
    
    # Code for explicit solution node 3, solved variable is d_6
    def ExplicitSolutionNode_node_3_solve_d_6_processor():
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
            th_0 = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[0] = -inv_Pz - l_1*r_13*math.cos(th_0) - l_1*r_33*math.sin(th_0) + l_2*r_23
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_d_6_processor()
    # Finish code for explicit solution node 2
    
    # Code for non-branch dispatcher node 4
    # Actually, there is no code
    
    # Code for explicit solution node 5, solved variable is th_4
    def ExplicitSolutionNode_node_5_solve_th_4_processor():
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
            th_0 = this_solution[2]
            condition_0: bool = (abs(l_3) >= zero_tolerance) or (abs(inv_Px + l_1*(r_11*math.cos(th_0) + r_31*math.sin(th_0)) - l_2*r_21) >= zero_tolerance) or (abs(inv_Py + l_1*(r_12*math.cos(th_0) + r_32*math.sin(th_0)) - l_2*r_22) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = l_3**(-1)
                x1 = math.cos(th_0)
                x2 = math.sin(th_0)
                # End of temp variables
                this_solution[9] = math.atan2(x0*(-inv_Px - l_1*(r_11*x1 + r_31*x2) + l_2*r_21), x0*(-inv_Py - l_1*(r_12*x1 + r_32*x2) + l_2*r_22))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_4_processor()
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
            d_6 = this_solution[0]
            condition_0: bool = (abs((Py - d_6*r_23 + l_2)/l_3) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.acos((-Py + d_6*r_23 - l_2)/l_3)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[7] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
            condition_1: bool = (abs((Py - d_6*r_23 + l_2)/l_3) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.acos((-Py + d_6*r_23 - l_2)/l_3)
                # End of temp variables
                this_solution[7] = -x0
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
            th_2 = this_solution[7]
            degenerate_valid_0 = (abs(th_2) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(14, node_input_i_idx_in_queue)
            
            th_2 = this_solution[7]
            degenerate_valid_1 = (abs(th_2 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(21, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(9, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor()
    # Finish code for solved_variable dispatcher node 8
    
    # Code for explicit solution node 21, solved variable is negative_th_3_positive_th_1__soa
    def ExplicitSolutionNode_node_21_solve_negative_th_3_positive_th_1__soa_processor():
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
            th_0 = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_13*math.sin(th_0) - r_33*math.cos(th_0)) >= zero_tolerance) or (abs(r_13*math.cos(th_0) + r_33*math.sin(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                # End of temp variables
                this_solution[1] = math.atan2(r_13*x0 + r_33*x1, r_13*x1 - r_33*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(22, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_21_solve_negative_th_3_positive_th_1__soa_processor()
    # Finish code for explicit solution node 21
    
    # Code for non-branch dispatcher node 22
    # Actually, there is no code
    
    # Code for explicit solution node 23, solved variable is th_1
    def ExplicitSolutionNode_node_23_solve_th_1_processor():
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
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[5] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(24, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_1_processor()
    # Finish code for explicit solution node 22
    
    # Code for non-branch dispatcher node 24
    # Actually, there is no code
    
    # Code for explicit solution node 25, solved variable is th_3
    def ExplicitSolutionNode_node_25_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[24]
        this_input_valid: bool = node_input_validity[24]
        if not this_input_valid:
            return
        
        # The solution of non-root node 25
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            negative_th_3_positive_th_1__soa = this_solution[1]
            th_1 = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[8] = -negative_th_3_positive_th_1__soa + th_1
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_3_processor()
    # Finish code for explicit solution node 24
    
    # Code for explicit solution node 14, solved variable is th_0th_1th_3_soa
    def ExplicitSolutionNode_node_14_solve_th_0th_1th_3_soa_processor():
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
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_13) >= zero_tolerance) or (abs(r_33) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[4] = math.atan2(r_13, -r_33)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_0th_1th_3_soa_processor()
    # Finish code for explicit solution node 14
    
    # Code for non-branch dispatcher node 15
    # Actually, there is no code
    
    # Code for explicit solution node 16, solved variable is th_1th_3_soa
    def ExplicitSolutionNode_node_16_solve_th_1th_3_soa_processor():
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
            th_0 = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_13*math.sin(th_0) - r_33*math.cos(th_0)) >= zero_tolerance) or (abs(r_13*math.cos(th_0) + r_33*math.sin(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                # End of temp variables
                this_solution[6] = math.atan2(r_13*x0 + r_33*x1, r_13*x1 - r_33*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(17, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_1th_3_soa_processor()
    # Finish code for explicit solution node 15
    
    # Code for non-branch dispatcher node 17
    # Actually, there is no code
    
    # Code for explicit solution node 18, solved variable is th_1
    def ExplicitSolutionNode_node_18_solve_th_1_processor():
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
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[5] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(19, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_1_processor()
    # Finish code for explicit solution node 17
    
    # Code for non-branch dispatcher node 19
    # Actually, there is no code
    
    # Code for explicit solution node 20, solved variable is th_3
    def ExplicitSolutionNode_node_20_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[19]
        this_input_valid: bool = node_input_validity[19]
        if not this_input_valid:
            return
        
        # The solution of non-root node 20
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[5]
            th_1th_3_soa = this_solution[6]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[8] = -th_1 + th_1th_3_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_20_solve_th_3_processor()
    # Finish code for explicit solution node 19
    
    # Code for explicit solution node 9, solved variable is th_3
    def ExplicitSolutionNode_node_9_solve_th_3_processor():
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
            th_2 = this_solution[7]
            th_4 = this_solution[9]
            condition_0: bool = (abs(r_23) >= zero_tolerance) or (abs(r_21*math.cos(th_4) - r_22*math.sin(th_4)) >= zero_tolerance) or (abs(math.sin(th_2)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_2)**(-1)
                # End of temp variables
                this_solution[8] = math.atan2(-r_23*x0, x0*(-r_21*math.cos(th_4) + r_22*math.sin(th_4)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_3_processor()
    # Finish code for explicit solution node 9
    
    # Code for non-branch dispatcher node 10
    # Actually, there is no code
    
    # Code for explicit solution node 11, solved variable is th_1
    def ExplicitSolutionNode_node_11_solve_th_1_processor():
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
            d_6 = this_solution[0]
            th_0 = this_solution[2]
            th_2 = this_solution[7]
            condition_0: bool = (abs(l_3*math.sin(th_2)) >= zero_tolerance) or (abs(Px*math.sin(th_0) - Pz*math.cos(th_0) + d_6*(-r_13*math.sin(th_0) + r_33*math.cos(th_0))) >= zero_tolerance) or (abs(Px*math.cos(th_0) + Pz*math.sin(th_0) - d_6*(r_13*math.cos(th_0) + r_33*math.sin(th_0)) - l_1) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                x2 = 1/(l_3*math.sin(th_2))
                # End of temp variables
                this_solution[5] = math.atan2(x2*(Px*x0 - Pz*x1 + d_6*(-r_13*x0 + r_33*x1)), x2*(-Px*x1 - Pz*x0 + d_6*(r_13*x1 + r_33*x0) + l_1))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(12, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1_processor()
    # Finish code for explicit solution node 10
    
    # Code for non-branch dispatcher node 12
    # Actually, there is no code
    
    # Code for explicit solution node 13, solved variable is th_0th_1_soa
    def ExplicitSolutionNode_node_13_solve_th_0th_1_soa_processor():
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
            th_0 = this_solution[2]
            th_1 = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[3] = th_0 + th_1
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_0th_1_soa_processor()
    # Finish code for explicit solution node 12
    
    # Collect the output
    ik_out: List[np.ndarray] = list()
    for i in range(len(solution_queue)):
        if not queue_element_validity[i]:
            continue
        ik_out_i = solution_queue[i]
        new_ik_i = np.zeros((robot_nq, 1))
        value_at_0 = ik_out_i[2]  # th_0
        new_ik_i[0] = value_at_0
        value_at_1 = ik_out_i[5]  # th_1
        new_ik_i[1] = value_at_1
        value_at_2 = ik_out_i[7]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[8]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[9]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[0]  # d_6
        new_ik_i[5] = value_at_5
        ik_out.append(new_ik_i)
    return ik_out


def arm_robo_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = T_ee
    ik_output_raw = arm_robo_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ee_pose_i = arm_robo_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_arm_robo():
    theta_in = np.random.random(size=(6, ))
    ee_pose = arm_robo_fk(theta_in)
    ik_output = arm_robo_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = arm_robo_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_arm_robo()
