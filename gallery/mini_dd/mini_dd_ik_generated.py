import numpy as np
import copy
import math
from typing import List, NewType

# Constants for solver
robot_nq: int = 5
n_tree_nodes: int = 10
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
l_3: float = 5.0
l_4: float = 2.0

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def mini_dd_ik_target_original_to_raw(T_ee: np.ndarray):
    return T_ee


def mini_dd_ik_target_raw_to_original(T_ee: np.ndarray):
    return T_ee


def mini_dd_fk(theta_input: np.ndarray):
    d_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]

    # Temp variable for efficiency
    x0 = math.sin(th_4)
    x1 = math.cos(th_1)
    x2 = math.sin(th_2)
    x3 = x1*x2
    x4 = math.cos(th_4)
    x5 = math.sin(th_1)
    x6 = math.sin(th_3)
    x7 = x5*x6
    x8 = math.cos(th_2)
    x9 = math.cos(th_3)
    x10 = x1*x9
    x11 = x10*x8 + x7
    x12 = x5*x9
    x13 = x1*x6
    x14 = x2*x9
    x15 = x2*x5
    x16 = -x12*x8 + x13
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = x0*x3 + x11*x4
    ee_pose[0, 1] = -x0*x11 + x3*x4
    ee_pose[0, 2] = x12 - x13*x8
    ee_pose[0, 3] = l_3*x1 - l_4*x3
    ee_pose[1, 0] = x0*x8 - x14*x4
    ee_pose[1, 1] = x0*x14 + x4*x8
    ee_pose[1, 2] = x2*x6
    ee_pose[1, 3] = -l_4*x8
    ee_pose[2, 0] = -x0*x15 + x16*x4
    ee_pose[2, 1] = -x0*x16 - x15*x4
    ee_pose[2, 2] = x10 + x7*x8
    ee_pose[2, 3] = d_0 - l_3*x5 + l_4*x15
    return ee_pose


def mini_dd_twist_jacobian(theta_input: np.ndarray):
    d_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]

    # Temp variable for efficiency
    x0 = math.sin(th_1)
    x1 = math.sin(th_2)
    x2 = math.cos(th_1)
    x3 = x1*x2
    x4 = math.cos(th_3)
    x5 = math.sin(th_3)
    x6 = math.cos(th_2)
    x7 = x5*x6
    x8 = x0*x4 - x2*x7
    x9 = x1*x5
    x10 = x0*x1
    x11 = x0*x7 + x2*x4
    x12 = l_4*x10
    x13 = d_0 - l_3*x0
    x14 = x12 + x13
    x15 = l_4*x6
    x16 = l_3*x2 - l_4*x3
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 5))
    jacobian_output[0, 2] = -x0
    jacobian_output[0, 3] = -x3
    jacobian_output[0, 4] = x8
    jacobian_output[1, 1] = 1
    jacobian_output[1, 3] = -x6
    jacobian_output[1, 4] = x9
    jacobian_output[2, 2] = -x2
    jacobian_output[2, 3] = x10
    jacobian_output[2, 4] = x11
    jacobian_output[3, 1] = -d_0
    jacobian_output[3, 3] = -x12*x6 + x14*x6
    jacobian_output[3, 4] = -x11*x15 - x14*x9
    jacobian_output[4, 2] = l_3*x2**2 - x0*x13
    jacobian_output[4, 3] = -x10*x16 - x14*x3
    jacobian_output[4, 4] = -x11*x16 + x14*x8
    jacobian_output[5, 0] = 1
    jacobian_output[5, 3] = -x15*x3 - x16*x6
    jacobian_output[5, 4] = x15*x8 + x16*x9
    return jacobian_output


def mini_dd_angular_velocity_jacobian(theta_input: np.ndarray):
    d_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]

    # Temp variable for efficiency
    x0 = math.sin(th_1)
    x1 = math.sin(th_2)
    x2 = math.cos(th_1)
    x3 = math.cos(th_3)
    x4 = math.sin(th_3)
    x5 = math.cos(th_2)
    x6 = x4*x5
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 5))
    jacobian_output[0, 2] = -x0
    jacobian_output[0, 3] = -x1*x2
    jacobian_output[0, 4] = x0*x3 - x2*x6
    jacobian_output[1, 1] = 1
    jacobian_output[1, 3] = -x5
    jacobian_output[1, 4] = x1*x4
    jacobian_output[2, 2] = -x2
    jacobian_output[2, 3] = x0*x1
    jacobian_output[2, 4] = x0*x6 + x2*x3
    return jacobian_output


def mini_dd_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
    d_0 = theta_input[0]
    th_1 = theta_input[1]
    th_2 = theta_input[2]
    th_3 = theta_input[3]
    th_4 = theta_input[4]
    p_on_ee_x: float = point_on_ee[0]
    p_on_ee_y: float = point_on_ee[1]
    p_on_ee_z: float = point_on_ee[2]

    # Temp variable for efficiency
    x0 = math.cos(th_1)
    x1 = p_on_ee_y*x0
    x2 = math.cos(th_2)
    x3 = math.sin(th_2)
    x4 = math.sin(th_1)
    x5 = p_on_ee_y*x4
    x6 = x3*x4
    x7 = l_4*x6
    x8 = d_0 - l_3*x4
    x9 = x7 + x8
    x10 = math.sin(th_3)
    x11 = x10*x3
    x12 = math.cos(th_3)
    x13 = x10*x2
    x14 = x0*x12 + x13*x4
    x15 = l_4*x2
    x16 = x0*x3
    x17 = l_3*x0 - l_4*x16
    x18 = -x0*x13 + x12*x4
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 5))
    jacobian_output[0, 1] = -d_0 + p_on_ee_z
    jacobian_output[0, 2] = x1
    jacobian_output[0, 3] = -p_on_ee_z*x2 - x2*x7 + x2*x9 - x3*x5
    jacobian_output[0, 4] = -p_on_ee_y*x14 + p_on_ee_z*x11 - x11*x9 - x14*x15
    jacobian_output[1, 2] = l_3*x0**2 - p_on_ee_x*x0 + p_on_ee_z*x4 - x4*x8
    jacobian_output[1, 3] = p_on_ee_x*x6 + p_on_ee_z*x16 - x16*x9 - x17*x6
    jacobian_output[1, 4] = p_on_ee_x*x14 - p_on_ee_z*x18 - x14*x17 + x18*x9
    jacobian_output[2, 0] = 1
    jacobian_output[2, 1] = -p_on_ee_x
    jacobian_output[2, 2] = -x5
    jacobian_output[2, 3] = p_on_ee_x*x2 - x1*x3 - x15*x16 - x17*x2
    jacobian_output[2, 4] = -p_on_ee_x*x11 + p_on_ee_y*x18 + x11*x17 + x15*x18
    return jacobian_output


def mini_dd_ik_solve_raw(T_ee: np.ndarray):
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
    for i in range(10):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True
    
    # Code for non-branch dispatcher node 0
    # Actually, there is no code
    
    # Code for explicit solution node 1, solved variable is th_2
    def ExplicitSolutionNode_node_1_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return
        
        # The explicit solution of root node
        condition_0: bool = (abs(Py/l_4) <= 1)
        if condition_0:
            # Temp variable for efficiency
            x0 = math.acos(-Py/l_4)
            # End of temp variables
            solution_0: IkSolution = make_ik_solution()
            solution_0[2] = x0
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)
            
        condition_1: bool = (abs(Py/l_4) <= 1)
        if condition_1:
            # Temp variable for efficiency
            x0 = math.acos(-Py/l_4)
            # End of temp variables
            solution_1: IkSolution = make_ik_solution()
            solution_1[2] = -x0
            appended_idx = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx)
            
    # Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_2_processor()
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
            th_2 = this_solution[2]
            checked_result: bool = (abs(l_3 - l_4*math.sin(th_2)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor()
    # Finish code for equation all-zero dispatcher node 2
    
    # Code for explicit solution node 3, solved variable is th_1
    def ExplicitSolutionNode_node_3_solve_th_1_processor():
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
            condition_0: bool = (abs(Px/(l_3 - l_4*math.sin(th_2))) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.acos(Px/(l_3 - l_4*math.sin(th_2)))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs(Px/(l_3 - l_4*math.sin(th_2))) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.acos(Px/(l_3 - l_4*math.sin(th_2)))
                # End of temp variables
                this_solution[1] = -x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_1_processor()
    # Finish code for explicit solution node 3
    
    # Code for non-branch dispatcher node 4
    # Actually, there is no code
    
    # Code for explicit solution node 5, solved variable is d_0
    def ExplicitSolutionNode_node_5_solve_d_0_processor():
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_1)
                # End of temp variables
                this_solution[0] = Pz + l_3*x0 - l_4*x0*math.sin(th_2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_d_0_processor()
    # Finish code for explicit solution node 4
    
    # Code for non-branch dispatcher node 6
    # Actually, there is no code
    
    # Code for explicit solution node 7, solved variable is th_3
    def ExplicitSolutionNode_node_7_solve_th_3_processor():
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_13*math.sin(th_1) + r_33*math.cos(th_1)) >= zero_tolerance) or (abs(-r_13*math.cos(th_1)*math.cos(th_2) + r_23*math.sin(th_2) + r_33*math.sin(th_1)*math.cos(th_2)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_1)
                x1 = math.cos(th_2)
                x2 = math.cos(th_1)
                # End of temp variables
                this_solution[3] = math.atan2(-r_13*x1*x2 + r_23*math.sin(th_2) + r_33*x0*x1, r_13*x0 + r_33*x2)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_3_processor()
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_2)*math.cos(th_1) + r_21*math.cos(th_2) - r_31*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance) or (abs(r_12*math.sin(th_2)*math.cos(th_1) + r_22*math.cos(th_2) - r_32*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_2)
                x1 = math.sin(th_2)
                x2 = x1*math.cos(th_1)
                x3 = x1*math.sin(th_1)
                # End of temp variables
                this_solution[4] = math.atan2(r_11*x2 + r_21*x0 - r_31*x3, r_12*x2 + r_22*x0 - r_32*x3)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_4_processor()
    # Finish code for explicit solution node 8
    
    # Collect the output
    ik_out: List[np.ndarray] = list()
    for i in range(len(solution_queue)):
        if not queue_element_validity[i]:
            continue
        ik_out_i = solution_queue[i]
        new_ik_i = np.zeros((robot_nq, 1))
        value_at_0 = ik_out_i[0]  # d_0
        new_ik_i[0] = value_at_0
        value_at_1 = ik_out_i[1]  # th_1
        new_ik_i[1] = value_at_1
        value_at_2 = ik_out_i[2]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[3]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[4]  # th_4
        new_ik_i[4] = value_at_4
        ik_out.append(new_ik_i)
    return ik_out


def mini_dd_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = T_ee
    ik_output_raw = mini_dd_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ee_pose_i = mini_dd_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_mini_dd():
    theta_in = np.random.random(size=(5, ))
    ee_pose = mini_dd_fk(theta_in)
    ik_output = mini_dd_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = mini_dd_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_mini_dd()
