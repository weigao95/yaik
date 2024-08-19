import numpy as np
import copy
import math
from typing import List, NewType

# Constants for solver
robot_nq: int = 7
n_tree_nodes: int = 58
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
d_1: float = 0.42
d_2: float = 0.4
d_3: float = 0.081
post_transform_d5: float = 0.045
pre_transform_d4: float = 0.36

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = 3.141592653589793
th_2_offset_original2raw: float = 0.0
th_3_offset_original2raw: float = -3.141592653589793
th_4_offset_original2raw: float = 0.0
th_5_offset_original2raw: float = -3.141592653589793
th_6_offset_original2raw: float = 3.141592653589793


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def kuka_iiwa_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 3] = Px - post_transform_d5*r_11
    ee_transformed[1, 0] = -r_23
    ee_transformed[1, 1] = r_22
    ee_transformed[1, 2] = r_21
    ee_transformed[1, 3] = Py - post_transform_d5*r_21
    ee_transformed[2, 0] = -r_33
    ee_transformed[2, 1] = r_32
    ee_transformed[2, 2] = r_31
    ee_transformed[2, 3] = Pz - post_transform_d5*r_31 - pre_transform_d4
    return ee_transformed


def kuka_iiwa_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 3] = Px + post_transform_d5*r_13
    ee_transformed[1, 0] = r_23
    ee_transformed[1, 1] = r_22
    ee_transformed[1, 2] = -r_21
    ee_transformed[1, 3] = Py + post_transform_d5*r_23
    ee_transformed[2, 0] = r_33
    ee_transformed[2, 1] = r_32
    ee_transformed[2, 2] = -r_31
    ee_transformed[2, 3] = Pz + post_transform_d5*r_33 + pre_transform_d4
    return ee_transformed


def kuka_iiwa_fk(theta_input: np.ndarray):
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
    x22 = -x0*x13 - x14*x21
    x23 = math.cos(th_6)
    x24 = -x15*x20 - x18*x19
    x25 = math.sin(th_6)
    x26 = x0*x21 - x13*x14
    x27 = x3*x6
    x28 = x16*x9 - x17
    x29 = x1*x27 - x28*x5
    x30 = -x11 - x8*x9
    x31 = x1*x28 + x27*x5
    x32 = -x15*x30 + x19*x31
    x33 = -x0*x29 - x14*x32
    x34 = -x15*x31 - x19*x30
    x35 = x0*x32 - x14*x29
    x36 = x10*x3
    x37 = x1*x9 + x36*x5
    x38 = x3*x7
    x39 = -x1*x36 + x5*x9
    x40 = -x15*x38 + x19*x39
    x41 = -x0*x37 - x14*x40
    x42 = -x15*x39 - x19*x38
    x43 = x0*x40 - x14*x37
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = x22
    ee_pose[0, 1] = -x23*x24 - x25*x26
    ee_pose[0, 2] = -x23*x26 + x24*x25
    ee_pose[0, 3] = -d_1*x4 + d_2*x13 + d_3*x22 + post_transform_d5*x22
    ee_pose[1, 0] = x33
    ee_pose[1, 1] = -x23*x34 - x25*x35
    ee_pose[1, 2] = -x23*x35 + x25*x34
    ee_pose[1, 3] = -d_1*x27 + d_2*x29 + d_3*x33 + post_transform_d5*x33
    ee_pose[2, 0] = x41
    ee_pose[2, 1] = -x23*x42 - x25*x43
    ee_pose[2, 2] = -x23*x43 + x25*x42
    ee_pose[2, 3] = -d_1*x9 + d_2*x37 + d_3*x41 + post_transform_d5*x41 + pre_transform_d4
    return ee_pose


def kuka_iiwa_twist_jacobian(theta_input: np.ndarray):
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
    x36 = d_1*x6
    x37 = pre_transform_d4 - x36
    x38 = d_1*x1**2
    x39 = d_2*x32 + x37
    x40 = d_1*x23
    x41 = d_2*x26 - x40
    x42 = d_3*x35 + x39
    x43 = d_3*x29 + x41
    x44 = d_1*x3
    x45 = d_2*x15 - x44
    x46 = d_3*x22 + x45
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
    jacobian_output[3, 1] = -pre_transform_d4*x2
    jacobian_output[3, 2] = x23*x36 + x23*x37
    jacobian_output[3, 3] = -x12*x38 - x24*x37
    jacobian_output[3, 4] = -x26*x39 + x32*x41
    jacobian_output[3, 5] = -x28*x39 + x34*x41
    jacobian_output[3, 6] = -x29*x42 + x35*x43
    jacobian_output[4, 1] = -pre_transform_d4*x0
    jacobian_output[4, 2] = -x3*x36 - x3*x37
    jacobian_output[4, 3] = x37*x9 + x38*x8
    jacobian_output[4, 4] = x15*x39 - x32*x45
    jacobian_output[4, 5] = x19*x39 - x34*x45
    jacobian_output[4, 6] = x22*x42 - x35*x46
    jacobian_output[5, 3] = -x24*x44 + x40*x9
    jacobian_output[5, 4] = -x15*x41 + x26*x45
    jacobian_output[5, 5] = -x19*x41 + x28*x45
    jacobian_output[5, 6] = -x22*x43 + x29*x46
    return jacobian_output


def kuka_iiwa_angular_velocity_jacobian(theta_input: np.ndarray):
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


def kuka_iiwa_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x6 = d_1*x2
    x7 = x3*x4
    x8 = pre_transform_d4 - x6
    x9 = math.sin(th_2)
    x10 = x3*x9
    x11 = x4*x9
    x12 = d_1*x3**2
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
    x24 = d_2*x19 + x8
    x25 = d_1*x7
    x26 = d_2*x23 - x25
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
    x37 = d_3*x35 + x24
    x38 = d_3*x36 + x26
    x39 = x0*x3
    x40 = -x2*x20 + x21
    x41 = x11 + x14*x2
    x42 = x16*x39 - x17*x41
    x43 = d_1*x39
    x44 = d_2*x42 - x43
    x45 = x16*x41 + x17*x39
    x46 = -x27*x40 - x28*x45
    x47 = -x33*x42 - x34*(x27*x45 - x28*x40)
    x48 = d_3*x47 + x44
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 0] = -p_on_ee_y
    jacobian_output[0, 1] = -pre_transform_d4*x0 + x1
    jacobian_output[0, 2] = p_on_ee_y*x2 - x3*x5 + x6*x7 + x7*x8
    jacobian_output[0, 3] = -p_on_ee_y*x10 + p_on_ee_z*x15 - x11*x12 - x15*x8
    jacobian_output[0, 4] = -p_on_ee_y*x19 + p_on_ee_z*x23 + x19*x26 - x23*x24
    jacobian_output[0, 5] = -p_on_ee_y*x30 + p_on_ee_z*x32 - x24*x32 + x26*x30
    jacobian_output[0, 6] = -p_on_ee_y*x35 + p_on_ee_z*x36 + x35*x38 - x36*x37
    jacobian_output[1, 0] = p_on_ee_x
    jacobian_output[1, 1] = -pre_transform_d4*x4 + x5
    jacobian_output[1, 2] = -p_on_ee_x*x2 + x1*x3 - x39*x6 - x39*x8
    jacobian_output[1, 3] = p_on_ee_x*x10 - p_on_ee_z*x40 + x12*x20 + x40*x8
    jacobian_output[1, 4] = p_on_ee_x*x19 - p_on_ee_z*x42 - x19*x44 + x24*x42
    jacobian_output[1, 5] = p_on_ee_x*x30 - p_on_ee_z*x46 + x24*x46 - x30*x44
    jacobian_output[1, 6] = p_on_ee_x*x35 - p_on_ee_z*x47 - x35*x48 + x37*x47
    jacobian_output[2, 1] = -p_on_ee_x*x0 - p_on_ee_y*x4
    jacobian_output[2, 2] = p_on_ee_x*x7 - p_on_ee_y*x39
    jacobian_output[2, 3] = -p_on_ee_x*x15 + p_on_ee_y*x40 - x15*x43 + x25*x40
    jacobian_output[2, 4] = -p_on_ee_x*x23 + p_on_ee_y*x42 + x23*x44 - x26*x42
    jacobian_output[2, 5] = -p_on_ee_x*x32 + p_on_ee_y*x46 - x26*x46 + x32*x44
    jacobian_output[2, 6] = -p_on_ee_x*x36 + p_on_ee_y*x47 + x36*x48 - x38*x47
    return jacobian_output


def kuka_iiwa_ik_solve_raw(T_ee: np.ndarray, th_0):
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
    for i in range(58):
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
        condition_0: bool = ((1/2)*abs((-d_1**2 - d_2**2 + (Px - d_3*r_13)**2 + (Py - d_3*r_23)**2 + (Pz - d_3*r_33)**2)/(d_1*d_2)) <= 1)
        if condition_0:
            # Temp variable for efficiency
            x0 = math.acos((1/2)*(d_1**2 + d_2**2 - (Px - d_3*r_13)**2 - (Py - d_3*r_23)**2 - (Pz - d_3*r_33)**2)/(d_1*d_2))
            # End of temp variables
            solution_0: IkSolution = make_ik_solution()
            solution_0[5] = x0
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)
            
        condition_1: bool = ((1/2)*abs((-d_1**2 - d_2**2 + (Px - d_3*r_13)**2 + (Py - d_3*r_23)**2 + (Pz - d_3*r_33)**2)/(d_1*d_2)) <= 1)
        if condition_1:
            # Temp variable for efficiency
            x0 = math.acos((1/2)*(d_1**2 + d_2**2 - (Px - d_3*r_13)**2 - (Py - d_3*r_23)**2 - (Pz - d_3*r_33)**2)/(d_1*d_2))
            # End of temp variables
            solution_1: IkSolution = make_ik_solution()
            solution_1[5] = -x0
            appended_idx = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx)
            
    # Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_3_processor()
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
            th_3 = this_solution[5]
            degenerate_valid_0 = (abs(th_3) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(12, node_input_i_idx_in_queue)
            
            th_3 = this_solution[5]
            degenerate_valid_1 = (abs(th_3 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(23, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(3, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_2_processor()
    # Finish code for solved_variable dispatcher node 2
    
    # Code for explicit solution node 23, solved variable is th_1
    def ExplicitSolutionNode_node_23_solve_th_1_processor():
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
            condition_0: bool = (abs(Pz - d_3*r_33) >= zero_tolerance) or (abs(d_1 + d_2) >= zero_tolerance) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) - d_3*(r_13*math.cos(th_0) + r_23*math.sin(th_0))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                # End of temp variables
                this_solution[2] = math.atan2((Px*x0 + Py*x1 - d_3*(r_13*x0 + r_23*x1))/(-d_1 - d_2), (-Pz + d_3*r_33)/(d_1 + d_2))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(24, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_1_processor()
    # Finish code for explicit solution node 23
    
    # Code for non-branch dispatcher node 24
    # Actually, there is no code
    
    # Code for explicit solution node 25, solved variable is th_5
    def ExplicitSolutionNode_node_25_solve_th_5_processor():
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
            condition_0: bool = (abs((d_3 + inv_Pz)/(d_1 + d_2)) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.acos((d_3 + inv_Pz)/(d_1 + d_2))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[8] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(26, appended_idx)
                
            condition_1: bool = (abs((d_3 + inv_Pz)/(d_1 + d_2)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.acos((d_3 + inv_Pz)/(d_1 + d_2))
                # End of temp variables
                this_solution[8] = -x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(26, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_5_processor()
    # Finish code for explicit solution node 24
    
    # Code for solved_variable dispatcher node 26
    def SolvedVariableDispatcherNode_node_26_processor():
        this_node_input_index: List[int] = node_input_index[26]
        this_input_valid: bool = node_input_validity[26]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_5 = this_solution[8]
            degenerate_valid_0 = (abs(th_5) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
            
            th_5 = this_solution[8]
            degenerate_valid_1 = (abs(th_5 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
            
            if not taken_by_degenerate:
                add_input_index_to(27, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_26_processor()
    # Finish code for solved_variable dispatcher node 26
    
    # Code for explicit solution node 27, solved variable is th_6
    def ExplicitSolutionNode_node_27_solve_th_6_processor():
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
            th_5 = this_solution[8]
            condition_0: bool = (abs(inv_Px) >= zero_tolerance) or (abs(inv_Py) >= zero_tolerance) or (abs((d_1 + d_2)*math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 1/((d_1 + d_2)*math.sin(th_5))
                # End of temp variables
                this_solution[9] = math.atan2(-inv_Py*x0, inv_Px*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(28, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_6_processor()
    # Finish code for explicit solution node 27
    
    # Code for solved_variable dispatcher node 28
    def SolvedVariableDispatcherNode_node_28_processor():
        this_node_input_index: List[int] = node_input_index[28]
        this_input_valid: bool = node_input_validity[28]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_5 = this_solution[8]
            degenerate_valid_0 = (abs(th_5) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
            
            th_5 = this_solution[8]
            degenerate_valid_1 = (abs(th_5 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
            
            if not taken_by_degenerate:
                add_input_index_to(29, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_28_processor()
    # Finish code for solved_variable dispatcher node 28
    
    # Code for explicit solution node 29, solved variable is th_2th_4_soa
    def ExplicitSolutionNode_node_29_solve_th_2th_4_soa_processor():
        this_node_input_index: List[int] = node_input_index[29]
        this_input_valid: bool = node_input_validity[29]
        if not this_input_valid:
            return
        
        # The solution of non-root node 29
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_5 = this_solution[8]
            condition_0: bool = (abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) >= zero_tolerance) or (abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance) or (abs(math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_5)**(-1)
                x1 = math.sin(th_0)
                x2 = math.cos(th_0)
                x3 = math.cos(th_1)
                # End of temp variables
                this_solution[4] = math.atan2(x0*(r_13*x1 - r_23*x2), x0*(r_13*x2*x3 + r_23*x1*x3 - r_33*math.sin(th_1)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(30, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_2th_4_soa_processor()
    # Finish code for explicit solution node 29
    
    # Code for non-branch dispatcher node 30
    # Actually, there is no code
    
    # Code for explicit solution node 31, solved variable is th_2
    def ExplicitSolutionNode_node_31_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[30]
        this_input_valid: bool = node_input_validity[30]
        if not this_input_valid:
            return
        
        # The solution of non-root node 31
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[3] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(32, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_2_processor()
    # Finish code for explicit solution node 30
    
    # Code for non-branch dispatcher node 32
    # Actually, there is no code
    
    # Code for explicit solution node 33, solved variable is th_4
    def ExplicitSolutionNode_node_33_solve_th_4_processor():
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
            th_2 = this_solution[3]
            th_2th_4_soa = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[6] = -th_2 + th_2th_4_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_33_solve_th_4_processor()
    # Finish code for explicit solution node 32
    
    # Code for explicit solution node 12, solved variable is th_1
    def ExplicitSolutionNode_node_12_solve_th_1_processor():
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
            condition_0: bool = (abs(Pz - d_3*r_33) >= zero_tolerance) or (abs(d_1 - d_2) >= zero_tolerance) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) - d_3*(r_13*math.cos(th_0) + r_23*math.sin(th_0))) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = (-d_1 + d_2)**(-1)
                x1 = math.cos(th_0)
                x2 = math.sin(th_0)
                # End of temp variables
                this_solution[2] = math.atan2(x0*(Px*x1 + Py*x2 - d_3*(r_13*x1 + r_23*x2)), x0*(Pz - d_3*r_33))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(13, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_12_solve_th_1_processor()
    # Finish code for explicit solution node 12
    
    # Code for non-branch dispatcher node 13
    # Actually, there is no code
    
    # Code for explicit solution node 14, solved variable is th_5
    def ExplicitSolutionNode_node_14_solve_th_5_processor():
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
            condition_0: bool = (abs((d_3 + inv_Pz)/(d_1 - d_2)) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.acos((d_3 + inv_Pz)/(-d_1 + d_2))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[8] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(15, appended_idx)
                
            condition_1: bool = (abs((d_3 + inv_Pz)/(d_1 - d_2)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.acos((d_3 + inv_Pz)/(-d_1 + d_2))
                # End of temp variables
                this_solution[8] = -x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(15, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_5_processor()
    # Finish code for explicit solution node 13
    
    # Code for solved_variable dispatcher node 15
    def SolvedVariableDispatcherNode_node_15_processor():
        this_node_input_index: List[int] = node_input_index[15]
        this_input_valid: bool = node_input_validity[15]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_5 = this_solution[8]
            degenerate_valid_0 = (abs(th_5) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(44, node_input_i_idx_in_queue)
            
            th_5 = this_solution[8]
            degenerate_valid_1 = (abs(th_5 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(51, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(16, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_15_processor()
    # Finish code for solved_variable dispatcher node 15
    
    # Code for explicit solution node 51, solved variable is th_4th_6_soa
    def ExplicitSolutionNode_node_51_solve_th_4th_6_soa_processor():
        this_node_input_index: List[int] = node_input_index[51]
        this_input_valid: bool = node_input_validity[51]
        if not this_input_valid:
            return
        
        # The solution of non-root node 51
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_3 = this_solution[5]
            condition_0: bool = (abs(inv_Px) >= zero_tolerance) or (abs(inv_Py) >= zero_tolerance) or (abs(d_1*math.sin(th_3)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 1/(d_1*math.sin(th_3))
                # End of temp variables
                this_solution[7] = math.atan2(inv_Py*x0, -inv_Px*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(52, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_51_solve_th_4th_6_soa_processor()
    # Finish code for explicit solution node 51
    
    # Code for solved_variable dispatcher node 52
    def SolvedVariableDispatcherNode_node_52_processor():
        this_node_input_index: List[int] = node_input_index[52]
        this_input_valid: bool = node_input_validity[52]
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
            
            th_3 = this_solution[5]
            degenerate_valid_1 = (abs(th_3 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
            
            if not taken_by_degenerate:
                add_input_index_to(53, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_52_processor()
    # Finish code for solved_variable dispatcher node 52
    
    # Code for explicit solution node 53, solved variable is th_2
    def ExplicitSolutionNode_node_53_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[53]
        this_input_valid: bool = node_input_validity[53]
        if not this_input_valid:
            return
        
        # The solution of non-root node 53
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_3 = this_solution[5]
            condition_0: bool = (abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) >= zero_tolerance) or (abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance) or (abs(math.sin(th_3)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_3)**(-1)
                x1 = math.cos(th_0)
                x2 = math.sin(th_0)
                x3 = math.cos(th_1)
                # End of temp variables
                this_solution[3] = math.atan2(x0*(-r_13*x2 + r_23*x1), x0*(-r_13*x1*x3 - r_23*x2*x3 + r_33*math.sin(th_1)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(54, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_53_solve_th_2_processor()
    # Finish code for explicit solution node 53
    
    # Code for non-branch dispatcher node 54
    # Actually, there is no code
    
    # Code for explicit solution node 55, solved variable is th_4
    def ExplicitSolutionNode_node_55_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[54]
        this_input_valid: bool = node_input_validity[54]
        if not this_input_valid:
            return
        
        # The solution of non-root node 55
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[6] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(56, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_55_solve_th_4_processor()
    # Finish code for explicit solution node 54
    
    # Code for non-branch dispatcher node 56
    # Actually, there is no code
    
    # Code for explicit solution node 57, solved variable is th_6
    def ExplicitSolutionNode_node_57_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[56]
        this_input_valid: bool = node_input_validity[56]
        if not this_input_valid:
            return
        
        # The solution of non-root node 57
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_4 = this_solution[6]
            th_4th_6_soa = this_solution[7]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[9] = -th_4 + th_4th_6_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_57_solve_th_6_processor()
    # Finish code for explicit solution node 56
    
    # Code for explicit solution node 44, solved variable is negative_th_6_positive_th_4__soa
    def ExplicitSolutionNode_node_44_solve_negative_th_6_positive_th_4__soa_processor():
        this_node_input_index: List[int] = node_input_index[44]
        this_input_valid: bool = node_input_validity[44]
        if not this_input_valid:
            return
        
        # The solution of non-root node 44
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_3 = this_solution[5]
            condition_0: bool = (abs(inv_Px) >= zero_tolerance) or (abs(inv_Py) >= zero_tolerance) or (abs(d_1*math.sin(th_3)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 1/(d_1*math.sin(th_3))
                # End of temp variables
                this_solution[1] = math.atan2(inv_Py*x0, inv_Px*x0)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(45, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_44_solve_negative_th_6_positive_th_4__soa_processor()
    # Finish code for explicit solution node 44
    
    # Code for solved_variable dispatcher node 45
    def SolvedVariableDispatcherNode_node_45_processor():
        this_node_input_index: List[int] = node_input_index[45]
        this_input_valid: bool = node_input_validity[45]
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
            
            th_3 = this_solution[5]
            degenerate_valid_1 = (abs(th_3 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
            
            if not taken_by_degenerate:
                add_input_index_to(46, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_45_processor()
    # Finish code for solved_variable dispatcher node 45
    
    # Code for explicit solution node 46, solved variable is th_2
    def ExplicitSolutionNode_node_46_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[46]
        this_input_valid: bool = node_input_validity[46]
        if not this_input_valid:
            return
        
        # The solution of non-root node 46
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_3 = this_solution[5]
            condition_0: bool = (abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) >= zero_tolerance) or (abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance) or (abs(math.sin(th_3)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_3)**(-1)
                x1 = math.sin(th_0)
                x2 = math.cos(th_0)
                x3 = math.cos(th_1)
                # End of temp variables
                this_solution[3] = math.atan2(x0*(r_13*x1 - r_23*x2), x0*(r_13*x2*x3 + r_23*x1*x3 - r_33*math.sin(th_1)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(47, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_46_solve_th_2_processor()
    # Finish code for explicit solution node 46
    
    # Code for non-branch dispatcher node 47
    # Actually, there is no code
    
    # Code for explicit solution node 48, solved variable is th_4
    def ExplicitSolutionNode_node_48_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[47]
        this_input_valid: bool = node_input_validity[47]
        if not this_input_valid:
            return
        
        # The solution of non-root node 48
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[6] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(49, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_48_solve_th_4_processor()
    # Finish code for explicit solution node 47
    
    # Code for non-branch dispatcher node 49
    # Actually, there is no code
    
    # Code for explicit solution node 50, solved variable is th_6
    def ExplicitSolutionNode_node_50_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[49]
        this_input_valid: bool = node_input_validity[49]
        if not this_input_valid:
            return
        
        # The solution of non-root node 50
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            negative_th_6_positive_th_4__soa = this_solution[1]
            th_4 = this_solution[6]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[9] = -negative_th_6_positive_th_4__soa + th_4
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_50_solve_th_6_processor()
    # Finish code for explicit solution node 49
    
    # Code for explicit solution node 16, solved variable is th_6
    def ExplicitSolutionNode_node_16_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[16]
        this_input_valid: bool = node_input_validity[16]
        if not this_input_valid:
            return
        
        # The solution of non-root node 16
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_5 = this_solution[8]
            condition_0: bool = (abs(inv_Px) >= zero_tolerance) or (abs(inv_Py) >= zero_tolerance) or (abs((d_1 - d_2)*math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_5)**(-1)
                # End of temp variables
                this_solution[9] = math.atan2(inv_Py*x0/(d_1 - d_2), inv_Px*x0/(-d_1 + d_2))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(17, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_6_processor()
    # Finish code for explicit solution node 16
    
    # Code for solved_variable dispatcher node 17
    def SolvedVariableDispatcherNode_node_17_processor():
        this_node_input_index: List[int] = node_input_index[17]
        this_input_valid: bool = node_input_validity[17]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            taken_by_degenerate: bool = False
            th_5 = this_solution[8]
            degenerate_valid_0 = (abs(th_5) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
            
            th_5 = this_solution[8]
            degenerate_valid_1 = (abs(th_5 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
            
            if not taken_by_degenerate:
                add_input_index_to(18, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_17_processor()
    # Finish code for solved_variable dispatcher node 17
    
    # Code for explicit solution node 18, solved variable is negative_th_4_positive_th_2__soa
    def ExplicitSolutionNode_node_18_solve_negative_th_4_positive_th_2__soa_processor():
        this_node_input_index: List[int] = node_input_index[18]
        this_input_valid: bool = node_input_validity[18]
        if not this_input_valid:
            return
        
        # The solution of non-root node 18
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_5 = this_solution[8]
            condition_0: bool = (abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) >= zero_tolerance) or (abs(r_13*math.cos(th_0)*math.cos(th_1) + r_23*math.sin(th_0)*math.cos(th_1) - r_33*math.sin(th_1)) >= zero_tolerance) or (abs(math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_5)**(-1)
                x1 = math.cos(th_0)
                x2 = math.sin(th_0)
                x3 = math.cos(th_1)
                # End of temp variables
                this_solution[0] = math.atan2(x0*(-r_13*x2 + r_23*x1), x0*(-r_13*x1*x3 - r_23*x2*x3 + r_33*math.sin(th_1)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(19, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_18_solve_negative_th_4_positive_th_2__soa_processor()
    # Finish code for explicit solution node 18
    
    # Code for non-branch dispatcher node 19
    # Actually, there is no code
    
    # Code for explicit solution node 20, solved variable is th_2
    def ExplicitSolutionNode_node_20_solve_th_2_processor():
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
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[3] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(21, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_20_solve_th_2_processor()
    # Finish code for explicit solution node 19
    
    # Code for non-branch dispatcher node 21
    # Actually, there is no code
    
    # Code for explicit solution node 22, solved variable is th_4
    def ExplicitSolutionNode_node_22_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[21]
        this_input_valid: bool = node_input_validity[21]
        if not this_input_valid:
            return
        
        # The solution of non-root node 22
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            negative_th_4_positive_th_2__soa = this_solution[0]
            th_2 = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[6] = -negative_th_4_positive_th_2__soa + th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_22_solve_th_4_processor()
    # Finish code for explicit solution node 21
    
    # Code for explicit solution node 3, solved variable is th_2
    def ExplicitSolutionNode_node_3_solve_th_2_processor():
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
            condition_0: bool = (abs((-Px*math.sin(th_0) + Py*math.cos(th_0) + d_3*(r_13*math.sin(th_0) - r_23*math.cos(th_0)))/(d_2*math.sin(th_3))) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = math.asin((-Px*x1 + Py*x0 - d_3*(-r_13*x1 + r_23*x0))/(d_2*math.sin(th_3)))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs((-Px*math.sin(th_0) + Py*math.cos(th_0) + d_3*(r_13*math.sin(th_0) - r_23*math.cos(th_0)))/(d_2*math.sin(th_3))) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.cos(th_0)
                x1 = math.sin(th_0)
                x2 = math.asin((-Px*x1 + Py*x0 - d_3*(-r_13*x1 + r_23*x0))/(d_2*math.sin(th_3)))
                # End of temp variables
                this_solution[3] = math.pi - x2
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_2_processor()
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
            checked_result: bool = (abs(Pz - d_3*r_33) <= 1.0e-6) and (abs(Px*math.cos(th_0) + Py*math.sin(th_0) - d_3*r_13*math.cos(th_0) - d_3*r_23*math.sin(th_0)) <= 1.0e-6)
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
            th_2 = this_solution[3]
            th_3 = this_solution[5]
            condition_0: bool = (abs(Pz - d_3*r_33) >= 1.0e-6) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) - d_3*r_13*math.cos(th_0) - d_3*r_23*math.sin(th_0)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = Pz - d_3*r_33
                x1 = d_2*math.sin(th_3)*math.cos(th_2)
                x2 = -d_1 + d_2*math.cos(th_3)
                x3 = math.cos(th_0)
                x4 = math.sin(th_0)
                x5 = -Px*x3 - Py*x4 + d_3*r_13*x3 + d_3*r_23*x4
                # End of temp variables
                this_solution[2] = math.atan2(x0*x1 - x2*x5, x0*x2 + x1*x5)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_1_processor()
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
            th_1 = this_solution[2]
            condition_0: bool = (abs((-d_1*(r_33*math.cos(th_1) + (r_13*math.cos(th_0) + r_23*math.sin(th_0))*math.sin(th_1)) + d_3 + inv_Pz)/d_2) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.acos((d_1*(-r_33*math.cos(th_1) - (r_13*math.cos(th_0) + r_23*math.sin(th_0))*math.sin(th_1)) + d_3 + inv_Pz)/d_2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[8] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
            condition_1: bool = (abs((-d_1*(r_33*math.cos(th_1) + (r_13*math.cos(th_0) + r_23*math.sin(th_0))*math.sin(th_1)) + d_3 + inv_Pz)/d_2) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.acos((d_1*(-r_33*math.cos(th_1) - (r_13*math.cos(th_0) + r_23*math.sin(th_0))*math.sin(th_1)) + d_3 + inv_Pz)/d_2)
                # End of temp variables
                this_solution[8] = -x0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_5_processor()
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
            th_5 = this_solution[8]
            degenerate_valid_0 = (abs(th_5) <= 1.0e-6)
            if degenerate_valid_0:
                taken_by_degenerate = True
                add_input_index_to(34, node_input_i_idx_in_queue)
            
            th_5 = this_solution[8]
            degenerate_valid_1 = (abs(th_5 - math.pi) <= 1.0e-6)
            if degenerate_valid_1:
                taken_by_degenerate = True
                add_input_index_to(39, node_input_i_idx_in_queue)
            
            if not taken_by_degenerate:
                add_input_index_to(9, node_input_i_idx_in_queue)
    
    # Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor()
    # Finish code for solved_variable dispatcher node 8
    
    # Code for explicit solution node 39, solved variable is th_4th_6_soa
    def ExplicitSolutionNode_node_39_solve_th_4th_6_soa_processor():
        this_node_input_index: List[int] = node_input_index[39]
        this_input_valid: bool = node_input_validity[39]
        if not this_input_valid:
            return
        
        # The solution of non-root node 39
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_2 = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_21*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_31*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance) or (abs(r_12*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_22*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_32*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_2)
                x1 = x0*math.sin(th_1)
                x2 = math.sin(th_0)
                x3 = math.cos(th_2)
                x4 = math.cos(th_0)
                x5 = x0*math.cos(th_1)
                x6 = x2*x3 - x4*x5
                x7 = x2*x5 + x3*x4
                # End of temp variables
                this_solution[7] = math.atan2(r_11*x6 - r_21*x7 + r_31*x1, r_12*x6 - r_22*x7 + r_32*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(40, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_39_solve_th_4th_6_soa_processor()
    # Finish code for explicit solution node 39
    
    # Code for non-branch dispatcher node 40
    # Actually, there is no code
    
    # Code for explicit solution node 41, solved variable is th_4
    def ExplicitSolutionNode_node_41_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[40]
        this_input_valid: bool = node_input_validity[40]
        if not this_input_valid:
            return
        
        # The solution of non-root node 41
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[6] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(42, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_41_solve_th_4_processor()
    # Finish code for explicit solution node 40
    
    # Code for non-branch dispatcher node 42
    # Actually, there is no code
    
    # Code for explicit solution node 43, solved variable is th_6
    def ExplicitSolutionNode_node_43_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[42]
        this_input_valid: bool = node_input_validity[42]
        if not this_input_valid:
            return
        
        # The solution of non-root node 43
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_4 = this_solution[6]
            th_4th_6_soa = this_solution[7]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[9] = -th_4 + th_4th_6_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_43_solve_th_6_processor()
    # Finish code for explicit solution node 42
    
    # Code for explicit solution node 34, solved variable is negative_th_6_positive_th_4__soa
    def ExplicitSolutionNode_node_34_solve_negative_th_6_positive_th_4__soa_processor():
        this_node_input_index: List[int] = node_input_index[34]
        this_input_valid: bool = node_input_validity[34]
        if not this_input_valid:
            return
        
        # The solution of non-root node 34
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_1 = this_solution[2]
            th_2 = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_21*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_31*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance) or (abs(r_12*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_22*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_32*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_2)
                x1 = x0*math.sin(th_1)
                x2 = math.cos(th_0)
                x3 = math.cos(th_2)
                x4 = math.sin(th_0)
                x5 = x0*math.cos(th_1)
                x6 = x2*x3 + x4*x5
                x7 = -x2*x5 + x3*x4
                # End of temp variables
                this_solution[1] = math.atan2(-r_11*x7 + r_21*x6 - r_31*x1, r_12*x7 - r_22*x6 + r_32*x1)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(35, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_34_solve_negative_th_6_positive_th_4__soa_processor()
    # Finish code for explicit solution node 34
    
    # Code for non-branch dispatcher node 35
    # Actually, there is no code
    
    # Code for explicit solution node 36, solved variable is th_4
    def ExplicitSolutionNode_node_36_solve_th_4_processor():
        this_node_input_index: List[int] = node_input_index[35]
        this_input_valid: bool = node_input_validity[35]
        if not this_input_valid:
            return
        
        # The solution of non-root node 36
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            condition_0: bool = True
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[6] = 0
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(37, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_36_solve_th_4_processor()
    # Finish code for explicit solution node 35
    
    # Code for non-branch dispatcher node 37
    # Actually, there is no code
    
    # Code for explicit solution node 38, solved variable is th_6
    def ExplicitSolutionNode_node_38_solve_th_6_processor():
        this_node_input_index: List[int] = node_input_index[37]
        this_input_valid: bool = node_input_validity[37]
        if not this_input_valid:
            return
        
        # The solution of non-root node 38
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            negative_th_6_positive_th_4__soa = this_solution[1]
            th_4 = this_solution[6]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[9] = -negative_th_6_positive_th_4__soa + th_4
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_38_solve_th_6_processor()
    # Finish code for explicit solution node 37
    
    # Code for explicit solution node 9, solved variable is th_6
    def ExplicitSolutionNode_node_9_solve_th_6_processor():
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
            th_1 = this_solution[2]
            th_5 = this_solution[8]
            condition_0: bool = (abs(d_2*math.sin(th_5)) >= zero_tolerance) or (abs(-d_1*(-r_31*math.cos(th_1) - (r_11*math.cos(th_0) + r_21*math.sin(th_0))*math.sin(th_1)) - inv_Px) >= zero_tolerance) or (abs(-d_1*(-r_32*math.cos(th_1) - (r_12*math.cos(th_0) + r_22*math.sin(th_0))*math.sin(th_1)) - inv_Py) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_1)
                x1 = math.sin(th_1)
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = 1/(d_2*math.sin(th_5))
                # End of temp variables
                this_solution[9] = math.atan2(x4*(-d_1*(-r_32*x0 - x1*(r_12*x2 + r_22*x3)) - inv_Py), x4*(d_1*(-r_31*x0 - x1*(r_11*x2 + r_21*x3)) + inv_Px))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_6_processor()
    # Finish code for explicit solution node 9
    
    # Code for non-branch dispatcher node 10
    # Actually, there is no code
    
    # Code for explicit solution node 11, solved variable is th_4
    def ExplicitSolutionNode_node_11_solve_th_4_processor():
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
            th_1 = this_solution[2]
            th_2 = this_solution[3]
            th_3 = this_solution[5]
            th_5 = this_solution[8]
            condition_0: bool = (abs(-r_13*((math.sin(th_1)*math.sin(th_3) + math.cos(th_1)*math.cos(th_2)*math.cos(th_3))*math.cos(th_0) + math.sin(th_0)*math.sin(th_2)*math.cos(th_3)) - r_23*((math.sin(th_1)*math.sin(th_3) + math.cos(th_1)*math.cos(th_2)*math.cos(th_3))*math.sin(th_0) - math.sin(th_2)*math.cos(th_0)*math.cos(th_3)) - r_33*(-math.sin(th_1)*math.cos(th_2)*math.cos(th_3) + math.sin(th_3)*math.cos(th_1))) >= zero_tolerance) or (abs(r_13*(math.sin(th_0)*math.cos(th_2) - math.sin(th_2)*math.cos(th_0)*math.cos(th_1)) - r_23*(math.sin(th_0)*math.sin(th_2)*math.cos(th_1) + math.cos(th_0)*math.cos(th_2)) + r_33*math.sin(th_1)*math.sin(th_2)) >= zero_tolerance) or (abs(math.sin(th_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_5)**(-1)
                x1 = math.sin(th_1)
                x2 = math.sin(th_2)
                x3 = math.sin(th_0)
                x4 = math.cos(th_2)
                x5 = math.cos(th_0)
                x6 = math.cos(th_1)
                x7 = x2*x6
                x8 = math.sin(th_3)
                x9 = math.cos(th_3)
                x10 = x4*x9
                x11 = x2*x9
                x12 = x1*x8 + x10*x6
                # End of temp variables
                this_solution[6] = math.atan2(x0*(r_13*(x3*x4 - x5*x7) - r_23*(x3*x7 + x4*x5) + r_33*x1*x2), x0*(-r_13*(x11*x3 + x12*x5) - r_23*(-x11*x5 + x12*x3) - r_33*(-x1*x10 + x6*x8)))
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_4_processor()
    # Finish code for explicit solution node 10
    
    # Collect the output
    ik_out: List[np.ndarray] = list()
    for i in range(len(solution_queue)):
        if not queue_element_validity[i]:
            continue
        ik_out_i = solution_queue[i]
        new_ik_i = np.zeros((robot_nq, 1))
        value_at_0 = th_0  # th_0
        new_ik_i[0] = value_at_0
        value_at_1 = ik_out_i[2]  # th_1
        new_ik_i[1] = value_at_1
        value_at_2 = ik_out_i[3]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[5]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[6]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[8]  # th_5
        new_ik_i[5] = value_at_5
        value_at_6 = ik_out_i[9]  # th_6
        new_ik_i[6] = value_at_6
        ik_out.append(new_ik_i)
    return ik_out


def kuka_iiwa_ik_solve(T_ee: np.ndarray, th_0):
    T_ee_raw_in = kuka_iiwa_ik_target_original_to_raw(T_ee)
    ik_output_raw = kuka_iiwa_ik_solve_raw(T_ee_raw_in, th_0)
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
        ee_pose_i = kuka_iiwa_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_kuka_iiwa():
    theta_in = np.random.random(size=(7, ))
    ee_pose = kuka_iiwa_fk(theta_in)
    ik_output = kuka_iiwa_ik_solve(ee_pose, th_0=theta_in[0])
    for i in range(len(ik_output)):
        ee_pose_i = kuka_iiwa_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_kuka_iiwa()
