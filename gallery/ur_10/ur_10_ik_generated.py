import numpy as np
import copy
import math
from typing import List, NewType

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 18
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_1: float = 0.612
a_3: float = 0.5723
d_0: float = 0.220941
d_2: float = -0.1719
d_4: float = 0.11485899999999999
d_5: float = 0.1157
d_6: float = 0.0922
pre_transform_special_symbol_23: float = 0.1273

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -0.0
th_2_offset_original2raw: float = 0.0
th_3_offset_original2raw: float = -0.0
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 0.0


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def ur_10_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = 1.0*r_11
    ee_transformed[0, 1] = 1.0*r_12 + 6.1232339957367697e-17*r_13
    ee_transformed[0, 2] = 6.1232339957367697e-17*r_12 - 1.0*r_13
    ee_transformed[0, 3] = -1.0*Px
    ee_transformed[1, 0] = 1.0*r_21
    ee_transformed[1, 1] = 1.0*r_22 + 6.1232339957367697e-17*r_23
    ee_transformed[1, 2] = 6.1232339957367697e-17*r_22 - 1.0*r_23
    ee_transformed[1, 3] = -1.0*Py
    ee_transformed[2, 0] = -1.0*r_31
    ee_transformed[2, 1] = -1.0*r_32 - 6.1232339957367697e-17*r_33
    ee_transformed[2, 2] = -6.1232339957367697e-17*r_32 + 1.0*r_33
    ee_transformed[2, 3] = 1.0*Pz - 1.0*pre_transform_special_symbol_23
    return ee_transformed


def ur_10_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = 1.0*r_11
    ee_transformed[0, 1] = 1.0*r_12 + 6.1232339957367697e-17*r_13
    ee_transformed[0, 2] = 6.1232339957367697e-17*r_12 - 1.0*r_13
    ee_transformed[0, 3] = -1.0*Px
    ee_transformed[1, 0] = 1.0*r_21
    ee_transformed[1, 1] = 1.0*r_22 + 6.1232339957367697e-17*r_23
    ee_transformed[1, 2] = 6.1232339957367697e-17*r_22 - 1.0*r_23
    ee_transformed[1, 3] = -1.0*Py
    ee_transformed[2, 0] = -1.0*r_31
    ee_transformed[2, 1] = -1.0*r_32 - 6.1232339957367697e-17*r_33
    ee_transformed[2, 2] = -6.1232339957367697e-17*r_32 + 1.0*r_33
    ee_transformed[2, 3] = 1.0*Pz + 1.0*pre_transform_special_symbol_23
    return ee_transformed


def ur_10_fk(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_5)
    x1 = math.sin(th_3)
    x2 = math.cos(th_0)
    x3 = math.cos(th_1)
    x4 = math.cos(th_2)
    x5 = x3*x4
    x6 = math.sin(th_1)
    x7 = math.sin(th_2)
    x8 = x6*x7
    x9 = x2*x5 - x2*x8
    x10 = math.cos(th_3)
    x11 = x4*x6
    x12 = x3*x7
    x13 = -x11*x2 - x12*x2
    x14 = -x1*x9 + x10*x13
    x15 = 1.0*x14
    x16 = math.cos(th_5)
    x17 = math.sin(th_0)
    x18 = math.sin(th_4)
    x19 = math.cos(th_4)
    x20 = x1*x13 + x10*x9
    x21 = x17*x18 + x19*x20
    x22 = 1.0*x21
    x23 = x17*x19
    x24 = x18*x20
    x25 = 6.12323399573677e-17*x16
    x26 = 6.12323399573677e-17*x0
    x27 = 1.0*x17
    x28 = 1.0*x2
    x29 = a_1*x3
    x30 = 1.0*a_3
    x31 = 1.0*d_6
    x32 = x17*x5 - x17*x8
    x33 = -x11*x17 - x12*x17
    x34 = -x1*x32 + x10*x33
    x35 = 1.0*x34
    x36 = x1*x33 + x10*x32
    x37 = -x18*x2 + x19*x36
    x38 = 1.0*x37
    x39 = x19*x2
    x40 = x18*x36
    x41 = -x5 + x8
    x42 = -x11 - x12
    x43 = -x1*x42 + x10*x41
    x44 = 1.0*x43
    x45 = x1*x41 + x10*x42
    x46 = x19*x45
    x47 = 1.0*x46
    x48 = x18*x45
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -x0*x15 + x16*x22
    ee_pose[0, 1] = -x0*x22 - x15*x16 + 6.12323399573677e-17*x23 - 6.12323399573677e-17*x24
    ee_pose[0, 2] = -x14*x25 - x21*x26 - 1.0*x23 + 1.0*x24
    ee_pose[0, 3] = d_0*x27 + d_2*x27 + d_4*x27 - d_5*x15 - x28*x29 - x30*x9 - x31*(x23 - x24)
    ee_pose[1, 0] = -x0*x35 + x16*x38
    ee_pose[1, 1] = -x0*x38 - x16*x35 - 6.12323399573677e-17*x39 - 6.12323399573677e-17*x40
    ee_pose[1, 2] = -x25*x34 - x26*x37 + 1.0*x39 + 1.0*x40
    ee_pose[1, 3] = -d_0*x28 - d_2*x28 - d_4*x28 - d_5*x35 - x27*x29 - x30*x32 - x31*(-x39 - x40)
    ee_pose[2, 0] = x0*x44 - x16*x47
    ee_pose[2, 1] = x0*x47 + x16*x44 + 6.12323399573677e-17*x48
    ee_pose[2, 2] = x25*x43 + x26*x46 - 1.0*x48
    ee_pose[2, 3] = -1.0*a_1*x6 + d_5*x44 + 1.0*pre_transform_special_symbol_23 + x30*x42 - x31*x48
    return ee_pose


def ur_10_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = 1.0*math.sin(th_0)
    x1 = math.cos(th_3)
    x2 = math.sin(th_1)
    x3 = math.cos(th_2)
    x4 = 1.0*math.cos(th_0)
    x5 = x3*x4
    x6 = math.cos(th_1)
    x7 = math.sin(th_2)
    x8 = x4*x7
    x9 = x2*x5 + x6*x8
    x10 = math.sin(th_3)
    x11 = x2*x8 - x5*x6
    x12 = x1*x9 - x10*x11
    x13 = math.cos(th_4)
    x14 = math.sin(th_4)
    x15 = -x0*x13 - x14*(x1*x11 + x10*x9)
    x16 = -x4
    x17 = x0*x3
    x18 = x0*x7
    x19 = x17*x2 + x18*x6
    x20 = -x17*x6 + x18*x2
    x21 = x1*x19 - x10*x20
    x22 = x13*x4 - x14*(x1*x20 + x10*x19)
    x23 = 1.0*x7
    x24 = 1.0*x3
    x25 = x2*x23 - x24*x6
    x26 = -x2*x24 - x23*x6
    x27 = x1*x25 - x10*x26
    x28 = x14*(x1*x26 + x10*x25)
    x29 = -1.0*a_1*x2 + pre_transform_special_symbol_23
    x30 = a_3*x26 + x29
    x31 = d_5*x27 + x30
    x32 = a_1*x6
    x33 = -d_0*x4 - d_2*x4 - x0*x32
    x34 = a_3*x20 - d_4*x4 + x33
    x35 = d_5*x21 + x34
    x36 = -d_6*x28 + x31
    x37 = d_6*x22 + x35
    x38 = d_0*x0 + d_2*x0 - x32*x4
    x39 = a_3*x11 + d_4*x0 + x38
    x40 = d_5*x12 + x39
    x41 = d_6*x15 + x40
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = x0
    jacobian_output[0, 2] = x0
    jacobian_output[0, 3] = x0
    jacobian_output[0, 4] = x12
    jacobian_output[0, 5] = x15
    jacobian_output[1, 1] = x16
    jacobian_output[1, 2] = x16
    jacobian_output[1, 3] = x16
    jacobian_output[1, 4] = x21
    jacobian_output[1, 5] = x22
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 4] = x27
    jacobian_output[2, 5] = -x28
    jacobian_output[3, 1] = pre_transform_special_symbol_23*x4
    jacobian_output[3, 2] = x29*x4
    jacobian_output[3, 3] = x30*x4
    jacobian_output[3, 4] = -x21*x31 + x27*x35
    jacobian_output[3, 5] = -x22*x36 - x28*x37
    jacobian_output[4, 1] = pre_transform_special_symbol_23*x0
    jacobian_output[4, 2] = x0*x29
    jacobian_output[4, 3] = x0*x30
    jacobian_output[4, 4] = x12*x31 - x27*x40
    jacobian_output[4, 5] = x15*x36 + x28*x41
    jacobian_output[5, 2] = -x0*x33 - x38*x4
    jacobian_output[5, 3] = -x0*x34 - x39*x4
    jacobian_output[5, 4] = -x12*x35 + x21*x40
    jacobian_output[5, 5] = -x15*x37 + x22*x41
    return jacobian_output


def ur_10_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = 1.0*math.sin(th_0)
    x1 = math.cos(th_3)
    x2 = math.sin(th_1)
    x3 = math.cos(th_2)
    x4 = 1.0*math.cos(th_0)
    x5 = x3*x4
    x6 = math.cos(th_1)
    x7 = math.sin(th_2)
    x8 = x4*x7
    x9 = x2*x5 + x6*x8
    x10 = math.sin(th_3)
    x11 = x2*x8 - x5*x6
    x12 = math.cos(th_4)
    x13 = math.sin(th_4)
    x14 = -x4
    x15 = x0*x3
    x16 = x0*x7
    x17 = x15*x2 + x16*x6
    x18 = -x15*x6 + x16*x2
    x19 = 1.0*x7
    x20 = 1.0*x3
    x21 = x19*x2 - x20*x6
    x22 = -x19*x6 - x2*x20
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 1] = x0
    jacobian_output[0, 2] = x0
    jacobian_output[0, 3] = x0
    jacobian_output[0, 4] = x1*x9 - x10*x11
    jacobian_output[0, 5] = -x0*x12 - x13*(x1*x11 + x10*x9)
    jacobian_output[1, 1] = x14
    jacobian_output[1, 2] = x14
    jacobian_output[1, 3] = x14
    jacobian_output[1, 4] = x1*x17 - x10*x18
    jacobian_output[1, 5] = x12*x4 - x13*(x1*x18 + x10*x17)
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 4] = x1*x21 - x10*x22
    jacobian_output[2, 5] = -x13*(x1*x22 + x10*x21)
    return jacobian_output


def ur_10_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x3 = -p_on_ee_z*x2
    x4 = math.sin(th_1)
    x5 = 1.0*x4
    x6 = -a_1*x5 + pre_transform_special_symbol_23
    x7 = math.cos(th_2)
    x8 = x5*x7
    x9 = math.sin(th_2)
    x10 = math.cos(th_1)
    x11 = 1.0*x10
    x12 = x11*x9
    x13 = -x12 - x8
    x14 = a_3*x13 + x6
    x15 = math.cos(th_3)
    x16 = x5*x9
    x17 = x11*x7
    x18 = x16 - x17
    x19 = math.sin(th_3)
    x20 = -x13*x19 + x15*x18
    x21 = math.sin(th_0)
    x22 = x12*x21 + x21*x8
    x23 = x16*x21 - x17*x21
    x24 = x15*x22 - x19*x23
    x25 = d_5*x20 + x14
    x26 = -a_1*x11*x21 - d_0*x2 - d_2*x2
    x27 = a_3*x23 - d_4*x2 + x26
    x28 = d_5*x24 + x27
    x29 = math.sin(th_4)
    x30 = x29*(x13*x15 + x18*x19)
    x31 = math.cos(th_4)
    x32 = x2*x31 - x29*(x15*x23 + x19*x22)
    x33 = -d_6*x30 + x25
    x34 = d_6*x32 + x28
    x35 = 1.0*p_on_ee_x
    x36 = 1.0*x21
    x37 = -p_on_ee_z*x36
    x38 = x2*x7
    x39 = x2*x9
    x40 = x10*x39 + x38*x4
    x41 = -x10*x38 + x39*x4
    x42 = x15*x40 - x19*x41
    x43 = -a_1*x10*x2 + d_0*x36 + d_2*x36
    x44 = a_3*x41 + d_4*x36 + x43
    x45 = d_5*x42 + x44
    x46 = -x29*(x15*x41 + x19*x40) - x31*x36
    x47 = d_6*x46 + x45
    x48 = x0*x21 + x1*x35
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = x2*x6 + x3
    jacobian_output[0, 3] = x14*x2 + x3
    jacobian_output[0, 4] = -p_on_ee_y*x20 + p_on_ee_z*x24 + x20*x28 - x24*x25
    jacobian_output[0, 5] = p_on_ee_y*x30 + p_on_ee_z*x32 - x30*x34 - x32*x33
    jacobian_output[1, 0] = x35
    jacobian_output[1, 1] = pre_transform_special_symbol_23*x36 + x37
    jacobian_output[1, 2] = x36*x6 + x37
    jacobian_output[1, 3] = x14*x36 + x37
    jacobian_output[1, 4] = p_on_ee_x*x20 - p_on_ee_z*x42 - x20*x45 + x25*x42
    jacobian_output[1, 5] = -p_on_ee_x*x30 - p_on_ee_z*x46 + x30*x47 + x33*x46
    jacobian_output[2, 1] = x48
    jacobian_output[2, 2] = -x2*x43 - x26*x36 + x48
    jacobian_output[2, 3] = -x2*x44 - x27*x36 + x48
    jacobian_output[2, 4] = -p_on_ee_x*x24 + p_on_ee_y*x42 + x24*x45 - x28*x42
    jacobian_output[2, 5] = -p_on_ee_x*x32 + p_on_ee_y*x46 + x32*x47 - x34*x46
    return jacobian_output


def ur_10_ik_solve_raw(T_ee: np.ndarray):
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
        condition_0: bool = (abs(Px - d_6*r_13) >= zero_tolerance) or (abs(Py - d_6*r_23) >= zero_tolerance) or (abs(d_0 + d_2 + d_4) >= zero_tolerance)
        if condition_0:
            # Temp variable for efficiency
            x0 = Px - d_6*r_13
            x1 = -Py + d_6*r_23
            x2 = math.atan2(x0, x1)
            x3 = -d_0 - d_2 - d_4
            x4 = math.sqrt(x0**2 + x1**2 - x3**2)
            # End of temp variables
            solution_0: IkSolution = make_ik_solution()
            solution_0[0] = x2 + math.atan2(x4, x3)
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)
            
        condition_1: bool = (abs(Px - d_6*r_13) >= zero_tolerance) or (abs(Py - d_6*r_23) >= zero_tolerance) or (abs(d_0 + d_2 + d_4) >= zero_tolerance)
        if condition_1:
            # Temp variable for efficiency
            x0 = Px - d_6*r_13
            x1 = -Py + d_6*r_23
            x2 = math.atan2(x0, x1)
            x3 = -d_0 - d_2 - d_4
            x4 = math.sqrt(x0**2 + x1**2 - x3**2)
            # End of temp variables
            solution_1: IkSolution = make_ik_solution()
            solution_1[0] = x2 + math.atan2(-x4, x3)
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
                x0 = math.acos(r_13*math.sin(th_0) - r_23*math.cos(th_0))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[7] = x0
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.acos(r_13*math.sin(th_0) - r_23*math.cos(th_0))
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
            condition_0: bool = ((1/2)*abs((a_1**2 + a_3**2 + d_0**2 + 2*d_0*d_2 + 2*d_0*d_4 + d_2**2 + 2*d_2*d_4 + d_4**2 - d_5**2 + 2*d_5*inv_Px*math.sin(th_5) + 2*d_5*inv_Py*math.cos(th_5) - d_6**2 - 2*d_6*inv_Pz - inv_Px**2 - inv_Py**2 - inv_Pz**2)/(a_1*a_3)) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*d_0
                x1 = 2*d_5
                x2 = math.acos((1/2)*(-a_1**2 - a_3**2 - d_0**2 - d_2**2 - 2*d_2*d_4 - d_2*x0 - d_4**2 - d_4*x0 + d_5**2 + d_6**2 + 2*d_6*inv_Pz + inv_Px**2 - inv_Px*x1*math.sin(th_5) + inv_Py**2 - inv_Py*x1*math.cos(th_5) + inv_Pz**2)/(a_1*a_3))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x2
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(10, appended_idx)
                
            condition_1: bool = ((1/2)*abs((a_1**2 + a_3**2 + d_0**2 + 2*d_0*d_2 + 2*d_0*d_4 + d_2**2 + 2*d_2*d_4 + d_4**2 - d_5**2 + 2*d_5*inv_Px*math.sin(th_5) + 2*d_5*inv_Py*math.cos(th_5) - d_6**2 - 2*d_6*inv_Pz - inv_Px**2 - inv_Py**2 - inv_Pz**2)/(a_1*a_3)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*d_0
                x1 = 2*d_5
                x2 = math.acos((1/2)*(-a_1**2 - a_3**2 - d_0**2 - d_2**2 - 2*d_2*d_4 - d_2*x0 - d_4**2 - d_4*x0 + d_5**2 + d_6**2 + 2*d_6*inv_Pz + inv_Px**2 - inv_Px*x1*math.sin(th_5) + inv_Py**2 - inv_Py*x1*math.cos(th_5) + inv_Pz**2)/(a_1*a_3))
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
            checked_result: bool = (abs(Pz + d_5*r_31*math.sin(th_5) + d_5*r_32*math.cos(th_5) - d_6*r_33) <= 1.0e-6) and (abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_5*r_11*math.sin(th_5)*math.cos(th_0) + d_5*r_12*math.cos(th_0)*math.cos(th_5) + d_5*r_21*math.sin(th_0)*math.sin(th_5) + d_5*r_22*math.sin(th_0)*math.cos(th_5) - d_6*r_13*math.cos(th_0) - d_6*r_23*math.sin(th_0)) <= 1.0e-6)
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
            th_2 = this_solution[4]
            th_5 = this_solution[8]
            condition_0: bool = (abs(Pz + d_5*r_31*math.sin(th_5) + d_5*r_32*math.cos(th_5) - d_6*r_33) >= 1.0e-6) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) + d_5*r_11*math.sin(th_5)*math.cos(th_0) + d_5*r_12*math.cos(th_0)*math.cos(th_5) + d_5*r_21*math.sin(th_0)*math.sin(th_5) + d_5*r_22*math.sin(th_0)*math.cos(th_5) - d_6*r_13*math.cos(th_0) - d_6*r_23*math.sin(th_0)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = -a_1 - a_3*math.cos(th_2)
                x1 = d_5*math.sin(th_5)
                x2 = d_5*math.cos(th_5)
                x3 = Pz - d_6*r_33 + r_31*x1 + r_32*x2
                x4 = math.cos(th_0)
                x5 = math.sin(th_0)
                x6 = -Px*x4 - Py*x5 + d_6*r_13*x4 + d_6*r_23*x5 - r_11*x1*x4 - r_12*x2*x4 - r_21*x1*x5 - r_22*x2*x5
                x7 = a_3*math.sin(th_2)
                # End of temp variables
                this_solution[1] = math.atan2(x0*x3 + x6*x7, x0*x6 - x3*x7)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(12, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1_processor()
    # Finish code for explicit solution node 11
    
    # Code for non-branch dispatcher node 12
    # Actually, there is no code
    
    # Code for explicit solution node 13, solved variable is th_2th_3_soa
    def ExplicitSolutionNode_node_13_solve_th_2th_3_soa_processor():
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
            th_1 = this_solution[1]
            th_1th_2th_3_soa = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[5] = -th_1 + th_1th_2th_3_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(14, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_2th_3_soa_processor()
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
            th_2 = this_solution[4]
            th_2th_3_soa = this_solution[5]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[6] = -th_2 + th_2th_3_soa
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(16, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_15_solve_th_3_processor()
    # Finish code for explicit solution node 14
    
    # Code for non-branch dispatcher node 16
    # Actually, there is no code
    
    # Code for explicit solution node 17, solved variable is th_1th_2_soa
    def ExplicitSolutionNode_node_17_solve_th_1th_2_soa_processor():
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
            th_1 = this_solution[1]
            th_2 = this_solution[4]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[2] = th_1 + th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_1th_2_soa_processor()
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


def ur_10_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = ur_10_ik_target_original_to_raw(T_ee)
    ik_output_raw = ur_10_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ee_pose_i = ur_10_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_ur_10():
    theta_in = np.random.random(size=(6, ))
    ee_pose = ur_10_fk(theta_in)
    ik_output = ur_10_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = ur_10_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_ur_10()
