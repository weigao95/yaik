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
a_0: float = 0.21
a_1: float = 0.32729955698106283
d_2: float = 1.35
d_3: float = 0.343
pre_transform_special_symbol_23: float = 0.67

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -0.9700825434839029
th_2_offset_original2raw: float = -2.5408788702787994
th_3_offset_original2raw: float = 3.141592653589793
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 1.5707963267948968


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def kawasaki_BX250L_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = 6.1232339957367599e-17*r_13 + 1.0*r_23 + 5.5511151231257802e-17*r_33
    ee_transformed[0, 1] = -6.1232339957367599e-17*r_12 - 1.0*r_22 - 5.5511151231257802e-17*r_32
    ee_transformed[0, 2] = 6.1232339957367599e-17*r_11 + 1.0*r_21 + 5.5511151231257802e-17*r_31
    ee_transformed[0, 3] = 6.1232339957367599e-17*Px + 1.0*Py + 5.5511151231257802e-17*Pz - 5.5511151231257802e-17*pre_transform_special_symbol_23
    ee_transformed[1, 0] = 1.0*r_13 - 6.1232339957367697e-17*r_23
    ee_transformed[1, 1] = -1.0*r_12 + 6.1232339957367697e-17*r_22
    ee_transformed[1, 2] = 1.0*r_11 - 6.1232339957367697e-17*r_21
    ee_transformed[1, 3] = 1.0*Px - 6.1232339957367697e-17*Py
    ee_transformed[2, 0] = -3.3990776836172297e-33*r_13 - 5.5511151231257802e-17*r_23 - 1.0*r_33
    ee_transformed[2, 1] = 3.3990776836172297e-33*r_12 + 5.5511151231257802e-17*r_22 + 1.0*r_32
    ee_transformed[2, 2] = -3.3990776836172297e-33*r_11 - 5.5511151231257802e-17*r_21 - 1.0*r_31
    ee_transformed[2, 3] = -3.3990776836172297e-33*Px - 5.5511151231257802e-17*Py - 1.0*Pz + 1.0*pre_transform_special_symbol_23
    return ee_transformed


def kawasaki_BX250L_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = 6.1232339957367599e-17*r_13 + 1.0*r_23 - 3.3990776836172297e-33*r_33
    ee_transformed[0, 1] = -6.1232339957367599e-17*r_12 - 1.0*r_22 + 3.3990776836172297e-33*r_32
    ee_transformed[0, 2] = 6.1232339957367599e-17*r_11 + 1.0*r_21 - 3.3990776836172297e-33*r_31
    ee_transformed[0, 3] = 6.1232339957367599e-17*Px + 1.0*Py - 3.3990776836172297e-33*Pz
    ee_transformed[1, 0] = 1.0*r_13 - 6.1232339957367697e-17*r_23 - 5.5511151231257802e-17*r_33
    ee_transformed[1, 1] = -1.0*r_12 + 6.1232339957367697e-17*r_22 + 5.5511151231257802e-17*r_32
    ee_transformed[1, 2] = 1.0*r_11 - 6.1232339957367697e-17*r_21 - 5.5511151231257802e-17*r_31
    ee_transformed[1, 3] = 1.0*Px - 6.1232339957367697e-17*Py - 5.5511151231257802e-17*Pz
    ee_transformed[2, 0] = 5.5511151231257802e-17*r_13 - 1.0*r_33
    ee_transformed[2, 1] = -5.5511151231257802e-17*r_12 + 1.0*r_32
    ee_transformed[2, 2] = 5.5511151231257802e-17*r_11 - 1.0*r_31
    ee_transformed[2, 3] = 5.5511151231257802e-17*Px - 1.0*Pz + 1.0*pre_transform_special_symbol_23
    return ee_transformed


def kawasaki_BX250L_fk(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.cos(th_4)
    x1 = math.sin(th_1)
    x2 = math.sin(th_2)
    x3 = x1*x2
    x4 = math.cos(th_1)
    x5 = math.cos(th_2)
    x6 = x4*x5
    x7 = -x3 - x6
    x8 = math.sin(th_4)
    x9 = math.cos(th_3)
    x10 = x1*x5
    x11 = x2*x4
    x12 = x10 - x11
    x13 = math.sin(th_0)
    x14 = x10*x13 - x11*x13
    x15 = x0*x14
    x16 = math.cos(th_0)
    x17 = x10*x16 - x11*x16
    x18 = x0*x17
    x19 = math.sin(th_3)
    x20 = x16*x3 + x16*x6
    x21 = x13*x19 + x20*x9
    x22 = x21*x8
    x23 = x13*x3 + x13*x6
    x24 = -x16*x19 + x23*x9
    x25 = x24*x8
    x26 = math.cos(th_5)
    x27 = x12*x19
    x28 = 3.39907768361723e-33*x27
    x29 = x13*x9 - x19*x20
    x30 = 6.12323399573676e-17*x29
    x31 = -x16*x9 - x19*x23
    x32 = 1.0*x31
    x33 = math.sin(th_5)
    x34 = x0*x12*x9 - x7*x8
    x35 = 3.39907768361723e-33*x34
    x36 = x0*x21 - x17*x8
    x37 = x0*x24 - x14*x8
    x38 = a_0*x16
    x39 = a_1*x1
    x40 = a_1*x4
    x41 = x16*x40
    x42 = d_2*x7
    x43 = 1.0*d_2
    x44 = d_2*x17
    x45 = x0*x7
    x46 = x12*x8*x9
    x47 = -x45 - x46
    x48 = d_3*x47
    x49 = -x18 - x22
    x50 = d_3*x49
    x51 = -x15 - x25
    x52 = 1.0*d_3
    x53 = 6.12323399573677e-17*x26
    x54 = 5.55111512312578e-17*x34
    x55 = 6.12323399573677e-17*x13
    x56 = 1.0*x27
    x57 = 5.55111512312578e-17*x29
    x58 = 1.0*x34
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = 3.39907768361723e-33*x0*x7 + 3.39907768361723e-33*x12*x8*x9 - 1.0*x15 - 6.12323399573676e-17*x18 - 6.12323399573676e-17*x22 - 1.0*x25
    ee_pose[0, 1] = x26*x28 + x26*x30 + x26*x32 - x33*x35 + 6.12323399573676e-17*x33*x36 + 1.0*x33*x37
    ee_pose[0, 2] = -x26*x35 + 6.12323399573676e-17*x26*x36 + 1.0*x26*x37 - x28*x33 - x30*x33 - x32*x33
    ee_pose[0, 3] = 1.0*a_0*x13 + 1.0*x13*x40 + x14*x43 + 6.12323399573676e-17*x38 - 3.39907768361723e-33*x39 + 6.12323399573676e-17*x41 - 3.39907768361723e-33*x42 + 6.12323399573676e-17*x44 - 3.39907768361723e-33*x48 + 6.12323399573676e-17*x50 + x51*x52
    ee_pose[1, 0] = 6.12323399573677e-17*x15 - 1.0*x18 - 1.0*x22 + 6.12323399573677e-17*x25 + 5.55111512312578e-17*x45 + 5.55111512312578e-17*x46
    ee_pose[1, 1] = 5.55111512312578e-17*x12*x19*x26 + 1.0*x26*x29 - x31*x53 + 1.0*x33*x36 - 6.12323399573677e-17*x33*x37 - x33*x54
    ee_pose[1, 2] = 1.0*x26*x36 - x26*x54 - 5.55111512312578e-17*x27*x33 - 1.0*x29*x33 + 6.12323399573677e-17*x31*x33 - x37*x53
    ee_pose[1, 3] = 1.0*a_0*x16 - a_0*x55 + 1.0*a_1*x16*x4 - 6.12323399573677e-17*d_2*x14 + 1.0*d_2*x17 + 1.0*d_3*x49 - 6.12323399573677e-17*d_3*x51 - 5.55111512312578e-17*x39 - x40*x55 - 5.55111512312578e-17*x42 - 5.55111512312578e-17*x48
    ee_pose[2, 0] = 1.0*x0*x7 + 1.0*x12*x8*x9 - 5.55111512312578e-17*x18 - 5.55111512312578e-17*x22
    ee_pose[2, 1] = x26*x56 + x26*x57 + 5.55111512312578e-17*x33*x36 - x33*x58
    ee_pose[2, 2] = 5.55111512312578e-17*x26*x36 - x26*x58 - x33*x56 - x33*x57
    ee_pose[2, 3] = 1.0*pre_transform_special_symbol_23 + 5.55111512312578e-17*x38 - 1.0*x39 + 5.55111512312578e-17*x41 - x43*x7 + 5.55111512312578e-17*x44 - x47*x52 + 5.55111512312578e-17*x50
    return ee_pose


def kawasaki_BX250L_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.cos(th_0)
    x2 = -1.0*x1
    x3 = 6.12323399573676e-17*x0 + x2
    x4 = -x3
    x5 = math.sin(th_2)
    x6 = math.sin(th_1)
    x7 = math.cos(th_1)
    x8 = 1.0*x0
    x9 = 6.12323399573676e-17*x1 + x8
    x10 = -3.39907768361723e-33*x6 + x7*x9
    x11 = math.cos(th_2)
    x12 = -x6*x9 - 3.39907768361723e-33*x7
    x13 = -x10*x5 - x11*x12
    x14 = math.cos(th_3)
    x15 = math.sin(th_3)
    x16 = x10*x11 - x12*x5
    x17 = -x14*x4 - x15*x16
    x18 = math.cos(th_4)
    x19 = math.sin(th_4)
    x20 = -x13*x18 - x19*(x14*x16 - x15*x4)
    x21 = 6.12323399573677e-17*x1 + x8
    x22 = -x21
    x23 = 5.55111512312578e-17*x6
    x24 = -6.12323399573677e-17*x0 - x2
    x25 = -x23 + x24*x7
    x26 = -x24*x6 - 5.55111512312578e-17*x7
    x27 = -x11*x26 - x25*x5
    x28 = x11*x25 - x26*x5
    x29 = -x14*x22 - x15*x28
    x30 = -x18*x27 - x19*(x14*x28 - x15*x22)
    x31 = 5.55111512312578e-17*x0
    x32 = -x1*x23 - 1.0*x7
    x33 = 5.55111512312578e-17*x1*x7 - 1.0*x6
    x34 = -x11*x32 - x33*x5
    x35 = x11*x33 - x32*x5
    x36 = 5.55111512312578e-17*x0*x14 - x15*x35
    x37 = -x18*x34 - x19*(x14*x35 + x15*x31)
    x38 = a_0*x24
    x39 = 5.55111512312578e-17*a_0*x1 + pre_transform_special_symbol_23
    x40 = a_1*x33 + x39
    x41 = a_1*x25 + x38
    x42 = d_2*x34 + x40
    x43 = d_2*x27 + x41
    x44 = d_3*x37 + x42
    x45 = d_3*x30 + x43
    x46 = a_0*x9
    x47 = a_1*x10 + x46
    x48 = d_2*x13 + x47
    x49 = d_3*x20 + x48
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 0] = -3.39907768361723e-33
    jacobian_output[0, 1] = x3
    jacobian_output[0, 2] = x4
    jacobian_output[0, 3] = x13
    jacobian_output[0, 4] = x17
    jacobian_output[0, 5] = x20
    jacobian_output[1, 0] = -5.55111512312578e-17
    jacobian_output[1, 1] = x21
    jacobian_output[1, 2] = x22
    jacobian_output[1, 3] = x27
    jacobian_output[1, 4] = x29
    jacobian_output[1, 5] = x30
    jacobian_output[2, 0] = -1.00000000000000
    jacobian_output[2, 1] = x31
    jacobian_output[2, 2] = -x31
    jacobian_output[2, 3] = x34
    jacobian_output[2, 4] = x36
    jacobian_output[2, 5] = x37
    jacobian_output[3, 0] = 5.55111512312578e-17*pre_transform_special_symbol_23
    jacobian_output[3, 1] = -x21*x39 + x31*x38
    jacobian_output[3, 2] = -x22*x40 - x31*x41
    jacobian_output[3, 3] = -x27*x42 + x34*x43
    jacobian_output[3, 4] = -x29*x42 + x36*x43
    jacobian_output[3, 5] = -x30*x44 + x37*x45
    jacobian_output[4, 0] = -3.39907768361723e-33*pre_transform_special_symbol_23
    jacobian_output[4, 1] = x3*x39 - x31*x46
    jacobian_output[4, 2] = x31*x47 + x4*x40
    jacobian_output[4, 3] = x13*x42 - x34*x48
    jacobian_output[4, 4] = x17*x42 - x36*x48
    jacobian_output[4, 5] = x20*x44 - x37*x49
    jacobian_output[5, 1] = a_0*x21*x9 - x3*x38
    jacobian_output[5, 2] = x22*x47 - x4*x41
    jacobian_output[5, 3] = -x13*x43 + x27*x48
    jacobian_output[5, 4] = -x17*x43 + x29*x48
    jacobian_output[5, 5] = -x20*x45 + x30*x49
    return jacobian_output


def kawasaki_BX250L_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = math.cos(th_0)
    x2 = -1.0*x1
    x3 = 6.12323399573676e-17*x0 + x2
    x4 = -x3
    x5 = math.sin(th_2)
    x6 = math.sin(th_1)
    x7 = math.cos(th_1)
    x8 = 1.0*x0
    x9 = 6.12323399573676e-17*x1 + x8
    x10 = -3.39907768361723e-33*x6 + x7*x9
    x11 = math.cos(th_2)
    x12 = -x6*x9 - 3.39907768361723e-33*x7
    x13 = -x10*x5 - x11*x12
    x14 = math.cos(th_3)
    x15 = math.sin(th_3)
    x16 = x10*x11 - x12*x5
    x17 = math.cos(th_4)
    x18 = math.sin(th_4)
    x19 = 6.12323399573677e-17*x1 + x8
    x20 = -x19
    x21 = 5.55111512312578e-17*x6
    x22 = -6.12323399573677e-17*x0 - x2
    x23 = -x21 + x22*x7
    x24 = -x22*x6 - 5.55111512312578e-17*x7
    x25 = -x11*x24 - x23*x5
    x26 = x11*x23 - x24*x5
    x27 = 5.55111512312578e-17*x0
    x28 = -x1*x21 - 1.0*x7
    x29 = 5.55111512312578e-17*x1*x7 - 1.0*x6
    x30 = -x11*x28 - x29*x5
    x31 = x11*x29 - x28*x5
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -3.39907768361723e-33
    jacobian_output[0, 1] = x3
    jacobian_output[0, 2] = x4
    jacobian_output[0, 3] = x13
    jacobian_output[0, 4] = -x14*x4 - x15*x16
    jacobian_output[0, 5] = -x13*x17 - x18*(x14*x16 - x15*x4)
    jacobian_output[1, 0] = -5.55111512312578e-17
    jacobian_output[1, 1] = x19
    jacobian_output[1, 2] = x20
    jacobian_output[1, 3] = x25
    jacobian_output[1, 4] = -x14*x20 - x15*x26
    jacobian_output[1, 5] = -x17*x25 - x18*(x14*x26 - x15*x20)
    jacobian_output[2, 0] = -1.00000000000000
    jacobian_output[2, 1] = x27
    jacobian_output[2, 2] = -x27
    jacobian_output[2, 3] = x30
    jacobian_output[2, 4] = 5.55111512312578e-17*x0*x14 - x15*x31
    jacobian_output[2, 5] = -x17*x30 - x18*(x14*x31 + x15*x27)
    return jacobian_output


def kawasaki_BX250L_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x0 = math.sin(th_0)
    x1 = 5.55111512312578e-17*x0
    x2 = p_on_ee_y*x1
    x3 = 1.0*x0
    x4 = math.cos(th_0)
    x5 = x3 + 6.12323399573677e-17*x4
    x6 = -1.0*x4
    x7 = -6.12323399573677e-17*x0 - x6
    x8 = a_0*x7
    x9 = 5.55111512312578e-17*a_0*x4 + pre_transform_special_symbol_23
    x10 = -x5
    x11 = math.sin(th_1)
    x12 = math.cos(th_1)
    x13 = -1.0*x11 + 5.55111512312578e-17*x12*x4
    x14 = a_1*x13 + x9
    x15 = 5.55111512312578e-17*x11
    x16 = x12*x7 - x15
    x17 = a_1*x16 + x8
    x18 = math.cos(th_2)
    x19 = -1.0*x12 - x15*x4
    x20 = math.sin(th_2)
    x21 = -x13*x20 - x18*x19
    x22 = -x11*x7 - 5.55111512312578e-17*x12
    x23 = -x16*x20 - x18*x22
    x24 = d_2*x21 + x14
    x25 = d_2*x23 + x17
    x26 = math.cos(th_3)
    x27 = math.sin(th_3)
    x28 = x13*x18 - x19*x20
    x29 = 5.55111512312578e-17*x0*x26 - x27*x28
    x30 = x16*x18 - x20*x22
    x31 = -x10*x26 - x27*x30
    x32 = math.cos(th_4)
    x33 = math.sin(th_4)
    x34 = -x21*x32 - x33*(x1*x27 + x26*x28)
    x35 = -x23*x32 - x33*(-x10*x27 + x26*x30)
    x36 = d_3*x34 + x24
    x37 = d_3*x35 + x25
    x38 = 5.55111512312578e-17*p_on_ee_x
    x39 = x0*x38
    x40 = 6.12323399573676e-17*x0 + x6
    x41 = x3 + 6.12323399573676e-17*x4
    x42 = a_0*x41
    x43 = -x40
    x44 = -3.39907768361723e-33*x11 + x12*x41
    x45 = a_1*x44 + x42
    x46 = -x11*x41 - 3.39907768361723e-33*x12
    x47 = -x18*x46 - x20*x44
    x48 = d_2*x47 + x45
    x49 = x18*x44 - x20*x46
    x50 = -x26*x43 - x27*x49
    x51 = -x32*x47 - x33*(x26*x49 - x27*x43)
    x52 = d_3*x51 + x48
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = 1.0*p_on_ee_y - 5.55111512312578e-17*p_on_ee_z + 5.55111512312578e-17*pre_transform_special_symbol_23
    jacobian_output[0, 1] = p_on_ee_z*x5 + x1*x8 - x2 - x5*x9
    jacobian_output[0, 2] = p_on_ee_z*x10 - x1*x17 - x10*x14 + x2
    jacobian_output[0, 3] = -p_on_ee_y*x21 + p_on_ee_z*x23 + x21*x25 - x23*x24
    jacobian_output[0, 4] = -p_on_ee_y*x29 + p_on_ee_z*x31 - x24*x31 + x25*x29
    jacobian_output[0, 5] = -p_on_ee_y*x34 + p_on_ee_z*x35 + x34*x37 - x35*x36
    jacobian_output[1, 0] = -1.0*p_on_ee_x + 3.39907768361723e-33*p_on_ee_z - 3.39907768361723e-33*pre_transform_special_symbol_23
    jacobian_output[1, 1] = -p_on_ee_z*x40 - x1*x42 + x39 + x40*x9
    jacobian_output[1, 2] = -p_on_ee_z*x43 + 5.55111512312578e-17*x0*x45 + x14*x43 - x39
    jacobian_output[1, 3] = p_on_ee_x*x21 - p_on_ee_z*x47 - x21*x48 + x24*x47
    jacobian_output[1, 4] = p_on_ee_x*x29 - p_on_ee_z*x50 + x24*x50 - x29*x48
    jacobian_output[1, 5] = p_on_ee_x*x34 - p_on_ee_z*x51 - x34*x52 + x36*x51
    jacobian_output[2, 0] = -3.39907768361723e-33*p_on_ee_y + x38
    jacobian_output[2, 1] = a_0*x41*x5 - p_on_ee_x*x5 + p_on_ee_y*x40 - x40*x8
    jacobian_output[2, 2] = -p_on_ee_x*x10 + p_on_ee_y*x43 + x10*x45 - x17*x43
    jacobian_output[2, 3] = -p_on_ee_x*x23 + p_on_ee_y*x47 + x23*x48 - x25*x47
    jacobian_output[2, 4] = -p_on_ee_x*x31 + p_on_ee_y*x50 - x25*x50 + x31*x48
    jacobian_output[2, 5] = -p_on_ee_x*x35 + p_on_ee_y*x51 + x35*x52 - x37*x51
    return jacobian_output


def kawasaki_BX250L_ik_solve_raw(T_ee: np.ndarray):
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
            solution_0[2] = x0
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)
            
        condition_1: bool = (abs(Px - d_3*r_13) >= zero_tolerance) or (abs(Py - d_3*r_23) >= zero_tolerance)
        if condition_1:
            # Temp variable for efficiency
            x0 = math.atan2(Py - d_3*r_23, Px - d_3*r_13)
            # End of temp variables
            solution_1: IkSolution = make_ik_solution()
            solution_1[2] = x0 + math.pi
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
            th_0 = this_solution[2]
            condition_0: bool = ((1/2)*abs((Px**2 - 2*Px*a_0*math.cos(th_0) - 2*Px*d_3*r_13 + Py**2 - 2*Py*a_0*math.sin(th_0) - 2*Py*d_3*r_23 + Pz**2 - 2*Pz*d_3*r_33 + a_0**2 + 2*a_0*d_3*r_13*math.cos(th_0) + 2*a_0*d_3*r_23*math.sin(th_0) - a_1**2 - d_2**2 + d_3**2*r_13**2 + d_3**2*r_23**2 + d_3**2*r_33**2)/(a_1*d_2)) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*Px
                x1 = d_3*r_13
                x2 = 2*Py
                x3 = d_3*r_23
                x4 = a_0*math.cos(th_0)
                x5 = a_0*math.sin(th_0)
                x6 = d_3**2
                x7 = safe_asin((1/2)*(Px**2 + Py**2 + Pz**2 - 2*Pz*d_3*r_33 + a_0**2 - a_1**2 - d_2**2 + r_13**2*x6 + r_23**2*x6 + r_33**2*x6 - x0*x1 - x0*x4 + 2*x1*x4 - x2*x3 - x2*x5 + 2*x3*x5)/(a_1*d_2))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = -x7
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = ((1/2)*abs((Px**2 - 2*Px*a_0*math.cos(th_0) - 2*Px*d_3*r_13 + Py**2 - 2*Py*a_0*math.sin(th_0) - 2*Py*d_3*r_23 + Pz**2 - 2*Pz*d_3*r_33 + a_0**2 + 2*a_0*d_3*r_13*math.cos(th_0) + 2*a_0*d_3*r_23*math.sin(th_0) - a_1**2 - d_2**2 + d_3**2*r_13**2 + d_3**2*r_23**2 + d_3**2*r_33**2)/(a_1*d_2)) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*Px
                x1 = d_3*r_13
                x2 = 2*Py
                x3 = d_3*r_23
                x4 = a_0*math.cos(th_0)
                x5 = a_0*math.sin(th_0)
                x6 = d_3**2
                x7 = safe_asin((1/2)*(Px**2 + Py**2 + Pz**2 - 2*Pz*d_3*r_33 + a_0**2 - a_1**2 - d_2**2 + r_13**2*x6 + r_23**2*x6 + r_33**2*x6 - x0*x1 - x0*x4 + 2*x1*x4 - x2*x3 - x2*x5 + 2*x3*x5)/(a_1*d_2))
                # End of temp variables
                this_solution[4] = x7 + math.pi
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
            th_0 = this_solution[2]
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
            th_0 = this_solution[2]
            th_2 = this_solution[4]
            condition_0: bool = (abs(Pz - d_3*r_33) >= 1.0e-6) or (abs(-Px*math.cos(th_0) - Py*math.sin(th_0) + a_0 + d_3*r_13*math.cos(th_0) + d_3*r_23*math.sin(th_0)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = -Pz + d_3*r_33
                x1 = -a_1 + d_2*math.sin(th_2)
                x2 = math.cos(th_0)
                x3 = math.sin(th_0)
                x4 = -Px*x2 - Py*x3 + a_0 + d_3*r_13*x2 + d_3*r_23*x3
                x5 = d_2*math.cos(th_2)
                # End of temp variables
                this_solution[3] = math.atan2(x0*x1 - x4*x5, x0*x5 + x1*x4)
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
            th_1 = this_solution[3]
            th_2 = this_solution[4]
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
            th_0 = this_solution[2]
            th_1 = this_solution[3]
            th_2 = this_solution[4]
            condition_0: bool = (abs(r_13*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_23*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.sin(th_0) - r_33*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) <= 1)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_1)
                x1 = math.sin(th_2)
                x2 = math.cos(th_1)
                x3 = math.cos(th_2)
                x4 = x0*x3 - x1*x2
                x5 = safe_acos(-r_13*x4*math.cos(th_0) - r_23*x4*math.sin(th_0) + r_33*(x0*x1 + x2*x3))
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[7] = x5
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(10, appended_idx)
                
            condition_1: bool = (abs(r_13*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_23*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.sin(th_0) - r_33*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) <= 1)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.sin(th_1)
                x1 = math.sin(th_2)
                x2 = math.cos(th_1)
                x3 = math.cos(th_2)
                x4 = x0*x3 - x1*x2
                x5 = safe_acos(-r_13*x4*math.cos(th_0) - r_23*x4*math.sin(th_0) + r_33*(x0*x1 + x2*x3))
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
            th_0 = this_solution[2]
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
            th_0 = this_solution[2]
            condition_0: bool = (1 >= zero_tolerance) or (abs(r_11*math.sin(th_0) - r_21*math.cos(th_0)) >= zero_tolerance) or (abs(r_12*math.sin(th_0) - r_22*math.cos(th_0)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_0)
                x1 = math.cos(th_0)
                # End of temp variables
                this_solution[1] = math.atan2(r_11*x0 - r_21*x1, -r_12*x0 + r_22*x1)
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
            negative_th_5_positive_th_3__soa = this_solution[1]
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
            th_0 = this_solution[2]
            th_1 = this_solution[3]
            th_2 = this_solution[4]
            th_4 = this_solution[7]
            condition_0: bool = (abs(r_13*math.sin(th_0) - r_23*math.cos(th_0)) >= zero_tolerance) or (abs(r_13*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.cos(th_0) + r_23*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))*math.sin(th_0) + r_33*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))) >= zero_tolerance) or (abs(math.sin(th_4)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_4)**(-1)
                x1 = math.sin(th_0)
                x2 = math.cos(th_0)
                x3 = math.sin(th_1)
                x4 = math.cos(th_2)
                x5 = math.sin(th_2)
                x6 = math.cos(th_1)
                x7 = x3*x5 + x4*x6
                # End of temp variables
                this_solution[5] = math.atan2(x0*(-r_13*x1 + r_23*x2), x0*(-r_13*x2*x7 - r_23*x1*x7 - r_33*(x3*x4 - x5*x6)))
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
            th_0 = this_solution[2]
            th_1 = this_solution[3]
            th_2 = this_solution[4]
            th_4 = this_solution[7]
            condition_0: bool = (abs(r_11*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_21*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.sin(th_0) - r_31*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) >= zero_tolerance) or (abs(r_12*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.cos(th_0) + r_22*(math.sin(th_1)*math.cos(th_2) - math.sin(th_2)*math.cos(th_1))*math.sin(th_0) - r_32*(math.sin(th_1)*math.sin(th_2) + math.cos(th_1)*math.cos(th_2))) >= zero_tolerance) or (abs(math.sin(th_4)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_4)**(-1)
                x1 = math.sin(th_1)
                x2 = math.sin(th_2)
                x3 = math.cos(th_1)
                x4 = math.cos(th_2)
                x5 = x1*x2 + x3*x4
                x6 = x1*x4 - x2*x3
                x7 = x6*math.cos(th_0)
                x8 = x6*math.sin(th_0)
                # End of temp variables
                this_solution[8] = math.atan2(x0*(r_12*x7 + r_22*x8 - r_32*x5), x0*(-r_11*x7 - r_21*x8 + r_31*x5))
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
        value_at_0 = ik_out_i[2]  # th_0
        new_ik_i[0] = value_at_0
        value_at_1 = ik_out_i[3]  # th_1
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


def kawasaki_BX250L_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = kawasaki_BX250L_ik_target_original_to_raw(T_ee)
    ik_output_raw = kawasaki_BX250L_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ee_pose_i = kawasaki_BX250L_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_kawasaki_BX250L():
    theta_in = np.random.random(size=(6, ))
    ee_pose = kawasaki_BX250L_fk(theta_in)
    ik_output = kawasaki_BX250L_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = kawasaki_BX250L_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_kawasaki_BX250L()
