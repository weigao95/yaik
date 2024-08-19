import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 6
n_tree_nodes: int = 14
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_0: float = 1.3
alpha_3: float = -1.0470795620393143
alpha_5: float = -1.0470795620393143
d_1: float = -1.4
d_2: float = -0.11397670814688408
d_4: float = -0.12300000000000003
pre_transform_special_symbol_23: float = 0.8

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -1.5707963267948966
th_2_offset_original2raw: float = -0.0
th_3_offset_original2raw: float = -1.5707963267948966
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = -1.5707963267948966


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def yaskawa_mpx3500_ik_target_original_to_raw(T_ee: np.ndarray):
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


def yaskawa_mpx3500_ik_target_raw_to_original(T_ee: np.ndarray):
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


def yaskawa_mpx3500_fk(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.cos(alpha_3)
    x1 = math.cos(th_0)
    x2 = math.sin(th_1)
    x3 = math.cos(th_2)
    x4 = x2*x3
    x5 = math.sin(th_2)
    x6 = math.cos(th_1)
    x7 = x5*x6
    x8 = x1*x4 - x1*x7
    x9 = x0*x8
    x10 = math.sin(alpha_3)
    x11 = math.sin(th_0)
    x12 = math.cos(th_3)
    x13 = math.sin(th_3)
    x14 = x2*x5
    x15 = x3*x6
    x16 = x1*x14 + x1*x15
    x17 = -x11*x12 - x13*x16
    x18 = x10*x17
    x19 = -x18 + x9
    x20 = 1.0*math.cos(alpha_5)
    x21 = x19*x20
    x22 = math.cos(th_4)
    x23 = x10*x8
    x24 = math.sin(th_4)
    x25 = -x11*x13 + x12*x16
    x26 = x0*x17
    x27 = x22*x23 + x22*x26 - x24*x25
    x28 = 1.0*math.sin(alpha_5)
    x29 = x27*x28
    x30 = math.cos(th_5)
    x31 = x19*x28
    x32 = math.sin(th_5)
    x33 = 1.0*x22*x25 + 1.0*x23*x24 + 1.0*x24*x26
    x34 = x20*x27
    x35 = 1.0*a_0
    x36 = x35*x6
    x37 = 1.0*d_1
    x38 = 1.0*d_2
    x39 = x11*x4 - x11*x7
    x40 = x0*x39
    x41 = x11*x14 + x11*x15
    x42 = x1*x12 - x13*x41
    x43 = x10*x42
    x44 = x40 - x43
    x45 = x20*x44
    x46 = x10*x39
    x47 = x1*x13 + x12*x41
    x48 = x0*x42
    x49 = x22*x46 + x22*x48 - x24*x47
    x50 = x28*x49
    x51 = x28*x44
    x52 = 1.0*x22*x47 + 1.0*x24*x46 + 1.0*x24*x48
    x53 = x20*x49
    x54 = x14 + x15
    x55 = x0*x54
    x56 = -x4 + x7
    x57 = x13*x56
    x58 = x10*x57
    x59 = x55 + x58
    x60 = x20*x59
    x61 = x10*x54
    x62 = x12*x56
    x63 = x0*x57
    x64 = x22*x61 - x22*x63 - x24*x62
    x65 = x28*x64
    x66 = x28*x59
    x67 = 1.0*x22*x62 + 1.0*x24*x61 - 1.0*x24*x63
    x68 = x20*x64
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -x21 + x29
    ee_pose[0, 1] = x30*x31 + x30*x34 - x32*x33
    ee_pose[0, 2] = x30*x33 + x31*x32 + x32*x34
    ee_pose[0, 3] = d_4*x21 - d_4*x29 + x1*x36 - x18*x38 + x37*x8 + x38*x9
    ee_pose[1, 0] = -x45 + x50
    ee_pose[1, 1] = x30*x51 + x30*x53 - x32*x52
    ee_pose[1, 2] = x30*x52 + x32*x51 + x32*x53
    ee_pose[1, 3] = d_4*x45 - d_4*x50 + x11*x36 + x37*x39 + x38*x40 - x38*x43
    ee_pose[2, 0] = -x60 + x65
    ee_pose[2, 1] = x30*x66 + x30*x68 - x32*x67
    ee_pose[2, 2] = x30*x67 + x32*x66 + x32*x68
    ee_pose[2, 3] = d_4*x60 - d_4*x65 + 1.0*pre_transform_special_symbol_23 - x2*x35 + x37*x54 + x38*x55 + x38*x58
    return ee_pose


def yaskawa_mpx3500_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_0)
    x1 = 1.0*x0
    x2 = math.sin(th_1)
    x3 = math.cos(th_2)
    x4 = math.cos(th_0)
    x5 = 1.0*x4
    x6 = x3*x5
    x7 = math.cos(th_1)
    x8 = math.sin(th_2)
    x9 = x5*x8
    x10 = x2*x6 - x7*x9
    x11 = math.cos(alpha_3)
    x12 = x10*x11
    x13 = math.sin(alpha_3)
    x14 = math.cos(th_3)
    x15 = math.sin(th_3)
    x16 = x2*x9 + x6*x7
    x17 = -x1*x14 - x15*x16
    x18 = x13*x17
    x19 = x12 - x18
    x20 = math.cos(alpha_5)
    x21 = x19*x20
    x22 = math.sin(alpha_5)
    x23 = math.cos(th_4)
    x24 = x13*x23
    x25 = math.sin(th_4)
    x26 = x11*x23
    x27 = x22*(x10*x24 + x17*x26 - x25*(-x1*x15 + x14*x16))
    x28 = x21 - x27
    x29 = x1*x3
    x30 = x1*x8
    x31 = x2*x29 - x30*x7
    x32 = x11*x31
    x33 = x2*x30 + x29*x7
    x34 = x14*x5 - x15*x33
    x35 = x13*x34
    x36 = x32 - x35
    x37 = x20*x36
    x38 = x22*(x24*x31 - x25*(x14*x33 + x15*x5) + x26*x34)
    x39 = x37 - x38
    x40 = 1.0*x2
    x41 = 1.0*x7
    x42 = x3*x41 + x40*x8
    x43 = x11*x42
    x44 = -x3*x40 + x41*x8
    x45 = x15*x44
    x46 = x13*x45
    x47 = x43 + x46
    x48 = x20*x47
    x49 = x22*(-x14*x25*x44 + x24*x42 - x26*x45)
    x50 = x48 - x49
    x51 = -a_0*x40 + pre_transform_special_symbol_23
    x52 = a_0*x7
    x53 = d_1*x31 + x1*x52
    x54 = d_1*x42 + x51
    x55 = d_2*x43 + d_2*x46 + x54
    x56 = d_2*x32 - d_2*x35 + x53
    x57 = d_4*x48 - d_4*x49 + x55
    x58 = d_4*x37 - d_4*x38 + x56
    x59 = d_1*x10 + x5*x52
    x60 = d_2*x12 - d_2*x18 + x59
    x61 = d_4*x21 - d_4*x27 + x60
    x62 = a_0*x41
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 6))
    jacobian_output[0, 1] = -x1
    jacobian_output[0, 2] = x1
    jacobian_output[0, 3] = x10
    jacobian_output[0, 4] = x19
    jacobian_output[0, 5] = x28
    jacobian_output[1, 1] = x5
    jacobian_output[1, 2] = -x5
    jacobian_output[1, 3] = x31
    jacobian_output[1, 4] = x36
    jacobian_output[1, 5] = x39
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x42
    jacobian_output[2, 4] = x47
    jacobian_output[2, 5] = x50
    jacobian_output[3, 1] = -pre_transform_special_symbol_23*x5
    jacobian_output[3, 2] = x5*x51
    jacobian_output[3, 3] = -x31*x54 + x42*x53
    jacobian_output[3, 4] = -x36*x55 + x47*x56
    jacobian_output[3, 5] = -x39*x57 + x50*x58
    jacobian_output[4, 1] = -pre_transform_special_symbol_23*x1
    jacobian_output[4, 2] = x1*x51
    jacobian_output[4, 3] = x10*x54 - x42*x59
    jacobian_output[4, 4] = x19*x55 - x47*x60
    jacobian_output[4, 5] = x28*x57 - x50*x61
    jacobian_output[5, 2] = -x0**2*x62 - x4**2*x62
    jacobian_output[5, 3] = -x10*x53 + x31*x59
    jacobian_output[5, 4] = -x19*x56 + x36*x60
    jacobian_output[5, 5] = -x28*x58 + x39*x61
    return jacobian_output


def yaskawa_mpx3500_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw

    # Temp variable for efficiency
    x0 = 1.0*math.sin(th_0)
    x1 = math.sin(th_1)
    x2 = math.cos(th_2)
    x3 = 1.0*math.cos(th_0)
    x4 = x2*x3
    x5 = math.cos(th_1)
    x6 = math.sin(th_2)
    x7 = x3*x6
    x8 = x1*x4 - x5*x7
    x9 = math.cos(alpha_3)
    x10 = math.sin(alpha_3)
    x11 = math.cos(th_3)
    x12 = math.sin(th_3)
    x13 = x1*x7 + x4*x5
    x14 = -x0*x11 - x12*x13
    x15 = -x10*x14 + x8*x9
    x16 = math.cos(alpha_5)
    x17 = math.sin(alpha_5)
    x18 = math.cos(th_4)
    x19 = x10*x18
    x20 = math.sin(th_4)
    x21 = x18*x9
    x22 = x0*x2
    x23 = x0*x6
    x24 = x1*x22 - x23*x5
    x25 = x1*x23 + x22*x5
    x26 = x11*x3 - x12*x25
    x27 = -x10*x26 + x24*x9
    x28 = 1.0*x1
    x29 = 1.0*x5
    x30 = x2*x29 + x28*x6
    x31 = -x2*x28 + x29*x6
    x32 = x12*x31
    x33 = x10*x32 + x30*x9
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 1] = -x0
    jacobian_output[0, 2] = x0
    jacobian_output[0, 3] = x8
    jacobian_output[0, 4] = x15
    jacobian_output[0, 5] = x15*x16 - x17*(x14*x21 + x19*x8 - x20*(-x0*x12 + x11*x13))
    jacobian_output[1, 1] = x3
    jacobian_output[1, 2] = -x3
    jacobian_output[1, 3] = x24
    jacobian_output[1, 4] = x27
    jacobian_output[1, 5] = x16*x27 - x17*(x19*x24 - x20*(x11*x25 + x12*x3) + x21*x26)
    jacobian_output[2, 0] = 1.00000000000000
    jacobian_output[2, 3] = x30
    jacobian_output[2, 4] = x33
    jacobian_output[2, 5] = x16*x33 - x17*(-x11*x20*x31 + x19*x30 - x21*x32)
    return jacobian_output


def yaskawa_mpx3500_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x6 = -a_0*x5 + pre_transform_special_symbol_23
    x7 = math.sin(th_2)
    x8 = x5*x7
    x9 = math.cos(th_2)
    x10 = math.cos(th_1)
    x11 = 1.0*x10
    x12 = x11*x9
    x13 = x12 + x8
    x14 = math.sin(th_0)
    x15 = x5*x9
    x16 = x11*x7
    x17 = x14*x15 - x14*x16
    x18 = a_0*x11
    x19 = d_1*x17 + x14*x18
    x20 = d_1*x13 + x6
    x21 = math.cos(alpha_3)
    x22 = x13*x21
    x23 = math.sin(alpha_3)
    x24 = math.sin(th_3)
    x25 = -x15 + x16
    x26 = x24*x25
    x27 = x23*x26
    x28 = x22 + x27
    x29 = x17*x21
    x30 = math.cos(th_3)
    x31 = x12*x14 + x14*x8
    x32 = x2*x30 - x24*x31
    x33 = x23*x32
    x34 = x29 - x33
    x35 = d_2*x22 + d_2*x27 + x20
    x36 = d_2*x29 - d_2*x33 + x19
    x37 = math.cos(alpha_5)
    x38 = x28*x37
    x39 = math.sin(alpha_5)
    x40 = math.cos(th_4)
    x41 = x23*x40
    x42 = math.sin(th_4)
    x43 = x21*x40
    x44 = x39*(x13*x41 - x25*x30*x42 - x26*x43)
    x45 = x38 - x44
    x46 = x34*x37
    x47 = x39*(x17*x41 + x32*x43 - x42*(x2*x24 + x30*x31))
    x48 = x46 - x47
    x49 = d_4*x38 - d_4*x44 + x35
    x50 = d_4*x46 - d_4*x47 + x36
    x51 = 1.0*p_on_ee_x
    x52 = 1.0*x14
    x53 = p_on_ee_z*x52
    x54 = x2*x9
    x55 = x10*x2
    x56 = x4*x54 - x55*x7
    x57 = a_0*x55 + d_1*x56
    x58 = x21*x56
    x59 = x10*x54 + x2*x4*x7
    x60 = -x24*x59 - x30*x52
    x61 = x23*x60
    x62 = x58 - x61
    x63 = d_2*x58 - d_2*x61 + x57
    x64 = x37*x62
    x65 = x39*(x41*x56 - x42*(-x24*x52 + x30*x59) + x43*x60)
    x66 = x64 - x65
    x67 = d_4*x64 - d_4*x65 + x63
    x68 = x1*x51
    x69 = x0*x14
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 6))
    jacobian_output[0, 0] = -x0
    jacobian_output[0, 1] = -pre_transform_special_symbol_23*x2 + x3
    jacobian_output[0, 2] = x2*x6 - x3
    jacobian_output[0, 3] = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x19 - x17*x20
    jacobian_output[0, 4] = -p_on_ee_y*x28 + p_on_ee_z*x34 + x28*x36 - x34*x35
    jacobian_output[0, 5] = -p_on_ee_y*x45 + p_on_ee_z*x48 + x45*x50 - x48*x49
    jacobian_output[1, 0] = x51
    jacobian_output[1, 1] = -pre_transform_special_symbol_23*x52 + x53
    jacobian_output[1, 2] = x52*x6 - x53
    jacobian_output[1, 3] = p_on_ee_x*x13 - p_on_ee_z*x56 - x13*x57 + x20*x56
    jacobian_output[1, 4] = p_on_ee_x*x28 - p_on_ee_z*x62 - x28*x63 + x35*x62
    jacobian_output[1, 5] = p_on_ee_x*x45 - p_on_ee_z*x66 - x45*x67 + x49*x66
    jacobian_output[2, 1] = -x68 - x69
    jacobian_output[2, 2] = -x1**2*x18 - x14**2*x18 + x68 + x69
    jacobian_output[2, 3] = -p_on_ee_x*x17 + p_on_ee_y*x56 + x17*x57 - x19*x56
    jacobian_output[2, 4] = -p_on_ee_x*x34 + p_on_ee_y*x62 + x34*x63 - x36*x62
    jacobian_output[2, 5] = -p_on_ee_x*x48 + p_on_ee_y*x66 + x48*x67 - x50*x66
    return jacobian_output


def yaskawa_mpx3500_ik_solve_raw(T_ee: np.ndarray):
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
        for tmp_sol_idx in range(7):
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
    for i in range(14):
        node_input_index.append(list())
        node_input_validity.append(False)
    def add_input_index_to(node_idx: int, solution_idx: int):
        node_input_index[node_idx].append(solution_idx)
        node_input_validity[node_idx] = True
    node_input_validity[0] = True
    
    # Code for non-branch dispatcher node 0
    # Actually, there is no code
    
    # Code for explicit solution node 1, solved variable is th_3
    def General6DoFNumericalReduceSolutionNode_node_1_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return
        
        # The general 6-dof solution of root node with semi-symbolic reduce
        R_l = np.zeros(shape=(8, 8))
        R_l[0, 0] = -d_2*r_21*math.sin(alpha_5)
        R_l[0, 1] = -d_2*r_22*math.sin(alpha_5)
        R_l[0, 2] = -d_2*r_11*math.sin(alpha_5)
        R_l[0, 3] = -d_2*r_12*math.sin(alpha_5)
        R_l[0, 4] = Py - d_2*r_23*math.cos(alpha_5) - d_4*r_23
        R_l[0, 5] = Px - d_2*r_13*math.cos(alpha_5) - d_4*r_13
        R_l[1, 0] = -d_2*r_11*math.sin(alpha_5)
        R_l[1, 1] = -d_2*r_12*math.sin(alpha_5)
        R_l[1, 2] = d_2*r_21*math.sin(alpha_5)
        R_l[1, 3] = d_2*r_22*math.sin(alpha_5)
        R_l[1, 4] = Px - d_2*r_13*math.cos(alpha_5) - d_4*r_13
        R_l[1, 5] = -Py + d_2*r_23*math.cos(alpha_5) + d_4*r_23
        R_l[2, 6] = -d_2*r_31*math.sin(alpha_5)
        R_l[2, 7] = -d_2*r_32*math.sin(alpha_5)
        R_l[3, 0] = r_21*math.sin(alpha_5)
        R_l[3, 1] = r_22*math.sin(alpha_5)
        R_l[3, 2] = r_11*math.sin(alpha_5)
        R_l[3, 3] = r_12*math.sin(alpha_5)
        R_l[3, 4] = r_23*math.cos(alpha_5)
        R_l[3, 5] = r_13*math.cos(alpha_5)
        R_l[4, 0] = r_11*math.sin(alpha_5)
        R_l[4, 1] = r_12*math.sin(alpha_5)
        R_l[4, 2] = -r_21*math.sin(alpha_5)
        R_l[4, 3] = -r_22*math.sin(alpha_5)
        R_l[4, 4] = r_13*math.cos(alpha_5)
        R_l[4, 5] = -r_23*math.cos(alpha_5)
        R_l[5, 6] = -2*Px*d_2*r_11*math.sin(alpha_5) - 2*Py*d_2*r_21*math.sin(alpha_5) - 2*Pz*d_2*r_31*math.sin(alpha_5) + 2*d_2**2*r_11*r_13*math.sin(alpha_5)*math.cos(alpha_5) + 2*d_2**2*r_21*r_23*math.sin(alpha_5)*math.cos(alpha_5) + 2*d_2**2*r_31*r_33*math.sin(alpha_5)*math.cos(alpha_5) + 2*d_2*d_4*r_11*r_13*math.sin(alpha_5) + 2*d_2*d_4*r_21*r_23*math.sin(alpha_5) + 2*d_2*d_4*r_31*r_33*math.sin(alpha_5)
        R_l[5, 7] = -2*Px*d_2*r_12*math.sin(alpha_5) - 2*Py*d_2*r_22*math.sin(alpha_5) - 2*Pz*d_2*r_32*math.sin(alpha_5) + 2*d_2**2*r_12*r_13*math.sin(alpha_5)*math.cos(alpha_5) + 2*d_2**2*r_22*r_23*math.sin(alpha_5)*math.cos(alpha_5) + 2*d_2**2*r_32*r_33*math.sin(alpha_5)*math.cos(alpha_5) + 2*d_2*d_4*r_12*r_13*math.sin(alpha_5) + 2*d_2*d_4*r_22*r_23*math.sin(alpha_5) + 2*d_2*d_4*r_32*r_33*math.sin(alpha_5)
        R_l[6, 0] = Px*r_31*math.sin(alpha_5) - Pz*r_11*math.sin(alpha_5) + d_4*r_11*r_33*math.sin(alpha_5) - d_4*r_13*r_31*math.sin(alpha_5)
        R_l[6, 1] = Px*r_32*math.sin(alpha_5) - Pz*r_12*math.sin(alpha_5) + d_4*r_12*r_33*math.sin(alpha_5) - d_4*r_13*r_32*math.sin(alpha_5)
        R_l[6, 2] = -Py*r_31*math.sin(alpha_5) + Pz*r_21*math.sin(alpha_5) - d_4*r_21*r_33*math.sin(alpha_5) + d_4*r_23*r_31*math.sin(alpha_5)
        R_l[6, 3] = -Py*r_32*math.sin(alpha_5) + Pz*r_22*math.sin(alpha_5) - d_4*r_22*r_33*math.sin(alpha_5) + d_4*r_23*r_32*math.sin(alpha_5)
        R_l[6, 4] = Px*r_33*math.cos(alpha_5) - Pz*r_13*math.cos(alpha_5)
        R_l[6, 5] = -Py*r_33*math.cos(alpha_5) + Pz*r_23*math.cos(alpha_5)
        R_l[7, 0] = -Py*r_31*math.sin(alpha_5) + Pz*r_21*math.sin(alpha_5) - d_4*r_21*r_33*math.sin(alpha_5) + d_4*r_23*r_31*math.sin(alpha_5)
        R_l[7, 1] = -Py*r_32*math.sin(alpha_5) + Pz*r_22*math.sin(alpha_5) - d_4*r_22*r_33*math.sin(alpha_5) + d_4*r_23*r_32*math.sin(alpha_5)
        R_l[7, 2] = -Px*r_31*math.sin(alpha_5) + Pz*r_11*math.sin(alpha_5) - d_4*r_11*r_33*math.sin(alpha_5) + d_4*r_13*r_31*math.sin(alpha_5)
        R_l[7, 3] = -Px*r_32*math.sin(alpha_5) + Pz*r_12*math.sin(alpha_5) - d_4*r_12*r_33*math.sin(alpha_5) + d_4*r_13*r_32*math.sin(alpha_5)
        R_l[7, 4] = -Py*r_33*math.cos(alpha_5) + Pz*r_23*math.cos(alpha_5)
        R_l[7, 5] = -Px*r_33*math.cos(alpha_5) + Pz*r_13*math.cos(alpha_5)
        try:
            R_l_mat_inv = np.linalg.inv(R_l)
        except:
            return
        R_l_inv_00 = R_l_mat_inv[0, 0]
        R_l_inv_01 = R_l_mat_inv[0, 1]
        R_l_inv_02 = R_l_mat_inv[0, 2]
        R_l_inv_03 = R_l_mat_inv[0, 3]
        R_l_inv_04 = R_l_mat_inv[0, 4]
        R_l_inv_05 = R_l_mat_inv[0, 5]
        R_l_inv_06 = R_l_mat_inv[0, 6]
        R_l_inv_07 = R_l_mat_inv[0, 7]
        R_l_inv_10 = R_l_mat_inv[1, 0]
        R_l_inv_11 = R_l_mat_inv[1, 1]
        R_l_inv_12 = R_l_mat_inv[1, 2]
        R_l_inv_13 = R_l_mat_inv[1, 3]
        R_l_inv_14 = R_l_mat_inv[1, 4]
        R_l_inv_15 = R_l_mat_inv[1, 5]
        R_l_inv_16 = R_l_mat_inv[1, 6]
        R_l_inv_17 = R_l_mat_inv[1, 7]
        R_l_inv_20 = R_l_mat_inv[2, 0]
        R_l_inv_21 = R_l_mat_inv[2, 1]
        R_l_inv_22 = R_l_mat_inv[2, 2]
        R_l_inv_23 = R_l_mat_inv[2, 3]
        R_l_inv_24 = R_l_mat_inv[2, 4]
        R_l_inv_25 = R_l_mat_inv[2, 5]
        R_l_inv_26 = R_l_mat_inv[2, 6]
        R_l_inv_27 = R_l_mat_inv[2, 7]
        R_l_inv_30 = R_l_mat_inv[3, 0]
        R_l_inv_31 = R_l_mat_inv[3, 1]
        R_l_inv_32 = R_l_mat_inv[3, 2]
        R_l_inv_33 = R_l_mat_inv[3, 3]
        R_l_inv_34 = R_l_mat_inv[3, 4]
        R_l_inv_35 = R_l_mat_inv[3, 5]
        R_l_inv_36 = R_l_mat_inv[3, 6]
        R_l_inv_37 = R_l_mat_inv[3, 7]
        R_l_inv_40 = R_l_mat_inv[4, 0]
        R_l_inv_41 = R_l_mat_inv[4, 1]
        R_l_inv_42 = R_l_mat_inv[4, 2]
        R_l_inv_43 = R_l_mat_inv[4, 3]
        R_l_inv_44 = R_l_mat_inv[4, 4]
        R_l_inv_45 = R_l_mat_inv[4, 5]
        R_l_inv_46 = R_l_mat_inv[4, 6]
        R_l_inv_47 = R_l_mat_inv[4, 7]
        R_l_inv_50 = R_l_mat_inv[5, 0]
        R_l_inv_51 = R_l_mat_inv[5, 1]
        R_l_inv_52 = R_l_mat_inv[5, 2]
        R_l_inv_53 = R_l_mat_inv[5, 3]
        R_l_inv_54 = R_l_mat_inv[5, 4]
        R_l_inv_55 = R_l_mat_inv[5, 5]
        R_l_inv_56 = R_l_mat_inv[5, 6]
        R_l_inv_57 = R_l_mat_inv[5, 7]
        R_l_inv_60 = R_l_mat_inv[6, 0]
        R_l_inv_61 = R_l_mat_inv[6, 1]
        R_l_inv_62 = R_l_mat_inv[6, 2]
        R_l_inv_63 = R_l_mat_inv[6, 3]
        R_l_inv_64 = R_l_mat_inv[6, 4]
        R_l_inv_65 = R_l_mat_inv[6, 5]
        R_l_inv_66 = R_l_mat_inv[6, 6]
        R_l_inv_67 = R_l_mat_inv[6, 7]
        R_l_inv_70 = R_l_mat_inv[7, 0]
        R_l_inv_71 = R_l_mat_inv[7, 1]
        R_l_inv_72 = R_l_mat_inv[7, 2]
        R_l_inv_73 = R_l_mat_inv[7, 3]
        R_l_inv_74 = R_l_mat_inv[7, 4]
        R_l_inv_75 = R_l_mat_inv[7, 5]
        R_l_inv_76 = R_l_mat_inv[7, 6]
        R_l_inv_77 = R_l_mat_inv[7, 7]
        
        # Temp variable for efficiency
        x0 = math.sin(alpha_5)
        x1 = r_31*x0
        x2 = r_32*x0
        x3 = R_l_inv_60*x1 + R_l_inv_70*x2
        x4 = a_0*x3
        x5 = math.sin(alpha_3)
        x6 = x5*(R_l_inv_64*x1 + R_l_inv_74*x2)
        x7 = math.cos(alpha_5)
        x8 = r_33*x7
        x9 = -x8
        x10 = R_l_inv_62*x1 + R_l_inv_72*x2
        x11 = d_4*r_33
        x12 = d_2*x8
        x13 = -Pz + x11 + x12
        x14 = -x10*x13
        x15 = R_l_inv_65*x1 + R_l_inv_75*x2
        x16 = a_0**2
        x17 = d_1**2
        x18 = Px**2
        x19 = Py**2
        x20 = Pz**2
        x21 = 2*d_4
        x22 = r_13*x21
        x23 = Px*x22
        x24 = r_23*x21
        x25 = Py*x24
        x26 = 2*Pz
        x27 = x11*x26
        x28 = Px*x7
        x29 = r_13*x28
        x30 = 2*d_2
        x31 = Py*x7
        x32 = r_23*x31
        x33 = d_4**2
        x34 = r_13**2
        x35 = x33*x34
        x36 = r_23**2
        x37 = x33*x36
        x38 = r_33**2
        x39 = x33*x38
        x40 = d_4*x7
        x41 = x34*x40
        x42 = x36*x40
        x43 = x38*x40
        x44 = r_11**2
        x45 = d_2**2
        x46 = x0**2
        x47 = x45*x46
        x48 = x44*x47
        x49 = x7**2
        x50 = x45*x49
        x51 = x34*x50
        x52 = r_21**2
        x53 = x47*x52
        x54 = x36*x50
        x55 = r_31**2
        x56 = x47*x55
        x57 = x38*x50
        x58 = x12*x26 + x16 + x17 - x18 - x19 - x20 + x23 + x25 + x27 + x29*x30 + x30*x32 - x30*x41 - x30*x42 - x30*x43 - x35 - x37 - x39 - x48 - x51 - x53 - x54 - x56 - x57
        x59 = -x15*x58
        x60 = x14 + x4 + x59 + x6 + x9
        x61 = R_l_inv_67*x1 + R_l_inv_77*x2
        x62 = math.cos(alpha_3)
        x63 = a_0*x62
        x64 = x61*x63
        x65 = -x64
        x66 = R_l_inv_66*x1 + R_l_inv_76*x2
        x67 = d_1*x5
        x68 = x66*x67
        x69 = -x68
        x70 = x65 + x69
        x71 = d_1*x10
        x72 = x62 - x71
        x73 = 4*d_1
        x74 = a_0*x73
        x75 = x15*x74
        x76 = 2*R_l_inv_63
        x77 = 2*R_l_inv_73
        x78 = x1*x76 + x2*x77
        x79 = x62*x78
        x80 = 2*d_1
        x81 = x3*x80
        x82 = -x79 - x81
        x83 = x75 + x82
        x84 = -x62 + x71
        x85 = x64 + x68
        x86 = x79 + x81
        x87 = 2*a_0
        x88 = x10*x87
        x89 = a_0*x5
        x90 = 2*x89
        x91 = x66*x90
        x92 = x88 + x91
        x93 = 4*x5
        x94 = d_1*x93
        x95 = x66*x94
        x96 = 4*x62
        x97 = -4*x71 + x96
        x98 = x14 + x59 + x84 + x9
        x99 = -x4
        x100 = x6 + x99
        x101 = x65 + x68
        x102 = x75 + x86
        x103 = x64 + x69
        x104 = x14 + x59 + x72 + x9
        x105 = d_1*x62
        x106 = Px*x0
        x107 = r_11*x106
        x108 = Py*x0
        x109 = r_21*x108
        x110 = Pz*x1
        x111 = r_11*x0
        x112 = d_4*r_13
        x113 = x111*x112
        x114 = r_21*x0
        x115 = d_4*r_23
        x116 = x114*x115
        x117 = x1*x11
        x118 = r_13*x7
        x119 = x118*x30
        x120 = r_23*x7
        x121 = x120*x30
        x122 = 2*x12
        x123 = -x1*x122 + x107 + x109 + x110 - x111*x119 - x113 - x114*x121 - x116 - x117
        x124 = r_12*x106
        x125 = r_22*x108
        x126 = Pz*x2
        x127 = r_12*x0
        x128 = x112*x127
        x129 = r_22*x0
        x130 = x115*x129
        x131 = x11*x2
        x132 = -x119*x127 - x121*x129 - x122*x2 + x124 + x125 + x126 - x128 - x130 - x131
        x133 = R_l_inv_62*x123 + R_l_inv_72*x132
        x134 = d_1*x133
        x135 = -x13*x133
        x136 = R_l_inv_65*x123 + R_l_inv_75*x132
        x137 = -x136*x58
        x138 = d_2*x46
        x139 = x138*x44
        x140 = d_2*x49
        x141 = x140*x34
        x142 = x138*x52
        x143 = x140*x36
        x144 = x138*x55
        x145 = x140*x38
        x146 = -x29
        x147 = -x32
        x148 = Pz*x8
        x149 = -x148
        x150 = x105 - x134 + x135 + x137 + x139 + x141 + x142 + x143 + x144 + x145 + x146 + x147 + x149 + x41 + x42 + x43
        x151 = R_l_inv_60*x123 + R_l_inv_70*x132
        x152 = a_0*x151
        x153 = x5*(R_l_inv_64*x123 + R_l_inv_74*x132)
        x154 = x152 + x153
        x155 = R_l_inv_67*x123 + R_l_inv_77*x132
        x156 = x155*x63
        x157 = -x156
        x158 = R_l_inv_66*x123 + R_l_inv_76*x132
        x159 = x158*x67
        x160 = -x159
        x161 = x157 + x160
        x162 = x123*x76 + x132*x77
        x163 = x162*x62
        x164 = x151*x80
        x165 = -x163 - x164
        x166 = x136*x74 - 2*x63
        x167 = x165 + x166
        x168 = x156 + x159
        x169 = x105 + x134 + x135 + x137 + x139 + x141 + x142 + x143 + x144 + x145 + x146 + x147 + x149 + x41 + x42 + x43
        x170 = x133*x87
        x171 = x158*x90
        x172 = x170 + x171
        x173 = x163 + x164
        x174 = -4*x134
        x175 = x158*x94
        x176 = -x152
        x177 = x153 + x176
        x178 = x157 + x159
        x179 = x166 + x173
        x180 = x156 + x160
        x181 = r_11*x108
        x182 = r_21*x106
        x183 = -x111*x115 + x112*x114 + x181 - x182
        x184 = r_12*x108
        x185 = r_22*x106
        x186 = x112*x129 - x115*x127 + x184 - x185
        x187 = x5*(R_l_inv_64*x183 + R_l_inv_74*x186)
        x188 = R_l_inv_62*x183 + R_l_inv_72*x186
        x189 = -x13*x188
        x190 = R_l_inv_65*x183 + R_l_inv_75*x186
        x191 = -x190*x58
        x192 = r_23*x28
        x193 = -r_13*x31
        x194 = R_l_inv_67*x183 + R_l_inv_77*x186
        x195 = x194*x63
        x196 = -x195
        x197 = x187 + x189 + x191 + x192 + x193 + x196
        x198 = R_l_inv_60*x183 + R_l_inv_70*x186
        x199 = a_0*x198
        x200 = R_l_inv_66*x183 + R_l_inv_76*x186
        x201 = x200*x67
        x202 = -x201
        x203 = x199 + x202
        x204 = d_1*x188
        x205 = -x204
        x206 = x205 + x89
        x207 = 2*x67
        x208 = -x207
        x209 = x190*x74
        x210 = x208 + x209
        x211 = x183*x76 + x186*x77
        x212 = x211*x62
        x213 = x198*x80
        x214 = -x212 - x213
        x215 = x201 + x204
        x216 = x187 + x189 + x191 + x192 + x193 + x195
        x217 = x212 + x213
        x218 = x188*x87
        x219 = x200*x90
        x220 = x218 + x219
        x221 = -4*x204
        x222 = x200*x94
        x223 = -x199
        x224 = -x89
        x225 = x223 + x224
        x226 = x207 + x209
        x227 = x114*x18
        x228 = x114*x20
        x229 = 2*Py
        x230 = x107*x229
        x231 = x110*x229
        x232 = x114*x19
        x233 = x107*x24
        x234 = x182*x22
        x235 = x181*x22
        x236 = x109*x24
        x237 = x117*x229
        x238 = x114*x27
        x239 = x110*x24
        x240 = x114*x35
        x241 = x114*x39
        x242 = 2*r_23
        x243 = x242*x33
        x244 = r_13*x111
        x245 = x243*x244
        x246 = r_33*x243
        x247 = x1*x246
        x248 = 4*d_2
        x249 = x118*x248
        x250 = x181*x249
        x251 = x120*x248
        x252 = x109*x251
        x253 = 4*x12
        x254 = Py*x253
        x255 = x1*x254
        x256 = x114*x37
        x257 = r_23*x248*x40
        x258 = x244*x257
        x259 = x115*x253
        x260 = x1*x259
        x261 = x0**3*x45
        x262 = r_21**3*x261
        x263 = x248*x42
        x264 = x114*x263
        x265 = r_21*x261
        x266 = x265*x44
        x267 = x265*x55
        x268 = x242*x50
        x269 = x244*x268
        x270 = r_33*x1
        x271 = x268*x270
        x272 = x114*x51
        x273 = 3*x54
        x274 = x114*x273
        x275 = x114*x57
        x276 = x227 + x228 - x230 - x231 - x232 + x233 - x234 + x235 + x236 + x237 - x238 + x239 + x240 + x241 - x245 - x247 + x250 + x252 + x255 - x256 - x258 - x260 - x262 - x264 - x266 - x267 - x269 - x271 - x272 - x274 - x275
        x277 = 2*Px
        x278 = r_13*x243
        x279 = 2*x33
        x280 = r_13*x279
        x281 = Px*x253
        x282 = r_13*x257
        x283 = x112*x253
        x284 = x248*x41
        x285 = r_11*x261
        x286 = r_13*x268
        x287 = 2*x50
        x288 = r_13*x287
        x289 = 3*x51
        x290 = -r_11**3*x261 + x1*x281 - x1*x283 + x107*x22 + x107*x249 + x109*x22 + x110*x22 - x110*x277 - x111*x18 + x111*x19 + x111*x20 - x111*x27 - x111*x284 - x111*x289 - x111*x35 + x111*x37 + x111*x39 - x111*x54 - x111*x57 - x114*x278 - x114*x282 - x114*x286 + x117*x277 - x181*x24 - x182*x229 + x182*x24 + x182*x251 - x270*x280 - x270*x288 - x285*x52 - x285*x55
        x291 = x129*x18
        x292 = x129*x20
        x293 = x124*x229
        x294 = x126*x229
        x295 = x129*x19
        x296 = x124*x24
        x297 = x185*x22
        x298 = x184*x22
        x299 = x125*x24
        x300 = x131*x229
        x301 = x129*x27
        x302 = x126*x24
        x303 = x129*x35
        x304 = x129*x39
        x305 = x127*x278
        x306 = x2*x246
        x307 = x184*x249
        x308 = x125*x251
        x309 = x2*x254
        x310 = x129*x37
        x311 = x127*x282
        x312 = x2*x259
        x313 = 2*x265
        x314 = r_11*x313
        x315 = r_12*x314
        x316 = r_31*r_32
        x317 = x313*x316
        x318 = x129*x263
        x319 = r_22*x261
        x320 = x319*x44
        x321 = 3*x319*x52
        x322 = x319*x55
        x323 = x127*x286
        x324 = r_33*x2
        x325 = x268*x324
        x326 = x129*x51
        x327 = x129*x273
        x328 = x129*x57
        x329 = x291 + x292 - x293 - x294 - x295 + x296 - x297 + x298 + x299 + x300 - x301 + x302 + x303 + x304 - x305 - x306 + x307 + x308 + x309 - x310 - x311 - x312 - x315 - x317 - x318 - x320 - x321 - x322 - x323 - x325 - x326 - x327 - x328
        x330 = r_33*x280
        x331 = 2*x285
        x332 = r_12*x261
        x333 = -r_22*x314 + x124*x22 + x124*x249 + x125*x22 + x126*x22 - x126*x277 - x127*x18 + x127*x19 + x127*x20 - x127*x27 - x127*x284 - x127*x289 - x127*x35 + x127*x37 + x127*x39 - x127*x54 - x127*x57 - x129*x278 - x129*x282 - x129*x286 + x131*x277 - x184*x24 - x185*x229 + x185*x24 + x185*x251 + x2*x281 - x2*x283 - x2*x330 - x288*x324 - x316*x331 - 3*x332*x44 - x332*x52 - x332*x55
        x334 = x120*x18
        x335 = x120*x20
        x336 = x229*x29
        x337 = x148*x229
        x338 = x120*x19
        x339 = x229*x41
        x340 = x229*x42
        x341 = x229*x43
        x342 = r_23**3
        x343 = x33*x7
        x344 = x342*x343
        x345 = x139*x229
        x346 = x141*x229
        x347 = x142*x229
        x348 = x143*x229
        x349 = x144*x229
        x350 = x145*x229
        x351 = x140*x21
        x352 = x342*x351
        x353 = x120*x35
        x354 = x120*x39
        x355 = x45*x7**3
        x356 = x342*x355
        x357 = x139*x24
        x358 = x141*x24
        x359 = x142*x24
        x360 = x144*x24
        x361 = x145*x24
        x362 = r_23*x355
        x363 = x34*x362
        x364 = x362*x38
        x365 = 2*r_21*x47
        x366 = r_11*x365
        x367 = x118*x366
        x368 = r_31*x8
        x369 = x365*x368
        x370 = x120*x48
        x371 = 3*x120*x53
        x372 = x120*x56
        x373 = x334 + x335 - x336 - x337 - x338 + x339 + x340 + x341 - x344 + x345 + x346 + x347 + x348 + x349 + x350 - x352 - x353 - x354 - x356 - x357 - x358 - x359 - x360 - x361 - x363 - x364 - x367 - x369 - x370 - x371 - x372
        x374 = r_13**3
        x375 = r_13*x355
        x376 = 2*r_11*x47
        x377 = -x118*x18 + x118*x19 + x118*x20 - x118*x37 - x118*x39 - 3*x118*x48 - x118*x53 - x118*x56 - x120*x366 - x139*x22 + x139*x277 + x141*x277 - x142*x22 + x142*x277 - x143*x22 + x143*x277 - x144*x22 + x144*x277 - x145*x22 + x145*x277 - x148*x277 - x192*x229 + x277*x41 + x277*x42 + x277*x43 - x343*x374 - x351*x374 - x355*x374 - x36*x375 - x368*x376 - x375*x38
        x378 = x5*(R_l_inv_04*x276 + R_l_inv_14*x329 + R_l_inv_24*x290 + R_l_inv_34*x333 + R_l_inv_44*x373 + R_l_inv_54*x377)
        x379 = R_l_inv_07*x276 + R_l_inv_17*x329 + R_l_inv_27*x290 + R_l_inv_37*x333 + R_l_inv_47*x373 + R_l_inv_57*x377
        x380 = x379*x63
        x381 = -x380
        x382 = x378 + x381
        x383 = R_l_inv_00*x276 + R_l_inv_10*x329 + R_l_inv_20*x290 + R_l_inv_30*x333 + R_l_inv_40*x373 + R_l_inv_50*x377
        x384 = a_0*x383
        x385 = x105*x87
        x386 = x384 + x385
        x387 = R_l_inv_06*x276 + R_l_inv_16*x329 + R_l_inv_26*x290 + R_l_inv_36*x333 + R_l_inv_46*x373 + R_l_inv_56*x377
        x388 = x387*x67
        x389 = -x388
        x390 = R_l_inv_02*x276 + R_l_inv_12*x329 + R_l_inv_22*x290 + R_l_inv_32*x333 + R_l_inv_42*x373 + R_l_inv_52*x377
        x391 = d_1*x390
        x392 = -x13*x390
        x393 = R_l_inv_05*x276 + R_l_inv_15*x329 + R_l_inv_25*x290 + R_l_inv_35*x333 + R_l_inv_45*x373 + R_l_inv_55*x377
        x394 = -x393*x58
        x395 = -x391 + x392 + x394
        x396 = x389 + x395
        x397 = x16*x62
        x398 = 2*x397
        x399 = -x398
        x400 = x393*x74
        x401 = 2*R_l_inv_03
        x402 = 2*R_l_inv_23
        x403 = 2*R_l_inv_13
        x404 = 2*R_l_inv_33
        x405 = 2*R_l_inv_43
        x406 = 2*R_l_inv_53
        x407 = x276*x401 + x290*x402 + x329*x403 + x333*x404 + x373*x405 + x377*x406
        x408 = x407*x62
        x409 = x383*x80
        x410 = x17*x62
        x411 = 2*x410
        x412 = -x408 - x409 - x411
        x413 = x399 + x400 + x412
        x414 = x378 + x380
        x415 = x386 + x388
        x416 = x391 + x392 + x394
        x417 = x390*x87
        x418 = x387*x90
        x419 = x417 + x418
        x420 = x408 + x409 + x411
        x421 = x399 + x420
        x422 = -4*x391
        x423 = x387*x94
        x424 = x398 + x412
        x425 = -x384 - x385
        x426 = x416 + x425
        x427 = x398 + x400 + x420
        x428 = -x227 - x228 + x230 + x231 + x232 - x233 + x234 - x235 - x236 - x237 + x238 - x239 - x240 - x241 + x245 + x247 - x250 - x252 - x255 + x256 + x258 + x260 + x262 + x264 + x266 + x267 + x269 + x271 + x272 + x274 + x275
        x429 = -x334 - x335 + x336 + x337 + x338 - x339 - x340 - x341 + x344 - x345 - x346 - x347 - x348 - x349 - x350 + x352 + x353 + x354 + x356 + x357 + x358 + x359 + x360 + x361 + x363 + x364 + x367 + x369 + x370 + x371 + x372
        x430 = -x291 - x292 + x293 + x294 + x295 - x296 + x297 - x298 - x299 - x300 + x301 - x302 - x303 - x304 + x305 + x306 - x307 - x308 - x309 + x310 + x311 + x312 + x315 + x317 + x318 + x320 + x321 + x322 + x323 + x325 + x326 + x327 + x328
        x431 = R_l_inv_00*x290 + R_l_inv_10*x333 + R_l_inv_20*x428 + R_l_inv_30*x430 + R_l_inv_40*x377 + R_l_inv_50*x429
        x432 = a_0*x431
        x433 = R_l_inv_06*x290 + R_l_inv_16*x333 + R_l_inv_26*x428 + R_l_inv_36*x430 + R_l_inv_46*x377 + R_l_inv_56*x429
        x434 = x433*x67
        x435 = -x434
        x436 = x432 + x435
        x437 = R_l_inv_07*x290 + R_l_inv_17*x333 + R_l_inv_27*x428 + R_l_inv_37*x430 + R_l_inv_47*x377 + R_l_inv_57*x429
        x438 = x437*x63
        x439 = -x438
        x440 = R_l_inv_02*x290 + R_l_inv_12*x333 + R_l_inv_22*x428 + R_l_inv_32*x430 + R_l_inv_42*x377 + R_l_inv_52*x429
        x441 = d_1*x440
        x442 = -x13*x440
        x443 = R_l_inv_05*x290 + R_l_inv_15*x333 + R_l_inv_25*x428 + R_l_inv_35*x430 + R_l_inv_45*x377 + R_l_inv_55*x429
        x444 = -x443*x58
        x445 = -x441 + x442 + x444
        x446 = x439 + x445
        x447 = x5*(R_l_inv_04*x290 + R_l_inv_14*x333 + R_l_inv_24*x428 + R_l_inv_34*x430 + R_l_inv_44*x377 + R_l_inv_54*x429)
        x448 = x16*x5
        x449 = x17*x5
        x450 = x447 - x448 - x449
        x451 = x290*x401 + x333*x403 + x377*x405 + x402*x428 + x404*x430 + x406*x429
        x452 = x451*x62
        x453 = x431*x80
        x454 = -x452 - x453
        x455 = x73*x89
        x456 = x443*x74
        x457 = x455 + x456
        x458 = x438 + x450
        x459 = x432 + x434
        x460 = x441 + x442 + x444
        x461 = x440*x87
        x462 = x433*x90
        x463 = x461 + x462
        x464 = x452 + x453
        x465 = -4*x441
        x466 = x433*x94
        x467 = -x432
        x468 = x434 + x467
        x469 = x439 + x460
        x470 = x435 + x467
        x471 = 2*x11
        x472 = Pz*x22
        x473 = Pz*x24
        x474 = r_33*x244
        x475 = Pz*x249
        x476 = Pz*x251
        x477 = x248*x43
        x478 = r_31*x261
        x479 = r_33*x268
        x480 = 3*x57
        x481 = -r_31**3*x261 + x1*x18 + x1*x19 - x1*x20 - x1*x23 - x1*x25 + x1*x35 + x1*x37 - x1*x39 - x1*x477 - x1*x480 - x1*x51 - x1*x54 - x107*x26 + x107*x471 - x109*x26 + x109*x471 + x110*x253 + x110*x471 + x111*x472 + x111*x475 - x113*x253 - x114*x246 + x114*x473 + x114*x476 - x114*x479 - x116*x253 - x279*x474 - x287*x474 - x44*x478 - x478*x52
        x482 = r_32*x261
        x483 = -r_12*r_31*x331 - r_22*r_31*x313 - r_33*x127*x288 - x124*x26 + x124*x471 - x125*x26 + x125*x471 + x126*x253 + x126*x471 - x127*x330 + x127*x472 + x127*x475 - x128*x253 - x129*x246 + x129*x473 + x129*x476 - x129*x479 - x130*x253 + x18*x2 + x19*x2 - x2*x20 - x2*x23 - x2*x25 + x2*x35 + x2*x37 - x2*x39 - x2*x477 - x2*x480 - x2*x51 - x2*x54 - x44*x482 - x482*x52 - 3*x482*x55
        x484 = R_l_inv_60*x481 + R_l_inv_70*x483
        x485 = a_0*x484
        x486 = x5*(R_l_inv_64*x481 + R_l_inv_74*x483)
        x487 = R_l_inv_62*x481 + R_l_inv_72*x483
        x488 = -x13*x487
        x489 = R_l_inv_65*x481 + R_l_inv_75*x483
        x490 = -x489*x58
        x491 = x20*x8
        x492 = r_33**3
        x493 = x355*x492
        x494 = x343*x492
        x495 = -x18*x8
        x496 = -x19*x8
        x497 = r_33*x355
        x498 = x34*x497
        x499 = x36*x497
        x500 = x35*x8
        x501 = x37*x8
        x502 = -x139*x26
        x503 = -x141*x26
        x504 = -x142*x26
        x505 = -x143*x26
        x506 = -x144*x26
        x507 = -x145*x26
        x508 = -x26*x41
        x509 = -x26*x42
        x510 = -x26*x43
        x511 = x26*x29
        x512 = x26*x32
        x513 = x351*x492
        x514 = x48*x8
        x515 = x53*x8
        x516 = x139*x471
        x517 = x141*x471
        x518 = x142*x471
        x519 = x143*x471
        x520 = x144*x471
        x521 = 3*x56*x8
        x522 = r_31*x118*x376
        x523 = r_31*x120*x365
        x524 = x485 + x486 + x488 + x490 + x491 + x493 + x494 + x495 + x496 + x498 + x499 + x500 + x501 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x517 + x518 + x519 + x520 + x521 + x522 + x523
        x525 = R_l_inv_67*x481 + R_l_inv_77*x483
        x526 = x525*x63
        x527 = -x526
        x528 = R_l_inv_66*x481 + R_l_inv_76*x483
        x529 = x528*x67
        x530 = -x529
        x531 = x527 + x530
        x532 = d_1*x487
        x533 = x397 - x410 - x532
        x534 = x489*x74
        x535 = x481*x76 + x483*x77
        x536 = x535*x62
        x537 = x484*x80
        x538 = -x536 - x537
        x539 = x534 + x538
        x540 = x526 + x529
        x541 = -x397 + x410 + x532
        x542 = x536 + x537
        x543 = x528*x90
        x544 = a_0*d_1*x96 + x487*x87
        x545 = x543 + x544
        x546 = x528*x94
        x547 = -x16*x96 - x17*x96 - 4*x532
        x548 = x488 + x490 + x491 + x493 + x494 + x495 + x496 + x498 + x499 + x500 + x501 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x517 + x518 + x519 + x520 + x521 + x522 + x523 + x541
        x549 = -x485
        x550 = x486 + x549
        x551 = x527 + x529
        x552 = x534 + x542
        x553 = x526 + x530
        x554 = x488 + x490 + x491 + x493 + x494 + x495 + x496 + x498 + x499 + x500 + x501 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x517 + x518 + x519 + x520 + x521 + x522 + x523 + x533
        x555 = -x207*x61
        x556 = x5*x78
        x557 = x555 - x556
        x558 = -x93
        x559 = 4*x89
        x560 = x559*x61
        x561 = x555 + x556
        x562 = 8*R_l_inv_63
        x563 = 8*R_l_inv_73
        x564 = x162*x5
        x565 = -x564
        x566 = -x155*x207
        x567 = x566 - x90
        x568 = x155*x559
        x569 = x566 + x90
        x570 = -x194*x207
        x571 = x211*x5
        x572 = x570 - x571
        x573 = x194*x559
        x574 = x570 + x571
        x575 = 2*x448
        x576 = 2*x449
        x577 = -x207*x379
        x578 = x407*x5
        x579 = -x575 + x576 + x577 - x578
        x580 = x379*x559
        x581 = x575 - x576 + x577 + x578
        x582 = -8*d_1*x89
        x583 = 8*R_l_inv_03
        x584 = 8*R_l_inv_23
        x585 = 8*R_l_inv_13
        x586 = 8*R_l_inv_33
        x587 = 8*R_l_inv_43
        x588 = 8*R_l_inv_53
        x589 = -x207*x437
        x590 = x451*x5
        x591 = x589 - x590
        x592 = x437*x559
        x593 = x589 + x590
        x594 = x5*x535
        x595 = -x594
        x596 = -x207*x525
        x597 = x455 + x596
        x598 = 4*x448
        x599 = -x598
        x600 = 4*x449
        x601 = -x600
        x602 = x525*x559
        x603 = -x455
        x604 = x596 + x603
        x605 = -x6
        x606 = x4 + x605
        x607 = x88 - x91
        x608 = x605 + x99
        x609 = -x153
        x610 = x152 + x609
        x611 = x170 - x171
        x612 = x176 + x609
        x613 = -x187
        x614 = x189 + x191 + x192 + x193 + x224 + x613
        x615 = x218 - x219
        x616 = x189 + x191 + x192 + x193 + x223 + x613
        x617 = -x378
        x618 = x381 + x617
        x619 = x380 + x617
        x620 = x417 - x418
        x621 = -x447 + x448 + x449
        x622 = x456 + x603
        x623 = x438 + x621
        x624 = x461 - x462
        x625 = -x486
        x626 = x485 + x625
        x627 = -x543 + x544
        x628 = x549 + x625
        # End of temp variable
        A = np.zeros(shape=(6, 9))
        A[0, 0] = x60 + x70 + x72
        A[0, 1] = x83
        A[0, 2] = x60 + x84 + x85
        A[0, 3] = x86 + x92
        A[0, 4] = -x95 + x97
        A[0, 5] = x82 + x92
        A[0, 6] = x100 + x101 + x98
        A[0, 7] = x102
        A[0, 8] = x100 + x103 + x104
        A[1, 0] = x150 + x154 + x161
        A[1, 1] = x167
        A[1, 2] = x154 + x168 + x169
        A[1, 3] = x172 + x173
        A[1, 4] = x174 - x175
        A[1, 5] = x165 + x172
        A[1, 6] = x169 + x177 + x178
        A[1, 7] = x179
        A[1, 8] = x150 + x177 + x180
        A[2, 0] = x197 + x203 + x206
        A[2, 1] = x210 + x214
        A[2, 2] = x199 + x215 + x216 + x89
        A[2, 3] = x207 + x217 + x220
        A[2, 4] = x221 - x222
        A[2, 5] = x208 + x214 + x220
        A[2, 6] = x197 + x215 + x225
        A[2, 7] = x217 + x226
        A[2, 8] = x202 + x205 + x216 + x225
        A[3, 0] = x382 + x386 + x396
        A[3, 1] = x413
        A[3, 2] = x414 + x415 + x416
        A[3, 3] = x419 + x421
        A[3, 4] = x422 - x423
        A[3, 5] = x419 + x424
        A[3, 6] = x382 + x388 + x426
        A[3, 7] = x427
        A[3, 8] = x396 + x414 + x425
        A[4, 0] = x436 + x446 + x450
        A[4, 1] = x454 + x457
        A[4, 2] = x458 + x459 + x460
        A[4, 3] = x463 + x464
        A[4, 4] = x465 - x466
        A[4, 5] = x454 + x463
        A[4, 6] = x450 + x468 + x469
        A[4, 7] = x457 + x464
        A[4, 8] = x445 + x458 + x470
        A[5, 0] = x524 + x531 + x533
        A[5, 1] = x539
        A[5, 2] = x524 + x540 + x541
        A[5, 3] = x542 + x545
        A[5, 4] = -x546 + x547
        A[5, 5] = x538 + x545
        A[5, 6] = x548 + x550 + x551
        A[5, 7] = x552
        A[5, 8] = x550 + x553 + x554
        B = np.zeros(shape=(6, 9))
        B[0, 0] = x557
        B[0, 1] = x558 + x560
        B[0, 2] = x561
        B[0, 3] = x93
        B[0, 4] = x5*(-x1*x562 - x2*x563)
        B[0, 5] = x558
        B[0, 6] = x561
        B[0, 7] = x560 + x93
        B[0, 8] = x557
        B[1, 0] = x565 + x567
        B[1, 1] = x568
        B[1, 2] = x564 + x569
        B[1, 4] = x5*(-x123*x562 - x132*x563)
        B[1, 6] = x564 + x567
        B[1, 7] = x568
        B[1, 8] = x565 + x569
        B[2, 0] = x572
        B[2, 1] = x573
        B[2, 2] = x574
        B[2, 4] = x5*(-x183*x562 - x186*x563)
        B[2, 6] = x574
        B[2, 7] = x573
        B[2, 8] = x572
        B[3, 0] = x579
        B[3, 1] = x580
        B[3, 2] = x581
        B[3, 3] = x582
        B[3, 4] = 8*x448 + 8*x449 - x5*(x276*x583 + x290*x584 + x329*x585 + x333*x586 + x373*x587 + x377*x588)
        B[3, 5] = x582
        B[3, 6] = x581
        B[3, 7] = x580
        B[3, 8] = x579
        B[4, 0] = x591
        B[4, 1] = x592
        B[4, 2] = x593
        B[4, 4] = x5*(-x290*x583 - x333*x585 - x377*x587 - x428*x584 - x429*x588 - x430*x586)
        B[4, 6] = x593
        B[4, 7] = x592
        B[4, 8] = x591
        B[5, 0] = x595 + x597
        B[5, 1] = x599 + x601 + x602
        B[5, 2] = x594 + x597
        B[5, 3] = x599 + x600
        B[5, 4] = x5*(-x481*x562 - x483*x563)
        B[5, 5] = x598 + x601
        B[5, 6] = x594 + x604
        B[5, 7] = x598 + x600 + x602
        B[5, 8] = x595 + x604
        C = np.zeros(shape=(6, 9))
        C[0, 0] = x101 + x104 + x606
        C[0, 1] = x83
        C[0, 2] = x103 + x606 + x98
        C[0, 3] = x607 + x86
        C[0, 4] = x95 + x97
        C[0, 5] = x607 + x82
        C[0, 6] = x608 + x70 + x98
        C[0, 7] = x102
        C[0, 8] = x104 + x608 + x85
        C[1, 0] = x150 + x178 + x610
        C[1, 1] = x167
        C[1, 2] = x169 + x180 + x610
        C[1, 3] = x173 + x611
        C[1, 4] = x174 + x175
        C[1, 5] = x165 + x611
        C[1, 6] = x161 + x169 + x612
        C[1, 7] = x179
        C[1, 8] = x150 + x168 + x612
        C[2, 0] = x196 + x199 + x201 + x205 + x614
        C[2, 1] = x214 + x226
        C[2, 2] = x195 + x203 + x204 + x614
        C[2, 3] = x208 + x217 + x615
        C[2, 4] = x221 + x222
        C[2, 5] = x207 + x214 + x615
        C[2, 6] = x196 + x202 + x204 + x616 + x89
        C[2, 7] = x210 + x217
        C[2, 8] = x195 + x201 + x206 + x616
        C[3, 0] = x395 + x415 + x618
        C[3, 1] = x413
        C[3, 2] = x386 + x389 + x416 + x619
        C[3, 3] = x421 + x620
        C[3, 4] = x422 + x423
        C[3, 5] = x424 + x620
        C[3, 6] = x389 + x426 + x618
        C[3, 7] = x427
        C[3, 8] = x388 + x395 + x425 + x619
        C[4, 0] = x446 + x459 + x621
        C[4, 1] = x454 + x622
        C[4, 2] = x436 + x460 + x623
        C[4, 3] = x464 + x624
        C[4, 4] = x465 + x466
        C[4, 5] = x454 + x624
        C[4, 6] = x469 + x470 + x621
        C[4, 7] = x464 + x622
        C[4, 8] = x445 + x468 + x623
        C[5, 0] = x551 + x554 + x626
        C[5, 1] = x539
        C[5, 2] = x548 + x553 + x626
        C[5, 3] = x542 + x627
        C[5, 4] = x546 + x547
        C[5, 5] = x538 + x627
        C[5, 6] = x531 + x548 + x628
        C[5, 7] = x552
        C[5, 8] = x540 + x554 + x628
        local_solutions = compute_solution_from_tanhalf_LME(A, B, C)
        for local_solutions_i in local_solutions:
            solution_i: IkSolution = make_ik_solution()
            solution_i[4] = local_solutions_i
            appended_idx = append_solution_to_queue(solution_i)
            add_input_index_to(2, appended_idx)
    # Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_3_processor()
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
            th_3 = this_solution[4]
            condition_0: bool = (2*abs(a_0*d_2*math.sin(alpha_3)*math.sin(th_3)) >= zero_tolerance) or (abs(2*a_0*d_1 + 2*a_0*d_2*math.cos(alpha_3)) >= zero_tolerance) or (abs(-a_0**2 - d_1**2 - 2*d_1*d_2*math.cos(alpha_3) - d_2**2 + d_4**2 + 2*d_4*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = 2*a_0
                x1 = d_2*math.cos(alpha_3)
                x2 = -d_1*x0 - x0*x1
                x3 = math.sin(alpha_3)
                x4 = math.sin(th_3)
                x5 = math.atan2(x2, d_2*x0*x3*x4)
                x6 = a_0**2
                x7 = d_2**2
                x8 = -d_1**2 - 2*d_1*x1 + d_4**2 + 2*d_4*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2 - x6 - x7
                x9 = safe_sqrt(x2**2 + 4*x3**2*x4**2*x6*x7 - x8**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[3] = x5 + math.atan2(x9, x8)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (2*abs(a_0*d_2*math.sin(alpha_3)*math.sin(th_3)) >= zero_tolerance) or (abs(2*a_0*d_1 + 2*a_0*d_2*math.cos(alpha_3)) >= zero_tolerance) or (abs(-a_0**2 - d_1**2 - 2*d_1*d_2*math.cos(alpha_3) - d_2**2 + d_4**2 + 2*d_4*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = 2*a_0
                x1 = d_2*math.cos(alpha_3)
                x2 = -d_1*x0 - x0*x1
                x3 = math.sin(alpha_3)
                x4 = math.sin(th_3)
                x5 = math.atan2(x2, d_2*x0*x3*x4)
                x6 = a_0**2
                x7 = d_2**2
                x8 = -d_1**2 - 2*d_1*x1 + d_4**2 + 2*d_4*inv_Pz + inv_Px**2 + inv_Py**2 + inv_Pz**2 - x6 - x7
                x9 = safe_sqrt(x2**2 + 4*x3**2*x4**2*x6*x7 - x8**2)
                # End of temp variables
                this_solution[3] = x5 + math.atan2(-x9, x8)
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
            th_3 = this_solution[4]
            checked_result: bool = (abs(d_2*math.sin(alpha_3)*math.cos(th_3)) <= 1.0e-6) and (abs(Px - d_4*r_13) <= 1.0e-6) and (abs(Py - d_4*r_23) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor()
    # Finish code for equation all-zero dispatcher node 4
    
    # Code for explicit solution node 5, solved variable is th_0
    def ExplicitSolutionNode_node_5_solve_th_0_processor():
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
            th_3 = this_solution[4]
            condition_0: bool = (abs(d_2*math.sin(alpha_3)*math.cos(th_3)) >= zero_tolerance) or (abs(Px - d_4*r_13) >= zero_tolerance) or (abs(Py - d_4*r_23) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = Px - d_4*r_13
                x1 = -Py + d_4*r_23
                x2 = math.atan2(x0, x1)
                x3 = math.sin(alpha_3)
                x4 = math.cos(th_3)
                x5 = safe_sqrt(-d_2**2*x3**2*x4**2 + x0**2 + x1**2)
                x6 = d_2*x3*x4
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x2 + math.atan2(x5, x6)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(6, appended_idx)
                
            condition_1: bool = (abs(d_2*math.sin(alpha_3)*math.cos(th_3)) >= zero_tolerance) or (abs(Px - d_4*r_13) >= zero_tolerance) or (abs(Py - d_4*r_23) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = Px - d_4*r_13
                x1 = -Py + d_4*r_23
                x2 = math.atan2(x0, x1)
                x3 = math.sin(alpha_3)
                x4 = math.cos(th_3)
                x5 = safe_sqrt(-d_2**2*x3**2*x4**2 + x0**2 + x1**2)
                x6 = d_2*x3*x4
                # End of temp variables
                this_solution[1] = x2 + math.atan2(-x5, x6)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_0_processor()
    # Finish code for explicit solution node 5
    
    # Code for equation all-zero dispatcher node 6
    def EquationAllZeroDispatcherNode_node_6_processor():
        this_node_input_index: List[int] = node_input_index[6]
        this_input_valid: bool = node_input_validity[6]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_0 = this_solution[1]
            checked_result: bool = (abs(Pz - d_4*r_33) <= 1.0e-6) and (abs(Px*math.cos(th_0) + Py*math.sin(th_0) - d_4*r_13*math.cos(th_0) - d_4*r_23*math.sin(th_0)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(7, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_6_processor()
    # Finish code for equation all-zero dispatcher node 6
    
    # Code for explicit solution node 7, solved variable is th_1
    def ExplicitSolutionNode_node_7_solve_th_1_processor():
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
            th_0 = this_solution[1]
            th_2 = this_solution[3]
            th_3 = this_solution[4]
            condition_0: bool = (abs(Pz - d_4*r_33) >= 1.0e-6) or (abs(Px*math.cos(th_0) + Py*math.sin(th_0) - d_4*r_13*math.cos(th_0) - d_4*r_23*math.sin(th_0)) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = Pz - d_4*r_33
                x1 = math.sin(th_2)
                x2 = d_2*math.cos(alpha_3)
                x3 = math.cos(th_2)
                x4 = d_2*math.sin(alpha_3)*math.sin(th_3)
                x5 = -a_0 + d_1*x1 + x1*x2 - x3*x4
                x6 = d_1*x3 + x1*x4 + x2*x3
                x7 = math.cos(th_0)
                x8 = math.sin(th_0)
                x9 = -Px*x7 - Py*x8 + d_4*r_13*x7 + d_4*r_23*x8
                # End of temp variables
                this_solution[2] = math.atan2(x0*x5 - x6*x9, x0*x6 + x5*x9)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_1_processor()
    # Finish code for explicit solution node 7
    
    # Code for non-branch dispatcher node 8
    # Actually, there is no code
    
    # Code for explicit solution node 9, solved variable is negative_th_2_positive_th_1__soa
    def ExplicitSolutionNode_node_9_solve_negative_th_2_positive_th_1__soa_processor():
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
            th_1 = this_solution[2]
            th_2 = this_solution[3]
            condition_0: bool = (1 >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                # End of temp variables
                this_solution[0] = th_1 - th_2
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_negative_th_2_positive_th_1__soa_processor()
    # Finish code for explicit solution node 8
    
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
            negative_th_2_positive_th_1__soa = this_solution[0]
            th_0 = this_solution[1]
            th_3 = this_solution[4]
            condition_0: bool = (abs(math.sin(alpha_3)*math.sin(alpha_5)) >= zero_tolerance) or (abs(-r_13*(-math.sin(th_0)*math.sin(th_3) + math.cos(negative_th_2_positive_th_1__soa)*math.cos(th_0)*math.cos(th_3)) - r_23*(math.sin(th_0)*math.cos(negative_th_2_positive_th_1__soa)*math.cos(th_3) + math.sin(th_3)*math.cos(th_0)) + r_33*math.sin(negative_th_2_positive_th_1__soa)*math.cos(th_3)) >= zero_tolerance) or (abs(r_13*math.sin(negative_th_2_positive_th_1__soa)*math.cos(th_0) + r_23*math.sin(negative_th_2_positive_th_1__soa)*math.sin(th_0) + r_33*math.cos(negative_th_2_positive_th_1__soa) - math.cos(alpha_3)*math.cos(alpha_5)) >= zero_tolerance) or (abs(math.sin(alpha_5)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(alpha_5)**(-1)
                x1 = math.sin(negative_th_2_positive_th_1__soa)
                x2 = math.cos(th_3)
                x3 = math.sin(th_3)
                x4 = math.cos(th_0)
                x5 = math.sin(th_0)
                x6 = math.cos(negative_th_2_positive_th_1__soa)
                x7 = x2*x6
                # End of temp variables
                this_solution[5] = math.atan2(x0*(r_13*(-x3*x5 + x4*x7) + r_23*(x3*x4 + x5*x7) - r_33*x1*x2), x0*(-r_13*x1*x4 - r_23*x1*x5 - r_33*x6 + math.cos(alpha_3)*math.cos(alpha_5))/math.sin(alpha_3))
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(12, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_4_processor()
    # Finish code for explicit solution node 10
    
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
            negative_th_2_positive_th_1__soa = this_solution[0]
            th_0 = this_solution[1]
            th_1 = this_solution[2]
            condition_0: bool = (abs(d_2*math.sin(alpha_5)) >= zero_tolerance) or (abs(-a_0*(-r_31*math.sin(th_1) + (r_11*math.cos(th_0) + r_21*math.sin(th_0))*math.cos(th_1)) - d_1*(r_11*math.sin(negative_th_2_positive_th_1__soa)*math.cos(th_0) + r_21*math.sin(negative_th_2_positive_th_1__soa)*math.sin(th_0) + r_31*math.cos(negative_th_2_positive_th_1__soa)) - inv_Px) >= zero_tolerance) or (abs(-a_0*(-r_32*math.sin(th_1) + (r_12*math.cos(th_0) + r_22*math.sin(th_0))*math.cos(th_1)) - d_1*(r_12*math.sin(negative_th_2_positive_th_1__soa)*math.cos(th_0) + r_22*math.sin(negative_th_2_positive_th_1__soa)*math.sin(th_0) + r_32*math.cos(negative_th_2_positive_th_1__soa)) - inv_Py) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(negative_th_2_positive_th_1__soa)
                x1 = math.sin(negative_th_2_positive_th_1__soa)
                x2 = math.cos(th_0)
                x3 = r_11*x2
                x4 = math.sin(th_0)
                x5 = r_21*x4
                x6 = math.sin(th_1)
                x7 = math.cos(th_1)
                x8 = 1/(d_2*math.sin(alpha_5))
                x9 = r_12*x2
                x10 = r_22*x4
                # End of temp variables
                this_solution[6] = math.atan2(x8*(-a_0*(-r_31*x6 + x7*(x3 + x5)) - d_1*(r_31*x0 + x1*x3 + x1*x5) - inv_Px), x8*(-a_0*(-r_32*x6 + x7*(x10 + x9)) - d_1*(r_32*x0 + x1*x10 + x1*x9) - inv_Py))
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
        value_at_2 = ik_out_i[3]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[4]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = ik_out_i[5]  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[6]  # th_5
        new_ik_i[5] = value_at_5
        ik_out.append(new_ik_i)
    return ik_out


def yaskawa_mpx3500_ik_solve(T_ee: np.ndarray):
    T_ee_raw_in = yaskawa_mpx3500_ik_target_original_to_raw(T_ee)
    ik_output_raw = yaskawa_mpx3500_ik_solve_raw(T_ee_raw_in)
    ik_output = list()
    for i in range(len(ik_output_raw)):
        ik_out_i = ik_output_raw[i]
        ik_out_i[0] -= th_0_offset_original2raw
        ik_out_i[1] -= th_1_offset_original2raw
        ik_out_i[2] -= th_2_offset_original2raw
        ik_out_i[3] -= th_3_offset_original2raw
        ik_out_i[4] -= th_4_offset_original2raw
        ik_out_i[5] -= th_5_offset_original2raw
        ee_pose_i = yaskawa_mpx3500_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_yaskawa_mpx3500():
    theta_in = np.random.random(size=(6, ))
    ee_pose = yaskawa_mpx3500_fk(theta_in)
    ik_output = yaskawa_mpx3500_ik_solve(ee_pose)
    for i in range(len(ik_output)):
        ee_pose_i = yaskawa_mpx3500_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_yaskawa_mpx3500()
