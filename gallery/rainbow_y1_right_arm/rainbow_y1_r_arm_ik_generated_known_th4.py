import numpy as np
import copy
import math
from typing import List, NewType
from python_run_import import *

# Constants for solver
robot_nq: int = 7
n_tree_nodes: int = 12
pose_tolerance: float = 1e-4
zero_tolerance: float = 1e-6

# Robot parameters
a_3: float = 0.031
d_2: float = -0.276
post_transform_s4: float = -0.1548
pre_transform_s0: float = 0.34202
pre_transform_s1: float = 0.939693
pre_transform_s2: float = -0.22
pre_transform_s3: float = 0.0800735

# Unknown offsets from original unknown value to raw value
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
th_0_offset_original2raw: float = 0.0
th_1_offset_original2raw: float = -1.2217300208404673
th_2_offset_original2raw: float = -1.5707963267948966
th_3_offset_original2raw: float = 3.141592653589793
th_4_offset_original2raw: float = 3.141592653589793
th_5_offset_original2raw: float = 3.141592653589793
th_6_offset_original2raw: float = 3.141592653589793


# The transformation between raw and original ee target
# Original value are the ones corresponded to robot (usually urdf/sdf)
# Raw value are the ones used in the solver
# ee_original = pre_transform * ee_raw * post_transform
# ee_raw = dh_forward_transform(theta_raw)
def rainbow_y1_r_arm_ik_target_original_to_raw(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = pre_transform_s0*r_21 + pre_transform_s1*r_31
    ee_transformed[0, 1] = pre_transform_s0*r_22 + pre_transform_s1*r_32
    ee_transformed[0, 2] = pre_transform_s0*r_23 + pre_transform_s1*r_33
    ee_transformed[0, 3] = -pre_transform_s0*pre_transform_s2 + pre_transform_s0*(Py - post_transform_s4*r_23) - pre_transform_s1*pre_transform_s3 + pre_transform_s1*(Pz - post_transform_s4*r_33)
    ee_transformed[1, 0] = r_11
    ee_transformed[1, 1] = r_12
    ee_transformed[1, 2] = r_13
    ee_transformed[1, 3] = Px - post_transform_s4*r_13
    ee_transformed[2, 0] = -pre_transform_s0*r_31 + pre_transform_s1*r_21
    ee_transformed[2, 1] = -pre_transform_s0*r_32 + pre_transform_s1*r_22
    ee_transformed[2, 2] = -pre_transform_s0*r_33 + pre_transform_s1*r_23
    ee_transformed[2, 3] = pre_transform_s0*pre_transform_s3 - pre_transform_s0*(Pz - post_transform_s4*r_33) - pre_transform_s1*pre_transform_s2 + pre_transform_s1*(Py - post_transform_s4*r_23)
    return ee_transformed


def rainbow_y1_r_arm_ik_target_raw_to_original(T_ee: np.ndarray):
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
    ee_transformed[0, 0] = r_21
    ee_transformed[0, 1] = r_22
    ee_transformed[0, 2] = r_23
    ee_transformed[0, 3] = Py + post_transform_s4*r_23
    ee_transformed[1, 0] = pre_transform_s0*r_11 + pre_transform_s1*r_31
    ee_transformed[1, 1] = pre_transform_s0*r_12 + pre_transform_s1*r_32
    ee_transformed[1, 2] = pre_transform_s0*r_13 + pre_transform_s1*r_33
    ee_transformed[1, 3] = pre_transform_s0*(Px + post_transform_s4*r_13) + pre_transform_s1*(Pz + post_transform_s4*r_33) + pre_transform_s2
    ee_transformed[2, 0] = -pre_transform_s0*r_31 + pre_transform_s1*r_11
    ee_transformed[2, 1] = -pre_transform_s0*r_32 + pre_transform_s1*r_12
    ee_transformed[2, 2] = -pre_transform_s0*r_33 + pre_transform_s1*r_13
    ee_transformed[2, 3] = -pre_transform_s0*(Pz + post_transform_s4*r_33) + pre_transform_s1*(Px + post_transform_s4*r_13) + pre_transform_s3
    return ee_transformed


def rainbow_y1_r_arm_fk(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.sin(th_6)
    x1 = math.cos(th_4)
    x2 = math.cos(th_0)
    x3 = math.cos(th_2)
    x4 = x2*x3
    x5 = math.cos(th_1)
    x6 = math.sin(th_0)
    x7 = math.sin(th_2)
    x8 = x6*x7
    x9 = -x4 - x5*x8
    x10 = math.sin(th_4)
    x11 = math.sin(th_3)
    x12 = math.sin(th_1)
    x13 = x12*x6
    x14 = math.cos(th_3)
    x15 = x2*x7
    x16 = x3*x6
    x17 = -x15 + x16*x5
    x18 = x11*x13 + x14*x17
    x19 = -x1*x9 - x10*x18
    x20 = math.cos(th_6)
    x21 = math.sin(th_5)
    x22 = -x11*x17 + x13*x14
    x23 = math.cos(th_5)
    x24 = x1*x18 - x10*x9
    x25 = -x21*x22 + x23*x24
    x26 = -x21*x24 - x22*x23
    x27 = x12*x7
    x28 = x12*x3
    x29 = x11*x5 - x14*x28
    x30 = -x1*x27 - x10*x29
    x31 = x11*x28 + x14*x5
    x32 = x1*x29 - x10*x27
    x33 = -x21*x31 + x23*x32
    x34 = -x0*x30 + x20*x33
    x35 = -x15*x5 + x16
    x36 = x12*x2
    x37 = x4*x5 + x8
    x38 = x11*x36 + x14*x37
    x39 = -x1*x35 - x10*x38
    x40 = -x11*x37 + x14*x36
    x41 = x1*x38 - x10*x35
    x42 = -x21*x40 + x23*x41
    x43 = -x0*x39 + x20*x42
    x44 = -x0*x33 - x20*x30
    x45 = -x0*x42 - x20*x39
    x46 = -x21*x32 - x23*x31
    x47 = -x21*x41 - x23*x40
    x48 = pre_transform_s0*x47 + pre_transform_s1*x46
    x49 = -a_3*x28 + a_3*x29 + d_2*x31 - d_2*x5
    x50 = a_3*x37 + a_3*x38 - d_2*x36 + d_2*x40
    x51 = -pre_transform_s0*x46 + pre_transform_s1*x47
    # End of temp variables
    ee_pose = np.eye(4)
    ee_pose[0, 0] = -x0*x19 + x20*x25
    ee_pose[0, 1] = -x0*x25 - x19*x20
    ee_pose[0, 2] = x26
    ee_pose[0, 3] = a_3*x17 + a_3*x18 - d_2*x13 + d_2*x22 + post_transform_s4*x26
    ee_pose[1, 0] = pre_transform_s0*x43 + pre_transform_s1*x34
    ee_pose[1, 1] = pre_transform_s0*x45 + pre_transform_s1*x44
    ee_pose[1, 2] = x48
    ee_pose[1, 3] = post_transform_s4*x48 + pre_transform_s0*x50 + pre_transform_s1*x49 + pre_transform_s2
    ee_pose[2, 0] = -pre_transform_s0*x34 + pre_transform_s1*x43
    ee_pose[2, 1] = -pre_transform_s0*x44 + pre_transform_s1*x45
    ee_pose[2, 2] = x51
    ee_pose[2, 3] = post_transform_s4*x51 - pre_transform_s0*x49 + pre_transform_s1*x50 + pre_transform_s3
    return ee_pose


def rainbow_y1_r_arm_twist_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.cos(th_0)
    x1 = math.sin(th_0)
    x2 = math.sin(th_1)
    x3 = x1*x2
    x4 = math.cos(th_2)
    x5 = math.sin(th_2)
    x6 = math.cos(th_1)
    x7 = x1*x6
    x8 = -x0*x4 - x5*x7
    x9 = math.cos(th_3)
    x10 = math.sin(th_3)
    x11 = -x0*x5 + x4*x7
    x12 = -x10*x11 + x3*x9
    x13 = math.cos(th_4)
    x14 = math.sin(th_4)
    x15 = x10*x3 + x11*x9
    x16 = -x13*x8 - x14*x15
    x17 = math.cos(th_5)
    x18 = math.sin(th_5)
    x19 = -x12*x17 - x18*(x13*x15 - x14*x8)
    x20 = pre_transform_s0*x1
    x21 = pre_transform_s1*x6
    x22 = pre_transform_s0*x2
    x23 = -x0*x22 - x21
    x24 = pre_transform_s1*x2
    x25 = pre_transform_s0*x6
    x26 = x0*x25 - x24
    x27 = x20*x4 - x26*x5
    x28 = x20*x5 + x26*x4
    x29 = -x10*x28 - x23*x9
    x30 = -x10*x23 + x28*x9
    x31 = -x13*x27 - x14*x30
    x32 = -x17*x29 - x18*(x13*x30 - x14*x27)
    x33 = pre_transform_s1*x1
    x34 = -x0*x24 + x25
    x35 = x0*x21 + x22
    x36 = x33*x4 - x35*x5
    x37 = x33*x5 + x35*x4
    x38 = -x10*x37 - x34*x9
    x39 = -x10*x34 + x37*x9
    x40 = -x13*x36 - x14*x39
    x41 = -x17*x38 - x18*(x13*x39 - x14*x36)
    x42 = d_2*x23
    x43 = pre_transform_s2 + x42
    x44 = d_2*x34
    x45 = pre_transform_s3 + x44
    x46 = a_3*x37 + x45
    x47 = a_3*x28 + x43
    x48 = a_3*x39 + d_2*x38 + x46
    x49 = a_3*x30 + d_2*x29 + x47
    x50 = a_3*x11 - d_2*x3
    x51 = a_3*x15 + d_2*x12 + x50
    # End of temp variables
    jacobian_output = np.zeros(shape=(6, 7))
    jacobian_output[0, 1] = x0
    jacobian_output[0, 2] = -x3
    jacobian_output[0, 3] = x8
    jacobian_output[0, 4] = x12
    jacobian_output[0, 5] = x16
    jacobian_output[0, 6] = x19
    jacobian_output[1, 0] = pre_transform_s1
    jacobian_output[1, 1] = -x20
    jacobian_output[1, 2] = x23
    jacobian_output[1, 3] = x27
    jacobian_output[1, 4] = x29
    jacobian_output[1, 5] = x31
    jacobian_output[1, 6] = x32
    jacobian_output[2, 0] = -pre_transform_s0
    jacobian_output[2, 1] = -x33
    jacobian_output[2, 2] = x34
    jacobian_output[2, 3] = x36
    jacobian_output[2, 4] = x38
    jacobian_output[2, 5] = x40
    jacobian_output[2, 6] = x41
    jacobian_output[3, 0] = -pre_transform_s0*pre_transform_s2 - pre_transform_s1*pre_transform_s3
    jacobian_output[3, 1] = -pre_transform_s2*x33 + pre_transform_s3*x20
    jacobian_output[3, 2] = -x23*x45 + x34*x43
    jacobian_output[3, 3] = -x27*x46 + x36*x47
    jacobian_output[3, 4] = -x29*x48 + x38*x49
    jacobian_output[3, 5] = -x31*x48 + x40*x49
    jacobian_output[3, 6] = -x32*x48 + x41*x49
    jacobian_output[4, 1] = pre_transform_s3*x0
    jacobian_output[4, 2] = x3*x44 - x3*x45
    jacobian_output[4, 3] = -x36*x50 + x46*x8
    jacobian_output[4, 4] = x12*x48 - x38*x51
    jacobian_output[4, 5] = x16*x48 - x40*x51
    jacobian_output[4, 6] = x19*x48 - x41*x51
    jacobian_output[5, 1] = -pre_transform_s2*x0
    jacobian_output[5, 2] = -x3*x42 + x3*x43
    jacobian_output[5, 3] = x27*x50 - x47*x8
    jacobian_output[5, 4] = -x12*x49 + x29*x51
    jacobian_output[5, 5] = -x16*x49 + x31*x51
    jacobian_output[5, 6] = -x19*x49 + x32*x51
    return jacobian_output


def rainbow_y1_r_arm_angular_velocity_jacobian(theta_input: np.ndarray):
    th_0 = theta_input[0] + th_0_offset_original2raw
    th_1 = theta_input[1] + th_1_offset_original2raw
    th_2 = theta_input[2] + th_2_offset_original2raw
    th_3 = theta_input[3] + th_3_offset_original2raw
    th_4 = theta_input[4] + th_4_offset_original2raw
    th_5 = theta_input[5] + th_5_offset_original2raw
    th_6 = theta_input[6] + th_6_offset_original2raw

    # Temp variable for efficiency
    x0 = math.cos(th_0)
    x1 = math.sin(th_0)
    x2 = math.sin(th_1)
    x3 = x1*x2
    x4 = math.cos(th_2)
    x5 = math.sin(th_2)
    x6 = math.cos(th_1)
    x7 = x1*x6
    x8 = -x0*x4 - x5*x7
    x9 = math.cos(th_3)
    x10 = math.sin(th_3)
    x11 = -x0*x5 + x4*x7
    x12 = -x10*x11 + x3*x9
    x13 = math.cos(th_4)
    x14 = math.sin(th_4)
    x15 = x10*x3 + x11*x9
    x16 = math.cos(th_5)
    x17 = math.sin(th_5)
    x18 = pre_transform_s0*x1
    x19 = pre_transform_s1*x6
    x20 = pre_transform_s0*x2
    x21 = -x0*x20 - x19
    x22 = pre_transform_s1*x2
    x23 = pre_transform_s0*x6
    x24 = x0*x23 - x22
    x25 = x18*x4 - x24*x5
    x26 = x18*x5 + x24*x4
    x27 = -x10*x26 - x21*x9
    x28 = -x10*x21 + x26*x9
    x29 = pre_transform_s1*x1
    x30 = -x0*x22 + x23
    x31 = x0*x19 + x20
    x32 = x29*x4 - x31*x5
    x33 = x29*x5 + x31*x4
    x34 = -x10*x33 - x30*x9
    x35 = -x10*x30 + x33*x9
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 1] = x0
    jacobian_output[0, 2] = -x3
    jacobian_output[0, 3] = x8
    jacobian_output[0, 4] = x12
    jacobian_output[0, 5] = -x13*x8 - x14*x15
    jacobian_output[0, 6] = -x12*x16 - x17*(x13*x15 - x14*x8)
    jacobian_output[1, 0] = pre_transform_s1
    jacobian_output[1, 1] = -x18
    jacobian_output[1, 2] = x21
    jacobian_output[1, 3] = x25
    jacobian_output[1, 4] = x27
    jacobian_output[1, 5] = -x13*x25 - x14*x28
    jacobian_output[1, 6] = -x16*x27 - x17*(x13*x28 - x14*x25)
    jacobian_output[2, 0] = -pre_transform_s0
    jacobian_output[2, 1] = -x29
    jacobian_output[2, 2] = x30
    jacobian_output[2, 3] = x32
    jacobian_output[2, 4] = x34
    jacobian_output[2, 5] = -x13*x32 - x14*x35
    jacobian_output[2, 6] = -x16*x34 - x17*(x13*x35 - x14*x32)
    return jacobian_output


def rainbow_y1_r_arm_transform_point_jacobian(theta_input: np.ndarray, point_on_ee: np.ndarray):
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
    x0 = math.sin(th_0)
    x1 = pre_transform_s1*x0
    x2 = pre_transform_s0*x0
    x3 = math.cos(th_1)
    x4 = pre_transform_s0*x3
    x5 = math.cos(th_0)
    x6 = math.sin(th_1)
    x7 = pre_transform_s1*x6
    x8 = x4 - x5*x7
    x9 = pre_transform_s1*x3
    x10 = pre_transform_s0*x6
    x11 = -x10*x5 - x9
    x12 = d_2*x11
    x13 = pre_transform_s2 + x12
    x14 = d_2*x8
    x15 = pre_transform_s3 + x14
    x16 = math.cos(th_2)
    x17 = math.sin(th_2)
    x18 = x10 + x5*x9
    x19 = x1*x16 - x17*x18
    x20 = x4*x5 - x7
    x21 = x16*x2 - x17*x20
    x22 = x1*x17 + x16*x18
    x23 = a_3*x22 + x15
    x24 = x16*x20 + x17*x2
    x25 = a_3*x24 + x13
    x26 = math.cos(th_3)
    x27 = math.sin(th_3)
    x28 = -x22*x27 - x26*x8
    x29 = -x11*x26 - x24*x27
    x30 = x22*x26 - x27*x8
    x31 = a_3*x30 + d_2*x28 + x23
    x32 = -x11*x27 + x24*x26
    x33 = a_3*x32 + d_2*x29 + x25
    x34 = math.cos(th_4)
    x35 = math.sin(th_4)
    x36 = -x19*x34 - x30*x35
    x37 = -x21*x34 - x32*x35
    x38 = math.cos(th_5)
    x39 = math.sin(th_5)
    x40 = -x28*x38 - x39*(-x19*x35 + x30*x34)
    x41 = -x29*x38 - x39*(-x21*x35 + x32*x34)
    x42 = p_on_ee_x*pre_transform_s0
    x43 = p_on_ee_x*pre_transform_s1
    x44 = x0*x6
    x45 = x0*x3
    x46 = -x16*x5 - x17*x45
    x47 = x16*x45 - x17*x5
    x48 = a_3*x47 - d_2*x44
    x49 = x26*x44 - x27*x47
    x50 = x26*x47 + x27*x44
    x51 = a_3*x50 + d_2*x49 + x48
    x52 = -x34*x46 - x35*x50
    x53 = -x38*x49 - x39*(x34*x50 - x35*x46)
    # End of temp variables
    jacobian_output = np.zeros(shape=(3, 7))
    jacobian_output[0, 0] = p_on_ee_y*pre_transform_s0 + p_on_ee_z*pre_transform_s1 - pre_transform_s0*pre_transform_s2 - pre_transform_s1*pre_transform_s3
    jacobian_output[0, 1] = p_on_ee_y*x1 - p_on_ee_z*x2 - pre_transform_s2*x1 + pre_transform_s3*x2
    jacobian_output[0, 2] = -p_on_ee_y*x8 + p_on_ee_z*x11 - x11*x15 + x13*x8
    jacobian_output[0, 3] = -p_on_ee_y*x19 + p_on_ee_z*x21 + x19*x25 - x21*x23
    jacobian_output[0, 4] = -p_on_ee_y*x28 + p_on_ee_z*x29 + x28*x33 - x29*x31
    jacobian_output[0, 5] = -p_on_ee_y*x36 + p_on_ee_z*x37 - x31*x37 + x33*x36
    jacobian_output[0, 6] = -p_on_ee_y*x40 + p_on_ee_z*x41 - x31*x41 + x33*x40
    jacobian_output[1, 0] = -x42
    jacobian_output[1, 1] = -p_on_ee_z*x5 + pre_transform_s3*x5 - x0*x43
    jacobian_output[1, 2] = p_on_ee_x*x8 + p_on_ee_z*x44 + x14*x44 - x15*x44
    jacobian_output[1, 3] = p_on_ee_x*x19 - p_on_ee_z*x46 - x19*x48 + x23*x46
    jacobian_output[1, 4] = p_on_ee_x*x28 - p_on_ee_z*x49 - x28*x51 + x31*x49
    jacobian_output[1, 5] = p_on_ee_x*x36 - p_on_ee_z*x52 + x31*x52 - x36*x51
    jacobian_output[1, 6] = p_on_ee_x*x40 - p_on_ee_z*x53 + x31*x53 - x40*x51
    jacobian_output[2, 0] = -x43
    jacobian_output[2, 1] = p_on_ee_y*x5 - pre_transform_s2*x5 + x0*x42
    jacobian_output[2, 2] = -p_on_ee_x*x11 - p_on_ee_y*x44 - x12*x44 + x13*x44
    jacobian_output[2, 3] = -p_on_ee_x*x21 + p_on_ee_y*x46 + x21*x48 - x25*x46
    jacobian_output[2, 4] = -p_on_ee_x*x29 + p_on_ee_y*x49 + x29*x51 - x33*x49
    jacobian_output[2, 5] = -p_on_ee_x*x37 + p_on_ee_y*x52 - x33*x52 + x37*x51
    jacobian_output[2, 6] = -p_on_ee_x*x41 + p_on_ee_y*x53 - x33*x53 + x41*x51
    return jacobian_output


def rainbow_y1_r_arm_ik_solve_raw(T_ee: np.ndarray, th_4):
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
    
    # Code for non-branch dispatcher node 0
    # Actually, there is no code
    
    # Code for explicit solution node 1, solved variable is th_3
    def ExplicitSolutionNode_node_1_solve_th_3_processor():
        this_node_input_index: List[int] = node_input_index[0]
        this_input_valid: bool = node_input_validity[0]
        if not this_input_valid:
            return
        
        # The explicit solution of root node
        condition_0: bool = (4*abs(a_3*d_2) >= zero_tolerance) or (abs(2*a_3**2 - 2*d_2**2) >= zero_tolerance) or (abs(-2*a_3**2 - 2*d_2**2 + inv_Px**2 + inv_Py**2 + inv_Pz**2) >= zero_tolerance)
        if condition_0:
            # Temp variable for efficiency
            x0 = a_3**2
            x1 = 2*x0
            x2 = d_2**2
            x3 = -2*x2
            x4 = x1 + x3
            x5 = math.atan2(-4*a_3*d_2, x4)
            x6 = inv_Px**2 + inv_Py**2 + inv_Pz**2 - x1 + x3
            x7 = safe_sqrt(16*x0*x2 + x4**2 - x6**2)
            # End of temp variables
            solution_0: IkSolution = make_ik_solution()
            solution_0[3] = x5 + math.atan2(x7, x6)
            appended_idx = append_solution_to_queue(solution_0)
            add_input_index_to(2, appended_idx)
            
        condition_1: bool = (4*abs(a_3*d_2) >= zero_tolerance) or (abs(2*a_3**2 - 2*d_2**2) >= zero_tolerance) or (abs(-2*a_3**2 - 2*d_2**2 + inv_Px**2 + inv_Py**2 + inv_Pz**2) >= zero_tolerance)
        if condition_1:
            # Temp variable for efficiency
            x0 = a_3**2
            x1 = 2*x0
            x2 = d_2**2
            x3 = -2*x2
            x4 = x1 + x3
            x5 = math.atan2(-4*a_3*d_2, x4)
            x6 = inv_Px**2 + inv_Py**2 + inv_Pz**2 - x1 + x3
            x7 = safe_sqrt(16*x0*x2 + x4**2 - x6**2)
            # End of temp variables
            solution_1: IkSolution = make_ik_solution()
            solution_1[3] = x5 + math.atan2(-x7, x6)
            appended_idx = append_solution_to_queue(solution_1)
            add_input_index_to(2, appended_idx)
            
    # Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_3_processor()
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
            th_3 = this_solution[3]
            checked_result: bool = (abs(inv_Pz) <= 1.0e-6) and (abs(a_3*math.sin(th_3) + d_2*math.cos(th_3) - d_2) <= 1.0e-6) and (abs(a_3*math.cos(th_3)*math.cos(th_4) + a_3*math.cos(th_4) - d_2*math.sin(th_3)*math.cos(th_4)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor()
    # Finish code for equation all-zero dispatcher node 2
    
    # Code for explicit solution node 3, solved variable is th_5
    def ExplicitSolutionNode_node_3_solve_th_5_processor():
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
            th_3 = this_solution[3]
            condition_0: bool = (abs(inv_Pz) >= zero_tolerance) or (abs(a_3*math.sin(th_3) + d_2*math.cos(th_3) - d_2) >= zero_tolerance) or (abs(a_3*math.cos(th_3)*math.cos(th_4) + a_3*math.cos(th_4) - d_2*math.sin(th_3)*math.cos(th_4)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_4)
                x1 = a_3*x0
                x2 = math.cos(th_3)
                x3 = math.sin(th_3)
                x4 = -d_2*x0*x3 + x1*x2 + x1
                x5 = -a_3*x3 - d_2*x2 + d_2
                x6 = math.atan2(x4, x5)
                x7 = safe_sqrt(-inv_Pz**2 + x4**2 + x5**2)
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[4] = x6 + math.atan2(x7, inv_Pz)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(4, appended_idx)
                
            condition_1: bool = (abs(inv_Pz) >= zero_tolerance) or (abs(a_3*math.sin(th_3) + d_2*math.cos(th_3) - d_2) >= zero_tolerance) or (abs(a_3*math.cos(th_3)*math.cos(th_4) + a_3*math.cos(th_4) - d_2*math.sin(th_3)*math.cos(th_4)) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.cos(th_4)
                x1 = a_3*x0
                x2 = math.cos(th_3)
                x3 = math.sin(th_3)
                x4 = -d_2*x0*x3 + x1*x2 + x1
                x5 = -a_3*x3 - d_2*x2 + d_2
                x6 = math.atan2(x4, x5)
                x7 = safe_sqrt(-inv_Pz**2 + x4**2 + x5**2)
                # End of temp variables
                this_solution[4] = x6 + math.atan2(-x7, inv_Pz)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(4, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_5_processor()
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
            checked_result: bool = (abs(inv_Px) <= 1.0e-6) and (abs(inv_Py) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor()
    # Finish code for equation all-zero dispatcher node 4
    
    # Code for explicit solution node 5, solved variable is th_6
    def ExplicitSolutionNode_node_5_solve_th_6_processor():
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
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            condition_0: bool = (abs(inv_Px) >= 1.0e-6) or (abs(inv_Py) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_3)
                x1 = d_2*x0
                x2 = math.cos(th_3)
                x3 = a_3*x2
                x4 = (-a_3 + x1 - x3)*math.sin(th_4)
                x5 = math.sin(th_5)
                x6 = d_2*x5
                x7 = math.cos(th_4)*math.cos(th_5)
                x8 = a_3*x0*x5 + a_3*x7 - x1*x7 + x2*x6 + x3*x7 - x6
                # End of temp variables
                this_solution[5] = math.atan2(inv_Px*x4 + inv_Py*x8, -inv_Px*x8 + inv_Py*x4)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(6, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_6_processor()
    # Finish code for explicit solution node 5
    
    # Code for non-branch dispatcher node 6
    # Actually, there is no code
    
    # Code for linear solution type2 node 7, solved variable is th_2
    def LinearSinCosType_2_SolverNode_node_7_solve_th_2_processor():
        this_node_input_index: List[int] = node_input_index[6]
        this_input_valid: bool = node_input_validity[6]
        if not this_input_valid:
            return
        
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            # Temp variable for efficiency
            x0 = math.cos(th_6)
            x1 = math.sin(th_3)
            x2 = math.sin(th_5)
            x3 = x1*x2
            x4 = x0*x3
            x5 = math.sin(th_6)
            x6 = math.sin(th_4)
            x7 = math.cos(th_3)
            x8 = x6*x7
            x9 = x5*x8
            x10 = math.cos(th_4)
            x11 = x0*x10
            x12 = math.cos(th_5)
            x13 = x12*x7
            x14 = x11*x13
            x15 = x10*x5
            x16 = x12*x6
            x17 = x0*x16
            x18 = x3*x5
            x19 = x0*x8
            x20 = x13*x15
            x21 = x16*x5
            x22 = x1*x12
            x23 = x10*x2*x7
            x24 = x2*x6
            x25 = d_2*x1
            x26 = a_3*x7
            # End of temp variables
            A_matrix = np.zeros(shape=(8, 4))
            A_matrix[0, 0] = r_11
            A_matrix[0, 1] = -r_21
            A_matrix[0, 2] = -x14 - x4 - x9
            A_matrix[0, 3] = -x15 + x17
            A_matrix[1, 0] = r_12
            A_matrix[1, 1] = -r_22
            A_matrix[1, 2] = x18 - x19 + x20
            A_matrix[1, 3] = -x11 - x21
            A_matrix[2, 0] = r_13
            A_matrix[2, 1] = -r_23
            A_matrix[2, 2] = -x22 + x23
            A_matrix[2, 3] = -x24
            A_matrix[3, 0] = Px
            A_matrix[3, 1] = -Py
            A_matrix[3, 2] = -a_3 + x25 - x26
            A_matrix[4, 0] = -r_11
            A_matrix[4, 1] = r_21
            A_matrix[4, 2] = x14 + x4 + x9
            A_matrix[4, 3] = x15 - x17
            A_matrix[5, 0] = -r_12
            A_matrix[5, 1] = r_22
            A_matrix[5, 2] = -x18 + x19 - x20
            A_matrix[5, 3] = x11 + x21
            A_matrix[6, 0] = -r_13
            A_matrix[6, 1] = r_23
            A_matrix[6, 2] = x22 - x23
            A_matrix[6, 3] = x24
            A_matrix[7, 0] = -Px
            A_matrix[7, 1] = Py
            A_matrix[7, 2] = a_3 - x25 + x26
            solution_tuple_0 = try_solve_linear_type2_specific_rows(A_matrix, 0, 1, 2)
            if solution_tuple_0 is not None:
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = solution_tuple_0[0]
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
                this_solution[2] = solution_tuple_0[1]
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
                continue
                
            solution_tuple_1 = try_solve_linear_type2_specific_rows(A_matrix, 0, 1, 3)
            if solution_tuple_1 is not None:
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = solution_tuple_1[0]
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
                this_solution[2] = solution_tuple_1[1]
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
                continue
                
            solution_tuple_2 = try_solve_linear_type2_specific_rows(A_matrix, 0, 1, 6)
            if solution_tuple_2 is not None:
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = solution_tuple_2[0]
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
                this_solution[2] = solution_tuple_2[1]
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
                continue
                
            solution_tuple_3 = try_solve_linear_type2_specific_rows(A_matrix, 0, 1, 7)
            if solution_tuple_3 is not None:
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = solution_tuple_3[0]
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
                this_solution[2] = solution_tuple_3[1]
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
                continue
                
            solution_tuple_4 = try_solve_linear_type2_specific_rows(A_matrix, 0, 2, 3)
            if solution_tuple_4 is not None:
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[2] = solution_tuple_4[0]
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(8, appended_idx)
                
                this_solution[2] = solution_tuple_4[1]
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(8, node_input_i_idx_in_queue)
                continue
                
    # Invoke the processor
    LinearSinCosType_2_SolverNode_node_7_solve_th_2_processor()
    # Finish code for explicit solution node 6
    
    # Code for equation all-zero dispatcher node 8
    def EquationAllZeroDispatcherNode_node_8_processor():
        this_node_input_index: List[int] = node_input_index[8]
        this_input_valid: bool = node_input_validity[8]
        if not this_input_valid:
            return
        for i in range(len(this_node_input_index)):
            node_input_i_idx_in_queue = this_node_input_index[i]
            if not queue_element_validity[node_input_i_idx_in_queue]:
                continue
            this_solution = solution_queue[node_input_i_idx_in_queue]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            checked_result: bool = (abs(r_33) <= 1.0e-6) and (abs(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) <= 1.0e-6) and (abs(math.sin(th_2)*math.sin(th_4)*math.sin(th_5) - math.sin(th_3)*math.cos(th_2)*math.cos(th_5) + math.sin(th_5)*math.cos(th_2)*math.cos(th_3)*math.cos(th_4)) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor()
    # Finish code for equation all-zero dispatcher node 8
    
    # Code for explicit solution node 9, solved variable is th_1
    def ExplicitSolutionNode_node_9_solve_th_1_processor():
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
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            condition_0: bool = (abs(r_33) >= zero_tolerance) or (abs(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= zero_tolerance) or (abs(math.sin(th_2)*math.sin(th_4)*math.sin(th_5) - math.sin(th_3)*math.cos(th_2)*math.cos(th_5) + math.sin(th_5)*math.cos(th_2)*math.cos(th_3)*math.cos(th_4)) >= zero_tolerance)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.sin(th_3)
                x1 = math.cos(th_2)
                x2 = math.cos(th_5)
                x3 = math.sin(th_5)
                x4 = math.cos(th_3)
                x5 = x3*math.cos(th_4)
                x6 = x0*x1*x2 - x1*x4*x5 - x3*math.sin(th_2)*math.sin(th_4)
                x7 = x0*x5 + x2*x4
                x8 = math.atan2(x6, x7)
                x9 = safe_sqrt(-r_33**2 + x6**2 + x7**2)
                x10 = -r_33
                # End of temp variables
                solution_0: IkSolution = copy.copy(this_solution)
                solution_0[1] = x8 + math.atan2(x9, x10)
                appended_idx = append_solution_to_queue(solution_0)
                add_input_index_to(10, appended_idx)
                
            condition_1: bool = (abs(r_33) >= zero_tolerance) or (abs(math.sin(th_3)*math.sin(th_5)*math.cos(th_4) + math.cos(th_3)*math.cos(th_5)) >= zero_tolerance) or (abs(math.sin(th_2)*math.sin(th_4)*math.sin(th_5) - math.sin(th_3)*math.cos(th_2)*math.cos(th_5) + math.sin(th_5)*math.cos(th_2)*math.cos(th_3)*math.cos(th_4)) >= zero_tolerance)
            if condition_1:
                # Temp variable for efficiency
                x0 = math.sin(th_3)
                x1 = math.cos(th_2)
                x2 = math.cos(th_5)
                x3 = math.sin(th_5)
                x4 = math.cos(th_3)
                x5 = x3*math.cos(th_4)
                x6 = x0*x1*x2 - x1*x4*x5 - x3*math.sin(th_2)*math.sin(th_4)
                x7 = x0*x5 + x2*x4
                x8 = math.atan2(x6, x7)
                x9 = safe_sqrt(-r_33**2 + x6**2 + x7**2)
                x10 = -r_33
                # End of temp variables
                this_solution[1] = x8 + math.atan2(-x9, x10)
                solution_queue[node_input_i_idx_in_queue] = this_solution
                add_input_index_to(10, node_input_i_idx_in_queue)
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_1_processor()
    # Finish code for explicit solution node 9
    
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
            checked_result: bool = (abs(r_11) <= 1.0e-6) and (abs(r_21) <= 1.0e-6)
            if not checked_result:  # To non-degenerate node
                add_input_index_to(11, node_input_i_idx_in_queue)
    
    # Invoke the processor
    EquationAllZeroDispatcherNode_node_10_processor()
    # Finish code for equation all-zero dispatcher node 10
    
    # Code for explicit solution node 11, solved variable is th_0
    def ExplicitSolutionNode_node_11_solve_th_0_processor():
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
            th_1 = this_solution[1]
            th_2 = this_solution[2]
            th_3 = this_solution[3]
            th_5 = this_solution[4]
            th_6 = this_solution[5]
            condition_0: bool = (abs(r_11) >= 1.0e-6) or (abs(r_21) >= 1.0e-6)
            if condition_0:
                # Temp variable for efficiency
                x0 = math.cos(th_2)
                x1 = math.cos(th_4)
                x2 = math.sin(th_6)
                x3 = x1*x2
                x4 = math.sin(th_2)
                x5 = math.sin(th_3)
                x6 = math.cos(th_6)
                x7 = x6*math.sin(th_5)
                x8 = x5*x7
                x9 = math.sin(th_4)
                x10 = x2*x9
                x11 = math.cos(th_3)
                x12 = x11*x4
                x13 = x6*math.cos(th_5)
                x14 = x13*x9
                x15 = x1*x13
                x16 = -x0*x14 + x0*x3 + x10*x12 + x12*x15 + x4*x8
                x17 = math.sin(th_1)
                x18 = math.cos(th_1)
                x19 = x18*x4
                x20 = x17*x5
                x21 = x0*x18
                x22 = x11*x21
                x23 = -x10*x20 - x10*x22 + x11*x17*x7 - x14*x19 - x15*x20 - x15*x22 + x19*x3 - x21*x8
                # End of temp variables
                this_solution[0] = math.atan2(r_11*x16 - r_21*x23, -r_11*x23 - r_21*x16)
                solution_queue[node_input_i_idx_in_queue] = this_solution
            else:
                queue_element_validity[node_input_i_idx_in_queue] = False
            
    # Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_0_processor()
    # Finish code for explicit solution node 11
    
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
        value_at_2 = ik_out_i[2]  # th_2
        new_ik_i[2] = value_at_2
        value_at_3 = ik_out_i[3]  # th_3
        new_ik_i[3] = value_at_3
        value_at_4 = th_4  # th_4
        new_ik_i[4] = value_at_4
        value_at_5 = ik_out_i[4]  # th_5
        new_ik_i[5] = value_at_5
        value_at_6 = ik_out_i[5]  # th_6
        new_ik_i[6] = value_at_6
        ik_out.append(new_ik_i)
    return ik_out


def rainbow_y1_r_arm_ik_solve(T_ee: np.ndarray, th_4):
    T_ee_raw_in = rainbow_y1_r_arm_ik_target_original_to_raw(T_ee)
    ik_output_raw = rainbow_y1_r_arm_ik_solve_raw(T_ee_raw_in, th_4 + th_4_offset_original2raw)
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
        ee_pose_i = rainbow_y1_r_arm_fk(ik_out_i)
        ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))
        if ee_pose_diff < pose_tolerance:
            ik_output.append(ik_out_i)
    return ik_output


def test_ik_solve_rainbow_y1_r_arm():
    theta_in = np.random.random(size=(7, ))
    ee_pose = rainbow_y1_r_arm_fk(theta_in)
    ik_output = rainbow_y1_r_arm_ik_solve(ee_pose, th_4=theta_in[4])
    for i in range(len(ik_output)):
        ee_pose_i = rainbow_y1_r_arm_fk(ik_output[i])
        ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))
        print('The pose difference is ', ee_pose_diff)


if __name__ == '__main__':
    test_ik_solve_rainbow_y1_r_arm()
