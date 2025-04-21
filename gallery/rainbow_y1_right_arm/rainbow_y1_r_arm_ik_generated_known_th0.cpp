#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct rainbow_y1_r_arm_ik {

// Constants for solver
static constexpr int robot_nq = 7;
static constexpr int max_n_solutions = 16;
static constexpr int n_tree_nodes = 60;
static constexpr int intermediate_solution_size = 13;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_3 = 0.031;
static constexpr double d_2 = -0.276;
static constexpr double post_transform_s4 = -0.1548;
static constexpr double pre_transform_s0 = 0.34202;
static constexpr double pre_transform_s1 = 0.939693;
static constexpr double pre_transform_s2 = -0.22;
static constexpr double pre_transform_s3 = 0.0800735;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -1.2217300208404673;
static constexpr double th_2_offset_original2raw = -1.5707963267948966;
static constexpr double th_3_offset_original2raw = 3.141592653589793;
static constexpr double th_4_offset_original2raw = 3.141592653589793;
static constexpr double th_5_offset_original2raw = 3.141592653589793;
static constexpr double th_6_offset_original2raw = 3.141592653589793;

// The transformation between raw and original ee target
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// ee_original = pre_transform * ee_raw * post_transform
// ee_raw = dh_forward_transform(theta_raw)
static Eigen::Matrix4d endEffectorTargetOriginalToRaw(const Eigen::Matrix4d& T_ee)
{
    const double r_11 = T_ee(0, 0);
    const double r_12 = T_ee(0, 1);
    const double r_13 = T_ee(0, 2);
    const double Px = T_ee(0, 3);
    const double r_21 = T_ee(1, 0);
    const double r_22 = T_ee(1, 1);
    const double r_23 = T_ee(1, 2);
    const double Py = T_ee(1, 3);
    const double r_31 = T_ee(2, 0);
    const double r_32 = T_ee(2, 1);
    const double r_33 = T_ee(2, 2);
    const double Pz = T_ee(2, 3);
    Eigen::Matrix4d ee_transformed;
    ee_transformed.setIdentity();
    ee_transformed(0, 0) = pre_transform_s0*r_21 + pre_transform_s1*r_31;
    ee_transformed(0, 1) = pre_transform_s0*r_22 + pre_transform_s1*r_32;
    ee_transformed(0, 2) = pre_transform_s0*r_23 + pre_transform_s1*r_33;
    ee_transformed(0, 3) = -pre_transform_s0*pre_transform_s2 + pre_transform_s0*(Py - post_transform_s4*r_23) - pre_transform_s1*pre_transform_s3 + pre_transform_s1*(Pz - post_transform_s4*r_33);
    ee_transformed(1, 0) = r_11;
    ee_transformed(1, 1) = r_12;
    ee_transformed(1, 2) = r_13;
    ee_transformed(1, 3) = Px - post_transform_s4*r_13;
    ee_transformed(2, 0) = -pre_transform_s0*r_31 + pre_transform_s1*r_21;
    ee_transformed(2, 1) = -pre_transform_s0*r_32 + pre_transform_s1*r_22;
    ee_transformed(2, 2) = -pre_transform_s0*r_33 + pre_transform_s1*r_23;
    ee_transformed(2, 3) = pre_transform_s0*pre_transform_s3 - pre_transform_s0*(Pz - post_transform_s4*r_33) - pre_transform_s1*pre_transform_s2 + pre_transform_s1*(Py - post_transform_s4*r_23);
    return ee_transformed;
}

static Eigen::Matrix4d endEffectorTargetRawToOriginal(const Eigen::Matrix4d& T_ee)
{
    const double r_11 = T_ee(0, 0);
    const double r_12 = T_ee(0, 1);
    const double r_13 = T_ee(0, 2);
    const double Px = T_ee(0, 3);
    const double r_21 = T_ee(1, 0);
    const double r_22 = T_ee(1, 1);
    const double r_23 = T_ee(1, 2);
    const double Py = T_ee(1, 3);
    const double r_31 = T_ee(2, 0);
    const double r_32 = T_ee(2, 1);
    const double r_33 = T_ee(2, 2);
    const double Pz = T_ee(2, 3);
    Eigen::Matrix4d ee_transformed;
    ee_transformed.setIdentity();
    ee_transformed(0, 0) = r_21;
    ee_transformed(0, 1) = r_22;
    ee_transformed(0, 2) = r_23;
    ee_transformed(0, 3) = Py + post_transform_s4*r_23;
    ee_transformed(1, 0) = pre_transform_s0*r_11 + pre_transform_s1*r_31;
    ee_transformed(1, 1) = pre_transform_s0*r_12 + pre_transform_s1*r_32;
    ee_transformed(1, 2) = pre_transform_s0*r_13 + pre_transform_s1*r_33;
    ee_transformed(1, 3) = pre_transform_s0*(Px + post_transform_s4*r_13) + pre_transform_s1*(Pz + post_transform_s4*r_33) + pre_transform_s2;
    ee_transformed(2, 0) = -pre_transform_s0*r_31 + pre_transform_s1*r_11;
    ee_transformed(2, 1) = -pre_transform_s0*r_32 + pre_transform_s1*r_12;
    ee_transformed(2, 2) = -pre_transform_s0*r_33 + pre_transform_s1*r_13;
    ee_transformed(2, 3) = -pre_transform_s0*(Pz + post_transform_s4*r_33) + pre_transform_s1*(Px + post_transform_s4*r_13) + pre_transform_s3;
    return ee_transformed;
}

///************* Below are the actual FK and IK implementations *******************
static Eigen::Matrix4d computeFK(const std::array<double, robot_nq>& theta_input_original)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_6);
    const double x1 = std::cos(th_4);
    const double x2 = std::sin(th_0);
    const double x3 = std::cos(th_2);
    const double x4 = x2*x3;
    const double x5 = std::cos(th_1);
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_0);
    const double x8 = x6*x7;
    const double x9 = x4 - x5*x8;
    const double x10 = std::sin(th_4);
    const double x11 = std::sin(th_3);
    const double x12 = std::sin(th_1);
    const double x13 = x12*x7;
    const double x14 = std::cos(th_3);
    const double x15 = x2*x6;
    const double x16 = x3*x7;
    const double x17 = x15 + x16*x5;
    const double x18 = x11*x13 + x14*x17;
    const double x19 = -x1*x9 - x10*x18;
    const double x20 = std::cos(th_6);
    const double x21 = std::sin(th_5);
    const double x22 = -x11*x17 + x13*x14;
    const double x23 = std::cos(th_5);
    const double x24 = x1*x18 - x10*x9;
    const double x25 = -x21*x22 + x23*x24;
    const double x26 = -x15*x5 - x16;
    const double x27 = x12*x2;
    const double x28 = x4*x5 - x8;
    const double x29 = x11*x27 + x14*x28;
    const double x30 = -x1*x26 - x10*x29;
    const double x31 = -x11*x28 + x14*x27;
    const double x32 = x1*x29 - x10*x26;
    const double x33 = -x21*x31 + x23*x32;
    const double x34 = x12*x6;
    const double x35 = x12*x3;
    const double x36 = x11*x5 - x14*x35;
    const double x37 = -x1*x34 - x10*x36;
    const double x38 = x11*x35 + x14*x5;
    const double x39 = x1*x36 - x10*x34;
    const double x40 = -x21*x38 + x23*x39;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x19 + x20*x25;
    ee_pose_raw(0, 1) = -x0*x25 - x19*x20;
    ee_pose_raw(0, 2) = -x21*x24 - x22*x23;
    ee_pose_raw(0, 3) = a_3*x17 + a_3*x18 - d_2*x13 + d_2*x22;
    ee_pose_raw(1, 0) = -x0*x30 + x20*x33;
    ee_pose_raw(1, 1) = -x0*x33 - x20*x30;
    ee_pose_raw(1, 2) = -x21*x32 - x23*x31;
    ee_pose_raw(1, 3) = a_3*x28 + a_3*x29 - d_2*x27 + d_2*x31;
    ee_pose_raw(2, 0) = -x0*x37 + x20*x40;
    ee_pose_raw(2, 1) = -x0*x40 - x20*x37;
    ee_pose_raw(2, 2) = -x21*x39 - x23*x38;
    ee_pose_raw(2, 3) = -a_3*x35 + a_3*x36 + d_2*x38 - d_2*x5;
    return endEffectorTargetRawToOriginal(ee_pose_raw);
}

static void computeTwistJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;
    
    // Temp variable for efficiency
    const double x0 = std::cos(th_0);
    const double x1 = std::sin(th_0);
    const double x2 = std::sin(th_1);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_2);
    const double x5 = std::sin(th_2);
    const double x6 = std::cos(th_1);
    const double x7 = x1*x6;
    const double x8 = -x0*x4 - x5*x7;
    const double x9 = std::cos(th_3);
    const double x10 = std::sin(th_3);
    const double x11 = -x0*x5 + x4*x7;
    const double x12 = -x10*x11 + x3*x9;
    const double x13 = std::cos(th_4);
    const double x14 = std::sin(th_4);
    const double x15 = x10*x3 + x11*x9;
    const double x16 = -x13*x8 - x14*x15;
    const double x17 = std::cos(th_5);
    const double x18 = std::sin(th_5);
    const double x19 = -x12*x17 - x18*(x13*x15 - x14*x8);
    const double x20 = pre_transform_s0*x1;
    const double x21 = pre_transform_s1*x6;
    const double x22 = pre_transform_s0*x2;
    const double x23 = -x0*x22 - x21;
    const double x24 = pre_transform_s1*x2;
    const double x25 = pre_transform_s0*x6;
    const double x26 = x0*x25 - x24;
    const double x27 = x20*x4 - x26*x5;
    const double x28 = x20*x5 + x26*x4;
    const double x29 = -x10*x28 - x23*x9;
    const double x30 = -x10*x23 + x28*x9;
    const double x31 = -x13*x27 - x14*x30;
    const double x32 = -x17*x29 - x18*(x13*x30 - x14*x27);
    const double x33 = pre_transform_s1*x1;
    const double x34 = -x0*x24 + x25;
    const double x35 = x0*x21 + x22;
    const double x36 = x33*x4 - x35*x5;
    const double x37 = x33*x5 + x35*x4;
    const double x38 = -x10*x37 - x34*x9;
    const double x39 = -x10*x34 + x37*x9;
    const double x40 = -x13*x36 - x14*x39;
    const double x41 = -x17*x38 - x18*(x13*x39 - x14*x36);
    const double x42 = d_2*x23;
    const double x43 = pre_transform_s2 + x42;
    const double x44 = d_2*x34;
    const double x45 = pre_transform_s3 + x44;
    const double x46 = a_3*x37 + x45;
    const double x47 = a_3*x28 + x43;
    const double x48 = a_3*x39 + d_2*x38 + x46;
    const double x49 = a_3*x30 + d_2*x29 + x47;
    const double x50 = a_3*x11 - d_2*x3;
    const double x51 = a_3*x15 + d_2*x12 + x50;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x0;
    jacobian(0, 2) = -x3;
    jacobian(0, 3) = x8;
    jacobian(0, 4) = x12;
    jacobian(0, 5) = x16;
    jacobian(0, 6) = x19;
    jacobian(1, 0) = pre_transform_s1;
    jacobian(1, 1) = -x20;
    jacobian(1, 2) = x23;
    jacobian(1, 3) = x27;
    jacobian(1, 4) = x29;
    jacobian(1, 5) = x31;
    jacobian(1, 6) = x32;
    jacobian(2, 0) = -pre_transform_s0;
    jacobian(2, 1) = -x33;
    jacobian(2, 2) = x34;
    jacobian(2, 3) = x36;
    jacobian(2, 4) = x38;
    jacobian(2, 5) = x40;
    jacobian(2, 6) = x41;
    jacobian(3, 0) = -pre_transform_s0*pre_transform_s2 - pre_transform_s1*pre_transform_s3;
    jacobian(3, 1) = -pre_transform_s2*x33 + pre_transform_s3*x20;
    jacobian(3, 2) = -x23*x45 + x34*x43;
    jacobian(3, 3) = -x27*x46 + x36*x47;
    jacobian(3, 4) = -x29*x48 + x38*x49;
    jacobian(3, 5) = -x31*x48 + x40*x49;
    jacobian(3, 6) = -x32*x48 + x41*x49;
    jacobian(4, 1) = pre_transform_s3*x0;
    jacobian(4, 2) = x3*x44 - x3*x45;
    jacobian(4, 3) = -x36*x50 + x46*x8;
    jacobian(4, 4) = x12*x48 - x38*x51;
    jacobian(4, 5) = x16*x48 - x40*x51;
    jacobian(4, 6) = x19*x48 - x41*x51;
    jacobian(5, 1) = -pre_transform_s2*x0;
    jacobian(5, 2) = -x3*x42 + x3*x43;
    jacobian(5, 3) = x27*x50 - x47*x8;
    jacobian(5, 4) = -x12*x49 + x29*x51;
    jacobian(5, 5) = -x16*x49 + x31*x51;
    jacobian(5, 6) = -x19*x49 + x32*x51;
    return;
}

static void computeAngularVelocityJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;
    
    // Temp variable for efficiency
    const double x0 = std::cos(th_0);
    const double x1 = std::sin(th_0);
    const double x2 = std::sin(th_1);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_2);
    const double x5 = std::sin(th_2);
    const double x6 = std::cos(th_1);
    const double x7 = x1*x6;
    const double x8 = -x0*x4 - x5*x7;
    const double x9 = std::cos(th_3);
    const double x10 = std::sin(th_3);
    const double x11 = -x0*x5 + x4*x7;
    const double x12 = -x10*x11 + x3*x9;
    const double x13 = std::cos(th_4);
    const double x14 = std::sin(th_4);
    const double x15 = x10*x3 + x11*x9;
    const double x16 = std::cos(th_5);
    const double x17 = std::sin(th_5);
    const double x18 = pre_transform_s0*x1;
    const double x19 = pre_transform_s1*x6;
    const double x20 = pre_transform_s0*x2;
    const double x21 = -x0*x20 - x19;
    const double x22 = pre_transform_s1*x2;
    const double x23 = pre_transform_s0*x6;
    const double x24 = x0*x23 - x22;
    const double x25 = x18*x4 - x24*x5;
    const double x26 = x18*x5 + x24*x4;
    const double x27 = -x10*x26 - x21*x9;
    const double x28 = -x10*x21 + x26*x9;
    const double x29 = pre_transform_s1*x1;
    const double x30 = -x0*x22 + x23;
    const double x31 = x0*x19 + x20;
    const double x32 = x29*x4 - x31*x5;
    const double x33 = x29*x5 + x31*x4;
    const double x34 = -x10*x33 - x30*x9;
    const double x35 = -x10*x30 + x33*x9;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x0;
    jacobian(0, 2) = -x3;
    jacobian(0, 3) = x8;
    jacobian(0, 4) = x12;
    jacobian(0, 5) = -x13*x8 - x14*x15;
    jacobian(0, 6) = -x12*x16 - x17*(x13*x15 - x14*x8);
    jacobian(1, 0) = pre_transform_s1;
    jacobian(1, 1) = -x18;
    jacobian(1, 2) = x21;
    jacobian(1, 3) = x25;
    jacobian(1, 4) = x27;
    jacobian(1, 5) = -x13*x25 - x14*x28;
    jacobian(1, 6) = -x16*x27 - x17*(x13*x28 - x14*x25);
    jacobian(2, 0) = -pre_transform_s0;
    jacobian(2, 1) = -x29;
    jacobian(2, 2) = x30;
    jacobian(2, 3) = x32;
    jacobian(2, 4) = x34;
    jacobian(2, 5) = -x13*x32 - x14*x35;
    jacobian(2, 6) = -x16*x34 - x17*(x13*x35 - x14*x32);
    return;
}

static void computeTransformPointJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Vector3d& point_on_ee, Eigen::Matrix<double, 3, robot_nq>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;
    const double p_on_ee_x = point_on_ee[0];
    const double p_on_ee_y = point_on_ee[1];
    const double p_on_ee_z = point_on_ee[2];
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = pre_transform_s1*x0;
    const double x2 = pre_transform_s0*x0;
    const double x3 = std::cos(th_1);
    const double x4 = pre_transform_s0*x3;
    const double x5 = std::cos(th_0);
    const double x6 = std::sin(th_1);
    const double x7 = pre_transform_s1*x6;
    const double x8 = x4 - x5*x7;
    const double x9 = pre_transform_s1*x3;
    const double x10 = pre_transform_s0*x6;
    const double x11 = -x10*x5 - x9;
    const double x12 = d_2*x11;
    const double x13 = pre_transform_s2 + x12;
    const double x14 = d_2*x8;
    const double x15 = pre_transform_s3 + x14;
    const double x16 = std::cos(th_2);
    const double x17 = std::sin(th_2);
    const double x18 = x10 + x5*x9;
    const double x19 = x1*x16 - x17*x18;
    const double x20 = x4*x5 - x7;
    const double x21 = x16*x2 - x17*x20;
    const double x22 = x1*x17 + x16*x18;
    const double x23 = a_3*x22 + x15;
    const double x24 = x16*x20 + x17*x2;
    const double x25 = a_3*x24 + x13;
    const double x26 = std::cos(th_3);
    const double x27 = std::sin(th_3);
    const double x28 = -x22*x27 - x26*x8;
    const double x29 = -x11*x26 - x24*x27;
    const double x30 = x22*x26 - x27*x8;
    const double x31 = a_3*x30 + d_2*x28 + x23;
    const double x32 = -x11*x27 + x24*x26;
    const double x33 = a_3*x32 + d_2*x29 + x25;
    const double x34 = std::cos(th_4);
    const double x35 = std::sin(th_4);
    const double x36 = -x19*x34 - x30*x35;
    const double x37 = -x21*x34 - x32*x35;
    const double x38 = std::cos(th_5);
    const double x39 = std::sin(th_5);
    const double x40 = -x28*x38 - x39*(-x19*x35 + x30*x34);
    const double x41 = -x29*x38 - x39*(-x21*x35 + x32*x34);
    const double x42 = p_on_ee_x*pre_transform_s0;
    const double x43 = p_on_ee_x*pre_transform_s1;
    const double x44 = x0*x6;
    const double x45 = x0*x3;
    const double x46 = -x16*x5 - x17*x45;
    const double x47 = x16*x45 - x17*x5;
    const double x48 = a_3*x47 - d_2*x44;
    const double x49 = x26*x44 - x27*x47;
    const double x50 = x26*x47 + x27*x44;
    const double x51 = a_3*x50 + d_2*x49 + x48;
    const double x52 = -x34*x46 - x35*x50;
    const double x53 = -x38*x49 - x39*(x34*x50 - x35*x46);
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = p_on_ee_y*pre_transform_s0 + p_on_ee_z*pre_transform_s1 - pre_transform_s0*pre_transform_s2 - pre_transform_s1*pre_transform_s3;
    jacobian(0, 1) = p_on_ee_y*x1 - p_on_ee_z*x2 - pre_transform_s2*x1 + pre_transform_s3*x2;
    jacobian(0, 2) = -p_on_ee_y*x8 + p_on_ee_z*x11 - x11*x15 + x13*x8;
    jacobian(0, 3) = -p_on_ee_y*x19 + p_on_ee_z*x21 + x19*x25 - x21*x23;
    jacobian(0, 4) = -p_on_ee_y*x28 + p_on_ee_z*x29 + x28*x33 - x29*x31;
    jacobian(0, 5) = -p_on_ee_y*x36 + p_on_ee_z*x37 - x31*x37 + x33*x36;
    jacobian(0, 6) = -p_on_ee_y*x40 + p_on_ee_z*x41 - x31*x41 + x33*x40;
    jacobian(1, 0) = -x42;
    jacobian(1, 1) = -p_on_ee_z*x5 + pre_transform_s3*x5 - x0*x43;
    jacobian(1, 2) = p_on_ee_x*x8 + p_on_ee_z*x44 + x14*x44 - x15*x44;
    jacobian(1, 3) = p_on_ee_x*x19 - p_on_ee_z*x46 - x19*x48 + x23*x46;
    jacobian(1, 4) = p_on_ee_x*x28 - p_on_ee_z*x49 - x28*x51 + x31*x49;
    jacobian(1, 5) = p_on_ee_x*x36 - p_on_ee_z*x52 + x31*x52 - x36*x51;
    jacobian(1, 6) = p_on_ee_x*x40 - p_on_ee_z*x53 + x31*x53 - x40*x51;
    jacobian(2, 0) = -x43;
    jacobian(2, 1) = p_on_ee_y*x5 - pre_transform_s2*x5 + x0*x42;
    jacobian(2, 2) = -p_on_ee_x*x11 - p_on_ee_y*x44 - x12*x44 + x13*x44;
    jacobian(2, 3) = -p_on_ee_x*x21 + p_on_ee_y*x46 + x21*x48 - x25*x46;
    jacobian(2, 4) = -p_on_ee_x*x29 + p_on_ee_y*x49 + x29*x51 - x33*x49;
    jacobian(2, 5) = -p_on_ee_x*x37 + p_on_ee_y*x52 - x33*x52 + x37*x51;
    jacobian(2, 6) = -p_on_ee_x*x41 + p_on_ee_y*x53 - x33*x53 + x41*x51;
    return;
}

static void computeRawIK(const Eigen::Matrix4d& T_ee, double th_0, SolutionQueue<intermediate_solution_size, max_n_solutions>& solution_queue, NodeIndexWorkspace<max_n_solutions>& node_index_workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    // Extracting the ik target symbols
    const double r_11 = T_ee(0, 0);
    const double r_12 = T_ee(0, 1);
    const double r_13 = T_ee(0, 2);
    const double Px = T_ee(0, 3);
    const double r_21 = T_ee(1, 0);
    const double r_22 = T_ee(1, 1);
    const double r_23 = T_ee(1, 2);
    const double Py = T_ee(1, 3);
    const double r_31 = T_ee(2, 0);
    const double r_32 = T_ee(2, 1);
    const double r_33 = T_ee(2, 2);
    const double Pz = T_ee(2, 3);
    const auto& ee_rotation = T_ee.block<3, 3>(0, 0);
    const auto& ee_translation = T_ee.block<3, 1>(0, 3);
    const Eigen::Vector3d inv_ee_translation = - ee_rotation.transpose() * ee_translation;
    const double inv_Px = inv_ee_translation(0);
    const double inv_Py = inv_ee_translation(1);
    const double inv_Pz = inv_ee_translation(2);
    
    solution_queue.reset();
    node_index_workspace.reset(n_tree_nodes);
    
    using RawSolution = IntermediateSolution<intermediate_solution_size>;
    auto make_raw_solution = []() -> RawSolution { return {}; };
    
    auto append_solution_to_queue = [&solution_queue](RawSolution solution_2_add) -> int {
        return solution_queue.push_solution(solution_2_add);
    };
    
    auto add_input_index_to = [&node_index_workspace](int node_idx, int solution_idx) -> void {
        if (solution_idx < 0) return;
        node_index_workspace.append_index_to_node(node_idx, solution_idx);
    };
    
    // Code for non-branch dispatcher node 0
    // Actually, there is no code
    
    // Code for explicit solution node 1, solved variable is th_3
    auto ExplicitSolutionNode_node_1_solve_th_3_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;
        
        // The explicit solution of root node
        const bool condition_0 = 4*std::fabs(a_3*d_2) >= zero_tolerance || std::fabs(2*std::pow(a_3, 2) - 2*std::pow(d_2, 2)) >= zero_tolerance || std::fabs(std::pow(Px, 2) + std::pow(Py, 2) + std::pow(Pz, 2) - 2*std::pow(a_3, 2) - 2*std::pow(d_2, 2)) >= zero_tolerance;
        if (condition_0)
        {
            // Temp variable for efficiency
            const double x0 = std::pow(a_3, 2);
            const double x1 = 2*x0;
            const double x2 = std::pow(d_2, 2);
            const double x3 = -2*x2;
            const double x4 = x1 + x3;
            const double x5 = std::atan2(-4*a_3*d_2, x4);
            const double x6 = std::pow(Px, 2) + std::pow(Py, 2) + std::pow(Pz, 2) - x1 + x3;
            const double x7 = safe_sqrt(16*x0*x2 + std::pow(x4, 2) - std::pow(x6, 2));
            // End of temp variables
            
            auto solution_0 = make_raw_solution();
            solution_0[8] = x5 + std::atan2(x7, x6);
            int appended_idx = append_solution_to_queue(solution_0);
            add_input_index_to(2, appended_idx);
        }
        
        const bool condition_1 = 4*std::fabs(a_3*d_2) >= zero_tolerance || std::fabs(2*std::pow(a_3, 2) - 2*std::pow(d_2, 2)) >= zero_tolerance || std::fabs(std::pow(Px, 2) + std::pow(Py, 2) + std::pow(Pz, 2) - 2*std::pow(a_3, 2) - 2*std::pow(d_2, 2)) >= zero_tolerance;
        if (condition_1)
        {
            // Temp variable for efficiency
            const double x0 = std::pow(a_3, 2);
            const double x1 = 2*x0;
            const double x2 = std::pow(d_2, 2);
            const double x3 = -2*x2;
            const double x4 = x1 + x3;
            const double x5 = std::atan2(-4*a_3*d_2, x4);
            const double x6 = std::pow(Px, 2) + std::pow(Py, 2) + std::pow(Pz, 2) - x1 + x3;
            const double x7 = safe_sqrt(16*x0*x2 + std::pow(x4, 2) - std::pow(x6, 2));
            // End of temp variables
            
            auto solution_1 = make_raw_solution();
            solution_1[8] = x5 + std::atan2(-x7, x6);
            int appended_idx = append_solution_to_queue(solution_1);
            add_input_index_to(2, appended_idx);
        }
        
    };
    // Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_3_processor();
    // Finish code for explicit solution node 0
    
    // Code for solved_variable dispatcher node 2
    auto SolvedVariableDispatcherNode_node_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(2);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(2);
        if (!this_input_valid)
            return;
        
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_3 = this_solution[8];
            
            const bool degenerate_valid_0 = std::fabs(th_3 - 2.9178925264888198 + M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(12, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_3 + M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(29, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(3, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_2_processor();
    // Finish code for solved_variable dispatcher node 2
    
    // Code for explicit solution node 29, solved variable is th_1
    auto ExplicitSolutionNode_node_29_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(29);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(29);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 29
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            
            const bool condition_0 = std::fabs(Pz) >= zero_tolerance || 2*std::fabs(d_2) >= zero_tolerance || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = (1.0/2.0)/d_2;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(-Px*std::cos(th_0) - Py*std::sin(th_0)), -Pz*x0);
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(30, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_1_processor();
    // Finish code for explicit solution node 29
    
    // Code for non-branch dispatcher node 30
    // Actually, there is no code
    
    // Code for explicit solution node 31, solved variable is th_5
    auto ExplicitSolutionNode_node_31_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(30);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(30);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 31
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            
            const bool condition_0 = (1.0/2.0)*std::fabs(inv_Pz/d_2) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = safe_acos((1.0/2.0)*inv_Pz/d_2);
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[11] = x0;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(32, appended_idx);
            }
            
            const bool condition_1 = (1.0/2.0)*std::fabs(inv_Pz/d_2) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = safe_acos((1.0/2.0)*inv_Pz/d_2);
                // End of temp variables
                const double tmp_sol_value = -x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[11] = tmp_sol_value;
                add_input_index_to(32, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_5_processor();
    // Finish code for explicit solution node 30
    
    // Code for solved_variable dispatcher node 32
    auto SolvedVariableDispatcherNode_node_32_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(32);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(32);
        if (!this_input_valid)
            return;
        
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_5 = this_solution[11];
            
            const bool degenerate_valid_0 = std::fabs(th_5) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
            }
            
            const bool degenerate_valid_1 = std::fabs(th_5 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(33, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_32_processor();
    // Finish code for solved_variable dispatcher node 32
    
    // Code for explicit solution node 33, solved variable is th_6
    auto ExplicitSolutionNode_node_33_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(33);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(33);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 33
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_5 = this_solution[11];
            
            const bool condition_0 = std::fabs(inv_Px) >= zero_tolerance || std::fabs(inv_Py) >= zero_tolerance || 2*std::fabs(d_2*std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = (1.0/2.0)/(d_2*std::sin(th_5));
                // End of temp variables
                const double tmp_sol_value = std::atan2(-inv_Py*x0, inv_Px*x0);
                solution_queue.get_solution(node_input_i_idx_in_queue)[12] = tmp_sol_value;
                add_input_index_to(34, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_33_solve_th_6_processor();
    // Finish code for explicit solution node 33
    
    // Code for solved_variable dispatcher node 34
    auto SolvedVariableDispatcherNode_node_34_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(34);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(34);
        if (!this_input_valid)
            return;
        
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_5 = this_solution[11];
            
            const bool degenerate_valid_0 = std::fabs(th_5) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
            }
            
            const bool degenerate_valid_1 = std::fabs(th_5 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(35, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_34_processor();
    // Finish code for solved_variable dispatcher node 34
    
    // Code for explicit solution node 35, solved variable is th_2th_4_soa
    auto ExplicitSolutionNode_node_35_solve_th_2th_4_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(35);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(35);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 35
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_5 = this_solution[11];
            
            const bool condition_0 = std::fabs(r_13*std::sin(th_0) - r_23*std::cos(th_0)) >= zero_tolerance || std::fabs(r_13*std::cos(th_0)*std::cos(th_1) + r_23*std::sin(th_0)*std::cos(th_1) - r_33*std::sin(th_1)) >= zero_tolerance || std::fabs(std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_5);
                const double x1 = std::sin(th_0);
                const double x2 = std::cos(th_0);
                const double x3 = std::cos(th_1);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(r_13*x1 - r_23*x2), x0*(r_13*x2*x3 + r_23*x1*x3 - r_33*std::sin(th_1)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(36, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_35_solve_th_2th_4_soa_processor();
    // Finish code for explicit solution node 35
    
    // Code for non-branch dispatcher node 36
    // Actually, there is no code
    
    // Code for explicit solution node 37, solved variable is th_2
    auto ExplicitSolutionNode_node_37_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(36);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(36);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 37
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(38, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_37_solve_th_2_processor();
    // Finish code for explicit solution node 36
    
    // Code for non-branch dispatcher node 38
    // Actually, there is no code
    
    // Code for explicit solution node 39, solved variable is th_4
    auto ExplicitSolutionNode_node_39_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(38);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(38);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 39
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[4];
            const double th_2th_4_soa = this_solution[5];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_2 + th_2th_4_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_39_solve_th_4_processor();
    // Finish code for explicit solution node 38
    
    // Code for explicit solution node 12, solved variable is th_2
    auto ExplicitSolutionNode_node_12_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(12);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(12);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 12
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            
            const bool condition_0 = std::fabs((Px*std::sin(th_0) - Py*std::cos(th_0))/(1.97508329336116*a_3 + 0.22183906555868099*d_2)) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = safe_asin((Px*std::sin(th_0) - Py*std::cos(th_0))/(1.97508329336116*a_3 + 0.22183906555868099*d_2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x0;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(13, appended_idx);
            }
            
            const bool condition_1 = std::fabs((Px*std::sin(th_0) - Py*std::cos(th_0))/(1.97508329336116*a_3 + 0.22183906555868099*d_2)) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = safe_asin((Px*std::sin(th_0) - Py*std::cos(th_0))/(1.97508329336116*a_3 + 0.22183906555868099*d_2));
                // End of temp variables
                const double tmp_sol_value = M_PI - x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(13, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_12_solve_th_2_processor();
    // Finish code for explicit solution node 12
    
    // Code for non-branch dispatcher node 13
    // Actually, there is no code
    
    // Code for explicit solution node 14, solved variable is th_1
    auto ExplicitSolutionNode_node_14_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(13);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(13);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 14
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            
            const bool condition_0 = std::fabs(Pz) >= zero_tolerance || std::fabs(0.22183906555868099*a_3 + 0.0249167066388374*d_2) >= zero_tolerance || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Px*std::cos(th_0) + Py*std::sin(th_0);
                const double x1 = std::atan2(x0, Pz);
                const double x2 = safe_sqrt(std::pow(Pz, 2) + std::pow(x0, 2) - 0.049212571007948999*std::pow(-a_3 - 0.11231884057971001*d_2, 2));
                const double x3 = -0.22183906555868099*a_3 - 0.0249167066388374*d_2;
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[2] = x1 + std::atan2(x2, x3);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(15, appended_idx);
            }
            
            const bool condition_1 = std::fabs(Pz) >= zero_tolerance || std::fabs(0.22183906555868099*a_3 + 0.0249167066388374*d_2) >= zero_tolerance || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = Px*std::cos(th_0) + Py*std::sin(th_0);
                const double x1 = std::atan2(x0, Pz);
                const double x2 = safe_sqrt(std::pow(Pz, 2) + std::pow(x0, 2) - 0.049212571007948999*std::pow(-a_3 - 0.11231884057971001*d_2, 2));
                const double x3 = -0.22183906555868099*a_3 - 0.0249167066388374*d_2;
                // End of temp variables
                const double tmp_sol_value = x1 + std::atan2(-x2, x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(15, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_1_processor();
    // Finish code for explicit solution node 13
    
    // Code for non-branch dispatcher node 15
    // Actually, there is no code
    
    // Code for explicit solution node 16, solved variable is th_5
    auto ExplicitSolutionNode_node_16_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(15);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(15);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 16
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = std::fabs(-r_13*((0.97508329336116295*std::sin(th_1) + 0.22183906555868099*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + 0.22183906555868099*std::sin(th_0)*std::sin(th_2)) - r_23*((0.97508329336116295*std::sin(th_1) + 0.22183906555868099*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - 0.22183906555868099*std::sin(th_2)*std::cos(th_0)) - r_33*(-0.22183906555868099*std::sin(th_1)*std::cos(th_2) + 0.97508329336116295*std::cos(th_1))) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_1);
                const double x2 = 0.22183906555868099*std::cos(th_2);
                const double x3 = std::sin(th_0);
                const double x4 = 0.22183906555868099*std::sin(th_2);
                const double x5 = std::cos(th_0);
                const double x6 = x0*x2 + 0.97508329336116295*x1;
                const double x7 = safe_acos(-r_13*(x3*x4 + x5*x6) - r_23*(x3*x6 - x4*x5) - r_33*(0.97508329336116295*x0 - x1*x2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[11] = x7;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(17, appended_idx);
            }
            
            const bool condition_1 = std::fabs(-r_13*((0.97508329336116295*std::sin(th_1) + 0.22183906555868099*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + 0.22183906555868099*std::sin(th_0)*std::sin(th_2)) - r_23*((0.97508329336116295*std::sin(th_1) + 0.22183906555868099*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - 0.22183906555868099*std::sin(th_2)*std::cos(th_0)) - r_33*(-0.22183906555868099*std::sin(th_1)*std::cos(th_2) + 0.97508329336116295*std::cos(th_1))) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_1);
                const double x2 = 0.22183906555868099*std::cos(th_2);
                const double x3 = std::sin(th_0);
                const double x4 = 0.22183906555868099*std::sin(th_2);
                const double x5 = std::cos(th_0);
                const double x6 = x0*x2 + 0.97508329336116295*x1;
                const double x7 = safe_acos(-r_13*(x3*x4 + x5*x6) - r_23*(x3*x6 - x4*x5) - r_33*(0.97508329336116295*x0 - x1*x2));
                // End of temp variables
                const double tmp_sol_value = -x7;
                solution_queue.get_solution(node_input_i_idx_in_queue)[11] = tmp_sol_value;
                add_input_index_to(17, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_5_processor();
    // Finish code for explicit solution node 15
    
    // Code for solved_variable dispatcher node 17
    auto SolvedVariableDispatcherNode_node_17_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(17);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(17);
        if (!this_input_valid)
            return;
        
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_5 = this_solution[11];
            
            const bool degenerate_valid_0 = std::fabs(th_5) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(50, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_5 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(55, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(18, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_17_processor();
    // Finish code for solved_variable dispatcher node 17
    
    // Code for explicit solution node 55, solved variable is th_4th_6_soa
    auto ExplicitSolutionNode_node_55_solve_th_4th_6_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(55);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(55);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 55
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_21*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_31*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(r_12*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_22*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_32*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_2);
                const double x1 = x0*std::sin(th_1);
                const double x2 = std::sin(th_0);
                const double x3 = std::cos(th_2);
                const double x4 = std::cos(th_0);
                const double x5 = x0*std::cos(th_1);
                const double x6 = x2*x3 - x4*x5;
                const double x7 = x2*x5 + x3*x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_11*x6 - r_21*x7 + r_31*x1, r_12*x6 - r_22*x7 + r_32*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[10] = tmp_sol_value;
                add_input_index_to(56, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_55_solve_th_4th_6_soa_processor();
    // Finish code for explicit solution node 55
    
    // Code for non-branch dispatcher node 56
    // Actually, there is no code
    
    // Code for explicit solution node 57, solved variable is th_4
    auto ExplicitSolutionNode_node_57_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(56);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(56);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 57
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
                add_input_index_to(58, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_57_solve_th_4_processor();
    // Finish code for explicit solution node 56
    
    // Code for non-branch dispatcher node 58
    // Actually, there is no code
    
    // Code for explicit solution node 59, solved variable is th_6
    auto ExplicitSolutionNode_node_59_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(58);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(58);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 59
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_4 = this_solution[9];
            const double th_4th_6_soa = this_solution[10];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_4 + th_4th_6_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[12] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_59_solve_th_6_processor();
    // Finish code for explicit solution node 58
    
    // Code for explicit solution node 50, solved variable is negative_th_6_positive_th_4__soa
    auto ExplicitSolutionNode_node_50_solve_negative_th_6_positive_th_4__soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(50);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(50);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 50
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_21*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_31*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(r_12*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_22*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_32*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_2);
                const double x1 = x0*std::sin(th_1);
                const double x2 = std::cos(th_0);
                const double x3 = std::cos(th_2);
                const double x4 = std::sin(th_0);
                const double x5 = x0*std::cos(th_1);
                const double x6 = x2*x3 + x4*x5;
                const double x7 = -x2*x5 + x3*x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x7 + r_21*x6 - r_31*x1, r_12*x7 - r_22*x6 + r_32*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(51, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_50_solve_negative_th_6_positive_th_4__soa_processor();
    // Finish code for explicit solution node 50
    
    // Code for non-branch dispatcher node 51
    // Actually, there is no code
    
    // Code for explicit solution node 52, solved variable is th_4
    auto ExplicitSolutionNode_node_52_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(51);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(51);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 52
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
                add_input_index_to(53, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_52_solve_th_4_processor();
    // Finish code for explicit solution node 51
    
    // Code for non-branch dispatcher node 53
    // Actually, there is no code
    
    // Code for explicit solution node 54, solved variable is th_6
    auto ExplicitSolutionNode_node_54_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(53);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(53);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 54
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double negative_th_6_positive_th_4__soa = this_solution[0];
            const double th_4 = this_solution[9];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -negative_th_6_positive_th_4__soa + th_4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[12] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_54_solve_th_6_processor();
    // Finish code for explicit solution node 53
    
    // Code for explicit solution node 18, solved variable is th_4
    auto ExplicitSolutionNode_node_18_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(18);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(18);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 18
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_5 = this_solution[11];
            
            const bool condition_0 = std::fabs(r_13*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_23*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_33*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(r_13*std::sin(th_1)*std::cos(th_0) + r_23*std::sin(th_0)*std::sin(th_1) + r_33*std::cos(th_1) + 0.97508329336116295*std::cos(th_5)) >= zero_tolerance || 0.22183906555868099*std::fabs(std::sin(th_5)) >= zero_tolerance || std::fabs(std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_5);
                const double x1 = std::sin(th_1);
                const double x2 = std::sin(th_2);
                const double x3 = std::sin(th_0);
                const double x4 = std::cos(th_2);
                const double x5 = std::cos(th_0);
                const double x6 = std::cos(th_1);
                const double x7 = x2*x6;
                const double x8 = 4.5077723235156597*x1;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(r_13*(x3*x4 - x5*x7) - r_23*(x3*x7 + x4*x5) + r_33*x1*x2), x0*(r_13*x5*x8 + r_23*x3*x8 + 4.5077723235156597*r_33*x6 + 4.3954534829359515*std::cos(th_5)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
                add_input_index_to(19, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_4_processor();
    // Finish code for explicit solution node 18
    
    // Code for non-branch dispatcher node 19
    // Actually, there is no code
    
    // Code for explicit solution node 20, solved variable is th_2th_4th_5_soa
    auto ExplicitSolutionNode_node_20_solve_th_2th_4th_5_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(19);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(19);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 20
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[9];
            const double th_5 = this_solution[11];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_2 + th_4 + th_5;
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(21, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_20_solve_th_2th_4th_5_soa_processor();
    // Finish code for explicit solution node 19
    
    // Code for non-branch dispatcher node 21
    // Actually, there is no code
    
    // Code for explicit solution node 22, solved variable is th_0th_2th_4_soa
    auto ExplicitSolutionNode_node_22_solve_th_0th_2th_4_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(21);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(21);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 22
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[9];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_0 + th_2 + th_4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(23, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_22_solve_th_0th_2th_4_soa_processor();
    // Finish code for explicit solution node 21
    
    // Code for non-branch dispatcher node 23
    // Actually, there is no code
    
    // Code for explicit solution node 24, solved variable is th_1th_2th_4_soa
    auto ExplicitSolutionNode_node_24_solve_th_1th_2th_4_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(23);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(23);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 24
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[9];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_1 + th_2 + th_4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(25, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_1th_2th_4_soa_processor();
    // Finish code for explicit solution node 23
    
    // Code for non-branch dispatcher node 25
    // Actually, there is no code
    
    // Code for explicit solution node 26, solved variable is th_6
    auto ExplicitSolutionNode_node_26_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(25);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(25);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 26
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[9];
            const double th_5 = this_solution[11];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(-r_11*((((-0.22183906555868099*std::sin(th_5) + 0.97508329336116295*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) + std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::cos(th_1) - (0.97508329336116295*std::sin(th_5) + 0.22183906555868099*std::cos(th_4)*std::cos(th_5))*std::sin(th_1))*std::cos(th_0) - ((0.22183906555868099*std::sin(th_5) - 0.97508329336116295*std::cos(th_4)*std::cos(th_5))*std::sin(th_2) + std::sin(th_4)*std::cos(th_2)*std::cos(th_5))*std::sin(th_0)) - r_21*((((-0.22183906555868099*std::sin(th_5) + 0.97508329336116295*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) + std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::cos(th_1) - (0.97508329336116295*std::sin(th_5) + 0.22183906555868099*std::cos(th_4)*std::cos(th_5))*std::sin(th_1))*std::sin(th_0) + ((0.22183906555868099*std::sin(th_5) - 0.97508329336116295*std::cos(th_4)*std::cos(th_5))*std::sin(th_2) + std::sin(th_4)*std::cos(th_2)*std::cos(th_5))*std::cos(th_0)) - r_31*((-(-0.22183906555868099*std::sin(th_5) + 0.97508329336116295*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) - std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::sin(th_1) - (0.97508329336116295*std::sin(th_5) + 0.22183906555868099*std::cos(th_4)*std::cos(th_5))*std::cos(th_1))) >= zero_tolerance || std::fabs(-r_12*((((-0.22183906555868099*std::sin(th_5) + 0.97508329336116295*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) + std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::cos(th_1) - (0.97508329336116295*std::sin(th_5) + 0.22183906555868099*std::cos(th_4)*std::cos(th_5))*std::sin(th_1))*std::cos(th_0) - ((0.22183906555868099*std::sin(th_5) - 0.97508329336116295*std::cos(th_4)*std::cos(th_5))*std::sin(th_2) + std::sin(th_4)*std::cos(th_2)*std::cos(th_5))*std::sin(th_0)) - r_22*((((-0.22183906555868099*std::sin(th_5) + 0.97508329336116295*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) + std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::cos(th_1) - (0.97508329336116295*std::sin(th_5) + 0.22183906555868099*std::cos(th_4)*std::cos(th_5))*std::sin(th_1))*std::sin(th_0) + ((0.22183906555868099*std::sin(th_5) - 0.97508329336116295*std::cos(th_4)*std::cos(th_5))*std::sin(th_2) + std::sin(th_4)*std::cos(th_2)*std::cos(th_5))*std::cos(th_0)) - r_32*((-(-0.22183906555868099*std::sin(th_5) + 0.97508329336116295*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) - std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::sin(th_1) - (0.97508329336116295*std::sin(th_5) + 0.22183906555868099*std::cos(th_4)*std::cos(th_5))*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_5);
                const double x3 = x2*std::cos(th_4);
                const double x4 = 0.97508329336116295*x1 + 0.22183906555868099*x3;
                const double x5 = std::sin(th_1);
                const double x6 = std::sin(th_2);
                const double x7 = x2*std::sin(th_4);
                const double x8 = x6*x7;
                const double x9 = std::cos(th_2);
                const double x10 = 0.22183906555868099*x1;
                const double x11 = 0.97508329336116295*x3;
                const double x12 = x9*(-x10 + x11);
                const double x13 = -x0*x4 + x5*(-x12 - x8);
                const double x14 = std::cos(th_0);
                const double x15 = x6*(x10 - x11) + x7*x9;
                const double x16 = std::sin(th_0);
                const double x17 = x0*(x12 + x8) - x4*x5;
                const double x18 = x14*x15 + x16*x17;
                const double x19 = x14*x17 - x15*x16;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_12*x19 - r_22*x18 - r_32*x13, r_11*x19 + r_21*x18 + r_31*x13);
                solution_queue.get_solution(node_input_i_idx_in_queue)[12] = tmp_sol_value;
                add_input_index_to(27, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_26_solve_th_6_processor();
    // Finish code for explicit solution node 25
    
    // Code for non-branch dispatcher node 27
    // Actually, there is no code
    
    // Code for explicit solution node 28, solved variable is th_2th_4th_6_soa
    auto ExplicitSolutionNode_node_28_solve_th_2th_4th_6_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(27);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(27);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 28
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[9];
            const double th_6 = this_solution[12];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_2 + th_4 + th_6;
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_28_solve_th_2th_4th_6_soa_processor();
    // Finish code for explicit solution node 27
    
    // Code for explicit solution node 3, solved variable is th_2
    auto ExplicitSolutionNode_node_3_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(3);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(3);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 3
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_3 = this_solution[8];
            
            const bool condition_0 = std::fabs((Px*std::sin(th_0) - Py*std::cos(th_0))/(a_3*std::cos(th_3) + a_3 - d_2*std::sin(th_3))) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = safe_asin((-Px*std::sin(th_0) + Py*std::cos(th_0))/(-a_3*std::cos(th_3) - a_3 + d_2*std::sin(th_3)));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x0;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = std::fabs((Px*std::sin(th_0) - Py*std::cos(th_0))/(a_3*std::cos(th_3) + a_3 - d_2*std::sin(th_3))) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = safe_asin((-Px*std::sin(th_0) + Py*std::cos(th_0))/(-a_3*std::cos(th_3) - a_3 + d_2*std::sin(th_3)));
                // End of temp variables
                const double tmp_sol_value = M_PI - x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(4, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_2_processor();
    // Finish code for explicit solution node 3
    
    // Code for equation all-zero dispatcher node 4
    auto EquationAllZeroDispatcherNode_node_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(4);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(4);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const bool checked_result = std::fabs(Pz) <= 9.9999999999999995e-7 && std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor();
    // Finish code for equation all-zero dispatcher node 4
    
    // Code for explicit solution node 5, solved variable is th_1
    auto ExplicitSolutionNode_node_5_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(5);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(5);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 5
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[4];
            const double th_3 = this_solution[8];
            
            const bool condition_0 = std::fabs(Pz) >= 9.9999999999999995e-7 || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_3);
                const double x1 = std::cos(th_3);
                const double x2 = (-a_3*x1 - a_3 + d_2*x0)*std::cos(th_2);
                const double x3 = -Px*std::cos(th_0) - Py*std::sin(th_0);
                const double x4 = a_3*x0 + d_2*x1 - d_2;
                // End of temp variables
                const double tmp_sol_value = std::atan2(Pz*x2 - x3*x4, Pz*x4 + x2*x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(6, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_1_processor();
    // Finish code for explicit solution node 5
    
    // Code for non-branch dispatcher node 6
    // Actually, there is no code
    
    // Code for explicit solution node 7, solved variable is th_5
    auto ExplicitSolutionNode_node_7_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(6);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(6);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 7
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_3 = this_solution[8];
            
            const bool condition_0 = std::fabs(r_13*((-std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + std::sin(th_0)*std::sin(th_2)*std::sin(th_3)) + r_23*((-std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - std::sin(th_2)*std::sin(th_3)*std::cos(th_0)) - r_33*(std::sin(th_1)*std::sin(th_3)*std::cos(th_2) + std::cos(th_1)*std::cos(th_3))) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::cos(th_3);
                const double x2 = std::sin(th_1);
                const double x3 = std::sin(th_3);
                const double x4 = x3*std::cos(th_2);
                const double x5 = std::sin(th_0);
                const double x6 = x3*std::sin(th_2);
                const double x7 = std::cos(th_0);
                const double x8 = x0*x4 - x1*x2;
                const double x9 = safe_acos(r_13*(x5*x6 + x7*x8) + r_23*(x5*x8 - x6*x7) - r_33*(x0*x1 + x2*x4));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[11] = x9;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(8, appended_idx);
            }
            
            const bool condition_1 = std::fabs(r_13*((-std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + std::sin(th_0)*std::sin(th_2)*std::sin(th_3)) + r_23*((-std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - std::sin(th_2)*std::sin(th_3)*std::cos(th_0)) - r_33*(std::sin(th_1)*std::sin(th_3)*std::cos(th_2) + std::cos(th_1)*std::cos(th_3))) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::cos(th_3);
                const double x2 = std::sin(th_1);
                const double x3 = std::sin(th_3);
                const double x4 = x3*std::cos(th_2);
                const double x5 = std::sin(th_0);
                const double x6 = x3*std::sin(th_2);
                const double x7 = std::cos(th_0);
                const double x8 = x0*x4 - x1*x2;
                const double x9 = safe_acos(r_13*(x5*x6 + x7*x8) + r_23*(x5*x8 - x6*x7) - r_33*(x0*x1 + x2*x4));
                // End of temp variables
                const double tmp_sol_value = -x9;
                solution_queue.get_solution(node_input_i_idx_in_queue)[11] = tmp_sol_value;
                add_input_index_to(8, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_5_processor();
    // Finish code for explicit solution node 6
    
    // Code for solved_variable dispatcher node 8
    auto SolvedVariableDispatcherNode_node_8_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(8);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(8);
        if (!this_input_valid)
            return;
        
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_5 = this_solution[11];
            
            const bool degenerate_valid_0 = std::fabs(th_5) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(40, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_5 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(45, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(9, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor();
    // Finish code for solved_variable dispatcher node 8
    
    // Code for explicit solution node 45, solved variable is th_4th_6_soa
    auto ExplicitSolutionNode_node_45_solve_th_4th_6_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(45);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(45);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 45
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_21*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_31*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(r_12*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_22*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_32*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_2);
                const double x1 = x0*std::sin(th_1);
                const double x2 = std::sin(th_0);
                const double x3 = std::cos(th_2);
                const double x4 = std::cos(th_0);
                const double x5 = x0*std::cos(th_1);
                const double x6 = x2*x3 - x4*x5;
                const double x7 = x2*x5 + x3*x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_11*x6 - r_21*x7 + r_31*x1, r_12*x6 - r_22*x7 + r_32*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[10] = tmp_sol_value;
                add_input_index_to(46, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_45_solve_th_4th_6_soa_processor();
    // Finish code for explicit solution node 45
    
    // Code for non-branch dispatcher node 46
    // Actually, there is no code
    
    // Code for explicit solution node 47, solved variable is th_4
    auto ExplicitSolutionNode_node_47_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(46);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(46);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 47
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
                add_input_index_to(48, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_47_solve_th_4_processor();
    // Finish code for explicit solution node 46
    
    // Code for non-branch dispatcher node 48
    // Actually, there is no code
    
    // Code for explicit solution node 49, solved variable is th_6
    auto ExplicitSolutionNode_node_49_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(48);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(48);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 49
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_4 = this_solution[9];
            const double th_4th_6_soa = this_solution[10];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_4 + th_4th_6_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[12] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_49_solve_th_6_processor();
    // Finish code for explicit solution node 48
    
    // Code for explicit solution node 40, solved variable is negative_th_6_positive_th_4__soa
    auto ExplicitSolutionNode_node_40_solve_negative_th_6_positive_th_4__soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(40);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(40);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 40
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_21*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_31*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(r_12*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_22*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_32*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_2);
                const double x1 = x0*std::sin(th_1);
                const double x2 = std::cos(th_0);
                const double x3 = std::cos(th_2);
                const double x4 = std::sin(th_0);
                const double x5 = x0*std::cos(th_1);
                const double x6 = x2*x3 + x4*x5;
                const double x7 = -x2*x5 + x3*x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x7 + r_21*x6 - r_31*x1, r_12*x7 - r_22*x6 + r_32*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(41, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_40_solve_negative_th_6_positive_th_4__soa_processor();
    // Finish code for explicit solution node 40
    
    // Code for non-branch dispatcher node 41
    // Actually, there is no code
    
    // Code for explicit solution node 42, solved variable is th_4
    auto ExplicitSolutionNode_node_42_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(41);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(41);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 42
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
                add_input_index_to(43, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_42_solve_th_4_processor();
    // Finish code for explicit solution node 41
    
    // Code for non-branch dispatcher node 43
    // Actually, there is no code
    
    // Code for explicit solution node 44, solved variable is th_6
    auto ExplicitSolutionNode_node_44_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(43);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(43);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 44
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double negative_th_6_positive_th_4__soa = this_solution[0];
            const double th_4 = this_solution[9];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -negative_th_6_positive_th_4__soa + th_4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[12] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_44_solve_th_6_processor();
    // Finish code for explicit solution node 43
    
    // Code for explicit solution node 9, solved variable is th_4
    auto ExplicitSolutionNode_node_9_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(9);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(9);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 9
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_3 = this_solution[8];
            const double th_5 = this_solution[11];
            
            const bool condition_0 = std::fabs(-r_13*((std::sin(th_1)*std::sin(th_3) + std::cos(th_1)*std::cos(th_2)*std::cos(th_3))*std::cos(th_0) + std::sin(th_0)*std::sin(th_2)*std::cos(th_3)) - r_23*((std::sin(th_1)*std::sin(th_3) + std::cos(th_1)*std::cos(th_2)*std::cos(th_3))*std::sin(th_0) - std::sin(th_2)*std::cos(th_0)*std::cos(th_3)) - r_33*(-std::sin(th_1)*std::cos(th_2)*std::cos(th_3) + std::sin(th_3)*std::cos(th_1))) >= zero_tolerance || std::fabs(r_13*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_23*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_33*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_5);
                const double x1 = std::sin(th_1);
                const double x2 = std::sin(th_2);
                const double x3 = std::sin(th_0);
                const double x4 = std::cos(th_2);
                const double x5 = std::cos(th_0);
                const double x6 = std::cos(th_1);
                const double x7 = x2*x6;
                const double x8 = std::sin(th_3);
                const double x9 = std::cos(th_3);
                const double x10 = x4*x9;
                const double x11 = x2*x9;
                const double x12 = x1*x8 + x10*x6;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(r_13*(x3*x4 - x5*x7) - r_23*(x3*x7 + x4*x5) + r_33*x1*x2), x0*(-r_13*(x11*x3 + x12*x5) - r_23*(-x11*x5 + x12*x3) - r_33*(-x1*x10 + x6*x8)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
                add_input_index_to(10, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_4_processor();
    // Finish code for explicit solution node 9
    
    // Code for non-branch dispatcher node 10
    // Actually, there is no code
    
    // Code for explicit solution node 11, solved variable is th_6
    auto ExplicitSolutionNode_node_11_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(10);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(10);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 11
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_3 = this_solution[8];
            const double th_5 = this_solution[11];
            
            const bool condition_0 = std::fabs(-r_11*((-std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + std::sin(th_0)*std::sin(th_2)*std::sin(th_3)) - r_21*((-std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - std::sin(th_2)*std::sin(th_3)*std::cos(th_0)) + r_31*(std::sin(th_1)*std::sin(th_3)*std::cos(th_2) + std::cos(th_1)*std::cos(th_3))) >= zero_tolerance || std::fabs(r_12*((std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) - std::sin(th_0)*std::sin(th_2)*std::sin(th_3)) + r_22*((std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) + std::sin(th_2)*std::sin(th_3)*std::cos(th_0)) + r_32*(std::sin(th_1)*std::sin(th_3)*std::cos(th_2) + std::cos(th_1)*std::cos(th_3))) >= zero_tolerance || std::fabs(std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_5);
                const double x1 = std::cos(th_1);
                const double x2 = std::cos(th_3);
                const double x3 = std::sin(th_1);
                const double x4 = std::sin(th_3);
                const double x5 = x4*std::cos(th_2);
                const double x6 = x1*x2 + x3*x5;
                const double x7 = std::cos(th_0);
                const double x8 = x4*std::sin(th_2);
                const double x9 = x7*x8;
                const double x10 = std::sin(th_0);
                const double x11 = x2*x3;
                const double x12 = x1*x5;
                const double x13 = x11 - x12;
                const double x14 = x10*x8;
                const double x15 = -x11 + x12;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(r_12*(x13*x7 - x14) + r_22*(x10*x13 + x9) + r_32*x6), x0*(r_11*(x14 + x15*x7) + r_21*(x10*x15 - x9) - r_31*x6));
                solution_queue.get_solution(node_input_i_idx_in_queue)[12] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_6_processor();
    // Finish code for explicit solution node 10
    
    // Collect the output
    for(int i = 0; i < solution_queue.size(); i++)
    {
        if(!solution_queue.solutions_validity[i])
            continue;
        const auto& raw_ik_out_i = solution_queue.get_solution(i);
        std::array<double, robot_nq> new_ik_i;
        const double value_at_0 = th_0;  // th_0
        new_ik_i[0] = value_at_0;
        const double value_at_1 = raw_ik_out_i[2];  // th_1
        new_ik_i[1] = value_at_1;
        const double value_at_2 = raw_ik_out_i[4];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[8];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[9];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[11];  // th_5
        new_ik_i[5] = value_at_5;
        const double value_at_6 = raw_ik_out_i[12];  // th_6
        new_ik_i[6] = value_at_6;
        ik_output.push_back(new_ik_i);
    }
}

static void computeRawIK(const Eigen::Matrix4d& T_ee_raw, double th_0, RawIKWorksace& workspace)
{
    workspace.raw_ik_out.clear();
    workspace.raw_ik_out.reserve(max_n_solutions);
    computeRawIK(T_ee_raw, th_0, workspace.solution_queue, workspace.node_index_workspace, workspace.raw_ik_out);
}

static void computeIKUnChecked(const Eigen::Matrix4d& T_ee, double th_0, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, th_0 + th_0_offset_original2raw, workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
        ik_out_i[0] -= th_0_offset_original2raw;
        ik_out_i[1] -= th_1_offset_original2raw;
        ik_out_i[2] -= th_2_offset_original2raw;
        ik_out_i[3] -= th_3_offset_original2raw;
        ik_out_i[4] -= th_4_offset_original2raw;
        ik_out_i[5] -= th_5_offset_original2raw;
        ik_out_i[6] -= th_6_offset_original2raw;
        ik_output.push_back(ik_out_i);
    }
}

static void wrapAngleToPi(std::vector<std::array<double, robot_nq>>& ik_output)
{
    for(int i = 0; i < ik_output.size(); i++)
    {
        // Wrap angular value to [-pi, pi]
        auto& solution_i = ik_output[i];
        // Revolute unknown th_0
        while(solution_i[0] > M_PI)
            solution_i[0] -= 2 * M_PI;
        while(solution_i[0] < - M_PI)
            solution_i[0] += 2 * M_PI;
        // Revolute unknown th_1
        while(solution_i[1] > M_PI)
            solution_i[1] -= 2 * M_PI;
        while(solution_i[1] < - M_PI)
            solution_i[1] += 2 * M_PI;
        // Revolute unknown th_2
        while(solution_i[2] > M_PI)
            solution_i[2] -= 2 * M_PI;
        while(solution_i[2] < - M_PI)
            solution_i[2] += 2 * M_PI;
        // Revolute unknown th_3
        while(solution_i[3] > M_PI)
            solution_i[3] -= 2 * M_PI;
        while(solution_i[3] < - M_PI)
            solution_i[3] += 2 * M_PI;
        // Revolute unknown th_4
        while(solution_i[4] > M_PI)
            solution_i[4] -= 2 * M_PI;
        while(solution_i[4] < - M_PI)
            solution_i[4] += 2 * M_PI;
        // Revolute unknown th_5
        while(solution_i[5] > M_PI)
            solution_i[5] -= 2 * M_PI;
        while(solution_i[5] < - M_PI)
            solution_i[5] += 2 * M_PI;
        // Revolute unknown th_6
        while(solution_i[6] > M_PI)
            solution_i[6] -= 2 * M_PI;
        while(solution_i[6] < - M_PI)
            solution_i[6] += 2 * M_PI;
    }
}

static void computeIK(const Eigen::Matrix4d& T_ee, double th_0, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, th_0 + th_0_offset_original2raw, workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
        ik_out_i[0] -= th_0_offset_original2raw;
        ik_out_i[1] -= th_1_offset_original2raw;
        ik_out_i[2] -= th_2_offset_original2raw;
        ik_out_i[3] -= th_3_offset_original2raw;
        ik_out_i[4] -= th_4_offset_original2raw;
        ik_out_i[5] -= th_5_offset_original2raw;
        ik_out_i[6] -= th_6_offset_original2raw;
        const Eigen::Matrix4d ee_pose_i = computeFK(ik_out_i);
        double ee_pose_diff = (ee_pose_i - T_ee).squaredNorm();
        if (ee_pose_diff < pose_tolerance)
            ik_output.push_back(ik_out_i);
    }
    
    if (!ik_output.empty())
    {
        wrapAngleToPi(ik_output);
        removeDuplicate<robot_nq>(ik_output, zero_tolerance);
        return;
    }
    
    // Disturbing method for degenerate handling
    Eigen::Matrix4d T_ee_raw_disturbed = yaik_cpp::disturbTransform(T_ee_raw);
    Eigen::Matrix4d T_ee_disturbed = endEffectorTargetRawToOriginal(T_ee_raw_disturbed);
    computeRawIK(T_ee_raw_disturbed, th_0 + th_0_offset_original2raw, workspace);
    const auto& raw_ik_out_disturb = workspace.raw_ik_out;
    for(int i = 0; i < raw_ik_out_disturb.size(); i++)
    {
        auto ik_out_i = raw_ik_out_disturb[i];
        ik_out_i[0] -= th_0_offset_original2raw;
        ik_out_i[1] -= th_1_offset_original2raw;
        ik_out_i[2] -= th_2_offset_original2raw;
        ik_out_i[3] -= th_3_offset_original2raw;
        ik_out_i[4] -= th_4_offset_original2raw;
        ik_out_i[5] -= th_5_offset_original2raw;
        ik_out_i[6] -= th_6_offset_original2raw;
        Eigen::Matrix4d ee_pose_i = computeFK(ik_out_i);
        double ee_pose_diff = (ee_pose_i - T_ee_disturbed).squaredNorm();
        if (ee_pose_diff > pose_tolerance_degenerate)
            continue;
        
        // Try numerical refinement
        yaik_cpp::numericalRefinement<robot_nq>(computeFK, computeTwistJacobian, T_ee, ik_out_i);
        ee_pose_i = computeFK(ik_out_i);
        ee_pose_diff = (ee_pose_i - T_ee).squaredNorm();
        if (ee_pose_diff < pose_tolerance_degenerate)
            ik_output.push_back(ik_out_i);
    }
    
    wrapAngleToPi(ik_output);
    removeDuplicate<robot_nq>(ik_output, zero_tolerance);
}

static std::vector<std::array<double, robot_nq>> computeIK(const Eigen::Matrix4d& T_ee, double th_0)
{
    std::vector<std::array<double, robot_nq>> ik_output;
    RawIKWorksace raw_ik_workspace;
    computeIK(T_ee, th_0, raw_ik_workspace, ik_output);
    return ik_output;
}

}; // struct rainbow_y1_r_arm_ik

// Code below for debug
void test_ik_solve_rainbow_y1_r_arm()
{
    std::array<double, rainbow_y1_r_arm_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = rainbow_y1_r_arm_ik::computeFK(theta);
    auto ik_output = rainbow_y1_r_arm_ik::computeIK(ee_pose, theta[0]);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = rainbow_y1_r_arm_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_rainbow_y1_r_arm();
}
