#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct franka_panda_ik {

// Constants for solver
static constexpr int robot_nq = 7;
static constexpr int max_n_solutions = 16;
static constexpr int n_tree_nodes = 16;
static constexpr int intermediate_solution_size = 6;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_3 = 0.00825;
static constexpr double a_5 = 0.088;
static constexpr double d_2 = 0.316;
static constexpr double d_4 = 0.384;
static constexpr double post_transform_d5 = 0.107;
static constexpr double post_transform_sqrt2_over2 = 0.707107;
static constexpr double pre_transform_d4 = 0.333;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = 3.141592653589793;
static constexpr double th_2_offset_original2raw = 3.141592653589793;
static constexpr double th_3_offset_original2raw = 3.141592653589793;
static constexpr double th_4_offset_original2raw = 0.0;
static constexpr double th_5_offset_original2raw = 3.141592653589793;
static constexpr double th_6_offset_original2raw = 0.0;

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
    ee_transformed(0, 0) = post_transform_sqrt2_over2*(r_11 + r_12);
    ee_transformed(0, 1) = post_transform_sqrt2_over2*(-r_11 + r_12);
    ee_transformed(0, 2) = r_13;
    ee_transformed(0, 3) = Px - post_transform_d5*r_13;
    ee_transformed(1, 0) = post_transform_sqrt2_over2*(r_21 + r_22);
    ee_transformed(1, 1) = post_transform_sqrt2_over2*(-r_21 + r_22);
    ee_transformed(1, 2) = r_23;
    ee_transformed(1, 3) = Py - post_transform_d5*r_23;
    ee_transformed(2, 0) = post_transform_sqrt2_over2*(r_31 + r_32);
    ee_transformed(2, 1) = post_transform_sqrt2_over2*(-r_31 + r_32);
    ee_transformed(2, 2) = r_33;
    ee_transformed(2, 3) = Pz - post_transform_d5*r_33 - pre_transform_d4;
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
    ee_transformed(0, 0) = post_transform_sqrt2_over2*(r_11 - r_12);
    ee_transformed(0, 1) = post_transform_sqrt2_over2*(r_11 + r_12);
    ee_transformed(0, 2) = r_13;
    ee_transformed(0, 3) = Px + post_transform_d5*r_13;
    ee_transformed(1, 0) = post_transform_sqrt2_over2*(r_21 - r_22);
    ee_transformed(1, 1) = post_transform_sqrt2_over2*(r_21 + r_22);
    ee_transformed(1, 2) = r_23;
    ee_transformed(1, 3) = Py + post_transform_d5*r_23;
    ee_transformed(2, 0) = post_transform_sqrt2_over2*(r_31 - r_32);
    ee_transformed(2, 1) = post_transform_sqrt2_over2*(r_31 + r_32);
    ee_transformed(2, 2) = r_33;
    ee_transformed(2, 3) = Pz + post_transform_d5*r_33 + pre_transform_d4;
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
    const double x5 = std::sin(th_2);
    const double x6 = std::cos(th_0);
    const double x7 = std::cos(th_1);
    const double x8 = -x4 + x5*x6*x7;
    const double x9 = std::sin(th_4);
    const double x10 = std::sin(th_3);
    const double x11 = std::sin(th_1);
    const double x12 = x11*x6;
    const double x13 = std::cos(th_3);
    const double x14 = x2*x5;
    const double x15 = x3*x6;
    const double x16 = x14 + x15*x7;
    const double x17 = -x10*x12 + x13*x16;
    const double x18 = x1*x8 - x17*x9;
    const double x19 = std::cos(th_6);
    const double x20 = std::sin(th_5);
    const double x21 = x10*x16 + x12*x13;
    const double x22 = std::cos(th_5);
    const double x23 = x1*x17 + x8*x9;
    const double x24 = -x20*x21 + x22*x23;
    const double x25 = x14*x7 + x15;
    const double x26 = x11*x2;
    const double x27 = x4*x7 - x5*x6;
    const double x28 = -x10*x26 + x13*x27;
    const double x29 = x1*x25 - x28*x9;
    const double x30 = x10*x27 + x13*x26;
    const double x31 = x1*x28 + x25*x9;
    const double x32 = -x20*x30 + x22*x31;
    const double x33 = x11*x5;
    const double x34 = x11*x3;
    const double x35 = -x10*x7 - x13*x34;
    const double x36 = -x1*x33 - x35*x9;
    const double x37 = x10*x34 - x13*x7;
    const double x38 = -x37;
    const double x39 = x1*x35 - x33*x9;
    const double x40 = -x20*x38 + x22*x39;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = x0*x18 + x19*x24;
    ee_pose_raw(0, 1) = -x0*x24 + x18*x19;
    ee_pose_raw(0, 2) = x20*x23 + x21*x22;
    ee_pose_raw(0, 3) = a_3*x16 + a_3*x17 + a_5*x24 - d_2*x12 + d_4*x21;
    ee_pose_raw(1, 0) = x0*x29 + x19*x32;
    ee_pose_raw(1, 1) = -x0*x32 + x19*x29;
    ee_pose_raw(1, 2) = x20*x31 + x22*x30;
    ee_pose_raw(1, 3) = a_3*x27 + a_3*x28 + a_5*x32 - d_2*x26 + d_4*x30;
    ee_pose_raw(2, 0) = x0*x36 + x19*x40;
    ee_pose_raw(2, 1) = -x0*x40 + x19*x36;
    ee_pose_raw(2, 2) = x20*x39 + x22*x38;
    ee_pose_raw(2, 3) = -a_3*x34 + a_3*x35 + a_5*x40 - d_2*x7 - d_4*x37;
    return endEffectorTargetRawToOriginal(ee_pose_raw);
}

static void computeTwistJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, 7>& jacobian)
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
    const double x0 = std::sin(th_0);
    const double x1 = std::sin(th_1);
    const double x2 = std::cos(th_0);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_2);
    const double x5 = x0*x4;
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_1);
    const double x8 = x2*x6*x7 - x5;
    const double x9 = std::cos(th_3);
    const double x10 = std::sin(th_3);
    const double x11 = x0*x6;
    const double x12 = x2*x4;
    const double x13 = x11 + x12*x7;
    const double x14 = x10*x13 + x3*x9;
    const double x15 = std::cos(th_4);
    const double x16 = std::sin(th_4);
    const double x17 = -x10*x3 + x13*x9;
    const double x18 = x15*x8 - x16*x17;
    const double x19 = std::cos(th_5);
    const double x20 = std::sin(th_5);
    const double x21 = x15*x17 + x16*x8;
    const double x22 = x14*x19 + x20*x21;
    const double x23 = x0*x1;
    const double x24 = x11*x7 + x12;
    const double x25 = -x2*x6 + x5*x7;
    const double x26 = x10*x25 + x23*x9;
    const double x27 = -x10*x23 + x25*x9;
    const double x28 = x15*x24 - x16*x27;
    const double x29 = x15*x27 + x16*x24;
    const double x30 = x19*x26 + x20*x29;
    const double x31 = x1*x6;
    const double x32 = x1*x4;
    const double x33 = x10*x32 - x7*x9;
    const double x34 = -x33;
    const double x35 = -x10*x7 - x32*x9;
    const double x36 = -x15*x31 - x16*x35;
    const double x37 = x15*x35 - x16*x31;
    const double x38 = x19*x34 + x20*x37;
    const double x39 = d_2*x7;
    const double x40 = -pre_transform_d4 + x39;
    const double x41 = -x40;
    const double x42 = a_3*x32;
    const double x43 = x40 + x42;
    const double x44 = -x43;
    const double x45 = a_3*x25 - d_2*x23;
    const double x46 = d_4*x33;
    const double x47 = a_3*x35 - x43 - x46;
    const double x48 = a_3*x27 + d_4*x26 + x45;
    const double x49 = a_3*x35 + a_5*(x19*x37 - x20*x34) + pre_transform_d4 - x39 - x42 - x46;
    const double x50 = a_5*(x19*x29 - x20*x26) + x48;
    const double x51 = a_3*x13 - d_2*x3;
    const double x52 = a_3*x17 + d_4*x14 + x51;
    const double x53 = a_5*(-x14*x20 + x19*x21) + x52;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = -x0;
    jacobian(0, 2) = -x3;
    jacobian(0, 3) = x8;
    jacobian(0, 4) = x14;
    jacobian(0, 5) = x18;
    jacobian(0, 6) = x22;
    jacobian(1, 1) = x2;
    jacobian(1, 2) = -x23;
    jacobian(1, 3) = x24;
    jacobian(1, 4) = x26;
    jacobian(1, 5) = x28;
    jacobian(1, 6) = x30;
    jacobian(2, 0) = 1;
    jacobian(2, 2) = -x7;
    jacobian(2, 3) = -x31;
    jacobian(2, 4) = x34;
    jacobian(2, 5) = x36;
    jacobian(2, 6) = x38;
    jacobian(3, 1) = -pre_transform_d4*x2;
    jacobian(3, 2) = x23*x39 + x23*x41;
    jacobian(3, 3) = -x24*x44 - x31*x45;
    jacobian(3, 4) = -x26*x47 + x34*x48;
    jacobian(3, 5) = -x28*x47 + x36*x48;
    jacobian(3, 6) = -x30*x49 + x38*x50;
    jacobian(4, 1) = -pre_transform_d4*x0;
    jacobian(4, 2) = -x3*x39 - x3*x41;
    jacobian(4, 3) = x31*x51 + x44*x8;
    jacobian(4, 4) = x14*x47 - x34*x52;
    jacobian(4, 5) = x18*x47 - x36*x52;
    jacobian(4, 6) = x22*x49 - x38*x53;
    jacobian(5, 3) = x24*x51 - x45*x8;
    jacobian(5, 4) = -x14*x48 + x26*x52;
    jacobian(5, 5) = -x18*x48 + x28*x52;
    jacobian(5, 6) = -x22*x50 + x30*x53;
    return;
}

static void computeAngularVelocityJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, 7>& jacobian)
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
    const double x0 = std::sin(th_0);
    const double x1 = std::sin(th_1);
    const double x2 = std::cos(th_0);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_2);
    const double x5 = x0*x4;
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_1);
    const double x8 = x2*x6*x7 - x5;
    const double x9 = std::cos(th_3);
    const double x10 = std::sin(th_3);
    const double x11 = x0*x6;
    const double x12 = x2*x4;
    const double x13 = x11 + x12*x7;
    const double x14 = x10*x13 + x3*x9;
    const double x15 = std::cos(th_4);
    const double x16 = std::sin(th_4);
    const double x17 = -x10*x3 + x13*x9;
    const double x18 = std::cos(th_5);
    const double x19 = std::sin(th_5);
    const double x20 = x0*x1;
    const double x21 = x11*x7 + x12;
    const double x22 = -x2*x6 + x5*x7;
    const double x23 = x10*x22 + x20*x9;
    const double x24 = -x10*x20 + x22*x9;
    const double x25 = x1*x6;
    const double x26 = x1*x4;
    const double x27 = -x10*x26 + x7*x9;
    const double x28 = -x10*x7 - x26*x9;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = -x0;
    jacobian(0, 2) = -x3;
    jacobian(0, 3) = x8;
    jacobian(0, 4) = x14;
    jacobian(0, 5) = x15*x8 - x16*x17;
    jacobian(0, 6) = x14*x18 + x19*(x15*x17 + x16*x8);
    jacobian(1, 1) = x2;
    jacobian(1, 2) = -x20;
    jacobian(1, 3) = x21;
    jacobian(1, 4) = x23;
    jacobian(1, 5) = x15*x21 - x16*x24;
    jacobian(1, 6) = x18*x23 + x19*(x15*x24 + x16*x21);
    jacobian(2, 0) = 1;
    jacobian(2, 2) = -x7;
    jacobian(2, 3) = -x25;
    jacobian(2, 4) = x27;
    jacobian(2, 5) = -x15*x25 - x16*x28;
    jacobian(2, 6) = x18*x27 + x19*(x15*x28 - x16*x25);
    return;
}

static void computeTransformPointJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Vector3d& point_on_ee, Eigen::Matrix<double, 6, 7>& jacobian)
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
    const double x0 = std::cos(th_0);
    const double x1 = std::cos(th_1);
    const double x2 = std::sin(th_1);
    const double x3 = std::sin(th_0);
    const double x4 = p_on_ee_z*x3;
    const double x5 = d_2*x1;
    const double x6 = x2*x3;
    const double x7 = -pre_transform_d4 + x5;
    const double x8 = -x7;
    const double x9 = std::sin(th_2);
    const double x10 = x2*x9;
    const double x11 = std::cos(th_2);
    const double x12 = x0*x11;
    const double x13 = x3*x9;
    const double x14 = x1*x13 + x12;
    const double x15 = x11*x2;
    const double x16 = a_3*x15;
    const double x17 = x16 + x7;
    const double x18 = -x17;
    const double x19 = x11*x3;
    const double x20 = -x0*x9 + x1*x19;
    const double x21 = a_3*x20 - d_2*x6;
    const double x22 = std::cos(th_3);
    const double x23 = std::sin(th_3);
    const double x24 = -x1*x22 + x15*x23;
    const double x25 = -x24;
    const double x26 = x20*x23 + x22*x6;
    const double x27 = d_4*x24;
    const double x28 = -x1*x23 - x15*x22;
    const double x29 = a_3*x28 - x17 - x27;
    const double x30 = x20*x22 - x23*x6;
    const double x31 = a_3*x30 + d_4*x26 + x21;
    const double x32 = std::cos(th_4);
    const double x33 = std::sin(th_4);
    const double x34 = -x10*x32 - x28*x33;
    const double x35 = x14*x32 - x30*x33;
    const double x36 = std::cos(th_5);
    const double x37 = std::sin(th_5);
    const double x38 = -x10*x33 + x28*x32;
    const double x39 = x25*x36 + x37*x38;
    const double x40 = x14*x33 + x30*x32;
    const double x41 = x26*x36 + x37*x40;
    const double x42 = a_3*x28 + a_5*(-x25*x37 + x36*x38) + pre_transform_d4 - x16 - x27 - x5;
    const double x43 = a_5*(-x26*x37 + x36*x40) + x31;
    const double x44 = x0*x2;
    const double x45 = x0*x1*x9 - x19;
    const double x46 = x1*x12 + x13;
    const double x47 = a_3*x46 - d_2*x44;
    const double x48 = x22*x44 + x23*x46;
    const double x49 = x22*x46 - x23*x44;
    const double x50 = a_3*x49 + d_4*x48 + x47;
    const double x51 = x32*x45 - x33*x49;
    const double x52 = x32*x49 + x33*x45;
    const double x53 = x36*x48 + x37*x52;
    const double x54 = a_5*(x36*x52 - x37*x48) + x50;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -p_on_ee_y;
    jacobian(0, 1) = p_on_ee_z*x0 - pre_transform_d4*x0;
    jacobian(0, 2) = p_on_ee_y*x1 - x2*x4 + x5*x6 + x6*x8;
    jacobian(0, 3) = p_on_ee_y*x10 + p_on_ee_z*x14 - x10*x21 - x14*x18;
    jacobian(0, 4) = -p_on_ee_y*x25 + p_on_ee_z*x26 + x25*x31 - x26*x29;
    jacobian(0, 5) = -p_on_ee_y*x34 + p_on_ee_z*x35 - x29*x35 + x31*x34;
    jacobian(0, 6) = -p_on_ee_y*x39 + p_on_ee_z*x41 + x39*x43 - x41*x42;
    jacobian(1, 0) = p_on_ee_x;
    jacobian(1, 1) = -pre_transform_d4*x3 + x4;
    jacobian(1, 2) = -p_on_ee_x*x1 + p_on_ee_z*x0*x2 - x44*x5 - x44*x8;
    jacobian(1, 3) = -p_on_ee_x*x10 - p_on_ee_z*x45 + x18*x45 + x2*x47*x9;
    jacobian(1, 4) = p_on_ee_x*x25 - p_on_ee_z*x48 - x25*x50 + x29*x48;
    jacobian(1, 5) = p_on_ee_x*x34 - p_on_ee_z*x51 + x29*x51 - x34*x50;
    jacobian(1, 6) = p_on_ee_x*x39 - p_on_ee_z*x53 - x39*x54 + x42*x53;
    jacobian(2, 1) = -p_on_ee_x*x0 - p_on_ee_y*x3;
    jacobian(2, 2) = p_on_ee_x*x6 - p_on_ee_y*x44;
    jacobian(2, 3) = -p_on_ee_x*x14 + p_on_ee_y*x45 + x14*x47 - x21*x45;
    jacobian(2, 4) = -p_on_ee_x*x26 + p_on_ee_y*x48 + x26*x50 - x31*x48;
    jacobian(2, 5) = -p_on_ee_x*x35 + p_on_ee_y*x51 - x31*x51 + x35*x50;
    jacobian(2, 6) = -p_on_ee_x*x41 + p_on_ee_y*x53 + x41*x54 - x43*x53;
    return;
}

static void computeRawIK(const Eigen::Matrix4d& T_ee, double th_3, SolutionQueue<intermediate_solution_size, max_n_solutions>& solution_queue, NodeIndexWorkspace<max_n_solutions>& node_index_workspace, std::vector<std::array<double, robot_nq>>& ik_output)
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
    
    // Code for equation all-zero dispatcher node 0
    auto EquationAllZeroDispatcherNode_node_0_processor= [&]()
    {
        const bool checked_result = 2*std::fabs(a_5*inv_Px) <= 9.9999999999999995e-7 && 2*std::fabs(a_5*inv_Py) <= 9.9999999999999995e-7 && std::fabs(2*std::pow(a_3, 2)*std::cos(th_3) + 2*std::pow(a_3, 2) + 2*a_3*d_2*std::sin(th_3) + 2*a_3*d_4*std::sin(th_3) - std::pow(a_5, 2) + std::pow(d_2, 2) - 2*d_2*d_4*std::cos(th_3) + std::pow(d_4, 2) - std::pow(inv_Px, 2) - std::pow(inv_Py, 2) - std::pow(inv_Pz, 2)) <= 9.9999999999999995e-7;
        if (!checked_result)  // To non-degenerate node
            node_index_workspace.node_input_validity_vector[1] = true;
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_0_processor();
    // Finish code for equation all-zero dispatcher node 0
    
    // Code for explicit solution node 1, solved variable is th_6
    auto ExplicitSolutionNode_node_1_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(1);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(1);
        if (!this_input_valid)
            return;
        
        // The explicit solution of root node
        const bool condition_0 = 2*std::fabs(a_5*inv_Px) >= zero_tolerance || 2*std::fabs(a_5*inv_Py) >= zero_tolerance || std::fabs(2*std::pow(a_3, 2)*std::cos(th_3) + 2*std::pow(a_3, 2) + 2*a_3*d_2*std::sin(th_3) + 2*a_3*d_4*std::sin(th_3) - std::pow(a_5, 2) + std::pow(d_2, 2) - 2*d_2*d_4*std::cos(th_3) + std::pow(d_4, 2) - std::pow(inv_Px, 2) - std::pow(inv_Py, 2) - std::pow(inv_Pz, 2)) >= zero_tolerance;
        if (condition_0)
        {
            // Temp variable for efficiency
            const double x0 = 2*a_5;
            const double x1 = std::atan2(inv_Py*x0, -inv_Px*x0);
            const double x2 = std::pow(inv_Px, 2);
            const double x3 = std::pow(a_5, 2);
            const double x4 = 4*x3;
            const double x5 = std::pow(inv_Py, 2);
            const double x6 = 2*std::pow(a_3, 2);
            const double x7 = a_3*std::sin(th_3);
            const double x8 = std::cos(th_3);
            const double x9 = -std::pow(d_2, 2) + 2*d_2*d_4*x8 - 2*d_2*x7 - std::pow(d_4, 2) - 2*d_4*x7 + std::pow(inv_Pz, 2) + x2 + x3 + x5 - x6*x8 - x6;
            const double x10 = std::sqrt(x2*x4 + x4*x5 - std::pow(x9, 2));
            // End of temp variables
            
            auto solution_0 = make_raw_solution();
            solution_0[5] = x1 + std::atan2(x10, x9);
            int appended_idx = append_solution_to_queue(solution_0);
            add_input_index_to(2, appended_idx);
        }
        
        const bool condition_1 = 2*std::fabs(a_5*inv_Px) >= zero_tolerance || 2*std::fabs(a_5*inv_Py) >= zero_tolerance || std::fabs(2*std::pow(a_3, 2)*std::cos(th_3) + 2*std::pow(a_3, 2) + 2*a_3*d_2*std::sin(th_3) + 2*a_3*d_4*std::sin(th_3) - std::pow(a_5, 2) + std::pow(d_2, 2) - 2*d_2*d_4*std::cos(th_3) + std::pow(d_4, 2) - std::pow(inv_Px, 2) - std::pow(inv_Py, 2) - std::pow(inv_Pz, 2)) >= zero_tolerance;
        if (condition_1)
        {
            // Temp variable for efficiency
            const double x0 = 2*a_5;
            const double x1 = std::atan2(inv_Py*x0, -inv_Px*x0);
            const double x2 = std::pow(inv_Px, 2);
            const double x3 = std::pow(a_5, 2);
            const double x4 = 4*x3;
            const double x5 = std::pow(inv_Py, 2);
            const double x6 = 2*std::pow(a_3, 2);
            const double x7 = a_3*std::sin(th_3);
            const double x8 = std::cos(th_3);
            const double x9 = -std::pow(d_2, 2) + 2*d_2*d_4*x8 - 2*d_2*x7 - std::pow(d_4, 2) - 2*d_4*x7 + std::pow(inv_Pz, 2) + x2 + x3 + x5 - x6*x8 - x6;
            const double x10 = std::sqrt(x2*x4 + x4*x5 - std::pow(x9, 2));
            // End of temp variables
            
            auto solution_1 = make_raw_solution();
            solution_1[5] = x1 + std::atan2(-x10, x9);
            int appended_idx = append_solution_to_queue(solution_1);
            add_input_index_to(2, appended_idx);
        }
        
    };
    // Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_6_processor();
    // Finish code for explicit solution node 1
    
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
            
            const bool degenerate_valid_0 = std::fabs(th_3 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(12, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_3 - 3.0893893222715398 + M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(13, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(3, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_2_processor();
    // Finish code for solved_variable dispatcher node 2
    
    // Code for explicit solution node 13, solved variable is th_4
    auto ExplicitSolutionNode_node_13_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(13);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(13);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 13
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_6 = this_solution[5];
            
            const bool condition_0 = std::fabs((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(1.99863771551522*a_3 - 0.0521796239019005*d_2)) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::asin((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(1.99863771551522*a_3 - 0.0521796239019005*d_2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[3] = x0;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(14, appended_idx);
            }
            
            const bool condition_1 = std::fabs((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(1.99863771551522*a_3 - 0.0521796239019005*d_2)) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::asin((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(1.99863771551522*a_3 - 0.0521796239019005*d_2));
                // End of temp variables
                const double tmp_sol_value = M_PI - x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(14, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_4_processor();
    // Finish code for explicit solution node 13
    
    // Code for non-branch dispatcher node 14
    // Actually, there is no code
    
    // Code for explicit solution node 15, solved variable is th_5
    auto ExplicitSolutionNode_node_15_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(14);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(14);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 15
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_4 = this_solution[3];
            const double th_6 = this_solution[5];
            
            const bool condition_0 = std::fabs(1.99863771551522*a_3*std::cos(th_4) - 0.0521796239019005*d_2*std::cos(th_4)) >= 9.9999999999999995e-7 || std::fabs(0.0521796239019005*a_3 + 0.99863771551521896*d_2 - 1.0*d_4) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_4);
                const double x1 = -1.99863771551522*a_3*x0 + 0.0521796239019005*d_2*x0;
                const double x2 = 0.0521796239019005*a_3 + 0.99863771551521896*d_2 - 1.0*d_4;
                const double x3 = 1.0*a_5 + 1.0*inv_Px*std::cos(th_6) - 1.0*inv_Py*std::sin(th_6);
                // End of temp variables
                const double tmp_sol_value = std::atan2(inv_Pz*x1 - x2*x3, inv_Pz*x2 + x1*x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_15_solve_th_5_processor();
    // Finish code for explicit solution node 14
    
    // Code for explicit solution node 12, solved variable is th_5
    auto ExplicitSolutionNode_node_12_solve_th_5_processor = [&]() -> void
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
            const double th_6 = this_solution[5];
            
            const bool condition_0 = std::fabs(inv_Pz) >= zero_tolerance || std::fabs(d_2 + d_4) >= zero_tolerance || std::fabs(a_5 + inv_Px*std::cos(th_6) - inv_Py*std::sin(th_6)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = d_2 + d_4;
                // End of temp variables
                const double tmp_sol_value = std::atan2((a_5 + inv_Px*std::cos(th_6) - inv_Py*std::sin(th_6))/x0, -inv_Pz/x0);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_12_solve_th_5_processor();
    // Finish code for explicit solution node 12
    
    // Code for explicit solution node 3, solved variable is th_4
    auto ExplicitSolutionNode_node_3_solve_th_4_processor = [&]() -> void
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
            const double th_6 = this_solution[5];
            
            const bool condition_0 = std::fabs((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(a_3*std::cos(th_3) + a_3 + d_2*std::sin(th_3))) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::asin((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(a_3*std::cos(th_3) + a_3 + d_2*std::sin(th_3)));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[3] = x0;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = std::fabs((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(a_3*std::cos(th_3) + a_3 + d_2*std::sin(th_3))) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::asin((inv_Px*std::sin(th_6) + inv_Py*std::cos(th_6))/(a_3*std::cos(th_3) + a_3 + d_2*std::sin(th_3)));
                // End of temp variables
                const double tmp_sol_value = M_PI - x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(4, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_4_processor();
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
            const double th_6 = this_solution[5];
            const bool checked_result = std::fabs(inv_Pz) <= 9.9999999999999995e-7 && std::fabs(a_5 + inv_Px*std::cos(th_6) - inv_Py*std::sin(th_6)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor();
    // Finish code for equation all-zero dispatcher node 4
    
    // Code for explicit solution node 5, solved variable is th_5
    auto ExplicitSolutionNode_node_5_solve_th_5_processor = [&]() -> void
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
            const double th_4 = this_solution[3];
            const double th_6 = this_solution[5];
            
            const bool condition_0 = std::fabs(inv_Pz) >= 9.9999999999999995e-7 || std::fabs(a_5 + inv_Px*std::cos(th_6) - inv_Py*std::sin(th_6)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_3);
                const double x2 = a_3*x0 + a_3 + d_2*x1;
                const double x3 = std::cos(th_4);
                const double x4 = a_3*x1 - d_2*x0 + d_4;
                const double x5 = -a_5 - inv_Px*std::cos(th_6) + inv_Py*std::sin(th_6);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-inv_Pz*x2*x3 - x4*x5, -inv_Pz*x4 + x2*x3*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(6, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_5_processor();
    // Finish code for explicit solution node 5
    
    // Code for non-branch dispatcher node 6
    // Actually, there is no code
    
    // Code for linear type2 solution node 7, solved variable is th_2
    auto LinearSinCosType_2_SolverNode_node_7_solve_th_2_processor = [&]() -> void
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
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const double th_6 = this_solution[5];
            // Temp variable for efficiency
            const double x0 = std::cos(th_6);
            const double x1 = std::sin(th_3);
            const double x2 = std::sin(th_5);
            const double x3 = x1*x2;
            const double x4 = std::sin(th_6);
            const double x5 = std::sin(th_4);
            const double x6 = std::cos(th_3);
            const double x7 = x5*x6;
            const double x8 = std::cos(th_4);
            const double x9 = std::cos(th_5);
            const double x10 = x6*x9;
            const double x11 = -x0*x10*x8 + x0*x3 + x4*x7;
            const double x12 = x4*x8;
            const double x13 = x5*x9;
            const double x14 = x0*x13 + x12;
            const double x15 = x0*x7 + x10*x12 - x3*x4;
            const double x16 = -x0*x8 + x13*x4;
            const double x17 = x6*x8;
            const double x18 = x1*x9 + x17*x2;
            const double x19 = x2*x5;
            const double x20 = a_3*x6 + a_3 - a_5*x1*x2 + a_5*x17*x9 + d_4*x1;
            const double x21 = a_5*x13;
            // End of temp variables
            Eigen::Matrix<double, 8, 4> A_matrix;
            A_matrix.setZero();
            A_matrix(0, 0) = r_11;
            A_matrix(0, 1) = -r_21;
            A_matrix(0, 2) = x11;
            A_matrix(0, 3) = x14;
            A_matrix(1, 0) = r_12;
            A_matrix(1, 1) = -r_22;
            A_matrix(1, 2) = x15;
            A_matrix(1, 3) = -x16;
            A_matrix(2, 0) = r_13;
            A_matrix(2, 1) = -r_23;
            A_matrix(2, 2) = -x18;
            A_matrix(2, 3) = x19;
            A_matrix(3, 0) = Px;
            A_matrix(3, 1) = -Py;
            A_matrix(3, 2) = -x20;
            A_matrix(3, 3) = x21;
            A_matrix(4, 0) = -r_11;
            A_matrix(4, 1) = r_21;
            A_matrix(4, 2) = -x11;
            A_matrix(4, 3) = -x14;
            A_matrix(5, 0) = -r_12;
            A_matrix(5, 1) = r_22;
            A_matrix(5, 2) = -x15;
            A_matrix(5, 3) = x16;
            A_matrix(6, 0) = -r_13;
            A_matrix(6, 1) = r_23;
            A_matrix(6, 2) = x18;
            A_matrix(6, 3) = -x19;
            A_matrix(7, 0) = -Px;
            A_matrix(7, 1) = Py;
            A_matrix(7, 2) = x20;
            A_matrix(7, 3) = -x21;
            // Code for solving
            {
                // Local variable for solving
                double var_solution_0 = 0.0;
                double var_solution_1 = 0.0;
                bool solved = ::yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<8>(A_matrix, 0, 1, 2, var_solution_0, var_solution_1);
                if(solved)
                {
                    RawSolution solution_0(this_solution);
                    solution_0[2] = var_solution_0;
                    int appended_idx = append_solution_to_queue(solution_0);
                    add_input_index_to(8, appended_idx);
                    solution_queue.get_solution(node_input_i_idx_in_queue)[2] = var_solution_1;
                    add_input_index_to(8, node_input_i_idx_in_queue);
                    continue;
                }
            }
            {
                // Local variable for solving
                double var_solution_0 = 0.0;
                double var_solution_1 = 0.0;
                bool solved = ::yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<8>(A_matrix, 0, 1, 3, var_solution_0, var_solution_1);
                if(solved)
                {
                    RawSolution solution_0(this_solution);
                    solution_0[2] = var_solution_0;
                    int appended_idx = append_solution_to_queue(solution_0);
                    add_input_index_to(8, appended_idx);
                    solution_queue.get_solution(node_input_i_idx_in_queue)[2] = var_solution_1;
                    add_input_index_to(8, node_input_i_idx_in_queue);
                    continue;
                }
            }
            {
                // Local variable for solving
                double var_solution_0 = 0.0;
                double var_solution_1 = 0.0;
                bool solved = ::yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<8>(A_matrix, 0, 1, 6, var_solution_0, var_solution_1);
                if(solved)
                {
                    RawSolution solution_0(this_solution);
                    solution_0[2] = var_solution_0;
                    int appended_idx = append_solution_to_queue(solution_0);
                    add_input_index_to(8, appended_idx);
                    solution_queue.get_solution(node_input_i_idx_in_queue)[2] = var_solution_1;
                    add_input_index_to(8, node_input_i_idx_in_queue);
                    continue;
                }
            }
            {
                // Local variable for solving
                double var_solution_0 = 0.0;
                double var_solution_1 = 0.0;
                bool solved = ::yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<8>(A_matrix, 0, 1, 7, var_solution_0, var_solution_1);
                if(solved)
                {
                    RawSolution solution_0(this_solution);
                    solution_0[2] = var_solution_0;
                    int appended_idx = append_solution_to_queue(solution_0);
                    add_input_index_to(8, appended_idx);
                    solution_queue.get_solution(node_input_i_idx_in_queue)[2] = var_solution_1;
                    add_input_index_to(8, node_input_i_idx_in_queue);
                    continue;
                }
            }
            {
                // Local variable for solving
                double var_solution_0 = 0.0;
                double var_solution_1 = 0.0;
                bool solved = ::yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<8>(A_matrix, 0, 2, 3, var_solution_0, var_solution_1);
                if(solved)
                {
                    RawSolution solution_0(this_solution);
                    solution_0[2] = var_solution_0;
                    int appended_idx = append_solution_to_queue(solution_0);
                    add_input_index_to(8, appended_idx);
                    solution_queue.get_solution(node_input_i_idx_in_queue)[2] = var_solution_1;
                    add_input_index_to(8, node_input_i_idx_in_queue);
                    continue;
                }
            }
        }
    };
    // Invoke the processor
    LinearSinCosType_2_SolverNode_node_7_solve_th_2_processor();
    // Finish code for general_6dof solution node 6
    
    // Code for equation all-zero dispatcher node 8
    auto EquationAllZeroDispatcherNode_node_8_processor = [&]() -> void
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
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            const bool checked_result = std::fabs(r_13) <= 9.9999999999999995e-7 && std::fabs(r_23) <= 9.9999999999999995e-7 && std::fabs(std::sin(th_2)*std::sin(th_3)*std::cos(th_5) + std::sin(th_2)*std::sin(th_5)*std::cos(th_3)*std::cos(th_4) - std::sin(th_4)*std::sin(th_5)*std::cos(th_2)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor();
    // Finish code for equation all-zero dispatcher node 8
    
    // Code for explicit solution node 9, solved variable is th_0
    auto ExplicitSolutionNode_node_9_solve_th_0_processor = [&]() -> void
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
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            
            const bool condition_0 = std::fabs(r_13) >= zero_tolerance || std::fabs(r_23) >= zero_tolerance || std::fabs(std::sin(th_2)*std::sin(th_3)*std::cos(th_5) + std::sin(th_2)*std::sin(th_5)*std::cos(th_3)*std::cos(th_4) - std::sin(th_4)*std::sin(th_5)*std::cos(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::atan2(r_13, -r_23);
                const double x1 = std::sin(th_2);
                const double x2 = std::sin(th_5);
                const double x3 = x1*x2*std::cos(th_3)*std::cos(th_4) + x1*std::sin(th_3)*std::cos(th_5) - x2*std::sin(th_4)*std::cos(th_2);
                const double x4 = std::sqrt(std::pow(r_13, 2) + std::pow(r_23, 2) - std::pow(x3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[0] = x0 + std::atan2(x4, x3);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(10, appended_idx);
            }
            
            const bool condition_1 = std::fabs(r_13) >= zero_tolerance || std::fabs(r_23) >= zero_tolerance || std::fabs(std::sin(th_2)*std::sin(th_3)*std::cos(th_5) + std::sin(th_2)*std::sin(th_5)*std::cos(th_3)*std::cos(th_4) - std::sin(th_4)*std::sin(th_5)*std::cos(th_2)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::atan2(r_13, -r_23);
                const double x1 = std::sin(th_2);
                const double x2 = std::sin(th_5);
                const double x3 = x1*x2*std::cos(th_3)*std::cos(th_4) + x1*std::sin(th_3)*std::cos(th_5) - x2*std::sin(th_4)*std::cos(th_2);
                const double x4 = std::sqrt(std::pow(r_13, 2) + std::pow(r_23, 2) - std::pow(x3, 2));
                // End of temp variables
                const double tmp_sol_value = x0 + std::atan2(-x4, x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(10, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_0_processor();
    // Finish code for explicit solution node 9
    
    // Code for equation all-zero dispatcher node 10
    auto EquationAllZeroDispatcherNode_node_10_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(10);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(10);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const bool checked_result = std::fabs(Pz) <= 9.9999999999999995e-7 && std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(11, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_10_processor();
    // Finish code for equation all-zero dispatcher node 10
    
    // Code for explicit solution node 11, solved variable is th_1
    auto ExplicitSolutionNode_node_11_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(11);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(11);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 11
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_2 = this_solution[2];
            const double th_4 = this_solution[3];
            const double th_5 = this_solution[4];
            
            const bool condition_0 = std::fabs(Pz) >= 9.9999999999999995e-7 || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = -Px*std::cos(th_0) - Py*std::sin(th_0);
                const double x1 = std::sin(th_3);
                const double x2 = std::cos(th_3);
                const double x3 = std::sin(th_5);
                const double x4 = a_5*std::cos(th_5);
                const double x5 = x4*std::cos(th_4);
                const double x6 = -a_3*x1 - a_5*x2*x3 - d_2 + d_4*x2 - x1*x5;
                const double x7 = std::cos(th_2);
                const double x8 = a_3*x7;
                const double x9 = a_5*x1*x3*x7 - d_4*x1*x7 - x2*x5*x7 - x2*x8 - x4*std::sin(th_2)*std::sin(th_4) - x8;
                // End of temp variables
                const double tmp_sol_value = std::atan2(Pz*x9 - x0*x6, Pz*x6 + x0*x9);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1_processor();
    // Finish code for explicit solution node 11
    
    // Collect the output
    for(int i = 0; i < solution_queue.size(); i++)
    {
        if(!solution_queue.solutions_validity[i])
            continue;
        const auto& raw_ik_out_i = solution_queue.get_solution(i);
        std::array<double, robot_nq> new_ik_i;
        const double value_at_0 = raw_ik_out_i[0];  // th_0
        new_ik_i[0] = value_at_0;
        const double value_at_1 = raw_ik_out_i[1];  // th_1
        new_ik_i[1] = value_at_1;
        const double value_at_2 = raw_ik_out_i[2];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = th_3;  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[3];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[4];  // th_5
        new_ik_i[5] = value_at_5;
        const double value_at_6 = raw_ik_out_i[5];  // th_6
        new_ik_i[6] = value_at_6;
        ik_output.push_back(new_ik_i);
    }
}

static void computeRawIK(const Eigen::Matrix4d& T_ee_raw, double th_3, RawIKWorksace& workspace)
{
    workspace.raw_ik_out.clear();
    workspace.raw_ik_out.reserve(max_n_solutions);
    computeRawIK(T_ee_raw, th_3, workspace.solution_queue, workspace.node_index_workspace, workspace.raw_ik_out);
}

static void computeIKUnChecked(const Eigen::Matrix4d& T_ee, double th_3, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, th_3 + th_3_offset_original2raw, workspace);
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

static void computeIK(const Eigen::Matrix4d& T_ee, double th_3, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, th_3 + th_3_offset_original2raw, workspace);
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
    if (!ik_output.empty()) return;
    
    // Disturbing method for degenerate handling
    Eigen::Matrix4d T_ee_raw_disturbed = yaik_cpp::disturbTransform(T_ee_raw);
    Eigen::Matrix4d T_ee_disturbed = endEffectorTargetRawToOriginal(T_ee_raw_disturbed);
    computeRawIK(T_ee_raw_disturbed, th_3 + th_3_offset_original2raw, workspace);
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
}

static std::vector<std::array<double, robot_nq>> computeIK(const Eigen::Matrix4d& T_ee, double th_3)
{
    std::vector<std::array<double, robot_nq>> ik_output;
    RawIKWorksace raw_ik_workspace;
    computeIK(T_ee, th_3, raw_ik_workspace, ik_output);
    return ik_output;
}

}; // struct franka_panda_ik

// Code below for debug
void test_ik_solve_franka_panda()
{
    std::array<double, franka_panda_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = franka_panda_ik::computeFK(theta);
    auto ik_output = franka_panda_ik::computeIK(ee_pose, theta[3]);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = franka_panda_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_franka_panda();
}
