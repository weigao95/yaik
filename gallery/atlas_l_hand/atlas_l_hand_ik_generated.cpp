#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct atlas_l_hand_ik {

// Constants for solver
static constexpr int robot_nq = 7;
static constexpr int max_n_solutions = 128;
static constexpr int n_tree_nodes = 52;
static constexpr int intermediate_solution_size = 12;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_1 = 0.11;
static constexpr double a_2 = 0.016;
static constexpr double a_3 = 0.0092;
static constexpr double a_4 = 0.00921;
static constexpr double d_2 = 0.306;
static constexpr double d_4 = 0.29955;
static constexpr double pre_transform_s0 = 0.1406;
static constexpr double pre_transform_s1 = 0.2256;
static constexpr double pre_transform_s2 = 0.2326;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -1.5707963267948966;
static constexpr double th_2_offset_original2raw = 3.141592653589793;
static constexpr double th_3_offset_original2raw = 3.141592653589793;
static constexpr double th_4_offset_original2raw = 3.141592653589793;
static constexpr double th_5_offset_original2raw = 3.141592653589793;
static constexpr double th_6_offset_original2raw = -1.5707963267948966;

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
    ee_transformed(0, 0) = -r_21;
    ee_transformed(0, 1) = -r_23;
    ee_transformed(0, 2) = -r_22;
    ee_transformed(0, 3) = Py - pre_transform_s1;
    ee_transformed(1, 0) = r_11;
    ee_transformed(1, 1) = r_13;
    ee_transformed(1, 2) = r_12;
    ee_transformed(1, 3) = -Px + pre_transform_s0;
    ee_transformed(2, 0) = -r_31;
    ee_transformed(2, 1) = -r_33;
    ee_transformed(2, 2) = -r_32;
    ee_transformed(2, 3) = Pz - pre_transform_s2;
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
    ee_transformed(0, 1) = r_23;
    ee_transformed(0, 2) = r_22;
    ee_transformed(0, 3) = -Py + pre_transform_s0;
    ee_transformed(1, 0) = -r_11;
    ee_transformed(1, 1) = -r_13;
    ee_transformed(1, 2) = -r_12;
    ee_transformed(1, 3) = Px + pre_transform_s1;
    ee_transformed(2, 0) = -r_31;
    ee_transformed(2, 1) = -r_33;
    ee_transformed(2, 2) = -r_32;
    ee_transformed(2, 3) = Pz + pre_transform_s2;
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
    const double x9 = -x4 - x5*x8;
    const double x10 = std::sin(th_4);
    const double x11 = std::sin(th_3);
    const double x12 = std::sin(th_1);
    const double x13 = x12*x7;
    const double x14 = std::cos(th_3);
    const double x15 = x2*x6;
    const double x16 = -x15 + x3*x5*x7;
    const double x17 = x11*x13 + x14*x16;
    const double x18 = -x1*x9 - x10*x17;
    const double x19 = std::cos(th_6);
    const double x20 = std::sin(th_5);
    const double x21 = -x11*x16 + x12*x14*x7;
    const double x22 = std::cos(th_5);
    const double x23 = x1*x17 - x10*x9;
    const double x24 = -x20*x21 + x22*x23;
    const double x25 = a_2*x5;
    const double x26 = -x15*x5 + x3*x7;
    const double x27 = x12*x2;
    const double x28 = x4*x5 + x8;
    const double x29 = x11*x27 + x14*x28;
    const double x30 = -x1*x26 - x10*x29;
    const double x31 = -x11*x28 + x12*x14*x2;
    const double x32 = x1*x29 - x10*x26;
    const double x33 = -x20*x31 + x22*x32;
    const double x34 = x12*x3;
    const double x35 = -x11*x5 + x14*x34;
    const double x36 = x1*x12*x6 - x10*x35;
    const double x37 = -x11*x34 - x14*x5;
    const double x38 = x1*x35 + x10*x12*x6;
    const double x39 = -x20*x37 + x22*x38;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x18 + x19*x24;
    ee_pose_raw(0, 1) = -x0*x24 - x18*x19;
    ee_pose_raw(0, 2) = -x20*x23 - x21*x22;
    ee_pose_raw(0, 3) = a_1*x7 + a_3*x16 + a_4*x17 - d_2*x13 + d_4*x21 + x25*x7;
    ee_pose_raw(1, 0) = -x0*x30 + x19*x33;
    ee_pose_raw(1, 1) = -x0*x33 - x19*x30;
    ee_pose_raw(1, 2) = -x20*x32 - x22*x31;
    ee_pose_raw(1, 3) = a_1*x2 + a_3*x28 + a_4*x29 - d_2*x27 + d_4*x31 + x2*x25;
    ee_pose_raw(2, 0) = -x0*x36 + x19*x39;
    ee_pose_raw(2, 1) = -x0*x39 - x19*x36;
    ee_pose_raw(2, 2) = -x20*x38 - x22*x37;
    ee_pose_raw(2, 3) = a_2*x12 + a_3*x34 + a_4*x35 + d_2*x5 + d_4*x37;
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
    const double x0 = std::cos(th_0);
    const double x1 = std::sin(th_0);
    const double x2 = std::sin(th_1);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_2);
    const double x5 = std::cos(th_1);
    const double x6 = std::sin(th_2);
    const double x7 = x1*x6;
    const double x8 = -x0*x4 + x5*x7;
    const double x9 = std::cos(th_3);
    const double x10 = std::sin(th_3);
    const double x11 = x0*x6;
    const double x12 = x1*x4;
    const double x13 = -x11 - x12*x5;
    const double x14 = -x10*x13 - x3*x9;
    const double x15 = std::cos(th_4);
    const double x16 = std::sin(th_4);
    const double x17 = -x10*x3 + x13*x9;
    const double x18 = -x15*x8 - x16*x17;
    const double x19 = std::cos(th_5);
    const double x20 = std::sin(th_5);
    const double x21 = -x14*x19 - x20*(x15*x17 - x16*x8);
    const double x22 = x0*x2;
    const double x23 = -x11*x5 - x12;
    const double x24 = x0*x4*x5 - x7;
    const double x25 = x0*x2*x9 - x10*x24;
    const double x26 = x10*x22 + x24*x9;
    const double x27 = -x15*x23 - x16*x26;
    const double x28 = -x19*x25 - x20*(x15*x26 - x16*x23);
    const double x29 = x2*x6;
    const double x30 = x2*x4;
    const double x31 = -x10*x30 - x5*x9;
    const double x32 = -x10*x5 + x30*x9;
    const double x33 = x15*x2*x6 - x16*x32;
    const double x34 = -x19*x31 - x20*(x15*x32 + x16*x29);
    const double x35 = a_2*x2 + d_2*x5 + pre_transform_s2;
    const double x36 = a_2*x5;
    const double x37 = a_1*x0 + pre_transform_s1;
    const double x38 = -d_2*x22 + x0*x36 + x37;
    const double x39 = a_3*x30 + x35;
    const double x40 = a_3*x24 + x38;
    const double x41 = a_4*x32 + d_4*x31 + x39;
    const double x42 = a_4*x26 + d_4*x25 + x40;
    const double x43 = -pre_transform_s0;
    const double x44 = x1*x36;
    const double x45 = a_1*x1;
    const double x46 = x43 + x45;
    const double x47 = d_2*x1*x2 - x44 - x46;
    const double x48 = a_3*x13 + d_2*x3 + pre_transform_s0 - x44 - x45;
    const double x49 = a_4*x17 + d_4*x14 + x48;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x0;
    jacobian(0, 2) = x3;
    jacobian(0, 3) = x8;
    jacobian(0, 4) = x14;
    jacobian(0, 5) = x18;
    jacobian(0, 6) = x21;
    jacobian(1, 1) = x1;
    jacobian(1, 2) = -x22;
    jacobian(1, 3) = x23;
    jacobian(1, 4) = x25;
    jacobian(1, 5) = x27;
    jacobian(1, 6) = x28;
    jacobian(2, 0) = 1;
    jacobian(2, 2) = x5;
    jacobian(2, 3) = -x29;
    jacobian(2, 4) = x31;
    jacobian(2, 5) = x33;
    jacobian(2, 6) = x34;
    jacobian(3, 0) = pre_transform_s1;
    jacobian(3, 1) = -pre_transform_s2*x1;
    jacobian(3, 2) = x22*x35 + x38*x5;
    jacobian(3, 3) = -x23*x39 - x29*x40;
    jacobian(3, 4) = -x25*x41 + x31*x42;
    jacobian(3, 5) = -x27*x41 + x33*x42;
    jacobian(3, 6) = -x28*x41 + x34*x42;
    jacobian(4, 0) = x43;
    jacobian(4, 1) = pre_transform_s2*x0;
    jacobian(4, 2) = x3*x35 - x47*x5;
    jacobian(4, 3) = x29*x48 + x39*x8;
    jacobian(4, 4) = x14*x41 - x31*x49;
    jacobian(4, 5) = x18*x41 - x33*x49;
    jacobian(4, 6) = x21*x41 - x34*x49;
    jacobian(5, 1) = -x0*x37 - x1*x46;
    jacobian(5, 2) = -x22*x47 - x3*x38;
    jacobian(5, 3) = x23*x48 - x40*x8;
    jacobian(5, 4) = -x14*x42 + x25*x49;
    jacobian(5, 5) = -x18*x42 + x27*x49;
    jacobian(5, 6) = -x21*x42 + x28*x49;
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
    const double x0 = std::cos(th_0);
    const double x1 = std::sin(th_0);
    const double x2 = std::sin(th_1);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_2);
    const double x5 = std::cos(th_1);
    const double x6 = std::sin(th_2);
    const double x7 = x1*x6;
    const double x8 = -x0*x4 + x5*x7;
    const double x9 = std::cos(th_3);
    const double x10 = std::sin(th_3);
    const double x11 = x0*x6;
    const double x12 = x1*x4;
    const double x13 = -x11 - x12*x5;
    const double x14 = -x10*x13 - x3*x9;
    const double x15 = std::cos(th_4);
    const double x16 = std::sin(th_4);
    const double x17 = -x10*x3 + x13*x9;
    const double x18 = std::cos(th_5);
    const double x19 = std::sin(th_5);
    const double x20 = x0*x2;
    const double x21 = -x11*x5 - x12;
    const double x22 = x0*x4*x5 - x7;
    const double x23 = x0*x2*x9 - x10*x22;
    const double x24 = x10*x20 + x22*x9;
    const double x25 = x2*x6;
    const double x26 = x2*x4;
    const double x27 = -x10*x26 - x5*x9;
    const double x28 = -x10*x5 + x26*x9;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x0;
    jacobian(0, 2) = x3;
    jacobian(0, 3) = x8;
    jacobian(0, 4) = x14;
    jacobian(0, 5) = -x15*x8 - x16*x17;
    jacobian(0, 6) = -x14*x18 - x19*(x15*x17 - x16*x8);
    jacobian(1, 1) = x1;
    jacobian(1, 2) = -x20;
    jacobian(1, 3) = x21;
    jacobian(1, 4) = x23;
    jacobian(1, 5) = -x15*x21 - x16*x24;
    jacobian(1, 6) = -x18*x23 - x19*(x15*x24 - x16*x21);
    jacobian(2, 0) = 1;
    jacobian(2, 2) = x5;
    jacobian(2, 3) = -x25;
    jacobian(2, 4) = x27;
    jacobian(2, 5) = x15*x2*x6 - x16*x28;
    jacobian(2, 6) = -x18*x27 - x19*(x15*x28 + x16*x25);
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
    const double x0 = std::sin(th_0);
    const double x1 = p_on_ee_z*x0;
    const double x2 = std::cos(th_1);
    const double x3 = std::sin(th_1);
    const double x4 = std::cos(th_0);
    const double x5 = p_on_ee_z*x4;
    const double x6 = a_2*x3 + d_2*x2 + pre_transform_s2;
    const double x7 = a_2*x2;
    const double x8 = x3*x4;
    const double x9 = a_1*x4 + pre_transform_s1;
    const double x10 = -d_2*x8 + x4*x7 + x9;
    const double x11 = std::sin(th_2);
    const double x12 = x11*x3;
    const double x13 = std::cos(th_2);
    const double x14 = x0*x13;
    const double x15 = x11*x4;
    const double x16 = -x14 - x15*x2;
    const double x17 = x13*x3;
    const double x18 = a_3*x17 + x6;
    const double x19 = x0*x11;
    const double x20 = x13*x2*x4 - x19;
    const double x21 = a_3*x20 + x10;
    const double x22 = std::cos(th_3);
    const double x23 = std::sin(th_3);
    const double x24 = -x17*x23 - x2*x22;
    const double x25 = -x20*x23 + x22*x3*x4;
    const double x26 = x17*x22 - x2*x23;
    const double x27 = a_4*x26 + d_4*x24 + x18;
    const double x28 = x20*x22 + x23*x8;
    const double x29 = a_4*x28 + d_4*x25 + x21;
    const double x30 = std::cos(th_4);
    const double x31 = std::sin(th_4);
    const double x32 = x11*x3*x30 - x26*x31;
    const double x33 = -x16*x30 - x28*x31;
    const double x34 = std::cos(th_5);
    const double x35 = std::sin(th_5);
    const double x36 = -x24*x34 - x35*(x12*x31 + x26*x30);
    const double x37 = -x25*x34 - x35*(-x16*x31 + x28*x30);
    const double x38 = -pre_transform_s0;
    const double x39 = x0*x3;
    const double x40 = x0*x7;
    const double x41 = a_1*x0;
    const double x42 = x38 + x41;
    const double x43 = d_2*x0*x3 - x40 - x42;
    const double x44 = -x13*x4 + x19*x2;
    const double x45 = -x14*x2 - x15;
    const double x46 = a_3*x45 + d_2*x39 + pre_transform_s0 - x40 - x41;
    const double x47 = -x22*x39 - x23*x45;
    const double x48 = x22*x45 - x23*x39;
    const double x49 = a_4*x48 + d_4*x47 + x46;
    const double x50 = -x30*x44 - x31*x48;
    const double x51 = -x34*x47 - x35*(x30*x48 - x31*x44);
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -p_on_ee_y + pre_transform_s1;
    jacobian(0, 1) = -pre_transform_s2*x0 + x1;
    jacobian(0, 2) = -p_on_ee_y*x2 + x10*x2 + x3*x4*x6 - x3*x5;
    jacobian(0, 3) = p_on_ee_y*x12 + p_on_ee_z*x16 - x12*x21 - x16*x18;
    jacobian(0, 4) = -p_on_ee_y*x24 + p_on_ee_z*x25 + x24*x29 - x25*x27;
    jacobian(0, 5) = -p_on_ee_y*x32 + p_on_ee_z*x33 - x27*x33 + x29*x32;
    jacobian(0, 6) = -p_on_ee_y*x36 + p_on_ee_z*x37 - x27*x37 + x29*x36;
    jacobian(1, 0) = p_on_ee_x + x38;
    jacobian(1, 1) = pre_transform_s2*x4 - x5;
    jacobian(1, 2) = p_on_ee_x*x2 - x1*x3 - x2*x43 + x39*x6;
    jacobian(1, 3) = -p_on_ee_x*x12 - p_on_ee_z*x44 + x11*x3*x46 + x18*x44;
    jacobian(1, 4) = p_on_ee_x*x24 - p_on_ee_z*x47 - x24*x49 + x27*x47;
    jacobian(1, 5) = p_on_ee_x*x32 - p_on_ee_z*x50 + x27*x50 - x32*x49;
    jacobian(1, 6) = p_on_ee_x*x36 - p_on_ee_z*x51 + x27*x51 - x36*x49;
    jacobian(2, 1) = -p_on_ee_x*x0 + p_on_ee_y*x4 - x0*x42 - x4*x9;
    jacobian(2, 2) = p_on_ee_x*x8 + p_on_ee_y*x39 - x10*x39 - x43*x8;
    jacobian(2, 3) = -p_on_ee_x*x16 + p_on_ee_y*x44 + x16*x46 - x21*x44;
    jacobian(2, 4) = -p_on_ee_x*x25 + p_on_ee_y*x47 + x25*x49 - x29*x47;
    jacobian(2, 5) = -p_on_ee_x*x33 + p_on_ee_y*x50 - x29*x50 + x33*x49;
    jacobian(2, 6) = -p_on_ee_x*x37 + p_on_ee_y*x51 - x29*x51 + x37*x49;
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
    
    // Code for general_6dof solution node 1, solved variable is th_1
    auto General6DoFNumericalReduceSolutionNode_node_1_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;
        
        // The general 6-dof solution of root node
        Eigen::Matrix<double, 8, 8> R_l;
        R_l.setZero();
        R_l(0, 1) = d_2;
        R_l(0, 3) = -a_3;
        R_l(0, 7) = -a_4;
        R_l(1, 0) = d_2;
        R_l(1, 2) = -a_3;
        R_l(1, 6) = -a_4;
        R_l(2, 4) = a_3;
        R_l(2, 5) = d_2;
        R_l(3, 1) = -1;
        R_l(4, 0) = -1;
        R_l(5, 5) = -1;
        R_l(6, 2) = a_4;
        R_l(6, 6) = a_3;
        R_l(7, 3) = -a_4;
        R_l(7, 7) = -a_3;
        Eigen::Matrix<double, 8, 8> R_l_mat_inv = R_l.inverse();
        for(auto r = 0; r < R_l_mat_inv.rows(); r++) {
            for(auto c = 0; c < R_l_mat_inv.cols(); c++) {
                if (std::isnan(R_l_mat_inv(r, c)) || (!std::isfinite(R_l_mat_inv(r, c)))) return;
            }
        }
        
        const double R_l_inv_00 = R_l_mat_inv(0, 0);
        const double R_l_inv_01 = R_l_mat_inv(0, 1);
        const double R_l_inv_02 = R_l_mat_inv(0, 2);
        const double R_l_inv_03 = R_l_mat_inv(0, 3);
        const double R_l_inv_04 = R_l_mat_inv(0, 4);
        const double R_l_inv_05 = R_l_mat_inv(0, 5);
        const double R_l_inv_06 = R_l_mat_inv(0, 6);
        const double R_l_inv_07 = R_l_mat_inv(0, 7);
        const double R_l_inv_10 = R_l_mat_inv(1, 0);
        const double R_l_inv_11 = R_l_mat_inv(1, 1);
        const double R_l_inv_12 = R_l_mat_inv(1, 2);
        const double R_l_inv_13 = R_l_mat_inv(1, 3);
        const double R_l_inv_14 = R_l_mat_inv(1, 4);
        const double R_l_inv_15 = R_l_mat_inv(1, 5);
        const double R_l_inv_16 = R_l_mat_inv(1, 6);
        const double R_l_inv_17 = R_l_mat_inv(1, 7);
        const double R_l_inv_20 = R_l_mat_inv(2, 0);
        const double R_l_inv_21 = R_l_mat_inv(2, 1);
        const double R_l_inv_22 = R_l_mat_inv(2, 2);
        const double R_l_inv_23 = R_l_mat_inv(2, 3);
        const double R_l_inv_24 = R_l_mat_inv(2, 4);
        const double R_l_inv_25 = R_l_mat_inv(2, 5);
        const double R_l_inv_26 = R_l_mat_inv(2, 6);
        const double R_l_inv_27 = R_l_mat_inv(2, 7);
        const double R_l_inv_30 = R_l_mat_inv(3, 0);
        const double R_l_inv_31 = R_l_mat_inv(3, 1);
        const double R_l_inv_32 = R_l_mat_inv(3, 2);
        const double R_l_inv_33 = R_l_mat_inv(3, 3);
        const double R_l_inv_34 = R_l_mat_inv(3, 4);
        const double R_l_inv_35 = R_l_mat_inv(3, 5);
        const double R_l_inv_36 = R_l_mat_inv(3, 6);
        const double R_l_inv_37 = R_l_mat_inv(3, 7);
        const double R_l_inv_40 = R_l_mat_inv(4, 0);
        const double R_l_inv_41 = R_l_mat_inv(4, 1);
        const double R_l_inv_42 = R_l_mat_inv(4, 2);
        const double R_l_inv_43 = R_l_mat_inv(4, 3);
        const double R_l_inv_44 = R_l_mat_inv(4, 4);
        const double R_l_inv_45 = R_l_mat_inv(4, 5);
        const double R_l_inv_46 = R_l_mat_inv(4, 6);
        const double R_l_inv_47 = R_l_mat_inv(4, 7);
        const double R_l_inv_50 = R_l_mat_inv(5, 0);
        const double R_l_inv_51 = R_l_mat_inv(5, 1);
        const double R_l_inv_52 = R_l_mat_inv(5, 2);
        const double R_l_inv_53 = R_l_mat_inv(5, 3);
        const double R_l_inv_54 = R_l_mat_inv(5, 4);
        const double R_l_inv_55 = R_l_mat_inv(5, 5);
        const double R_l_inv_56 = R_l_mat_inv(5, 6);
        const double R_l_inv_57 = R_l_mat_inv(5, 7);
        const double R_l_inv_60 = R_l_mat_inv(6, 0);
        const double R_l_inv_61 = R_l_mat_inv(6, 1);
        const double R_l_inv_62 = R_l_mat_inv(6, 2);
        const double R_l_inv_63 = R_l_mat_inv(6, 3);
        const double R_l_inv_64 = R_l_mat_inv(6, 4);
        const double R_l_inv_65 = R_l_mat_inv(6, 5);
        const double R_l_inv_66 = R_l_mat_inv(6, 6);
        const double R_l_inv_67 = R_l_mat_inv(6, 7);
        const double R_l_inv_70 = R_l_mat_inv(7, 0);
        const double R_l_inv_71 = R_l_mat_inv(7, 1);
        const double R_l_inv_72 = R_l_mat_inv(7, 2);
        const double R_l_inv_73 = R_l_mat_inv(7, 3);
        const double R_l_inv_74 = R_l_mat_inv(7, 4);
        const double R_l_inv_75 = R_l_mat_inv(7, 5);
        const double R_l_inv_76 = R_l_mat_inv(7, 6);
        const double R_l_inv_77 = R_l_mat_inv(7, 7);
        
        // Temp variable for efficiency
        const double x0 = std::pow(a_3, 2);
        const double x1 = -x0;
        const double x2 = std::pow(d_2, 2);
        const double x3 = -x2;
        const double x4 = std::pow(d_4, 2);
        const double x5 = std::pow(a_4, 2);
        const double x6 = -x5;
        const double x7 = std::pow(Pz, 2);
        const double x8 = std::pow(r_31, 2);
        const double x9 = x7*x8;
        const double x10 = std::pow(r_32, 2);
        const double x11 = x10*x7;
        const double x12 = std::pow(r_33, 2);
        const double x13 = x12*x7;
        const double x14 = std::pow(a_2, 2);
        const double x15 = x14*x8;
        const double x16 = x10*x14;
        const double x17 = x12*x14;
        const double x18 = std::sin(th_0);
        const double x19 = Px*x18;
        const double x20 = std::cos(th_0);
        const double x21 = Py*x20;
        const double x22 = x19 - x21;
        const double x23 = std::pow(x22, 2);
        const double x24 = r_11*x18 - r_21*x20;
        const double x25 = std::pow(x24, 2);
        const double x26 = x23*x25;
        const double x27 = r_12*x18 - r_22*x20;
        const double x28 = std::pow(x27, 2);
        const double x29 = x23*x28;
        const double x30 = r_13*x18 - r_23*x20;
        const double x31 = x23*std::pow(x30, 2);
        const double x32 = r_11*x20;
        const double x33 = r_21*x18;
        const double x34 = x32 + x33;
        const double x35 = std::pow(x34, 2);
        const double x36 = Px*x20 + Py*x18 - a_1*std::pow(x18, 2) - a_1*std::pow(x20, 2);
        const double x37 = std::pow(x36, 2);
        const double x38 = x35*x37;
        const double x39 = r_12*x20;
        const double x40 = r_22*x18;
        const double x41 = x39 + x40;
        const double x42 = std::pow(x41, 2);
        const double x43 = x37*x42;
        const double x44 = r_13*x20;
        const double x45 = r_23*x18;
        const double x46 = x44 + x45;
        const double x47 = std::pow(x46, 2);
        const double x48 = x37*x47;
        const double x49 = Pz*r_33;
        const double x50 = 2*x49;
        const double x51 = d_4*x50;
        const double x52 = -x51;
        const double x53 = R_l_inv_40*d_2;
        const double x54 = -R_l_inv_50*a_3 + x53;
        const double x55 = Pz*r_31;
        const double x56 = x22*x24;
        const double x57 = x34*x36;
        const double x58 = x55 + x56 + x57;
        const double x59 = a_4*x58;
        const double x60 = 2*x59;
        const double x61 = x54*x60;
        const double x62 = -x61;
        const double x63 = x22*x30;
        const double x64 = x36*x46;
        const double x65 = x49 + x63 + x64;
        const double x66 = R_l_inv_42*d_2;
        const double x67 = -R_l_inv_52*a_3 + x66;
        const double x68 = 2*a_4;
        const double x69 = x67*x68;
        const double x70 = x65*x69;
        const double x71 = -x70;
        const double x72 = 2*x63;
        const double x73 = d_4*x72;
        const double x74 = -x73;
        const double x75 = 2*x64;
        const double x76 = d_4*x75;
        const double x77 = -x76;
        const double x78 = d_4*x69;
        const double x79 = R_l_inv_41*d_2;
        const double x80 = -R_l_inv_51*a_3 + x79;
        const double x81 = Pz*r_32;
        const double x82 = x22*x27;
        const double x83 = x36*x41;
        const double x84 = x81 + x82 + x83;
        const double x85 = a_4*x84;
        const double x86 = 2*x80*x85;
        const double x87 = 2*x19 - 2*x21;
        const double x88 = x24*x57*x87;
        const double x89 = x27*x83*x87;
        const double x90 = x30*x87;
        const double x91 = x64*x90;
        const double x92 = r_32*x46;
        const double x93 = r_33*x41 - x92;
        const double x94 = a_2*x93;
        const double x95 = R_l_inv_46*d_2;
        const double x96 = -R_l_inv_56*a_3 + x95;
        const double x97 = x68*x96;
        const double x98 = x94*x97;
        const double x99 = -x98;
        const double x100 = r_31*x46;
        const double x101 = a_2*(r_33*x34 - x100);
        const double x102 = R_l_inv_47*d_2;
        const double x103 = -R_l_inv_57*a_3 + x102;
        const double x104 = x103*x68;
        const double x105 = x101*x104;
        const double x106 = -x105;
        const double x107 = 2*x55;
        const double x108 = x107*x56;
        const double x109 = x107*x57;
        const double x110 = 2*x81;
        const double x111 = x110*x82;
        const double x112 = x110*x83;
        const double x113 = x50*x63;
        const double x114 = x50*x64;
        const double x115 = x1 + x106 + x108 + x109 + x11 + x111 + x112 + x113 + x114 + x13 + x15 + x16 + x17 + x26 + x29 + x3 + x31 + x38 + x4 + x43 + x48 + x52 + x6 + x62 + x71 + x74 + x77 + x78 + x86 + x88 + x89 + x9 + x91 + x99;
        const double x116 = a_2*x46;
        const double x117 = 2*x116;
        const double x118 = d_4*x117;
        const double x119 = -x118;
        const double x120 = R_l_inv_43*d_2;
        const double x121 = -R_l_inv_53*a_3 + x120;
        const double x122 = a_4*r_31;
        const double x123 = 2*x122;
        const double x124 = x121*x123;
        const double x125 = -x124;
        const double x126 = R_l_inv_45*d_2;
        const double x127 = -R_l_inv_55*a_3 + x126;
        const double x128 = a_4*r_33;
        const double x129 = 2*x128;
        const double x130 = x127*x129;
        const double x131 = -x130;
        const double x132 = r_32*x63;
        const double x133 = r_32*x64;
        const double x134 = r_33*x22*x27 + r_33*x36*x41 - x132 - x133;
        const double x135 = a_4*x134;
        const double x136 = 2*x135*x96;
        const double x137 = -x136;
        const double x138 = a_2*x34;
        const double x139 = x138*x54*x68;
        const double x140 = -x139;
        const double x141 = x116*x69;
        const double x142 = -x141;
        const double x143 = d_4*r_31;
        const double x144 = x104*x143;
        const double x145 = -x144;
        const double x146 = x119 + x125 + x131 + x137 + x140 + x142 + x145;
        const double x147 = 2*x36;
        const double x148 = x147*x35;
        const double x149 = a_2*x148;
        const double x150 = x147*x42;
        const double x151 = a_2*x150;
        const double x152 = x147*x47;
        const double x153 = a_2*x152;
        const double x154 = x107*x138;
        const double x155 = a_2*x41;
        const double x156 = x110*x155;
        const double x157 = x116*x50;
        const double x158 = 2*x56;
        const double x159 = x138*x158;
        const double x160 = 2*x82;
        const double x161 = x155*x160;
        const double x162 = x116*x72;
        const double x163 = r_31*x63;
        const double x164 = r_31*x64;
        const double x165 = r_33*x22*x24 + r_33*x34*x36 - x163 - x164;
        const double x166 = a_4*x165;
        const double x167 = 2*x103*x166;
        const double x168 = R_l_inv_44*d_2;
        const double x169 = -R_l_inv_54*a_3 + x168;
        const double x170 = a_4*r_32;
        const double x171 = 2*x170;
        const double x172 = x169*x171;
        const double x173 = d_4*r_32;
        const double x174 = x173*x97;
        const double x175 = x155*x68*x80;
        const double x176 = -x167 + x172 - x174 + x175;
        const double x177 = x149 + x151 + x153 + x154 + x156 + x157 + x159 + x161 + x162 + x176;
        const double x178 = x54*x84;
        const double x179 = x101*x96;
        const double x180 = r_31*x169;
        const double x181 = x58*x80;
        const double x182 = x103*x134;
        const double x183 = x138*x80;
        const double x184 = x103*x94;
        const double x185 = x143*x96;
        const double x186 = -x185;
        const double x187 = x180 + x181 + x182 + x183 + x184 + x186;
        const double x188 = r_32*x121;
        const double x189 = x165*x96;
        const double x190 = x155*x54;
        const double x191 = x103*x173;
        const double x192 = x188 - x189 + x190 + x191;
        const double x193 = 4*a_4;
        const double x194 = -x172;
        const double x195 = -x175;
        const double x196 = x124 + x136 + x139 + x144 + x167 + x174 + x194 + x195;
        const double x197 = -x86;
        const double x198 = x1 + x108 + x109 + x11 + x111 + x112 + x113 + x114 + x13 + x15 + x16 + x17 + x26 + x29 + x3 + x31 + x38 + x4 + x43 + x48 + x6 + x61 + x78 + x88 + x89 + x9 + x91 + x98;
        const double x199 = x105 + x197 + x198 + x52 + x71 + x74 + x77;
        const double x200 = x119 + x131 + x142;
        const double x201 = x149 + x151 + x153 + x154 + x156 + x157 + x159 + x161 + x162;
        const double x202 = x121*x193;
        const double x203 = r_33*x202;
        const double x204 = r_31*x82;
        const double x205 = r_32*x56;
        const double x206 = r_31*x83;
        const double x207 = r_32*x57;
        const double x208 = x204 - x205 + x206 - x207;
        const double x209 = -x208;
        const double x210 = x193*x96;
        const double x211 = x209*x210;
        const double x212 = x193*x54;
        const double x213 = x116*x212;
        const double x214 = d_4*r_33;
        const double x215 = x103*x193;
        const double x216 = x214*x215;
        const double x217 = -4*a_4*x54*x65;
        const double x218 = 4*d_4;
        const double x219 = x138*x218;
        const double x220 = x127*x193;
        const double x221 = r_31*x220;
        const double x222 = r_31*x41;
        const double x223 = r_32*x34 - x222;
        const double x224 = -4*a_2*a_4*x223*x96;
        const double x225 = x193*x67;
        const double x226 = x138*x225;
        const double x227 = x217 + x219 + x221 + x224 + x226;
        const double x228 = x218*x55;
        const double x229 = x225*x58;
        const double x230 = x218*x56;
        const double x231 = x218*x57;
        const double x232 = x228 + x229 + x230 + x231;
        const double x233 = 8*d_4;
        const double x234 = x155*x233;
        const double x235 = 8*x127*x170;
        const double x236 = 8*a_4;
        const double x237 = x155*x236*x67;
        const double x238 = 8*x67;
        const double x239 = x233*x81 + x233*x82 + x233*x83 + x238*x85;
        const double x240 = a_2*x223;
        const double x241 = x203 + x211 + x213 + x216;
        const double x242 = x232 + x241;
        const double x243 = x118 + x130 + x141;
        const double x244 = x243 + x51 + x70 + x73 + x76;
        const double x245 = x106 + x198 + x86;
        const double x246 = x124 + x136 + x139 + x144;
        const double x247 = -x178 + x179;
        const double x248 = x125 + x137 + x140 + x145;
        const double x249 = x167 + x174 + x194 + x195;
        const double x250 = x1 + x105 + x108 + x109 + x11 + x111 + x112 + x113 + x114 + x13 + x15 + x16 + x17 + x197 + x26 + x29 + x3 + x31 + x38 + x4 + x43 + x48 + x6 + x62 + x78 + x88 + x89 + x9 + x91 + x99;
        const double x251 = -x214;
        const double x252 = R_l_inv_40*x59;
        const double x253 = R_l_inv_42*a_4;
        const double x254 = x253*x65;
        const double x255 = R_l_inv_43*x122;
        const double x256 = R_l_inv_45*x128;
        const double x257 = R_l_inv_46*x135;
        const double x258 = R_l_inv_40*a_4*x138;
        const double x259 = x116*x253;
        const double x260 = R_l_inv_46*a_4;
        const double x261 = x260*x94;
        const double x262 = R_l_inv_47*a_4;
        const double x263 = x143*x262;
        const double x264 = x251 + x252 + x254 + x255 + x256 + x257 + x258 + x259 + x261 + x263;
        const double x265 = Pz*x8;
        const double x266 = Pz*x10;
        const double x267 = Pz*x12;
        const double x268 = r_31*x56;
        const double x269 = r_31*x57;
        const double x270 = r_32*x82;
        const double x271 = r_32*x83;
        const double x272 = r_33*x63;
        const double x273 = r_33*x64;
        const double x274 = R_l_inv_41*x85;
        const double x275 = x101*x262;
        const double x276 = x265 + x266 + x267 + x268 + x269 + x270 + x271 + x272 + x273 - x274 + x275;
        const double x277 = R_l_inv_47*x166;
        const double x278 = R_l_inv_44*x170;
        const double x279 = x173*x260;
        const double x280 = R_l_inv_41*a_4*x155;
        const double x281 = x277 - x278 + x279 - x280;
        const double x282 = d_4*x253;
        const double x283 = r_31*x138;
        const double x284 = r_32*x155;
        const double x285 = r_33*x116;
        const double x286 = d_2 - x282 - x283 - x284 - x285;
        const double x287 = R_l_inv_40*x84;
        const double x288 = R_l_inv_46*x101;
        const double x289 = R_l_inv_41*x58;
        const double x290 = R_l_inv_44*r_31;
        const double x291 = R_l_inv_47*x134;
        const double x292 = R_l_inv_41*x138;
        const double x293 = R_l_inv_47*x94;
        const double x294 = R_l_inv_46*x143;
        const double x295 = -x294;
        const double x296 = x289 + x290 + x291 + x292 + x293 + x295;
        const double x297 = R_l_inv_43*r_32;
        const double x298 = R_l_inv_46*x165;
        const double x299 = R_l_inv_40*x155;
        const double x300 = R_l_inv_47*x173;
        const double x301 = x297 - x298 + x299 + x300;
        const double x302 = -x252;
        const double x303 = -x261;
        const double x304 = x254 + x302 + x303;
        const double x305 = -x255;
        const double x306 = -x257;
        const double x307 = -x258;
        const double x308 = -x263;
        const double x309 = x251 + x256 + x259 + x305 + x306 + x307 + x308;
        const double x310 = -x277;
        const double x311 = -x279;
        const double x312 = -x275;
        const double x313 = x274 + x278 + x280 + x310 + x311 + x312;
        const double x314 = x265 + x266 + x267 + x268 + x269 + x270 + x271 + x272 + x273 + x286 + x313;
        const double x315 = R_l_inv_42*x60;
        const double x316 = R_l_inv_43*x129;
        const double x317 = R_l_inv_46*x68;
        const double x318 = x209*x317;
        const double x319 = R_l_inv_40*x68;
        const double x320 = x116*x319;
        const double x321 = R_l_inv_47*x68;
        const double x322 = x214*x321;
        const double x323 = -x315 + x316 + x318 + x320 + x322;
        const double x324 = x319*x65;
        const double x325 = x240*x317;
        const double x326 = x324 + x325;
        const double x327 = 2*x143;
        const double x328 = R_l_inv_45*x123;
        const double x329 = 2*x138;
        const double x330 = x253*x329;
        const double x331 = x327 - x328 - x330;
        const double x332 = R_l_inv_42*x193;
        const double x333 = 4*x173;
        const double x334 = R_l_inv_45*x193;
        const double x335 = r_32*x334 + x155*x332 - x333;
        const double x336 = -x327 + x328 + x330;
        const double x337 = x315 + x316 + x318 + x320 + x322;
        const double x338 = -d_2 + x282 + x283 + x284 + x285;
        const double x339 = -x287 + x288;
        const double x340 = -x254;
        const double x341 = x252 + x261 + x340;
        const double x342 = x255 + x257 + x258 + x263;
        const double x343 = x214 - x256 - x259;
        const double x344 = x342 + x343;
        const double x345 = a_2*x222;
        const double x346 = r_32*x34;
        const double x347 = a_2*x346;
        const double x348 = x345 - x347;
        const double x349 = x208 + x348;
        const double x350 = r_33*x82;
        const double x351 = 2*x350;
        const double x352 = r_33*x83;
        const double x353 = 2*x352;
        const double x354 = 2*x132;
        const double x355 = 2*x133;
        const double x356 = r_33*x41;
        const double x357 = 2*a_2;
        const double x358 = x356*x357;
        const double x359 = x357*x92;
        const double x360 = -x358 + x359;
        const double x361 = -x351 - x353 + x354 + x355 + x360;
        const double x362 = r_33*x56;
        const double x363 = 4*x362;
        const double x364 = r_33*x57;
        const double x365 = 4*x364;
        const double x366 = 4*x163;
        const double x367 = 4*x164;
        const double x368 = 4*a_2;
        const double x369 = x100*x368;
        const double x370 = -4*a_2*r_33*x34 + x369;
        const double x371 = -x349;
        const double x372 = a_3*d_2;
        const double x373 = 2*x372;
        const double x374 = d_2*x68;
        const double x375 = x1 + x2 + x5;
        const double x376 = -R_l_inv_15*x375 + R_l_inv_35*x373 + R_l_inv_75*x374;
        const double x377 = r_33*x376;
        const double x378 = -R_l_inv_12*x375 + R_l_inv_32*x373 + R_l_inv_72*x374;
        const double x379 = x378*x65;
        const double x380 = -d_4*x378;
        const double x381 = x116*x378;
        const double x382 = x377 + x379 + x380 + x381;
        const double x383 = R_l_inv_17*x375;
        const double x384 = R_l_inv_37*x373 + R_l_inv_77*x374 - x383;
        const double x385 = x165*x384;
        const double x386 = -R_l_inv_14*x375 + R_l_inv_34*x373 + R_l_inv_74*x374;
        const double x387 = r_32*x386;
        const double x388 = -x387;
        const double x389 = -R_l_inv_11*x375 + R_l_inv_31*x373 + R_l_inv_71*x374;
        const double x390 = x389*x84;
        const double x391 = -x390;
        const double x392 = x101*x384;
        const double x393 = R_l_inv_16*x375;
        const double x394 = R_l_inv_36*x373 + R_l_inv_76*x374 - x393;
        const double x395 = x173*x394;
        const double x396 = x155*x389;
        const double x397 = -x396;
        const double x398 = d_4*x357;
        const double x399 = x100*x398;
        const double x400 = -x399;
        const double x401 = 2*d_4;
        const double x402 = x163*x401;
        const double x403 = -x402;
        const double x404 = x164*x401;
        const double x405 = -x404;
        const double x406 = r_33*x34;
        const double x407 = x398*x406;
        const double x408 = x158*x214;
        const double x409 = 2*x57;
        const double x410 = x214*x409;
        const double x411 = x385 + x388 + x391 + x392 + x395 + x397 + x400 + x403 + x405 + x407 + x408 + x410;
        const double x412 = r_31*x4;
        const double x413 = -R_l_inv_13*x375 + R_l_inv_33*x373 + R_l_inv_73*x374;
        const double x414 = r_31*x413;
        const double x415 = x134*x394;
        const double x416 = std::pow(r_31, 3);
        const double x417 = x416*x7;
        const double x418 = x14*x416;
        const double x419 = -R_l_inv_10*x375 + R_l_inv_30*x373 + R_l_inv_70*x374;
        const double x420 = x138*x419;
        const double x421 = x143*x384;
        const double x422 = r_31*x29;
        const double x423 = r_31*x31;
        const double x424 = r_31*x43;
        const double x425 = r_31*x48;
        const double x426 = r_31*x11;
        const double x427 = r_31*x13;
        const double x428 = r_31*x16;
        const double x429 = r_31*x17;
        const double x430 = r_31*x26;
        const double x431 = r_31*x38;
        const double x432 = 2*r_31;
        const double x433 = x14*x432;
        const double x434 = x35*x433;
        const double x435 = x42*x433;
        const double x436 = x433*x47;
        const double x437 = x158*x265;
        const double x438 = x265*x409;
        const double x439 = x158*x266;
        const double x440 = x266*x409;
        const double x441 = x158*x267;
        const double x442 = x267*x409;
        const double x443 = 2*r_32;
        const double x444 = x23*x24;
        const double x445 = x27*x444;
        const double x446 = x443*x445;
        const double x447 = 2*x37;
        const double x448 = x346*x41;
        const double x449 = x447*x448;
        const double x450 = 2*r_33;
        const double x451 = x30*x444;
        const double x452 = x450*x451;
        const double x453 = 2*x46;
        const double x454 = x37*x453;
        const double x455 = x406*x454;
        const double x456 = x268*x409;
        const double x457 = x160*x207;
        const double x458 = 2*x83;
        const double x459 = x205*x458;
        const double x460 = x364*x72;
        const double x461 = x362*x75;
        const double x462 = x204*x458;
        const double x463 = x163*x75;
        const double x464 = x419*x58;
        const double x465 = x394*x94;
        const double x466 = x266*x329;
        const double x467 = x267*x329;
        const double x468 = x265*x329;
        const double x469 = x35*x36;
        const double x470 = a_2*x432;
        const double x471 = x469*x470;
        const double x472 = x36*x470;
        const double x473 = x42*x472;
        const double x474 = x47*x472;
        const double x475 = x160*x347;
        const double x476 = a_2*x72;
        const double x477 = x406*x476;
        const double x478 = x268*x329;
        const double x479 = x160*x345;
        const double x480 = x100*x476;
        const double x481 = 2*x155;
        const double x482 = x205*x481;
        const double x483 = x117*x362;
        const double x484 = 4*x55;
        const double x485 = x284*x484;
        const double x486 = x285*x484;
        const double x487 = x464 + x465 - x466 - x467 + x468 + x471 + x473 + x474 - x475 - x477 + x478 + x479 + x480 + x482 + x483 + x485 + x486;
        const double x488 = x412 + x414 + x415 - x417 - x418 + x420 + x421 + x422 + x423 + x424 + x425 - x426 - x427 - x428 - x429 - x430 - x431 + x434 + x435 + x436 - x437 - x438 - x439 - x440 - x441 - x442 - x446 - x449 - x452 - x455 - x456 - x457 - x459 - x460 - x461 + x462 + x463 + x487;
        const double x489 = 4*x372;
        const double x490 = d_2*x193;
        const double x491 = R_l_inv_36*x489 + R_l_inv_76*x490 - 2*x393;
        const double x492 = x165*x491;
        const double x493 = x107 + x158 + x409;
        const double x494 = x389*x493;
        const double x495 = r_32*x4;
        const double x496 = -2*x495;
        const double x497 = 2*x413;
        const double x498 = -r_32*x497;
        const double x499 = std::pow(r_32, 3);
        const double x500 = x499*x7;
        const double x501 = 2*x500;
        const double x502 = x14*x499;
        const double x503 = 2*x502;
        const double x504 = 4*r_32;
        const double x505 = x14*x504;
        const double x506 = -x35*x505;
        const double x507 = -x42*x505;
        const double x508 = -x47*x505;
        const double x509 = -x419*x481;
        const double x510 = 2*x384;
        const double x511 = x510*x94;
        const double x512 = -x173*x510;
        const double x513 = r_32*x26;
        const double x514 = -2*x513;
        const double x515 = r_32*x31;
        const double x516 = -2*x515;
        const double x517 = r_32*x38;
        const double x518 = -2*x517;
        const double x519 = r_32*x48;
        const double x520 = -2*x519;
        const double x521 = r_32*x9;
        const double x522 = 2*x521;
        const double x523 = r_32*x13;
        const double x524 = 2*x523;
        const double x525 = r_32*x15;
        const double x526 = 2*x525;
        const double x527 = r_32*x17;
        const double x528 = 2*x527;
        const double x529 = r_32*x29;
        const double x530 = 2*x529;
        const double x531 = r_32*x43;
        const double x532 = 2*x531;
        const double x533 = a_2*x218;
        const double x534 = x356*x533;
        const double x535 = 4*x82;
        const double x536 = x265*x535;
        const double x537 = 4*x83;
        const double x538 = x265*x537;
        const double x539 = x266*x535;
        const double x540 = x266*x537;
        const double x541 = x267*x535;
        const double x542 = x267*x537;
        const double x543 = 4*r_31;
        const double x544 = x445*x543;
        const double x545 = 4*x37;
        const double x546 = x34*x545;
        const double x547 = x222*x546;
        const double x548 = 4*r_33;
        const double x549 = x23*x27*x30;
        const double x550 = x548*x549;
        const double x551 = x46*x545;
        const double x552 = x356*x551;
        const double x553 = 4*x57;
        const double x554 = -x205*x553;
        const double x555 = 4*x64;
        const double x556 = -x132*x555;
        const double x557 = 4*x204;
        const double x558 = x557*x57;
        const double x559 = 4*x56;
        const double x560 = x206*x559;
        const double x561 = x270*x537;
        const double x562 = x352*x63;
        const double x563 = 4*x562;
        const double x564 = x350*x555;
        const double x565 = x492 - x494 + x496 + x498 + x501 + x503 + x506 + x507 + x508 + x509 - x511 + x512 + x514 + x516 + x518 + x520 + x522 + x524 + x526 + x528 + x530 + x532 + x533*x92 - x534 + x536 + x538 + x539 + x540 + x541 + x542 + x544 + x547 + x550 + x552 + x554 + x556 + x558 + x560 + x561 + x563 + x564;
        const double x566 = R_l_inv_37*x489 + R_l_inv_77*x490 - 2*x383;
        const double x567 = x134*x566;
        const double x568 = x386*x432;
        const double x569 = x329*x389;
        const double x570 = x327*x394;
        const double x571 = x214*x535;
        const double x572 = x214*x537;
        const double x573 = x132*x218;
        const double x574 = x133*x218;
        const double x575 = -x567 - x568 - x569 + x570 - x571 - x572 + x573 + x574;
        const double x576 = x110 + x160 + x458;
        const double x577 = x419*x576;
        const double x578 = 2*x101;
        const double x579 = x394*x578;
        const double x580 = 4*x155;
        const double x581 = x266*x580;
        const double x582 = a_2*x504;
        const double x583 = x469*x582;
        const double x584 = x36*x582;
        const double x585 = x42*x584;
        const double x586 = x47*x584;
        const double x587 = x265*x580;
        const double x588 = x267*x580;
        const double x589 = 8*x55;
        const double x590 = x347*x589;
        const double x591 = 8*a_2;
        const double x592 = x49*x92;
        const double x593 = x591*x592;
        const double x594 = x138*x557;
        const double x595 = x347*x559;
        const double x596 = x270*x580;
        const double x597 = x368*x63;
        const double x598 = x597*x92;
        const double x599 = 4*x116;
        const double x600 = x350*x599;
        const double x601 = x345*x559;
        const double x602 = x356*x597;
        const double x603 = -x577 + x579 - x581 - x583 - x585 - x586 + x587 + x588 - x590 - x593 - x594 - x595 - x596 - x598 - x600 + x601 + x602;
        const double x604 = -x385;
        const double x605 = -x392;
        const double x606 = -x395;
        const double x607 = -x407;
        const double x608 = -x408;
        const double x609 = -x410;
        const double x610 = x387 + x390 + x396 + x399 + x402 + x404 + x604 + x605 + x606 + x607 + x608 + x609;
        const double x611 = -x412 - x414 - x415 + x417 + x418 - x420 - x421 - x422 - x423 - x424 - x425 + x426 + x427 + x428 + x429 + x430 + x431 - x434 - x435 - x436 + x437 + x438 + x439 + x440 + x441 + x442 + x446 + x449 + x452 + x455 + x456 + x457 + x459 + x460 + x461 - x462 - x463;
        const double x612 = -x464 - x465 + x466 + x467 - x468 - x471 - x473 - x474 + x475 + x477 - x478 - x479 - x480 - x482 - x483 - x485 - x486 + x611;
        const double x613 = x50 + x72 + x75;
        const double x614 = x265*x599;
        const double x615 = x266*x599;
        const double x616 = x369*x56;
        const double x617 = a_2*x535;
        const double x618 = x617*x92;
        const double x619 = x378*x493;
        const double x620 = x376*x432;
        const double x621 = x329*x378;
        const double x622 = x619 + x620 + x621;
        const double x623 = x209*x491;
        const double x624 = r_33*x4;
        const double x625 = 2*x624;
        const double x626 = r_33*x497;
        const double x627 = std::pow(r_33, 3);
        const double x628 = x627*x7;
        const double x629 = 2*x628;
        const double x630 = x14*x627;
        const double x631 = 2*x630;
        const double x632 = x14*x35;
        const double x633 = x548*x632;
        const double x634 = x14*x548;
        const double x635 = x42*x634;
        const double x636 = x47*x634;
        const double x637 = x117*x419;
        const double x638 = x214*x510;
        const double x639 = r_33*x26;
        const double x640 = 2*x639;
        const double x641 = r_33*x29;
        const double x642 = 2*x641;
        const double x643 = r_33*x38;
        const double x644 = 2*x643;
        const double x645 = r_33*x43;
        const double x646 = 2*x645;
        const double x647 = r_33*x9;
        const double x648 = 2*x647;
        const double x649 = r_33*x11;
        const double x650 = 2*x649;
        const double x651 = r_33*x15;
        const double x652 = 2*x651;
        const double x653 = r_33*x16;
        const double x654 = 2*x653;
        const double x655 = r_33*x31;
        const double x656 = 2*x655;
        const double x657 = r_33*x48;
        const double x658 = 2*x657;
        const double x659 = 4*x265;
        const double x660 = x63*x659;
        const double x661 = x64*x659;
        const double x662 = 4*x266;
        const double x663 = x63*x662;
        const double x664 = x64*x662;
        const double x665 = 4*x267;
        const double x666 = x63*x665;
        const double x667 = x64*x665;
        const double x668 = x451*x543;
        const double x669 = x100*x546;
        const double x670 = x504*x549;
        const double x671 = x41*x92;
        const double x672 = x545*x671;
        const double x673 = x363*x57;
        const double x674 = x350*x537;
        const double x675 = x366*x57;
        const double x676 = x367*x56;
        const double x677 = x132*x537;
        const double x678 = x133*x535;
        const double x679 = x272*x555;
        const double x680 = -x623 - x625 - x626 + x629 + x631 - x633 - x635 - x636 - x637 - x638 - x640 - x642 - x644 - x646 + x648 + x650 + x652 + x654 + x656 + x658 + x660 + x661 + x663 + x664 + x666 + x667 + x668 + x669 + x670 + x672 - x673 - x674 + x675 + x676 + x677 + x678 + x679;
        const double x681 = 4*x81;
        const double x682 = x535 + x537 + x681;
        const double x683 = x378*x682;
        const double x684 = x376*x504 + x378*x580;
        const double x685 = x223*x357;
        const double x686 = r_33*x368;
        const double x687 = x36*x686;
        const double x688 = x368*x406;
        const double x689 = 8*x49;
        const double x690 = x132*x580 + x138*x366 + x267*x599 + x272*x599 + x284*x689 + x356*x617 + x394*x685 + x406*x55*x591 + x419*x613 + x42*x687 + x469*x686 + x47*x687 + x56*x688 - x614 - x615 - x616 - x618;
        const double x691 = d_4*x378;
        const double x692 = x379 + x691;
        const double x693 = x377 + x381;
        const double x694 = x692 + x693;
        const double x695 = x567 + x568 + x569 - x570 + x571 + x572 - x573 - x574;
        const double x696 = -4*a_2*d_4*r_32*x46 + x492 + x494 + x496 + x498 + x501 + x503 + x506 + x507 + x508 + x509 + x511 + x512 + x514 + x516 + x518 + x520 + x522 + x524 + x526 + x528 + x530 + x532 + x534 + x536 + x538 + x539 + x540 + x541 + x542 + x544 + x547 + x550 + x552 + x554 + x556 + x558 + x560 + x561 + x563 + x564;
        const double x697 = -R_l_inv_03*x375 + R_l_inv_23*x373 + R_l_inv_63*x374;
        const double x698 = r_31*x697;
        const double x699 = -R_l_inv_05*x375 + R_l_inv_25*x373 + R_l_inv_65*x374;
        const double x700 = r_33*x699;
        const double x701 = -R_l_inv_00*x375 + R_l_inv_20*x373 + R_l_inv_60*x374;
        const double x702 = x58*x701;
        const double x703 = -R_l_inv_02*x375 + R_l_inv_22*x373 + R_l_inv_62*x374;
        const double x704 = x65*x703;
        const double x705 = R_l_inv_06*x375;
        const double x706 = R_l_inv_26*x373 + R_l_inv_66*x374 - x705;
        const double x707 = x134*x706;
        const double x708 = x138*x701;
        const double x709 = x116*x703;
        const double x710 = x706*x94;
        const double x711 = R_l_inv_07*x375;
        const double x712 = R_l_inv_27*x373 + R_l_inv_67*x374 - x711;
        const double x713 = x143*x712;
        const double x714 = d_4*x358;
        const double x715 = -x714;
        const double x716 = x160*x214;
        const double x717 = -x716;
        const double x718 = x214*x458;
        const double x719 = -x718;
        const double x720 = d_4*x359;
        const double x721 = d_4*x354;
        const double x722 = d_4*x355;
        const double x723 = x698 + x700 + x702 + x704 + x707 + x708 + x709 + x710 + x713 + x715 + x717 + x719 + x720 + x721 + x722;
        const double x724 = x165*x712;
        const double x725 = -R_l_inv_04*x375 + R_l_inv_24*x373 + R_l_inv_64*x374;
        const double x726 = r_32*x725;
        const double x727 = x173*x706;
        const double x728 = -R_l_inv_01*x375 + R_l_inv_21*x373 + R_l_inv_61*x374;
        const double x729 = x155*x728;
        const double x730 = x443*x632;
        const double x731 = x14*x443;
        const double x732 = x42*x731;
        const double x733 = x47*x731;
        const double x734 = x160*x265;
        const double x735 = x265*x458;
        const double x736 = x160*x266;
        const double x737 = x266*x458;
        const double x738 = x160*x267;
        const double x739 = x267*x458;
        const double x740 = x432*x445;
        const double x741 = x34*x447;
        const double x742 = x222*x741;
        const double x743 = x450*x549;
        const double x744 = x356*x454;
        const double x745 = x205*x409;
        const double x746 = x132*x75;
        const double x747 = x204*x409;
        const double x748 = x158*x206;
        const double x749 = x270*x458;
        const double x750 = x352*x72;
        const double x751 = x350*x75;
        const double x752 = -x495 + x500 + x502 - x513 - x515 - x517 - x519 + x521 + x523 + x525 + x527 + x529 + x531 + x724 - x726 + x727 - x729 - x730 - x732 - x733 + x734 + x735 + x736 + x737 + x738 + x739 + x740 + x742 + x743 + x744 - x745 - x746 + x747 + x748 + x749 + x750 + x751;
        const double x753 = -d_4*x703 + x752;
        const double x754 = x728*x84;
        const double x755 = x101*x712;
        const double x756 = x266*x481;
        const double x757 = a_2*r_32;
        const double x758 = x148*x757;
        const double x759 = x150*x757;
        const double x760 = x152*x757;
        const double x761 = x265*x481;
        const double x762 = x267*x481;
        const double x763 = x347*x484;
        const double x764 = x368*x592;
        const double x765 = x204*x329;
        const double x766 = x158*x347;
        const double x767 = x270*x481;
        const double x768 = x476*x92;
        const double x769 = x117*x350;
        const double x770 = x158*x345;
        const double x771 = x356*x476;
        const double x772 = -x754 + x755 - x756 - x758 - x759 - x760 + x761 + x762 - x763 - x764 - x765 - x766 - x767 - x768 - x769 + x770 + x771;
        const double x773 = x493*x728;
        const double x774 = R_l_inv_27*x489 + R_l_inv_67*x490 - 2*x711;
        const double x775 = -x134*x774;
        const double x776 = 2*x412;
        const double x777 = -x776;
        const double x778 = -x432*x725;
        const double x779 = -x329*x728;
        const double x780 = 2*x712;
        const double x781 = x780*x94;
        const double x782 = x327*x706;
        const double x783 = x14*x543;
        const double x784 = -x204*x537 + x205*x537 + x207*x535 + x268*x553 + x363*x64 + x365*x63 - x366*x64 + x406*x551 + 2*x417 + 2*x418 - x42*x783 - 2*x422 - 2*x423 - 2*x424 - 2*x425 + 2*x426 + 2*x427 + 2*x428 + 2*x429 + 2*x430 + 2*x431 + x445*x504 + x448*x545 + x451*x548 - x47*x783 - x543*x632 + x56*x659 + x56*x662 + x56*x665 + x57*x659 + x57*x662 + x57*x665;
        const double x785 = x138*x659;
        const double x786 = a_2*r_31;
        const double x787 = 4*x786;
        const double x788 = x469*x787;
        const double x789 = x36*x787;
        const double x790 = x42*x789;
        const double x791 = x47*x789;
        const double x792 = x138*x662;
        const double x793 = x138*x665;
        const double x794 = x284*x589;
        const double x795 = x285*x589;
        const double x796 = 4*x138*x268;
        const double x797 = x345*x535;
        const double x798 = x369*x63;
        const double x799 = x205*x580;
        const double x800 = x116*x363;
        const double x801 = x347*x535;
        const double x802 = x63*x688;
        const double x803 = -x785 - x788 - x790 - x791 + x792 + x793 - x794 - x795 - x796 - x797 - x798 - x799 - x800 + x801 + x802;
        const double x804 = -x773 + x775 + x777 + x778 + x779 - x781 + x782 + x784 + x803;
        const double x805 = x576*x701;
        const double x806 = x578*x706;
        const double x807 = x406*x533;
        const double x808 = x100*x533;
        const double x809 = -x805 + x806 - x807 + x808;
        const double x810 = R_l_inv_26*x489 + R_l_inv_66*x490 - 2*x705;
        const double x811 = x165*x810;
        const double x812 = 2*x697;
        const double x813 = r_32*x812;
        const double x814 = x481*x701;
        const double x815 = x173*x780;
        const double x816 = 4*x214;
        const double x817 = x56*x816;
        const double x818 = x57*x816;
        const double x819 = x163*x218;
        const double x820 = x164*x218;
        const double x821 = x811 - x813 - x814 - x815 - x817 - x818 + x819 + x820;
        const double x822 = d_4*x703;
        const double x823 = x752 + x772 + x822;
        const double x824 = -x704;
        const double x825 = x702 + x710 + x715 + x720 + x824;
        const double x826 = -x700 - x709;
        const double x827 = x698 + x707 + x708 + x713 + x717 + x719 + x721 + x722 + x826;
        const double x828 = x209*x810;
        const double x829 = x493*x703;
        const double x830 = r_33*x812;
        const double x831 = x117*x701;
        const double x832 = x214*x780;
        const double x833 = -x333*x56;
        const double x834 = -x333*x57;
        const double x835 = x143*x535;
        const double x836 = x143*x537;
        const double x837 = x828 - x829 + x830 + x831 + x832 + x833 + x834 + x835 + x836;
        const double x838 = x432*x699;
        const double x839 = x329*x703;
        const double x840 = -x838 - x839;
        const double x841 = x218*x347;
        const double x842 = x218*x345 + x613*x701 + x685*x706 - x841;
        const double x843 = x682*x703;
        const double x844 = x504*x699 + x580*x703;
        const double x845 = x838 + x839;
        const double x846 = x828 + x829 + x830 + x831 + x832 + x833 + x834 + x835 + x836;
        const double x847 = x754 - x755 + x756 + x758 + x759 + x760 - x761 - x762 + x763 + x764 + x765 + x766 + x767 + x768 + x769 - x770 - x771;
        const double x848 = x822 + x847;
        const double x849 = x805 - x806 + x807 - x808;
        const double x850 = -x811 + x813 + x814 + x815 + x817 + x818 - x819 - x820;
        const double x851 = -x702;
        const double x852 = -x710;
        const double x853 = -x720;
        const double x854 = x704 + x714 + x851 + x852 + x853;
        const double x855 = x700 + x709;
        const double x856 = -x698 - x707 - x708 - x713 + x716 + x718 - x721 - x722;
        const double x857 = x855 + x856;
        const double x858 = 2*a_3;
        const double x859 = x0 + x3 + x5;
        const double x860 = -R_l_inv_53*x859 + x120*x858;
        const double x861 = r_31*x860;
        const double x862 = R_l_inv_50*x859;
        const double x863 = x53*x858 - x862;
        const double x864 = x58*x863;
        const double x865 = R_l_inv_56*x859;
        const double x866 = x858*x95 - x865;
        const double x867 = x134*x866;
        const double x868 = R_l_inv_57*x859;
        const double x869 = x102*x858 - x868;
        const double x870 = x165*x869;
        const double x871 = -R_l_inv_54*x859 + x168*x858;
        const double x872 = r_32*x871;
        const double x873 = -x872;
        const double x874 = R_l_inv_51*x859;
        const double x875 = x79*x858 - x874;
        const double x876 = x84*x875;
        const double x877 = -x876;
        const double x878 = x138*x863;
        const double x879 = x101*x869;
        const double x880 = x866*x94;
        const double x881 = x143*x869;
        const double x882 = x173*x866;
        const double x883 = x155*x875;
        const double x884 = -x883;
        const double x885 = x861 + x864 + x867 + x870 + x873 + x877 + x878 + x879 + x880 + x881 + x882 + x884;
        const double x886 = -R_l_inv_55*x859 + x126*x858;
        const double x887 = r_33*x886;
        const double x888 = R_l_inv_52*x859;
        const double x889 = x66*x858 - x888;
        const double x890 = d_4*x889;
        const double x891 = x116*x889;
        const double x892 = -x639;
        const double x893 = -x641;
        const double x894 = -x643;
        const double x895 = -x645;
        const double x896 = x450*x632;
        const double x897 = -x896;
        const double x898 = x14*x450;
        const double x899 = x42*x898;
        const double x900 = -x899;
        const double x901 = x47*x898;
        const double x902 = -x901;
        const double x903 = x265*x72;
        const double x904 = x265*x75;
        const double x905 = x266*x72;
        const double x906 = x266*x75;
        const double x907 = x267*x72;
        const double x908 = x267*x75;
        const double x909 = x432*x451;
        const double x910 = x100*x741;
        const double x911 = x443*x549;
        const double x912 = x447*x671;
        const double x913 = x362*x409;
        const double x914 = -x913;
        const double x915 = x350*x458;
        const double x916 = -x915;
        const double x917 = x163*x409;
        const double x918 = x158*x164;
        const double x919 = x132*x458;
        const double x920 = x133*x160;
        const double x921 = x272*x75;
        const double x922 = a_3*x68 + x117*x214 + x138*x327 + x173*x481 + x624 + x628 + x630 + x647 + x649 + x651 + x653 + x655 + x657 + x887 - x890 + x891 + x892 + x893 + x894 + x895 + x897 + x900 + x902 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 + x914 + x916 + x917 + x918 + x919 + x920 + x921;
        const double x923 = x265*x401;
        const double x924 = x266*x401;
        const double x925 = x267*x401;
        const double x926 = x327*x56;
        const double x927 = x327*x57;
        const double x928 = x160*x173;
        const double x929 = x173*x458;
        const double x930 = x214*x72;
        const double x931 = x214*x75;
        const double x932 = -x923 - x924 - x925 - x926 - x927 - x928 - x929 - x930 - x931;
        const double x933 = x65*x889;
        const double x934 = x117*x267;
        const double x935 = r_33*x149;
        const double x936 = r_33*x151;
        const double x937 = r_33*x153;
        const double x938 = x117*x265;
        const double x939 = x117*x266;
        const double x940 = x55*x688;
        const double x941 = 4*x49;
        const double x942 = x284*x941;
        const double x943 = x163*x329;
        const double x944 = x155*x354;
        const double x945 = a_2*x158;
        const double x946 = x406*x945;
        const double x947 = a_2*x160;
        const double x948 = x356*x947;
        const double x949 = x117*x272;
        const double x950 = x100*x945;
        const double x951 = x92*x947;
        const double x952 = x933 - x934 - x935 - x936 - x937 + x938 + x939 - x940 - x942 - x943 - x944 - x946 - x948 - x949 + x950 + x951;
        const double x953 = 4*a_3;
        const double x954 = x53*x953 - 2*x862;
        const double x955 = x84*x954;
        const double x956 = x578*x866;
        const double x957 = x79*x953 - 2*x874;
        const double x958 = x58*x957;
        const double x959 = x102*x953 - 2*x868;
        const double x960 = x134*x959;
        const double x961 = x432*x871;
        const double x962 = x327*x866;
        const double x963 = -x962;
        const double x964 = x329*x875;
        const double x965 = 2*x869;
        const double x966 = x94*x965;
        const double x967 = x958 + x960 + x961 + x963 + x964 + x966;
        const double x968 = -2*x865 + x95*x953;
        const double x969 = x165*x968;
        const double x970 = 2*x860;
        const double x971 = r_32*x970;
        const double x972 = x481*x863;
        const double x973 = x173*x965;
        const double x974 = -x969 + x971 + x972 + x973;
        const double x975 = -2*a_3*a_4;
        const double x976 = -2*a_2*d_4*r_31*x34;
        const double x977 = -2*a_2*d_4*r_32*x41;
        const double x978 = -2*a_2*d_4*r_33*x46;
        const double x979 = x890 + x923 + x924 + x925 + x926 + x927 + x928 + x929 + x930 + x931 + x975 + x976 + x977 + x978;
        const double x980 = -x933 + x934 + x935 + x936 + x937 - x938 - x939 + x940 + x942 + x943 + x944 + x946 + x948 + x949 - x950 - x951;
        const double x981 = x65*x954;
        const double x982 = x432*x886;
        const double x983 = x685*x866;
        const double x984 = x329*x889;
        const double x985 = x776 + x784 - x981 + x982 - x983 + x984;
        const double x986 = x209*x968;
        const double x987 = r_33*x970;
        const double x988 = x117*x863;
        const double x989 = x214*x965;
        const double x990 = -x986 - x987 - x988 - x989;
        const double x991 = x58*(x66*x953 - 2*x888);
        const double x992 = x803 + x991;
        const double x993 = 8*a_3;
        const double x994 = x66*x993 - 4*x888;
        const double x995 = 8*x155;
        const double x996 = x266*x995;
        const double x997 = 8*x757;
        const double x998 = x469*x997;
        const double x999 = x36*x997;
        const double x1000 = x42*x999;
        const double x1001 = x47*x999;
        const double x1002 = 16*x55;
        const double x1003 = x1002*x347;
        const double x1004 = 16*a_2;
        const double x1005 = x1004*x592;
        const double x1006 = 8*x204;
        const double x1007 = x1006*x138;
        const double x1008 = 8*x56;
        const double x1009 = x1008*x347;
        const double x1010 = x270*x995;
        const double x1011 = x591*x63;
        const double x1012 = x1011*x92;
        const double x1013 = 8*x350;
        const double x1014 = x1013*x116;
        const double x1015 = 8*r_32;
        const double x1016 = x1015*x14;
        const double x1017 = 8*x265;
        const double x1018 = 8*x266;
        const double x1019 = 8*x267;
        const double x1020 = 8*r_31;
        const double x1021 = 8*x37;
        const double x1022 = 8*r_33;
        const double x1023 = x1006*x57 + x1008*x206 + x1013*x64 - x1015*x632 - x1016*x42 - x1016*x47 + x1017*x82 + x1017*x83 + x1018*x82 + x1018*x83 + x1019*x82 + x1019*x83 + x1020*x445 + x1021*x222*x34 + x1021*x356*x46 + x1022*x549 + x13*x504 - 8*x132*x64 + x15*x504 + x17*x504 - 8*x205*x57 - x26*x504 + 8*x270*x83 + x29*x504 - x31*x504 - x38*x504 + x43*x504 - x48*x504 + 4*x495 + 4*x500 + 4*x502 + x504*x886 + x504*x9 + 8*x562 + x580*x889;
        const double x1024 = x986 + x987 + x988 + x989;
        const double x1025 = x776 + x784 + x981 + x982 + x983 + x984;
        const double x1026 = -x879;
        const double x1027 = x1026 + x864 + x876 + x880;
        const double x1028 = -x870;
        const double x1029 = -x882;
        const double x1030 = x1028 + x1029 + x861 + x867 + x872 + x878 + x881 + x883;
        const double x1031 = x624 + x628 + x630 + x647 + x649 + x651 + x653 + x655 + x657 + x887 + x891 + x892 + x893 + x894 + x895 + x897 + x900 + x902 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 + x914 + x916 + x917 + x918 + x919 + x920 + x921 + x952 + x979;
        const double x1032 = -x955 + x956;
        const double x1033 = -x864;
        const double x1034 = -x880;
        const double x1035 = x1033 + x1034 + x877 + x879;
        const double x1036 = x870 + x873 + x882 + x884;
        const double x1037 = -x861 - x867 - x878 - x881;
        const double x1038 = x1036 + x1037;
        const double x1039 = x214*x368;
        const double x1040 = x265*x368;
        const double x1041 = x266*x368;
        const double x1042 = x267*x368;
        const double x1043 = x220*x46;
        const double x1044 = a_2*r_33;
        const double x1045 = x1044*x225;
        const double x1046 = x268*x368;
        const double x1047 = x269*x368;
        const double x1048 = x270*x368;
        const double x1049 = x271*x368;
        const double x1050 = x272*x368;
        const double x1051 = x273*x368;
        const double x1052 = -x1039 + x1040 + x1041 + x1042 + x1043 - x1045 + x1046 + x1047 + x1048 + x1049 + x1050 + x1051;
        const double x1053 = x169*x193*x41;
        const double x1054 = x46*x55;
        const double x1055 = x46*x56;
        const double x1056 = Pz*r_33*x34 - x1054 - x1055 + x22*x30*x34;
        const double x1057 = x1056*x215;
        const double x1058 = x193*x757*x80;
        const double x1059 = d_4*x41;
        const double x1060 = x1059*x210;
        const double x1061 = -x1053 - x1057 + x1058 + x1060;
        const double x1062 = x46*x81;
        const double x1063 = x46*x82;
        const double x1064 = Pz*r_33*x41 - x1062 - x1063 + x22*x30*x41;
        const double x1065 = x1064*x210;
        const double x1066 = x202*x34;
        const double x1067 = x212*x786;
        const double x1068 = d_4*x34;
        const double x1069 = x1068*x215;
        const double x1070 = -x1065 + x1066 - x1067 + x1069;
        const double x1071 = x121*x41;
        const double x1072 = x1056*x96;
        const double x1073 = x54*x757;
        const double x1074 = x103*x1059;
        const double x1075 = x103*x1064 + x1068*x96 - x169*x34 + x786*x80;
        const double x1076 = x1065 - x1066 + x1067 - x1069;
        const double x1077 = x1053 + x1057 - x1058 - x1060;
        const double x1078 = x41*x55;
        const double x1079 = x41*x56;
        const double x1080 = Pz*r_32*x34 - x1078 - x1079 + x22*x27*x34;
        const double x1081 = x1080*x236*x96;
        const double x1082 = x236*x46;
        const double x1083 = x128*x54*x591;
        const double x1084 = a_2*x122*x238 - x127*x236*x34 + x143*x591;
        const double x1085 = 16*x41;
        const double x1086 = x1039 + x1040 + x1041 + x1042 - x1043 + x1045 + x1046 + x1047 + x1048 + x1049 + x1050 + x1051;
        const double x1087 = d_4*x453;
        const double x1088 = x24*x34*x87;
        const double x1089 = x41*x87;
        const double x1090 = x1089*x27;
        const double x1091 = x46*x90;
        const double x1092 = x107*x34;
        const double x1093 = x110*x41;
        const double x1094 = x46*x50;
        const double x1095 = R_l_inv_45*x46*x68;
        const double x1096 = r_33*x357;
        const double x1097 = x1096*x253;
        const double x1098 = -x1087 + x1088 + x1090 + x1091 + x1092 + x1093 + x1094 + x1095 - x1097 + x148 + x150 + x152;
        const double x1099 = R_l_inv_44*x41*x68;
        const double x1100 = x1056*x321;
        const double x1101 = R_l_inv_41*a_2*x171;
        const double x1102 = x1059*x317;
        const double x1103 = -x1099 - x1100 + x1101 + x1102;
        const double x1104 = x1064*x317;
        const double x1105 = R_l_inv_43*x34*x68;
        const double x1106 = R_l_inv_40*a_2*x123;
        const double x1107 = x1068*x321;
        const double x1108 = -x1104 + x1105 - x1106 + x1107;
        const double x1109 = R_l_inv_43*x41;
        const double x1110 = R_l_inv_46*x1056;
        const double x1111 = R_l_inv_40*x757;
        const double x1112 = R_l_inv_47*x1059;
        const double x1113 = R_l_inv_41*x786 - R_l_inv_44*x34 + R_l_inv_46*x1068 + R_l_inv_47*x1064;
        const double x1114 = x1104 - x1105 + x1106 - x1107;
        const double x1115 = x1099 + x1100 - x1101 - x1102;
        const double x1116 = x193*x46;
        const double x1117 = R_l_inv_43*x1116;
        const double x1118 = R_l_inv_47*d_4*x1116;
        const double x1119 = 4*x1068;
        const double x1120 = x1119 + x332*x786 - x334*x34;
        const double x1121 = 8*x1059;
        const double x1122 = x1087 + x1088 + x1090 + x1091 + x1092 + x1093 + x1094 - x1095 + x1097 + x148 + x150 + x152;
        const double x1123 = -2*Pz*r_32*x34 + x107*x41 + x1089*x24 - x27*x34*x87;
        const double x1124 = -x1123;
        const double x1125 = x46*x681;
        const double x1126 = 4*x19 - 4*x21;
        const double x1127 = x1126*x27;
        const double x1128 = -4*Pz*r_33*x41 + x1125 - x1126*x30*x41 + x1127*x46;
        const double x1129 = 8*x1054;
        const double x1130 = x34*x49;
        const double x1131 = 8*x1130;
        const double x1132 = 8*x19 - 8*x21;
        const double x1133 = x1132*x24;
        const double x1134 = x1132*x30;
        const double x1135 = 2*x44 + 2*x45;
        const double x1136 = x1135*x376;
        const double x1137 = x1096*x378;
        const double x1138 = 2*x32 + 2*x33;
        const double x1139 = 2*x4;
        const double x1140 = std::pow(x34, 3);
        const double x1141 = 2*x34;
        const double x1142 = a_2*x419;
        const double x1143 = x1126*x24;
        const double x1144 = x1143*x36;
        const double x1145 = x36*x484;
        const double x1146 = x222*x7;
        const double x1147 = x548*x7;
        const double x1148 = 4*x41;
        const double x1149 = 4*x46;
        const double x1150 = x34*x82;
        const double x1151 = 4*x63;
        const double x1152 = 4*x34;
        const double x1153 = x55*x56;
        const double x1154 = x100*x1147 + x100*x634 + x1054*x1151 + x1055*x941 + x1064*x491 - x1068*x510 + x1078*x535 + x1079*x681 - x11*x1141 - x1130*x1151 - x1138*x413 + x1138*x43 + x1138*x48 - x1139*x34 + x1140*x447 - x1141*x13 + x1141*x15 - x1141*x16 - x1141*x17 + x1141*x26 - x1141*x29 - x1141*x31 + x1141*x9 + x1142*x432 + x1143*x469 + x1144*x42 + x1144*x47 + x1145*x42 + x1145*x47 + x1146*x504 + x1148*x445 + x1149*x451 - x1150*x681 + x1152*x1153 + x222*x505 + x469*x484;
        const double x1155 = -x1136 + x1137 + x1154;
        const double x1156 = 2*x39 + 2*x40;
        const double x1157 = x1156*x386;
        const double x1158 = x1056*x566;
        const double x1159 = 2*x757;
        const double x1160 = x1159*x389;
        const double x1161 = 2*x1059;
        const double x1162 = x1161*x394;
        const double x1163 = x1054*x218;
        const double x1164 = x1055*x218;
        const double x1165 = x1130*x218;
        const double x1166 = x34*x63;
        const double x1167 = x1166*x218;
        const double x1168 = x1157 + x1158 - x1160 - x1162 - x1163 - x1164 + x1165 + x1167;
        const double x1169 = 8*x372;
        const double x1170 = d_2*x236;
        const double x1171 = R_l_inv_37*x1169 + R_l_inv_77*x1170 - 4*x383;
        const double x1172 = 4*x32 + 4*x33;
        const double x1173 = x1172*x386;
        const double x1174 = x1062*x233;
        const double x1175 = x1063*x233;
        const double x1176 = x41*x49;
        const double x1177 = x41*x63;
        const double x1178 = 4*x39 + 4*x40;
        const double x1179 = R_l_inv_36*x1169 + R_l_inv_76*x1170 - 4*x393;
        const double x1180 = 4*x4;
        const double x1181 = std::pow(x41, 3);
        const double x1182 = 4*x1059;
        const double x1183 = x1132*x27;
        const double x1184 = x1183*x36;
        const double x1185 = 8*x81;
        const double x1186 = x1185*x36;
        const double x1187 = x1020*x346;
        const double x1188 = x1022*x92;
        const double x1189 = 8*x34;
        const double x1190 = 8*x46;
        const double x1191 = 8*x63;
        const double x1192 = x34*x81;
        const double x1193 = 8*x41;
        const double x1194 = x81*x82;
        const double x1195 = -x1008*x1078 + x1008*x1192 - x1056*x1179 + x1062*x1191 + x1063*x689 + x11*x1148 + x1142*x504 - x1148*x13 - x1148*x15 + x1148*x16 - x1148*x17 - x1148*x26 + x1148*x29 - x1148*x31 + x1148*x38 - x1148*x9 + x1150*x589 - x1176*x1191 - x1178*x413 + x1178*x48 - x1180*x41 + x1181*x545 - x1182*x384 + x1183*x469 + x1184*x42 + x1184*x47 + x1185*x469 + x1186*x42 + x1186*x47 + x1187*x14 + x1187*x7 + x1188*x14 + x1188*x7 + x1189*x445 + x1190*x549 + x1193*x1194;
        const double x1196 = x1136 - x1137 + x1154;
        const double x1197 = x1172*x376;
        const double x1198 = x378*x787;
        const double x1199 = 4*x44 + 4*x45;
        const double x1200 = std::pow(x46, 3);
        const double x1201 = x1134*x36;
        const double x1202 = x36*x689;
        const double x1203 = x1020*x406;
        const double x1204 = x356*x7;
        const double x1205 = 8*x82;
        const double x1206 = x49*x63;
        const double x1207 = -8*Pz*r_31*x22*x24*x46 - 8*Pz*r_32*x22*x27*x46 - 4*d_4*x384*x46 - 4*x10*x14*x46 - 4*x10*x46*x7 + x1015*x1204 + x1016*x356 + x1080*x1179 + x1131*x56 + x1134*x469 + x1149*x13 + x1149*x17 + x1149*x31 + x1149*x38 + x1149*x43 + x1166*x589 + x1176*x1205 + x1177*x1185 + x1189*x451 + x1190*x1206 + x1193*x549 - x1199*x413 + x1200*x545 + x1201*x42 + x1201*x47 + x1202*x42 + x1202*x47 + x1203*x14 + x1203*x7 - 4*x14*x46*x8 - 4*x23*x25*x46 - 4*x23*x28*x46 - 4*x4*x46 + x419*x686 - 4*x46*x7*x8 + x469*x689;
        const double x1208 = 8*x39 + 8*x40;
        const double x1209 = -x1157 - x1158 + x1160 + x1162 + x1163 + x1164 - x1165 - x1167;
        const double x1210 = x1135*x699;
        const double x1211 = x1096*x703;
        const double x1212 = 2*x41;
        const double x1213 = x1127*x36;
        const double x1214 = x36*x681;
        const double x1215 = x543*x7;
        const double x1216 = x46*x535;
        const double x1217 = -x1056*x774 - x1078*x559 + x11*x1212 + x1125*x63 + x1127*x469 - x1139*x41 + x1147*x92 + x1149*x549 + x1152*x445 + x1156*x48 - x1156*x725 + x1159*x728 + x1161*x706 - 4*x1176*x63 + x1181*x447 - x1212*x13 - x1212*x15 + x1212*x16 - x1212*x17 - x1212*x26 + x1212*x29 - x1212*x31 + x1212*x38 - x1212*x9 + x1213*x42 + x1213*x47 + x1214*x42 + x1214*x47 + x1215*x346 + x1216*x49 + x34*x535*x55 + x34*x56*x681 + x346*x783 + x41*x681*x82 + x469*x681 + x634*x92;
        const double x1218 = x1210 - x1211 + x1217;
        const double x1219 = x1138*x697;
        const double x1220 = x1064*x810;
        const double x1221 = a_2*x701;
        const double x1222 = x1221*x432;
        const double x1223 = x1068*x780;
        const double x1224 = d_4*x1125;
        const double x1225 = d_4*x1216;
        const double x1226 = x1176*x218;
        const double x1227 = x1177*x218;
        const double x1228 = x1219 - x1220 - x1222 + x1223 - x1224 - x1225 + x1226 + x1227;
        const double x1229 = x1178*x697;
        const double x1230 = R_l_inv_26*x1169 + R_l_inv_66*x1170 - 4*x705;
        const double x1231 = x1056*x1230;
        const double x1232 = x1182*x712;
        const double x1233 = x1221*x504;
        const double x1234 = x1054*x233;
        const double x1235 = x1055*x233;
        const double x1236 = x1130*x233;
        const double x1237 = x1166*x233;
        const double x1238 = x1180*x34;
        const double x1239 = x1133*x36;
        const double x1240 = x36*x589;
        const double x1241 = x100*x1022;
        const double x1242 = x1015*x1146 + x1016*x222 + x1055*x689 + x1078*x1205 + x1079*x1185 - x11*x1152 + x1129*x63 - x1131*x63 + x1133*x469 + x1140*x545 - x1152*x13 + x1152*x15 - x1152*x16 - x1152*x17 + x1152*x26 - x1152*x29 - x1152*x31 + x1152*x9 + x1153*x1189 + x1172*x43 + x1172*x48 + x1190*x451 - x1192*x1205 + x1193*x445 + x1239*x42 + x1239*x47 + x1240*x42 + x1240*x47 + x1241*x14 + x1241*x7 + x469*x589;
        const double x1243 = x1064*(R_l_inv_27*x1169 + R_l_inv_67*x1170 - 4*x711) + x1119*x706 - x1172*x725 - x1238 + x1242 + x728*x787;
        const double x1244 = -x1210 + x1211 + x1217;
        const double x1245 = x1172*x699;
        const double x1246 = x703*x787;
        const double x1247 = -8*Pz*d_4*r_32*x34 - 8*d_4*x22*x27*x34 - 4*d_4*x46*x712 + x1080*x1230 + x1121*x55 + x1121*x56 - x1199*x697 + x686*x701;
        const double x1248 = -x1219 + x1220 + x1222 - x1223 + x1224 + x1225 - x1226 - x1227;
        const double x1249 = x1138*x860;
        const double x1250 = x1135*x886;
        const double x1251 = x1064*x968;
        const double x1252 = x4*x453;
        const double x1253 = x1200*x447;
        const double x1254 = a_2*x863;
        const double x1255 = x1254*x432;
        const double x1256 = -x1096*x889;
        const double x1257 = -x453*x9;
        const double x1258 = -x11*x453;
        const double x1259 = -x15*x453;
        const double x1260 = -x16*x453;
        const double x1261 = -x26*x453;
        const double x1262 = -x29*x453;
        const double x1263 = x1068*x965;
        const double x1264 = x13*x453;
        const double x1265 = x17*x453;
        const double x1266 = x31*x453;
        const double x1267 = x38*x453;
        const double x1268 = x43*x453;
        const double x1269 = x1126*x30;
        const double x1270 = x1269*x469;
        const double x1271 = x1269*x36;
        const double x1272 = x1271*x42;
        const double x1273 = x1271*x47;
        const double x1274 = x469*x941;
        const double x1275 = x36*x941;
        const double x1276 = x1275*x42;
        const double x1277 = x1275*x47;
        const double x1278 = x1215*x406;
        const double x1279 = x406*x783;
        const double x1280 = x1204*x504;
        const double x1281 = x356*x505;
        const double x1282 = x1152*x451;
        const double x1283 = x1148*x549;
        const double x1284 = -x1054*x559;
        const double x1285 = -x1063*x681;
        const double x1286 = x1166*x484;
        const double x1287 = x1177*x681;
        const double x1288 = x1130*x559;
        const double x1289 = x1176*x535;
        const double x1290 = x1149*x1206;
        const double x1291 = x1249 + x1250 - x1251 + x1252 + x1253 - x1255 + x1256 + x1257 + x1258 + x1259 + x1260 + x1261 + x1262 + x1263 + x1264 + x1265 + x1266 + x1267 + x1268 + x1270 + x1272 + x1273 + x1274 + x1276 + x1277 + x1278 + x1279 + x1280 + x1281 + x1282 + x1283 + x1284 + x1285 + x1286 + x1287 + x1288 + x1289 + x1290;
        const double x1292 = x1156*x871;
        const double x1293 = x1056*x959;
        const double x1294 = x1159*x875;
        const double x1295 = x1161*x866;
        const double x1296 = -x1292 - x1293 + x1294 + x1295;
        const double x1297 = x218*x469;
        const double x1298 = x218*x36;
        const double x1299 = x1298*x42;
        const double x1300 = x1298*x47;
        const double x1301 = x1119*x55;
        const double x1302 = x1059*x681;
        const double x1303 = x218*x46;
        const double x1304 = x1303*x49;
        const double x1305 = x1119*x56;
        const double x1306 = x1059*x535;
        const double x1307 = x1303*x63;
        const double x1308 = -x1297 - x1299 - x1300 - x1301 - x1302 - x1304 - x1305 - x1306 - x1307;
        const double x1309 = x1178*x860;
        const double x1310 = -4*x865 + x95*x993;
        const double x1311 = x1056*x1310;
        const double x1312 = x1182*x869;
        const double x1313 = x1254*x504;
        const double x1314 = x1064*(x102*x993 - 4*x868) + x1119*x866 - x1172*x871 + x787*x875;
        const double x1315 = x1292 + x1293 - x1294 - x1295;
        const double x1316 = -x1249 + x1250 + x1251 + x1252 + x1253 + x1255 + x1256 + x1257 + x1258 + x1259 + x1260 + x1261 + x1262 - x1263 + x1264 + x1265 + x1266 + x1267 + x1268 + x1270 + x1272 + x1273 + x1274 + x1276 + x1277 + x1278 + x1279 + x1280 + x1281 + x1282 + x1283 + x1284 + x1285 + x1286 + x1287 + x1288 + x1289 + x1290;
        const double x1317 = x1199*x860;
        const double x1318 = x1303*x869;
        const double x1319 = x1172*x886 + x1238 + x1242 - x787*x889;
        const double x1320 = 16*x7;
        const double x1321 = r_31*x346;
        const double x1322 = r_33*x92;
        const double x1323 = 16*x14;
        const double x1324 = 16*x56;
        const double x1325 = 16*x63;
        const double x1326 = 16*x81;
        const double x1327 = x1326*x36;
        const double x1328 = x27*(16*x19 - 16*x21);
        const double x1329 = x1328*x36;
        const double x1330 = x1297 + x1299 + x1300 + x1301 + x1302 + x1304 + x1305 + x1306 + x1307;
        const double x1331 = -x149 - x151 - x153 - x154 - x156 - x157 - x159 - x161 - x162;
        const double x1332 = x1331 + x243;
        const double x1333 = x192 + x247;
        const double x1334 = x1331 + x51 + x70 + x73 + x76;
        const double x1335 = x265 + x266 + x267 + x268 + x269 + x270 + x271 + x272 + x273 + x274 + x281 + x312 + x338;
        const double x1336 = x301 + x339;
        const double x1337 = x276 + x278 + x280 + x310 + x311 + x338;
        const double x1338 = -x324 - x325;
        const double x1339 = -x204 + x205 - x206 + x207 + x348;
        const double x1340 = x351 + x353 - x354 - x355 + x360;
        const double x1341 = -x1339;
        const double x1342 = x380 + x487 + x611;
        const double x1343 = -x377 - x381;
        const double x1344 = x387 + x391 + x392 + x396 + x400 + x402 + x404 + x407 + x604 + x606 + x608 + x609;
        const double x1345 = x577 - x579 + x581 + x583 + x585 + x586 - x587 - x588 + x590 + x593 + x594 + x595 + x596 + x598 + x600 - x601 - x602;
        const double x1346 = x487 + x611;
        const double x1347 = -x379 + x693;
        const double x1348 = x680 + x690;
        const double x1349 = x385 + x388 + x390 + x395 + x397 + x399 + x403 + x405 + x408 + x410 + x605 + x607;
        const double x1350 = x752 + x848;
        const double x1351 = x785 + x788 + x790 + x791 - x792 - x793 + x794 + x795 + x796 + x797 + x798 + x799 + x800 - x801 - x802;
        const double x1352 = x1351 + x773 + x775 + x777 + x778 + x779 + x781 + x782 + x784;
        const double x1353 = x753 + x847;
        const double x1354 = -4*a_2*d_4*r_31*x41 - 2*a_2*x223*x706 - x613*x701 + x841;
        const double x1355 = x624 + x628 + x630 + x647 + x649 + x651 + x653 + x655 + x657 + x887 + x890 + x891 + x892 + x893 + x894 + x895 + x897 + x900 + x902 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 + x914 + x916 + x917 + x918 + x919 + x920 + x921 + x932 + x975 + x976 + x977 + x978 + x980;
        const double x1356 = x1032 + x974;
        const double x1357 = x1351 - x991;
        const double x1358 = x922 + x923 + x924 + x925 + x926 + x927 + x928 + x929 + x930 + x931 + x980;
        
        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = x115 + x146 + x177;
        A(0, 1) = x193*(-x178 + x179 - x187 - x192);
        A(0, 2) = x196 + x199 + x200 + x201;
        A(0, 3) = x203 + x211 + x213 + x216 - x227 - x232;
        A(0, 4) = -x234 - x235 - x237 - x239;
        A(0, 5) = x210*x240 + x212*x65 + x219 + x221 + x226 + x242;
        A(0, 6) = x177 + x244 + x245 + x246;
        A(0, 7) = x193*(-x187 + x188 - x189 + x190 + x191 - x247);
        A(0, 8) = x201 + x244 + x248 + x249 + x250;
        A(1, 0) = x264 + x276 + x281 + x286;
        A(1, 1) = x68*(x287 - x288 + x296 + x301);
        A(1, 2) = x304 + x309 + x314;
        A(1, 3) = -x323 - x326 - x331;
        A(1, 4) = x332*x84 + x335;
        A(1, 5) = -x326 - x336 - x337;
        A(1, 6) = -x264 + x265 + x266 + x267 + x268 + x269 + x270 + x271 + x272 + x273 - x313 - x338;
        A(1, 7) = x68*(x296 - x297 + x298 - x299 - x300 + x339);
        A(1, 8) = x314 + x341 + x344;
        A(2, 0) = x349;
        A(2, 2) = x349;
        A(2, 3) = x361;
        A(2, 4) = x363 + x365 - x366 - x367 - x370;
        A(2, 5) = -x361;
        A(2, 6) = x371;
        A(2, 8) = x371;
        A(3, 0) = -x382 - x411 - x488;
        A(3, 1) = x565 + x575 + x603;
        A(3, 2) = -x382 - x610 - x612;
        A(3, 3) = 8*Pz*a_2*r_31*r_33*x34 + 8*Pz*a_2*r_32*r_33*x41 + 4*Pz*a_2*x12*x46 + 4*a_2*r_31*x22*x30*x34 + 4*a_2*r_32*x22*x30*x41 + 4*a_2*r_33*x22*x24*x34 + 4*a_2*r_33*x22*x27*x41 + 4*a_2*r_33*x22*x30*x46 + 4*a_2*r_33*x35*x36 + 4*a_2*r_33*x36*x42 + 4*a_2*r_33*x36*x47 + 2*a_2*x223*x394 + x419*x613 - x614 - x615 - x616 - x618 - x622 - x680;
        A(3, 4) = -x683 - x684;
        A(3, 5) = x622 + x623 + x625 + x626 - x629 - x631 + x633 + x635 + x636 + x637 + x638 + x640 + x642 + x644 + x646 - x648 - x650 - x652 - x654 - x656 - x658 - x660 - x661 - x663 - x664 - x666 - x667 - x668 - x669 - x670 - x672 + x673 + x674 - x675 - x676 - x677 - x678 - x679 + x690;
        A(3, 6) = x488 + x610 + x694;
        A(3, 7) = -x603 - x695 - x696;
        A(3, 8) = x411 + x612 + x694;
        A(4, 0) = -x723 - x753 - x772;
        A(4, 1) = x804 + x809 + x821;
        A(4, 2) = x823 + x825 + x827;
        A(4, 3) = x837 + x840 + x842;
        A(4, 4) = -x843 - x844;
        A(4, 5) = x842 + x845 + x846;
        A(4, 6) = x495 - x500 - x502 + x513 + x515 + x517 + x519 - x521 - x523 - x525 - x527 - x529 - x531 + x723 - x724 + x726 - x727 + x729 + x730 + x732 + x733 - x734 - x735 - x736 - x737 - x738 - x739 - x740 - x742 - x743 - x744 + x745 + x746 - x747 - x748 - x749 - x750 - x751 + x848;
        A(4, 7) = x804 + x849 + x850;
        A(4, 8) = x823 + x854 + x857;
        A(5, 0) = x885 + x922 + x932 + x952;
        A(5, 1) = x955 - x956 + x967 + x974;
        A(5, 2) = x624 + x628 + x630 - x639 - x641 - x643 - x645 + x647 + x649 + x651 + x653 + x655 + x657 - x885 + x887 + x891 - x896 - x899 - x901 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 - x913 - x915 + x917 + x918 + x919 + x920 + x921 - x979 - x980;
        A(5, 3) = x985 + x990 + x992;
        A(5, 4) = -x1000 - x1001 - x1003 - x1005 - x1007 + x1008*x345 - x1009 - x1010 + x1011*x356 - x1012 - x1014 + x1023 + x265*x995 + x267*x995 + x84*x994 - x996 - x998;
        A(5, 5) = -x1024 - x1025 - x992;
        A(5, 6) = -x1027 - x1030 - x1031;
        A(5, 7) = x1032 + x967 + x969 - x971 - x972 - x973;
        A(5, 8) = -x1031 - x1035 - x1038;
        
        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = -x1052 - x1061 - x1070;
        B(0, 1) = x236*(-x1071 - x1072 + x1073 - x1074 + x1075);
        B(0, 2) = -x1052 - x1076 - x1077;
        B(0, 3) = d_4*x103*x1082 - x1081 + x1082*x121 - x1083 + x1084;
        B(0, 4) = -a_4*x1085*x127 + x1004*x170*x67 + x1004*x173;
        B(0, 5) = 8*a_4*d_4*x103*x46 + 8*a_4*x121*x46 - x1081 - x1083 - x1084;
        B(0, 6) = -x1061 - x1076 - x1086;
        B(0, 7) = x236*(x1071 + x1072 - x1073 + x1074 + x1075);
        B(0, 8) = -x1070 - x1077 - x1086;
        B(1, 0) = x1098 + x1103 + x1108;
        B(1, 1) = x193*(x1109 + x1110 - x1111 + x1112 - x1113);
        B(1, 2) = x1098 + x1114 + x1115;
        B(1, 3) = 4*R_l_inv_40*a_2*a_4*r_33 + 4*R_l_inv_46*a_4*x1080 - x1117 - x1118 - x1120;
        B(1, 4) = 8*R_l_inv_45*a_4*x41 - x1121 - x253*x997;
        B(1, 5) = R_l_inv_40*x1044*x193 + R_l_inv_46*x1080*x193 - x1117 - x1118 + x1120;
        B(1, 6) = x1103 + x1114 + x1122;
        B(1, 7) = x193*(-x1109 - x1110 + x1111 - x1112 - x1113);
        B(1, 8) = x1108 + x1115 + x1122;
        B(2, 0) = x1124;
        B(2, 2) = x1124;
        B(2, 3) = -x1128;
        B(2, 4) = x1129 - x1131 + x1133*x46 - x1134*x34;
        B(2, 5) = x1128;
        B(2, 6) = x1123;
        B(2, 8) = x1123;
        B(3, 0) = x1155 + x1168;
        B(3, 1) = x1064*x1171 + x1119*x394 - x1173 - x1174 - x1175 + x1176*x233 + x1177*x233 + x1195 + x389*x787;
        B(3, 2) = -x1168 - x1196;
        B(3, 3) = -x1197 + x1198 - x1207;
        B(3, 4) = -x1208*x376 + x378*x997;
        B(3, 5) = x1197 - x1198 - x1207;
        B(3, 6) = -x1155 - x1209;
        B(3, 7) = 8*Pz*d_4*r_33*x41 + 4*a_2*r_31*x389 + 8*d_4*x22*x30*x41 + 4*d_4*x34*x394 + x1064*x1171 - x1173 - x1174 - x1175 - x1195;
        B(3, 8) = x1196 + x1209;
        B(4, 0) = -x1218 - x1228;
        B(4, 1) = -x1229 - x1231 - x1232 + x1233 - x1234 - x1235 + x1236 + x1237 + x1243;
        B(4, 2) = x1228 + x1244;
        B(4, 3) = -x1245 + x1246 - x1247;
        B(4, 4) = -x1208*x699 + x703*x997;
        B(4, 5) = x1245 - x1246 - x1247;
        B(4, 6) = -x1244 - x1248;
        B(4, 7) = x1229 + x1231 + x1232 - x1233 + x1234 + x1235 - x1236 - x1237 + x1243;
        B(4, 8) = x1218 + x1248;
        B(5, 0) = x1291 + x1296 + x1308;
        B(5, 1) = x1309 + x1311 + x1312 - x1313 - x1314;
        B(5, 2) = x1308 + x1315 + x1316;
        B(5, 3) = x1080*x1310 - x1317 - x1318 + x1319 + x686*x863;
        B(5, 4) = x1002*x1150 + x1021*x1181 + x1062*x1325 + 16*x1063*x49 - x1078*x1324 + x1085*x1194 + x11*x1193 - x1176*x1325 + x1192*x1324 - x1193*x13 - x1193*x15 + x1193*x16 - x1193*x17 - x1193*x26 + x1193*x29 - x1193*x31 + x1193*x38 + x1193*x4 - x1193*x9 + x1208*x48 + x1208*x886 + x1320*x1321 + x1320*x1322 + x1321*x1323 + x1322*x1323 + x1326*x469 + x1327*x42 + x1327*x47 + x1328*x469 + x1329*x42 + x1329*x47 + 16*x34*x445 + 16*x46*x549 - x889*x997;
        B(5, 5) = 4*a_2*r_33*x863 + x1080*x1310 - x1317 - x1318 - x1319;
        B(5, 6) = -x1291 - x1315 - x1330;
        B(5, 7) = -x1309 - x1311 - x1312 + x1313 - x1314;
        B(5, 8) = -x1296 - x1316 - x1330;
        
        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = x115 + x1332 + x196;
        C(0, 1) = x193*(x1333 + x180 - x181 + x182 + x183 - x184 + x186);
        C(0, 2) = x1332 + x176 + x199 + x248;
        C(0, 3) = -x217 + x219 + x221 - x224 + x226 - x242;
        C(0, 4) = x234 + x235 + x237 - x239;
        C(0, 5) = -x227 + x228 + x229 + x230 + x231 - x241;
        C(0, 6) = x1334 + x146 + x245 + x249;
        C(0, 7) = x193*(-x1333 + x180 - x181 + x182 + x183 - x184 - x185);
        C(0, 8) = x1334 + x176 + x200 + x246 + x250;
        C(1, 0) = -x1335 - x251 - x256 - x259 - x302 - x303 - x340 - x342;
        C(1, 1) = x68*(R_l_inv_41*x58 + R_l_inv_47*a_2*x93 - x1336 - x290 - x291 - x292 - x295);
        C(1, 2) = -x1337 - x309 - x341;
        C(1, 3) = x1338 + x331 + x337;
        C(1, 4) = 4*R_l_inv_42*a_4*x84 - x335;
        C(1, 5) = x1338 + x323 + x336;
        C(1, 6) = -x1335 - x252 - x254 - x261 - x305 - x306 - x307 - x308 - x343;
        C(1, 7) = x68*(x1336 + x289 - x290 - x291 - x292 + x293 + x294);
        C(1, 8) = -x1337 - x304 - x344;
        C(2, 0) = x1339;
        C(2, 2) = x1339;
        C(2, 3) = x1340;
        C(2, 4) = -x363 - x365 + x366 + x367 - x370;
        C(2, 5) = -x1340;
        C(2, 6) = x1341;
        C(2, 8) = x1341;
        C(3, 0) = -x1342 - x1343 - x1344 - x379;
        C(3, 1) = -x1345 - x575 - x696;
        C(3, 2) = x1344 + x1346 + x1347 + x691;
        C(3, 3) = x1348 - x619 + x620 + x621;
        C(3, 4) = -x683 + x684;
        C(3, 5) = x1348 + x619 - x620 - x621;
        C(3, 6) = x1343 + x1346 + x1349 + x692;
        C(3, 7) = x1345 + x565 + x695;
        C(3, 8) = -x1342 - x1347 - x1349;
        C(4, 0) = x1350 + x698 + x707 + x708 + x713 + x714 + x717 + x719 + x721 + x722 + x824 + x851 + x852 + x853 + x855;
        C(4, 1) = -x1352 - x821 - x849;
        C(4, 2) = -x1353 - x827 - x854;
        C(4, 3) = -x1354 - x840 - x846;
        C(4, 4) = -x843 + x844;
        C(4, 5) = -x1354 - x837 - x845;
        C(4, 6) = x1350 + x702 + x704 + x710 + x715 + x720 + x826 + x856;
        C(4, 7) = -x1352 - x809 - x850;
        C(4, 8) = -x1353 - x825 - x857;
        C(5, 0) = -x1026 - x1033 - x1034 - x1036 - x1355 - x861 - x867 - x876 - x878 - x881;
        C(5, 1) = 2*a_2*x869*x93 - x1356 + x58*x957 - x960 - x961 - x963 - x964;
        C(5, 2) = -x1028 - x1029 - x1037 - x1355 - x864 - x872 - x877 - x879 - x880 - x883;
        C(5, 3) = -x1025 - x1357 - x990;
        C(5, 4) = 8*Pz*a_2*x12*x41 + 8*Pz*a_2*x41*x8 + 8*a_2*r_31*x22*x24*x41 + 8*a_2*r_33*x22*x30*x41 - x1000 - x1001 - x1003 - x1005 - x1007 - x1009 - x1010 - x1012 - x1014 - x1023 + x84*x994 - x996 - x998;
        C(5, 5) = x1024 + x1357 + x985;
        C(5, 6) = x1030 + x1035 + x1358;
        C(5, 7) = x1356 + x958 - x960 - x961 + x962 - x964 + x966;
        C(5, 8) = x1027 + x1038 + x1358;
        
        // Invoke the solver
        std::array<double, 16> solution_buffer;
        int n_solutions = yaik_cpp::general_6dof_internal::computeSolutionFromTanhalfLME(A, B, C, &solution_buffer);
        
        for(auto i = 0; i < n_solutions; i++)
        {
            auto solution_i = make_raw_solution();
            solution_i[2] = solution_buffer[i];
            int appended_idx = append_solution_to_queue(solution_i);
            add_input_index_to(2, appended_idx);
        };
    };
    // Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_1_processor();
    // Finish code for general_6dof solution node 0
    
    // Code for non-branch dispatcher node 2
    // Actually, there is no code
    
    // Code for explicit solution node 3, solved variable is th_3
    auto ExplicitSolutionNode_node_3_solve_th_3_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(2);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(2);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 3
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            
            const bool condition_0 = std::fabs(a_4) >= zero_tolerance || std::fabs(d_4) >= zero_tolerance || std::fabs(Px*std::sin(th_1)*std::cos(th_0) + Py*std::sin(th_0)*std::sin(th_1) - Pz*std::cos(th_1) - a_1*std::sin(th_1) + d_2) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::atan2(a_4, d_4);
                const double x1 = std::sin(th_1);
                const double x2 = Px*x1*std::cos(th_0) + Py*x1*std::sin(th_0) - Pz*std::cos(th_1) - a_1*x1 + d_2;
                const double x3 = std::sqrt(std::pow(a_4, 2) + std::pow(d_4, 2) - std::pow(x2, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[7] = x0 + std::atan2(x3, x2);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = std::fabs(a_4) >= zero_tolerance || std::fabs(d_4) >= zero_tolerance || std::fabs(Px*std::sin(th_1)*std::cos(th_0) + Py*std::sin(th_0)*std::sin(th_1) - Pz*std::cos(th_1) - a_1*std::sin(th_1) + d_2) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::atan2(a_4, d_4);
                const double x1 = std::sin(th_1);
                const double x2 = Px*x1*std::cos(th_0) + Py*x1*std::sin(th_0) - Pz*std::cos(th_1) - a_1*x1 + d_2;
                const double x3 = std::sqrt(std::pow(a_4, 2) + std::pow(d_4, 2) - std::pow(x2, 2));
                // End of temp variables
                const double tmp_sol_value = x0 + std::atan2(-x3, x2);
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
                add_input_index_to(4, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_3_processor();
    // Finish code for explicit solution node 2
    
    // Code for solved_variable dispatcher node 4
    auto SolvedVariableDispatcherNode_node_4_processor = [&]() -> void
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
            bool taken_by_degenerate = false;
            const double th_3 = this_solution[7];
            
            const bool degenerate_valid_0 = std::fabs(th_3 - M_PI + 3.0801531643337001) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(12, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_3 - 3.3383391319530303e-5 + M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(27, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(5, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_4_processor();
    // Finish code for solved_variable dispatcher node 4
    
    // Code for explicit solution node 27, solved variable is th_2
    auto ExplicitSolutionNode_node_27_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(27);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(27);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 27
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            
            const bool condition_0 = std::fabs(Px*std::sin(th_0) - Py*std::cos(th_0)) >= zero_tolerance || std::fabs(a_3 - 0.99999999944277496*a_4 + 3.3383391313329601e-5*d_4) >= zero_tolerance || std::fabs(1.0*a_3 - 0.99999999944277496*a_4 + 3.3383391313329601e-5*d_4) >= zero_tolerance || std::fabs(Px*std::cos(th_0)*std::cos(th_1) + Py*std::sin(th_0)*std::cos(th_1) + Pz*std::sin(th_1) - a_1*std::cos(th_1) - a_2) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double x2 = -0.99999999944277496*a_4 + 3.3383391313329601e-5*d_4;
                const double x3 = std::cos(th_1);
                // End of temp variables
                const double tmp_sol_value = std::atan2((-Px*x0 + Py*x1)/(1.0*a_3 + x2), (Px*x1*x3 + Py*x0*x3 + Pz*std::sin(th_1) - a_1*x3 - a_2)/(a_3 + x2));
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(28, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_2_processor();
    // Finish code for explicit solution node 27
    
    // Code for non-branch dispatcher node 28
    // Actually, there is no code
    
    // Code for explicit solution node 29, solved variable is th_5
    auto ExplicitSolutionNode_node_29_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(28);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(28);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 29
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = std::fabs(r_13*((0.99999999944277496*std::sin(th_1) - 3.3383391313329601e-5*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + 3.3383391313329601e-5*std::sin(th_0)*std::sin(th_2)) + r_23*((0.99999999944277496*std::sin(th_1) - 3.3383391313329601e-5*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - 3.3383391313329601e-5*std::sin(th_2)*std::cos(th_0)) - r_33*(3.3383391313329601e-5*std::sin(th_1)*std::cos(th_2) + 0.99999999944277496*std::cos(th_1))) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_1);
                const double x2 = 3.3383391313329601e-5*std::cos(th_2);
                const double x3 = std::sin(th_0);
                const double x4 = 3.3383391313329601e-5*std::sin(th_2);
                const double x5 = std::cos(th_0);
                const double x6 = -x0*x2 + 0.99999999944277496*x1;
                const double x7 = std::acos(r_13*(x3*x4 + x5*x6) + r_23*(x3*x6 - x4*x5) - r_33*(0.99999999944277496*x0 + x1*x2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[10] = x7;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(30, appended_idx);
            }
            
            const bool condition_1 = std::fabs(r_13*((0.99999999944277496*std::sin(th_1) - 3.3383391313329601e-5*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + 3.3383391313329601e-5*std::sin(th_0)*std::sin(th_2)) + r_23*((0.99999999944277496*std::sin(th_1) - 3.3383391313329601e-5*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - 3.3383391313329601e-5*std::sin(th_2)*std::cos(th_0)) - r_33*(3.3383391313329601e-5*std::sin(th_1)*std::cos(th_2) + 0.99999999944277496*std::cos(th_1))) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_1);
                const double x2 = 3.3383391313329601e-5*std::cos(th_2);
                const double x3 = std::sin(th_0);
                const double x4 = 3.3383391313329601e-5*std::sin(th_2);
                const double x5 = std::cos(th_0);
                const double x6 = -x0*x2 + 0.99999999944277496*x1;
                const double x7 = std::acos(r_13*(x3*x4 + x5*x6) + r_23*(x3*x6 - x4*x5) - r_33*(0.99999999944277496*x0 + x1*x2));
                // End of temp variables
                const double tmp_sol_value = -x7;
                solution_queue.get_solution(node_input_i_idx_in_queue)[10] = tmp_sol_value;
                add_input_index_to(30, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_5_processor();
    // Finish code for explicit solution node 28
    
    // Code for solved_variable dispatcher node 30
    auto SolvedVariableDispatcherNode_node_30_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(30);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(30);
        if (!this_input_valid)
            return;
        
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_5 = this_solution[10];
            
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
                add_input_index_to(31, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_30_processor();
    // Finish code for solved_variable dispatcher node 30
    
    // Code for explicit solution node 31, solved variable is th_4
    auto ExplicitSolutionNode_node_31_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(31);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(31);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 31
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_5 = this_solution[10];
            
            const bool condition_0 = std::fabs(r_13*(std::sin(th_0)*std::cos(th_2) + std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_23*(-std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_33*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(r_13*std::sin(th_1)*std::cos(th_0) + r_23*std::sin(th_0)*std::sin(th_1) - r_33*std::cos(th_1) - 0.99999999944277496*std::cos(th_5)) >= zero_tolerance || 3.3383391313329601e-5*std::fabs(std::sin(th_5)) >= zero_tolerance || std::fabs(std::sin(th_5)) >= zero_tolerance;
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
                const double x8 = 29955.015373189799*x1;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(-r_13*(x3*x4 + x5*x7) + r_23*(-x3*x7 + x4*x5) - r_33*x1*x2), x0*(r_13*x5*x8 + r_23*x3*x8 - 29955.015373189799*r_33*x6 - 29955.015356498116*std::cos(th_5)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
                add_input_index_to(32, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_4_processor();
    // Finish code for explicit solution node 31
    
    // Code for non-branch dispatcher node 32
    // Actually, there is no code
    
    // Code for explicit solution node 33, solved variable is th_2th_4th_5_soa
    auto ExplicitSolutionNode_node_33_solve_th_2th_4th_5_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(32);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(32);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 33
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[8];
            const double th_5 = this_solution[10];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_2 + th_4 + th_5;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(34, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_33_solve_th_2th_4th_5_soa_processor();
    // Finish code for explicit solution node 32
    
    // Code for non-branch dispatcher node 34
    // Actually, there is no code
    
    // Code for explicit solution node 35, solved variable is th_0th_2th_4_soa
    auto ExplicitSolutionNode_node_35_solve_th_0th_2th_4_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(34);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(34);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 35
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[8];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_0 + th_2 + th_4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(36, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_35_solve_th_0th_2th_4_soa_processor();
    // Finish code for explicit solution node 34
    
    // Code for non-branch dispatcher node 36
    // Actually, there is no code
    
    // Code for explicit solution node 37, solved variable is th_1th_2th_4_soa
    auto ExplicitSolutionNode_node_37_solve_th_1th_2th_4_soa_processor = [&]() -> void
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
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[8];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_1 + th_2 + th_4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(38, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_37_solve_th_1th_2th_4_soa_processor();
    // Finish code for explicit solution node 36
    
    // Code for non-branch dispatcher node 38
    // Actually, there is no code
    
    // Code for explicit solution node 39, solved variable is th_6
    auto ExplicitSolutionNode_node_39_solve_th_6_processor = [&]() -> void
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
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[8];
            const double th_5 = this_solution[10];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*(-(((3.3383391313329601e-5*std::sin(th_5) + 0.99999999944277496*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) - std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::cos(th_1) - (0.99999999944277496*std::sin(th_5) - 3.3383391313329601e-5*std::cos(th_4)*std::cos(th_5))*std::sin(th_1))*std::cos(th_0) + ((3.3383391313329601e-5*std::sin(th_5) + 0.99999999944277496*std::cos(th_4)*std::cos(th_5))*std::sin(th_2) + std::sin(th_4)*std::cos(th_2)*std::cos(th_5))*std::sin(th_0)) + r_21*(-(((3.3383391313329601e-5*std::sin(th_5) + 0.99999999944277496*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) - std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::cos(th_1) - (0.99999999944277496*std::sin(th_5) - 3.3383391313329601e-5*std::cos(th_4)*std::cos(th_5))*std::sin(th_1))*std::sin(th_0) - ((3.3383391313329601e-5*std::sin(th_5) + 0.99999999944277496*std::cos(th_4)*std::cos(th_5))*std::sin(th_2) + std::sin(th_4)*std::cos(th_2)*std::cos(th_5))*std::cos(th_0)) - r_31*(((3.3383391313329601e-5*std::sin(th_5) + 0.99999999944277496*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) - std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::sin(th_1) + (0.99999999944277496*std::sin(th_5) - 3.3383391313329601e-5*std::cos(th_4)*std::cos(th_5))*std::cos(th_1))) >= zero_tolerance || std::fabs(r_12*(-(((3.3383391313329601e-5*std::sin(th_5) + 0.99999999944277496*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) - std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::cos(th_1) - (0.99999999944277496*std::sin(th_5) - 3.3383391313329601e-5*std::cos(th_4)*std::cos(th_5))*std::sin(th_1))*std::cos(th_0) + ((3.3383391313329601e-5*std::sin(th_5) + 0.99999999944277496*std::cos(th_4)*std::cos(th_5))*std::sin(th_2) + std::sin(th_4)*std::cos(th_2)*std::cos(th_5))*std::sin(th_0)) + r_22*(-(((3.3383391313329601e-5*std::sin(th_5) + 0.99999999944277496*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) - std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::cos(th_1) - (0.99999999944277496*std::sin(th_5) - 3.3383391313329601e-5*std::cos(th_4)*std::cos(th_5))*std::sin(th_1))*std::sin(th_0) - ((3.3383391313329601e-5*std::sin(th_5) + 0.99999999944277496*std::cos(th_4)*std::cos(th_5))*std::sin(th_2) + std::sin(th_4)*std::cos(th_2)*std::cos(th_5))*std::cos(th_0)) - r_32*(((3.3383391313329601e-5*std::sin(th_5) + 0.99999999944277496*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) - std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::sin(th_1) + (0.99999999944277496*std::sin(th_5) - 3.3383391313329601e-5*std::cos(th_4)*std::cos(th_5))*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_4);
                const double x3 = std::cos(th_5);
                const double x4 = -0.99999999944277496*x1 + 3.3383391313329601e-5*x2*x3;
                const double x5 = std::sin(th_1);
                const double x6 = std::sin(th_2);
                const double x7 = x3*std::sin(th_4);
                const double x8 = std::cos(th_2);
                const double x9 = -3.3383391313329601e-5*x1 - 0.99999999944277496*x2*x3;
                const double x10 = x6*x7 + x8*x9;
                const double x11 = x0*x4 + x10*x5;
                const double x12 = std::cos(th_0);
                const double x13 = x6*x9 - x7*x8;
                const double x14 = std::sin(th_0);
                const double x15 = x0*x10 - x4*x5;
                const double x16 = x12*x13 + x14*x15;
                const double x17 = x12*x15 - x13*x14;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_12*x17 - r_22*x16 - r_32*x11, r_11*x17 + r_21*x16 + r_31*x11);
                solution_queue.get_solution(node_input_i_idx_in_queue)[11] = tmp_sol_value;
                add_input_index_to(40, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_39_solve_th_6_processor();
    // Finish code for explicit solution node 38
    
    // Code for non-branch dispatcher node 40
    // Actually, there is no code
    
    // Code for explicit solution node 41, solved variable is th_2th_4th_6_soa
    auto ExplicitSolutionNode_node_41_solve_th_2th_4th_6_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(40);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(40);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 41
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[8];
            const double th_6 = this_solution[11];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_2 + th_4 + th_6;
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_41_solve_th_2th_4th_6_soa_processor();
    // Finish code for explicit solution node 40
    
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
            const double th_1 = this_solution[2];
            
            const bool condition_0 = std::fabs(Px*std::sin(th_0) - Py*std::cos(th_0)) >= zero_tolerance || std::fabs(a_3 + 0.99811318822181105*a_4 - 0.06140084280929*d_4) >= zero_tolerance || std::fabs(1.0*a_3 + 0.99811318822181105*a_4 - 0.06140084280929*d_4) >= zero_tolerance || std::fabs(Px*std::cos(th_0)*std::cos(th_1) + Py*std::sin(th_0)*std::cos(th_1) + Pz*std::sin(th_1) - a_1*std::cos(th_1) - a_2) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double x2 = 0.99811318822181105*a_4 - 0.06140084280929*d_4;
                const double x3 = std::cos(th_1);
                // End of temp variables
                const double tmp_sol_value = std::atan2((-Px*x0 + Py*x1)/(1.0*a_3 + x2), (Px*x1*x3 + Py*x0*x3 + Pz*std::sin(th_1) - a_1*x3 - a_2)/(a_3 + x2));
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
    
    // Code for explicit solution node 14, solved variable is th_5
    auto ExplicitSolutionNode_node_14_solve_th_5_processor = [&]() -> void
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
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = std::fabs(r_13*((0.99811318822181105*std::sin(th_1) - 0.06140084280929*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + 0.06140084280929*std::sin(th_0)*std::sin(th_2)) + r_23*((0.99811318822181105*std::sin(th_1) - 0.06140084280929*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - 0.06140084280929*std::sin(th_2)*std::cos(th_0)) - r_33*(0.06140084280929*std::sin(th_1)*std::cos(th_2) + 0.99811318822181105*std::cos(th_1))) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_1);
                const double x2 = 0.06140084280929*std::cos(th_2);
                const double x3 = std::sin(th_0);
                const double x4 = 0.06140084280929*std::sin(th_2);
                const double x5 = std::cos(th_0);
                const double x6 = -x0*x2 + 0.99811318822181105*x1;
                const double x7 = std::acos(-r_13*(x3*x4 + x5*x6) - r_23*(x3*x6 - x4*x5) + r_33*(0.99811318822181105*x0 + x1*x2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[10] = x7;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(15, appended_idx);
            }
            
            const bool condition_1 = std::fabs(r_13*((0.99811318822181105*std::sin(th_1) - 0.06140084280929*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + 0.06140084280929*std::sin(th_0)*std::sin(th_2)) + r_23*((0.99811318822181105*std::sin(th_1) - 0.06140084280929*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - 0.06140084280929*std::sin(th_2)*std::cos(th_0)) - r_33*(0.06140084280929*std::sin(th_1)*std::cos(th_2) + 0.99811318822181105*std::cos(th_1))) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_1);
                const double x2 = 0.06140084280929*std::cos(th_2);
                const double x3 = std::sin(th_0);
                const double x4 = 0.06140084280929*std::sin(th_2);
                const double x5 = std::cos(th_0);
                const double x6 = -x0*x2 + 0.99811318822181105*x1;
                const double x7 = std::acos(-r_13*(x3*x4 + x5*x6) - r_23*(x3*x6 - x4*x5) + r_33*(0.99811318822181105*x0 + x1*x2));
                // End of temp variables
                const double tmp_sol_value = -x7;
                solution_queue.get_solution(node_input_i_idx_in_queue)[10] = tmp_sol_value;
                add_input_index_to(15, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_5_processor();
    // Finish code for explicit solution node 13
    
    // Code for solved_variable dispatcher node 15
    auto SolvedVariableDispatcherNode_node_15_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(15);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(15);
        if (!this_input_valid)
            return;
        
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_5 = this_solution[10];
            
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
                add_input_index_to(16, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_15_processor();
    // Finish code for solved_variable dispatcher node 15
    
    // Code for explicit solution node 16, solved variable is th_4
    auto ExplicitSolutionNode_node_16_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(16);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(16);
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
            const double th_5 = this_solution[10];
            
            const bool condition_0 = std::fabs(r_13*(std::sin(th_0)*std::cos(th_2) + std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_23*(-std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_33*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(r_13*std::sin(th_1)*std::cos(th_0) + r_23*std::sin(th_0)*std::sin(th_1) - r_33*std::cos(th_1) + 0.99811318822181105*std::cos(th_5)) >= zero_tolerance || 0.06140084280929*std::fabs(std::sin(th_5)) >= zero_tolerance || std::fabs(std::sin(th_5)) >= zero_tolerance;
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
                const double x8 = 16.286421394995902*x1;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(-r_13*(x3*x4 + x5*x7) + r_23*(-x3*x7 + x4*x5) - r_33*x1*x2), x0*(-r_13*x5*x8 - r_23*x3*x8 + 16.286421394995902*r_33*x6 - 16.255691983283274*std::cos(th_5)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
                add_input_index_to(17, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_4_processor();
    // Finish code for explicit solution node 16
    
    // Code for non-branch dispatcher node 17
    // Actually, there is no code
    
    // Code for explicit solution node 18, solved variable is th_2th_4th_5_soa
    auto ExplicitSolutionNode_node_18_solve_th_2th_4th_5_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(17);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(17);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 18
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[8];
            const double th_5 = this_solution[10];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_2 + th_4 + th_5;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(19, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_2th_4th_5_soa_processor();
    // Finish code for explicit solution node 17
    
    // Code for non-branch dispatcher node 19
    // Actually, there is no code
    
    // Code for explicit solution node 20, solved variable is th_0th_2th_4_soa
    auto ExplicitSolutionNode_node_20_solve_th_0th_2th_4_soa_processor = [&]() -> void
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
            const double th_4 = this_solution[8];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_0 + th_2 + th_4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(21, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_20_solve_th_0th_2th_4_soa_processor();
    // Finish code for explicit solution node 19
    
    // Code for non-branch dispatcher node 21
    // Actually, there is no code
    
    // Code for explicit solution node 22, solved variable is th_1th_2th_4_soa
    auto ExplicitSolutionNode_node_22_solve_th_1th_2th_4_soa_processor = [&]() -> void
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
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[8];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_1 + th_2 + th_4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(23, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_22_solve_th_1th_2th_4_soa_processor();
    // Finish code for explicit solution node 21
    
    // Code for non-branch dispatcher node 23
    // Actually, there is no code
    
    // Code for explicit solution node 24, solved variable is th_6
    auto ExplicitSolutionNode_node_24_solve_th_6_processor = [&]() -> void
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
            const double th_4 = this_solution[8];
            const double th_5 = this_solution[10];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*((((0.06140084280929*std::sin(th_5) + 0.99811318822181105*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) + std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::cos(th_1) - (0.99811318822181105*std::sin(th_5) - 0.06140084280929*std::cos(th_4)*std::cos(th_5))*std::sin(th_1))*std::cos(th_0) - ((0.06140084280929*std::sin(th_5) + 0.99811318822181105*std::cos(th_4)*std::cos(th_5))*std::sin(th_2) - std::sin(th_4)*std::cos(th_2)*std::cos(th_5))*std::sin(th_0)) + r_21*((((0.06140084280929*std::sin(th_5) + 0.99811318822181105*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) + std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::cos(th_1) - (0.99811318822181105*std::sin(th_5) - 0.06140084280929*std::cos(th_4)*std::cos(th_5))*std::sin(th_1))*std::sin(th_0) + ((0.06140084280929*std::sin(th_5) + 0.99811318822181105*std::cos(th_4)*std::cos(th_5))*std::sin(th_2) - std::sin(th_4)*std::cos(th_2)*std::cos(th_5))*std::cos(th_0)) + r_31*(((0.06140084280929*std::sin(th_5) + 0.99811318822181105*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) + std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::sin(th_1) + (0.99811318822181105*std::sin(th_5) - 0.06140084280929*std::cos(th_4)*std::cos(th_5))*std::cos(th_1))) >= zero_tolerance || std::fabs(r_12*((((0.06140084280929*std::sin(th_5) + 0.99811318822181105*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) + std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::cos(th_1) - (0.99811318822181105*std::sin(th_5) - 0.06140084280929*std::cos(th_4)*std::cos(th_5))*std::sin(th_1))*std::cos(th_0) - ((0.06140084280929*std::sin(th_5) + 0.99811318822181105*std::cos(th_4)*std::cos(th_5))*std::sin(th_2) - std::sin(th_4)*std::cos(th_2)*std::cos(th_5))*std::sin(th_0)) + r_22*((((0.06140084280929*std::sin(th_5) + 0.99811318822181105*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) + std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::cos(th_1) - (0.99811318822181105*std::sin(th_5) - 0.06140084280929*std::cos(th_4)*std::cos(th_5))*std::sin(th_1))*std::sin(th_0) + ((0.06140084280929*std::sin(th_5) + 0.99811318822181105*std::cos(th_4)*std::cos(th_5))*std::sin(th_2) - std::sin(th_4)*std::cos(th_2)*std::cos(th_5))*std::cos(th_0)) + r_32*(((0.06140084280929*std::sin(th_5) + 0.99811318822181105*std::cos(th_4)*std::cos(th_5))*std::cos(th_2) + std::sin(th_2)*std::sin(th_4)*std::cos(th_5))*std::sin(th_1) + (0.99811318822181105*std::sin(th_5) - 0.06140084280929*std::cos(th_4)*std::cos(th_5))*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_5);
                const double x3 = x2*std::cos(th_4);
                const double x4 = 0.99811318822181105*x1 - 0.06140084280929*x3;
                const double x5 = std::sin(th_1);
                const double x6 = std::sin(th_2);
                const double x7 = x2*std::sin(th_4);
                const double x8 = std::cos(th_2);
                const double x9 = 0.06140084280929*x1 + 0.99811318822181105*x3;
                const double x10 = x6*x7 + x8*x9;
                const double x11 = x0*x4 + x10*x5;
                const double x12 = std::cos(th_0);
                const double x13 = x6*x9 - x7*x8;
                const double x14 = std::sin(th_0);
                const double x15 = x0*x10 - x4*x5;
                const double x16 = x12*x13 + x14*x15;
                const double x17 = x12*x15 - x13*x14;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_12*x17 - r_22*x16 - r_32*x11, r_11*x17 + r_21*x16 + r_31*x11);
                solution_queue.get_solution(node_input_i_idx_in_queue)[11] = tmp_sol_value;
                add_input_index_to(25, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_6_processor();
    // Finish code for explicit solution node 23
    
    // Code for non-branch dispatcher node 25
    // Actually, there is no code
    
    // Code for explicit solution node 26, solved variable is th_2th_4th_6_soa
    auto ExplicitSolutionNode_node_26_solve_th_2th_4th_6_soa_processor = [&]() -> void
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
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[8];
            const double th_6 = this_solution[11];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_2 + th_4 + th_6;
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_26_solve_th_2th_4th_6_soa_processor();
    // Finish code for explicit solution node 25
    
    // Code for explicit solution node 5, solved variable is th_2
    auto ExplicitSolutionNode_node_5_solve_th_2_processor = [&]() -> void
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
            const double th_3 = this_solution[7];
            
            const bool condition_0 = std::fabs((Px*std::sin(th_0) - Py*std::cos(th_0))/(a_3 + a_4*std::cos(th_3) - d_4*std::sin(th_3))) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::asin((-Px*std::sin(th_0) + Py*std::cos(th_0))/(a_3 + a_4*std::cos(th_3) - d_4*std::sin(th_3)));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x0;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx);
            }
            
            const bool condition_1 = std::fabs((Px*std::sin(th_0) - Py*std::cos(th_0))/(a_3 + a_4*std::cos(th_3) - d_4*std::sin(th_3))) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::asin((-Px*std::sin(th_0) + Py*std::cos(th_0))/(a_3 + a_4*std::cos(th_3) - d_4*std::sin(th_3)));
                // End of temp variables
                const double tmp_sol_value = M_PI - x0;
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
    ExplicitSolutionNode_node_5_solve_th_2_processor();
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
            const double th_3 = this_solution[7];
            
            const bool condition_0 = std::fabs(r_13*((std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + std::sin(th_0)*std::sin(th_2)*std::sin(th_3)) + r_23*((std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - std::sin(th_2)*std::sin(th_3)*std::cos(th_0)) - r_33*(std::sin(th_1)*std::sin(th_3)*std::cos(th_2) + std::cos(th_1)*std::cos(th_3))) <= 1;
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
                const double x8 = -x0*x4 + x1*x2;
                const double x9 = std::acos(-r_13*(x5*x6 + x7*x8) - r_23*(x5*x8 - x6*x7) + r_33*(x0*x1 + x2*x4));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[10] = x9;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(8, appended_idx);
            }
            
            const bool condition_1 = std::fabs(r_13*((std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + std::sin(th_0)*std::sin(th_2)*std::sin(th_3)) + r_23*((std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - std::sin(th_2)*std::sin(th_3)*std::cos(th_0)) - r_33*(std::sin(th_1)*std::sin(th_3)*std::cos(th_2) + std::cos(th_1)*std::cos(th_3))) <= 1;
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
                const double x8 = -x0*x4 + x1*x2;
                const double x9 = std::acos(-r_13*(x5*x6 + x7*x8) - r_23*(x5*x8 - x6*x7) + r_33*(x0*x1 + x2*x4));
                // End of temp variables
                const double tmp_sol_value = -x9;
                solution_queue.get_solution(node_input_i_idx_in_queue)[10] = tmp_sol_value;
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
            const double th_5 = this_solution[10];
            
            const bool degenerate_valid_0 = std::fabs(th_5) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(42, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_5 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(47, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(9, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor();
    // Finish code for solved_variable dispatcher node 8
    
    // Code for explicit solution node 47, solved variable is th_4th_6_soa
    auto ExplicitSolutionNode_node_47_solve_th_4th_6_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(47);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(47);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 47
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*(std::sin(th_0)*std::cos(th_2) + std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_21*(-std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_31*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(r_12*(std::sin(th_0)*std::cos(th_2) + std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_22*(-std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_32*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_2);
                const double x1 = x0*std::sin(th_1);
                const double x2 = std::sin(th_0);
                const double x3 = std::cos(th_2);
                const double x4 = std::cos(th_0);
                const double x5 = x0*std::cos(th_1);
                const double x6 = x2*x3 + x4*x5;
                const double x7 = -x2*x5 + x3*x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x6 + r_21*x7 - r_31*x1, -r_12*x6 + r_22*x7 - r_32*x1);
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
    ExplicitSolutionNode_node_47_solve_th_4th_6_soa_processor();
    // Finish code for explicit solution node 47
    
    // Code for non-branch dispatcher node 48
    // Actually, there is no code
    
    // Code for explicit solution node 49, solved variable is th_4
    auto ExplicitSolutionNode_node_49_solve_th_4_processor = [&]() -> void
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
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
                add_input_index_to(50, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_49_solve_th_4_processor();
    // Finish code for explicit solution node 48
    
    // Code for non-branch dispatcher node 50
    // Actually, there is no code
    
    // Code for explicit solution node 51, solved variable is th_6
    auto ExplicitSolutionNode_node_51_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(50);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(50);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 51
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_4 = this_solution[8];
            const double th_4th_6_soa = this_solution[9];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_4 + th_4th_6_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[11] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_51_solve_th_6_processor();
    // Finish code for explicit solution node 50
    
    // Code for explicit solution node 42, solved variable is negative_th_6_positive_th_4__soa
    auto ExplicitSolutionNode_node_42_solve_negative_th_6_positive_th_4__soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(42);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(42);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 42
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*(std::sin(th_0)*std::cos(th_2) + std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_21*(-std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_31*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(r_12*(std::sin(th_0)*std::cos(th_2) + std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_22*(-std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_32*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_2);
                const double x1 = x0*std::sin(th_1);
                const double x2 = std::sin(th_0);
                const double x3 = std::cos(th_2);
                const double x4 = std::cos(th_0);
                const double x5 = x0*std::cos(th_1);
                const double x6 = x2*x3 + x4*x5;
                const double x7 = -x2*x5 + x3*x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_11*x6 - r_21*x7 + r_31*x1, -r_12*x6 + r_22*x7 - r_32*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(43, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_42_solve_negative_th_6_positive_th_4__soa_processor();
    // Finish code for explicit solution node 42
    
    // Code for non-branch dispatcher node 43
    // Actually, there is no code
    
    // Code for explicit solution node 44, solved variable is th_4
    auto ExplicitSolutionNode_node_44_solve_th_4_processor = [&]() -> void
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
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
                add_input_index_to(45, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_44_solve_th_4_processor();
    // Finish code for explicit solution node 43
    
    // Code for non-branch dispatcher node 45
    // Actually, there is no code
    
    // Code for explicit solution node 46, solved variable is th_6
    auto ExplicitSolutionNode_node_46_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(45);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(45);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 46
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double negative_th_6_positive_th_4__soa = this_solution[0];
            const double th_4 = this_solution[8];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -negative_th_6_positive_th_4__soa + th_4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[11] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_46_solve_th_6_processor();
    // Finish code for explicit solution node 45
    
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
            const double th_3 = this_solution[7];
            const double th_5 = this_solution[10];
            
            const bool condition_0 = std::fabs(r_13*((std::sin(th_1)*std::sin(th_3) + std::cos(th_1)*std::cos(th_2)*std::cos(th_3))*std::cos(th_0) - std::sin(th_0)*std::sin(th_2)*std::cos(th_3)) + r_23*((std::sin(th_1)*std::sin(th_3) + std::cos(th_1)*std::cos(th_2)*std::cos(th_3))*std::sin(th_0) + std::sin(th_2)*std::cos(th_0)*std::cos(th_3)) + r_33*(std::sin(th_1)*std::cos(th_2)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1))) >= zero_tolerance || std::fabs(r_13*(std::sin(th_0)*std::cos(th_2) + std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_23*(-std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_33*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(std::sin(th_5)) >= zero_tolerance;
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
                const double tmp_sol_value = std::atan2(x0*(-r_13*(x3*x4 + x5*x7) + r_23*(-x3*x7 + x4*x5) - r_33*x1*x2), x0*(-r_13*(-x11*x3 + x12*x5) - r_23*(x11*x5 + x12*x3) - r_33*(x1*x10 - x6*x8)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
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
            const double th_3 = this_solution[7];
            const double th_5 = this_solution[10];
            
            const bool condition_0 = std::fabs(-r_11*((-std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) - std::sin(th_0)*std::sin(th_2)*std::sin(th_3)) - r_21*((-std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) + std::sin(th_2)*std::sin(th_3)*std::cos(th_0)) - r_31*(std::sin(th_1)*std::sin(th_3)*std::cos(th_2) + std::cos(th_1)*std::cos(th_3))) >= zero_tolerance || std::fabs(r_12*((std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + std::sin(th_0)*std::sin(th_2)*std::sin(th_3)) + r_22*((std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - std::sin(th_2)*std::sin(th_3)*std::cos(th_0)) - r_32*(std::sin(th_1)*std::sin(th_3)*std::cos(th_2) + std::cos(th_1)*std::cos(th_3))) >= zero_tolerance || std::fabs(std::sin(th_5)) >= zero_tolerance;
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
                const double x7 = std::sin(th_0);
                const double x8 = x4*std::sin(th_2);
                const double x9 = x7*x8;
                const double x10 = std::cos(th_0);
                const double x11 = -x1*x5 + x2*x3;
                const double x12 = x10*x8;
                const double x13 = -x11;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(r_12*(x10*x11 + x9) + r_22*(x11*x7 - x12) - r_32*x6), x0*(r_11*(x10*x13 - x9) + r_21*(x12 + x13*x7) + r_31*x6));
                solution_queue.get_solution(node_input_i_idx_in_queue)[11] = tmp_sol_value;
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
        const double value_at_3 = raw_ik_out_i[7];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[8];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[10];  // th_5
        new_ik_i[5] = value_at_5;
        const double value_at_6 = raw_ik_out_i[11];  // th_6
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
    if (!ik_output.empty()) return;
    
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
}

static std::vector<std::array<double, robot_nq>> computeIK(const Eigen::Matrix4d& T_ee, double th_0)
{
    std::vector<std::array<double, robot_nq>> ik_output;
    RawIKWorksace raw_ik_workspace;
    computeIK(T_ee, th_0, raw_ik_workspace, ik_output);
    return ik_output;
}

}; // struct atlas_l_hand_ik

// Code below for debug
void test_ik_solve_atlas_l_hand()
{
    std::array<double, atlas_l_hand_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = atlas_l_hand_ik::computeFK(theta);
    auto ik_output = atlas_l_hand_ik::computeIK(ee_pose, theta[0]);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = atlas_l_hand_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_atlas_l_hand();
}
