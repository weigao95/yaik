#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct atlas_r_hand_ik {

// Constants for solver
static constexpr int robot_nq = 7;
static constexpr int max_n_solutions = 128;
static constexpr int n_tree_nodes = 22;
static constexpr int intermediate_solution_size = 8;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_1 = 0.11;
static constexpr double a_2 = 0.016;
static constexpr double a_3 = 0.0092;
static constexpr double a_4 = 0.00921;
static constexpr double d_2 = -0.306;
static constexpr double d_4 = -0.29955;
static constexpr double pre_transform_s0 = 0.1406;
static constexpr double pre_transform_s1 = -0.2256;
static constexpr double pre_transform_s2 = 0.2326;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = 1.5707963267948966;
static constexpr double th_2_offset_original2raw = 3.141592653589793;
static constexpr double th_3_offset_original2raw = 3.141592653589793;
static constexpr double th_4_offset_original2raw = 3.141592653589793;
static constexpr double th_5_offset_original2raw = 3.141592653589793;
static constexpr double th_6_offset_original2raw = 1.5707963267948966;

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
    ee_transformed(0, 0) = r_21;
    ee_transformed(0, 1) = -r_23;
    ee_transformed(0, 2) = -r_22;
    ee_transformed(0, 3) = -Py + pre_transform_s1;
    ee_transformed(1, 0) = -r_11;
    ee_transformed(1, 1) = r_13;
    ee_transformed(1, 2) = r_12;
    ee_transformed(1, 3) = Px - pre_transform_s0;
    ee_transformed(2, 0) = -r_31;
    ee_transformed(2, 1) = r_33;
    ee_transformed(2, 2) = r_32;
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
    ee_transformed(0, 0) = -r_21;
    ee_transformed(0, 1) = r_23;
    ee_transformed(0, 2) = r_22;
    ee_transformed(0, 3) = Py + pre_transform_s0;
    ee_transformed(1, 0) = r_11;
    ee_transformed(1, 1) = -r_13;
    ee_transformed(1, 2) = -r_12;
    ee_transformed(1, 3) = -Px + pre_transform_s1;
    ee_transformed(2, 0) = -r_31;
    ee_transformed(2, 1) = r_33;
    ee_transformed(2, 2) = r_32;
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
    const double x26 = a_2*x5;
    const double x27 = -x15*x5 - x16;
    const double x28 = x12*x2;
    const double x29 = x4*x5 - x8;
    const double x30 = x11*x28 + x14*x29;
    const double x31 = -x1*x27 - x10*x30;
    const double x32 = -x11*x29 + x14*x28;
    const double x33 = x1*x30 - x10*x27;
    const double x34 = -x21*x32 + x23*x33;
    const double x35 = x12*x6;
    const double x36 = x12*x3;
    const double x37 = x11*x5 - x14*x36;
    const double x38 = -x1*x35 - x10*x37;
    const double x39 = x11*x36 + x14*x5;
    const double x40 = x1*x37 - x10*x35;
    const double x41 = -x21*x39 + x23*x40;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x19 + x20*x25;
    ee_pose_raw(0, 1) = -x0*x25 - x19*x20;
    ee_pose_raw(0, 2) = -x21*x24 - x22*x23;
    ee_pose_raw(0, 3) = a_1*x7 + a_3*x17 + a_4*x18 - d_2*x13 + d_4*x22 + x26*x7;
    ee_pose_raw(1, 0) = -x0*x31 + x20*x34;
    ee_pose_raw(1, 1) = -x0*x34 - x20*x31;
    ee_pose_raw(1, 2) = -x21*x33 - x23*x32;
    ee_pose_raw(1, 3) = a_1*x2 + a_3*x29 + a_4*x30 - d_2*x28 + d_4*x32 + x2*x26;
    ee_pose_raw(2, 0) = -x0*x38 + x20*x41;
    ee_pose_raw(2, 1) = -x0*x41 - x20*x38;
    ee_pose_raw(2, 2) = -x21*x40 - x23*x39;
    ee_pose_raw(2, 3) = -a_2*x12 - a_3*x36 + a_4*x37 - d_2*x5 + d_4*x39;
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
    const double x5 = x0*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::sin(th_2);
    const double x8 = x1*x7;
    const double x9 = -x5 - x6*x8;
    const double x10 = std::cos(th_3);
    const double x11 = std::sin(th_3);
    const double x12 = x0*x7;
    const double x13 = x1*x4;
    const double x14 = -x12 + x13*x6;
    const double x15 = x10*x3 - x11*x14;
    const double x16 = std::cos(th_4);
    const double x17 = std::sin(th_4);
    const double x18 = x10*x14 + x11*x3;
    const double x19 = -x16*x9 - x17*x18;
    const double x20 = std::cos(th_5);
    const double x21 = std::sin(th_5);
    const double x22 = -x15*x20 - x21*(x16*x18 - x17*x9);
    const double x23 = x0*x2;
    const double x24 = x12*x6 - x13;
    const double x25 = -x5*x6 - x8;
    const double x26 = -x10*x23 - x11*x25;
    const double x27 = x10*x25 - x11*x23;
    const double x28 = -x16*x24 - x17*x27;
    const double x29 = -x20*x26 - x21*(x16*x27 - x17*x24);
    const double x30 = x2*x7;
    const double x31 = x2*x4;
    const double x32 = x10*x6 + x11*x31;
    const double x33 = -x10*x31 + x11*x6;
    const double x34 = -x16*x30 - x17*x33;
    const double x35 = -x20*x32 - x21*(x16*x33 - x17*x30);
    const double x36 = -a_2*x2 - d_2*x6 + pre_transform_s2;
    const double x37 = a_2*x6;
    const double x38 = -a_1*x0 + pre_transform_s1;
    const double x39 = d_2*x23 - x0*x37 + x38;
    const double x40 = -a_3*x31 + x36;
    const double x41 = a_3*x25 + x39;
    const double x42 = a_4*x33 + d_4*x32 + x40;
    const double x43 = a_4*x27 + d_4*x26 + x41;
    const double x44 = a_1*x1 + pre_transform_s0;
    const double x45 = -d_2*x3 + x1*x37 + x44;
    const double x46 = a_3*x14 + x45;
    const double x47 = a_4*x18 + d_4*x15 + x46;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x0;
    jacobian(0, 2) = -x3;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = x15;
    jacobian(0, 5) = x19;
    jacobian(0, 6) = x22;
    jacobian(1, 1) = x1;
    jacobian(1, 2) = x23;
    jacobian(1, 3) = x24;
    jacobian(1, 4) = x26;
    jacobian(1, 5) = x28;
    jacobian(1, 6) = x29;
    jacobian(2, 0) = 1;
    jacobian(2, 2) = -x6;
    jacobian(2, 3) = x30;
    jacobian(2, 4) = x32;
    jacobian(2, 5) = x34;
    jacobian(2, 6) = x35;
    jacobian(3, 0) = pre_transform_s1;
    jacobian(3, 1) = -pre_transform_s2*x1;
    jacobian(3, 2) = -x23*x36 - x39*x6;
    jacobian(3, 3) = -x24*x40 + x30*x41;
    jacobian(3, 4) = -x26*x42 + x32*x43;
    jacobian(3, 5) = -x28*x42 + x34*x43;
    jacobian(3, 6) = -x29*x42 + x35*x43;
    jacobian(4, 0) = -pre_transform_s0;
    jacobian(4, 1) = pre_transform_s2*x0;
    jacobian(4, 2) = -x3*x36 + x45*x6;
    jacobian(4, 3) = -x30*x46 + x40*x9;
    jacobian(4, 4) = x15*x42 - x32*x47;
    jacobian(4, 5) = x19*x42 - x34*x47;
    jacobian(4, 6) = x22*x42 - x35*x47;
    jacobian(5, 1) = -x0*x38 + x1*x44;
    jacobian(5, 2) = x23*x45 + x3*x39;
    jacobian(5, 3) = x24*x46 - x41*x9;
    jacobian(5, 4) = -x15*x43 + x26*x47;
    jacobian(5, 5) = -x19*x43 + x28*x47;
    jacobian(5, 6) = -x22*x43 + x29*x47;
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
    const double x5 = x0*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::sin(th_2);
    const double x8 = x1*x7;
    const double x9 = -x5 - x6*x8;
    const double x10 = std::cos(th_3);
    const double x11 = std::sin(th_3);
    const double x12 = x0*x7;
    const double x13 = x1*x4;
    const double x14 = -x12 + x13*x6;
    const double x15 = x10*x3 - x11*x14;
    const double x16 = std::cos(th_4);
    const double x17 = std::sin(th_4);
    const double x18 = x10*x14 + x11*x3;
    const double x19 = std::cos(th_5);
    const double x20 = std::sin(th_5);
    const double x21 = x0*x2;
    const double x22 = x12*x6 - x13;
    const double x23 = -x5*x6 - x8;
    const double x24 = -x10*x21 - x11*x23;
    const double x25 = x10*x23 - x11*x21;
    const double x26 = x2*x7;
    const double x27 = x2*x4;
    const double x28 = x10*x6 + x11*x27;
    const double x29 = -x10*x27 + x11*x6;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x0;
    jacobian(0, 2) = -x3;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = x15;
    jacobian(0, 5) = -x16*x9 - x17*x18;
    jacobian(0, 6) = -x15*x19 - x20*(x16*x18 - x17*x9);
    jacobian(1, 1) = x1;
    jacobian(1, 2) = x21;
    jacobian(1, 3) = x22;
    jacobian(1, 4) = x24;
    jacobian(1, 5) = -x16*x22 - x17*x25;
    jacobian(1, 6) = -x19*x24 - x20*(x16*x25 - x17*x22);
    jacobian(2, 0) = 1;
    jacobian(2, 2) = -x6;
    jacobian(2, 3) = x26;
    jacobian(2, 4) = x28;
    jacobian(2, 5) = -x16*x26 - x17*x29;
    jacobian(2, 6) = -x19*x28 - x20*(x16*x29 - x17*x26);
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
    const double x1 = p_on_ee_z*x0;
    const double x2 = std::cos(th_1);
    const double x3 = std::sin(th_1);
    const double x4 = std::cos(th_0);
    const double x5 = p_on_ee_z*x4;
    const double x6 = -a_2*x3 - d_2*x2 + pre_transform_s2;
    const double x7 = x3*x4;
    const double x8 = a_2*x2;
    const double x9 = -a_1*x4 + pre_transform_s1;
    const double x10 = d_2*x7 - x4*x8 + x9;
    const double x11 = std::sin(th_2);
    const double x12 = x11*x3;
    const double x13 = std::cos(th_2);
    const double x14 = x0*x13;
    const double x15 = x11*x4;
    const double x16 = -x14 + x15*x2;
    const double x17 = x13*x3;
    const double x18 = -a_3*x17 + x6;
    const double x19 = x0*x11;
    const double x20 = x13*x4;
    const double x21 = -x19 - x2*x20;
    const double x22 = a_3*x21 + x10;
    const double x23 = std::cos(th_3);
    const double x24 = std::sin(th_3);
    const double x25 = x17*x24 + x2*x23;
    const double x26 = -x21*x24 - x23*x7;
    const double x27 = -x17*x23 + x2*x24;
    const double x28 = a_4*x27 + d_4*x25 + x18;
    const double x29 = x21*x23 - x24*x7;
    const double x30 = a_4*x29 + d_4*x26 + x22;
    const double x31 = std::cos(th_4);
    const double x32 = std::sin(th_4);
    const double x33 = -x12*x31 - x27*x32;
    const double x34 = -x16*x31 - x29*x32;
    const double x35 = std::cos(th_5);
    const double x36 = std::sin(th_5);
    const double x37 = -x25*x35 - x36*(-x12*x32 + x27*x31);
    const double x38 = -x26*x35 - x36*(-x16*x32 + x29*x31);
    const double x39 = x0*x3;
    const double x40 = a_1*x0 + pre_transform_s0;
    const double x41 = -d_2*x39 + x0*x8 + x40;
    const double x42 = -x19*x2 - x20;
    const double x43 = x14*x2 - x15;
    const double x44 = a_3*x43 + x41;
    const double x45 = x23*x39 - x24*x43;
    const double x46 = x23*x43 + x24*x39;
    const double x47 = a_4*x46 + d_4*x45 + x44;
    const double x48 = -x31*x42 - x32*x46;
    const double x49 = -x35*x45 - x36*(x31*x46 - x32*x42);
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -p_on_ee_y + pre_transform_s1;
    jacobian(0, 1) = -pre_transform_s2*x0 + x1;
    jacobian(0, 2) = p_on_ee_y*x2 - x10*x2 + x3*x5 - x6*x7;
    jacobian(0, 3) = -p_on_ee_y*x12 + p_on_ee_z*x16 + x12*x22 - x16*x18;
    jacobian(0, 4) = -p_on_ee_y*x25 + p_on_ee_z*x26 + x25*x30 - x26*x28;
    jacobian(0, 5) = -p_on_ee_y*x33 + p_on_ee_z*x34 - x28*x34 + x30*x33;
    jacobian(0, 6) = -p_on_ee_y*x37 + p_on_ee_z*x38 - x28*x38 + x30*x37;
    jacobian(1, 0) = p_on_ee_x - pre_transform_s0;
    jacobian(1, 1) = pre_transform_s2*x4 - x5;
    jacobian(1, 2) = -p_on_ee_x*x2 + x1*x3 + x2*x41 - x39*x6;
    jacobian(1, 3) = p_on_ee_x*x12 - p_on_ee_z*x42 - x12*x44 + x18*x42;
    jacobian(1, 4) = p_on_ee_x*x25 - p_on_ee_z*x45 - x25*x47 + x28*x45;
    jacobian(1, 5) = p_on_ee_x*x33 - p_on_ee_z*x48 + x28*x48 - x33*x47;
    jacobian(1, 6) = p_on_ee_x*x37 - p_on_ee_z*x49 + x28*x49 - x37*x47;
    jacobian(2, 1) = -p_on_ee_x*x0 + p_on_ee_y*x4 + x0*x40 - x4*x9;
    jacobian(2, 2) = -p_on_ee_x*x7 - p_on_ee_y*x39 + x10*x39 + x41*x7;
    jacobian(2, 3) = -p_on_ee_x*x16 + p_on_ee_y*x42 + x16*x44 - x22*x42;
    jacobian(2, 4) = -p_on_ee_x*x26 + p_on_ee_y*x45 + x26*x47 - x30*x45;
    jacobian(2, 5) = -p_on_ee_x*x34 + p_on_ee_y*x48 - x30*x48 + x34*x47;
    jacobian(2, 6) = -p_on_ee_x*x38 + p_on_ee_y*x49 - x30*x49 + x38*x47;
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
    
    // Code for general_6dof solution node 1, solved variable is th_2
    auto General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;
        
        // The general 6-dof solution of root node
        Eigen::Matrix<double, 8, 8> R_l;
        R_l.setZero();
        R_l(0, 4) = -Pz;
        R_l(0, 5) = Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2);
        R_l(1, 4) = Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2);
        R_l(1, 5) = Pz;
        R_l(2, 0) = r_31;
        R_l(2, 1) = r_32;
        R_l(2, 2) = -r_11*std::cos(th_0) - r_21*std::sin(th_0);
        R_l(2, 3) = -r_12*std::cos(th_0) - r_22*std::sin(th_0);
        R_l(3, 0) = -r_11*std::cos(th_0) - r_21*std::sin(th_0);
        R_l(3, 1) = -r_12*std::cos(th_0) - r_22*std::sin(th_0);
        R_l(3, 2) = -r_31;
        R_l(3, 3) = -r_32;
        R_l(4, 6) = r_11*std::sin(th_0) - r_21*std::cos(th_0);
        R_l(4, 7) = r_12*std::sin(th_0) - r_22*std::cos(th_0);
        R_l(5, 6) = -Pz*r_31 - (-Px*std::sin(th_0) + Py*std::cos(th_0))*(-r_11*std::sin(th_0) + r_21*std::cos(th_0)) - (r_11*std::cos(th_0) + r_21*std::sin(th_0))*(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2));
        R_l(5, 7) = -Pz*r_32 - (-Px*std::sin(th_0) + Py*std::cos(th_0))*(-r_12*std::sin(th_0) + r_22*std::cos(th_0)) - (r_12*std::cos(th_0) + r_22*std::sin(th_0))*(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2));
        R_l(6, 0) = (-Px*std::sin(th_0) + Py*std::cos(th_0))*(r_11*std::cos(th_0) + r_21*std::sin(th_0)) - (-r_11*std::sin(th_0) + r_21*std::cos(th_0))*(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2));
        R_l(6, 1) = (-Px*std::sin(th_0) + Py*std::cos(th_0))*(r_12*std::cos(th_0) + r_22*std::sin(th_0)) - (-r_12*std::sin(th_0) + r_22*std::cos(th_0))*(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2));
        R_l(6, 2) = -Pz*(-r_11*std::sin(th_0) + r_21*std::cos(th_0)) + r_31*(-Px*std::sin(th_0) + Py*std::cos(th_0));
        R_l(6, 3) = -Pz*(-r_12*std::sin(th_0) + r_22*std::cos(th_0)) + r_32*(-Px*std::sin(th_0) + Py*std::cos(th_0));
        R_l(7, 0) = -Pz*(-r_11*std::sin(th_0) + r_21*std::cos(th_0)) + r_31*(-Px*std::sin(th_0) + Py*std::cos(th_0));
        R_l(7, 1) = -Pz*(-r_12*std::sin(th_0) + r_22*std::cos(th_0)) + r_32*(-Px*std::sin(th_0) + Py*std::cos(th_0));
        R_l(7, 2) = (Px*std::sin(th_0) - Py*std::cos(th_0))*(r_11*std::cos(th_0) + r_21*std::sin(th_0)) + (-r_11*std::sin(th_0) + r_21*std::cos(th_0))*(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2));
        R_l(7, 3) = (Px*std::sin(th_0) - Py*std::cos(th_0))*(r_12*std::cos(th_0) + r_22*std::sin(th_0)) + (-r_12*std::sin(th_0) + r_22*std::cos(th_0))*(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1*std::pow(std::sin(th_0), 2) - a_1*std::pow(std::cos(th_0), 2));
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
        const double x0 = std::sin(th_0);
        const double x1 = Px*x0;
        const double x2 = std::cos(th_0);
        const double x3 = Py*x2;
        const double x4 = x1 - x3;
        const double x5 = std::pow(Pz, 2);
        const double x6 = -x1 + x3;
        const double x7 = std::pow(x6, 2);
        const double x8 = Px*x2 + Py*x0 - a_1*std::pow(x0, 2) - a_1*std::pow(x2, 2);
        const double x9 = std::pow(x8, 2);
        const double x10 = -x5 - x7 - x9;
        const double x11 = 2*a_3;
        const double x12 = a_2*x11;
        const double x13 = -x12;
        const double x14 = 2*a_4;
        const double x15 = a_2*x14;
        const double x16 = x13 + x15;
        const double x17 = a_4*x11;
        const double x18 = 2*d_2;
        const double x19 = d_4*x18;
        const double x20 = -x17 + x19;
        const double x21 = std::pow(a_2, 2);
        const double x22 = std::pow(a_3, 2);
        const double x23 = std::pow(a_4, 2);
        const double x24 = std::pow(d_2, 2);
        const double x25 = std::pow(d_4, 2);
        const double x26 = x21 + x22 + x23 + x24 + x25;
        const double x27 = x20 + x26;
        const double x28 = x16 + x27;
        const double x29 = x10 + x28;
        const double x30 = 4*a_2;
        const double x31 = d_4*x30;
        const double x32 = 4*a_3;
        const double x33 = d_4*x32;
        const double x34 = 4*a_4;
        const double x35 = d_2*x34;
        const double x36 = -x35;
        const double x37 = -x33 + x36;
        const double x38 = x31 + x37;
        const double x39 = -x15;
        const double x40 = x10 + x39;
        const double x41 = x17 - x19;
        const double x42 = x26 + x41;
        const double x43 = x13 + x42;
        const double x44 = x40 + x43;
        const double x45 = r_11*x2 + r_21*x0;
        const double x46 = Pz*x45;
        const double x47 = r_31*x8;
        const double x48 = x46 - x47;
        const double x49 = r_12*x2 + r_22*x0;
        const double x50 = Pz*x49;
        const double x51 = r_32*x8;
        const double x52 = x50 - x51;
        const double x53 = R_l_inv_60*x48 + R_l_inv_70*x52;
        const double x54 = a_3*x53;
        const double x55 = R_l_inv_61*x48 + R_l_inv_71*x52;
        const double x56 = d_2*x55;
        const double x57 = d_4*x55;
        const double x58 = -a_2*x53;
        const double x59 = a_4*x53;
        const double x60 = -x59;
        const double x61 = x54 + x56 + x57 + x58 + x60;
        const double x62 = R_l_inv_67*x48 + R_l_inv_77*x52;
        const double x63 = a_4*x62;
        const double x64 = R_l_inv_66*x48 + R_l_inv_76*x52;
        const double x65 = d_4*x64;
        const double x66 = x63 + x65;
        const double x67 = a_2*x62;
        const double x68 = d_2*x64;
        const double x69 = R_l_inv_64*x48;
        const double x70 = -x69;
        const double x71 = R_l_inv_74*x52;
        const double x72 = -x71;
        const double x73 = a_3*x62;
        const double x74 = -x73;
        const double x75 = x67 + x68 + x70 + x72 + x74;
        const double x76 = 2*d_4;
        const double x77 = -x76;
        const double x78 = 2*R_l_inv_62;
        const double x79 = x48*x78;
        const double x80 = 2*R_l_inv_72;
        const double x81 = x52*x80;
        const double x82 = R_l_inv_65*x48 + R_l_inv_75*x52;
        const double x83 = 2*a_2;
        const double x84 = x82*x83;
        const double x85 = -x18 + x79 + x81 + x84;
        const double x86 = x77 + x85;
        const double x87 = x11*x82;
        const double x88 = x14*x82;
        const double x89 = -x87 + x88;
        const double x90 = -x63;
        const double x91 = -x65;
        const double x92 = x90 + x91;
        const double x93 = -x67;
        const double x94 = -x68;
        const double x95 = x69 + x71 + x73 + x93 + x94;
        const double x96 = x14*x64;
        const double x97 = -x96;
        const double x98 = x62*x76;
        const double x99 = -x14*x55;
        const double x100 = x53*x76;
        const double x101 = -x100 + x99;
        const double x102 = 4*d_2;
        const double x103 = 4*R_l_inv_63;
        const double x104 = 4*R_l_inv_73;
        const double x105 = -x102*x82 + x103*x48 + x104*x52 - x30;
        const double x106 = -x98;
        const double x107 = -x57;
        const double x108 = x56 + x58 + x59;
        const double x109 = x107 + x108 + x54;
        const double x110 = x18 - x79 - x81 - x84;
        const double x111 = x110 + x77;
        const double x112 = x87 + x88;
        const double x113 = -r_11*x0 + r_21*x2;
        const double x114 = 2*Pz;
        const double x115 = x114*x6;
        const double x116 = 2*x8;
        const double x117 = -r_31*x5 + r_31*x7 + r_31*x9 - x113*x115 - x116*x46;
        const double x118 = -r_12*x0 + r_22*x2;
        const double x119 = -r_32*x5 + r_32*x7 + r_32*x9 - x115*x118 - x116*x50;
        const double x120 = x8*(-2*x1 + 2*x3);
        const double x121 = x113*x120 + x114*x47 - x45*x5 - x45*x7 + x45*x9;
        const double x122 = x114*x51 + x118*x120 - x49*x5 - x49*x7 + x49*x9;
        const double x123 = R_l_inv_00*x117 + R_l_inv_10*x119 + R_l_inv_20*x121 + R_l_inv_30*x122;
        const double x124 = a_3*x123;
        const double x125 = R_l_inv_01*x117 + R_l_inv_11*x119 + R_l_inv_21*x121 + R_l_inv_31*x122;
        const double x126 = d_2*x125;
        const double x127 = d_4*x125;
        const double x128 = -a_2*x123;
        const double x129 = a_4*x123;
        const double x130 = -x129;
        const double x131 = x124 + x126 + x127 + x128 + x130;
        const double x132 = R_l_inv_07*x117 + R_l_inv_17*x119 + R_l_inv_27*x121 + R_l_inv_37*x122;
        const double x133 = a_4*x132;
        const double x134 = R_l_inv_06*x117 + R_l_inv_16*x119 + R_l_inv_26*x121 + R_l_inv_36*x122;
        const double x135 = d_4*x134;
        const double x136 = x133 + x135;
        const double x137 = a_2*x132;
        const double x138 = d_2*x134;
        const double x139 = R_l_inv_04*x117;
        const double x140 = -x139;
        const double x141 = R_l_inv_14*x119;
        const double x142 = -x141;
        const double x143 = R_l_inv_24*x121;
        const double x144 = -x143;
        const double x145 = R_l_inv_34*x122;
        const double x146 = -x145;
        const double x147 = a_3*x132;
        const double x148 = -x147;
        const double x149 = x137 + x138 + x140 + x142 + x144 + x146 + x148;
        const double x150 = 4*d_4;
        const double x151 = d_2*x150;
        const double x152 = -x151;
        const double x153 = a_4*x32;
        const double x154 = -x153;
        const double x155 = R_l_inv_05*x117 + R_l_inv_15*x119 + R_l_inv_25*x121 + R_l_inv_35*x122;
        const double x156 = x14*x155;
        const double x157 = a_4*x30;
        const double x158 = x152 + x154 + x156 + x157;
        const double x159 = a_3*x30;
        const double x160 = x11*x155;
        const double x161 = -x159 - x160;
        const double x162 = 2*x24;
        const double x163 = 2*x25;
        const double x164 = 2*x21;
        const double x165 = 2*x117;
        const double x166 = R_l_inv_02*x165;
        const double x167 = 2*x119;
        const double x168 = R_l_inv_12*x167;
        const double x169 = 2*x121;
        const double x170 = R_l_inv_22*x169;
        const double x171 = 2*x122;
        const double x172 = R_l_inv_32*x171;
        const double x173 = x155*x83;
        const double x174 = -x162 - x163 + x164 + x166 + x168 + x170 + x172 + x173;
        const double x175 = 2*x22;
        const double x176 = 2*x23;
        const double x177 = x175 + x176;
        const double x178 = x174 + x177;
        const double x179 = -x133;
        const double x180 = -x135;
        const double x181 = x179 + x180;
        const double x182 = -x137;
        const double x183 = -x138;
        const double x184 = x139 + x141 + x143 + x145 + x147 + x182 + x183;
        const double x185 = x134*x14;
        const double x186 = -x185;
        const double x187 = x132*x76;
        const double x188 = -x125*x14;
        const double x189 = x123*x76;
        const double x190 = x188 - x189;
        const double x191 = 8*d_2;
        const double x192 = a_3*x191;
        const double x193 = 8*d_4;
        const double x194 = a_4*x193;
        const double x195 = 4*x117;
        const double x196 = 4*x119;
        const double x197 = 4*x121;
        const double x198 = 4*x122;
        const double x199 = R_l_inv_03*x195 + R_l_inv_13*x196 + R_l_inv_23*x197 + R_l_inv_33*x198 - a_2*x191 - x102*x155;
        const double x200 = -x187;
        const double x201 = -x127;
        const double x202 = x126 + x128 + x129;
        const double x203 = x124 + x201 + x202;
        const double x204 = -x175 - x176;
        const double x205 = x162 + x163 - x164 - x166 - x168 - x170 - x172 - x173;
        const double x206 = x204 + x205;
        const double x207 = x159 + x160;
        const double x208 = R_l_inv_00*x121 + R_l_inv_10*x122 - R_l_inv_20*x117 - R_l_inv_30*x119;
        const double x209 = a_3*x208;
        const double x210 = R_l_inv_01*x121 + R_l_inv_11*x122 - R_l_inv_21*x117 - R_l_inv_31*x119;
        const double x211 = d_2*x210;
        const double x212 = d_4*x210;
        const double x213 = -a_2*x208;
        const double x214 = a_4*x208;
        const double x215 = -x214;
        const double x216 = x209 + x211 + x212 + x213 + x215;
        const double x217 = R_l_inv_07*x121 + R_l_inv_17*x122 - R_l_inv_27*x117 - R_l_inv_37*x119;
        const double x218 = a_4*x217;
        const double x219 = R_l_inv_06*x121 + R_l_inv_16*x122 - R_l_inv_26*x117 - R_l_inv_36*x119;
        const double x220 = d_4*x219;
        const double x221 = x218 + x220;
        const double x222 = R_l_inv_24*x117;
        const double x223 = R_l_inv_34*x119;
        const double x224 = a_2*x217;
        const double x225 = d_2*x219;
        const double x226 = R_l_inv_04*x121;
        const double x227 = -x226;
        const double x228 = R_l_inv_14*x122;
        const double x229 = -x228;
        const double x230 = a_3*x217;
        const double x231 = -x230;
        const double x232 = x222 + x223 + x224 + x225 + x227 + x229 + x231;
        const double x233 = d_2*x30;
        const double x234 = R_l_inv_22*x165;
        const double x235 = R_l_inv_32*x167;
        const double x236 = R_l_inv_02*x169;
        const double x237 = R_l_inv_12*x171;
        const double x238 = R_l_inv_05*x121 + R_l_inv_15*x122 - R_l_inv_25*x117 - R_l_inv_35*x119;
        const double x239 = x238*x83;
        const double x240 = -x233 - x234 - x235 + x236 + x237 + x239;
        const double x241 = -x31;
        const double x242 = x241 + x33;
        const double x243 = x14*x238 + x36;
        const double x244 = x242 + x243;
        const double x245 = a_4*x150;
        const double x246 = x11*x238;
        const double x247 = d_2*x32;
        const double x248 = -x245 - x246 + x247;
        const double x249 = -x218;
        const double x250 = -x220;
        const double x251 = x249 + x250;
        const double x252 = -x222;
        const double x253 = -x223;
        const double x254 = -x224;
        const double x255 = -x225;
        const double x256 = x226 + x228 + x230 + x252 + x253 + x254 + x255;
        const double x257 = x14*x219;
        const double x258 = -x257;
        const double x259 = x217*x76;
        const double x260 = -x14*x210;
        const double x261 = x208*x76;
        const double x262 = x260 - x261;
        const double x263 = 8*a_3;
        const double x264 = a_2*x263;
        const double x265 = 4*x22;
        const double x266 = 4*x24;
        const double x267 = -x265 + x266;
        const double x268 = 4*x21;
        const double x269 = 4*x25;
        const double x270 = 4*x23;
        const double x271 = -x268 - x269 + x270;
        const double x272 = R_l_inv_03*x197 + R_l_inv_13*x198 - R_l_inv_23*x195 - R_l_inv_33*x196 - x102*x238 + x267 + x271;
        const double x273 = -x259;
        const double x274 = -x212;
        const double x275 = x211 + x213 + x214;
        const double x276 = x209 + x274 + x275;
        const double x277 = x233 + x234 + x235 - x236 - x237 - x239;
        const double x278 = x245 + x246 - x247;
        const double x279 = r_31*x115 - x113*x5 + x113*x7 - x113*x9 + x120*x45;
        const double x280 = r_32*x115 - x118*x5 + x118*x7 - x118*x9 + x120*x49;
        const double x281 = R_l_inv_60*x279 + R_l_inv_70*x280;
        const double x282 = a_3*x281;
        const double x283 = R_l_inv_61*x279 + R_l_inv_71*x280;
        const double x284 = d_2*x283;
        const double x285 = d_4*x283;
        const double x286 = -a_2*x281;
        const double x287 = a_4*x281;
        const double x288 = -x287;
        const double x289 = x282 + x284 + x285 + x286 + x288;
        const double x290 = R_l_inv_67*x279 + R_l_inv_77*x280;
        const double x291 = a_4*x290;
        const double x292 = R_l_inv_66*x279 + R_l_inv_76*x280;
        const double x293 = d_4*x292;
        const double x294 = x291 + x293;
        const double x295 = a_2*x290;
        const double x296 = d_2*x292;
        const double x297 = R_l_inv_64*x279;
        const double x298 = -x297;
        const double x299 = R_l_inv_74*x280;
        const double x300 = -x299;
        const double x301 = a_3*x290;
        const double x302 = -x301;
        const double x303 = x295 + x296 + x298 + x300 + x302;
        const double x304 = x279*x78;
        const double x305 = x280*x80;
        const double x306 = R_l_inv_65*x279 + R_l_inv_75*x280;
        const double x307 = x306*x83;
        const double x308 = x304 + x305 + x307;
        const double x309 = x11*x306;
        const double x310 = x14*x306;
        const double x311 = -x309 + x310;
        const double x312 = -x293;
        const double x313 = x301 + x312;
        const double x314 = -x291 + x39;
        const double x315 = -x21 - x22 - x23 - x24 - x25 - x295 - x296 + x297 + x299;
        const double x316 = x315 + x41;
        const double x317 = x14*x292;
        const double x318 = -x317;
        const double x319 = x290*x76;
        const double x320 = -x14*x283;
        const double x321 = x281*x76;
        const double x322 = x320 - x321;
        const double x323 = -x102*x306 + x103*x279 + x104*x280;
        const double x324 = -x319;
        const double x325 = x317 + x35;
        const double x326 = x287 + x312;
        const double x327 = x284 - x285 + x286;
        const double x328 = x314 + x327;
        const double x329 = -x304 - x305 - x307;
        const double x330 = x309 + x310;
        const double x331 = x12 + x15;
        const double x332 = x287 + x301;
        const double x333 = x20 + x315;
        const double x334 = -x11;
        const double x335 = x14 + x334;
        const double x336 = -x14 + x334;
        const double x337 = 4*x69;
        const double x338 = 4*x71;
        const double x339 = 4*x67;
        const double x340 = 4*x68;
        const double x341 = -4*x65;
        const double x342 = x110 + x76;
        const double x343 = x76 + x85;
        const double x344 = x152 + x153;
        const double x345 = x174 + x204;
        const double x346 = 4*x139;
        const double x347 = 4*x141;
        const double x348 = 4*x143;
        const double x349 = 4*x145;
        const double x350 = x132*x30;
        const double x351 = 4*x138;
        const double x352 = -4*x135;
        const double x353 = x151 + x154;
        const double x354 = x177 + x205;
        const double x355 = a_3*x193;
        const double x356 = a_4*x191;
        const double x357 = 4*x222;
        const double x358 = 4*x223;
        const double x359 = 4*x226;
        const double x360 = 4*x228;
        const double x361 = x217*x30;
        const double x362 = 4*x225;
        const double x363 = -4*x220;
        const double x364 = x277 + x31;
        const double x365 = 8*a_2*a_4;
        const double x366 = x240 + x31;
        const double x367 = x290*x30;
        const double x368 = 4*x296;
        const double x369 = 4*x297;
        const double x370 = 4*x299;
        const double x371 = -a_4*x263 - d_4*x191 - 4*x293;
        const double x372 = x12 + x27;
        const double x373 = x372 + x40;
        const double x374 = x241 + x37;
        const double x375 = x331 + x42;
        const double x376 = x10 + x375;
        const double x377 = -x54;
        const double x378 = x108 + x377 + x57;
        const double x379 = x63 + x91;
        const double x380 = x69 + x71 + x74 + x93 + x94;
        const double x381 = x65 + x90;
        const double x382 = x67 + x68 + x70 + x72 + x73;
        const double x383 = x100 + x99;
        const double x384 = x107 + x377 + x56 + x58 + x60;
        const double x385 = -x124;
        const double x386 = x127 + x202 + x385;
        const double x387 = x133 + x180;
        const double x388 = x139 + x141 + x143 + x145 + x148 + x182 + x183;
        const double x389 = x151 + x153 + x156 + x157;
        const double x390 = x135 + x179;
        const double x391 = x137 + x138 + x140 + x142 + x144 + x146 + x147;
        const double x392 = x188 + x189;
        const double x393 = x126 + x128 + x130 + x201 + x385;
        const double x394 = -x209;
        const double x395 = x212 + x275 + x394;
        const double x396 = x218 + x250;
        const double x397 = x226 + x228 + x231 + x252 + x253 + x254 + x255;
        const double x398 = x243 + x33;
        const double x399 = x220 + x249;
        const double x400 = x222 + x223 + x224 + x225 + x227 + x229 + x230;
        const double x401 = x260 + x261;
        const double x402 = x211 + x213 + x215 + x274 + x394;
        const double x403 = -x282;
        const double x404 = x302 + x403;
        const double x405 = x284 + x285 + x286;
        const double x406 = x295 + x296 + x298 + x300 + x403;
        const double x407 = x320 + x321;
        
        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = x4;
        A(0, 2) = x4;
        A(0, 6) = x4;
        A(0, 8) = x4;
        A(1, 0) = x29;
        A(1, 2) = x29;
        A(1, 3) = x38;
        A(1, 5) = x38;
        A(1, 6) = x44;
        A(1, 8) = x44;
        A(2, 0) = x61 + x66 + x75;
        A(2, 1) = x86 + x89;
        A(2, 2) = x61 + x92 + x95;
        A(2, 3) = x101 + x97 + x98;
        A(2, 4) = x105 + x32;
        A(2, 5) = x101 + x106 + x96;
        A(2, 6) = x109 + x75 + x92;
        A(2, 7) = x111 + x112;
        A(2, 8) = x109 + x66 + x95;
        A(3, 0) = x131 + x136 + x149;
        A(3, 1) = x158 + x161 + x178;
        A(3, 2) = x131 + x181 + x184;
        A(3, 3) = x186 + x187 + x190;
        A(3, 4) = x192 + x194 + x199;
        A(3, 5) = x185 + x190 + x200;
        A(3, 6) = x149 + x181 + x203;
        A(3, 7) = x158 + x206 + x207;
        A(3, 8) = x136 + x184 + x203;
        A(4, 0) = x216 + x221 + x232;
        A(4, 1) = x240 + x244 + x248;
        A(4, 2) = x216 + x251 + x256;
        A(4, 3) = x258 + x259 + x262;
        A(4, 4) = x264 + x272;
        A(4, 5) = x257 + x262 + x273;
        A(4, 6) = x232 + x251 + x276;
        A(4, 7) = x244 + x277 + x278;
        A(4, 8) = x221 + x256 + x276;
        A(5, 0) = x28 + x289 + x294 + x303;
        A(5, 1) = x308 + x311;
        A(5, 2) = x12 + x289 + x313 + x314 + x316;
        A(5, 3) = x318 + x319 + x322 + x38;
        A(5, 4) = x323;
        A(5, 5) = x242 + x322 + x324 + x325;
        A(5, 6) = x282 + x303 + x326 + x328 + x43;
        A(5, 7) = x329 + x330;
        A(5, 8) = x282 + x294 + x327 + x331 + x332 + x333;
        
        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x335;
        B(0, 2) = x335;
        B(0, 3) = x150;
        B(0, 5) = x150;
        B(0, 6) = x336;
        B(0, 8) = x336;
        B(2, 0) = x86;
        B(2, 1) = x337 + x338 - x339 - x340 + x341;
        B(2, 2) = x342;
        B(2, 3) = x34;
        B(2, 4) = x263*x64;
        B(2, 5) = -x34;
        B(2, 6) = x343;
        B(2, 7) = -x337 - x338 + x339 + x340 + x341;
        B(2, 8) = x111;
        B(3, 0) = x344 + x345;
        B(3, 1) = x346 + x347 + x348 + x349 - x350 - x351 + x352;
        B(3, 2) = x353 + x354;
        B(3, 3) = x355 + x356;
        B(3, 4) = x134*x263;
        B(3, 5) = -x355 - x356;
        B(3, 6) = x345 + x353;
        B(3, 7) = -x346 - x347 - x348 - x349 + x350 + x351 + x352;
        B(3, 8) = x344 + x354;
        B(4, 0) = x240 + x241;
        B(4, 1) = -x357 - x358 + x359 + x360 - x361 - x362 + x363;
        B(4, 2) = x364;
        B(4, 3) = x365;
        B(4, 4) = x219*x263;
        B(4, 5) = -x365;
        B(4, 6) = x366;
        B(4, 7) = x357 + x358 - x359 - x360 + x361 + x362 + x363;
        B(4, 8) = x241 + x277;
        B(5, 0) = x308;
        B(5, 1) = x265 - x266 + x271 - x367 - x368 + x369 + x370 + x371;
        B(5, 2) = x329;
        B(5, 4) = 16*a_3*d_2 + 16*a_4*d_4 + x263*x292;
        B(5, 6) = x308;
        B(5, 7) = x267 + x268 + x269 - x270 + x367 + x368 - x369 - x370 + x371;
        B(5, 8) = x329;
        
        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = x4;
        C(0, 2) = x4;
        C(0, 6) = x4;
        C(0, 8) = x4;
        C(1, 0) = x373;
        C(1, 2) = x373;
        C(1, 3) = x374;
        C(1, 5) = x374;
        C(1, 6) = x376;
        C(1, 8) = x376;
        C(2, 0) = x378 + x379 + x380;
        C(2, 1) = x342 + x89;
        C(2, 2) = x378 + x381 + x382;
        C(2, 3) = x383 + x96 + x98;
        C(2, 4) = x105 - x32;
        C(2, 5) = x106 + x383 + x97;
        C(2, 6) = x380 + x381 + x384;
        C(2, 7) = x112 + x343;
        C(2, 8) = x379 + x382 + x384;
        C(3, 0) = x386 + x387 + x388;
        C(3, 1) = x161 + x206 + x389;
        C(3, 2) = x386 + x390 + x391;
        C(3, 3) = x185 + x187 + x392;
        C(3, 4) = -x192 - x194 + x199;
        C(3, 5) = x186 + x200 + x392;
        C(3, 6) = x388 + x390 + x393;
        C(3, 7) = x178 + x207 + x389;
        C(3, 8) = x387 + x391 + x393;
        C(4, 0) = x395 + x396 + x397;
        C(4, 1) = x248 + x364 + x398;
        C(4, 2) = x395 + x399 + x400;
        C(4, 3) = x257 + x259 + x401;
        C(4, 4) = -x264 + x272;
        C(4, 5) = x258 + x273 + x401;
        C(4, 6) = x397 + x399 + x402;
        C(4, 7) = x278 + x366 + x398;
        C(4, 8) = x396 + x400 + x402;
        C(5, 0) = x16 + x291 + x316 + x326 + x404 + x405;
        C(5, 1) = x311 + x329;
        C(5, 2) = x293 + x314 + x332 + x372 + x405 + x406;
        C(5, 3) = x31 + x319 + x325 + x33 + x407;
        C(5, 4) = x323;
        C(5, 5) = x318 + x324 + x374 + x407;
        C(5, 6) = x13 + x288 + x293 + x328 + x333 + x404;
        C(5, 7) = x308 + x330;
        C(5, 8) = x288 + x291 + x313 + x327 + x375 + x406;
        
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
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor();
    // Finish code for general_6dof solution node 0
    
    // Code for equation all-zero dispatcher node 2
    auto EquationAllZeroDispatcherNode_node_2_processor = [&]() -> void
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
            const double th_2 = this_solution[2];
            const bool checked_result = std::fabs(2*a_2*a_4*std::cos(th_2) + 2*a_3*a_4 - 2*d_2*d_4) <= 9.9999999999999995e-7 && std::fabs(2*a_2*d_4*std::cos(th_2) + 2*a_3*d_4 + 2*a_4*d_2) <= 9.9999999999999995e-7 && std::fabs(-std::pow(Px, 2) + 2*Px*a_1*std::cos(th_0) - std::pow(Py, 2) + 2*Py*a_1*std::sin(th_0) - std::pow(Pz, 2) - std::pow(a_1, 2) + std::pow(a_2, 2) + 2*a_2*a_3*std::cos(th_2) + std::pow(a_3, 2) + std::pow(a_4, 2) + std::pow(d_2, 2) + std::pow(d_4, 2)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor();
    // Finish code for equation all-zero dispatcher node 2
    
    // Code for explicit solution node 3, solved variable is th_3
    auto ExplicitSolutionNode_node_3_solve_th_3_processor = [&]() -> void
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
            const double th_2 = this_solution[2];
            
            const bool condition_0 = std::fabs(2*a_2*a_4*std::cos(th_2) + 2*a_3*a_4 - 2*d_2*d_4) >= zero_tolerance || std::fabs(2*a_2*d_4*std::cos(th_2) + 2*a_3*d_4 + 2*a_4*d_2) >= zero_tolerance || std::fabs(-std::pow(Px, 2) + 2*Px*a_1*std::cos(th_0) - std::pow(Py, 2) + 2*Py*a_1*std::sin(th_0) - std::pow(Pz, 2) - std::pow(a_1, 2) + std::pow(a_2, 2) + 2*a_2*a_3*std::cos(th_2) + std::pow(a_3, 2) + std::pow(a_4, 2) + std::pow(d_2, 2) + std::pow(d_4, 2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 2*d_4;
                const double x1 = 2*a_4;
                const double x2 = a_2*std::cos(th_2);
                const double x3 = -a_3*x0 - d_2*x1 - x0*x2;
                const double x4 = a_3*x1 - d_2*x0 + x1*x2;
                const double x5 = std::atan2(x3, x4);
                const double x6 = 2*a_1;
                const double x7 = std::pow(Px, 2) - Px*x6*std::cos(th_0) + std::pow(Py, 2) - Py*x6*std::sin(th_0) + std::pow(Pz, 2) + std::pow(a_1, 2) - std::pow(a_2, 2) - std::pow(a_3, 2) - 2*a_3*x2 - std::pow(a_4, 2) - std::pow(d_2, 2) - std::pow(d_4, 2);
                const double x8 = safe_sqrt(std::pow(x3, 2) + std::pow(x4, 2) - std::pow(x7, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[3] = x5 + std::atan2(x8, x7);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = std::fabs(2*a_2*a_4*std::cos(th_2) + 2*a_3*a_4 - 2*d_2*d_4) >= zero_tolerance || std::fabs(2*a_2*d_4*std::cos(th_2) + 2*a_3*d_4 + 2*a_4*d_2) >= zero_tolerance || std::fabs(-std::pow(Px, 2) + 2*Px*a_1*std::cos(th_0) - std::pow(Py, 2) + 2*Py*a_1*std::sin(th_0) - std::pow(Pz, 2) - std::pow(a_1, 2) + std::pow(a_2, 2) + 2*a_2*a_3*std::cos(th_2) + std::pow(a_3, 2) + std::pow(a_4, 2) + std::pow(d_2, 2) + std::pow(d_4, 2)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = 2*d_4;
                const double x1 = 2*a_4;
                const double x2 = a_2*std::cos(th_2);
                const double x3 = -a_3*x0 - d_2*x1 - x0*x2;
                const double x4 = a_3*x1 - d_2*x0 + x1*x2;
                const double x5 = std::atan2(x3, x4);
                const double x6 = 2*a_1;
                const double x7 = std::pow(Px, 2) - Px*x6*std::cos(th_0) + std::pow(Py, 2) - Py*x6*std::sin(th_0) + std::pow(Pz, 2) + std::pow(a_1, 2) - std::pow(a_2, 2) - std::pow(a_3, 2) - 2*a_3*x2 - std::pow(a_4, 2) - std::pow(d_2, 2) - std::pow(d_4, 2);
                const double x8 = safe_sqrt(std::pow(x3, 2) + std::pow(x4, 2) - std::pow(x7, 2));
                // End of temp variables
                const double tmp_sol_value = x5 + std::atan2(-x8, x7);
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
    ExplicitSolutionNode_node_3_solve_th_3_processor();
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
            const bool checked_result = std::fabs(Pz) <= 9.9999999999999995e-7 && std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1) <= 9.9999999999999995e-7;
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
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            
            const bool condition_0 = std::fabs(Pz) >= 9.9999999999999995e-7 || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - a_1) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_2);
                const double x1 = std::sin(th_3);
                const double x2 = std::cos(th_3);
                const double x3 = -a_2 - a_3*x0 - a_4*x0*x2 + d_4*x0*x1;
                const double x4 = -Px*std::cos(th_0) - Py*std::sin(th_0) + a_1;
                const double x5 = a_4*x1 - d_2 + d_4*x2;
                // End of temp variables
                const double tmp_sol_value = std::atan2(Pz*x3 - x4*x5, Pz*x5 + x3*x4);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            
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
                solution_0[6] = x9;
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
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
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
            const double th_5 = this_solution[6];
            
            const bool degenerate_valid_0 = std::fabs(th_5) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(12, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_5 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(17, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(9, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor();
    // Finish code for solved_variable dispatcher node 8
    
    // Code for explicit solution node 17, solved variable is th_4th_6_soa
    auto ExplicitSolutionNode_node_17_solve_th_4th_6_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(17);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(17);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 17
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            
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
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(18, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_4th_6_soa_processor();
    // Finish code for explicit solution node 17
    
    // Code for non-branch dispatcher node 18
    // Actually, there is no code
    
    // Code for explicit solution node 19, solved variable is th_4
    auto ExplicitSolutionNode_node_19_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(18);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(18);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 19
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
                add_input_index_to(20, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_4_processor();
    // Finish code for explicit solution node 18
    
    // Code for non-branch dispatcher node 20
    // Actually, there is no code
    
    // Code for explicit solution node 21, solved variable is th_6
    auto ExplicitSolutionNode_node_21_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(20);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(20);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 21
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_4 = this_solution[4];
            const double th_4th_6_soa = this_solution[5];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_4 + th_4th_6_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_6_processor();
    // Finish code for explicit solution node 20
    
    // Code for explicit solution node 12, solved variable is negative_th_6_positive_th_4__soa
    auto ExplicitSolutionNode_node_12_solve_negative_th_6_positive_th_4__soa_processor = [&]() -> void
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            
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
                add_input_index_to(13, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_12_solve_negative_th_6_positive_th_4__soa_processor();
    // Finish code for explicit solution node 12
    
    // Code for non-branch dispatcher node 13
    // Actually, there is no code
    
    // Code for explicit solution node 14, solved variable is th_4
    auto ExplicitSolutionNode_node_14_solve_th_4_processor = [&]() -> void
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
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(15, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_4_processor();
    // Finish code for explicit solution node 13
    
    // Code for non-branch dispatcher node 15
    // Actually, there is no code
    
    // Code for explicit solution node 16, solved variable is th_6
    auto ExplicitSolutionNode_node_16_solve_th_6_processor = [&]() -> void
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
            const double negative_th_6_positive_th_4__soa = this_solution[0];
            const double th_4 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -negative_th_6_positive_th_4__soa + th_4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_6_processor();
    // Finish code for explicit solution node 15
    
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            const double th_5 = this_solution[6];
            
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
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            const double th_5 = this_solution[6];
            
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
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
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
        const double value_at_1 = raw_ik_out_i[1];  // th_1
        new_ik_i[1] = value_at_1;
        const double value_at_2 = raw_ik_out_i[2];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[3];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[4];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[6];  // th_5
        new_ik_i[5] = value_at_5;
        const double value_at_6 = raw_ik_out_i[7];  // th_6
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

}; // struct atlas_r_hand_ik

// Code below for debug
void test_ik_solve_atlas_r_hand()
{
    std::array<double, atlas_r_hand_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = atlas_r_hand_ik::computeFK(theta);
    auto ik_output = atlas_r_hand_ik::computeIK(ee_pose, theta[0]);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = atlas_r_hand_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_atlas_r_hand();
}
