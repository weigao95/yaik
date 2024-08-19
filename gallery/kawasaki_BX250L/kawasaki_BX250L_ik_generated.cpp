#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct kawasaki_BX250L_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 16;
static constexpr int n_tree_nodes = 24;
static constexpr int intermediate_solution_size = 9;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_0 = 0.21;
static constexpr double a_1 = 0.32729955698106283;
static constexpr double d_2 = 1.35;
static constexpr double d_3 = 0.343;
static constexpr double pre_transform_special_symbol_23 = 0.67;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -0.9700825434839029;
static constexpr double th_2_offset_original2raw = -2.5408788702787994;
static constexpr double th_3_offset_original2raw = 3.141592653589793;
static constexpr double th_4_offset_original2raw = 3.141592653589793;
static constexpr double th_5_offset_original2raw = 1.5707963267948968;

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
    ee_transformed(0, 0) = 6.1232339957367599e-17*r_13 + 1.0*r_23 + 5.5511151231257802e-17*r_33;
    ee_transformed(0, 1) = -6.1232339957367599e-17*r_12 - 1.0*r_22 - 5.5511151231257802e-17*r_32;
    ee_transformed(0, 2) = 6.1232339957367599e-17*r_11 + 1.0*r_21 + 5.5511151231257802e-17*r_31;
    ee_transformed(0, 3) = 6.1232339957367599e-17*Px + 1.0*Py + 5.5511151231257802e-17*Pz - 5.5511151231257802e-17*pre_transform_special_symbol_23;
    ee_transformed(1, 0) = 1.0*r_13 - 6.1232339957367697e-17*r_23;
    ee_transformed(1, 1) = -1.0*r_12 + 6.1232339957367697e-17*r_22;
    ee_transformed(1, 2) = 1.0*r_11 - 6.1232339957367697e-17*r_21;
    ee_transformed(1, 3) = 1.0*Px - 6.1232339957367697e-17*Py;
    ee_transformed(2, 0) = -3.3990776836172297e-33*r_13 - 5.5511151231257802e-17*r_23 - 1.0*r_33;
    ee_transformed(2, 1) = 3.3990776836172297e-33*r_12 + 5.5511151231257802e-17*r_22 + 1.0*r_32;
    ee_transformed(2, 2) = -3.3990776836172297e-33*r_11 - 5.5511151231257802e-17*r_21 - 1.0*r_31;
    ee_transformed(2, 3) = -3.3990776836172297e-33*Px - 5.5511151231257802e-17*Py - 1.0*Pz + 1.0*pre_transform_special_symbol_23;
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
    ee_transformed(0, 0) = 6.1232339957367599e-17*r_13 + 1.0*r_23 - 3.3990776836172297e-33*r_33;
    ee_transformed(0, 1) = -6.1232339957367599e-17*r_12 - 1.0*r_22 + 3.3990776836172297e-33*r_32;
    ee_transformed(0, 2) = 6.1232339957367599e-17*r_11 + 1.0*r_21 - 3.3990776836172297e-33*r_31;
    ee_transformed(0, 3) = 6.1232339957367599e-17*Px + 1.0*Py - 3.3990776836172297e-33*Pz;
    ee_transformed(1, 0) = 1.0*r_13 - 6.1232339957367697e-17*r_23 - 5.5511151231257802e-17*r_33;
    ee_transformed(1, 1) = -1.0*r_12 + 6.1232339957367697e-17*r_22 + 5.5511151231257802e-17*r_32;
    ee_transformed(1, 2) = 1.0*r_11 - 6.1232339957367697e-17*r_21 - 5.5511151231257802e-17*r_31;
    ee_transformed(1, 3) = 1.0*Px - 6.1232339957367697e-17*Py - 5.5511151231257802e-17*Pz;
    ee_transformed(2, 0) = 5.5511151231257802e-17*r_13 - 1.0*r_33;
    ee_transformed(2, 1) = -5.5511151231257802e-17*r_12 + 1.0*r_32;
    ee_transformed(2, 2) = 5.5511151231257802e-17*r_11 - 1.0*r_31;
    ee_transformed(2, 3) = 5.5511151231257802e-17*Px - 1.0*Pz + 1.0*pre_transform_special_symbol_23;
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
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_5);
    const double x1 = std::sin(th_0);
    const double x2 = std::cos(th_3);
    const double x3 = std::sin(th_3);
    const double x4 = std::cos(th_0);
    const double x5 = std::sin(th_1);
    const double x6 = std::sin(th_2);
    const double x7 = x5*x6;
    const double x8 = std::cos(th_1);
    const double x9 = std::cos(th_2);
    const double x10 = x8*x9;
    const double x11 = x10*x4 + x4*x7;
    const double x12 = x1*x2 - x11*x3;
    const double x13 = std::cos(th_5);
    const double x14 = std::sin(th_4);
    const double x15 = x5*x9;
    const double x16 = x6*x8;
    const double x17 = x15*x4 - x16*x4;
    const double x18 = std::cos(th_4);
    const double x19 = x1*x3 + x11*x2;
    const double x20 = -x14*x17 + x18*x19;
    const double x21 = -x14*x19 - x17*x18;
    const double x22 = a_1*x8;
    const double x23 = x1*x10 + x1*x7;
    const double x24 = -x2*x4 - x23*x3;
    const double x25 = x1*x15 - x1*x16;
    const double x26 = x2*x23 - x3*x4;
    const double x27 = -x14*x25 + x18*x26;
    const double x28 = -x14*x26 - x18*x25;
    const double x29 = x15 - x16;
    const double x30 = -x10 - x7;
    const double x31 = -x14*x30 + x18*x2*x29;
    const double x32 = -x14*x2*x29 - x18*x30;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x12 + x13*x20;
    ee_pose_raw(0, 1) = -x0*x20 - x12*x13;
    ee_pose_raw(0, 2) = x21;
    ee_pose_raw(0, 3) = a_0*x4 + d_2*x17 + d_3*x21 + x22*x4;
    ee_pose_raw(1, 0) = -x0*x24 + x13*x27;
    ee_pose_raw(1, 1) = -x0*x27 - x13*x24;
    ee_pose_raw(1, 2) = x28;
    ee_pose_raw(1, 3) = a_0*x1 + d_2*x25 + d_3*x28 + x1*x22;
    ee_pose_raw(2, 0) = x0*x29*x3 + x13*x31;
    ee_pose_raw(2, 1) = -x0*x31 + x13*x29*x3;
    ee_pose_raw(2, 2) = x32;
    ee_pose_raw(2, 3) = a_1*x5 + d_2*x30 + d_3*x32;
    return endEffectorTargetRawToOriginal(ee_pose_raw);
}

static void computeTwistJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, 6>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = std::cos(th_0);
    const double x2 = -1.0*x1;
    const double x3 = 6.1232339957367599e-17*x0 + x2;
    const double x4 = -x3;
    const double x5 = std::sin(th_2);
    const double x6 = std::sin(th_1);
    const double x7 = std::cos(th_1);
    const double x8 = 1.0*x0;
    const double x9 = 6.1232339957367599e-17*x1 + x8;
    const double x10 = -3.3990776836172297e-33*x6 + x7*x9;
    const double x11 = std::cos(th_2);
    const double x12 = -x6*x9 - 3.3990776836172297e-33*x7;
    const double x13 = -x10*x5 - x11*x12;
    const double x14 = std::cos(th_3);
    const double x15 = std::sin(th_3);
    const double x16 = x10*x11 - x12*x5;
    const double x17 = -x14*x4 - x15*x16;
    const double x18 = std::cos(th_4);
    const double x19 = std::sin(th_4);
    const double x20 = -x13*x18 - x19*(x14*x16 - x15*x4);
    const double x21 = 6.1232339957367697e-17*x1 + x8;
    const double x22 = -x21;
    const double x23 = 5.5511151231257802e-17*x6;
    const double x24 = -6.1232339957367697e-17*x0 - x2;
    const double x25 = -x23 + x24*x7;
    const double x26 = -x24*x6 - 5.5511151231257802e-17*x7;
    const double x27 = -x11*x26 - x25*x5;
    const double x28 = x11*x25 - x26*x5;
    const double x29 = -x14*x22 - x15*x28;
    const double x30 = -x18*x27 - x19*(x14*x28 - x15*x22);
    const double x31 = 5.5511151231257802e-17*x0;
    const double x32 = -x1*x23 - 1.0*x7;
    const double x33 = 5.5511151231257802e-17*x1*x7 - 1.0*x6;
    const double x34 = -x11*x32 - x33*x5;
    const double x35 = x11*x33 - x32*x5;
    const double x36 = 5.5511151231257802e-17*x0*x14 - x15*x35;
    const double x37 = -x18*x34 - x19*(x14*x35 + x15*x31);
    const double x38 = a_0*x24;
    const double x39 = 5.5511151231257802e-17*a_0*x1 + pre_transform_special_symbol_23;
    const double x40 = a_1*x33 + x39;
    const double x41 = a_1*x25 + x38;
    const double x42 = d_2*x34 + x40;
    const double x43 = d_2*x27 + x41;
    const double x44 = d_3*x37 + x42;
    const double x45 = d_3*x30 + x43;
    const double x46 = a_0*x9;
    const double x47 = a_1*x10 + x46;
    const double x48 = d_2*x13 + x47;
    const double x49 = d_3*x20 + x48;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -3.3990776836172297e-33;
    jacobian(0, 1) = x3;
    jacobian(0, 2) = x4;
    jacobian(0, 3) = x13;
    jacobian(0, 4) = x17;
    jacobian(0, 5) = x20;
    jacobian(1, 0) = -5.5511151231257802e-17;
    jacobian(1, 1) = x21;
    jacobian(1, 2) = x22;
    jacobian(1, 3) = x27;
    jacobian(1, 4) = x29;
    jacobian(1, 5) = x30;
    jacobian(2, 0) = -1.0;
    jacobian(2, 1) = x31;
    jacobian(2, 2) = -x31;
    jacobian(2, 3) = x34;
    jacobian(2, 4) = x36;
    jacobian(2, 5) = x37;
    jacobian(3, 0) = 5.5511151231257802e-17*pre_transform_special_symbol_23;
    jacobian(3, 1) = -x21*x39 + x31*x38;
    jacobian(3, 2) = -x22*x40 - x31*x41;
    jacobian(3, 3) = -x27*x42 + x34*x43;
    jacobian(3, 4) = -x29*x42 + x36*x43;
    jacobian(3, 5) = -x30*x44 + x37*x45;
    jacobian(4, 0) = -3.3990776836172297e-33*pre_transform_special_symbol_23;
    jacobian(4, 1) = x3*x39 - x31*x46;
    jacobian(4, 2) = x31*x47 + x4*x40;
    jacobian(4, 3) = x13*x42 - x34*x48;
    jacobian(4, 4) = x17*x42 - x36*x48;
    jacobian(4, 5) = x20*x44 - x37*x49;
    jacobian(5, 1) = a_0*x21*x9 - x3*x38;
    jacobian(5, 2) = x22*x47 - x4*x41;
    jacobian(5, 3) = -x13*x43 + x27*x48;
    jacobian(5, 4) = -x17*x43 + x29*x48;
    jacobian(5, 5) = -x20*x45 + x30*x49;
    return;
}

static void computeAngularVelocityJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, 6>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = std::cos(th_0);
    const double x2 = -1.0*x1;
    const double x3 = 6.1232339957367599e-17*x0 + x2;
    const double x4 = -x3;
    const double x5 = std::sin(th_2);
    const double x6 = std::sin(th_1);
    const double x7 = std::cos(th_1);
    const double x8 = 1.0*x0;
    const double x9 = 6.1232339957367599e-17*x1 + x8;
    const double x10 = -3.3990776836172297e-33*x6 + x7*x9;
    const double x11 = std::cos(th_2);
    const double x12 = -x6*x9 - 3.3990776836172297e-33*x7;
    const double x13 = -x10*x5 - x11*x12;
    const double x14 = std::cos(th_3);
    const double x15 = std::sin(th_3);
    const double x16 = x10*x11 - x12*x5;
    const double x17 = std::cos(th_4);
    const double x18 = std::sin(th_4);
    const double x19 = 6.1232339957367697e-17*x1 + x8;
    const double x20 = -x19;
    const double x21 = 5.5511151231257802e-17*x6;
    const double x22 = -6.1232339957367697e-17*x0 - x2;
    const double x23 = -x21 + x22*x7;
    const double x24 = -x22*x6 - 5.5511151231257802e-17*x7;
    const double x25 = -x11*x24 - x23*x5;
    const double x26 = x11*x23 - x24*x5;
    const double x27 = 5.5511151231257802e-17*x0;
    const double x28 = -x1*x21 - 1.0*x7;
    const double x29 = 5.5511151231257802e-17*x1*x7 - 1.0*x6;
    const double x30 = -x11*x28 - x29*x5;
    const double x31 = x11*x29 - x28*x5;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -3.3990776836172297e-33;
    jacobian(0, 1) = x3;
    jacobian(0, 2) = x4;
    jacobian(0, 3) = x13;
    jacobian(0, 4) = -x14*x4 - x15*x16;
    jacobian(0, 5) = -x13*x17 - x18*(x14*x16 - x15*x4);
    jacobian(1, 0) = -5.5511151231257802e-17;
    jacobian(1, 1) = x19;
    jacobian(1, 2) = x20;
    jacobian(1, 3) = x25;
    jacobian(1, 4) = -x14*x20 - x15*x26;
    jacobian(1, 5) = -x17*x25 - x18*(x14*x26 - x15*x20);
    jacobian(2, 0) = -1.0;
    jacobian(2, 1) = x27;
    jacobian(2, 2) = -x27;
    jacobian(2, 3) = x30;
    jacobian(2, 4) = 5.5511151231257802e-17*x0*x14 - x15*x31;
    jacobian(2, 5) = -x17*x30 - x18*(x14*x31 + x15*x27);
    return;
}

static void computeTransformPointJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Vector3d& point_on_ee, Eigen::Matrix<double, 6, 6>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    const double th_5 = theta_input_original[5] + th_5_offset_original2raw;
    const double p_on_ee_x = point_on_ee[0];
    const double p_on_ee_y = point_on_ee[1];
    const double p_on_ee_z = point_on_ee[2];
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = 5.5511151231257802e-17*x0;
    const double x2 = p_on_ee_y*x1;
    const double x3 = 1.0*x0;
    const double x4 = std::cos(th_0);
    const double x5 = x3 + 6.1232339957367697e-17*x4;
    const double x6 = -1.0*x4;
    const double x7 = -6.1232339957367697e-17*x0 - x6;
    const double x8 = a_0*x7;
    const double x9 = 5.5511151231257802e-17*a_0*x4 + pre_transform_special_symbol_23;
    const double x10 = -x5;
    const double x11 = std::sin(th_1);
    const double x12 = std::cos(th_1);
    const double x13 = -1.0*x11 + 5.5511151231257802e-17*x12*x4;
    const double x14 = a_1*x13 + x9;
    const double x15 = 5.5511151231257802e-17*x11;
    const double x16 = x12*x7 - x15;
    const double x17 = a_1*x16 + x8;
    const double x18 = std::cos(th_2);
    const double x19 = -1.0*x12 - x15*x4;
    const double x20 = std::sin(th_2);
    const double x21 = -x13*x20 - x18*x19;
    const double x22 = -x11*x7 - 5.5511151231257802e-17*x12;
    const double x23 = -x16*x20 - x18*x22;
    const double x24 = d_2*x21 + x14;
    const double x25 = d_2*x23 + x17;
    const double x26 = std::cos(th_3);
    const double x27 = std::sin(th_3);
    const double x28 = x13*x18 - x19*x20;
    const double x29 = 5.5511151231257802e-17*x0*x26 - x27*x28;
    const double x30 = x16*x18 - x20*x22;
    const double x31 = -x10*x26 - x27*x30;
    const double x32 = std::cos(th_4);
    const double x33 = std::sin(th_4);
    const double x34 = -x21*x32 - x33*(x1*x27 + x26*x28);
    const double x35 = -x23*x32 - x33*(-x10*x27 + x26*x30);
    const double x36 = d_3*x34 + x24;
    const double x37 = d_3*x35 + x25;
    const double x38 = 5.5511151231257802e-17*p_on_ee_x;
    const double x39 = x0*x38;
    const double x40 = 6.1232339957367599e-17*x0 + x6;
    const double x41 = x3 + 6.1232339957367599e-17*x4;
    const double x42 = a_0*x41;
    const double x43 = -x40;
    const double x44 = -3.3990776836172297e-33*x11 + x12*x41;
    const double x45 = a_1*x44 + x42;
    const double x46 = -x11*x41 - 3.3990776836172297e-33*x12;
    const double x47 = -x18*x46 - x20*x44;
    const double x48 = d_2*x47 + x45;
    const double x49 = x18*x44 - x20*x46;
    const double x50 = -x26*x43 - x27*x49;
    const double x51 = -x32*x47 - x33*(x26*x49 - x27*x43);
    const double x52 = d_3*x51 + x48;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = 1.0*p_on_ee_y - 5.5511151231257802e-17*p_on_ee_z + 5.5511151231257802e-17*pre_transform_special_symbol_23;
    jacobian(0, 1) = p_on_ee_z*x5 + x1*x8 - x2 - x5*x9;
    jacobian(0, 2) = p_on_ee_z*x10 - x1*x17 - x10*x14 + x2;
    jacobian(0, 3) = -p_on_ee_y*x21 + p_on_ee_z*x23 + x21*x25 - x23*x24;
    jacobian(0, 4) = -p_on_ee_y*x29 + p_on_ee_z*x31 - x24*x31 + x25*x29;
    jacobian(0, 5) = -p_on_ee_y*x34 + p_on_ee_z*x35 + x34*x37 - x35*x36;
    jacobian(1, 0) = -1.0*p_on_ee_x + 3.3990776836172297e-33*p_on_ee_z - 3.3990776836172297e-33*pre_transform_special_symbol_23;
    jacobian(1, 1) = -p_on_ee_z*x40 - x1*x42 + x39 + x40*x9;
    jacobian(1, 2) = -p_on_ee_z*x43 + 5.5511151231257802e-17*x0*x45 + x14*x43 - x39;
    jacobian(1, 3) = p_on_ee_x*x21 - p_on_ee_z*x47 - x21*x48 + x24*x47;
    jacobian(1, 4) = p_on_ee_x*x29 - p_on_ee_z*x50 + x24*x50 - x29*x48;
    jacobian(1, 5) = p_on_ee_x*x34 - p_on_ee_z*x51 - x34*x52 + x36*x51;
    jacobian(2, 0) = -3.3990776836172297e-33*p_on_ee_y + x38;
    jacobian(2, 1) = a_0*x41*x5 - p_on_ee_x*x5 + p_on_ee_y*x40 - x40*x8;
    jacobian(2, 2) = -p_on_ee_x*x10 + p_on_ee_y*x43 + x10*x45 - x17*x43;
    jacobian(2, 3) = -p_on_ee_x*x23 + p_on_ee_y*x47 + x23*x48 - x25*x47;
    jacobian(2, 4) = -p_on_ee_x*x31 + p_on_ee_y*x50 - x25*x50 + x31*x48;
    jacobian(2, 5) = -p_on_ee_x*x35 + p_on_ee_y*x51 + x35*x52 - x37*x51;
    return;
}

static void computeRawIK(const Eigen::Matrix4d& T_ee, SolutionQueue<intermediate_solution_size, max_n_solutions>& solution_queue, NodeIndexWorkspace<max_n_solutions>& node_index_workspace, std::vector<std::array<double, robot_nq>>& ik_output)
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
        const bool checked_result = std::fabs(Px - d_3*r_13) <= 9.9999999999999995e-7 && std::fabs(Py - d_3*r_23) <= 9.9999999999999995e-7;
        if (!checked_result)  // To non-degenerate node
            node_index_workspace.node_input_validity_vector[1] = true;
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_0_processor();
    // Finish code for equation all-zero dispatcher node 0
    
    // Code for explicit solution node 1, solved variable is th_0
    auto ExplicitSolutionNode_node_1_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(1);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(1);
        if (!this_input_valid)
            return;
        
        // The explicit solution of root node
        const bool condition_0 = std::fabs(Px - d_3*r_13) >= zero_tolerance || std::fabs(Py - d_3*r_23) >= zero_tolerance;
        if (condition_0)
        {
            // Temp variable for efficiency
            const double x0 = std::atan2(Py - d_3*r_23, Px - d_3*r_13);
            // End of temp variables
            
            auto solution_0 = make_raw_solution();
            solution_0[2] = x0;
            int appended_idx = append_solution_to_queue(solution_0);
            add_input_index_to(2, appended_idx);
        }
        
        const bool condition_1 = std::fabs(Px - d_3*r_13) >= zero_tolerance || std::fabs(Py - d_3*r_23) >= zero_tolerance;
        if (condition_1)
        {
            // Temp variable for efficiency
            const double x0 = std::atan2(Py - d_3*r_23, Px - d_3*r_13);
            // End of temp variables
            
            auto solution_1 = make_raw_solution();
            solution_1[2] = x0 + M_PI;
            int appended_idx = append_solution_to_queue(solution_1);
            add_input_index_to(2, appended_idx);
        }
        
    };
    // Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_0_processor();
    // Finish code for explicit solution node 1
    
    // Code for non-branch dispatcher node 2
    // Actually, there is no code
    
    // Code for explicit solution node 3, solved variable is th_2
    auto ExplicitSolutionNode_node_3_solve_th_2_processor = [&]() -> void
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
            const double th_0 = this_solution[2];
            
            const bool condition_0 = (1.0/2.0)*std::fabs((std::pow(Px, 2) - 2*Px*a_0*std::cos(th_0) - 2*Px*d_3*r_13 + std::pow(Py, 2) - 2*Py*a_0*std::sin(th_0) - 2*Py*d_3*r_23 + std::pow(Pz, 2) - 2*Pz*d_3*r_33 + std::pow(a_0, 2) + 2*a_0*d_3*r_13*std::cos(th_0) + 2*a_0*d_3*r_23*std::sin(th_0) - std::pow(a_1, 2) - std::pow(d_2, 2) + std::pow(d_3, 2)*std::pow(r_13, 2) + std::pow(d_3, 2)*std::pow(r_23, 2) + std::pow(d_3, 2)*std::pow(r_33, 2))/(a_1*d_2)) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 2*Px;
                const double x1 = d_3*r_13;
                const double x2 = 2*Py;
                const double x3 = d_3*r_23;
                const double x4 = a_0*std::cos(th_0);
                const double x5 = a_0*std::sin(th_0);
                const double x6 = std::pow(d_3, 2);
                const double x7 = std::asin((1.0/2.0)*(std::pow(Px, 2) + std::pow(Py, 2) + std::pow(Pz, 2) - 2*Pz*d_3*r_33 + std::pow(a_0, 2) - std::pow(a_1, 2) - std::pow(d_2, 2) + std::pow(r_13, 2)*x6 + std::pow(r_23, 2)*x6 + std::pow(r_33, 2)*x6 - x0*x1 - x0*x4 + 2*x1*x4 - x2*x3 - x2*x5 + 2*x3*x5)/(a_1*d_2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = -x7;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = (1.0/2.0)*std::fabs((std::pow(Px, 2) - 2*Px*a_0*std::cos(th_0) - 2*Px*d_3*r_13 + std::pow(Py, 2) - 2*Py*a_0*std::sin(th_0) - 2*Py*d_3*r_23 + std::pow(Pz, 2) - 2*Pz*d_3*r_33 + std::pow(a_0, 2) + 2*a_0*d_3*r_13*std::cos(th_0) + 2*a_0*d_3*r_23*std::sin(th_0) - std::pow(a_1, 2) - std::pow(d_2, 2) + std::pow(d_3, 2)*std::pow(r_13, 2) + std::pow(d_3, 2)*std::pow(r_23, 2) + std::pow(d_3, 2)*std::pow(r_33, 2))/(a_1*d_2)) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = 2*Px;
                const double x1 = d_3*r_13;
                const double x2 = 2*Py;
                const double x3 = d_3*r_23;
                const double x4 = a_0*std::cos(th_0);
                const double x5 = a_0*std::sin(th_0);
                const double x6 = std::pow(d_3, 2);
                const double x7 = std::asin((1.0/2.0)*(std::pow(Px, 2) + std::pow(Py, 2) + std::pow(Pz, 2) - 2*Pz*d_3*r_33 + std::pow(a_0, 2) - std::pow(a_1, 2) - std::pow(d_2, 2) + std::pow(r_13, 2)*x6 + std::pow(r_23, 2)*x6 + std::pow(r_33, 2)*x6 - x0*x1 - x0*x4 + 2*x1*x4 - x2*x3 - x2*x5 + 2*x3*x5)/(a_1*d_2));
                // End of temp variables
                const double tmp_sol_value = x7 + M_PI;
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
    // Finish code for explicit solution node 2
    
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
            const double th_0 = this_solution[2];
            const bool checked_result = std::fabs(Pz - d_3*r_33) <= 9.9999999999999995e-7 && std::fabs(-Px*std::cos(th_0) - Py*std::sin(th_0) + a_0 + d_3*r_13*std::cos(th_0) + d_3*r_23*std::sin(th_0)) <= 9.9999999999999995e-7;
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
            const double th_0 = this_solution[2];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = std::fabs(Pz - d_3*r_33) >= 9.9999999999999995e-7 || std::fabs(-Px*std::cos(th_0) - Py*std::sin(th_0) + a_0 + d_3*r_13*std::cos(th_0) + d_3*r_23*std::sin(th_0)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = -Pz + d_3*r_33;
                const double x1 = -a_1 + d_2*std::sin(th_2);
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = -Px*x2 - Py*x3 + a_0 + d_3*r_13*x2 + d_3*r_23*x3;
                const double x5 = d_2*std::cos(th_2);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*x1 - x4*x5, x0*x5 + x1*x4);
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
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
    
    // Code for explicit solution node 7, solved variable is negative_th_2_positive_th_1__soa
    auto ExplicitSolutionNode_node_7_solve_negative_th_2_positive_th_1__soa_processor = [&]() -> void
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
            const double th_1 = this_solution[3];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_1 - th_2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(8, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_7_solve_negative_th_2_positive_th_1__soa_processor();
    // Finish code for explicit solution node 6
    
    // Code for non-branch dispatcher node 8
    // Actually, there is no code
    
    // Code for explicit solution node 9, solved variable is th_4
    auto ExplicitSolutionNode_node_9_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(8);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(8);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 9
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[2];
            const double th_1 = this_solution[3];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = std::fabs(r_13*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_23*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::sin(th_0) - r_33*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_1);
                const double x1 = std::sin(th_2);
                const double x2 = std::cos(th_1);
                const double x3 = std::cos(th_2);
                const double x4 = x0*x3 - x1*x2;
                const double x5 = std::acos(-r_13*x4*std::cos(th_0) - r_23*x4*std::sin(th_0) + r_33*(x0*x1 + x2*x3));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[7] = x5;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(10, appended_idx);
            }
            
            const bool condition_1 = std::fabs(r_13*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_23*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::sin(th_0) - r_33*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_1);
                const double x1 = std::sin(th_2);
                const double x2 = std::cos(th_1);
                const double x3 = std::cos(th_2);
                const double x4 = x0*x3 - x1*x2;
                const double x5 = std::acos(-r_13*x4*std::cos(th_0) - r_23*x4*std::sin(th_0) + r_33*(x0*x1 + x2*x3));
                // End of temp variables
                const double tmp_sol_value = -x5;
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
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
    // Finish code for explicit solution node 8
    
    // Code for solved_variable dispatcher node 10
    auto SolvedVariableDispatcherNode_node_10_processor = [&]() -> void
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
            bool taken_by_degenerate = false;
            const double th_4 = this_solution[7];
            
            const bool degenerate_valid_0 = std::fabs(th_4) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(14, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_4 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(19, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(11, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_10_processor();
    // Finish code for solved_variable dispatcher node 10
    
    // Code for explicit solution node 19, solved variable is th_3th_5_soa
    auto ExplicitSolutionNode_node_19_solve_th_3th_5_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(19);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(19);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 19
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[2];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(20, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_3th_5_soa_processor();
    // Finish code for explicit solution node 19
    
    // Code for non-branch dispatcher node 20
    // Actually, there is no code
    
    // Code for explicit solution node 21, solved variable is th_3
    auto ExplicitSolutionNode_node_21_solve_th_3_processor = [&]() -> void
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
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(22, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_3_processor();
    // Finish code for explicit solution node 20
    
    // Code for non-branch dispatcher node 22
    // Actually, there is no code
    
    // Code for explicit solution node 23, solved variable is th_5
    auto ExplicitSolutionNode_node_23_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(22);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(22);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 23
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_3 = this_solution[5];
            const double th_3th_5_soa = this_solution[6];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_3 + th_3th_5_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_5_processor();
    // Finish code for explicit solution node 22
    
    // Code for explicit solution node 14, solved variable is negative_th_5_positive_th_3__soa
    auto ExplicitSolutionNode_node_14_solve_negative_th_5_positive_th_3__soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(14);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(14);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 14
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[2];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_11*x0 - r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(15, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_14_solve_negative_th_5_positive_th_3__soa_processor();
    // Finish code for explicit solution node 14
    
    // Code for non-branch dispatcher node 15
    // Actually, there is no code
    
    // Code for explicit solution node 16, solved variable is th_3
    auto ExplicitSolutionNode_node_16_solve_th_3_processor = [&]() -> void
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
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(17, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_3_processor();
    // Finish code for explicit solution node 15
    
    // Code for non-branch dispatcher node 17
    // Actually, there is no code
    
    // Code for explicit solution node 18, solved variable is th_5
    auto ExplicitSolutionNode_node_18_solve_th_5_processor = [&]() -> void
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
            const double negative_th_5_positive_th_3__soa = this_solution[1];
            const double th_3 = this_solution[5];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -negative_th_5_positive_th_3__soa + th_3;
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_5_processor();
    // Finish code for explicit solution node 17
    
    // Code for explicit solution node 11, solved variable is th_3
    auto ExplicitSolutionNode_node_11_solve_th_3_processor = [&]() -> void
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
            const double th_0 = this_solution[2];
            const double th_1 = this_solution[3];
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[7];
            
            const bool condition_0 = std::fabs(r_13*std::sin(th_0) - r_23*std::cos(th_0)) >= zero_tolerance || std::fabs(r_13*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + r_23*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_0) + r_33*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))) >= zero_tolerance || std::fabs(std::sin(th_4)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_4);
                const double x1 = std::sin(th_0);
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_1);
                const double x4 = std::cos(th_2);
                const double x5 = std::sin(th_2);
                const double x6 = std::cos(th_1);
                const double x7 = x3*x5 + x4*x6;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(-r_13*x1 + r_23*x2), x0*(-r_13*x2*x7 - r_23*x1*x7 - r_33*(x3*x4 - x5*x6)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(12, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_3_processor();
    // Finish code for explicit solution node 11
    
    // Code for non-branch dispatcher node 12
    // Actually, there is no code
    
    // Code for explicit solution node 13, solved variable is th_5
    auto ExplicitSolutionNode_node_13_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(12);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(12);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 13
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[2];
            const double th_1 = this_solution[3];
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[7];
            
            const bool condition_0 = std::fabs(r_11*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_21*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::sin(th_0) - r_31*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) >= zero_tolerance || std::fabs(r_12*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_22*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::sin(th_0) - r_32*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) >= zero_tolerance || std::fabs(std::sin(th_4)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_4);
                const double x1 = std::sin(th_1);
                const double x2 = std::sin(th_2);
                const double x3 = std::cos(th_1);
                const double x4 = std::cos(th_2);
                const double x5 = x1*x2 + x3*x4;
                const double x6 = x1*x4 - x2*x3;
                const double x7 = x6*std::cos(th_0);
                const double x8 = x6*std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(r_12*x7 + r_22*x8 - r_32*x5), x0*(-r_11*x7 - r_21*x8 + r_31*x5));
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_5_processor();
    // Finish code for explicit solution node 12
    
    // Collect the output
    for(int i = 0; i < solution_queue.size(); i++)
    {
        if(!solution_queue.solutions_validity[i])
            continue;
        const auto& raw_ik_out_i = solution_queue.get_solution(i);
        std::array<double, robot_nq> new_ik_i;
        const double value_at_0 = raw_ik_out_i[2];  // th_0
        new_ik_i[0] = value_at_0;
        const double value_at_1 = raw_ik_out_i[3];  // th_1
        new_ik_i[1] = value_at_1;
        const double value_at_2 = raw_ik_out_i[4];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[5];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[7];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[8];  // th_5
        new_ik_i[5] = value_at_5;
        ik_output.push_back(new_ik_i);
    }
}

static void computeRawIK(const Eigen::Matrix4d& T_ee_raw, RawIKWorksace& workspace)
{
    workspace.raw_ik_out.clear();
    workspace.raw_ik_out.reserve(max_n_solutions);
    computeRawIK(T_ee_raw, workspace.solution_queue, workspace.node_index_workspace, workspace.raw_ik_out);
}

static void computeIKUnChecked(const Eigen::Matrix4d& T_ee, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, workspace);
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
        ik_output.push_back(ik_out_i);
    }
}

static void computeIK(const Eigen::Matrix4d& T_ee, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, workspace);
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
        const Eigen::Matrix4d ee_pose_i = computeFK(ik_out_i);
        double ee_pose_diff = (ee_pose_i - T_ee).squaredNorm();
        if (ee_pose_diff < pose_tolerance)
            ik_output.push_back(ik_out_i);
    }
    if (!ik_output.empty()) return;
    
    // Disturbing method for degenerate handling
    Eigen::Matrix4d T_ee_raw_disturbed = yaik_cpp::disturbTransform(T_ee_raw);
    Eigen::Matrix4d T_ee_disturbed = endEffectorTargetRawToOriginal(T_ee_raw_disturbed);
    computeRawIK(T_ee_raw_disturbed, workspace);
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

static std::vector<std::array<double, robot_nq>> computeIK(const Eigen::Matrix4d& T_ee)
{
    std::vector<std::array<double, robot_nq>> ik_output;
    RawIKWorksace raw_ik_workspace;
    computeIK(T_ee, raw_ik_workspace, ik_output);
    return ik_output;
}

}; // struct kawasaki_BX250L_ik

// Code below for debug
void test_ik_solve_kawasaki_BX250L()
{
    std::array<double, kawasaki_BX250L_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = kawasaki_BX250L_ik::computeFK(theta);
    auto ik_output = kawasaki_BX250L_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = kawasaki_BX250L_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_kawasaki_BX250L();
}