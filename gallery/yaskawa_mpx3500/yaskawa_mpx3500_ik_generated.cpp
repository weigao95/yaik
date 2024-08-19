#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct yaskawa_mpx3500_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 128;
static constexpr int n_tree_nodes = 14;
static constexpr int intermediate_solution_size = 7;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_0 = 1.3;
static constexpr double alpha_3 = -1.0470795620393143;
static constexpr double alpha_5 = -1.0470795620393143;
static constexpr double d_1 = -1.4;
static constexpr double d_2 = -0.11397670814688408;
static constexpr double d_4 = -0.12300000000000003;
static constexpr double pre_transform_special_symbol_23 = 0.8;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -1.5707963267948966;
static constexpr double th_2_offset_original2raw = -0.0;
static constexpr double th_3_offset_original2raw = -1.5707963267948966;
static constexpr double th_4_offset_original2raw = 3.141592653589793;
static constexpr double th_5_offset_original2raw = -1.5707963267948966;

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
    ee_transformed(0, 0) = 1.0*r_13;
    ee_transformed(0, 1) = 1.0*r_12;
    ee_transformed(0, 2) = -1.0*r_11;
    ee_transformed(0, 3) = 1.0*Px;
    ee_transformed(1, 0) = 1.0*r_23;
    ee_transformed(1, 1) = 1.0*r_22;
    ee_transformed(1, 2) = -1.0*r_21;
    ee_transformed(1, 3) = 1.0*Py;
    ee_transformed(2, 0) = 1.0*r_33;
    ee_transformed(2, 1) = 1.0*r_32;
    ee_transformed(2, 2) = -1.0*r_31;
    ee_transformed(2, 3) = 1.0*Pz - 1.0*pre_transform_special_symbol_23;
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
    ee_transformed(0, 0) = -1.0*r_13;
    ee_transformed(0, 1) = 1.0*r_12;
    ee_transformed(0, 2) = 1.0*r_11;
    ee_transformed(0, 3) = 1.0*Px;
    ee_transformed(1, 0) = -1.0*r_23;
    ee_transformed(1, 1) = 1.0*r_22;
    ee_transformed(1, 2) = 1.0*r_21;
    ee_transformed(1, 3) = 1.0*Py;
    ee_transformed(2, 0) = -1.0*r_33;
    ee_transformed(2, 1) = 1.0*r_32;
    ee_transformed(2, 2) = 1.0*r_31;
    ee_transformed(2, 3) = 1.0*Pz + 1.0*pre_transform_special_symbol_23;
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
    const double x1 = std::cos(alpha_3);
    const double x2 = std::cos(th_0);
    const double x3 = std::sin(th_1);
    const double x4 = std::cos(th_2);
    const double x5 = x3*x4;
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_1);
    const double x8 = x6*x7;
    const double x9 = x2*x5 - x2*x8;
    const double x10 = x1*x9;
    const double x11 = std::sin(alpha_3);
    const double x12 = std::sin(th_0);
    const double x13 = std::cos(th_3);
    const double x14 = std::sin(th_3);
    const double x15 = x3*x6;
    const double x16 = x4*x7;
    const double x17 = x15*x2 + x16*x2;
    const double x18 = -x12*x13 - x14*x17;
    const double x19 = x11*x18;
    const double x20 = x10 - x19;
    const double x21 = std::sin(alpha_5);
    const double x22 = x20*x21;
    const double x23 = std::cos(th_5);
    const double x24 = std::sin(th_4);
    const double x25 = x11*x9;
    const double x26 = std::cos(th_4);
    const double x27 = -x12*x14 + x13*x17;
    const double x28 = x1*x18;
    const double x29 = x24*x25 + x24*x28 + x26*x27;
    const double x30 = std::cos(alpha_5);
    const double x31 = -x24*x27 + x25*x26 + x26*x28;
    const double x32 = x30*x31;
    const double x33 = x20*x30;
    const double x34 = x21*x31;
    const double x35 = a_0*x7;
    const double x36 = x12*x5 - x12*x8;
    const double x37 = x1*x36;
    const double x38 = x12*x15 + x12*x16;
    const double x39 = x13*x2 - x14*x38;
    const double x40 = x11*x39;
    const double x41 = x37 - x40;
    const double x42 = x21*x41;
    const double x43 = x11*x36;
    const double x44 = x13*x38 + x14*x2;
    const double x45 = x1*x39;
    const double x46 = x24*x43 + x24*x45 + x26*x44;
    const double x47 = -x24*x44 + x26*x43 + x26*x45;
    const double x48 = x30*x47;
    const double x49 = x30*x41;
    const double x50 = x21*x47;
    const double x51 = x15 + x16;
    const double x52 = x1*x51;
    const double x53 = -x5 + x8;
    const double x54 = x14*x53;
    const double x55 = x11*x54;
    const double x56 = x52 + x55;
    const double x57 = x21*x56;
    const double x58 = x11*x51;
    const double x59 = x13*x53;
    const double x60 = x1*x54;
    const double x61 = x24*x58 - x24*x60 + x26*x59;
    const double x62 = -x24*x59 + x26*x58 - x26*x60;
    const double x63 = x30*x62;
    const double x64 = x30*x56;
    const double x65 = x21*x62;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = x0*x22 + x0*x32 + x23*x29;
    ee_pose_raw(0, 1) = -x0*x29 + x22*x23 + x23*x32;
    ee_pose_raw(0, 2) = x33 - x34;
    ee_pose_raw(0, 3) = d_1*x9 + d_2*x10 - d_2*x19 + d_4*x33 - d_4*x34 + x2*x35;
    ee_pose_raw(1, 0) = x0*x42 + x0*x48 + x23*x46;
    ee_pose_raw(1, 1) = -x0*x46 + x23*x42 + x23*x48;
    ee_pose_raw(1, 2) = x49 - x50;
    ee_pose_raw(1, 3) = d_1*x36 + d_2*x37 - d_2*x40 + d_4*x49 - d_4*x50 + x12*x35;
    ee_pose_raw(2, 0) = x0*x57 + x0*x63 + x23*x61;
    ee_pose_raw(2, 1) = -x0*x61 + x23*x57 + x23*x63;
    ee_pose_raw(2, 2) = x64 - x65;
    ee_pose_raw(2, 3) = -a_0*x3 + d_1*x51 + d_2*x52 + d_2*x55 + d_4*x64 - d_4*x65;
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
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = 1.0*x0;
    const double x2 = std::sin(th_1);
    const double x3 = std::cos(th_2);
    const double x4 = std::cos(th_0);
    const double x5 = 1.0*x4;
    const double x6 = x3*x5;
    const double x7 = std::cos(th_1);
    const double x8 = std::sin(th_2);
    const double x9 = x5*x8;
    const double x10 = x2*x6 - x7*x9;
    const double x11 = std::cos(alpha_3);
    const double x12 = x10*x11;
    const double x13 = std::sin(alpha_3);
    const double x14 = std::cos(th_3);
    const double x15 = std::sin(th_3);
    const double x16 = x2*x9 + x6*x7;
    const double x17 = -x1*x14 - x15*x16;
    const double x18 = x13*x17;
    const double x19 = x12 - x18;
    const double x20 = std::cos(alpha_5);
    const double x21 = x19*x20;
    const double x22 = std::sin(alpha_5);
    const double x23 = std::cos(th_4);
    const double x24 = x13*x23;
    const double x25 = std::sin(th_4);
    const double x26 = x11*x23;
    const double x27 = x22*(x10*x24 + x17*x26 - x25*(-x1*x15 + x14*x16));
    const double x28 = x21 - x27;
    const double x29 = x1*x3;
    const double x30 = x1*x8;
    const double x31 = x2*x29 - x30*x7;
    const double x32 = x11*x31;
    const double x33 = x2*x30 + x29*x7;
    const double x34 = x14*x5 - x15*x33;
    const double x35 = x13*x34;
    const double x36 = x32 - x35;
    const double x37 = x20*x36;
    const double x38 = x22*(x24*x31 - x25*(x14*x33 + x15*x5) + x26*x34);
    const double x39 = x37 - x38;
    const double x40 = 1.0*x2;
    const double x41 = 1.0*x7;
    const double x42 = x3*x41 + x40*x8;
    const double x43 = x11*x42;
    const double x44 = -x3*x40 + x41*x8;
    const double x45 = x15*x44;
    const double x46 = x13*x45;
    const double x47 = x43 + x46;
    const double x48 = x20*x47;
    const double x49 = x22*(-x14*x25*x44 + x24*x42 - x26*x45);
    const double x50 = x48 - x49;
    const double x51 = -a_0*x40 + pre_transform_special_symbol_23;
    const double x52 = a_0*x7;
    const double x53 = d_1*x31 + x1*x52;
    const double x54 = d_1*x42 + x51;
    const double x55 = d_2*x43 + d_2*x46 + x54;
    const double x56 = d_2*x32 - d_2*x35 + x53;
    const double x57 = d_4*x48 - d_4*x49 + x55;
    const double x58 = d_4*x37 - d_4*x38 + x56;
    const double x59 = d_1*x10 + x5*x52;
    const double x60 = d_2*x12 - d_2*x18 + x59;
    const double x61 = d_4*x21 - d_4*x27 + x60;
    const double x62 = a_0*x41;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = -x1;
    jacobian(0, 2) = x1;
    jacobian(0, 3) = x10;
    jacobian(0, 4) = x19;
    jacobian(0, 5) = x28;
    jacobian(1, 1) = x5;
    jacobian(1, 2) = -x5;
    jacobian(1, 3) = x31;
    jacobian(1, 4) = x36;
    jacobian(1, 5) = x39;
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x42;
    jacobian(2, 4) = x47;
    jacobian(2, 5) = x50;
    jacobian(3, 1) = -pre_transform_special_symbol_23*x5;
    jacobian(3, 2) = x5*x51;
    jacobian(3, 3) = -x31*x54 + x42*x53;
    jacobian(3, 4) = -x36*x55 + x47*x56;
    jacobian(3, 5) = -x39*x57 + x50*x58;
    jacobian(4, 1) = -pre_transform_special_symbol_23*x1;
    jacobian(4, 2) = x1*x51;
    jacobian(4, 3) = x10*x54 - x42*x59;
    jacobian(4, 4) = x19*x55 - x47*x60;
    jacobian(4, 5) = x28*x57 - x50*x61;
    jacobian(5, 2) = -std::pow(x0, 2)*x62 - std::pow(x4, 2)*x62;
    jacobian(5, 3) = -x10*x53 + x31*x59;
    jacobian(5, 4) = -x19*x56 + x36*x60;
    jacobian(5, 5) = -x28*x58 + x39*x61;
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
    
    // Temp variable for efficiency
    const double x0 = 1.0*std::sin(th_0);
    const double x1 = std::sin(th_1);
    const double x2 = std::cos(th_2);
    const double x3 = 1.0*std::cos(th_0);
    const double x4 = x2*x3;
    const double x5 = std::cos(th_1);
    const double x6 = std::sin(th_2);
    const double x7 = x3*x6;
    const double x8 = x1*x4 - x5*x7;
    const double x9 = std::cos(alpha_3);
    const double x10 = std::sin(alpha_3);
    const double x11 = std::cos(th_3);
    const double x12 = std::sin(th_3);
    const double x13 = x1*x7 + x4*x5;
    const double x14 = -x0*x11 - x12*x13;
    const double x15 = -x10*x14 + x8*x9;
    const double x16 = std::cos(alpha_5);
    const double x17 = std::sin(alpha_5);
    const double x18 = std::cos(th_4);
    const double x19 = x10*x18;
    const double x20 = std::sin(th_4);
    const double x21 = x18*x9;
    const double x22 = x0*x2;
    const double x23 = x0*x6;
    const double x24 = x1*x22 - x23*x5;
    const double x25 = x1*x23 + x22*x5;
    const double x26 = x11*x3 - x12*x25;
    const double x27 = -x10*x26 + x24*x9;
    const double x28 = 1.0*x1;
    const double x29 = 1.0*x5;
    const double x30 = x2*x29 + x28*x6;
    const double x31 = -x2*x28 + x29*x6;
    const double x32 = x12*x31;
    const double x33 = x10*x32 + x30*x9;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = -x0;
    jacobian(0, 2) = x0;
    jacobian(0, 3) = x8;
    jacobian(0, 4) = x15;
    jacobian(0, 5) = x15*x16 - x17*(x14*x21 + x19*x8 - x20*(-x0*x12 + x11*x13));
    jacobian(1, 1) = x3;
    jacobian(1, 2) = -x3;
    jacobian(1, 3) = x24;
    jacobian(1, 4) = x27;
    jacobian(1, 5) = x16*x27 - x17*(x19*x24 - x20*(x11*x25 + x12*x3) + x21*x26);
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x30;
    jacobian(2, 4) = x33;
    jacobian(2, 5) = x16*x33 - x17*(-x11*x20*x31 + x19*x30 - x21*x32);
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
    const double p_on_ee_x = point_on_ee[0];
    const double p_on_ee_y = point_on_ee[1];
    const double p_on_ee_z = point_on_ee[2];
    
    // Temp variable for efficiency
    const double x0 = 1.0*p_on_ee_y;
    const double x1 = std::cos(th_0);
    const double x2 = 1.0*x1;
    const double x3 = p_on_ee_z*x2;
    const double x4 = std::sin(th_1);
    const double x5 = 1.0*x4;
    const double x6 = -a_0*x5 + pre_transform_special_symbol_23;
    const double x7 = std::sin(th_2);
    const double x8 = x5*x7;
    const double x9 = std::cos(th_2);
    const double x10 = std::cos(th_1);
    const double x11 = 1.0*x10;
    const double x12 = x11*x9;
    const double x13 = x12 + x8;
    const double x14 = std::sin(th_0);
    const double x15 = x5*x9;
    const double x16 = x11*x7;
    const double x17 = x14*x15 - x14*x16;
    const double x18 = a_0*x11;
    const double x19 = d_1*x17 + x14*x18;
    const double x20 = d_1*x13 + x6;
    const double x21 = std::cos(alpha_3);
    const double x22 = x13*x21;
    const double x23 = std::sin(alpha_3);
    const double x24 = std::sin(th_3);
    const double x25 = -x15 + x16;
    const double x26 = x24*x25;
    const double x27 = x23*x26;
    const double x28 = x22 + x27;
    const double x29 = x17*x21;
    const double x30 = std::cos(th_3);
    const double x31 = x12*x14 + x14*x8;
    const double x32 = x2*x30 - x24*x31;
    const double x33 = x23*x32;
    const double x34 = x29 - x33;
    const double x35 = d_2*x22 + d_2*x27 + x20;
    const double x36 = d_2*x29 - d_2*x33 + x19;
    const double x37 = std::cos(alpha_5);
    const double x38 = x28*x37;
    const double x39 = std::sin(alpha_5);
    const double x40 = std::cos(th_4);
    const double x41 = x23*x40;
    const double x42 = std::sin(th_4);
    const double x43 = x21*x40;
    const double x44 = x39*(x13*x41 - x25*x30*x42 - x26*x43);
    const double x45 = x38 - x44;
    const double x46 = x34*x37;
    const double x47 = x39*(x17*x41 + x32*x43 - x42*(x2*x24 + x30*x31));
    const double x48 = x46 - x47;
    const double x49 = d_4*x38 - d_4*x44 + x35;
    const double x50 = d_4*x46 - d_4*x47 + x36;
    const double x51 = 1.0*p_on_ee_x;
    const double x52 = 1.0*x14;
    const double x53 = p_on_ee_z*x52;
    const double x54 = x2*x9;
    const double x55 = x10*x2;
    const double x56 = x4*x54 - x55*x7;
    const double x57 = a_0*x55 + d_1*x56;
    const double x58 = x21*x56;
    const double x59 = x10*x54 + x2*x4*x7;
    const double x60 = -x24*x59 - x30*x52;
    const double x61 = x23*x60;
    const double x62 = x58 - x61;
    const double x63 = d_2*x58 - d_2*x61 + x57;
    const double x64 = x37*x62;
    const double x65 = x39*(x41*x56 - x42*(-x24*x52 + x30*x59) + x43*x60);
    const double x66 = x64 - x65;
    const double x67 = d_4*x64 - d_4*x65 + x63;
    const double x68 = x1*x51;
    const double x69 = x0*x14;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = -pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = x2*x6 - x3;
    jacobian(0, 3) = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x19 - x17*x20;
    jacobian(0, 4) = -p_on_ee_y*x28 + p_on_ee_z*x34 + x28*x36 - x34*x35;
    jacobian(0, 5) = -p_on_ee_y*x45 + p_on_ee_z*x48 + x45*x50 - x48*x49;
    jacobian(1, 0) = x51;
    jacobian(1, 1) = -pre_transform_special_symbol_23*x52 + x53;
    jacobian(1, 2) = x52*x6 - x53;
    jacobian(1, 3) = p_on_ee_x*x13 - p_on_ee_z*x56 - x13*x57 + x20*x56;
    jacobian(1, 4) = p_on_ee_x*x28 - p_on_ee_z*x62 - x28*x63 + x35*x62;
    jacobian(1, 5) = p_on_ee_x*x45 - p_on_ee_z*x66 - x45*x67 + x49*x66;
    jacobian(2, 1) = -x68 - x69;
    jacobian(2, 2) = -std::pow(x1, 2)*x18 - std::pow(x14, 2)*x18 + x68 + x69;
    jacobian(2, 3) = -p_on_ee_x*x17 + p_on_ee_y*x56 + x17*x57 - x19*x56;
    jacobian(2, 4) = -p_on_ee_x*x34 + p_on_ee_y*x62 + x34*x63 - x36*x62;
    jacobian(2, 5) = -p_on_ee_x*x48 + p_on_ee_y*x66 + x48*x67 - x50*x66;
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
    
    // Code for non-branch dispatcher node 0
    // Actually, there is no code
    
    // Code for general_6dof solution node 1, solved variable is th_3
    auto General6DoFNumericalReduceSolutionNode_node_1_solve_th_3_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;
        
        // The general 6-dof solution of root node
        Eigen::Matrix<double, 8, 8> R_l;
        R_l.setZero();
        R_l(0, 0) = -d_2*r_21*std::sin(alpha_5);
        R_l(0, 1) = -d_2*r_22*std::sin(alpha_5);
        R_l(0, 2) = -d_2*r_11*std::sin(alpha_5);
        R_l(0, 3) = -d_2*r_12*std::sin(alpha_5);
        R_l(0, 4) = Py - d_2*r_23*std::cos(alpha_5) - d_4*r_23;
        R_l(0, 5) = Px - d_2*r_13*std::cos(alpha_5) - d_4*r_13;
        R_l(1, 0) = -d_2*r_11*std::sin(alpha_5);
        R_l(1, 1) = -d_2*r_12*std::sin(alpha_5);
        R_l(1, 2) = d_2*r_21*std::sin(alpha_5);
        R_l(1, 3) = d_2*r_22*std::sin(alpha_5);
        R_l(1, 4) = Px - d_2*r_13*std::cos(alpha_5) - d_4*r_13;
        R_l(1, 5) = -Py + d_2*r_23*std::cos(alpha_5) + d_4*r_23;
        R_l(2, 6) = -d_2*r_31*std::sin(alpha_5);
        R_l(2, 7) = -d_2*r_32*std::sin(alpha_5);
        R_l(3, 0) = r_21*std::sin(alpha_5);
        R_l(3, 1) = r_22*std::sin(alpha_5);
        R_l(3, 2) = r_11*std::sin(alpha_5);
        R_l(3, 3) = r_12*std::sin(alpha_5);
        R_l(3, 4) = r_23*std::cos(alpha_5);
        R_l(3, 5) = r_13*std::cos(alpha_5);
        R_l(4, 0) = r_11*std::sin(alpha_5);
        R_l(4, 1) = r_12*std::sin(alpha_5);
        R_l(4, 2) = -r_21*std::sin(alpha_5);
        R_l(4, 3) = -r_22*std::sin(alpha_5);
        R_l(4, 4) = r_13*std::cos(alpha_5);
        R_l(4, 5) = -r_23*std::cos(alpha_5);
        R_l(5, 6) = -2*Px*d_2*r_11*std::sin(alpha_5) - 2*Py*d_2*r_21*std::sin(alpha_5) - 2*Pz*d_2*r_31*std::sin(alpha_5) + 2*std::pow(d_2, 2)*r_11*r_13*std::sin(alpha_5)*std::cos(alpha_5) + 2*std::pow(d_2, 2)*r_21*r_23*std::sin(alpha_5)*std::cos(alpha_5) + 2*std::pow(d_2, 2)*r_31*r_33*std::sin(alpha_5)*std::cos(alpha_5) + 2*d_2*d_4*r_11*r_13*std::sin(alpha_5) + 2*d_2*d_4*r_21*r_23*std::sin(alpha_5) + 2*d_2*d_4*r_31*r_33*std::sin(alpha_5);
        R_l(5, 7) = -2*Px*d_2*r_12*std::sin(alpha_5) - 2*Py*d_2*r_22*std::sin(alpha_5) - 2*Pz*d_2*r_32*std::sin(alpha_5) + 2*std::pow(d_2, 2)*r_12*r_13*std::sin(alpha_5)*std::cos(alpha_5) + 2*std::pow(d_2, 2)*r_22*r_23*std::sin(alpha_5)*std::cos(alpha_5) + 2*std::pow(d_2, 2)*r_32*r_33*std::sin(alpha_5)*std::cos(alpha_5) + 2*d_2*d_4*r_12*r_13*std::sin(alpha_5) + 2*d_2*d_4*r_22*r_23*std::sin(alpha_5) + 2*d_2*d_4*r_32*r_33*std::sin(alpha_5);
        R_l(6, 0) = Px*r_31*std::sin(alpha_5) - Pz*r_11*std::sin(alpha_5) + d_4*r_11*r_33*std::sin(alpha_5) - d_4*r_13*r_31*std::sin(alpha_5);
        R_l(6, 1) = Px*r_32*std::sin(alpha_5) - Pz*r_12*std::sin(alpha_5) + d_4*r_12*r_33*std::sin(alpha_5) - d_4*r_13*r_32*std::sin(alpha_5);
        R_l(6, 2) = -Py*r_31*std::sin(alpha_5) + Pz*r_21*std::sin(alpha_5) - d_4*r_21*r_33*std::sin(alpha_5) + d_4*r_23*r_31*std::sin(alpha_5);
        R_l(6, 3) = -Py*r_32*std::sin(alpha_5) + Pz*r_22*std::sin(alpha_5) - d_4*r_22*r_33*std::sin(alpha_5) + d_4*r_23*r_32*std::sin(alpha_5);
        R_l(6, 4) = Px*r_33*std::cos(alpha_5) - Pz*r_13*std::cos(alpha_5);
        R_l(6, 5) = -Py*r_33*std::cos(alpha_5) + Pz*r_23*std::cos(alpha_5);
        R_l(7, 0) = -Py*r_31*std::sin(alpha_5) + Pz*r_21*std::sin(alpha_5) - d_4*r_21*r_33*std::sin(alpha_5) + d_4*r_23*r_31*std::sin(alpha_5);
        R_l(7, 1) = -Py*r_32*std::sin(alpha_5) + Pz*r_22*std::sin(alpha_5) - d_4*r_22*r_33*std::sin(alpha_5) + d_4*r_23*r_32*std::sin(alpha_5);
        R_l(7, 2) = -Px*r_31*std::sin(alpha_5) + Pz*r_11*std::sin(alpha_5) - d_4*r_11*r_33*std::sin(alpha_5) + d_4*r_13*r_31*std::sin(alpha_5);
        R_l(7, 3) = -Px*r_32*std::sin(alpha_5) + Pz*r_12*std::sin(alpha_5) - d_4*r_12*r_33*std::sin(alpha_5) + d_4*r_13*r_32*std::sin(alpha_5);
        R_l(7, 4) = -Py*r_33*std::cos(alpha_5) + Pz*r_23*std::cos(alpha_5);
        R_l(7, 5) = -Px*r_33*std::cos(alpha_5) + Pz*r_13*std::cos(alpha_5);
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
        const double x0 = std::sin(alpha_5);
        const double x1 = r_31*x0;
        const double x2 = r_32*x0;
        const double x3 = R_l_inv_60*x1 + R_l_inv_70*x2;
        const double x4 = a_0*x3;
        const double x5 = std::sin(alpha_3);
        const double x6 = x5*(R_l_inv_64*x1 + R_l_inv_74*x2);
        const double x7 = std::cos(alpha_5);
        const double x8 = r_33*x7;
        const double x9 = -x8;
        const double x10 = R_l_inv_62*x1 + R_l_inv_72*x2;
        const double x11 = d_4*r_33;
        const double x12 = d_2*x8;
        const double x13 = -Pz + x11 + x12;
        const double x14 = -x10*x13;
        const double x15 = R_l_inv_65*x1 + R_l_inv_75*x2;
        const double x16 = std::pow(a_0, 2);
        const double x17 = std::pow(d_1, 2);
        const double x18 = std::pow(Px, 2);
        const double x19 = std::pow(Py, 2);
        const double x20 = std::pow(Pz, 2);
        const double x21 = 2*d_4;
        const double x22 = r_13*x21;
        const double x23 = Px*x22;
        const double x24 = r_23*x21;
        const double x25 = Py*x24;
        const double x26 = 2*Pz;
        const double x27 = x11*x26;
        const double x28 = Px*x7;
        const double x29 = r_13*x28;
        const double x30 = 2*d_2;
        const double x31 = Py*x7;
        const double x32 = r_23*x31;
        const double x33 = std::pow(d_4, 2);
        const double x34 = std::pow(r_13, 2);
        const double x35 = x33*x34;
        const double x36 = std::pow(r_23, 2);
        const double x37 = x33*x36;
        const double x38 = std::pow(r_33, 2);
        const double x39 = x33*x38;
        const double x40 = d_4*x7;
        const double x41 = x34*x40;
        const double x42 = x36*x40;
        const double x43 = x38*x40;
        const double x44 = std::pow(r_11, 2);
        const double x45 = std::pow(d_2, 2);
        const double x46 = std::pow(x0, 2);
        const double x47 = x45*x46;
        const double x48 = x44*x47;
        const double x49 = std::pow(x7, 2);
        const double x50 = x45*x49;
        const double x51 = x34*x50;
        const double x52 = std::pow(r_21, 2);
        const double x53 = x47*x52;
        const double x54 = x36*x50;
        const double x55 = std::pow(r_31, 2);
        const double x56 = x47*x55;
        const double x57 = x38*x50;
        const double x58 = x12*x26 + x16 + x17 - x18 - x19 - x20 + x23 + x25 + x27 + x29*x30 + x30*x32 - x30*x41 - x30*x42 - x30*x43 - x35 - x37 - x39 - x48 - x51 - x53 - x54 - x56 - x57;
        const double x59 = -x15*x58;
        const double x60 = x14 + x4 + x59 + x6 + x9;
        const double x61 = R_l_inv_67*x1 + R_l_inv_77*x2;
        const double x62 = std::cos(alpha_3);
        const double x63 = a_0*x62;
        const double x64 = x61*x63;
        const double x65 = -x64;
        const double x66 = R_l_inv_66*x1 + R_l_inv_76*x2;
        const double x67 = d_1*x5;
        const double x68 = x66*x67;
        const double x69 = -x68;
        const double x70 = x65 + x69;
        const double x71 = d_1*x10;
        const double x72 = x62 - x71;
        const double x73 = 4*d_1;
        const double x74 = a_0*x73;
        const double x75 = x15*x74;
        const double x76 = 2*R_l_inv_63;
        const double x77 = 2*R_l_inv_73;
        const double x78 = x1*x76 + x2*x77;
        const double x79 = x62*x78;
        const double x80 = 2*d_1;
        const double x81 = x3*x80;
        const double x82 = -x79 - x81;
        const double x83 = x75 + x82;
        const double x84 = -x62 + x71;
        const double x85 = x64 + x68;
        const double x86 = x79 + x81;
        const double x87 = 2*a_0;
        const double x88 = x10*x87;
        const double x89 = a_0*x5;
        const double x90 = 2*x89;
        const double x91 = x66*x90;
        const double x92 = x88 + x91;
        const double x93 = 4*x5;
        const double x94 = d_1*x93;
        const double x95 = x66*x94;
        const double x96 = 4*x62;
        const double x97 = -4*x71 + x96;
        const double x98 = x14 + x59 + x84 + x9;
        const double x99 = -x4;
        const double x100 = x6 + x99;
        const double x101 = x65 + x68;
        const double x102 = x75 + x86;
        const double x103 = x64 + x69;
        const double x104 = x14 + x59 + x72 + x9;
        const double x105 = d_1*x62;
        const double x106 = Px*x0;
        const double x107 = r_11*x106;
        const double x108 = Py*x0;
        const double x109 = r_21*x108;
        const double x110 = Pz*x1;
        const double x111 = r_11*x0;
        const double x112 = d_4*r_13;
        const double x113 = x111*x112;
        const double x114 = r_21*x0;
        const double x115 = d_4*r_23;
        const double x116 = x114*x115;
        const double x117 = x1*x11;
        const double x118 = r_13*x7;
        const double x119 = x118*x30;
        const double x120 = r_23*x7;
        const double x121 = x120*x30;
        const double x122 = 2*x12;
        const double x123 = -x1*x122 + x107 + x109 + x110 - x111*x119 - x113 - x114*x121 - x116 - x117;
        const double x124 = r_12*x106;
        const double x125 = r_22*x108;
        const double x126 = Pz*x2;
        const double x127 = r_12*x0;
        const double x128 = x112*x127;
        const double x129 = r_22*x0;
        const double x130 = x115*x129;
        const double x131 = x11*x2;
        const double x132 = -x119*x127 - x121*x129 - x122*x2 + x124 + x125 + x126 - x128 - x130 - x131;
        const double x133 = R_l_inv_62*x123 + R_l_inv_72*x132;
        const double x134 = d_1*x133;
        const double x135 = -x13*x133;
        const double x136 = R_l_inv_65*x123 + R_l_inv_75*x132;
        const double x137 = -x136*x58;
        const double x138 = d_2*x46;
        const double x139 = x138*x44;
        const double x140 = d_2*x49;
        const double x141 = x140*x34;
        const double x142 = x138*x52;
        const double x143 = x140*x36;
        const double x144 = x138*x55;
        const double x145 = x140*x38;
        const double x146 = -x29;
        const double x147 = -x32;
        const double x148 = Pz*x8;
        const double x149 = -x148;
        const double x150 = x105 - x134 + x135 + x137 + x139 + x141 + x142 + x143 + x144 + x145 + x146 + x147 + x149 + x41 + x42 + x43;
        const double x151 = R_l_inv_60*x123 + R_l_inv_70*x132;
        const double x152 = a_0*x151;
        const double x153 = x5*(R_l_inv_64*x123 + R_l_inv_74*x132);
        const double x154 = x152 + x153;
        const double x155 = R_l_inv_67*x123 + R_l_inv_77*x132;
        const double x156 = x155*x63;
        const double x157 = -x156;
        const double x158 = R_l_inv_66*x123 + R_l_inv_76*x132;
        const double x159 = x158*x67;
        const double x160 = -x159;
        const double x161 = x157 + x160;
        const double x162 = x123*x76 + x132*x77;
        const double x163 = x162*x62;
        const double x164 = x151*x80;
        const double x165 = -x163 - x164;
        const double x166 = x136*x74 - 2*x63;
        const double x167 = x165 + x166;
        const double x168 = x156 + x159;
        const double x169 = x105 + x134 + x135 + x137 + x139 + x141 + x142 + x143 + x144 + x145 + x146 + x147 + x149 + x41 + x42 + x43;
        const double x170 = x133*x87;
        const double x171 = x158*x90;
        const double x172 = x170 + x171;
        const double x173 = x163 + x164;
        const double x174 = -4*x134;
        const double x175 = x158*x94;
        const double x176 = -x152;
        const double x177 = x153 + x176;
        const double x178 = x157 + x159;
        const double x179 = x166 + x173;
        const double x180 = x156 + x160;
        const double x181 = r_11*x108;
        const double x182 = r_21*x106;
        const double x183 = -x111*x115 + x112*x114 + x181 - x182;
        const double x184 = r_12*x108;
        const double x185 = r_22*x106;
        const double x186 = x112*x129 - x115*x127 + x184 - x185;
        const double x187 = x5*(R_l_inv_64*x183 + R_l_inv_74*x186);
        const double x188 = R_l_inv_62*x183 + R_l_inv_72*x186;
        const double x189 = -x13*x188;
        const double x190 = R_l_inv_65*x183 + R_l_inv_75*x186;
        const double x191 = -x190*x58;
        const double x192 = r_23*x28;
        const double x193 = -r_13*x31;
        const double x194 = R_l_inv_67*x183 + R_l_inv_77*x186;
        const double x195 = x194*x63;
        const double x196 = -x195;
        const double x197 = x187 + x189 + x191 + x192 + x193 + x196;
        const double x198 = R_l_inv_60*x183 + R_l_inv_70*x186;
        const double x199 = a_0*x198;
        const double x200 = R_l_inv_66*x183 + R_l_inv_76*x186;
        const double x201 = x200*x67;
        const double x202 = -x201;
        const double x203 = x199 + x202;
        const double x204 = d_1*x188;
        const double x205 = -x204;
        const double x206 = x205 + x89;
        const double x207 = 2*x67;
        const double x208 = -x207;
        const double x209 = x190*x74;
        const double x210 = x208 + x209;
        const double x211 = x183*x76 + x186*x77;
        const double x212 = x211*x62;
        const double x213 = x198*x80;
        const double x214 = -x212 - x213;
        const double x215 = x201 + x204;
        const double x216 = x187 + x189 + x191 + x192 + x193 + x195;
        const double x217 = x212 + x213;
        const double x218 = x188*x87;
        const double x219 = x200*x90;
        const double x220 = x218 + x219;
        const double x221 = -4*x204;
        const double x222 = x200*x94;
        const double x223 = -x199;
        const double x224 = -x89;
        const double x225 = x223 + x224;
        const double x226 = x207 + x209;
        const double x227 = x114*x18;
        const double x228 = x114*x20;
        const double x229 = 2*Py;
        const double x230 = x107*x229;
        const double x231 = x110*x229;
        const double x232 = x114*x19;
        const double x233 = x107*x24;
        const double x234 = x182*x22;
        const double x235 = x181*x22;
        const double x236 = x109*x24;
        const double x237 = x117*x229;
        const double x238 = x114*x27;
        const double x239 = x110*x24;
        const double x240 = x114*x35;
        const double x241 = x114*x39;
        const double x242 = 2*r_23;
        const double x243 = x242*x33;
        const double x244 = r_13*x111;
        const double x245 = x243*x244;
        const double x246 = r_33*x243;
        const double x247 = x1*x246;
        const double x248 = 4*d_2;
        const double x249 = x118*x248;
        const double x250 = x181*x249;
        const double x251 = x120*x248;
        const double x252 = x109*x251;
        const double x253 = 4*x12;
        const double x254 = Py*x253;
        const double x255 = x1*x254;
        const double x256 = x114*x37;
        const double x257 = r_23*x248*x40;
        const double x258 = x244*x257;
        const double x259 = x115*x253;
        const double x260 = x1*x259;
        const double x261 = std::pow(x0, 3)*x45;
        const double x262 = std::pow(r_21, 3)*x261;
        const double x263 = x248*x42;
        const double x264 = x114*x263;
        const double x265 = r_21*x261;
        const double x266 = x265*x44;
        const double x267 = x265*x55;
        const double x268 = x242*x50;
        const double x269 = x244*x268;
        const double x270 = r_33*x1;
        const double x271 = x268*x270;
        const double x272 = x114*x51;
        const double x273 = 3*x54;
        const double x274 = x114*x273;
        const double x275 = x114*x57;
        const double x276 = x227 + x228 - x230 - x231 - x232 + x233 - x234 + x235 + x236 + x237 - x238 + x239 + x240 + x241 - x245 - x247 + x250 + x252 + x255 - x256 - x258 - x260 - x262 - x264 - x266 - x267 - x269 - x271 - x272 - x274 - x275;
        const double x277 = 2*Px;
        const double x278 = r_13*x243;
        const double x279 = 2*x33;
        const double x280 = r_13*x279;
        const double x281 = Px*x253;
        const double x282 = r_13*x257;
        const double x283 = x112*x253;
        const double x284 = x248*x41;
        const double x285 = r_11*x261;
        const double x286 = r_13*x268;
        const double x287 = 2*x50;
        const double x288 = r_13*x287;
        const double x289 = 3*x51;
        const double x290 = -std::pow(r_11, 3)*x261 + x1*x281 - x1*x283 + x107*x22 + x107*x249 + x109*x22 + x110*x22 - x110*x277 - x111*x18 + x111*x19 + x111*x20 - x111*x27 - x111*x284 - x111*x289 - x111*x35 + x111*x37 + x111*x39 - x111*x54 - x111*x57 - x114*x278 - x114*x282 - x114*x286 + x117*x277 - x181*x24 - x182*x229 + x182*x24 + x182*x251 - x270*x280 - x270*x288 - x285*x52 - x285*x55;
        const double x291 = x129*x18;
        const double x292 = x129*x20;
        const double x293 = x124*x229;
        const double x294 = x126*x229;
        const double x295 = x129*x19;
        const double x296 = x124*x24;
        const double x297 = x185*x22;
        const double x298 = x184*x22;
        const double x299 = x125*x24;
        const double x300 = x131*x229;
        const double x301 = x129*x27;
        const double x302 = x126*x24;
        const double x303 = x129*x35;
        const double x304 = x129*x39;
        const double x305 = x127*x278;
        const double x306 = x2*x246;
        const double x307 = x184*x249;
        const double x308 = x125*x251;
        const double x309 = x2*x254;
        const double x310 = x129*x37;
        const double x311 = x127*x282;
        const double x312 = x2*x259;
        const double x313 = 2*x265;
        const double x314 = r_11*x313;
        const double x315 = r_12*x314;
        const double x316 = r_31*r_32;
        const double x317 = x313*x316;
        const double x318 = x129*x263;
        const double x319 = r_22*x261;
        const double x320 = x319*x44;
        const double x321 = 3*x319*x52;
        const double x322 = x319*x55;
        const double x323 = x127*x286;
        const double x324 = r_33*x2;
        const double x325 = x268*x324;
        const double x326 = x129*x51;
        const double x327 = x129*x273;
        const double x328 = x129*x57;
        const double x329 = x291 + x292 - x293 - x294 - x295 + x296 - x297 + x298 + x299 + x300 - x301 + x302 + x303 + x304 - x305 - x306 + x307 + x308 + x309 - x310 - x311 - x312 - x315 - x317 - x318 - x320 - x321 - x322 - x323 - x325 - x326 - x327 - x328;
        const double x330 = r_33*x280;
        const double x331 = 2*x285;
        const double x332 = r_12*x261;
        const double x333 = -r_22*x314 + x124*x22 + x124*x249 + x125*x22 + x126*x22 - x126*x277 - x127*x18 + x127*x19 + x127*x20 - x127*x27 - x127*x284 - x127*x289 - x127*x35 + x127*x37 + x127*x39 - x127*x54 - x127*x57 - x129*x278 - x129*x282 - x129*x286 + x131*x277 - x184*x24 - x185*x229 + x185*x24 + x185*x251 + x2*x281 - x2*x283 - x2*x330 - x288*x324 - x316*x331 - 3*x332*x44 - x332*x52 - x332*x55;
        const double x334 = x120*x18;
        const double x335 = x120*x20;
        const double x336 = x229*x29;
        const double x337 = x148*x229;
        const double x338 = x120*x19;
        const double x339 = x229*x41;
        const double x340 = x229*x42;
        const double x341 = x229*x43;
        const double x342 = std::pow(r_23, 3);
        const double x343 = x33*x7;
        const double x344 = x342*x343;
        const double x345 = x139*x229;
        const double x346 = x141*x229;
        const double x347 = x142*x229;
        const double x348 = x143*x229;
        const double x349 = x144*x229;
        const double x350 = x145*x229;
        const double x351 = x140*x21;
        const double x352 = x342*x351;
        const double x353 = x120*x35;
        const double x354 = x120*x39;
        const double x355 = x45*std::pow(x7, 3);
        const double x356 = x342*x355;
        const double x357 = x139*x24;
        const double x358 = x141*x24;
        const double x359 = x142*x24;
        const double x360 = x144*x24;
        const double x361 = x145*x24;
        const double x362 = r_23*x355;
        const double x363 = x34*x362;
        const double x364 = x362*x38;
        const double x365 = 2*r_21*x47;
        const double x366 = r_11*x365;
        const double x367 = x118*x366;
        const double x368 = r_31*x8;
        const double x369 = x365*x368;
        const double x370 = x120*x48;
        const double x371 = 3*x120*x53;
        const double x372 = x120*x56;
        const double x373 = x334 + x335 - x336 - x337 - x338 + x339 + x340 + x341 - x344 + x345 + x346 + x347 + x348 + x349 + x350 - x352 - x353 - x354 - x356 - x357 - x358 - x359 - x360 - x361 - x363 - x364 - x367 - x369 - x370 - x371 - x372;
        const double x374 = std::pow(r_13, 3);
        const double x375 = r_13*x355;
        const double x376 = 2*r_11*x47;
        const double x377 = -x118*x18 + x118*x19 + x118*x20 - x118*x37 - x118*x39 - 3*x118*x48 - x118*x53 - x118*x56 - x120*x366 - x139*x22 + x139*x277 + x141*x277 - x142*x22 + x142*x277 - x143*x22 + x143*x277 - x144*x22 + x144*x277 - x145*x22 + x145*x277 - x148*x277 - x192*x229 + x277*x41 + x277*x42 + x277*x43 - x343*x374 - x351*x374 - x355*x374 - x36*x375 - x368*x376 - x375*x38;
        const double x378 = x5*(R_l_inv_04*x276 + R_l_inv_14*x329 + R_l_inv_24*x290 + R_l_inv_34*x333 + R_l_inv_44*x373 + R_l_inv_54*x377);
        const double x379 = R_l_inv_07*x276 + R_l_inv_17*x329 + R_l_inv_27*x290 + R_l_inv_37*x333 + R_l_inv_47*x373 + R_l_inv_57*x377;
        const double x380 = x379*x63;
        const double x381 = -x380;
        const double x382 = x378 + x381;
        const double x383 = R_l_inv_00*x276 + R_l_inv_10*x329 + R_l_inv_20*x290 + R_l_inv_30*x333 + R_l_inv_40*x373 + R_l_inv_50*x377;
        const double x384 = a_0*x383;
        const double x385 = x105*x87;
        const double x386 = x384 + x385;
        const double x387 = R_l_inv_06*x276 + R_l_inv_16*x329 + R_l_inv_26*x290 + R_l_inv_36*x333 + R_l_inv_46*x373 + R_l_inv_56*x377;
        const double x388 = x387*x67;
        const double x389 = -x388;
        const double x390 = R_l_inv_02*x276 + R_l_inv_12*x329 + R_l_inv_22*x290 + R_l_inv_32*x333 + R_l_inv_42*x373 + R_l_inv_52*x377;
        const double x391 = d_1*x390;
        const double x392 = -x13*x390;
        const double x393 = R_l_inv_05*x276 + R_l_inv_15*x329 + R_l_inv_25*x290 + R_l_inv_35*x333 + R_l_inv_45*x373 + R_l_inv_55*x377;
        const double x394 = -x393*x58;
        const double x395 = -x391 + x392 + x394;
        const double x396 = x389 + x395;
        const double x397 = x16*x62;
        const double x398 = 2*x397;
        const double x399 = -x398;
        const double x400 = x393*x74;
        const double x401 = 2*R_l_inv_03;
        const double x402 = 2*R_l_inv_23;
        const double x403 = 2*R_l_inv_13;
        const double x404 = 2*R_l_inv_33;
        const double x405 = 2*R_l_inv_43;
        const double x406 = 2*R_l_inv_53;
        const double x407 = x276*x401 + x290*x402 + x329*x403 + x333*x404 + x373*x405 + x377*x406;
        const double x408 = x407*x62;
        const double x409 = x383*x80;
        const double x410 = x17*x62;
        const double x411 = 2*x410;
        const double x412 = -x408 - x409 - x411;
        const double x413 = x399 + x400 + x412;
        const double x414 = x378 + x380;
        const double x415 = x386 + x388;
        const double x416 = x391 + x392 + x394;
        const double x417 = x390*x87;
        const double x418 = x387*x90;
        const double x419 = x417 + x418;
        const double x420 = x408 + x409 + x411;
        const double x421 = x399 + x420;
        const double x422 = -4*x391;
        const double x423 = x387*x94;
        const double x424 = x398 + x412;
        const double x425 = -x384 - x385;
        const double x426 = x416 + x425;
        const double x427 = x398 + x400 + x420;
        const double x428 = -x227 - x228 + x230 + x231 + x232 - x233 + x234 - x235 - x236 - x237 + x238 - x239 - x240 - x241 + x245 + x247 - x250 - x252 - x255 + x256 + x258 + x260 + x262 + x264 + x266 + x267 + x269 + x271 + x272 + x274 + x275;
        const double x429 = -x334 - x335 + x336 + x337 + x338 - x339 - x340 - x341 + x344 - x345 - x346 - x347 - x348 - x349 - x350 + x352 + x353 + x354 + x356 + x357 + x358 + x359 + x360 + x361 + x363 + x364 + x367 + x369 + x370 + x371 + x372;
        const double x430 = -x291 - x292 + x293 + x294 + x295 - x296 + x297 - x298 - x299 - x300 + x301 - x302 - x303 - x304 + x305 + x306 - x307 - x308 - x309 + x310 + x311 + x312 + x315 + x317 + x318 + x320 + x321 + x322 + x323 + x325 + x326 + x327 + x328;
        const double x431 = R_l_inv_00*x290 + R_l_inv_10*x333 + R_l_inv_20*x428 + R_l_inv_30*x430 + R_l_inv_40*x377 + R_l_inv_50*x429;
        const double x432 = a_0*x431;
        const double x433 = R_l_inv_06*x290 + R_l_inv_16*x333 + R_l_inv_26*x428 + R_l_inv_36*x430 + R_l_inv_46*x377 + R_l_inv_56*x429;
        const double x434 = x433*x67;
        const double x435 = -x434;
        const double x436 = x432 + x435;
        const double x437 = R_l_inv_07*x290 + R_l_inv_17*x333 + R_l_inv_27*x428 + R_l_inv_37*x430 + R_l_inv_47*x377 + R_l_inv_57*x429;
        const double x438 = x437*x63;
        const double x439 = -x438;
        const double x440 = R_l_inv_02*x290 + R_l_inv_12*x333 + R_l_inv_22*x428 + R_l_inv_32*x430 + R_l_inv_42*x377 + R_l_inv_52*x429;
        const double x441 = d_1*x440;
        const double x442 = -x13*x440;
        const double x443 = R_l_inv_05*x290 + R_l_inv_15*x333 + R_l_inv_25*x428 + R_l_inv_35*x430 + R_l_inv_45*x377 + R_l_inv_55*x429;
        const double x444 = -x443*x58;
        const double x445 = -x441 + x442 + x444;
        const double x446 = x439 + x445;
        const double x447 = x5*(R_l_inv_04*x290 + R_l_inv_14*x333 + R_l_inv_24*x428 + R_l_inv_34*x430 + R_l_inv_44*x377 + R_l_inv_54*x429);
        const double x448 = x16*x5;
        const double x449 = x17*x5;
        const double x450 = x447 - x448 - x449;
        const double x451 = x290*x401 + x333*x403 + x377*x405 + x402*x428 + x404*x430 + x406*x429;
        const double x452 = x451*x62;
        const double x453 = x431*x80;
        const double x454 = -x452 - x453;
        const double x455 = x73*x89;
        const double x456 = x443*x74;
        const double x457 = x455 + x456;
        const double x458 = x438 + x450;
        const double x459 = x432 + x434;
        const double x460 = x441 + x442 + x444;
        const double x461 = x440*x87;
        const double x462 = x433*x90;
        const double x463 = x461 + x462;
        const double x464 = x452 + x453;
        const double x465 = -4*x441;
        const double x466 = x433*x94;
        const double x467 = -x432;
        const double x468 = x434 + x467;
        const double x469 = x439 + x460;
        const double x470 = x435 + x467;
        const double x471 = 2*x11;
        const double x472 = Pz*x22;
        const double x473 = Pz*x24;
        const double x474 = r_33*x244;
        const double x475 = Pz*x249;
        const double x476 = Pz*x251;
        const double x477 = x248*x43;
        const double x478 = r_31*x261;
        const double x479 = r_33*x268;
        const double x480 = 3*x57;
        const double x481 = -std::pow(r_31, 3)*x261 + x1*x18 + x1*x19 - x1*x20 - x1*x23 - x1*x25 + x1*x35 + x1*x37 - x1*x39 - x1*x477 - x1*x480 - x1*x51 - x1*x54 - x107*x26 + x107*x471 - x109*x26 + x109*x471 + x110*x253 + x110*x471 + x111*x472 + x111*x475 - x113*x253 - x114*x246 + x114*x473 + x114*x476 - x114*x479 - x116*x253 - x279*x474 - x287*x474 - x44*x478 - x478*x52;
        const double x482 = r_32*x261;
        const double x483 = -r_12*r_31*x331 - r_22*r_31*x313 - r_33*x127*x288 - x124*x26 + x124*x471 - x125*x26 + x125*x471 + x126*x253 + x126*x471 - x127*x330 + x127*x472 + x127*x475 - x128*x253 - x129*x246 + x129*x473 + x129*x476 - x129*x479 - x130*x253 + x18*x2 + x19*x2 - x2*x20 - x2*x23 - x2*x25 + x2*x35 + x2*x37 - x2*x39 - x2*x477 - x2*x480 - x2*x51 - x2*x54 - x44*x482 - x482*x52 - 3*x482*x55;
        const double x484 = R_l_inv_60*x481 + R_l_inv_70*x483;
        const double x485 = a_0*x484;
        const double x486 = x5*(R_l_inv_64*x481 + R_l_inv_74*x483);
        const double x487 = R_l_inv_62*x481 + R_l_inv_72*x483;
        const double x488 = -x13*x487;
        const double x489 = R_l_inv_65*x481 + R_l_inv_75*x483;
        const double x490 = -x489*x58;
        const double x491 = x20*x8;
        const double x492 = std::pow(r_33, 3);
        const double x493 = x355*x492;
        const double x494 = x343*x492;
        const double x495 = -x18*x8;
        const double x496 = -x19*x8;
        const double x497 = r_33*x355;
        const double x498 = x34*x497;
        const double x499 = x36*x497;
        const double x500 = x35*x8;
        const double x501 = x37*x8;
        const double x502 = -x139*x26;
        const double x503 = -x141*x26;
        const double x504 = -x142*x26;
        const double x505 = -x143*x26;
        const double x506 = -x144*x26;
        const double x507 = -x145*x26;
        const double x508 = -x26*x41;
        const double x509 = -x26*x42;
        const double x510 = -x26*x43;
        const double x511 = x26*x29;
        const double x512 = x26*x32;
        const double x513 = x351*x492;
        const double x514 = x48*x8;
        const double x515 = x53*x8;
        const double x516 = x139*x471;
        const double x517 = x141*x471;
        const double x518 = x142*x471;
        const double x519 = x143*x471;
        const double x520 = x144*x471;
        const double x521 = 3*x56*x8;
        const double x522 = r_31*x118*x376;
        const double x523 = r_31*x120*x365;
        const double x524 = x485 + x486 + x488 + x490 + x491 + x493 + x494 + x495 + x496 + x498 + x499 + x500 + x501 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x517 + x518 + x519 + x520 + x521 + x522 + x523;
        const double x525 = R_l_inv_67*x481 + R_l_inv_77*x483;
        const double x526 = x525*x63;
        const double x527 = -x526;
        const double x528 = R_l_inv_66*x481 + R_l_inv_76*x483;
        const double x529 = x528*x67;
        const double x530 = -x529;
        const double x531 = x527 + x530;
        const double x532 = d_1*x487;
        const double x533 = x397 - x410 - x532;
        const double x534 = x489*x74;
        const double x535 = x481*x76 + x483*x77;
        const double x536 = x535*x62;
        const double x537 = x484*x80;
        const double x538 = -x536 - x537;
        const double x539 = x534 + x538;
        const double x540 = x526 + x529;
        const double x541 = -x397 + x410 + x532;
        const double x542 = x536 + x537;
        const double x543 = x528*x90;
        const double x544 = a_0*d_1*x96 + x487*x87;
        const double x545 = x543 + x544;
        const double x546 = x528*x94;
        const double x547 = -x16*x96 - x17*x96 - 4*x532;
        const double x548 = x488 + x490 + x491 + x493 + x494 + x495 + x496 + x498 + x499 + x500 + x501 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x517 + x518 + x519 + x520 + x521 + x522 + x523 + x541;
        const double x549 = -x485;
        const double x550 = x486 + x549;
        const double x551 = x527 + x529;
        const double x552 = x534 + x542;
        const double x553 = x526 + x530;
        const double x554 = x488 + x490 + x491 + x493 + x494 + x495 + x496 + x498 + x499 + x500 + x501 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512 + x513 + x514 + x515 + x516 + x517 + x518 + x519 + x520 + x521 + x522 + x523 + x533;
        const double x555 = -x207*x61;
        const double x556 = x5*x78;
        const double x557 = x555 - x556;
        const double x558 = -x93;
        const double x559 = 4*x89;
        const double x560 = x559*x61;
        const double x561 = x555 + x556;
        const double x562 = 8*R_l_inv_63;
        const double x563 = 8*R_l_inv_73;
        const double x564 = x162*x5;
        const double x565 = -x564;
        const double x566 = -x155*x207;
        const double x567 = x566 - x90;
        const double x568 = x155*x559;
        const double x569 = x566 + x90;
        const double x570 = -x194*x207;
        const double x571 = x211*x5;
        const double x572 = x570 - x571;
        const double x573 = x194*x559;
        const double x574 = x570 + x571;
        const double x575 = 2*x448;
        const double x576 = 2*x449;
        const double x577 = -x207*x379;
        const double x578 = x407*x5;
        const double x579 = -x575 + x576 + x577 - x578;
        const double x580 = x379*x559;
        const double x581 = x575 - x576 + x577 + x578;
        const double x582 = -8*d_1*x89;
        const double x583 = 8*R_l_inv_03;
        const double x584 = 8*R_l_inv_23;
        const double x585 = 8*R_l_inv_13;
        const double x586 = 8*R_l_inv_33;
        const double x587 = 8*R_l_inv_43;
        const double x588 = 8*R_l_inv_53;
        const double x589 = -x207*x437;
        const double x590 = x451*x5;
        const double x591 = x589 - x590;
        const double x592 = x437*x559;
        const double x593 = x589 + x590;
        const double x594 = x5*x535;
        const double x595 = -x594;
        const double x596 = -x207*x525;
        const double x597 = x455 + x596;
        const double x598 = 4*x448;
        const double x599 = -x598;
        const double x600 = 4*x449;
        const double x601 = -x600;
        const double x602 = x525*x559;
        const double x603 = -x455;
        const double x604 = x596 + x603;
        const double x605 = -x6;
        const double x606 = x4 + x605;
        const double x607 = x88 - x91;
        const double x608 = x605 + x99;
        const double x609 = -x153;
        const double x610 = x152 + x609;
        const double x611 = x170 - x171;
        const double x612 = x176 + x609;
        const double x613 = -x187;
        const double x614 = x189 + x191 + x192 + x193 + x224 + x613;
        const double x615 = x218 - x219;
        const double x616 = x189 + x191 + x192 + x193 + x223 + x613;
        const double x617 = -x378;
        const double x618 = x381 + x617;
        const double x619 = x380 + x617;
        const double x620 = x417 - x418;
        const double x621 = -x447 + x448 + x449;
        const double x622 = x456 + x603;
        const double x623 = x438 + x621;
        const double x624 = x461 - x462;
        const double x625 = -x486;
        const double x626 = x485 + x625;
        const double x627 = -x543 + x544;
        const double x628 = x549 + x625;
        
        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = x60 + x70 + x72;
        A(0, 1) = x83;
        A(0, 2) = x60 + x84 + x85;
        A(0, 3) = x86 + x92;
        A(0, 4) = -x95 + x97;
        A(0, 5) = x82 + x92;
        A(0, 6) = x100 + x101 + x98;
        A(0, 7) = x102;
        A(0, 8) = x100 + x103 + x104;
        A(1, 0) = x150 + x154 + x161;
        A(1, 1) = x167;
        A(1, 2) = x154 + x168 + x169;
        A(1, 3) = x172 + x173;
        A(1, 4) = x174 - x175;
        A(1, 5) = x165 + x172;
        A(1, 6) = x169 + x177 + x178;
        A(1, 7) = x179;
        A(1, 8) = x150 + x177 + x180;
        A(2, 0) = x197 + x203 + x206;
        A(2, 1) = x210 + x214;
        A(2, 2) = x199 + x215 + x216 + x89;
        A(2, 3) = x207 + x217 + x220;
        A(2, 4) = x221 - x222;
        A(2, 5) = x208 + x214 + x220;
        A(2, 6) = x197 + x215 + x225;
        A(2, 7) = x217 + x226;
        A(2, 8) = x202 + x205 + x216 + x225;
        A(3, 0) = x382 + x386 + x396;
        A(3, 1) = x413;
        A(3, 2) = x414 + x415 + x416;
        A(3, 3) = x419 + x421;
        A(3, 4) = x422 - x423;
        A(3, 5) = x419 + x424;
        A(3, 6) = x382 + x388 + x426;
        A(3, 7) = x427;
        A(3, 8) = x396 + x414 + x425;
        A(4, 0) = x436 + x446 + x450;
        A(4, 1) = x454 + x457;
        A(4, 2) = x458 + x459 + x460;
        A(4, 3) = x463 + x464;
        A(4, 4) = x465 - x466;
        A(4, 5) = x454 + x463;
        A(4, 6) = x450 + x468 + x469;
        A(4, 7) = x457 + x464;
        A(4, 8) = x445 + x458 + x470;
        A(5, 0) = x524 + x531 + x533;
        A(5, 1) = x539;
        A(5, 2) = x524 + x540 + x541;
        A(5, 3) = x542 + x545;
        A(5, 4) = -x546 + x547;
        A(5, 5) = x538 + x545;
        A(5, 6) = x548 + x550 + x551;
        A(5, 7) = x552;
        A(5, 8) = x550 + x553 + x554;
        
        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x557;
        B(0, 1) = x558 + x560;
        B(0, 2) = x561;
        B(0, 3) = x93;
        B(0, 4) = x5*(-x1*x562 - x2*x563);
        B(0, 5) = x558;
        B(0, 6) = x561;
        B(0, 7) = x560 + x93;
        B(0, 8) = x557;
        B(1, 0) = x565 + x567;
        B(1, 1) = x568;
        B(1, 2) = x564 + x569;
        B(1, 4) = x5*(-x123*x562 - x132*x563);
        B(1, 6) = x564 + x567;
        B(1, 7) = x568;
        B(1, 8) = x565 + x569;
        B(2, 0) = x572;
        B(2, 1) = x573;
        B(2, 2) = x574;
        B(2, 4) = x5*(-x183*x562 - x186*x563);
        B(2, 6) = x574;
        B(2, 7) = x573;
        B(2, 8) = x572;
        B(3, 0) = x579;
        B(3, 1) = x580;
        B(3, 2) = x581;
        B(3, 3) = x582;
        B(3, 4) = 8*x448 + 8*x449 - x5*(x276*x583 + x290*x584 + x329*x585 + x333*x586 + x373*x587 + x377*x588);
        B(3, 5) = x582;
        B(3, 6) = x581;
        B(3, 7) = x580;
        B(3, 8) = x579;
        B(4, 0) = x591;
        B(4, 1) = x592;
        B(4, 2) = x593;
        B(4, 4) = x5*(-x290*x583 - x333*x585 - x377*x587 - x428*x584 - x429*x588 - x430*x586);
        B(4, 6) = x593;
        B(4, 7) = x592;
        B(4, 8) = x591;
        B(5, 0) = x595 + x597;
        B(5, 1) = x599 + x601 + x602;
        B(5, 2) = x594 + x597;
        B(5, 3) = x599 + x600;
        B(5, 4) = x5*(-x481*x562 - x483*x563);
        B(5, 5) = x598 + x601;
        B(5, 6) = x594 + x604;
        B(5, 7) = x598 + x600 + x602;
        B(5, 8) = x595 + x604;
        
        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = x101 + x104 + x606;
        C(0, 1) = x83;
        C(0, 2) = x103 + x606 + x98;
        C(0, 3) = x607 + x86;
        C(0, 4) = x95 + x97;
        C(0, 5) = x607 + x82;
        C(0, 6) = x608 + x70 + x98;
        C(0, 7) = x102;
        C(0, 8) = x104 + x608 + x85;
        C(1, 0) = x150 + x178 + x610;
        C(1, 1) = x167;
        C(1, 2) = x169 + x180 + x610;
        C(1, 3) = x173 + x611;
        C(1, 4) = x174 + x175;
        C(1, 5) = x165 + x611;
        C(1, 6) = x161 + x169 + x612;
        C(1, 7) = x179;
        C(1, 8) = x150 + x168 + x612;
        C(2, 0) = x196 + x199 + x201 + x205 + x614;
        C(2, 1) = x214 + x226;
        C(2, 2) = x195 + x203 + x204 + x614;
        C(2, 3) = x208 + x217 + x615;
        C(2, 4) = x221 + x222;
        C(2, 5) = x207 + x214 + x615;
        C(2, 6) = x196 + x202 + x204 + x616 + x89;
        C(2, 7) = x210 + x217;
        C(2, 8) = x195 + x201 + x206 + x616;
        C(3, 0) = x395 + x415 + x618;
        C(3, 1) = x413;
        C(3, 2) = x386 + x389 + x416 + x619;
        C(3, 3) = x421 + x620;
        C(3, 4) = x422 + x423;
        C(3, 5) = x424 + x620;
        C(3, 6) = x389 + x426 + x618;
        C(3, 7) = x427;
        C(3, 8) = x388 + x395 + x425 + x619;
        C(4, 0) = x446 + x459 + x621;
        C(4, 1) = x454 + x622;
        C(4, 2) = x436 + x460 + x623;
        C(4, 3) = x464 + x624;
        C(4, 4) = x465 + x466;
        C(4, 5) = x454 + x624;
        C(4, 6) = x469 + x470 + x621;
        C(4, 7) = x464 + x622;
        C(4, 8) = x445 + x468 + x623;
        C(5, 0) = x551 + x554 + x626;
        C(5, 1) = x539;
        C(5, 2) = x548 + x553 + x626;
        C(5, 3) = x542 + x627;
        C(5, 4) = x546 + x547;
        C(5, 5) = x538 + x627;
        C(5, 6) = x531 + x548 + x628;
        C(5, 7) = x552;
        C(5, 8) = x540 + x554 + x628;
        
        // Invoke the solver
        std::array<double, 16> solution_buffer;
        int n_solutions = yaik_cpp::general_6dof_internal::computeSolutionFromTanhalfLME(A, B, C, &solution_buffer);
        
        for(auto i = 0; i < n_solutions; i++)
        {
            auto solution_i = make_raw_solution();
            solution_i[4] = solution_buffer[i];
            int appended_idx = append_solution_to_queue(solution_i);
            add_input_index_to(2, appended_idx);
        };
    };
    // Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_3_processor();
    // Finish code for general_6dof solution node 0
    
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
            const double th_3 = this_solution[4];
            
            const bool condition_0 = 2*std::fabs(a_0*d_2*std::sin(alpha_3)*std::sin(th_3)) >= zero_tolerance || std::fabs(2*a_0*d_1 + 2*a_0*d_2*std::cos(alpha_3)) >= zero_tolerance || std::fabs(-std::pow(a_0, 2) - std::pow(d_1, 2) - 2*d_1*d_2*std::cos(alpha_3) - std::pow(d_2, 2) + std::pow(d_4, 2) + 2*d_4*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_0;
                const double x1 = d_2*std::cos(alpha_3);
                const double x2 = -d_1*x0 - x0*x1;
                const double x3 = std::sin(alpha_3);
                const double x4 = std::sin(th_3);
                const double x5 = std::atan2(x2, d_2*x0*x3*x4);
                const double x6 = std::pow(a_0, 2);
                const double x7 = std::pow(d_2, 2);
                const double x8 = -std::pow(d_1, 2) - 2*d_1*x1 + std::pow(d_4, 2) + 2*d_4*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2) - x6 - x7;
                const double x9 = safe_sqrt(std::pow(x2, 2) + 4*std::pow(x3, 2)*std::pow(x4, 2)*x6*x7 - std::pow(x8, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[3] = x5 + std::atan2(x9, x8);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = 2*std::fabs(a_0*d_2*std::sin(alpha_3)*std::sin(th_3)) >= zero_tolerance || std::fabs(2*a_0*d_1 + 2*a_0*d_2*std::cos(alpha_3)) >= zero_tolerance || std::fabs(-std::pow(a_0, 2) - std::pow(d_1, 2) - 2*d_1*d_2*std::cos(alpha_3) - std::pow(d_2, 2) + std::pow(d_4, 2) + 2*d_4*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_0;
                const double x1 = d_2*std::cos(alpha_3);
                const double x2 = -d_1*x0 - x0*x1;
                const double x3 = std::sin(alpha_3);
                const double x4 = std::sin(th_3);
                const double x5 = std::atan2(x2, d_2*x0*x3*x4);
                const double x6 = std::pow(a_0, 2);
                const double x7 = std::pow(d_2, 2);
                const double x8 = -std::pow(d_1, 2) - 2*d_1*x1 + std::pow(d_4, 2) + 2*d_4*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2) - x6 - x7;
                const double x9 = safe_sqrt(std::pow(x2, 2) + 4*std::pow(x3, 2)*std::pow(x4, 2)*x6*x7 - std::pow(x8, 2));
                // End of temp variables
                const double tmp_sol_value = x5 + std::atan2(-x9, x8);
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
            const double th_3 = this_solution[4];
            const bool checked_result = std::fabs(d_2*std::sin(alpha_3)*std::cos(th_3)) <= 9.9999999999999995e-7 && std::fabs(Px - d_4*r_13) <= 9.9999999999999995e-7 && std::fabs(Py - d_4*r_23) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor();
    // Finish code for equation all-zero dispatcher node 4
    
    // Code for explicit solution node 5, solved variable is th_0
    auto ExplicitSolutionNode_node_5_solve_th_0_processor = [&]() -> void
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
            const double th_3 = this_solution[4];
            
            const bool condition_0 = std::fabs(d_2*std::sin(alpha_3)*std::cos(th_3)) >= zero_tolerance || std::fabs(Px - d_4*r_13) >= zero_tolerance || std::fabs(Py - d_4*r_23) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Px - d_4*r_13;
                const double x1 = -Py + d_4*r_23;
                const double x2 = std::atan2(x0, x1);
                const double x3 = std::sin(alpha_3);
                const double x4 = std::cos(th_3);
                const double x5 = safe_sqrt(-std::pow(d_2, 2)*std::pow(x3, 2)*std::pow(x4, 2) + std::pow(x0, 2) + std::pow(x1, 2));
                const double x6 = d_2*x3*x4;
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x2 + std::atan2(x5, x6);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx);
            }
            
            const bool condition_1 = std::fabs(d_2*std::sin(alpha_3)*std::cos(th_3)) >= zero_tolerance || std::fabs(Px - d_4*r_13) >= zero_tolerance || std::fabs(Py - d_4*r_23) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = Px - d_4*r_13;
                const double x1 = -Py + d_4*r_23;
                const double x2 = std::atan2(x0, x1);
                const double x3 = std::sin(alpha_3);
                const double x4 = std::cos(th_3);
                const double x5 = safe_sqrt(-std::pow(d_2, 2)*std::pow(x3, 2)*std::pow(x4, 2) + std::pow(x0, 2) + std::pow(x1, 2));
                const double x6 = d_2*x3*x4;
                // End of temp variables
                const double tmp_sol_value = x2 + std::atan2(-x5, x6);
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
    ExplicitSolutionNode_node_5_solve_th_0_processor();
    // Finish code for explicit solution node 5
    
    // Code for equation all-zero dispatcher node 6
    auto EquationAllZeroDispatcherNode_node_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(6);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(6);
        if (!this_input_valid)
            return;
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[1];
            const bool checked_result = std::fabs(Pz - d_4*r_33) <= 9.9999999999999995e-7 && std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - d_4*r_13*std::cos(th_0) - d_4*r_23*std::sin(th_0)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(7, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_6_processor();
    // Finish code for equation all-zero dispatcher node 6
    
    // Code for explicit solution node 7, solved variable is th_1
    auto ExplicitSolutionNode_node_7_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(7);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(7);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 7
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[1];
            const double th_2 = this_solution[3];
            const double th_3 = this_solution[4];
            
            const bool condition_0 = std::fabs(Pz - d_4*r_33) >= 9.9999999999999995e-7 || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - d_4*r_13*std::cos(th_0) - d_4*r_23*std::sin(th_0)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Pz - d_4*r_33;
                const double x1 = std::sin(th_2);
                const double x2 = d_2*std::cos(alpha_3);
                const double x3 = std::cos(th_2);
                const double x4 = d_2*std::sin(alpha_3)*std::sin(th_3);
                const double x5 = -a_0 + d_1*x1 + x1*x2 - x3*x4;
                const double x6 = d_1*x3 + x1*x4 + x2*x3;
                const double x7 = std::cos(th_0);
                const double x8 = std::sin(th_0);
                const double x9 = -Px*x7 - Py*x8 + d_4*r_13*x7 + d_4*r_23*x8;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*x5 - x6*x9, x0*x6 + x5*x9);
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(8, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_1_processor();
    // Finish code for explicit solution node 7
    
    // Code for non-branch dispatcher node 8
    // Actually, there is no code
    
    // Code for explicit solution node 9, solved variable is negative_th_2_positive_th_1__soa
    auto ExplicitSolutionNode_node_9_solve_negative_th_2_positive_th_1__soa_processor = [&]() -> void
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
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_1 - th_2;
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
    ExplicitSolutionNode_node_9_solve_negative_th_2_positive_th_1__soa_processor();
    // Finish code for explicit solution node 8
    
    // Code for non-branch dispatcher node 10
    // Actually, there is no code
    
    // Code for explicit solution node 11, solved variable is th_4
    auto ExplicitSolutionNode_node_11_solve_th_4_processor = [&]() -> void
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
            const double negative_th_2_positive_th_1__soa = this_solution[0];
            const double th_0 = this_solution[1];
            const double th_3 = this_solution[4];
            
            const bool condition_0 = std::fabs(std::sin(alpha_3)*std::sin(alpha_5)) >= zero_tolerance || std::fabs(-r_13*(-std::sin(th_0)*std::sin(th_3) + std::cos(negative_th_2_positive_th_1__soa)*std::cos(th_0)*std::cos(th_3)) - r_23*(std::sin(th_0)*std::cos(negative_th_2_positive_th_1__soa)*std::cos(th_3) + std::sin(th_3)*std::cos(th_0)) + r_33*std::sin(negative_th_2_positive_th_1__soa)*std::cos(th_3)) >= zero_tolerance || std::fabs(r_13*std::sin(negative_th_2_positive_th_1__soa)*std::cos(th_0) + r_23*std::sin(negative_th_2_positive_th_1__soa)*std::sin(th_0) + r_33*std::cos(negative_th_2_positive_th_1__soa) - std::cos(alpha_3)*std::cos(alpha_5)) >= zero_tolerance || std::fabs(std::sin(alpha_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(alpha_5);
                const double x1 = std::sin(negative_th_2_positive_th_1__soa);
                const double x2 = std::cos(th_3);
                const double x3 = std::sin(th_3);
                const double x4 = std::cos(th_0);
                const double x5 = std::sin(th_0);
                const double x6 = std::cos(negative_th_2_positive_th_1__soa);
                const double x7 = x2*x6;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(r_13*(-x3*x5 + x4*x7) + r_23*(x3*x4 + x5*x7) - r_33*x1*x2), x0*(-r_13*x1*x4 - r_23*x1*x5 - r_33*x6 + std::cos(alpha_3)*std::cos(alpha_5))/std::sin(alpha_3));
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
    ExplicitSolutionNode_node_11_solve_th_4_processor();
    // Finish code for explicit solution node 10
    
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
            const double negative_th_2_positive_th_1__soa = this_solution[0];
            const double th_0 = this_solution[1];
            const double th_1 = this_solution[2];
            
            const bool condition_0 = std::fabs(d_2*std::sin(alpha_5)) >= zero_tolerance || std::fabs(-a_0*(-r_31*std::sin(th_1) + (r_11*std::cos(th_0) + r_21*std::sin(th_0))*std::cos(th_1)) - d_1*(r_11*std::sin(negative_th_2_positive_th_1__soa)*std::cos(th_0) + r_21*std::sin(negative_th_2_positive_th_1__soa)*std::sin(th_0) + r_31*std::cos(negative_th_2_positive_th_1__soa)) - inv_Px) >= zero_tolerance || std::fabs(-a_0*(-r_32*std::sin(th_1) + (r_12*std::cos(th_0) + r_22*std::sin(th_0))*std::cos(th_1)) - d_1*(r_12*std::sin(negative_th_2_positive_th_1__soa)*std::cos(th_0) + r_22*std::sin(negative_th_2_positive_th_1__soa)*std::sin(th_0) + r_32*std::cos(negative_th_2_positive_th_1__soa)) - inv_Py) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(negative_th_2_positive_th_1__soa);
                const double x1 = std::sin(negative_th_2_positive_th_1__soa);
                const double x2 = std::cos(th_0);
                const double x3 = r_11*x2;
                const double x4 = std::sin(th_0);
                const double x5 = r_21*x4;
                const double x6 = std::sin(th_1);
                const double x7 = std::cos(th_1);
                const double x8 = 1/(d_2*std::sin(alpha_5));
                const double x9 = r_12*x2;
                const double x10 = r_22*x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x8*(-a_0*(-r_31*x6 + x7*(x3 + x5)) - d_1*(r_31*x0 + x1*x3 + x1*x5) - inv_Px), x8*(-a_0*(-r_32*x6 + x7*(x10 + x9)) - d_1*(r_32*x0 + x1*x10 + x1*x9) - inv_Py));
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
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
        const double value_at_0 = raw_ik_out_i[1];  // th_0
        new_ik_i[0] = value_at_0;
        const double value_at_1 = raw_ik_out_i[2];  // th_1
        new_ik_i[1] = value_at_1;
        const double value_at_2 = raw_ik_out_i[3];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[4];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[5];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[6];  // th_5
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
    
    if (!ik_output.empty())
    {
        wrapAngleToPi(ik_output);
        removeDuplicate<robot_nq>(ik_output, zero_tolerance);
        return;
    }
    
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
    
    wrapAngleToPi(ik_output);
    removeDuplicate<robot_nq>(ik_output, zero_tolerance);
}

static std::vector<std::array<double, robot_nq>> computeIK(const Eigen::Matrix4d& T_ee)
{
    std::vector<std::array<double, robot_nq>> ik_output;
    RawIKWorksace raw_ik_workspace;
    computeIK(T_ee, raw_ik_workspace, ik_output);
    return ik_output;
}

}; // struct yaskawa_mpx3500_ik

// Code below for debug
void test_ik_solve_yaskawa_mpx3500()
{
    std::array<double, yaskawa_mpx3500_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = yaskawa_mpx3500_ik::computeFK(theta);
    auto ik_output = yaskawa_mpx3500_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = yaskawa_mpx3500_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_yaskawa_mpx3500();
}
