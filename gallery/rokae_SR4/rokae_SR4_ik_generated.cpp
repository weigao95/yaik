#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct rokae_SR4_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 128;
static constexpr int n_tree_nodes = 38;
static constexpr int intermediate_solution_size = 8;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_0 = 0.4031128874149275;
static constexpr double a_1 = 0.05;
static constexpr double d_2 = 0.4;
static constexpr double d_3 = -0.136;
static constexpr double d_4 = 0.1035;
static constexpr double pre_transform_special_symbol_23 = 0.355;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -1.4464413322481353;
static constexpr double th_2_offset_original2raw = -0.1243549945467616;
static constexpr double th_3_offset_original2raw = 3.141592653589793;
static constexpr double th_4_offset_original2raw = 3.141592653589793;
static constexpr double th_5_offset_original2raw = 0.0;

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
    ee_transformed(0, 1) = -1.0*r_12;
    ee_transformed(0, 2) = 1.0*r_11;
    ee_transformed(0, 3) = 1.0*Px;
    ee_transformed(1, 0) = 1.0*r_23;
    ee_transformed(1, 1) = -1.0*r_22;
    ee_transformed(1, 2) = 1.0*r_21;
    ee_transformed(1, 3) = 1.0*Py;
    ee_transformed(2, 0) = 2.7755575615628901e-17*r_13 + 1.0*r_33;
    ee_transformed(2, 1) = -2.7755575615628901e-17*r_12 - 1.0*r_32;
    ee_transformed(2, 2) = 2.7755575615628901e-17*r_11 + 1.0*r_31;
    ee_transformed(2, 3) = 2.7755575615628901e-17*Px + 1.0*Pz - 1.0*pre_transform_special_symbol_23;
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
    ee_transformed(0, 0) = 1.0*r_13 + 2.7755575615628901e-17*r_33;
    ee_transformed(0, 1) = -1.0*r_12 - 2.7755575615628901e-17*r_32;
    ee_transformed(0, 2) = 1.0*r_11 + 2.7755575615628901e-17*r_31;
    ee_transformed(0, 3) = 1.0*Px + 2.7755575615628901e-17*Pz;
    ee_transformed(1, 0) = 1.0*r_23;
    ee_transformed(1, 1) = -1.0*r_22;
    ee_transformed(1, 2) = 1.0*r_21;
    ee_transformed(1, 3) = 1.0*Py;
    ee_transformed(2, 0) = 1.0*r_33;
    ee_transformed(2, 1) = -1.0*r_32;
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
    const double x1 = std::sin(th_0);
    const double x2 = std::cos(th_3);
    const double x3 = std::sin(th_3);
    const double x4 = std::cos(th_0);
    const double x5 = std::sin(th_1);
    const double x6 = std::sin(th_2);
    const double x7 = x5*x6;
    const double x8 = std::cos(th_1);
    const double x9 = std::cos(th_2);
    const double x10 = -x4*x7 + x4*x8*x9;
    const double x11 = x1*x2 - x10*x3;
    const double x12 = std::cos(th_5);
    const double x13 = std::sin(th_4);
    const double x14 = x5*x9;
    const double x15 = x6*x8;
    const double x16 = -x14*x4 - x15*x4;
    const double x17 = std::cos(th_4);
    const double x18 = x1*x3 + x10*x2;
    const double x19 = -x13*x16 + x17*x18;
    const double x20 = -x13*x18 - x16*x17;
    const double x21 = a_0*x8;
    const double x22 = -x1*x7 + x1*x8*x9;
    const double x23 = -x2*x4 - x22*x3;
    const double x24 = -x1*x14 - x1*x15;
    const double x25 = x2*x22 - x3*x4;
    const double x26 = -x13*x24 + x17*x25;
    const double x27 = -x13*x25 - x17*x24;
    const double x28 = -x14 - x15;
    const double x29 = x28*x3;
    const double x30 = x7 - x8*x9;
    const double x31 = -x13*x30 + x17*x2*x28;
    const double x32 = -x13*x2*x28 - x17*x30;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x11 + x12*x19;
    ee_pose_raw(0, 1) = -x0*x19 - x11*x12;
    ee_pose_raw(0, 2) = x20;
    ee_pose_raw(0, 3) = a_1*x10 + d_2*x16 + d_3*x11 + d_4*x20 + x21*x4;
    ee_pose_raw(1, 0) = -x0*x23 + x12*x26;
    ee_pose_raw(1, 1) = -x0*x26 - x12*x23;
    ee_pose_raw(1, 2) = x27;
    ee_pose_raw(1, 3) = a_1*x22 + d_2*x24 + d_3*x23 + d_4*x27 + x1*x21;
    ee_pose_raw(2, 0) = x0*x29 + x12*x31;
    ee_pose_raw(2, 1) = -x0*x31 + x12*x28*x3;
    ee_pose_raw(2, 2) = x32;
    ee_pose_raw(2, 3) = -a_0*x5 + a_1*x28 + d_2*x30 - d_3*x29 + d_4*x32;
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
    const double x2 = -x1;
    const double x3 = std::cos(th_2);
    const double x4 = std::cos(th_1);
    const double x5 = std::sin(th_1);
    const double x6 = std::cos(th_0);
    const double x7 = 1.0*x6;
    const double x8 = -2.7755575615628901e-17*x4 - x5*x7;
    const double x9 = std::sin(th_2);
    const double x10 = 1.0*x4*x6 - 2.7755575615628901e-17*x5;
    const double x11 = -x10*x9 + x3*x8;
    const double x12 = std::cos(th_3);
    const double x13 = std::sin(th_3);
    const double x14 = x10*x3 + x8*x9;
    const double x15 = 1.0*x0*x12 - x13*x14;
    const double x16 = std::cos(th_4);
    const double x17 = std::sin(th_4);
    const double x18 = -x11*x16 - x17*(x1*x13 + x12*x14);
    const double x19 = x1*x5;
    const double x20 = x1*x4;
    const double x21 = -x19*x3 - x20*x9;
    const double x22 = 1.0*x0*x3*x4 - x19*x9;
    const double x23 = -x12*x7 - x13*x22;
    const double x24 = -x16*x21 - x17*(x12*x22 - x13*x7);
    const double x25 = 1.0*x9;
    const double x26 = 1.0*x3;
    const double x27 = x25*x5 - x26*x4;
    const double x28 = -x25*x4 - x26*x5;
    const double x29 = x13*x28;
    const double x30 = -x12*x17*x28 - x16*x27;
    const double x31 = 1.0*a_0;
    const double x32 = x31*x5;
    const double x33 = pre_transform_special_symbol_23 - x32;
    const double x34 = a_1*x28 + d_2*x27 + pre_transform_special_symbol_23 - x32;
    const double x35 = a_0*x20 + a_1*x22 + d_2*x21;
    const double x36 = -d_3*x29 + x34;
    const double x37 = d_3*x23 + x35;
    const double x38 = d_4*x30 + x36;
    const double x39 = d_4*x24 + x37;
    const double x40 = a_0*x10;
    const double x41 = a_1*x14 + d_2*x11 + x40;
    const double x42 = d_3*x15 + x41;
    const double x43 = d_4*x18 + x42;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = 2.7755575615628901e-17;
    jacobian(0, 1) = x2;
    jacobian(0, 2) = x2;
    jacobian(0, 3) = x11;
    jacobian(0, 4) = x15;
    jacobian(0, 5) = x18;
    jacobian(1, 1) = x7;
    jacobian(1, 2) = x7;
    jacobian(1, 3) = x21;
    jacobian(1, 4) = x23;
    jacobian(1, 5) = x24;
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x27;
    jacobian(2, 4) = -x29;
    jacobian(2, 5) = x30;
    jacobian(3, 1) = -pre_transform_special_symbol_23*x7;
    jacobian(3, 2) = -x33*x7;
    jacobian(3, 3) = -x21*x34 + x27*x35;
    jacobian(3, 4) = -x23*x36 - x29*x37;
    jacobian(3, 5) = -x24*x38 + x30*x39;
    jacobian(4, 0) = 2.7755575615628901e-17*pre_transform_special_symbol_23;
    jacobian(4, 1) = -pre_transform_special_symbol_23*x1;
    jacobian(4, 2) = -x1*x33;
    jacobian(4, 3) = x11*x34 - x27*x41;
    jacobian(4, 4) = x15*x36 + x29*x42;
    jacobian(4, 5) = x18*x38 - x30*x43;
    jacobian(5, 2) = std::pow(x0, 2)*x31*x4 + x40*x7;
    jacobian(5, 3) = -x11*x35 + x21*x41;
    jacobian(5, 4) = -x15*x37 + x23*x42;
    jacobian(5, 5) = -x18*x39 + x24*x43;
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
    const double x0 = std::sin(th_0);
    const double x1 = 1.0*x0;
    const double x2 = -x1;
    const double x3 = std::cos(th_2);
    const double x4 = std::cos(th_1);
    const double x5 = std::sin(th_1);
    const double x6 = std::cos(th_0);
    const double x7 = 1.0*x6;
    const double x8 = -2.7755575615628901e-17*x4 - x5*x7;
    const double x9 = std::sin(th_2);
    const double x10 = 1.0*x4*x6 - 2.7755575615628901e-17*x5;
    const double x11 = -x10*x9 + x3*x8;
    const double x12 = std::cos(th_3);
    const double x13 = std::sin(th_3);
    const double x14 = x10*x3 + x8*x9;
    const double x15 = std::cos(th_4);
    const double x16 = std::sin(th_4);
    const double x17 = x1*x5;
    const double x18 = -x1*x4*x9 - x17*x3;
    const double x19 = 1.0*x0*x3*x4 - x17*x9;
    const double x20 = 1.0*x9;
    const double x21 = 1.0*x3;
    const double x22 = x20*x5 - x21*x4;
    const double x23 = -x20*x4 - x21*x5;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = 2.7755575615628901e-17;
    jacobian(0, 1) = x2;
    jacobian(0, 2) = x2;
    jacobian(0, 3) = x11;
    jacobian(0, 4) = 1.0*x0*x12 - x13*x14;
    jacobian(0, 5) = -x11*x15 - x16*(x1*x13 + x12*x14);
    jacobian(1, 1) = x7;
    jacobian(1, 2) = x7;
    jacobian(1, 3) = x18;
    jacobian(1, 4) = -x12*x7 - x13*x19;
    jacobian(1, 5) = -x15*x18 - x16*(x12*x19 - x13*x7);
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x22;
    jacobian(2, 4) = -x13*x23;
    jacobian(2, 5) = -x12*x16*x23 - x15*x22;
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
    const double x6 = a_0*x5;
    const double x7 = pre_transform_special_symbol_23 - x6;
    const double x8 = std::sin(th_2);
    const double x9 = x5*x8;
    const double x10 = std::cos(th_2);
    const double x11 = std::cos(th_1);
    const double x12 = 1.0*x11;
    const double x13 = -x10*x12 + x9;
    const double x14 = std::sin(th_0);
    const double x15 = x10*x5;
    const double x16 = x12*x8;
    const double x17 = -x14*x15 - x14*x16;
    const double x18 = -x15 - x16;
    const double x19 = a_1*x18 + d_2*x13 + pre_transform_special_symbol_23 - x6;
    const double x20 = a_0*x12;
    const double x21 = 1.0*x10*x11*x14 - x14*x9;
    const double x22 = a_1*x21 + d_2*x17 + x14*x20;
    const double x23 = std::sin(th_3);
    const double x24 = x18*x23;
    const double x25 = std::cos(th_3);
    const double x26 = -x2*x25 - x21*x23;
    const double x27 = -d_3*x24 + x19;
    const double x28 = d_3*x26 + x22;
    const double x29 = std::cos(th_4);
    const double x30 = std::sin(th_4);
    const double x31 = -x13*x29 - x18*x25*x30;
    const double x32 = -x17*x29 - x30*(-x2*x23 + x21*x25);
    const double x33 = d_4*x31 + x27;
    const double x34 = d_4*x32 + x28;
    const double x35 = 1.0*p_on_ee_x;
    const double x36 = 1.0*x14;
    const double x37 = p_on_ee_z*x36;
    const double x38 = -2.7755575615628901e-17*x11 - x2*x4;
    const double x39 = 1.0*x1*x11 - 2.7755575615628901e-17*x4;
    const double x40 = x10*x38 - x39*x8;
    const double x41 = a_0*x39;
    const double x42 = x10*x39 + x38*x8;
    const double x43 = a_1*x42 + d_2*x40 + x41;
    const double x44 = 1.0*x14*x25 - x23*x42;
    const double x45 = d_3*x44 + x43;
    const double x46 = -x29*x40 - x30*(x23*x36 + x25*x42);
    const double x47 = d_4*x46 + x45;
    const double x48 = x1*x35;
    const double x49 = x0*x14;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = -pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = -x2*x7 + x3;
    jacobian(0, 3) = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x22 - x17*x19;
    jacobian(0, 4) = p_on_ee_y*x24 + p_on_ee_z*x26 - x24*x28 - x26*x27;
    jacobian(0, 5) = -p_on_ee_y*x31 + p_on_ee_z*x32 + x31*x34 - x32*x33;
    jacobian(1, 0) = -2.7755575615628901e-17*p_on_ee_z + 2.7755575615628901e-17*pre_transform_special_symbol_23 + x35;
    jacobian(1, 1) = -pre_transform_special_symbol_23*x36 + x37;
    jacobian(1, 2) = -x36*x7 + x37;
    jacobian(1, 3) = p_on_ee_x*x13 - p_on_ee_z*x40 - x13*x43 + x19*x40;
    jacobian(1, 4) = -p_on_ee_x*x24 - p_on_ee_z*x44 + x18*x23*x45 + x27*x44;
    jacobian(1, 5) = p_on_ee_x*x31 - p_on_ee_z*x46 - x31*x47 + x33*x46;
    jacobian(2, 0) = 2.7755575615628901e-17*p_on_ee_y;
    jacobian(2, 1) = -x48 - x49;
    jacobian(2, 2) = std::pow(x14, 2)*x20 + x2*x41 - x48 - x49;
    jacobian(2, 3) = -p_on_ee_x*x17 + p_on_ee_y*x40 + x17*x43 - x22*x40;
    jacobian(2, 4) = -p_on_ee_x*x26 + p_on_ee_y*x44 + x26*x45 - x28*x44;
    jacobian(2, 5) = -p_on_ee_x*x32 + p_on_ee_y*x46 + x32*x47 - x34*x46;
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
    
    // Code for general_6dof solution node 1, solved variable is th_0
    auto General6DoFNumericalReduceSolutionNode_node_1_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;
        
        // The general 6-dof solution of root node
        Eigen::Matrix<double, 8, 8> R_l;
        R_l.setZero();
        R_l(0, 3) = -a_0;
        R_l(0, 7) = -a_1;
        R_l(1, 2) = -a_0;
        R_l(1, 6) = -a_1;
        R_l(2, 4) = a_0;
        R_l(3, 6) = -1;
        R_l(4, 7) = 1;
        R_l(5, 5) = 2*a_0*a_1;
        R_l(6, 1) = -a_0;
        R_l(7, 0) = -a_0;
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
        const double x0 = -r_23;
        const double x1 = 2*r_21;
        const double x2 = -x1;
        const double x3 = 4*r_22;
        const double x4 = d_3*r_22;
        const double x5 = -x4;
        const double x6 = d_2*r_23;
        const double x7 = std::pow(r_21, 2);
        const double x8 = Py*x7;
        const double x9 = std::pow(r_22, 2);
        const double x10 = Py*x9;
        const double x11 = std::pow(r_23, 2);
        const double x12 = Py*x11;
        const double x13 = d_4*r_23;
        const double x14 = Px*r_11;
        const double x15 = r_21*x14;
        const double x16 = Px*r_12;
        const double x17 = r_22*x16;
        const double x18 = Px*r_13;
        const double x19 = r_23*x18;
        const double x20 = Pz*r_31;
        const double x21 = r_21*x20;
        const double x22 = Pz*r_32;
        const double x23 = r_22*x22;
        const double x24 = Pz*r_33;
        const double x25 = r_23*x24;
        const double x26 = x10 + x12 - x13 + x15 + x17 + x19 + x21 + x23 + x25 + x8;
        const double x27 = x26 - x6;
        const double x28 = x27 + x5;
        const double x29 = d_3*x1;
        const double x30 = x27 + x4;
        const double x31 = d_2*x1;
        const double x32 = -x31;
        const double x33 = d_2*x3;
        const double x34 = x26 + x6;
        const double x35 = x34 + x5;
        const double x36 = x34 + x4;
        const double x37 = Py*r_21;
        const double x38 = x14 + x20 + x37;
        const double x39 = R_l_inv_50*a_0;
        const double x40 = x38*x39;
        const double x41 = Py*r_23;
        const double x42 = -d_4 + x18 + x24 + x41;
        const double x43 = R_l_inv_52*a_0;
        const double x44 = x42*x43;
        const double x45 = d_4*r_21;
        const double x46 = r_23*x14;
        const double x47 = r_23*x20;
        const double x48 = r_21*x18;
        const double x49 = r_21*x24;
        const double x50 = x45 + x46 + x47 - x48 - x49;
        const double x51 = R_l_inv_57*a_0;
        const double x52 = x50*x51;
        const double x53 = a_0*r_22;
        const double x54 = R_l_inv_54*x53;
        const double x55 = -x54;
        const double x56 = R_l_inv_56*d_2*x53;
        const double x57 = R_l_inv_55*a_0;
        const double x58 = 2*d_2;
        const double x59 = x42*x58;
        const double x60 = x57*x59;
        const double x61 = x40 + x44 + x52 + x55 + x56 + x60;
        const double x62 = d_3*r_21;
        const double x63 = R_l_inv_53*a_0;
        const double x64 = r_21*x63;
        const double x65 = d_4*r_22;
        const double x66 = r_23*x16;
        const double x67 = r_23*x22;
        const double x68 = r_22*x18;
        const double x69 = r_22*x24;
        const double x70 = x65 + x66 + x67 - x68 - x69;
        const double x71 = R_l_inv_56*a_0;
        const double x72 = x70*x71;
        const double x73 = Py*r_22;
        const double x74 = x16 + x22 + x73;
        const double x75 = R_l_inv_51*a_0;
        const double x76 = x74*x75;
        const double x77 = d_2*r_21;
        const double x78 = x51*x77;
        const double x79 = 2*d_3;
        const double x80 = x57*x74;
        const double x81 = x79*x80;
        const double x82 = x62 + x64 + x72 - x76 + x78 + x81;
        const double x83 = d_3*r_23;
        const double x84 = x71*x83;
        const double x85 = r_22*x14;
        const double x86 = r_22*x20;
        const double x87 = r_21*x16;
        const double x88 = r_21*x22;
        const double x89 = x85 + x86 - x87 - x88;
        const double x90 = -x84 + x89;
        const double x91 = d_3*x75;
        const double x92 = std::pow(d_2, 2);
        const double x93 = std::pow(d_3, 2);
        const double x94 = std::pow(d_4, 2);
        const double x95 = std::pow(a_0, 2);
        const double x96 = std::pow(a_1, 2);
        const double x97 = 2*d_4;
        const double x98 = 2*x13;
        const double x99 = Py*x1;
        const double x100 = 2*x16;
        const double x101 = 2*x18;
        const double x102 = 2*x14;
        const double x103 = 2*x22;
        const double x104 = 2*x24;
        const double x105 = std::pow(Px, 2);
        const double x106 = std::pow(r_11, 2);
        const double x107 = x105*x106;
        const double x108 = std::pow(r_12, 2);
        const double x109 = x105*x108;
        const double x110 = std::pow(r_13, 2);
        const double x111 = x105*x110;
        const double x112 = std::pow(Py, 2);
        const double x113 = x112*x7;
        const double x114 = x112*x9;
        const double x115 = x11*x112;
        const double x116 = std::pow(Pz, 2);
        const double x117 = std::pow(r_31, 2);
        const double x118 = x116*x117;
        const double x119 = std::pow(r_32, 2);
        const double x120 = x116*x119;
        const double x121 = std::pow(r_33, 2)*x116;
        const double x122 = -Py*x98 + x100*x22 + x100*x73 + x101*x24 + x101*x41 + x102*x20 + x103*x73 + x104*x41 + x107 + x109 + x111 + x113 + x114 + x115 + x118 + x120 + x121 + x14*x99 - x18*x97 + x20*x99 - x24*x97 + x92 + x93 + x94 - x95 - x96;
        const double x123 = -R_l_inv_52*a_0*d_2 - R_l_inv_55*a_0*x122 - a_1 + x91;
        const double x124 = 2*x38;
        const double x125 = x124*x75;
        const double x126 = 2*x39;
        const double x127 = x126*x74;
        const double x128 = R_l_inv_54*a_0;
        const double x129 = x1*x128;
        const double x130 = 2*x51;
        const double x131 = x130*x70;
        const double x132 = x31*x71;
        const double x133 = -x132;
        const double x134 = x127 + x129 + x131 + x133;
        const double x135 = 2*x4;
        const double x136 = 2*x71;
        const double x137 = x136*x50;
        const double x138 = 2*R_l_inv_53*x53;
        const double x139 = R_l_inv_57*x53*x58;
        const double x140 = x135 - x137 + x138 + x139;
        const double x141 = d_2*x43;
        const double x142 = x122*x57;
        const double x143 = -x91;
        const double x144 = a_1 + x141 + x142 + x143 + x40;
        const double x145 = -x85;
        const double x146 = -x86;
        const double x147 = -x44;
        const double x148 = -x60;
        const double x149 = x145 + x146 + x147 + x148 + x84 + x87 + x88;
        const double x150 = x52 + x55 + x56;
        const double x151 = 2*x83;
        const double x152 = 2*r_23;
        const double x153 = x152*x63;
        const double x154 = x136*x89;
        const double x155 = x130*x6;
        const double x156 = x126*x42 + x151 + x153 + x154 + x155;
        const double x157 = x124*x43;
        const double x158 = 4*d_2;
        const double x159 = x38*x57;
        const double x160 = x158*x159;
        const double x161 = -x157 - x160;
        const double x162 = 2*x65;
        const double x163 = 2*x66;
        const double x164 = 2*x67;
        const double x165 = 2*x68;
        const double x166 = 2*x69;
        const double x167 = x29*x71;
        const double x168 = -x162 - x163 - x164 + x165 + x166 + x167;
        const double x169 = 4*x43*x74;
        const double x170 = 8*d_2;
        const double x171 = x170*x80;
        const double x172 = 4*x45;
        const double x173 = 4*x48;
        const double x174 = 4*x46;
        const double x175 = d_3*x3;
        const double x176 = x172 - x173 + x174 + x175*x71 + 4*x47 - 4*x49;
        const double x177 = x157 + x160;
        const double x178 = x162 + x163 + x164 - x165 - x166 - x167;
        const double x179 = -x52;
        const double x180 = -x56;
        const double x181 = x179 + x180 + x44 + x54 + x60;
        const double x182 = x76 - x81;
        const double x183 = x182 + x62 + x64 + x72 + x78 + x90;
        const double x184 = 4*d_3;
        const double x185 = -x125 + x159*x184;
        const double x186 = x140 + x185;
        const double x187 = x123 + x182 + x62 + x64 + x72 + x78;
        const double x188 = a_0*a_1;
        const double x189 = 2*x188;
        const double x190 = x95 + x96;
        const double x191 = R_l_inv_62*x190;
        const double x192 = R_l_inv_22*x189 + x191;
        const double x193 = d_2*x192;
        const double x194 = std::pow(r_21, 3)*x112;
        const double x195 = R_l_inv_25*x189 + R_l_inv_65*x190;
        const double x196 = x122*x195;
        const double x197 = R_l_inv_61*x190;
        const double x198 = R_l_inv_21*x189 + x197;
        const double x199 = d_3*x198;
        const double x200 = -x199;
        const double x201 = -r_21*x92;
        const double x202 = -r_21*x93;
        const double x203 = -r_21*x94;
        const double x204 = R_l_inv_23*x189 + R_l_inv_63*x190;
        const double x205 = -r_21*x204;
        const double x206 = R_l_inv_60*x190;
        const double x207 = x38*(R_l_inv_20*x189 + x206);
        const double x208 = -x207;
        const double x209 = R_l_inv_66*x190;
        const double x210 = R_l_inv_26*x189 + x209;
        const double x211 = -x210*x70;
        const double x212 = r_21*x107;
        const double x213 = r_21*x114;
        const double x214 = r_21*x115;
        const double x215 = r_21*x118;
        const double x216 = R_l_inv_67*x190;
        const double x217 = R_l_inv_27*x189 + x216;
        const double x218 = -x217*x77;
        const double x219 = -r_21*x109;
        const double x220 = -r_21*x111;
        const double x221 = -r_21*x120;
        const double x222 = -r_21*x121;
        const double x223 = -x14*x98;
        const double x224 = -x20*x98;
        const double x225 = x102*x8;
        const double x226 = x10*x102;
        const double x227 = x102*x12;
        const double x228 = d_4*x1;
        const double x229 = x18*x228;
        const double x230 = 2*x20;
        const double x231 = x230*x8;
        const double x232 = x10*x230;
        const double x233 = x12*x230;
        const double x234 = x228*x24;
        const double x235 = 2*r_11;
        const double x236 = r_12*x105;
        const double x237 = r_22*x236;
        const double x238 = x235*x237;
        const double x239 = 2*r_13;
        const double x240 = r_23*x105;
        const double x241 = r_11*x239*x240;
        const double x242 = r_31*x116;
        const double x243 = r_32*x242;
        const double x244 = 2*r_22;
        const double x245 = x243*x244;
        const double x246 = r_33*x242;
        const double x247 = x152*x246;
        const double x248 = x16*x22;
        const double x249 = -x1*x248;
        const double x250 = x18*x24;
        const double x251 = -x1*x250;
        const double x252 = x14*x20;
        const double x253 = x1*x252;
        const double x254 = x102*x23;
        const double x255 = x102*x25;
        const double x256 = x17*x230;
        const double x257 = x19*x230;
        const double x258 = x193 + x194 + x196 + x200 + x201 + x202 + x203 + x205 + x208 + x211 + x212 + x213 + x214 + x215 + x218 + x219 + x220 + x221 + x222 + x223 + x224 + x225 + x226 + x227 + x229 + x231 + x232 + x233 + x234 + x238 + x241 + x245 + x247 + x249 + x251 + x253 + x254 + x255 + x256 + x257;
        const double x259 = x198*x74;
        const double x260 = x192*x42;
        const double x261 = -x260;
        const double x262 = x195*x59;
        const double x263 = -x262;
        const double x264 = x195*x74;
        const double x265 = x264*x79;
        const double x266 = -x265;
        const double x267 = x259 + x261 + x263 + x266;
        const double x268 = x210*x83;
        const double x269 = x135*x14;
        const double x270 = x135*x20;
        const double x271 = x16*x29;
        const double x272 = x22*x29;
        const double x273 = x268 - x269 - x270 + x271 + x272;
        const double x274 = R_l_inv_24*x189 + R_l_inv_64*x190;
        const double x275 = r_22*x274;
        const double x276 = x217*x50;
        const double x277 = d_2*r_22;
        const double x278 = x210*x277;
        const double x279 = d_4*x31;
        const double x280 = 2*x6;
        const double x281 = x14*x280;
        const double x282 = x20*x280;
        const double x283 = x18*x31;
        const double x284 = x24*x31;
        const double x285 = x275 - x276 - x278 - x279 - x281 - x282 + x283 + x284;
        const double x286 = x273 + x285;
        const double x287 = 4*x188;
        const double x288 = R_l_inv_26*x287 + 2*x209;
        const double x289 = x288*x50;
        const double x290 = R_l_inv_20*x287 + 2*x206;
        const double x291 = x290*x74;
        const double x292 = r_22*x92;
        const double x293 = -2*x292;
        const double x294 = r_22*x93;
        const double x295 = -2*x294;
        const double x296 = r_22*x94;
        const double x297 = -2*x296;
        const double x298 = 2*x204;
        const double x299 = -r_22*x298;
        const double x300 = std::pow(r_22, 3)*x112;
        const double x301 = 2*x300;
        const double x302 = r_22*x58;
        const double x303 = -x217*x302;
        const double x304 = r_22*x107;
        const double x305 = -2*x304;
        const double x306 = r_22*x111;
        const double x307 = -2*x306;
        const double x308 = r_22*x118;
        const double x309 = -2*x308;
        const double x310 = r_22*x121;
        const double x311 = -2*x310;
        const double x312 = r_22*x109;
        const double x313 = 2*x312;
        const double x314 = r_22*x113;
        const double x315 = 2*x314;
        const double x316 = r_22*x115;
        const double x317 = 2*x316;
        const double x318 = r_22*x120;
        const double x319 = 2*x318;
        const double x320 = 4*x16;
        const double x321 = -x13*x320;
        const double x322 = 4*x22;
        const double x323 = -x13*x322;
        const double x324 = x320*x8;
        const double x325 = x10*x320;
        const double x326 = x12*x320;
        const double x327 = d_4*x3;
        const double x328 = x18*x327;
        const double x329 = x322*x8;
        const double x330 = x10*x322;
        const double x331 = x12*x322;
        const double x332 = x24*x327;
        const double x333 = 4*r_11;
        const double x334 = r_21*x333;
        const double x335 = x236*x334;
        const double x336 = 4*r_12;
        const double x337 = r_13*x240;
        const double x338 = x336*x337;
        const double x339 = 4*r_21;
        const double x340 = x243*x339;
        const double x341 = 4*r_23;
        const double x342 = r_32*r_33*x116;
        const double x343 = x341*x342;
        const double x344 = -x252*x3;
        const double x345 = -x250*x3;
        const double x346 = x15*x322;
        const double x347 = x21*x320;
        const double x348 = x248*x3;
        const double x349 = x25*x320;
        const double x350 = x19*x322;
        const double x351 = x289 - x291 + x293 + x295 + x297 + x299 + x301 + x303 + x305 + x307 + x309 + x311 + x313 + x315 + x317 + x319 + x321 + x323 + x324 + x325 + x326 + x328 + x329 + x330 + x331 + x332 + x335 + x338 + x340 + x343 + x344 + x345 + x346 + x347 + x348 + x349 + x350;
        const double x352 = x38*(R_l_inv_21*x287 + 2*x197);
        const double x353 = x195*x38;
        const double x354 = x184*x353 - x352;
        const double x355 = R_l_inv_27*x287 + 2*x216;
        const double x356 = x355*x70;
        const double x357 = x1*x274;
        const double x358 = d_4*x33;
        const double x359 = x210*x31;
        const double x360 = x320*x6;
        const double x361 = x322*x6;
        const double x362 = x18*x33;
        const double x363 = x24*x33;
        const double x364 = -x356 - x357 - x358 + x359 - x360 - x361 + x362 + x363;
        const double x365 = -x268;
        const double x366 = -x271;
        const double x367 = -x272;
        const double x368 = x260 + x262 + x269 + x270 + x365 + x366 + x367;
        const double x369 = -d_2*x192;
        const double x370 = -x122*x195;
        const double x371 = x194 + x199 + x201 + x202 + x203 + x205 + x208 + x211 + x212 + x213 + x214 + x215 + x218 + x219 + x220 + x221 + x222 + x223 + x224 + x225 + x226 + x227 + x229 + x231 + x232 + x233 + x234 + x238 + x241 + x245 + x247 + x249 + x251 + x253 + x254 + x255 + x256 + x257 + x369 + x370;
        const double x372 = x259 + x266;
        const double x373 = x38*(R_l_inv_22*x287 + 2*x191);
        const double x374 = x158*x353;
        const double x375 = -x290*x42 + x373 + x374;
        const double x376 = x210*x29;
        const double x377 = d_4*x175;
        const double x378 = x175*x18;
        const double x379 = x175*x24;
        const double x380 = x184*x66;
        const double x381 = x184*x67;
        const double x382 = -x376 + x377 - x378 - x379 + x380 + x381;
        const double x383 = x288*x89;
        const double x384 = r_23*x92;
        const double x385 = 2*x384;
        const double x386 = r_23*x93;
        const double x387 = 2*x386;
        const double x388 = r_23*x298;
        const double x389 = r_23*x94;
        const double x390 = 2*x389;
        const double x391 = std::pow(r_23, 3)*x112;
        const double x392 = 2*x391;
        const double x393 = 4*d_4;
        const double x394 = x393*x8;
        const double x395 = x10*x393;
        const double x396 = x12*x393;
        const double x397 = x217*x280;
        const double x398 = r_23*x107;
        const double x399 = 2*x398;
        const double x400 = r_23*x109;
        const double x401 = 2*x400;
        const double x402 = r_23*x118;
        const double x403 = 2*x402;
        const double x404 = r_23*x120;
        const double x405 = 2*x404;
        const double x406 = r_23*x111;
        const double x407 = 2*x406;
        const double x408 = r_23*x113;
        const double x409 = 2*x408;
        const double x410 = r_23*x114;
        const double x411 = 2*x410;
        const double x412 = r_23*x121;
        const double x413 = 2*x412;
        const double x414 = x14*x172;
        const double x415 = x16*x327;
        const double x416 = 4*x18;
        const double x417 = x13*x416;
        const double x418 = x172*x20;
        const double x419 = x22*x327;
        const double x420 = 4*x24;
        const double x421 = x13*x420;
        const double x422 = x416*x8;
        const double x423 = x10*x416;
        const double x424 = x12*x416;
        const double x425 = x420*x8;
        const double x426 = x10*x420;
        const double x427 = x12*x420;
        const double x428 = r_13*x105;
        const double x429 = x334*x428;
        const double x430 = r_13*x3;
        const double x431 = x236*x430;
        const double x432 = x246*x339;
        const double x433 = x3*x342;
        const double x434 = x174*x20;
        const double x435 = x322*x66;
        const double x436 = x15*x420;
        const double x437 = x16*x3;
        const double x438 = x24*x437;
        const double x439 = x21*x416;
        const double x440 = x22*x3;
        const double x441 = x18*x440;
        const double x442 = x19*x420;
        const double x443 = -x383 - x385 - x387 - x388 + x390 + x392 - x394 - x395 - x396 - x397 - x399 - x401 - x403 - x405 + x407 + x409 + x411 + x413 - x414 - x415 - x417 - x418 - x419 - x421 + x422 + x423 + x424 + x425 + x426 + x427 + x429 + x431 + x432 + x433 - x434 - x435 + x436 + x438 + x439 + x441 + x442;
        const double x444 = 8*x188;
        const double x445 = x74*(R_l_inv_22*x444 + 4*x191);
        const double x446 = x170*x264;
        const double x447 = 8*d_4;
        const double x448 = 8*x62;
        const double x449 = 8*d_3;
        const double x450 = x175*x210 - x18*x448 - x24*x448 + x447*x62 + x449*x46 + x449*x47;
        const double x451 = x290*x42 + x373 + x374;
        const double x452 = x383 + x385 + x387 + x388 - x390 - x392 + x394 + x395 + x396 + x397 + x399 + x401 + x403 + x405 - x407 - x409 - x411 - x413 + x414 + x415 + x417 + x418 + x419 + x421 - x422 - x423 - x424 - x425 - x426 - x427 - x429 - x431 - x432 - x433 + x434 + x435 - x436 - x438 - x439 - x441 - x442;
        const double x453 = -x259 + x265;
        const double x454 = x261 + x263 + x453;
        const double x455 = -x275 + x276 + x278 + x279 + x281 + x282 - x283 - x284;
        const double x456 = x273 + x455;
        const double x457 = -4*d_3*x195*x38 + x352;
        const double x458 = x356 + x357 + x358 - x359 + x360 + x361 - x362 - x363;
        const double x459 = R_l_inv_77*x190;
        const double x460 = R_l_inv_37*x189 + x459;
        const double x461 = x460*x50;
        const double x462 = R_l_inv_34*x189 + R_l_inv_74*x190;
        const double x463 = r_22*x462;
        const double x464 = R_l_inv_76*x190;
        const double x465 = R_l_inv_36*x189 + x464;
        const double x466 = x277*x465;
        const double x467 = x100*x8;
        const double x468 = x10*x100;
        const double x469 = x100*x12;
        const double x470 = x162*x18;
        const double x471 = x103*x8;
        const double x472 = x10*x103;
        const double x473 = x103*x12;
        const double x474 = x162*x24;
        const double x475 = r_11*x236;
        const double x476 = x1*x475;
        const double x477 = r_23*x236*x239;
        const double x478 = x1*x243;
        const double x479 = x152*x342;
        const double x480 = x16*x98;
        const double x481 = x22*x98;
        const double x482 = x1*x14;
        const double x483 = x22*x482;
        const double x484 = x1*x20;
        const double x485 = x16*x484;
        const double x486 = x103*x17;
        const double x487 = x100*x25;
        const double x488 = x103*x19;
        const double x489 = x230*x85;
        const double x490 = x165*x24;
        const double x491 = R_l_inv_71*x190;
        const double x492 = R_l_inv_31*x189 + x491;
        const double x493 = d_3*x492;
        const double x494 = R_l_inv_70*x190;
        const double x495 = x38*(R_l_inv_30*x189 + x494);
        const double x496 = R_l_inv_72*x190;
        const double x497 = R_l_inv_32*x189 + x496;
        const double x498 = x42*x497;
        const double x499 = d_2*x497;
        const double x500 = -x499;
        const double x501 = x492*x74;
        const double x502 = -x501;
        const double x503 = R_l_inv_35*x189 + R_l_inv_75*x190;
        const double x504 = x122*x503;
        const double x505 = -x504;
        const double x506 = x503*x59;
        const double x507 = x503*x74;
        const double x508 = x507*x79;
        const double x509 = x493 + x495 + x498 + x500 + x502 + x505 + x506 + x508;
        const double x510 = R_l_inv_33*x189 + R_l_inv_73*x190;
        const double x511 = r_21*x510;
        const double x512 = x465*x70;
        const double x513 = x460*x77;
        const double x514 = x465*x83;
        const double x515 = -x514;
        const double x516 = x6*x79;
        const double x517 = -x516;
        const double x518 = x58*x65;
        const double x519 = x58*x68;
        const double x520 = -x519;
        const double x521 = x58*x69;
        const double x522 = -x521;
        const double x523 = x16*x280;
        const double x524 = x22*x280;
        const double x525 = x511 + x512 + x513 + x515 + x517 + x518 + x520 + x522 + x523 + x524;
        const double x526 = x13*x79;
        const double x527 = x79*x8;
        const double x528 = x10*x79;
        const double x529 = x12*x79;
        const double x530 = x14*x29;
        const double x531 = x135*x16;
        const double x532 = x19*x79;
        const double x533 = x20*x29;
        const double x534 = x135*x22;
        const double x535 = x25*x79;
        const double x536 = -x526 + x527 + x528 + x529 + x530 + x531 + x532 + x533 + x534 + x535;
        const double x537 = R_l_inv_30*x287 + 2*x494;
        const double x538 = x537*x74;
        const double x539 = R_l_inv_37*x287 + 2*x459;
        const double x540 = x539*x70;
        const double x541 = x1*x92;
        const double x542 = -x541;
        const double x543 = x1*x93;
        const double x544 = x1*x462;
        const double x545 = -x31*x465;
        const double x546 = 4*x14;
        const double x547 = 4*x20;
        const double x548 = x1*x107 - x1*x109 - x1*x111 + x1*x114 + x1*x115 + x1*x118 - x1*x120 - x1*x121 - x1*x94 + x10*x546 + x10*x547 + x12*x546 + x12*x547 - x13*x546 - x13*x547 + x14*x440 + x15*x547 + x172*x18 + x172*x24 - x173*x24 + x19*x547 + 2*x194 + x20*x437 + x243*x3 + x246*x341 + x25*x546 + x3*x475 - x322*x87 + x333*x337 + x546*x8 + x547*x8;
        const double x549 = x538 + x540 + x542 + x543 + x544 + x545 + x548;
        const double x550 = R_l_inv_31*x287 + 2*x491;
        const double x551 = x38*x503;
        const double x552 = x184*x551;
        const double x553 = x38*x550 - x552;
        const double x554 = R_l_inv_36*x287 + 2*x464;
        const double x555 = x50*x554;
        const double x556 = 2*x510;
        const double x557 = r_22*x556;
        const double x558 = d_2*x172;
        const double x559 = x302*x460;
        const double x560 = x546*x6;
        const double x561 = x547*x6;
        const double x562 = x416*x77;
        const double x563 = x420*x77;
        const double x564 = -x555 + x557 - x558 + x559 - x560 - x561 + x562 + x563;
        const double x565 = -x292;
        const double x566 = -x296;
        const double x567 = -x461;
        const double x568 = -x466;
        const double x569 = -x304;
        const double x570 = -x306;
        const double x571 = -x308;
        const double x572 = -x310;
        const double x573 = -x508;
        const double x574 = -x480;
        const double x575 = -x481;
        const double x576 = -x489;
        const double x577 = -x490;
        const double x578 = x294 + x300 + x312 + x314 + x316 + x318 + x463 + x467 + x468 + x469 + x470 + x471 + x472 + x473 + x474 + x476 + x477 + x478 + x479 + x483 + x485 + x486 + x487 + x488 + x498 + x501 + x506 + x565 + x566 + x567 + x568 + x569 + x570 + x571 + x572 + x573 + x574 + x575 + x576 + x577;
        const double x579 = -x495;
        const double x580 = x493 + x500 + x505 + x579;
        const double x581 = -x511 - x512 - x513 - x518 + x519 + x521 - x523 - x524;
        const double x582 = x515 + x517 + x536 + x581;
        const double x583 = x38*(R_l_inv_32*x287 + 2*x496);
        const double x584 = x158*x551;
        const double x585 = x42*x537;
        const double x586 = x29*x465;
        const double x587 = x158*x62;
        const double x588 = x585 + x586 + x587;
        const double x589 = x554*x89;
        const double x590 = r_23*x556;
        const double x591 = x280*x460;
        const double x592 = x320*x77;
        const double x593 = x322*x77;
        const double x594 = x14*x33;
        const double x595 = x20*x33;
        const double x596 = x589 + x590 + x591 - x592 - x593 + x594 + x595;
        const double x597 = R_l_inv_32*x444 + 4*x496;
        const double x598 = x170*x4 + x175*x465;
        const double x599 = x583 + x584;
        const double x600 = x596 + x599;
        const double x601 = -d_3*x492 + x499 + x504;
        const double x602 = x495 + x601;
        const double x603 = x526 - x527 - x528 - x529 - x530 - x531 - x532 - x533 - x534 - x535;
        const double x604 = x525 + x603;
        const double x605 = -x538 + x540 + x542 + x543 + x544 + x545 + x548;
        const double x606 = x555 - x557 + x558 - x559 + x560 + x561 - x562 - x563;
        const double x607 = -x498 - x506;
        const double x608 = x294 + x300 + x312 + x314 + x316 + x318 + x463 + x467 + x468 + x469 + x470 + x471 + x472 + x473 + x474 + x476 + x477 + x478 + x479 + x483 + x485 + x486 + x487 + x488 + x514 + x516 + x565 + x566 + x567 + x568 + x569 + x570 + x571 + x572 + x574 + x575 + x576 + x577;
        const double x609 = x511 + x512 + x513 + x518 + x520 + x522 + x523 + x524 + x536 + x608;
        const double x610 = x4*x58;
        const double x611 = -2*d_2*d_4*r_23;
        const double x612 = x58*x8;
        const double x613 = x10*x58;
        const double x614 = x12*x58;
        const double x615 = x14*x31;
        const double x616 = x17*x58;
        const double x617 = x18*x280;
        const double x618 = x20*x31;
        const double x619 = x23*x58;
        const double x620 = x24*x280;
        const double x621 = -x610 + x611 + x612 + x613 + x614 + x615 + x616 + x617 + x618 + x619 + x620;
        const double x622 = d_4*x135;
        const double x623 = x66*x79;
        const double x624 = x67*x79;
        const double x625 = x135*x18;
        const double x626 = x135*x24;
        const double x627 = -x622 - x623 - x624 + x625 + x626;
        const double x628 = x8*x97;
        const double x629 = x10*x97;
        const double x630 = x12*x97;
        const double x631 = x101*x8;
        const double x632 = x10*x101;
        const double x633 = x101*x12;
        const double x634 = x104*x8;
        const double x635 = x10*x104;
        const double x636 = x104*x12;
        const double x637 = r_11*x1*x428;
        const double x638 = x237*x239;
        const double x639 = x1*x246;
        const double x640 = x244*x342;
        const double x641 = x14*x228;
        const double x642 = x16*x162;
        const double x643 = x18*x98;
        const double x644 = x20*x228;
        const double x645 = x162*x22;
        const double x646 = x24*x98;
        const double x647 = x24*x482;
        const double x648 = x104*x17;
        const double x649 = x18*x484;
        const double x650 = x101*x23;
        const double x651 = x104*x19;
        const double x652 = x230*x46;
        const double x653 = x163*x22;
        const double x654 = -x384 + x386 - x389 - x391 + x398 + x400 + x402 + x404 - x406 - x408 - x410 - x412 + x628 + x629 + x630 - x631 - x632 - x633 - x634 - x635 - x636 - x637 - x638 - x639 - x640 + x641 + x642 + x643 + x644 + x645 + x646 - x647 - x648 - x649 - x650 - x651 + x652 + x653;
        const double x655 = x621 + x627 + x654;
        const double x656 = x50 + x77;
        const double x657 = x622 + x623 + x624 - x625 - x626;
        const double x658 = x610 + x611 + x612 + x613 + x614 + x615 + x616 + x617 + x618 + x619 + x620;
        const double x659 = x654 + x657 + x658;
        const double x660 = x14*x175;
        const double x661 = x175*x20;
        const double x662 = x320*x62;
        const double x663 = x322*x62;
        const double x664 = x541 - x543 + x548;
        const double x665 = -x660 - x661 + x662 + x663 + x664;
        const double x666 = 8*x13;
        const double x667 = 8*x65;
        const double x668 = 8*x22;
        const double x669 = 8*x20;
        const double x670 = 8*x16;
        const double x671 = 8*x24;
        const double x672 = 8*r_12;
        const double x673 = r_11*r_21;
        const double x674 = 8*x243;
        const double x675 = 8*r_23;
        const double x676 = r_21*x674 + x10*x668 + x10*x670 + x105*x672*x673 - x107*x3 + x109*x3 - x111*x3 + x113*x3 + x115*x3 - x118*x3 + x12*x668 + x12*x670 + x120*x3 - x121*x3 + x15*x668 - x16*x666 + x17*x668 + x18*x667 + x19*x668 + x21*x670 - x22*x666 + x24*x667 + x25*x670 + x3*x92 - x3*x93 - x3*x94 + 4*x300 + x337*x672 + x342*x675 + x668*x8 - x669*x85 + x670*x8 - x671*x68;
        const double x677 = x660 + x661 - x662 - x663 + x664;
        const double x678 = x384 - x386 + x389 + x391 - x398 - x400 - x402 - x404 + x406 + x408 + x410 + x412 - x628 - x629 - x630 + x631 + x632 + x633 + x634 + x635 + x636 + x637 + x638 + x639 + x640 - x641 - x642 - x643 - x644 - x645 - x646 + x647 + x648 + x649 + x650 + x651 - x652 - x653;
        const double x679 = x621 + x657 + x678;
        const double x680 = x50 - x77;
        const double x681 = x627 + x658 + x678;
        const double x682 = -x239;
        const double x683 = r_12*x79;
        const double x684 = -x683;
        const double x685 = d_2*x239;
        const double x686 = 2*Px;
        const double x687 = 2*r_12;
        const double x688 = -d_4*x239 + r_11*x99 + x106*x686 + x108*x686 + x110*x686 + x20*x235 + x22*x687 + x239*x24 + x239*x41 + x687*x73;
        const double x689 = -x685 + x688;
        const double x690 = d_3*x333;
        const double x691 = d_2*x333;
        const double x692 = d_2*x672;
        const double x693 = x685 + x688;
        const double x694 = x128*x687;
        const double x695 = d_4*r_11;
        const double x696 = r_13*x37;
        const double x697 = r_13*x20;
        const double x698 = r_11*x41;
        const double x699 = r_11*x24;
        const double x700 = x695 + x696 + x697 - x698 - x699;
        const double x701 = x130*x700;
        const double x702 = r_12*x58;
        const double x703 = x702*x71;
        const double x704 = d_4*r_12;
        const double x705 = r_13*x73;
        const double x706 = r_13*x22;
        const double x707 = r_12*x41;
        const double x708 = r_12*x24;
        const double x709 = x704 + x705 + x706 - x707 - x708;
        const double x710 = r_11*x58;
        const double x711 = r_11*x79 + x136*x709 + x235*x63 + x51*x710;
        const double x712 = -x694 + x701 + x703 + x711;
        const double x713 = r_11*x73;
        const double x714 = 2*x713;
        const double x715 = r_11*x22;
        const double x716 = 2*x715;
        const double x717 = r_12*x99;
        const double x718 = r_12*x20;
        const double x719 = 2*x718;
        const double x720 = d_3*x239;
        const double x721 = x71*x720;
        const double x722 = -x714 - x716 + x717 + x719 - x721;
        const double x723 = x128*x333;
        const double x724 = 4*x51*x709;
        const double x725 = r_12*x184;
        const double x726 = 4*x71;
        const double x727 = d_2*x336;
        const double x728 = x336*x63 + x51*x727 - x700*x726 + x725;
        const double x729 = x714 + x716 - x717 - x719 + x721;
        const double x730 = 4*x704;
        const double x731 = Py*x430;
        const double x732 = 4*x706;
        const double x733 = 4*x707;
        const double x734 = 4*x708;
        const double x735 = x690*x71;
        const double x736 = r_13*x184;
        const double x737 = r_12*x37;
        const double x738 = x713 + x715 - x718 - x737;
        const double x739 = 4*r_13;
        const double x740 = d_2*x739;
        const double x741 = x51*x740 + x63*x739 - x726*x738 + x736;
        const double x742 = 8*x695;
        const double x743 = 8*x698;
        const double x744 = 8*x696;
        const double x745 = 8*x699;
        const double x746 = 8*x697;
        const double x747 = d_3*x672;
        const double x748 = x694 - x701 - x703 + x711;
        const double x749 = x210*x720;
        const double x750 = Py*x336;
        const double x751 = x62*x750;
        const double x752 = x20*x725;
        const double x753 = Py*x175;
        const double x754 = r_11*x753;
        const double x755 = x22*x690;
        const double x756 = std::pow(r_11, 3);
        const double x757 = 2*x105;
        const double x758 = Px*x106;
        const double x759 = 4*x37;
        const double x760 = Px*x759;
        const double x761 = Px*x547;
        const double x762 = Py*x333;
        const double x763 = d_4*x24;
        const double x764 = r_21*x112;
        const double x765 = r_12*x3;
        const double x766 = Py*x3;
        const double x767 = x24*x41;
        const double x768 = x20*x37;
        const double x769 = 4*x41;
        const double x770 = -r_11*x298 + r_23*x739*x764 + x108*x760 + x108*x761 + x109*x235 + x110*x760 + x110*x761 + x111*x235 + x113*x235 - x114*x235 - x115*x235 + x118*x235 - x120*x235 - x121*x235 + x13*x762 - x217*x710 - x235*x92 - x235*x93 - x235*x94 + x243*x336 + x246*x739 - x288*x709 + x322*x737 + x333*x763 - x333*x767 + x333*x768 - x393*x696 - x393*x697 + x420*x696 + x547*x758 + x697*x769 - x715*x766 + x718*x766 + x756*x757 + x758*x759 + x764*x765;
        const double x771 = x749 - x751 - x752 + x754 + x755 + x770;
        const double x772 = x355*x700;
        const double x773 = x274*x687;
        const double x774 = d_4*x691;
        const double x775 = r_12*x210;
        const double x776 = x58*x775;
        const double x777 = x158*x696;
        const double x778 = x158*x697;
        const double x779 = x6*x762;
        const double x780 = x24*x691;
        const double x781 = -x772 + x773 - x774 - x776 - x777 - x778 + x779 + x780;
        const double x782 = x709*(R_l_inv_27*x444 + 4*x216);
        const double x783 = x274*x333;
        const double x784 = d_4*x692;
        const double x785 = x170*x705;
        const double x786 = x170*x706;
        const double x787 = Py*x672;
        const double x788 = R_l_inv_26*x444 + 4*x209;
        const double x789 = std::pow(r_12, 3);
        const double x790 = 4*x105;
        const double x791 = 8*x73;
        const double x792 = Px*x791;
        const double x793 = Px*x668;
        const double x794 = r_22*x112;
        const double x795 = x673*x794;
        const double x796 = r_13*x675;
        const double x797 = 8*r_13;
        const double x798 = 8*x37;
        const double x799 = x672*x73;
        const double x800 = 8*x41;
        const double x801 = -d_2*x217*x336 + r_11*x674 + x107*x336 + x108*x792 + x108*x793 + x110*x792 + x110*x793 + x111*x336 - x113*x336 + x114*x336 - x115*x336 - x118*x336 + x120*x336 - x121*x336 + x13*x787 - x204*x336 + x22*x799 - x336*x92 - x336*x93 - x336*x94 + x342*x797 - x447*x705 - x447*x706 + x668*x758 + x669*x713 + x671*x705 + x672*x763 - x672*x767 - x672*x768 + x700*x788 + x706*x800 + x715*x798 + x758*x791 + x789*x790 + x794*x796 + 8*x795;
        const double x802 = -x749 + x751 + x752 - x754 - x755 + x770;
        const double x803 = x210*x690;
        const double x804 = d_4*x747;
        const double x805 = x41*x747;
        const double x806 = x24*x747;
        const double x807 = Py*x4;
        const double x808 = x797*x807;
        const double x809 = x449*x706;
        const double x810 = std::pow(r_13, 3);
        const double x811 = Px*x800;
        const double x812 = Px*x671;
        const double x813 = x112*x673;
        const double x814 = 8*r_11;
        const double x815 = r_23*x794;
        const double x816 = x22*x672;
        const double x817 = -8*Px*d_4*x106 - 8*Px*d_4*x108 - 8*Px*d_4*x110 - 8*Py*Pz*r_13*r_21*r_31 - 8*Py*Pz*r_13*r_22*r_32 - 8*Py*d_4*r_11*r_21 - 8*Py*d_4*r_12*r_22 - 8*Py*d_4*r_13*r_23 - 8*Pz*d_4*r_11*r_31 - 8*Pz*d_4*r_12*r_32 - 8*Pz*d_4*r_13*r_33 - 4*d_2*r_13*x217 - 4*r_13*x112*x7 - 4*r_13*x112*x9 - 4*r_13*x116*x117 - 4*r_13*x116*x119 - 4*r_13*x204 - 4*r_13*x92 - 4*r_13*x93 + x107*x739 + x108*x811 + x108*x812 + x109*x739 + x110*x811 + x110*x812 + x115*x739 + x121*x739 + x20*x743 + x24*x799 + x246*x814 + x342*x672 + x37*x745 + x41*x816 + x671*x758 + x672*x815 + x675*x813 + x738*x788 + x739*x94 + x758*x800 + x767*x797 + x790*x810;
        const double x818 = x772 - x773 + x774 + x776 + x777 + x778 - x779 - x780;
        const double x819 = -x539*x700;
        const double x820 = -x687*x92;
        const double x821 = -x687*x94;
        const double x822 = x687*x93;
        const double x823 = x462*x687;
        const double x824 = x757*x789;
        const double x825 = -x465*x702;
        const double x826 = -x113*x687;
        const double x827 = -x115*x687;
        const double x828 = -x118*x687;
        const double x829 = -x121*x687;
        const double x830 = x465*x720;
        const double x831 = x107*x687;
        const double x832 = x111*x687;
        const double x833 = x114*x687;
        const double x834 = x120*x687;
        const double x835 = d_2*x736;
        const double x836 = -d_4*x731;
        const double x837 = -x393*x706;
        const double x838 = x758*x766;
        const double x839 = Px*x766;
        const double x840 = x108*x839;
        const double x841 = x110*x839;
        const double x842 = x322*x758;
        const double x843 = Px*x322;
        const double x844 = x108*x843;
        const double x845 = x110*x843;
        const double x846 = x13*x750;
        const double x847 = x24*x730;
        const double x848 = x3*x813;
        const double x849 = x243*x333;
        const double x850 = r_23*x112;
        const double x851 = x430*x850;
        const double x852 = x342*x739;
        const double x853 = -x547*x737;
        const double x854 = -x24*x733;
        const double x855 = x333*x37;
        const double x856 = x22*x855;
        const double x857 = r_11*x20*x766;
        const double x858 = Py*x22*x765;
        const double x859 = x24*x731;
        const double x860 = x41*x732;
        const double x861 = x819 + x820 + x821 + x822 + x823 + x824 + x825 + x826 + x827 + x828 + x829 + x830 + x831 + x832 + x833 + x834 + x835 + x836 + x837 + x838 + x840 + x841 + x842 + x844 + x845 + x846 + x847 + x848 + x849 + x851 + x852 + x853 + x854 + x856 + x857 + x858 + x859 + x860;
        const double x862 = x554*x709;
        const double x863 = r_11*x556;
        const double x864 = d_2*x730;
        const double x865 = x460*x710;
        const double x866 = Py*r_13*x33;
        const double x867 = x158*x706;
        const double x868 = x6*x750;
        const double x869 = x158*x708;
        const double x870 = -x862 - x863 - x864 - x865 - x866 - x867 + x868 + x869;
        const double x871 = d_4*x736;
        const double x872 = -4*Px*d_3*x106 - 4*Px*d_3*x108 - 4*Px*d_3*x110 - 4*Py*d_3*r_11*r_21 - 4*Py*d_3*r_12*r_22 - 4*Py*d_3*r_13*r_23 - 4*Pz*d_3*r_11*r_31 - 4*Pz*d_3*r_12*r_32 - 4*Pz*d_3*r_13*r_33 + x871;
        const double x873 = R_l_inv_36*x444 + 4*x464;
        const double x874 = x700*x873;
        const double x875 = x336*x510;
        const double x876 = d_2*x742;
        const double x877 = d_2*x460;
        const double x878 = x336*x877;
        const double x879 = x170*x696;
        const double x880 = x170*x697;
        const double x881 = Py*x6;
        const double x882 = x814*x881;
        const double x883 = d_2*r_11;
        const double x884 = x671*x883;
        const double x885 = x333*x92;
        const double x886 = x333*x93;
        const double x887 = Px*x798;
        const double x888 = Px*x669;
        const double x889 = Py*r_11*x666 + r_22*x672*x764 + x108*x887 + x108*x888 + x109*x333 + x110*x887 + x110*x888 + x111*x333 + x113*x333 - x114*x333 - x115*x333 + x118*x333 - x120*x333 - x121*x333 + x20*x799 + x24*x742 - x24*x743 + x24*x744 + x243*x672 + x246*x797 - x333*x94 + x37*x816 + x41*x746 - x447*x696 - x447*x697 - x668*x713 + x669*x758 + x756*x790 + x758*x798 + x764*x796 + x768*x814;
        const double x890 = x333*x462 - x465*x691 + x709*(R_l_inv_37*x444 + 4*x459) - x885 + x886 + x889;
        const double x891 = Px*x184;
        const double x892 = r_12*x753 + x106*x891 + x108*x891 + x110*x891 + x20*x690 + x22*x725 + x24*x736 + x41*x736 + x62*x762 - x871;
        const double x893 = x819 + x820 + x821 + x822 + x823 + x824 + x825 + x826 + x827 + x828 + x829 - x830 + x831 + x832 + x833 + x834 - x835 + x836 + x837 + x838 + x840 + x841 + x842 + x844 + x845 + x846 + x847 + x848 + x849 + x851 + x852 + x853 + x854 + x856 + x857 + x858 + x859 + x860;
        const double x894 = x510*x739;
        const double x895 = x739*x877;
        const double x896 = x37*x692;
        const double x897 = x20*x692;
        const double x898 = x449*x883 + x465*x690;
        const double x899 = x862 + x863 + x864 + x865 + x866 + x867 - x868 - x869;
        const double x900 = Px*x158;
        const double x901 = x106*x900;
        const double x902 = x108*x900;
        const double x903 = x110*x900;
        const double x904 = x37*x691;
        const double x905 = Py*r_12;
        const double x906 = x33*x905;
        const double x907 = x739*x881;
        const double x908 = x20*x691;
        const double x909 = x22*x727;
        const double x910 = x24*x740;
        const double x911 = d_2*x725;
        const double x912 = x184*x704;
        const double x913 = x184*x707;
        const double x914 = -x913;
        const double x915 = x184*x708;
        const double x916 = -x915;
        const double x917 = r_13*x753;
        const double x918 = x184*x706;
        const double x919 = x911 + x912 + x914 + x916 + x917 + x918;
        const double x920 = x239*x93;
        const double x921 = x239*x92;
        const double x922 = x239*x94;
        const double x923 = x757*x810;
        const double x924 = Px*x393;
        const double x925 = x106*x924;
        const double x926 = x108*x924;
        const double x927 = x110*x924;
        const double x928 = x113*x239;
        const double x929 = x114*x239;
        const double x930 = x118*x239;
        const double x931 = x120*x239;
        const double x932 = x107*x239;
        const double x933 = x109*x239;
        const double x934 = x115*x239;
        const double x935 = x121*x239;
        const double x936 = d_4*x333;
        const double x937 = x37*x936;
        const double x938 = x704*x766;
        const double x939 = Py*x13*x739;
        const double x940 = x20*x936;
        const double x941 = x22*x730;
        const double x942 = x739*x763;
        const double x943 = x758*x769;
        const double x944 = Px*x769;
        const double x945 = x108*x944;
        const double x946 = x110*x944;
        const double x947 = x420*x758;
        const double x948 = Px*x420;
        const double x949 = x108*x948;
        const double x950 = x110*x948;
        const double x951 = x334*x850;
        const double x952 = x246*x333;
        const double x953 = x765*x850;
        const double x954 = x336*x342;
        const double x955 = x547*x696;
        const double x956 = x706*x766;
        const double x957 = x24*x855;
        const double x958 = x20*x333*x41;
        const double x959 = x708*x766;
        const double x960 = x22*x733;
        const double x961 = x739*x767;
        const double x962 = -x920 + x921 + x922 + x923 - x925 - x926 - x927 - x928 - x929 - x930 - x931 + x932 + x933 + x934 + x935 - x937 - x938 - x939 - x940 - x941 - x942 + x943 + x945 + x946 + x947 + x949 + x950 + x951 + x952 + x953 + x954 - x955 - x956 + x957 + x958 + x959 + x960 + x961;
        const double x963 = -4*d_2*d_4*r_13 + x901 + x902 + x903 + x904 + x906 + x907 + x908 + x909 + x910;
        const double x964 = x62*x787;
        const double x965 = x20*x747;
        const double x966 = x885 - x886 + x889;
        const double x967 = 16*d_4;
        const double x968 = 16*x24;
        const double x969 = 16*x20;
        const double x970 = 16*x73;
        const double x971 = Px*x970;
        const double x972 = 16*x22;
        const double x973 = Px*x972;
        const double x974 = 16*r_13;
        const double x975 = x962 + x963;
        const double x976 = -x29;
        const double x977 = -x40;
        const double x978 = x150 + x977;
        const double x979 = a_1 + x141 + x142 + x143;
        const double x980 = x145 + x146 + x84 + x87 + x88;
        const double x981 = -2*R_l_inv_50*a_0*x42 + x151 + x153 + x154 + x155;
        const double x982 = x979 + x980;
        const double x983 = x194 + x201 + x202 + x203 + x205 + x207 + x211 + x212 + x213 + x214 + x215 + x218 + x219 + x220 + x221 + x222 + x223 + x224 + x225 + x226 + x227 + x229 + x231 + x232 + x233 + x234 + x238 + x241 + x245 + x247 + x249 + x251 + x253 + x254 + x255 + x256 + x257;
        const double x984 = x260 + x262 + x983;
        const double x985 = x199 + x369 + x370;
        const double x986 = x289 + x291 + x293 + x295 + x297 + x299 + x301 + x303 + x305 + x307 + x309 + x311 + x313 + x315 + x317 + x319 + x321 + x323 + x324 + x325 + x326 + x328 + x329 + x330 + x331 + x332 + x335 + x338 + x340 + x343 + x344 + x345 + x346 + x347 + x348 + x349 + x350;
        const double x987 = x193 + x196 + x200;
        const double x988 = x269 + x270 + x365 + x366 + x367 + x983;
        const double x989 = x376 - x377 + x378 + x379 - x380 - x381;
        const double x990 = -x38*x550 + x552;
        const double x991 = x502 + x508;
        const double x992 = x294 + x300 + x312 + x314 + x316 + x318 + x463 + x467 + x468 + x469 + x470 + x471 + x472 + x473 + x474 + x476 + x477 + x478 + x479 + x483 + x485 + x486 + x487 + x488 + x565 + x566 + x567 + x568 + x569 + x570 + x571 + x572 + x574 + x575 + x576 + x577 + x607 + x991;
        
        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = x0;
        A(0, 2) = x0;
        A(0, 3) = x2;
        A(0, 4) = -x3;
        A(0, 5) = x1;
        A(0, 6) = r_23;
        A(0, 8) = r_23;
        A(1, 0) = x28;
        A(1, 1) = x29;
        A(1, 2) = x30;
        A(1, 3) = x32;
        A(1, 4) = -x33;
        A(1, 5) = x31;
        A(1, 6) = x35;
        A(1, 7) = x29;
        A(1, 8) = x36;
        A(2, 0) = -x123 - x61 - x82 - x90;
        A(2, 1) = 4*R_l_inv_55*a_0*d_3*x38 - x125 - x134 - x140;
        A(2, 2) = x144 + x149 + x150 + x82;
        A(2, 3) = x156 + x161 + x168;
        A(2, 4) = -x169 - x171 + x176;
        A(2, 5) = x156 + x177 + x178;
        A(2, 6) = x144 + x181 + x183;
        A(2, 7) = x127 - x129 - x131 + x132 + x186;
        A(2, 8) = -x149 - x179 - x180 - x187 - x40 - x54;
        A(3, 0) = x258 + x267 + x286;
        A(3, 1) = x351 + x354 + x364;
        A(3, 2) = -x285 - x368 - x371 - x372;
        A(3, 3) = -x375 - x382 - x443;
        A(3, 4) = -x445 - x446 + x450;
        A(3, 5) = x382 + x451 + x452;
        A(3, 6) = -x371 - x454 - x456;
        A(3, 7) = -x351 - x457 - x458;
        A(3, 8) = x258 + x368 + x453 + x455;
        A(4, 0) = x292 - x294 + x296 - x300 + x304 + x306 + x308 + x310 - x312 - x314 - x316 - x318 + x461 - x463 + x466 - x467 - x468 - x469 - x470 - x471 - x472 - x473 - x474 - x476 - x477 - x478 - x479 + x480 + x481 - x483 - x485 - x486 - x487 - x488 + x489 + x490 + x509 + x525 + x536;
        A(4, 1) = x549 + x553 + x564;
        A(4, 2) = x578 + x580 + x582;
        A(4, 3) = x583 + x584 - x588 - x596;
        A(4, 4) = 8*d_2*x503*x74 + x597*x74 - x598;
        A(4, 5) = -x585 + x586 + x587 - x600;
        A(4, 6) = -x578 - x602 - x604;
        A(4, 7) = x553 + x605 + x606;
        A(4, 8) = x493 + x495 + x500 + x501 + x505 + x573 + x607 + x609;
        A(5, 0) = -x655;
        A(5, 1) = -x184*x656;
        A(5, 2) = -x659;
        A(5, 3) = x665;
        A(5, 4) = x676;
        A(5, 5) = -x677;
        A(5, 6) = -x679;
        A(5, 7) = x184*x680;
        A(5, 8) = -x681;
        
        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x682;
        B(0, 2) = x682;
        B(0, 3) = -x333;
        B(0, 4) = -x672;
        B(0, 5) = x333;
        B(0, 6) = x239;
        B(0, 8) = x239;
        B(1, 0) = x684 + x689;
        B(1, 1) = x690;
        B(1, 2) = x683 + x689;
        B(1, 3) = -x691;
        B(1, 4) = -x692;
        B(1, 5) = x691;
        B(1, 6) = x684 + x693;
        B(1, 7) = x690;
        B(1, 8) = x683 + x693;
        B(2, 0) = -x712 - x722;
        B(2, 1) = 4*R_l_inv_56*a_0*d_2*r_11 - x723 - x724 - x728;
        B(2, 2) = x712 + x729;
        B(2, 3) = -x730 - x731 - x732 + x733 + x734 + x735 + x741;
        B(2, 4) = x71*x747 + x742 - x743 + x744 - x745 + x746;
        B(2, 5) = x730 + x731 + x732 - x733 - x734 - x735 + x741;
        B(2, 6) = x722 + x748;
        B(2, 7) = x691*x71 - x723 - x724 + x728;
        B(2, 8) = -x729 - x748;
        B(3, 0) = x771 + x781;
        B(3, 1) = x210*x691 + x24*x692 + x6*x787 - x782 - x783 - x784 - x785 - x786 + x801;
        B(3, 2) = -x781 - x802;
        B(3, 3) = x803 - x804 + x805 + x806 - x808 - x809 - x817;
        B(3, 4) = x449*(r_13*x99 + x20*x239 + 2*x695 - 2*x698 - 2*x699 + x775);
        B(3, 5) = -x803 + x804 - x805 - x806 + x808 + x809 - x817;
        B(3, 6) = -x771 - x818;
        B(3, 7) = 8*Py*d_2*r_12*r_23 + 8*Pz*d_2*r_12*r_33 + 4*d_2*r_11*x210 - x782 - x783 - x784 - x785 - x786 - x801;
        B(3, 8) = x802 + x818;
        B(4, 0) = -x861 - x870 - x872;
        B(4, 1) = -x874 + x875 - x876 + x878 - x879 - x880 + x882 + x884 + x890;
        B(4, 2) = x870 + x892 + x893;
        B(4, 3) = 8*Py*d_2*r_11*r_22 + 8*Pz*d_2*r_11*r_32 + x738*x873 - x894 - x895 - x896 - x897 - x898;
        B(4, 4) = x747*(-x465 - x58);
        B(4, 5) = x668*x883 + x738*x873 + x791*x883 - x894 - x895 - x896 - x897 + x898;
        B(4, 6) = -x872 - x893 - x899;
        B(4, 7) = x874 - x875 + x876 - x878 + x879 + x880 - x882 - x884 + x890;
        B(4, 8) = x861 + x892 + x899;
        B(5, 0) = d_4*x740 - x901 - x902 - x903 - x904 - x906 - x907 - x908 - x909 - x910 + x919 + x962;
        B(5, 1) = x449*(-x700 - x883);
        B(5, 2) = -x919 - x920 + x921 + x922 + x923 - x925 - x926 - x927 - x928 - x929 - x930 - x931 + x932 + x933 + x934 + x935 - x937 - x938 - x939 - x940 - x941 - x942 + x943 + x945 + x946 + x947 + x949 + x950 + x951 + x952 + x953 + x954 - x955 - x956 + x957 + x958 + x959 + x960 + x961 - x963;
        B(5, 3) = x449*x715 + x807*x814 - x964 - x965 + x966;
        B(5, 4) = 16*r_11*x243 + r_12*x22*x970 + 8*x105*x789 + x107*x672 + x108*x971 + x108*x973 + x110*x971 + x110*x973 + x111*x672 - x113*x672 + x114*x672 - x115*x672 - x118*x672 + x120*x672 - x121*x672 + 16*x13*x905 + x342*x974 + 16*x37*x715 + 16*x41*x706 + x672*x92 - x672*x93 - x672*x94 + x704*x968 - x705*x967 + x705*x968 - x706*x967 - x707*x968 + x713*x969 - x737*x969 + x758*x970 + x758*x972 + 16*x795 + x815*x974;
        B(5, 5) = 8*Py*d_3*r_11*r_22 + 8*Pz*d_3*r_11*r_32 - x964 - x965 - x966;
        B(5, 6) = x911 - x912 - x914 - x916 - x917 - x918 - x975;
        B(5, 7) = x449*(x695 + x696 + x697 - x698 - x699 - x883);
        B(5, 8) = -x911 + x912 - x913 - x915 + x917 + x918 - x975;
        
        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = r_23;
        C(0, 2) = r_23;
        C(0, 3) = x1;
        C(0, 4) = x3;
        C(0, 5) = x2;
        C(0, 6) = x0;
        C(0, 8) = x0;
        C(1, 0) = -x28;
        C(1, 1) = x976;
        C(1, 2) = -x30;
        C(1, 3) = x31;
        C(1, 4) = x33;
        C(1, 5) = x32;
        C(1, 6) = -x35;
        C(1, 7) = x976;
        C(1, 8) = -x36;
        C(2, 0) = x147 + x148 + x183 + x978 + x979;
        C(2, 1) = -x127 + x129 + x131 + x133 + x186;
        C(2, 2) = -x187 - x44 - x60 - x978 - x980;
        C(2, 3) = -x168 - x177 - x981;
        C(2, 4) = -x169 - x171 - x176;
        C(2, 5) = -x161 - x178 - x981;
        C(2, 6) = x182 + x61 - x62 - x64 - x72 - x78 + x982;
        C(2, 7) = x134 - x135 + x137 - x138 - x139 + x185;
        C(2, 8) = x181 + x82 + x977 + x982;
        C(3, 0) = -x286 - x453 - x984 - x985;
        C(3, 1) = -x364 - x457 - x986;
        C(3, 2) = x285 + x454 + x987 + x988;
        C(3, 3) = -x375 - x452 - x989;
        C(3, 4) = -x445 - x446 - x450;
        C(3, 5) = x443 + x451 + x989;
        C(3, 6) = x372 + x456 + x984 + x987;
        C(3, 7) = x354 + x458 + x986;
        C(3, 8) = -x267 - x455 - x985 - x988;
        C(4, 0) = x509 + x581 + x603 + x608;
        C(4, 1) = -x564 - x605 - x990;
        C(4, 2) = -x582 - x602 - x992;
        C(4, 3) = -x585 + x586 + x587 + x600;
        C(4, 4) = x170*x507 + x597*x74 + x598;
        C(4, 5) = -x588 + x589 + x590 + x591 - x592 - x593 + x594 + x595 - x599;
        C(4, 6) = x580 + x604 + x992;
        C(4, 7) = -x549 - x606 - x990;
        C(4, 8) = -x498 - x506 - x579 - x601 - x609 - x991;
        C(5, 0) = x655;
        C(5, 1) = x184*x656;
        C(5, 2) = x659;
        C(5, 3) = -x665;
        C(5, 4) = -x676;
        C(5, 5) = x677;
        C(5, 6) = x679;
        C(5, 7) = -x184*x680;
        C(5, 8) = x681;
        
        // Invoke the solver
        std::array<double, 16> solution_buffer;
        int n_solutions = yaik_cpp::general_6dof_internal::computeSolutionFromTanhalfLME(A, B, C, &solution_buffer);
        
        for(auto i = 0; i < n_solutions; i++)
        {
            auto solution_i = make_raw_solution();
            solution_i[0] = solution_buffer[i];
            int appended_idx = append_solution_to_queue(solution_i);
            add_input_index_to(2, appended_idx);
        };
    };
    // Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_0_processor();
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
            const double th_0 = this_solution[0];
            
            const bool condition_0 = std::fabs((-Px*std::sin(th_0) + Py*std::cos(th_0) + d_4*(r_13*std::sin(th_0) - r_23*std::cos(th_0)))/d_3) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = safe_acos((Px*x1 - Py*x0 - d_4*(r_13*x1 - r_23*x0))/d_3);
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[5] = x2;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = std::fabs((-Px*std::sin(th_0) + Py*std::cos(th_0) + d_4*(r_13*std::sin(th_0) - r_23*std::cos(th_0)))/d_3) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = safe_acos((Px*x1 - Py*x0 - d_4*(r_13*x1 - r_23*x0))/d_3);
                // End of temp variables
                const double tmp_sol_value = -x2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
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
    
    // Code for non-branch dispatcher node 4
    // Actually, there is no code
    
    // Code for explicit solution node 5, solved variable is th_2
    auto ExplicitSolutionNode_node_5_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(4);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(4);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 5
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_3 = this_solution[5];
            
            const bool condition_0 = 2*std::fabs(a_0*d_2) >= zero_tolerance || std::fabs(2*a_0*a_1 - 2*a_0*d_3*std::sin(th_3)) >= zero_tolerance || std::fabs(-std::pow(a_0, 2) - std::pow(a_1, 2) + 2*a_1*d_3*std::sin(th_3) - std::pow(d_2, 2) - std::pow(d_3, 2) + std::pow(d_4, 2) + 2*d_4*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_0;
                const double x1 = d_3*std::sin(th_3);
                const double x2 = a_1*x0 - x0*x1;
                const double x3 = std::atan2(-d_2*x0, x2);
                const double x4 = std::pow(a_0, 2);
                const double x5 = std::pow(d_2, 2);
                const double x6 = -std::pow(a_1, 2) + 2*a_1*x1 - std::pow(d_3, 2) + std::pow(d_4, 2) + 2*d_4*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2) - x4 - x5;
                const double x7 = safe_sqrt(std::pow(x2, 2) + 4*x4*x5 - std::pow(x6, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x3 + std::atan2(x7, x6);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx);
            }
            
            const bool condition_1 = 2*std::fabs(a_0*d_2) >= zero_tolerance || std::fabs(2*a_0*a_1 - 2*a_0*d_3*std::sin(th_3)) >= zero_tolerance || std::fabs(-std::pow(a_0, 2) - std::pow(a_1, 2) + 2*a_1*d_3*std::sin(th_3) - std::pow(d_2, 2) - std::pow(d_3, 2) + std::pow(d_4, 2) + 2*d_4*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_0;
                const double x1 = d_3*std::sin(th_3);
                const double x2 = a_1*x0 - x0*x1;
                const double x3 = std::atan2(-d_2*x0, x2);
                const double x4 = std::pow(a_0, 2);
                const double x5 = std::pow(d_2, 2);
                const double x6 = -std::pow(a_1, 2) + 2*a_1*x1 - std::pow(d_3, 2) + std::pow(d_4, 2) + 2*d_4*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2) - x4 - x5;
                const double x7 = safe_sqrt(std::pow(x2, 2) + 4*x4*x5 - std::pow(x6, 2));
                // End of temp variables
                const double tmp_sol_value = x3 + std::atan2(-x7, x6);
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
    // Finish code for explicit solution node 4
    
    // Code for solved_variable dispatcher node 6
    auto SolvedVariableDispatcherNode_node_6_processor = [&]() -> void
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
            bool taken_by_degenerate = false;
            const double th_3 = this_solution[5];
            
            const bool degenerate_valid_0 = std::fabs(th_3) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(14, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_3 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(19, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(7, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_6_processor();
    // Finish code for solved_variable dispatcher node 6
    
    // Code for explicit solution node 19, solved variable is th_1th_2th_4_soa
    auto ExplicitSolutionNode_node_19_solve_th_1th_2th_4_soa_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_33) >= zero_tolerance || std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_13*std::cos(th_0) + r_23*std::sin(th_0), r_33);
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(20, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_1th_2th_4_soa_processor();
    // Finish code for explicit solution node 19
    
    // Code for non-branch dispatcher node 20
    // Actually, there is no code
    
    // Code for explicit solution node 21, solved variable is th_5
    auto ExplicitSolutionNode_node_21_solve_th_5_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
                add_input_index_to(22, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_5_processor();
    // Finish code for explicit solution node 20
    
    // Code for solved_variable dispatcher node 22
    auto SolvedVariableDispatcherNode_node_22_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(22);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(22);
        if (!this_input_valid)
            return;
        
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_2 = this_solution[4];
            
            const bool degenerate_valid_0 = std::fabs(th_2 - M_PI + 1.44644133224814) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(32, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_2 - 2*M_PI + 1.44644133224814) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(35, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(23, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_22_processor();
    // Finish code for solved_variable dispatcher node 22
    
    // Code for explicit solution node 35, solved variable is th_4
    auto ExplicitSolutionNode_node_35_solve_th_4_processor = [&]() -> void
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
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_0 + d_2) >= zero_tolerance || std::fabs(d_4 + inv_Pz) >= zero_tolerance || std::fabs(inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5);
                const double x1 = d_4 + inv_Pz;
                const double x2 = std::atan2(x0, x1);
                const double x3 = 0.99227787671366796*a_0 + d_2;
                const double x4 = safe_sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[6] = x2 + std::atan2(x4, x3);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(36, appended_idx);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_0 + d_2) >= zero_tolerance || std::fabs(d_4 + inv_Pz) >= zero_tolerance || std::fabs(inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5);
                const double x1 = d_4 + inv_Pz;
                const double x2 = std::atan2(x0, x1);
                const double x3 = 0.99227787671366796*a_0 + d_2;
                const double x4 = safe_sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                const double tmp_sol_value = x2 + std::atan2(-x4, x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(36, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_35_solve_th_4_processor();
    // Finish code for explicit solution node 35
    
    // Code for non-branch dispatcher node 36
    // Actually, there is no code
    
    // Code for explicit solution node 37, solved variable is th_1
    auto ExplicitSolutionNode_node_37_solve_th_1_processor = [&]() -> void
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
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_1 - 0.12403473458920899*d_2) >= zero_tolerance || std::fabs(a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2) >= zero_tolerance || std::fabs(Pz + d_3*r_31*std::sin(th_5) + d_3*r_32*std::cos(th_5) - d_4*r_33) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_1 + 0.12403473458920899*d_2);
                const double x2 = -Pz - d_3*r_31*std::sin(th_5) - d_3*r_32*std::cos(th_5) + d_4*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_1 + 0.125*d_2, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x1 + std::atan2(x3, x2);
                int appended_idx = append_solution_to_queue(solution_0);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_1 - 0.12403473458920899*d_2) >= zero_tolerance || std::fabs(a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2) >= zero_tolerance || std::fabs(Pz + d_3*r_31*std::sin(th_5) + d_3*r_32*std::cos(th_5) - d_4*r_33) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_1 + 0.12403473458920899*d_2);
                const double x2 = -Pz - d_3*r_31*std::sin(th_5) - d_3*r_32*std::cos(th_5) + d_4*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_1 + 0.125*d_2, 2));
                // End of temp variables
                const double tmp_sol_value = x1 + std::atan2(-x3, x2);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_37_solve_th_1_processor();
    // Finish code for explicit solution node 36
    
    // Code for explicit solution node 32, solved variable is th_1
    auto ExplicitSolutionNode_node_32_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(32);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(32);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 32
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_1 - 0.12403473458920899*d_2) >= zero_tolerance || std::fabs(-a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2) >= zero_tolerance || std::fabs(Pz + d_3*r_31*std::sin(th_5) + d_3*r_32*std::cos(th_5) - d_4*r_33) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = -a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_1 + 0.12403473458920899*d_2);
                const double x2 = Pz + d_3*r_31*std::sin(th_5) + d_3*r_32*std::cos(th_5) - d_4*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_1 + 0.125*d_2, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x1 + std::atan2(x3, x2);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(33, appended_idx);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_1 - 0.12403473458920899*d_2) >= zero_tolerance || std::fabs(-a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2) >= zero_tolerance || std::fabs(Pz + d_3*r_31*std::sin(th_5) + d_3*r_32*std::cos(th_5) - d_4*r_33) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = -a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_1 + 0.12403473458920899*d_2);
                const double x2 = Pz + d_3*r_31*std::sin(th_5) + d_3*r_32*std::cos(th_5) - d_4*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_1 + 0.125*d_2, 2));
                // End of temp variables
                const double tmp_sol_value = x1 + std::atan2(-x3, x2);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(33, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_32_solve_th_1_processor();
    // Finish code for explicit solution node 32
    
    // Code for non-branch dispatcher node 33
    // Actually, there is no code
    
    // Code for explicit solution node 34, solved variable is th_4
    auto ExplicitSolutionNode_node_34_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(33);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(33);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 34
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[5];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(-r_13*((-0.99227787671366796*std::sin(th_1) - 0.124034734589208*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) - r_23*((-0.99227787671366796*std::sin(th_1) - 0.124034734589208*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(0.124034734589208*std::sin(th_1) - 0.99227787671366796*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance || std::fabs(-r_13*(-0.12403473458920899*std::sin(th_1) + 0.99227787671366796*std::cos(th_1))*std::cos(th_0) - r_23*(-0.12403473458920899*std::sin(th_1) + 0.99227787671366796*std::cos(th_1))*std::sin(th_0) + r_33*(0.99227787671366796*std::sin(th_1) + 0.12403473458920899*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_1);
                const double x3 = -0.99227787671366796*x2;
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_3);
                const double x6 = std::cos(th_0);
                const double x7 = 0.99227787671366796*x1;
                const double x8 = x0*(-0.124034734589208*x2 - x7);
                const double x9 = -0.12403473458920899*x1 - x3;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*(0.124034734589208*x1 + x3), r_13*x6*x9 + r_23*x4*x9 - r_33*(0.12403473458920899*x2 + x7));
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_34_solve_th_4_processor();
    // Finish code for explicit solution node 33
    
    // Code for explicit solution node 23, solved variable is th_1
    auto ExplicitSolutionNode_node_23_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(23);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(23);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 23
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_1th_2th_4_soa = this_solution[3];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = std::fabs(a_1*std::sin(th_2) + d_2*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_0 + a_1*std::cos(th_2) - d_2*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Pz - d_4*r_33;
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = a_0 + a_1*x1 - d_2*x2;
                const double x4 = -a_1*x2 - d_2*x1;
                const double x5 = Px*std::cos(th_0) + Py*std::sin(th_0) - d_4*std::sin(th_1th_2th_4_soa);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x3 + x4*x5, x0*x4 + x3*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(24, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_1_processor();
    // Finish code for explicit solution node 23
    
    // Code for non-branch dispatcher node 24
    // Actually, there is no code
    
    // Code for explicit solution node 25, solved variable is th_4
    auto ExplicitSolutionNode_node_25_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(24);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(24);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 25
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[1];
            const double th_1th_2th_4_soa = this_solution[3];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_1 + th_1th_2th_4_soa - th_2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_4_processor();
    // Finish code for explicit solution node 24
    
    // Code for explicit solution node 14, solved variable is th_5
    auto ExplicitSolutionNode_node_14_solve_th_5_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
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
    // Finish code for explicit solution node 14
    
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
            const double th_2 = this_solution[4];
            
            const bool degenerate_valid_0 = std::fabs(th_2 - M_PI + 1.44644133224814) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(26, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_2 - 2*M_PI + 1.44644133224814) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(29, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(16, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_15_processor();
    // Finish code for solved_variable dispatcher node 15
    
    // Code for explicit solution node 29, solved variable is th_4
    auto ExplicitSolutionNode_node_29_solve_th_4_processor = [&]() -> void
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
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_0 + d_2) >= zero_tolerance || std::fabs(d_4 + inv_Pz) >= zero_tolerance || std::fabs(inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5);
                const double x1 = d_4 + inv_Pz;
                const double x2 = std::atan2(x0, x1);
                const double x3 = 0.99227787671366796*a_0 + d_2;
                const double x4 = safe_sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[6] = x2 + std::atan2(x4, x3);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(30, appended_idx);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_0 + d_2) >= zero_tolerance || std::fabs(d_4 + inv_Pz) >= zero_tolerance || std::fabs(inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5);
                const double x1 = d_4 + inv_Pz;
                const double x2 = std::atan2(x0, x1);
                const double x3 = 0.99227787671366796*a_0 + d_2;
                const double x4 = safe_sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                const double tmp_sol_value = x2 + std::atan2(-x4, x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(30, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_4_processor();
    // Finish code for explicit solution node 29
    
    // Code for non-branch dispatcher node 30
    // Actually, there is no code
    
    // Code for explicit solution node 31, solved variable is th_1
    auto ExplicitSolutionNode_node_31_solve_th_1_processor = [&]() -> void
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
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_1 - 0.12403473458920899*d_2) >= zero_tolerance || std::fabs(a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2) >= zero_tolerance || std::fabs(Pz + d_3*r_31*std::sin(th_5) + d_3*r_32*std::cos(th_5) - d_4*r_33) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_1 + 0.12403473458920899*d_2);
                const double x2 = -Pz - d_3*r_31*std::sin(th_5) - d_3*r_32*std::cos(th_5) + d_4*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_1 + 0.125*d_2, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x1 + std::atan2(x3, x2);
                int appended_idx = append_solution_to_queue(solution_0);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_1 - 0.12403473458920899*d_2) >= zero_tolerance || std::fabs(a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2) >= zero_tolerance || std::fabs(Pz + d_3*r_31*std::sin(th_5) + d_3*r_32*std::cos(th_5) - d_4*r_33) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_1 + 0.12403473458920899*d_2);
                const double x2 = -Pz - d_3*r_31*std::sin(th_5) - d_3*r_32*std::cos(th_5) + d_4*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_1 + 0.125*d_2, 2));
                // End of temp variables
                const double tmp_sol_value = x1 + std::atan2(-x3, x2);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_1_processor();
    // Finish code for explicit solution node 30
    
    // Code for explicit solution node 26, solved variable is th_1
    auto ExplicitSolutionNode_node_26_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(26);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(26);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 26
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_1 - 0.12403473458920899*d_2) >= zero_tolerance || std::fabs(-a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2) >= zero_tolerance || std::fabs(Pz + d_3*r_31*std::sin(th_5) + d_3*r_32*std::cos(th_5) - d_4*r_33) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = -a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_1 + 0.12403473458920899*d_2);
                const double x2 = Pz + d_3*r_31*std::sin(th_5) + d_3*r_32*std::cos(th_5) - d_4*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_1 + 0.125*d_2, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x1 + std::atan2(x3, x2);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(27, appended_idx);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_1 - 0.12403473458920899*d_2) >= zero_tolerance || std::fabs(-a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2) >= zero_tolerance || std::fabs(Pz + d_3*r_31*std::sin(th_5) + d_3*r_32*std::cos(th_5) - d_4*r_33) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = -a_0 + 0.12403473458920899*a_1 + 0.99227787671366796*d_2;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_1 + 0.12403473458920899*d_2);
                const double x2 = Pz + d_3*r_31*std::sin(th_5) + d_3*r_32*std::cos(th_5) - d_4*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_1 + 0.125*d_2, 2));
                // End of temp variables
                const double tmp_sol_value = x1 + std::atan2(-x3, x2);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(27, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_26_solve_th_1_processor();
    // Finish code for explicit solution node 26
    
    // Code for non-branch dispatcher node 27
    // Actually, there is no code
    
    // Code for explicit solution node 28, solved variable is th_4
    auto ExplicitSolutionNode_node_28_solve_th_4_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[5];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(-r_13*((-0.99227787671366796*std::sin(th_1) - 0.124034734589208*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) - r_23*((-0.99227787671366796*std::sin(th_1) - 0.124034734589208*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(0.124034734589208*std::sin(th_1) - 0.99227787671366796*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance || std::fabs(-r_13*(-0.12403473458920899*std::sin(th_1) + 0.99227787671366796*std::cos(th_1))*std::cos(th_0) - r_23*(-0.12403473458920899*std::sin(th_1) + 0.99227787671366796*std::cos(th_1))*std::sin(th_0) + r_33*(0.99227787671366796*std::sin(th_1) + 0.12403473458920899*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_1);
                const double x3 = -0.99227787671366796*x2;
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_3);
                const double x6 = std::cos(th_0);
                const double x7 = 0.99227787671366796*x1;
                const double x8 = x0*(-0.124034734589208*x2 - x7);
                const double x9 = -0.12403473458920899*x1 - x3;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*(0.124034734589208*x1 + x3), r_13*x6*x9 + r_23*x4*x9 - r_33*(0.12403473458920899*x2 + x7));
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_28_solve_th_4_processor();
    // Finish code for explicit solution node 27
    
    // Code for explicit solution node 16, solved variable is th_1
    auto ExplicitSolutionNode_node_16_solve_th_1_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = std::fabs(a_1*std::sin(th_2) + d_2*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_0 + a_1*std::cos(th_2) - d_2*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Pz - d_4*r_33;
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = a_0 + a_1*x1 - d_2*x2;
                const double x4 = -a_1*x2 - d_2*x1;
                const double x5 = std::cos(th_0);
                const double x6 = std::sin(th_0);
                const double x7 = Px*x5 + Py*x6 - d_4*r_13*x5 - d_4*r_23*x6;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x3 + x4*x7, x0*x4 + x3*x7);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(17, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_1_processor();
    // Finish code for explicit solution node 16
    
    // Code for non-branch dispatcher node 17
    // Actually, there is no code
    
    // Code for explicit solution node 18, solved variable is th_4
    auto ExplicitSolutionNode_node_18_solve_th_4_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(-r_13*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::cos(th_0) - r_23*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_0) + r_33*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))) >= zero_tolerance || std::fabs(r_13*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_23*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::sin(th_0) + r_33*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_1);
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = std::cos(th_1);
                const double x4 = x0*x1 + x2*x3;
                const double x5 = -x0*x2 + x1*x3;
                const double x6 = r_13*std::cos(th_0);
                const double x7 = r_23*std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_33*x4 - x5*x6 - x5*x7, r_33*x5 + x4*x6 + x4*x7);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_4_processor();
    // Finish code for explicit solution node 17
    
    // Code for explicit solution node 7, solved variable is th_4
    auto ExplicitSolutionNode_node_7_solve_th_4_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_3 = this_solution[5];
            
            const bool condition_0 = std::fabs((r_13*std::sin(th_0) - r_23*std::cos(th_0))/std::sin(th_3)) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = safe_asin((-r_13*std::sin(th_0) + r_23*std::cos(th_0))/std::sin(th_3));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[6] = x0;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(8, appended_idx);
            }
            
            const bool condition_1 = std::fabs((r_13*std::sin(th_0) - r_23*std::cos(th_0))/std::sin(th_3)) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = safe_asin((-r_13*std::sin(th_0) + r_23*std::cos(th_0))/std::sin(th_3));
                // End of temp variables
                const double tmp_sol_value = M_PI - x0;
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
    ExplicitSolutionNode_node_7_solve_th_4_processor();
    // Finish code for explicit solution node 7
    
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
            const double th_0 = this_solution[0];
            const bool checked_result = std::fabs(Pz) <= 9.9999999999999995e-7 && std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor();
    // Finish code for equation all-zero dispatcher node 8
    
    // Code for explicit solution node 9, solved variable is th_1th_2_soa
    auto ExplicitSolutionNode_node_9_solve_th_1th_2_soa_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_2 = this_solution[4];
            const double th_3 = this_solution[5];
            const double th_4 = this_solution[6];
            
            const bool condition_0 = std::fabs(Pz) >= 9.9999999999999995e-7 || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = -a_0*std::cos(th_2) - a_1 + d_3*std::sin(th_3) + d_4*std::sin(th_4)*std::cos(th_3);
                const double x1 = -Px*std::cos(th_0) - Py*std::sin(th_0);
                const double x2 = a_0*std::sin(th_2) - d_2 + d_4*std::cos(th_4);
                // End of temp variables
                const double tmp_sol_value = std::atan2(Pz*x0 - x1*x2, Pz*x2 + x0*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(10, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_1th_2_soa_processor();
    // Finish code for explicit solution node 9
    
    // Code for non-branch dispatcher node 10
    // Actually, there is no code
    
    // Code for explicit solution node 11, solved variable is th_1
    auto ExplicitSolutionNode_node_11_solve_th_1_processor = [&]() -> void
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
            const double th_1th_2_soa = this_solution[2];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_1th_2_soa - th_2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(12, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1_processor();
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
            const double th_0 = this_solution[0];
            const double th_1th_2_soa = this_solution[2];
            const double th_3 = this_solution[5];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*(std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)*std::cos(th_1th_2_soa)) - r_21*(std::sin(th_0)*std::sin(th_3)*std::cos(th_1th_2_soa) + std::cos(th_0)*std::cos(th_3)) + r_31*std::sin(th_1th_2_soa)*std::sin(th_3)) >= zero_tolerance || std::fabs(r_12*(std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)*std::cos(th_1th_2_soa)) - r_22*(std::sin(th_0)*std::sin(th_3)*std::cos(th_1th_2_soa) + std::cos(th_0)*std::cos(th_3)) + r_32*std::sin(th_1th_2_soa)*std::sin(th_3)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_3);
                const double x1 = x0*std::sin(th_1th_2_soa);
                const double x2 = std::sin(th_0);
                const double x3 = std::cos(th_3);
                const double x4 = std::cos(th_0);
                const double x5 = x0*std::cos(th_1th_2_soa);
                const double x6 = x2*x3 - x4*x5;
                const double x7 = x2*x5 + x3*x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x6 + r_21*x7 - r_31*x1, -r_12*x6 + r_22*x7 - r_32*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
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
        const double value_at_0 = raw_ik_out_i[0];  // th_0
        new_ik_i[0] = value_at_0;
        const double value_at_1 = raw_ik_out_i[1];  // th_1
        new_ik_i[1] = value_at_1;
        const double value_at_2 = raw_ik_out_i[4];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[5];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[6];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[7];  // th_5
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

}; // struct rokae_SR4_ik

// Code below for debug
void test_ik_solve_rokae_SR4()
{
    std::array<double, rokae_SR4_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = rokae_SR4_ik::computeFK(theta);
    auto ik_output = rokae_SR4_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = rokae_SR4_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_rokae_SR4();
}
