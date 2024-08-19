#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct abb_crb15000_10_1_52_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 128;
static constexpr int n_tree_nodes = 28;
static constexpr int intermediate_solution_size = 8;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_0 = 0.15;
static constexpr double a_1 = 0.707;
static constexpr double a_2 = 0.11;
static constexpr double a_4 = 0.08;
static constexpr double d_3 = 0.637;
static constexpr double d_5 = 0.101;
static constexpr double pre_transform_special_symbol_23 = 0.4;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -1.5707963267948966;
static constexpr double th_2_offset_original2raw = 0.0;
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
    ee_transformed(2, 0) = 1.0*r_33;
    ee_transformed(2, 1) = -1.0*r_32;
    ee_transformed(2, 2) = 1.0*r_31;
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
    ee_transformed(0, 0) = 1.0*r_13;
    ee_transformed(0, 1) = -1.0*r_12;
    ee_transformed(0, 2) = 1.0*r_11;
    ee_transformed(0, 3) = 1.0*Px;
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
    const double x21 = a_1*x8;
    const double x22 = -x1*x7 + x1*x8*x9;
    const double x23 = -x2*x4 - x22*x3;
    const double x24 = -x1*x14 - x1*x15;
    const double x25 = x2*x22 - x3*x4;
    const double x26 = -x13*x24 + x17*x25;
    const double x27 = -x13*x25 - x17*x24;
    const double x28 = -x14 - x15;
    const double x29 = x7 - x8*x9;
    const double x30 = -x13*x29 + x17*x2*x28;
    const double x31 = -x13*x2*x28 - x17*x29;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x11 + x12*x19;
    ee_pose_raw(0, 1) = -x0*x19 - x11*x12;
    ee_pose_raw(0, 2) = x20;
    ee_pose_raw(0, 3) = a_0*x4 + a_2*x10 + a_4*x19 + d_3*x16 + d_5*x20 + x21*x4;
    ee_pose_raw(1, 0) = -x0*x23 + x12*x26;
    ee_pose_raw(1, 1) = -x0*x26 - x12*x23;
    ee_pose_raw(1, 2) = x27;
    ee_pose_raw(1, 3) = a_0*x1 + a_2*x22 + a_4*x26 + d_3*x24 + d_5*x27 + x1*x21;
    ee_pose_raw(2, 0) = x0*x28*x3 + x12*x30;
    ee_pose_raw(2, 1) = -x0*x30 + x12*x28*x3;
    ee_pose_raw(2, 2) = x31;
    ee_pose_raw(2, 3) = -a_1*x5 + a_2*x28 + a_4*x30 + d_3*x29 + d_5*x31;
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
    const double x4 = std::sin(th_1);
    const double x5 = std::cos(th_0);
    const double x6 = 1.0*x5;
    const double x7 = x4*x6;
    const double x8 = std::sin(th_2);
    const double x9 = std::cos(th_1);
    const double x10 = x6*x9;
    const double x11 = -x10*x8 - x3*x7;
    const double x12 = std::cos(th_3);
    const double x13 = std::sin(th_3);
    const double x14 = 1.0*x3*x5*x9 - x7*x8;
    const double x15 = 1.0*x0*x12 - x13*x14;
    const double x16 = std::cos(th_4);
    const double x17 = std::sin(th_4);
    const double x18 = x1*x13 + x12*x14;
    const double x19 = -x11*x16 - x17*x18;
    const double x20 = x1*x4;
    const double x21 = x1*x9;
    const double x22 = -x20*x3 - x21*x8;
    const double x23 = 1.0*x0*x3*x9 - x20*x8;
    const double x24 = -x12*x6 - x13*x23;
    const double x25 = x12*x23 - x13*x6;
    const double x26 = -x16*x22 - x17*x25;
    const double x27 = 1.0*x8;
    const double x28 = 1.0*x3;
    const double x29 = x27*x4 - x28*x9;
    const double x30 = -x27*x9 - x28*x4;
    const double x31 = x13*x30;
    const double x32 = -x12*x17*x30 - x16*x29;
    const double x33 = 1.0*a_1*x4;
    const double x34 = pre_transform_special_symbol_23 - x33;
    const double x35 = a_2*x30 + d_3*x29 + pre_transform_special_symbol_23 - x33;
    const double x36 = a_0*x1 + a_1*x21;
    const double x37 = a_2*x23 + d_3*x22 + x36;
    const double x38 = a_4*(x12*x16*x30 - x17*x29) + d_5*x32 + x35;
    const double x39 = a_4*(x16*x25 - x17*x22) + d_5*x26 + x37;
    const double x40 = a_0*x6 + a_1*x10;
    const double x41 = a_2*x14 + d_3*x11 + x40;
    const double x42 = a_4*(-x11*x17 + x16*x18) + d_5*x19 + x41;
    const double x43 = 1.0*a_0;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x2;
    jacobian(0, 2) = x2;
    jacobian(0, 3) = x11;
    jacobian(0, 4) = x15;
    jacobian(0, 5) = x19;
    jacobian(1, 1) = x6;
    jacobian(1, 2) = x6;
    jacobian(1, 3) = x22;
    jacobian(1, 4) = x24;
    jacobian(1, 5) = x26;
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x29;
    jacobian(2, 4) = -x31;
    jacobian(2, 5) = x32;
    jacobian(3, 1) = -pre_transform_special_symbol_23*x6;
    jacobian(3, 2) = -x34*x6;
    jacobian(3, 3) = -x22*x35 + x29*x37;
    jacobian(3, 4) = -x24*x35 - x31*x37;
    jacobian(3, 5) = -x26*x38 + x32*x39;
    jacobian(4, 1) = -pre_transform_special_symbol_23*x1;
    jacobian(4, 2) = -x1*x34;
    jacobian(4, 3) = x11*x35 - x29*x41;
    jacobian(4, 4) = x15*x35 + x31*x41;
    jacobian(4, 5) = x19*x38 - x32*x42;
    jacobian(5, 1) = std::pow(x0, 2)*x43 + x43*std::pow(x5, 2);
    jacobian(5, 2) = x1*x36 + x40*x6;
    jacobian(5, 3) = -x11*x37 + x22*x41;
    jacobian(5, 4) = -x15*x37 + x24*x41;
    jacobian(5, 5) = -x19*x39 + x26*x42;
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
    const double x4 = std::sin(th_1);
    const double x5 = std::cos(th_0);
    const double x6 = 1.0*x5;
    const double x7 = x4*x6;
    const double x8 = std::sin(th_2);
    const double x9 = std::cos(th_1);
    const double x10 = -x3*x7 - x6*x8*x9;
    const double x11 = std::cos(th_3);
    const double x12 = std::sin(th_3);
    const double x13 = 1.0*x3*x5*x9 - x7*x8;
    const double x14 = std::cos(th_4);
    const double x15 = std::sin(th_4);
    const double x16 = x1*x4;
    const double x17 = -x1*x8*x9 - x16*x3;
    const double x18 = 1.0*x0*x3*x9 - x16*x8;
    const double x19 = 1.0*x8;
    const double x20 = 1.0*x3;
    const double x21 = x19*x4 - x20*x9;
    const double x22 = -x19*x9 - x20*x4;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x2;
    jacobian(0, 2) = x2;
    jacobian(0, 3) = x10;
    jacobian(0, 4) = 1.0*x0*x11 - x12*x13;
    jacobian(0, 5) = -x10*x14 - x15*(x1*x12 + x11*x13);
    jacobian(1, 1) = x6;
    jacobian(1, 2) = x6;
    jacobian(1, 3) = x17;
    jacobian(1, 4) = -x11*x6 - x12*x18;
    jacobian(1, 5) = -x14*x17 - x15*(x11*x18 - x12*x6);
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x21;
    jacobian(2, 4) = -x12*x22;
    jacobian(2, 5) = -x11*x15*x22 - x14*x21;
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
    const double x6 = a_1*x5;
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
    const double x19 = a_2*x18 + d_3*x13 + pre_transform_special_symbol_23 - x6;
    const double x20 = 1.0*x10*x11*x14 - x14*x9;
    const double x21 = 1.0*x14;
    const double x22 = a_0*x21 + a_1*x12*x14;
    const double x23 = a_2*x20 + d_3*x17 + x22;
    const double x24 = std::sin(th_3);
    const double x25 = x18*x24;
    const double x26 = std::cos(th_3);
    const double x27 = -x2*x26 - x20*x24;
    const double x28 = std::cos(th_4);
    const double x29 = std::sin(th_4);
    const double x30 = -x13*x28 - x18*x26*x29;
    const double x31 = -x2*x24 + x20*x26;
    const double x32 = -x17*x28 - x29*x31;
    const double x33 = a_4*(-x13*x29 + x18*x26*x28) + d_5*x30 + x19;
    const double x34 = a_4*(-x17*x29 + x28*x31) + d_5*x32 + x23;
    const double x35 = 1.0*p_on_ee_x;
    const double x36 = p_on_ee_z*x21;
    const double x37 = x2*x4;
    const double x38 = x11*x2;
    const double x39 = -x10*x37 - x38*x8;
    const double x40 = 1.0*x1*x10*x11 - x37*x8;
    const double x41 = a_0*x2 + a_1*x38;
    const double x42 = a_2*x40 + d_3*x39 + x41;
    const double x43 = 1.0*x14*x26 - x24*x40;
    const double x44 = x21*x24 + x26*x40;
    const double x45 = -x28*x39 - x29*x44;
    const double x46 = a_4*(x28*x44 - x29*x39) + d_5*x45 + x42;
    const double x47 = x1*x35;
    const double x48 = x0*x14;
    const double x49 = 1.0*a_0;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = -pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = -x2*x7 + x3;
    jacobian(0, 3) = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x23 - x17*x19;
    jacobian(0, 4) = p_on_ee_y*x25 + p_on_ee_z*x27 - x19*x27 - x23*x25;
    jacobian(0, 5) = -p_on_ee_y*x30 + p_on_ee_z*x32 + x30*x34 - x32*x33;
    jacobian(1, 0) = x35;
    jacobian(1, 1) = -pre_transform_special_symbol_23*x21 + x36;
    jacobian(1, 2) = -x21*x7 + x36;
    jacobian(1, 3) = p_on_ee_x*x13 - p_on_ee_z*x39 - x13*x42 + x19*x39;
    jacobian(1, 4) = -p_on_ee_x*x25 - p_on_ee_z*x43 + x18*x24*x42 + x19*x43;
    jacobian(1, 5) = p_on_ee_x*x30 - p_on_ee_z*x45 - x30*x46 + x33*x45;
    jacobian(2, 1) = std::pow(x1, 2)*x49 + std::pow(x14, 2)*x49 - x47 - x48;
    jacobian(2, 2) = 1.0*x1*x41 + 1.0*x14*x22 - x47 - x48;
    jacobian(2, 3) = -p_on_ee_x*x17 + p_on_ee_y*x39 + x17*x42 - x23*x39;
    jacobian(2, 4) = -p_on_ee_x*x27 + p_on_ee_y*x43 - x23*x43 + x27*x42;
    jacobian(2, 5) = -p_on_ee_x*x32 + p_on_ee_y*x45 + x32*x46 - x34*x45;
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
    
    // Code for general_6dof solution node 1, solved variable is th_5
    auto General6DoFNumericalReduceSolutionNode_node_1_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;
        
        // The general 6-dof solution of root node
        Eigen::Matrix<double, 8, 8> R_l;
        R_l.setZero();
        R_l(0, 3) = -a_1;
        R_l(0, 7) = -a_2;
        R_l(1, 2) = -a_1;
        R_l(1, 6) = -a_2;
        R_l(2, 4) = a_1;
        R_l(3, 6) = -1;
        R_l(4, 7) = 1;
        R_l(5, 5) = 2*a_1*a_2;
        R_l(6, 1) = -a_1;
        R_l(7, 0) = -a_1;
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
        const double x3 = 2*r_13;
        const double x4 = -x3;
        const double x5 = 4*r_11;
        const double x6 = d_3*r_23;
        const double x7 = -x6;
        const double x8 = a_4*r_21;
        const double x9 = d_5*r_23;
        const double x10 = Px*r_11;
        const double x11 = r_21*x10;
        const double x12 = Px*r_12;
        const double x13 = r_22*x12;
        const double x14 = Px*r_13;
        const double x15 = r_23*x14;
        const double x16 = Pz*r_31;
        const double x17 = r_21*x16;
        const double x18 = Pz*r_32;
        const double x19 = r_22*x18;
        const double x20 = Pz*r_33;
        const double x21 = r_23*x20;
        const double x22 = std::pow(r_21, 2);
        const double x23 = Py*x22;
        const double x24 = std::pow(r_22, 2);
        const double x25 = Py*x24;
        const double x26 = std::pow(r_23, 2);
        const double x27 = Py*x26;
        const double x28 = x11 + x13 + x15 + x17 + x19 + x21 + x23 + x25 + x27 - x9;
        const double x29 = x28 + x8;
        const double x30 = x29 + x7;
        const double x31 = a_0*r_11;
        const double x32 = r_21*x31;
        const double x33 = a_0*r_12;
        const double x34 = r_22*x33;
        const double x35 = a_0*r_13;
        const double x36 = r_23*x35;
        const double x37 = -x32 - x34 - x36;
        const double x38 = d_3*x1;
        const double x39 = -x38;
        const double x40 = x29 + x6;
        const double x41 = d_3*x3;
        const double x42 = -x41;
        const double x43 = 2*a_4;
        const double x44 = r_11*x43;
        const double x45 = d_5*x3;
        const double x46 = std::pow(r_11, 2);
        const double x47 = Px*x46;
        const double x48 = 2*x47;
        const double x49 = std::pow(r_12, 2);
        const double x50 = Px*x49;
        const double x51 = 2*x50;
        const double x52 = std::pow(r_13, 2);
        const double x53 = Px*x52;
        const double x54 = 2*x53;
        const double x55 = Py*x1;
        const double x56 = Py*r_22;
        const double x57 = r_12*x56;
        const double x58 = Py*r_23;
        const double x59 = x3*x58;
        const double x60 = r_11*x16;
        const double x61 = r_12*x18;
        const double x62 = x20*x3;
        const double x63 = r_11*x55 - x45 + x48 + x51 + x54 + 2*x57 + x59 + 2*x60 + 2*x61 + x62;
        const double x64 = x44 + x63;
        const double x65 = d_3*x5;
        const double x66 = x32 + x34 + x36;
        const double x67 = -a_2;
        const double x68 = a_4*r_22;
        const double x69 = Py*r_21;
        const double x70 = x10 + x16 + x69;
        const double x71 = R_l_inv_50*a_1;
        const double x72 = x70*x71;
        const double x73 = a_1*r_21;
        const double x74 = R_l_inv_53*x73;
        const double x75 = d_5*r_22;
        const double x76 = r_23*x12;
        const double x77 = r_23*x18;
        const double x78 = r_22*x14;
        const double x79 = r_22*x20;
        const double x80 = x75 + x76 + x77 - x78 - x79;
        const double x81 = R_l_inv_56*a_1;
        const double x82 = x80*x81;
        const double x83 = -R_l_inv_52*a_1*d_3;
        const double x84 = std::pow(a_4, 2);
        const double x85 = std::pow(d_3, 2);
        const double x86 = std::pow(d_5, 2);
        const double x87 = std::pow(a_1, 2);
        const double x88 = std::pow(a_2, 2);
        const double x89 = 2*x9;
        const double x90 = 2*d_5;
        const double x91 = 2*x12;
        const double x92 = 2*x10;
        const double x93 = 2*x18;
        const double x94 = 2*x20;
        const double x95 = std::pow(Px, 2);
        const double x96 = x46*x95;
        const double x97 = x49*x95;
        const double x98 = x52*x95;
        const double x99 = std::pow(Py, 2);
        const double x100 = x22*x99;
        const double x101 = x24*x99;
        const double x102 = x26*x99;
        const double x103 = std::pow(Pz, 2);
        const double x104 = std::pow(r_31, 2);
        const double x105 = x103*x104;
        const double x106 = std::pow(r_32, 2);
        const double x107 = x103*x106;
        const double x108 = std::pow(r_33, 2)*x103;
        const double x109 = std::pow(a_0, 2);
        const double x110 = x109*x22;
        const double x111 = x109*x24;
        const double x112 = x109*x26;
        const double x113 = -Px*x45 + Px*x59 + Px*x62 - Py*x89 + x10*x55 + x100 + x101 + x102 + x105 + x107 + x108 + x110 + x111 + x112 + x16*x55 + x16*x92 + x18*x91 - x20*x90 + x56*x91 + x56*x93 + x58*x94 + x84 + x85 + x86 - x87 - x88 + x96 + x97 + x98;
        const double x114 = -R_l_inv_55*a_1*x113;
        const double x115 = x31*x71;
        const double x116 = r_12*r_23;
        const double x117 = r_13*r_22;
        const double x118 = x116 - x117;
        const double x119 = a_0*x118;
        const double x120 = x119*x81;
        const double x121 = R_l_inv_57*a_1;
        const double x122 = a_4*r_23;
        const double x123 = x121*x122;
        const double x124 = R_l_inv_57*d_3*x73;
        const double x125 = R_l_inv_55*a_1;
        const double x126 = d_5*r_13;
        const double x127 = r_11*x69;
        const double x128 = r_13*x58;
        const double x129 = r_13*x20;
        const double x130 = 2*a_0;
        const double x131 = x130*(-x126 + x127 + x128 + x129 + x47 + x50 + x53 + x57 + x60 + x61);
        const double x132 = x125*x131;
        const double x133 = -x132;
        const double x134 = x114 + x115 + x120 + x123 + x124 + x133 + x67 + x68 + x72 + x74 + x82 + x83;
        const double x135 = a_4*x71;
        const double x136 = -d_5 + x14 + x20 + x58;
        const double x137 = R_l_inv_52*a_1;
        const double x138 = x136*x137;
        const double x139 = d_5*r_21;
        const double x140 = r_23*x10;
        const double x141 = r_23*x16;
        const double x142 = r_21*x14;
        const double x143 = r_21*x20;
        const double x144 = x139 + x140 + x141 - x142 - x143;
        const double x145 = x121*x144;
        const double x146 = r_11*r_22;
        const double x147 = a_0*x146;
        const double x148 = R_l_inv_54*a_1;
        const double x149 = r_22*x148;
        const double x150 = -x149;
        const double x151 = r_12*r_21;
        const double x152 = a_0*x151;
        const double x153 = -x152;
        const double x154 = d_3*r_22;
        const double x155 = x154*x81;
        const double x156 = R_l_inv_51*a_1;
        const double x157 = x156*x33;
        const double x158 = -x157;
        const double x159 = 2*x136;
        const double x160 = d_3*x125;
        const double x161 = x159*x160;
        const double x162 = x31*x43;
        const double x163 = x125*x162;
        const double x164 = -x163;
        const double x165 = x135 + x138 + x145 + x147 + x150 + x153 + x155 + x158 + x161 + x164;
        const double x166 = x137*x35;
        const double x167 = a_0*x125;
        const double x168 = x167*x41;
        const double x169 = r_22*x10;
        const double x170 = r_22*x16;
        const double x171 = r_21*x12;
        const double x172 = r_21*x18;
        const double x173 = x169 + x170 - x171 - x172;
        const double x174 = x166 + x168 + x173;
        const double x175 = x12 + x18 + x56;
        const double x176 = x156*x175;
        const double x177 = r_11*r_23;
        const double x178 = r_13*r_21;
        const double x179 = a_0*(x177 - x178);
        const double x180 = x121*x179;
        const double x181 = x43*x70;
        const double x182 = x125*x181;
        const double x183 = -x176 + x180 - x182;
        const double x184 = R_l_inv_53*a_1;
        const double x185 = 2*r_23;
        const double x186 = x184*x185;
        const double x187 = 2*x81;
        const double x188 = x173*x187;
        const double x189 = a_0*x71;
        const double x190 = x189*x3;
        const double x191 = 2*x121;
        const double x192 = x191*x6;
        const double x193 = -2*R_l_inv_50*a_1*x136;
        const double x194 = r_22*x3;
        const double x195 = a_0*x194;
        const double x196 = -x195;
        const double x197 = x137*x43;
        const double x198 = 2*x70;
        const double x199 = x137*x198;
        const double x200 = a_0*x116;
        const double x201 = 2*x200;
        const double x202 = x146 - x151;
        const double x203 = -2*R_l_inv_56*a_0*a_1*x202;
        const double x204 = 4*a_4;
        const double x205 = x160*x204;
        const double x206 = 4*x70;
        const double x207 = x160*x206;
        const double x208 = x193 + x196 + x197 + x199 + x201 + x203 + x205 + x207;
        const double x209 = 2*x75;
        const double x210 = Px*x194;
        const double x211 = 2*x79;
        const double x212 = 2*x76;
        const double x213 = 2*x77;
        const double x214 = 2*x68;
        const double x215 = x214*x81;
        const double x216 = 2*x31;
        const double x217 = x137*x216;
        const double x218 = x167*x65;
        const double x219 = x209 - x210 - x211 + x212 + x213 - x215 + x217 + x218;
        const double x220 = d_3*x137;
        const double x221 = x113*x125;
        const double x222 = -x180;
        const double x223 = a_2 + x120 + x176 + x182 + x220 + x221 + x222 + x72;
        const double x224 = x115 - x123 + x124 + x132 + x68 + x74 + x82;
        const double x225 = -x145 + x149 - x155 + x157 + x163;
        const double x226 = x135 + x138 + x147 + x153 + x161 + x225;
        const double x227 = r_11*x56;
        const double x228 = 2*x227;
        const double x229 = r_11*x18;
        const double x230 = 2*x229;
        const double x231 = r_12*x55;
        const double x232 = r_12*x16;
        const double x233 = 2*x232;
        const double x234 = a_0*x185;
        const double x235 = x137*x234;
        const double x236 = 4*a_0;
        const double x237 = x125*x236;
        const double x238 = x237*x6;
        const double x239 = 2*r_12;
        const double x240 = x148*x239;
        const double x241 = -x240;
        const double x242 = d_5*r_11;
        const double x243 = r_13*x69;
        const double x244 = r_13*x16;
        const double x245 = r_11*x58;
        const double x246 = r_11*x20;
        const double x247 = x242 + x243 + x244 - x245 - x246;
        const double x248 = x191*x247;
        const double x249 = r_22*x130;
        const double x250 = x156*x249;
        const double x251 = d_3*r_12;
        const double x252 = x187*x251;
        const double x253 = a_4*x3;
        const double x254 = x121*x253;
        const double x255 = x236*x28;
        const double x256 = x125*x255;
        const double x257 = x237*x8;
        const double x258 = x241 + x248 + x250 + x252 + x254 + x256 + x257;
        const double x259 = r_12*x43;
        const double x260 = 2*r_11;
        const double x261 = x184*x260;
        const double x262 = d_5*r_12;
        const double x263 = r_13*x56;
        const double x264 = r_13*x18;
        const double x265 = r_12*x58;
        const double x266 = r_12*x20;
        const double x267 = x262 + x263 + x264 - x265 - x266;
        const double x268 = x187*x267;
        const double x269 = x1*x189;
        const double x270 = d_3*x260;
        const double x271 = x121*x270;
        const double x272 = x259 + x261 + x268 - x269 + x271;
        const double x273 = r_12*x69;
        const double x274 = x227 + x229 - x232 - x273;
        const double x275 = 4*x81;
        const double x276 = x274*x275;
        const double x277 = 4*r_13;
        const double x278 = 4*r_23;
        const double x279 = x189*x278;
        const double x280 = d_3*x277;
        const double x281 = 4*x262;
        const double x282 = 4*x263;
        const double x283 = 4*x264;
        const double x284 = 4*x265;
        const double x285 = r_21*x236;
        const double x286 = 4*r_12;
        const double x287 = a_4*x286;
        const double x288 = d_3*r_21;
        const double x289 = 8*x288;
        const double x290 = x137*x285 + x167*x289 + 4*x266 - x281 - x282 - x283 + x284 + x287*x81;
        const double x291 = x228 + x230 - x231 - x233 + x235 + x238;
        const double x292 = -x72;
        const double x293 = -x120;
        const double x294 = x176 + x182 + x222 + x292 + x293;
        const double x295 = x123 + x133;
        const double x296 = a_2 + x115 + x124 + x220 + x221 + x295 + x68 + x74 + x82;
        const double x297 = -x135 - x138 - x147 + x152 - x161;
        const double x298 = x145 + x150 + x155 + x158 + x164 + x297;
        const double x299 = x186 + x188 + x190 + x192;
        const double x300 = -x166 - x168 - x169 - x170 + x171 + x172;
        const double x301 = x165 + x300;
        const double x302 = a_1*a_2;
        const double x303 = 2*x302;
        const double x304 = x87 + x88;
        const double x305 = R_l_inv_62*x304;
        const double x306 = R_l_inv_22*x303 + x305;
        const double x307 = d_3*x306;
        const double x308 = R_l_inv_24*x303 + R_l_inv_64*x304;
        const double x309 = r_22*x308;
        const double x310 = R_l_inv_61*x304;
        const double x311 = R_l_inv_21*x303 + x310;
        const double x312 = x175*x311;
        const double x313 = R_l_inv_25*x303 + R_l_inv_65*x304;
        const double x314 = x113*x313;
        const double x315 = R_l_inv_67*x304;
        const double x316 = R_l_inv_27*x303 + x315;
        const double x317 = x144*x316;
        const double x318 = -x317;
        const double x319 = x311*x33;
        const double x320 = x179*x316;
        const double x321 = -x320;
        const double x322 = x122*x316;
        const double x323 = -x322;
        const double x324 = R_l_inv_66*x304;
        const double x325 = R_l_inv_26*x303 + x324;
        const double x326 = x154*x325;
        const double x327 = -x326;
        const double x328 = x43*x6;
        const double x329 = -x328;
        const double x330 = d_5*x38;
        const double x331 = -x330;
        const double x332 = x131*x313;
        const double x333 = x181*x313;
        const double x334 = 2*x6;
        const double x335 = x10*x334;
        const double x336 = -x335;
        const double x337 = x16*x334;
        const double x338 = -x337;
        const double x339 = x216*x6;
        const double x340 = -x339;
        const double x341 = x14*x38;
        const double x342 = x20*x38;
        const double x343 = x162*x313;
        const double x344 = x35*x38;
        const double x345 = x307 + x309 + x312 + x314 + x318 + x319 + x321 + x323 + x327 + x329 + x331 + x332 + x333 + x336 + x338 + x340 + x341 + x342 + x343 + x344;
        const double x346 = r_21*x84;
        const double x347 = std::pow(r_21, 3);
        const double x348 = x347*x99;
        const double x349 = x109*x347;
        const double x350 = R_l_inv_60*x304;
        const double x351 = R_l_inv_20*x303 + x350;
        const double x352 = a_4*x351;
        const double x353 = r_21*x85;
        const double x354 = -x353;
        const double x355 = r_21*x86;
        const double x356 = -x355;
        const double x357 = R_l_inv_23*x303 + R_l_inv_63*x304;
        const double x358 = r_21*x357;
        const double x359 = -x358;
        const double x360 = x136*x306;
        const double x361 = x325*x80;
        const double x362 = -x361;
        const double x363 = r_21*x96;
        const double x364 = r_21*x101;
        const double x365 = r_21*x102;
        const double x366 = r_21*x105;
        const double x367 = r_21*x111;
        const double x368 = r_21*x112;
        const double x369 = x31*x351;
        const double x370 = -x369;
        const double x371 = x288*x316;
        const double x372 = -x371;
        const double x373 = r_21*x97;
        const double x374 = -x373;
        const double x375 = r_21*x98;
        const double x376 = -x375;
        const double x377 = r_21*x107;
        const double x378 = -x377;
        const double x379 = r_21*x108;
        const double x380 = -x379;
        const double x381 = d_3*x313;
        const double x382 = x159*x381;
        const double x383 = x1*x109;
        const double x384 = x383*x46;
        const double x385 = -x384;
        const double x386 = x383*x49;
        const double x387 = -x386;
        const double x388 = x383*x52;
        const double x389 = -x388;
        const double x390 = x10*x89;
        const double x391 = -x390;
        const double x392 = x16*x89;
        const double x393 = -x392;
        const double x394 = a_4*x1;
        const double x395 = x31*x394;
        const double x396 = x214*x33;
        const double x397 = x122*x3;
        const double x398 = a_0*x397;
        const double x399 = x23*x92;
        const double x400 = x25*x92;
        const double x401 = x27*x92;
        const double x402 = d_5*x1;
        const double x403 = x14*x402;
        const double x404 = 2*x16;
        const double x405 = x23*x404;
        const double x406 = x25*x404;
        const double x407 = x27*x404;
        const double x408 = x20*x402;
        const double x409 = x146*x239*x95;
        const double x410 = x3*x95;
        const double x411 = x177*x410;
        const double x412 = 2*r_22;
        const double x413 = r_31*x103;
        const double x414 = r_32*x413;
        const double x415 = x412*x414;
        const double x416 = r_33*x413;
        const double x417 = x185*x416;
        const double x418 = x1*x18;
        const double x419 = x12*x418;
        const double x420 = -x419;
        const double x421 = x1*x20;
        const double x422 = x14*x421;
        const double x423 = -x422;
        const double x424 = x1*x16;
        const double x425 = x10*x424;
        const double x426 = x19*x92;
        const double x427 = x21*x92;
        const double x428 = x13*x404;
        const double x429 = Px*x3;
        const double x430 = x141*x429;
        const double x431 = x346 + x348 + x349 - x352 + x354 + x356 + x359 - x360 + x362 + x363 + x364 + x365 + x366 + x367 + x368 + x370 + x372 + x374 + x376 + x378 + x380 - x382 + x385 + x387 + x389 + x391 + x393 - x395 - x396 - x398 + x399 + x400 + x401 + x403 + x405 + x406 + x407 + x408 + x409 + x411 + x415 + x417 + x420 + x423 + x425 + x426 + x427 + x428 + x430;
        const double x432 = x306*x35;
        const double x433 = x43*x9;
        const double x434 = x23*x43;
        const double x435 = x25*x43;
        const double x436 = x27*x43;
        const double x437 = a_0*x313;
        const double x438 = x41*x437;
        const double x439 = x10*x394;
        const double x440 = x12*x214;
        const double x441 = Px*x397;
        const double x442 = x16*x394;
        const double x443 = x18*x214;
        const double x444 = x21*x43;
        const double x445 = -x432 - x433 + x434 + x435 + x436 - x438 + x439 + x440 + x441 + x442 + x443 + x444;
        const double x446 = x351*x70;
        const double x447 = x119*x325;
        const double x448 = a_0*x1;
        const double x449 = x448*x47;
        const double x450 = x448*x50;
        const double x451 = x448*x53;
        const double x452 = x216*x23;
        const double x453 = x31*x89;
        const double x454 = x216*x25;
        const double x455 = x216*x27;
        const double x456 = x126*x448;
        const double x457 = 4*x56;
        const double x458 = x152*x457;
        const double x459 = 4*x69;
        const double x460 = x36*x459;
        const double x461 = x31*x424;
        const double x462 = x33*x418;
        const double x463 = x34*x404;
        const double x464 = x35*x421;
        const double x465 = a_0*x3;
        const double x466 = x141*x465;
        const double x467 = x147*x93;
        const double x468 = x130*x177;
        const double x469 = x20*x468;
        const double x470 = -x446 - x447 - x449 - x450 - x451 - x452 - x453 + x454 + x455 + x456 - x458 - x460 - x461 - x462 - x463 - x464 - x466 + x467 + x469;
        const double x471 = 4*x302;
        const double x472 = R_l_inv_20*x471 + 2*x350;
        const double x473 = x136*x472;
        const double x474 = x70*(R_l_inv_22*x471 + 2*x305);
        const double x475 = -x306*x43;
        const double x476 = -x204*x381;
        const double x477 = x206*x381;
        const double x478 = x130*x202;
        const double x479 = x325*x478;
        const double x480 = 4*x35;
        const double x481 = -x23*x480;
        const double x482 = -x25*x480;
        const double x483 = 4*x8;
        const double x484 = x35*x483;
        const double x485 = x139*x5;
        const double x486 = -a_0*x485;
        const double x487 = 4*x33;
        const double x488 = -x487*x75;
        const double x489 = -x480*x9;
        const double x490 = a_0*x278;
        const double x491 = x47*x490;
        const double x492 = x490*x50;
        const double x493 = x490*x53;
        const double x494 = x27*x480;
        const double x495 = x122*x5;
        const double x496 = a_0*x495;
        const double x497 = x178*x236;
        const double x498 = -x16*x497;
        const double x499 = a_0*x117;
        const double x500 = 4*x18;
        const double x501 = -x499*x500;
        const double x502 = a_0*x5;
        const double x503 = x143*x502;
        const double x504 = x141*x502;
        const double x505 = 4*r_22;
        const double x506 = x20*x505;
        const double x507 = x33*x506;
        const double x508 = x200*x500;
        const double x509 = x21*x480;
        const double x510 = 8*x177;
        const double x511 = a_0*x510*x69;
        const double x512 = 8*x56;
        const double x513 = x200*x512;
        const double x514 = x473 - x474 + x475 + x476 - x477 + x479 + x481 + x482 - x484 + x486 + x488 + x489 + x491 + x492 + x493 + x494 + x496 + x498 + x501 + x503 + x504 + x507 + x508 + x509 + x511 + x513;
        const double x515 = x216*x306;
        const double x516 = x214*x325;
        const double x517 = 4*d_5;
        const double x518 = x517*x8;
        const double x519 = x14*x483;
        const double x520 = x20*x483;
        const double x521 = x437*x65;
        const double x522 = Px*x495;
        const double x523 = x141*x204;
        const double x524 = -x515 + x516 + x518 - x519 - x520 - x521 + x522 + x523;
        const double x525 = R_l_inv_26*x471 + 2*x324;
        const double x526 = x173*x525;
        const double x527 = r_23*x86;
        const double x528 = 2*x527;
        const double x529 = std::pow(r_23, 3);
        const double x530 = x529*x99;
        const double x531 = 2*x530;
        const double x532 = x109*x529;
        const double x533 = 2*x532;
        const double x534 = r_23*x84;
        const double x535 = 2*x534;
        const double x536 = r_23*x85;
        const double x537 = 2*x536;
        const double x538 = 2*x357;
        const double x539 = r_23*x538;
        const double x540 = r_23*x98;
        const double x541 = 2*x540;
        const double x542 = r_23*x100;
        const double x543 = 2*x542;
        const double x544 = r_23*x101;
        const double x545 = 2*x544;
        const double x546 = r_23*x108;
        const double x547 = 2*x546;
        const double x548 = r_23*x110;
        const double x549 = 2*x548;
        const double x550 = r_23*x111;
        const double x551 = 2*x550;
        const double x552 = a_0*x351;
        const double x553 = x3*x552;
        const double x554 = x316*x334;
        const double x555 = r_23*x96;
        const double x556 = 2*x555;
        const double x557 = r_23*x97;
        const double x558 = 2*x557;
        const double x559 = r_23*x105;
        const double x560 = 2*x559;
        const double x561 = r_23*x107;
        const double x562 = 2*x561;
        const double x563 = x23*x517;
        const double x564 = x25*x517;
        const double x565 = x27*x517;
        const double x566 = x109*x46;
        const double x567 = x278*x566;
        const double x568 = x109*x278;
        const double x569 = x49*x568;
        const double x570 = x52*x568;
        const double x571 = 4*x14;
        const double x572 = x23*x571;
        const double x573 = x25*x571;
        const double x574 = x27*x571;
        const double x575 = 4*x20;
        const double x576 = x23*x575;
        const double x577 = x25*x575;
        const double x578 = x27*x575;
        const double x579 = x5*x95;
        const double x580 = x178*x579;
        const double x581 = 4*x95;
        const double x582 = r_12*x117*x581;
        const double x583 = 4*r_21;
        const double x584 = x416*x583;
        const double x585 = r_32*r_33*x103;
        const double x586 = x505*x585;
        const double x587 = Px*x485;
        const double x588 = 4*x75;
        const double x589 = x12*x588;
        const double x590 = x571*x9;
        const double x591 = 4*x139;
        const double x592 = x16*x591;
        const double x593 = x18*x588;
        const double x594 = x575*x9;
        const double x595 = Px*x5;
        const double x596 = x143*x595;
        const double x597 = x12*x506;
        const double x598 = x17*x571;
        const double x599 = x18*x505;
        const double x600 = x14*x599;
        const double x601 = x15*x575;
        const double x602 = x141*x595;
        const double x603 = x500*x76;
        const double x604 = x526 - x528 - x531 - x533 + x535 + x537 + x539 - x541 - x543 - x545 - x547 - x549 - x551 + x553 + x554 + x556 + x558 + x560 + x562 + x563 + x564 + x565 + x567 + x569 + x570 - x572 - x573 - x574 - x576 - x577 - x578 - x580 - x582 - x584 - x586 + x587 + x589 + x590 + x592 + x593 + x594 - x596 - x597 - x598 - x600 - x601 + x602 + x603;
        const double x605 = -x434;
        const double x606 = -x435;
        const double x607 = -x436;
        const double x608 = -x439;
        const double x609 = -x440;
        const double x610 = -x441;
        const double x611 = -x442;
        const double x612 = -x443;
        const double x613 = -x444;
        const double x614 = x352 + x360 + x382 + x395 + x396 + x398 + x432 + x433 + x438 + x605 + x606 + x607 + x608 + x609 + x610 + x611 + x612 + x613;
        const double x615 = x446 + x447 + x449 + x450 + x451 + x452 + x453 - x454 - x455 - x456 + x458 + x460 + x461 + x462 + x463 + x464 + x466 - x467 - x469;
        const double x616 = -x267*x525;
        const double x617 = -x260*x85;
        const double x618 = -x260*x86;
        const double x619 = -r_11*x538;
        const double x620 = x260*x84;
        const double x621 = std::pow(r_11, 3);
        const double x622 = 2*x95;
        const double x623 = x621*x622;
        const double x624 = x255*x313;
        const double x625 = a_4*x280;
        const double x626 = x253*x316;
        const double x627 = -x270*x316;
        const double x628 = -x101*x260;
        const double x629 = -x102*x260;
        const double x630 = -x107*x260;
        const double x631 = -x108*x260;
        const double x632 = -x111*x260;
        const double x633 = -x112*x260;
        const double x634 = x1*x552;
        const double x635 = x260*x97;
        const double x636 = x260*x98;
        const double x637 = x100*x260;
        const double x638 = x105*x260;
        const double x639 = x110*x260;
        const double x640 = 4*x126;
        const double x641 = -x640*x69;
        const double x642 = -x16*x640;
        const double x643 = x459*x47;
        const double x644 = x459*x50;
        const double x645 = x459*x53;
        const double x646 = 4*x16;
        const double x647 = x47*x646;
        const double x648 = x50*x646;
        const double x649 = x53*x646;
        const double x650 = Py*x5;
        const double x651 = x650*x9;
        const double x652 = d_5*x5;
        const double x653 = x20*x652;
        const double x654 = x505*x99;
        const double x655 = x151*x654;
        const double x656 = x109*x505;
        const double x657 = x151*x656;
        const double x658 = x286*x414;
        const double x659 = x278*x99;
        const double x660 = x178*x659;
        const double x661 = x178*x568;
        const double x662 = x277*x416;
        const double x663 = x18*x5;
        const double x664 = -x56*x663;
        const double x665 = x20*x5;
        const double x666 = -x58*x665;
        const double x667 = x16*x5;
        const double x668 = x667*x69;
        const double x669 = x459*x61;
        const double x670 = x57*x646;
        const double x671 = x129*x459;
        const double x672 = x128*x646;
        const double x673 = x616 + x617 + x618 + x619 + x620 + x623 - x624 - x625 - x626 + x627 + x628 + x629 + x630 + x631 + x632 + x633 + x634 + x635 + x636 + x637 + x638 + x639 + x641 + x642 + x643 + x644 + x645 + x647 + x648 + x649 + x651 + x653 + x655 + x657 + x658 + x660 + x661 + x662 + x664 + x666 + x668 + x669 + x670 + x671 + x672;
        const double x674 = R_l_inv_27*x471 + 2*x315;
        const double x675 = x247*x674;
        const double x676 = x239*x308;
        const double x677 = d_5*x65;
        const double x678 = x249*x311;
        const double x679 = 2*x251;
        const double x680 = x325*x679;
        const double x681 = 4*d_3;
        const double x682 = x243*x681;
        const double x683 = x244*x681;
        const double x684 = x236*x313;
        const double x685 = x684*x8;
        const double x686 = x6*x650;
        const double x687 = x20*x65;
        const double x688 = -x675 + x676 - x677 - x678 - x680 - x682 - x683 - x685 + x686 + x687;
        const double x689 = x126*x204;
        const double x690 = x234*x306;
        const double x691 = x204*x47;
        const double x692 = x204*x50;
        const double x693 = x204*x53;
        const double x694 = x650*x8;
        const double x695 = Py*x286;
        const double x696 = x68*x695;
        const double x697 = x128*x204;
        const double x698 = a_4*x5;
        const double x699 = x16*x698;
        const double x700 = x204*x61;
        const double x701 = x129*x204;
        const double x702 = x6*x684;
        const double x703 = -x689 + x690 + x691 + x692 + x693 + x694 + x696 + x697 + x699 + x700 + x701 + x702;
        const double x704 = 8*x242;
        const double x705 = a_4*x704;
        const double x706 = x285*x306;
        const double x707 = x287*x325;
        const double x708 = Py*x8;
        const double x709 = 8*r_13;
        const double x710 = x708*x709;
        const double x711 = 8*a_4;
        const double x712 = x244*x711;
        const double x713 = x289*x437;
        const double x714 = x245*x711;
        const double x715 = x246*x711;
        const double x716 = 8*x302;
        const double x717 = R_l_inv_26*x716 + 4*x324;
        const double x718 = std::pow(r_13, 3);
        const double x719 = 8*x58;
        const double x720 = 8*x20;
        const double x721 = r_21*x99;
        const double x722 = r_21*x109;
        const double x723 = 8*r_11;
        const double x724 = 8*r_22;
        const double x725 = x724*x99;
        const double x726 = x109*x724;
        const double x727 = 8*r_12;
        const double x728 = x56*x727;
        const double x729 = x18*x727;
        const double x730 = -8*Px*d_5*x46 - 8*Px*d_5*x49 - 8*Px*d_5*x52 - 8*Py*Pz*r_13*r_21*r_31 - 8*Py*Pz*r_13*r_22*r_32 - 8*Py*d_5*r_11*r_21 - 8*Py*d_5*r_12*r_22 - 8*Py*d_5*r_13*r_23 - 8*Pz*d_5*r_11*r_31 - 8*Pz*d_5*r_12*r_32 - 8*Pz*d_5*r_13*r_33 - 4*d_3*r_13*x316 - 4*r_13*x103*x104 - 4*r_13*x103*x106 - 4*r_13*x109*x22 - 4*r_13*x109*x24 - 4*r_13*x22*x99 - 4*r_13*x24*x99 - 4*r_13*x357 - 4*r_13*x84 - 4*r_13*x85 + x102*x277 + x108*x277 + x112*x277 + x116*x725 + x116*x726 + x127*x720 + x128*x720 + x20*x728 + x274*x717 + x277*x86 + x277*x96 + x277*x97 + x278*x552 + x416*x723 + x47*x719 + x47*x720 + x50*x719 + x50*x720 + x510*x721 + x510*x722 + x53*x719 + x53*x720 + x58*x729 + x581*x718 + x585*x727 + x60*x719;
        const double x731 = x675 - x676 + x677 + x678 + x680 + x682 + x683 + x685 - x686 - x687;
        const double x732 = x616 + x617 + x618 + x619 + x620 + x623 + x624 + x625 + x626 + x627 + x628 + x629 + x630 + x631 + x632 + x633 + x634 + x635 + x636 + x637 + x638 + x639 + x641 + x642 + x643 + x644 + x645 + x647 + x648 + x649 + x651 + x653 + x655 + x657 + x658 + x660 + x661 + x662 + x664 + x666 + x668 + x669 + x670 + x671 + x672;
        const double x733 = -d_3*x306;
        const double x734 = -x113*x313;
        const double x735 = x323 + x329 + x332 + x615 + x733 + x734;
        const double x736 = -x312;
        const double x737 = -x333;
        const double x738 = -x344;
        const double x739 = x309 + x318 + x319 + x320 + x327 + x331 + x336 + x338 + x339 + x341 + x342 + x343 + x736 + x737 + x738;
        const double x740 = x346 + x348 + x349 + x352 + x354 + x356 + x359 + x360 + x362 + x363 + x364 + x365 + x366 + x367 + x368 + x370 + x372 + x374 + x376 + x378 + x380 + x382 + x385 + x387 + x389 + x391 + x393 + x395 + x396 + x398 + x399 + x400 + x401 + x403 + x405 + x406 + x407 + x408 + x409 + x411 + x415 + x417 + x420 + x423 + x425 + x426 + x427 + x428 + x430 + x445;
        const double x741 = x515 - x516 - x518 + x519 + x520 + x521 - x522 - x523;
        const double x742 = -x526 + x528 + x531 + x533 - x535 - x537 - x539 + x541 + x543 + x545 + x547 + x549 + x551 - x553 - x554 - x556 - x558 - x560 - x562 - x563 - x564 - x565 - x567 - x569 - x570 + x572 + x573 + x574 + x576 + x577 + x578 + x580 + x582 + x584 + x586 - x587 - x589 - x590 - x592 - x593 - x594 + x596 + x597 + x598 + x600 + x601 - x602 - x603;
        const double x743 = -x332;
        const double x744 = x307 + x314 + x322 + x328 + x615 + x743;
        const double x745 = x312 + x321 + x333 + x340 + x344;
        const double x746 = -x309 + x317 - x319 + x326 + x330 + x335 + x337 - x341 - x342 - x343;
        const double x747 = x745 + x746;
        const double x748 = r_22*x84;
        const double x749 = r_22*x85;
        const double x750 = r_22*x86;
        const double x751 = R_l_inv_77*x304;
        const double x752 = R_l_inv_37*x303 + x751;
        const double x753 = x144*x752;
        const double x754 = R_l_inv_34*x303 + R_l_inv_74*x304;
        const double x755 = r_22*x754;
        const double x756 = std::pow(r_22, 3);
        const double x757 = x756*x99;
        const double x758 = x109*x756;
        const double x759 = R_l_inv_76*x304;
        const double x760 = R_l_inv_36*x303 + x759;
        const double x761 = x154*x760;
        const double x762 = r_22*x96;
        const double x763 = r_22*x98;
        const double x764 = r_22*x105;
        const double x765 = r_22*x108;
        const double x766 = R_l_inv_71*x304;
        const double x767 = R_l_inv_31*x303 + x766;
        const double x768 = x33*x767;
        const double x769 = r_22*x97;
        const double x770 = r_22*x100;
        const double x771 = r_22*x102;
        const double x772 = r_22*x107;
        const double x773 = r_22*x110;
        const double x774 = r_22*x112;
        const double x775 = x412*x566;
        const double x776 = x109*x412;
        const double x777 = x49*x776;
        const double x778 = x52*x776;
        const double x779 = x23*x91;
        const double x780 = x25*x91;
        const double x781 = x27*x91;
        const double x782 = x429*x75;
        const double x783 = x23*x93;
        const double x784 = x25*x93;
        const double x785 = x27*x93;
        const double x786 = x20*x209;
        const double x787 = R_l_inv_35*x303 + R_l_inv_75*x304;
        const double x788 = x162*x787;
        const double x789 = r_12*x95;
        const double x790 = r_11*x1;
        const double x791 = x789*x790;
        const double x792 = x116*x410;
        const double x793 = x1*x414;
        const double x794 = x185*x585;
        const double x795 = x12*x89;
        const double x796 = x18*x89;
        const double x797 = x10*x418;
        const double x798 = x12*x424;
        const double x799 = x13*x93;
        const double x800 = x21*x91;
        const double x801 = x429*x77;
        const double x802 = x169*x404;
        const double x803 = x429*x79;
        const double x804 = R_l_inv_70*x304;
        const double x805 = R_l_inv_30*x303 + x804;
        const double x806 = a_4*x805;
        const double x807 = R_l_inv_72*x304;
        const double x808 = R_l_inv_32*x303 + x807;
        const double x809 = x136*x808;
        const double x810 = d_3*x808;
        const double x811 = -x810;
        const double x812 = x113*x787;
        const double x813 = -x812;
        const double x814 = x35*x808;
        const double x815 = x122*x752;
        const double x816 = x131*x787;
        const double x817 = -x816;
        const double x818 = d_3*x787;
        const double x819 = x159*x818;
        const double x820 = x12*x394;
        const double x821 = -x820;
        const double x822 = x18*x394;
        const double x823 = -x822;
        const double x824 = x33*x394;
        const double x825 = -x824;
        const double x826 = x10*x214;
        const double x827 = x16*x214;
        const double x828 = x214*x31;
        const double x829 = a_0*x787;
        const double x830 = x41*x829;
        const double x831 = x806 + x809 + x811 + x813 + x814 + x815 + x817 + x819 + x821 + x823 + x825 + x826 + x827 + x828 + x830;
        const double x832 = R_l_inv_33*x303 + R_l_inv_73*x304;
        const double x833 = r_21*x832;
        const double x834 = x70*x805;
        const double x835 = x760*x80;
        const double x836 = x31*x805;
        const double x837 = x119*x760;
        const double x838 = x288*x752;
        const double x839 = d_3*x209;
        const double x840 = x154*x429;
        const double x841 = -x840;
        const double x842 = x154*x94;
        const double x843 = -x842;
        const double x844 = x154*x465;
        const double x845 = -x844;
        const double x846 = x12*x334;
        const double x847 = x18*x334;
        const double x848 = x33*x334;
        const double x849 = x833 + x834 + x835 + x836 + x837 + x838 + x839 + x841 + x843 + x845 + x846 + x847 + x848;
        const double x850 = x175*x767;
        const double x851 = x179*x752;
        const double x852 = x181*x787;
        const double x853 = 2*x33;
        const double x854 = x23*x853;
        const double x855 = x27*x853;
        const double x856 = x465*x75;
        const double x857 = a_0*r_22;
        const double x858 = x48*x857;
        const double x859 = x51*x857;
        const double x860 = x54*x857;
        const double x861 = x25*x853;
        const double x862 = x33*x89;
        const double x863 = x33*x424;
        const double x864 = x20*x201;
        const double x865 = x31*x418;
        const double x866 = x147*x404;
        const double x867 = x19*x853;
        const double x868 = x465*x79;
        const double x869 = x465*x77;
        const double x870 = r_21*x502*x56;
        const double x871 = x36*x457;
        const double x872 = -x850 + x851 - x852 - x854 - x855 - x856 + x858 + x859 + x860 + x861 + x862 - x863 - x864 + x865 + x866 + x867 + x868 + x869 + x870 + x871;
        const double x873 = R_l_inv_36*x471 + 2*x759;
        const double x874 = x173*x873;
        const double x875 = x70*(R_l_inv_32*x471 + 2*x807);
        const double x876 = 2*x832;
        const double x877 = r_23*x876;
        const double x878 = x206*x818;
        const double x879 = a_0*x805;
        const double x880 = x3*x879;
        const double x881 = x334*x752;
        const double x882 = -x171*x681;
        const double x883 = -x172*x681;
        const double x884 = x154*x5;
        const double x885 = Px*x884;
        const double x886 = d_3*x505;
        const double x887 = x16*x886;
        const double x888 = x874 - x875 + x877 - x878 + x880 + x881 + x882 + x883 + x885 + x887;
        const double x889 = x216*x808;
        const double x890 = x214*x760;
        const double x891 = x68*x681;
        const double x892 = x65*x829;
        const double x893 = -x889 + x890 + x891 - x892;
        const double x894 = R_l_inv_30*x471 + 2*x804;
        const double x895 = x136*x894;
        const double x896 = x478*x760;
        const double x897 = a_0*x884;
        const double x898 = -4*a_0*d_3*r_12*r_21 - 4*a_4*d_3*x787 - 2*a_4*x808 + x895 + x896 + x897;
        const double x899 = -x748;
        const double x900 = -x749;
        const double x901 = -x750;
        const double x902 = -x753;
        const double x903 = -x761;
        const double x904 = -x762;
        const double x905 = -x763;
        const double x906 = -x764;
        const double x907 = -x765;
        const double x908 = -x775;
        const double x909 = -x777;
        const double x910 = -x778;
        const double x911 = -x795;
        const double x912 = -x796;
        const double x913 = -2*a_0*a_4*r_11*r_22;
        const double x914 = -x802;
        const double x915 = -x803;
        const double x916 = x755 + x757 + x758 + x768 + x769 + x770 + x771 + x772 + x773 + x774 + x779 + x780 + x781 + x782 + x783 + x784 + x785 + x786 + x788 + x791 + x792 + x793 + x794 + x797 + x798 + x799 + x800 + x801 + x810 + x812 + x814 + x824 + x830 + x899 + x900 + x901 + x902 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 + x913 + x914 + x915;
        const double x917 = x806 + x809 + x819;
        const double x918 = -x815 + x816 + x820 + x822 - x826 - x827;
        const double x919 = x917 + x918;
        const double x920 = x850 - x851 + x852 + x854 + x855 + x856 - x858 - x859 - x860 - x861 - x862 + x863 + x864 - x865 - x866 - x867 - x868 - x869 - x870 - x871;
        const double x921 = x849 + x920;
        const double x922 = R_l_inv_37*x471 + 2*x751;
        const double x923 = -x247*x922;
        const double x924 = -x239*x84;
        const double x925 = -x239*x85;
        const double x926 = -x239*x86;
        const double x927 = x239*x754;
        const double x928 = std::pow(r_12, 3);
        const double x929 = x622*x928;
        const double x930 = -x249*x767;
        const double x931 = -x679*x760;
        const double x932 = -x100*x239;
        const double x933 = -x102*x239;
        const double x934 = -x105*x239;
        const double x935 = -x108*x239;
        const double x936 = -x110*x239;
        const double x937 = -x112*x239;
        const double x938 = x234*x808;
        const double x939 = x239*x96;
        const double x940 = x239*x98;
        const double x941 = x101*x239;
        const double x942 = x107*x239;
        const double x943 = x111*x239;
        const double x944 = -x56*x640;
        const double x945 = -x18*x640;
        const double x946 = x236*x787;
        const double x947 = -x8*x946;
        const double x948 = x457*x47;
        const double x949 = x457*x50;
        const double x950 = x457*x53;
        const double x951 = x47*x500;
        const double x952 = x50*x500;
        const double x953 = x500*x53;
        const double x954 = x695*x9;
        const double x955 = x20*x281;
        const double x956 = x6*x946;
        const double x957 = r_22*x5;
        const double x958 = x721*x957;
        const double x959 = x722*x957;
        const double x960 = x414*x5;
        const double x961 = x117*x659;
        const double x962 = x117*x568;
        const double x963 = x277*x585;
        const double x964 = -x273*x646;
        const double x965 = -x20*x284;
        const double x966 = x663*x69;
        const double x967 = x56*x667;
        const double x968 = x500*x57;
        const double x969 = x129*x457;
        const double x970 = x128*x500;
        const double x971 = x923 + x924 + x925 + x926 + x927 + x929 + x930 + x931 + x932 + x933 + x934 + x935 + x936 + x937 + x938 + x939 + x940 + x941 + x942 + x943 + x944 + x945 + x947 + x948 + x949 + x950 + x951 + x952 + x953 + x954 + x955 + x956 + x958 + x959 + x960 + x961 + x962 + x963 + x964 + x965 + x966 + x967 + x968 + x969 + x970;
        const double x972 = x650*x68;
        const double x973 = x18*x698;
        const double x974 = -4*Py*a_4*r_12*r_21 - 4*Pz*a_4*r_12*r_31 - 4*a_0*x28*x787 - 2*a_4*r_13*x752 + x972 + x973;
        const double x975 = x267*x873;
        const double x976 = x260*x832;
        const double x977 = d_3*x281;
        const double x978 = x270*x752;
        const double x979 = x1*x879;
        const double x980 = d_3*x282;
        const double x981 = d_3*x283;
        const double x982 = x6*x695;
        const double x983 = 4*x251;
        const double x984 = x20*x983;
        const double x985 = -x975 - x976 - x977 - x978 + x979 - x980 - x981 + x982 + x984;
        const double x986 = R_l_inv_36*x716 + 4*x759;
        const double x987 = x277*x832;
        const double x988 = x280*x752;
        const double x989 = d_3*x727;
        const double x990 = x69*x989;
        const double x991 = x16*x989;
        const double x992 = a_4*x989 + x285*x808 + x287*x760 + x289*x829;
        const double x993 = x975 + x976 + x977 + x978 - x979 + x980 + x981 - x982 - x984;
        const double x994 = x923 + x924 + x925 + x926 + x927 + x929 + x930 + x931 + x932 + x933 + x934 + x935 + x936 + x937 - x938 + x939 + x940 + x941 + x942 + x943 + x944 + x945 + x947 + x948 + x949 + x950 + x951 + x952 + x953 + x954 + x955 - x956 + x958 + x959 + x960 + x961 + x962 + x963 + x964 + x965 + x966 + x967 + x968 + x969 + x970;
        const double x995 = -x833;
        const double x996 = -x835;
        const double x997 = -x836;
        const double x998 = -x838;
        const double x999 = -x839;
        const double x1000 = -x846;
        const double x1001 = -x847;
        const double x1002 = x1000 + x1001 + x834 + x837 + x840 + x842 + x845 + x848 + x872 + x995 + x996 + x997 + x998 + x999;
        const double x1003 = x755 + x757 + x758 + x768 + x769 + x770 + x771 + x772 + x773 + x774 + x779 + x780 + x781 + x782 + x783 + x784 + x785 + x786 + x788 + x791 + x792 + x793 + x794 + x797 + x798 + x799 + x800 + x801 - x814 - x830 + x899 + x900 + x901 + x902 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 + x914 + x915;
        const double x1004 = x811 + x813 + x825 + x828;
        const double x1005 = x1003 + x1004;
        const double x1006 = x152*x681 + x204*x818 + x43*x808 - x895 - x896 - x897;
        const double x1007 = x874 + x875 + x877 + x878 + x880 + x881 + x882 + x883 + x885 + x887;
        const double x1008 = -x806 - x809 - x819;
        const double x1009 = -x834;
        const double x1010 = -x837;
        const double x1011 = -x848;
        const double x1012 = x1009 + x1010 + x1011 + x833 + x835 + x836 + x838 + x839 + x841 + x843 + x844 + x846 + x847 + x872;
        const double x1013 = x755 + x757 + x758 + x768 + x769 + x770 + x771 + x772 + x773 + x774 + x779 + x780 + x781 + x782 + x783 + x784 + x785 + x786 + x788 + x791 + x792 + x793 + x794 + x797 + x798 + x799 + x800 + x801 + x899 + x900 + x901 + x902 + x903 + x904 + x905 + x906 + x907 + x908 + x909 + x910 + x911 + x912 + x914 + x915;
        const double x1014 = x6*x90;
        const double x1015 = -x1014;
        const double x1016 = 2*d_3;
        const double x1017 = x1016*x23;
        const double x1018 = x1016*x25;
        const double x1019 = x1016*x27;
        const double x1020 = a_4*x38;
        const double x1021 = -2*a_0*d_3*r_11*r_21;
        const double x1022 = -2*a_0*d_3*r_12*r_22;
        const double x1023 = -2*a_0*d_3*r_13*r_23;
        const double x1024 = x10*x38;
        const double x1025 = x154*x91;
        const double x1026 = x429*x6;
        const double x1027 = x16*x38;
        const double x1028 = x154*x93;
        const double x1029 = x20*x334;
        const double x1030 = x1015 + x1017 + x1018 + x1019 + x1020 + x1021 + x1022 + x1023 + x1024 + x1025 + x1026 + x1027 + x1028 + x1029;
        const double x1031 = d_5*x394;
        const double x1032 = x14*x394;
        const double x1033 = -x1032;
        const double x1034 = x20*x394;
        const double x1035 = -x1034;
        const double x1036 = x35*x394;
        const double x1037 = -x1036;
        const double x1038 = x140*x43;
        const double x1039 = x141*x43;
        const double x1040 = a_0*x177*x43;
        const double x1041 = x1031 + x1033 + x1035 + x1037 + x1038 + x1039 + x1040;
        const double x1042 = x23*x465;
        const double x1043 = x25*x465;
        const double x1044 = x242*x448;
        const double x1045 = x209*x33;
        const double x1046 = x465*x9;
        const double x1047 = a_0*r_23;
        const double x1048 = x1047*x48;
        const double x1049 = x1047*x51;
        const double x1050 = x1047*x54;
        const double x1051 = x27*x465;
        const double x1052 = x35*x424;
        const double x1053 = x19*x465;
        const double x1054 = x31*x421;
        const double x1055 = x16*x468;
        const double x1056 = x34*x94;
        const double x1057 = x18*x201;
        const double x1058 = x21*x465;
        const double x1059 = r_23*x5;
        const double x1060 = a_0*x1059*x69;
        const double x1061 = x200*x457;
        const double x1062 = -x1042 - x1043 - x1044 - x1045 - x1046 + x1048 + x1049 + x1050 + x1051 - x1052 - x1053 + x1054 + x1055 + x1056 + x1057 + x1058 + x1060 + x1061;
        const double x1063 = x23*x90;
        const double x1064 = x25*x90;
        const double x1065 = x27*x90;
        const double x1066 = x185*x566;
        const double x1067 = x109*x185;
        const double x1068 = x1067*x49;
        const double x1069 = x1067*x52;
        const double x1070 = x23*x429;
        const double x1071 = x25*x429;
        const double x1072 = x27*x429;
        const double x1073 = x23*x94;
        const double x1074 = x25*x94;
        const double x1075 = x27*x94;
        const double x1076 = r_13*x95;
        const double x1077 = x1076*x790;
        const double x1078 = x194*x789;
        const double x1079 = x1*x416;
        const double x1080 = x412*x585;
        const double x1081 = x10*x402;
        const double x1082 = x12*x209;
        const double x1083 = x429*x9;
        const double x1084 = x16*x402;
        const double x1085 = x18*x209;
        const double x1086 = x20*x89;
        const double x1087 = x10*x421;
        const double x1088 = x13*x94;
        const double x1089 = x14*x424;
        const double x1090 = x19*x429;
        const double x1091 = x21*x429;
        const double x1092 = x140*x404;
        const double x1093 = x18*x212;
        const double x1094 = x1063 + x1064 + x1065 + x1066 + x1068 + x1069 - x1070 - x1071 - x1072 - x1073 - x1074 - x1075 - x1077 - x1078 - x1079 - x1080 + x1081 + x1082 + x1083 + x1084 + x1085 + x1086 - x1087 - x1088 - x1089 - x1090 - x1091 + x1092 + x1093 - x527 - x530 - x532 + x534 - x536 - x540 - x542 - x544 - x546 - x548 - x550 + x555 + x557 + x559 + x561;
        const double x1095 = x1062 + x1094;
        const double x1096 = x1*x84;
        const double x1097 = x1*x85;
        const double x1098 = x502*x8;
        const double x1099 = x487*x68;
        const double x1100 = x204*x36;
        const double x1101 = x109*x583;
        const double x1102 = x5*x9;
        const double x1103 = 4*x142;
        const double x1104 = x16*x505;
        const double x1105 = -Px*x1102 + r_12*r_22*x579 + x1*x101 + x1*x102 + x1*x105 - x1*x107 - x1*x108 + x1*x111 + x1*x112 - x1*x86 + x1*x96 - x1*x97 - x1*x98 + x1059*x1076 - x1101*x49 - x1101*x52 - x1103*x20 + x1104*x12 + x14*x591 + x15*x646 + x17*x595 - x171*x500 + x19*x595 + x20*x591 + x21*x595 + x23*x595 + x23*x646 + x25*x595 + x25*x646 + x27*x595 + x27*x646 + x278*x416 + 2*x348 + 2*x349 + x414*x505 - x566*x583 - x646*x9;
        const double x1106 = x1096 + x1097 - x1098 - x1099 - x1100 + x1105;
        const double x1107 = x204*x9;
        const double x1108 = x204*x23;
        const double x1109 = x204*x25;
        const double x1110 = x204*x27;
        const double x1111 = x595*x8;
        const double x1112 = 4*x12;
        const double x1113 = x1112*x68;
        const double x1114 = x15*x204;
        const double x1115 = x16*x483;
        const double x1116 = x500*x68;
        const double x1117 = x204*x21;
        const double x1118 = -x1107 + x1108 + x1109 + x1110 + x1111 + x1113 + x1114 + x1115 + x1116 + x1117;
        const double x1119 = x285*x47;
        const double x1120 = x285*x50;
        const double x1121 = x285*x53;
        const double x1122 = x23*x502;
        const double x1123 = a_0*x1102;
        const double x1124 = x25*x502;
        const double x1125 = x27*x502;
        const double x1126 = x35*x591;
        const double x1127 = x152*x512;
        const double x1128 = 8*x69;
        const double x1129 = x1128*x36;
        const double x1130 = x17*x502;
        const double x1131 = x152*x500;
        const double x1132 = x1104*x33;
        const double x1133 = x20*x497;
        const double x1134 = x36*x646;
        const double x1135 = x19*x502;
        const double x1136 = x21*x502;
        const double x1137 = -x1119 - x1120 - x1121 - x1122 - x1123 + x1124 + x1125 + x1126 - x1127 - x1129 - x1130 - x1131 - x1132 - x1133 - x1134 + x1135 + x1136;
        const double x1138 = -x1031;
        const double x1139 = -x1038;
        const double x1140 = -x1039;
        const double x1141 = -x1040;
        const double x1142 = x1032 + x1034 + x1036 + x1138 + x1139 + x1140 + x1141;
        const double x1143 = x1042 + x1043 + x1044 + x1045 + x1046 - x1048 - x1049 - x1050 - x1051 + x1052 + x1053 - x1054 - x1055 - x1056 - x1057 - x1058 - x1060 - x1061;
        const double x1144 = -x1063 - x1064 - x1065 - x1066 - x1068 - x1069 + x1070 + x1071 + x1072 + x1073 + x1074 + x1075 + x1077 + x1078 + x1079 + x1080 - x1081 - x1082 - x1083 - x1084 - x1085 - x1086 + x1087 + x1088 + x1089 + x1090 + x1091 - x1092 - x1093 + x527 + x530 + x532 - x534 + x536 + x540 + x542 + x544 + x546 + x548 + x550 - x555 - x557 - x559 - x561;
        const double x1145 = x1143 + x1144;
        const double x1146 = x3*x85;
        const double x1147 = x3*x86;
        const double x1148 = x622*x718;
        const double x1149 = x3*x84;
        const double x1150 = x3*x96;
        const double x1151 = x3*x97;
        const double x1152 = x102*x3;
        const double x1153 = x108*x3;
        const double x1154 = x112*x3;
        const double x1155 = x100*x3;
        const double x1156 = x101*x3;
        const double x1157 = x105*x3;
        const double x1158 = x107*x3;
        const double x1159 = x110*x3;
        const double x1160 = x111*x3;
        const double x1161 = x47*x517;
        const double x1162 = x50*x517;
        const double x1163 = x517*x53;
        const double x1164 = 4*x58;
        const double x1165 = x1164*x47;
        const double x1166 = x1164*x50;
        const double x1167 = x1164*x53;
        const double x1168 = x47*x575;
        const double x1169 = x50*x575;
        const double x1170 = x53*x575;
        const double x1171 = x1059*x721;
        const double x1172 = x1059*x722;
        const double x1173 = x416*x5;
        const double x1174 = x116*x654;
        const double x1175 = x116*x656;
        const double x1176 = x286*x585;
        const double x1177 = x652*x69;
        const double x1178 = x281*x56;
        const double x1179 = Py*x9;
        const double x1180 = x1179*x277;
        const double x1181 = x16*x652;
        const double x1182 = x18*x281;
        const double x1183 = x126*x575;
        const double x1184 = x665*x69;
        const double x1185 = x58*x667;
        const double x1186 = x57*x575;
        const double x1187 = x1164*x61;
        const double x1188 = x128*x575;
        const double x1189 = x243*x646;
        const double x1190 = x18*x282;
        const double x1191 = a_4*x65;
        const double x1192 = d_5*x698;
        const double x1193 = x58*x698;
        const double x1194 = -x1193;
        const double x1195 = x20*x698;
        const double x1196 = -x1195;
        const double x1197 = x277*x708;
        const double x1198 = x204*x244;
        const double x1199 = x1191 + x1192 + x1194 + x1196 + x1197 + x1198;
        const double x1200 = x47*x681;
        const double x1201 = x50*x681;
        const double x1202 = x53*x681;
        const double x1203 = x65*x69;
        const double x1204 = x56*x983;
        const double x1205 = Py*x6;
        const double x1206 = x1205*x277;
        const double x1207 = x16*x65;
        const double x1208 = x18*x983;
        const double x1209 = x129*x681;
        const double x1210 = -4*d_3*d_5*r_13 + x1200 + x1201 + x1202 + x1203 + x1204 + x1206 + x1207 + x1208 + x1209;
        const double x1211 = x126*x711;
        const double x1212 = Py*x68;
        const double x1213 = x5*x84;
        const double x1214 = x5*x85;
        const double x1215 = 8*x126;
        const double x1216 = 8*x16;
        const double x1217 = 8*r_23;
        const double x1218 = x1217*x178;
        const double x1219 = 8*x18;
        const double x1220 = 8*x245;
        const double x1221 = x100*x5 - x101*x5 - x102*x5 + x105*x5 - x107*x5 - x108*x5 + x109*x1218 + x110*x5 - x111*x5 - x112*x5 + x1128*x129 + x1128*x47 + x1128*x50 + x1128*x53 + x1179*x723 - x1215*x16 - x1215*x69 + x1216*x127 + x1216*x128 + x1216*x47 + x1216*x50 + x1216*x53 + x1218*x99 - x1219*x227 - x1220*x20 + x151*x725 + x151*x726 + x16*x728 + x20*x704 + x414*x727 + x416*x709 - x5*x86 + x5*x97 + x5*x98 + x581*x621 + x69*x729;
        const double x1222 = x1213 + x1214 + x1221;
        const double x1223 = x1146 + x1147 + x1148 - x1149 + x1150 + x1151 + x1152 + x1153 + x1154 - x1155 - x1156 - x1157 - x1158 - x1159 - x1160 - x1161 - x1162 - x1163 + x1165 + x1166 + x1167 + x1168 + x1169 + x1170 + x1171 + x1172 + x1173 + x1174 + x1175 + x1176 - x1177 - x1178 - x1180 - x1181 - x1182 - x1183 + x1184 + x1185 + x1186 + x1187 + x1188 - x1189 - x1190;
        const double x1224 = x1210 + x1223;
        const double x1225 = x1020 + x1031 + x1033 + x1035 + x1036 + x1038 + x1039 + x1141;
        const double x1226 = x1015 + x1017 + x1018 + x1019 + x1024 + x1025 + x1026 + x1027 + x1028 + x1029;
        const double x1227 = x1226 + x154*x853 + x31*x38 + x465*x6;
        const double x1228 = x1096 + x1097 + x1098 + x1099 + x1100 + x1105;
        const double x1229 = x1119 + x1120 + x1121 + x1122 + x1123 - x1124 - x1125 - x1126 + x1127 + x1129 + x1130 + x1131 + x1132 + x1133 + x1134 - x1135 - x1136;
        const double x1230 = x1062 + x1144;
        const double x1231 = x1227 + x1230;
        const double x1232 = -x214;
        const double x1233 = 2*x71;
        const double x1234 = x1233*x175;
        const double x1235 = x179*x187;
        const double x1236 = x156*x198;
        const double x1237 = x1*x148;
        const double x1238 = x191*x80;
        const double x1239 = -4*R_l_inv_55*a_1*a_4*x175;
        const double x1240 = x38*x81;
        const double x1241 = -x1240;
        const double x1242 = x156*x216;
        const double x1243 = x119*x191;
        const double x1244 = x204*x33;
        const double x1245 = x1244*x125;
        const double x1246 = -x1245;
        const double x1247 = x1236 + x1237 + x1238 + x1239 + x1241 + x1242 + x1243 + x1246;
        const double x1248 = x144*x187;
        const double x1249 = x184*x412;
        const double x1250 = x1233*x33;
        const double x1251 = x154*x191;
        const double x1252 = -x1248 + x1249 + x1250 + x1251 - x394;
        const double x1253 = Px*x1059;
        const double x1254 = 4*x141;
        const double x1255 = 4*x143;
        const double x1256 = 4*x137;
        const double x1257 = x1256*x33;
        const double x1258 = x275*x8;
        const double x1259 = x167*x989;
        const double x1260 = 8*x175;
        const double x1261 = -4*a_0*r_11*r_23 + x1256*x175 + x1260*x160 + x497;
        const double x1262 = -x1234 + x1235;
        const double x1263 = x148*x5;
        const double x1264 = 4*x121;
        const double x1265 = x1264*x267;
        const double x1266 = 8*x68;
        const double x1267 = x1266*x167;
        const double x1268 = -x1264*x251 - x184*x286 + x189*x505 + x247*x275 + x698;
        const double x1269 = 8*x243;
        const double x1270 = 8*x246;
        const double x1271 = 8*x244;
        const double x1272 = a_0*x724;
        const double x1273 = r_11*x711;
        const double x1274 = 16*x154;
        const double x1275 = x175*x204;
        const double x1276 = x1252 + x1262;
        const double x1277 = x144*x525;
        const double x1278 = x70*(R_l_inv_21*x471 + 2*x310);
        const double x1279 = -2*x749;
        const double x1280 = -2*x750;
        const double x1281 = -r_22*x538;
        const double x1282 = 2*x748;
        const double x1283 = 2*x757;
        const double x1284 = 2*x758;
        const double x1285 = -x505*x566;
        const double x1286 = -x49*x656;
        const double x1287 = -x52*x656;
        const double x1288 = -x351*x853;
        const double x1289 = 2*x316;
        const double x1290 = x119*x1289;
        const double x1291 = -x1289*x154;
        const double x1292 = -2*x762;
        const double x1293 = -2*x763;
        const double x1294 = -2*x764;
        const double x1295 = -2*x765;
        const double x1296 = 2*x769;
        const double x1297 = 2*x770;
        const double x1298 = 2*x771;
        const double x1299 = 2*x772;
        const double x1300 = 2*x773;
        const double x1301 = 2*x774;
        const double x1302 = -x1112*x9;
        const double x1303 = -x500*x9;
        const double x1304 = x487*x6;
        const double x1305 = x1112*x23;
        const double x1306 = x1112*x25;
        const double x1307 = x1112*x27;
        const double x1308 = x571*x75;
        const double x1309 = x23*x500;
        const double x1310 = x25*x500;
        const double x1311 = x27*x500;
        const double x1312 = x575*x75;
        const double x1313 = x151*x579;
        const double x1314 = x116*x277*x95;
        const double x1315 = x414*x583;
        const double x1316 = x278*x585;
        const double x1317 = -x170*x595;
        const double x1318 = -x14*x506;
        const double x1319 = x172*x595;
        const double x1320 = x1112*x17;
        const double x1321 = x12*x599;
        const double x1322 = x1112*x21;
        const double x1323 = x15*x500;
        const double x1324 = x1275*x313 + x1277 - x1278 + x1279 + x1280 + x1281 + x1282 + x1283 + x1284 + x1285 + x1286 + x1287 + x1288 - x1290 + x1291 + x1292 + x1293 + x1294 + x1295 + x1296 + x1297 + x1298 + x1299 + x1300 + x1301 + x1302 + x1303 - x1304 + x1305 + x1306 + x1307 + x1308 + x1309 + x1310 + x1311 + x1312 + x1313 + x1314 + x1315 + x1316 + x1317 + x1318 + x1319 + x1320 + x1321 + x1322 + x1323 + x499*x681;
        const double x1325 = x674*x80;
        const double x1326 = x1*x308;
        const double x1327 = x681*x75;
        const double x1328 = x216*x311;
        const double x1329 = x325*x38;
        const double x1330 = x1112*x6;
        const double x1331 = x500*x6;
        const double x1332 = x14*x886;
        const double x1333 = x20*x886;
        const double x1334 = x1244*x313;
        const double x1335 = -x1325 - x1326 - x1327 - x1328 + x1329 - x1330 - x1331 + x1332 + x1333 + x1334;
        const double x1336 = x175*x472;
        const double x1337 = 2*x179;
        const double x1338 = x1337*x325;
        const double x1339 = a_0*x505;
        const double x1340 = x1339*x47;
        const double x1341 = x1339*x50;
        const double x1342 = x1339*x53;
        const double x1343 = x25*x487;
        const double x1344 = x487*x9;
        const double x1345 = x23*x487;
        const double x1346 = x27*x487;
        const double x1347 = x480*x75;
        const double x1348 = x32*x512;
        const double x1349 = x36*x512;
        const double x1350 = x172*x502;
        const double x1351 = x170*x502;
        const double x1352 = x33*x599;
        const double x1353 = x499*x575;
        const double x1354 = x36*x500;
        const double x1355 = x152*x646;
        const double x1356 = x200*x575;
        const double x1357 = -x1336 + x1338 - x1340 - x1341 - x1342 - x1343 - x1344 + x1345 + x1346 + x1347 - x1348 - x1349 - x1350 - x1351 - x1352 - x1353 - x1354 + x1355 + x1356;
        const double x1358 = 8*d_5*x68;
        const double x1359 = x306*x487;
        const double x1360 = x325*x483;
        const double x1361 = x711*x76;
        const double x1362 = x711*x77;
        const double x1363 = x1266*x14;
        const double x1364 = x68*x720;
        const double x1365 = x437*x989;
        const double x1366 = -8*a_0*a_4*r_12*r_23 + x1260*x381 + x1266*x35 + x175*(R_l_inv_22*x716 + 4*x305);
        const double x1367 = x1325 + x1326 + x1327 + x1328 - x1329 + x1330 + x1331 - x1332 - x1333 - x1334;
        const double x1368 = -4*a_0*d_3*r_13*r_22 - 4*a_4*x175*x313 + x1277 + x1278 + x1279 + x1280 + x1281 + x1282 + x1283 + x1284 + x1285 + x1286 + x1287 + x1288 + x1290 + x1291 + x1292 + x1293 + x1294 + x1295 + x1296 + x1297 + x1298 + x1299 + x1300 + x1301 + x1302 + x1303 + x1304 + x1305 + x1306 + x1307 + x1308 + x1309 + x1310 + x1311 + x1312 + x1313 + x1314 + x1315 + x1316 + x1317 + x1318 + x1319 + x1320 + x1321 + x1322 + x1323;
        const double x1369 = x267*(R_l_inv_27*x716 + 4*x315);
        const double x1370 = x308*x5;
        const double x1371 = 8*d_3;
        const double x1372 = x1371*x262;
        const double x1373 = x1371*x263;
        const double x1374 = x1371*x264;
        const double x1375 = x1266*x437;
        const double x1376 = 8*x146;
        const double x1377 = x117*x1217;
        const double x1378 = -x100*x286 + x101*x286 - x102*x286 - x105*x286 + x107*x286 - x108*x286 + x109*x1377 - x110*x286 + x111*x286 - x112*x286 + x1179*x727 - x1215*x18 - x1215*x56 + x1219*x127 + x1219*x128 + x1219*x47 + x1219*x50 + x1219*x53 + x129*x512 + x1376*x721 + x1376*x722 + x1377*x99 - x16*x69*x727 - x20*x58*x727 + x247*x717 + x262*x720 - x286*x357 + x286*x84 - x286*x85 - x286*x86 + x286*x96 + x286*x98 - x316*x983 + x414*x723 + x47*x512 + x50*x512 + x505*x552 + x512*x53 + x512*x60 + x56*x729 + x581*x928 + x585*x709;
        const double x1379 = 16*a_4;
        const double x1380 = 16*r_13;
        const double x1381 = x1336 - x1338 + x1340 + x1341 + x1342 + x1343 + x1344 - x1345 - x1346 - x1347 + x1348 + x1349 + x1350 + x1351 + x1352 + x1353 + x1354 - x1355 - x1356;
        const double x1382 = R_l_inv_31*x471 + 2*x766;
        const double x1383 = x80*x922;
        const double x1384 = -x1096;
        const double x1385 = -x1097;
        const double x1386 = x1*x754;
        const double x1387 = x1275*x787;
        const double x1388 = -x38*x760;
        const double x1389 = x216*x767;
        const double x1390 = 2*x752;
        const double x1391 = -x1244*x787;
        const double x1392 = x1105 + x1137 + x119*x1390 + x1382*x70 + x1383 + x1384 + x1385 + x1386 - x1387 + x1388 + x1389 + x1391;
        const double x1393 = x175*x894;
        const double x1394 = x1337*x760;
        const double x1395 = x502*x6;
        const double x1396 = d_3*x497;
        const double x1397 = x1393 - x1394 - x1395 + x1396;
        const double x1398 = x144*x873;
        const double x1399 = r_22*x876;
        const double x1400 = d_3*x591;
        const double x1401 = x805*x853;
        const double x1402 = x1390*x154;
        const double x1403 = x595*x6;
        const double x1404 = x6*x646;
        const double x1405 = x142*x681;
        const double x1406 = x143*x681;
        const double x1407 = -x1398 + x1399 - x1400 + x1401 + x1402 - x1403 - x1404 + x1405 + x1406;
        const double x1408 = R_l_inv_32*x716 + 4*x807;
        const double x1409 = x1371*x8 + x483*x760 + x487*x808 + x829*x989;
        const double x1410 = -x1393 + x1394 + x1395 - x1396;
        const double x1411 = x1398 - x1399 + x1400 - x1401 - x1402 + x1403 + x1404 - x1405 - x1406;
        const double x1412 = x247*x986;
        const double x1413 = x286*x832;
        const double x1414 = d_3*x704;
        const double x1415 = x505*x879;
        const double x1416 = x752*x983;
        const double x1417 = d_3*x1269;
        const double x1418 = d_3*x1271;
        const double x1419 = x1205*x723;
        const double x1420 = d_3*x1270;
        const double x1421 = -x1213 - x1214 + x1221 + x1266*x829 + x267*(R_l_inv_37*x716 + 4*x751) - x285*x767 + x5*x754 - x65*x760;
        const double x1422 = -2*a_0*x118*x752 + x1105 + x1229 - x1382*x70 + x1383 + x1384 + x1385 + x1386 + x1387 + x1388 + x1389 + x1391;
        const double x1423 = x154 + x80;
        const double x1424 = x200 - x499;
        const double x1425 = x1272*x47;
        const double x1426 = x1272*x50;
        const double x1427 = x1272*x53;
        const double x1428 = a_0*x727;
        const double x1429 = x1428*x25;
        const double x1430 = x1428*x9;
        const double x1431 = 8*x75;
        const double x1432 = 16*x56;
        const double x1433 = x1432*x32;
        const double x1434 = x1432*x36;
        const double x1435 = x1219*x32;
        const double x1436 = x1216*x147;
        const double x1437 = x1428*x19;
        const double x1438 = x499*x720;
        const double x1439 = x1219*x36;
        const double x1440 = 8*x12;
        const double x1441 = 8*x95;
        const double x1442 = r_11*x1441*x151 + r_13*x116*x1441 + 8*r_21*x414 + x100*x505 + x102*x505 - x105*x505 + x107*x505 - x108*x505 + x11*x1219 + x110*x505 + x112*x505 - x1216*x169 + x1217*x585 + x1219*x13 + x1219*x15 + x1219*x23 + x1219*x25 + x1219*x27 - x1219*x9 + x14*x1431 + x1440*x17 + x1440*x21 + x1440*x23 + x1440*x25 + x1440*x27 - x1440*x9 - x49*x726 - x505*x96 + x505*x97 - x505*x98 - x52*x726 - x566*x724 + x720*x75 - x720*x78 + 4*x748 + 4*x749 - 4*x750 + 4*x757 + 4*x758;
        const double x1443 = -x154 + x80;
        const double x1444 = 16*x126;
        const double x1445 = 16*x20;
        const double x1446 = 16*x18;
        const double x1447 = 16*x146;
        const double x1448 = 16*r_23*x117;
        const double x1449 = -x200 + x499;
        const double x1450 = x28 - x8;
        const double x1451 = x1450 + x37;
        const double x1452 = -x44 + x63;
        const double x1453 = x1450 + x66;
        const double x1454 = x183 + x300;
        const double x1455 = x219 + x299;
        const double x1456 = x272 + x291;
        const double x1457 = x346 + x348 + x349 + x354 + x356 + x359 + x362 + x363 + x364 + x365 + x366 + x367 + x368 + x370 + x372 + x374 + x376 + x378 + x380 + x385 + x387 + x389 + x391 + x393 + x399 + x400 + x401 + x403 + x405 + x406 + x407 + x408 + x409 + x411 + x415 + x417 + x420 + x423 + x425 + x426 + x427 + x428 + x430 + x470 + x614;
        const double x1458 = x473 + x474 + x475 + x476 + x477 + x479 + x481 + x482 + x484 + x486 + x488 + x489 + x491 + x492 + x493 + x494 - x496 + x498 + x501 + x503 + x504 + x507 + x508 + x509 + x511 + x513;
        const double x1459 = x689 - x690 - x691 - x692 - x693 - x694 - x696 - x697 - x699 - x700 - x701 - x702;
        const double x1460 = x431 + x432 + x433 + x438 + x605 + x606 + x607 + x608 + x609 + x610 + x611 + x612 + x613;
        const double x1461 = x889 - x890 - x891 + x892;
        const double x1462 = x815 + x817 + x821 + x823 + x826 + x827;
        const double x1463 = x1008 + x1462;
        const double x1464 = x204*x232 + x253*x752 + x255*x787 + x695*x8 - x972 - x973;
        const double x1465 = -x1020;
        const double x1466 = x1021 + x1022 + x1023;
        const double x1467 = x1226 + x1465 + x1466;
        const double x1468 = x1107 - x1108 - x1109 - x1110 - x1111 - x1113 - x1114 - x1115 - x1116 - x1117;
        
        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = x0;
        A(0, 1) = x2;
        A(0, 2) = r_23;
        A(0, 3) = x4;
        A(0, 4) = -x5;
        A(0, 5) = x3;
        A(0, 6) = r_23;
        A(0, 7) = x1;
        A(0, 8) = x0;
        A(1, 0) = x30 + x37;
        A(1, 1) = x39;
        A(1, 2) = x37 + x40;
        A(1, 3) = x42 + x64;
        A(1, 4) = -x65;
        A(1, 5) = x41 + x64;
        A(1, 6) = -x30 - x66;
        A(1, 7) = x38;
        A(1, 8) = -x40 - x66;
        A(2, 0) = -x134 - x165 - x174 - x183;
        A(2, 1) = x186 + x188 + x190 + x192 - x208 - x219;
        A(2, 2) = x174 + x223 + x224 + x226;
        A(2, 3) = x228 + x230 - x231 - x233 + x235 + x238 - x258 - x272;
        A(2, 4) = x121*x280 + x184*x277 - x276 - x279 + x290;
        A(2, 5) = -x258 + x259 + x261 + x268 - x269 + x271 - x291;
        A(2, 6) = x174 + x294 + x296 + x298;
        A(2, 7) = -x208 + x209 - x210 - x211 + x212 + x213 - x215 + x217 + x218 - x299;
        A(2, 8) = -x115 - x124 + x223 + x295 + x301 - x68 - x74 - x82;
        A(3, 0) = x345 + x431 + x445 + x470;
        A(3, 1) = x514 + x524 + x604;
        A(3, 2) = x345 - x346 - x348 - x349 + x353 + x355 + x358 + x361 - x363 - x364 - x365 - x366 - x367 - x368 + x369 + x371 + x373 + x375 + x377 + x379 + x384 + x386 + x388 + x390 + x392 - x399 - x400 - x401 - x403 - x405 - x406 - x407 - x408 - x409 - x411 - x415 - x417 + x419 + x422 - x425 - x426 - x427 - x428 - x430 + x614 + x615;
        A(3, 3) = x673 + x688 + x703;
        A(3, 4) = x705 + x706 + x707 + x710 + x712 + x713 - x714 - x715 - x730;
        A(3, 5) = -x703 - x731 - x732;
        A(3, 6) = -x735 - x739 - x740;
        A(3, 7) = x514 + x741 + x742;
        A(3, 8) = x740 + x744 + x747;
        A(4, 0) = x748 + x749 + x750 + x753 - x755 - x757 - x758 + x761 + x762 + x763 + x764 + x765 - x768 - x769 - x770 - x771 - x772 - x773 - x774 + x775 + x777 + x778 - x779 - x780 - x781 - x782 - x783 - x784 - x785 - x786 - x788 - x791 - x792 - x793 - x794 + x795 + x796 - x797 - x798 - x799 - x800 - x801 + x802 + x803 + x831 + x849 + x872;
        A(4, 1) = -x888 - x893 - x898;
        A(4, 2) = -x916 - x919 - x921;
        A(4, 3) = -x971 - x974 - x985;
        A(4, 4) = 8*Py*d_3*r_11*r_22 + 8*Pz*d_3*r_11*r_32 + 4*a_0*r_23*x805 + x274*x986 - x987 - x988 - x990 - x991 - x992;
        A(4, 5) = -x974 - x993 - x994;
        A(4, 6) = x1002 + x1005 + x919;
        A(4, 7) = x1006 + x1007 + x893;
        A(4, 8) = x1004 + x1008 + x1012 + x1013 + x814 + x830 + x918;
        A(5, 0) = -x1030 - x1041 - x1095;
        A(5, 1) = x1106 + x1118 + x1137;
        A(5, 2) = -x1030 - x1142 - x1145;
        A(5, 3) = x1146 + x1147 + x1148 - x1149 + x1150 + x1151 + x1152 + x1153 + x1154 - x1155 - x1156 - x1157 - x1158 - x1159 - x1160 - x1161 - x1162 - x1163 + x1165 + x1166 + x1167 + x1168 + x1169 + x1170 + x1171 + x1172 + x1173 + x1174 + x1175 + x1176 - x1177 - x1178 - x1180 - x1181 - x1182 - x1183 + x1184 + x1185 + x1186 + x1187 + x1188 - x1189 - x1190 - x1199 - x1210;
        A(5, 4) = a_4*x729 - x1211 + x1212*x727 + x1222 + x128*x711 + x129*x711 + x47*x711 + x50*x711 + x53*x711 + x60*x711 + x708*x723;
        A(5, 5) = -x1191 + x1192 - x1193 - x1195 + x1197 + x1198 - x1224;
        A(5, 6) = x1094 + x1143 + x1225 + x1227;
        A(5, 7) = -x1118 - x1228 - x1229;
        A(5, 8) = x1020 + x1032 + x1034 + x1037 + x1040 + x1138 + x1139 + x1140 + x1231;
        
        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 1) = -x505;
        B(0, 4) = -x727;
        B(0, 7) = x505;
        B(1, 0) = x214;
        B(1, 1) = -x886;
        B(1, 2) = x214;
        B(1, 3) = x287;
        B(1, 4) = -x989;
        B(1, 5) = x287;
        B(1, 6) = x1232;
        B(1, 7) = x886;
        B(1, 8) = x1232;
        B(2, 0) = -x1234 + x1235 - x1247 - x1252;
        B(2, 1) = -x1103 + x1253 + x1254 - x1255 - x1257 - x1258 - x1259 - x1261 + x591;
        B(2, 2) = -x1247 - x1248 + x1249 + x1250 + x1251 - x1262 - x394;
        B(2, 3) = -x1263 - x1265 - x1267 + x1268 + x156*x285 + x65*x81;
        B(2, 4) = -x1220 + x1269 - x1270 + x1271 + x1272*x137 - x1273*x81 + x1274*x167 + x704;
        B(2, 5) = 4*R_l_inv_51*a_0*a_1*r_21 + 4*R_l_inv_56*a_1*d_3*r_11 - x1263 - x1265 - x1267 - x1268;
        B(2, 6) = -x1236 + x1237 + x1238 + x1241 + x1242 - x1243 + x1246 + x125*x1275 + x1276;
        B(2, 7) = x1103 - x1253 - x1254 + x1255 + x1257 + x1258 + x1259 - x1261 - x591;
        B(2, 8) = -x1236 + x1237 + x1238 - x1239 - x1240 + x1242 - x1243 - x1245 - x1276;
        B(3, 0) = x1324 + x1335 + x1357;
        B(3, 1) = x1358 - x1359 - x1360 + x1361 + x1362 - x1363 - x1364 - x1365 - x1366;
        B(3, 2) = -x1357 - x1367 - x1368;
        B(3, 3) = x1205*x727 - x1369 - x1370 - x1372 - x1373 - x1374 - x1375 + x1378 + x20*x989 + x285*x311 + x325*x65;
        B(3, 4) = x1212*x1380 + x1272*x306 - x1273*x325 + x1274*x437 + x1379*x262 + x1379*x264 - x1379*x265 - x1379*x266;
        B(3, 5) = 8*Py*d_3*r_12*r_23 + 8*Pz*d_3*r_12*r_33 + 4*a_0*r_21*x311 + 4*d_3*r_11*x325 - x1369 - x1370 - x1372 - x1373 - x1374 - x1375 - x1378;
        B(3, 6) = -x1335 - x1368 - x1381;
        B(3, 7) = -x1358 + x1359 + x1360 - x1361 - x1362 + x1363 + x1364 + x1365 - x1366;
        B(3, 8) = x1324 + x1367 + x1381;
        B(4, 0) = x1392 + x1397 + x1407;
        B(4, 1) = x1260*x818 + x1408*x175 + x1409;
        B(4, 2) = x1392 + x1410 + x1411;
        B(4, 3) = -x1412 + x1413 - x1414 - x1415 + x1416 - x1417 - x1418 + x1419 + x1420 + x1421;
        B(4, 4) = 16*a_4*d_3*r_11 + 8*a_4*r_11*x760 - x1272*x808 - x1274*x829;
        B(4, 5) = x1412 - x1413 + x1414 + x1415 - x1416 + x1417 + x1418 - x1419 - x1420 + x1421;
        B(4, 6) = -x1407 - x1410 - x1422;
        B(4, 7) = 8*d_3*x175*x787 + x1408*x175 - x1409;
        B(4, 8) = -x1397 - x1411 - x1422;
        B(5, 0) = x204*(-x1423 - x1424);
        B(5, 1) = x1216*x152 - x1425 - x1426 - x1427 + x1428*x23 + x1428*x27 - x1429 - x1430 + x1431*x35 - x1433 - x1434 - x1435 - x1436 - x1437 - x1438 - x1439 + x1442 + x200*x720;
        B(5, 2) = x204*(x1424 + x1443);
        B(5, 3) = x711*(-x251 - x267);
        B(5, 4) = 16*r_11*x414 + 16*r_12*x1179 - x100*x727 + x101*x727 - x102*x727 - x105*x727 + x107*x727 - x108*x727 + x109*x1448 - x110*x727 + x111*x727 - x112*x727 + x127*x1446 + x128*x1446 + x129*x1432 + x1380*x585 + x1432*x47 + x1432*x50 + x1432*x53 + x1432*x60 + x1441*x928 - x1444*x18 - x1444*x56 + x1445*x262 - x1445*x265 + x1446*x47 + x1446*x50 + x1446*x53 + x1446*x57 + x1447*x721 + x1447*x722 + x1448*x99 - 16*x16*x273 + x727*x84 + x727*x85 - x727*x86 + x727*x96 + x727*x98;
        B(5, 5) = x711*(-x251 + x262 + x263 + x264 - x265 - x266);
        B(5, 6) = x204*(x1423 + x1449);
        B(5, 7) = 8*Py*a_0*r_12*x22 + 8*Py*a_0*r_12*x26 + 8*Pz*a_0*r_12*r_21*r_31 + 8*Pz*a_0*r_12*r_23*r_33 + 8*a_0*d_5*r_13*r_22 - x1425 - x1426 - x1427 - x1429 - x1430 - x1433 - x1434 - x1435 - x1436 - x1437 - x1438 - x1439 - x1442;
        B(5, 8) = x204*(-x1443 - x1449);
        
        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = x0;
        C(0, 1) = x1;
        C(0, 2) = r_23;
        C(0, 3) = x4;
        C(0, 4) = x5;
        C(0, 5) = x3;
        C(0, 6) = r_23;
        C(0, 7) = x2;
        C(0, 8) = x0;
        C(1, 0) = x1451 + x7;
        C(1, 1) = x38;
        C(1, 2) = x1451 + x6;
        C(1, 3) = x1452 + x42;
        C(1, 4) = x65;
        C(1, 5) = x1452 + x41;
        C(1, 6) = -x1453 - x7;
        C(1, 7) = x39;
        C(1, 8) = -x1453 - x6;
        C(2, 0) = a_2 + x120 + x1454 + x220 + x221 + x224 + x298 + x72;
        C(2, 1) = a_0*x187*x202 + x1455 + x159*x71 + x196 - x197 + x199 + x201 - x205 + x207;
        C(2, 2) = -x134 - x176 - x182 - x222 - x225 - x297 - x300;
        C(2, 3) = x1456 + x241 + x248 + x250 + x252 - x254 - x256 + x257;
        C(2, 4) = 4*R_l_inv_53*a_1*r_13 + 4*R_l_inv_57*a_1*d_3*r_13 - x276 - x279 - x290;
        C(2, 5) = -x1456 - x240 + x248 + x250 + x252 - x254 - x256 + x257;
        C(2, 6) = -x114 - x224 - x294 - x301 - x67 - x83;
        C(2, 7) = -x1455 - x193 - x195 - x197 + x199 + x201 - x203 - x205 + x207;
        C(2, 8) = x1454 + x226 + x292 + x293 + x296;
        C(3, 0) = -x1457 - x309 - x318 - x319 - x322 - x327 - x328 - x331 - x336 - x338 - x341 - x342 - x343 - x733 - x734 - x743 - x745;
        C(3, 1) = x1458 + x604 + x741;
        C(3, 2) = x1457 + x307 + x314 + x320 + x323 + x329 + x332 + x339 + x736 + x737 + x738 + x746;
        C(3, 3) = -x1459 - x688 - x732;
        C(3, 4) = -x705 - x706 - x707 - x710 - x712 - x713 + x714 + x715 - x730;
        C(3, 5) = x1459 + x673 + x731;
        C(3, 6) = x1460 + x739 + x744;
        C(3, 7) = x1458 + x524 + x742;
        C(3, 8) = -x1460 - x735 - x747;
        C(4, 0) = x1000 + x1001 + x1009 + x1010 + x1011 + x1013 + x831 + x840 + x842 + x844 + x920 + x995 + x996 + x997 + x998 + x999;
        C(4, 1) = -x1007 - x1461 - x898;
        C(4, 2) = x1005 + x1463 + x921;
        C(4, 3) = x1464 + x985 + x994;
        C(4, 4) = x1371*x227 + x1371*x229 + x274*x986 + x278*x879 - x987 - x988 - x990 - x991 + x992;
        C(4, 5) = x1464 + x971 + x993;
        C(4, 6) = -x1002 - x1463 - x916;
        C(4, 7) = x1006 + x1461 + x888;
        C(4, 8) = -x1003 - x1012 - x1462 - x810 - x812 - x824 - x913 - x917;
        C(5, 0) = -x1095 - x1142 - x1467;
        C(5, 1) = -x1137 - x1228 - x1468;
        C(5, 2) = -x1041 - x1145 - x1467;
        C(5, 3) = x1199 - x1200 - x1201 - x1202 - x1203 - x1204 - x1206 - x1207 - x1208 - x1209 + x1223 + x126*x681;
        C(5, 4) = 8*Px*a_4*x46 + 8*Px*a_4*x49 + 8*Px*a_4*x52 + 8*Py*a_4*r_11*r_21 + 8*Py*a_4*r_12*r_22 + 8*Py*a_4*r_13*r_23 + 8*Pz*a_4*r_11*r_31 + 8*Pz*a_4*r_12*r_32 + 8*Pz*a_4*r_13*r_33 - x1211 - x1222;
        C(5, 5) = x1191 - x1192 - x1194 - x1196 - x1197 - x1198 - x1224;
        C(5, 6) = -x1014 + x1017 + x1018 + x1019 + x1024 + x1025 + x1026 + x1027 + x1028 + x1029 - x1225 - x1230 - x1466;
        C(5, 7) = x1106 + x1229 + x1468;
        C(5, 8) = x1031 + x1033 + x1035 + x1036 + x1038 + x1039 + x1141 + x1231 + x1465;
        
        // Invoke the solver
        std::array<double, 16> solution_buffer;
        int n_solutions = yaik_cpp::general_6dof_internal::computeSolutionFromTanhalfLME(A, B, C, &solution_buffer);
        
        for(auto i = 0; i < n_solutions; i++)
        {
            auto solution_i = make_raw_solution();
            solution_i[7] = solution_buffer[i];
            int appended_idx = append_solution_to_queue(solution_i);
            add_input_index_to(2, appended_idx);
        };
    };
    // Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_5_processor();
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
            const double th_5 = this_solution[7];
            const bool checked_result = std::fabs(Px - a_4*r_11*std::cos(th_5) + a_4*r_12*std::sin(th_5) - d_5*r_13) <= 9.9999999999999995e-7 && std::fabs(Py - a_4*r_21*std::cos(th_5) + a_4*r_22*std::sin(th_5) - d_5*r_23) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor();
    // Finish code for equation all-zero dispatcher node 2
    
    // Code for explicit solution node 3, solved variable is th_0
    auto ExplicitSolutionNode_node_3_solve_th_0_processor = [&]() -> void
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
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(Px - a_4*r_11*std::cos(th_5) + a_4*r_12*std::sin(th_5) - d_5*r_13) >= zero_tolerance || std::fabs(Py - a_4*r_21*std::cos(th_5) + a_4*r_22*std::sin(th_5) - d_5*r_23) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_4*std::sin(th_5);
                const double x1 = a_4*std::cos(th_5);
                const double x2 = std::atan2(Py - d_5*r_23 - r_21*x1 + r_22*x0, Px - d_5*r_13 - r_11*x1 + r_12*x0);
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[0] = x2;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = std::fabs(Px - a_4*r_11*std::cos(th_5) + a_4*r_12*std::sin(th_5) - d_5*r_13) >= zero_tolerance || std::fabs(Py - a_4*r_21*std::cos(th_5) + a_4*r_22*std::sin(th_5) - d_5*r_23) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = a_4*std::sin(th_5);
                const double x1 = a_4*std::cos(th_5);
                const double x2 = std::atan2(Py - d_5*r_23 - r_21*x1 + r_22*x0, Px - d_5*r_13 - r_11*x1 + r_12*x0);
                // End of temp variables
                const double tmp_sol_value = x2 + M_PI;
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(4, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_0_processor();
    // Finish code for explicit solution node 3
    
    // Code for non-branch dispatcher node 4
    // Actually, there is no code
    
    // Code for explicit solution node 5, solved variable is th_3
    auto ExplicitSolutionNode_node_5_solve_th_3_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs((r_11*std::sin(th_0) - r_21*std::cos(th_0))*std::sin(th_5) + (r_12*std::sin(th_0) - r_22*std::cos(th_0))*std::cos(th_5)) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double x2 = safe_acos((-r_11*x0 + r_21*x1)*std::sin(th_5) + (-r_12*x0 + r_22*x1)*std::cos(th_5));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[5] = x2;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx);
            }
            
            const bool condition_1 = std::fabs((r_11*std::sin(th_0) - r_21*std::cos(th_0))*std::sin(th_5) + (r_12*std::sin(th_0) - r_22*std::cos(th_0))*std::cos(th_5)) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double x2 = safe_acos((-r_11*x0 + r_21*x1)*std::sin(th_5) + (-r_12*x0 + r_22*x1)*std::cos(th_5));
                // End of temp variables
                const double tmp_sol_value = -x2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(6, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_3_processor();
    // Finish code for explicit solution node 4
    
    // Code for non-branch dispatcher node 6
    // Actually, there is no code
    
    // Code for explicit solution node 7, solved variable is th_2
    auto ExplicitSolutionNode_node_7_solve_th_2_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_5 = this_solution[7];
            
            const bool condition_0 = 2*std::fabs(a_1*a_2) >= zero_tolerance || 2*std::fabs(a_1*d_3) >= zero_tolerance || std::fabs(std::pow(Px, 2) - 2*Px*a_0*std::cos(th_0) - 2*Px*a_4*r_11*std::cos(th_5) + 2*Px*a_4*r_12*std::sin(th_5) - 2*Px*d_5*r_13 + std::pow(Py, 2) - 2*Py*a_0*std::sin(th_0) - 2*Py*a_4*r_21*std::cos(th_5) + 2*Py*a_4*r_22*std::sin(th_5) - 2*Py*d_5*r_23 + std::pow(Pz, 2) - 2*Pz*a_4*r_31*std::cos(th_5) + 2*Pz*a_4*r_32*std::sin(th_5) - 2*Pz*d_5*r_33 + std::pow(a_0, 2) + a_0*a_4*r_11*std::cos(th_0 - th_5) + a_0*a_4*r_11*std::cos(th_0 + th_5) + a_0*a_4*r_12*std::sin(th_0 - th_5) - a_0*a_4*r_12*std::sin(th_0 + th_5) + a_0*a_4*r_21*std::sin(th_0 - th_5) + a_0*a_4*r_21*std::sin(th_0 + th_5) - a_0*a_4*r_22*std::cos(th_0 - th_5) + a_0*a_4*r_22*std::cos(th_0 + th_5) + 2*a_0*d_5*r_13*std::cos(th_0) + 2*a_0*d_5*r_23*std::sin(th_0) - std::pow(a_1, 2) - std::pow(a_2, 2) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_11, 2)*std::cos(2*th_5) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_11, 2) - std::pow(a_4, 2)*r_11*r_12*std::sin(2*th_5) - 1.0/2.0*std::pow(a_4, 2)*std::pow(r_12, 2)*std::cos(2*th_5) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_12, 2) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_21, 2)*std::cos(2*th_5) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_21, 2) - std::pow(a_4, 2)*r_21*r_22*std::sin(2*th_5) - 1.0/2.0*std::pow(a_4, 2)*std::pow(r_22, 2)*std::cos(2*th_5) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_22, 2) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_31, 2)*std::cos(2*th_5) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_31, 2) - std::pow(a_4, 2)*r_31*r_32*std::sin(2*th_5) - 1.0/2.0*std::pow(a_4, 2)*std::pow(r_32, 2)*std::cos(2*th_5) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_32, 2) + 2*a_4*d_5*r_11*r_13*std::cos(th_5) - 2*a_4*d_5*r_12*r_13*std::sin(th_5) + 2*a_4*d_5*r_21*r_23*std::cos(th_5) - 2*a_4*d_5*r_22*r_23*std::sin(th_5) + 2*a_4*d_5*r_31*r_33*std::cos(th_5) - 2*a_4*d_5*r_32*r_33*std::sin(th_5) - std::pow(d_3, 2) + std::pow(d_5, 2)*std::pow(r_13, 2) + std::pow(d_5, 2)*std::pow(r_23, 2) + std::pow(d_5, 2)*std::pow(r_33, 2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_1;
                const double x1 = std::atan2(-d_3*x0, a_2*x0);
                const double x2 = std::pow(a_2, 2);
                const double x3 = std::pow(a_1, 2);
                const double x4 = 4*x3;
                const double x5 = std::pow(d_3, 2);
                const double x6 = 2*Px;
                const double x7 = d_5*r_13;
                const double x8 = 2*Py;
                const double x9 = d_5*r_23;
                const double x10 = 2*Pz;
                const double x11 = d_5*r_33;
                const double x12 = a_0*std::cos(th_0);
                const double x13 = a_0*std::sin(th_0);
                const double x14 = std::pow(d_5, 2);
                const double x15 = a_4*std::cos(th_5);
                const double x16 = r_11*x15;
                const double x17 = a_4*std::sin(th_5);
                const double x18 = r_12*x17;
                const double x19 = r_21*x15;
                const double x20 = r_22*x17;
                const double x21 = r_31*x15;
                const double x22 = r_32*x17;
                const double x23 = 2*x7;
                const double x24 = 2*x9;
                const double x25 = std::pow(a_4, 2);
                const double x26 = (1.0/2.0)*x25;
                const double x27 = std::pow(r_11, 2)*x26;
                const double x28 = std::pow(r_12, 2)*x26;
                const double x29 = std::pow(r_21, 2)*x26;
                const double x30 = std::pow(r_22, 2)*x26;
                const double x31 = std::pow(r_31, 2)*x26;
                const double x32 = std::pow(r_32, 2)*x26;
                const double x33 = th_0 + th_5;
                const double x34 = std::cos(x33);
                const double x35 = a_0*a_4;
                const double x36 = r_11*x35;
                const double x37 = x35*std::sin(x33);
                const double x38 = r_22*x35;
                const double x39 = 2*x11;
                const double x40 = th_0 - th_5;
                const double x41 = std::cos(x40);
                const double x42 = x35*std::sin(x40);
                const double x43 = 2*th_5;
                const double x44 = x25*std::sin(x43);
                const double x45 = std::cos(x43);
                const double x46 = std::pow(Px, 2) + std::pow(Py, 2) + std::pow(Pz, 2) + std::pow(a_0, 2) - r_11*r_12*x44 - r_12*x37 + r_12*x42 + std::pow(r_13, 2)*x14 - r_21*r_22*x44 + r_21*x37 + r_21*x42 + std::pow(r_23, 2)*x14 - r_31*r_32*x44 + std::pow(r_33, 2)*x14 - x10*x11 - x10*x21 + x10*x22 + x12*x23 - x12*x6 + x13*x24 - x13*x8 + x16*x23 - x16*x6 - x18*x23 + x18*x6 + x19*x24 - x19*x8 - x2 - x20*x24 + x20*x8 + x21*x39 - x22*x39 + x27*x45 + x27 - x28*x45 + x28 + x29*x45 + x29 - x3 - x30*x45 + x30 + x31*x45 + x31 - x32*x45 + x32 + x34*x36 + x34*x38 + x36*x41 - x38*x41 - x5 - x6*x7 - x8*x9;
                const double x47 = safe_sqrt(x2*x4 + x4*x5 - std::pow(x46, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x1 + std::atan2(x47, x46);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(8, appended_idx);
            }
            
            const bool condition_1 = 2*std::fabs(a_1*a_2) >= zero_tolerance || 2*std::fabs(a_1*d_3) >= zero_tolerance || std::fabs(std::pow(Px, 2) - 2*Px*a_0*std::cos(th_0) - 2*Px*a_4*r_11*std::cos(th_5) + 2*Px*a_4*r_12*std::sin(th_5) - 2*Px*d_5*r_13 + std::pow(Py, 2) - 2*Py*a_0*std::sin(th_0) - 2*Py*a_4*r_21*std::cos(th_5) + 2*Py*a_4*r_22*std::sin(th_5) - 2*Py*d_5*r_23 + std::pow(Pz, 2) - 2*Pz*a_4*r_31*std::cos(th_5) + 2*Pz*a_4*r_32*std::sin(th_5) - 2*Pz*d_5*r_33 + std::pow(a_0, 2) + a_0*a_4*r_11*std::cos(th_0 - th_5) + a_0*a_4*r_11*std::cos(th_0 + th_5) + a_0*a_4*r_12*std::sin(th_0 - th_5) - a_0*a_4*r_12*std::sin(th_0 + th_5) + a_0*a_4*r_21*std::sin(th_0 - th_5) + a_0*a_4*r_21*std::sin(th_0 + th_5) - a_0*a_4*r_22*std::cos(th_0 - th_5) + a_0*a_4*r_22*std::cos(th_0 + th_5) + 2*a_0*d_5*r_13*std::cos(th_0) + 2*a_0*d_5*r_23*std::sin(th_0) - std::pow(a_1, 2) - std::pow(a_2, 2) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_11, 2)*std::cos(2*th_5) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_11, 2) - std::pow(a_4, 2)*r_11*r_12*std::sin(2*th_5) - 1.0/2.0*std::pow(a_4, 2)*std::pow(r_12, 2)*std::cos(2*th_5) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_12, 2) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_21, 2)*std::cos(2*th_5) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_21, 2) - std::pow(a_4, 2)*r_21*r_22*std::sin(2*th_5) - 1.0/2.0*std::pow(a_4, 2)*std::pow(r_22, 2)*std::cos(2*th_5) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_22, 2) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_31, 2)*std::cos(2*th_5) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_31, 2) - std::pow(a_4, 2)*r_31*r_32*std::sin(2*th_5) - 1.0/2.0*std::pow(a_4, 2)*std::pow(r_32, 2)*std::cos(2*th_5) + (1.0/2.0)*std::pow(a_4, 2)*std::pow(r_32, 2) + 2*a_4*d_5*r_11*r_13*std::cos(th_5) - 2*a_4*d_5*r_12*r_13*std::sin(th_5) + 2*a_4*d_5*r_21*r_23*std::cos(th_5) - 2*a_4*d_5*r_22*r_23*std::sin(th_5) + 2*a_4*d_5*r_31*r_33*std::cos(th_5) - 2*a_4*d_5*r_32*r_33*std::sin(th_5) - std::pow(d_3, 2) + std::pow(d_5, 2)*std::pow(r_13, 2) + std::pow(d_5, 2)*std::pow(r_23, 2) + std::pow(d_5, 2)*std::pow(r_33, 2)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_1;
                const double x1 = std::atan2(-d_3*x0, a_2*x0);
                const double x2 = std::pow(a_2, 2);
                const double x3 = std::pow(a_1, 2);
                const double x4 = 4*x3;
                const double x5 = std::pow(d_3, 2);
                const double x6 = 2*Px;
                const double x7 = d_5*r_13;
                const double x8 = 2*Py;
                const double x9 = d_5*r_23;
                const double x10 = 2*Pz;
                const double x11 = d_5*r_33;
                const double x12 = a_0*std::cos(th_0);
                const double x13 = a_0*std::sin(th_0);
                const double x14 = std::pow(d_5, 2);
                const double x15 = a_4*std::cos(th_5);
                const double x16 = r_11*x15;
                const double x17 = a_4*std::sin(th_5);
                const double x18 = r_12*x17;
                const double x19 = r_21*x15;
                const double x20 = r_22*x17;
                const double x21 = r_31*x15;
                const double x22 = r_32*x17;
                const double x23 = 2*x7;
                const double x24 = 2*x9;
                const double x25 = std::pow(a_4, 2);
                const double x26 = (1.0/2.0)*x25;
                const double x27 = std::pow(r_11, 2)*x26;
                const double x28 = std::pow(r_12, 2)*x26;
                const double x29 = std::pow(r_21, 2)*x26;
                const double x30 = std::pow(r_22, 2)*x26;
                const double x31 = std::pow(r_31, 2)*x26;
                const double x32 = std::pow(r_32, 2)*x26;
                const double x33 = th_0 + th_5;
                const double x34 = std::cos(x33);
                const double x35 = a_0*a_4;
                const double x36 = r_11*x35;
                const double x37 = x35*std::sin(x33);
                const double x38 = r_22*x35;
                const double x39 = 2*x11;
                const double x40 = th_0 - th_5;
                const double x41 = std::cos(x40);
                const double x42 = x35*std::sin(x40);
                const double x43 = 2*th_5;
                const double x44 = x25*std::sin(x43);
                const double x45 = std::cos(x43);
                const double x46 = std::pow(Px, 2) + std::pow(Py, 2) + std::pow(Pz, 2) + std::pow(a_0, 2) - r_11*r_12*x44 - r_12*x37 + r_12*x42 + std::pow(r_13, 2)*x14 - r_21*r_22*x44 + r_21*x37 + r_21*x42 + std::pow(r_23, 2)*x14 - r_31*r_32*x44 + std::pow(r_33, 2)*x14 - x10*x11 - x10*x21 + x10*x22 + x12*x23 - x12*x6 + x13*x24 - x13*x8 + x16*x23 - x16*x6 - x18*x23 + x18*x6 + x19*x24 - x19*x8 - x2 - x20*x24 + x20*x8 + x21*x39 - x22*x39 + x27*x45 + x27 - x28*x45 + x28 + x29*x45 + x29 - x3 - x30*x45 + x30 + x31*x45 + x31 - x32*x45 + x32 + x34*x36 + x34*x38 + x36*x41 - x38*x41 - x5 - x6*x7 - x8*x9;
                const double x47 = safe_sqrt(x2*x4 + x4*x5 - std::pow(x46, 2));
                // End of temp variables
                const double tmp_sol_value = x1 + std::atan2(-x47, x46);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(8, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_2_processor();
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
                add_input_index_to(17, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(9, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor();
    // Finish code for solved_variable dispatcher node 8
    
    // Code for explicit solution node 17, solved variable is th_1th_2th_4_soa
    auto ExplicitSolutionNode_node_17_solve_th_1th_2th_4_soa_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_33) >= zero_tolerance || std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_13*std::cos(th_0) + r_23*std::sin(th_0), r_33);
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(18, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_1th_2th_4_soa_processor();
    // Finish code for explicit solution node 17
    
    // Code for solved_variable dispatcher node 18
    auto SolvedVariableDispatcherNode_node_18_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(18);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(18);
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
            
            const bool degenerate_valid_0 = std::fabs(th_2 - M_PI + 1.39979827560533) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(22, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_2 - 2*M_PI + 1.39979827560533) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(25, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(19, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_18_processor();
    // Finish code for solved_variable dispatcher node 18
    
    // Code for explicit solution node 25, solved variable is th_1
    auto ExplicitSolutionNode_node_25_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(25);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(25);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 25
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(0.98541542341935695*a_2 - 0.17016592869094099*d_3) >= zero_tolerance || std::fabs(a_1 + 0.17016592869094099*a_2 + 0.98541542341935695*d_3) >= zero_tolerance || std::fabs(Pz - a_4*r_31*std::cos(th_5) + a_4*r_32*std::sin(th_5) - d_5*r_33) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_1 + 0.17016592869094099*a_2 + 0.98541542341935695*d_3;
                const double x1 = std::atan2(x0, -0.98541542341935695*a_2 + 0.17016592869094099*d_3);
                const double x2 = -Pz + a_4*r_31*std::cos(th_5) - a_4*r_32*std::sin(th_5) + d_5*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.97104355671275*std::pow(-a_2 + 0.172684458398744*d_3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x1 + std::atan2(x3, x2);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(26, appended_idx);
            }
            
            const bool condition_1 = std::fabs(0.98541542341935695*a_2 - 0.17016592869094099*d_3) >= zero_tolerance || std::fabs(a_1 + 0.17016592869094099*a_2 + 0.98541542341935695*d_3) >= zero_tolerance || std::fabs(Pz - a_4*r_31*std::cos(th_5) + a_4*r_32*std::sin(th_5) - d_5*r_33) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = a_1 + 0.17016592869094099*a_2 + 0.98541542341935695*d_3;
                const double x1 = std::atan2(x0, -0.98541542341935695*a_2 + 0.17016592869094099*d_3);
                const double x2 = -Pz + a_4*r_31*std::cos(th_5) - a_4*r_32*std::sin(th_5) + d_5*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.97104355671275*std::pow(-a_2 + 0.172684458398744*d_3, 2));
                // End of temp variables
                const double tmp_sol_value = x1 + std::atan2(-x3, x2);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(26, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_1_processor();
    // Finish code for explicit solution node 25
    
    // Code for non-branch dispatcher node 26
    // Actually, there is no code
    
    // Code for explicit solution node 27, solved variable is th_4
    auto ExplicitSolutionNode_node_27_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(26);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(26);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 27
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[5];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_13*((0.98541542341935695*std::sin(th_1) + 0.17016592869094099*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) + r_23*((0.98541542341935695*std::sin(th_1) + 0.17016592869094099*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(0.17016592869094099*std::sin(th_1) - 0.98541542341935695*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance || std::fabs(-r_13*(-0.17016592869094099*std::sin(th_1) + 0.98541542341935695*std::cos(th_1))*std::cos(th_0) - r_23*(-0.17016592869094099*std::sin(th_1) + 0.98541542341935695*std::cos(th_1))*std::sin(th_0) + r_33*(0.98541542341935695*std::sin(th_1) + 0.17016592869094099*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_1);
                const double x3 = -0.17016592869094099*x1 + 0.98541542341935695*x2;
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_3);
                const double x6 = std::cos(th_0);
                const double x7 = 0.98541542341935695*x1 + 0.17016592869094099*x2;
                const double x8 = x0*x7;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*x3, -r_13*x3*x6 - r_23*x3*x4 + r_33*x7);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_4_processor();
    // Finish code for explicit solution node 26
    
    // Code for explicit solution node 22, solved variable is th_1
    auto ExplicitSolutionNode_node_22_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(22);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(22);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 22
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(0.98541542341935695*a_2 - 0.17016592869094099*d_3) >= zero_tolerance || std::fabs(-a_1 + 0.17016592869094099*a_2 + 0.98541542341935695*d_3) >= zero_tolerance || std::fabs(Pz - a_4*r_31*std::cos(th_5) + a_4*r_32*std::sin(th_5) - d_5*r_33) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = -a_1 + 0.17016592869094099*a_2 + 0.98541542341935695*d_3;
                const double x1 = std::atan2(x0, -0.98541542341935695*a_2 + 0.17016592869094099*d_3);
                const double x2 = Pz - a_4*r_31*std::cos(th_5) + a_4*r_32*std::sin(th_5) - d_5*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.97104355671275*std::pow(-a_2 + 0.172684458398744*d_3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x1 + std::atan2(x3, x2);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(23, appended_idx);
            }
            
            const bool condition_1 = std::fabs(0.98541542341935695*a_2 - 0.17016592869094099*d_3) >= zero_tolerance || std::fabs(-a_1 + 0.17016592869094099*a_2 + 0.98541542341935695*d_3) >= zero_tolerance || std::fabs(Pz - a_4*r_31*std::cos(th_5) + a_4*r_32*std::sin(th_5) - d_5*r_33) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = -a_1 + 0.17016592869094099*a_2 + 0.98541542341935695*d_3;
                const double x1 = std::atan2(x0, -0.98541542341935695*a_2 + 0.17016592869094099*d_3);
                const double x2 = Pz - a_4*r_31*std::cos(th_5) + a_4*r_32*std::sin(th_5) - d_5*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.97104355671275*std::pow(-a_2 + 0.172684458398744*d_3, 2));
                // End of temp variables
                const double tmp_sol_value = x1 + std::atan2(-x3, x2);
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
    ExplicitSolutionNode_node_22_solve_th_1_processor();
    // Finish code for explicit solution node 22
    
    // Code for non-branch dispatcher node 23
    // Actually, there is no code
    
    // Code for explicit solution node 24, solved variable is th_4
    auto ExplicitSolutionNode_node_24_solve_th_4_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[5];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(-r_13*((-0.98541542341935695*std::sin(th_1) - 0.17016592869094099*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) - r_23*((-0.98541542341935695*std::sin(th_1) - 0.17016592869094099*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(0.17016592869094099*std::sin(th_1) - 0.98541542341935695*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance || std::fabs(-r_13*(-0.17016592869094099*std::sin(th_1) + 0.98541542341935695*std::cos(th_1))*std::cos(th_0) - r_23*(-0.17016592869094099*std::sin(th_1) + 0.98541542341935695*std::cos(th_1))*std::sin(th_0) + r_33*(0.98541542341935695*std::sin(th_1) + 0.17016592869094099*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_1);
                const double x3 = 0.17016592869094099*x1 - 0.98541542341935695*x2;
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_3);
                const double x6 = std::cos(th_0);
                const double x7 = 0.98541542341935695*x1 + 0.17016592869094099*x2;
                const double x8 = -x0*x7;
                const double x9 = -x3;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*x3, r_13*x6*x9 + r_23*x4*x9 - r_33*x7);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_4_processor();
    // Finish code for explicit solution node 23
    
    // Code for explicit solution node 19, solved variable is th_1
    auto ExplicitSolutionNode_node_19_solve_th_1_processor = [&]() -> void
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
            const double th_1th_2th_4_soa = this_solution[3];
            const double th_2 = this_solution[4];
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(a_2*std::sin(th_2) + d_3*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_1 + a_2*std::cos(th_2) - d_3*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_2);
                const double x1 = std::sin(th_2);
                const double x2 = a_1 + a_2*x0 - d_3*x1;
                const double x3 = Pz - a_4*r_31*std::cos(th_5) + a_4*r_32*std::sin(th_5) - d_5*r_33;
                const double x4 = -a_2*x1 - d_3*x0;
                const double x5 = Px*std::cos(th_0) + Py*std::sin(th_0) - a_0 + a_4*std::cos(th_1th_2th_4_soa) - d_5*std::sin(th_1th_2th_4_soa);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x2*x3 + x4*x5, x2*x5 + x3*x4);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(20, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_19_solve_th_1_processor();
    // Finish code for explicit solution node 19
    
    // Code for non-branch dispatcher node 20
    // Actually, there is no code
    
    // Code for explicit solution node 21, solved variable is th_4
    auto ExplicitSolutionNode_node_21_solve_th_4_processor = [&]() -> void
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
    ExplicitSolutionNode_node_21_solve_th_4_processor();
    // Finish code for explicit solution node 20
    
    // Code for explicit solution node 14, solved variable is th_1
    auto ExplicitSolutionNode_node_14_solve_th_1_processor = [&]() -> void
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
            const double th_2 = this_solution[4];
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(a_2*std::sin(th_2) + d_3*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_1 + a_2*std::cos(th_2) - d_3*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_2);
                const double x1 = std::sin(th_2);
                const double x2 = a_1 + a_2*x0 - d_3*x1;
                const double x3 = std::sin(th_5);
                const double x4 = a_4*std::cos(th_5);
                const double x5 = Pz + a_4*r_32*x3 - d_5*r_33 - r_31*x4;
                const double x6 = -a_2*x1 - d_3*x0;
                const double x7 = std::cos(th_0);
                const double x8 = std::sin(th_0);
                const double x9 = Px*x7 + Py*x8 - a_0 + a_4*r_12*x3*x7 + a_4*r_22*x3*x8 - d_5*r_13*x7 - d_5*r_23*x8 - r_11*x4*x7 - r_21*x4*x8;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x2*x5 + x6*x9, x2*x9 + x5*x6);
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
    ExplicitSolutionNode_node_14_solve_th_1_processor();
    // Finish code for explicit solution node 14
    
    // Code for non-branch dispatcher node 15
    // Actually, there is no code
    
    // Code for explicit solution node 16, solved variable is th_4
    auto ExplicitSolutionNode_node_16_solve_th_4_processor = [&]() -> void
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
    ExplicitSolutionNode_node_16_solve_th_4_processor();
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
            const double th_0 = this_solution[0];
            const double th_3 = this_solution[5];
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(r_13*std::sin(th_0) - r_23*std::cos(th_0)) >= zero_tolerance || std::fabs((r_11*std::sin(th_0) - r_21*std::cos(th_0))*std::cos(th_5) + (-r_12*std::sin(th_0) + r_22*std::cos(th_0))*std::sin(th_5)) >= zero_tolerance || std::fabs(std::sin(th_3)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_3);
                const double x1 = std::sin(th_0);
                const double x2 = std::cos(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(-r_13*x1 + r_23*x2), x0*(-(-r_11*x1 + r_21*x2)*std::cos(th_5) + (-r_12*x1 + r_22*x2)*std::sin(th_5)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
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
    
    // Code for explicit solution node 11, solved variable is th_1th_2_soa
    auto ExplicitSolutionNode_node_11_solve_th_1th_2_soa_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_3 = this_solution[5];
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(r_31*std::sin(th_5) + r_32*std::cos(th_5)) >= zero_tolerance || std::fabs((r_11*std::cos(th_0) + r_21*std::sin(th_0))*std::sin(th_5) + (r_12*std::cos(th_0) + r_22*std::sin(th_0))*std::cos(th_5)) >= zero_tolerance || std::fabs(std::sin(th_3)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_3);
                const double x1 = std::sin(th_5);
                const double x2 = std::cos(th_5);
                const double x3 = std::cos(th_0);
                const double x4 = std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(-r_31*x1 - r_32*x2), x0*(-x1*(-r_11*x3 - r_21*x4) + x2*(r_12*x3 + r_22*x4)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(12, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_1th_2_soa_processor();
    // Finish code for explicit solution node 10
    
    // Code for non-branch dispatcher node 12
    // Actually, there is no code
    
    // Code for explicit solution node 13, solved variable is th_1
    auto ExplicitSolutionNode_node_13_solve_th_1_processor = [&]() -> void
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
            const double th_1th_2_soa = this_solution[2];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_1th_2_soa - th_2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_1_processor();
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

}; // struct abb_crb15000_10_1_52_ik

// Code below for debug
void test_ik_solve_abb_crb15000_10_1_52()
{
    std::array<double, abb_crb15000_10_1_52_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = abb_crb15000_10_1_52_ik::computeFK(theta);
    auto ik_output = abb_crb15000_10_1_52_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = abb_crb15000_10_1_52_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_abb_crb15000_10_1_52();
}
