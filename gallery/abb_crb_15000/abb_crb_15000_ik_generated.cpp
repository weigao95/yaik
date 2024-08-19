#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct abb_crb_15000_ik {

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
static constexpr double a_0 = 0.444;
static constexpr double a_1 = 0.11;
static constexpr double a_3 = 0.08;
static constexpr double d_2 = 0.47;
static constexpr double d_4 = 0.101;
static constexpr double pre_transform_special_symbol_23 = 0.265;

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
    const double x5 = std::cos(th_1);
    const double x6 = std::cos(th_2);
    const double x7 = x5*x6;
    const double x8 = std::sin(th_1);
    const double x9 = std::sin(th_2);
    const double x10 = x8*x9;
    const double x11 = -x10*x4 + x4*x7;
    const double x12 = x1*x2 - x11*x3;
    const double x13 = std::cos(th_5);
    const double x14 = std::sin(th_4);
    const double x15 = x6*x8;
    const double x16 = x5*x9;
    const double x17 = -x15*x4 - x16*x4;
    const double x18 = std::cos(th_4);
    const double x19 = x1*x3 + x11*x2;
    const double x20 = -x14*x17 + x18*x19;
    const double x21 = -x14*x19 - x17*x18;
    const double x22 = a_0*x5;
    const double x23 = -x1*x10 + x1*x7;
    const double x24 = -x2*x4 - x23*x3;
    const double x25 = -x1*x15 - x1*x16;
    const double x26 = x2*x23 - x3*x4;
    const double x27 = -x14*x25 + x18*x26;
    const double x28 = -x14*x26 - x18*x25;
    const double x29 = -x15 - x16;
    const double x30 = x29*x3;
    const double x31 = x10 - x7;
    const double x32 = x2*x29;
    const double x33 = -x14*x31 + x18*x32;
    const double x34 = -x14*x32 - x18*x31;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x12 + x13*x20;
    ee_pose_raw(0, 1) = -x0*x20 - x12*x13;
    ee_pose_raw(0, 2) = x21;
    ee_pose_raw(0, 3) = a_1*x11 + a_3*x20 + d_2*x17 + d_4*x21 + x22*x4;
    ee_pose_raw(1, 0) = -x0*x24 + x13*x27;
    ee_pose_raw(1, 1) = -x0*x27 - x13*x24;
    ee_pose_raw(1, 2) = x28;
    ee_pose_raw(1, 3) = a_1*x23 + a_3*x27 + d_2*x25 + d_4*x28 + x1*x22;
    ee_pose_raw(2, 0) = x0*x30 + x13*x33;
    ee_pose_raw(2, 1) = -x0*x33 + x13*x30;
    ee_pose_raw(2, 2) = x34;
    ee_pose_raw(2, 3) = -a_0*x8 + a_1*x29 + a_3*x33 + d_2*x31 + d_4*x34;
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
    const double x14 = x10*x3 - x7*x8;
    const double x15 = x1*x12 - x13*x14;
    const double x16 = std::cos(th_4);
    const double x17 = std::sin(th_4);
    const double x18 = x1*x13 + x12*x14;
    const double x19 = -x11*x16 - x17*x18;
    const double x20 = x1*x4;
    const double x21 = x1*x9;
    const double x22 = -x20*x3 - x21*x8;
    const double x23 = -x20*x8 + x21*x3;
    const double x24 = -x12*x6 - x13*x23;
    const double x25 = x12*x23 - x13*x6;
    const double x26 = -x16*x22 - x17*x25;
    const double x27 = 1.0*x8;
    const double x28 = 1.0*x3;
    const double x29 = x27*x4 - x28*x9;
    const double x30 = -x27*x9 - x28*x4;
    const double x31 = x13*x30;
    const double x32 = x12*x30;
    const double x33 = -x16*x29 - x17*x32;
    const double x34 = 1.0*a_0;
    const double x35 = pre_transform_special_symbol_23 - x34*x4;
    const double x36 = a_1*x30 + d_2*x29 + x35;
    const double x37 = a_0*x21 + a_1*x23 + d_2*x22;
    const double x38 = a_3*(x16*x32 - x17*x29) + d_4*x33 + x36;
    const double x39 = a_3*(x16*x25 - x17*x22) + d_4*x26 + x37;
    const double x40 = a_0*x10 + a_1*x14 + d_2*x11;
    const double x41 = a_3*(-x11*x17 + x16*x18) + d_4*x19 + x40;
    const double x42 = x34*x9;
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
    jacobian(2, 5) = x33;
    jacobian(3, 1) = -pre_transform_special_symbol_23*x6;
    jacobian(3, 2) = -x35*x6;
    jacobian(3, 3) = -x22*x36 + x29*x37;
    jacobian(3, 4) = -x24*x36 - x31*x37;
    jacobian(3, 5) = -x26*x38 + x33*x39;
    jacobian(4, 1) = -pre_transform_special_symbol_23*x1;
    jacobian(4, 2) = -x1*x35;
    jacobian(4, 3) = x11*x36 - x29*x40;
    jacobian(4, 4) = x15*x36 + x31*x40;
    jacobian(4, 5) = x19*x38 - x33*x41;
    jacobian(5, 2) = std::pow(x0, 2)*x42 + x42*std::pow(x5, 2);
    jacobian(5, 3) = -x11*x37 + x22*x40;
    jacobian(5, 4) = -x15*x37 + x24*x40;
    jacobian(5, 5) = -x19*x39 + x26*x41;
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
    const double x0 = 1.0*std::sin(th_0);
    const double x1 = -x0;
    const double x2 = std::cos(th_2);
    const double x3 = std::sin(th_1);
    const double x4 = 1.0*std::cos(th_0);
    const double x5 = x3*x4;
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_1);
    const double x8 = x4*x7;
    const double x9 = -x2*x5 - x6*x8;
    const double x10 = std::cos(th_3);
    const double x11 = std::sin(th_3);
    const double x12 = x2*x8 - x5*x6;
    const double x13 = std::cos(th_4);
    const double x14 = std::sin(th_4);
    const double x15 = x0*x3;
    const double x16 = x0*x7;
    const double x17 = -x15*x2 - x16*x6;
    const double x18 = -x15*x6 + x16*x2;
    const double x19 = 1.0*x6;
    const double x20 = 1.0*x2;
    const double x21 = x19*x3 - x20*x7;
    const double x22 = -x19*x7 - x20*x3;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x1;
    jacobian(0, 2) = x1;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = x0*x10 - x11*x12;
    jacobian(0, 5) = -x13*x9 - x14*(x0*x11 + x10*x12);
    jacobian(1, 1) = x4;
    jacobian(1, 2) = x4;
    jacobian(1, 3) = x17;
    jacobian(1, 4) = -x10*x4 - x11*x18;
    jacobian(1, 5) = -x13*x17 - x14*(x10*x18 - x11*x4);
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x21;
    jacobian(2, 4) = -x11*x22;
    jacobian(2, 5) = -x10*x14*x22 - x13*x21;
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
    const double x13 = -x12 + x8;
    const double x14 = std::sin(th_0);
    const double x15 = x5*x9;
    const double x16 = x11*x7;
    const double x17 = -x14*x15 - x14*x16;
    const double x18 = -x15 - x16;
    const double x19 = a_1*x18 + d_2*x13 + x6;
    const double x20 = a_0*x11;
    const double x21 = x12*x14 - x14*x8;
    const double x22 = a_1*x21 + d_2*x17 + x14*x20;
    const double x23 = std::sin(th_3);
    const double x24 = x18*x23;
    const double x25 = std::cos(th_3);
    const double x26 = -x2*x25 - x21*x23;
    const double x27 = std::cos(th_4);
    const double x28 = std::sin(th_4);
    const double x29 = x18*x25;
    const double x30 = -x13*x27 - x28*x29;
    const double x31 = -x2*x23 + x21*x25;
    const double x32 = -x17*x27 - x28*x31;
    const double x33 = a_3*(-x13*x28 + x27*x29) + d_4*x30 + x19;
    const double x34 = a_3*(-x17*x28 + x27*x31) + d_4*x32 + x22;
    const double x35 = 1.0*p_on_ee_x;
    const double x36 = 1.0*x14;
    const double x37 = p_on_ee_z*x36;
    const double x38 = x2*x4;
    const double x39 = x10*x2;
    const double x40 = -x38*x9 - x39*x7;
    const double x41 = -x38*x7 + x39*x9;
    const double x42 = a_0*x39 + a_1*x41 + d_2*x40;
    const double x43 = -x23*x41 + x25*x36;
    const double x44 = x23*x36 + x25*x41;
    const double x45 = -x27*x40 - x28*x44;
    const double x46 = a_3*(x27*x44 - x28*x40) + d_4*x45 + x42;
    const double x47 = -x0*x14 - x1*x35;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = -pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = -x2*x6 + x3;
    jacobian(0, 3) = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x22 - x17*x19;
    jacobian(0, 4) = p_on_ee_y*x24 + p_on_ee_z*x26 - x19*x26 - x22*x24;
    jacobian(0, 5) = -p_on_ee_y*x30 + p_on_ee_z*x32 + x30*x34 - x32*x33;
    jacobian(1, 0) = x35;
    jacobian(1, 1) = -pre_transform_special_symbol_23*x36 + x37;
    jacobian(1, 2) = -x36*x6 + x37;
    jacobian(1, 3) = p_on_ee_x*x13 - p_on_ee_z*x40 - x13*x42 + x19*x40;
    jacobian(1, 4) = -p_on_ee_x*x24 - p_on_ee_z*x43 + x19*x43 + x24*x42;
    jacobian(1, 5) = p_on_ee_x*x30 - p_on_ee_z*x45 - x30*x46 + x33*x45;
    jacobian(2, 1) = x47;
    jacobian(2, 2) = std::pow(x1, 2)*x20 + std::pow(x14, 2)*x20 + x47;
    jacobian(2, 3) = -p_on_ee_x*x17 + p_on_ee_y*x40 + x17*x42 - x22*x40;
    jacobian(2, 4) = -p_on_ee_x*x26 + p_on_ee_y*x43 - x22*x43 + x26*x42;
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
        const double x4 = a_3*r_21;
        const double x5 = d_2*r_23;
        const double x6 = -x5;
        const double x7 = x4 + x6;
        const double x8 = std::pow(r_21, 2);
        const double x9 = Py*x8;
        const double x10 = std::pow(r_22, 2);
        const double x11 = Py*x10;
        const double x12 = std::pow(r_23, 2);
        const double x13 = Py*x12;
        const double x14 = d_4*r_23;
        const double x15 = Px*r_11;
        const double x16 = r_21*x15;
        const double x17 = Px*r_12;
        const double x18 = r_22*x17;
        const double x19 = Px*r_13;
        const double x20 = r_23*x19;
        const double x21 = Pz*r_31;
        const double x22 = r_21*x21;
        const double x23 = Pz*r_32;
        const double x24 = r_22*x23;
        const double x25 = Pz*r_33;
        const double x26 = r_23*x25;
        const double x27 = x11 + x13 - x14 + x16 + x18 + x20 + x22 + x24 + x26 + x9;
        const double x28 = a_3*r_22;
        const double x29 = 2*x28;
        const double x30 = -x4;
        const double x31 = x27 + x30;
        const double x32 = d_2*x1;
        const double x33 = -x32;
        const double x34 = d_2*x3;
        const double x35 = x4 + x5;
        const double x36 = Py*r_22;
        const double x37 = x17 + x23 + x36;
        const double x38 = a_0*x37;
        const double x39 = R_l_inv_51*x38;
        const double x40 = R_l_inv_52*a_0;
        const double x41 = d_2*x40;
        const double x42 = a_0*r_22;
        const double x43 = R_l_inv_54*x42;
        const double x44 = std::pow(a_3, 2);
        const double x45 = std::pow(d_2, 2);
        const double x46 = std::pow(d_4, 2);
        const double x47 = std::pow(a_0, 2);
        const double x48 = std::pow(a_1, 2);
        const double x49 = 2*d_4;
        const double x50 = 2*x14;
        const double x51 = Py*x1;
        const double x52 = 2*x17;
        const double x53 = Py*r_23;
        const double x54 = 2*x19;
        const double x55 = 2*x15;
        const double x56 = 2*x23;
        const double x57 = 2*x25;
        const double x58 = std::pow(Px, 2);
        const double x59 = std::pow(r_11, 2);
        const double x60 = x58*x59;
        const double x61 = std::pow(r_12, 2);
        const double x62 = x58*x61;
        const double x63 = std::pow(r_13, 2);
        const double x64 = x58*x63;
        const double x65 = std::pow(Py, 2);
        const double x66 = x65*x8;
        const double x67 = x10*x65;
        const double x68 = x12*x65;
        const double x69 = std::pow(Pz, 2);
        const double x70 = std::pow(r_31, 2)*x69;
        const double x71 = std::pow(r_32, 2)*x69;
        const double x72 = std::pow(r_33, 2)*x69;
        const double x73 = -Py*x50 + x15*x51 - x19*x49 + x21*x51 + x21*x55 + x23*x52 - x25*x49 + x25*x54 + x36*x52 + x36*x56 + x44 + x45 + x46 - x47 - x48 + x53*x54 + x53*x57 + x60 + x62 + x64 + x66 + x67 + x68 + x70 + x71 + x72;
        const double x74 = R_l_inv_55*a_0;
        const double x75 = x73*x74;
        const double x76 = d_4*r_21;
        const double x77 = r_23*x15;
        const double x78 = r_23*x21;
        const double x79 = r_21*x19;
        const double x80 = r_21*x25;
        const double x81 = x76 + x77 + x78 - x79 - x80;
        const double x82 = R_l_inv_57*a_0;
        const double x83 = x81*x82;
        const double x84 = -x83;
        const double x85 = d_2*r_22;
        const double x86 = R_l_inv_56*a_0;
        const double x87 = x85*x86;
        const double x88 = -x87;
        const double x89 = a_3*r_23;
        const double x90 = x82*x89;
        const double x91 = -x90;
        const double x92 = 2*a_3;
        const double x93 = Py*r_21;
        const double x94 = x15 + x21 + x93;
        const double x95 = x74*x94;
        const double x96 = x92*x95;
        const double x97 = a_1 + x39 + x41 + x43 + x75 + x84 + x88 + x91 + x96;
        const double x98 = -x28;
        const double x99 = R_l_inv_50*a_0;
        const double x100 = x94*x99;
        const double x101 = -x100;
        const double x102 = a_0*r_21;
        const double x103 = R_l_inv_53*x102;
        const double x104 = -x103;
        const double x105 = d_4*r_22;
        const double x106 = r_23*x17;
        const double x107 = r_23*x23;
        const double x108 = r_22*x19;
        const double x109 = r_22*x25;
        const double x110 = x105 + x106 + x107 - x108 - x109;
        const double x111 = x110*x86;
        const double x112 = -x111;
        const double x113 = R_l_inv_57*d_2*x102;
        const double x114 = -x113;
        const double x115 = x101 + x104 + x112 + x114 + x98;
        const double x116 = r_21*x17;
        const double x117 = r_21*x23;
        const double x118 = r_22*x15;
        const double x119 = -x118;
        const double x120 = r_22*x21;
        const double x121 = -x120;
        const double x122 = a_3*x99;
        const double x123 = -d_4 + x19 + x25 + x53;
        const double x124 = x123*x40;
        const double x125 = 2*d_2;
        const double x126 = x123*x125;
        const double x127 = x126*x74;
        const double x128 = -x122 - x124 - x127;
        const double x129 = x116 + x117 + x119 + x121 + x128;
        const double x130 = 2*R_l_inv_50*x38;
        const double x131 = 2*x94;
        const double x132 = -R_l_inv_51*a_0*x131;
        const double x133 = 4*a_3;
        const double x134 = R_l_inv_55*x38;
        const double x135 = x133*x134;
        const double x136 = -x130 + x132 + x135;
        const double x137 = R_l_inv_54*a_0;
        const double x138 = x1*x137;
        const double x139 = 2*x82;
        const double x140 = x110*x139;
        const double x141 = x32*x86;
        const double x142 = -x138 - x140 + x141;
        const double x143 = a_3*x1;
        const double x144 = 2*R_l_inv_53;
        const double x145 = x144*x42;
        const double x146 = 2*x86;
        const double x147 = x146*x81;
        const double x148 = x139*x85;
        const double x149 = x143 - x145 + x147 - x148;
        const double x150 = x100 + x103 + x111 + x113 + x28;
        const double x151 = a_1 - x39 + x41 + x75 - x96;
        const double x152 = -x43 + x83 + x87;
        const double x153 = x151 + x152 + x91;
        const double x154 = -x40*x92;
        const double x155 = x131*x40;
        const double x156 = 2*x123*x99;
        const double x157 = -d_2*x133*x74;
        const double x158 = 4*d_2;
        const double x159 = x158*x95;
        const double x160 = x154 - x155 + x156 + x157 - x159;
        const double x161 = a_0*r_23*x144;
        const double x162 = -x116 - x117 + x118 + x120;
        const double x163 = x146*x162;
        const double x164 = x139*x5;
        const double x165 = x161 + x163 + x164;
        const double x166 = 2*x105;
        const double x167 = 2*x106;
        const double x168 = 2*x107;
        const double x169 = 2*x108;
        const double x170 = 2*x109;
        const double x171 = x29*x86;
        const double x172 = -x166 - x167 - x168 + x169 + x170 + x171;
        const double x173 = 4*x76;
        const double x174 = 4*x79;
        const double x175 = 4*x80;
        const double x176 = 4*x77;
        const double x177 = 4*x78;
        const double x178 = 4*x4;
        const double x179 = x178*x86;
        const double x180 = 8*d_2;
        const double x181 = -4*R_l_inv_52*x38 - x134*x180;
        const double x182 = x154 + x155 + x156 + x157 + x159;
        const double x183 = x166 + x167 + x168 - x169 - x170 - x171;
        const double x184 = x122 + x124 + x127;
        const double x185 = x162 + x184;
        const double x186 = x130 + x132 + x135;
        const double x187 = -x143 + x145 - x147 + x148;
        const double x188 = a_0*a_1;
        const double x189 = 2*x188;
        const double x190 = x47 + x48;
        const double x191 = R_l_inv_62*x190;
        const double x192 = R_l_inv_22*x189 + x191;
        const double x193 = d_2*x192;
        const double x194 = R_l_inv_61*x190;
        const double x195 = x37*(R_l_inv_21*x189 + x194);
        const double x196 = R_l_inv_25*x189 + R_l_inv_65*x190;
        const double x197 = x196*x73;
        const double x198 = R_l_inv_60*x190;
        const double x199 = R_l_inv_20*x189 + x198;
        const double x200 = a_3*x199;
        const double x201 = -x200;
        const double x202 = x199*x94;
        const double x203 = -x202;
        const double x204 = x123*x192;
        const double x205 = -x204;
        const double x206 = x126*x196;
        const double x207 = -x206;
        const double x208 = x196*x94;
        const double x209 = x208*x92;
        const double x210 = x193 + x195 + x197 + x201 + x203 + x205 + x207 + x209;
        const double x211 = R_l_inv_24*x189 + R_l_inv_64*x190;
        const double x212 = r_22*x211;
        const double x213 = R_l_inv_67*x190;
        const double x214 = R_l_inv_27*x189 + x213;
        const double x215 = x214*x81;
        const double x216 = -x215;
        const double x217 = x214*x89;
        const double x218 = -x217;
        const double x219 = R_l_inv_66*x190;
        const double x220 = R_l_inv_26*x189 + x219;
        const double x221 = x220*x85;
        const double x222 = -x221;
        const double x223 = x5*x92;
        const double x224 = -x223;
        const double x225 = d_4*x32;
        const double x226 = -x225;
        const double x227 = x5*x55;
        const double x228 = -x227;
        const double x229 = 2*x21;
        const double x230 = x229*x5;
        const double x231 = -x230;
        const double x232 = x19*x32;
        const double x233 = x25*x32;
        const double x234 = x212 + x216 + x218 + x222 + x224 + x226 + x228 + x231 + x232 + x233;
        const double x235 = x14*x92;
        const double x236 = x9*x92;
        const double x237 = x11*x92;
        const double x238 = x13*x92;
        const double x239 = x143*x15;
        const double x240 = x17*x29;
        const double x241 = x20*x92;
        const double x242 = x143*x21;
        const double x243 = x23*x29;
        const double x244 = x26*x92;
        const double x245 = -x235 + x236 + x237 + x238 + x239 + x240 + x241 + x242 + x243 + x244;
        const double x246 = r_21*x44;
        const double x247 = std::pow(r_21, 3)*x65;
        const double x248 = r_21*x45;
        const double x249 = r_21*x46;
        const double x250 = R_l_inv_23*x189 + R_l_inv_63*x190;
        const double x251 = r_21*x250;
        const double x252 = x110*x220;
        const double x253 = r_21*x60;
        const double x254 = r_21*x67;
        const double x255 = r_21*x68;
        const double x256 = r_21*x70;
        const double x257 = d_2*r_21;
        const double x258 = x214*x257;
        const double x259 = r_21*x62;
        const double x260 = r_21*x64;
        const double x261 = r_21*x71;
        const double x262 = r_21*x72;
        const double x263 = x15*x50;
        const double x264 = x21*x50;
        const double x265 = x55*x9;
        const double x266 = x11*x55;
        const double x267 = x13*x55;
        const double x268 = d_4*x1;
        const double x269 = x19*x268;
        const double x270 = x229*x9;
        const double x271 = x11*x229;
        const double x272 = x13*x229;
        const double x273 = x25*x268;
        const double x274 = 2*r_11;
        const double x275 = r_12*x58;
        const double x276 = r_22*x275;
        const double x277 = x274*x276;
        const double x278 = 2*r_13;
        const double x279 = r_23*x58;
        const double x280 = r_11*x278*x279;
        const double x281 = 2*r_31;
        const double x282 = r_32*x69;
        const double x283 = r_22*x282;
        const double x284 = x281*x283;
        const double x285 = r_23*r_33;
        const double x286 = x285*x69;
        const double x287 = x281*x286;
        const double x288 = x17*x23;
        const double x289 = x1*x288;
        const double x290 = x19*x25;
        const double x291 = x1*x290;
        const double x292 = x15*x21;
        const double x293 = x1*x292;
        const double x294 = x24*x55;
        const double x295 = x26*x55;
        const double x296 = x18*x229;
        const double x297 = x20*x229;
        const double x298 = x246 + x247 - x248 - x249 - x251 - x252 + x253 + x254 + x255 + x256 - x258 - x259 - x260 - x261 - x262 - x263 - x264 + x265 + x266 + x267 + x269 + x270 + x271 + x272 + x273 + x277 + x280 + x284 + x287 - x289 - x291 + x293 + x294 + x295 + x296 + x297;
        const double x299 = x245 + x298;
        const double x300 = 4*x188;
        const double x301 = R_l_inv_20*x300 + 2*x198;
        const double x302 = x301*x37;
        const double x303 = -x94*(R_l_inv_21*x300 + 2*x194);
        const double x304 = x133*x196;
        const double x305 = x304*x37;
        const double x306 = -x302 + x303 + x305;
        const double x307 = R_l_inv_27*x300 + 2*x213;
        const double x308 = x110*x307;
        const double x309 = x1*x211;
        const double x310 = d_4*x34;
        const double x311 = x220*x32;
        const double x312 = 4*x17;
        const double x313 = x312*x5;
        const double x314 = 4*x23;
        const double x315 = x314*x5;
        const double x316 = x19*x34;
        const double x317 = x25*x34;
        const double x318 = -x308 - x309 - x310 + x311 - x313 - x315 + x316 + x317;
        const double x319 = R_l_inv_26*x300 + 2*x219;
        const double x320 = x319*x81;
        const double x321 = r_22*x45;
        const double x322 = 2*x321;
        const double x323 = r_22*x46;
        const double x324 = 2*x323;
        const double x325 = 2*x250;
        const double x326 = r_22*x325;
        const double x327 = r_22*x44;
        const double x328 = 2*x327;
        const double x329 = std::pow(r_22, 3)*x65;
        const double x330 = 2*x329;
        const double x331 = 2*x214;
        const double x332 = x331*x85;
        const double x333 = r_22*x60;
        const double x334 = 2*x333;
        const double x335 = r_22*x64;
        const double x336 = 2*x335;
        const double x337 = r_22*x70;
        const double x338 = 2*x337;
        const double x339 = r_22*x72;
        const double x340 = 2*x339;
        const double x341 = r_22*x62;
        const double x342 = 2*x341;
        const double x343 = r_22*x66;
        const double x344 = 2*x343;
        const double x345 = r_22*x68;
        const double x346 = 2*x345;
        const double x347 = r_22*x71;
        const double x348 = 2*x347;
        const double x349 = x14*x312;
        const double x350 = x14*x314;
        const double x351 = x312*x9;
        const double x352 = x11*x312;
        const double x353 = x13*x312;
        const double x354 = d_4*x3;
        const double x355 = x19*x354;
        const double x356 = x314*x9;
        const double x357 = x11*x314;
        const double x358 = x13*x314;
        const double x359 = x25*x354;
        const double x360 = 4*r_11;
        const double x361 = r_21*x360;
        const double x362 = x275*x361;
        const double x363 = 4*r_12;
        const double x364 = r_13*x279;
        const double x365 = x363*x364;
        const double x366 = 4*x282;
        const double x367 = r_21*r_31;
        const double x368 = x366*x367;
        const double x369 = x285*x366;
        const double x370 = x292*x3;
        const double x371 = x290*x3;
        const double x372 = x16*x314;
        const double x373 = x22*x312;
        const double x374 = x288*x3;
        const double x375 = x26*x312;
        const double x376 = x20*x314;
        const double x377 = x320 - x322 - x324 - x326 + x328 + x330 - x332 - x334 - x336 - x338 - x340 + x342 + x344 + x346 + x348 - x349 - x350 + x351 + x352 + x353 + x355 + x356 + x357 + x358 + x359 + x362 + x365 + x368 + x369 - x370 - x371 + x372 + x373 + x374 + x375 + x376;
        const double x378 = -x195;
        const double x379 = -x209;
        const double x380 = x193 + x197 + x201 + x202 + x205 + x207 + x378 + x379;
        const double x381 = -x212 + x215 + x221 + x225 + x227 + x230 - x232 - x233;
        const double x382 = x218 + x224 + x381;
        const double x383 = -x246 - x247 + x248 + x249 + x251 + x252 - x253 - x254 - x255 - x256 + x258 + x259 + x260 + x261 + x262 + x263 + x264 - x265 - x266 - x267 - x269 - x270 - x271 - x272 - x273 - x277 - x280 - x284 - x287 + x289 + x291 - x293 - x294 - x295 - x296 - x297;
        const double x384 = x245 + x383;
        const double x385 = x123*x301;
        const double x386 = x94*(R_l_inv_22*x300 + 2*x191);
        const double x387 = -x192*x92;
        const double x388 = -d_2*x304;
        const double x389 = x158*x208;
        const double x390 = x385 - x386 + x387 + x388 - x389;
        const double x391 = x220*x29;
        const double x392 = 4*d_4;
        const double x393 = x392*x4;
        const double x394 = x178*x19;
        const double x395 = x178*x25;
        const double x396 = x133*x77;
        const double x397 = x133*x78;
        const double x398 = x391 + x393 - x394 - x395 + x396 + x397;
        const double x399 = x162*x319;
        const double x400 = r_23*x46;
        const double x401 = 2*x400;
        const double x402 = std::pow(r_23, 3)*x65;
        const double x403 = 2*x402;
        const double x404 = r_23*x44;
        const double x405 = 2*x404;
        const double x406 = r_23*x45;
        const double x407 = 2*x406;
        const double x408 = r_23*x325;
        const double x409 = r_23*x64;
        const double x410 = 2*x409;
        const double x411 = r_23*x66;
        const double x412 = 2*x411;
        const double x413 = r_23*x67;
        const double x414 = 2*x413;
        const double x415 = r_23*x72;
        const double x416 = 2*x415;
        const double x417 = x331*x5;
        const double x418 = r_23*x60;
        const double x419 = 2*x418;
        const double x420 = r_23*x62;
        const double x421 = 2*x420;
        const double x422 = r_23*x70;
        const double x423 = 2*x422;
        const double x424 = r_23*x71;
        const double x425 = 2*x424;
        const double x426 = x392*x9;
        const double x427 = x11*x392;
        const double x428 = x13*x392;
        const double x429 = 4*x19;
        const double x430 = x429*x9;
        const double x431 = x11*x429;
        const double x432 = x13*x429;
        const double x433 = 4*x25;
        const double x434 = x433*x9;
        const double x435 = x11*x433;
        const double x436 = x13*x433;
        const double x437 = r_13*x58;
        const double x438 = x361*x437;
        const double x439 = r_13*x3;
        const double x440 = x275*x439;
        const double x441 = r_33*x69;
        const double x442 = 4*x367*x441;
        const double x443 = x282*x3;
        const double x444 = r_33*x443;
        const double x445 = x15*x173;
        const double x446 = x17*x354;
        const double x447 = x14*x429;
        const double x448 = x173*x21;
        const double x449 = x23*x354;
        const double x450 = x14*x433;
        const double x451 = x16*x433;
        const double x452 = x17*x3;
        const double x453 = x25*x452;
        const double x454 = x22*x429;
        const double x455 = x23*x3;
        const double x456 = x19*x455;
        const double x457 = x20*x433;
        const double x458 = x176*x21;
        const double x459 = x106*x314;
        const double x460 = x399 - x401 - x403 + x405 + x407 + x408 - x410 - x412 - x414 - x416 + x417 + x419 + x421 + x423 + x425 + x426 + x427 + x428 - x430 - x431 - x432 - x434 - x435 - x436 - x438 - x440 - x442 - x444 + x445 + x446 + x447 + x448 + x449 + x450 - x451 - x453 - x454 - x456 - x457 + x458 + x459;
        const double x461 = x178*x220;
        const double x462 = 8*d_4;
        const double x463 = x28*x462;
        const double x464 = 8*x28;
        const double x465 = x19*x464;
        const double x466 = x25*x464;
        const double x467 = 8*a_3;
        const double x468 = x106*x467;
        const double x469 = x107*x467;
        const double x470 = 8*x188;
        const double x471 = x180*x37;
        const double x472 = -x196*x471 - x37*(R_l_inv_22*x470 + 4*x191);
        const double x473 = x385 + x386 + x387 + x388 + x389;
        const double x474 = -x391 - x393 + x394 + x395 - x396 - x397;
        const double x475 = x193 + x197 + x200 + x204 + x206;
        const double x476 = x195 + x202 + x209 + x475;
        const double x477 = x235 - x236 - x237 - x238 - x239 - x240 - x241 - x242 - x243 - x244;
        const double x478 = x383 + x477;
        const double x479 = x302 + x303 + x305;
        const double x480 = -x320 + x322 + x324 + x326 - x328 - x330 + x332 + x334 + x336 + x338 + x340 - x342 - x344 - x346 - x348 + x349 + x350 - x351 - x352 - x353 - x355 - x356 - x357 - x358 - x359 - x362 - x365 - x368 - x369 + x370 + x371 - x372 - x373 - x374 - x375 - x376;
        const double x481 = x298 + x477;
        const double x482 = x203 + x378 + x379 + x475;
        const double x483 = R_l_inv_70*x190;
        const double x484 = R_l_inv_30*x189 + x483;
        const double x485 = a_3*x484;
        const double x486 = x484*x94;
        const double x487 = R_l_inv_72*x190;
        const double x488 = R_l_inv_32*x189 + x487;
        const double x489 = x123*x488;
        const double x490 = -d_2*x488;
        const double x491 = R_l_inv_71*x190;
        const double x492 = x37*(R_l_inv_31*x189 + x491);
        const double x493 = -x492;
        const double x494 = R_l_inv_35*x189 + R_l_inv_75*x190;
        const double x495 = -x494*x73;
        const double x496 = x494*x94;
        const double x497 = x496*x92;
        const double x498 = -x497;
        const double x499 = x126*x494;
        const double x500 = x485 + x486 + x489 + x490 + x493 + x495 + x498 + x499;
        const double x501 = R_l_inv_77*x190;
        const double x502 = R_l_inv_37*x189 + x501;
        const double x503 = x502*x89;
        const double x504 = x143*x17;
        const double x505 = x143*x23;
        const double x506 = x15*x29;
        const double x507 = x21*x29;
        const double x508 = x503 - x504 - x505 + x506 + x507;
        const double x509 = R_l_inv_33*x189 + R_l_inv_73*x190;
        const double x510 = r_21*x509;
        const double x511 = R_l_inv_76*x190;
        const double x512 = R_l_inv_36*x189 + x511;
        const double x513 = x110*x512;
        const double x514 = x257*x502;
        const double x515 = x105*x125;
        const double x516 = x54*x85;
        const double x517 = x57*x85;
        const double x518 = x5*x52;
        const double x519 = x5*x56;
        const double x520 = x510 + x513 + x514 + x515 - x516 - x517 + x518 + x519;
        const double x521 = x508 + x520;
        const double x522 = x502*x81;
        const double x523 = R_l_inv_34*x189 + R_l_inv_74*x190;
        const double x524 = r_22*x523;
        const double x525 = x512*x85;
        const double x526 = x52*x9;
        const double x527 = x11*x52;
        const double x528 = x13*x52;
        const double x529 = x166*x19;
        const double x530 = x56*x9;
        const double x531 = x11*x56;
        const double x532 = x13*x56;
        const double x533 = x166*x25;
        const double x534 = r_11*x275;
        const double x535 = x1*x534;
        const double x536 = r_23*x275*x278;
        const double x537 = r_31*x1;
        const double x538 = x282*x537;
        const double x539 = x282*x285;
        const double x540 = 2*x539;
        const double x541 = x17*x50;
        const double x542 = x23*x50;
        const double x543 = x1*x15;
        const double x544 = x23*x543;
        const double x545 = x1*x21;
        const double x546 = x17*x545;
        const double x547 = x18*x56;
        const double x548 = x26*x52;
        const double x549 = x20*x56;
        const double x550 = x118*x229;
        const double x551 = x169*x25;
        const double x552 = x321 + x323 + x327 - x329 + x333 + x335 + x337 + x339 - x341 - x343 - x345 - x347 + x522 - x524 + x525 - x526 - x527 - x528 - x529 - x530 - x531 - x532 - x533 - x535 - x536 - x538 - x540 + x541 + x542 - x544 - x546 - x547 - x548 - x549 + x550 + x551;
        const double x553 = R_l_inv_30*x300 + 2*x483;
        const double x554 = x37*x553;
        const double x555 = x94*(R_l_inv_31*x300 + 2*x491);
        const double x556 = x133*x494;
        const double x557 = -x37*x556;
        const double x558 = x554 + x555 + x557;
        const double x559 = R_l_inv_37*x300 + 2*x501;
        const double x560 = x110*x559;
        const double x561 = x1*x523;
        const double x562 = x32*x512;
        const double x563 = x1*x46;
        const double x564 = 2*x247;
        const double x565 = x1*x62;
        const double x566 = x1*x64;
        const double x567 = x1*x71;
        const double x568 = x1*x72;
        const double x569 = x1*x60;
        const double x570 = x1*x67;
        const double x571 = x1*x68;
        const double x572 = x1*x70;
        const double x573 = 4*x15;
        const double x574 = x14*x573;
        const double x575 = 4*x21;
        const double x576 = x14*x575;
        const double x577 = x573*x9;
        const double x578 = x11*x573;
        const double x579 = x13*x573;
        const double x580 = x173*x19;
        const double x581 = x575*x9;
        const double x582 = x11*x575;
        const double x583 = x13*x575;
        const double x584 = x173*x25;
        const double x585 = x3*x534;
        const double x586 = x360*x364;
        const double x587 = r_31*x443;
        const double x588 = 4*r_31*x286;
        const double x589 = x116*x314;
        const double x590 = x174*x25;
        const double x591 = x16*x575;
        const double x592 = x15*x455;
        const double x593 = x26*x573;
        const double x594 = x21*x452;
        const double x595 = x20*x575;
        const double x596 = -x563 + x564 - x565 - x566 - x567 - x568 + x569 + x570 + x571 + x572 - x574 - x576 + x577 + x578 + x579 + x580 + x581 + x582 + x583 + x584 + x585 + x586 + x587 + x588 - x589 - x590 + x591 + x592 + x593 + x594 + x595;
        const double x597 = x1*x44;
        const double x598 = x1*x45;
        const double x599 = -x597 - x598;
        const double x600 = x560 + x561 - x562 + x596 + x599;
        const double x601 = R_l_inv_36*x300 + 2*x511;
        const double x602 = x601*x81;
        const double x603 = 2*x509;
        const double x604 = r_22*x603;
        const double x605 = d_2*x173;
        const double x606 = 2*x502;
        const double x607 = x606*x85;
        const double x608 = x5*x573;
        const double x609 = x5*x575;
        const double x610 = d_2*x174;
        const double x611 = d_2*x175;
        const double x612 = -x602 + x604 - x605 + x607 - x608 - x609 + x610 + x611;
        const double x613 = -x486;
        const double x614 = x485 + x489 + x490 + x492 + x495 + x497 + x499 + x613;
        const double x615 = -x510;
        const double x616 = -x513;
        const double x617 = -x514;
        const double x618 = -x515;
        const double x619 = -x518;
        const double x620 = -x519;
        const double x621 = x508 + x516 + x517 + x615 + x616 + x617 + x618 + x619 + x620;
        const double x622 = -x321 - x323 - x327 + x329 - x333 - x335 - x337 - x339 + x341 + x343 + x345 + x347 - x522 + x524 - x525 + x526 + x527 + x528 + x529 + x530 + x531 + x532 + x533 + x535 + x536 + x538 + x540 - x541 - x542 + x544 + x546 + x547 + x548 + x549 - x550 - x551;
        const double x623 = x94*(R_l_inv_32*x300 + 2*x487);
        const double x624 = -x123*x553;
        const double x625 = x488*x92;
        const double x626 = d_2*x556;
        const double x627 = x158*x496;
        const double x628 = x623 + x624 + x625 + x626 + x627;
        const double x629 = a_3*x34;
        const double x630 = x29*x512;
        const double x631 = -x629 - x630;
        const double x632 = x162*x601;
        const double x633 = r_23*x603;
        const double x634 = x5*x606;
        const double x635 = x15*x34;
        const double x636 = x21*x34;
        const double x637 = x116*x158;
        const double x638 = x117*x158;
        const double x639 = -x632 - x633 - x634 - x635 - x636 + x637 + x638;
        const double x640 = x178*x512;
        const double x641 = x180*x4;
        const double x642 = x37*(R_l_inv_32*x470 + 4*x487) + x471*x494;
        const double x643 = x629 + x630;
        const double x644 = -x623 + x624 + x625 + x626 - x627;
        const double x645 = -x485 - x489 + x490 + x495 - x499;
        const double x646 = x552 + x645;
        const double x647 = x493 + x498 + x613;
        const double x648 = -x554 + x555 + x557;
        const double x649 = x602 - x604 + x605 - x607 + x608 + x609 - x610 - x611;
        const double x650 = x622 + x645;
        const double x651 = x486 + x492 + x497;
        const double x652 = a_3*x32;
        const double x653 = -x652;
        const double x654 = d_4*x143;
        const double x655 = -x654;
        const double x656 = x77*x92;
        const double x657 = -x656;
        const double x658 = x78*x92;
        const double x659 = -x658;
        const double x660 = x143*x19;
        const double x661 = x143*x25;
        const double x662 = x653 + x655 + x657 + x659 + x660 + x661;
        const double x663 = x125*x9;
        const double x664 = x11*x125;
        const double x665 = x125*x13;
        const double x666 = x49*x5;
        const double x667 = x15*x32;
        const double x668 = x52*x85;
        const double x669 = x5*x54;
        const double x670 = x21*x32;
        const double x671 = x56*x85;
        const double x672 = x5*x57;
        const double x673 = -x663 - x664 - x665 + x666 - x667 - x668 - x669 - x670 - x671 - x672;
        const double x674 = x49*x9;
        const double x675 = x11*x49;
        const double x676 = x13*x49;
        const double x677 = x15*x268;
        const double x678 = x166*x17;
        const double x679 = x19*x50;
        const double x680 = x21*x268;
        const double x681 = x166*x23;
        const double x682 = x25*x50;
        const double x683 = x54*x9;
        const double x684 = x11*x54;
        const double x685 = x13*x54;
        const double x686 = x57*x9;
        const double x687 = x11*x57;
        const double x688 = x13*x57;
        const double x689 = r_11*x1*x437;
        const double x690 = x276*x278;
        const double x691 = x441*x537;
        const double x692 = 2*r_33*x283;
        const double x693 = x229*x77;
        const double x694 = x167*x23;
        const double x695 = x25*x543;
        const double x696 = x18*x57;
        const double x697 = x19*x545;
        const double x698 = x24*x54;
        const double x699 = x20*x57;
        const double x700 = x400 + x402 - x404 + x406 + x409 + x411 + x413 + x415 - x418 - x420 - x422 - x424 - x674 - x675 - x676 - x677 - x678 - x679 - x680 - x681 - x682 + x683 + x684 + x685 + x686 + x687 + x688 + x689 + x690 + x691 + x692 - x693 - x694 + x695 + x696 + x697 + x698 + x699;
        const double x701 = x673 + x700;
        const double x702 = -x85;
        const double x703 = -x105 - x106 - x107 + x108 + x109;
        const double x704 = x654 + x656 + x658 - x660 - x661;
        const double x705 = x652 + x704;
        const double x706 = x133*x14;
        const double x707 = x133*x9;
        const double x708 = x11*x133;
        const double x709 = x13*x133;
        const double x710 = x15*x178;
        const double x711 = a_3*x3;
        const double x712 = x17*x711;
        const double x713 = x133*x20;
        const double x714 = x178*x21;
        const double x715 = x23*x711;
        const double x716 = x133*x26;
        const double x717 = -x706 + x707 + x708 + x709 + x710 + x712 + x713 + x714 + x715 + x716;
        const double x718 = x597 + x598;
        const double x719 = x596 + x718;
        const double x720 = x3*x44;
        const double x721 = x3*x45;
        const double x722 = x3*x46;
        const double x723 = 8*x14;
        const double x724 = x17*x723;
        const double x725 = 8*x105;
        const double x726 = x19*x725;
        const double x727 = x25*x725;
        const double x728 = x23*x723;
        const double x729 = 8*x23;
        const double x730 = x16*x729;
        const double x731 = 8*x21;
        const double x732 = x118*x731;
        const double x733 = 8*x17;
        const double x734 = x22*x733;
        const double x735 = x18*x729;
        const double x736 = x26*x733;
        const double x737 = 8*x25;
        const double x738 = x108*x737;
        const double x739 = x20*x729;
        const double x740 = 4*x329;
        const double x741 = x733*x9;
        const double x742 = x11*x733;
        const double x743 = x13*x733;
        const double x744 = 8*r_12;
        const double x745 = r_11*r_21;
        const double x746 = x58*x744*x745;
        const double x747 = x364*x744;
        const double x748 = x729*x9;
        const double x749 = x11*x729;
        const double x750 = x13*x729;
        const double x751 = 8*x282;
        const double x752 = x367*x751;
        const double x753 = 8*x539;
        const double x754 = x3*x60;
        const double x755 = x3*x62;
        const double x756 = x3*x64;
        const double x757 = x3*x66;
        const double x758 = x3*x68;
        const double x759 = x3*x70;
        const double x760 = x3*x71;
        const double x761 = x3*x72;
        const double x762 = x563 - x564 + x565 + x566 + x567 + x568 - x569 - x570 - x571 - x572 + x574 + x576 - x577 - x578 - x579 - x580 - x581 - x582 - x583 - x584 - x585 - x586 - x587 - x588 + x589 + x590 - x591 - x592 - x593 - x594 - x595;
        const double x763 = x599 + x762;
        const double x764 = x653 + x704;
        const double x765 = -x400 - x402 + x404 - x406 - x409 - x411 - x413 - x415 + x418 + x420 + x422 + x424 + x674 + x675 + x676 + x677 + x678 + x679 + x680 + x681 + x682 - x683 - x684 - x685 - x686 - x687 - x688 - x689 - x690 - x691 - x692 + x693 + x694 - x695 - x696 - x697 - x698 - x699;
        const double x766 = x673 + x765;
        const double x767 = x652 + x655 + x657 + x659 + x660 + x661;
        const double x768 = -x278;
        const double x769 = r_11*x92;
        const double x770 = d_2*x278;
        const double x771 = 2*Px;
        const double x772 = 2*r_12;
        const double x773 = -d_4*x278 + r_11*x51 + x21*x274 + x23*x772 + x25*x278 + x278*x53 + x36*x772 + x59*x771 + x61*x771 + x63*x771;
        const double x774 = -x770 + x773;
        const double x775 = r_12*x133;
        const double x776 = -x769;
        const double x777 = d_2*x360;
        const double x778 = d_2*x744;
        const double x779 = x770 + x773;
        const double x780 = d_4*r_11;
        const double x781 = r_13*x93;
        const double x782 = r_13*x21;
        const double x783 = r_11*x53;
        const double x784 = r_11*x25;
        const double x785 = x780 + x781 + x782 - x783 - x784;
        const double x786 = x139*x785;
        const double x787 = x137*x772;
        const double x788 = r_12*x125;
        const double x789 = x788*x86;
        const double x790 = a_3*x278;
        const double x791 = -x790*x82;
        const double x792 = -x786 + x787 - x789 + x791;
        const double x793 = r_12*x51;
        const double x794 = r_12*x21;
        const double x795 = 2*x794;
        const double x796 = r_11*x36;
        const double x797 = 2*x796;
        const double x798 = r_11*x23;
        const double x799 = 2*x798;
        const double x800 = -x793 - x795 + x797 + x799;
        const double x801 = r_12*x92;
        const double x802 = R_l_inv_53*a_0;
        const double x803 = x274*x802;
        const double x804 = d_4*r_12;
        const double x805 = r_13*x36;
        const double x806 = r_13*x23;
        const double x807 = r_12*x53;
        const double x808 = r_12*x25;
        const double x809 = x804 + x805 + x806 - x807 - x808;
        const double x810 = x146*x809;
        const double x811 = r_11*x125;
        const double x812 = x811*x82;
        const double x813 = -x801 - x803 - x810 - x812;
        const double x814 = a_3*x360;
        const double x815 = x363*x802;
        const double x816 = 4*x86;
        const double x817 = x785*x816;
        const double x818 = d_2*r_12;
        const double x819 = 4*x82;
        const double x820 = x818*x819;
        const double x821 = -x137*x360 + x777*x86 - x809*x819;
        const double x822 = x801 + x803 + x810 + x812;
        const double x823 = x786 - x787 + x789 + x791;
        const double x824 = 4*x804;
        const double x825 = Py*x439;
        const double x826 = 4*x806;
        const double x827 = 4*x807;
        const double x828 = 4*x808;
        const double x829 = x775*x86;
        const double x830 = r_12*x93;
        const double x831 = -x794 + x796 + x798 - x830;
        const double x832 = 4*r_13;
        const double x833 = d_2*x832;
        const double x834 = x802*x832 - x816*x831 + x82*x833;
        const double x835 = 8*x780;
        const double x836 = 8*x783;
        const double x837 = 8*x781;
        const double x838 = 8*x784;
        const double x839 = 8*x782;
        const double x840 = r_11*x467;
        const double x841 = x793 + x795 - x797 - x799;
        const double x842 = x307*x785;
        const double x843 = x211*x772;
        const double x844 = r_13*x133;
        const double x845 = -d_2*x844;
        const double x846 = d_4*x777;
        const double x847 = -x214*x790;
        const double x848 = x220*x788;
        const double x849 = x158*x781;
        const double x850 = x158*x782;
        const double x851 = Py*x360;
        const double x852 = x5*x851;
        const double x853 = x25*x777;
        const double x854 = -x842 + x843 + x845 - x846 + x847 - x848 - x849 - x850 + x852 + x853;
        const double x855 = d_4*x844;
        const double x856 = Px*x133;
        const double x857 = x59*x856;
        const double x858 = x61*x856;
        const double x859 = x63*x856;
        const double x860 = x4*x851;
        const double x861 = Py*x711;
        const double x862 = r_12*x861;
        const double x863 = x53*x844;
        const double x864 = x21*x814;
        const double x865 = x23*x775;
        const double x866 = x25*x844;
        const double x867 = -x855 + x857 + x858 + x859 + x860 + x862 + x863 + x864 + x865 + x866;
        const double x868 = x319*x809;
        const double x869 = x274*x45;
        const double x870 = x274*x46;
        const double x871 = r_11*x325;
        const double x872 = x274*x44;
        const double x873 = std::pow(r_11, 3);
        const double x874 = 2*x58;
        const double x875 = x873*x874;
        const double x876 = x214*x811;
        const double x877 = x274*x67;
        const double x878 = x274*x68;
        const double x879 = x274*x71;
        const double x880 = x274*x72;
        const double x881 = x274*x62;
        const double x882 = x274*x64;
        const double x883 = x274*x66;
        const double x884 = x274*x70;
        const double x885 = x392*x781;
        const double x886 = x392*x782;
        const double x887 = Px*x59;
        const double x888 = 4*x93;
        const double x889 = x887*x888;
        const double x890 = Px*x888;
        const double x891 = x61*x890;
        const double x892 = x63*x890;
        const double x893 = x575*x887;
        const double x894 = Px*x575;
        const double x895 = x61*x894;
        const double x896 = x63*x894;
        const double x897 = x14*x851;
        const double x898 = d_4*x25;
        const double x899 = x360*x898;
        const double x900 = r_21*x65;
        const double x901 = r_12*x3;
        const double x902 = x900*x901;
        const double x903 = r_31*x282;
        const double x904 = x363*x903;
        const double x905 = r_23*x900;
        const double x906 = x832*x905;
        const double x907 = r_31*x441;
        const double x908 = x832*x907;
        const double x909 = Py*x3;
        const double x910 = x798*x909;
        const double x911 = x25*x53;
        const double x912 = x360*x911;
        const double x913 = x21*x93;
        const double x914 = x360*x913;
        const double x915 = x314*x830;
        const double x916 = x794*x909;
        const double x917 = x433*x781;
        const double x918 = 4*x53;
        const double x919 = x782*x918;
        const double x920 = -x868 - x869 - x870 - x871 + x872 + x875 - x876 - x877 - x878 - x879 - x880 + x881 + x882 + x883 + x884 - x885 - x886 + x889 + x891 + x892 + x893 + x895 + x896 + x897 + x899 + x902 + x904 + x906 + x908 - x910 - x912 + x914 + x915 + x916 + x917 + x919;
        const double x921 = R_l_inv_26*x470 + 4*x219;
        const double x922 = x785*x921;
        const double x923 = x363*x45;
        const double x924 = x363*x46;
        const double x925 = x250*x363;
        const double x926 = x363*x44;
        const double x927 = std::pow(r_12, 3);
        const double x928 = 4*x58;
        const double x929 = x927*x928;
        const double x930 = 4*x818;
        const double x931 = x214*x930;
        const double x932 = x363*x66;
        const double x933 = x363*x68;
        const double x934 = x363*x70;
        const double x935 = x363*x72;
        const double x936 = x363*x60;
        const double x937 = x363*x64;
        const double x938 = x363*x67;
        const double x939 = x363*x71;
        const double x940 = x462*x805;
        const double x941 = x462*x806;
        const double x942 = 8*x36;
        const double x943 = x887*x942;
        const double x944 = Px*x942;
        const double x945 = x61*x944;
        const double x946 = x63*x944;
        const double x947 = x729*x887;
        const double x948 = Px*x729;
        const double x949 = x61*x948;
        const double x950 = x63*x948;
        const double x951 = Py*x744;
        const double x952 = x14*x951;
        const double x953 = x744*x898;
        const double x954 = r_22*x65;
        const double x955 = 8*x954;
        const double x956 = x745*x955;
        const double x957 = r_11*r_31*x751;
        const double x958 = r_13*r_23;
        const double x959 = x955*x958;
        const double x960 = r_13*r_33;
        const double x961 = x751*x960;
        const double x962 = x744*x913;
        const double x963 = x744*x911;
        const double x964 = 8*x93;
        const double x965 = x798*x964;
        const double x966 = x731*x796;
        const double x967 = x23*x744;
        const double x968 = x36*x967;
        const double x969 = x737*x805;
        const double x970 = 8*x53;
        const double x971 = x806*x970;
        const double x972 = -d_4*x778 - x180*x805 - x180*x806 - x211*x360 + x220*x777 + x25*x778 + x5*x951 - x809*(R_l_inv_27*x470 + 4*x213);
        const double x973 = x842 - x843 + x845 + x846 + x847 + x848 + x849 + x850 - x852 - x853;
        const double x974 = x868 + x869 + x870 + x871 - x872 - x875 + x876 + x877 + x878 + x879 + x880 - x881 - x882 - x883 - x884 + x885 + x886 - x889 - x891 - x892 - x893 - x895 - x896 - x897 - x899 - x902 - x904 - x906 - x908 + x910 + x912 - x914 - x915 - x916 - x917 - x919;
        const double x975 = x220*x775;
        const double x976 = a_3*x835;
        const double x977 = x467*x783;
        const double x978 = x467*x784;
        const double x979 = Py*r_13;
        const double x980 = 8*x4*x979;
        const double x981 = x467*x782;
        const double x982 = std::pow(r_13, 3);
        const double x983 = Px*x462;
        const double x984 = Px*x970;
        const double x985 = Px*x737;
        const double x986 = x65*x745;
        const double x987 = 8*r_11;
        const double x988 = r_33*x282;
        const double x989 = d_4*x744;
        const double x990 = r_13*x25;
        const double x991 = x36*x744;
        const double x992 = -r_23*x744*x954 - 8*r_23*x986 + x21*x835 - x21*x836 + x21*x837 + x214*x833 + x23*x989 - x25*x991 + x250*x832 + x36*x989 + x44*x832 + x45*x832 - x46*x832 + x462*x990 - x53*x967 + x59*x983 - x60*x832 + x61*x983 - x61*x984 - x61*x985 - x62*x832 + x63*x983 - x63*x984 - x63*x985 + x66*x832 + x67*x832 - x68*x832 + x70*x832 + x71*x832 - x72*x832 + x723*x979 + x729*x805 - x737*x887 - x744*x988 - x831*x921 + x835*x93 - x838*x93 - x887*x970 - x907*x987 - x928*x982 - x970*x990;
        const double x993 = x855 - x857 - x858 - x859 - x860 - x862 - x863 - x864 - x865 - x866;
        const double x994 = x601*x809;
        const double x995 = r_11*x603;
        const double x996 = x502*x811;
        const double x997 = d_2*x824;
        const double x998 = Py*x363;
        const double x999 = x5*x998;
        const double x1000 = x25*x930;
        const double x1001 = x34*x979;
        const double x1002 = x158*x806;
        const double x1003 = -r_11*x861 + x21*x775 - x23*x814 + x4*x998 + x502*x790;
        const double x1004 = -x1000 + x1001 + x1002 + x1003 + x994 + x995 + x996 + x997 - x999;
        const double x1005 = x559*x785;
        const double x1006 = x523*x772;
        const double x1007 = x874*x927;
        const double x1008 = x44*x772;
        const double x1009 = x45*x772;
        const double x1010 = x46*x772;
        const double x1011 = x60*x772;
        const double x1012 = x64*x772;
        const double x1013 = x67*x772;
        const double x1014 = x71*x772;
        const double x1015 = x512*x788;
        const double x1016 = x66*x772;
        const double x1017 = x68*x772;
        const double x1018 = x70*x772;
        const double x1019 = x72*x772;
        const double x1020 = x887*x909;
        const double x1021 = Px*x909;
        const double x1022 = x1021*x61;
        const double x1023 = x1021*x63;
        const double x1024 = x314*x887;
        const double x1025 = Px*x314;
        const double x1026 = x1025*x61;
        const double x1027 = x1025*x63;
        const double x1028 = x14*x998;
        const double x1029 = x25*x824;
        const double x1030 = x3*x986;
        const double x1031 = x360*x903;
        const double x1032 = r_23*x65;
        const double x1033 = x1032*x439;
        const double x1034 = x832*x988;
        const double x1035 = d_4*x825;
        const double x1036 = x392*x806;
        const double x1037 = x360*x93;
        const double x1038 = x1037*x23;
        const double x1039 = r_11*x21*x909;
        const double x1040 = Py*x23*x901;
        const double x1041 = x25*x825;
        const double x1042 = x53*x826;
        const double x1043 = x575*x830;
        const double x1044 = x25*x827;
        const double x1045 = x1005 - x1006 - x1007 + x1008 + x1009 + x1010 - x1011 - x1012 - x1013 - x1014 + x1015 + x1016 + x1017 + x1018 + x1019 - x1020 - x1022 - x1023 - x1024 - x1026 - x1027 - x1028 - x1029 - x1030 - x1031 - x1033 - x1034 + x1035 + x1036 - x1038 - x1039 - x1040 - x1041 - x1042 + x1043 + x1044;
        const double x1046 = R_l_inv_36*x470 + 4*x511;
        const double x1047 = x1046*x785;
        const double x1048 = x363*x509;
        const double x1049 = d_2*x835;
        const double x1050 = x502*x930;
        const double x1051 = x180*x781;
        const double x1052 = x180*x782;
        const double x1053 = Py*x5;
        const double x1054 = x1053*x987;
        const double x1055 = x180*x784;
        const double x1056 = x360*x46;
        const double x1057 = x873*x928;
        const double x1058 = x360*x67;
        const double x1059 = x360*x68;
        const double x1060 = x360*x71;
        const double x1061 = x360*x72;
        const double x1062 = x360*x62;
        const double x1063 = x360*x64;
        const double x1064 = x360*x66;
        const double x1065 = x360*x70;
        const double x1066 = x462*x781;
        const double x1067 = x462*x782;
        const double x1068 = x887*x964;
        const double x1069 = Px*x964;
        const double x1070 = x1069*x61;
        const double x1071 = x1069*x63;
        const double x1072 = x731*x887;
        const double x1073 = Px*x731;
        const double x1074 = x1073*x61;
        const double x1075 = x1073*x63;
        const double x1076 = Py*r_11*x723;
        const double x1077 = x25*x835;
        const double x1078 = r_22*x744*x900;
        const double x1079 = x744*x903;
        const double x1080 = 8*r_13;
        const double x1081 = x1080*x905;
        const double x1082 = x1080*x907;
        const double x1083 = x729*x796;
        const double x1084 = x25*x836;
        const double x1085 = x913*x987;
        const double x1086 = x93*x967;
        const double x1087 = x21*x991;
        const double x1088 = x25*x837;
        const double x1089 = x53*x839;
        const double x1090 = -x1056 + x1057 - x1058 - x1059 - x1060 - x1061 + x1062 + x1063 + x1064 + x1065 - x1066 - x1067 + x1068 + x1070 + x1071 + x1072 + x1074 + x1075 + x1076 + x1077 + x1078 + x1079 + x1081 + x1082 - x1083 - x1084 + x1085 + x1086 + x1087 + x1088 + x1089;
        const double x1091 = x360*x44;
        const double x1092 = x360*x45;
        const double x1093 = -x1091 - x1092;
        const double x1094 = x1090 + x1093 + x360*x523 - x512*x777 + x809*(R_l_inv_37*x470 + 4*x501);
        const double x1095 = x1000 - x1001 - x1002 + x1003 - x994 - x995 - x996 - x997 + x999;
        const double x1096 = -x1005 + x1006 + x1007 - x1008 - x1009 - x1010 + x1011 + x1012 + x1013 + x1014 - x1015 - x1016 - x1017 - x1018 - x1019 + x1020 + x1022 + x1023 + x1024 + x1026 + x1027 + x1028 + x1029 + x1030 + x1031 + x1033 + x1034 - x1035 - x1036 + x1038 + x1039 + x1040 + x1041 + x1042 - x1043 - x1044;
        const double x1097 = a_3*x778;
        const double x1098 = x512*x775;
        const double x1099 = x1046*x831 + x180*x796 + x180*x798 - x21*x778 - x502*x833 - x509*x832 - x778*x93;
        const double x1100 = Px*x158;
        const double x1101 = -x1100*x59;
        const double x1102 = -x1100*x61;
        const double x1103 = -x1100*x63;
        const double x1104 = a_3*x777;
        const double x1105 = d_4*x833;
        const double x1106 = -x777*x93;
        const double x1107 = -x818*x909;
        const double x1108 = -x1053*x832;
        const double x1109 = -x21*x777;
        const double x1110 = -x23*x930;
        const double x1111 = -x25*x833;
        const double x1112 = x1101 + x1102 + x1103 - x1104 + x1105 + x1106 + x1107 + x1108 + x1109 + x1110 + x1111;
        const double x1113 = d_4*x814;
        const double x1114 = Py*x832;
        const double x1115 = x1114*x4;
        const double x1116 = x133*x782;
        const double x1117 = x53*x814;
        const double x1118 = x25*x814;
        const double x1119 = -x1113 - x1115 - x1116 + x1117 + x1118;
        const double x1120 = x278*x44;
        const double x1121 = x278*x45;
        const double x1122 = x278*x46;
        const double x1123 = x874*x982;
        const double x1124 = Px*x392;
        const double x1125 = x1124*x59;
        const double x1126 = x1124*x61;
        const double x1127 = x1124*x63;
        const double x1128 = x278*x66;
        const double x1129 = x278*x67;
        const double x1130 = x278*x70;
        const double x1131 = x278*x71;
        const double x1132 = x278*x60;
        const double x1133 = x278*x62;
        const double x1134 = x278*x68;
        const double x1135 = x278*x72;
        const double x1136 = d_4*x360;
        const double x1137 = x1136*x93;
        const double x1138 = x804*x909;
        const double x1139 = x1114*x14;
        const double x1140 = x1136*x21;
        const double x1141 = x23*x824;
        const double x1142 = x832*x898;
        const double x1143 = x887*x918;
        const double x1144 = Px*x918;
        const double x1145 = x1144*x61;
        const double x1146 = x1144*x63;
        const double x1147 = x433*x887;
        const double x1148 = Px*x433;
        const double x1149 = x1148*x61;
        const double x1150 = x1148*x63;
        const double x1151 = x1032*x361;
        const double x1152 = x360*x907;
        const double x1153 = x1032*x901;
        const double x1154 = x363*x988;
        const double x1155 = x575*x781;
        const double x1156 = x806*x909;
        const double x1157 = x1037*x25;
        const double x1158 = x21*x360*x53;
        const double x1159 = x808*x909;
        const double x1160 = x23*x827;
        const double x1161 = x832*x911;
        const double x1162 = -x1120 + x1121 + x1122 + x1123 - x1125 - x1126 - x1127 - x1128 - x1129 - x1130 - x1131 + x1132 + x1133 + x1134 + x1135 - x1137 - x1138 - x1139 - x1140 - x1141 - x1142 + x1143 + x1145 + x1146 + x1147 + x1149 + x1150 + x1151 + x1152 + x1153 + x1154 - x1155 - x1156 + x1157 + x1158 + x1159 + x1160 + x1161;
        const double x1163 = -x818;
        const double x1164 = x1113 + x1115 + x1116 - x1117 - x1118;
        const double x1165 = x1101 + x1102 + x1103 + x1104 + x1105 + x1106 + x1107 + x1108 + x1109 + x1110 + x1111;
        const double x1166 = r_13*x467;
        const double x1167 = Px*x467;
        const double x1168 = Py*x4*x987 + a_3*x967 - d_4*x1166 + x1166*x25 + x1166*x53 + x1167*x59 + x1167*x61 + x1167*x63 + x21*x840 + x28*x951;
        const double x1169 = 16*d_4;
        const double x1170 = 16*x25;
        const double x1171 = 16*x21;
        const double x1172 = 16*x36;
        const double x1173 = Px*x1172;
        const double x1174 = 16*x23;
        const double x1175 = Px*x1174;
        const double x1176 = 16*x954;
        const double x1177 = x1120 - x1121 - x1122 - x1123 + x1125 + x1126 + x1127 + x1128 + x1129 + x1130 + x1131 - x1132 - x1133 - x1134 - x1135 + x1137 + x1138 + x1139 + x1140 + x1141 + x1142 - x1143 - x1145 - x1146 - x1147 - x1149 - x1150 - x1151 - x1152 - x1153 - x1154 + x1155 + x1156 - x1157 - x1158 - x1159 - x1160 - x1161;
        const double x1178 = -x11 - x13 + x14 - x16 - x18 - x20 - x22 - x24 - x26 - x9;
        const double x1179 = x1178 + x30;
        const double x1180 = -x29;
        const double x1181 = x128 + x162 + x90;
        const double x1182 = x101 + x103 + x111 + x113 + x28;
        const double x1183 = a_1 + x152 + x39 + x41 + x75 + x96;
        const double x1184 = x138 + x140 - x141;
        const double x1185 = x151 + x43 + x84 + x88;
        const double x1186 = x100 + x104 + x112 + x114 + x98;
        const double x1187 = -x161 - x163 - x164;
        const double x1188 = x116 + x117 + x119 + x121 + x184 + x90;
        const double x1189 = x217 + x223;
        const double x1190 = x1189 + x381;
        const double x1191 = x308 + x309 + x310 - x311 + x313 + x315 - x316 - x317;
        const double x1192 = x1189 + x212 + x216 + x222 + x226 + x228 + x231 + x232 + x233;
        const double x1193 = -x399 + x401 + x403 - x405 - x407 - x408 + x410 + x412 + x414 + x416 - x417 - x419 - x421 - x423 - x425 - x426 - x427 - x428 + x430 + x431 + x432 + x434 + x435 + x436 + x438 + x440 + x442 + x444 - x445 - x446 - x447 - x448 - x449 - x450 + x451 + x453 + x454 + x456 + x457 - x458 - x459;
        const double x1194 = -x503 + x504 + x505 - x506 - x507;
        const double x1195 = x1194 + x516 + x517 + x615 + x616 + x617 + x618 + x619 + x620;
        const double x1196 = -x560 - x561 + x562 + x718 + x762;
        const double x1197 = x1194 + x520;
        const double x1198 = x632 + x633 + x634 + x635 + x636 - x637 - x638;
        const double x1199 = x663 + x664 + x665 - x666 + x667 + x668 + x669 + x670 + x671 + x672;
        const double x1200 = x1199 + x765;
        const double x1201 = x706 - x707 - x708 - x709 - x710 - x712 - x713 - x714 - x715 - x716;
        const double x1202 = x1199 + x700;
        
        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = x0;
        A(0, 2) = x0;
        A(0, 3) = x2;
        A(0, 4) = -x3;
        A(0, 5) = x1;
        A(0, 6) = r_23;
        A(0, 8) = r_23;
        A(1, 0) = x27 + x7;
        A(1, 1) = x29;
        A(1, 2) = x31 + x6;
        A(1, 3) = x33;
        A(1, 4) = -x34;
        A(1, 5) = x32;
        A(1, 6) = x27 + x35;
        A(1, 7) = x29;
        A(1, 8) = x31 + x5;
        A(2, 0) = x115 + x129 + x97;
        A(2, 1) = x136 + x142 + x149;
        A(2, 2) = x129 + x150 + x153;
        A(2, 3) = x160 + x165 + x172;
        A(2, 4) = x173 - x174 - x175 + x176 + x177 - x179 + x181;
        A(2, 5) = x165 + x182 + x183;
        A(2, 6) = x150 + x185 + x97;
        A(2, 7) = x142 + x186 + x187;
        A(2, 8) = x115 + x153 + x185;
        A(3, 0) = x210 + x234 + x299;
        A(3, 1) = x306 + x318 + x377;
        A(3, 2) = x380 + x382 + x384;
        A(3, 3) = x390 + x398 + x460;
        A(3, 4) = -x461 + x463 - x465 - x466 + x468 + x469 + x472;
        A(3, 5) = x460 + x473 + x474;
        A(3, 6) = x234 + x476 + x478;
        A(3, 7) = x318 + x479 + x480;
        A(3, 8) = x382 + x481 + x482;
        A(4, 0) = x500 + x521 + x552;
        A(4, 1) = x558 + x600 + x612;
        A(4, 2) = x614 + x621 + x622;
        A(4, 3) = x628 + x631 + x639;
        A(4, 4) = x640 + x641 + x642;
        A(4, 5) = x639 + x643 + x644;
        A(4, 6) = x621 + x646 + x647;
        A(4, 7) = x600 + x648 + x649;
        A(4, 8) = x521 + x650 + x651;
        A(5, 0) = x662 + x701;
        A(5, 1) = x133*(x702 + x703);
        A(5, 2) = x701 + x705;
        A(5, 3) = x717 + x719;
        A(5, 4) = x720 + x721 - x722 - x724 + x726 + x727 - x728 + x730 - x732 + x734 + x735 + x736 - x738 + x739 + x740 + x741 + x742 + x743 + x746 + x747 + x748 + x749 + x750 + x752 + x753 - x754 + x755 - x756 + x757 + x758 - x759 + x760 - x761;
        A(5, 5) = x717 + x763;
        A(5, 6) = x764 + x766;
        A(5, 7) = x133*(x110 + x702);
        A(5, 8) = x766 + x767;
        
        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x768;
        B(0, 2) = x768;
        B(0, 3) = -x360;
        B(0, 4) = -x744;
        B(0, 5) = x360;
        B(0, 6) = x278;
        B(0, 8) = x278;
        B(1, 0) = x769 + x774;
        B(1, 1) = x775;
        B(1, 2) = x774 + x776;
        B(1, 3) = -x777;
        B(1, 4) = -x778;
        B(1, 5) = x777;
        B(1, 6) = x769 + x779;
        B(1, 7) = x775;
        B(1, 8) = x776 + x779;
        B(2, 0) = x792 + x800 + x813;
        B(2, 1) = x814 - x815 + x817 - x820 + x821;
        B(2, 2) = x800 + x822 + x823;
        B(2, 3) = -x824 - x825 - x826 + x827 + x828 + x829 + x834;
        B(2, 4) = x835 - x836 + x837 - x838 + x839 - x840*x86;
        B(2, 5) = x824 + x825 + x826 - x827 - x828 - x829 + x834;
        B(2, 6) = x792 + x822 + x841;
        B(2, 7) = -x814 + x815 - x817 + x820 + x821;
        B(2, 8) = x813 + x823 + x841;
        B(3, 0) = x854 + x867 + x920;
        B(3, 1) = x922 - x923 - x924 - x925 + x926 + x929 - x931 - x932 - x933 - x934 - x935 + x936 + x937 + x938 + x939 - x940 - x941 + x943 + x945 + x946 + x947 + x949 + x950 + x952 + x953 + x956 + x957 + x959 + x961 - x962 - x963 + x965 + x966 + x968 + x969 + x971 + x972;
        B(3, 2) = x867 + x973 + x974;
        B(3, 3) = x975 + x976 - x977 - x978 + x980 + x981 + x992;
        B(3, 4) = x467*(-r_11*x220 + x23*x278 + x278*x36 + 2*x804 - 2*x807 - 2*x808);
        B(3, 5) = -x975 - x976 + x977 + x978 - x980 - x981 + x992;
        B(3, 6) = x854 + x974 + x993;
        B(3, 7) = -x922 + x923 + x924 + x925 - x926 - x929 + x931 + x932 + x933 + x934 + x935 - x936 - x937 - x938 - x939 + x940 + x941 - x943 - x945 - x946 - x947 - x949 - x950 - x952 - x953 - x956 - x957 - x959 - x961 + x962 + x963 - x965 - x966 - x968 - x969 - x971 + x972;
        B(3, 8) = x920 + x973 + x993;
        B(4, 0) = x1004 + x1045;
        B(4, 1) = -x1047 + x1048 - x1049 + x1050 - x1051 - x1052 + x1054 + x1055 + x1094;
        B(4, 2) = x1095 + x1096;
        B(4, 3) = -x1097 - x1098 + x1099;
        B(4, 4) = x840*(x125 + x512);
        B(4, 5) = x1097 + x1098 + x1099;
        B(4, 6) = x1045 + x1095;
        B(4, 7) = x1047 - x1048 + x1049 - x1050 + x1051 + x1052 - x1054 - x1055 + x1094;
        B(4, 8) = x1004 + x1096;
        B(5, 0) = x1112 + x1119 + x1162;
        B(5, 1) = x467*(x1163 - x804 - x805 - x806 + x807 + x808);
        B(5, 2) = x1162 + x1164 + x1165;
        B(5, 3) = x1090 + x1091 + x1092 + x1168;
        B(5, 4) = 16*Py*r_12*x14 + 16*r_11*x903 + r_12*x1172*x23 - x1169*x805 - x1169*x806 + x1170*x804 + x1170*x805 - x1170*x807 + x1171*x796 - x1171*x830 + x1172*x887 + x1173*x61 + x1173*x63 + x1174*x887 + x1175*x61 + x1175*x63 + x1176*x745 + x1176*x958 + 16*x282*x960 + x44*x744 + x45*x744 - x46*x744 + 16*x53*x806 + 8*x58*x927 + x60*x744 + x64*x744 - x66*x744 + x67*x744 - x68*x744 - x70*x744 + x71*x744 - x72*x744 + 16*x798*x93;
        B(5, 5) = x1056 - x1057 + x1058 + x1059 + x1060 + x1061 - x1062 - x1063 - x1064 - x1065 + x1066 + x1067 - x1068 - x1070 - x1071 - x1072 - x1074 - x1075 - x1076 - x1077 - x1078 - x1079 - x1081 - x1082 + x1083 + x1084 - x1085 - x1086 - x1087 - x1088 - x1089 + x1093 + x1168;
        B(5, 6) = x1112 + x1164 + x1177;
        B(5, 7) = x467*(x1163 + x809);
        B(5, 8) = x1119 + x1165 + x1177;
        
        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = r_23;
        C(0, 2) = r_23;
        C(0, 3) = x1;
        C(0, 4) = x3;
        C(0, 5) = x2;
        C(0, 6) = x0;
        C(0, 8) = x0;
        C(1, 0) = x1179 + x5;
        C(1, 1) = x1180;
        C(1, 2) = x1178 + x35;
        C(1, 3) = x32;
        C(1, 4) = x34;
        C(1, 5) = x33;
        C(1, 6) = x1179 + x6;
        C(1, 7) = x1180;
        C(1, 8) = x1178 + x7;
        C(2, 0) = x1181 + x1182 + x1183;
        C(2, 1) = x1184 + x136 + x187;
        C(2, 2) = x1181 + x1185 + x1186;
        C(2, 3) = x1187 + x160 + x183;
        C(2, 4) = -x173 + x174 + x175 - x176 - x177 + x179 + x181;
        C(2, 5) = x1187 + x172 + x182;
        C(2, 6) = x1183 + x1186 + x1188;
        C(2, 7) = x1184 + x149 + x186;
        C(2, 8) = x1182 + x1185 + x1188;
        C(3, 0) = x1190 + x210 + x478;
        C(3, 1) = x1191 + x306 + x480;
        C(3, 2) = x1192 + x380 + x481;
        C(3, 3) = x1193 + x390 + x474;
        C(3, 4) = x461 - x463 + x465 + x466 - x468 - x469 + x472;
        C(3, 5) = x1193 + x398 + x473;
        C(3, 6) = x1190 + x299 + x476;
        C(3, 7) = x1191 + x377 + x479;
        C(3, 8) = x1192 + x384 + x482;
        C(4, 0) = x1195 + x500 + x622;
        C(4, 1) = x1196 + x558 + x649;
        C(4, 2) = x1197 + x552 + x614;
        C(4, 3) = x1198 + x628 + x643;
        C(4, 4) = -x640 - x641 + x642;
        C(4, 5) = x1198 + x631 + x644;
        C(4, 6) = x1197 + x647 + x650;
        C(4, 7) = x1196 + x612 + x648;
        C(4, 8) = x1195 + x646 + x651;
        C(5, 0) = x1200 + x705;
        C(5, 1) = x133*(x110 + x85);
        C(5, 2) = x1200 + x662;
        C(5, 3) = x1201 + x763;
        C(5, 4) = -x720 - x721 + x722 + x724 - x726 - x727 + x728 - x730 + x732 - x734 - x735 - x736 + x738 - x739 - x740 - x741 - x742 - x743 - x746 - x747 - x748 - x749 - x750 - x752 - x753 + x754 - x755 + x756 - x757 - x758 + x759 - x760 + x761;
        C(5, 5) = x1201 + x719;
        C(5, 6) = x1202 + x767;
        C(5, 7) = x133*(x703 + x85);
        C(5, 8) = x1202 + x764;
        
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
            const double th_0 = this_solution[0];
            const bool checked_result = std::fabs(a_3*r_11*std::sin(th_0) - a_3*r_21*std::cos(th_0)) <= 9.9999999999999995e-7 && std::fabs(a_3*r_12*std::sin(th_0) - a_3*r_22*std::cos(th_0)) <= 9.9999999999999995e-7 && std::fabs(Px*std::sin(th_0) - Py*std::cos(th_0) - d_4*r_13*std::sin(th_0) + d_4*r_23*std::cos(th_0)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor();
    // Finish code for equation all-zero dispatcher node 2
    
    // Code for explicit solution node 3, solved variable is th_5
    auto ExplicitSolutionNode_node_3_solve_th_5_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            
            const bool condition_0 = std::fabs(a_3*r_11*std::sin(th_0) - a_3*r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(a_3*r_12*std::sin(th_0) - a_3*r_22*std::cos(th_0)) >= zero_tolerance || std::fabs(Px*std::sin(th_0) - Py*std::cos(th_0) - d_4*r_13*std::sin(th_0) + d_4*r_23*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = a_3*x0;
                const double x2 = std::cos(th_0);
                const double x3 = a_3*x2;
                const double x4 = r_12*x1 - r_22*x3;
                const double x5 = -r_11*x1 + r_21*x3;
                const double x6 = std::atan2(x4, x5);
                const double x7 = -Px*x0 + Py*x2 + d_4*r_13*x0 - d_4*r_23*x2;
                const double x8 = std::sqrt(std::pow(x4, 2) + std::pow(x5, 2) - std::pow(x7, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[7] = x6 + std::atan2(x8, x7);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = std::fabs(a_3*r_11*std::sin(th_0) - a_3*r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(a_3*r_12*std::sin(th_0) - a_3*r_22*std::cos(th_0)) >= zero_tolerance || std::fabs(Px*std::sin(th_0) - Py*std::cos(th_0) - d_4*r_13*std::sin(th_0) + d_4*r_23*std::cos(th_0)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = a_3*x0;
                const double x2 = std::cos(th_0);
                const double x3 = a_3*x2;
                const double x4 = r_12*x1 - r_22*x3;
                const double x5 = -r_11*x1 + r_21*x3;
                const double x6 = std::atan2(x4, x5);
                const double x7 = -Px*x0 + Py*x2 + d_4*r_13*x0 - d_4*r_23*x2;
                const double x8 = std::sqrt(std::pow(x4, 2) + std::pow(x5, 2) - std::pow(x7, 2));
                // End of temp variables
                const double tmp_sol_value = x6 + std::atan2(-x8, x7);
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
    ExplicitSolutionNode_node_3_solve_th_5_processor();
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
            
            const bool condition_0 = std::fabs((-r_11*std::sin(th_0) + r_21*std::cos(th_0))*std::sin(th_5) + (-r_12*std::sin(th_0) + r_22*std::cos(th_0))*std::cos(th_5)) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = std::acos((-r_11*x1 + r_21*x0)*std::sin(th_5) + (-r_12*x1 + r_22*x0)*std::cos(th_5));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[5] = x2;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx);
            }
            
            const bool condition_1 = std::fabs((-r_11*std::sin(th_0) + r_21*std::cos(th_0))*std::sin(th_5) + (-r_12*std::sin(th_0) + r_22*std::cos(th_0))*std::cos(th_5)) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = std::acos((-r_11*x1 + r_21*x0)*std::sin(th_5) + (-r_12*x1 + r_22*x0)*std::cos(th_5));
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
            const double th_5 = this_solution[7];
            
            const bool condition_0 = 2*std::fabs(a_0*a_1) >= zero_tolerance || 2*std::fabs(a_0*d_2) >= zero_tolerance || std::fabs(std::pow(Px, 2) - 2*Px*a_3*r_11*std::cos(th_5) + 2*Px*a_3*r_12*std::sin(th_5) - 2*Px*d_4*r_13 + std::pow(Py, 2) - 2*Py*a_3*r_21*std::cos(th_5) + 2*Py*a_3*r_22*std::sin(th_5) - 2*Py*d_4*r_23 + std::pow(Pz, 2) - 2*Pz*a_3*r_31*std::cos(th_5) + 2*Pz*a_3*r_32*std::sin(th_5) - 2*Pz*d_4*r_33 - std::pow(a_0, 2) - std::pow(a_1, 2) + std::pow(a_3, 2)*std::pow(r_11, 2)*std::pow(std::cos(th_5), 2) - std::pow(a_3, 2)*r_11*r_12*std::sin(2*th_5) - std::pow(a_3, 2)*std::pow(r_12, 2)*std::pow(std::cos(th_5), 2) + std::pow(a_3, 2)*std::pow(r_12, 2) + std::pow(a_3, 2)*std::pow(r_21, 2)*std::pow(std::cos(th_5), 2) - std::pow(a_3, 2)*r_21*r_22*std::sin(2*th_5) - std::pow(a_3, 2)*std::pow(r_22, 2)*std::pow(std::cos(th_5), 2) + std::pow(a_3, 2)*std::pow(r_22, 2) + std::pow(a_3, 2)*std::pow(r_31, 2)*std::pow(std::cos(th_5), 2) - std::pow(a_3, 2)*r_31*r_32*std::sin(2*th_5) - std::pow(a_3, 2)*std::pow(r_32, 2)*std::pow(std::cos(th_5), 2) + std::pow(a_3, 2)*std::pow(r_32, 2) + 2*a_3*d_4*r_11*r_13*std::cos(th_5) - 2*a_3*d_4*r_12*r_13*std::sin(th_5) + 2*a_3*d_4*r_21*r_23*std::cos(th_5) - 2*a_3*d_4*r_22*r_23*std::sin(th_5) + 2*a_3*d_4*r_31*r_33*std::cos(th_5) - 2*a_3*d_4*r_32*r_33*std::sin(th_5) - std::pow(d_2, 2) + std::pow(d_4, 2)*std::pow(r_13, 2) + std::pow(d_4, 2)*std::pow(r_23, 2) + std::pow(d_4, 2)*std::pow(r_33, 2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_0;
                const double x1 = std::atan2(-d_2*x0, a_1*x0);
                const double x2 = std::pow(a_1, 2);
                const double x3 = std::pow(a_0, 2);
                const double x4 = 4*x3;
                const double x5 = std::pow(d_2, 2);
                const double x6 = 2*d_4;
                const double x7 = r_13*x6;
                const double x8 = r_23*x6;
                const double x9 = r_33*x6;
                const double x10 = std::pow(a_3, 2);
                const double x11 = std::pow(r_12, 2)*x10;
                const double x12 = std::pow(r_22, 2)*x10;
                const double x13 = std::pow(r_32, 2)*x10;
                const double x14 = std::pow(d_4, 2);
                const double x15 = std::cos(th_5);
                const double x16 = 2*a_3;
                const double x17 = x15*x16;
                const double x18 = std::sin(th_5);
                const double x19 = x16*x18;
                const double x20 = a_3*x15;
                const double x21 = a_3*x18;
                const double x22 = std::pow(x15, 2);
                const double x23 = x10*x22;
                const double x24 = x10*std::sin(2*th_5);
                const double x25 = std::pow(Px, 2) - Px*r_11*x17 + Px*r_12*x19 - Px*x7 + std::pow(Py, 2) - Py*r_21*x17 + Py*r_22*x19 - Py*x8 + std::pow(Pz, 2) - Pz*r_31*x17 + Pz*r_32*x19 - Pz*x9 + std::pow(r_11, 2)*x23 - r_11*r_12*x24 + r_11*x20*x7 - r_12*x21*x7 + std::pow(r_13, 2)*x14 + std::pow(r_21, 2)*x23 - r_21*r_22*x24 + r_21*x20*x8 - r_22*x21*x8 + std::pow(r_23, 2)*x14 + std::pow(r_31, 2)*x23 - r_31*r_32*x24 + r_31*x20*x9 - r_32*x21*x9 + std::pow(r_33, 2)*x14 - x11*x22 + x11 - x12*x22 + x12 - x13*x22 + x13 - x2 - x3 - x5;
                const double x26 = std::sqrt(x2*x4 - std::pow(x25, 2) + x4*x5);
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x1 + std::atan2(x26, x25);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(8, appended_idx);
            }
            
            const bool condition_1 = 2*std::fabs(a_0*a_1) >= zero_tolerance || 2*std::fabs(a_0*d_2) >= zero_tolerance || std::fabs(std::pow(Px, 2) - 2*Px*a_3*r_11*std::cos(th_5) + 2*Px*a_3*r_12*std::sin(th_5) - 2*Px*d_4*r_13 + std::pow(Py, 2) - 2*Py*a_3*r_21*std::cos(th_5) + 2*Py*a_3*r_22*std::sin(th_5) - 2*Py*d_4*r_23 + std::pow(Pz, 2) - 2*Pz*a_3*r_31*std::cos(th_5) + 2*Pz*a_3*r_32*std::sin(th_5) - 2*Pz*d_4*r_33 - std::pow(a_0, 2) - std::pow(a_1, 2) + std::pow(a_3, 2)*std::pow(r_11, 2)*std::pow(std::cos(th_5), 2) - std::pow(a_3, 2)*r_11*r_12*std::sin(2*th_5) - std::pow(a_3, 2)*std::pow(r_12, 2)*std::pow(std::cos(th_5), 2) + std::pow(a_3, 2)*std::pow(r_12, 2) + std::pow(a_3, 2)*std::pow(r_21, 2)*std::pow(std::cos(th_5), 2) - std::pow(a_3, 2)*r_21*r_22*std::sin(2*th_5) - std::pow(a_3, 2)*std::pow(r_22, 2)*std::pow(std::cos(th_5), 2) + std::pow(a_3, 2)*std::pow(r_22, 2) + std::pow(a_3, 2)*std::pow(r_31, 2)*std::pow(std::cos(th_5), 2) - std::pow(a_3, 2)*r_31*r_32*std::sin(2*th_5) - std::pow(a_3, 2)*std::pow(r_32, 2)*std::pow(std::cos(th_5), 2) + std::pow(a_3, 2)*std::pow(r_32, 2) + 2*a_3*d_4*r_11*r_13*std::cos(th_5) - 2*a_3*d_4*r_12*r_13*std::sin(th_5) + 2*a_3*d_4*r_21*r_23*std::cos(th_5) - 2*a_3*d_4*r_22*r_23*std::sin(th_5) + 2*a_3*d_4*r_31*r_33*std::cos(th_5) - 2*a_3*d_4*r_32*r_33*std::sin(th_5) - std::pow(d_2, 2) + std::pow(d_4, 2)*std::pow(r_13, 2) + std::pow(d_4, 2)*std::pow(r_23, 2) + std::pow(d_4, 2)*std::pow(r_33, 2)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_0;
                const double x1 = std::atan2(-d_2*x0, a_1*x0);
                const double x2 = std::pow(a_1, 2);
                const double x3 = std::pow(a_0, 2);
                const double x4 = 4*x3;
                const double x5 = std::pow(d_2, 2);
                const double x6 = 2*d_4;
                const double x7 = r_13*x6;
                const double x8 = r_23*x6;
                const double x9 = r_33*x6;
                const double x10 = std::pow(a_3, 2);
                const double x11 = std::pow(r_12, 2)*x10;
                const double x12 = std::pow(r_22, 2)*x10;
                const double x13 = std::pow(r_32, 2)*x10;
                const double x14 = std::pow(d_4, 2);
                const double x15 = std::cos(th_5);
                const double x16 = 2*a_3;
                const double x17 = x15*x16;
                const double x18 = std::sin(th_5);
                const double x19 = x16*x18;
                const double x20 = a_3*x15;
                const double x21 = a_3*x18;
                const double x22 = std::pow(x15, 2);
                const double x23 = x10*x22;
                const double x24 = x10*std::sin(2*th_5);
                const double x25 = std::pow(Px, 2) - Px*r_11*x17 + Px*r_12*x19 - Px*x7 + std::pow(Py, 2) - Py*r_21*x17 + Py*r_22*x19 - Py*x8 + std::pow(Pz, 2) - Pz*r_31*x17 + Pz*r_32*x19 - Pz*x9 + std::pow(r_11, 2)*x23 - r_11*r_12*x24 + r_11*x20*x7 - r_12*x21*x7 + std::pow(r_13, 2)*x14 + std::pow(r_21, 2)*x23 - r_21*r_22*x24 + r_21*x20*x8 - r_22*x21*x8 + std::pow(r_23, 2)*x14 + std::pow(r_31, 2)*x23 - r_31*r_32*x24 + r_31*x20*x9 - r_32*x21*x9 + std::pow(r_33, 2)*x14 - x11*x22 + x11 - x12*x22 + x12 - x13*x22 + x13 - x2 - x3 - x5;
                const double x26 = std::sqrt(x2*x4 - std::pow(x25, 2) + x4*x5);
                // End of temp variables
                const double tmp_sol_value = x1 + std::atan2(-x26, x25);
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
            
            const bool degenerate_valid_0 = std::fabs(th_2 - M_PI + 1.3408918986629901) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(22, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_2 - 2*M_PI + 1.3408918986629901) <= 9.9999999999999995e-7;
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
            const double th_0 = this_solution[0];
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(0.973688178796424*a_1 - 0.22788446737788701*d_2) >= 9.9999999999999995e-7 || std::fabs(a_0 + 0.22788446737788701*a_1 + 0.973688178796424*d_2) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_0 + 0.22788446737788701*a_1 + 0.973688178796424*d_2;
                const double x1 = a_3*std::sin(th_5);
                const double x2 = a_3*std::cos(th_5);
                const double x3 = Pz - d_4*r_33 - r_31*x2 + r_32*x1;
                const double x4 = 0.973688178796424*a_1 - 0.22788446737788701*d_2;
                const double x5 = std::cos(th_0);
                const double x6 = std::sin(th_0);
                const double x7 = Px*x5 + Py*x6 - d_4*r_13*x5 - d_4*r_23*x6 - r_11*x2*x5 + r_12*x1*x5 - r_21*x2*x6 + r_22*x1*x6;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x3 + x4*x7, x0*x7 + x3*x4);
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
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(-r_13*((0.973688178796424*std::sin(th_1) + 0.22788446737788601*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) - r_23*((0.973688178796424*std::sin(th_1) + 0.22788446737788601*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(-0.22788446737788601*std::sin(th_1) + 0.973688178796424*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance || std::fabs(-r_13*(-0.22788446737788701*std::sin(th_1) + 0.973688178796424*std::cos(th_1))*std::cos(th_0) - r_23*(-0.22788446737788701*std::sin(th_1) + 0.973688178796424*std::cos(th_1))*std::sin(th_0) + r_33*(0.973688178796424*std::sin(th_1) + 0.22788446737788701*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_1);
                const double x3 = 0.973688178796424*x2;
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_3);
                const double x6 = std::cos(th_0);
                const double x7 = 0.973688178796424*x1;
                const double x8 = x0*(0.22788446737788601*x2 + x7);
                const double x9 = -0.22788446737788701*x1 + x3;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*(-0.22788446737788601*x1 + x3), -r_13*x6*x9 - r_23*x4*x9 + r_33*(0.22788446737788701*x2 + x7));
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
            const double th_0 = this_solution[0];
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(0.973688178796424*a_1 - 0.22788446737788701*d_2) >= 9.9999999999999995e-7 || std::fabs(-a_0 + 0.22788446737788701*a_1 + 0.973688178796424*d_2) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_0 - 0.22788446737788701*a_1 - 0.973688178796424*d_2;
                const double x1 = a_3*std::sin(th_5);
                const double x2 = a_3*std::cos(th_5);
                const double x3 = Pz - d_4*r_33 - r_31*x2 + r_32*x1;
                const double x4 = -0.973688178796424*a_1 + 0.22788446737788701*d_2;
                const double x5 = std::cos(th_0);
                const double x6 = std::sin(th_0);
                const double x7 = Px*x5 + Py*x6 - d_4*r_13*x5 - d_4*r_23*x6 - r_11*x2*x5 + r_12*x1*x5 - r_21*x2*x6 + r_22*x1*x6;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x3 + x4*x7, x0*x7 + x3*x4);
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
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(-r_13*((-0.973688178796424*std::sin(th_1) - 0.22788446737788601*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) - r_23*((-0.973688178796424*std::sin(th_1) - 0.22788446737788601*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(0.22788446737788601*std::sin(th_1) - 0.973688178796424*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance || std::fabs(-r_13*(-0.22788446737788701*std::sin(th_1) + 0.973688178796424*std::cos(th_1))*std::cos(th_0) - r_23*(-0.22788446737788701*std::sin(th_1) + 0.973688178796424*std::cos(th_1))*std::sin(th_0) + r_33*(0.973688178796424*std::sin(th_1) + 0.22788446737788701*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_1);
                const double x3 = 0.973688178796424*x2;
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_3);
                const double x6 = std::cos(th_0);
                const double x7 = 0.973688178796424*x1;
                const double x8 = x0*(-0.22788446737788601*x2 - x7);
                const double x9 = -0.22788446737788701*x1 + x3;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_13*(x4*x5 + x6*x8) - r_23*(x4*x8 - x5*x6) - r_33*x0*(0.22788446737788601*x1 - x3), r_13*x6*x9 + r_23*x4*x9 - r_33*(0.22788446737788701*x2 + x7));
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
            
            const bool condition_0 = std::fabs(a_1*std::sin(th_2) + d_2*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_0 + a_1*std::cos(th_2) - d_2*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_2);
                const double x1 = std::cos(th_2);
                const double x2 = -a_1*x0 - d_2*x1;
                const double x3 = Px*std::cos(th_0) + Py*std::sin(th_0) + a_3*std::cos(th_1th_2th_4_soa) - d_4*std::sin(th_1th_2th_4_soa);
                const double x4 = a_0 + a_1*x1 - d_2*x0;
                const double x5 = Pz - a_3*r_31*std::cos(th_5) + a_3*r_32*std::sin(th_5) - d_4*r_33;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x2*x3 - x4*x5, x2*x5 + x3*x4);
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
            
            const bool condition_0 = std::fabs(a_1*std::sin(th_2) + d_2*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_0 + a_1*std::cos(th_2) - d_2*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_2);
                const double x1 = std::sin(th_2);
                const double x2 = a_0 + a_1*x0 - d_2*x1;
                const double x3 = a_3*std::sin(th_5);
                const double x4 = a_3*std::cos(th_5);
                const double x5 = Pz - d_4*r_33 - r_31*x4 + r_32*x3;
                const double x6 = -a_1*x1 - d_2*x0;
                const double x7 = std::cos(th_0);
                const double x8 = std::sin(th_0);
                const double x9 = Px*x7 + Py*x8 - d_4*r_13*x7 - d_4*r_23*x8 - r_11*x4*x7 + r_12*x3*x7 - r_21*x4*x8 + r_22*x3*x8;
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
                const double x1 = std::cos(th_0);
                const double x2 = std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(-r_13*x2 + r_23*x1), x0*(-(-r_11*x2 + r_21*x1)*std::cos(th_5) + (-r_12*x2 + r_22*x1)*std::sin(th_5)));
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

}; // struct abb_crb_15000_ik

// Code below for debug
void test_ik_solve_abb_crb_15000()
{
    std::array<double, abb_crb_15000_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = abb_crb_15000_ik::computeFK(theta);
    auto ik_output = abb_crb_15000_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = abb_crb_15000_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_abb_crb_15000();
}
