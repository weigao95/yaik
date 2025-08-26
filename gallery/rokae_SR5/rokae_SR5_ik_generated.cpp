#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct rokae_SR5_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 128;
static constexpr int n_tree_nodes = 34;
static constexpr int intermediate_solution_size = 6;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_2 = 0.403113;
static constexpr double a_3 = 0.05;
static constexpr double d_3 = 0.4;
static constexpr double d_4 = -0.136;
static constexpr double d_5 = 0.1035;
static constexpr double pre_transform_s0 = 0.328;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -1.44644;
static constexpr double th_2_offset_original2raw = 1.69515;
static constexpr double th_3_offset_original2raw = 0.0;
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
    ee_transformed(0, 0) = r_11;
    ee_transformed(0, 1) = r_12;
    ee_transformed(0, 2) = r_13;
    ee_transformed(0, 3) = Px;
    ee_transformed(1, 0) = r_21;
    ee_transformed(1, 1) = r_22;
    ee_transformed(1, 2) = r_23;
    ee_transformed(1, 3) = Py;
    ee_transformed(2, 0) = r_31;
    ee_transformed(2, 1) = r_32;
    ee_transformed(2, 2) = r_33;
    ee_transformed(2, 3) = Pz - pre_transform_s0;
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
    ee_transformed(0, 0) = r_11;
    ee_transformed(0, 1) = r_12;
    ee_transformed(0, 2) = r_13;
    ee_transformed(0, 3) = Px;
    ee_transformed(1, 0) = r_21;
    ee_transformed(1, 1) = r_22;
    ee_transformed(1, 2) = r_23;
    ee_transformed(1, 3) = Py;
    ee_transformed(2, 0) = r_31;
    ee_transformed(2, 1) = r_32;
    ee_transformed(2, 2) = r_33;
    ee_transformed(2, 3) = Pz + pre_transform_s0;
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
    const double x15 = x6*x8;
    const double x16 = x15*x4;
    const double x17 = x5*x9;
    const double x18 = x17*x4;
    const double x19 = x16 - x18;
    const double x20 = std::cos(th_4);
    const double x21 = x1*x3 + x11*x2;
    const double x22 = -x14*x19 + x20*x21;
    const double x23 = -x14*x21 - x19*x20;
    const double x24 = a_2*x8;
    const double x25 = x1*x10 + x1*x7;
    const double x26 = -x2*x4 - x25*x3;
    const double x27 = x1*x15;
    const double x28 = x1*x17;
    const double x29 = x27 - x28;
    const double x30 = x2*x25 - x3*x4;
    const double x31 = -x14*x29 + x20*x30;
    const double x32 = -x14*x30 - x20*x29;
    const double x33 = x15 - x17;
    const double x34 = x3*x33;
    const double x35 = -x10 - x7;
    const double x36 = x2*x33;
    const double x37 = -x14*x35 + x20*x36;
    const double x38 = -x14*x36 - x20*x35;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x12 + x13*x22;
    ee_pose_raw(0, 1) = -x0*x22 - x12*x13;
    ee_pose_raw(0, 2) = x23;
    ee_pose_raw(0, 3) = a_3*x11 - d_3*(-x16 + x18) + d_4*x12 + d_5*x23 + x24*x4;
    ee_pose_raw(1, 0) = -x0*x26 + x13*x31;
    ee_pose_raw(1, 1) = -x0*x31 - x13*x26;
    ee_pose_raw(1, 2) = x32;
    ee_pose_raw(1, 3) = a_3*x25 - d_3*(-x27 + x28) + d_4*x26 + d_5*x32 + x1*x24;
    ee_pose_raw(2, 0) = x0*x34 + x13*x37;
    ee_pose_raw(2, 1) = -x0*x37 + x13*x34;
    ee_pose_raw(2, 2) = x38;
    ee_pose_raw(2, 3) = -a_2*x5 + a_3*x33 - d_3*(x10 + x7) - d_4*x34 + d_5*x38;
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
    const double x1 = std::cos(th_0);
    const double x2 = std::sin(th_2);
    const double x3 = std::cos(th_1);
    const double x4 = x2*x3;
    const double x5 = x1*x4;
    const double x6 = std::sin(th_1);
    const double x7 = std::cos(th_2);
    const double x8 = x6*x7;
    const double x9 = x1*x8;
    const double x10 = x5 - x9;
    const double x11 = std::cos(th_3);
    const double x12 = std::sin(th_3);
    const double x13 = x2*x6;
    const double x14 = x3*x7;
    const double x15 = x1*x13 + x1*x14;
    const double x16 = x0*x11 - x12*x15;
    const double x17 = std::cos(th_4);
    const double x18 = std::sin(th_4);
    const double x19 = -x10*x17 - x18*(x0*x12 + x11*x15);
    const double x20 = x0*x4;
    const double x21 = x0*x8;
    const double x22 = x20 - x21;
    const double x23 = x0*x13 + x0*x14;
    const double x24 = -x1*x11 - x12*x23;
    const double x25 = -x17*x22 - x18*(-x1*x12 + x11*x23);
    const double x26 = -x13 - x14;
    const double x27 = x4 - x8;
    const double x28 = x12*x27;
    const double x29 = -x11*x18*x27 - x17*x26;
    const double x30 = -a_2*x6 + pre_transform_s0;
    const double x31 = a_3*x27 - d_3*(x13 + x14) + x30;
    const double x32 = a_2*x3;
    const double x33 = a_3*x23 - d_3*(-x20 + x21) + x0*x32;
    const double x34 = -d_4*x28 + x31;
    const double x35 = d_4*x24 + x33;
    const double x36 = d_5*x29 + x34;
    const double x37 = d_5*x25 + x35;
    const double x38 = a_3*x15 - d_3*(-x5 + x9) + x1*x32;
    const double x39 = d_4*x16 + x38;
    const double x40 = d_5*x19 + x39;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = -x0;
    jacobian(0, 2) = x0;
    jacobian(0, 3) = x10;
    jacobian(0, 4) = x16;
    jacobian(0, 5) = x19;
    jacobian(1, 1) = x1;
    jacobian(1, 2) = -x1;
    jacobian(1, 3) = x22;
    jacobian(1, 4) = x24;
    jacobian(1, 5) = x25;
    jacobian(2, 0) = 1;
    jacobian(2, 3) = x26;
    jacobian(2, 4) = -x28;
    jacobian(2, 5) = x29;
    jacobian(3, 1) = -pre_transform_s0*x1;
    jacobian(3, 2) = x1*x30;
    jacobian(3, 3) = -x22*x31 + x26*x33;
    jacobian(3, 4) = -x24*x34 - x28*x35;
    jacobian(3, 5) = -x25*x36 + x29*x37;
    jacobian(4, 1) = -pre_transform_s0*x0;
    jacobian(4, 2) = x0*x30;
    jacobian(4, 3) = x10*x31 - x26*x38;
    jacobian(4, 4) = x16*x34 + x28*x39;
    jacobian(4, 5) = x19*x36 - x29*x40;
    jacobian(5, 2) = -std::pow(x0, 2)*x32 - std::pow(x1, 2)*x32;
    jacobian(5, 3) = -x10*x33 + x22*x38;
    jacobian(5, 4) = -x16*x35 + x24*x39;
    jacobian(5, 5) = -x19*x37 + x25*x40;
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
    const double x1 = std::cos(th_0);
    const double x2 = std::sin(th_2);
    const double x3 = std::cos(th_1);
    const double x4 = x2*x3;
    const double x5 = std::sin(th_1);
    const double x6 = std::cos(th_2);
    const double x7 = x5*x6;
    const double x8 = x1*x4 - x1*x7;
    const double x9 = std::cos(th_3);
    const double x10 = std::sin(th_3);
    const double x11 = x2*x5;
    const double x12 = x3*x6;
    const double x13 = x1*x11 + x1*x12;
    const double x14 = std::cos(th_4);
    const double x15 = std::sin(th_4);
    const double x16 = x0*x4 - x0*x7;
    const double x17 = x0*x11 + x0*x12;
    const double x18 = -x11 - x12;
    const double x19 = x4 - x7;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = -x0;
    jacobian(0, 2) = x0;
    jacobian(0, 3) = x8;
    jacobian(0, 4) = x0*x9 - x10*x13;
    jacobian(0, 5) = -x14*x8 - x15*(x0*x10 + x13*x9);
    jacobian(1, 1) = x1;
    jacobian(1, 2) = -x1;
    jacobian(1, 3) = x16;
    jacobian(1, 4) = -x1*x9 - x10*x17;
    jacobian(1, 5) = -x14*x16 - x15*(-x1*x10 + x17*x9);
    jacobian(2, 0) = 1;
    jacobian(2, 3) = x18;
    jacobian(2, 4) = -x10*x19;
    jacobian(2, 5) = -x14*x18 - x15*x19*x9;
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
    const double x0 = std::cos(th_0);
    const double x1 = p_on_ee_z*x0;
    const double x2 = std::sin(th_1);
    const double x3 = -a_2*x2 + pre_transform_s0;
    const double x4 = std::sin(th_2);
    const double x5 = x2*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::cos(th_2);
    const double x8 = x6*x7;
    const double x9 = -x5 - x8;
    const double x10 = std::sin(th_0);
    const double x11 = x4*x6;
    const double x12 = x10*x11;
    const double x13 = x2*x7;
    const double x14 = x10*x13;
    const double x15 = x12 - x14;
    const double x16 = x11 - x13;
    const double x17 = a_3*x16 - d_3*(x5 + x8) + x3;
    const double x18 = a_2*x6;
    const double x19 = x10*x5 + x10*x8;
    const double x20 = a_3*x19 - d_3*(-x12 + x14) + x10*x18;
    const double x21 = std::sin(th_3);
    const double x22 = x16*x21;
    const double x23 = std::cos(th_3);
    const double x24 = -x0*x23 - x19*x21;
    const double x25 = -d_4*x22 + x17;
    const double x26 = d_4*x24 + x20;
    const double x27 = std::cos(th_4);
    const double x28 = std::sin(th_4);
    const double x29 = -x16*x23*x28 - x27*x9;
    const double x30 = -x15*x27 - x28*(-x0*x21 + x19*x23);
    const double x31 = d_5*x29 + x25;
    const double x32 = d_5*x30 + x26;
    const double x33 = p_on_ee_z*x10;
    const double x34 = x0*x11;
    const double x35 = x0*x13;
    const double x36 = x34 - x35;
    const double x37 = x0*x5 + x0*x8;
    const double x38 = a_3*x37 - d_3*(-x34 + x35) + x0*x18;
    const double x39 = x10*x23 - x21*x37;
    const double x40 = d_4*x39 + x38;
    const double x41 = -x27*x36 - x28*(x10*x21 + x23*x37);
    const double x42 = d_5*x41 + x40;
    const double x43 = p_on_ee_x*x0;
    const double x44 = p_on_ee_y*x10;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -p_on_ee_y;
    jacobian(0, 1) = -pre_transform_s0*x0 + x1;
    jacobian(0, 2) = x0*x3 - x1;
    jacobian(0, 3) = -p_on_ee_y*x9 + p_on_ee_z*x15 - x15*x17 + x20*x9;
    jacobian(0, 4) = p_on_ee_y*x22 + p_on_ee_z*x24 - x22*x26 - x24*x25;
    jacobian(0, 5) = -p_on_ee_y*x29 + p_on_ee_z*x30 + x29*x32 - x30*x31;
    jacobian(1, 0) = p_on_ee_x;
    jacobian(1, 1) = -pre_transform_s0*x10 + x33;
    jacobian(1, 2) = x10*x3 - x33;
    jacobian(1, 3) = p_on_ee_x*x9 - p_on_ee_z*x36 + x17*x36 - x38*x9;
    jacobian(1, 4) = -p_on_ee_x*x22 - p_on_ee_z*x39 + x22*x40 + x25*x39;
    jacobian(1, 5) = p_on_ee_x*x29 - p_on_ee_z*x41 - x29*x42 + x31*x41;
    jacobian(2, 1) = -x43 - x44;
    jacobian(2, 2) = -std::pow(x0, 2)*x18 - std::pow(x10, 2)*x18 + x43 + x44;
    jacobian(2, 3) = -p_on_ee_x*x15 + p_on_ee_y*x36 + x15*x38 - x20*x36;
    jacobian(2, 4) = -p_on_ee_x*x24 + p_on_ee_y*x39 + x24*x40 - x26*x39;
    jacobian(2, 5) = -p_on_ee_x*x30 + p_on_ee_y*x41 + x30*x42 - x32*x41;
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
        R_l(0, 0) = d_4*r_21;
        R_l(0, 1) = d_4*r_22;
        R_l(0, 2) = d_4*r_11;
        R_l(0, 3) = d_4*r_12;
        R_l(0, 4) = Py - d_5*r_23;
        R_l(0, 5) = Px - d_5*r_13;
        R_l(1, 0) = d_4*r_11;
        R_l(1, 1) = d_4*r_12;
        R_l(1, 2) = -d_4*r_21;
        R_l(1, 3) = -d_4*r_22;
        R_l(1, 4) = Px - d_5*r_13;
        R_l(1, 5) = -Py + d_5*r_23;
        R_l(2, 6) = d_4*r_31;
        R_l(2, 7) = d_4*r_32;
        R_l(3, 0) = -r_21;
        R_l(3, 1) = -r_22;
        R_l(3, 2) = -r_11;
        R_l(3, 3) = -r_12;
        R_l(4, 0) = -r_11;
        R_l(4, 1) = -r_12;
        R_l(4, 2) = r_21;
        R_l(4, 3) = r_22;
        R_l(5, 6) = 2*Px*d_4*r_11 + 2*Py*d_4*r_21 + 2*Pz*d_4*r_31 - 2*d_4*d_5*r_11*r_13 - 2*d_4*d_5*r_21*r_23 - 2*d_4*d_5*r_31*r_33;
        R_l(5, 7) = 2*Px*d_4*r_12 + 2*Py*d_4*r_22 + 2*Pz*d_4*r_32 - 2*d_4*d_5*r_12*r_13 - 2*d_4*d_5*r_22*r_23 - 2*d_4*d_5*r_32*r_33;
        R_l(6, 0) = -Px*r_31 + Pz*r_11 - d_5*r_11*r_33 + d_5*r_13*r_31;
        R_l(6, 1) = -Px*r_32 + Pz*r_12 - d_5*r_12*r_33 + d_5*r_13*r_32;
        R_l(6, 2) = Py*r_31 - Pz*r_21 + d_5*r_21*r_33 - d_5*r_23*r_31;
        R_l(6, 3) = Py*r_32 - Pz*r_22 + d_5*r_22*r_33 - d_5*r_23*r_32;
        R_l(7, 0) = Py*r_31 - Pz*r_21 + d_5*r_21*r_33 - d_5*r_23*r_31;
        R_l(7, 1) = Py*r_32 - Pz*r_22 + d_5*r_22*r_33 - d_5*r_23*r_32;
        R_l(7, 2) = Px*r_31 - Pz*r_11 + d_5*r_11*r_33 - d_5*r_13*r_31;
        R_l(7, 3) = Px*r_32 - Pz*r_12 + d_5*r_12*r_33 - d_5*r_13*r_32;
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
        const double x0 = R_l_inv_66*r_31 + R_l_inv_76*r_32;
        const double x1 = d_3*x0;
        const double x2 = -x1;
        const double x3 = d_5*r_33;
        const double x4 = Pz - x3;
        const double x5 = R_l_inv_62*r_31 + R_l_inv_72*r_32;
        const double x6 = -x4*x5;
        const double x7 = R_l_inv_65*r_31 + R_l_inv_75*r_32;
        const double x8 = std::pow(Px, 2);
        const double x9 = std::pow(Py, 2);
        const double x10 = std::pow(Pz, 2);
        const double x11 = std::pow(d_4, 2);
        const double x12 = std::pow(r_11, 2);
        const double x13 = x11*x12;
        const double x14 = std::pow(r_21, 2);
        const double x15 = x11*x14;
        const double x16 = std::pow(r_31, 2);
        const double x17 = x11*x16;
        const double x18 = std::pow(d_5, 2);
        const double x19 = std::pow(r_13, 2)*x18;
        const double x20 = std::pow(r_23, 2)*x18;
        const double x21 = std::pow(r_33, 2)*x18;
        const double x22 = d_5*r_13;
        const double x23 = 2*Px;
        const double x24 = x22*x23;
        const double x25 = d_5*r_23;
        const double x26 = 2*Py;
        const double x27 = x25*x26;
        const double x28 = 2*Pz;
        const double x29 = x28*x3;
        const double x30 = std::pow(a_2, 2);
        const double x31 = std::pow(a_3, 2);
        const double x32 = std::pow(d_3, 2);
        const double x33 = -x30 - x31 - x32;
        const double x34 = x10 + x13 + x15 + x17 + x19 + x20 + x21 - x24 - x27 - x29 + x33 + x8 + x9;
        const double x35 = -x34*x7;
        const double x36 = 2*a_3;
        const double x37 = a_2*x36;
        const double x38 = x37*x7;
        const double x39 = -x38;
        const double x40 = x2 + x35 + x39 + x6;
        const double x41 = R_l_inv_60*r_31 + R_l_inv_70*r_32;
        const double x42 = a_3*x41;
        const double x43 = d_3*x5;
        const double x44 = x42 - x43;
        const double x45 = a_2*x41;
        const double x46 = -x45;
        const double x47 = R_l_inv_64*r_31;
        const double x48 = R_l_inv_74*r_32;
        const double x49 = -x47 - x48;
        const double x50 = x46 + x49;
        const double x51 = x0*x36;
        const double x52 = 2*a_2;
        const double x53 = x0*x52;
        const double x54 = -x53;
        const double x55 = -x5*x52;
        const double x56 = x36*x5;
        const double x57 = 2*d_3;
        const double x58 = x41*x57;
        const double x59 = x56 + x58;
        const double x60 = x55 + x59;
        const double x61 = -x42;
        const double x62 = x43 + x45 + x61;
        const double x63 = x1 + x35 + x39 + x6;
        const double x64 = 2*R_l_inv_63;
        const double x65 = r_31*x64;
        const double x66 = 2*R_l_inv_73;
        const double x67 = r_32*x66;
        const double x68 = R_l_inv_67*r_31 + R_l_inv_77*r_32;
        const double x69 = x57*x68;
        const double x70 = -x65 - x67 + x69;
        const double x71 = x65 + x67 + x69;
        const double x72 = x47 + x48;
        const double x73 = x46 + x72;
        const double x74 = -x51;
        const double x75 = d_4*x12;
        const double x76 = d_4*x14;
        const double x77 = d_4*x16;
        const double x78 = Px*r_11;
        const double x79 = Py*r_21;
        const double x80 = Pz*r_31;
        const double x81 = r_11*x22;
        const double x82 = r_21*x25;
        const double x83 = r_31*x3;
        const double x84 = x78 + x79 + x80 - x81 - x82 - x83;
        const double x85 = Px*r_12;
        const double x86 = Py*r_22;
        const double x87 = Pz*r_32;
        const double x88 = r_12*x22;
        const double x89 = r_22*x25;
        const double x90 = r_32*x3;
        const double x91 = x85 + x86 + x87 - x88 - x89 - x90;
        const double x92 = R_l_inv_66*x84 + R_l_inv_76*x91;
        const double x93 = d_3*x92;
        const double x94 = -x93;
        const double x95 = R_l_inv_62*x84 + R_l_inv_72*x91;
        const double x96 = -x4*x95;
        const double x97 = R_l_inv_65*x84 + R_l_inv_75*x91;
        const double x98 = -x34*x97;
        const double x99 = x37*x97;
        const double x100 = -x99;
        const double x101 = x100 + x75 + x76 + x77 + x94 + x96 + x98;
        const double x102 = R_l_inv_60*x84 + R_l_inv_70*x91;
        const double x103 = a_3*x102;
        const double x104 = d_3*x95;
        const double x105 = x103 - x104;
        const double x106 = a_2*x102;
        const double x107 = -x106;
        const double x108 = R_l_inv_64*x84;
        const double x109 = R_l_inv_74*x91;
        const double x110 = -x108 - x109;
        const double x111 = x107 + x110;
        const double x112 = x36*x92;
        const double x113 = x52*x92;
        const double x114 = -x113;
        const double x115 = -x52*x95;
        const double x116 = x36*x95;
        const double x117 = x102*x57;
        const double x118 = x116 + x117;
        const double x119 = x115 + x118;
        const double x120 = -x103;
        const double x121 = x104 + x106 + x120;
        const double x122 = x100 + x75 + x76 + x77 + x93 + x96 + x98;
        const double x123 = R_l_inv_67*x84 + R_l_inv_77*x91;
        const double x124 = x123*x57 - x36;
        const double x125 = x124 + x52;
        const double x126 = x64*x84;
        const double x127 = x66*x91;
        const double x128 = -x126 - x127;
        const double x129 = x126 + x127;
        const double x130 = x108 + x109;
        const double x131 = x107 + x130;
        const double x132 = -x112;
        const double x133 = Px*r_21;
        const double x134 = Py*r_11;
        const double x135 = r_11*x25 - r_21*x22 + x133 - x134;
        const double x136 = Px*r_22;
        const double x137 = Py*r_12;
        const double x138 = r_12*x25 - r_22*x22 + x136 - x137;
        const double x139 = R_l_inv_62*x135 + R_l_inv_72*x138;
        const double x140 = d_3*x139;
        const double x141 = x139*x4;
        const double x142 = R_l_inv_65*x135 + R_l_inv_75*x138;
        const double x143 = x142*x34;
        const double x144 = R_l_inv_60*x135 + R_l_inv_70*x138;
        const double x145 = a_3*x144;
        const double x146 = x140 + x141 + x143 - x145;
        const double x147 = R_l_inv_64*x135;
        const double x148 = R_l_inv_74*x138;
        const double x149 = x142*x37;
        const double x150 = x147 + x148 + x149;
        const double x151 = -a_3;
        const double x152 = a_2*x144;
        const double x153 = R_l_inv_66*x135 + R_l_inv_76*x138;
        const double x154 = d_3*x153;
        const double x155 = x151 + x152 + x154;
        const double x156 = x153*x52;
        const double x157 = x153*x36;
        const double x158 = -x157 - x57;
        const double x159 = x139*x52;
        const double x160 = x139*x36;
        const double x161 = x144*x57;
        const double x162 = -x160 - x161;
        const double x163 = x159 + x162;
        const double x164 = -x154;
        const double x165 = -a_2;
        const double x166 = a_3 + x165;
        const double x167 = x164 + x166;
        const double x168 = -x152;
        const double x169 = -x140 + x141 + x143 + x145;
        const double x170 = x168 + x169;
        const double x171 = x135*x64;
        const double x172 = x138*x66;
        const double x173 = R_l_inv_67*x135 + R_l_inv_77*x138;
        const double x174 = -x173*x57;
        const double x175 = x171 + x172 + x174;
        const double x176 = -x171 - x172 + x174;
        const double x177 = -x147 - x148;
        const double x178 = x149 + x177;
        const double x179 = -x156;
        const double x180 = x157 + x57;
        const double x181 = x151 + x154;
        const double x182 = Py*x12 + Py*x14 + Py*x16 - x12*x25 - x14*x25 - x16*x25;
        const double x183 = 2*d_4;
        const double x184 = x182*x183;
        const double x185 = Px*x12 + Px*x14 + Px*x16 - x12*x22 - x14*x22 - x16*x22;
        const double x186 = x183*x185;
        const double x187 = 2*x25;
        const double x188 = 2*x22;
        const double x189 = 2*r_11;
        const double x190 = r_23*x18;
        const double x191 = r_13*x190;
        const double x192 = 2*r_31;
        const double x193 = r_33*x190;
        const double x194 = std::pow(r_21, 3)*x11 - r_21*x10 + r_21*x13 + r_21*x17 - r_21*x19 + r_21*x20 - r_21*x21 + r_21*x29 - r_21*x8 + r_21*x9 + x133*x188 - x134*x188 - x187*x78 - x187*x79 - x187*x80 + x189*x191 + x192*x193 + x26*x78 + x26*x80 - x26*x83;
        const double x195 = 2*x191;
        const double x196 = r_13*r_33*x18;
        const double x197 = std::pow(r_11, 3)*x11 - r_11*x10 + r_11*x15 + r_11*x17 + r_11*x19 - r_11*x20 - r_11*x21 + r_11*x29 + r_11*x8 - r_11*x9 + r_21*x195 - x133*x187 + x134*x187 - x188*x78 - x188*x79 - x188*x80 + x192*x196 + x23*x79 + x23*x80 - x23*x83;
        const double x198 = r_21*x11;
        const double x199 = x189*x198;
        const double x200 = x192*x198;
        const double x201 = 2*r_32;
        const double x202 = r_12*x195 + r_12*x199 - r_22*x10 + r_22*x13 + 3*r_22*x15 + r_22*x17 - r_22*x19 + r_22*x20 - r_22*x21 + r_22*x29 - r_22*x8 + r_22*x9 + r_32*x200 + x136*x188 - x137*x188 - x187*x85 - x187*x86 - x187*x87 + x193*x201 + x26*x85 + x26*x87 - x26*x90;
        const double x203 = r_31*x11*x189;
        const double x204 = -r_12*x10 + 3*r_12*x13 + r_12*x15 + r_12*x17 + r_12*x19 - r_12*x20 - r_12*x21 + r_12*x29 + r_12*x8 - r_12*x9 + r_22*x195 + r_22*x199 + r_32*x203 - x136*x187 + x137*x187 - x188*x85 - x188*x86 - x188*x87 + x196*x201 + x23*x86 + x23*x87 - x23*x90;
        const double x205 = R_l_inv_06*x194 + R_l_inv_16*x202 + R_l_inv_26*x197 + R_l_inv_36*x204 + R_l_inv_46*x184 + R_l_inv_56*x186;
        const double x206 = d_3*x205;
        const double x207 = R_l_inv_02*x194 + R_l_inv_12*x202 + R_l_inv_22*x197 + R_l_inv_32*x204 + R_l_inv_42*x184 + R_l_inv_52*x186;
        const double x208 = x207*x4;
        const double x209 = R_l_inv_05*x194 + R_l_inv_15*x202 + R_l_inv_25*x197 + R_l_inv_35*x204 + R_l_inv_45*x184 + R_l_inv_55*x186;
        const double x210 = x209*x34;
        const double x211 = x209*x37;
        const double x212 = x206 + x208 + x210 + x211;
        const double x213 = R_l_inv_00*x194 + R_l_inv_10*x202 + R_l_inv_20*x197 + R_l_inv_30*x204 + R_l_inv_40*x184 + R_l_inv_50*x186;
        const double x214 = a_2*x213;
        const double x215 = d_3*x207;
        const double x216 = a_3*x213;
        const double x217 = x215 - x216;
        const double x218 = x214 + x217;
        const double x219 = R_l_inv_04*x194;
        const double x220 = R_l_inv_14*x202;
        const double x221 = R_l_inv_24*x197;
        const double x222 = R_l_inv_34*x204;
        const double x223 = R_l_inv_44*x184;
        const double x224 = R_l_inv_54*x186;
        const double x225 = x219 + x220 + x221 + x222 + x223 + x224;
        const double x226 = x205*x36;
        const double x227 = -x226;
        const double x228 = x205*x52;
        const double x229 = x207*x52;
        const double x230 = x207*x36;
        const double x231 = x213*x57;
        const double x232 = -x230 - x231;
        const double x233 = x229 + x232;
        const double x234 = -x214;
        const double x235 = -x215;
        const double x236 = x216 + x234 + x235;
        const double x237 = -x206;
        const double x238 = x208 + x210 + x211 + x237;
        const double x239 = 4*a_3;
        const double x240 = a_2*x239;
        const double x241 = R_l_inv_07*x194 + R_l_inv_17*x202 + R_l_inv_27*x197 + R_l_inv_37*x204 + R_l_inv_47*x184 + R_l_inv_57*x186;
        const double x242 = -x241*x57;
        const double x243 = -x240 + x242;
        const double x244 = 2*x32;
        const double x245 = 2*x30;
        const double x246 = 2*x31;
        const double x247 = 2*x194;
        const double x248 = R_l_inv_03*x247;
        const double x249 = 2*x202;
        const double x250 = R_l_inv_13*x249;
        const double x251 = 2*x197;
        const double x252 = R_l_inv_23*x251;
        const double x253 = 2*x204;
        const double x254 = R_l_inv_33*x253;
        const double x255 = 4*d_4;
        const double x256 = x182*x255;
        const double x257 = R_l_inv_43*x256;
        const double x258 = x185*x255;
        const double x259 = R_l_inv_53*x258;
        const double x260 = -x244 + x245 + x246 + x248 + x250 + x252 + x254 + x257 + x259;
        const double x261 = 8*d_3;
        const double x262 = x240 + x242;
        const double x263 = x244 - x245 - x246 - x248 - x250 - x252 - x254 - x257 - x259;
        const double x264 = -x219 - x220 - x221 - x222 - x223 - x224;
        const double x265 = -x228;
        const double x266 = R_l_inv_06*x197 + R_l_inv_16*x204 - R_l_inv_26*x194 - R_l_inv_36*x202 + R_l_inv_46*x186 - R_l_inv_56*x184;
        const double x267 = d_3*x266;
        const double x268 = R_l_inv_02*x197 + R_l_inv_12*x204 - R_l_inv_22*x194 - R_l_inv_32*x202 + R_l_inv_42*x186 - R_l_inv_52*x184;
        const double x269 = x268*x4;
        const double x270 = R_l_inv_05*x197 + R_l_inv_15*x204 - R_l_inv_25*x194 - R_l_inv_35*x202 + R_l_inv_45*x186 - R_l_inv_55*x184;
        const double x271 = x270*x34;
        const double x272 = x270*x37;
        const double x273 = x267 + x269 + x271 + x272;
        const double x274 = R_l_inv_00*x197 + R_l_inv_10*x204 - R_l_inv_20*x194 - R_l_inv_30*x202 + R_l_inv_40*x186 - R_l_inv_50*x184;
        const double x275 = a_2*x274;
        const double x276 = x275 + x37;
        const double x277 = d_3*x268;
        const double x278 = a_3*x274;
        const double x279 = x277 - x278;
        const double x280 = R_l_inv_04*x197;
        const double x281 = R_l_inv_14*x204;
        const double x282 = R_l_inv_24*x194;
        const double x283 = R_l_inv_34*x202;
        const double x284 = R_l_inv_54*x184;
        const double x285 = R_l_inv_44*x186;
        const double x286 = x280 + x281 - x282 - x283 - x284 + x285 + x33;
        const double x287 = x279 + x286;
        const double x288 = x268*x52;
        const double x289 = x266*x52;
        const double x290 = x288 + x289;
        const double x291 = x266*x36;
        const double x292 = -x291;
        const double x293 = x268*x36;
        const double x294 = x274*x57;
        const double x295 = -x293 - x294;
        const double x296 = x292 + x295;
        const double x297 = -x275;
        const double x298 = -x277;
        const double x299 = x278 + x297 + x298;
        const double x300 = -x267;
        const double x301 = x269 + x271;
        const double x302 = x272 + x300 + x301;
        const double x303 = R_l_inv_43*x258;
        const double x304 = R_l_inv_53*x256;
        const double x305 = R_l_inv_03*x251;
        const double x306 = R_l_inv_23*x247;
        const double x307 = R_l_inv_13*x253;
        const double x308 = R_l_inv_33*x249;
        const double x309 = R_l_inv_07*x197 + R_l_inv_17*x204 - R_l_inv_27*x194 - R_l_inv_37*x202 + R_l_inv_47*x186 - R_l_inv_57*x184;
        const double x310 = -x309*x57;
        const double x311 = x303 - x304 + x305 - x306 + x307 - x308 + x310;
        const double x312 = -x303 + x304 - x305 + x306 - x307 + x308 + x310;
        const double x313 = -x37;
        const double x314 = x275 + x313;
        const double x315 = -x280 - x281 + x282 + x283 + x284 - x285 + x30 + x31 + x32;
        const double x316 = x279 + x315;
        const double x317 = x291 + x295;
        const double x318 = x288 - x289;
        const double x319 = 2*x3;
        const double x320 = 2*x193;
        const double x321 = r_21*x320 + std::pow(r_31, 3)*x11 + r_31*x10 + r_31*x13 + r_31*x15 - r_31*x19 - r_31*x20 + r_31*x21 + r_31*x24 + r_31*x27 - r_31*x8 - r_31*x9 + x189*x196 + x28*x78 + x28*x79 - x28*x81 - x28*x82 - x319*x78 - x319*x79 - x319*x80;
        const double x322 = 2*r_12*x196 + r_12*x203 + r_22*x200 + r_22*x320 + r_32*x10 + r_32*x13 + r_32*x15 + 3*r_32*x17 - r_32*x19 - r_32*x20 + r_32*x21 + r_32*x24 + r_32*x27 - r_32*x8 - r_32*x9 + x28*x85 + x28*x86 - x28*x88 - x28*x89 - x319*x85 - x319*x86 - x319*x87;
        const double x323 = R_l_inv_66*x321 + R_l_inv_76*x322;
        const double x324 = d_3*x323;
        const double x325 = R_l_inv_62*x321 + R_l_inv_72*x322;
        const double x326 = x325*x4;
        const double x327 = R_l_inv_65*x321 + R_l_inv_75*x322;
        const double x328 = x327*x34;
        const double x329 = -x28*x75;
        const double x330 = -x28*x76;
        const double x331 = -x28*x77;
        const double x332 = x327*x37;
        const double x333 = x319*x75;
        const double x334 = x319*x76;
        const double x335 = x319*x77;
        const double x336 = x324 + x326 + x328 + x329 + x330 + x331 + x332 + x333 + x334 + x335;
        const double x337 = d_3*x325;
        const double x338 = R_l_inv_60*x321 + R_l_inv_70*x322;
        const double x339 = a_3*x338;
        const double x340 = x337 - x339;
        const double x341 = a_2*x338;
        const double x342 = R_l_inv_64*x321;
        const double x343 = R_l_inv_74*x322;
        const double x344 = x342 + x343;
        const double x345 = x341 + x344;
        const double x346 = x323*x36;
        const double x347 = -x346;
        const double x348 = x323*x52;
        const double x349 = x325*x52;
        const double x350 = x325*x36;
        const double x351 = x338*x57;
        const double x352 = -x350 - x351;
        const double x353 = x349 + x352;
        const double x354 = -x341;
        const double x355 = -x337;
        const double x356 = x339 + x354 + x355;
        const double x357 = -x324;
        const double x358 = x326 + x328 + x329 + x330 + x331 + x332 + x333 + x334 + x335 + x357;
        const double x359 = R_l_inv_67*x321 + R_l_inv_77*x322;
        const double x360 = -x359*x57;
        const double x361 = 4*a_2;
        const double x362 = d_3*x361;
        const double x363 = x360 + x362;
        const double x364 = d_3*x239;
        const double x365 = x321*x64;
        const double x366 = x322*x66;
        const double x367 = -x364 + x365 + x366;
        const double x368 = 4*x30;
        const double x369 = -8*a_2*a_3;
        const double x370 = 4*x32;
        const double x371 = 4*x31;
        const double x372 = -x370 + x371;
        const double x373 = -x362;
        const double x374 = x360 + x373;
        const double x375 = x364 - x365 - x366;
        const double x376 = -x342 - x343;
        const double x377 = x341 + x376;
        const double x378 = -x348;
        const double x379 = x362*x7;
        const double x380 = x379 + x74;
        const double x381 = -x56 - x58;
        const double x382 = 4*x1;
        const double x383 = 4*x42 - 4*x43;
        const double x384 = x379 + x51;
        const double x385 = x361*x68;
        const double x386 = 8*R_l_inv_63;
        const double x387 = 8*R_l_inv_73;
        const double x388 = x362*x97;
        const double x389 = x132 + x388;
        const double x390 = -x116 - x117;
        const double x391 = 4*x93;
        const double x392 = 4*x103 - 4*x104;
        const double x393 = x112 + x388;
        const double x394 = x123*x361;
        const double x395 = -x142*x362;
        const double x396 = x160 + x161;
        const double x397 = x395 + x396;
        const double x398 = 4*x154;
        const double x399 = 4*x140 - 4*x145;
        const double x400 = x162 + x395;
        const double x401 = -x173*x361;
        const double x402 = -x209*x362;
        const double x403 = x226 + x402;
        const double x404 = x230 + x231;
        const double x405 = 4*x206;
        const double x406 = 4*x215 - 4*x216;
        const double x407 = x227 + x402;
        const double x408 = a_3*x261;
        const double x409 = -x241*x361;
        const double x410 = 16*d_4;
        const double x411 = x182*x410;
        const double x412 = x185*x410;
        const double x413 = 8*x194;
        const double x414 = 8*x197;
        const double x415 = 8*x202;
        const double x416 = 8*x204;
        const double x417 = -x270*x362;
        const double x418 = x373 + x417;
        const double x419 = x293 + x294;
        const double x420 = x291 + x419;
        const double x421 = 4*x267;
        const double x422 = 4*x277 - 4*x278;
        const double x423 = -x309*x361;
        const double x424 = x362 + x417;
        const double x425 = x292 + x419;
        const double x426 = -x327*x362;
        const double x427 = x346 + x426;
        const double x428 = x350 + x351;
        const double x429 = 4*x324;
        const double x430 = 4*x337 - 4*x339;
        const double x431 = x347 + x426;
        const double x432 = -x359*x361;
        const double x433 = x370 - x371;
        const double x434 = -x368;
        const double x435 = x35 + x38 + x6;
        const double x436 = x1 + x435;
        const double x437 = x43 + x61;
        const double x438 = x381 + x55;
        const double x439 = x2 + x435;
        const double x440 = x44 + x45;
        const double x441 = x75 + x76 + x77 + x96 + x98 + x99;
        const double x442 = x441 + x93;
        const double x443 = x104 + x120;
        const double x444 = x115 + x390;
        const double x445 = x441 + x94;
        const double x446 = x105 + x106;
        const double x447 = x124 - x52;
        const double x448 = -x149;
        const double x449 = x169 + x448;
        const double x450 = x147 + x148;
        const double x451 = a_2 + a_3;
        const double x452 = x164 + x451;
        const double x453 = x159 + x396;
        const double x454 = x146 + x168 + x448;
        const double x455 = -x211;
        const double x456 = x208 + x210 + x225 + x455;
        const double x457 = x214 + x216 + x235;
        const double x458 = x229 + x404;
        const double x459 = x217 + x234;
        const double x460 = x208 + x210 + x264 + x455;
        const double x461 = -x272 + x301;
        const double x462 = x300 + x461;
        const double x463 = x278 + x298;
        const double x464 = x267 + x461;
        const double x465 = x326 + x328 + x329 + x330 + x331 - x332 + x333 + x334 + x335;
        const double x466 = x357 + x465;
        const double x467 = x339 + x355;
        const double x468 = x349 + x428;
        const double x469 = x324 + x465;
        const double x470 = x340 + x354;
        
        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = x40 + x44 + x50;
        A(0, 1) = x51 + x54 + x60;
        A(0, 2) = x49 + x62 + x63;
        A(0, 3) = x70;
        A(0, 4) = -4;
        A(0, 5) = x71;
        A(0, 6) = x44 + x63 + x73;
        A(0, 7) = x53 + x60 + x74;
        A(0, 8) = x40 + x62 + x72;
        A(1, 0) = x101 + x105 + x111;
        A(1, 1) = x112 + x114 + x119;
        A(1, 2) = x110 + x121 + x122;
        A(1, 3) = x125 + x128;
        A(1, 5) = x125 + x129;
        A(1, 6) = x105 + x122 + x131;
        A(1, 7) = x113 + x119 + x132;
        A(1, 8) = x101 + x121 + x130;
        A(2, 0) = a_2 + x146 + x150 + x155;
        A(2, 1) = x156 + x158 + x163;
        A(2, 2) = x150 + x167 + x170;
        A(2, 3) = x175;
        A(2, 5) = x176;
        A(2, 6) = x146 + x152 + x167 + x178;
        A(2, 7) = x163 + x179 + x180;
        A(2, 8) = a_2 + x170 + x178 + x181;
        A(3, 0) = x212 + x218 + x225;
        A(3, 1) = x227 + x228 + x233;
        A(3, 2) = x225 + x236 + x238;
        A(3, 3) = x243 + x260;
        A(3, 4) = x166*x261;
        A(3, 5) = x262 + x263;
        A(3, 6) = x218 + x238 + x264;
        A(3, 7) = x226 + x233 + x265;
        A(3, 8) = x212 + x236 + x264;
        A(4, 0) = x273 + x276 + x287;
        A(4, 1) = x290 + x296;
        A(4, 2) = x286 + x299 + x302 + x37;
        A(4, 3) = x311;
        A(4, 5) = x312;
        A(4, 6) = x302 + x314 + x316;
        A(4, 7) = x317 + x318;
        A(4, 8) = x273 + x299 + x313 + x315;
        A(5, 0) = x336 + x340 + x345;
        A(5, 1) = x347 + x348 + x353;
        A(5, 2) = x344 + x356 + x358;
        A(5, 3) = x363 + x367;
        A(5, 4) = x368 + x369 + x372;
        A(5, 5) = x374 + x375;
        A(5, 6) = x340 + x358 + x377;
        A(5, 7) = x346 + x353 + x378;
        A(5, 8) = x336 + x356 + x376;
        
        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x380 + x381;
        B(0, 1) = -x382 + x383;
        B(0, 2) = x384 + x59;
        B(0, 3) = x385 + 4;
        B(0, 4) = -r_31*x386 - r_32*x387;
        B(0, 5) = x385 - 4;
        B(0, 6) = x381 + x384;
        B(0, 7) = x382 + x383;
        B(0, 8) = x380 + x59;
        B(1, 0) = x389 + x390;
        B(1, 1) = -x391 + x392;
        B(1, 2) = x118 + x393;
        B(1, 3) = x394;
        B(1, 4) = -x386*x84 - x387*x91;
        B(1, 5) = x394;
        B(1, 6) = x390 + x393;
        B(1, 7) = x391 + x392;
        B(1, 8) = x118 + x389;
        B(2, 0) = x180 + x397;
        B(2, 1) = -x239 + x398 + x399;
        B(2, 2) = x158 + x400;
        B(2, 3) = x401;
        B(2, 4) = x135*x386 + x138*x387;
        B(2, 5) = x401;
        B(2, 6) = x158 + x397;
        B(2, 7) = x239 - x398 + x399;
        B(2, 8) = x180 + x400;
        B(3, 0) = x403 + x404;
        B(3, 1) = x405 + x406;
        B(3, 2) = x232 + x407;
        B(3, 3) = -x408 + x409;
        B(3, 4) = R_l_inv_03*x413 + R_l_inv_13*x415 + R_l_inv_23*x414 + R_l_inv_33*x416 + R_l_inv_43*x411 + R_l_inv_53*x412 - 8*x30 + 8*x31 - 8*x32;
        B(3, 5) = x408 + x409;
        B(3, 6) = x404 + x407;
        B(3, 7) = -x405 + x406;
        B(3, 8) = x232 + x403;
        B(4, 0) = x418 + x420;
        B(4, 1) = x421 + x422;
        B(4, 2) = x296 + x418;
        B(4, 3) = x423;
        B(4, 4) = R_l_inv_03*x414 + R_l_inv_13*x416 - R_l_inv_23*x413 - R_l_inv_33*x415 + R_l_inv_43*x412 - R_l_inv_53*x411;
        B(4, 5) = x423;
        B(4, 6) = x424 + x425;
        B(4, 7) = -x421 + x422;
        B(4, 8) = x317 + x424;
        B(5, 0) = x427 + x428;
        B(5, 1) = x429 + x430;
        B(5, 2) = x352 + x431;
        B(5, 3) = x368 + x432 + x433;
        B(5, 4) = -16*a_3*d_3 + x321*x386 + x322*x387;
        B(5, 5) = x372 + x432 + x434;
        B(5, 6) = x428 + x431;
        B(5, 7) = -x429 + x430;
        B(5, 8) = x352 + x427;
        
        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = x436 + x437 + x50;
        C(0, 1) = x438 + x54 + x74;
        C(0, 2) = x439 + x440 + x49;
        C(0, 3) = x71;
        C(0, 4) = 4;
        C(0, 5) = x70;
        C(0, 6) = x437 + x439 + x73;
        C(0, 7) = x438 + x51 + x53;
        C(0, 8) = x436 + x440 + x72;
        C(1, 0) = x111 + x442 + x443;
        C(1, 1) = x114 + x132 + x444;
        C(1, 2) = x110 + x445 + x446;
        C(1, 3) = x129 + x447;
        C(1, 5) = x128 + x447;
        C(1, 6) = x131 + x443 + x445;
        C(1, 7) = x112 + x113 + x444;
        C(1, 8) = x130 + x442 + x446;
        C(2, 0) = x152 + x449 + x450 + x452;
        C(2, 1) = x156 + x180 + x453;
        C(2, 2) = x165 + x181 + x450 + x454;
        C(2, 3) = x176;
        C(2, 5) = x175;
        C(2, 6) = x155 + x165 + x177 + x449;
        C(2, 7) = x158 + x179 + x453;
        C(2, 8) = x177 + x452 + x454;
        C(3, 0) = x237 + x456 + x457;
        C(3, 1) = x226 + x228 + x458;
        C(3, 2) = x206 + x456 + x459;
        C(3, 3) = x243 + x263;
        C(3, 4) = -x261*x451;
        C(3, 5) = x260 + x262;
        C(3, 6) = x206 + x457 + x460;
        C(3, 7) = x227 + x265 + x458;
        C(3, 8) = x237 + x459 + x460;
        C(4, 0) = x286 + x314 + x462 + x463;
        C(4, 1) = x290 + x420;
        C(4, 2) = x287 + x297 + x313 + x464;
        C(4, 3) = x312;
        C(4, 5) = x311;
        C(4, 6) = x276 + x315 + x463 + x464;
        C(4, 7) = x318 + x425;
        C(4, 8) = x297 + x316 + x37 + x462;
        C(5, 0) = x345 + x466 + x467;
        C(5, 1) = x346 + x348 + x468;
        C(5, 2) = x344 + x469 + x470;
        C(5, 3) = x363 + x375;
        C(5, 4) = x369 + x433 + x434;
        C(5, 5) = x367 + x374;
        C(5, 6) = x377 + x467 + x469;
        C(5, 7) = x347 + x378 + x468;
        C(5, 8) = x376 + x466 + x470;
        
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
            const double th_2 = this_solution[2];
            
            const bool degenerate_valid_0 = std::fabs(th_2 - 1.69515128643052) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
            }
            
            const bool degenerate_valid_1 = std::fabs(th_2 + 1.69515128643052) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(3, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_2_processor();
    // Finish code for solved_variable dispatcher node 2
    
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
            
            const bool condition_0 = std::fabs((std::pow(a_2, 2) + 2*a_2*a_3*std::cos(th_2) + 2*a_2*d_3*std::sin(th_2) + std::pow(a_3, 2) + std::pow(d_3, 2) + std::pow(d_4, 2) - std::pow(inv_Px, 2) - std::pow(inv_Py, 2) - std::pow(d_5 + inv_Pz, 2))/(2*a_2*d_4*std::cos(th_2) + 2*a_3*d_4)) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 2*d_4;
                const double x1 = a_2*std::cos(th_2);
                const double x2 = safe_asin((-std::pow(a_2, 2) - 2*a_2*d_3*std::sin(th_2) - std::pow(a_3, 2) - 2*a_3*x1 - std::pow(d_3, 2) - std::pow(d_4, 2) + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(d_5 + inv_Pz, 2))/(-a_3*x0 - x0*x1));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[3] = x2;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = std::fabs((std::pow(a_2, 2) + 2*a_2*a_3*std::cos(th_2) + 2*a_2*d_3*std::sin(th_2) + std::pow(a_3, 2) + std::pow(d_3, 2) + std::pow(d_4, 2) - std::pow(inv_Px, 2) - std::pow(inv_Py, 2) - std::pow(d_5 + inv_Pz, 2))/(2*a_2*d_4*std::cos(th_2) + 2*a_3*d_4)) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = 2*d_4;
                const double x1 = a_2*std::cos(th_2);
                const double x2 = safe_asin((-std::pow(a_2, 2) - 2*a_2*d_3*std::sin(th_2) - std::pow(a_3, 2) - 2*a_3*x1 - std::pow(d_3, 2) - std::pow(d_4, 2) + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(d_5 + inv_Pz, 2))/(-a_3*x0 - x0*x1));
                // End of temp variables
                const double tmp_sol_value = M_PI - x2;
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
            const double th_3 = this_solution[3];
            const bool checked_result = std::fabs(d_4*std::cos(th_3)) <= 9.9999999999999995e-7 && std::fabs(Px - d_5*r_13) <= 9.9999999999999995e-7 && std::fabs(Py - d_5*r_23) <= 9.9999999999999995e-7;
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
            const double th_3 = this_solution[3];
            
            const bool condition_0 = std::fabs(d_4*std::cos(th_3)) >= zero_tolerance || std::fabs(Px - d_5*r_13) >= zero_tolerance || std::fabs(Py - d_5*r_23) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Px - d_5*r_13;
                const double x1 = -Py + d_5*r_23;
                const double x2 = std::atan2(x0, x1);
                const double x3 = std::cos(th_3);
                const double x4 = safe_sqrt(-std::pow(d_4, 2)*std::pow(x3, 2) + std::pow(x0, 2) + std::pow(x1, 2));
                const double x5 = d_4*x3;
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[0] = x2 + std::atan2(x4, x5);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx);
            }
            
            const bool condition_1 = std::fabs(d_4*std::cos(th_3)) >= zero_tolerance || std::fabs(Px - d_5*r_13) >= zero_tolerance || std::fabs(Py - d_5*r_23) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = Px - d_5*r_13;
                const double x1 = -Py + d_5*r_23;
                const double x2 = std::atan2(x0, x1);
                const double x3 = std::cos(th_3);
                const double x4 = safe_sqrt(-std::pow(d_4, 2)*std::pow(x3, 2) + std::pow(x0, 2) + std::pow(x1, 2));
                const double x5 = d_4*x3;
                // End of temp variables
                const double tmp_sol_value = x2 + std::atan2(-x4, x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
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
            const double th_3 = this_solution[3];
            
            const bool degenerate_valid_0 = std::fabs(th_3) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(12, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_3 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(17, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(7, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_6_processor();
    // Finish code for solved_variable dispatcher node 6
    
    // Code for explicit solution node 17, solved variable is th_5
    auto ExplicitSolutionNode_node_17_solve_th_5_processor = [&]() -> void
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
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1);
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
    ExplicitSolutionNode_node_17_solve_th_5_processor();
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
            const double th_2 = this_solution[2];
            
            const bool degenerate_valid_0 = std::fabs(th_2 - 1.44644133224814) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(28, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(-th_2 + 1.44644133224814 + M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(31, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(19, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_18_processor();
    // Finish code for solved_variable dispatcher node 18
    
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
            const double th_5 = this_solution[5];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_2 - d_3) >= zero_tolerance || std::fabs(d_5 + inv_Pz) >= zero_tolerance || std::fabs(inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5);
                const double x1 = d_5 + inv_Pz;
                const double x2 = std::atan2(x0, x1);
                const double x3 = -0.99227787671366796*a_2 + d_3;
                const double x4 = safe_sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x2 + std::atan2(x4, x3);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(32, appended_idx);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_2 - d_3) >= zero_tolerance || std::fabs(d_5 + inv_Pz) >= zero_tolerance || std::fabs(inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5);
                const double x1 = d_5 + inv_Pz;
                const double x2 = std::atan2(x0, x1);
                const double x3 = -0.99227787671366796*a_2 + d_3;
                const double x4 = safe_sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                const double tmp_sol_value = x2 + std::atan2(-x4, x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
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
    
    // Code for explicit solution node 33, solved variable is th_1
    auto ExplicitSolutionNode_node_33_solve_th_1_processor = [&]() -> void
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
            const double th_5 = this_solution[5];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_3 - 0.12403473458920899*d_3) >= zero_tolerance || std::fabs(-a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = -a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_3 + 0.12403473458920899*d_3);
                const double x2 = Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_3 + 0.125*d_3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x1 + std::atan2(x3, x2);
                int appended_idx = append_solution_to_queue(solution_0);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_3 - 0.12403473458920899*d_3) >= zero_tolerance || std::fabs(-a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = -a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_3 + 0.12403473458920899*d_3);
                const double x2 = Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_3 + 0.125*d_3, 2));
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
    ExplicitSolutionNode_node_33_solve_th_1_processor();
    // Finish code for explicit solution node 32
    
    // Code for explicit solution node 28, solved variable is th_4
    auto ExplicitSolutionNode_node_28_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(28);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(28);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 28
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_5 = this_solution[5];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_2 + d_3) >= zero_tolerance || std::fabs(d_5 + inv_Pz) >= zero_tolerance || std::fabs(inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5);
                const double x1 = d_5 + inv_Pz;
                const double x2 = std::atan2(x0, x1);
                const double x3 = 0.99227787671366796*a_2 + d_3;
                const double x4 = safe_sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x2 + std::atan2(x4, x3);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(29, appended_idx);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_2 + d_3) >= zero_tolerance || std::fabs(d_5 + inv_Pz) >= zero_tolerance || std::fabs(inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5);
                const double x1 = d_5 + inv_Pz;
                const double x2 = std::atan2(x0, x1);
                const double x3 = 0.99227787671366796*a_2 + d_3;
                const double x4 = safe_sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                const double tmp_sol_value = x2 + std::atan2(-x4, x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(29, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_28_solve_th_4_processor();
    // Finish code for explicit solution node 28
    
    // Code for non-branch dispatcher node 29
    // Actually, there is no code
    
    // Code for explicit solution node 30, solved variable is th_1
    auto ExplicitSolutionNode_node_30_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(29);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(29);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 30
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_5 = this_solution[5];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_3 - 0.12403473458920899*d_3) >= zero_tolerance || std::fabs(a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_3 + 0.12403473458920899*d_3);
                const double x2 = -Pz - d_4*r_31*std::sin(th_5) - d_4*r_32*std::cos(th_5) + d_5*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_3 + 0.125*d_3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x1 + std::atan2(x3, x2);
                int appended_idx = append_solution_to_queue(solution_0);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_3 - 0.12403473458920899*d_3) >= zero_tolerance || std::fabs(a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_3 + 0.12403473458920899*d_3);
                const double x2 = -Pz - d_4*r_31*std::sin(th_5) - d_4*r_32*std::cos(th_5) + d_5*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_3 + 0.125*d_3, 2));
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
    ExplicitSolutionNode_node_30_solve_th_1_processor();
    // Finish code for explicit solution node 29
    
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
            const double th_2 = this_solution[2];
            
            const bool condition_0 = std::fabs(a_3*std::sin(th_2) - d_3*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_2 + a_3*std::cos(th_2) + d_3*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = d_5*r_33;
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = a_2 + a_3*x1 + d_3*x2;
                const double x4 = a_3*x2 - d_3*x1;
                const double x5 = std::cos(th_0);
                const double x6 = std::sin(th_0);
                const double x7 = Px*x5 + Py*x6 - d_5*r_13*x5 - d_5*r_23*x6;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x3*(-Pz + x0) + x4*x7, x3*x7 + x4*(Pz - x0));
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
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_13*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + r_23*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - r_33*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))) >= zero_tolerance || std::fabs(r_13*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_23*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::sin(th_0) + r_33*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_1);
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = std::cos(th_1);
                const double x4 = x0*x1 - x2*x3;
                const double x5 = x0*x2 + x1*x3;
                const double x6 = r_13*std::cos(th_0);
                const double x7 = r_23*std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_33*x4 + x5*x6 + x5*x7, r_33*x5 + x4*x6 + x4*x7);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
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
            const double th_0 = this_solution[0];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x1 + r_21*x0, -r_12*x1 + r_22*x0);
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(13, node_input_i_idx_in_queue);
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
    
    // Code for solved_variable dispatcher node 13
    auto SolvedVariableDispatcherNode_node_13_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(13);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(13);
        if (!this_input_valid)
            return;
        
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_2 = this_solution[2];
            
            const bool degenerate_valid_0 = std::fabs(th_2 - 1.44644133224814) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(22, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(-th_2 + 1.44644133224814 + M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(25, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(14, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_13_processor();
    // Finish code for solved_variable dispatcher node 13
    
    // Code for explicit solution node 25, solved variable is th_4
    auto ExplicitSolutionNode_node_25_solve_th_4_processor = [&]() -> void
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
            const double th_5 = this_solution[5];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_2 - d_3) >= zero_tolerance || std::fabs(d_5 + inv_Pz) >= zero_tolerance || std::fabs(inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5);
                const double x1 = d_5 + inv_Pz;
                const double x2 = std::atan2(x0, x1);
                const double x3 = -0.99227787671366796*a_2 + d_3;
                const double x4 = safe_sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x2 + std::atan2(x4, x3);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(26, appended_idx);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_2 - d_3) >= zero_tolerance || std::fabs(d_5 + inv_Pz) >= zero_tolerance || std::fabs(inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5);
                const double x1 = d_5 + inv_Pz;
                const double x2 = std::atan2(x0, x1);
                const double x3 = -0.99227787671366796*a_2 + d_3;
                const double x4 = safe_sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                const double tmp_sol_value = x2 + std::atan2(-x4, x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(26, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_4_processor();
    // Finish code for explicit solution node 25
    
    // Code for non-branch dispatcher node 26
    // Actually, there is no code
    
    // Code for explicit solution node 27, solved variable is th_1
    auto ExplicitSolutionNode_node_27_solve_th_1_processor = [&]() -> void
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
            const double th_5 = this_solution[5];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_3 - 0.12403473458920899*d_3) >= zero_tolerance || std::fabs(-a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = -a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_3 + 0.12403473458920899*d_3);
                const double x2 = Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_3 + 0.125*d_3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x1 + std::atan2(x3, x2);
                int appended_idx = append_solution_to_queue(solution_0);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_3 - 0.12403473458920899*d_3) >= zero_tolerance || std::fabs(-a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = -a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_3 + 0.12403473458920899*d_3);
                const double x2 = Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_3 + 0.125*d_3, 2));
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
    ExplicitSolutionNode_node_27_solve_th_1_processor();
    // Finish code for explicit solution node 26
    
    // Code for explicit solution node 22, solved variable is th_4
    auto ExplicitSolutionNode_node_22_solve_th_4_processor = [&]() -> void
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
            const double th_5 = this_solution[5];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_2 + d_3) >= zero_tolerance || std::fabs(d_5 + inv_Pz) >= zero_tolerance || std::fabs(inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5);
                const double x1 = d_5 + inv_Pz;
                const double x2 = std::atan2(x0, x1);
                const double x3 = 0.99227787671366796*a_2 + d_3;
                const double x4 = safe_sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x2 + std::atan2(x4, x3);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(23, appended_idx);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_2 + d_3) >= zero_tolerance || std::fabs(d_5 + inv_Pz) >= zero_tolerance || std::fabs(inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = inv_Px*std::cos(th_5) - inv_Py*std::sin(th_5);
                const double x1 = d_5 + inv_Pz;
                const double x2 = std::atan2(x0, x1);
                const double x3 = 0.99227787671366796*a_2 + d_3;
                const double x4 = safe_sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                const double tmp_sol_value = x2 + std::atan2(-x4, x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(23, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_22_solve_th_4_processor();
    // Finish code for explicit solution node 22
    
    // Code for non-branch dispatcher node 23
    // Actually, there is no code
    
    // Code for explicit solution node 24, solved variable is th_1
    auto ExplicitSolutionNode_node_24_solve_th_1_processor = [&]() -> void
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
            const double th_5 = this_solution[5];
            
            const bool condition_0 = std::fabs(0.99227787671366796*a_3 - 0.12403473458920899*d_3) >= zero_tolerance || std::fabs(a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_3 + 0.12403473458920899*d_3);
                const double x2 = -Pz - d_4*r_31*std::sin(th_5) - d_4*r_32*std::cos(th_5) + d_5*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_3 + 0.125*d_3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x1 + std::atan2(x3, x2);
                int appended_idx = append_solution_to_queue(solution_0);
            }
            
            const bool condition_1 = std::fabs(0.99227787671366796*a_3 - 0.12403473458920899*d_3) >= zero_tolerance || std::fabs(a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = a_2 + 0.12403473458920899*a_3 + 0.99227787671366796*d_3;
                const double x1 = std::atan2(x0, -0.99227787671366796*a_3 + 0.12403473458920899*d_3);
                const double x2 = -Pz - d_4*r_31*std::sin(th_5) - d_4*r_32*std::cos(th_5) + d_5*r_33;
                const double x3 = safe_sqrt(std::pow(x0, 2) - std::pow(x2, 2) + 0.984615384615385*std::pow(-a_3 + 0.125*d_3, 2));
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
    ExplicitSolutionNode_node_24_solve_th_1_processor();
    // Finish code for explicit solution node 23
    
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
            const double th_2 = this_solution[2];
            
            const bool condition_0 = std::fabs(a_3*std::sin(th_2) - d_3*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_2 + a_3*std::cos(th_2) + d_3*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = d_5*r_33;
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = a_2 + a_3*x1 + d_3*x2;
                const double x4 = a_3*x2 - d_3*x1;
                const double x5 = std::cos(th_0);
                const double x6 = std::sin(th_0);
                const double x7 = Px*x5 + Py*x6 - d_5*r_13*x5 - d_5*r_23*x6;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x3*(-Pz + x0) + x4*x7, x3*x7 + x4*(Pz - x0));
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
            const double th_2 = this_solution[2];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_13*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::cos(th_0) + r_23*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_0) - r_33*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))) >= zero_tolerance || std::fabs(r_13*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_23*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::sin(th_0) + r_33*(std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_1);
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = std::cos(th_1);
                const double x4 = x0*x1 - x2*x3;
                const double x5 = x0*x2 + x1*x3;
                const double x6 = r_13*std::cos(th_0);
                const double x7 = r_23*std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_33*x4 - x5*x6 - x5*x7, r_33*x5 + x4*x6 + x4*x7);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
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
            const double th_3 = this_solution[3];
            
            const bool condition_0 = std::fabs((r_13*std::sin(th_0) - r_23*std::cos(th_0))/std::sin(th_3)) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = safe_asin((-r_13*std::sin(th_0) + r_23*std::cos(th_0))/std::sin(th_3));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x0;
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
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            const double th_4 = this_solution[4];
            const bool checked_result = std::fabs(a_2*std::sin(th_3)*std::cos(th_2) + a_3*std::sin(th_3) - d_4) <= 9.9999999999999995e-7 && std::fabs(a_2*std::sin(th_2)*std::sin(th_4) - a_2*std::cos(th_2)*std::cos(th_3)*std::cos(th_4) - a_3*std::cos(th_3)*std::cos(th_4) + d_3*std::sin(th_4)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor();
    // Finish code for equation all-zero dispatcher node 8
    
    // Code for explicit solution node 9, solved variable is th_5
    auto ExplicitSolutionNode_node_9_solve_th_5_processor = [&]() -> void
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
            const double th_3 = this_solution[3];
            const double th_4 = this_solution[4];
            
            const bool condition_0 = std::fabs(a_2*std::sin(th_3)*std::cos(th_2) + a_3*std::sin(th_3) - d_4) >= 9.9999999999999995e-7 || std::fabs(a_2*std::sin(th_2)*std::sin(th_4) - a_2*std::cos(th_2)*std::cos(th_3)*std::cos(th_4) - a_3*std::cos(th_3)*std::cos(th_4) + d_3*std::sin(th_4)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_3);
                const double x1 = a_2*std::cos(th_2);
                const double x2 = -a_3*x0 + d_4 - x0*x1;
                const double x3 = std::sin(th_4);
                const double x4 = std::cos(th_3)*std::cos(th_4);
                const double x5 = a_2*x3*std::sin(th_2) - a_3*x4 + d_3*x3 - x1*x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(inv_Px*x2 - inv_Py*x5, inv_Px*x5 + inv_Py*x2);
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(10, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_5_processor();
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
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            const bool checked_result = std::fabs(-a_3*std::sin(th_2) + d_3*std::cos(th_2) + d_4*std::sin(th_2)*std::sin(th_3)) <= 9.9999999999999995e-7 && std::fabs(a_2 + a_3*std::cos(th_2) + d_3*std::sin(th_2) - d_4*std::sin(th_3)*std::cos(th_2)) <= 9.9999999999999995e-7;
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
            const double th_3 = this_solution[3];
            
            const bool condition_0 = std::fabs(-a_3*std::sin(th_2) + d_3*std::cos(th_2) + d_4*std::sin(th_2)*std::sin(th_3)) >= 9.9999999999999995e-7 || std::fabs(a_2 + a_3*std::cos(th_2) + d_3*std::sin(th_2) - d_4*std::sin(th_3)*std::cos(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = d_5*r_33;
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = d_4*std::sin(th_3);
                const double x4 = a_2 + a_3*x1 + d_3*x2 - x1*x3;
                const double x5 = a_3*x2 - d_3*x1 - x2*x3;
                const double x6 = std::cos(th_0);
                const double x7 = std::sin(th_0);
                const double x8 = Px*x6 + Py*x7 - d_5*r_13*x6 - d_5*r_23*x7;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x4*(-Pz + x0) + x5*x8, x4*x8 + x5*(Pz - x0));
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
        const double value_at_3 = raw_ik_out_i[3];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[4];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[5];  // th_5
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

}; // struct rokae_SR5_ik

// Code below for debug
void test_ik_solve_rokae_SR5()
{
    std::array<double, rokae_SR5_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = rokae_SR5_ik::computeFK(theta);
    auto ik_output = rokae_SR5_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = rokae_SR5_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_rokae_SR5();
}
