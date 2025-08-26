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
        R_l(0, 3) = -a_2;
        R_l(0, 7) = -a_3;
        R_l(1, 2) = -a_2;
        R_l(1, 6) = -a_3;
        R_l(2, 4) = -a_2;
        R_l(3, 6) = -1;
        R_l(4, 7) = 1;
        R_l(5, 5) = 2*a_2*a_3;
        R_l(6, 1) = a_2;
        R_l(7, 0) = a_2;
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
        const double x4 = d_3*r_23;
        const double x5 = -x4;
        const double x6 = d_4*r_22;
        const double x7 = -x6;
        const double x8 = x5 + x7;
        const double x9 = std::pow(r_21, 2);
        const double x10 = Py*x9;
        const double x11 = std::pow(r_22, 2);
        const double x12 = Py*x11;
        const double x13 = std::pow(r_23, 2);
        const double x14 = Py*x13;
        const double x15 = d_5*r_23;
        const double x16 = Px*r_11;
        const double x17 = r_21*x16;
        const double x18 = Px*r_12;
        const double x19 = r_22*x18;
        const double x20 = Px*r_13;
        const double x21 = r_23*x20;
        const double x22 = Pz*r_31;
        const double x23 = r_21*x22;
        const double x24 = Pz*r_32;
        const double x25 = r_22*x24;
        const double x26 = Pz*r_33;
        const double x27 = r_23*x26;
        const double x28 = x10 + x12 + x14 - x15 + x17 + x19 + x21 + x23 + x25 + x27;
        const double x29 = d_4*x1;
        const double x30 = x28 + x6;
        const double x31 = d_3*x1;
        const double x32 = -x31;
        const double x33 = d_3*x3;
        const double x34 = x4 + x7;
        const double x35 = Py*r_22;
        const double x36 = x18 + x24 + x35;
        const double x37 = R_l_inv_51*a_2;
        const double x38 = x36*x37;
        const double x39 = R_l_inv_52*a_2;
        const double x40 = d_3*x39;
        const double x41 = a_2*r_22;
        const double x42 = R_l_inv_54*x41;
        const double x43 = std::pow(d_3, 2);
        const double x44 = std::pow(d_4, 2);
        const double x45 = std::pow(d_5, 2);
        const double x46 = std::pow(a_2, 2);
        const double x47 = std::pow(a_3, 2);
        const double x48 = 2*d_5;
        const double x49 = 2*x15;
        const double x50 = Py*x1;
        const double x51 = 2*x18;
        const double x52 = Py*r_23;
        const double x53 = 2*x20;
        const double x54 = 2*x16;
        const double x55 = 2*x24;
        const double x56 = 2*x26;
        const double x57 = std::pow(Px, 2);
        const double x58 = std::pow(r_11, 2);
        const double x59 = x57*x58;
        const double x60 = std::pow(r_12, 2);
        const double x61 = x57*x60;
        const double x62 = std::pow(r_13, 2);
        const double x63 = x57*x62;
        const double x64 = std::pow(Py, 2);
        const double x65 = x64*x9;
        const double x66 = x11*x64;
        const double x67 = x13*x64;
        const double x68 = std::pow(Pz, 2);
        const double x69 = std::pow(r_31, 2)*x68;
        const double x70 = std::pow(r_32, 2)*x68;
        const double x71 = std::pow(r_33, 2)*x68;
        const double x72 = -Py*x49 + x16*x50 - x20*x48 + x22*x50 + x22*x54 + x24*x51 - x26*x48 + x26*x53 + x35*x51 + x35*x55 + x43 + x44 + x45 - x46 - x47 + x52*x53 + x52*x56 + x59 + x61 + x63 + x65 + x66 + x67 + x69 + x70 + x71;
        const double x73 = R_l_inv_55*a_2;
        const double x74 = x72*x73;
        const double x75 = -d_4*x37;
        const double x76 = d_5*r_21;
        const double x77 = r_23*x16;
        const double x78 = r_23*x22;
        const double x79 = r_21*x20;
        const double x80 = r_21*x26;
        const double x81 = x76 + x77 + x78 - x79 - x80;
        const double x82 = R_l_inv_57*a_2;
        const double x83 = x81*x82;
        const double x84 = -x83;
        const double x85 = R_l_inv_56*d_3*x41;
        const double x86 = -x85;
        const double x87 = 2*d_4;
        const double x88 = x36*x73;
        const double x89 = x87*x88;
        const double x90 = -x89;
        const double x91 = a_3 + x38 + x40 + x42 + x74 + x75 + x84 + x86 + x90;
        const double x92 = d_4*r_21;
        const double x93 = -x92;
        const double x94 = Py*r_21;
        const double x95 = x16 + x22 + x94;
        const double x96 = R_l_inv_50*a_2;
        const double x97 = x95*x96;
        const double x98 = -x97;
        const double x99 = R_l_inv_53*a_2;
        const double x100 = r_21*x99;
        const double x101 = -x100;
        const double x102 = d_5*r_22;
        const double x103 = r_23*x18;
        const double x104 = r_23*x24;
        const double x105 = r_22*x20;
        const double x106 = r_22*x26;
        const double x107 = x102 + x103 + x104 - x105 - x106;
        const double x108 = R_l_inv_56*a_2;
        const double x109 = x107*x108;
        const double x110 = -x109;
        const double x111 = d_3*r_21;
        const double x112 = x111*x82;
        const double x113 = -x112;
        const double x114 = x101 + x110 + x113 + x93 + x98;
        const double x115 = r_21*x18;
        const double x116 = r_21*x24;
        const double x117 = r_22*x16;
        const double x118 = -x117;
        const double x119 = r_22*x22;
        const double x120 = -x119;
        const double x121 = d_4*r_23;
        const double x122 = x108*x121;
        const double x123 = -d_5 + x20 + x26 + x52;
        const double x124 = x123*x39;
        const double x125 = 2*d_3;
        const double x126 = x123*x125;
        const double x127 = x126*x73;
        const double x128 = -x124 - x127;
        const double x129 = x115 + x116 + x118 + x120 + x122 + x128;
        const double x130 = 2*x96;
        const double x131 = x130*x36;
        const double x132 = 2*x95;
        const double x133 = -x132*x37;
        const double x134 = 4*d_4;
        const double x135 = x73*x95;
        const double x136 = x134*x135;
        const double x137 = -x131 + x133 + x136;
        const double x138 = R_l_inv_54*a_2;
        const double x139 = x1*x138;
        const double x140 = 2*x82;
        const double x141 = x107*x140;
        const double x142 = x108*x31;
        const double x143 = -x139 - x141 + x142;
        const double x144 = 2*x6;
        const double x145 = 2*R_l_inv_53*x41;
        const double x146 = 2*x108;
        const double x147 = x146*x81;
        const double x148 = R_l_inv_57*x125*x41;
        const double x149 = -x144 - x145 + x147 - x148;
        const double x150 = x100 + x109 + x112 + x92 + x97;
        const double x151 = a_3 - x38 + x40 + x74 + x75 + x89;
        const double x152 = -x42 + x83 + x85;
        const double x153 = x151 + x152;
        const double x154 = x132*x39;
        const double x155 = x123*x130;
        const double x156 = 4*d_3;
        const double x157 = x135*x156;
        const double x158 = -x154 + x155 - x157;
        const double x159 = 2*x121;
        const double x160 = 2*r_23;
        const double x161 = x160*x99;
        const double x162 = -x115 - x116 + x117 + x119;
        const double x163 = x146*x162;
        const double x164 = x140*x4;
        const double x165 = x159 + x161 + x163 + x164;
        const double x166 = 2*x102;
        const double x167 = 2*x103;
        const double x168 = 2*x104;
        const double x169 = 2*x105;
        const double x170 = 2*x106;
        const double x171 = x108*x29;
        const double x172 = -x166 - x167 - x168 + x169 + x170 + x171;
        const double x173 = 4*x76;
        const double x174 = 4*x79;
        const double x175 = 4*x80;
        const double x176 = 4*x77;
        const double x177 = 4*x78;
        const double x178 = d_4*x3;
        const double x179 = x108*x178;
        const double x180 = 8*d_3;
        const double x181 = -x180*x88 - 4*x36*x39;
        const double x182 = x154 + x155 + x157;
        const double x183 = x166 + x167 + x168 - x169 - x170 - x171;
        const double x184 = -x122 + x162;
        const double x185 = x124 + x127;
        const double x186 = x184 + x185;
        const double x187 = x131 + x133 + x136;
        const double x188 = x144 + x145 - x147 + x148;
        const double x189 = a_2*a_3;
        const double x190 = 2*x189;
        const double x191 = x46 + x47;
        const double x192 = R_l_inv_62*x191;
        const double x193 = R_l_inv_22*x190 + x192;
        const double x194 = d_3*x193;
        const double x195 = R_l_inv_61*x191;
        const double x196 = R_l_inv_21*x190 + x195;
        const double x197 = x196*x36;
        const double x198 = R_l_inv_25*x190 + R_l_inv_65*x191;
        const double x199 = x198*x72;
        const double x200 = -d_4*x196;
        const double x201 = R_l_inv_60*x191;
        const double x202 = x95*(R_l_inv_20*x190 + x201);
        const double x203 = -x202;
        const double x204 = x123*x193;
        const double x205 = -x204;
        const double x206 = x126*x198;
        const double x207 = -x206;
        const double x208 = x198*x36;
        const double x209 = x208*x87;
        const double x210 = -x209;
        const double x211 = x194 + x197 + x199 + x200 + x203 + x205 + x207 + x210;
        const double x212 = R_l_inv_66*x191;
        const double x213 = R_l_inv_26*x190 + x212;
        const double x214 = x121*x213;
        const double x215 = x144*x16;
        const double x216 = x144*x22;
        const double x217 = x18*x29;
        const double x218 = x24*x29;
        const double x219 = x214 - x215 - x216 + x217 + x218;
        const double x220 = R_l_inv_24*x190 + R_l_inv_64*x191;
        const double x221 = r_22*x220;
        const double x222 = R_l_inv_67*x191;
        const double x223 = R_l_inv_27*x190 + x222;
        const double x224 = x223*x81;
        const double x225 = d_3*r_22;
        const double x226 = x213*x225;
        const double x227 = d_5*x31;
        const double x228 = 2*x4;
        const double x229 = x16*x228;
        const double x230 = x22*x228;
        const double x231 = x20*x31;
        const double x232 = x26*x31;
        const double x233 = x221 - x224 - x226 - x227 - x229 - x230 + x231 + x232;
        const double x234 = x219 + x233;
        const double x235 = std::pow(r_21, 3)*x64;
        const double x236 = r_21*x43;
        const double x237 = r_21*x44;
        const double x238 = r_21*x45;
        const double x239 = R_l_inv_23*x190 + R_l_inv_63*x191;
        const double x240 = r_21*x239;
        const double x241 = x107*x213;
        const double x242 = r_21*x59;
        const double x243 = r_21*x66;
        const double x244 = r_21*x67;
        const double x245 = r_21*x69;
        const double x246 = x111*x223;
        const double x247 = r_21*x61;
        const double x248 = r_21*x63;
        const double x249 = r_21*x70;
        const double x250 = r_21*x71;
        const double x251 = x16*x49;
        const double x252 = x22*x49;
        const double x253 = x10*x54;
        const double x254 = x12*x54;
        const double x255 = x14*x54;
        const double x256 = d_5*x1;
        const double x257 = x20*x256;
        const double x258 = 2*x22;
        const double x259 = x10*x258;
        const double x260 = x12*x258;
        const double x261 = x14*x258;
        const double x262 = x256*x26;
        const double x263 = 2*r_11;
        const double x264 = r_12*x57;
        const double x265 = r_22*x264;
        const double x266 = x263*x265;
        const double x267 = 2*r_13;
        const double x268 = r_23*x57;
        const double x269 = r_11*x267*x268;
        const double x270 = r_31*x68;
        const double x271 = r_32*x270;
        const double x272 = 2*r_22;
        const double x273 = x271*x272;
        const double x274 = r_33*x270;
        const double x275 = x160*x274;
        const double x276 = x18*x24;
        const double x277 = x1*x276;
        const double x278 = x20*x26;
        const double x279 = x1*x278;
        const double x280 = x16*x22;
        const double x281 = x1*x280;
        const double x282 = x25*x54;
        const double x283 = x27*x54;
        const double x284 = x19*x258;
        const double x285 = x21*x258;
        const double x286 = x235 - x236 - x237 - x238 - x240 - x241 + x242 + x243 + x244 + x245 - x246 - x247 - x248 - x249 - x250 - x251 - x252 + x253 + x254 + x255 + x257 + x259 + x260 + x261 + x262 + x266 + x269 + x273 + x275 - x277 - x279 + x281 + x282 + x283 + x284 + x285;
        const double x287 = 4*x189;
        const double x288 = R_l_inv_20*x287 + 2*x201;
        const double x289 = x288*x36;
        const double x290 = -x95*(R_l_inv_21*x287 + 2*x195);
        const double x291 = x198*x95;
        const double x292 = x134*x291;
        const double x293 = -x289 + x290 + x292;
        const double x294 = R_l_inv_27*x287 + 2*x222;
        const double x295 = x107*x294;
        const double x296 = x1*x220;
        const double x297 = d_5*x33;
        const double x298 = x213*x31;
        const double x299 = 4*x18;
        const double x300 = x299*x4;
        const double x301 = 4*x24;
        const double x302 = x301*x4;
        const double x303 = x20*x33;
        const double x304 = x26*x33;
        const double x305 = -x295 - x296 - x297 + x298 - x300 - x302 + x303 + x304;
        const double x306 = R_l_inv_26*x287 + 2*x212;
        const double x307 = x306*x81;
        const double x308 = r_22*x43;
        const double x309 = 2*x308;
        const double x310 = r_22*x44;
        const double x311 = 2*x310;
        const double x312 = r_22*x45;
        const double x313 = 2*x312;
        const double x314 = 2*x239;
        const double x315 = r_22*x314;
        const double x316 = std::pow(r_22, 3)*x64;
        const double x317 = 2*x316;
        const double x318 = r_22*x125;
        const double x319 = x223*x318;
        const double x320 = r_22*x59;
        const double x321 = 2*x320;
        const double x322 = r_22*x63;
        const double x323 = 2*x322;
        const double x324 = r_22*x69;
        const double x325 = 2*x324;
        const double x326 = r_22*x71;
        const double x327 = 2*x326;
        const double x328 = r_22*x61;
        const double x329 = 2*x328;
        const double x330 = r_22*x65;
        const double x331 = 2*x330;
        const double x332 = r_22*x67;
        const double x333 = 2*x332;
        const double x334 = r_22*x70;
        const double x335 = 2*x334;
        const double x336 = x15*x299;
        const double x337 = x15*x301;
        const double x338 = x10*x299;
        const double x339 = x12*x299;
        const double x340 = x14*x299;
        const double x341 = d_5*x3;
        const double x342 = x20*x341;
        const double x343 = x10*x301;
        const double x344 = x12*x301;
        const double x345 = x14*x301;
        const double x346 = x26*x341;
        const double x347 = 4*r_11;
        const double x348 = r_21*x347;
        const double x349 = x264*x348;
        const double x350 = 4*r_12;
        const double x351 = r_13*x268;
        const double x352 = x350*x351;
        const double x353 = 4*r_21;
        const double x354 = x271*x353;
        const double x355 = 4*r_23;
        const double x356 = r_32*r_33*x68;
        const double x357 = x355*x356;
        const double x358 = x280*x3;
        const double x359 = x278*x3;
        const double x360 = x17*x301;
        const double x361 = x23*x299;
        const double x362 = x276*x3;
        const double x363 = x27*x299;
        const double x364 = x21*x301;
        const double x365 = x307 - x309 - x311 - x313 - x315 + x317 - x319 - x321 - x323 - x325 - x327 + x329 + x331 + x333 + x335 - x336 - x337 + x338 + x339 + x340 + x342 + x343 + x344 + x345 + x346 + x349 + x352 + x354 + x357 - x358 - x359 + x360 + x361 + x362 + x363 + x364;
        const double x366 = -x197;
        const double x367 = x194 + x199 + x200 + x202 + x205 + x207 + x209 + x366;
        const double x368 = -x221 + x224 + x226 + x227 + x229 + x230 - x231 - x232;
        const double x369 = x219 + x368;
        const double x370 = -x235 + x236 + x237 + x238 + x240 + x241 - x242 - x243 - x244 - x245 + x246 + x247 + x248 + x249 + x250 + x251 + x252 - x253 - x254 - x255 - x257 - x259 - x260 - x261 - x262 - x266 - x269 - x273 - x275 + x277 + x279 - x281 - x282 - x283 - x284 - x285;
        const double x371 = x123*x288;
        const double x372 = x95*(R_l_inv_22*x287 + 2*x192);
        const double x373 = x156*x291;
        const double x374 = x371 - x372 - x373;
        const double x375 = d_5*x178;
        const double x376 = x213*x29;
        const double x377 = x103*x134;
        const double x378 = x104*x134;
        const double x379 = x178*x20;
        const double x380 = x178*x26;
        const double x381 = -x375 + x376 - x377 - x378 + x379 + x380;
        const double x382 = x162*x306;
        const double x383 = r_23*x45;
        const double x384 = 2*x383;
        const double x385 = std::pow(r_23, 3)*x64;
        const double x386 = 2*x385;
        const double x387 = r_23*x43;
        const double x388 = 2*x387;
        const double x389 = r_23*x44;
        const double x390 = 2*x389;
        const double x391 = r_23*x314;
        const double x392 = r_23*x63;
        const double x393 = 2*x392;
        const double x394 = r_23*x65;
        const double x395 = 2*x394;
        const double x396 = r_23*x66;
        const double x397 = 2*x396;
        const double x398 = r_23*x71;
        const double x399 = 2*x398;
        const double x400 = x223*x228;
        const double x401 = r_23*x59;
        const double x402 = 2*x401;
        const double x403 = r_23*x61;
        const double x404 = 2*x403;
        const double x405 = r_23*x69;
        const double x406 = 2*x405;
        const double x407 = r_23*x70;
        const double x408 = 2*x407;
        const double x409 = 4*d_5;
        const double x410 = x10*x409;
        const double x411 = x12*x409;
        const double x412 = x14*x409;
        const double x413 = 4*x20;
        const double x414 = x10*x413;
        const double x415 = x12*x413;
        const double x416 = x14*x413;
        const double x417 = 4*x26;
        const double x418 = x10*x417;
        const double x419 = x12*x417;
        const double x420 = x14*x417;
        const double x421 = r_13*x57;
        const double x422 = x348*x421;
        const double x423 = r_13*x3;
        const double x424 = x264*x423;
        const double x425 = x274*x353;
        const double x426 = x3*x356;
        const double x427 = x16*x173;
        const double x428 = x18*x341;
        const double x429 = x15*x413;
        const double x430 = x173*x22;
        const double x431 = x24*x341;
        const double x432 = x15*x417;
        const double x433 = x17*x417;
        const double x434 = x18*x3;
        const double x435 = x26*x434;
        const double x436 = x23*x413;
        const double x437 = x24*x3;
        const double x438 = x20*x437;
        const double x439 = x21*x417;
        const double x440 = x176*x22;
        const double x441 = x103*x301;
        const double x442 = x382 - x384 - x386 + x388 + x390 + x391 - x393 - x395 - x397 - x399 + x400 + x402 + x404 + x406 + x408 + x410 + x411 + x412 - x414 - x415 - x416 - x418 - x419 - x420 - x422 - x424 - x425 - x426 + x427 + x428 + x429 + x430 + x431 + x432 - x433 - x435 - x436 - x438 - x439 + x440 + x441;
        const double x443 = x178*x213;
        const double x444 = 8*d_5;
        const double x445 = x444*x92;
        const double x446 = 8*x92;
        const double x447 = x20*x446;
        const double x448 = x26*x446;
        const double x449 = 8*d_4;
        const double x450 = x449*x77;
        const double x451 = x449*x78;
        const double x452 = 8*x189;
        const double x453 = -x180*x208 - x36*(R_l_inv_22*x452 + 4*x192);
        const double x454 = x371 + x372 + x373;
        const double x455 = x375 - x376 + x377 + x378 - x379 - x380;
        const double x456 = -x214;
        const double x457 = -x217;
        const double x458 = -x218;
        const double x459 = x215 + x216 + x370 + x456 + x457 + x458;
        const double x460 = x194 + x199 + x200 + x204 + x206;
        const double x461 = x197 + x202 + x210 + x460;
        const double x462 = x289 + x290 + x292;
        const double x463 = -x307 + x309 + x311 + x313 + x315 - x317 + x319 + x321 + x323 + x325 + x327 - x329 - x331 - x333 - x335 + x336 + x337 - x338 - x339 - x340 - x342 - x343 - x344 - x345 - x346 - x349 - x352 - x354 - x357 + x358 + x359 - x360 - x361 - x362 - x363 - x364;
        const double x464 = x203 + x209 + x366 + x460;
        const double x465 = x215 + x216 + x286 + x456 + x457 + x458;
        const double x466 = R_l_inv_71*x191;
        const double x467 = R_l_inv_31*x190 + x466;
        const double x468 = d_4*x467;
        const double x469 = R_l_inv_70*x191;
        const double x470 = x95*(R_l_inv_30*x190 + x469);
        const double x471 = R_l_inv_72*x191;
        const double x472 = R_l_inv_32*x190 + x471;
        const double x473 = x123*x472;
        const double x474 = -d_3*x472;
        const double x475 = x36*x467;
        const double x476 = -x475;
        const double x477 = R_l_inv_35*x190 + R_l_inv_75*x191;
        const double x478 = -x477*x72;
        const double x479 = x126*x477;
        const double x480 = x36*x477;
        const double x481 = x480*x87;
        const double x482 = x468 + x470 + x473 + x474 + x476 + x478 + x479 + x481;
        const double x483 = R_l_inv_33*x190 + R_l_inv_73*x191;
        const double x484 = r_21*x483;
        const double x485 = R_l_inv_76*x191;
        const double x486 = R_l_inv_36*x190 + x485;
        const double x487 = x107*x486;
        const double x488 = R_l_inv_77*x191;
        const double x489 = R_l_inv_37*x190 + x488;
        const double x490 = x111*x489;
        const double x491 = x121*x486;
        const double x492 = -x491;
        const double x493 = x4*x87;
        const double x494 = -x493;
        const double x495 = x102*x125;
        const double x496 = x105*x125;
        const double x497 = -x496;
        const double x498 = x106*x125;
        const double x499 = -x498;
        const double x500 = x18*x228;
        const double x501 = x228*x24;
        const double x502 = x484 + x487 + x490 + x492 + x494 + x495 + x497 + x499 + x500 + x501;
        const double x503 = x15*x87;
        const double x504 = x10*x87;
        const double x505 = x12*x87;
        const double x506 = x14*x87;
        const double x507 = x16*x29;
        const double x508 = x144*x18;
        const double x509 = x21*x87;
        const double x510 = x22*x29;
        const double x511 = x144*x24;
        const double x512 = x27*x87;
        const double x513 = -x503 + x504 + x505 + x506 + x507 + x508 + x509 + x510 + x511 + x512;
        const double x514 = x489*x81;
        const double x515 = R_l_inv_34*x190 + R_l_inv_74*x191;
        const double x516 = r_22*x515;
        const double x517 = x225*x486;
        const double x518 = x10*x51;
        const double x519 = x12*x51;
        const double x520 = x14*x51;
        const double x521 = x166*x20;
        const double x522 = x10*x55;
        const double x523 = x12*x55;
        const double x524 = x14*x55;
        const double x525 = x166*x26;
        const double x526 = r_11*x264;
        const double x527 = x1*x526;
        const double x528 = r_23*x264*x267;
        const double x529 = x1*x271;
        const double x530 = x160*x356;
        const double x531 = x18*x49;
        const double x532 = x24*x49;
        const double x533 = x1*x16;
        const double x534 = x24*x533;
        const double x535 = x1*x22;
        const double x536 = x18*x535;
        const double x537 = x19*x55;
        const double x538 = x27*x51;
        const double x539 = x21*x55;
        const double x540 = x117*x258;
        const double x541 = x169*x26;
        const double x542 = x308 - x310 + x312 - x316 + x320 + x322 + x324 + x326 - x328 - x330 - x332 - x334 + x514 - x516 + x517 - x518 - x519 - x520 - x521 - x522 - x523 - x524 - x525 - x527 - x528 - x529 - x530 + x531 + x532 - x534 - x536 - x537 - x538 - x539 + x540 + x541;
        const double x543 = x513 + x542;
        const double x544 = R_l_inv_30*x287 + 2*x469;
        const double x545 = x36*x544;
        const double x546 = x95*(R_l_inv_31*x287 + 2*x466);
        const double x547 = x477*x95;
        const double x548 = -x134*x547;
        const double x549 = x545 + x546 + x548;
        const double x550 = R_l_inv_37*x287 + 2*x488;
        const double x551 = x107*x550;
        const double x552 = x1*x515;
        const double x553 = x31*x486;
        const double x554 = x1*x45;
        const double x555 = 2*x235;
        const double x556 = x1*x61;
        const double x557 = x1*x63;
        const double x558 = x1*x70;
        const double x559 = x1*x71;
        const double x560 = x1*x59;
        const double x561 = x1*x66;
        const double x562 = x1*x67;
        const double x563 = x1*x69;
        const double x564 = 4*x16;
        const double x565 = x15*x564;
        const double x566 = 4*x22;
        const double x567 = x15*x566;
        const double x568 = x10*x564;
        const double x569 = x12*x564;
        const double x570 = x14*x564;
        const double x571 = x173*x20;
        const double x572 = x10*x566;
        const double x573 = x12*x566;
        const double x574 = x14*x566;
        const double x575 = x173*x26;
        const double x576 = x3*x526;
        const double x577 = x347*x351;
        const double x578 = x271*x3;
        const double x579 = x274*x355;
        const double x580 = x115*x301;
        const double x581 = x174*x26;
        const double x582 = x17*x566;
        const double x583 = x16*x437;
        const double x584 = x27*x564;
        const double x585 = x22*x434;
        const double x586 = x21*x566;
        const double x587 = -x554 + x555 - x556 - x557 - x558 - x559 + x560 + x561 + x562 + x563 - x565 - x567 + x568 + x569 + x570 + x571 + x572 + x573 + x574 + x575 + x576 + x577 + x578 + x579 - x580 - x581 + x582 + x583 + x584 + x585 + x586;
        const double x588 = x1*x43;
        const double x589 = x1*x44;
        const double x590 = -x588 + x589;
        const double x591 = x551 + x552 - x553 + x587 + x590;
        const double x592 = R_l_inv_36*x287 + 2*x485;
        const double x593 = x592*x81;
        const double x594 = 2*x483;
        const double x595 = r_22*x594;
        const double x596 = d_3*x173;
        const double x597 = x318*x489;
        const double x598 = x4*x564;
        const double x599 = x4*x566;
        const double x600 = x111*x413;
        const double x601 = x111*x417;
        const double x602 = -x593 + x595 - x596 + x597 - x598 - x599 + x600 + x601;
        const double x603 = -x470;
        const double x604 = -x481;
        const double x605 = x468 + x473 + x474 + x475 + x478 + x479 + x603 + x604;
        const double x606 = -x484 - x487 - x490 - x495 + x496 + x498 - x500 - x501;
        const double x607 = x492 + x494 + x606;
        const double x608 = -x308 + x310 - x312 + x316 - x320 - x322 - x324 - x326 + x328 + x330 + x332 + x334 - x514 + x516 - x517 + x518 + x519 + x520 + x521 + x522 + x523 + x524 + x525 + x527 + x528 + x529 + x530 - x531 - x532 + x534 + x536 + x537 + x538 + x539 - x540 - x541;
        const double x609 = x513 + x608;
        const double x610 = x95*(R_l_inv_32*x287 + 2*x471);
        const double x611 = -x123*x544;
        const double x612 = x156*x547;
        const double x613 = x610 + x611 + x612;
        const double x614 = x156*x92;
        const double x615 = x29*x486;
        const double x616 = -x614 - x615;
        const double x617 = x162*x592;
        const double x618 = r_23*x594;
        const double x619 = x228*x489;
        const double x620 = x16*x33;
        const double x621 = x22*x33;
        const double x622 = x111*x299;
        const double x623 = x111*x301;
        const double x624 = -x617 - x618 - x619 - x620 - x621 + x622 + x623;
        const double x625 = x180*x6;
        const double x626 = x178*x486;
        const double x627 = x180*x480 + x36*(R_l_inv_32*x452 + 4*x471);
        const double x628 = x614 + x615;
        const double x629 = -x610 + x611 - x612;
        const double x630 = x491 + x493;
        const double x631 = x606 + x630;
        const double x632 = x468 - x473 + x474 + x478 - x479;
        const double x633 = x476 + x481 + x603 + x632;
        const double x634 = -x545 + x546 + x548;
        const double x635 = x593 - x595 + x596 - x597 + x598 + x599 - x600 - x601;
        const double x636 = x470 + x475 + x604 + x632;
        const double x637 = x484 + x487 + x490 + x495 + x497 + x499 + x500 + x501 + x630;
        const double x638 = x125*x6;
        const double x639 = d_5*x144;
        const double x640 = x144*x20;
        const double x641 = -x640;
        const double x642 = x144*x26;
        const double x643 = -x642;
        const double x644 = x103*x87;
        const double x645 = x104*x87;
        const double x646 = x638 + x639 + x641 + x643 + x644 + x645;
        const double x647 = x10*x125;
        const double x648 = x12*x125;
        const double x649 = x125*x14;
        const double x650 = x4*x48;
        const double x651 = x16*x31;
        const double x652 = x125*x19;
        const double x653 = x20*x228;
        const double x654 = x22*x31;
        const double x655 = x125*x25;
        const double x656 = x228*x26;
        const double x657 = -x647 - x648 - x649 + x650 - x651 - x652 - x653 - x654 - x655 - x656;
        const double x658 = x10*x48;
        const double x659 = x12*x48;
        const double x660 = x14*x48;
        const double x661 = x16*x256;
        const double x662 = x166*x18;
        const double x663 = x20*x49;
        const double x664 = x22*x256;
        const double x665 = x166*x24;
        const double x666 = x26*x49;
        const double x667 = x10*x53;
        const double x668 = x12*x53;
        const double x669 = x14*x53;
        const double x670 = x10*x56;
        const double x671 = x12*x56;
        const double x672 = x14*x56;
        const double x673 = r_11*x1*x421;
        const double x674 = x265*x267;
        const double x675 = x1*x274;
        const double x676 = x272*x356;
        const double x677 = x258*x77;
        const double x678 = x167*x24;
        const double x679 = x26*x533;
        const double x680 = x19*x56;
        const double x681 = x20*x535;
        const double x682 = x25*x53;
        const double x683 = x21*x56;
        const double x684 = x383 + x385 + x387 - x389 + x392 + x394 + x396 + x398 - x401 - x403 - x405 - x407 - x658 - x659 - x660 - x661 - x662 - x663 - x664 - x665 - x666 + x667 + x668 + x669 + x670 + x671 + x672 + x673 + x674 + x675 + x676 - x677 - x678 + x679 + x680 + x681 + x682 + x683;
        const double x685 = x657 + x684;
        const double x686 = -x111;
        const double x687 = -x76 - x77 - x78 + x79 + x80;
        const double x688 = -x638;
        const double x689 = -x639 + x640 + x642 - x644 - x645;
        const double x690 = x688 + x689;
        const double x691 = x16*x178;
        const double x692 = x178*x22;
        const double x693 = x299*x92;
        const double x694 = x301*x92;
        const double x695 = -x691 - x692 + x693 + x694;
        const double x696 = x588 - x589;
        const double x697 = x587 + x696;
        const double x698 = x3*x43;
        const double x699 = x3*x44;
        const double x700 = x3*x45;
        const double x701 = 8*x15;
        const double x702 = x18*x701;
        const double x703 = 8*x102;
        const double x704 = x20*x703;
        const double x705 = x26*x703;
        const double x706 = x24*x701;
        const double x707 = 8*x24;
        const double x708 = x17*x707;
        const double x709 = 8*x22;
        const double x710 = x117*x709;
        const double x711 = 8*x18;
        const double x712 = x23*x711;
        const double x713 = x19*x707;
        const double x714 = x27*x711;
        const double x715 = 8*x26;
        const double x716 = x105*x715;
        const double x717 = x21*x707;
        const double x718 = 4*x316;
        const double x719 = x10*x711;
        const double x720 = x12*x711;
        const double x721 = x14*x711;
        const double x722 = 8*r_12;
        const double x723 = r_11*r_21;
        const double x724 = x57*x722*x723;
        const double x725 = x351*x722;
        const double x726 = x10*x707;
        const double x727 = x12*x707;
        const double x728 = x14*x707;
        const double x729 = 8*x271;
        const double x730 = r_21*x729;
        const double x731 = 8*r_23;
        const double x732 = x356*x731;
        const double x733 = x3*x59;
        const double x734 = x3*x61;
        const double x735 = x3*x63;
        const double x736 = x3*x65;
        const double x737 = x3*x67;
        const double x738 = x3*x69;
        const double x739 = x3*x70;
        const double x740 = x3*x71;
        const double x741 = x554 - x555 + x556 + x557 + x558 + x559 - x560 - x561 - x562 - x563 + x565 + x567 - x568 - x569 - x570 - x571 - x572 - x573 - x574 - x575 - x576 - x577 - x578 - x579 + x580 + x581 - x582 - x583 - x584 - x585 - x586;
        const double x742 = x590 + x741;
        const double x743 = x638 + x689;
        const double x744 = -x383 - x385 - x387 + x389 - x392 - x394 - x396 - x398 + x401 + x403 + x405 + x407 + x658 + x659 + x660 + x661 + x662 + x663 + x664 + x665 + x666 - x667 - x668 - x669 - x670 - x671 - x672 - x673 - x674 - x675 - x676 + x677 + x678 - x679 - x680 - x681 - x682 - x683;
        const double x745 = x657 + x744;
        const double x746 = x639 + x641 + x643 + x644 + x645 + x688;
        const double x747 = -x267;
        const double x748 = r_12*x87;
        const double x749 = -x748;
        const double x750 = d_3*x267;
        const double x751 = 2*Px;
        const double x752 = 2*r_12;
        const double x753 = -d_5*x267 + r_11*x50 + x22*x263 + x24*x752 + x26*x267 + x267*x52 + x35*x752 + x58*x751 + x60*x751 + x62*x751;
        const double x754 = -x750 + x753;
        const double x755 = d_4*x347;
        const double x756 = d_3*x347;
        const double x757 = d_3*x722;
        const double x758 = x750 + x753;
        const double x759 = d_5*r_11;
        const double x760 = r_13*x94;
        const double x761 = r_13*x22;
        const double x762 = r_11*x52;
        const double x763 = r_11*x26;
        const double x764 = x759 + x760 + x761 - x762 - x763;
        const double x765 = x140*x764;
        const double x766 = x138*x752;
        const double x767 = r_12*x125;
        const double x768 = x108*x767;
        const double x769 = -x765 + x766 - x768;
        const double x770 = r_11*x87;
        const double x771 = x263*x99;
        const double x772 = d_5*r_12;
        const double x773 = r_13*x35;
        const double x774 = r_13*x24;
        const double x775 = r_12*x52;
        const double x776 = r_12*x26;
        const double x777 = x772 + x773 + x774 - x775 - x776;
        const double x778 = x146*x777;
        const double x779 = r_11*x125;
        const double x780 = x779*x82;
        const double x781 = -x770 - x771 - x778 - x780;
        const double x782 = r_12*x50;
        const double x783 = r_12*x22;
        const double x784 = 2*x783;
        const double x785 = r_11*x35;
        const double x786 = 2*x785;
        const double x787 = r_11*x24;
        const double x788 = 2*x787;
        const double x789 = d_4*x267;
        const double x790 = x108*x789;
        const double x791 = -x782 - x784 + x786 + x788 + x790;
        const double x792 = r_12*x134;
        const double x793 = x350*x99;
        const double x794 = 4*x108;
        const double x795 = x764*x794;
        const double x796 = d_3*x350;
        const double x797 = x796*x82;
        const double x798 = x108*x756 - x138*x347 - 4*x777*x82;
        const double x799 = x765 - x766 + x768;
        const double x800 = x770 + x771 + x778 + x780;
        const double x801 = 4*x772;
        const double x802 = Py*x423;
        const double x803 = 4*x774;
        const double x804 = 4*x775;
        const double x805 = 4*x776;
        const double x806 = x108*x755;
        const double x807 = r_13*x134;
        const double x808 = r_12*x94;
        const double x809 = -x783 + x785 + x787 - x808;
        const double x810 = 4*r_13;
        const double x811 = d_3*x810;
        const double x812 = -x794*x809 + x807 + x810*x99 + x811*x82;
        const double x813 = 8*x759;
        const double x814 = 8*x762;
        const double x815 = 8*x760;
        const double x816 = 8*x763;
        const double x817 = 8*x761;
        const double x818 = d_4*x722;
        const double x819 = x782 + x784 - x786 - x788 - x790;
        const double x820 = x213*x789;
        const double x821 = Py*x350;
        const double x822 = x821*x92;
        const double x823 = x22*x792;
        const double x824 = Py*x178;
        const double x825 = r_11*x824;
        const double x826 = x24*x755;
        const double x827 = x820 - x822 - x823 + x825 + x826;
        const double x828 = x294*x764;
        const double x829 = x220*x752;
        const double x830 = d_5*x756;
        const double x831 = r_12*x213;
        const double x832 = x125*x831;
        const double x833 = x156*x760;
        const double x834 = x156*x761;
        const double x835 = Py*x347;
        const double x836 = x4*x835;
        const double x837 = x26*x756;
        const double x838 = -x828 + x829 - x830 - x832 - x833 - x834 + x836 + x837;
        const double x839 = x306*x777;
        const double x840 = x263*x43;
        const double x841 = x263*x44;
        const double x842 = x263*x45;
        const double x843 = r_11*x314;
        const double x844 = std::pow(r_11, 3);
        const double x845 = 2*x57;
        const double x846 = x844*x845;
        const double x847 = x223*x779;
        const double x848 = x263*x66;
        const double x849 = x263*x67;
        const double x850 = x263*x70;
        const double x851 = x263*x71;
        const double x852 = x263*x61;
        const double x853 = x263*x63;
        const double x854 = x263*x65;
        const double x855 = x263*x69;
        const double x856 = x409*x760;
        const double x857 = x409*x761;
        const double x858 = Px*x58;
        const double x859 = 4*x94;
        const double x860 = x858*x859;
        const double x861 = Px*x859;
        const double x862 = x60*x861;
        const double x863 = x62*x861;
        const double x864 = x566*x858;
        const double x865 = Px*x566;
        const double x866 = x60*x865;
        const double x867 = x62*x865;
        const double x868 = x15*x835;
        const double x869 = d_5*x26;
        const double x870 = x347*x869;
        const double x871 = r_21*x64;
        const double x872 = r_12*x3;
        const double x873 = x871*x872;
        const double x874 = x271*x350;
        const double x875 = r_23*x810*x871;
        const double x876 = x274*x810;
        const double x877 = Py*x3;
        const double x878 = x787*x877;
        const double x879 = x26*x52;
        const double x880 = x347*x879;
        const double x881 = x22*x94;
        const double x882 = x347*x881;
        const double x883 = x301*x808;
        const double x884 = x783*x877;
        const double x885 = x417*x760;
        const double x886 = 4*x52;
        const double x887 = x761*x886;
        const double x888 = -x839 - x840 - x841 - x842 - x843 + x846 - x847 - x848 - x849 - x850 - x851 + x852 + x853 + x854 + x855 - x856 - x857 + x860 + x862 + x863 + x864 + x866 + x867 + x868 + x870 + x873 + x874 + x875 + x876 - x878 - x880 + x882 + x883 + x884 + x885 + x887;
        const double x889 = R_l_inv_26*x452 + 4*x212;
        const double x890 = x764*x889;
        const double x891 = x350*x43;
        const double x892 = x350*x44;
        const double x893 = x350*x45;
        const double x894 = x239*x350;
        const double x895 = std::pow(r_12, 3);
        const double x896 = 4*x57;
        const double x897 = x895*x896;
        const double x898 = d_3*x223;
        const double x899 = x350*x898;
        const double x900 = x350*x65;
        const double x901 = x350*x67;
        const double x902 = x350*x69;
        const double x903 = x350*x71;
        const double x904 = x350*x59;
        const double x905 = x350*x63;
        const double x906 = x350*x66;
        const double x907 = x350*x70;
        const double x908 = x444*x773;
        const double x909 = x444*x774;
        const double x910 = 8*x35;
        const double x911 = x858*x910;
        const double x912 = Px*x910;
        const double x913 = x60*x912;
        const double x914 = x62*x912;
        const double x915 = x707*x858;
        const double x916 = Px*x707;
        const double x917 = x60*x916;
        const double x918 = x62*x916;
        const double x919 = Py*x722;
        const double x920 = x15*x919;
        const double x921 = x722*x869;
        const double x922 = r_22*x64;
        const double x923 = x723*x922;
        const double x924 = 8*x923;
        const double x925 = r_11*x729;
        const double x926 = r_13*x731;
        const double x927 = x922*x926;
        const double x928 = 8*r_13;
        const double x929 = x356*x928;
        const double x930 = x722*x881;
        const double x931 = x722*x879;
        const double x932 = 8*x94;
        const double x933 = x787*x932;
        const double x934 = x709*x785;
        const double x935 = x35*x722;
        const double x936 = x24*x935;
        const double x937 = x715*x773;
        const double x938 = 8*x52;
        const double x939 = x774*x938;
        const double x940 = -d_5*x757 - x180*x773 - x180*x774 + x213*x756 - x220*x347 + x26*x757 + x4*x919 - x777*(R_l_inv_27*x452 + 4*x222);
        const double x941 = x828 - x829 + x830 + x832 + x833 + x834 - x836 - x837;
        const double x942 = x839 + x840 + x841 + x842 + x843 - x846 + x847 + x848 + x849 + x850 + x851 - x852 - x853 - x854 - x855 + x856 + x857 - x860 - x862 - x863 - x864 - x866 - x867 - x868 - x870 - x873 - x874 - x875 - x876 + x878 + x880 - x882 - x883 - x884 - x885 - x887;
        const double x943 = d_5*x818;
        const double x944 = x213*x755;
        const double x945 = Py*x6;
        const double x946 = x928*x945;
        const double x947 = x449*x774;
        const double x948 = x52*x818;
        const double x949 = x26*x818;
        const double x950 = std::pow(r_13, 3);
        const double x951 = Px*x444;
        const double x952 = Px*x938;
        const double x953 = Px*x715;
        const double x954 = x64*x723;
        const double x955 = 8*r_11;
        const double x956 = r_23*x922;
        const double x957 = d_5*x722;
        const double x958 = Py*r_13;
        const double x959 = x24*x722;
        const double x960 = r_13*x26*x444 + x22*x813 - x22*x814 + x22*x815 + x239*x810 + x24*x957 - x26*x935 - x274*x955 + x35*x957 - x356*x722 + x43*x810 + x44*x810 - x45*x810 - x52*x959 + x58*x951 - x59*x810 + x60*x951 - x60*x952 - x60*x953 - x61*x810 + x62*x951 - x62*x952 - x62*x953 + x65*x810 + x66*x810 - x67*x810 + x69*x810 + x70*x810 + x701*x958 + x707*x773 - x71*x810 - x715*x858 - x722*x956 - x731*x954 - x809*x889 + x810*x898 + x813*x94 - x816*x94 - x858*x938 - x879*x928 - x896*x950;
        const double x961 = -x820 + x822 + x823 - x825 - x826;
        const double x962 = d_3*x807;
        const double x963 = -d_5*x807;
        const double x964 = x486*x789;
        const double x965 = Px*x134;
        const double x966 = x58*x965;
        const double x967 = x60*x965;
        const double x968 = x62*x965;
        const double x969 = x835*x92;
        const double x970 = r_12*x824;
        const double x971 = x52*x807;
        const double x972 = x22*x755;
        const double x973 = x24*x792;
        const double x974 = x26*x807;
        const double x975 = -x962 + x963 - x964 + x966 + x967 + x968 + x969 + x970 + x971 + x972 + x973 + x974;
        const double x976 = x592*x777;
        const double x977 = r_11*x594;
        const double x978 = x489*x779;
        const double x979 = d_3*x801;
        const double x980 = x4*x821;
        const double x981 = x156*x776;
        const double x982 = x33*x958;
        const double x983 = x156*x774;
        const double x984 = x976 + x977 + x978 + x979 - x980 - x981 + x982 + x983;
        const double x985 = x550*x764;
        const double x986 = x44*x752;
        const double x987 = x515*x752;
        const double x988 = x845*x895;
        const double x989 = x43*x752;
        const double x990 = x45*x752;
        const double x991 = x59*x752;
        const double x992 = x63*x752;
        const double x993 = x66*x752;
        const double x994 = x70*x752;
        const double x995 = x486*x767;
        const double x996 = x65*x752;
        const double x997 = x67*x752;
        const double x998 = x69*x752;
        const double x999 = x71*x752;
        const double x1000 = x858*x877;
        const double x1001 = Px*x877;
        const double x1002 = x1001*x60;
        const double x1003 = x1001*x62;
        const double x1004 = x301*x858;
        const double x1005 = Px*x301;
        const double x1006 = x1005*x60;
        const double x1007 = x1005*x62;
        const double x1008 = x15*x821;
        const double x1009 = x26*x801;
        const double x1010 = x3*x954;
        const double x1011 = x271*x347;
        const double x1012 = r_23*x64;
        const double x1013 = x1012*x423;
        const double x1014 = x356*x810;
        const double x1015 = d_5*x802;
        const double x1016 = x409*x774;
        const double x1017 = x347*x94;
        const double x1018 = x1017*x24;
        const double x1019 = r_11*x22*x877;
        const double x1020 = Py*x24*x872;
        const double x1021 = x26*x802;
        const double x1022 = x52*x803;
        const double x1023 = x566*x808;
        const double x1024 = x26*x804;
        const double x1025 = -x1000 - x1002 - x1003 - x1004 - x1006 - x1007 - x1008 - x1009 - x1010 - x1011 - x1013 - x1014 + x1015 + x1016 - x1018 - x1019 - x1020 - x1021 - x1022 + x1023 + x1024 + x985 - x986 - x987 - x988 + x989 + x990 - x991 - x992 - x993 - x994 + x995 + x996 + x997 + x998 + x999;
        const double x1026 = R_l_inv_36*x452 + 4*x485;
        const double x1027 = x1026*x764;
        const double x1028 = x350*x483;
        const double x1029 = d_3*x813;
        const double x1030 = d_3*x489;
        const double x1031 = x1030*x350;
        const double x1032 = x180*x760;
        const double x1033 = x180*x761;
        const double x1034 = Py*x4;
        const double x1035 = x1034*x955;
        const double x1036 = d_3*r_11;
        const double x1037 = x1036*x715;
        const double x1038 = x347*x45;
        const double x1039 = x844*x896;
        const double x1040 = x347*x66;
        const double x1041 = x347*x67;
        const double x1042 = x347*x70;
        const double x1043 = x347*x71;
        const double x1044 = x347*x61;
        const double x1045 = x347*x63;
        const double x1046 = x347*x65;
        const double x1047 = x347*x69;
        const double x1048 = x444*x760;
        const double x1049 = x444*x761;
        const double x1050 = x858*x932;
        const double x1051 = Px*x932;
        const double x1052 = x1051*x60;
        const double x1053 = x1051*x62;
        const double x1054 = x709*x858;
        const double x1055 = Px*x709;
        const double x1056 = x1055*x60;
        const double x1057 = x1055*x62;
        const double x1058 = Py*r_11*x701;
        const double x1059 = x26*x813;
        const double x1060 = r_22*x722*x871;
        const double x1061 = x271*x722;
        const double x1062 = x871*x926;
        const double x1063 = x274*x928;
        const double x1064 = x707*x785;
        const double x1065 = x26*x814;
        const double x1066 = x881*x955;
        const double x1067 = x94*x959;
        const double x1068 = x22*x935;
        const double x1069 = x26*x815;
        const double x1070 = x52*x817;
        const double x1071 = -x1038 + x1039 - x1040 - x1041 - x1042 - x1043 + x1044 + x1045 + x1046 + x1047 - x1048 - x1049 + x1050 + x1052 + x1053 + x1054 + x1056 + x1057 + x1058 + x1059 + x1060 + x1061 + x1062 + x1063 - x1064 - x1065 + x1066 + x1067 + x1068 + x1069 + x1070;
        const double x1072 = x347*x43;
        const double x1073 = x347*x44;
        const double x1074 = -x1072 + x1073;
        const double x1075 = x1071 + x1074 + x347*x515 - x486*x756 + x777*(R_l_inv_37*x452 + 4*x488);
        const double x1076 = -x976 - x977 - x978 - x979 + x980 + x981 - x982 - x983;
        const double x1077 = x1000 + x1002 + x1003 + x1004 + x1006 + x1007 + x1008 + x1009 + x1010 + x1011 + x1013 + x1014 - x1015 - x1016 + x1018 + x1019 + x1020 + x1021 + x1022 - x1023 - x1024 - x985 + x986 + x987 + x988 - x989 - x990 + x991 + x992 + x993 + x994 - x995 - x996 - x997 - x998 - x999;
        const double x1078 = x1036*x449;
        const double x1079 = x486*x755;
        const double x1080 = x1026*x809 - x1030*x810 + x1036*x707 + x1036*x910 - x22*x757 - x483*x810 - x757*x94;
        const double x1081 = x962 + x963 + x964 + x966 + x967 + x968 + x969 + x970 + x971 + x972 + x973 + x974;
        const double x1082 = Px*x156;
        const double x1083 = -x1082*x58;
        const double x1084 = -x1082*x60;
        const double x1085 = -x1082*x62;
        const double x1086 = d_3*x792;
        const double x1087 = d_5*x811;
        const double x1088 = -x756*x94;
        const double x1089 = Py*r_12;
        const double x1090 = -x1089*x33;
        const double x1091 = -x1034*x810;
        const double x1092 = -x22*x756;
        const double x1093 = -x24*x796;
        const double x1094 = -x26*x811;
        const double x1095 = x1083 + x1084 + x1085 + x1086 + x1087 + x1088 + x1090 + x1091 + x1092 + x1093 + x1094;
        const double x1096 = x134*x772;
        const double x1097 = x134*x775;
        const double x1098 = x134*x776;
        const double x1099 = r_13*x824;
        const double x1100 = x134*x774;
        const double x1101 = x1096 - x1097 - x1098 + x1099 + x1100;
        const double x1102 = x267*x44;
        const double x1103 = x267*x43;
        const double x1104 = x267*x45;
        const double x1105 = x845*x950;
        const double x1106 = Px*x409;
        const double x1107 = x1106*x58;
        const double x1108 = x1106*x60;
        const double x1109 = x1106*x62;
        const double x1110 = x267*x65;
        const double x1111 = x267*x66;
        const double x1112 = x267*x69;
        const double x1113 = x267*x70;
        const double x1114 = x267*x59;
        const double x1115 = x267*x61;
        const double x1116 = x267*x67;
        const double x1117 = x267*x71;
        const double x1118 = d_5*x347;
        const double x1119 = x1118*x94;
        const double x1120 = x772*x877;
        const double x1121 = Py*x15*x810;
        const double x1122 = x1118*x22;
        const double x1123 = x24*x801;
        const double x1124 = x810*x869;
        const double x1125 = x858*x886;
        const double x1126 = Px*x886;
        const double x1127 = x1126*x60;
        const double x1128 = x1126*x62;
        const double x1129 = x417*x858;
        const double x1130 = Px*x417;
        const double x1131 = x1130*x60;
        const double x1132 = x1130*x62;
        const double x1133 = x1012*x348;
        const double x1134 = x274*x347;
        const double x1135 = x1012*x872;
        const double x1136 = x350*x356;
        const double x1137 = x566*x760;
        const double x1138 = x774*x877;
        const double x1139 = x1017*x26;
        const double x1140 = x22*x347*x52;
        const double x1141 = x776*x877;
        const double x1142 = x24*x804;
        const double x1143 = x810*x879;
        const double x1144 = -x1102 + x1103 + x1104 + x1105 - x1107 - x1108 - x1109 - x1110 - x1111 - x1112 - x1113 + x1114 + x1115 + x1116 + x1117 - x1119 - x1120 - x1121 - x1122 - x1123 - x1124 + x1125 + x1127 + x1128 + x1129 + x1131 + x1132 + x1133 + x1134 + x1135 + x1136 - x1137 - x1138 + x1139 + x1140 + x1141 + x1142 + x1143;
        const double x1145 = -x1036;
        const double x1146 = -x1096 + x1097 + x1098 - x1099 - x1100;
        const double x1147 = x1083 + x1084 + x1085 - x1086 + x1087 + x1088 + x1090 + x1091 + x1092 + x1093 + x1094;
        const double x1148 = -x22*x818 + x449*x787 - x919*x92 + x945*x955;
        const double x1149 = 16*d_5;
        const double x1150 = 16*x26;
        const double x1151 = 16*x22;
        const double x1152 = 16*x35;
        const double x1153 = Px*x1152;
        const double x1154 = 16*x24;
        const double x1155 = Px*x1154;
        const double x1156 = 16*r_13;
        const double x1157 = x1102 - x1103 - x1104 - x1105 + x1107 + x1108 + x1109 + x1110 + x1111 + x1112 + x1113 - x1114 - x1115 - x1116 - x1117 + x1119 + x1120 + x1121 + x1122 + x1123 + x1124 - x1125 - x1127 - x1128 - x1129 - x1131 - x1132 - x1133 - x1134 - x1135 - x1136 + x1137 + x1138 - x1139 - x1140 - x1141 - x1142 - x1143;
        const double x1158 = -x10 - x12 - x14 + x15 - x17 - x19 - x21 - x23 - x25 - x27;
        const double x1159 = x1158 + x6;
        const double x1160 = -x29;
        const double x1161 = x128 + x184;
        const double x1162 = x100 + x109 + x112 + x92 + x98;
        const double x1163 = a_3 + x152 + x38 + x40 + x74 + x75 + x90;
        const double x1164 = x139 + x141 - x142;
        const double x1165 = x151 + x42 + x84 + x86;
        const double x1166 = x101 + x110 + x113 + x93 + x97;
        const double x1167 = -x159 - x161 - x163 - x164;
        const double x1168 = x115 + x116 + x118 + x120 + x122 + x185;
        const double x1169 = x295 + x296 + x297 - x298 + x300 + x302 - x303 - x304;
        const double x1170 = -x382 + x384 + x386 - x388 - x390 - x391 + x393 + x395 + x397 + x399 - x400 - x402 - x404 - x406 - x408 - x410 - x411 - x412 + x414 + x415 + x416 + x418 + x419 + x420 + x422 + x424 + x425 + x426 - x427 - x428 - x429 - x430 - x431 - x432 + x433 + x435 + x436 + x438 + x439 - x440 - x441;
        const double x1171 = x503 - x504 - x505 - x506 - x507 - x508 - x509 - x510 - x511 - x512;
        const double x1172 = x1171 + x608;
        const double x1173 = -x551 - x552 + x553 + x696 + x741;
        const double x1174 = x1171 + x542;
        const double x1175 = x617 + x618 + x619 + x620 + x621 - x622 - x623;
        const double x1176 = x647 + x648 + x649 - x650 + x651 + x652 + x653 + x654 + x655 + x656;
        const double x1177 = x1176 + x744;
        const double x1178 = x691 + x692 - x693 - x694;
        const double x1179 = x1176 + x684;
        
        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = x0;
        A(0, 2) = x0;
        A(0, 3) = x2;
        A(0, 4) = -x3;
        A(0, 5) = x1;
        A(0, 6) = r_23;
        A(0, 8) = r_23;
        A(1, 0) = x28 + x8;
        A(1, 1) = x29;
        A(1, 2) = x30 + x5;
        A(1, 3) = x32;
        A(1, 4) = -x33;
        A(1, 5) = x31;
        A(1, 6) = x28 + x34;
        A(1, 7) = x29;
        A(1, 8) = x30 + x4;
        A(2, 0) = x114 + x129 + x91;
        A(2, 1) = x137 + x143 + x149;
        A(2, 2) = x129 + x150 + x153;
        A(2, 3) = x158 + x165 + x172;
        A(2, 4) = x173 - x174 - x175 + x176 + x177 + x179 + x181;
        A(2, 5) = x165 + x182 + x183;
        A(2, 6) = x150 + x186 + x91;
        A(2, 7) = x143 + x187 + x188;
        A(2, 8) = x114 + x153 + x186;
        A(3, 0) = x211 + x234 + x286;
        A(3, 1) = x293 + x305 + x365;
        A(3, 2) = x367 + x369 + x370;
        A(3, 3) = x374 + x381 + x442;
        A(3, 4) = x443 + x445 - x447 - x448 + x450 + x451 + x453;
        A(3, 5) = x442 + x454 + x455;
        A(3, 6) = x233 + x459 + x461;
        A(3, 7) = x305 + x462 + x463;
        A(3, 8) = x368 + x464 + x465;
        A(4, 0) = x482 + x502 + x543;
        A(4, 1) = x549 + x591 + x602;
        A(4, 2) = x605 + x607 + x609;
        A(4, 3) = x613 + x616 + x624;
        A(4, 4) = -x625 - x626 + x627;
        A(4, 5) = x624 + x628 + x629;
        A(4, 6) = x543 + x631 + x633;
        A(4, 7) = x591 + x634 + x635;
        A(4, 8) = x609 + x636 + x637;
        A(5, 0) = x646 + x685;
        A(5, 1) = x134*(x686 + x687);
        A(5, 2) = x685 + x690;
        A(5, 3) = x695 + x697;
        A(5, 4) = x698 - x699 - x700 - x702 + x704 + x705 - x706 + x708 - x710 + x712 + x713 + x714 - x716 + x717 + x718 + x719 + x720 + x721 + x724 + x725 + x726 + x727 + x728 + x730 + x732 - x733 + x734 - x735 + x736 + x737 - x738 + x739 - x740;
        A(5, 5) = x695 + x742;
        A(5, 6) = x743 + x745;
        A(5, 7) = x134*(x686 + x81);
        A(5, 8) = x745 + x746;
        
        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x747;
        B(0, 2) = x747;
        B(0, 3) = -x347;
        B(0, 4) = -x722;
        B(0, 5) = x347;
        B(0, 6) = x267;
        B(0, 8) = x267;
        B(1, 0) = x749 + x754;
        B(1, 1) = x755;
        B(1, 2) = x748 + x754;
        B(1, 3) = -x756;
        B(1, 4) = -x757;
        B(1, 5) = x756;
        B(1, 6) = x749 + x758;
        B(1, 7) = x755;
        B(1, 8) = x748 + x758;
        B(2, 0) = x769 + x781 + x791;
        B(2, 1) = -x792 - x793 + x795 - x797 + x798;
        B(2, 2) = x791 + x799 + x800;
        B(2, 3) = -x801 - x802 - x803 + x804 + x805 + x806 + x812;
        B(2, 4) = x108*x818 + x813 - x814 + x815 - x816 + x817;
        B(2, 5) = x801 + x802 + x803 - x804 - x805 - x806 + x812;
        B(2, 6) = x769 + x800 + x819;
        B(2, 7) = x792 + x793 - x795 + x797 + x798;
        B(2, 8) = x781 + x799 + x819;
        B(3, 0) = x827 + x838 + x888;
        B(3, 1) = x890 - x891 - x892 - x893 - x894 + x897 - x899 - x900 - x901 - x902 - x903 + x904 + x905 + x906 + x907 - x908 - x909 + x911 + x913 + x914 + x915 + x917 + x918 + x920 + x921 + x924 + x925 + x927 + x929 - x930 - x931 + x933 + x934 + x936 + x937 + x939 + x940;
        B(3, 2) = x827 + x941 + x942;
        B(3, 3) = -x943 + x944 - x946 - x947 + x948 + x949 + x960;
        B(3, 4) = x449*(r_13*x50 + x22*x267 + 2*x759 - 2*x762 - 2*x763 + x831);
        B(3, 5) = x943 - x944 + x946 + x947 - x948 - x949 + x960;
        B(3, 6) = x838 + x942 + x961;
        B(3, 7) = -x890 + x891 + x892 + x893 + x894 - x897 + x899 + x900 + x901 + x902 + x903 - x904 - x905 - x906 - x907 + x908 + x909 - x911 - x913 - x914 - x915 - x917 - x918 - x920 - x921 - x924 - x925 - x927 - x929 + x930 + x931 - x933 - x934 - x936 - x937 - x939 + x940;
        B(3, 8) = x888 + x941 + x961;
        B(4, 0) = x1025 + x975 + x984;
        B(4, 1) = -x1027 + x1028 - x1029 + x1031 - x1032 - x1033 + x1035 + x1037 + x1075;
        B(4, 2) = x1076 + x1077 + x975;
        B(4, 3) = -x1078 - x1079 + x1080;
        B(4, 4) = -x818*(x125 + x486);
        B(4, 5) = x1078 + x1079 + x1080;
        B(4, 6) = x1025 + x1076 + x1081;
        B(4, 7) = x1027 - x1028 + x1029 - x1031 + x1032 + x1033 - x1035 - x1037 + x1075;
        B(4, 8) = x1077 + x1081 + x984;
        B(5, 0) = x1095 + x1101 + x1144;
        B(5, 1) = x449*(x1145 - x759 - x760 - x761 + x762 + x763);
        B(5, 2) = x1144 + x1146 + x1147;
        B(5, 3) = x1071 + x1072 - x1073 + x1148;
        B(5, 4) = 16*r_11*x271 + r_12*x1152*x24 + 16*x1089*x15 - x1149*x773 - x1149*x774 + x1150*x772 + x1150*x773 - x1150*x775 + x1151*x785 - x1151*x808 + x1152*x858 + x1153*x60 + x1153*x62 + x1154*x858 + x1155*x60 + x1155*x62 + x1156*x356 + x1156*x956 + x43*x722 - x44*x722 - x45*x722 + 16*x52*x774 + 8*x57*x895 + x59*x722 + x63*x722 - x65*x722 + x66*x722 - x67*x722 - x69*x722 + x70*x722 - x71*x722 + 16*x787*x94 + 16*x923;
        B(5, 5) = x1038 - x1039 + x1040 + x1041 + x1042 + x1043 - x1044 - x1045 - x1046 - x1047 + x1048 + x1049 - x1050 - x1052 - x1053 - x1054 - x1056 - x1057 - x1058 - x1059 - x1060 - x1061 - x1062 - x1063 + x1064 + x1065 - x1066 - x1067 - x1068 - x1069 - x1070 + x1074 + x1148;
        B(5, 6) = x1095 + x1146 + x1157;
        B(5, 7) = x449*(x1145 + x764);
        B(5, 8) = x1101 + x1147 + x1157;
        
        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = r_23;
        C(0, 2) = r_23;
        C(0, 3) = x1;
        C(0, 4) = x3;
        C(0, 5) = x2;
        C(0, 6) = x0;
        C(0, 8) = x0;
        C(1, 0) = x1159 + x4;
        C(1, 1) = x1160;
        C(1, 2) = x1158 + x34;
        C(1, 3) = x31;
        C(1, 4) = x33;
        C(1, 5) = x32;
        C(1, 6) = x1159 + x5;
        C(1, 7) = x1160;
        C(1, 8) = x1158 + x8;
        C(2, 0) = x1161 + x1162 + x1163;
        C(2, 1) = x1164 + x137 + x188;
        C(2, 2) = x1161 + x1165 + x1166;
        C(2, 3) = x1167 + x158 + x183;
        C(2, 4) = -x173 + x174 + x175 - x176 - x177 - x179 + x181;
        C(2, 5) = x1167 + x172 + x182;
        C(2, 6) = x1163 + x1166 + x1168;
        C(2, 7) = x1164 + x149 + x187;
        C(2, 8) = x1162 + x1165 + x1168;
        C(3, 0) = x211 + x368 + x459;
        C(3, 1) = x1169 + x293 + x463;
        C(3, 2) = x233 + x367 + x465;
        C(3, 3) = x1170 + x374 + x455;
        C(3, 4) = -x443 - x445 + x447 + x448 - x450 - x451 + x453;
        C(3, 5) = x1170 + x381 + x454;
        C(3, 6) = x286 + x369 + x461;
        C(3, 7) = x1169 + x365 + x462;
        C(3, 8) = x234 + x370 + x464;
        C(4, 0) = x1172 + x482 + x631;
        C(4, 1) = x1173 + x549 + x635;
        C(4, 2) = x1174 + x605 + x637;
        C(4, 3) = x1175 + x613 + x628;
        C(4, 4) = x625 + x626 + x627;
        C(4, 5) = x1175 + x616 + x629;
        C(4, 6) = x1172 + x502 + x633;
        C(4, 7) = x1173 + x602 + x634;
        C(4, 8) = x1174 + x607 + x636;
        C(5, 0) = x1177 + x690;
        C(5, 1) = x134*(x111 + x81);
        C(5, 2) = x1177 + x646;
        C(5, 3) = x1178 + x742;
        C(5, 4) = -x698 + x699 + x700 + x702 - x704 - x705 + x706 - x708 + x710 - x712 - x713 - x714 + x716 - x717 - x718 - x719 - x720 - x721 - x724 - x725 - x726 - x727 - x728 - x730 - x732 + x733 - x734 + x735 - x736 - x737 + x738 - x739 + x740;
        C(5, 5) = x1178 + x697;
        C(5, 6) = x1179 + x746;
        C(5, 7) = x134*(x111 + x687);
        C(5, 8) = x1179 + x743;
        
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
            
            const bool condition_0 = std::fabs((-Px*std::sin(th_0) + Py*std::cos(th_0) + d_5*(r_13*std::sin(th_0) - r_23*std::cos(th_0)))/d_4) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double x2 = safe_acos((Px*x0 - Py*x1 - d_5*(r_13*x0 - r_23*x1))/d_4);
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[3] = x2;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = std::fabs((-Px*std::sin(th_0) + Py*std::cos(th_0) + d_5*(r_13*std::sin(th_0) - r_23*std::cos(th_0)))/d_4) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double x2 = safe_acos((Px*x0 - Py*x1 - d_5*(r_13*x0 - r_23*x1))/d_4);
                // End of temp variables
                const double tmp_sol_value = -x2;
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
            const double th_3 = this_solution[3];
            
            const bool condition_0 = 2*std::fabs(a_2*d_3) >= zero_tolerance || std::fabs(2*a_2*a_3 - 2*a_2*d_4*std::sin(th_3)) >= zero_tolerance || std::fabs(-std::pow(a_2, 2) - std::pow(a_3, 2) + 2*a_3*d_4*std::sin(th_3) - std::pow(d_3, 2) - std::pow(d_4, 2) + std::pow(d_5, 2) + 2*d_5*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_2;
                const double x1 = d_4*std::sin(th_3);
                const double x2 = a_3*x0 - x0*x1;
                const double x3 = std::atan2(d_3*x0, x2);
                const double x4 = std::pow(a_2, 2);
                const double x5 = std::pow(d_3, 2);
                const double x6 = -std::pow(a_3, 2) + 2*a_3*x1 - std::pow(d_4, 2) + std::pow(d_5, 2) + 2*d_5*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2) - x4 - x5;
                const double x7 = safe_sqrt(std::pow(x2, 2) + 4*x4*x5 - std::pow(x6, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[2] = x3 + std::atan2(x7, x6);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx);
            }
            
            const bool condition_1 = 2*std::fabs(a_2*d_3) >= zero_tolerance || std::fabs(2*a_2*a_3 - 2*a_2*d_4*std::sin(th_3)) >= zero_tolerance || std::fabs(-std::pow(a_2, 2) - std::pow(a_3, 2) + 2*a_3*d_4*std::sin(th_3) - std::pow(d_3, 2) - std::pow(d_4, 2) + std::pow(d_5, 2) + 2*d_5*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_2;
                const double x1 = d_4*std::sin(th_3);
                const double x2 = a_3*x0 - x0*x1;
                const double x3 = std::atan2(d_3*x0, x2);
                const double x4 = std::pow(a_2, 2);
                const double x5 = std::pow(d_3, 2);
                const double x6 = -std::pow(a_3, 2) + 2*a_3*x1 - std::pow(d_4, 2) + std::pow(d_5, 2) + 2*d_5*inv_Pz + std::pow(inv_Px, 2) + std::pow(inv_Py, 2) + std::pow(inv_Pz, 2) - x4 - x5;
                const double x7 = safe_sqrt(std::pow(x2, 2) + 4*x4*x5 - std::pow(x6, 2));
                // End of temp variables
                const double tmp_sol_value = x3 + std::atan2(-x7, x6);
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
            const double th_0 = this_solution[0];
            const bool checked_result = std::fabs(Pz - d_5*r_33) <= 9.9999999999999995e-7 && std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - d_5*r_13*std::cos(th_0) - d_5*r_23*std::sin(th_0)) <= 9.9999999999999995e-7;
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
            
            const bool condition_0 = std::fabs(Pz - d_5*r_33) >= 9.9999999999999995e-7 || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - d_5*r_13*std::cos(th_0) - d_5*r_23*std::sin(th_0)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Pz - d_5*r_33;
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = d_4*std::sin(th_3);
                const double x4 = -a_2 - a_3*x1 - d_3*x2 + x1*x3;
                const double x5 = a_3*x2 - d_3*x1 - x2*x3;
                const double x6 = std::cos(th_0);
                const double x7 = std::sin(th_0);
                const double x8 = -Px*x6 - Py*x7 + d_5*r_13*x6 + d_5*r_23*x7;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*x4 - x5*x8, x0*x5 + x4*x8);
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
