#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct dense_cvr_038_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 128;
static constexpr int n_tree_nodes = 35;
static constexpr int intermediate_solution_size = 8;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_0 = 0.165;
static constexpr double a_2 = 0.012;
static constexpr double d_1 = 0.02;
static constexpr double d_3 = 0.1775;
static constexpr double d_4 = -0.0445;
static constexpr double d_5 = 0.045;
static constexpr double pre_transform_special_symbol_23 = 0.18;

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
    ee_pose_raw(0, 3) = a_2*x11 - d_1*x1 + d_3*x17 + d_4*x12 + d_5*x21 + x22*x4;
    ee_pose_raw(1, 0) = -x0*x24 + x13*x27;
    ee_pose_raw(1, 1) = -x0*x27 - x13*x24;
    ee_pose_raw(1, 2) = x28;
    ee_pose_raw(1, 3) = a_2*x23 + d_1*x4 + d_3*x25 + d_4*x24 + d_5*x28 + x1*x22;
    ee_pose_raw(2, 0) = x0*x30 + x13*x33;
    ee_pose_raw(2, 1) = -x0*x33 + x13*x30;
    ee_pose_raw(2, 2) = x34;
    ee_pose_raw(2, 3) = -a_0*x8 + a_2*x29 + d_3*x31 - d_4*x30 + d_5*x34;
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
    const double x13 = x0*x10 - x11*x12;
    const double x14 = std::cos(th_4);
    const double x15 = std::sin(th_4);
    const double x16 = -x14*x9 - x15*(x0*x11 + x10*x12);
    const double x17 = x0*x3;
    const double x18 = x0*x7;
    const double x19 = -x17*x2 - x18*x6;
    const double x20 = -x17*x6 + x18*x2;
    const double x21 = -x10*x4 - x11*x20;
    const double x22 = -x14*x19 - x15*(x10*x20 - x11*x4);
    const double x23 = 1.0*x6;
    const double x24 = 1.0*x2;
    const double x25 = x23*x3 - x24*x7;
    const double x26 = -x23*x7 - x24*x3;
    const double x27 = x11*x26;
    const double x28 = -x10*x15*x26 - x14*x25;
    const double x29 = -1.0*a_0*x3 + pre_transform_special_symbol_23;
    const double x30 = a_2*x26 + d_3*x25 + x29;
    const double x31 = a_0*x18 + d_1*x4;
    const double x32 = a_2*x20 + d_3*x19 + x31;
    const double x33 = -d_4*x27 + x30;
    const double x34 = d_4*x21 + x32;
    const double x35 = d_5*x28 + x33;
    const double x36 = d_5*x22 + x34;
    const double x37 = a_0*x8 - d_1*x0;
    const double x38 = a_2*x12 + d_3*x9 + x37;
    const double x39 = d_4*x13 + x38;
    const double x40 = d_5*x16 + x39;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x1;
    jacobian(0, 2) = x1;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = x13;
    jacobian(0, 5) = x16;
    jacobian(1, 1) = x4;
    jacobian(1, 2) = x4;
    jacobian(1, 3) = x19;
    jacobian(1, 4) = x21;
    jacobian(1, 5) = x22;
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x25;
    jacobian(2, 4) = -x27;
    jacobian(2, 5) = x28;
    jacobian(3, 1) = -pre_transform_special_symbol_23*x4;
    jacobian(3, 2) = -x29*x4;
    jacobian(3, 3) = -x19*x30 + x25*x32;
    jacobian(3, 4) = -x21*x33 - x27*x34;
    jacobian(3, 5) = -x22*x35 + x28*x36;
    jacobian(4, 1) = -pre_transform_special_symbol_23*x0;
    jacobian(4, 2) = -x0*x29;
    jacobian(4, 3) = -x25*x38 + x30*x9;
    jacobian(4, 4) = x13*x33 + x27*x39;
    jacobian(4, 5) = x16*x35 - x28*x40;
    jacobian(5, 2) = x0*x31 + x37*x4;
    jacobian(5, 3) = x19*x38 - x32*x9;
    jacobian(5, 4) = -x13*x34 + x21*x39;
    jacobian(5, 5) = -x16*x36 + x22*x40;
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
    const double x19 = a_2*x18 + d_3*x13 + x6;
    const double x20 = x12*x14 - x14*x8;
    const double x21 = a_0*x11*x14 + d_1*x2;
    const double x22 = a_2*x20 + d_3*x17 + x21;
    const double x23 = std::sin(th_3);
    const double x24 = x18*x23;
    const double x25 = std::cos(th_3);
    const double x26 = -x2*x25 - x20*x23;
    const double x27 = -d_4*x24 + x19;
    const double x28 = d_4*x26 + x22;
    const double x29 = std::cos(th_4);
    const double x30 = std::sin(th_4);
    const double x31 = -x13*x29 - x18*x25*x30;
    const double x32 = -x17*x29 - x30*(-x2*x23 + x20*x25);
    const double x33 = d_5*x31 + x27;
    const double x34 = d_5*x32 + x28;
    const double x35 = 1.0*p_on_ee_x;
    const double x36 = 1.0*x14;
    const double x37 = p_on_ee_z*x36;
    const double x38 = x2*x4;
    const double x39 = x10*x2;
    const double x40 = -x38*x9 - x39*x7;
    const double x41 = -x38*x7 + x39*x9;
    const double x42 = a_0*x39 - d_1*x36;
    const double x43 = a_2*x41 + d_3*x40 + x42;
    const double x44 = -x23*x41 + x25*x36;
    const double x45 = d_4*x44 + x43;
    const double x46 = -x29*x40 - x30*(x23*x36 + x25*x41);
    const double x47 = d_5*x46 + x45;
    const double x48 = -x0*x14 - x1*x35;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = -pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = -x2*x6 + x3;
    jacobian(0, 3) = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x22 - x17*x19;
    jacobian(0, 4) = p_on_ee_y*x24 + p_on_ee_z*x26 - x24*x28 - x26*x27;
    jacobian(0, 5) = -p_on_ee_y*x31 + p_on_ee_z*x32 + x31*x34 - x32*x33;
    jacobian(1, 0) = x35;
    jacobian(1, 1) = -pre_transform_special_symbol_23*x36 + x37;
    jacobian(1, 2) = -x36*x6 + x37;
    jacobian(1, 3) = p_on_ee_x*x13 - p_on_ee_z*x40 - x13*x43 + x19*x40;
    jacobian(1, 4) = -p_on_ee_x*x24 - p_on_ee_z*x44 + x24*x45 + x27*x44;
    jacobian(1, 5) = p_on_ee_x*x31 - p_on_ee_z*x46 - x31*x47 + x33*x46;
    jacobian(2, 1) = x48;
    jacobian(2, 2) = x2*x42 + x21*x36 + x48;
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
        R_l(0, 6) = d_1;
        R_l(0, 7) = -a_2;
        R_l(1, 2) = -a_0;
        R_l(1, 6) = -a_2;
        R_l(1, 7) = -d_1;
        R_l(2, 4) = a_0;
        R_l(3, 6) = -1;
        R_l(4, 7) = 1;
        R_l(5, 5) = 2*a_0*a_2;
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
        const double x4 = d_3*r_23;
        const double x5 = -x4;
        const double x6 = d_4*r_22;
        const double x7 = d_1 - x6;
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
        const double x30 = d_1 + x6;
        const double x31 = x28 + x30;
        const double x32 = d_3*x1;
        const double x33 = -x32;
        const double x34 = d_3*x3;
        const double x35 = x4 + x7;
        const double x36 = Py*r_22;
        const double x37 = x18 + x24 + x36;
        const double x38 = R_l_inv_51*a_0;
        const double x39 = x37*x38;
        const double x40 = R_l_inv_52*a_0;
        const double x41 = d_3*x40;
        const double x42 = a_0*r_22;
        const double x43 = R_l_inv_54*x42;
        const double x44 = std::pow(d_3, 2);
        const double x45 = std::pow(d_4, 2);
        const double x46 = std::pow(d_5, 2);
        const double x47 = std::pow(a_0, 2);
        const double x48 = std::pow(a_2, 2);
        const double x49 = -std::pow(d_1, 2);
        const double x50 = 2*d_5;
        const double x51 = Py*x15;
        const double x52 = Py*x1;
        const double x53 = 2*x18;
        const double x54 = Py*r_23;
        const double x55 = 2*x20;
        const double x56 = 2*x16;
        const double x57 = 2*x36;
        const double x58 = 2*x54;
        const double x59 = std::pow(Px, 2);
        const double x60 = std::pow(r_11, 2);
        const double x61 = x59*x60;
        const double x62 = std::pow(r_12, 2);
        const double x63 = x59*x62;
        const double x64 = std::pow(r_13, 2);
        const double x65 = x59*x64;
        const double x66 = std::pow(Py, 2);
        const double x67 = x66*x9;
        const double x68 = x11*x66;
        const double x69 = x13*x66;
        const double x70 = std::pow(Pz, 2);
        const double x71 = std::pow(r_31, 2)*x70;
        const double x72 = std::pow(r_32, 2)*x70;
        const double x73 = std::pow(r_33, 2)*x70;
        const double x74 = x16*x52 - x20*x50 + x22*x52 + x22*x56 + x24*x53 + x24*x57 - x26*x50 + x26*x55 + x26*x58 + x36*x53 + x44 + x45 + x46 - x47 - x48 + x49 - 2*x51 + x54*x55 + x61 + x63 + x65 + x67 + x68 + x69 + x71 + x72 + x73;
        const double x75 = R_l_inv_55*a_0;
        const double x76 = x74*x75;
        const double x77 = -d_4*x38;
        const double x78 = d_5*r_21;
        const double x79 = r_23*x16;
        const double x80 = r_23*x22;
        const double x81 = r_21*x20;
        const double x82 = r_21*x26;
        const double x83 = x78 + x79 + x80 - x81 - x82;
        const double x84 = R_l_inv_57*a_0;
        const double x85 = x83*x84;
        const double x86 = -x85;
        const double x87 = R_l_inv_56*d_3*x42;
        const double x88 = -x87;
        const double x89 = 2*d_4;
        const double x90 = x37*x75;
        const double x91 = x89*x90;
        const double x92 = -x91;
        const double x93 = a_2 + x39 + x41 + x43 + x76 + x77 + x86 + x88 + x92;
        const double x94 = d_4*r_21;
        const double x95 = -x94;
        const double x96 = Py*r_21;
        const double x97 = x16 + x22 + x96;
        const double x98 = R_l_inv_50*a_0;
        const double x99 = x97*x98;
        const double x100 = -x99;
        const double x101 = a_0*r_21;
        const double x102 = R_l_inv_53*x101;
        const double x103 = -x102;
        const double x104 = d_5*r_22;
        const double x105 = r_23*x18;
        const double x106 = r_23*x24;
        const double x107 = r_22*x20;
        const double x108 = r_22*x26;
        const double x109 = x104 + x105 + x106 - x107 - x108;
        const double x110 = R_l_inv_56*a_0;
        const double x111 = x109*x110;
        const double x112 = -x111;
        const double x113 = d_3*x101;
        const double x114 = R_l_inv_57*x113;
        const double x115 = -x114;
        const double x116 = x100 + x103 + x112 + x115 + x95;
        const double x117 = r_21*x18;
        const double x118 = r_21*x24;
        const double x119 = r_22*x16;
        const double x120 = -x119;
        const double x121 = r_22*x22;
        const double x122 = -x121;
        const double x123 = d_4*r_23;
        const double x124 = x110*x123;
        const double x125 = -d_5 + x20 + x26 + x54;
        const double x126 = x125*x40;
        const double x127 = 2*d_3;
        const double x128 = x125*x127;
        const double x129 = x128*x75;
        const double x130 = -x126 - x129;
        const double x131 = x117 + x118 + x120 + x122 + x124 + x130;
        const double x132 = 2*x98;
        const double x133 = x132*x37;
        const double x134 = 2*x97;
        const double x135 = -x134*x38;
        const double x136 = 4*d_4;
        const double x137 = x75*x97;
        const double x138 = x136*x137;
        const double x139 = -x133 + x135 + x138;
        const double x140 = R_l_inv_54*a_0;
        const double x141 = x1*x140;
        const double x142 = 2*x84;
        const double x143 = x109*x142;
        const double x144 = x110*x32;
        const double x145 = -x141 - x143 + x144;
        const double x146 = 2*x6;
        const double x147 = 2*R_l_inv_53;
        const double x148 = x147*x42;
        const double x149 = 2*x110;
        const double x150 = x149*x83;
        const double x151 = x127*x42;
        const double x152 = R_l_inv_57*x151;
        const double x153 = -x146 - x148 + x150 - x152;
        const double x154 = x102 + x111 + x114 + x94 + x99;
        const double x155 = a_2 - x39 + x41 + x76 + x77 + x91;
        const double x156 = -x43 + x85 + x87;
        const double x157 = x155 + x156;
        const double x158 = x134*x40;
        const double x159 = x125*x132;
        const double x160 = 4*d_3;
        const double x161 = x137*x160;
        const double x162 = -x158 + x159 - x161;
        const double x163 = 2*x123;
        const double x164 = a_0*r_23;
        const double x165 = x147*x164;
        const double x166 = -x117 - x118 + x119 + x121;
        const double x167 = x149*x166;
        const double x168 = x142*x4;
        const double x169 = x163 + x165 + x167 + x168;
        const double x170 = 2*x104;
        const double x171 = 2*x105;
        const double x172 = 2*x106;
        const double x173 = 2*x107;
        const double x174 = 2*x108;
        const double x175 = x110*x29;
        const double x176 = -x170 - x171 - x172 + x173 + x174 + x175;
        const double x177 = 4*x78;
        const double x178 = 4*x81;
        const double x179 = 4*x82;
        const double x180 = 4*x79;
        const double x181 = 4*x80;
        const double x182 = d_4*x3;
        const double x183 = x110*x182;
        const double x184 = 4*x37;
        const double x185 = 8*d_3;
        const double x186 = -x184*x40 - x185*x90;
        const double x187 = x158 + x159 + x161;
        const double x188 = x170 + x171 + x172 - x173 - x174 - x175;
        const double x189 = -x124 + x166;
        const double x190 = x126 + x129;
        const double x191 = x189 + x190;
        const double x192 = x133 + x135 + x138;
        const double x193 = x146 + x148 - x150 + x152;
        const double x194 = 2*a_0;
        const double x195 = a_2*x194;
        const double x196 = d_1*x194;
        const double x197 = a_2*d_1;
        const double x198 = 2*x197;
        const double x199 = x47 + x48 + x49;
        const double x200 = R_l_inv_22*x195 + R_l_inv_32*x196 + R_l_inv_62*x199 + R_l_inv_72*x198;
        const double x201 = d_3*x200;
        const double x202 = R_l_inv_21*x195 + R_l_inv_31*x196 + R_l_inv_61*x199 + R_l_inv_71*x198;
        const double x203 = x202*x37;
        const double x204 = R_l_inv_25*x195 + R_l_inv_35*x196 + R_l_inv_65*x199 + R_l_inv_75*x198;
        const double x205 = x204*x74;
        const double x206 = -d_4*x202;
        const double x207 = R_l_inv_20*x195 + R_l_inv_30*x196 + R_l_inv_60*x199 + R_l_inv_70*x198;
        const double x208 = x207*x97;
        const double x209 = -x208;
        const double x210 = x125*x200;
        const double x211 = -x210;
        const double x212 = x128*x204;
        const double x213 = -x212;
        const double x214 = x204*x37;
        const double x215 = x214*x89;
        const double x216 = -x215;
        const double x217 = x201 + x203 + x205 + x206 + x209 + x211 + x213 + x216;
        const double x218 = R_l_inv_66*x199;
        const double x219 = R_l_inv_26*x195 + R_l_inv_36*x196 + R_l_inv_76*x198 + x218;
        const double x220 = x123*x219;
        const double x221 = x146*x16;
        const double x222 = x146*x22;
        const double x223 = x18*x29;
        const double x224 = x24*x29;
        const double x225 = x220 - x221 - x222 + x223 + x224;
        const double x226 = R_l_inv_24*x195 + R_l_inv_34*x196 + R_l_inv_64*x199 + R_l_inv_74*x198;
        const double x227 = r_22*x226;
        const double x228 = R_l_inv_67*x199;
        const double x229 = R_l_inv_27*x195 + R_l_inv_37*x196 + R_l_inv_77*x198 + x228;
        const double x230 = x229*x83;
        const double x231 = d_3*r_22;
        const double x232 = x219*x231;
        const double x233 = d_5*x32;
        const double x234 = x4*x56;
        const double x235 = 2*x22;
        const double x236 = x235*x4;
        const double x237 = x20*x32;
        const double x238 = x26*x32;
        const double x239 = x227 - x230 - x232 - x233 - x234 - x236 + x237 + x238;
        const double x240 = x225 + x239;
        const double x241 = std::pow(r_21, 3)*x66;
        const double x242 = r_21*x44;
        const double x243 = r_21*x45;
        const double x244 = r_21*x46;
        const double x245 = R_l_inv_23*x195 + R_l_inv_33*x196 + R_l_inv_63*x199 + R_l_inv_73*x198;
        const double x246 = r_21*x245;
        const double x247 = x109*x219;
        const double x248 = r_21*x61;
        const double x249 = r_21*x68;
        const double x250 = r_21*x69;
        const double x251 = r_21*x71;
        const double x252 = d_3*r_21;
        const double x253 = x229*x252;
        const double x254 = r_21*x63;
        const double x255 = r_21*x65;
        const double x256 = r_21*x72;
        const double x257 = r_21*x73;
        const double x258 = x15*x56;
        const double x259 = x15*x235;
        const double x260 = x10*x56;
        const double x261 = x12*x56;
        const double x262 = x14*x56;
        const double x263 = d_5*x1;
        const double x264 = x20*x263;
        const double x265 = x10*x235;
        const double x266 = x12*x235;
        const double x267 = x14*x235;
        const double x268 = x26*x263;
        const double x269 = 2*r_11;
        const double x270 = r_12*x59;
        const double x271 = r_22*x270;
        const double x272 = x269*x271;
        const double x273 = 2*r_13;
        const double x274 = r_23*x59;
        const double x275 = r_11*x273*x274;
        const double x276 = 2*r_31;
        const double x277 = r_32*x70;
        const double x278 = r_22*x277;
        const double x279 = x276*x278;
        const double x280 = r_23*r_33;
        const double x281 = x280*x70;
        const double x282 = x276*x281;
        const double x283 = x1*x18;
        const double x284 = x24*x283;
        const double x285 = x20*x26;
        const double x286 = x1*x285;
        const double x287 = x16*x22;
        const double x288 = x1*x287;
        const double x289 = x25*x56;
        const double x290 = x27*x56;
        const double x291 = x19*x235;
        const double x292 = x21*x235;
        const double x293 = x241 - x242 - x243 - x244 - x246 - x247 + x248 + x249 + x250 + x251 - x253 - x254 - x255 - x256 - x257 - x258 - x259 + x260 + x261 + x262 + x264 + x265 + x266 + x267 + x268 + x272 + x275 + x279 + x282 - x284 - x286 + x288 + x289 + x290 + x291 + x292;
        const double x294 = x235 + x52 + x56;
        const double x295 = -x202*x294;
        const double x296 = 2*x24;
        const double x297 = x296 + x53 + x57;
        const double x298 = x207*x297;
        const double x299 = x204*x97;
        const double x300 = x136*x299;
        const double x301 = x295 - x298 + x300;
        const double x302 = 4*a_0;
        const double x303 = a_2*x302;
        const double x304 = d_1*x302;
        const double x305 = 4*x197;
        const double x306 = R_l_inv_27*x303 + R_l_inv_37*x304 + R_l_inv_77*x305 + 2*x228;
        const double x307 = x109*x306;
        const double x308 = x1*x226;
        const double x309 = d_5*x34;
        const double x310 = x219*x32;
        const double x311 = 4*x18;
        const double x312 = x311*x4;
        const double x313 = 4*x24;
        const double x314 = x313*x4;
        const double x315 = x20*x34;
        const double x316 = x26*x34;
        const double x317 = -x307 - x308 - x309 + x310 - x312 - x314 + x315 + x316;
        const double x318 = R_l_inv_26*x303 + R_l_inv_36*x304 + R_l_inv_76*x305 + 2*x218;
        const double x319 = x318*x83;
        const double x320 = r_22*x44;
        const double x321 = 2*x320;
        const double x322 = r_22*x45;
        const double x323 = 2*x322;
        const double x324 = r_22*x46;
        const double x325 = 2*x324;
        const double x326 = 2*x245;
        const double x327 = r_22*x326;
        const double x328 = std::pow(r_22, 3)*x66;
        const double x329 = 2*x328;
        const double x330 = r_22*x127;
        const double x331 = x229*x330;
        const double x332 = r_22*x61;
        const double x333 = 2*x332;
        const double x334 = r_22*x65;
        const double x335 = 2*x334;
        const double x336 = r_22*x71;
        const double x337 = 2*x336;
        const double x338 = r_22*x73;
        const double x339 = 2*x338;
        const double x340 = r_22*x63;
        const double x341 = 2*x340;
        const double x342 = r_22*x67;
        const double x343 = 2*x342;
        const double x344 = r_22*x69;
        const double x345 = 2*x344;
        const double x346 = r_22*x72;
        const double x347 = 2*x346;
        const double x348 = x15*x311;
        const double x349 = x15*x313;
        const double x350 = x10*x311;
        const double x351 = x12*x311;
        const double x352 = x14*x311;
        const double x353 = d_5*x3;
        const double x354 = x20*x353;
        const double x355 = x10*x313;
        const double x356 = x12*x313;
        const double x357 = x14*x313;
        const double x358 = x26*x353;
        const double x359 = 4*r_11;
        const double x360 = r_21*x359;
        const double x361 = x270*x360;
        const double x362 = 4*r_12;
        const double x363 = r_13*x274;
        const double x364 = x362*x363;
        const double x365 = 4*x277;
        const double x366 = r_21*r_31;
        const double x367 = x365*x366;
        const double x368 = x280*x365;
        const double x369 = x287*x3;
        const double x370 = x285*x3;
        const double x371 = x17*x313;
        const double x372 = x23*x311;
        const double x373 = x18*x3;
        const double x374 = x24*x373;
        const double x375 = x27*x311;
        const double x376 = x21*x313;
        const double x377 = x319 - x321 - x323 - x325 - x327 + x329 - x331 - x333 - x335 - x337 - x339 + x341 + x343 + x345 + x347 - x348 - x349 + x350 + x351 + x352 + x354 + x355 + x356 + x357 + x358 + x361 + x364 + x367 + x368 - x369 - x370 + x371 + x372 + x374 + x375 + x376;
        const double x378 = -x203;
        const double x379 = x201 + x205 + x206 + x208 + x211 + x213 + x215 + x378;
        const double x380 = -x227 + x230 + x232 + x233 + x234 + x236 - x237 - x238;
        const double x381 = x225 + x380;
        const double x382 = -x241 + x242 + x243 + x244 + x246 + x247 - x248 - x249 - x250 - x251 + x253 + x254 + x255 + x256 + x257 + x258 + x259 - x260 - x261 - x262 - x264 - x265 - x266 - x267 - x268 - x272 - x275 - x279 - x282 + x284 + x286 - x288 - x289 - x290 - x291 - x292;
        const double x383 = 2*x26;
        const double x384 = x383 - x50 + x55 + x58;
        const double x385 = x207*x384;
        const double x386 = x200*x294;
        const double x387 = x160*x299;
        const double x388 = x385 - x386 - x387;
        const double x389 = d_5*x182;
        const double x390 = x219*x29;
        const double x391 = x105*x136;
        const double x392 = x106*x136;
        const double x393 = x182*x20;
        const double x394 = x182*x26;
        const double x395 = -x389 + x390 - x391 - x392 + x393 + x394;
        const double x396 = x1*x24;
        const double x397 = 2*x119 + 2*x121 - x283 - x396;
        const double x398 = x219*x397;
        const double x399 = r_23*x46;
        const double x400 = 2*x399;
        const double x401 = std::pow(r_23, 3)*x66;
        const double x402 = 2*x401;
        const double x403 = r_23*x44;
        const double x404 = 2*x403;
        const double x405 = r_23*x45;
        const double x406 = 2*x405;
        const double x407 = r_23*x326;
        const double x408 = r_23*x65;
        const double x409 = 2*x408;
        const double x410 = r_23*x67;
        const double x411 = 2*x410;
        const double x412 = r_23*x68;
        const double x413 = 2*x412;
        const double x414 = r_23*x73;
        const double x415 = 2*x414;
        const double x416 = 2*x4;
        const double x417 = x229*x416;
        const double x418 = r_23*x61;
        const double x419 = 2*x418;
        const double x420 = r_23*x63;
        const double x421 = 2*x420;
        const double x422 = r_23*x71;
        const double x423 = 2*x422;
        const double x424 = r_23*x72;
        const double x425 = 2*x424;
        const double x426 = 4*d_5;
        const double x427 = x10*x426;
        const double x428 = x12*x426;
        const double x429 = x14*x426;
        const double x430 = 4*x20;
        const double x431 = x10*x430;
        const double x432 = x12*x430;
        const double x433 = x14*x430;
        const double x434 = 4*x26;
        const double x435 = x10*x434;
        const double x436 = x12*x434;
        const double x437 = x14*x434;
        const double x438 = r_13*x59;
        const double x439 = x360*x438;
        const double x440 = x270*x3;
        const double x441 = r_13*x440;
        const double x442 = r_33*x70;
        const double x443 = 4*x366*x442;
        const double x444 = x277*x3;
        const double x445 = r_33*x444;
        const double x446 = x16*x177;
        const double x447 = x18*x353;
        const double x448 = x15*x430;
        const double x449 = x177*x22;
        const double x450 = x24*x353;
        const double x451 = x15*x434;
        const double x452 = x17*x434;
        const double x453 = x26*x373;
        const double x454 = x23*x430;
        const double x455 = x24*x3;
        const double x456 = x20*x455;
        const double x457 = x21*x434;
        const double x458 = x180*x22;
        const double x459 = x105*x313;
        const double x460 = x398 - x400 - x402 + x404 + x406 + x407 - x409 - x411 - x413 - x415 + x417 + x419 + x421 + x423 + x425 + x427 + x428 + x429 - x431 - x432 - x433 - x435 - x436 - x437 - x439 - x441 - x443 - x445 + x446 + x447 + x448 + x449 + x450 + x451 - x452 - x453 - x454 - x456 - x457 + x458 + x459;
        const double x461 = x182*x219;
        const double x462 = 8*d_5;
        const double x463 = x462*x94;
        const double x464 = 8*x94;
        const double x465 = x20*x464;
        const double x466 = x26*x464;
        const double x467 = 8*d_4;
        const double x468 = x467*x79;
        const double x469 = x467*x80;
        const double x470 = Py*x3;
        const double x471 = x311 + x313 + x470;
        const double x472 = -x185*x214 - x200*x471;
        const double x473 = x385 + x386 + x387;
        const double x474 = x389 - x390 + x391 + x392 - x393 - x394;
        const double x475 = -x220;
        const double x476 = -x223;
        const double x477 = -x224;
        const double x478 = x221 + x222 + x382 + x475 + x476 + x477;
        const double x479 = x201 + x205 + x206 + x210 + x212;
        const double x480 = x203 + x208 + x216 + x479;
        const double x481 = x295 + x298 + x300;
        const double x482 = -x319 + x321 + x323 + x325 + x327 - x329 + x331 + x333 + x335 + x337 + x339 - x341 - x343 - x345 - x347 + x348 + x349 - x350 - x351 - x352 - x354 - x355 - x356 - x357 - x358 - x361 - x364 - x367 - x368 + x369 + x370 - x371 - x372 - x374 - x375 - x376;
        const double x483 = x209 + x215 + x378 + x479;
        const double x484 = x221 + x222 + x293 + x475 + x476 + x477;
        const double x485 = R_l_inv_22*x196 - R_l_inv_32*x195 + R_l_inv_62*x198 - R_l_inv_72*x199;
        const double x486 = d_3*x485;
        const double x487 = R_l_inv_21*x196 - R_l_inv_31*x195 + R_l_inv_61*x198 - R_l_inv_71*x199;
        const double x488 = x37*x487;
        const double x489 = R_l_inv_25*x196 - R_l_inv_35*x195 + R_l_inv_65*x198 - R_l_inv_75*x199;
        const double x490 = x489*x74;
        const double x491 = -d_4*x487;
        const double x492 = R_l_inv_20*x196 - R_l_inv_30*x195 + R_l_inv_60*x198 - R_l_inv_70*x199;
        const double x493 = x492*x97;
        const double x494 = -x493;
        const double x495 = x125*x485;
        const double x496 = -x495;
        const double x497 = x128*x489;
        const double x498 = -x497;
        const double x499 = x37*x489;
        const double x500 = x499*x89;
        const double x501 = -x500;
        const double x502 = x486 + x488 + x490 + x491 + x494 + x496 + x498 + x501;
        const double x503 = R_l_inv_23*x196 - R_l_inv_33*x195 + R_l_inv_63*x198 - R_l_inv_73*x199;
        const double x504 = r_21*x503;
        const double x505 = -x504;
        const double x506 = R_l_inv_76*x199;
        const double x507 = R_l_inv_26*x196 - R_l_inv_36*x195 + R_l_inv_66*x198 - x506;
        const double x508 = x109*x507;
        const double x509 = -x508;
        const double x510 = x123*x507;
        const double x511 = R_l_inv_77*x199;
        const double x512 = R_l_inv_27*x196 - R_l_inv_37*x195 + R_l_inv_67*x198 - x511;
        const double x513 = x252*x512;
        const double x514 = -x513;
        const double x515 = x4*x89;
        const double x516 = -x515;
        const double x517 = x104*x127;
        const double x518 = x107*x127;
        const double x519 = -x518;
        const double x520 = x108*x127;
        const double x521 = -x520;
        const double x522 = x4*x53;
        const double x523 = x296*x4;
        const double x524 = x505 + x509 + x510 + x514 + x516 + x517 + x519 + x521 + x522 + x523;
        const double x525 = x15*x89;
        const double x526 = x10*x89;
        const double x527 = x12*x89;
        const double x528 = x14*x89;
        const double x529 = x16*x29;
        const double x530 = x146*x18;
        const double x531 = x21*x89;
        const double x532 = x22*x29;
        const double x533 = x146*x24;
        const double x534 = x27*x89;
        const double x535 = -x525 + x526 + x527 + x528 + x529 + x530 + x531 + x532 + x533 + x534;
        const double x536 = R_l_inv_24*x196 - R_l_inv_34*x195 + R_l_inv_64*x198 - R_l_inv_74*x199;
        const double x537 = r_22*x536;
        const double x538 = x512*x83;
        const double x539 = x231*x507;
        const double x540 = x10*x53;
        const double x541 = x12*x53;
        const double x542 = x14*x53;
        const double x543 = x170*x20;
        const double x544 = x10*x296;
        const double x545 = x12*x296;
        const double x546 = x14*x296;
        const double x547 = x170*x26;
        const double x548 = r_11*x1;
        const double x549 = x270*x548;
        const double x550 = r_23*x270*x273;
        const double x551 = r_31*x1;
        const double x552 = x277*x551;
        const double x553 = x277*x280;
        const double x554 = 2*x553;
        const double x555 = x15*x53;
        const double x556 = x15*x296;
        const double x557 = x16*x396;
        const double x558 = x22*x283;
        const double x559 = x19*x296;
        const double x560 = x27*x53;
        const double x561 = x21*x296;
        const double x562 = x119*x235;
        const double x563 = x107*x383;
        const double x564 = x320 - x322 + x324 - x328 + x332 + x334 + x336 + x338 - x340 - x342 - x344 - x346 + x537 - x538 - x539 - x540 - x541 - x542 - x543 - x544 - x545 - x546 - x547 - x549 - x550 - x552 - x554 + x555 + x556 - x557 - x558 - x559 - x560 - x561 + x562 + x563;
        const double x565 = x535 + x564;
        const double x566 = -x294*x487;
        const double x567 = x297*x492;
        const double x568 = x489*x97;
        const double x569 = x136*x568;
        const double x570 = x566 - x567 + x569;
        const double x571 = R_l_inv_27*x304 - R_l_inv_37*x303 + R_l_inv_67*x305 - 2*x511;
        const double x572 = x109*x571;
        const double x573 = x1*x536;
        const double x574 = x32*x507;
        const double x575 = x1*x46;
        const double x576 = 2*x241;
        const double x577 = x1*x63;
        const double x578 = x1*x65;
        const double x579 = x1*x72;
        const double x580 = x1*x73;
        const double x581 = x1*x61;
        const double x582 = x1*x68;
        const double x583 = x1*x69;
        const double x584 = x1*x71;
        const double x585 = 4*x16;
        const double x586 = x15*x585;
        const double x587 = 4*x22;
        const double x588 = x15*x587;
        const double x589 = x10*x585;
        const double x590 = x12*x585;
        const double x591 = x14*x585;
        const double x592 = x177*x20;
        const double x593 = x10*x587;
        const double x594 = x12*x587;
        const double x595 = x14*x587;
        const double x596 = x177*x26;
        const double x597 = r_11*x440;
        const double x598 = x359*x363;
        const double x599 = r_31*x444;
        const double x600 = 4*r_31*x281;
        const double x601 = x117*x313;
        const double x602 = x178*x26;
        const double x603 = x17*x587;
        const double x604 = x16*x455;
        const double x605 = x27*x585;
        const double x606 = x22*x373;
        const double x607 = x21*x587;
        const double x608 = -x575 + x576 - x577 - x578 - x579 - x580 + x581 + x582 + x583 + x584 - x586 - x588 + x589 + x590 + x591 + x592 + x593 + x594 + x595 + x596 + x597 + x598 + x599 + x600 - x601 - x602 + x603 + x604 + x605 + x606 + x607;
        const double x609 = x1*x44;
        const double x610 = x1*x45;
        const double x611 = -x609 + x610;
        const double x612 = -x572 - x573 + x574 + x608 + x611;
        const double x613 = R_l_inv_26*x304 - R_l_inv_36*x303 + R_l_inv_66*x305 - 2*x506;
        const double x614 = x613*x83;
        const double x615 = 2*x503;
        const double x616 = r_22*x615;
        const double x617 = d_3*x177;
        const double x618 = x330*x512;
        const double x619 = x4*x585;
        const double x620 = x4*x587;
        const double x621 = d_3*x178;
        const double x622 = d_3*x179;
        const double x623 = x614 - x616 - x617 - x618 - x619 - x620 + x621 + x622;
        const double x624 = -x488;
        const double x625 = x486 + x490 + x491 + x493 + x496 + x498 + x500 + x624;
        const double x626 = x504 + x508 + x513 - x517 + x518 + x520 - x522 - x523;
        const double x627 = x510 + x516 + x626;
        const double x628 = -x320 + x322 - x324 + x328 - x332 - x334 - x336 - x338 + x340 + x342 + x344 + x346 - x537 + x538 + x539 + x540 + x541 + x542 + x543 + x544 + x545 + x546 + x547 + x549 + x550 + x552 + x554 - x555 - x556 + x557 + x558 + x559 + x560 + x561 - x562 - x563;
        const double x629 = x535 + x628;
        const double x630 = x384*x492;
        const double x631 = x294*x485;
        const double x632 = x160*x568;
        const double x633 = x630 - x631 - x632;
        const double x634 = x160*x94;
        const double x635 = -x634;
        const double x636 = x29*x507;
        const double x637 = x635 + x636;
        const double x638 = x397*x507;
        const double x639 = r_23*x615;
        const double x640 = x416*x512;
        const double x641 = x16*x34;
        const double x642 = x22*x34;
        const double x643 = x117*x160;
        const double x644 = x118*x160;
        const double x645 = x638 + x639 + x640 - x641 - x642 + x643 + x644;
        const double x646 = x185*x6;
        const double x647 = x182*x507;
        const double x648 = -x185*x499 - x471*x485;
        const double x649 = x634 - x636;
        const double x650 = x630 + x631 + x632;
        const double x651 = -x510 + x515;
        const double x652 = x626 + x651;
        const double x653 = x486 + x490 + x491 + x495 + x497;
        const double x654 = x488 + x493 + x501 + x653;
        const double x655 = x566 + x567 + x569;
        const double x656 = -x614 + x616 + x617 + x618 + x619 + x620 - x621 - x622;
        const double x657 = x494 + x500 + x624 + x653;
        const double x658 = x505 + x509 + x514 + x517 + x519 + x521 + x522 + x523 + x651;
        const double x659 = R_l_inv_41*x196*x37;
        const double x660 = -x659;
        const double x661 = a_0*d_1;
        const double x662 = -R_l_inv_42*x127*x661;
        const double x663 = -R_l_inv_45*x196*x74;
        const double x664 = R_l_inv_40*x661;
        const double x665 = x134*x664;
        const double x666 = R_l_inv_41*x661*x89;
        const double x667 = R_l_inv_42*x125*x196;
        const double x668 = x125*x304;
        const double x669 = R_l_inv_45*d_3;
        const double x670 = x668*x669;
        const double x671 = R_l_inv_45*x661;
        const double x672 = x136*x37*x671;
        const double x673 = x660 + x662 + x663 + x665 + x666 + x667 + x670 + x672;
        const double x674 = x127*x6;
        const double x675 = x50*x6;
        const double x676 = x146*x20;
        const double x677 = -x676;
        const double x678 = x146*x26;
        const double x679 = -x678;
        const double x680 = R_l_inv_44*d_1;
        const double x681 = 2*x42*x680;
        const double x682 = -x681;
        const double x683 = x105*x89;
        const double x684 = x106*x89;
        const double x685 = R_l_inv_43*x661;
        const double x686 = x1*x685;
        const double x687 = R_l_inv_46*x109*x196;
        const double x688 = R_l_inv_47*x196*x83;
        const double x689 = R_l_inv_46*d_1*x151;
        const double x690 = R_l_inv_47*x661;
        const double x691 = x32*x690;
        const double x692 = x674 + x675 + x677 + x679 + x682 + x683 + x684 + x686 + x687 + x688 + x689 + x691;
        const double x693 = x10*x127;
        const double x694 = x12*x127;
        const double x695 = x127*x14;
        const double x696 = x4*x50;
        const double x697 = x16*x32;
        const double x698 = x127*x19;
        const double x699 = x4*x55;
        const double x700 = x22*x32;
        const double x701 = x127*x25;
        const double x702 = x383*x4;
        const double x703 = -x693 - x694 - x695 + x696 - x697 - x698 - x699 - x700 - x701 - x702;
        const double x704 = x10*x50;
        const double x705 = x12*x50;
        const double x706 = x14*x50;
        const double x707 = x16*x263;
        const double x708 = x170*x18;
        const double x709 = x15*x55;
        const double x710 = x22*x263;
        const double x711 = x170*x24;
        const double x712 = x15*x383;
        const double x713 = x10*x55;
        const double x714 = x12*x55;
        const double x715 = x14*x55;
        const double x716 = x10*x383;
        const double x717 = x12*x383;
        const double x718 = x14*x383;
        const double x719 = x438*x548;
        const double x720 = x271*x273;
        const double x721 = x442*x551;
        const double x722 = 2*r_33*x278;
        const double x723 = x235*x79;
        const double x724 = x105*x296;
        const double x725 = R_l_inv_46*x661;
        const double x726 = x163*x725;
        const double x727 = x1*x16*x26;
        const double x728 = x19*x383;
        const double x729 = x1*x20*x22;
        const double x730 = x25*x55;
        const double x731 = x21*x383;
        const double x732 = x399 + x401 + x403 - x405 + x408 + x410 + x412 + x414 - x418 - x420 - x422 - x424 - x704 - x705 - x706 - x707 - x708 - x709 - x710 - x711 - x712 + x713 + x714 + x715 + x716 + x717 + x718 + x719 + x720 + x721 + x722 - x723 - x724 - x726 + x727 + x728 + x729 + x730 + x731;
        const double x733 = x703 + x732;
        const double x734 = x184*x664;
        const double x735 = x304*x97;
        const double x736 = R_l_inv_41*x735;
        const double x737 = x671*x97;
        const double x738 = -x467*x737;
        const double x739 = x734 + x736 + x738;
        const double x740 = 4*x101*x680;
        const double x741 = R_l_inv_47*x304;
        const double x742 = x109*x741;
        const double x743 = 4*d_1;
        const double x744 = R_l_inv_46*x113*x743;
        const double x745 = x635 + x740 + x742 - x744;
        const double x746 = x426*x94;
        const double x747 = x136*x79;
        const double x748 = x136*x80;
        const double x749 = R_l_inv_46*x304;
        const double x750 = x749*x83;
        const double x751 = x430*x94;
        const double x752 = x434*x94;
        const double x753 = x3*x685;
        const double x754 = x34*x690;
        const double x755 = -x746 - x747 - x748 - x750 + x751 + x752 + x753 + x754;
        const double x756 = -x665;
        const double x757 = -x672;
        const double x758 = x659 + x662 + x663 + x666 + x667 + x670 + x756 + x757;
        const double x759 = -x674 + x681 - x688 - x689;
        const double x760 = -x675 + x676 + x678 - x683 - x684 - x686 - x687 - x691;
        const double x761 = x759 + x760;
        const double x762 = -R_l_inv_40*x668;
        const double x763 = R_l_inv_42*x735;
        const double x764 = x185*x737;
        const double x765 = x762 + x763 + x764;
        const double x766 = x16*x182;
        const double x767 = x182*x22;
        const double x768 = R_l_inv_43*x164*x743;
        const double x769 = x166*x749;
        const double x770 = x311*x94;
        const double x771 = x313*x94;
        const double x772 = x4*x741;
        const double x773 = -x766 - x767 - x768 - x769 + x770 + x771 - x772;
        const double x774 = x749*x94;
        const double x775 = x609 - x610;
        const double x776 = x608 - x774 + x775;
        const double x777 = x3*x45;
        const double x778 = x3*x46;
        const double x779 = x3*x44;
        const double x780 = 4*x328;
        const double x781 = x3*x61;
        const double x782 = x3*x65;
        const double x783 = x3*x71;
        const double x784 = x3*x73;
        const double x785 = x3*x63;
        const double x786 = x3*x67;
        const double x787 = x3*x69;
        const double x788 = x3*x72;
        const double x789 = 8*x15;
        const double x790 = x18*x789;
        const double x791 = x24*x789;
        const double x792 = 8*x18;
        const double x793 = x10*x792;
        const double x794 = x12*x792;
        const double x795 = x14*x792;
        const double x796 = 8*x104;
        const double x797 = x20*x796;
        const double x798 = 8*x24;
        const double x799 = x10*x798;
        const double x800 = x12*x798;
        const double x801 = x14*x798;
        const double x802 = x26*x796;
        const double x803 = 8*r_12;
        const double x804 = r_11*r_21;
        const double x805 = x59*x803*x804;
        const double x806 = x363*x803;
        const double x807 = 8*x277;
        const double x808 = x366*x807;
        const double x809 = 8*x553;
        const double x810 = 8*x22;
        const double x811 = x119*x810;
        const double x812 = 8*x26;
        const double x813 = x107*x812;
        const double x814 = 8*a_0;
        const double x815 = d_1*x814;
        const double x816 = R_l_inv_46*x815;
        const double x817 = x6*x816;
        const double x818 = x17*x798;
        const double x819 = x23*x792;
        const double x820 = x19*x798;
        const double x821 = x27*x792;
        const double x822 = x21*x798;
        const double x823 = R_l_inv_42*x37*x815 + 16*x37*x661*x669;
        const double x824 = x762 - x763 - x764;
        const double x825 = x575 - x576 + x577 + x578 + x579 + x580 - x581 - x582 - x583 - x584 + x586 + x588 - x589 - x590 - x591 - x592 - x593 - x594 - x595 - x596 - x597 - x598 - x599 - x600 + x601 + x602 - x603 - x604 - x605 - x606 - x607;
        const double x826 = x611 + x774 + x825;
        const double x827 = -x667;
        const double x828 = -x670;
        const double x829 = x660 + x662 + x663 + x666 + x672 + x756 + x827 + x828;
        const double x830 = x674 + x682 + x688 + x689 + x760;
        const double x831 = -x399 - x401 - x403 + x405 - x408 - x410 - x412 - x414 + x418 + x420 + x422 + x424 + x704 + x705 + x706 + x707 + x708 + x709 + x710 + x711 + x712 - x713 - x714 - x715 - x716 - x717 - x718 - x719 - x720 - x721 - x722 + x723 + x724 + x726 - x727 - x728 - x729 - x730 - x731;
        const double x832 = x703 + x831;
        const double x833 = -x734 + x736 + x738;
        const double x834 = x746 + x747 + x748 + x750 - x751 - x752 - x753 - x754;
        const double x835 = x659 + x662 + x663 + x665 + x666 + x757 + x827 + x828;
        const double x836 = x675 + x677 + x679 + x683 + x684 + x686 + x687 + x691 + x759;
        const double x837 = -x273;
        const double x838 = r_12*x89;
        const double x839 = -x838;
        const double x840 = d_3*x273;
        const double x841 = 2*Px;
        const double x842 = r_11*x235 + r_11*x52 + r_12*x296 + r_12*x57 - r_13*x50 + x26*x273 + x273*x54 + x60*x841 + x62*x841 + x64*x841;
        const double x843 = -x840 + x842;
        const double x844 = d_4*x359;
        const double x845 = d_3*x359;
        const double x846 = d_3*x803;
        const double x847 = x840 + x842;
        const double x848 = d_5*r_11;
        const double x849 = r_13*x96;
        const double x850 = r_13*x22;
        const double x851 = r_11*x54;
        const double x852 = r_11*x26;
        const double x853 = x848 + x849 + x850 - x851 - x852;
        const double x854 = x142*x853;
        const double x855 = 2*r_12;
        const double x856 = x140*x855;
        const double x857 = r_12*x127;
        const double x858 = x110*x857;
        const double x859 = -x854 + x856 - x858;
        const double x860 = r_11*x89;
        const double x861 = R_l_inv_53*a_0;
        const double x862 = x269*x861;
        const double x863 = d_5*r_12;
        const double x864 = r_13*x36;
        const double x865 = r_13*x24;
        const double x866 = r_12*x54;
        const double x867 = r_12*x26;
        const double x868 = x863 + x864 + x865 - x866 - x867;
        const double x869 = x149*x868;
        const double x870 = r_11*x127;
        const double x871 = x84*x870;
        const double x872 = -x860 - x862 - x869 - x871;
        const double x873 = r_12*x52;
        const double x874 = r_12*x235;
        const double x875 = r_11*x57;
        const double x876 = r_11*x296;
        const double x877 = d_4*x273;
        const double x878 = x110*x877;
        const double x879 = -x873 - x874 + x875 + x876 + x878;
        const double x880 = r_12*x136;
        const double x881 = x362*x861;
        const double x882 = 4*x110;
        const double x883 = x853*x882;
        const double x884 = d_3*x362;
        const double x885 = x84*x884;
        const double x886 = x110*x845 - x140*x359 - 4*x84*x868;
        const double x887 = x854 - x856 + x858;
        const double x888 = x860 + x862 + x869 + x871;
        const double x889 = 4*x863;
        const double x890 = r_13*x470;
        const double x891 = r_13*x313;
        const double x892 = 4*x866;
        const double x893 = 4*x867;
        const double x894 = x110*x844;
        const double x895 = r_13*x136;
        const double x896 = r_11*x36;
        const double x897 = r_11*x24;
        const double x898 = r_12*x96;
        const double x899 = r_12*x22;
        const double x900 = x896 + x897 - x898 - x899;
        const double x901 = 4*r_13;
        const double x902 = d_3*x901;
        const double x903 = x84*x902 + x861*x901 - x882*x900 + x895;
        const double x904 = 8*x848;
        const double x905 = 8*x851;
        const double x906 = 8*x849;
        const double x907 = 8*x852;
        const double x908 = 8*x850;
        const double x909 = d_4*x803;
        const double x910 = x873 + x874 - x875 - x876 - x878;
        const double x911 = x219*x877;
        const double x912 = Py*x362;
        const double x913 = x912*x94;
        const double x914 = x22*x880;
        const double x915 = r_11*x470;
        const double x916 = d_4*x915;
        const double x917 = x24*x844;
        const double x918 = x911 - x913 - x914 + x916 + x917;
        const double x919 = x306*x853;
        const double x920 = x226*x855;
        const double x921 = d_5*x845;
        const double x922 = r_12*x219;
        const double x923 = x127*x922;
        const double x924 = x160*x849;
        const double x925 = x160*x850;
        const double x926 = Py*x4;
        const double x927 = x359*x926;
        const double x928 = x26*x845;
        const double x929 = -x919 + x920 - x921 - x923 - x924 - x925 + x927 + x928;
        const double x930 = x318*x868;
        const double x931 = x269*x44;
        const double x932 = x269*x45;
        const double x933 = x269*x46;
        const double x934 = r_11*x326;
        const double x935 = std::pow(r_11, 3);
        const double x936 = 2*x59;
        const double x937 = x935*x936;
        const double x938 = x229*x870;
        const double x939 = x269*x68;
        const double x940 = x269*x69;
        const double x941 = x269*x72;
        const double x942 = x269*x73;
        const double x943 = x269*x63;
        const double x944 = x269*x65;
        const double x945 = x269*x67;
        const double x946 = x269*x71;
        const double x947 = x426*x849;
        const double x948 = x426*x850;
        const double x949 = Px*x60;
        const double x950 = 4*x96;
        const double x951 = x949*x950;
        const double x952 = Px*x950;
        const double x953 = x62*x952;
        const double x954 = x64*x952;
        const double x955 = x587*x949;
        const double x956 = Px*x587;
        const double x957 = x62*x956;
        const double x958 = x64*x956;
        const double x959 = x359*x51;
        const double x960 = d_5*x26;
        const double x961 = x359*x960;
        const double x962 = r_21*x66;
        const double x963 = r_12*x3;
        const double x964 = x962*x963;
        const double x965 = r_31*x277;
        const double x966 = x362*x965;
        const double x967 = r_23*x962;
        const double x968 = x901*x967;
        const double x969 = r_31*x442;
        const double x970 = x901*x969;
        const double x971 = x470*x897;
        const double x972 = x26*x54;
        const double x973 = x359*x972;
        const double x974 = x22*x96;
        const double x975 = x359*x974;
        const double x976 = x313*x898;
        const double x977 = x470*x899;
        const double x978 = x434*x849;
        const double x979 = 4*x54;
        const double x980 = x850*x979;
        const double x981 = -x930 - x931 - x932 - x933 - x934 + x937 - x938 - x939 - x940 - x941 - x942 + x943 + x944 + x945 + x946 - x947 - x948 + x951 + x953 + x954 + x955 + x957 + x958 + x959 + x961 + x964 + x966 + x968 + x970 - x971 - x973 + x975 + x976 + x977 + x978 + x980;
        const double x982 = a_2*x814;
        const double x983 = 8*x197;
        const double x984 = x853*(R_l_inv_26*x982 + R_l_inv_36*x815 + R_l_inv_76*x983 + 4*x218);
        const double x985 = x362*x44;
        const double x986 = x362*x45;
        const double x987 = x362*x46;
        const double x988 = x245*x362;
        const double x989 = std::pow(r_12, 3);
        const double x990 = 4*x59;
        const double x991 = x989*x990;
        const double x992 = d_3*x229;
        const double x993 = x362*x992;
        const double x994 = x362*x67;
        const double x995 = x362*x69;
        const double x996 = x362*x71;
        const double x997 = x362*x73;
        const double x998 = x362*x61;
        const double x999 = x362*x65;
        const double x1000 = x362*x68;
        const double x1001 = x362*x72;
        const double x1002 = x462*x864;
        const double x1003 = x462*x865;
        const double x1004 = 8*x36;
        const double x1005 = x1004*x949;
        const double x1006 = Px*x1004;
        const double x1007 = x1006*x62;
        const double x1008 = x1006*x64;
        const double x1009 = x798*x949;
        const double x1010 = Px*x798;
        const double x1011 = x1010*x62;
        const double x1012 = x1010*x64;
        const double x1013 = x51*x803;
        const double x1014 = x803*x960;
        const double x1015 = r_22*x66;
        const double x1016 = 8*x1015;
        const double x1017 = x1016*x804;
        const double x1018 = r_11*r_31*x807;
        const double x1019 = r_13*r_23;
        const double x1020 = x1016*x1019;
        const double x1021 = r_13*r_33;
        const double x1022 = x1021*x807;
        const double x1023 = x803*x974;
        const double x1024 = x803*x972;
        const double x1025 = 8*x96;
        const double x1026 = x1025*x897;
        const double x1027 = x810*x896;
        const double x1028 = x36*x803;
        const double x1029 = x1028*x24;
        const double x1030 = x812*x864;
        const double x1031 = 8*x54;
        const double x1032 = x1031*x865;
        const double x1033 = -d_5*x846 - x185*x864 - x185*x865 + x219*x845 - x226*x359 + x26*x846 + x803*x926 - x868*(R_l_inv_27*x982 + R_l_inv_37*x815 + R_l_inv_77*x983 + 4*x228);
        const double x1034 = x919 - x920 + x921 + x923 + x924 + x925 - x927 - x928;
        const double x1035 = x930 + x931 + x932 + x933 + x934 - x937 + x938 + x939 + x940 + x941 + x942 - x943 - x944 - x945 - x946 + x947 + x948 - x951 - x953 - x954 - x955 - x957 - x958 - x959 - x961 - x964 - x966 - x968 - x970 + x971 + x973 - x975 - x976 - x977 - x978 - x980;
        const double x1036 = d_5*x909;
        const double x1037 = x219*x844;
        const double x1038 = 8*r_13;
        const double x1039 = Py*x6;
        const double x1040 = x1038*x1039;
        const double x1041 = x467*x865;
        const double x1042 = x54*x909;
        const double x1043 = x26*x909;
        const double x1044 = 4*x898;
        const double x1045 = x24*x359;
        const double x1046 = -x1044 + x1045 - 4*x899 + x915;
        const double x1047 = std::pow(r_13, 3);
        const double x1048 = Px*x462;
        const double x1049 = Px*x1031;
        const double x1050 = Px*x812;
        const double x1051 = x66*x804;
        const double x1052 = 8*r_11;
        const double x1053 = r_33*x277;
        const double x1054 = d_5*x803;
        const double x1055 = r_13*x26;
        const double x1056 = x24*x803;
        const double x1057 = -r_23*x1015*x803 - 8*r_23*x1051 - x1028*x26 - x1031*x1055 - x1031*x949 + x1038*x51 - x1046*x219 - x1047*x990 + x1048*x60 + x1048*x62 + x1048*x64 - x1049*x62 - x1049*x64 - x1050*x62 - x1050*x64 - x1052*x969 - x1053*x803 + x1054*x24 + x1054*x36 + x1055*x462 - x1056*x54 + x22*x904 - x22*x905 + x22*x906 + x245*x901 + x44*x901 + x45*x901 - x46*x901 - x61*x901 - x63*x901 + x67*x901 + x68*x901 - x69*x901 + x71*x901 + x72*x901 - x73*x901 + x798*x864 - x812*x949 + x901*x992 + x904*x96 - x907*x96;
        const double x1058 = -x911 + x913 + x914 - x916 - x917;
        const double x1059 = d_3*x895;
        const double x1060 = -d_5*x895;
        const double x1061 = x507*x877;
        const double x1062 = Px*x136;
        const double x1063 = x1062*x60;
        const double x1064 = x1062*x62;
        const double x1065 = x1062*x64;
        const double x1066 = Py*x94;
        const double x1067 = x1066*x359;
        const double x1068 = d_4*r_12;
        const double x1069 = x1068*x470;
        const double x1070 = x54*x895;
        const double x1071 = x22*x844;
        const double x1072 = x1068*x313;
        const double x1073 = x26*x895;
        const double x1074 = -x1059 + x1060 + x1061 + x1063 + x1064 + x1065 + x1067 + x1069 + x1070 + x1071 + x1072 + x1073;
        const double x1075 = x613*x868;
        const double x1076 = r_11*x615;
        const double x1077 = x512*x870;
        const double x1078 = d_3*x889;
        const double x1079 = x4*x912;
        const double x1080 = x160*x867;
        const double x1081 = Py*x34;
        const double x1082 = r_13*x1081;
        const double x1083 = d_3*x891;
        const double x1084 = -x1075 - x1076 - x1077 + x1078 - x1079 - x1080 + x1082 + x1083;
        const double x1085 = x571*x853;
        const double x1086 = x45*x855;
        const double x1087 = x936*x989;
        const double x1088 = x44*x855;
        const double x1089 = x46*x855;
        const double x1090 = x536*x855;
        const double x1091 = x507*x857;
        const double x1092 = x61*x855;
        const double x1093 = x65*x855;
        const double x1094 = x68*x855;
        const double x1095 = x72*x855;
        const double x1096 = x67*x855;
        const double x1097 = x69*x855;
        const double x1098 = x71*x855;
        const double x1099 = x73*x855;
        const double x1100 = x470*x949;
        const double x1101 = Px*x470;
        const double x1102 = x1101*x62;
        const double x1103 = x1101*x64;
        const double x1104 = x313*x949;
        const double x1105 = Px*x313;
        const double x1106 = x1105*x62;
        const double x1107 = x1105*x64;
        const double x1108 = x362*x51;
        const double x1109 = x26*x889;
        const double x1110 = x1051*x3;
        const double x1111 = x359*x965;
        const double x1112 = x1019*x3*x66;
        const double x1113 = x1053*x901;
        const double x1114 = d_5*x890;
        const double x1115 = d_5*x891;
        const double x1116 = x1045*x96;
        const double x1117 = x22*x915;
        const double x1118 = r_12*x24;
        const double x1119 = x1118*x470;
        const double x1120 = x26*x890;
        const double x1121 = x54*x891;
        const double x1122 = x1044*x22;
        const double x1123 = x26*x892;
        const double x1124 = -x1085 - x1086 - x1087 + x1088 + x1089 + x1090 - x1091 - x1092 - x1093 - x1094 - x1095 + x1096 + x1097 + x1098 + x1099 - x1100 - x1102 - x1103 - x1104 - x1106 - x1107 - x1108 - x1109 - x1110 - x1111 - x1112 - x1113 + x1114 + x1115 - x1116 - x1117 - x1119 - x1120 - x1121 + x1122 + x1123;
        const double x1125 = x853*(R_l_inv_26*x815 - R_l_inv_36*x982 + R_l_inv_66*x983 - 4*x506);
        const double x1126 = x362*x503;
        const double x1127 = d_3*x904;
        const double x1128 = d_3*x512;
        const double x1129 = x1128*x362;
        const double x1130 = x185*x849;
        const double x1131 = x185*x850;
        const double x1132 = x1052*x926;
        const double x1133 = x185*x852;
        const double x1134 = x359*x46;
        const double x1135 = x935*x990;
        const double x1136 = x359*x68;
        const double x1137 = x359*x69;
        const double x1138 = x359*x72;
        const double x1139 = x359*x73;
        const double x1140 = x359*x63;
        const double x1141 = x359*x65;
        const double x1142 = x359*x67;
        const double x1143 = x359*x71;
        const double x1144 = x462*x849;
        const double x1145 = x462*x850;
        const double x1146 = x1025*x949;
        const double x1147 = Px*x1025;
        const double x1148 = x1147*x62;
        const double x1149 = x1147*x64;
        const double x1150 = x810*x949;
        const double x1151 = Px*x810;
        const double x1152 = x1151*x62;
        const double x1153 = x1151*x64;
        const double x1154 = x1052*x51;
        const double x1155 = x26*x904;
        const double x1156 = r_22*x803*x962;
        const double x1157 = x803*x965;
        const double x1158 = x1038*x967;
        const double x1159 = x1038*x969;
        const double x1160 = x798*x896;
        const double x1161 = x26*x905;
        const double x1162 = x1052*x974;
        const double x1163 = x1056*x96;
        const double x1164 = x1028*x22;
        const double x1165 = x26*x906;
        const double x1166 = x54*x908;
        const double x1167 = -x1134 + x1135 - x1136 - x1137 - x1138 - x1139 + x1140 + x1141 + x1142 + x1143 - x1144 - x1145 + x1146 + x1148 + x1149 + x1150 + x1152 + x1153 + x1154 + x1155 + x1156 + x1157 + x1158 + x1159 - x1160 - x1161 + x1162 + x1163 + x1164 + x1165 + x1166;
        const double x1168 = x359*x44;
        const double x1169 = x359*x45;
        const double x1170 = -x1168 + x1169;
        const double x1171 = x1167 + x1170 - x359*x536 + x507*x845 - x868*(R_l_inv_27*x815 - R_l_inv_37*x982 + R_l_inv_67*x983 - 4*x511);
        const double x1172 = x1075 + x1076 + x1077 - x1078 + x1079 + x1080 - x1082 - x1083;
        const double x1173 = x1085 + x1086 + x1087 - x1088 - x1089 - x1090 + x1091 + x1092 + x1093 + x1094 + x1095 - x1096 - x1097 - x1098 - x1099 + x1100 + x1102 + x1103 + x1104 + x1106 + x1107 + x1108 + x1109 + x1110 + x1111 + x1112 + x1113 - x1114 - x1115 + x1116 + x1117 + x1119 + x1120 + x1121 - x1122 - x1123;
        const double x1174 = r_11*x467;
        const double x1175 = d_3*x1174;
        const double x1176 = -x1175;
        const double x1177 = x507*x844;
        const double x1178 = -x1046*x507 + x1128*x901 + x185*x896 + x185*x897 - x22*x846 + x503*x901 - x846*x96;
        const double x1179 = x1059 + x1060 - x1061 + x1063 + x1064 + x1065 + x1067 + x1069 + x1070 + x1071 + x1072 + x1073;
        const double x1180 = Px*x160;
        const double x1181 = -x1180*x60;
        const double x1182 = -x1180*x62;
        const double x1183 = -x1180*x64;
        const double x1184 = d_3*x880;
        const double x1185 = d_5*x902;
        const double x1186 = -x845*x96;
        const double x1187 = -r_12*x1081;
        const double x1188 = -x901*x926;
        const double x1189 = -x22*x845;
        const double x1190 = -d_3*r_12*x313;
        const double x1191 = -x26*x902;
        const double x1192 = R_l_inv_44*x362*x661;
        const double x1193 = x741*x853;
        const double x1194 = x725*x884;
        const double x1195 = x1181 + x1182 + x1183 + x1184 + x1185 + x1186 + x1187 + x1188 + x1189 + x1190 + x1191 - x1192 + x1193 + x1194;
        const double x1196 = x136*x863;
        const double x1197 = x136*x866;
        const double x1198 = x136*x867;
        const double x1199 = d_4*x890;
        const double x1200 = d_4*x891;
        const double x1201 = x359*x685;
        const double x1202 = x749*x868;
        const double x1203 = x690*x845;
        const double x1204 = x1196 - x1197 - x1198 + x1199 + x1200 + x1201 + x1202 + x1203;
        const double x1205 = x273*x45;
        const double x1206 = x273*x44;
        const double x1207 = x273*x46;
        const double x1208 = x1047*x936;
        const double x1209 = Px*x426;
        const double x1210 = x1209*x60;
        const double x1211 = x1209*x62;
        const double x1212 = x1209*x64;
        const double x1213 = x273*x67;
        const double x1214 = x273*x68;
        const double x1215 = x273*x71;
        const double x1216 = x273*x72;
        const double x1217 = x273*x61;
        const double x1218 = x273*x63;
        const double x1219 = x273*x69;
        const double x1220 = x273*x73;
        const double x1221 = d_5*x359;
        const double x1222 = x1221*x96;
        const double x1223 = x470*x863;
        const double x1224 = x51*x901;
        const double x1225 = x1221*x22;
        const double x1226 = x313*x863;
        const double x1227 = x901*x960;
        const double x1228 = x949*x979;
        const double x1229 = Px*x979;
        const double x1230 = x1229*x62;
        const double x1231 = x1229*x64;
        const double x1232 = x434*x949;
        const double x1233 = Px*x434;
        const double x1234 = x1233*x62;
        const double x1235 = x1233*x64;
        const double x1236 = r_23*x66;
        const double x1237 = x1236*x360;
        const double x1238 = x359*x969;
        const double x1239 = x1236*x963;
        const double x1240 = x1053*x362;
        const double x1241 = x587*x849;
        const double x1242 = x470*x865;
        const double x1243 = x725*x895;
        const double x1244 = x26*x359*x96;
        const double x1245 = x22*x359*x54;
        const double x1246 = x470*x867;
        const double x1247 = x313*x866;
        const double x1248 = x901*x972;
        const double x1249 = -x1205 + x1206 + x1207 + x1208 - x1210 - x1211 - x1212 - x1213 - x1214 - x1215 - x1216 + x1217 + x1218 + x1219 + x1220 - x1222 - x1223 - x1224 - x1225 - x1226 - x1227 + x1228 + x1230 + x1231 + x1232 + x1234 + x1235 + x1237 + x1238 + x1239 + x1240 - x1241 - x1242 - x1243 + x1244 + x1245 + x1246 + x1247 + x1248;
        const double x1250 = d_4*x904;
        const double x1251 = Py*r_13*x464;
        const double x1252 = x467*x850;
        const double x1253 = x816*x853;
        const double x1254 = x467*x851;
        const double x1255 = x467*x852;
        const double x1256 = x685*x803;
        const double x1257 = x690*x846;
        const double x1258 = R_l_inv_44*r_11*x815 + R_l_inv_47*x815*x868 - r_11*x185*x725 + x1176;
        const double x1259 = -x1196 + x1197 + x1198 - x1199 - x1200 - x1201 - x1202 - x1203;
        const double x1260 = x1181 + x1182 + x1183 - x1184 + x1185 + x1186 + x1187 + x1188 + x1189 + x1190 + x1191 + x1192 - x1193 - x1194;
        const double x1261 = x1174*x725;
        const double x1262 = -R_l_inv_43*r_13*x815 - r_13*x185*x690 + x1039*x1052 - x1066*x803 - x22*x909 + x467*x897 + x816*x900;
        const double x1263 = 16*d_5;
        const double x1264 = 16*x26;
        const double x1265 = 16*x22;
        const double x1266 = 16*x36;
        const double x1267 = Px*x1266;
        const double x1268 = 16*x24;
        const double x1269 = Px*x1268;
        const double x1270 = 16*x1015;
        const double x1271 = x1205 - x1206 - x1207 - x1208 + x1210 + x1211 + x1212 + x1213 + x1214 + x1215 + x1216 - x1217 - x1218 - x1219 - x1220 + x1222 + x1223 + x1224 + x1225 + x1226 + x1227 - x1228 - x1230 - x1231 - x1232 - x1234 - x1235 - x1237 - x1238 - x1239 - x1240 + x1241 + x1242 + x1243 - x1244 - x1245 - x1246 - x1247 - x1248;
        const double x1272 = -x10 - x12 - x14 + x15 - x17 - x19 - x21 - x23 - x25 - x27;
        const double x1273 = x1272 + x30;
        const double x1274 = -x29;
        const double x1275 = x130 + x189;
        const double x1276 = x100 + x102 + x111 + x114 + x94;
        const double x1277 = a_2 + x156 + x39 + x41 + x76 + x77 + x92;
        const double x1278 = x141 + x143 - x144;
        const double x1279 = x155 + x43 + x86 + x88;
        const double x1280 = x103 + x112 + x115 + x95 + x99;
        const double x1281 = -x163 - x165 - x167 - x168;
        const double x1282 = x117 + x118 + x120 + x122 + x124 + x190;
        const double x1283 = x307 + x308 + x309 - x310 + x312 + x314 - x315 - x316;
        const double x1284 = -x398 + x400 + x402 - x404 - x406 - x407 + x409 + x411 + x413 + x415 - x417 - x419 - x421 - x423 - x425 - x427 - x428 - x429 + x431 + x432 + x433 + x435 + x436 + x437 + x439 + x441 + x443 + x445 - x446 - x447 - x448 - x449 - x450 - x451 + x452 + x453 + x454 + x456 + x457 - x458 - x459;
        const double x1285 = x525 - x526 - x527 - x528 - x529 - x530 - x531 - x532 - x533 - x534;
        const double x1286 = x1285 + x628;
        const double x1287 = x572 + x573 - x574 + x775 + x825;
        const double x1288 = x1285 + x564;
        const double x1289 = -x638 - x639 - x640 + x641 + x642 - x643 - x644;
        const double x1290 = x693 + x694 + x695 - x696 + x697 + x698 + x699 + x700 + x701 + x702;
        const double x1291 = x1290 + x831;
        const double x1292 = x634 - x740 - x742 + x744;
        const double x1293 = x766 + x767 + x768 + x769 - x770 - x771 + x772;
        const double x1294 = x1290 + x732;
        
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
        A(1, 2) = x31 + x5;
        A(1, 3) = x33;
        A(1, 4) = -x34;
        A(1, 5) = x32;
        A(1, 6) = x28 + x35;
        A(1, 7) = x29;
        A(1, 8) = x31 + x4;
        A(2, 0) = x116 + x131 + x93;
        A(2, 1) = x139 + x145 + x153;
        A(2, 2) = x131 + x154 + x157;
        A(2, 3) = x162 + x169 + x176;
        A(2, 4) = x177 - x178 - x179 + x180 + x181 + x183 + x186;
        A(2, 5) = x169 + x187 + x188;
        A(2, 6) = x154 + x191 + x93;
        A(2, 7) = x145 + x192 + x193;
        A(2, 8) = x116 + x157 + x191;
        A(3, 0) = x217 + x240 + x293;
        A(3, 1) = x301 + x317 + x377;
        A(3, 2) = x379 + x381 + x382;
        A(3, 3) = x388 + x395 + x460;
        A(3, 4) = x461 + x463 - x465 - x466 + x468 + x469 + x472;
        A(3, 5) = x460 + x473 + x474;
        A(3, 6) = x239 + x478 + x480;
        A(3, 7) = x317 + x481 + x482;
        A(3, 8) = x380 + x483 + x484;
        A(4, 0) = x502 + x524 + x565;
        A(4, 1) = x570 + x612 + x623;
        A(4, 2) = x625 + x627 + x629;
        A(4, 3) = x633 + x637 + x645;
        A(4, 4) = -x646 + x647 + x648;
        A(4, 5) = x645 + x649 + x650;
        A(4, 6) = x565 + x652 + x654;
        A(4, 7) = x612 + x655 + x656;
        A(4, 8) = x629 + x657 + x658;
        A(5, 0) = x673 + x692 + x733;
        A(5, 1) = x739 + x745 + x755;
        A(5, 2) = x733 + x758 + x761;
        A(5, 3) = x765 + x773 + x776;
        A(5, 4) = -x777 - x778 + x779 + x780 - x781 - x782 - x783 - x784 + x785 + x786 + x787 + x788 - x790 - x791 + x793 + x794 + x795 + x797 + x799 + x800 + x801 + x802 + x805 + x806 + x808 + x809 - x811 - x813 - x817 + x818 + x819 + x820 + x821 + x822 + x823;
        A(5, 5) = x773 + x824 + x826;
        A(5, 6) = x829 + x830 + x832;
        A(5, 7) = x745 + x833 + x834;
        A(5, 8) = x832 + x835 + x836;
        
        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x837;
        B(0, 2) = x837;
        B(0, 3) = -x359;
        B(0, 4) = -x803;
        B(0, 5) = x359;
        B(0, 6) = x273;
        B(0, 8) = x273;
        B(1, 0) = x839 + x843;
        B(1, 1) = x844;
        B(1, 2) = x838 + x843;
        B(1, 3) = -x845;
        B(1, 4) = -x846;
        B(1, 5) = x845;
        B(1, 6) = x839 + x847;
        B(1, 7) = x844;
        B(1, 8) = x838 + x847;
        B(2, 0) = x859 + x872 + x879;
        B(2, 1) = -x880 - x881 + x883 - x885 + x886;
        B(2, 2) = x879 + x887 + x888;
        B(2, 3) = -x889 - x890 - x891 + x892 + x893 + x894 + x903;
        B(2, 4) = x110*x909 + x904 - x905 + x906 - x907 + x908;
        B(2, 5) = x889 + x890 + x891 - x892 - x893 - x894 + x903;
        B(2, 6) = x859 + x888 + x910;
        B(2, 7) = x880 + x881 - x883 + x885 + x886;
        B(2, 8) = x872 + x887 + x910;
        B(3, 0) = x918 + x929 + x981;
        B(3, 1) = x1000 + x1001 - x1002 - x1003 + x1005 + x1007 + x1008 + x1009 + x1011 + x1012 + x1013 + x1014 + x1017 + x1018 + x1020 + x1022 - x1023 - x1024 + x1026 + x1027 + x1029 + x1030 + x1032 + x1033 + x984 - x985 - x986 - x987 - x988 + x991 - x993 - x994 - x995 - x996 - x997 + x998 + x999;
        B(3, 2) = x1034 + x1035 + x918;
        B(3, 3) = -x1036 + x1037 - x1040 - x1041 + x1042 + x1043 + x1057;
        B(3, 4) = x467*(-r_11*x383 + r_11*x50 - r_11*x58 + r_13*x52 + x22*x273 + x922);
        B(3, 5) = x1036 - x1037 + x1040 + x1041 - x1042 - x1043 + x1057;
        B(3, 6) = x1035 + x1058 + x929;
        B(3, 7) = -x1000 - x1001 + x1002 + x1003 - x1005 - x1007 - x1008 - x1009 - x1011 - x1012 - x1013 - x1014 - x1017 - x1018 - x1020 - x1022 + x1023 + x1024 - x1026 - x1027 - x1029 - x1030 - x1032 + x1033 - x984 + x985 + x986 + x987 + x988 - x991 + x993 + x994 + x995 + x996 + x997 - x998 - x999;
        B(3, 8) = x1034 + x1058 + x981;
        B(4, 0) = x1074 + x1084 + x1124;
        B(4, 1) = x1125 - x1126 - x1127 - x1129 - x1130 - x1131 + x1132 + x1133 + x1171;
        B(4, 2) = x1074 + x1172 + x1173;
        B(4, 3) = x1176 + x1177 + x1178;
        B(4, 4) = x909*(-x127 + x507);
        B(4, 5) = x1175 - x1177 + x1178;
        B(4, 6) = x1124 + x1172 + x1179;
        B(4, 7) = -x1125 + x1126 + x1127 + x1129 + x1130 + x1131 - x1132 - x1133 + x1171;
        B(4, 8) = x1084 + x1173 + x1179;
        B(5, 0) = x1195 + x1204 + x1249;
        B(5, 1) = -x1250 - x1251 - x1252 - x1253 + x1254 + x1255 + x1256 + x1257 + x1258;
        B(5, 2) = x1249 + x1259 + x1260;
        B(5, 3) = x1167 + x1168 - x1169 - x1261 + x1262;
        B(5, 4) = 16*r_11*x965 + 16*r_12*x51 + x1019*x1270 + 16*x1021*x277 - 16*x1068*x725 + x1118*x1266 - x1263*x864 - x1263*x865 + x1264*x863 + x1264*x864 - x1264*x866 + x1265*x896 - x1265*x898 + x1266*x949 + x1267*x62 + x1267*x64 + x1268*x949 + x1269*x62 + x1269*x64 + x1270*x804 + x44*x803 - x45*x803 - x46*x803 + 16*x54*x865 + 8*x59*x989 + x61*x803 + x65*x803 - x67*x803 + x68*x803 - x69*x803 - x71*x803 + x72*x803 - x73*x803 + 16*x897*x96;
        B(5, 5) = x1134 - x1135 + x1136 + x1137 + x1138 + x1139 - x1140 - x1141 - x1142 - x1143 + x1144 + x1145 - x1146 - x1148 - x1149 - x1150 - x1152 - x1153 - x1154 - x1155 - x1156 - x1157 - x1158 - x1159 + x1160 + x1161 - x1162 - x1163 - x1164 - x1165 - x1166 + x1170 + x1261 + x1262;
        B(5, 6) = x1195 + x1259 + x1271;
        B(5, 7) = x1250 + x1251 + x1252 + x1253 - x1254 - x1255 - x1256 - x1257 + x1258;
        B(5, 8) = x1204 + x1260 + x1271;
        
        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = r_23;
        C(0, 2) = r_23;
        C(0, 3) = x1;
        C(0, 4) = x3;
        C(0, 5) = x2;
        C(0, 6) = x0;
        C(0, 8) = x0;
        C(1, 0) = x1273 + x4;
        C(1, 1) = x1274;
        C(1, 2) = x1272 + x35;
        C(1, 3) = x32;
        C(1, 4) = x34;
        C(1, 5) = x33;
        C(1, 6) = x1273 + x5;
        C(1, 7) = x1274;
        C(1, 8) = x1272 + x8;
        C(2, 0) = x1275 + x1276 + x1277;
        C(2, 1) = x1278 + x139 + x193;
        C(2, 2) = x1275 + x1279 + x1280;
        C(2, 3) = x1281 + x162 + x188;
        C(2, 4) = -x177 + x178 + x179 - x180 - x181 - x183 + x186;
        C(2, 5) = x1281 + x176 + x187;
        C(2, 6) = x1277 + x1280 + x1282;
        C(2, 7) = x1278 + x153 + x192;
        C(2, 8) = x1276 + x1279 + x1282;
        C(3, 0) = x217 + x380 + x478;
        C(3, 1) = x1283 + x301 + x482;
        C(3, 2) = x239 + x379 + x484;
        C(3, 3) = x1284 + x388 + x474;
        C(3, 4) = -x461 - x463 + x465 + x466 - x468 - x469 + x472;
        C(3, 5) = x1284 + x395 + x473;
        C(3, 6) = x293 + x381 + x480;
        C(3, 7) = x1283 + x377 + x481;
        C(3, 8) = x240 + x382 + x483;
        C(4, 0) = x1286 + x502 + x652;
        C(4, 1) = x1287 + x570 + x656;
        C(4, 2) = x1288 + x625 + x658;
        C(4, 3) = x1289 + x633 + x649;
        C(4, 4) = x646 - x647 + x648;
        C(4, 5) = x1289 + x637 + x650;
        C(4, 6) = x1286 + x524 + x654;
        C(4, 7) = x1287 + x623 + x655;
        C(4, 8) = x1288 + x627 + x657;
        C(5, 0) = x1291 + x673 + x761;
        C(5, 1) = x1292 + x739 + x834;
        C(5, 2) = x1291 + x692 + x758;
        C(5, 3) = x1293 + x765 + x826;
        C(5, 4) = x777 + x778 - x779 - x780 + x781 + x782 + x783 + x784 - x785 - x786 - x787 - x788 + x790 + x791 - x793 - x794 - x795 - x797 - x799 - x800 - x801 - x802 - x805 - x806 - x808 - x809 + x811 + x813 + x817 - x818 - x819 - x820 - x821 - x822 + x823;
        C(5, 5) = x1293 + x776 + x824;
        C(5, 6) = x1294 + x829 + x836;
        C(5, 7) = x1292 + x755 + x833;
        C(5, 8) = x1294 + x830 + x835;
        
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
            
            const bool condition_0 = std::fabs((Px*std::sin(th_0) - Py*std::cos(th_0) + d_1 - d_5*(r_13*std::sin(th_0) - r_23*std::cos(th_0)))/d_4) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double x2 = std::acos((Px*x0 - Py*x1 + d_1 + d_5*(-r_13*x0 + r_23*x1))/d_4);
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[5] = x2;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = std::fabs((Px*std::sin(th_0) - Py*std::cos(th_0) + d_1 - d_5*(r_13*std::sin(th_0) - r_23*std::cos(th_0)))/d_4) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double x2 = std::acos((Px*x0 - Py*x1 + d_1 + d_5*(-r_13*x0 + r_23*x1))/d_4);
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
            
            const bool condition_0 = 2*std::fabs(a_0*d_3) >= zero_tolerance || std::fabs(2*a_0*a_2 - 2*a_0*d_4*std::sin(th_3)) >= zero_tolerance || std::fabs(std::pow(Px, 2) - 2*Px*d_5*r_13 + std::pow(Py, 2) - 2*Py*d_5*r_23 + std::pow(Pz, 2) - 2*Pz*d_5*r_33 - std::pow(a_0, 2) - std::pow(a_2, 2) + 2*a_2*d_4*std::sin(th_3) - std::pow(d_1, 2) + 2*d_1*d_4*std::cos(th_3) - std::pow(d_3, 2) - std::pow(d_4, 2) + std::pow(d_5, 2)*std::pow(r_13, 2) + std::pow(d_5, 2)*std::pow(r_23, 2) + std::pow(d_5, 2)*std::pow(r_33, 2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_0;
                const double x1 = d_4*std::sin(th_3);
                const double x2 = a_2*x0 - x0*x1;
                const double x3 = std::atan2(-d_3*x0, x2);
                const double x4 = std::pow(a_0, 2);
                const double x5 = std::pow(d_3, 2);
                const double x6 = 2*d_5;
                const double x7 = std::pow(d_5, 2);
                const double x8 = std::pow(Px, 2) - Px*r_13*x6 + std::pow(Py, 2) - Py*r_23*x6 + std::pow(Pz, 2) - Pz*r_33*x6 - std::pow(a_2, 2) + 2*a_2*x1 - std::pow(d_1, 2) + 2*d_1*d_4*std::cos(th_3) - std::pow(d_4, 2) + std::pow(r_13, 2)*x7 + std::pow(r_23, 2)*x7 + std::pow(r_33, 2)*x7 - x4 - x5;
                const double x9 = std::sqrt(std::pow(x2, 2) + 4*x4*x5 - std::pow(x8, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x3 + std::atan2(x9, x8);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx);
            }
            
            const bool condition_1 = 2*std::fabs(a_0*d_3) >= zero_tolerance || std::fabs(2*a_0*a_2 - 2*a_0*d_4*std::sin(th_3)) >= zero_tolerance || std::fabs(std::pow(Px, 2) - 2*Px*d_5*r_13 + std::pow(Py, 2) - 2*Py*d_5*r_23 + std::pow(Pz, 2) - 2*Pz*d_5*r_33 - std::pow(a_0, 2) - std::pow(a_2, 2) + 2*a_2*d_4*std::sin(th_3) - std::pow(d_1, 2) + 2*d_1*d_4*std::cos(th_3) - std::pow(d_3, 2) - std::pow(d_4, 2) + std::pow(d_5, 2)*std::pow(r_13, 2) + std::pow(d_5, 2)*std::pow(r_23, 2) + std::pow(d_5, 2)*std::pow(r_33, 2)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_0;
                const double x1 = d_4*std::sin(th_3);
                const double x2 = a_2*x0 - x0*x1;
                const double x3 = std::atan2(-d_3*x0, x2);
                const double x4 = std::pow(a_0, 2);
                const double x5 = std::pow(d_3, 2);
                const double x6 = 2*d_5;
                const double x7 = std::pow(d_5, 2);
                const double x8 = std::pow(Px, 2) - Px*r_13*x6 + std::pow(Py, 2) - Py*r_23*x6 + std::pow(Pz, 2) - Pz*r_33*x6 - std::pow(a_2, 2) + 2*a_2*x1 - std::pow(d_1, 2) + 2*d_1*d_4*std::cos(th_3) - std::pow(d_4, 2) + std::pow(r_13, 2)*x7 + std::pow(r_23, 2)*x7 + std::pow(r_33, 2)*x7 - x4 - x5;
                const double x9 = std::sqrt(std::pow(x2, 2) + 4*x4*x5 - std::pow(x8, 2));
                // End of temp variables
                const double tmp_sol_value = x3 + std::atan2(-x9, x8);
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
            
            const bool degenerate_valid_0 = std::fabs(th_2 - M_PI + 1.5032934091316701) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(32, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_2 - 2*M_PI + 1.5032934091316701) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(23, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_22_processor();
    // Finish code for solved_variable dispatcher node 22
    
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
            const double th_0 = this_solution[0];
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(0.99772254304723296*a_2 - 0.067451664882066498*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) + d_4*r_11*std::sin(th_5)*std::cos(th_0) + d_4*r_12*std::cos(th_0)*std::cos(th_5) + d_4*r_21*std::sin(th_0)*std::sin(th_5) + d_4*r_22*std::sin(th_0)*std::cos(th_5) - d_5*r_13*std::cos(th_0) - d_5*r_23*std::sin(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = d_4*std::sin(th_5);
                const double x3 = d_4*std::cos(th_5);
                const double x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3;
                const double x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3;
                const double x6 = std::atan2(x4, x5);
                const double x7 = std::sqrt(std::pow(x4, 2) + std::pow(x5, 2) - 0.99545027290463695*std::pow(-a_2 + 0.067605633802817006*d_3, 2));
                const double x8 = -0.99772254304723296*a_2 + 0.067451664882066498*d_3;
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x6 + std::atan2(x7, x8);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(33, appended_idx);
            }
            
            const bool condition_1 = std::fabs(0.99772254304723296*a_2 - 0.067451664882066498*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) + d_4*r_11*std::sin(th_5)*std::cos(th_0) + d_4*r_12*std::cos(th_0)*std::cos(th_5) + d_4*r_21*std::sin(th_0)*std::sin(th_5) + d_4*r_22*std::sin(th_0)*std::cos(th_5) - d_5*r_13*std::cos(th_0) - d_5*r_23*std::sin(th_0)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = d_4*std::sin(th_5);
                const double x3 = d_4*std::cos(th_5);
                const double x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3;
                const double x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3;
                const double x6 = std::atan2(x4, x5);
                const double x7 = std::sqrt(std::pow(x4, 2) + std::pow(x5, 2) - 0.99545027290463695*std::pow(-a_2 + 0.067605633802817006*d_3, 2));
                const double x8 = -0.99772254304723296*a_2 + 0.067451664882066498*d_3;
                // End of temp variables
                const double tmp_sol_value = x6 + std::atan2(-x7, x8);
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
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(-r_13*((-0.99772254304723296*std::sin(th_1) - 0.067451664882066498*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) - r_23*((-0.99772254304723296*std::sin(th_1) - 0.067451664882066498*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(0.067451664882066498*std::sin(th_1) - 0.99772254304723296*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance || std::fabs(-r_13*(-0.067451664882066498*std::sin(th_1) + 0.99772254304723296*std::cos(th_1))*std::cos(th_0) - r_23*(-0.067451664882066498*std::sin(th_1) + 0.99772254304723296*std::cos(th_1))*std::sin(th_0) + r_33*(0.99772254304723296*std::sin(th_1) + 0.067451664882066498*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = 0.067451664882066498*x1;
                const double x3 = std::cos(th_1);
                const double x4 = 0.99772254304723296*x3;
                const double x5 = std::sin(th_0);
                const double x6 = std::sin(th_3);
                const double x7 = std::cos(th_0);
                const double x8 = 0.99772254304723296*x1;
                const double x9 = 0.067451664882066498*x3;
                const double x10 = x0*(-x8 - x9);
                const double x11 = -x2 + x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_13*(x10*x7 + x5*x6) - r_23*(x10*x5 - x6*x7) - r_33*x0*(x2 - x4), r_13*x11*x7 + r_23*x11*x5 - r_33*(x8 + x9));
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
            
            const bool condition_0 = std::fabs(a_2*std::sin(th_2) + d_3*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_0 + a_2*std::cos(th_2) - d_3*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = d_5*r_33;
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = a_0 + a_2*x1 - d_3*x2;
                const double x4 = -a_2*x2 - d_3*x1;
                const double x5 = Px*std::cos(th_0) + Py*std::sin(th_0) - d_5*std::sin(th_1th_2th_4_soa);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x3*(-Pz + x0) + x4*x5, x3*x5 + x4*(Pz - x0));
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
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x1 + r_21*x0, -r_12*x1 + r_22*x0);
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
            
            const bool degenerate_valid_0 = std::fabs(th_2 - M_PI + 1.5032934091316701) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(26, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_2 - 2*M_PI + 1.5032934091316701) <= 9.9999999999999995e-7;
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
    
    // Code for explicit solution node 29, solved variable is th_1
    auto ExplicitSolutionNode_node_29_solve_th_1_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(0.99772254304723296*a_2 - 0.067451664882066498*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) + d_4*r_11*std::sin(th_5)*std::cos(th_0) + d_4*r_12*std::cos(th_0)*std::cos(th_5) + d_4*r_21*std::sin(th_0)*std::sin(th_5) + d_4*r_22*std::sin(th_0)*std::cos(th_5) - d_5*r_13*std::cos(th_0) - d_5*r_23*std::sin(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = d_4*std::sin(th_5);
                const double x3 = d_4*std::cos(th_5);
                const double x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3;
                const double x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3;
                const double x6 = std::atan2(x4, x5);
                const double x7 = std::sqrt(std::pow(x4, 2) + std::pow(x5, 2) - 0.99545027290463695*std::pow(a_2 - 0.067605633802817006*d_3, 2));
                const double x8 = 0.99772254304723296*a_2 - 0.067451664882066498*d_3;
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x6 + std::atan2(x7, x8);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(30, appended_idx);
            }
            
            const bool condition_1 = std::fabs(0.99772254304723296*a_2 - 0.067451664882066498*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) + d_4*r_11*std::sin(th_5)*std::cos(th_0) + d_4*r_12*std::cos(th_0)*std::cos(th_5) + d_4*r_21*std::sin(th_0)*std::sin(th_5) + d_4*r_22*std::sin(th_0)*std::cos(th_5) - d_5*r_13*std::cos(th_0) - d_5*r_23*std::sin(th_0)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = d_4*std::sin(th_5);
                const double x3 = d_4*std::cos(th_5);
                const double x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3;
                const double x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3;
                const double x6 = std::atan2(x4, x5);
                const double x7 = std::sqrt(std::pow(x4, 2) + std::pow(x5, 2) - 0.99545027290463695*std::pow(a_2 - 0.067605633802817006*d_3, 2));
                const double x8 = 0.99772254304723296*a_2 - 0.067451664882066498*d_3;
                // End of temp variables
                const double tmp_sol_value = x6 + std::atan2(-x7, x8);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(30, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_1_processor();
    // Finish code for explicit solution node 29
    
    // Code for non-branch dispatcher node 30
    // Actually, there is no code
    
    // Code for explicit solution node 31, solved variable is th_4
    auto ExplicitSolutionNode_node_31_solve_th_4_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[5];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(-r_13*((0.99772254304723296*std::sin(th_1) + 0.067451664882066498*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) - r_23*((0.99772254304723296*std::sin(th_1) + 0.067451664882066498*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(-0.067451664882066498*std::sin(th_1) + 0.99772254304723296*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance || std::fabs(-r_13*(-0.067451664882066498*std::sin(th_1) + 0.99772254304723296*std::cos(th_1))*std::cos(th_0) - r_23*(-0.067451664882066498*std::sin(th_1) + 0.99772254304723296*std::cos(th_1))*std::sin(th_0) + r_33*(0.99772254304723296*std::sin(th_1) + 0.067451664882066498*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_1);
                const double x3 = -0.067451664882066498*x1 + 0.99772254304723296*x2;
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_3);
                const double x6 = std::cos(th_0);
                const double x7 = 0.99772254304723296*x1 + 0.067451664882066498*x2;
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
    ExplicitSolutionNode_node_31_solve_th_4_processor();
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
            const double th_0 = this_solution[0];
            const double th_5 = this_solution[7];
            
            const bool condition_0 = std::fabs(0.99772254304723296*a_2 - 0.067451664882066498*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) + d_4*r_11*std::sin(th_5)*std::cos(th_0) + d_4*r_12*std::cos(th_0)*std::cos(th_5) + d_4*r_21*std::sin(th_0)*std::sin(th_5) + d_4*r_22*std::sin(th_0)*std::cos(th_5) - d_5*r_13*std::cos(th_0) - d_5*r_23*std::sin(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = d_4*std::sin(th_5);
                const double x3 = d_4*std::cos(th_5);
                const double x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3;
                const double x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3;
                const double x6 = std::atan2(x4, x5);
                const double x7 = std::sqrt(std::pow(x4, 2) + std::pow(x5, 2) - 0.99545027290463695*std::pow(-a_2 + 0.067605633802817006*d_3, 2));
                const double x8 = -0.99772254304723296*a_2 + 0.067451664882066498*d_3;
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x6 + std::atan2(x7, x8);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(27, appended_idx);
            }
            
            const bool condition_1 = std::fabs(0.99772254304723296*a_2 - 0.067451664882066498*d_3) >= zero_tolerance || std::fabs(Pz + d_4*r_31*std::sin(th_5) + d_4*r_32*std::cos(th_5) - d_5*r_33) >= zero_tolerance || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) + d_4*r_11*std::sin(th_5)*std::cos(th_0) + d_4*r_12*std::cos(th_0)*std::cos(th_5) + d_4*r_21*std::sin(th_0)*std::sin(th_5) + d_4*r_22*std::sin(th_0)*std::cos(th_5) - d_5*r_13*std::cos(th_0) - d_5*r_23*std::sin(th_0)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = d_4*std::sin(th_5);
                const double x3 = d_4*std::cos(th_5);
                const double x4 = Px*x0 + Py*x1 - d_5*r_13*x0 - d_5*r_23*x1 + r_11*x0*x2 + r_12*x0*x3 + r_21*x1*x2 + r_22*x1*x3;
                const double x5 = Pz - d_5*r_33 + r_31*x2 + r_32*x3;
                const double x6 = std::atan2(x4, x5);
                const double x7 = std::sqrt(std::pow(x4, 2) + std::pow(x5, 2) - 0.99545027290463695*std::pow(-a_2 + 0.067605633802817006*d_3, 2));
                const double x8 = -0.99772254304723296*a_2 + 0.067451664882066498*d_3;
                // End of temp variables
                const double tmp_sol_value = x6 + std::atan2(-x7, x8);
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
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(-r_13*((-0.99772254304723296*std::sin(th_1) - 0.067451664882066498*std::cos(th_1))*std::cos(th_0)*std::cos(th_3) + std::sin(th_0)*std::sin(th_3)) - r_23*((-0.99772254304723296*std::sin(th_1) - 0.067451664882066498*std::cos(th_1))*std::sin(th_0)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) - r_33*(0.067451664882066498*std::sin(th_1) - 0.99772254304723296*std::cos(th_1))*std::cos(th_3)) >= zero_tolerance || std::fabs(-r_13*(-0.067451664882066498*std::sin(th_1) + 0.99772254304723296*std::cos(th_1))*std::cos(th_0) - r_23*(-0.067451664882066498*std::sin(th_1) + 0.99772254304723296*std::cos(th_1))*std::sin(th_0) + r_33*(0.99772254304723296*std::sin(th_1) + 0.067451664882066498*std::cos(th_1))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = 0.067451664882066498*x1;
                const double x3 = std::cos(th_1);
                const double x4 = 0.99772254304723296*x3;
                const double x5 = std::sin(th_0);
                const double x6 = std::sin(th_3);
                const double x7 = std::cos(th_0);
                const double x8 = 0.99772254304723296*x1;
                const double x9 = 0.067451664882066498*x3;
                const double x10 = x0*(-x8 - x9);
                const double x11 = -x2 + x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_13*(x10*x7 + x5*x6) - r_23*(x10*x5 - x6*x7) - r_33*x0*(x2 - x4), r_13*x11*x7 + r_23*x11*x5 - r_33*(x8 + x9));
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
            
            const bool condition_0 = std::fabs(a_2*std::sin(th_2) + d_3*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_0 + a_2*std::cos(th_2) - d_3*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = d_5*r_33;
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_2);
                const double x3 = a_0 + a_2*x1 - d_3*x2;
                const double x4 = -a_2*x2 - d_3*x1;
                const double x5 = std::cos(th_0);
                const double x6 = std::sin(th_0);
                const double x7 = Px*x5 + Py*x6 - d_5*r_13*x5 - d_5*r_23*x6;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x3*(-Pz + x0) + x4*x7, x3*x7 + x4*(Pz - x0));
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
                const double x0 = std::asin((-r_13*std::sin(th_0) + r_23*std::cos(th_0))/std::sin(th_3));
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
                const double x0 = std::asin((-r_13*std::sin(th_0) + r_23*std::cos(th_0))/std::sin(th_3));
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
                const double x0 = -a_0*std::cos(th_2) - a_2 + d_4*std::sin(th_3) + d_5*std::sin(th_4)*std::cos(th_3);
                const double x1 = -Px*std::cos(th_0) - Py*std::sin(th_0);
                const double x2 = a_0*std::sin(th_2) - d_3 + d_5*std::cos(th_4);
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
                const double x2 = std::cos(th_0);
                const double x3 = std::cos(th_3);
                const double x4 = std::sin(th_0);
                const double x5 = x0*std::cos(th_1th_2_soa);
                const double x6 = x2*x3 + x4*x5;
                const double x7 = -x2*x5 + x3*x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x7 + r_21*x6 - r_31*x1, -r_12*x7 + r_22*x6 - r_32*x1);
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

}; // struct dense_cvr_038_ik

// Code below for debug
void test_ik_solve_dense_cvr_038()
{
    std::array<double, dense_cvr_038_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = dense_cvr_038_ik::computeFK(theta);
    auto ik_output = dense_cvr_038_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = dense_cvr_038_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_dense_cvr_038();
}
