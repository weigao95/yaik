#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct yaskawa_HC10_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 128;
static constexpr int n_tree_nodes = 28;
static constexpr int intermediate_solution_size = 6;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_0 = 0.7;
static constexpr double d_1 = -0.5;
static constexpr double d_2 = -0.162;
static constexpr double d_3 = -0.13;
static constexpr double pre_transform_special_symbol_23 = 0.275;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -1.5707963267948966;
static constexpr double th_2_offset_original2raw = 0.0;
static constexpr double th_3_offset_original2raw = 3.141592653589793;
static constexpr double th_4_offset_original2raw = 3.141592653589793;
static constexpr double th_5_offset_original2raw = 3.141592653589793;

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
    const double x12 = -x1*x2 - x11*x3;
    const double x13 = std::cos(th_5);
    const double x14 = std::sin(th_4);
    const double x15 = x5*x9;
    const double x16 = x6*x8;
    const double x17 = x15*x4 - x16*x4;
    const double x18 = std::cos(th_4);
    const double x19 = -x1*x3 + x11*x2;
    const double x20 = -x14*x17 + x18*x19;
    const double x21 = -x14*x19 - x17*x18;
    const double x22 = a_0*x8;
    const double x23 = x1*x10 + x1*x7;
    const double x24 = x2*x4 - x23*x3;
    const double x25 = x1*x15 - x1*x16;
    const double x26 = x2*x23 + x3*x4;
    const double x27 = -x14*x25 + x18*x26;
    const double x28 = -x14*x26 - x18*x25;
    const double x29 = -x15 + x16;
    const double x30 = x29*x3;
    const double x31 = x10 + x7;
    const double x32 = x2*x29;
    const double x33 = -x14*x31 + x18*x32;
    const double x34 = -x14*x32 - x18*x31;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x12 + x13*x20;
    ee_pose_raw(0, 1) = -x0*x20 - x12*x13;
    ee_pose_raw(0, 2) = x21;
    ee_pose_raw(0, 3) = d_1*x17 + d_2*x12 + d_3*x21 + x22*x4;
    ee_pose_raw(1, 0) = -x0*x24 + x13*x27;
    ee_pose_raw(1, 1) = -x0*x27 - x13*x24;
    ee_pose_raw(1, 2) = x28;
    ee_pose_raw(1, 3) = d_1*x25 + d_2*x24 + d_3*x28 + x1*x22;
    ee_pose_raw(2, 0) = x0*x30 + x13*x33;
    ee_pose_raw(2, 1) = -x0*x33 + x13*x30;
    ee_pose_raw(2, 2) = x34;
    ee_pose_raw(2, 3) = -a_0*x5 + d_1*x31 - d_2*x30 + d_3*x34;
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
    const double x2 = std::sin(th_1);
    const double x3 = std::cos(th_2);
    const double x4 = std::cos(th_0);
    const double x5 = 1.0*x4;
    const double x6 = x3*x5;
    const double x7 = std::cos(th_1);
    const double x8 = std::sin(th_2);
    const double x9 = x5*x8;
    const double x10 = x2*x6 - x7*x9;
    const double x11 = std::cos(th_3);
    const double x12 = std::sin(th_3);
    const double x13 = x2*x9 + x6*x7;
    const double x14 = -x1*x11 - x12*x13;
    const double x15 = std::cos(th_4);
    const double x16 = std::sin(th_4);
    const double x17 = -x10*x15 - x16*(-x1*x12 + x11*x13);
    const double x18 = x1*x3;
    const double x19 = x1*x8;
    const double x20 = x18*x2 - x19*x7;
    const double x21 = x18*x7 + x19*x2;
    const double x22 = x11*x5 - x12*x21;
    const double x23 = -x15*x20 - x16*(x11*x21 + x12*x5);
    const double x24 = 1.0*x2;
    const double x25 = 1.0*x7;
    const double x26 = x24*x8 + x25*x3;
    const double x27 = -x24*x3 + x25*x8;
    const double x28 = x12*x27;
    const double x29 = -x11*x16*x27 - x15*x26;
    const double x30 = -a_0*x24 + pre_transform_special_symbol_23;
    const double x31 = a_0*x7;
    const double x32 = d_1*x20 + x1*x31;
    const double x33 = d_1*x26 + x30;
    const double x34 = -d_2*x28 + x33;
    const double x35 = d_2*x22 + x32;
    const double x36 = d_3*x29 + x34;
    const double x37 = d_3*x23 + x35;
    const double x38 = d_1*x10 + x31*x5;
    const double x39 = d_2*x14 + x38;
    const double x40 = d_3*x17 + x39;
    const double x41 = a_0*x25;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = -x1;
    jacobian(0, 2) = x1;
    jacobian(0, 3) = x10;
    jacobian(0, 4) = x14;
    jacobian(0, 5) = x17;
    jacobian(1, 1) = x5;
    jacobian(1, 2) = -x5;
    jacobian(1, 3) = x20;
    jacobian(1, 4) = x22;
    jacobian(1, 5) = x23;
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x26;
    jacobian(2, 4) = -x28;
    jacobian(2, 5) = x29;
    jacobian(3, 1) = -pre_transform_special_symbol_23*x5;
    jacobian(3, 2) = x30*x5;
    jacobian(3, 3) = -x20*x33 + x26*x32;
    jacobian(3, 4) = -x22*x34 - x28*x35;
    jacobian(3, 5) = -x23*x36 + x29*x37;
    jacobian(4, 1) = -pre_transform_special_symbol_23*x1;
    jacobian(4, 2) = x1*x30;
    jacobian(4, 3) = x10*x33 - x26*x38;
    jacobian(4, 4) = x14*x34 + x28*x39;
    jacobian(4, 5) = x17*x36 - x29*x40;
    jacobian(5, 2) = -std::pow(x0, 2)*x41 - std::pow(x4, 2)*x41;
    jacobian(5, 3) = -x10*x32 + x20*x38;
    jacobian(5, 4) = -x14*x35 + x22*x39;
    jacobian(5, 5) = -x17*x37 + x23*x40;
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
    const double x1 = std::sin(th_1);
    const double x2 = std::cos(th_2);
    const double x3 = 1.0*std::cos(th_0);
    const double x4 = x2*x3;
    const double x5 = std::cos(th_1);
    const double x6 = std::sin(th_2);
    const double x7 = x3*x6;
    const double x8 = x1*x4 - x5*x7;
    const double x9 = std::cos(th_3);
    const double x10 = std::sin(th_3);
    const double x11 = x1*x7 + x4*x5;
    const double x12 = std::cos(th_4);
    const double x13 = std::sin(th_4);
    const double x14 = x0*x2;
    const double x15 = x0*x6;
    const double x16 = x1*x14 - x15*x5;
    const double x17 = x1*x15 + x14*x5;
    const double x18 = 1.0*x1;
    const double x19 = 1.0*x5;
    const double x20 = x18*x6 + x19*x2;
    const double x21 = -x18*x2 + x19*x6;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = -x0;
    jacobian(0, 2) = x0;
    jacobian(0, 3) = x8;
    jacobian(0, 4) = -x0*x9 - x10*x11;
    jacobian(0, 5) = -x12*x8 - x13*(-x0*x10 + x11*x9);
    jacobian(1, 1) = x3;
    jacobian(1, 2) = -x3;
    jacobian(1, 3) = x16;
    jacobian(1, 4) = -x10*x17 + x3*x9;
    jacobian(1, 5) = -x12*x16 - x13*(x10*x3 + x17*x9);
    jacobian(2, 0) = 1.0;
    jacobian(2, 3) = x20;
    jacobian(2, 4) = -x10*x21;
    jacobian(2, 5) = -x12*x20 - x13*x21*x9;
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
    const double x13 = x12 + x8;
    const double x14 = std::sin(th_0);
    const double x15 = x5*x9;
    const double x16 = x11*x7;
    const double x17 = x14*x15 - x14*x16;
    const double x18 = a_0*x11;
    const double x19 = d_1*x17 + x14*x18;
    const double x20 = d_1*x13 + x6;
    const double x21 = std::sin(th_3);
    const double x22 = -x15 + x16;
    const double x23 = x21*x22;
    const double x24 = std::cos(th_3);
    const double x25 = x12*x14 + x14*x8;
    const double x26 = x2*x24 - x21*x25;
    const double x27 = -d_2*x23 + x20;
    const double x28 = d_2*x26 + x19;
    const double x29 = std::cos(th_4);
    const double x30 = std::sin(th_4);
    const double x31 = -x13*x29 - x22*x24*x30;
    const double x32 = -x17*x29 - x30*(x2*x21 + x24*x25);
    const double x33 = d_3*x31 + x27;
    const double x34 = d_3*x32 + x28;
    const double x35 = 1.0*p_on_ee_x;
    const double x36 = 1.0*x14;
    const double x37 = p_on_ee_z*x36;
    const double x38 = x2*x9;
    const double x39 = x10*x2;
    const double x40 = x38*x4 - x39*x7;
    const double x41 = a_0*x39 + d_1*x40;
    const double x42 = x10*x38 + x2*x4*x7;
    const double x43 = -x21*x42 - x24*x36;
    const double x44 = d_2*x43 + x41;
    const double x45 = -x29*x40 - x30*(-x21*x36 + x24*x42);
    const double x46 = d_3*x45 + x44;
    const double x47 = x1*x35;
    const double x48 = x0*x14;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = -pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = x2*x6 - x3;
    jacobian(0, 3) = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x19 - x17*x20;
    jacobian(0, 4) = p_on_ee_y*x23 + p_on_ee_z*x26 - x23*x28 - x26*x27;
    jacobian(0, 5) = -p_on_ee_y*x31 + p_on_ee_z*x32 + x31*x34 - x32*x33;
    jacobian(1, 0) = x35;
    jacobian(1, 1) = -pre_transform_special_symbol_23*x36 + x37;
    jacobian(1, 2) = x36*x6 - x37;
    jacobian(1, 3) = p_on_ee_x*x13 - p_on_ee_z*x40 - x13*x41 + x20*x40;
    jacobian(1, 4) = -p_on_ee_x*x23 - p_on_ee_z*x43 + x23*x44 + x27*x43;
    jacobian(1, 5) = p_on_ee_x*x31 - p_on_ee_z*x45 - x31*x46 + x33*x45;
    jacobian(2, 1) = -x47 - x48;
    jacobian(2, 2) = -std::pow(x1, 2)*x18 - std::pow(x14, 2)*x18 + x47 + x48;
    jacobian(2, 3) = -p_on_ee_x*x17 + p_on_ee_y*x40 + x17*x41 - x19*x40;
    jacobian(2, 4) = -p_on_ee_x*x26 + p_on_ee_y*x43 + x26*x44 - x28*x43;
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
        R_l(0, 0) = d_2*r_21;
        R_l(0, 1) = d_2*r_22;
        R_l(0, 2) = d_2*r_11;
        R_l(0, 3) = d_2*r_12;
        R_l(0, 4) = Py - d_3*r_23;
        R_l(0, 5) = Px - d_3*r_13;
        R_l(1, 0) = d_2*r_11;
        R_l(1, 1) = d_2*r_12;
        R_l(1, 2) = -d_2*r_21;
        R_l(1, 3) = -d_2*r_22;
        R_l(1, 4) = Px - d_3*r_13;
        R_l(1, 5) = -Py + d_3*r_23;
        R_l(2, 6) = d_2*r_31;
        R_l(2, 7) = d_2*r_32;
        R_l(3, 0) = -r_21;
        R_l(3, 1) = -r_22;
        R_l(3, 2) = -r_11;
        R_l(3, 3) = -r_12;
        R_l(4, 0) = -r_11;
        R_l(4, 1) = -r_12;
        R_l(4, 2) = r_21;
        R_l(4, 3) = r_22;
        R_l(5, 6) = 2*Px*d_2*r_11 + 2*Py*d_2*r_21 + 2*Pz*d_2*r_31 - 2*d_2*d_3*r_11*r_13 - 2*d_2*d_3*r_21*r_23 - 2*d_2*d_3*r_31*r_33;
        R_l(5, 7) = 2*Px*d_2*r_12 + 2*Py*d_2*r_22 + 2*Pz*d_2*r_32 - 2*d_2*d_3*r_12*r_13 - 2*d_2*d_3*r_22*r_23 - 2*d_2*d_3*r_32*r_33;
        R_l(6, 0) = -Px*r_31 + Pz*r_11 - d_3*r_11*r_33 + d_3*r_13*r_31;
        R_l(6, 1) = -Px*r_32 + Pz*r_12 - d_3*r_12*r_33 + d_3*r_13*r_32;
        R_l(6, 2) = Py*r_31 - Pz*r_21 + d_3*r_21*r_33 - d_3*r_23*r_31;
        R_l(6, 3) = Py*r_32 - Pz*r_22 + d_3*r_22*r_33 - d_3*r_23*r_32;
        R_l(7, 0) = Py*r_31 - Pz*r_21 + d_3*r_21*r_33 - d_3*r_23*r_31;
        R_l(7, 1) = Py*r_32 - Pz*r_22 + d_3*r_22*r_33 - d_3*r_23*r_32;
        R_l(7, 2) = Px*r_31 - Pz*r_11 + d_3*r_11*r_33 - d_3*r_13*r_31;
        R_l(7, 3) = Px*r_32 - Pz*r_12 + d_3*r_12*r_33 - d_3*r_13*r_32;
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
        const double x0 = R_l_inv_62*r_31 + R_l_inv_72*r_32;
        const double x1 = d_1*x0;
        const double x2 = R_l_inv_60*r_31 + R_l_inv_70*r_32;
        const double x3 = a_0*x2;
        const double x4 = -x3;
        const double x5 = x1 + x4;
        const double x6 = R_l_inv_66*r_31;
        const double x7 = R_l_inv_76*r_32;
        const double x8 = x6 + x7;
        const double x9 = d_1*x8;
        const double x10 = d_3*r_33;
        const double x11 = Pz - x10;
        const double x12 = -x0*x11;
        const double x13 = R_l_inv_65*r_31 + R_l_inv_75*r_32;
        const double x14 = std::pow(Px, 2);
        const double x15 = std::pow(Py, 2);
        const double x16 = std::pow(Pz, 2);
        const double x17 = std::pow(d_2, 2);
        const double x18 = std::pow(r_11, 2);
        const double x19 = x17*x18;
        const double x20 = std::pow(r_21, 2);
        const double x21 = x17*x20;
        const double x22 = std::pow(r_31, 2);
        const double x23 = x17*x22;
        const double x24 = std::pow(d_3, 2);
        const double x25 = std::pow(r_13, 2)*x24;
        const double x26 = std::pow(r_23, 2)*x24;
        const double x27 = std::pow(r_33, 2)*x24;
        const double x28 = d_3*r_13;
        const double x29 = 2*Px;
        const double x30 = x28*x29;
        const double x31 = d_3*r_23;
        const double x32 = 2*Py;
        const double x33 = x31*x32;
        const double x34 = 2*Pz;
        const double x35 = x10*x34;
        const double x36 = std::pow(a_0, 2);
        const double x37 = std::pow(d_1, 2);
        const double x38 = -x36 - x37;
        const double x39 = x14 + x15 + x16 + x19 + x21 + x23 + x25 + x26 + x27 - x30 - x33 - x35 + x38;
        const double x40 = -x13*x39;
        const double x41 = x12 + x40 - x9;
        const double x42 = R_l_inv_64*r_31;
        const double x43 = R_l_inv_74*r_32;
        const double x44 = x42 + x43;
        const double x45 = x41 + x44;
        const double x46 = 2*a_0;
        const double x47 = x46*x8;
        const double x48 = -x0*x46;
        const double x49 = 2*d_1;
        const double x50 = x2*x49;
        const double x51 = x48 - x50;
        const double x52 = -x1;
        const double x53 = x3 + x52;
        const double x54 = x12 + x40 + x9;
        const double x55 = x44 + x54;
        const double x56 = 2*R_l_inv_63;
        const double x57 = r_31*x56;
        const double x58 = 2*R_l_inv_73;
        const double x59 = r_32*x58;
        const double x60 = R_l_inv_67*r_31 + R_l_inv_77*r_32;
        const double x61 = -x49*x60;
        const double x62 = -x57 - x59 + x61;
        const double x63 = x57 + x59 + x61;
        const double x64 = -x42 - x43;
        const double x65 = x54 + x64;
        const double x66 = -x47;
        const double x67 = x41 + x64;
        const double x68 = Px*r_11;
        const double x69 = Py*r_21;
        const double x70 = Pz*r_31;
        const double x71 = r_11*x28;
        const double x72 = r_21*x31;
        const double x73 = r_31*x10;
        const double x74 = x68 + x69 + x70 - x71 - x72 - x73;
        const double x75 = Px*r_12;
        const double x76 = Py*r_22;
        const double x77 = Pz*r_32;
        const double x78 = r_12*x28;
        const double x79 = r_22*x31;
        const double x80 = r_32*x10;
        const double x81 = x75 + x76 + x77 - x78 - x79 - x80;
        const double x82 = R_l_inv_62*x74 + R_l_inv_72*x81;
        const double x83 = d_1*x82;
        const double x84 = R_l_inv_60*x74 + R_l_inv_70*x81;
        const double x85 = a_0*x84;
        const double x86 = -x85;
        const double x87 = x83 + x86;
        const double x88 = d_2*x18;
        const double x89 = d_2*x20;
        const double x90 = d_2*x22;
        const double x91 = R_l_inv_66*x74;
        const double x92 = R_l_inv_76*x81;
        const double x93 = x91 + x92;
        const double x94 = d_1*x93;
        const double x95 = -x11*x82;
        const double x96 = R_l_inv_65*x74 + R_l_inv_75*x81;
        const double x97 = -x39*x96;
        const double x98 = x88 + x89 + x90 - x94 + x95 + x97;
        const double x99 = R_l_inv_64*x74;
        const double x100 = R_l_inv_74*x81;
        const double x101 = x100 + x99;
        const double x102 = x101 + x98;
        const double x103 = x46*x93;
        const double x104 = -x46*x82;
        const double x105 = x49*x84;
        const double x106 = x104 - x105;
        const double x107 = -x83;
        const double x108 = x107 + x85;
        const double x109 = x88 + x89 + x90 + x94 + x95 + x97;
        const double x110 = x101 + x109;
        const double x111 = R_l_inv_67*x74 + R_l_inv_77*x81;
        const double x112 = -x111*x49;
        const double x113 = x112 + x46;
        const double x114 = x56*x74;
        const double x115 = x58*x81;
        const double x116 = -x114 - x115;
        const double x117 = x114 + x115;
        const double x118 = -x100 - x99;
        const double x119 = x109 + x118;
        const double x120 = -x103;
        const double x121 = x118 + x98;
        const double x122 = Px*r_21;
        const double x123 = Py*r_11;
        const double x124 = r_11*x31 - r_21*x28 + x122 - x123;
        const double x125 = R_l_inv_62*x124;
        const double x126 = Px*r_22;
        const double x127 = Py*r_12;
        const double x128 = r_12*x31 - r_22*x28 + x126 - x127;
        const double x129 = R_l_inv_72*x128;
        const double x130 = x125 + x129;
        const double x131 = x11*x130;
        const double x132 = R_l_inv_65*x124 + R_l_inv_75*x128;
        const double x133 = x132*x39;
        const double x134 = R_l_inv_64*x124;
        const double x135 = R_l_inv_74*x128;
        const double x136 = x131 + x133 - x134 - x135;
        const double x137 = R_l_inv_60*x124;
        const double x138 = R_l_inv_70*x128;
        const double x139 = x137 + x138;
        const double x140 = a_0*x139;
        const double x141 = d_1*x130;
        const double x142 = -x141;
        const double x143 = x140 + x142;
        const double x144 = -a_0;
        const double x145 = R_l_inv_66*x124 + R_l_inv_76*x128;
        const double x146 = d_1*x145;
        const double x147 = x144 + x146;
        const double x148 = -x49;
        const double x149 = x145*x46;
        const double x150 = -x149;
        const double x151 = x130*x46;
        const double x152 = x139*x49;
        const double x153 = x151 + x152;
        const double x154 = -x146;
        const double x155 = a_0 + x154;
        const double x156 = -x140;
        const double x157 = x141 + x156;
        const double x158 = x124*x56;
        const double x159 = x128*x58;
        const double x160 = R_l_inv_67*x124 + R_l_inv_77*x128;
        const double x161 = x160*x49;
        const double x162 = x158 + x159 + x161;
        const double x163 = -x158 - x159 + x161;
        const double x164 = x131 + x133 + x134 + x135;
        const double x165 = Py*x18 + Py*x20 + Py*x22 - x18*x31 - x20*x31 - x22*x31;
        const double x166 = 2*d_2;
        const double x167 = x165*x166;
        const double x168 = R_l_inv_40*x167;
        const double x169 = Px*x18 + Px*x20 + Px*x22 - x18*x28 - x20*x28 - x22*x28;
        const double x170 = x166*x169;
        const double x171 = R_l_inv_50*x170;
        const double x172 = 2*x31;
        const double x173 = 2*x28;
        const double x174 = 2*r_11;
        const double x175 = r_23*x24;
        const double x176 = r_13*x175;
        const double x177 = 2*r_31;
        const double x178 = r_33*x175;
        const double x179 = std::pow(r_21, 3)*x17 - r_21*x14 + r_21*x15 - r_21*x16 + r_21*x19 + r_21*x23 - r_21*x25 + r_21*x26 - r_21*x27 + r_21*x35 + x122*x173 - x123*x173 - x172*x68 - x172*x69 - x172*x70 + x174*x176 + x177*x178 + x32*x68 + x32*x70 - x32*x73;
        const double x180 = R_l_inv_00*x179;
        const double x181 = 2*x176;
        const double x182 = r_13*r_33*x24;
        const double x183 = std::pow(r_11, 3)*x17 + r_11*x14 - r_11*x15 - r_11*x16 + r_11*x21 + r_11*x23 + r_11*x25 - r_11*x26 - r_11*x27 + r_11*x35 + r_21*x181 - x122*x172 + x123*x172 - x173*x68 - x173*x69 - x173*x70 + x177*x182 + x29*x69 + x29*x70 - x29*x73;
        const double x184 = R_l_inv_20*x183;
        const double x185 = r_21*x17;
        const double x186 = x174*x185;
        const double x187 = x177*x185;
        const double x188 = 2*r_32;
        const double x189 = r_12*x181 + r_12*x186 - r_22*x14 + r_22*x15 - r_22*x16 + r_22*x19 + 3*r_22*x21 + r_22*x23 - r_22*x25 + r_22*x26 - r_22*x27 + r_22*x35 + r_32*x187 + x126*x173 - x127*x173 - x172*x75 - x172*x76 - x172*x77 + x178*x188 + x32*x75 + x32*x77 - x32*x80;
        const double x190 = R_l_inv_10*x189;
        const double x191 = r_31*x17*x174;
        const double x192 = r_12*x14 - r_12*x15 - r_12*x16 + 3*r_12*x19 + r_12*x21 + r_12*x23 + r_12*x25 - r_12*x26 - r_12*x27 + r_12*x35 + r_22*x181 + r_22*x186 + r_32*x191 - x126*x172 + x127*x172 - x173*x75 - x173*x76 - x173*x77 + x182*x188 + x29*x76 + x29*x77 - x29*x80;
        const double x193 = R_l_inv_30*x192;
        const double x194 = x168 + x171 + x180 + x184 + x190 + x193;
        const double x195 = a_0*x194;
        const double x196 = R_l_inv_42*x167;
        const double x197 = R_l_inv_52*x170;
        const double x198 = R_l_inv_02*x179;
        const double x199 = R_l_inv_22*x183;
        const double x200 = R_l_inv_12*x189;
        const double x201 = R_l_inv_32*x192;
        const double x202 = x196 + x197 + x198 + x199 + x200 + x201;
        const double x203 = d_1*x202;
        const double x204 = -x203;
        const double x205 = x195 + x204;
        const double x206 = R_l_inv_06*x179 + R_l_inv_16*x189 + R_l_inv_26*x183 + R_l_inv_36*x192 + R_l_inv_46*x167 + R_l_inv_56*x170;
        const double x207 = d_1*x206;
        const double x208 = x11*x202;
        const double x209 = R_l_inv_05*x179 + R_l_inv_15*x189 + R_l_inv_25*x183 + R_l_inv_35*x192 + R_l_inv_45*x167 + R_l_inv_55*x170;
        const double x210 = x209*x39;
        const double x211 = x207 + x208 + x210;
        const double x212 = R_l_inv_04*x179;
        const double x213 = R_l_inv_14*x189;
        const double x214 = R_l_inv_24*x183;
        const double x215 = R_l_inv_34*x192;
        const double x216 = R_l_inv_44*x167;
        const double x217 = R_l_inv_54*x170;
        const double x218 = -x212 - x213 - x214 - x215 - x216 - x217;
        const double x219 = x211 + x218;
        const double x220 = x206*x46;
        const double x221 = -x220;
        const double x222 = x202*x46;
        const double x223 = x194*x49;
        const double x224 = x222 + x223;
        const double x225 = -x195;
        const double x226 = x203 + x225;
        const double x227 = -x207 + x208 + x210;
        const double x228 = x218 + x227;
        const double x229 = 2*x36;
        const double x230 = 2*x37;
        const double x231 = 4*d_2;
        const double x232 = x165*x231;
        const double x233 = R_l_inv_43*x232;
        const double x234 = x169*x231;
        const double x235 = R_l_inv_53*x234;
        const double x236 = 2*x179;
        const double x237 = R_l_inv_03*x236;
        const double x238 = 2*x183;
        const double x239 = R_l_inv_23*x238;
        const double x240 = 2*x189;
        const double x241 = R_l_inv_13*x240;
        const double x242 = 2*x192;
        const double x243 = R_l_inv_33*x242;
        const double x244 = R_l_inv_07*x179 + R_l_inv_17*x189 + R_l_inv_27*x183 + R_l_inv_37*x192 + R_l_inv_47*x167 + R_l_inv_57*x170;
        const double x245 = x244*x49;
        const double x246 = x229 - x230 + x233 + x235 + x237 + x239 + x241 + x243 + x245;
        const double x247 = a_0*d_1;
        const double x248 = 8*x247;
        const double x249 = -x229 + x230 - x233 - x235 - x237 - x239 - x241 - x243 + x245;
        const double x250 = x212 + x213 + x214 + x215 + x216 + x217;
        const double x251 = x227 + x250;
        const double x252 = x211 + x250;
        const double x253 = R_l_inv_40*x170;
        const double x254 = R_l_inv_50*x167;
        const double x255 = R_l_inv_00*x183;
        const double x256 = R_l_inv_20*x179;
        const double x257 = R_l_inv_10*x192;
        const double x258 = R_l_inv_30*x189;
        const double x259 = x253 - x254 + x255 - x256 + x257 - x258;
        const double x260 = a_0*x259;
        const double x261 = R_l_inv_42*x170;
        const double x262 = R_l_inv_52*x167;
        const double x263 = R_l_inv_02*x183;
        const double x264 = R_l_inv_22*x179;
        const double x265 = R_l_inv_12*x192;
        const double x266 = R_l_inv_32*x189;
        const double x267 = x261 - x262 + x263 - x264 + x265 - x266;
        const double x268 = d_1*x267;
        const double x269 = -x268;
        const double x270 = x260 + x269;
        const double x271 = R_l_inv_46*x170;
        const double x272 = R_l_inv_56*x167;
        const double x273 = R_l_inv_06*x183;
        const double x274 = R_l_inv_26*x179;
        const double x275 = R_l_inv_16*x192;
        const double x276 = R_l_inv_36*x189;
        const double x277 = x271 - x272 + x273 - x274 + x275 - x276;
        const double x278 = d_1*x277;
        const double x279 = x11*x267;
        const double x280 = R_l_inv_05*x183 + R_l_inv_15*x192 - R_l_inv_25*x179 - R_l_inv_35*x189 + R_l_inv_45*x170 - R_l_inv_55*x167;
        const double x281 = x280*x39;
        const double x282 = x278 + x279 + x281;
        const double x283 = R_l_inv_24*x179;
        const double x284 = R_l_inv_34*x189;
        const double x285 = R_l_inv_04*x183;
        const double x286 = R_l_inv_14*x192;
        const double x287 = R_l_inv_44*x170;
        const double x288 = R_l_inv_54*x167;
        const double x289 = x283 + x284 - x285 - x286 - x287 + x288 + x36 + x37;
        const double x290 = x282 + x289;
        const double x291 = x277*x46;
        const double x292 = -x291;
        const double x293 = x267*x46;
        const double x294 = x259*x49;
        const double x295 = x293 + x294;
        const double x296 = -x260;
        const double x297 = x268 + x296;
        const double x298 = -x278 + x279 + x281;
        const double x299 = x289 + x298;
        const double x300 = R_l_inv_43*x234;
        const double x301 = R_l_inv_53*x232;
        const double x302 = R_l_inv_03*x238;
        const double x303 = R_l_inv_23*x236;
        const double x304 = R_l_inv_13*x242;
        const double x305 = R_l_inv_33*x240;
        const double x306 = R_l_inv_07*x183 + R_l_inv_17*x192 - R_l_inv_27*x179 - R_l_inv_37*x189 + R_l_inv_47*x170 - R_l_inv_57*x167;
        const double x307 = x306*x49;
        const double x308 = x300 - x301 + x302 - x303 + x304 - x305 + x307;
        const double x309 = -x300 + x301 - x302 + x303 - x304 + x305 + x307;
        const double x310 = -x283 - x284 + x285 + x286 + x287 - x288 + x38;
        const double x311 = x298 + x310;
        const double x312 = x282 + x310;
        const double x313 = 2*x10;
        const double x314 = 2*x178;
        const double x315 = r_21*x314 + std::pow(r_31, 3)*x17 - r_31*x14 - r_31*x15 + r_31*x16 + r_31*x19 + r_31*x21 - r_31*x25 - r_31*x26 + r_31*x27 + r_31*x30 + r_31*x33 + x174*x182 - x313*x68 - x313*x69 - x313*x70 + x34*x68 + x34*x69 - x34*x71 - x34*x72;
        const double x316 = R_l_inv_60*x315;
        const double x317 = 2*r_12*x182 + r_12*x191 + r_22*x187 + r_22*x314 - r_32*x14 - r_32*x15 + r_32*x16 + r_32*x19 + r_32*x21 + 3*r_32*x23 - r_32*x25 - r_32*x26 + r_32*x27 + r_32*x30 + r_32*x33 - x313*x75 - x313*x76 - x313*x77 + x34*x75 + x34*x76 - x34*x78 - x34*x79;
        const double x318 = R_l_inv_70*x317;
        const double x319 = x316 + x318;
        const double x320 = a_0*x319;
        const double x321 = R_l_inv_62*x315;
        const double x322 = R_l_inv_72*x317;
        const double x323 = x321 + x322;
        const double x324 = d_1*x323;
        const double x325 = -x324;
        const double x326 = x320 + x325;
        const double x327 = R_l_inv_66*x315 + R_l_inv_76*x317;
        const double x328 = d_1*x327;
        const double x329 = x11*x323;
        const double x330 = R_l_inv_65*x315 + R_l_inv_75*x317;
        const double x331 = x330*x39;
        const double x332 = -x34*x88;
        const double x333 = -x34*x89;
        const double x334 = -x34*x90;
        const double x335 = x313*x88;
        const double x336 = x313*x89;
        const double x337 = x313*x90;
        const double x338 = x328 + x329 + x331 + x332 + x333 + x334 + x335 + x336 + x337;
        const double x339 = R_l_inv_64*x315;
        const double x340 = R_l_inv_74*x317;
        const double x341 = -x339 - x340;
        const double x342 = x338 + x341;
        const double x343 = x327*x46;
        const double x344 = -x343;
        const double x345 = x323*x46;
        const double x346 = x319*x49;
        const double x347 = x345 + x346;
        const double x348 = -x320;
        const double x349 = x324 + x348;
        const double x350 = -x328 + x329 + x331 + x332 + x333 + x334 + x335 + x336 + x337;
        const double x351 = x341 + x350;
        const double x352 = 4*x247;
        const double x353 = R_l_inv_67*x315 + R_l_inv_77*x317;
        const double x354 = x353*x49;
        const double x355 = -x352 + x354;
        const double x356 = x315*x56;
        const double x357 = x317*x58;
        const double x358 = x356 + x357;
        const double x359 = 4*x36;
        const double x360 = 4*x37;
        const double x361 = -x360;
        const double x362 = -x356 - x357;
        const double x363 = x352 + x354;
        const double x364 = x339 + x340;
        const double x365 = x350 + x364;
        const double x366 = x338 + x364;
        const double x367 = x13*x46;
        const double x368 = x49*(x2 - x367);
        const double x369 = 4*d_1;
        const double x370 = -x49*(x2 + x367);
        const double x371 = 4*a_0;
        const double x372 = x371*x60;
        const double x373 = 8*R_l_inv_63;
        const double x374 = 8*R_l_inv_73;
        const double x375 = x46*x96;
        const double x376 = x49*(-x375 + x84);
        const double x377 = -x49*(x375 + x84);
        const double x378 = x111*x371;
        const double x379 = x132*x46;
        const double x380 = x379 + 1;
        const double x381 = -x137 - x138;
        const double x382 = x379 - 1;
        const double x383 = -x160*x371;
        const double x384 = x209*x46;
        const double x385 = x49*(-x168 - x171 - x180 - x184 - x190 - x193 + x384);
        const double x386 = x49*(x194 + x384);
        const double x387 = -x244*x371;
        const double x388 = 16*d_2;
        const double x389 = x165*x388;
        const double x390 = x169*x388;
        const double x391 = 8*x179;
        const double x392 = 8*x183;
        const double x393 = 8*x189;
        const double x394 = 8*x192;
        const double x395 = -x46;
        const double x396 = x280*x46;
        const double x397 = x395 + x396;
        const double x398 = -x253 + x254 - x255 + x256 - x257 + x258;
        const double x399 = -x261 + x262 - x263 + x264 - x265 + x266;
        const double x400 = -x306*x371;
        const double x401 = x396 + x46;
        const double x402 = x330*x46;
        const double x403 = x49*(-x316 - x318 + x402);
        const double x404 = x49*(x319 + x402);
        const double x405 = -x353*x371;
        const double x406 = -x359;
        const double x407 = x4 + x52;
        const double x408 = x48 + x50;
        const double x409 = x1 + x3;
        const double x410 = x107 + x86;
        const double x411 = x104 + x105;
        const double x412 = x83 + x85;
        const double x413 = x112 + x395;
        const double x414 = x140 + x141;
        const double x415 = x144 + x154;
        const double x416 = x151 - x152;
        const double x417 = a_0 + x146;
        const double x418 = x142 + x156;
        const double x419 = x195 + x203;
        const double x420 = x222 - x223;
        const double x421 = x204 + x225;
        const double x422 = x260 + x268;
        const double x423 = x293 - x294;
        const double x424 = x269 + x296;
        const double x425 = x320 + x324;
        const double x426 = x345 - x346;
        const double x427 = x325 + x348;
        
        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = x45 + x5;
        A(0, 1) = x47 + x51;
        A(0, 2) = x53 + x55;
        A(0, 3) = x62;
        A(0, 4) = -4;
        A(0, 5) = x63;
        A(0, 6) = x5 + x65;
        A(0, 7) = x51 + x66;
        A(0, 8) = x53 + x67;
        A(1, 0) = x102 + x87;
        A(1, 1) = x103 + x106;
        A(1, 2) = x108 + x110;
        A(1, 3) = x113 + x116;
        A(1, 5) = x113 + x117;
        A(1, 6) = x119 + x87;
        A(1, 7) = x106 + x120;
        A(1, 8) = x108 + x121;
        A(2, 0) = x136 + x143 + x147;
        A(2, 1) = x148 + x150 + x153;
        A(2, 2) = x136 + x155 + x157;
        A(2, 3) = x162;
        A(2, 5) = x163;
        A(2, 6) = x143 + x155 + x164;
        A(2, 7) = x149 + x153 + x49;
        A(2, 8) = x147 + x157 + x164;
        A(3, 0) = x205 + x219;
        A(3, 1) = x221 + x224;
        A(3, 2) = x226 + x228;
        A(3, 3) = x246;
        A(3, 4) = x248;
        A(3, 5) = x249;
        A(3, 6) = x205 + x251;
        A(3, 7) = x220 + x224;
        A(3, 8) = x226 + x252;
        A(4, 0) = x270 + x290;
        A(4, 1) = x292 + x295;
        A(4, 2) = x297 + x299;
        A(4, 3) = x308;
        A(4, 5) = x309;
        A(4, 6) = x270 + x311;
        A(4, 7) = x291 + x295;
        A(4, 8) = x297 + x312;
        A(5, 0) = x326 + x342;
        A(5, 1) = x344 + x347;
        A(5, 2) = x349 + x351;
        A(5, 3) = x355 + x358;
        A(5, 4) = x359 + x361;
        A(5, 5) = x362 + x363;
        A(5, 6) = x326 + x365;
        A(5, 7) = x343 + x347;
        A(5, 8) = x349 + x366;
        
        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x368;
        B(0, 1) = x369*(x0 - x6 - x7);
        B(0, 2) = x370;
        B(0, 3) = x372 + 4;
        B(0, 4) = -r_31*x373 - r_32*x374;
        B(0, 5) = x372 - 4;
        B(0, 6) = x368;
        B(0, 7) = x369*(x0 + x8);
        B(0, 8) = x370;
        B(1, 0) = x376;
        B(1, 1) = x369*(x82 - x91 - x92);
        B(1, 2) = x377;
        B(1, 3) = x378;
        B(1, 4) = -x373*x74 - x374*x81;
        B(1, 5) = x378;
        B(1, 6) = x376;
        B(1, 7) = x369*(x82 + x93);
        B(1, 8) = x377;
        B(2, 0) = x49*(x380 + x381);
        B(2, 1) = x369*(-x125 - x129 + x145);
        B(2, 2) = x49*(x139 + x382);
        B(2, 3) = x383;
        B(2, 4) = x124*x373 + x128*x374;
        B(2, 5) = x383;
        B(2, 6) = x49*(x381 + x382);
        B(2, 7) = -x369*(x130 + x145);
        B(2, 8) = x49*(x139 + x380);
        B(3, 0) = x385;
        B(3, 1) = x369*(-x196 - x197 - x198 - x199 - x200 - x201 + x206);
        B(3, 2) = x386;
        B(3, 3) = x387;
        B(3, 4) = R_l_inv_03*x391 + R_l_inv_13*x393 + R_l_inv_23*x392 + R_l_inv_33*x394 + R_l_inv_43*x389 + R_l_inv_53*x390 - 8*x36 - 8*x37;
        B(3, 5) = x387;
        B(3, 6) = x385;
        B(3, 7) = -x369*(x202 + x206);
        B(3, 8) = x386;
        B(4, 0) = x49*(x397 + x398);
        B(4, 1) = x369*(x277 + x399);
        B(4, 2) = x49*(x259 + x397);
        B(4, 3) = x400;
        B(4, 4) = R_l_inv_03*x392 + R_l_inv_13*x394 - R_l_inv_23*x391 - R_l_inv_33*x393 + R_l_inv_43*x390 - R_l_inv_53*x389;
        B(4, 5) = x400;
        B(4, 6) = x49*(x398 + x401);
        B(4, 7) = x369*(-x271 + x272 - x273 + x274 - x275 + x276 + x399);
        B(4, 8) = x49*(x259 + x401);
        B(5, 0) = x403;
        B(5, 1) = x369*(-x321 - x322 + x327);
        B(5, 2) = x404;
        B(5, 3) = x359 + x360 + x405;
        B(5, 4) = x315*x373 + x317*x374;
        B(5, 5) = x361 + x405 + x406;
        B(5, 6) = x403;
        B(5, 7) = -x369*(x323 + x327);
        B(5, 8) = x404;
        
        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = x407 + x55;
        C(0, 1) = x408 + x47;
        C(0, 2) = x409 + x45;
        C(0, 3) = x63;
        C(0, 4) = 4;
        C(0, 5) = x62;
        C(0, 6) = x407 + x67;
        C(0, 7) = x408 + x66;
        C(0, 8) = x409 + x65;
        C(1, 0) = x110 + x410;
        C(1, 1) = x103 + x411;
        C(1, 2) = x102 + x412;
        C(1, 3) = x117 + x413;
        C(1, 5) = x116 + x413;
        C(1, 6) = x121 + x410;
        C(1, 7) = x120 + x411;
        C(1, 8) = x119 + x412;
        C(2, 0) = x136 + x414 + x415;
        C(2, 1) = x150 + x416 + x49;
        C(2, 2) = x136 + x417 + x418;
        C(2, 3) = x163;
        C(2, 5) = x162;
        C(2, 6) = x164 + x414 + x417;
        C(2, 7) = x148 + x149 + x416;
        C(2, 8) = x164 + x415 + x418;
        C(3, 0) = x228 + x419;
        C(3, 1) = x221 + x420;
        C(3, 2) = x219 + x421;
        C(3, 3) = x249;
        C(3, 4) = x248;
        C(3, 5) = x246;
        C(3, 6) = x252 + x419;
        C(3, 7) = x220 + x420;
        C(3, 8) = x251 + x421;
        C(4, 0) = x299 + x422;
        C(4, 1) = x292 + x423;
        C(4, 2) = x290 + x424;
        C(4, 3) = x309;
        C(4, 5) = x308;
        C(4, 6) = x312 + x422;
        C(4, 7) = x291 + x423;
        C(4, 8) = x311 + x424;
        C(5, 0) = x351 + x425;
        C(5, 1) = x344 + x426;
        C(5, 2) = x342 + x427;
        C(5, 3) = x355 + x362;
        C(5, 4) = x360 + x406;
        C(5, 5) = x358 + x363;
        C(5, 6) = x366 + x425;
        C(5, 7) = x343 + x426;
        C(5, 8) = x365 + x427;
        
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
            
            const bool degenerate_valid_0 = std::fabs(th_2 - 1.0/2.0*M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
            }
            
            const bool degenerate_valid_1 = std::fabs(th_2 + M_PI_2) <= 9.9999999999999995e-7;
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
            
            const bool condition_0 = (1.0/2.0)*std::fabs((std::pow(Px, 2) - 2*Px*d_3*r_13 + std::pow(Py, 2) - 2*Py*d_3*r_23 + std::pow(Pz, 2) - 2*Pz*d_3*r_33 - std::pow(a_0, 2) + 2*a_0*d_1*std::sin(th_2) - std::pow(d_1, 2) - std::pow(d_2, 2) + std::pow(d_3, 2)*std::pow(r_13, 2) + std::pow(d_3, 2)*std::pow(r_23, 2) + std::pow(d_3, 2)*std::pow(r_33, 2))/(a_0*d_2*std::cos(th_2))) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 2*d_3;
                const double x1 = std::pow(d_3, 2);
                const double x2 = std::asin((1.0/2.0)*(std::pow(Px, 2) - Px*r_13*x0 + std::pow(Py, 2) - Py*r_23*x0 + std::pow(Pz, 2) - Pz*r_33*x0 - std::pow(a_0, 2) + 2*a_0*d_1*std::sin(th_2) - std::pow(d_1, 2) - std::pow(d_2, 2) + std::pow(r_13, 2)*x1 + std::pow(r_23, 2)*x1 + std::pow(r_33, 2)*x1)/(a_0*d_2*std::cos(th_2)));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[3] = -x2;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = (1.0/2.0)*std::fabs((std::pow(Px, 2) - 2*Px*d_3*r_13 + std::pow(Py, 2) - 2*Py*d_3*r_23 + std::pow(Pz, 2) - 2*Pz*d_3*r_33 - std::pow(a_0, 2) + 2*a_0*d_1*std::sin(th_2) - std::pow(d_1, 2) - std::pow(d_2, 2) + std::pow(d_3, 2)*std::pow(r_13, 2) + std::pow(d_3, 2)*std::pow(r_23, 2) + std::pow(d_3, 2)*std::pow(r_33, 2))/(a_0*d_2*std::cos(th_2))) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = 2*d_3;
                const double x1 = std::pow(d_3, 2);
                const double x2 = std::asin((1.0/2.0)*(std::pow(Px, 2) - Px*r_13*x0 + std::pow(Py, 2) - Py*r_23*x0 + std::pow(Pz, 2) - Pz*r_33*x0 - std::pow(a_0, 2) + 2*a_0*d_1*std::sin(th_2) - std::pow(d_1, 2) - std::pow(d_2, 2) + std::pow(r_13, 2)*x1 + std::pow(r_23, 2)*x1 + std::pow(r_33, 2)*x1)/(a_0*d_2*std::cos(th_2)));
                // End of temp variables
                const double tmp_sol_value = x2 + M_PI;
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
            const bool checked_result = std::fabs(d_2*std::cos(th_3)) <= 9.9999999999999995e-7 && std::fabs(Px - d_3*r_13) <= 9.9999999999999995e-7 && std::fabs(Py - d_3*r_23) <= 9.9999999999999995e-7;
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
            
            const bool condition_0 = std::fabs(d_2*std::cos(th_3)) >= zero_tolerance || std::fabs(Px - d_3*r_13) >= zero_tolerance || std::fabs(Py - d_3*r_23) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = -Px + d_3*r_13;
                const double x1 = Py - d_3*r_23;
                const double x2 = std::atan2(x0, x1);
                const double x3 = std::cos(th_3);
                const double x4 = std::sqrt(-std::pow(d_2, 2)*std::pow(x3, 2) + std::pow(x0, 2) + std::pow(x1, 2));
                const double x5 = d_2*x3;
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[0] = x2 + std::atan2(x4, x5);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx);
            }
            
            const bool condition_1 = std::fabs(d_2*std::cos(th_3)) >= zero_tolerance || std::fabs(Px - d_3*r_13) >= zero_tolerance || std::fabs(Py - d_3*r_23) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = -Px + d_3*r_13;
                const double x1 = Py - d_3*r_23;
                const double x2 = std::atan2(x0, x1);
                const double x3 = std::cos(th_3);
                const double x4 = std::sqrt(-std::pow(d_2, 2)*std::pow(x3, 2) + std::pow(x0, 2) + std::pow(x1, 2));
                const double x5 = d_2*x3;
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
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x1 + r_21*x0, -r_12*x1 + r_22*x0);
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
            
            const bool degenerate_valid_0 = std::fabs(th_2 + M_PI_2) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
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
            const double th_3 = this_solution[3];
            
            const bool condition_0 = std::fabs(d_2*std::sin(th_3)) >= 9.9999999999999995e-7 || std::fabs(a_0 + d_1) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_0 + d_1;
                const double x1 = Pz - d_3*r_33;
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3;
                const double x5 = d_2*std::sin(th_3);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5);
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
            const double th_3 = this_solution[3];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_13*(std::sin(th_0)*std::sin(th_3) + std::sin(th_1)*std::cos(th_0)*std::cos(th_3)) - r_23*(-std::sin(th_0)*std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_0)) + r_33*std::cos(th_1)*std::cos(th_3)) >= zero_tolerance || std::fabs(r_13*std::cos(th_0)*std::cos(th_1) + r_23*std::sin(th_0)*std::cos(th_1) - r_33*std::sin(th_1)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::cos(th_3);
                const double x2 = std::sin(th_0);
                const double x3 = std::sin(th_3);
                const double x4 = std::cos(th_0);
                const double x5 = std::sin(th_1);
                const double x6 = x1*x5;
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_13*(x2*x3 + x4*x6) - r_23*(-x2*x6 + x3*x4) + r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
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
            
            const bool condition_0 = std::fabs(d_1*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_0 - d_1*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Pz - d_3*r_33;
                const double x1 = a_0 - d_1*std::sin(th_2);
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3;
                const double x5 = d_1*std::cos(th_2);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x1 + x4*x5, x0*x5 + x1*x4);
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
                const double tmp_sol_value = std::atan2(-r_33*x4 + x5*x6 + x5*x7, -r_33*x5 - x4*x6 - x4*x7);
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
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1);
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
            
            const bool degenerate_valid_0 = std::fabs(th_2 + M_PI_2) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(22, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(14, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_13_processor();
    // Finish code for solved_variable dispatcher node 13
    
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
            const double th_3 = this_solution[3];
            
            const bool condition_0 = std::fabs(d_2*std::sin(th_3)) >= 9.9999999999999995e-7 || std::fabs(a_0 + d_1) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_0 + d_1;
                const double x1 = Pz - d_3*r_33;
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3;
                const double x5 = d_2*std::sin(th_3);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5);
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
            const double th_3 = this_solution[3];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_13*(std::sin(th_0)*std::sin(th_3) + std::sin(th_1)*std::cos(th_0)*std::cos(th_3)) - r_23*(-std::sin(th_0)*std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_0)) + r_33*std::cos(th_1)*std::cos(th_3)) >= zero_tolerance || std::fabs(r_13*std::cos(th_0)*std::cos(th_1) + r_23*std::sin(th_0)*std::cos(th_1) - r_33*std::sin(th_1)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::cos(th_3);
                const double x2 = std::sin(th_0);
                const double x3 = std::sin(th_3);
                const double x4 = std::cos(th_0);
                const double x5 = std::sin(th_1);
                const double x6 = x1*x5;
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_13*(x2*x3 + x4*x6) - r_23*(-x2*x6 + x3*x4) + r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
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
            
            const bool condition_0 = std::fabs(d_1*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_0 - d_1*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Pz - d_3*r_33;
                const double x1 = a_0 - d_1*std::sin(th_2);
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = Px*x2 + Py*x3 - d_3*r_13*x2 - d_3*r_23*x3;
                const double x5 = d_1*std::cos(th_2);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x1 + x4*x5, x0*x5 + x1*x4);
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
                const double tmp_sol_value = std::atan2(r_33*x4 - x5*x6 - x5*x7, -r_33*x5 - x4*x6 - x4*x7);
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
                const double x0 = std::asin((r_13*std::sin(th_0) - r_23*std::cos(th_0))/std::sin(th_3));
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
                const double x0 = std::asin((r_13*std::sin(th_0) - r_23*std::cos(th_0))/std::sin(th_3));
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
            const bool checked_result = std::fabs(d_1*std::cos(th_2) - d_2*std::sin(th_2)*std::sin(th_3)) <= 9.9999999999999995e-7 && std::fabs(-a_0 + d_1*std::sin(th_2) + d_2*std::sin(th_3)*std::cos(th_2)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(9, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_8_processor();
    // Finish code for equation all-zero dispatcher node 8
    
    // Code for explicit solution node 9, solved variable is th_1
    auto ExplicitSolutionNode_node_9_solve_th_1_processor = [&]() -> void
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
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            
            const bool condition_0 = std::fabs(d_1*std::cos(th_2) - d_2*std::sin(th_2)*std::sin(th_3)) >= 9.9999999999999995e-7 || std::fabs(-a_0 + d_1*std::sin(th_2) + d_2*std::sin(th_3)*std::cos(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = d_3*r_33;
                const double x1 = std::sin(th_2);
                const double x2 = std::cos(th_2);
                const double x3 = d_2*std::sin(th_3);
                const double x4 = a_0 - d_1*x1 - x2*x3;
                const double x5 = d_1*x2 - x1*x3;
                const double x6 = std::cos(th_0);
                const double x7 = std::sin(th_0);
                const double x8 = Px*x6 + Py*x7 - d_3*r_13*x6 - d_3*r_23*x7;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x4*(-Pz + x0) + x5*x8, x4*x8 + x5*(Pz - x0));
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(10, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_1_processor();
    // Finish code for explicit solution node 9
    
    // Code for non-branch dispatcher node 10
    // Actually, there is no code
    
    // Code for explicit solution node 11, solved variable is th_5
    auto ExplicitSolutionNode_node_11_solve_th_5_processor = [&]() -> void
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            const double th_3 = this_solution[3];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*((std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_3)*std::cos(th_0) + std::sin(th_0)*std::cos(th_3)) - r_21*((-std::sin(th_1)*std::sin(th_2) - std::cos(th_1)*std::cos(th_2))*std::sin(th_0)*std::sin(th_3) + std::cos(th_0)*std::cos(th_3)) - r_31*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::sin(th_3)) >= zero_tolerance || std::fabs(r_12*((std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_3)*std::cos(th_0) + std::sin(th_0)*std::cos(th_3)) - r_22*((-std::sin(th_1)*std::sin(th_2) - std::cos(th_1)*std::cos(th_2))*std::sin(th_0)*std::sin(th_3) + std::cos(th_0)*std::cos(th_3)) - r_32*(std::sin(th_1)*std::cos(th_2) - std::sin(th_2)*std::cos(th_1))*std::sin(th_3)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_2);
                const double x3 = std::sin(th_2);
                const double x4 = std::cos(th_1);
                const double x5 = x0*(x1*x2 - x3*x4);
                const double x6 = std::sin(th_0);
                const double x7 = std::cos(th_3);
                const double x8 = std::cos(th_0);
                const double x9 = x1*x3;
                const double x10 = x2*x4;
                const double x11 = x0*x8*(x10 + x9) + x6*x7;
                const double x12 = x0*x6*(-x10 - x9) + x7*x8;
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_11*x11 - r_21*x12 - r_31*x5, r_12*x11 - r_22*x12 - r_32*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_5_processor();
    // Finish code for explicit solution node 10
    
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

}; // struct yaskawa_HC10_ik

// Code below for debug
void test_ik_solve_yaskawa_HC10()
{
    std::array<double, yaskawa_HC10_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = yaskawa_HC10_ik::computeFK(theta);
    auto ik_output = yaskawa_HC10_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = yaskawa_HC10_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_yaskawa_HC10();
}
