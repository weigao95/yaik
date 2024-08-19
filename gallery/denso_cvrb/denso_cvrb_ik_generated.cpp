#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct denso_cvrb_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 128;
static constexpr int n_tree_nodes = 36;
static constexpr int intermediate_solution_size = 7;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_0 = 0.71;
static constexpr double d_1 = -0.05;
static constexpr double d_2 = 0.59;
static constexpr double d_3 = 0.1;
static constexpr double d_4 = 0.16;
static constexpr double pre_transform_special_symbol_23 = 0.22;

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
    ee_pose_raw(0, 3) = -d_1*x1 + d_2*x17 + d_3*x12 + d_4*x21 + x22*x4;
    ee_pose_raw(1, 0) = -x0*x24 + x13*x27;
    ee_pose_raw(1, 1) = -x0*x27 - x13*x24;
    ee_pose_raw(1, 2) = x28;
    ee_pose_raw(1, 3) = d_1*x4 + d_2*x25 + d_3*x24 + d_4*x28 + x1*x22;
    ee_pose_raw(2, 0) = x0*x30 + x13*x33;
    ee_pose_raw(2, 1) = -x0*x33 + x13*x30;
    ee_pose_raw(2, 2) = x34;
    ee_pose_raw(2, 3) = -a_0*x8 + d_2*x31 - d_3*x30 + d_4*x34;
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
    const double x30 = d_2*x25 + x29;
    const double x31 = a_0*x18 + d_1*x4;
    const double x32 = d_2*x19 + x31;
    const double x33 = -d_3*x27 + x30;
    const double x34 = d_3*x21 + x32;
    const double x35 = d_4*x28 + x33;
    const double x36 = d_4*x22 + x34;
    const double x37 = a_0*x8 - d_1*x0;
    const double x38 = d_2*x9 + x37;
    const double x39 = d_3*x13 + x38;
    const double x40 = d_4*x16 + x39;
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
    const double x18 = d_2*x13 + x6;
    const double x19 = a_0*x11*x14 + d_1*x2;
    const double x20 = d_2*x17 + x19;
    const double x21 = std::sin(th_3);
    const double x22 = -x15 - x16;
    const double x23 = x21*x22;
    const double x24 = std::cos(th_3);
    const double x25 = x12*x14 - x14*x8;
    const double x26 = -x2*x24 - x21*x25;
    const double x27 = -d_3*x23 + x18;
    const double x28 = d_3*x26 + x20;
    const double x29 = std::cos(th_4);
    const double x30 = std::sin(th_4);
    const double x31 = -x13*x29 - x22*x24*x30;
    const double x32 = -x17*x29 - x30*(-x2*x21 + x24*x25);
    const double x33 = d_4*x31 + x27;
    const double x34 = d_4*x32 + x28;
    const double x35 = 1.0*p_on_ee_x;
    const double x36 = 1.0*x14;
    const double x37 = p_on_ee_z*x36;
    const double x38 = x2*x4;
    const double x39 = x10*x2;
    const double x40 = -x38*x9 - x39*x7;
    const double x41 = a_0*x39 - d_1*x36;
    const double x42 = d_2*x40 + x41;
    const double x43 = -x38*x7 + x39*x9;
    const double x44 = -x21*x43 + x24*x36;
    const double x45 = d_3*x44 + x42;
    const double x46 = -x29*x40 - x30*(x21*x36 + x24*x43);
    const double x47 = d_4*x46 + x45;
    const double x48 = -x0*x14 - x1*x35;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = -pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = -x2*x6 + x3;
    jacobian(0, 3) = -p_on_ee_y*x13 + p_on_ee_z*x17 + x13*x20 - x17*x18;
    jacobian(0, 4) = p_on_ee_y*x23 + p_on_ee_z*x26 - x23*x28 - x26*x27;
    jacobian(0, 5) = -p_on_ee_y*x31 + p_on_ee_z*x32 + x31*x34 - x32*x33;
    jacobian(1, 0) = x35;
    jacobian(1, 1) = -pre_transform_special_symbol_23*x36 + x37;
    jacobian(1, 2) = -x36*x6 + x37;
    jacobian(1, 3) = p_on_ee_x*x13 - p_on_ee_z*x40 - x13*x42 + x18*x40;
    jacobian(1, 4) = -p_on_ee_x*x23 - p_on_ee_z*x44 + x23*x45 + x27*x44;
    jacobian(1, 5) = p_on_ee_x*x31 - p_on_ee_z*x46 - x31*x47 + x33*x46;
    jacobian(2, 1) = x48;
    jacobian(2, 2) = x19*x36 + x2*x41 + x48;
    jacobian(2, 3) = -p_on_ee_x*x17 + p_on_ee_y*x40 + x17*x42 - x20*x40;
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
        R_l(0, 0) = d_3*r_21;
        R_l(0, 1) = d_3*r_22;
        R_l(0, 2) = d_3*r_11;
        R_l(0, 3) = d_3*r_12;
        R_l(0, 4) = Py - d_4*r_23;
        R_l(0, 5) = Px - d_4*r_13;
        R_l(1, 0) = d_3*r_11;
        R_l(1, 1) = d_3*r_12;
        R_l(1, 2) = -d_3*r_21;
        R_l(1, 3) = -d_3*r_22;
        R_l(1, 4) = Px - d_4*r_13;
        R_l(1, 5) = -Py + d_4*r_23;
        R_l(2, 6) = d_3*r_31;
        R_l(2, 7) = d_3*r_32;
        R_l(3, 0) = -r_21;
        R_l(3, 1) = -r_22;
        R_l(3, 2) = -r_11;
        R_l(3, 3) = -r_12;
        R_l(4, 0) = -r_11;
        R_l(4, 1) = -r_12;
        R_l(4, 2) = r_21;
        R_l(4, 3) = r_22;
        R_l(5, 6) = 2*Px*d_3*r_11 + 2*Py*d_3*r_21 + 2*Pz*d_3*r_31 - 2*d_3*d_4*r_11*r_13 - 2*d_3*d_4*r_21*r_23 - 2*d_3*d_4*r_31*r_33;
        R_l(5, 7) = 2*Px*d_3*r_12 + 2*Py*d_3*r_22 + 2*Pz*d_3*r_32 - 2*d_3*d_4*r_12*r_13 - 2*d_3*d_4*r_22*r_23 - 2*d_3*d_4*r_32*r_33;
        R_l(6, 0) = -Px*r_31 + Pz*r_11 - d_4*r_11*r_33 + d_4*r_13*r_31;
        R_l(6, 1) = -Px*r_32 + Pz*r_12 - d_4*r_12*r_33 + d_4*r_13*r_32;
        R_l(6, 2) = Py*r_31 - Pz*r_21 + d_4*r_21*r_33 - d_4*r_23*r_31;
        R_l(6, 3) = Py*r_32 - Pz*r_22 + d_4*r_22*r_33 - d_4*r_23*r_32;
        R_l(7, 0) = Py*r_31 - Pz*r_21 + d_4*r_21*r_33 - d_4*r_23*r_31;
        R_l(7, 1) = Py*r_32 - Pz*r_22 + d_4*r_22*r_33 - d_4*r_23*r_32;
        R_l(7, 2) = Px*r_31 - Pz*r_11 + d_4*r_11*r_33 - d_4*r_13*r_31;
        R_l(7, 3) = Px*r_32 - Pz*r_12 + d_4*r_12*r_33 - d_4*r_13*r_32;
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
        const double x0 = R_l_inv_60*r_31 + R_l_inv_70*r_32;
        const double x1 = a_0*x0;
        const double x2 = -x1;
        const double x3 = R_l_inv_62*r_31 + R_l_inv_72*r_32;
        const double x4 = d_2*x3;
        const double x5 = -x4;
        const double x6 = x2 + x5;
        const double x7 = -d_1*(R_l_inv_61*r_31 + R_l_inv_71*r_32);
        const double x8 = R_l_inv_66*r_31;
        const double x9 = R_l_inv_76*r_32;
        const double x10 = x8 + x9;
        const double x11 = d_2*x10;
        const double x12 = d_4*r_33;
        const double x13 = Pz - x12;
        const double x14 = -x13*x3;
        const double x15 = R_l_inv_65*r_31 + R_l_inv_75*r_32;
        const double x16 = std::pow(Px, 2);
        const double x17 = std::pow(Py, 2);
        const double x18 = std::pow(Pz, 2);
        const double x19 = std::pow(d_1, 2);
        const double x20 = -x19;
        const double x21 = std::pow(d_3, 2);
        const double x22 = std::pow(r_11, 2);
        const double x23 = x21*x22;
        const double x24 = std::pow(r_21, 2);
        const double x25 = x21*x24;
        const double x26 = std::pow(r_31, 2);
        const double x27 = x21*x26;
        const double x28 = std::pow(d_4, 2);
        const double x29 = std::pow(r_13, 2)*x28;
        const double x30 = std::pow(r_23, 2)*x28;
        const double x31 = std::pow(r_33, 2)*x28;
        const double x32 = d_4*r_13;
        const double x33 = 2*Px;
        const double x34 = x32*x33;
        const double x35 = d_4*r_23;
        const double x36 = 2*Py;
        const double x37 = x35*x36;
        const double x38 = 2*Pz;
        const double x39 = x12*x38;
        const double x40 = std::pow(a_0, 2);
        const double x41 = std::pow(d_2, 2);
        const double x42 = -x40 - x41;
        const double x43 = x16 + x17 + x18 + x20 + x23 + x25 + x27 + x29 + x30 + x31 - x34 - x37 - x39 + x42;
        const double x44 = -x15*x43;
        const double x45 = -x11 + x14 + x44 + x7;
        const double x46 = R_l_inv_64*r_31;
        const double x47 = R_l_inv_74*r_32;
        const double x48 = -x46 - x47;
        const double x49 = x45 + x48;
        const double x50 = 2*a_0;
        const double x51 = x10*x50;
        const double x52 = -x51;
        const double x53 = -x3*x50;
        const double x54 = 2*d_2;
        const double x55 = x0*x54;
        const double x56 = x53 + x55;
        const double x57 = x1 + x4;
        const double x58 = x11 + x14 + x44 + x7;
        const double x59 = x48 + x58;
        const double x60 = 2*R_l_inv_63;
        const double x61 = r_31*x60;
        const double x62 = 2*R_l_inv_73;
        const double x63 = r_32*x62;
        const double x64 = R_l_inv_67*r_31 + R_l_inv_77*r_32;
        const double x65 = x54*x64;
        const double x66 = -x61 - x63 + x65;
        const double x67 = 4*d_1;
        const double x68 = x10*x67;
        const double x69 = x68 - 4;
        const double x70 = x61 + x63 + x65;
        const double x71 = x46 + x47;
        const double x72 = x58 + x71;
        const double x73 = x45 + x71;
        const double x74 = Px*r_11;
        const double x75 = Py*r_21;
        const double x76 = Pz*r_31;
        const double x77 = r_11*x32;
        const double x78 = r_21*x35;
        const double x79 = r_31*x12;
        const double x80 = x74 + x75 + x76 - x77 - x78 - x79;
        const double x81 = Px*r_12;
        const double x82 = Py*r_22;
        const double x83 = Pz*r_32;
        const double x84 = r_12*x32;
        const double x85 = r_22*x35;
        const double x86 = r_32*x12;
        const double x87 = x81 + x82 + x83 - x84 - x85 - x86;
        const double x88 = R_l_inv_60*x80 + R_l_inv_70*x87;
        const double x89 = a_0*x88;
        const double x90 = -x89;
        const double x91 = R_l_inv_62*x80 + R_l_inv_72*x87;
        const double x92 = d_2*x91;
        const double x93 = -x92;
        const double x94 = x90 + x93;
        const double x95 = d_3*x22;
        const double x96 = d_3*x24;
        const double x97 = d_3*x26;
        const double x98 = -d_1*(R_l_inv_61*x80 + R_l_inv_71*x87);
        const double x99 = R_l_inv_66*x80;
        const double x100 = R_l_inv_76*x87;
        const double x101 = x100 + x99;
        const double x102 = d_2*x101;
        const double x103 = -x13*x91;
        const double x104 = R_l_inv_65*x80 + R_l_inv_75*x87;
        const double x105 = -x104*x43;
        const double x106 = -x102 + x103 + x105 + x95 + x96 + x97 + x98;
        const double x107 = R_l_inv_64*x80;
        const double x108 = R_l_inv_74*x87;
        const double x109 = d_1 - x107 - x108;
        const double x110 = x106 + x109;
        const double x111 = x101*x50;
        const double x112 = -x111;
        const double x113 = -x50*x91;
        const double x114 = x54*x88;
        const double x115 = x113 + x114;
        const double x116 = x89 + x92;
        const double x117 = x102 + x103 + x105 + x95 + x96 + x97 + x98;
        const double x118 = x109 + x117;
        const double x119 = R_l_inv_67*x80 + R_l_inv_77*x87;
        const double x120 = x119*x54;
        const double x121 = x120 + x50;
        const double x122 = x60*x80;
        const double x123 = x62*x87;
        const double x124 = -x122 - x123;
        const double x125 = x101*x67;
        const double x126 = x122 + x123;
        const double x127 = -d_1 + x107 + x108;
        const double x128 = x117 + x127;
        const double x129 = x106 + x127;
        const double x130 = Px*r_21;
        const double x131 = Py*r_11;
        const double x132 = r_11*x35 - r_21*x32 + x130 - x131;
        const double x133 = R_l_inv_64*x132;
        const double x134 = Px*r_22;
        const double x135 = Py*r_12;
        const double x136 = r_12*x35 - r_22*x32 + x134 - x135;
        const double x137 = R_l_inv_74*x136;
        const double x138 = d_1*(R_l_inv_61*x132 + R_l_inv_71*x136);
        const double x139 = R_l_inv_62*x132;
        const double x140 = R_l_inv_72*x136;
        const double x141 = x139 + x140;
        const double x142 = x13*x141;
        const double x143 = R_l_inv_65*x132 + R_l_inv_75*x136;
        const double x144 = x143*x43;
        const double x145 = x133 + x137 + x138 + x142 + x144;
        const double x146 = R_l_inv_60*x132;
        const double x147 = R_l_inv_70*x136;
        const double x148 = x146 + x147;
        const double x149 = a_0*x148;
        const double x150 = d_2*x141;
        const double x151 = x149 + x150;
        const double x152 = R_l_inv_66*x132 + R_l_inv_76*x136;
        const double x153 = d_2*x152;
        const double x154 = a_0 + x153;
        const double x155 = -x54;
        const double x156 = x152*x50;
        const double x157 = x148*x54;
        const double x158 = x141*x50;
        const double x159 = -x157 + x158;
        const double x160 = -a_0;
        const double x161 = -x153;
        const double x162 = x160 + x161;
        const double x163 = -x149;
        const double x164 = -x150;
        const double x165 = x163 + x164;
        const double x166 = 2*d_1;
        const double x167 = -x166;
        const double x168 = x132*x60;
        const double x169 = x136*x62;
        const double x170 = R_l_inv_67*x132 + R_l_inv_77*x136;
        const double x171 = -x170*x54;
        const double x172 = x167 + x168 + x169 + x171;
        const double x173 = x152*x67;
        const double x174 = -x173;
        const double x175 = x166 - x168 - x169 + x171;
        const double x176 = -x133 - x137 + x138 + x142 + x144;
        const double x177 = -x156;
        const double x178 = Py*x22 + Py*x24 + Py*x26 - x22*x35 - x24*x35 - x26*x35;
        const double x179 = 2*d_3;
        const double x180 = x178*x179;
        const double x181 = R_l_inv_40*x180;
        const double x182 = Px*x22 + Px*x24 + Px*x26 - x22*x32 - x24*x32 - x26*x32;
        const double x183 = x179*x182;
        const double x184 = R_l_inv_50*x183;
        const double x185 = 2*x35;
        const double x186 = 2*x32;
        const double x187 = 2*r_11;
        const double x188 = r_23*x28;
        const double x189 = r_13*x188;
        const double x190 = 2*r_31;
        const double x191 = r_33*x188;
        const double x192 = std::pow(r_21, 3)*x21 - r_21*x16 + r_21*x17 - r_21*x18 + r_21*x23 + r_21*x27 - r_21*x29 + r_21*x30 - r_21*x31 + r_21*x39 + x130*x186 - x131*x186 - x185*x74 - x185*x75 - x185*x76 + x187*x189 + x190*x191 + x36*x74 + x36*x76 - x36*x79;
        const double x193 = R_l_inv_00*x192;
        const double x194 = 2*x189;
        const double x195 = r_13*r_33*x28;
        const double x196 = std::pow(r_11, 3)*x21 + r_11*x16 - r_11*x17 - r_11*x18 + r_11*x25 + r_11*x27 + r_11*x29 - r_11*x30 - r_11*x31 + r_11*x39 + r_21*x194 - x130*x185 + x131*x185 - x186*x74 - x186*x75 - x186*x76 + x190*x195 + x33*x75 + x33*x76 - x33*x79;
        const double x197 = R_l_inv_20*x196;
        const double x198 = r_21*x21;
        const double x199 = x187*x198;
        const double x200 = x190*x198;
        const double x201 = 2*r_32;
        const double x202 = r_12*x194 + r_12*x199 - r_22*x16 + r_22*x17 - r_22*x18 + r_22*x23 + 3*r_22*x25 + r_22*x27 - r_22*x29 + r_22*x30 - r_22*x31 + r_22*x39 + r_32*x200 + x134*x186 - x135*x186 - x185*x81 - x185*x82 - x185*x83 + x191*x201 + x36*x81 + x36*x83 - x36*x86;
        const double x203 = R_l_inv_10*x202;
        const double x204 = r_31*x187*x21;
        const double x205 = r_12*x16 - r_12*x17 - r_12*x18 + 3*r_12*x23 + r_12*x25 + r_12*x27 + r_12*x29 - r_12*x30 - r_12*x31 + r_12*x39 + r_22*x194 + r_22*x199 + r_32*x204 - x134*x185 + x135*x185 - x186*x81 - x186*x82 - x186*x83 + x195*x201 + x33*x82 + x33*x83 - x33*x86;
        const double x206 = R_l_inv_30*x205;
        const double x207 = x181 + x184 + x193 + x197 + x203 + x206;
        const double x208 = a_0*x207;
        const double x209 = d_1*(R_l_inv_01*x192 + R_l_inv_11*x202 + R_l_inv_21*x196 + R_l_inv_31*x205 + R_l_inv_41*x180 + R_l_inv_51*x183);
        const double x210 = R_l_inv_42*x180;
        const double x211 = R_l_inv_52*x183;
        const double x212 = R_l_inv_02*x192;
        const double x213 = R_l_inv_22*x196;
        const double x214 = R_l_inv_12*x202;
        const double x215 = R_l_inv_32*x205;
        const double x216 = x210 + x211 + x212 + x213 + x214 + x215;
        const double x217 = d_2*x216;
        const double x218 = x13*x216;
        const double x219 = R_l_inv_05*x192 + R_l_inv_15*x202 + R_l_inv_25*x196 + R_l_inv_35*x205 + R_l_inv_45*x180 + R_l_inv_55*x183;
        const double x220 = x219*x43;
        const double x221 = x208 + x209 + x217 + x218 + x220;
        const double x222 = R_l_inv_06*x192 + R_l_inv_16*x202 + R_l_inv_26*x196 + R_l_inv_36*x205 + R_l_inv_46*x180 + R_l_inv_56*x183;
        const double x223 = d_2*x222;
        const double x224 = d_1*x50;
        const double x225 = x223 + x224;
        const double x226 = R_l_inv_04*x192;
        const double x227 = R_l_inv_14*x202;
        const double x228 = R_l_inv_24*x196;
        const double x229 = R_l_inv_34*x205;
        const double x230 = R_l_inv_44*x180;
        const double x231 = R_l_inv_54*x183;
        const double x232 = x226 + x227 + x228 + x229 + x230 + x231;
        const double x233 = d_2*x67;
        const double x234 = -x233;
        const double x235 = x222*x50;
        const double x236 = x207*x54;
        const double x237 = x216*x50;
        const double x238 = -x236 + x237;
        const double x239 = -x223;
        const double x240 = -x224;
        const double x241 = x239 + x240;
        const double x242 = -x208;
        const double x243 = -x217;
        const double x244 = x209 + x218 + x220 + x242 + x243;
        const double x245 = 2*x40;
        const double x246 = 2*x19;
        const double x247 = 2*x41;
        const double x248 = 4*d_3;
        const double x249 = x178*x248;
        const double x250 = R_l_inv_43*x249;
        const double x251 = x182*x248;
        const double x252 = R_l_inv_53*x251;
        const double x253 = 2*x192;
        const double x254 = R_l_inv_03*x253;
        const double x255 = 2*x196;
        const double x256 = R_l_inv_23*x255;
        const double x257 = 2*x202;
        const double x258 = R_l_inv_13*x257;
        const double x259 = 2*x205;
        const double x260 = R_l_inv_33*x259;
        const double x261 = R_l_inv_07*x192 + R_l_inv_17*x202 + R_l_inv_27*x196 + R_l_inv_37*x205 + R_l_inv_47*x180 + R_l_inv_57*x183;
        const double x262 = -x261*x54;
        const double x263 = x245 - x246 - x247 + x250 + x252 + x254 + x256 + x258 + x260 + x262;
        const double x264 = a_0*d_2;
        const double x265 = -8*x264;
        const double x266 = x222*x67;
        const double x267 = -x266;
        const double x268 = -x245 + x246 + x247 - x250 - x252 - x254 - x256 - x258 - x260 + x262;
        const double x269 = -x226 - x227 - x228 - x229 - x230 - x231;
        const double x270 = -x235;
        const double x271 = R_l_inv_40*x183;
        const double x272 = R_l_inv_50*x180;
        const double x273 = R_l_inv_00*x196;
        const double x274 = R_l_inv_20*x192;
        const double x275 = R_l_inv_10*x205;
        const double x276 = R_l_inv_30*x202;
        const double x277 = x271 - x272 + x273 - x274 + x275 - x276;
        const double x278 = a_0*x277;
        const double x279 = R_l_inv_42*x183;
        const double x280 = R_l_inv_52*x180;
        const double x281 = R_l_inv_02*x196;
        const double x282 = R_l_inv_22*x192;
        const double x283 = R_l_inv_12*x205;
        const double x284 = R_l_inv_32*x202;
        const double x285 = x279 - x280 + x281 - x282 + x283 - x284;
        const double x286 = d_2*x285;
        const double x287 = x278 + x286;
        const double x288 = d_1*(R_l_inv_01*x196 + R_l_inv_11*x205 - R_l_inv_21*x192 - R_l_inv_31*x202 + R_l_inv_41*x183 - R_l_inv_51*x180);
        const double x289 = R_l_inv_46*x183;
        const double x290 = R_l_inv_56*x180;
        const double x291 = R_l_inv_06*x196;
        const double x292 = R_l_inv_26*x192;
        const double x293 = R_l_inv_16*x205;
        const double x294 = R_l_inv_36*x202;
        const double x295 = x289 - x290 + x291 - x292 + x293 - x294;
        const double x296 = d_2*x295;
        const double x297 = x13*x285;
        const double x298 = R_l_inv_05*x196 + R_l_inv_15*x205 - R_l_inv_25*x192 - R_l_inv_35*x202 + R_l_inv_45*x183 - R_l_inv_55*x180;
        const double x299 = x298*x43;
        const double x300 = x288 + x296 + x297 + x299;
        const double x301 = R_l_inv_04*x196;
        const double x302 = R_l_inv_14*x205;
        const double x303 = R_l_inv_24*x192;
        const double x304 = R_l_inv_34*x202;
        const double x305 = R_l_inv_54*x180;
        const double x306 = R_l_inv_44*x183;
        const double x307 = x19 + x301 + x302 - x303 - x304 - x305 + x306 + x42;
        const double x308 = x300 + x307;
        const double x309 = x295*x50;
        const double x310 = x277*x54;
        const double x311 = x285*x50;
        const double x312 = -x310 + x311;
        const double x313 = -x278;
        const double x314 = -x286;
        const double x315 = x313 + x314;
        const double x316 = x288 - x296 + x297 + x299;
        const double x317 = x307 + x316;
        const double x318 = a_0*x67;
        const double x319 = R_l_inv_07*x196 + R_l_inv_17*x205 - R_l_inv_27*x192 - R_l_inv_37*x202 + R_l_inv_47*x183 - R_l_inv_57*x180;
        const double x320 = -x319*x54;
        const double x321 = x318 + x320;
        const double x322 = R_l_inv_23*x253;
        const double x323 = R_l_inv_33*x257;
        const double x324 = R_l_inv_03*x255;
        const double x325 = R_l_inv_13*x259;
        const double x326 = R_l_inv_53*x249;
        const double x327 = R_l_inv_43*x251;
        const double x328 = -x322 - x323 + x324 + x325 - x326 + x327;
        const double x329 = x295*x67;
        const double x330 = -x329;
        const double x331 = x322 + x323 - x324 - x325 + x326 - x327;
        const double x332 = x20 - x301 - x302 + x303 + x304 + x305 - x306 + x40 + x41;
        const double x333 = x316 + x332;
        const double x334 = -x309;
        const double x335 = x300 + x332;
        const double x336 = 2*x12;
        const double x337 = 2*x191;
        const double x338 = r_21*x337 + std::pow(r_31, 3)*x21 - r_31*x16 - r_31*x17 + r_31*x18 + r_31*x23 + r_31*x25 - r_31*x29 - r_31*x30 + r_31*x31 + r_31*x34 + r_31*x37 + x187*x195 - x336*x74 - x336*x75 - x336*x76 + x38*x74 + x38*x75 - x38*x77 - x38*x78;
        const double x339 = R_l_inv_60*x338;
        const double x340 = 2*r_12*x195 + r_12*x204 + r_22*x200 + r_22*x337 - r_32*x16 - r_32*x17 + r_32*x18 + r_32*x23 + r_32*x25 + 3*r_32*x27 - r_32*x29 - r_32*x30 + r_32*x31 + r_32*x34 + r_32*x37 - x336*x81 - x336*x82 - x336*x83 + x38*x81 + x38*x82 - x38*x84 - x38*x85;
        const double x341 = R_l_inv_70*x340;
        const double x342 = x339 + x341;
        const double x343 = a_0*x342;
        const double x344 = d_1*(R_l_inv_61*x338 + R_l_inv_71*x340);
        const double x345 = R_l_inv_62*x338;
        const double x346 = R_l_inv_72*x340;
        const double x347 = x345 + x346;
        const double x348 = d_2*x347;
        const double x349 = x13*x347;
        const double x350 = R_l_inv_65*x338 + R_l_inv_75*x340;
        const double x351 = x350*x43;
        const double x352 = -x38*x95;
        const double x353 = -x38*x96;
        const double x354 = -x38*x97;
        const double x355 = x336*x95;
        const double x356 = x336*x96;
        const double x357 = x336*x97;
        const double x358 = x344 + x348 + x349 + x351 + x352 + x353 + x354 + x355 + x356 + x357;
        const double x359 = x343 + x358;
        const double x360 = R_l_inv_64*x338;
        const double x361 = R_l_inv_74*x340;
        const double x362 = x360 + x361;
        const double x363 = R_l_inv_66*x338 + R_l_inv_76*x340;
        const double x364 = d_2*x363;
        const double x365 = d_1*x54;
        const double x366 = x364 + x365;
        const double x367 = x362 + x366;
        const double x368 = x342*x54;
        const double x369 = x347*x50;
        const double x370 = -x368 + x369;
        const double x371 = x363*x50;
        const double x372 = x318 + x371;
        const double x373 = -x343;
        const double x374 = -x364 - x365;
        const double x375 = x373 + x374;
        const double x376 = x344 - x348 + x349 + x351 + x352 + x353 + x354 + x355 + x356 + x357;
        const double x377 = x362 + x376;
        const double x378 = R_l_inv_67*x338 + R_l_inv_77*x340;
        const double x379 = -x378*x54;
        const double x380 = 4*x264;
        const double x381 = x379 + x380;
        const double x382 = x338*x60;
        const double x383 = x340*x62;
        const double x384 = x382 + x383;
        const double x385 = 4*x40;
        const double x386 = 4*x19;
        const double x387 = 4*x41;
        const double x388 = x363*x67;
        const double x389 = -x386 - x387 - x388;
        const double x390 = -x382 - x383;
        const double x391 = x379 - x380;
        const double x392 = -x360 - x361;
        const double x393 = -x318;
        const double x394 = -x371 + x393;
        const double x395 = x366 + x376 + x392;
        const double x396 = x15*x50;
        const double x397 = x54*(x0 - x396);
        const double x398 = 4*d_2;
        const double x399 = -x54*(x0 + x396);
        const double x400 = 4*a_0;
        const double x401 = -x400*x64;
        const double x402 = 8*R_l_inv_63;
        const double x403 = 8*R_l_inv_73;
        const double x404 = 4 - x68;
        const double x405 = x104*x50;
        const double x406 = x54*(-x405 + x88);
        const double x407 = -x54*(x405 + x88);
        const double x408 = -x119*x400;
        const double x409 = -x125;
        const double x410 = x143*x50;
        const double x411 = x410 - 1;
        const double x412 = -x146 - x147;
        const double x413 = x410 + 1;
        const double x414 = x170*x400;
        const double x415 = x219*x50;
        const double x416 = x167 + x415;
        const double x417 = -x181 - x184 - x193 - x197 - x203 - x206;
        const double x418 = x166 + x415;
        const double x419 = x261*x400;
        const double x420 = 16*d_3;
        const double x421 = R_l_inv_43*x420;
        const double x422 = R_l_inv_53*x420;
        const double x423 = 8*R_l_inv_03;
        const double x424 = 8*R_l_inv_23;
        const double x425 = 8*R_l_inv_13;
        const double x426 = 8*R_l_inv_33;
        const double x427 = x298*x50;
        const double x428 = x427 + x50;
        const double x429 = -x271 + x272 - x273 + x274 - x275 + x276;
        const double x430 = -x279 + x280 - x281 + x282 - x283 + x284;
        const double x431 = x319*x400;
        const double x432 = -x50;
        const double x433 = x427 + x432;
        const double x434 = x350*x50;
        const double x435 = x54*(-x339 - x341 + x434);
        const double x436 = x166 + x363;
        const double x437 = x54*(x342 + x434);
        const double x438 = -x385;
        const double x439 = x378*x400;
        const double x440 = x386 + x387 + x388;
        const double x441 = x2 + x4;
        const double x442 = x53 - x55;
        const double x443 = x1 + x5;
        const double x444 = x90 + x92;
        const double x445 = x113 - x114;
        const double x446 = x89 + x93;
        const double x447 = x120 + x432;
        const double x448 = x149 + x164;
        const double x449 = a_0 + x161;
        const double x450 = x157 + x158;
        const double x451 = x153 + x160;
        const double x452 = x150 + x163;
        const double x453 = x209 + x218 + x220 + x232;
        const double x454 = x208 + x243;
        const double x455 = x224 + x239;
        const double x456 = x236 + x237;
        const double x457 = x223 + x240;
        const double x458 = x217 + x242;
        const double x459 = x209 + x218 + x220 + x269;
        const double x460 = x278 + x314;
        const double x461 = x310 + x311;
        const double x462 = x286 + x313;
        const double x463 = x320 + x393;
        const double x464 = x368 + x369;
        
        Eigen::Matrix<double, 6, 9> A;
        A.setZero();
        A(0, 0) = x49 + x6;
        A(0, 1) = x52 + x56;
        A(0, 2) = x57 + x59;
        A(0, 3) = x66;
        A(0, 4) = x69;
        A(0, 5) = x70;
        A(0, 6) = x6 + x72;
        A(0, 7) = x51 + x56;
        A(0, 8) = x57 + x73;
        A(1, 0) = x110 + x94;
        A(1, 1) = x112 + x115;
        A(1, 2) = x116 + x118;
        A(1, 3) = x121 + x124;
        A(1, 4) = x125;
        A(1, 5) = x121 + x126;
        A(1, 6) = x128 + x94;
        A(1, 7) = x111 + x115;
        A(1, 8) = x116 + x129;
        A(2, 0) = x145 + x151 + x154;
        A(2, 1) = x155 + x156 + x159;
        A(2, 2) = x145 + x162 + x165;
        A(2, 3) = x172;
        A(2, 4) = x174;
        A(2, 5) = x175;
        A(2, 6) = x151 + x162 + x176;
        A(2, 7) = x159 + x177 + x54;
        A(2, 8) = x154 + x165 + x176;
        A(3, 0) = x221 + x225 + x232;
        A(3, 1) = x234 + x235 + x238;
        A(3, 2) = x232 + x241 + x244;
        A(3, 3) = x263;
        A(3, 4) = x265 + x267;
        A(3, 5) = x268;
        A(3, 6) = x221 + x241 + x269;
        A(3, 7) = x233 + x238 + x270;
        A(3, 8) = x225 + x244 + x269;
        A(4, 0) = x287 + x308;
        A(4, 1) = x309 + x312;
        A(4, 2) = x315 + x317;
        A(4, 3) = x321 + x328;
        A(4, 4) = x330;
        A(4, 5) = x321 + x331;
        A(4, 6) = x287 + x333;
        A(4, 7) = x312 + x334;
        A(4, 8) = x315 + x335;
        A(5, 0) = x359 + x367;
        A(5, 1) = x370 + x372;
        A(5, 2) = x375 + x377;
        A(5, 3) = x381 + x384;
        A(5, 4) = x385 + x389;
        A(5, 5) = x390 + x391;
        A(5, 6) = x359 + x374 + x392;
        A(5, 7) = x370 + x394;
        A(5, 8) = x373 + x395;
        
        Eigen::Matrix<double, 6, 9> B;
        B.setZero();
        B(0, 0) = x397;
        B(0, 1) = x398*(x10 + x3);
        B(0, 2) = x399;
        B(0, 3) = x401 + x69;
        B(0, 4) = r_31*x402 + r_32*x403;
        B(0, 5) = x401 + x404;
        B(0, 6) = x397;
        B(0, 7) = x398*(x3 - x8 - x9);
        B(0, 8) = x399;
        B(1, 0) = x406;
        B(1, 1) = x398*(x101 + x91);
        B(1, 2) = x407;
        B(1, 3) = x125 + x408;
        B(1, 4) = x402*x80 + x403*x87;
        B(1, 5) = x408 + x409;
        B(1, 6) = x406;
        B(1, 7) = x398*(-x100 + x91 - x99);
        B(1, 8) = x407;
        B(2, 0) = x54*(x411 + x412);
        B(2, 1) = -x398*(x141 + x152);
        B(2, 2) = x54*(x148 + x413);
        B(2, 3) = x174 + x414;
        B(2, 4) = 8*d_1 - x132*x402 - x136*x403;
        B(2, 5) = x173 + x414;
        B(2, 6) = x54*(x412 + x413);
        B(2, 7) = x398*(-x139 - x140 + x152);
        B(2, 8) = x54*(x148 + x411);
        B(3, 0) = x54*(x416 + x417);
        B(3, 1) = -x398*(x216 + x222);
        B(3, 2) = x54*(x207 + x418);
        B(3, 3) = x267 + x419;
        B(3, 4) = -x178*x421 - x182*x422 + 8*x19 - x192*x423 - x196*x424 - x202*x425 - x205*x426 + 8*x40 + 8*x41;
        B(3, 5) = x266 + x419;
        B(3, 6) = x54*(x417 + x418);
        B(3, 7) = x398*(-x210 - x211 - x212 - x213 - x214 - x215 + x222);
        B(3, 8) = x54*(x207 + x416);
        B(4, 0) = x54*(x428 + x429);
        B(4, 1) = x398*(-x289 + x290 - x291 + x292 - x293 + x294 + x430);
        B(4, 2) = x54*(x277 + x428);
        B(4, 3) = x330 + x431;
        B(4, 4) = x178*x422 - x182*x421 + x192*x424 - x196*x423 + x202*x426 - x205*x425;
        B(4, 5) = x329 + x431;
        B(4, 6) = x54*(x429 + x433);
        B(4, 7) = x398*(x295 + x430);
        B(4, 8) = x54*(x277 + x433);
        B(5, 0) = x435;
        B(5, 1) = -x398*(x347 + x436);
        B(5, 2) = x437;
        B(5, 3) = x389 + x438 + x439;
        B(5, 4) = -x338*x402 - x340*x403;
        B(5, 5) = x385 + x439 + x440;
        B(5, 6) = x435;
        B(5, 7) = x398*(-x345 - x346 + x436);
        B(5, 8) = x437;
        
        Eigen::Matrix<double, 6, 9> C;
        C.setZero();
        C(0, 0) = x441 + x59;
        C(0, 1) = x442 + x52;
        C(0, 2) = x443 + x49;
        C(0, 3) = x70;
        C(0, 4) = x404;
        C(0, 5) = x66;
        C(0, 6) = x441 + x73;
        C(0, 7) = x442 + x51;
        C(0, 8) = x443 + x72;
        C(1, 0) = x118 + x444;
        C(1, 1) = x112 + x445;
        C(1, 2) = x110 + x446;
        C(1, 3) = x126 + x447;
        C(1, 4) = x409;
        C(1, 5) = x124 + x447;
        C(1, 6) = x129 + x444;
        C(1, 7) = x111 + x445;
        C(1, 8) = x128 + x446;
        C(2, 0) = x145 + x448 + x449;
        C(2, 1) = x156 + x450 + x54;
        C(2, 2) = x145 + x451 + x452;
        C(2, 3) = x175;
        C(2, 4) = x173;
        C(2, 5) = x172;
        C(2, 6) = x176 + x448 + x451;
        C(2, 7) = x155 + x177 + x450;
        C(2, 8) = x176 + x449 + x452;
        C(3, 0) = x453 + x454 + x455;
        C(3, 1) = x233 + x235 + x456;
        C(3, 2) = x453 + x457 + x458;
        C(3, 3) = x268;
        C(3, 4) = x265 + x266;
        C(3, 5) = x263;
        C(3, 6) = x454 + x457 + x459;
        C(3, 7) = x234 + x270 + x456;
        C(3, 8) = x455 + x458 + x459;
        C(4, 0) = x317 + x460;
        C(4, 1) = x309 + x461;
        C(4, 2) = x308 + x462;
        C(4, 3) = x331 + x463;
        C(4, 4) = x329;
        C(4, 5) = x328 + x463;
        C(4, 6) = x335 + x460;
        C(4, 7) = x334 + x461;
        C(4, 8) = x333 + x462;
        C(5, 0) = x343 + x374 + x377;
        C(5, 1) = x372 + x464;
        C(5, 2) = x358 + x367 + x373;
        C(5, 3) = x381 + x390;
        C(5, 4) = x438 + x440;
        C(5, 5) = x384 + x391;
        C(5, 6) = x343 + x395;
        C(5, 7) = x394 + x464;
        C(5, 8) = x358 + x375 + x392;
        
        // Invoke the solver
        std::array<double, 16> solution_buffer;
        int n_solutions = yaik_cpp::general_6dof_internal::computeSolutionFromTanhalfLME(A, B, C, &solution_buffer);
        
        for(auto i = 0; i < n_solutions; i++)
        {
            auto solution_i = make_raw_solution();
            solution_i[3] = solution_buffer[i];
            int appended_idx = append_solution_to_queue(solution_i);
            add_input_index_to(2, appended_idx);
        };
    };
    // Invoke the processor
    General6DoFNumericalReduceSolutionNode_node_1_solve_th_2_processor();
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
            const double th_2 = this_solution[3];
            
            const bool condition_0 = 2*std::fabs(d_1*d_3) >= zero_tolerance || 2*std::fabs(a_0*d_3*std::cos(th_2)) >= zero_tolerance || std::fabs(std::pow(Px, 2) - 2*Px*d_4*r_13 + std::pow(Py, 2) - 2*Py*d_4*r_23 + std::pow(Pz, 2) - 2*Pz*d_4*r_33 - std::pow(a_0, 2) + 2*a_0*d_2*std::sin(th_2) - std::pow(d_1, 2) - std::pow(d_2, 2) - std::pow(d_3, 2) + std::pow(d_4, 2)*std::pow(r_13, 2) + std::pow(d_4, 2)*std::pow(r_23, 2) + std::pow(d_4, 2)*std::pow(r_33, 2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_2);
                const double x1 = 2*d_3;
                const double x2 = std::atan2(-a_0*x0*x1, -d_1*x1);
                const double x3 = std::pow(d_1, 2);
                const double x4 = std::pow(d_3, 2);
                const double x5 = 4*x4;
                const double x6 = std::pow(a_0, 2);
                const double x7 = 2*d_4;
                const double x8 = std::pow(d_4, 2);
                const double x9 = std::pow(Px, 2) - Px*r_13*x7 + std::pow(Py, 2) - Py*r_23*x7 + std::pow(Pz, 2) - Pz*r_33*x7 + 2*a_0*d_2*std::sin(th_2) - std::pow(d_2, 2) + std::pow(r_13, 2)*x8 + std::pow(r_23, 2)*x8 + std::pow(r_33, 2)*x8 - x3 - x4 - x6;
                const double x10 = std::sqrt(std::pow(x0, 2)*x5*x6 + x3*x5 - std::pow(x9, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x2 + std::atan2(x10, x9);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = 2*std::fabs(d_1*d_3) >= zero_tolerance || 2*std::fabs(a_0*d_3*std::cos(th_2)) >= zero_tolerance || std::fabs(std::pow(Px, 2) - 2*Px*d_4*r_13 + std::pow(Py, 2) - 2*Py*d_4*r_23 + std::pow(Pz, 2) - 2*Pz*d_4*r_33 - std::pow(a_0, 2) + 2*a_0*d_2*std::sin(th_2) - std::pow(d_1, 2) - std::pow(d_2, 2) - std::pow(d_3, 2) + std::pow(d_4, 2)*std::pow(r_13, 2) + std::pow(d_4, 2)*std::pow(r_23, 2) + std::pow(d_4, 2)*std::pow(r_33, 2)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_2);
                const double x1 = 2*d_3;
                const double x2 = std::atan2(-a_0*x0*x1, -d_1*x1);
                const double x3 = std::pow(d_1, 2);
                const double x4 = std::pow(d_3, 2);
                const double x5 = 4*x4;
                const double x6 = std::pow(a_0, 2);
                const double x7 = 2*d_4;
                const double x8 = std::pow(d_4, 2);
                const double x9 = std::pow(Px, 2) - Px*r_13*x7 + std::pow(Py, 2) - Py*r_23*x7 + std::pow(Pz, 2) - Pz*r_33*x7 + 2*a_0*d_2*std::sin(th_2) - std::pow(d_2, 2) + std::pow(r_13, 2)*x8 + std::pow(r_23, 2)*x8 + std::pow(r_33, 2)*x8 - x3 - x4 - x6;
                const double x10 = std::sqrt(std::pow(x0, 2)*x5*x6 + x3*x5 - std::pow(x9, 2));
                // End of temp variables
                const double tmp_sol_value = x2 + std::atan2(-x10, x9);
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
    ExplicitSolutionNode_node_3_solve_th_3_processor();
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
            const bool checked_result = std::fabs(Px - d_4*r_13) <= 9.9999999999999995e-7 && std::fabs(Py - d_4*r_23) <= 9.9999999999999995e-7 && std::fabs(d_1 - d_3*std::cos(th_3)) <= 9.9999999999999995e-7;
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
            
            const bool condition_0 = std::fabs(Px - d_4*r_13) >= zero_tolerance || std::fabs(Py - d_4*r_23) >= zero_tolerance || std::fabs(d_1 - d_3*std::cos(th_3)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Px - d_4*r_13;
                const double x1 = -Py + d_4*r_23;
                const double x2 = std::atan2(x0, x1);
                const double x3 = -d_1 + d_3*std::cos(th_3);
                const double x4 = std::sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[0] = x2 + std::atan2(x4, x3);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(6, appended_idx);
            }
            
            const bool condition_1 = std::fabs(Px - d_4*r_13) >= zero_tolerance || std::fabs(Py - d_4*r_23) >= zero_tolerance || std::fabs(d_1 - d_3*std::cos(th_3)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = Px - d_4*r_13;
                const double x1 = -Py + d_4*r_23;
                const double x2 = std::atan2(x0, x1);
                const double x3 = -d_1 + d_3*std::cos(th_3);
                const double x4 = std::sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
                // End of temp variables
                const double tmp_sol_value = x2 + std::atan2(-x4, x3);
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
            const double th_3 = this_solution[4];
            
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
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
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
    
    // Code for non-branch dispatcher node 18
    // Actually, there is no code
    
    // Code for explicit solution node 19, solved variable is th_5
    auto ExplicitSolutionNode_node_19_solve_th_5_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1);
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
    ExplicitSolutionNode_node_19_solve_th_5_processor();
    // Finish code for explicit solution node 18
    
    // Code for solved_variable dispatcher node 20
    auto SolvedVariableDispatcherNode_node_20_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(20);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(20);
        if (!this_input_valid)
            return;
        
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_2 = this_solution[3];
            
            const bool degenerate_valid_0 = std::fabs(th_2 - 1.0/2.0*M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(30, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_2 + M_PI_2) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(33, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(21, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_20_processor();
    // Finish code for solved_variable dispatcher node 20
    
    // Code for explicit solution node 33, solved variable is th_1
    auto ExplicitSolutionNode_node_33_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(33);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(33);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 33
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_3 = this_solution[4];
            
            const bool condition_0 = std::fabs(d_3*std::sin(th_3)) >= 9.9999999999999995e-7 || std::fabs(a_0 + d_2) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_0 + d_2;
                const double x1 = Pz - d_4*r_33;
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3;
                const double x5 = d_3*std::sin(th_3);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x1 - x4*x5, x0*x4 - x1*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(34, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_33_solve_th_1_processor();
    // Finish code for explicit solution node 33
    
    // Code for non-branch dispatcher node 34
    // Actually, there is no code
    
    // Code for explicit solution node 35, solved variable is th_4
    auto ExplicitSolutionNode_node_35_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(34);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(34);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 35
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_13*(std::sin(th_0)*std::sin(th_3) + std::sin(th_1)*std::cos(th_0)*std::cos(th_3)) + r_23*(std::sin(th_0)*std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) + r_33*std::cos(th_1)*std::cos(th_3)) >= zero_tolerance || std::fabs(r_13*std::cos(th_0)*std::cos(th_1) + r_23*std::sin(th_0)*std::cos(th_1) - r_33*std::sin(th_1)) >= zero_tolerance;
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
                const double tmp_sol_value = std::atan2(-r_13*(x2*x3 + x4*x6) - r_23*(x2*x6 - x3*x4) - r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_35_solve_th_4_processor();
    // Finish code for explicit solution node 34
    
    // Code for explicit solution node 30, solved variable is th_1
    auto ExplicitSolutionNode_node_30_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(30);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(30);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 30
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_0 = this_solution[0];
            const double th_3 = this_solution[4];
            
            const bool condition_0 = std::fabs(d_3*std::sin(th_3)) >= 9.9999999999999995e-7 || std::fabs(a_0 - d_2) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_0 - d_2;
                const double x1 = Pz - d_4*r_33;
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3;
                const double x5 = d_3*std::sin(th_3);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(31, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_30_solve_th_1_processor();
    // Finish code for explicit solution node 30
    
    // Code for non-branch dispatcher node 31
    // Actually, there is no code
    
    // Code for explicit solution node 32, solved variable is th_4
    auto ExplicitSolutionNode_node_32_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(31);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(31);
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
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(-r_13*(std::sin(th_0)*std::sin(th_3) - std::sin(th_1)*std::cos(th_0)*std::cos(th_3)) + r_23*(std::sin(th_0)*std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_0)) + r_33*std::cos(th_1)*std::cos(th_3)) >= zero_tolerance || std::fabs(r_13*std::cos(th_0)*std::cos(th_1) + r_23*std::sin(th_0)*std::cos(th_1) - r_33*std::sin(th_1)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::cos(th_3);
                const double x2 = std::sin(th_3);
                const double x3 = std::cos(th_0);
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_1);
                const double x6 = x1*x5;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_13*(x2*x4 - x3*x6) + r_23*(x2*x3 + x4*x6) + r_33*x0*x1, r_13*x0*x3 + r_23*x0*x4 - r_33*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_32_solve_th_4_processor();
    // Finish code for explicit solution node 31
    
    // Code for explicit solution node 21, solved variable is th_1
    auto ExplicitSolutionNode_node_21_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(21);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(21);
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
            const double th_1th_2th_4_soa = this_solution[2];
            const double th_2 = this_solution[3];
            
            const bool condition_0 = std::fabs(d_2*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_0 - d_2*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Pz - d_4*r_33;
                const double x1 = a_0 - d_2*std::sin(th_2);
                const double x2 = Px*std::cos(th_0) + Py*std::sin(th_0) - d_4*std::sin(th_1th_2th_4_soa);
                const double x3 = d_2*std::cos(th_2);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x1 - x2*x3, -x0*x3 + x1*x2);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(22, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_1_processor();
    // Finish code for explicit solution node 21
    
    // Code for non-branch dispatcher node 22
    // Actually, there is no code
    
    // Code for explicit solution node 23, solved variable is th_4
    auto ExplicitSolutionNode_node_23_solve_th_4_processor = [&]() -> void
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
            const double th_1 = this_solution[1];
            const double th_1th_2th_4_soa = this_solution[2];
            const double th_2 = this_solution[3];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_1 + th_1th_2th_4_soa - th_2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_4_processor();
    // Finish code for explicit solution node 22
    
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
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
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
            const double th_2 = this_solution[3];
            
            const bool degenerate_valid_0 = std::fabs(th_2 - 1.0/2.0*M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(24, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_2 + M_PI_2) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(27, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(14, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_13_processor();
    // Finish code for solved_variable dispatcher node 13
    
    // Code for explicit solution node 27, solved variable is th_1
    auto ExplicitSolutionNode_node_27_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(27);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(27);
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
            const double th_3 = this_solution[4];
            
            const bool condition_0 = std::fabs(d_3*std::sin(th_3)) >= 9.9999999999999995e-7 || std::fabs(a_0 + d_2) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_0 + d_2;
                const double x1 = Pz - d_4*r_33;
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3;
                const double x5 = d_3*std::sin(th_3);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x1 - x4*x5, x0*x4 - x1*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(28, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_1_processor();
    // Finish code for explicit solution node 27
    
    // Code for non-branch dispatcher node 28
    // Actually, there is no code
    
    // Code for explicit solution node 29, solved variable is th_4
    auto ExplicitSolutionNode_node_29_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(28);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(28);
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
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_13*(std::sin(th_0)*std::sin(th_3) + std::sin(th_1)*std::cos(th_0)*std::cos(th_3)) + r_23*(std::sin(th_0)*std::sin(th_1)*std::cos(th_3) - std::sin(th_3)*std::cos(th_0)) + r_33*std::cos(th_1)*std::cos(th_3)) >= zero_tolerance || std::fabs(r_13*std::cos(th_0)*std::cos(th_1) + r_23*std::sin(th_0)*std::cos(th_1) - r_33*std::sin(th_1)) >= zero_tolerance;
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
                const double tmp_sol_value = std::atan2(-r_13*(x2*x3 + x4*x6) - r_23*(x2*x6 - x3*x4) - r_33*x0*x1, -r_13*x0*x4 - r_23*x0*x2 + r_33*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_4_processor();
    // Finish code for explicit solution node 28
    
    // Code for explicit solution node 24, solved variable is th_1
    auto ExplicitSolutionNode_node_24_solve_th_1_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(24);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(24);
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
            const double th_3 = this_solution[4];
            
            const bool condition_0 = std::fabs(d_3*std::sin(th_3)) >= 9.9999999999999995e-7 || std::fabs(a_0 - d_2) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = a_0 - d_2;
                const double x1 = Pz - d_4*r_33;
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3;
                const double x5 = d_3*std::sin(th_3);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x1 + x4*x5, x0*x4 + x1*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(25, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_24_solve_th_1_processor();
    // Finish code for explicit solution node 24
    
    // Code for non-branch dispatcher node 25
    // Actually, there is no code
    
    // Code for explicit solution node 26, solved variable is th_4
    auto ExplicitSolutionNode_node_26_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(25);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(25);
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
            const double th_1 = this_solution[1];
            const double th_3 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(-r_13*(std::sin(th_0)*std::sin(th_3) - std::sin(th_1)*std::cos(th_0)*std::cos(th_3)) + r_23*(std::sin(th_0)*std::sin(th_1)*std::cos(th_3) + std::sin(th_3)*std::cos(th_0)) + r_33*std::cos(th_1)*std::cos(th_3)) >= zero_tolerance || std::fabs(r_13*std::cos(th_0)*std::cos(th_1) + r_23*std::sin(th_0)*std::cos(th_1) - r_33*std::sin(th_1)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::cos(th_3);
                const double x2 = std::sin(th_3);
                const double x3 = std::cos(th_0);
                const double x4 = std::sin(th_0);
                const double x5 = std::sin(th_1);
                const double x6 = x1*x5;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_13*(x2*x4 - x3*x6) + r_23*(x2*x3 + x4*x6) + r_33*x0*x1, r_13*x0*x3 + r_23*x0*x4 - r_33*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_26_solve_th_4_processor();
    // Finish code for explicit solution node 25
    
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
            const double th_2 = this_solution[3];
            
            const bool condition_0 = std::fabs(d_2*std::cos(th_2)) >= 9.9999999999999995e-7 || std::fabs(a_0 - d_2*std::sin(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Pz - d_4*r_33;
                const double x1 = a_0 - d_2*std::sin(th_2);
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = Px*x2 + Py*x3 - d_4*r_13*x2 - d_4*r_23*x3;
                const double x5 = d_2*std::cos(th_2);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-x0*x1 - x4*x5, -x0*x5 + x1*x4);
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
            const double th_2 = this_solution[3];
            
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
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
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
            const double th_3 = this_solution[4];
            
            const bool condition_0 = std::fabs((r_13*std::sin(th_0) - r_23*std::cos(th_0))/std::sin(th_3)) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::asin((-r_13*std::sin(th_0) + r_23*std::cos(th_0))/std::sin(th_3));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[5] = x0;
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
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
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
            const double th_2 = this_solution[3];
            const double th_3 = this_solution[4];
            const bool checked_result = std::fabs(d_2*std::cos(th_2) - d_3*std::sin(th_2)*std::sin(th_3)) <= 9.9999999999999995e-7 && std::fabs(-a_0 + d_2*std::sin(th_2) + d_3*std::sin(th_3)*std::cos(th_2)) <= 9.9999999999999995e-7;
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
            const double th_2 = this_solution[3];
            const double th_3 = this_solution[4];
            
            const bool condition_0 = std::fabs(d_2*std::cos(th_2) - d_3*std::sin(th_2)*std::sin(th_3)) >= 9.9999999999999995e-7 || std::fabs(-a_0 + d_2*std::sin(th_2) + d_3*std::sin(th_3)*std::cos(th_2)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = d_4*r_33;
                const double x1 = std::sin(th_2);
                const double x2 = std::cos(th_2);
                const double x3 = d_3*std::sin(th_3);
                const double x4 = a_0 - d_2*x1 - x2*x3;
                const double x5 = -d_2*x2 + x1*x3;
                const double x6 = std::cos(th_0);
                const double x7 = std::sin(th_0);
                const double x8 = Px*x6 + Py*x7 - d_4*r_13*x6 - d_4*r_23*x7;
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
            const double th_2 = this_solution[3];
            const double th_3 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(-r_11*((std::sin(th_1)*std::sin(th_2) - std::cos(th_1)*std::cos(th_2))*std::sin(th_3)*std::cos(th_0) + std::sin(th_0)*std::cos(th_3)) + r_21*((-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_0)*std::sin(th_3) + std::cos(th_0)*std::cos(th_3)) - r_31*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::sin(th_3)) >= zero_tolerance || std::fabs(-r_12*((std::sin(th_1)*std::sin(th_2) - std::cos(th_1)*std::cos(th_2))*std::sin(th_3)*std::cos(th_0) + std::sin(th_0)*std::cos(th_3)) + r_22*((-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_0)*std::sin(th_3) + std::cos(th_0)*std::cos(th_3)) - r_32*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::sin(th_3)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_3);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_2);
                const double x3 = std::sin(th_2);
                const double x4 = std::cos(th_1);
                const double x5 = x0*(x1*x2 + x3*x4);
                const double x6 = std::cos(th_0);
                const double x7 = std::cos(th_3);
                const double x8 = std::sin(th_0);
                const double x9 = x2*x4;
                const double x10 = x1*x3;
                const double x11 = x0*x8*(-x10 + x9) + x6*x7;
                const double x12 = x0*x6*(x10 - x9) + x7*x8;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x12 + r_21*x11 - r_31*x5, -r_12*x12 + r_22*x11 - r_32*x5);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
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

}; // struct denso_cvrb_ik

// Code below for debug
void test_ik_solve_denso_cvrb()
{
    std::array<double, denso_cvrb_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = denso_cvrb_ik::computeFK(theta);
    auto ik_output = denso_cvrb_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = denso_cvrb_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_denso_cvrb();
}
