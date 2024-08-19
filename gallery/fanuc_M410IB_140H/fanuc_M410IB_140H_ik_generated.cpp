#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct fanuc_M410IB_140H_ik {

// Constants for solver
static constexpr int robot_nq = 5;
static constexpr int max_n_solutions = 16;
static constexpr int n_tree_nodes = 12;
static constexpr int intermediate_solution_size = 6;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_0 = 0.24;
static constexpr double a_1 = 1.16;
static constexpr double a_2 = 1.7364907140552177;
static constexpr double d_3 = -0.215;
static constexpr double pre_transform_special_symbol_23 = 0.7200000000000002;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -1.5707963267948966;
static constexpr double th_2_offset_original2raw = -1.484307426876522;
static constexpr double th_3_offset_original2raw = -0.08648889991837498;
static constexpr double th_4_offset_original2raw = 3.141592653589793;

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
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = std::sin(th_4);
    const double x2 = std::cos(th_4);
    const double x3 = std::cos(th_3);
    const double x4 = std::cos(th_0);
    const double x5 = std::sin(th_1);
    const double x6 = std::sin(th_2);
    const double x7 = x5*x6;
    const double x8 = std::cos(th_1);
    const double x9 = std::cos(th_2);
    const double x10 = x8*x9;
    const double x11 = x10*x4 + x4*x7;
    const double x12 = std::sin(th_3);
    const double x13 = x5*x9;
    const double x14 = x6*x8;
    const double x15 = x13*x4 - x14*x4;
    const double x16 = x11*x3 + x12*x15;
    const double x17 = -x11*x12 + x15*x3;
    const double x18 = a_1*x8;
    const double x19 = x0*x10 + x0*x7;
    const double x20 = x0*x13 - x0*x14;
    const double x21 = x12*x20 + x19*x3;
    const double x22 = -x12*x19 + x20*x3;
    const double x23 = x10 + x7;
    const double x24 = -x13 + x6*x8;
    const double x25 = x12*x23 + x24*x3;
    const double x26 = -x12*x24 + x23*x3;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x1 + x16*x2;
    ee_pose_raw(0, 1) = -x0*x2 - x1*x16;
    ee_pose_raw(0, 2) = x17;
    ee_pose_raw(0, 3) = a_0*x4 + a_2*x11 + d_3*x17 + x18*x4;
    ee_pose_raw(1, 0) = x1*x4 + x2*x21;
    ee_pose_raw(1, 1) = -x1*x21 + x2*x4;
    ee_pose_raw(1, 2) = x22;
    ee_pose_raw(1, 3) = a_0*x0 + a_2*x19 + d_3*x22 + x0*x18;
    ee_pose_raw(2, 0) = x2*x25;
    ee_pose_raw(2, 1) = -x1*x25;
    ee_pose_raw(2, 2) = x26;
    ee_pose_raw(2, 3) = -a_1*x5 + a_2*x24 + d_3*x26;
    return endEffectorTargetRawToOriginal(ee_pose_raw);
}

static void computeTwistJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, 5>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = 1.0*x0;
    const double x2 = std::sin(th_3);
    const double x3 = std::sin(th_1);
    const double x4 = std::sin(th_2);
    const double x5 = std::cos(th_0);
    const double x6 = 1.0*x5;
    const double x7 = x4*x6;
    const double x8 = std::cos(th_1);
    const double x9 = std::cos(th_2);
    const double x10 = x6*x9;
    const double x11 = x10*x8 + x3*x7;
    const double x12 = std::cos(th_3);
    const double x13 = -x11*x2 + x12*(x10*x3 - x7*x8);
    const double x14 = -x6;
    const double x15 = x1*x4;
    const double x16 = x1*x9;
    const double x17 = x15*x3 + x16*x8;
    const double x18 = x12*(-x15*x8 + x16*x3) - x17*x2;
    const double x19 = 1.0*x3;
    const double x20 = -x19*x9 + 1.0*x4*x8;
    const double x21 = x12*(x19*x4 + 1.0*x8*x9) - x2*x20;
    const double x22 = a_1*x19;
    const double x23 = pre_transform_special_symbol_23 - x22;
    const double x24 = a_2*x20 + pre_transform_special_symbol_23 - x22;
    const double x25 = d_3*x21 + x24;
    const double x26 = a_1*x8;
    const double x27 = a_0*x1 + x1*x26;
    const double x28 = a_2*x17 + x27;
    const double x29 = d_3*x18 + x28;
    const double x30 = a_0*x6 + x26*x6;
    const double x31 = a_2*x11 + x30;
    const double x32 = d_3*x13 + x31;
    const double x33 = 1.0*a_0;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = -x1;
    jacobian(0, 2) = x1;
    jacobian(0, 3) = x1;
    jacobian(0, 4) = x13;
    jacobian(1, 1) = x6;
    jacobian(1, 2) = x14;
    jacobian(1, 3) = x14;
    jacobian(1, 4) = x18;
    jacobian(2, 0) = 1.0;
    jacobian(2, 4) = x21;
    jacobian(3, 1) = -pre_transform_special_symbol_23*x6;
    jacobian(3, 2) = x23*x6;
    jacobian(3, 3) = x24*x6;
    jacobian(3, 4) = -x18*x25 + x21*x29;
    jacobian(4, 1) = -pre_transform_special_symbol_23*x1;
    jacobian(4, 2) = x1*x23;
    jacobian(4, 3) = x1*x24;
    jacobian(4, 4) = x13*x25 - x21*x32;
    jacobian(5, 1) = std::pow(x0, 2)*x33 + x33*std::pow(x5, 2);
    jacobian(5, 2) = -x1*x27 - x30*x6;
    jacobian(5, 3) = -x1*x28 - x31*x6;
    jacobian(5, 4) = -x13*x29 + x18*x32;
    return;
}

static void computeAngularVelocityJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, 5>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
    
    // Temp variable for efficiency
    const double x0 = 1.0*std::sin(th_0);
    const double x1 = std::sin(th_3);
    const double x2 = std::sin(th_1);
    const double x3 = std::sin(th_2);
    const double x4 = 1.0*std::cos(th_0);
    const double x5 = x3*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::cos(th_2);
    const double x8 = x4*x7;
    const double x9 = std::cos(th_3);
    const double x10 = -x4;
    const double x11 = x0*x3;
    const double x12 = x0*x7;
    const double x13 = 1.0*x2;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = -x0;
    jacobian(0, 2) = x0;
    jacobian(0, 3) = x0;
    jacobian(0, 4) = -x1*(x2*x5 + x6*x8) + x9*(x2*x8 - x5*x6);
    jacobian(1, 1) = x4;
    jacobian(1, 2) = x10;
    jacobian(1, 3) = x10;
    jacobian(1, 4) = -x1*(x11*x2 + x12*x6) + x9*(-x11*x6 + x12*x2);
    jacobian(2, 0) = 1.0;
    jacobian(2, 4) = -x1*(-x13*x7 + 1.0*x3*x6) + x9*(x13*x3 + 1.0*x6*x7);
    return;
}

static void computeTransformPointJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Vector3d& point_on_ee, Eigen::Matrix<double, 6, 5>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0] + th_0_offset_original2raw;
    const double th_1 = theta_input_original[1] + th_1_offset_original2raw;
    const double th_2 = theta_input_original[2] + th_2_offset_original2raw;
    const double th_3 = theta_input_original[3] + th_3_offset_original2raw;
    const double th_4 = theta_input_original[4] + th_4_offset_original2raw;
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
    const double x8 = std::cos(th_2);
    const double x9 = x5*x8;
    const double x10 = std::sin(th_2);
    const double x11 = std::cos(th_1);
    const double x12 = 1.0*x10*x11 - x9;
    const double x13 = a_2*x12 + pre_transform_special_symbol_23 - x6;
    const double x14 = std::cos(th_3);
    const double x15 = x10*x5;
    const double x16 = 1.0*x11;
    const double x17 = x16*x8;
    const double x18 = std::sin(th_3);
    const double x19 = -x12*x18 + x14*(x15 + x17);
    const double x20 = std::sin(th_0);
    const double x21 = x15*x20 + x17*x20;
    const double x22 = x14*(-x10*x16*x20 + x20*x9) - x18*x21;
    const double x23 = d_3*x19 + x13;
    const double x24 = 1.0*x20;
    const double x25 = a_0*x24 + a_1*x16*x20;
    const double x26 = a_2*x21 + x25;
    const double x27 = d_3*x22 + x26;
    const double x28 = 1.0*p_on_ee_x;
    const double x29 = p_on_ee_z*x24;
    const double x30 = x10*x2;
    const double x31 = x2*x8;
    const double x32 = x11*x31 + x30*x4;
    const double x33 = x14*(-x11*x30 + x31*x4) - x18*x32;
    const double x34 = a_0*x2 + a_1*x11*x2;
    const double x35 = a_2*x32 + x34;
    const double x36 = d_3*x33 + x35;
    const double x37 = x1*x28;
    const double x38 = x0*x20;
    const double x39 = 1.0*a_0;
    const double x40 = x37 + x38;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = -pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = 1.0*x1*x7 - x3;
    jacobian(0, 3) = 1.0*x1*x13 - x3;
    jacobian(0, 4) = -p_on_ee_y*x19 + p_on_ee_z*x22 + x19*x27 - x22*x23;
    jacobian(1, 0) = x28;
    jacobian(1, 1) = -pre_transform_special_symbol_23*x24 + x29;
    jacobian(1, 2) = 1.0*x20*x7 - x29;
    jacobian(1, 3) = 1.0*x13*x20 - x29;
    jacobian(1, 4) = p_on_ee_x*x19 - p_on_ee_z*x33 - x19*x36 + x23*x33;
    jacobian(2, 1) = std::pow(x1, 2)*x39 + std::pow(x20, 2)*x39 - x37 - x38;
    jacobian(2, 2) = -x2*x34 - x24*x25 + x40;
    jacobian(2, 3) = -x2*x35 - x24*x26 + x40;
    jacobian(2, 4) = -p_on_ee_x*x22 + p_on_ee_y*x33 + x22*x36 - x27*x33;
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
            solution_0[1] = x0;
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
            solution_1[1] = x0 + M_PI;
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
            const double th_0 = this_solution[1];
            
            const bool condition_0 = (1.0/2.0)*std::fabs((std::pow(Px, 2) - 2*Px*a_0*std::cos(th_0) - 2*Px*d_3*r_13 + std::pow(Py, 2) - 2*Py*a_0*std::sin(th_0) - 2*Py*d_3*r_23 + std::pow(Pz, 2) - 2*Pz*d_3*r_33 + std::pow(a_0, 2) + 2*a_0*d_3*r_13*std::cos(th_0) + 2*a_0*d_3*r_23*std::sin(th_0) - std::pow(a_1, 2) - std::pow(a_2, 2) + std::pow(d_3, 2)*std::pow(r_13, 2) + std::pow(d_3, 2)*std::pow(r_23, 2) + std::pow(d_3, 2)*std::pow(r_33, 2))/(a_1*a_2)) <= 1;
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
                const double x7 = std::acos((1.0/2.0)*(std::pow(Px, 2) + std::pow(Py, 2) + std::pow(Pz, 2) - 2*Pz*d_3*r_33 + std::pow(a_0, 2) - std::pow(a_1, 2) - std::pow(a_2, 2) + std::pow(r_13, 2)*x6 + std::pow(r_23, 2)*x6 + std::pow(r_33, 2)*x6 - x0*x1 - x0*x4 + 2*x1*x4 - x2*x3 - x2*x5 + 2*x3*x5)/(a_1*a_2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[3] = x7;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = (1.0/2.0)*std::fabs((std::pow(Px, 2) - 2*Px*a_0*std::cos(th_0) - 2*Px*d_3*r_13 + std::pow(Py, 2) - 2*Py*a_0*std::sin(th_0) - 2*Py*d_3*r_23 + std::pow(Pz, 2) - 2*Pz*d_3*r_33 + std::pow(a_0, 2) + 2*a_0*d_3*r_13*std::cos(th_0) + 2*a_0*d_3*r_23*std::sin(th_0) - std::pow(a_1, 2) - std::pow(a_2, 2) + std::pow(d_3, 2)*std::pow(r_13, 2) + std::pow(d_3, 2)*std::pow(r_23, 2) + std::pow(d_3, 2)*std::pow(r_33, 2))/(a_1*a_2)) <= 1;
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
                const double x7 = std::acos((1.0/2.0)*(std::pow(Px, 2) + std::pow(Py, 2) + std::pow(Pz, 2) - 2*Pz*d_3*r_33 + std::pow(a_0, 2) - std::pow(a_1, 2) - std::pow(a_2, 2) + std::pow(r_13, 2)*x6 + std::pow(r_23, 2)*x6 + std::pow(r_33, 2)*x6 - x0*x1 - x0*x4 + 2*x1*x4 - x2*x3 - x2*x5 + 2*x3*x5)/(a_1*a_2));
                // End of temp variables
                const double tmp_sol_value = -x7;
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
            const double th_0 = this_solution[1];
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
            const double th_0 = this_solution[1];
            const double th_2 = this_solution[3];
            
            const bool condition_0 = std::fabs(Pz - d_3*r_33) >= 9.9999999999999995e-7 || std::fabs(-Px*std::cos(th_0) - Py*std::sin(th_0) + a_0 + d_3*r_13*std::cos(th_0) + d_3*r_23*std::sin(th_0)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Pz - d_3*r_33;
                const double x1 = -a_1 - a_2*std::cos(th_2);
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = -Px*x2 - Py*x3 + a_0 + d_3*r_13*x2 + d_3*r_23*x3;
                const double x5 = a_2*std::sin(th_2);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*x1 - x4*x5, x0*x5 + x1*x4);
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
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            
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
            const double th_0 = this_solution[1];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x0 + r_21*x1, -r_12*x0 + r_22*x1);
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
    ExplicitSolutionNode_node_9_solve_th_4_processor();
    // Finish code for explicit solution node 8
    
    // Code for non-branch dispatcher node 10
    // Actually, there is no code
    
    // Code for explicit solution node 11, solved variable is th_3
    auto ExplicitSolutionNode_node_11_solve_th_3_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            
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
    ExplicitSolutionNode_node_11_solve_th_3_processor();
    // Finish code for explicit solution node 10
    
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

}; // struct fanuc_M410IB_140H_ik

// Code below for debug
void test_ik_solve_fanuc_M410IB_140H()
{
    std::array<double, fanuc_M410IB_140H_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = fanuc_M410IB_140H_ik::computeFK(theta);
    auto ik_output = fanuc_M410IB_140H_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = fanuc_M410IB_140H_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_fanuc_M410IB_140H();
}
