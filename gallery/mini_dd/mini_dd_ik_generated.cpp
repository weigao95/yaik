#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct mini_dd_ik {

// Constants for solver
static constexpr int robot_nq = 5;
static constexpr int max_n_solutions = 16;
static constexpr int n_tree_nodes = 10;
static constexpr int intermediate_solution_size = 5;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double l_3 = 5.0;
static constexpr double l_4 = 2.0;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw

// The transformation between raw and original ee target
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// ee_original = pre_transform * ee_raw * post_transform
// ee_raw = dh_forward_transform(theta_raw)
static Eigen::Matrix4d endEffectorTargetOriginalToRaw(const Eigen::Matrix4d& T_ee)
{
    return T_ee;
}

static Eigen::Matrix4d endEffectorTargetRawToOriginal(const Eigen::Matrix4d& T_ee)
{
    return T_ee;
}

///************* Below are the actual FK and IK implementations *******************
static Eigen::Matrix4d computeFK(const std::array<double, robot_nq>& theta_input_original)
{
    // Extract the variables
    const double d_0 = theta_input_original[0];
    const double th_1 = theta_input_original[1];
    const double th_2 = theta_input_original[2];
    const double th_3 = theta_input_original[3];
    const double th_4 = theta_input_original[4];
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_4);
    const double x1 = std::cos(th_1);
    const double x2 = std::sin(th_2);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_4);
    const double x5 = std::sin(th_1);
    const double x6 = std::sin(th_3);
    const double x7 = x5*x6;
    const double x8 = std::cos(th_2);
    const double x9 = std::cos(th_3);
    const double x10 = x1*x9;
    const double x11 = x10*x8 + x7;
    const double x12 = x5*x9;
    const double x13 = x1*x6;
    const double x14 = x2*x9;
    const double x15 = x2*x5;
    const double x16 = -x12*x8 + x13;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = x0*x3 + x11*x4;
    ee_pose_raw(0, 1) = -x0*x11 + x3*x4;
    ee_pose_raw(0, 2) = x12 - x13*x8;
    ee_pose_raw(0, 3) = l_3*x1 - l_4*x3;
    ee_pose_raw(1, 0) = x0*x8 - x14*x4;
    ee_pose_raw(1, 1) = x0*x14 + x4*x8;
    ee_pose_raw(1, 2) = x2*x6;
    ee_pose_raw(1, 3) = -l_4*x8;
    ee_pose_raw(2, 0) = -x0*x15 + x16*x4;
    ee_pose_raw(2, 1) = -x0*x16 - x15*x4;
    ee_pose_raw(2, 2) = x10 + x7*x8;
    ee_pose_raw(2, 3) = d_0 - l_3*x5 + l_4*x15;
    return endEffectorTargetRawToOriginal(ee_pose_raw);
}

static void computeTwistJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, 5>& jacobian)
{
    // Extract the variables
    const double d_0 = theta_input_original[0];
    const double th_1 = theta_input_original[1];
    const double th_2 = theta_input_original[2];
    const double th_3 = theta_input_original[3];
    const double th_4 = theta_input_original[4];
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_1);
    const double x1 = std::sin(th_2);
    const double x2 = std::cos(th_1);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_3);
    const double x5 = std::sin(th_3);
    const double x6 = std::cos(th_2);
    const double x7 = x5*x6;
    const double x8 = x0*x4 - x2*x7;
    const double x9 = x1*x5;
    const double x10 = x0*x1;
    const double x11 = x0*x7 + x2*x4;
    const double x12 = l_4*x10;
    const double x13 = d_0 - l_3*x0;
    const double x14 = x12 + x13;
    const double x15 = l_4*x6;
    const double x16 = l_3*x2 - l_4*x3;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 2) = -x0;
    jacobian(0, 3) = -x3;
    jacobian(0, 4) = x8;
    jacobian(1, 1) = 1;
    jacobian(1, 3) = -x6;
    jacobian(1, 4) = x9;
    jacobian(2, 2) = -x2;
    jacobian(2, 3) = x10;
    jacobian(2, 4) = x11;
    jacobian(3, 1) = -d_0;
    jacobian(3, 3) = -x12*x6 + x14*x6;
    jacobian(3, 4) = -x11*x15 - x14*x9;
    jacobian(4, 2) = l_3*std::pow(x2, 2) - x0*x13;
    jacobian(4, 3) = -x10*x16 - x14*x3;
    jacobian(4, 4) = -x11*x16 + x14*x8;
    jacobian(5, 0) = 1;
    jacobian(5, 3) = -x15*x3 - x16*x6;
    jacobian(5, 4) = x15*x8 + x16*x9;
    return;
}

static void computeAngularVelocityJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, 5>& jacobian)
{
    // Extract the variables
    const double d_0 = theta_input_original[0];
    const double th_1 = theta_input_original[1];
    const double th_2 = theta_input_original[2];
    const double th_3 = theta_input_original[3];
    const double th_4 = theta_input_original[4];
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_1);
    const double x1 = std::sin(th_2);
    const double x2 = std::cos(th_1);
    const double x3 = std::cos(th_3);
    const double x4 = std::sin(th_3);
    const double x5 = std::cos(th_2);
    const double x6 = x4*x5;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 2) = -x0;
    jacobian(0, 3) = -x1*x2;
    jacobian(0, 4) = x0*x3 - x2*x6;
    jacobian(1, 1) = 1;
    jacobian(1, 3) = -x5;
    jacobian(1, 4) = x1*x4;
    jacobian(2, 2) = -x2;
    jacobian(2, 3) = x0*x1;
    jacobian(2, 4) = x0*x6 + x2*x3;
    return;
}

static void computeTransformPointJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Vector3d& point_on_ee, Eigen::Matrix<double, 6, 5>& jacobian)
{
    // Extract the variables
    const double d_0 = theta_input_original[0];
    const double th_1 = theta_input_original[1];
    const double th_2 = theta_input_original[2];
    const double th_3 = theta_input_original[3];
    const double th_4 = theta_input_original[4];
    const double p_on_ee_x = point_on_ee[0];
    const double p_on_ee_y = point_on_ee[1];
    const double p_on_ee_z = point_on_ee[2];
    
    // Temp variable for efficiency
    const double x0 = std::cos(th_1);
    const double x1 = p_on_ee_y*x0;
    const double x2 = std::cos(th_2);
    const double x3 = std::sin(th_2);
    const double x4 = std::sin(th_1);
    const double x5 = p_on_ee_y*x4;
    const double x6 = x3*x4;
    const double x7 = l_4*x6;
    const double x8 = d_0 - l_3*x4;
    const double x9 = x7 + x8;
    const double x10 = std::sin(th_3);
    const double x11 = x10*x3;
    const double x12 = std::cos(th_3);
    const double x13 = x10*x2;
    const double x14 = x0*x12 + x13*x4;
    const double x15 = l_4*x2;
    const double x16 = x0*x3;
    const double x17 = l_3*x0 - l_4*x16;
    const double x18 = -x0*x13 + x12*x4;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = -d_0 + p_on_ee_z;
    jacobian(0, 2) = x1;
    jacobian(0, 3) = -p_on_ee_z*x2 - x2*x7 + x2*x9 - x3*x5;
    jacobian(0, 4) = -p_on_ee_y*x14 + p_on_ee_z*x11 - x11*x9 - x14*x15;
    jacobian(1, 2) = l_3*std::pow(x0, 2) - p_on_ee_x*x0 + p_on_ee_z*x4 - x4*x8;
    jacobian(1, 3) = p_on_ee_x*x6 + p_on_ee_z*x16 - x16*x9 - x17*x6;
    jacobian(1, 4) = p_on_ee_x*x14 - p_on_ee_z*x18 - x14*x17 + x18*x9;
    jacobian(2, 0) = 1;
    jacobian(2, 1) = -p_on_ee_x;
    jacobian(2, 2) = -x5;
    jacobian(2, 3) = p_on_ee_x*x2 - x1*x3 - x15*x16 - x17*x2;
    jacobian(2, 4) = -p_on_ee_x*x11 + p_on_ee_y*x18 + x11*x17 + x15*x18;
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
    
    // Code for explicit solution node 1, solved variable is th_2
    auto ExplicitSolutionNode_node_1_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;
        
        // The explicit solution of root node
        const bool condition_0 = std::fabs(Py/l_4) <= 1;
        if (condition_0)
        {
            // Temp variable for efficiency
            const double x0 = std::acos(-Py/l_4);
            // End of temp variables
            
            auto solution_0 = make_raw_solution();
            solution_0[2] = x0;
            int appended_idx = append_solution_to_queue(solution_0);
            add_input_index_to(2, appended_idx);
        }
        
        const bool condition_1 = std::fabs(Py/l_4) <= 1;
        if (condition_1)
        {
            // Temp variable for efficiency
            const double x0 = std::acos(-Py/l_4);
            // End of temp variables
            
            auto solution_1 = make_raw_solution();
            solution_1[2] = -x0;
            int appended_idx = append_solution_to_queue(solution_1);
            add_input_index_to(2, appended_idx);
        }
        
    };
    // Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_2_processor();
    // Finish code for explicit solution node 0
    
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
            const double th_2 = this_solution[2];
            const bool checked_result = std::fabs(l_3 - l_4*std::sin(th_2)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(3, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_2_processor();
    // Finish code for equation all-zero dispatcher node 2
    
    // Code for explicit solution node 3, solved variable is th_1
    auto ExplicitSolutionNode_node_3_solve_th_1_processor = [&]() -> void
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
            
            const bool condition_0 = std::fabs(Px/(l_3 - l_4*std::sin(th_2))) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::acos(Px/(l_3 - l_4*std::sin(th_2)));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[1] = x0;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = std::fabs(Px/(l_3 - l_4*std::sin(th_2))) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::acos(Px/(l_3 - l_4*std::sin(th_2)));
                // End of temp variables
                const double tmp_sol_value = -x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(4, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_3_solve_th_1_processor();
    // Finish code for explicit solution node 3
    
    // Code for non-branch dispatcher node 4
    // Actually, there is no code
    
    // Code for explicit solution node 5, solved variable is d_0
    auto ExplicitSolutionNode_node_5_solve_d_0_processor = [&]() -> void
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_1);
                // End of temp variables
                const double tmp_sol_value = Pz + l_3*x0 - l_4*x0*std::sin(th_2);
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
    ExplicitSolutionNode_node_5_solve_d_0_processor();
    // Finish code for explicit solution node 4
    
    // Code for non-branch dispatcher node 6
    // Actually, there is no code
    
    // Code for explicit solution node 7, solved variable is th_3
    auto ExplicitSolutionNode_node_7_solve_th_3_processor = [&]() -> void
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_13*std::sin(th_1) + r_33*std::cos(th_1)) >= zero_tolerance || std::fabs(-r_13*std::cos(th_1)*std::cos(th_2) + r_23*std::sin(th_2) + r_33*std::sin(th_1)*std::cos(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_1);
                const double x1 = std::cos(th_2);
                const double x2 = std::cos(th_1);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_13*x1*x2 + r_23*std::sin(th_2) + r_33*x0*x1, r_13*x0 + r_33*x2);
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(8, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_3_processor();
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
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[2];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_2)*std::cos(th_1) + r_21*std::cos(th_2) - r_31*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(r_12*std::sin(th_2)*std::cos(th_1) + r_22*std::cos(th_2) - r_32*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_2);
                const double x1 = std::sin(th_2);
                const double x2 = x1*std::cos(th_1);
                const double x3 = x1*std::sin(th_1);
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_11*x2 + r_21*x0 - r_31*x3, r_12*x2 + r_22*x0 - r_32*x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
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
    
    // Collect the output
    for(int i = 0; i < solution_queue.size(); i++)
    {
        if(!solution_queue.solutions_validity[i])
            continue;
        const auto& raw_ik_out_i = solution_queue.get_solution(i);
        std::array<double, robot_nq> new_ik_i;
        const double value_at_0 = raw_ik_out_i[0];  // d_0
        new_ik_i[0] = value_at_0;
        const double value_at_1 = raw_ik_out_i[1];  // th_1
        new_ik_i[1] = value_at_1;
        const double value_at_2 = raw_ik_out_i[2];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[3];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[4];  // th_4
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
    const Eigen::Matrix4d& T_ee_raw = T_ee;
    computeRawIK(T_ee_raw, workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
        ik_output.push_back(ik_out_i);
    }
}

static void computeIK(const Eigen::Matrix4d& T_ee, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    const Eigen::Matrix4d& T_ee_raw = T_ee;
    computeRawIK(T_ee_raw, workspace);
    const auto& raw_ik_out = workspace.raw_ik_out;
    ik_output.clear();
    for(int i = 0; i < raw_ik_out.size(); i++)
    {
        auto ik_out_i = raw_ik_out[i];
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

}; // struct mini_dd_ik

// Code below for debug
void test_ik_solve_mini_dd()
{
    std::array<double, mini_dd_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = mini_dd_ik::computeFK(theta);
    auto ik_output = mini_dd_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = mini_dd_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_mini_dd();
}
