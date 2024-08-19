#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct puma_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 16;
static constexpr int n_tree_nodes = 24;
static constexpr int intermediate_solution_size = 9;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_2 = 0.432;
static constexpr double a_3 = 0.0203;
static constexpr double d_1 = 0.6;
static constexpr double d_3 = 0.1245;
static constexpr double d_4 = 0.432;

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
    const double th_0 = theta_input_original[0];
    const double th_1 = theta_input_original[1];
    const double th_2 = theta_input_original[2];
    const double th_3 = theta_input_original[3];
    const double th_4 = theta_input_original[4];
    const double th_5 = theta_input_original[5];
    
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
    const double x21 = a_2*x5;
    const double x22 = -x1*x10 + x1*x7;
    const double x23 = -x2*x4 - x22*x3;
    const double x24 = -x1*x15 - x1*x16;
    const double x25 = x2*x22 - x3*x4;
    const double x26 = -x14*x24 + x18*x25;
    const double x27 = -x15 - x16;
    const double x28 = x27*x3;
    const double x29 = x10 - x7;
    const double x30 = x2*x27;
    const double x31 = -x14*x29 + x18*x30;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = x0*x12 + x13*x20;
    ee_pose_raw(0, 1) = -x0*x20 + x12*x13;
    ee_pose_raw(0, 2) = x14*x19 + x17*x18;
    ee_pose_raw(0, 3) = a_3*x11 - d_3*x1 + d_4*x17 + x21*x4;
    ee_pose_raw(1, 0) = x0*x23 + x13*x26;
    ee_pose_raw(1, 1) = -x0*x26 + x13*x23;
    ee_pose_raw(1, 2) = x14*x25 + x18*x24;
    ee_pose_raw(1, 3) = a_3*x22 + d_3*x4 + d_4*x24 + x1*x21;
    ee_pose_raw(2, 0) = -x0*x28 + x13*x31;
    ee_pose_raw(2, 1) = -x0*x31 - x13*x28;
    ee_pose_raw(2, 2) = x14*x30 + x18*x29;
    ee_pose_raw(2, 3) = -a_2*x8 + a_3*x27 + d_1 + d_4*x29;
    return endEffectorTargetRawToOriginal(ee_pose_raw);
}

static void computeTwistJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, 6>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0];
    const double th_1 = theta_input_original[1];
    const double th_2 = theta_input_original[2];
    const double th_3 = theta_input_original[3];
    const double th_4 = theta_input_original[4];
    const double th_5 = theta_input_original[5];
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = -x0;
    const double x2 = std::cos(th_0);
    const double x3 = std::sin(th_1);
    const double x4 = std::cos(th_2);
    const double x5 = x3*x4;
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_1);
    const double x8 = x6*x7;
    const double x9 = -x2*x5 - x2*x8;
    const double x10 = std::cos(th_3);
    const double x11 = std::sin(th_3);
    const double x12 = x4*x7;
    const double x13 = x3*x6;
    const double x14 = x12*x2 - x13*x2;
    const double x15 = x0*x10 - x11*x14;
    const double x16 = std::cos(th_4);
    const double x17 = std::sin(th_4);
    const double x18 = x16*x9 + x17*(x0*x11 + x10*x14);
    const double x19 = -x0*x5 - x0*x8;
    const double x20 = x0*x12 - x0*x13;
    const double x21 = -x10*x2 - x11*x20;
    const double x22 = x16*x19 + x17*(x10*x20 - x11*x2);
    const double x23 = -x12 + x13;
    const double x24 = -x5 - x8;
    const double x25 = x11*x24;
    const double x26 = x10*x17*x24 + x16*x23;
    const double x27 = -a_2*x3 + d_1;
    const double x28 = a_3*x24 + d_4*x23 + x27;
    const double x29 = a_2*x7;
    const double x30 = d_3*x2 + x0*x29;
    const double x31 = a_3*x20 + d_4*x19 + x30;
    const double x32 = -d_3*x0 + x2*x29;
    const double x33 = a_3*x14 + d_4*x9 + x32;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x1;
    jacobian(0, 2) = x1;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = x15;
    jacobian(0, 5) = x18;
    jacobian(1, 1) = x2;
    jacobian(1, 2) = x2;
    jacobian(1, 3) = x19;
    jacobian(1, 4) = x21;
    jacobian(1, 5) = x22;
    jacobian(2, 0) = 1;
    jacobian(2, 3) = x23;
    jacobian(2, 4) = -x25;
    jacobian(2, 5) = x26;
    jacobian(3, 1) = -d_1*x2;
    jacobian(3, 2) = -x2*x27;
    jacobian(3, 3) = -x19*x28 + x23*x31;
    jacobian(3, 4) = -x21*x28 - x25*x31;
    jacobian(3, 5) = -x22*x28 + x26*x31;
    jacobian(4, 1) = -d_1*x0;
    jacobian(4, 2) = -x0*x27;
    jacobian(4, 3) = -x23*x33 + x28*x9;
    jacobian(4, 4) = x15*x28 + x25*x33;
    jacobian(4, 5) = x18*x28 - x26*x33;
    jacobian(5, 2) = x0*x30 + x2*x32;
    jacobian(5, 3) = x19*x33 - x31*x9;
    jacobian(5, 4) = -x15*x31 + x21*x33;
    jacobian(5, 5) = -x18*x31 + x22*x33;
    return;
}

static void computeAngularVelocityJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Matrix<double, 6, 6>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0];
    const double th_1 = theta_input_original[1];
    const double th_2 = theta_input_original[2];
    const double th_3 = theta_input_original[3];
    const double th_4 = theta_input_original[4];
    const double th_5 = theta_input_original[5];
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = -x0;
    const double x2 = std::cos(th_0);
    const double x3 = std::sin(th_1);
    const double x4 = std::cos(th_2);
    const double x5 = x3*x4;
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_1);
    const double x8 = x6*x7;
    const double x9 = -x2*x5 - x2*x8;
    const double x10 = std::cos(th_3);
    const double x11 = std::sin(th_3);
    const double x12 = x4*x7;
    const double x13 = x3*x6;
    const double x14 = x12*x2 - x13*x2;
    const double x15 = std::cos(th_4);
    const double x16 = std::sin(th_4);
    const double x17 = -x0*x5 - x0*x8;
    const double x18 = x0*x12 - x0*x13;
    const double x19 = -x12 + x13;
    const double x20 = -x5 - x8;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x1;
    jacobian(0, 2) = x1;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = x0*x10 - x11*x14;
    jacobian(0, 5) = x15*x9 + x16*(x0*x11 + x10*x14);
    jacobian(1, 1) = x2;
    jacobian(1, 2) = x2;
    jacobian(1, 3) = x17;
    jacobian(1, 4) = -x10*x2 - x11*x18;
    jacobian(1, 5) = x15*x17 + x16*(x10*x18 - x11*x2);
    jacobian(2, 0) = 1;
    jacobian(2, 3) = x19;
    jacobian(2, 4) = -x11*x20;
    jacobian(2, 5) = x10*x16*x20 + x15*x19;
    return;
}

static void computeTransformPointJacobian(const std::array<double, robot_nq>& theta_input_original, Eigen::Vector3d& point_on_ee, Eigen::Matrix<double, 6, 6>& jacobian)
{
    // Extract the variables
    const double th_0 = theta_input_original[0];
    const double th_1 = theta_input_original[1];
    const double th_2 = theta_input_original[2];
    const double th_3 = theta_input_original[3];
    const double th_4 = theta_input_original[4];
    const double th_5 = theta_input_original[5];
    const double p_on_ee_x = point_on_ee[0];
    const double p_on_ee_y = point_on_ee[1];
    const double p_on_ee_z = point_on_ee[2];
    
    // Temp variable for efficiency
    const double x0 = std::cos(th_0);
    const double x1 = p_on_ee_z*x0;
    const double x2 = std::sin(th_1);
    const double x3 = -a_2*x2 + d_1;
    const double x4 = std::sin(th_2);
    const double x5 = x2*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::cos(th_2);
    const double x8 = x6*x7;
    const double x9 = x5 - x8;
    const double x10 = std::sin(th_0);
    const double x11 = x2*x7;
    const double x12 = x4*x6;
    const double x13 = -x10*x11 - x10*x12;
    const double x14 = -x11 - x12;
    const double x15 = a_3*x14 + d_4*x9 + x3;
    const double x16 = -x10*x5 + x10*x8;
    const double x17 = a_2*x6;
    const double x18 = d_3*x0 + x10*x17;
    const double x19 = a_3*x16 + d_4*x13 + x18;
    const double x20 = std::sin(th_3);
    const double x21 = x14*x20;
    const double x22 = std::cos(th_3);
    const double x23 = -x0*x22 - x16*x20;
    const double x24 = std::cos(th_4);
    const double x25 = std::sin(th_4);
    const double x26 = x14*x22*x25 + x24*x9;
    const double x27 = x13*x24 + x25*(-x0*x20 + x16*x22);
    const double x28 = p_on_ee_z*x10;
    const double x29 = -x0*x11 - x0*x12;
    const double x30 = -x0*x5 + x0*x8;
    const double x31 = -d_3*x10 + x0*x17;
    const double x32 = a_3*x30 + d_4*x29 + x31;
    const double x33 = x10*x22 - x20*x30;
    const double x34 = x24*x29 + x25*(x10*x20 + x22*x30);
    const double x35 = -p_on_ee_x*x0 - p_on_ee_y*x10;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -p_on_ee_y;
    jacobian(0, 1) = -d_1*x0 + x1;
    jacobian(0, 2) = -x0*x3 + x1;
    jacobian(0, 3) = -p_on_ee_y*x9 + p_on_ee_z*x13 - x13*x15 + x19*x9;
    jacobian(0, 4) = p_on_ee_y*x21 + p_on_ee_z*x23 - x15*x23 - x19*x21;
    jacobian(0, 5) = -p_on_ee_y*x26 + p_on_ee_z*x27 - x15*x27 + x19*x26;
    jacobian(1, 0) = p_on_ee_x;
    jacobian(1, 1) = -d_1*x10 + x28;
    jacobian(1, 2) = -x10*x3 + x28;
    jacobian(1, 3) = p_on_ee_x*x9 - p_on_ee_z*x29 + x15*x29 - x32*x9;
    jacobian(1, 4) = -p_on_ee_x*x21 - p_on_ee_z*x33 + x15*x33 + x21*x32;
    jacobian(1, 5) = p_on_ee_x*x26 - p_on_ee_z*x34 + x15*x34 - x26*x32;
    jacobian(2, 1) = x35;
    jacobian(2, 2) = x0*x31 + x10*x18 + x35;
    jacobian(2, 3) = -p_on_ee_x*x13 + p_on_ee_y*x29 + x13*x32 - x19*x29;
    jacobian(2, 4) = -p_on_ee_x*x23 + p_on_ee_y*x33 - x19*x33 + x23*x32;
    jacobian(2, 5) = -p_on_ee_x*x27 + p_on_ee_y*x34 - x19*x34 + x27*x32;
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
    
    // Code for explicit solution node 1, solved variable is th_0
    auto ExplicitSolutionNode_node_1_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;
        
        // The explicit solution of root node
        const bool condition_0 = std::fabs(Px) >= zero_tolerance || std::fabs(Py) >= zero_tolerance || std::fabs(d_3) >= zero_tolerance;
        if (condition_0)
        {
            // Temp variable for efficiency
            const double x0 = std::atan2(Px, -Py);
            const double x1 = std::sqrt(std::pow(Px, 2) + std::pow(Py, 2) - std::pow(d_3, 2));
            const double x2 = -d_3;
            // End of temp variables
            
            auto solution_0 = make_raw_solution();
            solution_0[1] = x0 + std::atan2(x1, x2);
            int appended_idx = append_solution_to_queue(solution_0);
            add_input_index_to(2, appended_idx);
        }
        
        const bool condition_1 = std::fabs(Px) >= zero_tolerance || std::fabs(Py) >= zero_tolerance || std::fabs(d_3) >= zero_tolerance;
        if (condition_1)
        {
            // Temp variable for efficiency
            const double x0 = std::atan2(Px, -Py);
            const double x1 = std::sqrt(std::pow(Px, 2) + std::pow(Py, 2) - std::pow(d_3, 2));
            const double x2 = -d_3;
            // End of temp variables
            
            auto solution_1 = make_raw_solution();
            solution_1[1] = x0 + std::atan2(-x1, x2);
            int appended_idx = append_solution_to_queue(solution_1);
            add_input_index_to(2, appended_idx);
        }
        
    };
    // Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_0_processor();
    // Finish code for explicit solution node 0
    
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
            
            const bool condition_0 = 2*std::fabs(a_2*a_3) >= zero_tolerance || 2*std::fabs(a_2*d_4) >= zero_tolerance || std::fabs(-std::pow(Px, 2) - std::pow(Py, 2) - std::pow(Pz, 2) + 2*Pz*d_1 + std::pow(a_2, 2) + std::pow(a_3, 2) - std::pow(d_1, 2) + std::pow(d_3, 2) + std::pow(d_4, 2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_2;
                const double x1 = std::atan2(-d_4*x0, a_3*x0);
                const double x2 = std::pow(a_3, 2);
                const double x3 = std::pow(a_2, 2);
                const double x4 = 4*x3;
                const double x5 = std::pow(d_4, 2);
                const double x6 = std::pow(Px, 2) + std::pow(Py, 2) + std::pow(Pz, 2) - 2*Pz*d_1 + std::pow(d_1, 2) - std::pow(d_3, 2) - x2 - x3 - x5;
                const double x7 = std::sqrt(x2*x4 + x4*x5 - std::pow(x6, 2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x1 + std::atan2(x7, x6);
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = 2*std::fabs(a_2*a_3) >= zero_tolerance || 2*std::fabs(a_2*d_4) >= zero_tolerance || std::fabs(-std::pow(Px, 2) - std::pow(Py, 2) - std::pow(Pz, 2) + 2*Pz*d_1 + std::pow(a_2, 2) + std::pow(a_3, 2) - std::pow(d_1, 2) + std::pow(d_3, 2) + std::pow(d_4, 2)) >= zero_tolerance;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = 2*a_2;
                const double x1 = std::atan2(-d_4*x0, a_3*x0);
                const double x2 = std::pow(a_3, 2);
                const double x3 = std::pow(a_2, 2);
                const double x4 = 4*x3;
                const double x5 = std::pow(d_4, 2);
                const double x6 = std::pow(Px, 2) + std::pow(Py, 2) + std::pow(Pz, 2) - 2*Pz*d_1 + std::pow(d_1, 2) - std::pow(d_3, 2) - x2 - x3 - x5;
                const double x7 = std::sqrt(x2*x4 + x4*x5 - std::pow(x6, 2));
                // End of temp variables
                const double tmp_sol_value = x1 + std::atan2(-x7, x6);
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
            const bool checked_result = std::fabs(Pz - d_1) <= 9.9999999999999995e-7 && std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) <= 9.9999999999999995e-7;
            if (!checked_result)  // To non-degenerate node
                add_input_index_to(5, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    EquationAllZeroDispatcherNode_node_4_processor();
    // Finish code for equation all-zero dispatcher node 4
    
    // Code for explicit solution node 5, solved variable is th_1th_2_soa
    auto ExplicitSolutionNode_node_5_solve_th_1th_2_soa_processor = [&]() -> void
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
            const double th_2 = this_solution[4];
            
            const bool condition_0 = std::fabs(Pz - d_1) >= 9.9999999999999995e-7 || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Pz - d_1;
                const double x1 = -a_2*std::cos(th_2) - a_3;
                const double x2 = a_2*std::sin(th_2) - d_4;
                const double x3 = -Px*std::cos(th_0) - Py*std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*x1 - x2*x3, x0*x2 + x1*x3);
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(6, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_1th_2_soa_processor();
    // Finish code for explicit solution node 5
    
    // Code for non-branch dispatcher node 6
    // Actually, there is no code
    
    // Code for explicit solution node 7, solved variable is th_1
    auto ExplicitSolutionNode_node_7_solve_th_1_processor = [&]() -> void
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
            const double th_1th_2_soa = this_solution[3];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_1th_2_soa - th_2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(8, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_1_processor();
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
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = std::fabs(-r_13*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::cos(th_0) - r_23*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::sin(th_0) - r_33*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_1);
                const double x3 = std::sin(th_2);
                const double x4 = x0*x3 + x1*x2;
                const double x5 = std::acos(-r_13*x4*std::cos(th_0) - r_23*x4*std::sin(th_0) - r_33*(x0*x1 - x2*x3));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[7] = x5;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(10, appended_idx);
            }
            
            const bool condition_1 = std::fabs(-r_13*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::cos(th_0) - r_23*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::sin(th_0) - r_33*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::cos(th_2);
                const double x2 = std::sin(th_1);
                const double x3 = std::sin(th_2);
                const double x4 = x0*x3 + x1*x2;
                const double x5 = std::acos(-r_13*x4*std::cos(th_0) - r_23*x4*std::sin(th_0) - r_33*(x0*x1 - x2*x3));
                // End of temp variables
                const double tmp_sol_value = -x5;
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
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
    
    // Code for solved_variable dispatcher node 10
    auto SolvedVariableDispatcherNode_node_10_processor = [&]() -> void
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
            bool taken_by_degenerate = false;
            const double th_4 = this_solution[7];
            
            const bool degenerate_valid_0 = std::fabs(th_4) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(14, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_4 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(19, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(11, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_10_processor();
    // Finish code for solved_variable dispatcher node 10
    
    // Code for explicit solution node 19, solved variable is negative_th_5_positive_th_3__soa
    auto ExplicitSolutionNode_node_19_solve_negative_th_5_positive_th_3__soa_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x1 + r_21*x0, r_12*x1 - r_22*x0);
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(20, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_19_solve_negative_th_5_positive_th_3__soa_processor();
    // Finish code for explicit solution node 19
    
    // Code for non-branch dispatcher node 20
    // Actually, there is no code
    
    // Code for explicit solution node 21, solved variable is th_3
    auto ExplicitSolutionNode_node_21_solve_th_3_processor = [&]() -> void
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
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(22, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_21_solve_th_3_processor();
    // Finish code for explicit solution node 20
    
    // Code for non-branch dispatcher node 22
    // Actually, there is no code
    
    // Code for explicit solution node 23, solved variable is th_5
    auto ExplicitSolutionNode_node_23_solve_th_5_processor = [&]() -> void
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
            const double negative_th_5_positive_th_3__soa = this_solution[0];
            const double th_3 = this_solution[5];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -negative_th_5_positive_th_3__soa + th_3;
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_23_solve_th_5_processor();
    // Finish code for explicit solution node 22
    
    // Code for explicit solution node 14, solved variable is th_3th_5_soa
    auto ExplicitSolutionNode_node_14_solve_th_3th_5_soa_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_11*x0 - r_21*x1, r_12*x0 - r_22*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(15, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_3th_5_soa_processor();
    // Finish code for explicit solution node 14
    
    // Code for non-branch dispatcher node 15
    // Actually, there is no code
    
    // Code for explicit solution node 16, solved variable is th_3
    auto ExplicitSolutionNode_node_16_solve_th_3_processor = [&]() -> void
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
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(17, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_3_processor();
    // Finish code for explicit solution node 15
    
    // Code for non-branch dispatcher node 17
    // Actually, there is no code
    
    // Code for explicit solution node 18, solved variable is th_5
    auto ExplicitSolutionNode_node_18_solve_th_5_processor = [&]() -> void
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
            const double th_3 = this_solution[5];
            const double th_3th_5_soa = this_solution[6];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_3 + th_3th_5_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_5_processor();
    // Finish code for explicit solution node 17
    
    // Code for explicit solution node 11, solved variable is th_3
    auto ExplicitSolutionNode_node_11_solve_th_3_processor = [&]() -> void
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
            const double th_0 = this_solution[1];
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[7];
            
            const bool condition_0 = std::fabs(r_13*std::sin(th_0) - r_23*std::cos(th_0)) >= zero_tolerance || std::fabs(-r_13*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::cos(th_0) - r_23*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))*std::sin(th_0) + r_33*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))) >= zero_tolerance || std::fabs(std::sin(th_4)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_4);
                const double x1 = std::sin(th_0);
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_1);
                const double x4 = std::cos(th_2);
                const double x5 = std::sin(th_2);
                const double x6 = std::cos(th_1);
                const double x7 = -x3*x5 + x4*x6;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(r_13*x1 - r_23*x2), x0*(r_13*x2*x7 + r_23*x1*x7 - r_33*(x3*x4 + x5*x6)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(12, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_3_processor();
    // Finish code for explicit solution node 11
    
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
            const double th_0 = this_solution[1];
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[4];
            const double th_4 = this_solution[7];
            
            const bool condition_0 = std::fabs(-r_11*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::cos(th_0) - r_21*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::sin(th_0) - r_31*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) >= zero_tolerance || std::fabs(r_12*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::cos(th_0) + r_22*(std::sin(th_1)*std::cos(th_2) + std::sin(th_2)*std::cos(th_1))*std::sin(th_0) + r_32*(-std::sin(th_1)*std::sin(th_2) + std::cos(th_1)*std::cos(th_2))) >= zero_tolerance || std::fabs(std::sin(th_4)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_4);
                const double x1 = std::cos(th_1);
                const double x2 = std::cos(th_2);
                const double x3 = std::sin(th_1);
                const double x4 = std::sin(th_2);
                const double x5 = x1*x2 - x3*x4;
                const double x6 = x1*x4 + x2*x3;
                const double x7 = x6*std::cos(th_0);
                const double x8 = x6*std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(-r_12*x7 - r_22*x8 - r_32*x5), x0*(r_11*x7 + r_21*x8 + r_31*x5));
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
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
        const double value_at_0 = raw_ik_out_i[1];  // th_0
        new_ik_i[0] = value_at_0;
        const double value_at_1 = raw_ik_out_i[2];  // th_1
        new_ik_i[1] = value_at_1;
        const double value_at_2 = raw_ik_out_i[4];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[5];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[7];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[8];  // th_5
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

}; // struct puma_ik

// Code below for debug
void test_ik_solve_puma()
{
    std::array<double, puma_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = puma_ik::computeFK(theta);
    auto ik_output = puma_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = puma_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_puma();
}
