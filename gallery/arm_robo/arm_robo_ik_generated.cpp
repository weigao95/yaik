#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct arm_robo_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 16;
static constexpr int n_tree_nodes = 26;
static constexpr int intermediate_solution_size = 10;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double l_1 = 0.19681;
static constexpr double l_2 = 0.251;
static constexpr double l_3 = 0.145423;

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
    const double d_6 = theta_input_original[5];
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_4);
    const double x1 = std::sin(th_2);
    const double x2 = std::cos(th_0);
    const double x3 = std::cos(th_1);
    const double x4 = x2*x3;
    const double x5 = std::sin(th_0);
    const double x6 = std::sin(th_1);
    const double x7 = x5*x6;
    const double x8 = x4 - x7;
    const double x9 = x1*x8;
    const double x10 = std::cos(th_4);
    const double x11 = std::sin(th_3);
    const double x12 = x2*x6 + x3*x5;
    const double x13 = x11*x12;
    const double x14 = std::cos(th_3);
    const double x15 = std::cos(th_2);
    const double x16 = x15*x8;
    const double x17 = -x13 + x14*x16;
    const double x18 = x12*x14;
    const double x19 = x11*x16 + x18;
    const double x20 = x1*x14;
    const double x21 = x1*x11;
    const double x22 = x1*x12;
    const double x23 = -x4 + x7;
    const double x24 = -x11*x23 + x15*x18;
    const double x25 = x13*x15 + x14*x23;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x9 + x10*x17;
    ee_pose_raw(0, 1) = -x0*x17 - x10*x9;
    ee_pose_raw(0, 2) = x19;
    ee_pose_raw(0, 3) = d_6*x19 + l_1*x2 - l_3*x9;
    ee_pose_raw(1, 0) = -x0*x15 - x10*x20;
    ee_pose_raw(1, 1) = x0*x20 - x10*x15;
    ee_pose_raw(1, 2) = -x21;
    ee_pose_raw(1, 3) = -d_6*x21 - l_2 - l_3*x15;
    ee_pose_raw(2, 0) = -x0*x22 + x10*x24;
    ee_pose_raw(2, 1) = -x0*x24 - x10*x22;
    ee_pose_raw(2, 2) = x25;
    ee_pose_raw(2, 3) = d_6*x25 + l_1*x5 - l_3*x22;
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
    const double d_6 = theta_input_original[5];
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = std::cos(th_1);
    const double x2 = std::sin(th_1);
    const double x3 = std::cos(th_0);
    const double x4 = x0*x1 + x2*x3;
    const double x5 = std::sin(th_2);
    const double x6 = x1*x3;
    const double x7 = x0*x2;
    const double x8 = x6 - x7;
    const double x9 = x5*x8;
    const double x10 = std::cos(th_3);
    const double x11 = std::sin(th_3);
    const double x12 = std::cos(th_2);
    const double x13 = x11*x12;
    const double x14 = x10*x4 + x13*x8;
    const double x15 = x11*x5;
    const double x16 = -x15;
    const double x17 = -x6 + x7;
    const double x18 = x4*x5;
    const double x19 = x10*x17 + x13*x4;
    const double x20 = l_1*x0;
    const double x21 = -l_3*x18 + x20;
    const double x22 = -l_2 - l_3*x12;
    const double x23 = l_1*x3;
    const double x24 = -l_3*x9 + x23;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 2) = x4;
    jacobian(0, 3) = -x9;
    jacobian(0, 4) = x14;
    jacobian(1, 0) = -1;
    jacobian(1, 1) = -1;
    jacobian(1, 3) = -x12;
    jacobian(1, 4) = x16;
    jacobian(2, 2) = x17;
    jacobian(2, 3) = -x18;
    jacobian(2, 4) = x19;
    jacobian(3, 1) = x20;
    jacobian(3, 2) = -l_2*x17;
    jacobian(3, 3) = x12*x21 - x18*x22;
    jacobian(3, 4) = x15*x21 + x19*x22;
    jacobian(3, 5) = x14;
    jacobian(4, 2) = -x17*x23 + x20*x4;
    jacobian(4, 3) = x18*x24 - x21*x9;
    jacobian(4, 4) = x14*x21 - x19*x24;
    jacobian(4, 5) = x16;
    jacobian(5, 1) = -x23;
    jacobian(5, 2) = l_2*x4;
    jacobian(5, 3) = -x12*x24 + x22*x9;
    jacobian(5, 4) = -x14*x22 - x15*x24;
    jacobian(5, 5) = x19;
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
    const double d_6 = theta_input_original[5];
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = std::cos(th_1);
    const double x2 = std::sin(th_1);
    const double x3 = std::cos(th_0);
    const double x4 = x0*x1 + x2*x3;
    const double x5 = std::sin(th_2);
    const double x6 = x1*x3;
    const double x7 = x0*x2;
    const double x8 = x6 - x7;
    const double x9 = std::cos(th_3);
    const double x10 = std::sin(th_3);
    const double x11 = std::cos(th_2);
    const double x12 = x10*x11;
    const double x13 = -x6 + x7;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 2) = x4;
    jacobian(0, 3) = -x5*x8;
    jacobian(0, 4) = x12*x8 + x4*x9;
    jacobian(1, 0) = -1;
    jacobian(1, 1) = -1;
    jacobian(1, 3) = -x11;
    jacobian(1, 4) = -x10*x5;
    jacobian(2, 2) = x13;
    jacobian(2, 3) = -x4*x5;
    jacobian(2, 4) = x12*x4 + x13*x9;
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
    const double d_6 = theta_input_original[5];
    const double p_on_ee_x = point_on_ee[0];
    const double p_on_ee_y = point_on_ee[1];
    const double p_on_ee_z = point_on_ee[2];
    
    // Temp variable for efficiency
    const double x0 = -p_on_ee_z;
    const double x1 = std::sin(th_0);
    const double x2 = l_1*x1;
    const double x3 = std::sin(th_1);
    const double x4 = x1*x3;
    const double x5 = std::cos(th_0);
    const double x6 = std::cos(th_1);
    const double x7 = x5*x6;
    const double x8 = x4 - x7;
    const double x9 = std::cos(th_2);
    const double x10 = std::sin(th_2);
    const double x11 = x1*x6 + x3*x5;
    const double x12 = p_on_ee_y*x11;
    const double x13 = x10*x11;
    const double x14 = -l_3*x13 + x2;
    const double x15 = -l_2 - l_3*x9;
    const double x16 = std::sin(th_3);
    const double x17 = x10*x16;
    const double x18 = std::cos(th_3);
    const double x19 = x16*x9;
    const double x20 = x11*x19 + x18*x8;
    const double x21 = -x4 + x7;
    const double x22 = x11*x18 + x19*x21;
    const double x23 = l_1*x5;
    const double x24 = x10*x21;
    const double x25 = -l_3*x24 + x23;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = x0;
    jacobian(0, 1) = x0 + x2;
    jacobian(0, 2) = -l_2*x8 - p_on_ee_y*x8;
    jacobian(0, 3) = -p_on_ee_z*x9 + x10*x12 - x13*x15 + x14*x9;
    jacobian(0, 4) = -p_on_ee_y*x20 - p_on_ee_z*x17 + x14*x17 + x15*x20;
    jacobian(0, 5) = x22;
    jacobian(1, 2) = p_on_ee_x*x8 - p_on_ee_z*x11 + x11*x2 - x23*x8;
    jacobian(1, 3) = -p_on_ee_x*x13 + p_on_ee_z*x24 + x13*x25 - x14*x24;
    jacobian(1, 4) = p_on_ee_x*x20 - p_on_ee_z*x22 + x14*x22 - x20*x25;
    jacobian(1, 5) = -x17;
    jacobian(2, 0) = p_on_ee_x;
    jacobian(2, 1) = p_on_ee_x - x23;
    jacobian(2, 2) = l_2*x11 + x12;
    jacobian(2, 3) = p_on_ee_x*x9 - p_on_ee_y*x24 + x15*x24 - x25*x9;
    jacobian(2, 4) = p_on_ee_x*x17 + p_on_ee_y*x22 - x15*x22 - x17*x25;
    jacobian(2, 5) = x20;
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
    
    // Code for polynomial solution node 0, solved variable is th_0
    auto PolynomialSolutionNode_node_1_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;
        
        // The polynomial solution of root node
        const double poly_coefficient_0_num = (std::pow(inv_Px, 2) - 2*inv_Px*l_1*r_11 - 2*inv_Px*l_2*r_21 + std::pow(inv_Py, 2) - 2*inv_Py*l_1*r_12 - 2*inv_Py*l_2*r_22 + std::pow(l_1, 2)*std::pow(r_11, 2) + std::pow(l_1, 2)*std::pow(r_12, 2) + 2*l_1*l_2*r_11*r_21 + 2*l_1*l_2*r_12*r_22 + std::pow(l_2, 2)*std::pow(r_21, 2) + std::pow(l_2, 2)*std::pow(r_22, 2) - std::pow(l_3, 2))*(std::pow(inv_Px, 2) + 2*inv_Px*l_1*r_11 - 2*inv_Px*l_2*r_21 + std::pow(inv_Py, 2) + 2*inv_Py*l_1*r_12 - 2*inv_Py*l_2*r_22 + std::pow(l_1, 2)*std::pow(r_11, 2) + std::pow(l_1, 2)*std::pow(r_12, 2) - 2*l_1*l_2*r_11*r_21 - 2*l_1*l_2*r_12*r_22 + std::pow(l_2, 2)*std::pow(r_21, 2) + std::pow(l_2, 2)*std::pow(r_22, 2) - std::pow(l_3, 2));
        const double poly_coefficient_0_denom = 1;
        const double poly_coefficient_0 = poly_coefficient_0_num / poly_coefficient_0_denom;
        const double poly_coefficient_1_num = 4*l_1*(std::pow(inv_Px, 3)*r_31 + std::pow(inv_Px, 2)*inv_Py*r_32 - 3*std::pow(inv_Px, 2)*l_2*r_21*r_31 - std::pow(inv_Px, 2)*l_2*r_22*r_32 + inv_Px*std::pow(inv_Py, 2)*r_31 - 2*inv_Px*inv_Py*l_2*r_21*r_32 - 2*inv_Px*inv_Py*l_2*r_22*r_31 - inv_Px*std::pow(l_1, 2)*std::pow(r_11, 2)*r_31 - 2*inv_Px*std::pow(l_1, 2)*r_11*r_12*r_32 + inv_Px*std::pow(l_1, 2)*std::pow(r_12, 2)*r_31 + 3*inv_Px*std::pow(l_2, 2)*std::pow(r_21, 2)*r_31 + 2*inv_Px*std::pow(l_2, 2)*r_21*r_22*r_32 + inv_Px*std::pow(l_2, 2)*std::pow(r_22, 2)*r_31 - inv_Px*std::pow(l_3, 2)*r_31 + std::pow(inv_Py, 3)*r_32 - std::pow(inv_Py, 2)*l_2*r_21*r_31 - 3*std::pow(inv_Py, 2)*l_2*r_22*r_32 + inv_Py*std::pow(l_1, 2)*std::pow(r_11, 2)*r_32 - 2*inv_Py*std::pow(l_1, 2)*r_11*r_12*r_31 - inv_Py*std::pow(l_1, 2)*std::pow(r_12, 2)*r_32 + inv_Py*std::pow(l_2, 2)*std::pow(r_21, 2)*r_32 + 2*inv_Py*std::pow(l_2, 2)*r_21*r_22*r_31 + 3*inv_Py*std::pow(l_2, 2)*std::pow(r_22, 2)*r_32 - inv_Py*std::pow(l_3, 2)*r_32 + std::pow(l_1, 2)*l_2*std::pow(r_11, 2)*r_21*r_31 - std::pow(l_1, 2)*l_2*std::pow(r_11, 2)*r_22*r_32 + 2*std::pow(l_1, 2)*l_2*r_11*r_12*r_21*r_32 + 2*std::pow(l_1, 2)*l_2*r_11*r_12*r_22*r_31 - std::pow(l_1, 2)*l_2*std::pow(r_12, 2)*r_21*r_31 + std::pow(l_1, 2)*l_2*std::pow(r_12, 2)*r_22*r_32 - std::pow(l_2, 3)*std::pow(r_21, 3)*r_31 - std::pow(l_2, 3)*std::pow(r_21, 2)*r_22*r_32 - std::pow(l_2, 3)*r_21*std::pow(r_22, 2)*r_31 - std::pow(l_2, 3)*std::pow(r_22, 3)*r_32 + l_2*std::pow(l_3, 2)*r_21*r_31 + l_2*std::pow(l_3, 2)*r_22*r_32);
        const double poly_coefficient_1_denom = 1;
        const double poly_coefficient_1 = poly_coefficient_1_num / poly_coefficient_1_denom;
        const double poly_coefficient_2_num = 2*std::pow(l_1, 2)*(std::pow(inv_Px, 2)*std::pow(r_11, 2) - std::pow(inv_Px, 2)*std::pow(r_12, 2) + 3*std::pow(inv_Px, 2)*std::pow(r_31, 2) + std::pow(inv_Px, 2)*std::pow(r_32, 2) + 4*inv_Px*inv_Py*r_11*r_12 + 4*inv_Px*inv_Py*r_31*r_32 - 2*inv_Px*l_2*std::pow(r_11, 2)*r_21 - 4*inv_Px*l_2*r_11*r_12*r_22 + 2*inv_Px*l_2*std::pow(r_12, 2)*r_21 - 6*inv_Px*l_2*r_21*std::pow(r_31, 2) - 2*inv_Px*l_2*r_21*std::pow(r_32, 2) - 4*inv_Px*l_2*r_22*r_31*r_32 - std::pow(inv_Py, 2)*std::pow(r_11, 2) + std::pow(inv_Py, 2)*std::pow(r_12, 2) + std::pow(inv_Py, 2)*std::pow(r_31, 2) + 3*std::pow(inv_Py, 2)*std::pow(r_32, 2) + 2*inv_Py*l_2*std::pow(r_11, 2)*r_22 - 4*inv_Py*l_2*r_11*r_12*r_21 - 2*inv_Py*l_2*std::pow(r_12, 2)*r_22 - 4*inv_Py*l_2*r_21*r_31*r_32 - 2*inv_Py*l_2*r_22*std::pow(r_31, 2) - 6*inv_Py*l_2*r_22*std::pow(r_32, 2) - std::pow(l_1, 2)*std::pow(r_11, 4) - 2*std::pow(l_1, 2)*std::pow(r_11, 2)*std::pow(r_12, 2) - std::pow(l_1, 2)*std::pow(r_11, 2)*std::pow(r_31, 2) + std::pow(l_1, 2)*std::pow(r_11, 2)*std::pow(r_32, 2) - 4*std::pow(l_1, 2)*r_11*r_12*r_31*r_32 - std::pow(l_1, 2)*std::pow(r_12, 4) + std::pow(l_1, 2)*std::pow(r_12, 2)*std::pow(r_31, 2) - std::pow(l_1, 2)*std::pow(r_12, 2)*std::pow(r_32, 2) + std::pow(l_2, 2)*std::pow(r_11, 2)*std::pow(r_21, 2) - std::pow(l_2, 2)*std::pow(r_11, 2)*std::pow(r_22, 2) + 4*std::pow(l_2, 2)*r_11*r_12*r_21*r_22 - std::pow(l_2, 2)*std::pow(r_12, 2)*std::pow(r_21, 2) + std::pow(l_2, 2)*std::pow(r_12, 2)*std::pow(r_22, 2) + 3*std::pow(l_2, 2)*std::pow(r_21, 2)*std::pow(r_31, 2) + std::pow(l_2, 2)*std::pow(r_21, 2)*std::pow(r_32, 2) + 4*std::pow(l_2, 2)*r_21*r_22*r_31*r_32 + std::pow(l_2, 2)*std::pow(r_22, 2)*std::pow(r_31, 2) + 3*std::pow(l_2, 2)*std::pow(r_22, 2)*std::pow(r_32, 2) + std::pow(l_3, 2)*std::pow(r_11, 2) + std::pow(l_3, 2)*std::pow(r_12, 2) - std::pow(l_3, 2)*std::pow(r_31, 2) - std::pow(l_3, 2)*std::pow(r_32, 2));
        const double poly_coefficient_2_denom = 1;
        const double poly_coefficient_2 = poly_coefficient_2_num / poly_coefficient_2_denom;
        const double poly_coefficient_3_num = 4*std::pow(l_1, 3)*(inv_Px*std::pow(r_11, 2)*r_31 + 2*inv_Px*r_11*r_12*r_32 - inv_Px*std::pow(r_12, 2)*r_31 + inv_Px*std::pow(r_31, 3) + inv_Px*r_31*std::pow(r_32, 2) - inv_Py*std::pow(r_11, 2)*r_32 + 2*inv_Py*r_11*r_12*r_31 + inv_Py*std::pow(r_12, 2)*r_32 + inv_Py*std::pow(r_31, 2)*r_32 + inv_Py*std::pow(r_32, 3) - l_2*std::pow(r_11, 2)*r_21*r_31 + l_2*std::pow(r_11, 2)*r_22*r_32 - 2*l_2*r_11*r_12*r_21*r_32 - 2*l_2*r_11*r_12*r_22*r_31 + l_2*std::pow(r_12, 2)*r_21*r_31 - l_2*std::pow(r_12, 2)*r_22*r_32 - l_2*r_21*std::pow(r_31, 3) - l_2*r_21*r_31*std::pow(r_32, 2) - l_2*r_22*std::pow(r_31, 2)*r_32 - l_2*r_22*std::pow(r_32, 3));
        const double poly_coefficient_3_denom = 1;
        const double poly_coefficient_3 = poly_coefficient_3_num / poly_coefficient_3_denom;
        const double poly_coefficient_4_num = std::pow(l_1, 4)*(std::pow(r_11, 2) - 2*r_11*r_32 + std::pow(r_12, 2) + 2*r_12*r_31 + std::pow(r_31, 2) + std::pow(r_32, 2))*(std::pow(r_11, 2) + 2*r_11*r_32 + std::pow(r_12, 2) - 2*r_12*r_31 + std::pow(r_31, 2) + std::pow(r_32, 2));
        const double poly_coefficient_4_denom = 1;
        const double poly_coefficient_4 = poly_coefficient_4_num / poly_coefficient_4_denom;
        std::array<double, 4 + 1> p_coefficients;
        p_coefficients[4] = poly_coefficient_0;
        p_coefficients[3] = poly_coefficient_1;
        p_coefficients[2] = poly_coefficient_2;
        p_coefficients[1] = poly_coefficient_3;
        p_coefficients[0] = poly_coefficient_4;
        
        // Invoke the solver. Note that p_coefficient[0] is the highest order
        const auto poly_roots = computePolynomialRealRoots<4>(p_coefficients);
        
        // Result collection
        for(int root_idx = 0; root_idx < poly_roots.size(); root_idx++)
        {
            const auto& this_root_record = poly_roots[root_idx];
            if(!this_root_record.is_valid)
                continue;
            const double this_root = this_root_record.value;
            if (std::abs(this_root) > 1)
                continue;
            const double first_angle = std::asin(this_root);
            const double second_angle = M_PI - std::asin(this_root);
            auto solution_0 = make_raw_solution();
            solution_0[2] = first_angle;
            auto solution_1 = make_raw_solution();
            solution_1[2] = second_angle;
            int appended_idx_0 = append_solution_to_queue(solution_0);
            int appended_idx_1 = append_solution_to_queue(solution_1);
            add_input_index_to(2, appended_idx_0);
            add_input_index_to(2, appended_idx_1);
        }
    };
    
    // Invoke the processor
    PolynomialSolutionNode_node_1_solve_th_0_processor();
    // Finish code for polynomial solution node 0
    
    // Code for non-branch dispatcher node 2
    // Actually, there is no code
    
    // Code for explicit solution node 3, solved variable is d_6
    auto ExplicitSolutionNode_node_3_solve_d_6_processor = [&]() -> void
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
            const double th_0 = this_solution[2];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -inv_Pz - l_1*r_13*std::cos(th_0) - l_1*r_33*std::sin(th_0) + l_2*r_23;
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
    ExplicitSolutionNode_node_3_solve_d_6_processor();
    // Finish code for explicit solution node 2
    
    // Code for non-branch dispatcher node 4
    // Actually, there is no code
    
    // Code for explicit solution node 5, solved variable is th_4
    auto ExplicitSolutionNode_node_5_solve_th_4_processor = [&]() -> void
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
            const double th_0 = this_solution[2];
            
            const bool condition_0 = std::fabs(l_3) >= zero_tolerance || std::fabs(inv_Px + l_1*(r_11*std::cos(th_0) + r_31*std::sin(th_0)) - l_2*r_21) >= zero_tolerance || std::fabs(inv_Py + l_1*(r_12*std::cos(th_0) + r_32*std::sin(th_0)) - l_2*r_22) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/l_3;
                const double x1 = std::cos(th_0);
                const double x2 = std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(-inv_Px - l_1*(r_11*x1 + r_31*x2) + l_2*r_21), x0*(-inv_Py - l_1*(r_12*x1 + r_32*x2) + l_2*r_22));
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
                add_input_index_to(6, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_5_solve_th_4_processor();
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
            const double d_6 = this_solution[0];
            
            const bool condition_0 = std::fabs((Py - d_6*r_23 + l_2)/l_3) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::acos((-Py + d_6*r_23 - l_2)/l_3);
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[7] = x0;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(8, appended_idx);
            }
            
            const bool condition_1 = std::fabs((Py - d_6*r_23 + l_2)/l_3) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::acos((-Py + d_6*r_23 - l_2)/l_3);
                // End of temp variables
                const double tmp_sol_value = -x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
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
            const double th_2 = this_solution[7];
            
            const bool degenerate_valid_0 = std::fabs(th_2) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(14, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_2 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(21, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(9, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor();
    // Finish code for solved_variable dispatcher node 8
    
    // Code for explicit solution node 21, solved variable is negative_th_3_positive_th_1__soa
    auto ExplicitSolutionNode_node_21_solve_negative_th_3_positive_th_1__soa_processor = [&]() -> void
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
            const double th_0 = this_solution[2];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_13*std::sin(th_0) - r_33*std::cos(th_0)) >= zero_tolerance || std::fabs(r_13*std::cos(th_0) + r_33*std::sin(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_13*x0 + r_33*x1, r_13*x1 - r_33*x0);
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
    ExplicitSolutionNode_node_21_solve_negative_th_3_positive_th_1__soa_processor();
    // Finish code for explicit solution node 21
    
    // Code for non-branch dispatcher node 22
    // Actually, there is no code
    
    // Code for explicit solution node 23, solved variable is th_1
    auto ExplicitSolutionNode_node_23_solve_th_1_processor = [&]() -> void
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
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
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
    // Finish code for explicit solution node 22
    
    // Code for non-branch dispatcher node 24
    // Actually, there is no code
    
    // Code for explicit solution node 25, solved variable is th_3
    auto ExplicitSolutionNode_node_25_solve_th_3_processor = [&]() -> void
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
            const double negative_th_3_positive_th_1__soa = this_solution[1];
            const double th_1 = this_solution[5];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -negative_th_3_positive_th_1__soa + th_1;
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_3_processor();
    // Finish code for explicit solution node 24
    
    // Code for explicit solution node 14, solved variable is th_0th_1th_3_soa
    auto ExplicitSolutionNode_node_14_solve_th_0th_1th_3_soa_processor = [&]() -> void
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
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_13) >= zero_tolerance || std::fabs(r_33) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_13, -r_33);
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(15, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_14_solve_th_0th_1th_3_soa_processor();
    // Finish code for explicit solution node 14
    
    // Code for non-branch dispatcher node 15
    // Actually, there is no code
    
    // Code for explicit solution node 16, solved variable is th_1th_3_soa
    auto ExplicitSolutionNode_node_16_solve_th_1th_3_soa_processor = [&]() -> void
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
            const double th_0 = this_solution[2];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_13*std::sin(th_0) - r_33*std::cos(th_0)) >= zero_tolerance || std::fabs(r_13*std::cos(th_0) + r_33*std::sin(th_0)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_13*x0 + r_33*x1, r_13*x1 - r_33*x0);
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(17, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_1th_3_soa_processor();
    // Finish code for explicit solution node 15
    
    // Code for non-branch dispatcher node 17
    // Actually, there is no code
    
    // Code for explicit solution node 18, solved variable is th_1
    auto ExplicitSolutionNode_node_18_solve_th_1_processor = [&]() -> void
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
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(19, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_18_solve_th_1_processor();
    // Finish code for explicit solution node 17
    
    // Code for non-branch dispatcher node 19
    // Actually, there is no code
    
    // Code for explicit solution node 20, solved variable is th_3
    auto ExplicitSolutionNode_node_20_solve_th_3_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(19);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(19);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 20
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[5];
            const double th_1th_3_soa = this_solution[6];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_1 + th_1th_3_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_20_solve_th_3_processor();
    // Finish code for explicit solution node 19
    
    // Code for explicit solution node 9, solved variable is th_3
    auto ExplicitSolutionNode_node_9_solve_th_3_processor = [&]() -> void
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
            const double th_2 = this_solution[7];
            const double th_4 = this_solution[9];
            
            const bool condition_0 = std::fabs(r_23) >= zero_tolerance || std::fabs(r_21*std::cos(th_4) - r_22*std::sin(th_4)) >= zero_tolerance || std::fabs(std::sin(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_2);
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_23*x0, x0*(-r_21*std::cos(th_4) + r_22*std::sin(th_4)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
                add_input_index_to(10, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_3_processor();
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
            const double d_6 = this_solution[0];
            const double th_0 = this_solution[2];
            const double th_2 = this_solution[7];
            
            const bool condition_0 = std::fabs(l_3*std::sin(th_2)) >= zero_tolerance || std::fabs(Px*std::sin(th_0) - Pz*std::cos(th_0) + d_6*(-r_13*std::sin(th_0) + r_33*std::cos(th_0))) >= zero_tolerance || std::fabs(Px*std::cos(th_0) + Pz*std::sin(th_0) - d_6*(r_13*std::cos(th_0) + r_33*std::sin(th_0)) - l_1) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_0);
                const double x1 = std::cos(th_0);
                const double x2 = 1/(l_3*std::sin(th_2));
                // End of temp variables
                const double tmp_sol_value = std::atan2(x2*(Px*x0 - Pz*x1 + d_6*(-r_13*x0 + r_33*x1)), x2*(-Px*x1 - Pz*x0 + d_6*(r_13*x1 + r_33*x0) + l_1));
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
    ExplicitSolutionNode_node_11_solve_th_1_processor();
    // Finish code for explicit solution node 10
    
    // Code for non-branch dispatcher node 12
    // Actually, there is no code
    
    // Code for explicit solution node 13, solved variable is th_0th_1_soa
    auto ExplicitSolutionNode_node_13_solve_th_0th_1_soa_processor = [&]() -> void
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
            const double th_0 = this_solution[2];
            const double th_1 = this_solution[5];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_0 + th_1;
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_0th_1_soa_processor();
    // Finish code for explicit solution node 12
    
    // Collect the output
    for(int i = 0; i < solution_queue.size(); i++)
    {
        if(!solution_queue.solutions_validity[i])
            continue;
        const auto& raw_ik_out_i = solution_queue.get_solution(i);
        std::array<double, robot_nq> new_ik_i;
        const double value_at_0 = raw_ik_out_i[2];  // th_0
        new_ik_i[0] = value_at_0;
        const double value_at_1 = raw_ik_out_i[5];  // th_1
        new_ik_i[1] = value_at_1;
        const double value_at_2 = raw_ik_out_i[7];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[8];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[9];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[0];  // d_6
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

}; // struct arm_robo_ik

// Code below for debug
void test_ik_solve_arm_robo()
{
    std::array<double, arm_robo_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = arm_robo_ik::computeFK(theta);
    auto ik_output = arm_robo_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = arm_robo_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_arm_robo();
}
