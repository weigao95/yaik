#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct ur_5_ik {

// Constants for solver
static constexpr int robot_nq = 6;
static constexpr int max_n_solutions = 16;
static constexpr int n_tree_nodes = 18;
static constexpr int intermediate_solution_size = 9;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double a_1 = 0.425;
static constexpr double a_3 = 0.39225;
static constexpr double d_0 = 0.220941;
static constexpr double d_2 = -0.1719;
static constexpr double d_4 = 0.060108999999999996;
static constexpr double d_5 = 0.09465;
static constexpr double d_6 = 0.0823;
static constexpr double pre_transform_special_symbol_23 = 0.089159;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = -0.0;
static constexpr double th_2_offset_original2raw = 0.0;
static constexpr double th_3_offset_original2raw = -0.0;
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
    ee_transformed(0, 0) = 1.0*r_11;
    ee_transformed(0, 1) = 1.0*r_12 + 6.1232339957367697e-17*r_13;
    ee_transformed(0, 2) = 6.1232339957367697e-17*r_12 - 1.0*r_13;
    ee_transformed(0, 3) = -1.0*Px;
    ee_transformed(1, 0) = 1.0*r_21;
    ee_transformed(1, 1) = 1.0*r_22 + 6.1232339957367697e-17*r_23;
    ee_transformed(1, 2) = 6.1232339957367697e-17*r_22 - 1.0*r_23;
    ee_transformed(1, 3) = -1.0*Py;
    ee_transformed(2, 0) = -1.0*r_31;
    ee_transformed(2, 1) = -1.0*r_32 - 6.1232339957367697e-17*r_33;
    ee_transformed(2, 2) = -6.1232339957367697e-17*r_32 + 1.0*r_33;
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
    ee_transformed(0, 0) = 1.0*r_11;
    ee_transformed(0, 1) = 1.0*r_12 + 6.1232339957367697e-17*r_13;
    ee_transformed(0, 2) = 6.1232339957367697e-17*r_12 - 1.0*r_13;
    ee_transformed(0, 3) = -1.0*Px;
    ee_transformed(1, 0) = 1.0*r_21;
    ee_transformed(1, 1) = 1.0*r_22 + 6.1232339957367697e-17*r_23;
    ee_transformed(1, 2) = 6.1232339957367697e-17*r_22 - 1.0*r_23;
    ee_transformed(1, 3) = -1.0*Py;
    ee_transformed(2, 0) = -1.0*r_31;
    ee_transformed(2, 1) = -1.0*r_32 - 6.1232339957367697e-17*r_33;
    ee_transformed(2, 2) = -6.1232339957367697e-17*r_32 + 1.0*r_33;
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
    const double x1 = std::sin(th_3);
    const double x2 = std::cos(th_0);
    const double x3 = std::cos(th_1);
    const double x4 = std::cos(th_2);
    const double x5 = x3*x4;
    const double x6 = std::sin(th_1);
    const double x7 = std::sin(th_2);
    const double x8 = x6*x7;
    const double x9 = x2*x5 - x2*x8;
    const double x10 = std::cos(th_3);
    const double x11 = x4*x6;
    const double x12 = x3*x7;
    const double x13 = -x11*x2 - x12*x2;
    const double x14 = -x1*x9 + x10*x13;
    const double x15 = std::cos(th_5);
    const double x16 = std::sin(th_0);
    const double x17 = std::sin(th_4);
    const double x18 = std::cos(th_4);
    const double x19 = x1*x13 + x10*x9;
    const double x20 = x16*x17 + x18*x19;
    const double x21 = x16*x18 - x17*x19;
    const double x22 = a_1*x3;
    const double x23 = x16*x5 - x16*x8;
    const double x24 = -x11*x16 - x12*x16;
    const double x25 = -x1*x23 + x10*x24;
    const double x26 = x1*x24 + x10*x23;
    const double x27 = -x17*x2 + x18*x26;
    const double x28 = -x17*x26 - x18*x2;
    const double x29 = -x5 + x8;
    const double x30 = -x11 - x12;
    const double x31 = -x1*x30 + x10*x29;
    const double x32 = x1*x29 + x10*x30;
    const double x33 = x18*x32;
    const double x34 = x17*x32;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x14 + x15*x20;
    ee_pose_raw(0, 1) = -x0*x20 - x14*x15;
    ee_pose_raw(0, 2) = x21;
    ee_pose_raw(0, 3) = a_3*x9 - d_0*x16 - d_2*x16 - d_4*x16 + d_5*x14 + d_6*x21 + x2*x22;
    ee_pose_raw(1, 0) = -x0*x25 + x15*x27;
    ee_pose_raw(1, 1) = -x0*x27 - x15*x25;
    ee_pose_raw(1, 2) = x28;
    ee_pose_raw(1, 3) = a_3*x23 + d_0*x2 + d_2*x2 + d_4*x2 + d_5*x25 + d_6*x28 + x16*x22;
    ee_pose_raw(2, 0) = -x0*x31 + x15*x33;
    ee_pose_raw(2, 1) = -x0*x33 - x15*x31;
    ee_pose_raw(2, 2) = -x34;
    ee_pose_raw(2, 3) = -a_1*x6 + a_3*x30 + d_5*x31 - d_6*x34;
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
    const double x1 = std::cos(th_3);
    const double x2 = std::sin(th_1);
    const double x3 = std::cos(th_2);
    const double x4 = 1.0*std::cos(th_0);
    const double x5 = x3*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::sin(th_2);
    const double x8 = x4*x7;
    const double x9 = x2*x5 + x6*x8;
    const double x10 = std::sin(th_3);
    const double x11 = x2*x8 - x5*x6;
    const double x12 = x1*x9 - x10*x11;
    const double x13 = std::cos(th_4);
    const double x14 = std::sin(th_4);
    const double x15 = -x0*x13 - x14*(x1*x11 + x10*x9);
    const double x16 = -x4;
    const double x17 = x0*x3;
    const double x18 = x0*x7;
    const double x19 = x17*x2 + x18*x6;
    const double x20 = -x17*x6 + x18*x2;
    const double x21 = x1*x19 - x10*x20;
    const double x22 = x13*x4 - x14*(x1*x20 + x10*x19);
    const double x23 = 1.0*x7;
    const double x24 = 1.0*x3;
    const double x25 = x2*x23 - x24*x6;
    const double x26 = -x2*x24 - x23*x6;
    const double x27 = x1*x25 - x10*x26;
    const double x28 = x14*(x1*x26 + x10*x25);
    const double x29 = -1.0*a_1*x2 + pre_transform_special_symbol_23;
    const double x30 = a_3*x26 + x29;
    const double x31 = d_5*x27 + x30;
    const double x32 = a_1*x6;
    const double x33 = -d_0*x4 - d_2*x4 - x0*x32;
    const double x34 = a_3*x20 - d_4*x4 + x33;
    const double x35 = d_5*x21 + x34;
    const double x36 = -d_6*x28 + x31;
    const double x37 = d_6*x22 + x35;
    const double x38 = d_0*x0 + d_2*x0 - x32*x4;
    const double x39 = a_3*x11 + d_4*x0 + x38;
    const double x40 = d_5*x12 + x39;
    const double x41 = d_6*x15 + x40;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x0;
    jacobian(0, 2) = x0;
    jacobian(0, 3) = x0;
    jacobian(0, 4) = x12;
    jacobian(0, 5) = x15;
    jacobian(1, 1) = x16;
    jacobian(1, 2) = x16;
    jacobian(1, 3) = x16;
    jacobian(1, 4) = x21;
    jacobian(1, 5) = x22;
    jacobian(2, 0) = 1.0;
    jacobian(2, 4) = x27;
    jacobian(2, 5) = -x28;
    jacobian(3, 1) = pre_transform_special_symbol_23*x4;
    jacobian(3, 2) = x29*x4;
    jacobian(3, 3) = x30*x4;
    jacobian(3, 4) = -x21*x31 + x27*x35;
    jacobian(3, 5) = -x22*x36 - x28*x37;
    jacobian(4, 1) = pre_transform_special_symbol_23*x0;
    jacobian(4, 2) = x0*x29;
    jacobian(4, 3) = x0*x30;
    jacobian(4, 4) = x12*x31 - x27*x40;
    jacobian(4, 5) = x15*x36 + x28*x41;
    jacobian(5, 2) = -x0*x33 - x38*x4;
    jacobian(5, 3) = -x0*x34 - x39*x4;
    jacobian(5, 4) = -x12*x35 + x21*x40;
    jacobian(5, 5) = -x15*x37 + x22*x41;
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
    const double x1 = std::cos(th_3);
    const double x2 = std::sin(th_1);
    const double x3 = std::cos(th_2);
    const double x4 = 1.0*std::cos(th_0);
    const double x5 = x3*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::sin(th_2);
    const double x8 = x4*x7;
    const double x9 = x2*x5 + x6*x8;
    const double x10 = std::sin(th_3);
    const double x11 = x2*x8 - x5*x6;
    const double x12 = std::cos(th_4);
    const double x13 = std::sin(th_4);
    const double x14 = -x4;
    const double x15 = x0*x3;
    const double x16 = x0*x7;
    const double x17 = x15*x2 + x16*x6;
    const double x18 = -x15*x6 + x16*x2;
    const double x19 = 1.0*x7;
    const double x20 = 1.0*x3;
    const double x21 = x19*x2 - x20*x6;
    const double x22 = -x19*x6 - x2*x20;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = x0;
    jacobian(0, 2) = x0;
    jacobian(0, 3) = x0;
    jacobian(0, 4) = x1*x9 - x10*x11;
    jacobian(0, 5) = -x0*x12 - x13*(x1*x11 + x10*x9);
    jacobian(1, 1) = x14;
    jacobian(1, 2) = x14;
    jacobian(1, 3) = x14;
    jacobian(1, 4) = x1*x17 - x10*x18;
    jacobian(1, 5) = x12*x4 - x13*(x1*x18 + x10*x17);
    jacobian(2, 0) = 1.0;
    jacobian(2, 4) = x1*x21 - x10*x22;
    jacobian(2, 5) = -x13*(x1*x22 + x10*x21);
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
    const double x3 = -p_on_ee_z*x2;
    const double x4 = std::sin(th_1);
    const double x5 = 1.0*x4;
    const double x6 = -a_1*x5 + pre_transform_special_symbol_23;
    const double x7 = std::cos(th_2);
    const double x8 = x5*x7;
    const double x9 = std::sin(th_2);
    const double x10 = std::cos(th_1);
    const double x11 = 1.0*x10;
    const double x12 = x11*x9;
    const double x13 = -x12 - x8;
    const double x14 = a_3*x13 + x6;
    const double x15 = std::cos(th_3);
    const double x16 = x5*x9;
    const double x17 = x11*x7;
    const double x18 = x16 - x17;
    const double x19 = std::sin(th_3);
    const double x20 = -x13*x19 + x15*x18;
    const double x21 = std::sin(th_0);
    const double x22 = x12*x21 + x21*x8;
    const double x23 = x16*x21 - x17*x21;
    const double x24 = x15*x22 - x19*x23;
    const double x25 = d_5*x20 + x14;
    const double x26 = -a_1*x11*x21 - d_0*x2 - d_2*x2;
    const double x27 = a_3*x23 - d_4*x2 + x26;
    const double x28 = d_5*x24 + x27;
    const double x29 = std::sin(th_4);
    const double x30 = x29*(x13*x15 + x18*x19);
    const double x31 = std::cos(th_4);
    const double x32 = x2*x31 - x29*(x15*x23 + x19*x22);
    const double x33 = -d_6*x30 + x25;
    const double x34 = d_6*x32 + x28;
    const double x35 = 1.0*p_on_ee_x;
    const double x36 = 1.0*x21;
    const double x37 = -p_on_ee_z*x36;
    const double x38 = x2*x7;
    const double x39 = x2*x9;
    const double x40 = x10*x39 + x38*x4;
    const double x41 = -x10*x38 + x39*x4;
    const double x42 = x15*x40 - x19*x41;
    const double x43 = -a_1*x10*x2 + d_0*x36 + d_2*x36;
    const double x44 = a_3*x41 + d_4*x36 + x43;
    const double x45 = d_5*x42 + x44;
    const double x46 = -x29*(x15*x41 + x19*x40) - x31*x36;
    const double x47 = d_6*x46 + x45;
    const double x48 = x0*x21 + x1*x35;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -x0;
    jacobian(0, 1) = pre_transform_special_symbol_23*x2 + x3;
    jacobian(0, 2) = x2*x6 + x3;
    jacobian(0, 3) = x14*x2 + x3;
    jacobian(0, 4) = -p_on_ee_y*x20 + p_on_ee_z*x24 + x20*x28 - x24*x25;
    jacobian(0, 5) = p_on_ee_y*x30 + p_on_ee_z*x32 - x30*x34 - x32*x33;
    jacobian(1, 0) = x35;
    jacobian(1, 1) = pre_transform_special_symbol_23*x36 + x37;
    jacobian(1, 2) = x36*x6 + x37;
    jacobian(1, 3) = x14*x36 + x37;
    jacobian(1, 4) = p_on_ee_x*x20 - p_on_ee_z*x42 - x20*x45 + x25*x42;
    jacobian(1, 5) = -p_on_ee_x*x30 - p_on_ee_z*x46 + x30*x47 + x33*x46;
    jacobian(2, 1) = x48;
    jacobian(2, 2) = -x2*x43 - x26*x36 + x48;
    jacobian(2, 3) = -x2*x44 - x27*x36 + x48;
    jacobian(2, 4) = -p_on_ee_x*x24 + p_on_ee_y*x42 + x24*x45 - x28*x42;
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
    
    // Code for explicit solution node 1, solved variable is th_0
    auto ExplicitSolutionNode_node_1_solve_th_0_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;
        
        // The explicit solution of root node
        const bool condition_0 = std::fabs(Px - d_6*r_13) >= zero_tolerance || std::fabs(Py - d_6*r_23) >= zero_tolerance || std::fabs(d_0 + d_2 + d_4) >= zero_tolerance;
        if (condition_0)
        {
            // Temp variable for efficiency
            const double x0 = Px - d_6*r_13;
            const double x1 = -Py + d_6*r_23;
            const double x2 = std::atan2(x0, x1);
            const double x3 = -d_0 - d_2 - d_4;
            const double x4 = std::sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
            // End of temp variables
            
            auto solution_0 = make_raw_solution();
            solution_0[0] = x2 + std::atan2(x4, x3);
            int appended_idx = append_solution_to_queue(solution_0);
            add_input_index_to(2, appended_idx);
        }
        
        const bool condition_1 = std::fabs(Px - d_6*r_13) >= zero_tolerance || std::fabs(Py - d_6*r_23) >= zero_tolerance || std::fabs(d_0 + d_2 + d_4) >= zero_tolerance;
        if (condition_1)
        {
            // Temp variable for efficiency
            const double x0 = Px - d_6*r_13;
            const double x1 = -Py + d_6*r_23;
            const double x2 = std::atan2(x0, x1);
            const double x3 = -d_0 - d_2 - d_4;
            const double x4 = std::sqrt(std::pow(x0, 2) + std::pow(x1, 2) - std::pow(x3, 2));
            // End of temp variables
            
            auto solution_1 = make_raw_solution();
            solution_1[0] = x2 + std::atan2(-x4, x3);
            int appended_idx = append_solution_to_queue(solution_1);
            add_input_index_to(2, appended_idx);
        }
        
    };
    // Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_0_processor();
    // Finish code for explicit solution node 0
    
    // Code for non-branch dispatcher node 2
    // Actually, there is no code
    
    // Code for explicit solution node 3, solved variable is th_4
    auto ExplicitSolutionNode_node_3_solve_th_4_processor = [&]() -> void
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
            
            const bool condition_0 = std::fabs(r_13*std::sin(th_0) - r_23*std::cos(th_0)) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::acos(r_13*std::sin(th_0) - r_23*std::cos(th_0));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[7] = x0;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = std::fabs(r_13*std::sin(th_0) - r_23*std::cos(th_0)) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::acos(r_13*std::sin(th_0) - r_23*std::cos(th_0));
                // End of temp variables
                const double tmp_sol_value = -x0;
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
    ExplicitSolutionNode_node_3_solve_th_4_processor();
    // Finish code for explicit solution node 2
    
    // Code for solved_variable dispatcher node 4
    auto SolvedVariableDispatcherNode_node_4_processor = [&]() -> void
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
            bool taken_by_degenerate = false;
            const double th_4 = this_solution[7];
            
            const bool degenerate_valid_0 = std::fabs(th_4) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
            }
            
            const bool degenerate_valid_1 = std::fabs(th_4 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(5, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_4_processor();
    // Finish code for solved_variable dispatcher node 4
    
    // Code for explicit solution node 5, solved variable is th_1th_2th_3_soa
    auto ExplicitSolutionNode_node_5_solve_th_1th_2th_3_soa_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_4 = this_solution[7];
            
            const bool condition_0 = std::fabs(r_33) >= zero_tolerance || std::fabs(r_13*std::cos(th_0) + r_23*std::sin(th_0)) >= zero_tolerance || std::fabs(std::sin(th_4)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_4);
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_33*x0, x0*(-r_13*std::cos(th_0) - r_23*std::sin(th_0)));
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
    ExplicitSolutionNode_node_5_solve_th_1th_2th_3_soa_processor();
    // Finish code for explicit solution node 5
    
    // Code for non-branch dispatcher node 6
    // Actually, there is no code
    
    // Code for explicit solution node 7, solved variable is th_5
    auto ExplicitSolutionNode_node_7_solve_th_5_processor = [&]() -> void
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
            const double th_0 = this_solution[0];
            const double th_4 = this_solution[7];
            
            const bool condition_0 = std::fabs(r_11*std::sin(th_0) - r_21*std::cos(th_0)) >= zero_tolerance || std::fabs(r_12*std::sin(th_0) - r_22*std::cos(th_0)) >= zero_tolerance || std::fabs(std::sin(th_4)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_4);
                const double x1 = std::cos(th_0);
                const double x2 = std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(-r_12*x2 + r_22*x1), x0*(r_11*x2 - r_21*x1));
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
                add_input_index_to(8, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_7_solve_th_5_processor();
    // Finish code for explicit solution node 6
    
    // Code for non-branch dispatcher node 8
    // Actually, there is no code
    
    // Code for explicit solution node 9, solved variable is th_2
    auto ExplicitSolutionNode_node_9_solve_th_2_processor = [&]() -> void
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
            const double th_5 = this_solution[8];
            
            const bool condition_0 = (1.0/2.0)*std::fabs((std::pow(a_1, 2) + std::pow(a_3, 2) + std::pow(d_0, 2) + 2*d_0*d_2 + 2*d_0*d_4 + std::pow(d_2, 2) + 2*d_2*d_4 + std::pow(d_4, 2) - std::pow(d_5, 2) + 2*d_5*inv_Px*std::sin(th_5) + 2*d_5*inv_Py*std::cos(th_5) - std::pow(d_6, 2) - 2*d_6*inv_Pz - std::pow(inv_Px, 2) - std::pow(inv_Py, 2) - std::pow(inv_Pz, 2))/(a_1*a_3)) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 2*d_0;
                const double x1 = 2*d_5;
                const double x2 = std::acos((1.0/2.0)*(-std::pow(a_1, 2) - std::pow(a_3, 2) - std::pow(d_0, 2) - std::pow(d_2, 2) - 2*d_2*d_4 - d_2*x0 - std::pow(d_4, 2) - d_4*x0 + std::pow(d_5, 2) + std::pow(d_6, 2) + 2*d_6*inv_Pz + std::pow(inv_Px, 2) - inv_Px*x1*std::sin(th_5) + std::pow(inv_Py, 2) - inv_Py*x1*std::cos(th_5) + std::pow(inv_Pz, 2))/(a_1*a_3));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[4] = x2;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(10, appended_idx);
            }
            
            const bool condition_1 = (1.0/2.0)*std::fabs((std::pow(a_1, 2) + std::pow(a_3, 2) + std::pow(d_0, 2) + 2*d_0*d_2 + 2*d_0*d_4 + std::pow(d_2, 2) + 2*d_2*d_4 + std::pow(d_4, 2) - std::pow(d_5, 2) + 2*d_5*inv_Px*std::sin(th_5) + 2*d_5*inv_Py*std::cos(th_5) - std::pow(d_6, 2) - 2*d_6*inv_Pz - std::pow(inv_Px, 2) - std::pow(inv_Py, 2) - std::pow(inv_Pz, 2))/(a_1*a_3)) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = 2*d_0;
                const double x1 = 2*d_5;
                const double x2 = std::acos((1.0/2.0)*(-std::pow(a_1, 2) - std::pow(a_3, 2) - std::pow(d_0, 2) - std::pow(d_2, 2) - 2*d_2*d_4 - d_2*x0 - std::pow(d_4, 2) - d_4*x0 + std::pow(d_5, 2) + std::pow(d_6, 2) + 2*d_6*inv_Pz + std::pow(inv_Px, 2) - inv_Px*x1*std::sin(th_5) + std::pow(inv_Py, 2) - inv_Py*x1*std::cos(th_5) + std::pow(inv_Pz, 2))/(a_1*a_3));
                // End of temp variables
                const double tmp_sol_value = -x2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(10, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_2_processor();
    // Finish code for explicit solution node 8
    
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
            const double th_5 = this_solution[8];
            const bool checked_result = std::fabs(Pz + d_5*r_31*std::sin(th_5) + d_5*r_32*std::cos(th_5) - d_6*r_33) <= 9.9999999999999995e-7 && std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) + d_5*r_11*std::sin(th_5)*std::cos(th_0) + d_5*r_12*std::cos(th_0)*std::cos(th_5) + d_5*r_21*std::sin(th_0)*std::sin(th_5) + d_5*r_22*std::sin(th_0)*std::cos(th_5) - d_6*r_13*std::cos(th_0) - d_6*r_23*std::sin(th_0)) <= 9.9999999999999995e-7;
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
            const double th_2 = this_solution[4];
            const double th_5 = this_solution[8];
            
            const bool condition_0 = std::fabs(Pz + d_5*r_31*std::sin(th_5) + d_5*r_32*std::cos(th_5) - d_6*r_33) >= 9.9999999999999995e-7 || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) + d_5*r_11*std::sin(th_5)*std::cos(th_0) + d_5*r_12*std::cos(th_0)*std::cos(th_5) + d_5*r_21*std::sin(th_0)*std::sin(th_5) + d_5*r_22*std::sin(th_0)*std::cos(th_5) - d_6*r_13*std::cos(th_0) - d_6*r_23*std::sin(th_0)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = -a_1 - a_3*std::cos(th_2);
                const double x1 = d_5*std::sin(th_5);
                const double x2 = d_5*std::cos(th_5);
                const double x3 = Pz - d_6*r_33 + r_31*x1 + r_32*x2;
                const double x4 = std::cos(th_0);
                const double x5 = std::sin(th_0);
                const double x6 = -Px*x4 - Py*x5 + d_6*r_13*x4 + d_6*r_23*x5 - r_11*x1*x4 - r_12*x2*x4 - r_21*x1*x5 - r_22*x2*x5;
                const double x7 = a_3*std::sin(th_2);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*x3 + x6*x7, x0*x6 - x3*x7);
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
    // Finish code for explicit solution node 11
    
    // Code for non-branch dispatcher node 12
    // Actually, there is no code
    
    // Code for explicit solution node 13, solved variable is th_2th_3_soa
    auto ExplicitSolutionNode_node_13_solve_th_2th_3_soa_processor = [&]() -> void
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
            const double th_1 = this_solution[1];
            const double th_1th_2th_3_soa = this_solution[3];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_1 + th_1th_2th_3_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[5] = tmp_sol_value;
                add_input_index_to(14, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_13_solve_th_2th_3_soa_processor();
    // Finish code for explicit solution node 12
    
    // Code for non-branch dispatcher node 14
    // Actually, there is no code
    
    // Code for explicit solution node 15, solved variable is th_3
    auto ExplicitSolutionNode_node_15_solve_th_3_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(14);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(14);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 15
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_2 = this_solution[4];
            const double th_2th_3_soa = this_solution[5];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_2 + th_2th_3_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(16, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_15_solve_th_3_processor();
    // Finish code for explicit solution node 14
    
    // Code for non-branch dispatcher node 16
    // Actually, there is no code
    
    // Code for explicit solution node 17, solved variable is th_1th_2_soa
    auto ExplicitSolutionNode_node_17_solve_th_1th_2_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(16);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(16);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 17
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[1];
            const double th_2 = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = th_1 + th_2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_17_solve_th_1th_2_soa_processor();
    // Finish code for explicit solution node 16
    
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
        const double value_at_3 = raw_ik_out_i[6];  // th_3
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

}; // struct ur_5_ik

// Code below for debug
void test_ik_solve_ur_5()
{
    std::array<double, ur_5_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = ur_5_ik::computeFK(theta);
    auto ik_output = ur_5_ik::computeIK(ee_pose);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = ur_5_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_ur_5();
}
