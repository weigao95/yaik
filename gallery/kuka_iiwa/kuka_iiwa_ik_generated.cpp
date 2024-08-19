#include "yaik_cpp_common.h"

using namespace yaik_cpp;

struct kuka_iiwa_ik {

// Constants for solver
static constexpr int robot_nq = 7;
static constexpr int max_n_solutions = 16;
static constexpr int n_tree_nodes = 58;
static constexpr int intermediate_solution_size = 10;
static constexpr double pose_tolerance = 1e-6;
static constexpr double pose_tolerance_degenerate = 1e-4;
static constexpr double zero_tolerance = 1e-6;
using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate<intermediate_solution_size, max_n_solutions, robot_nq>;

// Robot parameters
static constexpr double d_1 = 0.42;
static constexpr double d_2 = 0.4;
static constexpr double d_3 = 0.081;
static constexpr double post_transform_d5 = 0.045;
static constexpr double pre_transform_d4 = 0.36;

// Unknown offsets from original unknown value to raw value
// Original value are the ones corresponded to robot (usually urdf/sdf)
// Raw value are the ones used in the solver
// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw
static constexpr double th_0_offset_original2raw = 0.0;
static constexpr double th_1_offset_original2raw = 3.141592653589793;
static constexpr double th_2_offset_original2raw = 0.0;
static constexpr double th_3_offset_original2raw = -3.141592653589793;
static constexpr double th_4_offset_original2raw = 0.0;
static constexpr double th_5_offset_original2raw = -3.141592653589793;
static constexpr double th_6_offset_original2raw = 3.141592653589793;

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
    ee_transformed(0, 0) = -r_13;
    ee_transformed(0, 1) = r_12;
    ee_transformed(0, 2) = r_11;
    ee_transformed(0, 3) = Px - post_transform_d5*r_11;
    ee_transformed(1, 0) = -r_23;
    ee_transformed(1, 1) = r_22;
    ee_transformed(1, 2) = r_21;
    ee_transformed(1, 3) = Py - post_transform_d5*r_21;
    ee_transformed(2, 0) = -r_33;
    ee_transformed(2, 1) = r_32;
    ee_transformed(2, 2) = r_31;
    ee_transformed(2, 3) = Pz - post_transform_d5*r_31 - pre_transform_d4;
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
    ee_transformed(0, 0) = r_13;
    ee_transformed(0, 1) = r_12;
    ee_transformed(0, 2) = -r_11;
    ee_transformed(0, 3) = Px + post_transform_d5*r_13;
    ee_transformed(1, 0) = r_23;
    ee_transformed(1, 1) = r_22;
    ee_transformed(1, 2) = -r_21;
    ee_transformed(1, 3) = Py + post_transform_d5*r_23;
    ee_transformed(2, 0) = r_33;
    ee_transformed(2, 1) = r_32;
    ee_transformed(2, 2) = -r_31;
    ee_transformed(2, 3) = Pz + post_transform_d5*r_33 + pre_transform_d4;
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
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_6);
    const double x1 = std::cos(th_4);
    const double x2 = std::sin(th_0);
    const double x3 = std::cos(th_2);
    const double x4 = x2*x3;
    const double x5 = std::cos(th_1);
    const double x6 = std::sin(th_2);
    const double x7 = std::cos(th_0);
    const double x8 = x6*x7;
    const double x9 = x4 - x5*x8;
    const double x10 = std::sin(th_4);
    const double x11 = std::sin(th_3);
    const double x12 = std::sin(th_1);
    const double x13 = x12*x7;
    const double x14 = std::cos(th_3);
    const double x15 = x2*x6;
    const double x16 = x3*x7;
    const double x17 = x15 + x16*x5;
    const double x18 = x11*x13 + x14*x17;
    const double x19 = -x1*x9 - x10*x18;
    const double x20 = std::cos(th_6);
    const double x21 = std::sin(th_5);
    const double x22 = -x11*x17 + x13*x14;
    const double x23 = std::cos(th_5);
    const double x24 = x1*x18 - x10*x9;
    const double x25 = -x21*x22 + x23*x24;
    const double x26 = -x21*x24 - x22*x23;
    const double x27 = -x15*x5 - x16;
    const double x28 = x12*x2;
    const double x29 = x4*x5 - x8;
    const double x30 = x11*x28 + x14*x29;
    const double x31 = -x1*x27 - x10*x30;
    const double x32 = -x11*x29 + x14*x28;
    const double x33 = x1*x30 - x10*x27;
    const double x34 = -x21*x32 + x23*x33;
    const double x35 = -x21*x33 - x23*x32;
    const double x36 = x12*x6;
    const double x37 = x12*x3;
    const double x38 = x11*x5 - x14*x37;
    const double x39 = -x1*x36 - x10*x38;
    const double x40 = x11*x37 + x14*x5;
    const double x41 = x1*x38 - x10*x36;
    const double x42 = -x21*x40 + x23*x41;
    const double x43 = -x21*x41 - x23*x40;
    // End of temp variables
    Eigen::Matrix4d ee_pose_raw;
    ee_pose_raw.setIdentity();
    ee_pose_raw(0, 0) = -x0*x19 + x20*x25;
    ee_pose_raw(0, 1) = -x0*x25 - x19*x20;
    ee_pose_raw(0, 2) = x26;
    ee_pose_raw(0, 3) = -d_1*x13 + d_2*x22 + d_3*x26;
    ee_pose_raw(1, 0) = -x0*x31 + x20*x34;
    ee_pose_raw(1, 1) = -x0*x34 - x20*x31;
    ee_pose_raw(1, 2) = x35;
    ee_pose_raw(1, 3) = -d_1*x28 + d_2*x32 + d_3*x35;
    ee_pose_raw(2, 0) = -x0*x39 + x20*x42;
    ee_pose_raw(2, 1) = -x0*x42 - x20*x39;
    ee_pose_raw(2, 2) = x43;
    ee_pose_raw(2, 3) = -d_1*x5 + d_2*x40 + d_3*x43;
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
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = std::sin(th_1);
    const double x2 = std::cos(th_0);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_2);
    const double x5 = x0*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::sin(th_2);
    const double x8 = x2*x7;
    const double x9 = x5 - x6*x8;
    const double x10 = std::cos(th_3);
    const double x11 = std::sin(th_3);
    const double x12 = x0*x7;
    const double x13 = x2*x4;
    const double x14 = x12 + x13*x6;
    const double x15 = x10*x3 - x11*x14;
    const double x16 = std::cos(th_4);
    const double x17 = std::sin(th_4);
    const double x18 = x10*x14 + x11*x3;
    const double x19 = -x16*x9 - x17*x18;
    const double x20 = std::cos(th_5);
    const double x21 = std::sin(th_5);
    const double x22 = -x15*x20 - x21*(x16*x18 - x17*x9);
    const double x23 = x0*x1;
    const double x24 = -x12*x6 - x13;
    const double x25 = x5*x6 - x8;
    const double x26 = x10*x23 - x11*x25;
    const double x27 = x10*x25 + x11*x23;
    const double x28 = -x16*x24 - x17*x27;
    const double x29 = -x20*x26 - x21*(x16*x27 - x17*x24);
    const double x30 = x1*x7;
    const double x31 = x1*x4;
    const double x32 = x10*x6 + x11*x31;
    const double x33 = -x10*x31 + x11*x6;
    const double x34 = -x16*x30 - x17*x33;
    const double x35 = -x20*x32 - x21*(x16*x33 - x17*x30);
    const double x36 = d_1*x6;
    const double x37 = pre_transform_d4 - x36;
    const double x38 = d_1*std::pow(x1, 2);
    const double x39 = d_2*x32 + x37;
    const double x40 = d_1*x23;
    const double x41 = d_2*x26 - x40;
    const double x42 = d_3*x35 + x39;
    const double x43 = d_3*x29 + x41;
    const double x44 = d_1*x3;
    const double x45 = d_2*x15 - x44;
    const double x46 = d_3*x22 + x45;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = -x0;
    jacobian(0, 2) = -x3;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = x15;
    jacobian(0, 5) = x19;
    jacobian(0, 6) = x22;
    jacobian(1, 1) = x2;
    jacobian(1, 2) = -x23;
    jacobian(1, 3) = x24;
    jacobian(1, 4) = x26;
    jacobian(1, 5) = x28;
    jacobian(1, 6) = x29;
    jacobian(2, 0) = 1;
    jacobian(2, 2) = -x6;
    jacobian(2, 3) = x30;
    jacobian(2, 4) = x32;
    jacobian(2, 5) = x34;
    jacobian(2, 6) = x35;
    jacobian(3, 1) = -pre_transform_d4*x2;
    jacobian(3, 2) = x23*x36 + x23*x37;
    jacobian(3, 3) = -x12*x38 - x24*x37;
    jacobian(3, 4) = -x26*x39 + x32*x41;
    jacobian(3, 5) = -x28*x39 + x34*x41;
    jacobian(3, 6) = -x29*x42 + x35*x43;
    jacobian(4, 1) = -pre_transform_d4*x0;
    jacobian(4, 2) = -x3*x36 - x3*x37;
    jacobian(4, 3) = x37*x9 + x38*x8;
    jacobian(4, 4) = x15*x39 - x32*x45;
    jacobian(4, 5) = x19*x39 - x34*x45;
    jacobian(4, 6) = x22*x42 - x35*x46;
    jacobian(5, 3) = -x24*x44 + x40*x9;
    jacobian(5, 4) = -x15*x41 + x26*x45;
    jacobian(5, 5) = -x19*x41 + x28*x45;
    jacobian(5, 6) = -x22*x43 + x29*x46;
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
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;
    
    // Temp variable for efficiency
    const double x0 = std::sin(th_0);
    const double x1 = std::sin(th_1);
    const double x2 = std::cos(th_0);
    const double x3 = x1*x2;
    const double x4 = std::cos(th_2);
    const double x5 = x0*x4;
    const double x6 = std::cos(th_1);
    const double x7 = std::sin(th_2);
    const double x8 = x2*x7;
    const double x9 = x5 - x6*x8;
    const double x10 = std::cos(th_3);
    const double x11 = std::sin(th_3);
    const double x12 = x0*x7;
    const double x13 = x2*x4;
    const double x14 = x12 + x13*x6;
    const double x15 = x10*x3 - x11*x14;
    const double x16 = std::cos(th_4);
    const double x17 = std::sin(th_4);
    const double x18 = x10*x14 + x11*x3;
    const double x19 = std::cos(th_5);
    const double x20 = std::sin(th_5);
    const double x21 = x0*x1;
    const double x22 = -x12*x6 - x13;
    const double x23 = x5*x6 - x8;
    const double x24 = x10*x21 - x11*x23;
    const double x25 = x10*x23 + x11*x21;
    const double x26 = x1*x7;
    const double x27 = x1*x4;
    const double x28 = x10*x6 + x11*x27;
    const double x29 = -x10*x27 + x11*x6;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 1) = -x0;
    jacobian(0, 2) = -x3;
    jacobian(0, 3) = x9;
    jacobian(0, 4) = x15;
    jacobian(0, 5) = -x16*x9 - x17*x18;
    jacobian(0, 6) = -x15*x19 - x20*(x16*x18 - x17*x9);
    jacobian(1, 1) = x2;
    jacobian(1, 2) = -x21;
    jacobian(1, 3) = x22;
    jacobian(1, 4) = x24;
    jacobian(1, 5) = -x16*x22 - x17*x25;
    jacobian(1, 6) = -x19*x24 - x20*(x16*x25 - x17*x22);
    jacobian(2, 0) = 1;
    jacobian(2, 2) = -x6;
    jacobian(2, 3) = x26;
    jacobian(2, 4) = x28;
    jacobian(2, 5) = -x16*x26 - x17*x29;
    jacobian(2, 6) = -x19*x28 - x20*(x16*x29 - x17*x26);
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
    const double th_6 = theta_input_original[6] + th_6_offset_original2raw;
    const double p_on_ee_x = point_on_ee[0];
    const double p_on_ee_y = point_on_ee[1];
    const double p_on_ee_z = point_on_ee[2];
    
    // Temp variable for efficiency
    const double x0 = std::cos(th_0);
    const double x1 = p_on_ee_z*x0;
    const double x2 = std::cos(th_1);
    const double x3 = std::sin(th_1);
    const double x4 = std::sin(th_0);
    const double x5 = p_on_ee_z*x4;
    const double x6 = d_1*x2;
    const double x7 = x3*x4;
    const double x8 = pre_transform_d4 - x6;
    const double x9 = std::sin(th_2);
    const double x10 = x3*x9;
    const double x11 = x4*x9;
    const double x12 = d_1*std::pow(x3, 2);
    const double x13 = std::cos(th_2);
    const double x14 = x0*x13;
    const double x15 = -x11*x2 - x14;
    const double x16 = std::cos(th_3);
    const double x17 = std::sin(th_3);
    const double x18 = x13*x3;
    const double x19 = x16*x2 + x17*x18;
    const double x20 = x0*x9;
    const double x21 = x13*x4;
    const double x22 = x2*x21 - x20;
    const double x23 = x16*x7 - x17*x22;
    const double x24 = d_2*x19 + x8;
    const double x25 = d_1*x7;
    const double x26 = d_2*x23 - x25;
    const double x27 = std::cos(th_4);
    const double x28 = std::sin(th_4);
    const double x29 = -x16*x18 + x17*x2;
    const double x30 = -x10*x27 - x28*x29;
    const double x31 = x16*x22 + x17*x7;
    const double x32 = -x15*x27 - x28*x31;
    const double x33 = std::cos(th_5);
    const double x34 = std::sin(th_5);
    const double x35 = -x19*x33 - x34*(-x10*x28 + x27*x29);
    const double x36 = -x23*x33 - x34*(-x15*x28 + x27*x31);
    const double x37 = d_3*x35 + x24;
    const double x38 = d_3*x36 + x26;
    const double x39 = x0*x3;
    const double x40 = -x2*x20 + x21;
    const double x41 = x11 + x14*x2;
    const double x42 = x16*x39 - x17*x41;
    const double x43 = d_1*x39;
    const double x44 = d_2*x42 - x43;
    const double x45 = x16*x41 + x17*x39;
    const double x46 = -x27*x40 - x28*x45;
    const double x47 = -x33*x42 - x34*(x27*x45 - x28*x40);
    const double x48 = d_3*x47 + x44;
    // End of temp variables
    
    jacobian.setZero();
    jacobian(0, 0) = -p_on_ee_y;
    jacobian(0, 1) = -pre_transform_d4*x0 + x1;
    jacobian(0, 2) = p_on_ee_y*x2 - x3*x5 + x6*x7 + x7*x8;
    jacobian(0, 3) = -p_on_ee_y*x10 + p_on_ee_z*x15 - x11*x12 - x15*x8;
    jacobian(0, 4) = -p_on_ee_y*x19 + p_on_ee_z*x23 + x19*x26 - x23*x24;
    jacobian(0, 5) = -p_on_ee_y*x30 + p_on_ee_z*x32 - x24*x32 + x26*x30;
    jacobian(0, 6) = -p_on_ee_y*x35 + p_on_ee_z*x36 + x35*x38 - x36*x37;
    jacobian(1, 0) = p_on_ee_x;
    jacobian(1, 1) = -pre_transform_d4*x4 + x5;
    jacobian(1, 2) = -p_on_ee_x*x2 + x1*x3 - x39*x6 - x39*x8;
    jacobian(1, 3) = p_on_ee_x*x10 - p_on_ee_z*x40 + x12*x20 + x40*x8;
    jacobian(1, 4) = p_on_ee_x*x19 - p_on_ee_z*x42 - x19*x44 + x24*x42;
    jacobian(1, 5) = p_on_ee_x*x30 - p_on_ee_z*x46 + x24*x46 - x30*x44;
    jacobian(1, 6) = p_on_ee_x*x35 - p_on_ee_z*x47 - x35*x48 + x37*x47;
    jacobian(2, 1) = -p_on_ee_x*x0 - p_on_ee_y*x4;
    jacobian(2, 2) = p_on_ee_x*x7 - p_on_ee_y*x39;
    jacobian(2, 3) = -p_on_ee_x*x15 + p_on_ee_y*x40 - x15*x43 + x25*x40;
    jacobian(2, 4) = -p_on_ee_x*x23 + p_on_ee_y*x42 + x23*x44 - x26*x42;
    jacobian(2, 5) = -p_on_ee_x*x32 + p_on_ee_y*x46 - x26*x46 + x32*x44;
    jacobian(2, 6) = -p_on_ee_x*x36 + p_on_ee_y*x47 + x36*x48 - x38*x47;
    return;
}

static void computeRawIK(const Eigen::Matrix4d& T_ee, double th_0, SolutionQueue<intermediate_solution_size, max_n_solutions>& solution_queue, NodeIndexWorkspace<max_n_solutions>& node_index_workspace, std::vector<std::array<double, robot_nq>>& ik_output)
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
    
    // Code for explicit solution node 1, solved variable is th_3
    auto ExplicitSolutionNode_node_1_solve_th_3_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(0);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(0);
        if (!this_input_valid)
            return;
        
        // The explicit solution of root node
        const bool condition_0 = (1.0/2.0)*std::fabs((-std::pow(d_1, 2) - std::pow(d_2, 2) + std::pow(Px - d_3*r_13, 2) + std::pow(Py - d_3*r_23, 2) + std::pow(Pz - d_3*r_33, 2))/(d_1*d_2)) <= 1;
        if (condition_0)
        {
            // Temp variable for efficiency
            const double x0 = safe_acos((1.0/2.0)*(std::pow(d_1, 2) + std::pow(d_2, 2) - std::pow(Px - d_3*r_13, 2) - std::pow(Py - d_3*r_23, 2) - std::pow(Pz - d_3*r_33, 2))/(d_1*d_2));
            // End of temp variables
            
            auto solution_0 = make_raw_solution();
            solution_0[5] = x0;
            int appended_idx = append_solution_to_queue(solution_0);
            add_input_index_to(2, appended_idx);
        }
        
        const bool condition_1 = (1.0/2.0)*std::fabs((-std::pow(d_1, 2) - std::pow(d_2, 2) + std::pow(Px - d_3*r_13, 2) + std::pow(Py - d_3*r_23, 2) + std::pow(Pz - d_3*r_33, 2))/(d_1*d_2)) <= 1;
        if (condition_1)
        {
            // Temp variable for efficiency
            const double x0 = safe_acos((1.0/2.0)*(std::pow(d_1, 2) + std::pow(d_2, 2) - std::pow(Px - d_3*r_13, 2) - std::pow(Py - d_3*r_23, 2) - std::pow(Pz - d_3*r_33, 2))/(d_1*d_2));
            // End of temp variables
            
            auto solution_1 = make_raw_solution();
            solution_1[5] = -x0;
            int appended_idx = append_solution_to_queue(solution_1);
            add_input_index_to(2, appended_idx);
        }
        
    };
    // Invoke the processor
    ExplicitSolutionNode_node_1_solve_th_3_processor();
    // Finish code for explicit solution node 0
    
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
            const double th_3 = this_solution[5];
            
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
                add_input_index_to(23, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(3, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_2_processor();
    // Finish code for solved_variable dispatcher node 2
    
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
            
            const bool condition_0 = std::fabs(Pz - d_3*r_33) >= zero_tolerance || std::fabs(d_1 + d_2) >= zero_tolerance || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - d_3*(r_13*std::cos(th_0) + r_23*std::sin(th_0))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2((Px*x0 + Py*x1 - d_3*(r_13*x0 + r_23*x1))/(-d_1 - d_2), (-Pz + d_3*r_33)/(d_1 + d_2));
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
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
    
    // Code for explicit solution node 25, solved variable is th_5
    auto ExplicitSolutionNode_node_25_solve_th_5_processor = [&]() -> void
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
            
            const bool condition_0 = std::fabs((d_3 + inv_Pz)/(d_1 + d_2)) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = safe_acos((d_3 + inv_Pz)/(d_1 + d_2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[8] = x0;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(26, appended_idx);
            }
            
            const bool condition_1 = std::fabs((d_3 + inv_Pz)/(d_1 + d_2)) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = safe_acos((d_3 + inv_Pz)/(d_1 + d_2));
                // End of temp variables
                const double tmp_sol_value = -x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
                add_input_index_to(26, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_25_solve_th_5_processor();
    // Finish code for explicit solution node 24
    
    // Code for solved_variable dispatcher node 26
    auto SolvedVariableDispatcherNode_node_26_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(26);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(26);
        if (!this_input_valid)
            return;
        
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_5 = this_solution[8];
            
            const bool degenerate_valid_0 = std::fabs(th_5) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
            }
            
            const bool degenerate_valid_1 = std::fabs(th_5 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(27, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_26_processor();
    // Finish code for solved_variable dispatcher node 26
    
    // Code for explicit solution node 27, solved variable is th_6
    auto ExplicitSolutionNode_node_27_solve_th_6_processor = [&]() -> void
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
            const double th_5 = this_solution[8];
            
            const bool condition_0 = std::fabs(inv_Px) >= zero_tolerance || std::fabs(inv_Py) >= zero_tolerance || std::fabs((d_1 + d_2)*std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1/((d_1 + d_2)*std::sin(th_5));
                // End of temp variables
                const double tmp_sol_value = std::atan2(-inv_Py*x0, inv_Px*x0);
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
                add_input_index_to(28, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_27_solve_th_6_processor();
    // Finish code for explicit solution node 27
    
    // Code for solved_variable dispatcher node 28
    auto SolvedVariableDispatcherNode_node_28_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(28);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(28);
        if (!this_input_valid)
            return;
        
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_5 = this_solution[8];
            
            const bool degenerate_valid_0 = std::fabs(th_5) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
            }
            
            const bool degenerate_valid_1 = std::fabs(th_5 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(29, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_28_processor();
    // Finish code for solved_variable dispatcher node 28
    
    // Code for explicit solution node 29, solved variable is th_2th_4_soa
    auto ExplicitSolutionNode_node_29_solve_th_2th_4_soa_processor = [&]() -> void
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
            const double th_1 = this_solution[2];
            const double th_5 = this_solution[8];
            
            const bool condition_0 = std::fabs(r_13*std::sin(th_0) - r_23*std::cos(th_0)) >= zero_tolerance || std::fabs(r_13*std::cos(th_0)*std::cos(th_1) + r_23*std::sin(th_0)*std::cos(th_1) - r_33*std::sin(th_1)) >= zero_tolerance || std::fabs(std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_5);
                const double x1 = std::sin(th_0);
                const double x2 = std::cos(th_0);
                const double x3 = std::cos(th_1);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(r_13*x1 - r_23*x2), x0*(r_13*x2*x3 + r_23*x1*x3 - r_33*std::sin(th_1)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[4] = tmp_sol_value;
                add_input_index_to(30, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_29_solve_th_2th_4_soa_processor();
    // Finish code for explicit solution node 29
    
    // Code for non-branch dispatcher node 30
    // Actually, there is no code
    
    // Code for explicit solution node 31, solved variable is th_2
    auto ExplicitSolutionNode_node_31_solve_th_2_processor = [&]() -> void
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
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(32, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_31_solve_th_2_processor();
    // Finish code for explicit solution node 30
    
    // Code for non-branch dispatcher node 32
    // Actually, there is no code
    
    // Code for explicit solution node 33, solved variable is th_4
    auto ExplicitSolutionNode_node_33_solve_th_4_processor = [&]() -> void
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
            const double th_2 = this_solution[3];
            const double th_2th_4_soa = this_solution[4];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_2 + th_2th_4_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_33_solve_th_4_processor();
    // Finish code for explicit solution node 32
    
    // Code for explicit solution node 12, solved variable is th_1
    auto ExplicitSolutionNode_node_12_solve_th_1_processor = [&]() -> void
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
            
            const bool condition_0 = std::fabs(Pz - d_3*r_33) >= zero_tolerance || std::fabs(d_1 - d_2) >= zero_tolerance || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - d_3*(r_13*std::cos(th_0) + r_23*std::sin(th_0))) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/(-d_1 + d_2);
                const double x1 = std::cos(th_0);
                const double x2 = std::sin(th_0);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(Px*x1 + Py*x2 - d_3*(r_13*x1 + r_23*x2)), x0*(Pz - d_3*r_33));
                solution_queue.get_solution(node_input_i_idx_in_queue)[2] = tmp_sol_value;
                add_input_index_to(13, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_12_solve_th_1_processor();
    // Finish code for explicit solution node 12
    
    // Code for non-branch dispatcher node 13
    // Actually, there is no code
    
    // Code for explicit solution node 14, solved variable is th_5
    auto ExplicitSolutionNode_node_14_solve_th_5_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(13);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(13);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 14
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            
            const bool condition_0 = std::fabs((d_3 + inv_Pz)/(d_1 - d_2)) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = safe_acos((d_3 + inv_Pz)/(-d_1 + d_2));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[8] = x0;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(15, appended_idx);
            }
            
            const bool condition_1 = std::fabs((d_3 + inv_Pz)/(d_1 - d_2)) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = safe_acos((d_3 + inv_Pz)/(-d_1 + d_2));
                // End of temp variables
                const double tmp_sol_value = -x0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[8] = tmp_sol_value;
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
    // Finish code for explicit solution node 13
    
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
            const double th_5 = this_solution[8];
            
            const bool degenerate_valid_0 = std::fabs(th_5) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(44, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_5 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(51, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(16, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_15_processor();
    // Finish code for solved_variable dispatcher node 15
    
    // Code for explicit solution node 51, solved variable is th_4th_6_soa
    auto ExplicitSolutionNode_node_51_solve_th_4th_6_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(51);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(51);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 51
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_3 = this_solution[5];
            
            const bool condition_0 = std::fabs(inv_Px) >= zero_tolerance || std::fabs(inv_Py) >= zero_tolerance || std::fabs(d_1*std::sin(th_3)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1/(d_1*std::sin(th_3));
                // End of temp variables
                const double tmp_sol_value = std::atan2(inv_Py*x0, -inv_Px*x0);
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
                add_input_index_to(52, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_51_solve_th_4th_6_soa_processor();
    // Finish code for explicit solution node 51
    
    // Code for solved_variable dispatcher node 52
    auto SolvedVariableDispatcherNode_node_52_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(52);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(52);
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
            }
            
            const bool degenerate_valid_1 = std::fabs(th_3 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(53, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_52_processor();
    // Finish code for solved_variable dispatcher node 52
    
    // Code for explicit solution node 53, solved variable is th_2
    auto ExplicitSolutionNode_node_53_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(53);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(53);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 53
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_3 = this_solution[5];
            
            const bool condition_0 = std::fabs(r_13*std::sin(th_0) - r_23*std::cos(th_0)) >= zero_tolerance || std::fabs(r_13*std::cos(th_0)*std::cos(th_1) + r_23*std::sin(th_0)*std::cos(th_1) - r_33*std::sin(th_1)) >= zero_tolerance || std::fabs(std::sin(th_3)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_3);
                const double x1 = std::cos(th_0);
                const double x2 = std::sin(th_0);
                const double x3 = std::cos(th_1);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(-r_13*x2 + r_23*x1), x0*(-r_13*x1*x3 - r_23*x2*x3 + r_33*std::sin(th_1)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(54, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_53_solve_th_2_processor();
    // Finish code for explicit solution node 53
    
    // Code for non-branch dispatcher node 54
    // Actually, there is no code
    
    // Code for explicit solution node 55, solved variable is th_4
    auto ExplicitSolutionNode_node_55_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(54);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(54);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 55
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
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(56, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_55_solve_th_4_processor();
    // Finish code for explicit solution node 54
    
    // Code for non-branch dispatcher node 56
    // Actually, there is no code
    
    // Code for explicit solution node 57, solved variable is th_6
    auto ExplicitSolutionNode_node_57_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(56);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(56);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 57
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_4 = this_solution[6];
            const double th_4th_6_soa = this_solution[7];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_4 + th_4th_6_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_57_solve_th_6_processor();
    // Finish code for explicit solution node 56
    
    // Code for explicit solution node 44, solved variable is negative_th_6_positive_th_4__soa
    auto ExplicitSolutionNode_node_44_solve_negative_th_6_positive_th_4__soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(44);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(44);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 44
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_3 = this_solution[5];
            
            const bool condition_0 = std::fabs(inv_Px) >= zero_tolerance || std::fabs(inv_Py) >= zero_tolerance || std::fabs(d_1*std::sin(th_3)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1/(d_1*std::sin(th_3));
                // End of temp variables
                const double tmp_sol_value = std::atan2(inv_Py*x0, inv_Px*x0);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(45, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_44_solve_negative_th_6_positive_th_4__soa_processor();
    // Finish code for explicit solution node 44
    
    // Code for solved_variable dispatcher node 45
    auto SolvedVariableDispatcherNode_node_45_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(45);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(45);
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
            }
            
            const bool degenerate_valid_1 = std::fabs(th_3 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(46, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_45_processor();
    // Finish code for solved_variable dispatcher node 45
    
    // Code for explicit solution node 46, solved variable is th_2
    auto ExplicitSolutionNode_node_46_solve_th_2_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(46);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(46);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 46
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_3 = this_solution[5];
            
            const bool condition_0 = std::fabs(r_13*std::sin(th_0) - r_23*std::cos(th_0)) >= zero_tolerance || std::fabs(r_13*std::cos(th_0)*std::cos(th_1) + r_23*std::sin(th_0)*std::cos(th_1) - r_33*std::sin(th_1)) >= zero_tolerance || std::fabs(std::sin(th_3)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_3);
                const double x1 = std::sin(th_0);
                const double x2 = std::cos(th_0);
                const double x3 = std::cos(th_1);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(r_13*x1 - r_23*x2), x0*(r_13*x2*x3 + r_23*x1*x3 - r_33*std::sin(th_1)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(47, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_46_solve_th_2_processor();
    // Finish code for explicit solution node 46
    
    // Code for non-branch dispatcher node 47
    // Actually, there is no code
    
    // Code for explicit solution node 48, solved variable is th_4
    auto ExplicitSolutionNode_node_48_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(47);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(47);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 48
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
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(49, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_48_solve_th_4_processor();
    // Finish code for explicit solution node 47
    
    // Code for non-branch dispatcher node 49
    // Actually, there is no code
    
    // Code for explicit solution node 50, solved variable is th_6
    auto ExplicitSolutionNode_node_50_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(49);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(49);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 50
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double negative_th_6_positive_th_4__soa = this_solution[1];
            const double th_4 = this_solution[6];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -negative_th_6_positive_th_4__soa + th_4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_50_solve_th_6_processor();
    // Finish code for explicit solution node 49
    
    // Code for explicit solution node 16, solved variable is th_6
    auto ExplicitSolutionNode_node_16_solve_th_6_processor = [&]() -> void
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
            const double th_5 = this_solution[8];
            
            const bool condition_0 = std::fabs(inv_Px) >= zero_tolerance || std::fabs(inv_Py) >= zero_tolerance || std::fabs((d_1 - d_2)*std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_5);
                // End of temp variables
                const double tmp_sol_value = std::atan2(inv_Py*x0/(d_1 - d_2), inv_Px*x0/(-d_1 + d_2));
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
                add_input_index_to(17, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_16_solve_th_6_processor();
    // Finish code for explicit solution node 16
    
    // Code for solved_variable dispatcher node 17
    auto SolvedVariableDispatcherNode_node_17_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(17);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(17);
        if (!this_input_valid)
            return;
        
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            bool taken_by_degenerate = false;
            const double th_5 = this_solution[8];
            
            const bool degenerate_valid_0 = std::fabs(th_5) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
            }
            
            const bool degenerate_valid_1 = std::fabs(th_5 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(18, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_17_processor();
    // Finish code for solved_variable dispatcher node 17
    
    // Code for explicit solution node 18, solved variable is negative_th_4_positive_th_2__soa
    auto ExplicitSolutionNode_node_18_solve_negative_th_4_positive_th_2__soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(18);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(18);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 18
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_5 = this_solution[8];
            
            const bool condition_0 = std::fabs(r_13*std::sin(th_0) - r_23*std::cos(th_0)) >= zero_tolerance || std::fabs(r_13*std::cos(th_0)*std::cos(th_1) + r_23*std::sin(th_0)*std::cos(th_1) - r_33*std::sin(th_1)) >= zero_tolerance || std::fabs(std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_5);
                const double x1 = std::cos(th_0);
                const double x2 = std::sin(th_0);
                const double x3 = std::cos(th_1);
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(-r_13*x2 + r_23*x1), x0*(-r_13*x1*x3 - r_23*x2*x3 + r_33*std::sin(th_1)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[0] = tmp_sol_value;
                add_input_index_to(19, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_18_solve_negative_th_4_positive_th_2__soa_processor();
    // Finish code for explicit solution node 18
    
    // Code for non-branch dispatcher node 19
    // Actually, there is no code
    
    // Code for explicit solution node 20, solved variable is th_2
    auto ExplicitSolutionNode_node_20_solve_th_2_processor = [&]() -> void
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
            
            const bool condition_0 = true;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = 0;
                solution_queue.get_solution(node_input_i_idx_in_queue)[3] = tmp_sol_value;
                add_input_index_to(21, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_20_solve_th_2_processor();
    // Finish code for explicit solution node 19
    
    // Code for non-branch dispatcher node 21
    // Actually, there is no code
    
    // Code for explicit solution node 22, solved variable is th_4
    auto ExplicitSolutionNode_node_22_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(21);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(21);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 22
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double negative_th_4_positive_th_2__soa = this_solution[0];
            const double th_2 = this_solution[3];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -negative_th_4_positive_th_2__soa + th_2;
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_22_solve_th_4_processor();
    // Finish code for explicit solution node 21
    
    // Code for explicit solution node 3, solved variable is th_2
    auto ExplicitSolutionNode_node_3_solve_th_2_processor = [&]() -> void
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
            const double th_3 = this_solution[5];
            
            const bool condition_0 = std::fabs((-Px*std::sin(th_0) + Py*std::cos(th_0) + d_3*(r_13*std::sin(th_0) - r_23*std::cos(th_0)))/(d_2*std::sin(th_3))) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = safe_asin((-Px*x1 + Py*x0 - d_3*(-r_13*x1 + r_23*x0))/(d_2*std::sin(th_3)));
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[3] = x2;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(4, appended_idx);
            }
            
            const bool condition_1 = std::fabs((-Px*std::sin(th_0) + Py*std::cos(th_0) + d_3*(r_13*std::sin(th_0) - r_23*std::cos(th_0)))/(d_2*std::sin(th_3))) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_0);
                const double x1 = std::sin(th_0);
                const double x2 = safe_asin((-Px*x1 + Py*x0 - d_3*(-r_13*x1 + r_23*x0))/(d_2*std::sin(th_3)));
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
    ExplicitSolutionNode_node_3_solve_th_2_processor();
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
            const bool checked_result = std::fabs(Pz - d_3*r_33) <= 9.9999999999999995e-7 && std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - d_3*r_13*std::cos(th_0) - d_3*r_23*std::sin(th_0)) <= 9.9999999999999995e-7;
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
            const double th_2 = this_solution[3];
            const double th_3 = this_solution[5];
            
            const bool condition_0 = std::fabs(Pz - d_3*r_33) >= 9.9999999999999995e-7 || std::fabs(Px*std::cos(th_0) + Py*std::sin(th_0) - d_3*r_13*std::cos(th_0) - d_3*r_23*std::sin(th_0)) >= 9.9999999999999995e-7;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = Pz - d_3*r_33;
                const double x1 = d_2*std::sin(th_3)*std::cos(th_2);
                const double x2 = -d_1 + d_2*std::cos(th_3);
                const double x3 = std::cos(th_0);
                const double x4 = std::sin(th_0);
                const double x5 = -Px*x3 - Py*x4 + d_3*r_13*x3 + d_3*r_23*x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*x1 - x2*x5, x0*x2 + x1*x5);
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
            const double th_1 = this_solution[2];
            
            const bool condition_0 = std::fabs((-d_1*(r_33*std::cos(th_1) + (r_13*std::cos(th_0) + r_23*std::sin(th_0))*std::sin(th_1)) + d_3 + inv_Pz)/d_2) <= 1;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = safe_acos((d_1*(-r_33*std::cos(th_1) - (r_13*std::cos(th_0) + r_23*std::sin(th_0))*std::sin(th_1)) + d_3 + inv_Pz)/d_2);
                // End of temp variables
                RawSolution solution_0(this_solution);
                solution_0[8] = x0;
                int appended_idx = append_solution_to_queue(solution_0);
                add_input_index_to(8, appended_idx);
            }
            
            const bool condition_1 = std::fabs((-d_1*(r_33*std::cos(th_1) + (r_13*std::cos(th_0) + r_23*std::sin(th_0))*std::sin(th_1)) + d_3 + inv_Pz)/d_2) <= 1;
            if (condition_1)
            {
                // Temp variable for efficiency
                const double x0 = safe_acos((d_1*(-r_33*std::cos(th_1) - (r_13*std::cos(th_0) + r_23*std::sin(th_0))*std::sin(th_1)) + d_3 + inv_Pz)/d_2);
                // End of temp variables
                const double tmp_sol_value = -x0;
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
            const double th_5 = this_solution[8];
            
            const bool degenerate_valid_0 = std::fabs(th_5) <= 9.9999999999999995e-7;
            if (degenerate_valid_0)
            {
                taken_by_degenerate = true;
                add_input_index_to(34, node_input_i_idx_in_queue);
            }
            
            const bool degenerate_valid_1 = std::fabs(th_5 - M_PI) <= 9.9999999999999995e-7;
            if (degenerate_valid_1)
            {
                taken_by_degenerate = true;
                add_input_index_to(39, node_input_i_idx_in_queue);
            }
            
            if (!taken_by_degenerate)
                add_input_index_to(9, node_input_i_idx_in_queue);
        }
    };
    
    // Invoke the processor
    SolvedVariableDispatcherNode_node_8_processor();
    // Finish code for solved_variable dispatcher node 8
    
    // Code for explicit solution node 39, solved variable is th_4th_6_soa
    auto ExplicitSolutionNode_node_39_solve_th_4th_6_soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(39);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(39);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 39
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_21*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_31*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(r_12*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_22*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_32*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_2);
                const double x1 = x0*std::sin(th_1);
                const double x2 = std::sin(th_0);
                const double x3 = std::cos(th_2);
                const double x4 = std::cos(th_0);
                const double x5 = x0*std::cos(th_1);
                const double x6 = x2*x3 - x4*x5;
                const double x7 = x2*x5 + x3*x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(r_11*x6 - r_21*x7 + r_31*x1, r_12*x6 - r_22*x7 + r_32*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[7] = tmp_sol_value;
                add_input_index_to(40, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_39_solve_th_4th_6_soa_processor();
    // Finish code for explicit solution node 39
    
    // Code for non-branch dispatcher node 40
    // Actually, there is no code
    
    // Code for explicit solution node 41, solved variable is th_4
    auto ExplicitSolutionNode_node_41_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(40);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(40);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 41
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
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(42, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_41_solve_th_4_processor();
    // Finish code for explicit solution node 40
    
    // Code for non-branch dispatcher node 42
    // Actually, there is no code
    
    // Code for explicit solution node 43, solved variable is th_6
    auto ExplicitSolutionNode_node_43_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(42);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(42);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 43
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_4 = this_solution[6];
            const double th_4th_6_soa = this_solution[7];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -th_4 + th_4th_6_soa;
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_43_solve_th_6_processor();
    // Finish code for explicit solution node 42
    
    // Code for explicit solution node 34, solved variable is negative_th_6_positive_th_4__soa
    auto ExplicitSolutionNode_node_34_solve_negative_th_6_positive_th_4__soa_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(34);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(34);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 34
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            
            const bool condition_0 = 1 >= zero_tolerance || std::fabs(r_11*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_21*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_31*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(r_12*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_22*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_32*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::sin(th_2);
                const double x1 = x0*std::sin(th_1);
                const double x2 = std::cos(th_0);
                const double x3 = std::cos(th_2);
                const double x4 = std::sin(th_0);
                const double x5 = x0*std::cos(th_1);
                const double x6 = x2*x3 + x4*x5;
                const double x7 = -x2*x5 + x3*x4;
                // End of temp variables
                const double tmp_sol_value = std::atan2(-r_11*x7 + r_21*x6 - r_31*x1, r_12*x7 - r_22*x6 + r_32*x1);
                solution_queue.get_solution(node_input_i_idx_in_queue)[1] = tmp_sol_value;
                add_input_index_to(35, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_34_solve_negative_th_6_positive_th_4__soa_processor();
    // Finish code for explicit solution node 34
    
    // Code for non-branch dispatcher node 35
    // Actually, there is no code
    
    // Code for explicit solution node 36, solved variable is th_4
    auto ExplicitSolutionNode_node_36_solve_th_4_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(35);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(35);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 36
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
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
                add_input_index_to(37, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_36_solve_th_4_processor();
    // Finish code for explicit solution node 35
    
    // Code for non-branch dispatcher node 37
    // Actually, there is no code
    
    // Code for explicit solution node 38, solved variable is th_6
    auto ExplicitSolutionNode_node_38_solve_th_6_processor = [&]() -> void
    {
        const auto& this_node_input_index = node_index_workspace.get_input_indices_for_node(37);
        const bool this_input_valid = node_index_workspace.is_input_indices_valid_for_node(37);
        if (!this_input_valid)
            return;
        
        // The solution of non-root node 38
        for(int i = 0; i < this_node_input_index.size(); i++)
        {
            int node_input_i_idx_in_queue = this_node_input_index[i];
            if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))
                continue;
            const auto& this_solution = solution_queue.get_solution(node_input_i_idx_in_queue);
            const double negative_th_6_positive_th_4__soa = this_solution[1];
            const double th_4 = this_solution[6];
            
            const bool condition_0 = 1 >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                // End of temp variables
                const double tmp_sol_value = -negative_th_6_positive_th_4__soa + th_4;
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_38_solve_th_6_processor();
    // Finish code for explicit solution node 37
    
    // Code for explicit solution node 9, solved variable is th_6
    auto ExplicitSolutionNode_node_9_solve_th_6_processor = [&]() -> void
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
            const double th_1 = this_solution[2];
            const double th_5 = this_solution[8];
            
            const bool condition_0 = std::fabs(d_2*std::sin(th_5)) >= zero_tolerance || std::fabs(-d_1*(-r_31*std::cos(th_1) - (r_11*std::cos(th_0) + r_21*std::sin(th_0))*std::sin(th_1)) - inv_Px) >= zero_tolerance || std::fabs(-d_1*(-r_32*std::cos(th_1) - (r_12*std::cos(th_0) + r_22*std::sin(th_0))*std::sin(th_1)) - inv_Py) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = std::cos(th_1);
                const double x1 = std::sin(th_1);
                const double x2 = std::cos(th_0);
                const double x3 = std::sin(th_0);
                const double x4 = 1/(d_2*std::sin(th_5));
                // End of temp variables
                const double tmp_sol_value = std::atan2(x4*(-d_1*(-r_32*x0 - x1*(r_12*x2 + r_22*x3)) - inv_Py), x4*(d_1*(-r_31*x0 - x1*(r_11*x2 + r_21*x3)) + inv_Px));
                solution_queue.get_solution(node_input_i_idx_in_queue)[9] = tmp_sol_value;
                add_input_index_to(10, node_input_i_idx_in_queue);
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_9_solve_th_6_processor();
    // Finish code for explicit solution node 9
    
    // Code for non-branch dispatcher node 10
    // Actually, there is no code
    
    // Code for explicit solution node 11, solved variable is th_4
    auto ExplicitSolutionNode_node_11_solve_th_4_processor = [&]() -> void
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
            const double th_1 = this_solution[2];
            const double th_2 = this_solution[3];
            const double th_3 = this_solution[5];
            const double th_5 = this_solution[8];
            
            const bool condition_0 = std::fabs(-r_13*((std::sin(th_1)*std::sin(th_3) + std::cos(th_1)*std::cos(th_2)*std::cos(th_3))*std::cos(th_0) + std::sin(th_0)*std::sin(th_2)*std::cos(th_3)) - r_23*((std::sin(th_1)*std::sin(th_3) + std::cos(th_1)*std::cos(th_2)*std::cos(th_3))*std::sin(th_0) - std::sin(th_2)*std::cos(th_0)*std::cos(th_3)) - r_33*(-std::sin(th_1)*std::cos(th_2)*std::cos(th_3) + std::sin(th_3)*std::cos(th_1))) >= zero_tolerance || std::fabs(r_13*(std::sin(th_0)*std::cos(th_2) - std::sin(th_2)*std::cos(th_0)*std::cos(th_1)) - r_23*(std::sin(th_0)*std::sin(th_2)*std::cos(th_1) + std::cos(th_0)*std::cos(th_2)) + r_33*std::sin(th_1)*std::sin(th_2)) >= zero_tolerance || std::fabs(std::sin(th_5)) >= zero_tolerance;
            if (condition_0)
            {
                // Temp variable for efficiency
                const double x0 = 1.0/std::sin(th_5);
                const double x1 = std::sin(th_1);
                const double x2 = std::sin(th_2);
                const double x3 = std::sin(th_0);
                const double x4 = std::cos(th_2);
                const double x5 = std::cos(th_0);
                const double x6 = std::cos(th_1);
                const double x7 = x2*x6;
                const double x8 = std::sin(th_3);
                const double x9 = std::cos(th_3);
                const double x10 = x4*x9;
                const double x11 = x2*x9;
                const double x12 = x1*x8 + x10*x6;
                // End of temp variables
                const double tmp_sol_value = std::atan2(x0*(r_13*(x3*x4 - x5*x7) - r_23*(x3*x7 + x4*x5) + r_33*x1*x2), x0*(-r_13*(x11*x3 + x12*x5) - r_23*(-x11*x5 + x12*x3) - r_33*(-x1*x10 + x6*x8)));
                solution_queue.get_solution(node_input_i_idx_in_queue)[6] = tmp_sol_value;
            }
            else
            {
                solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;
            }
        }
    };
    // Invoke the processor
    ExplicitSolutionNode_node_11_solve_th_4_processor();
    // Finish code for explicit solution node 10
    
    // Collect the output
    for(int i = 0; i < solution_queue.size(); i++)
    {
        if(!solution_queue.solutions_validity[i])
            continue;
        const auto& raw_ik_out_i = solution_queue.get_solution(i);
        std::array<double, robot_nq> new_ik_i;
        const double value_at_0 = th_0;  // th_0
        new_ik_i[0] = value_at_0;
        const double value_at_1 = raw_ik_out_i[2];  // th_1
        new_ik_i[1] = value_at_1;
        const double value_at_2 = raw_ik_out_i[3];  // th_2
        new_ik_i[2] = value_at_2;
        const double value_at_3 = raw_ik_out_i[5];  // th_3
        new_ik_i[3] = value_at_3;
        const double value_at_4 = raw_ik_out_i[6];  // th_4
        new_ik_i[4] = value_at_4;
        const double value_at_5 = raw_ik_out_i[8];  // th_5
        new_ik_i[5] = value_at_5;
        const double value_at_6 = raw_ik_out_i[9];  // th_6
        new_ik_i[6] = value_at_6;
        ik_output.push_back(new_ik_i);
    }
}

static void computeRawIK(const Eigen::Matrix4d& T_ee_raw, double th_0, RawIKWorksace& workspace)
{
    workspace.raw_ik_out.clear();
    workspace.raw_ik_out.reserve(max_n_solutions);
    computeRawIK(T_ee_raw, th_0, workspace.solution_queue, workspace.node_index_workspace, workspace.raw_ik_out);
}

static void computeIKUnChecked(const Eigen::Matrix4d& T_ee, double th_0, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, th_0 + th_0_offset_original2raw, workspace);
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
        ik_out_i[6] -= th_6_offset_original2raw;
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
        // Revolute unknown th_6
        while(solution_i[6] > M_PI)
            solution_i[6] -= 2 * M_PI;
        while(solution_i[6] < - M_PI)
            solution_i[6] += 2 * M_PI;
    }
}

static void computeIK(const Eigen::Matrix4d& T_ee, double th_0, RawIKWorksace& workspace, std::vector<std::array<double, robot_nq>>& ik_output)
{
    const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);
    computeRawIK(T_ee_raw, th_0 + th_0_offset_original2raw, workspace);
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
        ik_out_i[6] -= th_6_offset_original2raw;
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
    computeRawIK(T_ee_raw_disturbed, th_0 + th_0_offset_original2raw, workspace);
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
        ik_out_i[6] -= th_6_offset_original2raw;
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

static std::vector<std::array<double, robot_nq>> computeIK(const Eigen::Matrix4d& T_ee, double th_0)
{
    std::vector<std::array<double, robot_nq>> ik_output;
    RawIKWorksace raw_ik_workspace;
    computeIK(T_ee, th_0, raw_ik_workspace, ik_output);
    return ik_output;
}

}; // struct kuka_iiwa_ik

// Code below for debug
void test_ik_solve_kuka_iiwa()
{
    std::array<double, kuka_iiwa_ik::robot_nq> theta;
    std::random_device rd;
    std::uniform_real_distribution<double> distribution;
    for(auto i = 0; i < theta.size(); i++)
        theta[i] = distribution(rd);
    const Eigen::Matrix4d ee_pose = kuka_iiwa_ik::computeFK(theta);
    auto ik_output = kuka_iiwa_ik::computeIK(ee_pose, theta[0]);
    for(int i = 0; i < ik_output.size(); i++)
    {
        Eigen::Matrix4d ee_pose_i = kuka_iiwa_ik::computeFK(ik_output[i]);
        double ee_pose_diff = (ee_pose_i - ee_pose).norm();
        std::cout << "For solution " << i << " Pose different with ground-truth " << ee_pose_diff << std::endl;
    }
}

int main()
{
    test_ik_solve_kuka_iiwa();
}
