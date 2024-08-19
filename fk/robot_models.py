from fk.fk_equations import DHEntry
from solver.equation_types import Unknown, UnknownType
from fk.robots import RobotDescription, RobotAuxiliaryData
from solver.equation_utils import default_unknowns, make_default_unknown, default_unknowns_with_offset
import numpy as np
import sympy as sp


def puma_robot() -> RobotDescription:
    n_dofs = 6
    robot = RobotDescription("puma")
    robot.unknowns = default_unknowns(n_dofs)
    a_2 = sp.Symbol('a_2')
    a_3 = sp.Symbol('a_3')
    d_1 = sp.Symbol('d_1')
    d_3 = sp.Symbol('d_3')
    d_4 = sp.Symbol('d_4')
    dh_0 = DHEntry(0, 0, d_1, robot.unknowns[0].symbol)
    dh_1 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[1].symbol)
    dh_2 = DHEntry(0, a_2, d_3, robot.unknowns[2].symbol)
    dh_3 = DHEntry(-sp.pi / 2, a_3, d_4, robot.unknowns[3].symbol)
    dh_4 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[4].symbol)
    dh_5 = DHEntry(sp.pi / 2, 0, 0, robot.unknowns[5].symbol)
    robot.dh_params = [dh_0, dh_1, dh_2, dh_3, dh_4, dh_5]

    # Make the robot
    robot.symbolic_parameters = {d_1, a_2, a_3, d_3, d_4}
    robot.parameters_value = {d_1: 0.6, a_2: 0.432, a_3: 0.0203, d_3: 0.1245, d_4: 0.432}
    robot.parameters_bound = dict()
    return robot


def arm_robo() -> RobotDescription:
    n_dofs = 6
    robot = RobotDescription("arm_robo")
    # For the unknown
    l_1 = sp.Symbol('l_1')
    l_2 = sp.Symbol('l_2')
    l_3 = sp.Symbol('l_3')
    d_6 = sp.Symbol('d_6')
    unknowns = default_unknowns(n_dofs - 1)
    unknowns.append(Unknown(d_6, UnknownType.Translational.name))
    robot.unknowns = unknowns

    dh_0 = DHEntry(sp.pi / 2, 0, l_2, unknowns[0].symbol)
    dh_1 = DHEntry(0, l_1, 0, unknowns[1].symbol)
    dh_2 = DHEntry(sp.pi / 2, 0, 0, unknowns[2].symbol)
    dh_3 = DHEntry(-sp.pi / 2, 0, l_3, unknowns[3].symbol)
    dh_4 = DHEntry(sp.pi / 2, 0, 0, unknowns[4].symbol)
    dh_5 = DHEntry(0, 0, d_6, 0)
    robot.dh_params = [dh_0, dh_1, dh_2, dh_3, dh_4, dh_5]
    robot.symbolic_parameters = {l_1, l_2, l_3}
    robot.parameters_value = {l_1: 0.19681, l_2: 0.251, l_3: 0.145423}
    robot.parameters_bound = dict()
    return robot


def mini_dd_robot() -> RobotDescription:
    n_dofs = 5
    robot = RobotDescription("mini_dd")
    unknown_d_0 = make_default_unknown(0, False)
    unknown_th_1 = make_default_unknown(1)
    unknown_th_2 = make_default_unknown(2)
    unknown_th_3 = make_default_unknown(3)
    unknown_th_4 = make_default_unknown(4)
    robot.unknowns = [unknown_d_0, unknown_th_1, unknown_th_2, unknown_th_3, unknown_th_4]

    # The dh entry
    l_3 = sp.Symbol('l_3')
    l_4 = sp.Symbol('l_4')
    dh_0 = DHEntry(0, 0, unknown_d_0.symbol, 0)
    dh_1 = DHEntry(-sp.pi / 2, 0, 0, unknown_th_1.symbol)
    dh_2 = DHEntry(-sp.pi / 2, l_3, 0, unknown_th_2.symbol)
    dh_3 = DHEntry(-sp.pi / 2, 0, l_4, unknown_th_3.symbol)
    dh_4 = DHEntry(-sp.pi / 2, 0, 0, unknown_th_4.symbol)
    robot.dh_params = [dh_0, dh_1, dh_2, dh_3, dh_4]
    robot.symbolic_parameters = {l_3, l_4}
    robot.parameters_value = {l_3: 5.0, l_4: 2.0}
    robot.parameters_bound = dict()
    return robot


def test_robot() -> RobotDescription:
    n_dofs = 6
    robot = RobotDescription("test_bot")
    robot.unknowns = default_unknowns(n_dofs)
    a_0 = sp.Symbol('a_0')
    a_1 = sp.Symbol('a_1')
    d_2 = sp.Symbol('a_2')
    a_3 = sp.Symbol('a_3')

    dh_0 = DHEntry(0, 0, 0, robot.unknowns[0].symbol)
    dh_1 = DHEntry(0, a_1, 0, robot.unknowns[1].symbol)
    dh_2 = DHEntry(sp.pi / 2, 0, d_2, robot.unknowns[2].symbol)
    dh_3 = DHEntry(0, a_3, 0, robot.unknowns[3].symbol)
    dh_4 = DHEntry(sp.pi / 2, 0, 0, robot.unknowns[4].symbol)
    dh_5 = DHEntry(0, 0, 0, robot.unknowns[5].symbol)
    robot.dh_params = [dh_0, dh_1, dh_2, dh_3, dh_4, dh_5]
    robot.symbolic_parameters = {a_0, a_1, d_2, a_3}
    robot.parameters_value = {a_0: 0.3, a_1: 1.0, d_2: .2, a_3: 1.5}
    return robot


def yaskawa_HC10_robot() -> RobotDescription:
    n_dofs = 6
    robot = RobotDescription("yaskawa_HC10")
    robot.unknowns = default_unknowns_with_offset(n_dofs, offset=0)
    a_0, d_1, d_2, d_3 = sp.symbols('a_0 d_1 d_2 d_3')
    dh_0 = DHEntry(0, 0, 0, robot.unknowns[0].symbol)
    dh_1 = DHEntry(- sp.pi / 2, 0, 0, robot.unknowns[1].symbol)
    dh_2 = DHEntry(- sp.pi, a_0, 0, robot.unknowns[2].symbol)
    dh_3 = DHEntry(- sp.pi / 2, 0, d_1, robot.unknowns[3].symbol)
    dh_4 = DHEntry(- sp.pi / 2, 0, d_2, robot.unknowns[4].symbol)
    dh_5 = DHEntry(- sp.pi / 2, 0, d_3, robot.unknowns[5].symbol)
    robot.dh_params = [dh_0, dh_1, dh_2, dh_3, dh_4, dh_5]
    robot.symbolic_parameters = {a_0, d_1, d_2, d_3}
    robot.parameters_value = {a_0: 0.7, d_1: 0.5, d_2: 0.162, d_3: 0.13}
    return robot


def ur10_urdf_robot() -> RobotDescription:
    n_dofs = 6
    robot = RobotDescription("ur10_urdf")
    robot.unknowns = default_unknowns(n_dofs)

    # The dh entry
    d_1 = sp.Symbol('d_1')
    a_2 = sp.Symbol('a_2')
    d_2 = sp.Symbol('d_2')
    a_3 = sp.Symbol('a_3')
    d_3 = sp.Symbol('d_3')
    d_4 = sp.Symbol('d_4')
    pre_transform_d4 = sp.Symbol('pre_transform_d4')
    post_transform_d5 = sp.Symbol('post_transform_d5')
    dh_0 = DHEntry(0, 0, 0, robot.unknowns[0].symbol)
    dh_1 = DHEntry(-sp.pi / 2, 0, d_1, robot.unknowns[1].symbol)
    dh_2 = DHEntry(0, a_2, d_2, robot.unknowns[2].symbol)
    dh_3 = DHEntry(0, a_3, d_3, robot.unknowns[3].symbol)
    dh_4 = DHEntry(-sp.pi / 2, 0, d_4, robot.unknowns[4].symbol)
    dh_5 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[5].symbol)
    robot.dh_params = [dh_0, dh_1, dh_2, dh_3, dh_4, dh_5]
    robot.symbolic_parameters = {d_1, a_2, d_2, a_3, d_3, d_4}
    robot.parameters_value = {d_1: 0.220941, a_2: 0.612, d_2: -0.1719, a_3: 0.5723, d_3: 0.1149, d_4: 0.1157,
                              pre_transform_d4: 0.1273, post_transform_d5: 0.0922}
    robot.parameters_bound = dict()

    # Add auxiliary data
    pi_float = float(np.pi)
    robot.auxiliary_data = RobotAuxiliaryData()
    robot.auxiliary_data.unknown_offset = [0.0, 0.0, 0.0, 0.0, pi_float, pi_float]
    robot.auxiliary_data.pre_transform_sp = sp.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, pre_transform_d4],
        [0, 0, 0, 1]
    ])
    robot.auxiliary_data.post_transform_sp = sp.Matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, post_transform_d5],
        [0, 0, 0, 1]
    ])

    # Finished
    return robot


def kuka_iiwa_robot() -> RobotDescription:
    n_dofs = 7
    robot = RobotDescription("kuka_iiwa")
    robot.unknowns = default_unknowns(n_dofs)

    # The dh entry
    d_1 = sp.Symbol('d_1')
    d_2 = sp.Symbol('d_2')
    d_3 = sp.Symbol('d_3')
    pre_transform_d4 = sp.Symbol('pre_transform_d4')
    post_transform_d5 = sp.Symbol('post_transform_d5')
    dh_0 = DHEntry(0, 0, 0, robot.unknowns[0].symbol)
    dh_1 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[1].symbol)
    dh_2 = DHEntry(-sp.pi / 2, 0, d_1, robot.unknowns[2].symbol)
    dh_3 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[3].symbol)
    dh_4 = DHEntry(-sp.pi / 2, 0, d_2, robot.unknowns[4].symbol)
    dh_5 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[5].symbol)
    dh_6 = DHEntry(-sp.pi / 2, 0, d_3, robot.unknowns[6].symbol)
    robot.dh_params = [dh_0, dh_1, dh_2, dh_3, dh_4, dh_5, dh_6]
    robot.symbolic_parameters = {d_1, d_2, d_3}
    robot.parameters_value = {d_1: 0.42, d_2: 0.4, d_3: 0.081, pre_transform_d4: 0.36, post_transform_d5: 0.045}
    robot.parameters_bound = dict()
    robot.unknown_as_parameter_more_dof = [robot.unknowns[0].symbol]

    # Add auxiliary data
    pi_float = float(np.pi)
    robot.auxiliary_data = RobotAuxiliaryData()
    robot.auxiliary_data.unknown_offset = [0.0, pi_float, 0.0, -pi_float, 0.0, -pi_float, pi_float]
    robot.auxiliary_data.pre_transform_sp = sp.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, pre_transform_d4],
        [0, 0, 0, 1]
    ])
    robot.auxiliary_data.post_transform_sp = sp.Matrix([
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, post_transform_d5],
        [0, 0, 0, 1]
    ])

    # Finished
    return robot


def pr2_r_gripper_palm() -> RobotDescription:
    n_dofs = 7
    robot = RobotDescription("pr2_r_gripper_palm")
    robot.unknowns = default_unknowns(n_dofs)

    # The dh entry
    a_1 = sp.Symbol('a_1')
    d_2 = sp.Symbol('d_2')
    d_4 = sp.Symbol('d_4')
    pre_transform_s23 = sp.Symbol('pre_transform_s23')
    dh_0 = DHEntry(0, 0, 0, robot.unknowns[0].symbol)
    dh_1 = DHEntry(-sp.pi / 2, a_1, 0, robot.unknowns[1].symbol)
    dh_2 = DHEntry(-sp.pi / 2, 0, d_2, robot.unknowns[2].symbol)
    dh_3 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[3].symbol)
    dh_4 = DHEntry(-sp.pi / 2, 0, d_4, robot.unknowns[4].symbol)
    dh_5 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[5].symbol)
    dh_6 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[6].symbol)
    robot.dh_params = [dh_0, dh_1, dh_2, dh_3, dh_4, dh_5, dh_6]
    robot.symbolic_parameters = {a_1, d_2, d_4}
    robot.parameters_value = {a_1: 0.1, d_2: 0.4, d_4: 0.321, pre_transform_s23: -0.188}
    robot.parameters_bound = dict()
    robot.unknown_as_parameter_more_dof = [robot.unknowns[0].symbol]

    # Add auxiliary data
    pi_float = float(np.pi)
    robot.auxiliary_data = RobotAuxiliaryData()
    robot.auxiliary_data.unknown_offset = [0.0, - 0.5 * pi_float, pi_float, pi_float, pi_float, pi_float, 0.0]
    robot.auxiliary_data.pre_transform_sp = sp.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, pre_transform_s23],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    robot.auxiliary_data.post_transform_sp = sp.Matrix([
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])

    # Finished
    return robot


def franka_panda_robot() -> RobotDescription:
    n_dofs = 7
    robot = RobotDescription("franka_panda")
    robot.unknowns = default_unknowns(n_dofs)

    # The dh entry
    a_3 = sp.Symbol('a_3')
    a_5 = sp.Symbol('a_5')
    d_2 = sp.Symbol('d_2')
    d_4 = sp.Symbol('d_4')
    pre_transform_d4 = sp.Symbol('pre_transform_d4')
    post_transform_d5 = sp.Symbol('post_transform_d5')
    post_transform_sqrt2_over2 = sp.Symbol('post_transform_sqrt2_over2')
    dh_0 = DHEntry(0, 0, 0, robot.unknowns[0].symbol)
    dh_1 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[1].symbol)
    dh_2 = DHEntry(-sp.pi / 2, 0, d_2, robot.unknowns[2].symbol)
    dh_3 = DHEntry(sp.pi / 2, a_3, 0, robot.unknowns[3].symbol)
    dh_4 = DHEntry(sp.pi / 2, a_3, d_4, robot.unknowns[4].symbol)
    dh_5 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[5].symbol)
    dh_6 = DHEntry(sp.pi / 2, a_5, 0, robot.unknowns[6].symbol)
    robot.dh_params = [dh_0, dh_1, dh_2, dh_3, dh_4, dh_5, dh_6]
    robot.symbolic_parameters = {d_2, a_5, a_3, d_4, pre_transform_d4, post_transform_d5, post_transform_sqrt2_over2}
    robot.parameters_value = {d_2: 0.316, d_4: 0.384, a_3: 0.00825, a_5: 0.088,
                              pre_transform_d4: 0.333, post_transform_d5: 0.107,
                              post_transform_sqrt2_over2: 0.707107}
    robot.parameters_bound = dict()
    robot.unknown_as_parameter_more_dof = [robot.unknowns[3].symbol]

    # Add auxiliary data
    pi_float = float(np.pi)
    robot.auxiliary_data = RobotAuxiliaryData()
    robot.auxiliary_data.unknown_offset = [0.0, pi_float, pi_float, pi_float, 0.0, pi_float, 0.0]
    robot.auxiliary_data.pre_transform_sp = sp.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, pre_transform_d4],
        [0, 0, 0, 1]
    ])
    robot.auxiliary_data.post_transform_sp = sp.Matrix([
        [post_transform_sqrt2_over2, post_transform_sqrt2_over2, 0, 0],
        [-post_transform_sqrt2_over2, post_transform_sqrt2_over2, 0, 0],
        [0, 0, 1, post_transform_d5],
        [0, 0, 0, 1]
    ])

    # Finished
    return robot


def atlas_l_hand():
    n_dofs = 7
    robot = RobotDescription("atlas_l_hand")
    robot.unknowns = default_unknowns(n_dofs)

    # The dh entry
    a_1 = sp.Symbol('a_1')
    a_2 = sp.Symbol('a_2')
    d_2 = sp.Symbol('d_2')
    a_3 = sp.Symbol('a_3')
    a_4 = sp.Symbol('a_4')
    d_4 = sp.Symbol('d_4')
    pre_transform_s0 = sp.Symbol('pre_transform_s0')
    pre_transform_s1 = sp.Symbol('pre_transform_s1')
    pre_transform_s2 = sp.Symbol('pre_transform_s2')
    dh_0 = DHEntry(0, 0, 0, robot.unknowns[0].symbol)
    dh_1 = DHEntry(sp.pi / 2, a_1, 0, robot.unknowns[1].symbol)
    dh_2 = DHEntry(-sp.pi / 2, a_2, d_2, robot.unknowns[2].symbol)
    dh_3 = DHEntry(-sp.pi / 2, a_3, 0, robot.unknowns[3].symbol)
    dh_4 = DHEntry(-sp.pi / 2, a_4, d_4, robot.unknowns[4].symbol)
    dh_5 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[5].symbol)
    dh_6 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[6].symbol)
    robot.dh_params = [dh_0, dh_1, dh_2, dh_3, dh_4, dh_5, dh_6]
    robot.symbolic_parameters = {a_1, a_2, d_2, a_3, a_4, d_4}
    robot.parameters_value = {
        a_1: 0.11,
        a_2: 0.016, d_2: 0.306,
        a_3: 0.0092,
        a_4: 0.00921, d_4: 0.29955,
        pre_transform_s0: 0.1406,
        pre_transform_s1: 0.2256,
        pre_transform_s2: 0.2326}
    robot.parameters_bound = dict()
    robot.unknown_as_parameter_more_dof = [robot.unknowns[0].symbol]

    # Add auxiliary data
    pi_float = float(np.pi)
    robot.auxiliary_data = RobotAuxiliaryData()
    robot.auxiliary_data.unknown_offset = [
        0.0,
        - 0.5 * pi_float,
        pi_float, pi_float, pi_float, pi_float,
        - 0.5 * pi_float]
    robot.auxiliary_data.pre_transform_sp = sp.Matrix([
        [0, -1, 0, pre_transform_s0],
        [1, 0, 0, pre_transform_s1],
        [0, 0, 1, pre_transform_s2],
        [0, 0, 0, 1]
    ])
    robot.auxiliary_data.post_transform_sp = sp.Matrix([
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])

    return robot


def atlas_r_hand():
    n_dofs = 7
    robot = RobotDescription("atlas_r_hand")
    robot.unknowns = default_unknowns(n_dofs)

    # The dh entry
    a_1 = sp.Symbol('a_1')
    a_2 = sp.Symbol('a_2')
    d_2 = sp.Symbol('d_2')
    a_3 = sp.Symbol('a_3')
    a_4 = sp.Symbol('a_4')
    d_4 = sp.Symbol('d_4')
    pre_transform_s0 = sp.Symbol('pre_transform_s0')
    pre_transform_s1 = sp.Symbol('pre_transform_s1')
    pre_transform_s2 = sp.Symbol('pre_transform_s2')
    dh_0 = DHEntry(0, 0, 0, robot.unknowns[0].symbol)
    dh_1 = DHEntry(-sp.pi / 2, a_1, 0, robot.unknowns[1].symbol)
    dh_2 = DHEntry(-sp.pi / 2, a_2, d_2, robot.unknowns[2].symbol)
    dh_3 = DHEntry(-sp.pi / 2, a_3, 0, robot.unknowns[3].symbol)
    dh_4 = DHEntry(-sp.pi / 2, a_4, d_4, robot.unknowns[4].symbol)
    dh_5 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[5].symbol)
    dh_6 = DHEntry(-sp.pi / 2, 0, 0, robot.unknowns[6].symbol)
    robot.dh_params = [dh_0, dh_1, dh_2, dh_3, dh_4, dh_5, dh_6]
    robot.symbolic_parameters = {a_1, a_2, d_2, a_3, a_4, d_4}
    robot.parameters_value = {
        a_1: 0.11,
        a_2: 0.016,
        d_2: -0.306,
        a_3: 0.0092,
        a_4: 0.00921,
        d_4: -0.29955,
        pre_transform_s0: 0.1406,
        pre_transform_s1: -0.2256,
        pre_transform_s2: 0.2326}
    robot.parameters_bound = dict()
    robot.unknown_as_parameter_more_dof = [robot.unknowns[0].symbol]

    # Add auxiliary data
    pi_float = float(np.pi)
    robot.auxiliary_data = RobotAuxiliaryData()
    robot.auxiliary_data.unknown_offset = [
        0.0,
        0.5 * pi_float,
        pi_float, pi_float, pi_float, pi_float,
        0.5 * pi_float]
    robot.auxiliary_data.pre_transform_sp = sp.Matrix([
        [0, 1, 0, pre_transform_s0],
        [-1, 0, 0, pre_transform_s1],
        [0, 0, 1, pre_transform_s2],
        [0, 0, 0, 1]
    ])
    robot.auxiliary_data.post_transform_sp = sp.Matrix([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    return robot
