import fk.chain_transform as chain_transform
from fk.chain_transform import ModifiedDHEntry as NumericalDHEntry
from fk.robots import DHEntry as SymbolicDHEntry
from fk.robots import RobotDescription, RobotAuxiliaryData
from solver.equation_utils import default_unknowns, make_default_unknown
from typing import Optional, Dict
import sympy as sp
import numpy as np


def process_special_angle(angle_value: float) -> Optional[sp.Expr]:
    max_multiple_or_pi_div_2 = 20
    for i in range(max_multiple_or_pi_div_2):
        value_i = 0.5 * i * float(np.pi)
        if abs(angle_value - value_i) < 1e-6:
            return sp.simplify(i * sp.pi / 2)
        if abs(angle_value + value_i) < 1e-6:
            return sp.simplify(- i * sp.pi / 2)
    return None


def convert_to_symbolic_dh(
        numerical_dh: NumericalDHEntry,
        unknown_for_theta: sp.Symbol,
        value_map: Dict[sp.Symbol, float]) -> SymbolicDHEntry:
    # Init as special angles
    symbolic_dh = SymbolicDHEntry(sp.S.Zero, sp.S.Zero, sp.S.Zero, sp.S.Zero)
    symbolic_dh.theta = unknown_for_theta

    # For translation elements
    if abs(numerical_dh.a) > 1e-6:
        symbol_name = 'a_{idx}'.format(idx=len(value_map))
        symbolic_dh.a = sp.Symbol(symbol_name)
        value_map[symbolic_dh.a] = numerical_dh.a
    if abs(numerical_dh.d) > 1e-6:
        symbol_name = 'd_{idx}'.format(idx=len(value_map))
        symbolic_dh.d = sp.Symbol(symbol_name)
        value_map[symbolic_dh.d] = numerical_dh.d

    # Do a
    alpha_in_symbol = process_special_angle(numerical_dh.alpha)
    if alpha_in_symbol is None:
        symbol_name = 'alpha_{idx}'.format(idx=len(value_map))
        symbolic_dh.alpha = sp.Symbol(symbol_name)
        value_map[symbolic_dh.alpha] = numerical_dh.alpha
    else:
        symbolic_dh.alpha = alpha_in_symbol

    # Don't need to do anything on theta_offset
    return symbolic_dh


def make_auxiliary_data(robot: RobotDescription, robot_dh: chain_transform.RobotDH):
    """
    Add pre/post transform and joint offset to the robot. Maybe create new symbolic value on non-special value
    """
    def is_special_value(value: float) -> bool:
        special_value_in_transform_matrix = [-1, 0, 1]
        for elem in special_value_in_transform_matrix:
            if abs(elem - value) < 1e-6:
                return True
        return False

    # The processing of matrix
    def process_transform_matrix(np_matrix: np.ndarray, name_prefix: str):
        symbol_map: Dict[sp.Symbol, float] = dict()
        sp_matrix = sp.Matrix(np_matrix)
        for r in range(4):
            for c in range(4):
                if is_special_value(np_matrix[r, c]):
                    continue
                symbol_name_rc = name_prefix + '_{r}{c}'.format(r=r, c=c)
                symbol_rc = sp.Symbol(symbol_name_rc)
                sp_matrix[r, c] = symbol_rc
                symbol_map[symbol_rc] = float(np_matrix[r, c])
        return sp_matrix, symbol_map

    # Invoke the functor
    pre_transform_matrix, pre_transform_map = \
        process_transform_matrix(robot_dh.pre_transform, 'pre_transform_special_symbol')
    post_transform_matrix, post_transform_map = \
        process_transform_matrix(robot_dh.post_transform, 'post_transform_special_symbol')

    # Add to aux data
    robot.auxiliary_data = RobotAuxiliaryData()
    robot.auxiliary_data.pre_transform_sp = pre_transform_matrix
    robot.auxiliary_data.post_transform_sp = post_transform_matrix
    robot.auxiliary_data.unknown_offset = [float(elem.theta_offset) for elem in robot_dh.dh_parameters]

    # Add to dict
    robot.parameters_value.update(pre_transform_map)
    robot.parameters_value.update(post_transform_map)
    robot.symbolic_parameters.update([elem for elem in pre_transform_map])
    robot.symbolic_parameters.update([elem for elem in post_transform_map])


def convert_chain_into_robot(robot_name: str, chain: chain_transform.ChainTransform) -> Optional[RobotDescription]:
    """
    Convert a robot described by a kinematic chain into a solvable format.
    Currently, only revolute joint wrt x/y/z/-x/-y/-z are supported
    """
    robot_dh = chain_transform.try_convert_to_dh(chain)
    if robot_dh is None:
        return None

    # Not one, into symbolic form
    robot = RobotDescription(robot_name)
    n_dof = len(robot_dh.dh_parameters)
    assert n_dof == chain.chain_length
    robot.unknowns = default_unknowns(n_dof)

    # Get the dh
    robot.dh_params = list()
    robot.parameters_value = dict()
    for i in range(n_dof):
        symbolic_dh_i = convert_to_symbolic_dh(
            robot_dh.dh_parameters[i],
            robot.unknowns[i].symbol,
            robot.parameters_value)
        robot.dh_params.append(symbolic_dh_i)

    # No bound
    robot.parameters_bound = dict()

    # Add value to symbolic parameters
    robot.symbolic_parameters = set()
    for symbol_key in robot.parameters_value:
        robot.symbolic_parameters.add(symbol_key)

    # The auxiliary data
    make_auxiliary_data(robot, robot_dh)
    return robot


def run_convert():
    import yaml
    import fk.chain_models as chain_models
    robot_chain = chain_models.abb_irb_6700_205_2_80()
    robot_name = 'spherical_wrist_six_axis'
    robot = convert_chain_into_robot(robot_name, robot_chain)
    robot_dict = robot.to_dict()

    # Save it
    save_path = robot.name + '.yaml'
    with open(save_path, 'w') as file_stream:
        yaml.dump(robot_dict, file_stream, Dumper=yaml.CDumper)
    file_stream.close()


if __name__ == '__main__':
    run_convert()
