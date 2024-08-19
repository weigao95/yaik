from fk.fk_equations import ik_target_4x4
from solver.general_6dof import dh_utils
from solver.general_6dof.general_closure_equation import UnReducedMatrixClosureEquation
import solver.general_6dof.general_closure_equation as closure
from solver.general_6dof.matrix_closure_equation import ReducedRaghavanRothEquationFormatter
import fk.robots as robots
from typing import List, Optional, Tuple
from utility import symbolic_utils
from solver.equation_utils import cast_expr_to_float
import sympy as sp
import attr


def ik_target_4x4_internal() -> sp.Matrix:
    """
    The "temporary" rhs target for the general 6dof solve. We use this matrix to simplify the processing
    and substitute back to actual rhs_target after solve. The actual rhs_target is just ik_target_4x4 for
    6-DoF robot, but may contain other term for 7 or more DoFs robot.
    """
    m = sp.zeros(4)
    for i in [1, 2, 3]:
        for j in [1, 2, 3]:
            v = 'r_tmp' + '{:1}{:1}'.format(i, j)
            m[i - 1, j - 1] = sp.var(v)

    m[0, 3] = sp.Symbol('Px_tmp')
    m[1, 3] = sp.Symbol('Py_tmp')
    m[2, 3] = sp.Symbol('Pz_tmp')
    m[3, 0] = 0
    m[3, 1] = 0
    m[3, 2] = 0
    m[3, 3] = 1
    return m


@attr.s
class ReduceInput(object):
    lhs_matrix: sp.Matrix = attr.ib()
    rhs_matrix: sp.Matrix = attr.ib()
    rhs_var_to_remove: dh_utils.RevoluteVariable = attr.ib()


def build_reduce_inputs(
        robot: robots.RobotDescription) -> Optional[Tuple[List[ReduceInput], List[dh_utils.RevoluteVariable], sp.Matrix]]:
    attempt_last_N_unknowns = build_reduce_input_last_N_unknowns_as_parameters(robot)
    if attempt_last_N_unknowns is not None:
        return attempt_last_N_unknowns
    attempt_first_N_unknowns = build_reduce_input_first_N_unknowns_as_parameters(robot)
    return attempt_first_N_unknowns


def first_N_unkonwns_are_parameters(robot: robots.RobotDescription) -> Optional[int]:
    """
    If the only unknowns treated as parameter are robot.unknown[0], robot.unknown[1] ...
    This function return the # of unknowns. If there are other unknowns, return None
    """
    if len(robot.dh_params) != len(robot.unknowns):
        return None
    if len(robot.unknowns) != 6 + len(robot.unknown_as_parameter_more_dof):
        return None

    n_unknown_as_parameter_from_start = 0
    for i in range(len(robot.unknowns)):
        if robot.unknowns[i].symbol in robot.unknown_as_parameter_more_dof:
            n_unknown_as_parameter_from_start += 1
            continue
        else:
            break
    if n_unknown_as_parameter_from_start != len(robot.unknown_as_parameter_more_dof):
        return None
    else:
        return n_unknown_as_parameter_from_start


def last_N_unknowns_are_parameters(robot: robots.RobotDescription) -> Optional[int]:
    """
    If the only unknowns treated as parameter are robot.unknown[-1], robot.unknown[-2] ...
    This function return the # of unknowns. If there are other unknowns, return None
    """
    if len(robot.dh_params) != len(robot.unknowns):
        return None
    if len(robot.unknowns) != 6 + len(robot.unknown_as_parameter_more_dof):
        return None

    n_unknown_as_parameter_from_end = 0
    for i in reversed(range(len(robot.unknowns))):
        if robot.unknowns[i].symbol in robot.unknown_as_parameter_more_dof:
            n_unknown_as_parameter_from_end += 1
            continue
        else:
            break
    if n_unknown_as_parameter_from_end != len(robot.unknown_as_parameter_more_dof):
        return None
    else:
        return n_unknown_as_parameter_from_end


def generate_reduce_equation(
        dh_params: List[dh_utils.DHEntry],
        revolute_vars: List[dh_utils.RevoluteVariable],
        offset: int = 0) -> Optional[Tuple[List[ReduceInput], List[dh_utils.RevoluteVariable]]]:
    """
    Take 6 dh-parameters and unknowns from dh_params and revolute_vars with start at @param offset (include offset)
    Construct the equation for reduction.
    """
    if len(dh_params) != len(revolute_vars):
        return None
    if offset + 5 >= len(dh_params):
        return None

    # These are variables to solve
    A1v, A1s = dh_utils.reflected_variable_and_structure_transform(dh_params[offset + 0])
    A1 = A1v * A1s
    A2v, A2s = dh_utils.reflected_variable_and_structure_transform(dh_params[offset + 1])
    A2 = A2v * A2s
    A3v, A3s = dh_utils.reflected_variable_and_structure_transform(dh_params[offset + 2])
    A3 = A3v * A3s
    A4v, A4s = dh_utils.reflected_variable_and_structure_transform(dh_params[offset + 3])
    A4 = A4v * A4s
    A5v, A5s = dh_utils.reflected_variable_and_structure_transform(dh_params[offset + 4])
    A5 = A5v * A5s
    A6v, A6s = dh_utils.reflected_variable_and_structure_transform(dh_params[offset + 5])
    A6 = A6v * A6s

    # The rhs
    inv_A1 = symbolic_utils.inverse_transform(A1)
    inv_A2 = symbolic_utils.inverse_transform(A2)
    inv_A3 = symbolic_utils.inverse_transform(A3)
    inv_A4 = symbolic_utils.inverse_transform(A4)
    inv_A5 = symbolic_utils.inverse_transform(A5)
    inv_A6 = symbolic_utils.inverse_transform(A6)
    inv_A1v = sp.transpose(A1v)
    inv_A2v = sp.transpose(A2v)
    inv_A3v = sp.transpose(A3v)
    inv_A4v = sp.transpose(A4v)
    inv_A5v = sp.transpose(A5v)
    inv_A6v = sp.transpose(A6v)

    # The reducer inputs
    rhs_target = ik_target_4x4_internal()
    reduce_input_list: List[ReduceInput] = list()
    lhs_matrix = A3s * A4 * A5 * A6 * symbolic_utils.inverse_transform(rhs_target)
    rhs_matrix = inv_A3v * inv_A2 * inv_A1
    var2remove = revolute_vars[offset + 0]
    reduce_input_list.append(ReduceInput(lhs_matrix=lhs_matrix, rhs_matrix=rhs_matrix, rhs_var_to_remove=var2remove))

    lhs_matrix = A4s * A5 * A6 * symbolic_utils.inverse_transform(rhs_target) * A1
    rhs_matrix = inv_A4v * inv_A3 * inv_A2
    var2remove = revolute_vars[offset + 1]
    reduce_input_list.append(ReduceInput(lhs_matrix=lhs_matrix, rhs_matrix=rhs_matrix, rhs_var_to_remove=var2remove))

    lhs_matrix = A5s * A6 * symbolic_utils.inverse_transform(rhs_target) * A1 * A2
    rhs_matrix = inv_A5v * inv_A4 * inv_A3
    var2remove = revolute_vars[offset + 2]
    reduce_input_list.append(ReduceInput(lhs_matrix=lhs_matrix, rhs_matrix=rhs_matrix, rhs_var_to_remove=var2remove))

    lhs_matrix = A6s * symbolic_utils.inverse_transform(rhs_target) * A1 * A2 * A3
    rhs_matrix = inv_A6v * inv_A5 * inv_A4
    var2remove = revolute_vars[offset + 3]
    reduce_input_list.append(ReduceInput(lhs_matrix=lhs_matrix, rhs_matrix=rhs_matrix, rhs_var_to_remove=var2remove))

    lhs_matrix = A1s * A2 * A3 * A4
    rhs_matrix = inv_A1v * rhs_target * inv_A6 * inv_A5
    var2remove = revolute_vars[offset + 4]
    reduce_input_list.append(ReduceInput(lhs_matrix=lhs_matrix, rhs_matrix=rhs_matrix, rhs_var_to_remove=var2remove))

    lhs_matrix = A2s * A3 * A4 * A5
    rhs_matrix = inv_A2v * inv_A1 * rhs_target * inv_A6
    var2remove = revolute_vars[offset + 5]
    reduce_input_list.append(ReduceInput(lhs_matrix=lhs_matrix, rhs_matrix=rhs_matrix, rhs_var_to_remove=var2remove))

    # OK
    return reduce_input_list, revolute_vars[offset:offset + 6]


def build_reduce_input_last_N_unknowns_as_parameters(robot: robots.RobotDescription) -> \
        Optional[Tuple[List[ReduceInput], List[dh_utils.RevoluteVariable], sp.Matrix]]:
    n_unknown_as_parameter_from_end = last_N_unknowns_are_parameters(robot)
    if n_unknown_as_parameter_from_end is None:
        return None

    dh_params = dh_utils.modified_dh_to_classic(robot.dh_params)
    revolute_vars = dh_utils.RevoluteVariable.convert_from_robot_unknowns(robot)
    if revolute_vars is None or dh_params is None:
        return None

    # The beginning
    A_last = sp.eye(4)
    for i in range(6, len(robot.unknowns)):
        A_i_v, A_i_s = dh_utils.reflected_variable_and_structure_transform(dh_params[i])
        A_i = A_i_v * A_i_s
        A_last = A_last * A_i
    inv_A_last = symbolic_utils.inverse_transform(A_last)
    rhs_target_subst = ik_target_4x4() * inv_A_last

    equation_out = generate_reduce_equation(dh_params, revolute_vars, offset=0)
    if equation_out is None:
        return None

    # Finished
    reduce_input_list, _ = equation_out
    return reduce_input_list, revolute_vars[0:6], rhs_target_subst


def build_reduce_input_first_N_unknowns_as_parameters(robot: robots.RobotDescription) -> \
        Optional[Tuple[List[ReduceInput], List[dh_utils.RevoluteVariable], sp.Matrix]]:
    n_unknown_as_parameter_from_start = first_N_unkonwns_are_parameters(robot)
    if n_unknown_as_parameter_from_start is None:
        return None

    dh_params = dh_utils.modified_dh_to_classic(robot.dh_params)
    revolute_vars = dh_utils.RevoluteVariable.convert_from_robot_unknowns(robot)
    if revolute_vars is None or dh_params is None:
        return None

    # The beginning
    A_first = sp.eye(4)
    for i in range(n_unknown_as_parameter_from_start):
        A_i_v, A_i_s = dh_utils.reflected_variable_and_structure_transform(dh_params[i])
        A_i = A_i_v * A_i_s
        A_first = A_first * A_i
    inv_A_first = symbolic_utils.inverse_transform(A_first)
    rhs_target_subst = inv_A_first * ik_target_4x4()

    equation_out = generate_reduce_equation(dh_params, revolute_vars, offset=n_unknown_as_parameter_from_start)
    if equation_out is None:
        return None

    # Finished
    reduce_input_list, revolute_vars_to_solve = equation_out
    return reduce_input_list, revolute_vars_to_solve, rhs_target_subst


def is_solvable(robot: robots.RobotDescription, logging: bool = False) -> bool:
    """
    Determine is the robot solvable by the general solver.
    """
    # 6-dof or special type of more dofs
    dof_ok = False
    if last_N_unknowns_are_parameters(robot) is not None:
        dof_ok = True
    if first_N_unkonwns_are_parameters(robot) is not None:
        dof_ok = True

    if not dof_ok:
        if logging:
            print('Cannot handle the current type of robot robots')
        return False

    # All revolute
    revolute_vars = dh_utils.RevoluteVariable.convert_from_robot_unknowns(robot)
    if revolute_vars is None:
        if logging:
            print('The general 6-dof solver can only handle revolute joints')
        return False

    # Convert to classic dh
    dh_params = dh_utils.modified_dh_to_classic(robot.dh_params)
    if dh_params is None:
        if logging:
            print('Cannot convert from modified dh to classic dh, maybe because dh_0.alpha/a is not zero. '
                  'In that case, please set them as zero and move the term to pre-transform.')
        return False

    # Initial test passed
    return True
