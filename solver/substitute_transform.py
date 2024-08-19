from solver.equation_types import ScalarEquation, ScalarEquationType
from solver.equation_utils import count_unknowns
from typing import List, Dict, Optional
import random
import sympy as sp
from sympy import S
import multiprocessing


def substitute_processor(dict_in: Dict) -> Optional[ScalarEquation]:
    """
    The processor for substitute transform
    :param dict_in: the input dictionary with keys 'equ_i', 'equ_j' and 'unknowns'
    :return: If a valid substitute transform is found, return it. Else return None
    """
    unknowns: List[sp.Symbol] = dict_in['unknowns']
    equ_i: ScalarEquation = dict_in['equ_i']
    equ_j: ScalarEquation = dict_in['equ_j']
    rhs_i = equ_i.rhs
    rhs_j = equ_j.rhs

    # substituted type
    substituted_type = ScalarEquationType.Default.name
    if equ_i.is_sum_of_angle and equ_j.is_sum_of_angle:
        substituted_type = ScalarEquationType.SumOfAngle.name

    # Do not need further processing
    n_unknowns_old = count_unknowns(equ_i.lhs, equ_i.rhs, unknowns)
    if n_unknowns_old == 1:
        return None

    if rhs_i != rhs_j and rhs_i != S.Zero and rhs_i.has(rhs_j):
        new_rhs_i = rhs_i.subs(rhs_j, equ_j.lhs)
        n_unknowns_new = count_unknowns(equ_i.lhs, new_rhs_i, unknowns)
        if n_unknowns_old > n_unknowns_new >= 1 and n_unknowns_new <= 2 and sp.simplify(
                equ_i.lhs - new_rhs_i, doit=False) != S.Zero:
            new_equ = ScalarEquation(equ_i.lhs, new_rhs_i, substituted_type)
            return new_equ

    if rhs_i != rhs_j and rhs_i != S.Zero and rhs_i.has(-rhs_j):
        new_rhs_i = rhs_i.subs(-rhs_j, -equ_j.lhs)
        n_unknowns_new = count_unknowns(equ_i.lhs, new_rhs_i, unknowns)
        if n_unknowns_old > n_unknowns_new >= 1 and n_unknowns_new <= 2 and sp.simplify(
                equ_i.lhs - new_rhs_i, doit=False) != S.Zero:
            new_equ = ScalarEquation(equ_i.lhs, new_rhs_i, substituted_type)
            return new_equ

    return None


def substitute_transform_parallel(scalar_exprs: List[ScalarEquation],
                                  unknowns: List[sp.Symbol],
                                  max_candidate_pair: int = 30000) -> List[ScalarEquation]:
    # Need to skip some of them
    n_equation_pairs = len(scalar_exprs) * (len(scalar_exprs) - 1)
    if n_equation_pairs <= 2:
        return list()
    n_max_pair = max_candidate_pair
    accept_ratio = float(n_max_pair) / float(n_equation_pairs)
    accept_ratio = min(1.0, accept_ratio)

    # Make the input
    input_args = list()
    for i in range(len(scalar_exprs)):
        equ_i = scalar_exprs[i]
        for j in range(len(scalar_exprs)):
            # Should we skip this pair
            skip_ij = False
            if i == j:
                skip_ij = True
            if random.uniform(0.0, 1.0) > accept_ratio:
                skip_ij = True
            if scalar_exprs[i].equation_type == ScalarEquationType.SumOfAngle.name \
                    or scalar_exprs[j].equation_type == ScalarEquationType.SumOfAngle.name:
                skip_ij = False

            # If yes, then ignore this pair
            if skip_ij:
                continue
            equ_j = scalar_exprs[j]
            dict_in = dict()
            dict_in['equ_i'] = equ_i
            dict_in['equ_j'] = equ_j
            dict_in['unknowns'] = unknowns
            dict_in['i'] = i
            dict_in['j'] = j
            input_args.append(dict_in)

    # OK, do the mapping
    n_process = min(32, len(input_args))
    n_process = max(n_process, 1)
    print('Try substitute transform. The candidate number: ', len(input_args))
    output: List[Optional[ScalarEquation]] = list()
    with multiprocessing.Pool(n_process) as pool:
        output = pool.map(substitute_processor, input_args)

    # Collect the result
    transformed_expressions = list()
    transformed_expr_idx = set()
    for k in range(len(input_args)):
        arg_k = input_args[k]
        output_k = output[k]
        if output_k is None:
            continue
        equ_idx = arg_k['i']
        transformed_expressions.append(output_k)
        transformed_expr_idx.add(equ_idx)

    # Log
    if len(transformed_expr_idx) > 0:
        print('Found substitute transform in expressions ', transformed_expr_idx)

    # Return the new equations
    return transformed_expressions


def substitute_transform_inplace(scalar_exprs: List[ScalarEquation], unknowns: List[sp.Symbol]):
    transformed_expr_idx = set()
    for i in range(len(scalar_exprs)):
        equ_i = scalar_exprs[i]
        for j in range(len(scalar_exprs)):
            if i == j:
                continue
            equ_j = scalar_exprs[j]
            dict_in = dict()
            dict_in['equ_i'] = equ_i
            dict_in['equ_j'] = equ_j
            dict_in['unknowns'] = unknowns
            new_equ = substitute_processor(dict_in)
            if new_equ is None:
                continue

            # Update scalar expr i
            scalar_exprs[i] = new_equ
            transformed_expr_idx.add(i)
            break

    if len(transformed_expr_idx) > 0:
        print('Found substitute transform in exprs')
        print(transformed_expr_idx)


# Code for testing
def test_substitute():
    from fk.robots import puma_robot, RobotDescription
    import fk.fk_equations as fk_equations
    import fk.kinematics_dh as kinematics_dh
    import solver.equation_utils as equation_utils
    from solver.soa_transform import sum_of_angle_transform_parallel

    # Use puma to debug
    robot = puma_robot()
    thetas = robot.unknowns
    unknowns = [elem.symbol for elem in thetas]
    fk_out = kinematics_dh.forward_kinematics_dh(robot.dh_params, unknowns)
    # Substitute the FK result, python pass by ref
    # fk_equations.fk_substitute_value(fk_out, robot)

    raw_fk_equs = fk_equations.build_fk_matrix_equations(fk_out)
    scalar_equs = equation_utils.collect_scalar_equations(raw_fk_equs)
    accumulator = sum_of_angle_transform_parallel(scalar_equs, unknowns)

    # Add to new unknowns
    unknowns.extend(accumulator.new_soa_var)
    scalar_equs.extend(accumulator.soa_equation)
    substitute_transform_inplace(scalar_equs, unknowns)
    print(accumulator.new_soa_var)


if __name__ == '__main__':
    test_substitute()
