import sympy as sp
from typing import List, Optional, Dict, Tuple
import attr
import copy
import multiprocessing
from solver.equation_types import ScalarEquation, ScalarEquationType
from solver.equation_utils import cast_expr_to_float


@attr.s
class SumOfAngleAccumulator(object):
    new_soa_var: List[sp.Symbol] = attr.ib()
    soa_equation: List[ScalarEquation] = attr.ib()
    soa_expansion_map: Dict[sp.Symbol, sp.Expr] = attr.ib()
    soa_substitute_map: Dict[sp.Expr, sp.Symbol] = attr.ib()


def get_soa_variable_name(variables_to_sum: List[str]) -> str:
    """
    The very simple sum case: a + b + c.
    This name should be unique as a key.
    """
    # Collect the summed variables that it contains
    summed_var_names: List[str] = list()
    for v in variables_to_sum:
        summed_var_names.append(v)
    summed_var_names.sort()

    # Make a new variables
    new_soa_var_name = ''
    for name in summed_var_names:
        new_soa_var_name += name
    new_soa_var_name += '_soa'
    return new_soa_var_name


def soa_detect_no_minus(
        nonzero_summed_terms: List[sp.Expr],
        unknowns: List[sp.Symbol]) -> Optional[Tuple[sp.Symbol, sp.Expr]]:
    """
    A list of expr is summed together.
    Detect whether they are sum of plain symbols (without any other exprs such as minus), and at least two
    unknowns are summed together.
    :return: None if we cannot find it; else the first element is new_soa_variable, the second is new_soa_expr
    """
    # Check size: no more than 3
    if len(nonzero_summed_terms) >= 4 or len(nonzero_summed_terms) <= 1:
        return None

    # Count the # of variables
    n_variables = 0
    symbols_in_d: List[sp.Expr] = list()
    is_sum_of_symbol = True
    sum_expr: sp.Expr = sp.S.Zero
    for expr_k in nonzero_summed_terms:
        sum_expr = sum_expr + expr_k
        if expr_k.is_Symbol:
            if expr_k in unknowns:
                n_variables += 1
            symbols_in_d.append(expr_k)
        else:
            is_sum_of_symbol = False
            break

    # This is a plain sum
    if (n_variables == 2 or n_variables == 3) and is_sum_of_symbol:
        # Get the new variable
        variable_names_to_sum = [elem.name for elem in symbols_in_d]
        new_soa_var_name = get_soa_variable_name(variable_names_to_sum)
        new_soa_var = sp.Symbol(new_soa_var_name)
        new_soa_expr = sum_expr
        return new_soa_var, new_soa_expr

    # Find nothing
    return None


def soa_detect_minus(
        nonzero_summed_terms: List[sp.Expr],
        unknowns: List[sp.Symbol]) -> Optional[Tuple[sp.Symbol, sp.Expr]]:
    """
    a - b situation. Must be exactly two variable, one minus another
    :return: None if we cannot find it; else the first element is new_soa_variable, the second is new_soa_expr
    """
    # Check size: no more than 3
    if len(nonzero_summed_terms) != 2:
        return None

    found_positive = False
    found_negative = False
    n_variables = 0
    sign_variable_tuple_list: List[Tuple[int, sp.Symbol]] = list()
    sum_expr: sp.Expr = sp.S.Zero
    for expr_k in nonzero_summed_terms:
        # Summing does not depend on term type
        sum_expr = sum_expr + expr_k

        # If this is a positive term candidate
        if expr_k.is_Symbol:
            found_positive = True
            symbol_k: sp.Symbol = expr_k
            sign_variable_tuple_list.append((1, symbol_k))
            if symbol_k in unknowns:
                n_variables += 1
        elif expr_k.is_Mul:  # negative term candidate
            multiple_args = expr_k.args
            if len(multiple_args) != 2:
                continue

            # Should be 2 multiple
            coefficient = cast_expr_to_float(multiple_args[0])
            if coefficient is None or (abs(coefficient + 1.0) > 1e-10):
                continue

            # coefficient is -1
            variable_candidate: sp.Expr = multiple_args[1]
            if variable_candidate.is_Symbol:
                found_negative = True
                symbol_k: sp.Symbol = variable_candidate
                sign_variable_tuple_list.append((-1, symbol_k))
                if symbol_k in unknowns:
                    n_variables += 1

    # Check if 2 variables with 1 positive and 1 negative
    if n_variables == 2 and found_positive and found_negative and len(sign_variable_tuple_list) == 2:
        signed_variable_name_list = list()
        for sign_and_var in sign_variable_tuple_list:
            var_sign, var_symbol = sign_and_var
            if var_sign > 0:
                signed_variable_name_list.append('positive_' + var_symbol.name + '_')
            else:
                signed_variable_name_list.append('negative_' + var_symbol.name + '_')

        # Get the new variable
        new_soa_var_name = get_soa_variable_name(signed_variable_name_list)
        new_soa_var = sp.Symbol(new_soa_var_name)
        new_soa_expr = sum_expr
        return new_soa_var, new_soa_expr

    # Find nothing
    return None


def sum_of_angle_substitute(expr_to_substitute: sp.Expr, unknowns: List[sp.Symbol],
                            soa_accumulator: List[Tuple[sp.Symbol, sp.Expr]],
                            lock: Optional[multiprocessing.Lock] = None) -> sp.Expr:
    """
    Find the pattern in the form of sin(theta_1 + theta_2), where theta_* are
    the unknowns to solve. If found, created new sum-of-angle variables and
    treat it as the new unknown.
    A lock is required when inserting into the accumulator.
    :param expr_to_substitute:
    :param unknowns:
    :param soa_accumulator:
    :param lock:
    :return:
    """
    # Do not touch the original expr
    substituted_expr = copy.deepcopy(expr_to_substitute)

    # Find all sin/cos of angle sums up to three angles
    aw = sp.Wild('aw')
    bw = sp.Wild('bw')
    cw = sp.Wild('cw')
    matches = expr_to_substitute.find(sp.sin(aw)) | expr_to_substitute.find(sp.cos(aw))

    # For each sin/cos terms
    for m in matches:
        # m = cos(theta_1 + theta_2)
        # d_cos = {aw: theta_1, bw: theta_2, cw: 0}
        d_cos: Dict[sp.Wild, sp.Symbol] = m.match(sp.cos(aw + bw + cw))
        d_sin: Dict[sp.Wild, sp.Symbol] = m.match(sp.sin(aw + bw + cw))
        d_map: Dict[sp.Wild, sp.Symbol] = d_cos
        if (d_map is not None) and (d_sin is not None):
            d_map.update(d_sin)
        if (d_map is None) and (d_sin is not None):
            d_map = d_sin

        # Nothing matched, just continue
        if d_map is None:
            continue

        # Get the non-zero terms in match
        assert d_map is not None and len(d_map) == 3
        nonzero_matched_term_array: List[sp.Expr] = list()
        for key in d_map.keys():
            value = d_map[key]
            if value == sp.S.Zero:
                continue
            else:
                nonzero_matched_term_array.append(value)

        # Find sum of angle
        new_soa_var: Optional[sp.Symbol] = None
        new_soa_expr: Optional[sp.Expr] = None

        # Detect plain sum of symbols
        no_minus_detect = soa_detect_no_minus(nonzero_matched_term_array, unknowns)
        if no_minus_detect is not None:
            new_soa_var, new_soa_expr = no_minus_detect
        else:  # no_minus_detect is None
            minus_detect = soa_detect_minus(nonzero_matched_term_array, unknowns)
            if minus_detect is not None:
                new_soa_var, new_soa_expr = minus_detect

        # Check if this new variable has been declared before
        # This need the lock
        if (new_soa_var is not None) and (new_soa_expr is not None):
            variable_exist = False
            if lock is not None:
                lock.acquire()
            for elem in soa_accumulator:
                var, _ = elem
                if var.name == new_soa_var.name:
                    variable_exist = True
                    break
            if lock is not None:
                lock.release()

            # Check the unknowns
            for var in unknowns:
                if var.name == new_soa_var.name:
                    variable_exist = True
                    break

            # Add to the accumulator
            if not variable_exist:
                if lock is not None:
                    lock.acquire()
                soa_accumulator.append((new_soa_var, new_soa_expr))
                print('sum_of_angles_sub: created new variable: ', new_soa_var)
                print('Create equation: ', new_soa_expr)
                if lock is not None:
                    lock.release()

            # Do substitute,
            # Use d_map[aw] + d_map[bw] + d_map[cw] instead of new_soa_expr, as new_soa_expr can be
            # in different form (although the semantic is exactly the same)
            substituted_expr = substituted_expr.subs(d_map[aw] + d_map[bw] + d_map[cw], new_soa_var)

    # OK
    substituted_expr = sp.expand_trig(substituted_expr)
    return substituted_expr


def soa_parallel_processor(equations: List[ScalarEquation],
                           unknowns: List[sp.Symbol], lock: multiprocessing.Lock,
                           result_list: List[Tuple[sp.Symbol, sp.Expr]]):
    for i in range(len(equations)):
        equ_i_raw = equations[i]
        # simplify should catch c1s2+s1c2
        lhs = sp.trigsimp(equ_i_raw.lhs)
        rhs = sp.trigsimp(equ_i_raw.rhs)
        lhs = sum_of_angle_substitute(lhs, unknowns, result_list, lock)
        rhs = sum_of_angle_substitute(rhs, unknowns, result_list, lock)
        # assign back
        new_equ_i = ScalarEquation(lhs, rhs, equation_type=equ_i_raw.equation_type)
        equations[i] = new_equ_i


def sum_of_angle_transform_parallel(equations: List[ScalarEquation],
                                    unknowns: List[sp.Symbol]) -> SumOfAngleAccumulator:
    # Make the lock and result
    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    result_list = manager.list()
    n_processor = min(32, len(equations))
    n_processor = max(n_processor, 1)
    n_elem_per_processor = int(len(equations) / n_processor)
    print('Try sum-of-angle transform. Use n processors: ', n_processor)

    # Make the input
    input_equations: List[List[ScalarEquation]] = list()
    processor_list = list()
    offset = 0
    for i in range(n_processor):
        # Collect the equations
        processor_i_counter = 0
        processor_i_equations = manager.list()
        while processor_i_counter < n_elem_per_processor and offset < len(equations):
            processor_i_equations.append(equations[offset])
            offset += 1
            processor_i_counter += 1

        # Make all remaining elements to the last processor
        if i == n_processor - 1:
            while offset < len(equations):
                processor_i_equations.append(equations[offset])
                offset += 1

        # Make the processor
        input_equations.append(processor_i_equations)
        processor = multiprocessing.Process(target=soa_parallel_processor,
                                            args=(input_equations[i], unknowns, lock, result_list))
        processor_list.append(processor)

    # Run the processor
    for p in processor_list:
        p.start()
    for p in processor_list:
        p.join()

    # Collect the result
    accumulator = SumOfAngleAccumulator([], [], dict(), dict())
    for elem in result_list:
        new_soa_var, expr_rhs = elem
        accumulator.new_soa_var.append(new_soa_var)
        accumulator.soa_equation.append(ScalarEquation(new_soa_var, expr_rhs, ScalarEquationType.SumOfAngle.name))
        accumulator.soa_expansion_map[new_soa_var] = expr_rhs
        accumulator.soa_substitute_map[expr_rhs] = new_soa_var

    offset = 0
    for i in range(n_processor):
        processor_i_equs = input_equations[i]
        for j in range(len(processor_i_equs)):
            equations[offset] = processor_i_equs[j]
            offset += 1
    assert offset == len(equations)
    return accumulator


def sum_of_angle_transform(equations: List[ScalarEquation], unknowns: List[sp.Symbol]) -> SumOfAngleAccumulator:
    """
    Do transform of equations in place, and append the result to accumulator
    :param equations:
    :param unknowns:
    :return:
    """
    accumulator = SumOfAngleAccumulator([], [], dict(), dict())
    result_list = list()
    for i in range(len(equations)):
        equ_i_raw = equations[i]
        # simplify should catch c1s2+s1c2
        lhs = sp.simplify(equ_i_raw.lhs)
        rhs = sp.simplify(equ_i_raw.rhs)
        lhs = sum_of_angle_substitute(lhs, unknowns, result_list)
        rhs = sum_of_angle_substitute(rhs, unknowns, result_list)
        # assign back
        new_equ_i = ScalarEquation(lhs, rhs)
        equations[i] = new_equ_i

    # Put back to accumulator
    for elem in result_list:
        new_soa_var, expr_rhs = elem
        accumulator.new_soa_var.append(new_soa_var)
        accumulator.soa_equation.append(ScalarEquation(new_soa_var, expr_rhs, ScalarEquationType.SumOfAngle.name))
        accumulator.soa_expansion_map[new_soa_var] = expr_rhs
        accumulator.soa_substitute_map[expr_rhs] = new_soa_var
    return accumulator


# Code for testing
def test_soa():
    th_0, th_1 = sp.symbols('th_0, th_1')
    accumulator = list()
    expr_in = th_0 + th_1
    expr_out = sum_of_angle_substitute(expr_in, [th_0, th_1], accumulator)
    print('Substitute result ', expr_out)
    print(accumulator)
    output = soa_detect_minus([th_0, -th_1], [th_0, th_1])
    print(output)


if __name__ == '__main__':
    test_soa()
