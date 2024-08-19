import sympy as sp
import copy
import random
import multiprocessing
from typing import List, Optional, Dict
from fk.fk_equations import expr_include_translation_target
from solver.equation_types import MatrixEquation, ScalarEquation, SumOfSquareHint, ScalarEquationType
from solver.equation_utils import count_unknowns_expr, count_unknowns, append_equation_no_duplicate
from solver.equation_utils import all_symbols_in_expr, all_symbols_in_equation
from solver.soa_transform import SumOfAngleAccumulator


def process_expr(expr_1: sp.Expr,
                 expr_2: sp.Expr,
                 expr_3: Optional[sp.Expr],
                 accumulator: Optional[SumOfAngleAccumulator]) -> sp.Expr:
    """
    Do sum-of-square for expr_1/expr_2/expr_3 and simplify the equation.
    Need to be careful with sp.simplify as it create new angular variables.
    Return the resulting sum-of-square equation.
    """
    if expr_3 is None:
        sos_expr = expr_1 * expr_1 + expr_2 * expr_2
    else:
        sos_expr = expr_1 * expr_1 + expr_2 * expr_2 + expr_3 * expr_3
    sos_expr = sos_expr.simplify()
    if accumulator is not None:
        sos_expr = sos_expr.subs(accumulator.soa_substitute_map)
    # To avoid sin(2x) or sin(x + y), which causes trouble in polynomial solve
    sos_expr = sp.expand_trig(sos_expr)
    return sos_expr


def process_expr_pair(lhs_1: sp.Expr, lhs_2: sp.Expr, lhs_3: Optional[sp.Expr],
                      rhs_1: sp.Expr, rhs_2: sp.Expr, rhs_3: Optional[sp.Expr],
                      soa_accumulator: Optional[SumOfAngleAccumulator],
                      unknowns: List[sp.Symbol]) -> Optional[ScalarEquation]:
    """
    Test whether the scalar equations defined above lead to a reduction of unknown numbers
    from >= 1 to 1. If so, return the new 1-unknown equations, else return None.
    """
    sos_expr_lhs = process_expr(lhs_1, lhs_2, lhs_3, soa_accumulator)
    if count_unknowns_expr(sos_expr_lhs, unknowns) == 0:
        # Try rhs
        sos_expr_rhs = process_expr(rhs_1, rhs_2, rhs_3, soa_accumulator)

        if count_unknowns_expr(sos_expr_rhs, unknowns) == 1:
            new_equ = ScalarEquation(sos_expr_lhs, sos_expr_rhs, ScalarEquationType.SumOfSquare.name)
            return new_equ
    elif count_unknowns_expr(sos_expr_lhs, unknowns) == 1:
        # Another case
        sos_expr_rhs = process_expr(rhs_1, rhs_2, rhs_3, soa_accumulator)
        if count_unknowns_expr(sos_expr_rhs, unknowns) == 0:
            new_equ = ScalarEquation(sos_expr_lhs, sos_expr_rhs, ScalarEquationType.SumOfSquare.name)
            return new_equ

    # Nothing for this pair
    return None


def parallel_processor_dict(dict_in: Dict):
    equ_1 = dict_in['equ_1']
    equ_2 = dict_in['equ_2']
    equ_1_lhs = equ_1.lhs
    equ_1_rhs = equ_1.rhs
    equ_2_lhs = equ_2.lhs
    equ_2_rhs = equ_2.rhs
    equ_3_lhs = None
    equ_3_rhs = None
    if 'equ_3' in dict_in:
        equ_3 = dict_in['equ_3']
        equ_3_lhs = equ_3.lhs
        equ_3_rhs = equ_3.rhs
    unknowns = dict_in['unknowns']
    if 'accumulator' in dict_in:
        accumulator = dict_in['accumulator']
    else:
        accumulator = None
    sos_equ = process_expr_pair(
        equ_1_lhs, equ_2_lhs, equ_3_lhs,
        equ_1_rhs, equ_2_rhs, equ_3_rhs,
        accumulator, unknowns)

    # Log info
    index = dict_in['index']
    if index > 0 and index % 30 == 0:
        print('Candidate finished index ', dict_in['index'], ' output is ', sos_equ)
    return sos_equ


def parallel_processor_list_output(dict_in: Dict, output_list: List[ScalarEquation], lock=None):
    this_output = parallel_processor_dict(dict_in)
    if this_output is None:
        return

    # Append a result
    if lock is not None:
        lock.acquire()
    output_list.append(this_output)
    if lock is not None:
        lock.release()


def parallel_run(input_args: List[Dict], timeout_seconds: int = 30) -> List[ScalarEquation]:
    n_processor = min(32, int(len(input_args)))
    n_processor = max(n_processor, 1)
    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    shared_result_list = manager.list()
    processed_offset = 0
    while processed_offset < len(input_args):
        processor_list = list()
        for i in range(n_processor):
            offset_i = processed_offset + i
            if offset_i < len(input_args):
                # OK, make the processor
                processor = multiprocessing.Process(
                    target=parallel_processor_list_output,
                    args=(input_args[offset_i], shared_result_list, lock))
                processor_list.append(processor)

        # Start the processor with time-out
        for p in processor_list:
            p.start()
        for p in processor_list:
            p.join(timeout=timeout_seconds)
            p.terminate()  # Explicit kill a process if the reduction takes too long

        # Update the offset
        processed_offset += n_processor

    # To usual list
    output_list = list()
    for elem in shared_result_list:
        output_list.append(copy.deepcopy(elem))
    return output_list


def sum_of_square_transform_pairwise(equations: List[ScalarEquation],
                                     unknowns: List[sp.Symbol],
                                     accumulator: Optional[SumOfAngleAccumulator],
                                     only_include_position_expr: bool = True,
                                     max_ops_in_candidate: int = 50,
                                     max_pairwise_sos_candidates: int = 1000) -> List[ScalarEquation]:
    """
    Search for a pair of equations (expr_1, expr_2) whose sum-of-square is simpler
    than those (expr_1, expr_2), which can be solved directly.
    For efficiency, this method parallelize on process.
    """
    result: List[ScalarEquation] = list()
    equation_to_test = list()
    for expr in equations:
        if expr.is_sum_of_square or expr.is_sum_of_angle:
            continue
        if count_unknowns(expr.lhs, expr.rhs, unknowns) != 2:
            continue
        if expr.lhs.count_ops() + expr.rhs.count_ops() > max_ops_in_candidate:
            continue
        if not only_include_position_expr:
            equation_to_test.append(expr)
            continue
        # Need to check positions
        if expr_include_translation_target(expr.lhs):
            equation_to_test.append(expr)
            continue
        if expr_include_translation_target(expr.rhs):
            equation_to_test.append(expr)
            continue

    # No equation
    if len(equation_to_test) <= 1:
        return list()

    # Collect the args
    parallel_args = list()
    for i in range(len(equation_to_test)):
        for j in range(i + 1, len(equation_to_test)):
            input_dict = dict()
            input_dict['index'] = len(parallel_args)
            input_dict['equ_1'] = equation_to_test[i]
            input_dict['equ_2'] = equation_to_test[j]
            input_dict['unknowns'] = unknowns
            input_dict['accumulator'] = accumulator
            parallel_args.append(input_dict)

    # Try parallel processing
    if len(parallel_args) > max_pairwise_sos_candidates:
        sampled_parallel_args = random.sample(parallel_args, max_pairwise_sos_candidates)
        parallel_args = sampled_parallel_args
        for i in range(len(parallel_args)):
            parallel_args[i]['index'] = i
    print('Try pairwise sum-of-square transform. The candidate number: ', len(parallel_args))

    # Run processor
    sos_output = parallel_run(parallel_args)

    # Collect the result
    for elem in sos_output:
        if elem is None:
            continue
        appended = append_equation_no_duplicate(result, elem)
        if appended:
            print('Found pairwise sum-of-square equation')
            print(elem)

    print('After pairwise sos, the number of found expr is ', len(result))
    return result


def collect_sos_hint_translation(matrix_equations: List[MatrixEquation]) -> List[SumOfSquareHint]:
    """
    The translational expression might be a good sum-of-square hint
    :param matrix_equations:
    :return:
    """
    sos_hint: List[SumOfSquareHint] = list()
    for i in range(len(matrix_equations)):
        mat_equ_i = matrix_equations[i]
        mat_rhs = mat_equ_i.Ts
        mat_lhs = mat_equ_i.Td
        equ_1 = ScalarEquation(mat_lhs[0, 3], mat_rhs[0, 3])
        equ_2 = ScalarEquation(mat_lhs[1, 3], mat_rhs[1, 3])
        equ_3 = ScalarEquation(mat_lhs[2, 3], mat_rhs[2, 3])
        hint = SumOfSquareHint(equ_1, equ_2, equ_3)
        sos_hint.append(hint)
    return sos_hint


def sum_of_square_transform_translation(sos_hints: List[SumOfSquareHint],
                                        soa_accumulator: SumOfAngleAccumulator,
                                        unknowns: List[sp.Symbol]) -> List[ScalarEquation]:
    """
    Perform sum-of-square for three equations who are the xyz component of a matrix equation.
    """
    print('Try translational sum-of-square transform. The candidate number: ', len(sos_hints))

    # Collect the args
    parallel_args = list()
    for i in range(len(sos_hints)):
        input_dict = dict()
        hint_i = sos_hints[i]
        input_dict['index'] = len(parallel_args)
        input_dict['equ_1'] = hint_i.equ_1
        input_dict['equ_2'] = hint_i.equ_2
        if hint_i.equ_3 is not None:
            input_dict['equ_3'] = hint_i.equ_3
        input_dict['unknowns'] = unknowns
        input_dict['accumulator'] = soa_accumulator
        parallel_args.append(input_dict)

    # Run processor
    sos_output = parallel_run(parallel_args)

    # Collect the result
    result: List[ScalarEquation] = list()
    for elem in sos_output:
        if elem is None:
            continue
        else:
            if elem not in result:
                print('Found translational sum-of-square equation')
                print(elem)
                result.append(elem)

    print('After translational sos, the number of found expr is ', len(result))
    return result


# Test code
def test_sos_transform():
    from fk.fk_equations import Px, Py, Pz
    equs = list()
    th_1, th_2, th_23 = sp.symbols('th_1 th_2 th_23')
    a_2, a_3, d_4 = sp.symbols('a_2 a_3 d_4')
    lhs_1 = sp.cos(th_1) * Px + sp.sin(th_1) * Py
    rhs_1 = a_3 * sp.cos(th_23) - d_4 * sp.sin(th_23) + a_2 * sp.cos(th_2)
    equs.append(ScalarEquation(lhs_1, rhs_1))
    all_symbols = all_symbols_in_expr(lhs_1)
    print(all_symbols)
    all_symbols = all_symbols_in_equation(lhs_1, rhs_1)
    print(all_symbols)

    lhs_2 = - Pz
    rhs_2 = a_3 * sp.sin(th_23) + d_4 * sp.cos(th_23) + a_2 * sp.sin(th_2)
    equs.append(ScalarEquation(lhs_2, rhs_2))

    # OK
    unknowns = [th_1, th_2, th_23]
    out = sum_of_square_transform_pairwise(equs, unknowns, None, False)
    print(out)


if __name__ == '__main__':
    test_sos_transform()
