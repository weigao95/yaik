from solver.equation_types import Unknown, UnknownType, CollectedEquations, ScalarEquation
from solver.equation_utils import cast_expr_to_float
from solver.soa_transform import SumOfAngleAccumulator
from solver.degenerate_analyse.analyser import \
    DegenerateAnalyser, DegenerateAnalyserContext, DegenerateAnalyserOption
from solver.degenerate_analyse.numerical_analyser import NumericalCheckDegenerateAnalyzer
from solver.solved_variable import VariableSolution, DegenerateType, NumericalAnalysedResultType, SolutionMethod
from typing import List, Set, Dict, Tuple, Optional
import sympy as sp
import multiprocessing
import copy


def check_zero_processor(input_dict) -> bool:
    """
    Check the scalar equation applied to input_dict yield a almost-zero float
    """
    equation_to_check: ScalarEquation = input_dict['scalar_equation']
    test_map = input_dict['test_map']
    lhs_value = cast_expr_to_float(equation_to_check.lhs.subs(test_map))
    rhs_value = cast_expr_to_float(equation_to_check.rhs.subs(test_map))
    if lhs_value is None or rhs_value is None:
        return False
    lhs_minus_rhs = lhs_value - rhs_value
    return abs(lhs_minus_rhs) < 1e-6


def detect_soa_solution(
        collected_equations: CollectedEquations,
        unknowns: List[Unknown],
        soa_accumulator: SumOfAngleAccumulator,
        context: DegenerateAnalyserContext,
        option: DegenerateAnalyserOption) -> Optional[VariableSolution]:
    """
    In degenerated branch, we might counter situations that only
    th_0 + th_1 is determined, but the value of th_0 and th_1 cannot
    be solved. This subroutine detects this situation.
    This subroutine only called when all solutions are degenerate.
    :return: None implies no such situation.
             Else the solution of one variable will be returned as zero
    """
    # Can not happen on the main branch
    if option.main_branch_solution:
        return None

    # There must be numerical test data
    if len(context.numerical_test_cases) == 0:
        print('There might be sum-of-angle solutions, but cannot be determined'
              'as I do not have numerical test data')
        return None

    # Must be two rotational unknown
    if len(unknowns) != 2:
        return None
    for unknown_i in unknowns:
        if unknown_i.unknown_type != UnknownType.Revolute.name:
            return None

    # OK, get the unknown
    unknown_0 = unknowns[0]
    unknown_1 = unknowns[1]
    print('Try detecting sum-of-angle solutions')

    # There must be a soa variable/expr contains them solved
    # This expr may be unknown_0 + unknown_1 / unknown_0 - unknown_1 / ...
    soa_variable_with_01: Optional[sp.Symbol] = None
    soa_expr_with_01: Optional[sp.Expr] = None
    for i in range(len(context.solved_variables)):
        parent_solution_i = context.solved_variables[i]
        solved_variable_i = parent_solution_i.solved_variable

        # Check whether this is a soa variable
        if solved_variable_i not in soa_accumulator.new_soa_var:
            continue

        # Now this is a soa variable
        assert solved_variable_i in soa_accumulator.soa_expansion_map
        soa_expr = soa_accumulator.soa_expansion_map[solved_variable_i]
        assert soa_expr is not None

        # Must be exactly two symbols for unknown_0/1
        soa_expr_symbols = soa_expr.free_symbols
        if len(soa_expr_symbols) == 2 \
                and unknown_0.symbol in soa_expr_symbols \
                and unknown_1.symbol in soa_expr_symbols:
            soa_expr_with_01 = soa_expr
            soa_variable_with_01 = solved_variable_i
            break

    # No soa expression found, just return
    if soa_expr_with_01 is None:
        return None

    # Perform numerical analysis
    assert soa_variable_with_01 is not None
    assert soa_expr_with_01 is not None
    numerical_test_cases: List[Dict[sp.Symbol, float]] = context.numerical_test_cases

    # Make the test map
    n_numerical_test = min(500, len(numerical_test_cases))
    updated_test_map_list: List[Dict[sp.Symbol, float]] = list()
    for i in range(n_numerical_test):
        test_map_i = copy.deepcopy(numerical_test_cases[i])
        # Add existing solution (soa variables) if not already in it
        for j in range(len(context.solved_variables)):
            solved_variable_j = context.solved_variables[j].solved_variable
            if solved_variable_j not in test_map_i:
                if solved_variable_j not in soa_accumulator.new_soa_var:
                    print('Variable {var} should be in numerical test cases.'.format(var=solved_variable_j.name))
                    return None
                soa_expr_for_j = soa_accumulator.soa_expansion_map[solved_variable_j]
                soa_expr_for_j_value = soa_expr_for_j.subs(test_map_i)
                soa_expr_for_j_value = cast_expr_to_float(soa_expr_for_j_value)
                if soa_expr_for_j_value is None:
                    return None
                else:
                    test_map_i[solved_variable_j] = soa_expr_for_j_value

        # Test the solution while setting th_0 to sp.S.Zero
        soa_variable_with_01_value: float = test_map_i[soa_variable_with_01]
        unknown_0_value = 0.0

        # Unknown 1 must be soa_01 value or - soa_01_value, just check
        expr_should_be_zero = soa_expr_with_01 - soa_variable_with_01_value
        expr_should_be_zero = expr_should_be_zero.subs({
            soa_variable_with_01: soa_variable_with_01_value,
            unknown_0.symbol: unknown_0_value
        })

        # Case 1: unknown 1 == soa_variable
        unknown_1_value: Optional[float] = None
        case_1_value = cast_expr_to_float(expr_should_be_zero.subs({unknown_1.symbol: soa_variable_with_01_value}))
        if case_1_value is not None and abs(case_1_value) < 1e-10:
            unknown_1_value = soa_variable_with_01_value

        # Case 2: unknown 1 == - soa_variable
        case_2_value = cast_expr_to_float(expr_should_be_zero.subs({unknown_1.symbol: -soa_variable_with_01_value}))
        if case_2_value is not None and abs(case_2_value) < 1e-10:
            unknown_1_value = - soa_variable_with_01_value

        # Should be one of them
        if unknown_1_value is None:
            return None

        # Check expr with this subst value
        test_map_i[unknown_0.symbol] = unknown_0_value
        test_map_i[unknown_1.symbol] = unknown_1_value
        updated_test_map_list.append(test_map_i)

    # This should be in parallel
    n_equations = len(collected_equations.other_equations) + len(collected_equations.two_unknown_equations)
    print('The number of input args is ', n_equations * len(updated_test_map_list))
    input_args = list()
    for equ_k in collected_equations.other_equations + collected_equations.two_unknown_equations:
        for test_map_i in updated_test_map_list:
            input_dict = dict()
            input_dict['scalar_equation'] = equ_k
            input_dict['test_map'] = test_map_i
            input_args.append(input_dict)

            # We don't need these serial code
            # lhs_minus_rhs = equ_k.lhs - equ_k.rhs
            # value_k = lhs_minus_rhs.subs(test_map_i)
            # value_k = cast_expr_to_float(value_k)
            # if value_k is None or abs(value_k) > 1e-6:
            #    return None

    # Run processing in parallel
    n_process = min(32, len(input_args))
    n_process = max(n_process, 1)
    with multiprocessing.Pool(n_process) as pool:
        output = pool.map(check_zero_processor, input_args)

    # Check output
    for k in range(len(output)):
        if not output[k]:
            return None

    # All test passed
    print('Find a degenerate soa solution')
    return VariableSolution.make_explicit_solution(
        solved_variable=unknown_0.symbol,
        solutions=[sp.S.Zero],
        solution_method=SolutionMethod.DefaultMethod.name,
        argument_valid_checkers=[sp.S.BooleanTrue])


# Debug code
def test_soa_solution():
    from solver.equation_types import ScalarEquation
    # Make the variable
    th_0, th_1 = sp.symbols('th_0 th_1')
    soa_var = sp.Symbol('th_0th_1_soa')
    th_0_unknown = Unknown(th_0, UnknownType.Revolute.name)
    th_1_unknown = Unknown(th_1, UnknownType.Revolute.name)

    # Make the accumulator
    accumulator = SumOfAngleAccumulator([], [], dict(), dict())
    accumulator.new_soa_var.append(soa_var)
    accumulator.soa_equation.append(ScalarEquation(soa_var, th_0 + th_1))
    accumulator.soa_substitute_map[th_0 + th_1] = soa_var
    accumulator.soa_expansion_map[soa_var] = th_0 + th_1

    # Make the equations
    collected_equation = CollectedEquations([], [ScalarEquation(soa_var, th_0 + th_1)], [])
    # collected_equation.other_equations.append(ScalarEquation(soa_var, sp.sin(th_0)))

    # The context
    context = DegenerateAnalyserContext()
    context.solved_variables = [VariableSolution.make_explicit_solution(
        soa_var, [sp.S.Zero], 'fake_method_to_test', argument_valid_checkers=[sp.S.BooleanTrue])]
    context.all_unknown_variables = [soa_var, th_0, th_1]
    context.numerical_test_cases = [{soa_var: 10.0, th_0: 1.0, th_1: 9.0}]

    # The option
    option = DegenerateAnalyserOption()
    option.main_branch_solution = False

    # Try it
    detection = detect_soa_solution(collected_equation, [th_0_unknown, th_1_unknown], accumulator, context, option)
    print(detection)


if __name__ == '__main__':
    test_soa_solution()
