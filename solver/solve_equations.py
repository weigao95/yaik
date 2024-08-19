import solver.equation_utils as equation_utils
from solver.equation_types import Unknown, UnknownType, EquationInput, NumericalAnalyseContext, CollectedEquations
import solver.soa_transform as soa_transform
import solver.substitute_transform as substitute_transform
from solver.unary_solvers.unary_solver_step import UnarySolverStep
from solver.linear_solver.linear_sin_cos_type2 import linear_sin_cos_type2_try_solve
from solver.polynomial_solver.polynomial_solve import polynomial_try_solve
import solver.sos_transform as sos_transform
from solver.solved_variable import VariableSolution, DegenerateType, NumericalAnalysedResultType
from solver.select_solution import select_solution, DegenerateAnalyserContext, DegenerateAnalyserOption
from solver.solver_snapshot import SolverSnapshot
from solver.degenerate_analyse.detect_soa_solution import detect_soa_solution
from typing import List, Dict, Set, Tuple, Optional
import attr
import sympy as sp
import copy


@attr.s
class SolveEquationOption(object):
    # We are solving the main-branch or degenerate branch
    on_main_branch: bool = True

    # Used in solving subset equation that we might stop the solving if the solution is not "good".
    # The "good-ness" is determined by the method below
    stop_on_bad_solution: bool = False

    # If we cannot find the solution, we can either directly return or increase the timeout and try
    # again. If this option is True, we would directly return
    stop_on_first_unsolved: bool = False

    # Skip substitute transform if sufficient one-unknowns
    sufficient_one_unknown_equations_to_skip_substitute: int = 20
    substitute_max_candidate_pair: int = 30000

    # For polynomial
    disable_polynomial_solution: bool = False
    initial_polynomial_timeout: int = 60

    # For sos transform
    pairwise_sos_max_candidate_ops: int = 100
    max_pairwise_sos_candidates: int = 1000

    @staticmethod
    def is_good_solution(solution: VariableSolution) -> bool:
        """
        Check whether a solution should be considered as good in SolveEquationOption.stop_on_bad_solution
        Only 3 cases:
        1. AlwaysNonDegenerate
        2. DegenerateOnVariableValue
        3. NumericalAlwaysNonDegenerate
        """
        if not solution.is_explicit_solution:
            return False
        degenerate_record = solution.degenerate_record
        if degenerate_record.type == DegenerateType.AlwaysNonDegenerate.name \
                or degenerate_record.type == DegenerateType.DegenerateOnVariableValue.name:
            return True
        if degenerate_record.numerical_analysed_result == NumericalAnalysedResultType.NumericalAlwaysNonDegenerate.name:
            return True

        # Nothing else
        return False

    @staticmethod
    def is_bad_solution(solution: VariableSolution) -> bool:
        return not SolveEquationOption.is_good_solution(solution)


def solve_equations(equation_input: EquationInput,
                    unknown_variables: List[Unknown],
                    all_parameters: Set[sp.Symbol],
                    numerical_context: NumericalAnalyseContext,
                    option=SolveEquationOption()) -> List[VariableSolution]:
    """
    Solve the given polynomial equations for the unknown_variables. Everything appears in the equation_input
    must be contained in all_parameters, while the parameter_value is not necessarily complete.
    :param equation_input:
    :param unknown_variables:
    :param all_parameters:
    :param numerical_context:
    :param option:
    :return:
    """
    # Make copy to avoid mutate the original element
    scalar_equations = copy.deepcopy(equation_input.scalar_equations)
    current_unknowns = copy.deepcopy(unknown_variables)
    unknown_symbols = [elem.symbol for elem in current_unknowns]

    # Perform soa transform, only once
    if len(current_unknowns) >= 2:
        accumulator = soa_transform.sum_of_angle_transform_parallel(scalar_equations, unknown_symbols)
        scalar_equations.extend(accumulator.soa_equation)
        for new_var in accumulator.new_soa_var:
            current_unknowns.append(Unknown(new_var, UnknownType.Revolute.name))
            unknown_symbols.append(new_var)
    else:
        # Nothing would happen in soa, thus just return the empty record
        accumulator = soa_transform.SumOfAngleAccumulator([], [], dict(), dict())

    # Ok, run the solver loop
    count = 0
    unary_solver_step = UnarySolverStep.make_default()
    solved_variables: List[VariableSolution] = list()
    polynomial_timeout_second = option.initial_polynomial_timeout
    max_polynomial_timeout = option.initial_polynomial_timeout * 64
    max_iterations = len(unknown_symbols) + 5

    # Local functor
    def select_loop_solution(
            loop_equations: CollectedEquations,
            loop_solutions: List[VariableSolution],
            loop_solved_variable: List[VariableSolution],
            loop_remaining_unknowns: List[Unknown],
            simple_solution_only: bool,
            use_numerical_test: bool) -> Optional[VariableSolution]:
        # Select which solution to go
        test_cases = list()
        if use_numerical_test:
            test_cases = numerical_context.numerical_test_cases

        # Make the context
        all_unknown_symbols = [elem.symbol for elem in unknown_variables]
        degenerate_context = DegenerateAnalyserContext()
        degenerate_context.solved_variables = loop_solved_variable
        degenerate_context.all_unknown_variables = all_unknown_symbols
        degenerate_context.all_parameters = all_parameters
        degenerate_context.parameter_bounds = numerical_context.parameter_bounds
        degenerate_context.parameter_value = numerical_context.parameter_values
        degenerate_context.numerical_test_cases = test_cases

        # Make the option
        degenerate_option = DegenerateAnalyserOption()
        degenerate_option.main_branch_solution = option.on_main_branch

        # Invoke the selection
        selected_sol = select_solution(
            loop_solutions,
            degenerate_context=degenerate_context,
            degenerate_option=degenerate_option)

        # Detect soa solution
        if selected_sol is None and (not simple_solution_only) and (not option.on_main_branch):
            selected_sol = detect_soa_solution(
                loop_equations, loop_remaining_unknowns, accumulator,
                context=degenerate_context, option=degenerate_option)

        # Nothing selected, just return
        if selected_sol is None:
            return None

        # If we request simple solution
        if simple_solution_only:
            # Only explicit
            if not selected_sol.is_explicit_solution:
                return None

            # Must be always non-degenerate
            degenerate_record = selected_sol.degenerate_record
            if degenerate_record.type != DegenerateType.AlwaysNonDegenerate.name:
                return None

            # The expressions cannot be too complex
            max_ops_in_simple_solution = 20
            for solution_i in selected_sol.explicit_solutions:
                if solution_i.count_ops() > max_ops_in_simple_solution:
                    return None
        # Ok
        return selected_sol

    while len(unknown_symbols) > 0 and count < max_iterations:
        # Try with simple solve
        collected_equations = equation_utils.scalar_equation_by_unknowns(scalar_equations, unknown_symbols)
        solutions: List[VariableSolution] = list()
        use_complex_processing = True
        if len(collected_equations.one_unknown_equations) > 0:
            print('Start the rule solver directly')
            simple_solutions = unary_solver_step.solve_step_parallel(collected_equations, unknown_symbols, with_timeout=True)
            solutions.extend(simple_solutions)
            select_simple_solution = select_loop_solution(
                collected_equations,
                simple_solutions, solved_variables,
                loop_remaining_unknowns=current_unknowns,
                simple_solution_only=True,
                use_numerical_test=False)
            if select_simple_solution is not None:
                print('Find a good solution in direct solve, thus skip SOS and substitute transform')
                use_complex_processing = False

        # We don't have good solutions with simple solve
        if use_complex_processing:
            # Logging
            print('Cannot find a good solution in direct (simple) solve. We need more processing')

            # Run pairwise sos
            pairwise_sos_expr = sos_transform.sum_of_square_transform_pairwise(
                scalar_equations,
                unknown_symbols,
                accumulator,
                only_include_position_expr=True,
                max_ops_in_candidate=option.pairwise_sos_max_candidate_ops,
                max_pairwise_sos_candidates=option.max_pairwise_sos_candidates)
            scalar_equations.extend(pairwise_sos_expr)

            # Run translational sos
            translation_sos_expr = sos_transform.sum_of_square_transform_translation(
                equation_input.sum_of_square_hint,
                accumulator,
                unknown_symbols
            )
            scalar_equations.extend(translation_sos_expr)

            # Try substitute transform
            sufficient_one_unknown_equations = option.sufficient_one_unknown_equations_to_skip_substitute
            collected_equations = equation_utils.scalar_equation_by_unknowns(scalar_equations, unknown_symbols)
            if len(collected_equations.one_unknown_equations) > sufficient_one_unknown_equations:
                # Logging and do nothing
                print('We have {n} one-unknown equations, thus skip substitute transform'.
                      format(n=len(collected_equations.one_unknown_equations)))
                pass
            else:
                # run substitute
                new_equations = substitute_transform.substitute_transform_parallel(
                    scalar_equations, unknown_symbols,
                    max_candidate_pair=option.substitute_max_candidate_pair)
                scalar_equations.extend(new_equations)
                collected_equations = equation_utils.scalar_equation_by_unknowns(scalar_equations, unknown_symbols)

            # Solve again
            print('Start the rule solver with SOS and substitute')
            # collected_equations.print_stdout()
            solutions_with_more_processing = unary_solver_step.solve_step_parallel(collected_equations, unknown_symbols, with_timeout=True)
            solutions.extend(solutions_with_more_processing)

            # Try linear type 2
            if len(solutions) == 0:
                # Log info
                print('No explicit solution, try linear solve')

                # Start it
                linear_solution = linear_sin_cos_type2_try_solve(
                    collected_equations, current_unknowns, all_parameters, numerical_context)
                if linear_solution is not None:
                    solutions.append(linear_solution)

            # Try polynomial solver if no solution find
            if (len(solutions) == 0) and (not option.stop_on_bad_solution) and (not option.disable_polynomial_solution):
                # Log info
                print('No explicit solution, try polynomial solve with timeout ', polynomial_timeout_second, 'seconds')

                # Go polynomial
                poly_solutions = polynomial_try_solve(
                    collected_equations,
                    current_unknowns,
                    all_parameters,
                    numerical_context=numerical_context,
                    timeout_seconds=polynomial_timeout_second)
                solutions.extend(poly_solutions)

        # Collect the result
        if len(solutions) == 0:
            if option.stop_on_first_unsolved:
                print('Fail to find a solution in iteration {count} '
                      'while stop_on_first_unsolved=True, just return.'.format(count=count))
                return solved_variables
            if option.disable_polynomial_solution:
                print('Fail to find a solution in iteration {count} while '
                      'disable_polynomial_solution=True, just return.'.format(count=count))
                return solved_variables

            # Continue with larger timeout
            print('Fail to find a solution in iteration {count}, try larger timeout.'.format(count=count))
            count += 1
            polynomial_timeout_second *= 4
            if polynomial_timeout_second > max_polynomial_timeout or option.stop_on_bad_solution:
                # Stop here as we cannot find more solutions
                return solved_variables
            continue
        else:
            # Select which solution to go
            sol = select_loop_solution(
                collected_equations,
                loop_solutions=solutions,
                loop_solved_variable=solved_variables,
                loop_remaining_unknowns=current_unknowns,
                simple_solution_only=False,
                use_numerical_test=True)

            # No solution selected, just return
            if sol is None or (option.stop_on_bad_solution and option.is_bad_solution(sol)):
                return solved_variables

            # We have at least one valid solutions
            solved_variables.append(sol)
            print('Variable solved ', sol.solved_variable)
            if sol.is_explicit_solution:
                print('The solution(s) is ', sol.explicit_solutions)

            # Remove the solved symbol
            assert sol.solved_variable in unknown_symbols
            unknown_symbols.remove(sol.solved_variable)
            new_unknowns = list()
            for symbol in unknown_symbols:
                unknown_for_symbol = equation_utils.find_unknown(current_unknowns, symbol.name)
                assert unknown_for_symbol is not None
                new_unknowns.append(unknown_for_symbol)
            current_unknowns = new_unknowns

            # Logging
            print('Move to next iterations with unknowns: ', unknown_symbols)
            count += 1

    # return the solutions
    return solved_variables


def solve_from_snapshot(solver_snapshot: SolverSnapshot, option=SolveEquationOption()) -> List[VariableSolution]:
    """
    Just a handy caller
    """
    equation_input = EquationInput(solver_snapshot.scalar_equations, solver_snapshot.sos_hints)
    numerical_context = NumericalAnalyseContext()
    numerical_context.parameter_values = solver_snapshot.parameter_values
    numerical_context.parameter_bounds = solver_snapshot.parameter_bounds
    numerical_context.numerical_test_cases = solver_snapshot.numerical_test_cases
    solutions = solve_equations(
        equation_input,
        solver_snapshot.unknowns,
        solver_snapshot.all_parameters,
        numerical_context=numerical_context,
        option=option
    )
    return solutions
