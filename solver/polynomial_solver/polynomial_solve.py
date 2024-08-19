from solver.polynomial_solver.find_candidate import collect_equation_by_variables
from solver.polynomial_solver.verify_polynomial_solution import verify_polynomial_solution
from solver.equation_types import CollectedEquations, Unknown, NumericalAnalyseContext
import solver.equation_utils as equation_utils
from solver.solved_variable import VariableSolution
import multiprocessing
import sympy as sp
from typing import List, Tuple, Optional, Set

# Make sage as optional dependency
try:
    from solver.polynomial_solver.groebner_reduce_sage import groebner_reduce_sage as groebner_reduce
except ImportError:
    from solver.polynomial_solver.groebner_reduce_sympy import groebner_reduce_sympy as groebner_reduce


def solve_processor(equations: List[sp.Expr],
                    unknowns: List[Unknown],
                    all_parameters: Set[sp.Symbol],
                    output_list: List,
                    numerical_context: Optional[NumericalAnalyseContext] = None,
                    lock: Optional[multiprocessing.Lock] = None,
                    reduce_to_cos: bool = True,
                    logging: bool = False):
    # Hand in to different methods
    this_output = groebner_reduce(equations, unknowns, all_parameters, reduce_to_cos)

    # Check if we succeed
    if this_output is None:
        return

    # Do verify
    if numerical_context is not None and len(numerical_context.numerical_test_cases) > 0:
        print('Find candidate solution, now do verification')
        all_case_match, pass_ratio = verify_polynomial_solution(
            equations, this_output, numerical_context.numerical_test_cases)
        if pass_ratio < 0.7:
            print(f'Find an incorrect polynomial solution, match ratio is {pass_ratio}')
            return
        else:
            print('Verification OK')

    # Append a result
    if lock is not None:
        lock.acquire()
    output_list.append(this_output)
    if lock is not None:
        lock.release()

    # Log
    if logging:
        print('Find a value solution', [elem.symbol.name for elem in unknowns])
        for equation in equations:
            print(equation)
        print(all_parameters)
        print(this_output[0])


def polynomial_try_solve(collected_equations: CollectedEquations,
                         unknowns: List[Unknown],
                         all_parameters: Set[sp.Symbol],
                         use_permutation: bool = True,
                         numerical_context: Optional[NumericalAnalyseContext] = None,
                         timeout_seconds: int = 30) -> List[VariableSolution]:
    # Collect the equations and flatten it
    variable_to_equation = collect_equation_by_variables(collected_equations, unknowns)
    variable_tuples: List[Tuple[str]] = list()
    variable_equations: List[List[sp.Expr]] = list()
    for key in variable_to_equation:
        value = variable_to_equation[key]
        variable_tuples.append(key)
        variable_equations.append(value)

    # Process-level details
    n_processor = min(32, len(variable_tuples))
    n_processor = max(n_processor, 1)
    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    result_list = manager.list()

    # Make the input arguments
    argument_list: List[Tuple] = list()
    for i in range(len(variable_tuples)):
        # Collect the result
        variable_names_i = variable_tuples[i]
        equations_i = variable_equations[i]

        # Get the unknowns
        unknowns_i_origin: List[Unknown] = list()
        for name in variable_names_i:
            unknown_with_name = equation_utils.find_unknown(unknowns, name)
            assert unknown_with_name is not None
            unknowns_i_origin.append(unknown_with_name)

        # input argument
        if not use_permutation:
            input_args = (equations_i, unknowns_i_origin, all_parameters, result_list, numerical_context, lock)
            argument_list.append(input_args)
        else:
            input_args_0 = (equations_i, unknowns_i_origin, all_parameters, result_list, numerical_context, lock, True)
            input_args_1 = (equations_i, unknowns_i_origin, all_parameters, result_list, numerical_context, lock, False)
            argument_list.append(input_args_0)
            argument_list.append(input_args_1)

    # Note that this is a two-level loop
    processed_offset = 0
    while processed_offset < len(argument_list):
        processor_list = list()
        for i in range(n_processor):
            offset_i = processed_offset + i
            if offset_i < len(argument_list):
                # OK, make the processor
                processor = multiprocessing.Process(
                    target=solve_processor,
                    args=argument_list[offset_i])
                processor_list.append(processor)

        # Start the processor with time-out of 5 minutes
        for p in processor_list:
            p.start()
        for p in processor_list:
            p.join(timeout=timeout_seconds)
            p.terminate()  # Explicit kill a process if the reduction takes too long

        # Update the offset
        processed_offset += n_processor

    # Process of the output
    solutions: List[VariableSolution] = list()
    for i in range(len(result_list)):
        poly_dict, _, solved_unknown, solution_method = result_list[i]
        this_solution = VariableSolution.make_polynomial_solution(
            solved_unknown.symbol, solution_method, poly_dict)
        print('Polynomial solution {idx} which solves {x}, which is a {n}-order polynomial'.
              format(idx=i, x=solved_unknown.symbol.name, n=len(poly_dict)-1))
        print(poly_dict)
        solutions.append(this_solution)

    # The solutions are verified if data are provided
    return solutions
