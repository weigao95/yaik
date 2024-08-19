from fk.robots import RobotDescription, Unknown
from solver.build_equations import IkEquationCollection
from solver.solved_variable import VariableSolution
from solver.equation_types import EquationInput
import solver.equation_utils as equation_utils
from solver.solve_equations import solve_equations, SolveEquationOption
import solver.general_6dof.numerical_reduce_solver as general_6dof_solver
from typing import List, Tuple, Dict, Set
import sympy as sp


def solve_subset_equations(ik_equation: IkEquationCollection,
                           numerical_test_cases: List[Dict[sp.Symbol, float]] = list()) -> List[VariableSolution]:
    """
    Iterate over the equation dict, which maps from unknowns to their dependent equations.
    Try to solve it and return the solution if one was found.
    """
    # Check the input
    sub_equation_dict = ik_equation.sub_equation_dict
    if len(sub_equation_dict) == 0:
        return list()

    # Sort the key according to the # of unknowns
    sub_equation_size2key = [(len(key), key) for key in sub_equation_dict.keys()]
    sorted_keys = sorted(sub_equation_size2key, key=lambda x: x[0])
    sorted_keys = [elem[1] for elem in sorted_keys]

    # Start the iteration from the one with smallest # of unknowns
    for key in sorted_keys:
        # Collect the variables
        subset_vars: List[Unknown] = list()
        for elem in key:
            unknown_for_elem = None
            for i in range(len(ik_equation.used_unknowns)):
                if ik_equation.used_unknowns[i].symbol.name == elem:
                    unknown_for_elem = ik_equation.used_unknowns[i]
            # The unknown might present in use_unknowns
            # Or it is treated as a parameter
            if unknown_for_elem is not None:
                subset_vars.append(unknown_for_elem)
            else:
                # It might have been moved to symbolic parameters, check it
                elem_symbol = sp.Symbol(elem)
                assert elem_symbol in ik_equation.unknown_as_parameter_more_dof
                assert elem_symbol in ik_equation.parameter_bound_dict

        # Logging
        print('Try solve ', [elem.symbol.name for elem in subset_vars])
        if len(ik_equation.unknown_as_parameter_more_dof) > 0:
            print('A subset of unknowns are treated as parameters: ',
                  [elem.name for elem in ik_equation.unknown_as_parameter_more_dof])

        # Start the solving
        equation_input = EquationInput([], [])
        translational_equations, cos_equations = ik_equation.sub_equation_dict[key]
        equation_utils.append_intersecting_axis_equations_to_input(
            translational_equations, equation_input, cos_equations)

        # We would stop on bad solutions here
        solve_option = SolveEquationOption()
        solve_option.stop_on_bad_solution = True

        # Ok
        numerical_context = ik_equation.make_numerical_context(numerical_test_cases)
        key_solutions = solve_equations(
            equation_input,
            subset_vars,
            ik_equation.all_symbol_set,
            numerical_context=numerical_context,
            option=solve_option)

        # If ok
        if len(key_solutions) > 0:
            print('I have solved a subset of unknowns: ', [var.solved_variable.name for var in key_solutions])
            return key_solutions

    # We cannot solve it
    return list()


def collect_remaining_unknowns(
        ik_equation: IkEquationCollection,
        solved_variables: List[VariableSolution]) -> List[Unknown]:
    """
    Maybe a subset of unknowns has been solved, get the remaining unknowns.
    """
    current_unknowns: List[Unknown] = list()
    for unknown in ik_equation.used_unknowns:
        found_in_solved = False
        for solved_unknown in solved_variables:
            if solved_unknown.solved_variable.name == unknown.symbol.name:
                found_in_solved = True
                break
        if not found_in_solved:
            current_unknowns.append(unknown)
    return current_unknowns


def solve_main_branch(
        robot: RobotDescription,
        ik_equation: IkEquationCollection,
        numerical_test_cases: List[Dict[sp.Symbol, float]] = list(),
        try_intersecting_axis_equation: bool = True,
        use_polynomial_solver: bool = True) -> List[VariableSolution]:
    """
    Try solving the ik problem by first trying the subset of variables.
    Then complete with solving the remaining parts.
    :param robot:
    :param ik_equation: the processed ik equations
    :param numerical_test_cases: a list of sympy.subs map for testing the solution
    :param try_intersecting_axis_equation: should we try solving a subset of intersecting axis equation
    :param use_polynomial_solver: should we use polynomial solver
    """
    # Solve the subset-equations
    solutions = list()
    if try_intersecting_axis_equation:
        solutions = solve_subset_equations(ik_equation, numerical_test_cases)

    # Maybe a subset has been solved, get the remaining unknowns
    current_unknowns: List[Unknown] = collect_remaining_unknowns(ik_equation, solutions)

    # Log some info
    print('Now we need to solve ', [elem.symbol.name for elem in current_unknowns])

    # The option for general solver
    general_6dof_solvable = general_6dof_solver.is_solvable(robot)
    solve_equation_option = SolveEquationOption()
    if general_6dof_solvable and (len(solutions) == 0):
        print('We would try general solver if analytical/polynomial solver does not work')
        solve_equation_option.stop_on_first_unsolved = True
    if not use_polynomial_solver:
        solve_equation_option.disable_polynomial_solution = True

    # Solve the remaining equations
    equation_input = EquationInput(ik_equation.scalar_equations, ik_equation.sos_hint)
    numerical_context = ik_equation.make_numerical_context(numerical_test_cases)
    remaining_solutions = solve_equations(
        equation_input,
        current_unknowns,
        ik_equation.all_symbol_set,
        numerical_context=numerical_context,
        option=solve_equation_option
    )

    # Try with general solver
    if len(solutions) == 0 and len(remaining_solutions) == 0 and general_6dof_solvable:
        print('Try general solver as we cannot find new solutions')
        return solve_with_general_6dof(robot, ik_equation, numerical_test_cases)
    else:
        # Maybe we have solved, do a checking?
        solutions.extend(remaining_solutions)
        return solutions


def solve_with_general_6dof(
        robot: RobotDescription,
        ik_equation: IkEquationCollection,
        numerical_test_cases: List[Dict[sp.Symbol, float]] = list()) -> List[VariableSolution]:
    general_6dof_solution_list = general_6dof_solver.try_solve_parallel_v2(robot)
    if len(general_6dof_solution_list) == 0:
        return list()

    # Select one from the general 6-dof
    for enabled_polynomial in [False, True]:
        solved_var_set: Set[sp.Symbol] = set()
        for solution_i in general_6dof_solution_list:
            # We do not need to try again if the same variable has been solved
            if solution_i.solved_variable in solved_var_set:
                continue
            solved_var_set.add(solution_i.solved_variable)

            # Try with this one
            current_solutions = [VariableSolution(solution_impl=solution_i)]
            current_unknowns = collect_remaining_unknowns(ik_equation, current_solutions)
            print('Now we try to solve ', [elem.symbol.name for elem in current_unknowns],
                  ' as ', solution_i.solved_variable.name, 'is solved')

            # Build the option
            option = SolveEquationOption()
            if not enabled_polynomial:
                option.disable_polynomial_solution = True
                option.stop_on_first_unsolved = True

            # Solve the remaining equations
            equation_input = EquationInput(ik_equation.scalar_equations, ik_equation.sos_hint)
            numerical_context = ik_equation.make_numerical_context(numerical_test_cases)
            remaining_solutions = solve_equations(
                equation_input,
                current_unknowns,
                ik_equation.all_symbol_set,
                numerical_context=numerical_context,
                option=option
            )

            # If solved
            if len(remaining_solutions) > 0:
                current_solutions.extend(remaining_solutions)
                return current_solutions

    # None of them are solvable
    return list()
