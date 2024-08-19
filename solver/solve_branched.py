from solver.equation_types import ScalarEquation
from solver.solved_variable import VariableSolution, DegenerateType, SolutionDegenerateRecord, SolutionMethod
from solver.solver_snapshot import SolverSnapshot
from solver.degenerate_analyse.numerical_check_data import generate_test_cases
import solver.equation_utils as equation_utils
import solver.build_equations as build_equations
from solver.build_equations import IkEquationCollection
from solver.solve_equations import solve_from_snapshot
from fk.robots import RobotDescription, Unknown
from typing import List, Dict, Tuple, Optional
import sympy as sp


def construct_variable_branched_equations(
        ik_equation: IkEquationCollection,
        solved_variables_from_root: List[VariableSolution],
        degenerate_solution_list: List[Tuple[sp.Symbol, sp.Expr]]
) -> Tuple[SolverSnapshot, List[VariableSolution]]:
    """
    Given a degenerate solution, check whether it provides solution for some variables,
    and return the remaining equation/variables to solve.
    """
    # General parameters
    all_symbol_set = ik_equation.all_symbol_set
    parameter_values = ik_equation.parameter_value_dict
    parameter_bounds = ik_equation.parameter_bound_dict
    last_solution = solved_variables_from_root[-1]
    assert last_solution.degenerate_record.is_degenerate_on_variable_values

    # Functor to checker whether this is a solved unknown
    # We cannot include the last solution, as it assumes degenerate
    def is_solved_unknown(symbol_name: str) -> bool:
        for k in range(len(solved_variables_from_root) - 1):
            solved_var = solved_variables_from_root[k]
            if solved_var.solved_variable.name == symbol_name:
                return True
        return False

    # The unknowns
    original_unknowns = ik_equation.used_unknowns
    original_unknown_names = [elem.symbol.name for elem in original_unknowns]

    # Append unsolved variable to solution
    solution_from_degenerate: List[VariableSolution] = list()
    for elem in degenerate_solution_list:
        var, sol = elem
        # If this is a un-solved unknown, where is rare
        if var.name in original_unknown_names and (not is_solved_unknown(var.name)):
            this_solution = VariableSolution.make_explicit_solution(
                solved_variable=var,
                solutions=[sol],
                solution_method=SolutionMethod.OneVariableAlgebra.name,
                solve_from_equations=None, arguments=[], argument_valid_checkers=[sp.S.BooleanTrue])
            solution_from_degenerate.append(this_solution)
            continue

    # The new unknowns
    remaining_unknowns: List[Unknown] = list()
    solution_from_degenerate_symbol_names: List[str] = [elem.solved_variable.name for elem in solution_from_degenerate]
    for unknown in original_unknowns:
        if is_solved_unknown(unknown.symbol.name):
            continue
        if unknown.symbol.name in solution_from_degenerate_symbol_names:
            continue
        # This is not solved yet
        remaining_unknowns.append(unknown)

    # Equations
    new_scalar_equations: List[ScalarEquation] = list()
    for i in range(len(ik_equation.scalar_equations)):
        fk_equation_i = ik_equation.scalar_equations[i]
        new_lhs = fk_equation_i.lhs.subs(degenerate_solution_list)
        new_rhs = fk_equation_i.rhs.subs(degenerate_solution_list)
        if equation_utils.count_unknowns(new_lhs, new_rhs, [elem.symbol for elem in remaining_unknowns]) == 0:
            continue
        # new_lhs = sp.expand_trig(sp.simplify(new_lhs))
        # new_rhs = sp.expand_trig(sp.simplify(new_rhs))
        new_equation_i = ScalarEquation(new_lhs, new_rhs, fk_equation_i.equation_type)
        new_scalar_equations.append(new_equation_i)

    # Given back the equations
    solver_input = SolverSnapshot(
        scalar_equations=new_scalar_equations,
        sos_hints=list(),
        unknowns=remaining_unknowns,
        all_parameters=all_symbol_set
    )
    solver_input.parameter_values = parameter_values
    solver_input.parameter_bounds = parameter_bounds
    return solver_input, solution_from_degenerate


def branched_equations_at_solved_variable_degeneration(
        ik_equation: IkEquationCollection,
        solved_variables_from_root: List[VariableSolution]) -> List[Tuple[SolverSnapshot, List[VariableSolution]]]:
    """
    Given a set of solved variables, generate the solver input for the DEGENERATED branch.
    In this function, the dependency comes from variable value.
    :param ik_equation
    :param solved_variables_from_root:
    :return: A list of solver input, one for each degenerate solution
    """
    assert len(solved_variables_from_root) > 0
    last_solution = solved_variables_from_root[-1]
    degenerate_record: SolutionDegenerateRecord = last_solution.degenerate_record
    assert degenerate_record.type == DegenerateType.DegenerateOnVariableValue.name

    # Find the # of solutions
    n_solutions = degenerate_record.count_number_variable_solutions()

    # Zero case
    if n_solutions is None or n_solutions == 0:
        return list()

    # Solver equation list
    solver_input_list: List[Tuple[SolverSnapshot, List[VariableSolution]]] = list()
    for i in range(n_solutions):
        solution_list_i = degenerate_record.get_variable_solution(i)
        solver_input_i, new_solution_i = construct_variable_branched_equations(
            ik_equation, solved_variables_from_root, solution_list_i)

        # Add test cases
        solution_dict_i = dict()
        all_solution_are_numerical = True
        for k in range(len(solution_list_i)):
            var_k, sol_k = solution_list_i[k]
            sol_k_value: Optional[float] = equation_utils.cast_expr_to_float(sol_k)
            if sol_k_value is None:
                print('The solution is not a number, ', sol_k)
                all_solution_are_numerical = False
            else:
                solution_dict_i[var_k] = sol_k_value

        # We can generate test cases by ourselves
        n_numerical_test_cases = 1000
        if all_solution_are_numerical:
            solver_input_i.numerical_test_cases = list()
            numerical_test_cases = generate_test_cases(
                ik_equation,
                n_test_cases=n_numerical_test_cases,
                subset_variable_value=solution_dict_i)
            solver_input_i.numerical_test_cases = numerical_test_cases
        else:
            # Find a subset from the given input
            print('Warning: Test case generation for non-numerical solution has not been implemented')
            solver_input_i.numerical_test_cases = list()

        # Append to result and done with i
        solver_input_list.append((solver_input_i, new_solution_i))

    # Ok
    return solver_input_list
