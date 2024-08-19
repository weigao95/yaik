from solver.degenerate_analyse.analyser import \
    DegenerateAnalyser, DegenerateAnalyserContext, DegenerateAnalyserOption
from solver.degenerate_analyse.numerical_analyser import NumericalCheckDegenerateAnalyzer
from solver.solved_variable import VariableSolution, DegenerateType, NumericalAnalysedResultType
from solver.solved_variable import VariableSolutionClassKey
from typing import List, Set, Dict, Tuple, Optional

solution_preference_order = {
    VariableSolutionClassKey.ExplicitSolution.name: 0,
    VariableSolutionClassKey.PolynomialSolution.name: 1,
    VariableSolutionClassKey.LinearSinCosType_2.name: 2,
    VariableSolutionClassKey.General6DoFNumericalReduce.name: 3
}


def select_solution_in_type(
        input_solutions: List[VariableSolution],
        degenerate_type: Optional[str]) -> Optional[VariableSolution]:
    """
    Given a list of input solutions, select the one with the least number of solutions/operations
    :param input_solutions: the input solution to select from
    :param degenerate_type: If not None, then only select solution that matched the type.
                            If None, then ignore it.
    """
    solution_selected: Optional[VariableSolution] = None
    for k in range(len(input_solutions)):
        solution_k = input_solutions[k]
        if (degenerate_type is None) or solution_k.degenerate_record.type == degenerate_type:
            pass
        else:
            # Ignore this one
            continue

        # Init
        if solution_selected is None:
            solution_selected = solution_k
            continue

        # From low preference to high preference
        assert solution_selected is not None
        solution_selected_type = solution_selected.impl_class_key
        solution_k_type = solution_k.impl_class_key
        assert solution_selected_type in solution_preference_order
        assert solution_k_type in solution_selected_type
        solution_selected_order = solution_preference_order[solution_selected_type]
        solution_k_order = solution_preference_order[solution_k_type]

        # Do select based on different order
        if solution_selected_order < solution_k_order:
            continue
        elif solution_selected_order > solution_k_order:
            solution_selected = solution_k
            continue

        # Now the order is the same
        assert solution_selected_type == solution_k_type
        if solution_selected.is_explicit_solution:
            assert solution_k.is_explicit_solution
            if solution_k.num_solutions() < solution_selected.num_solutions():
                solution_selected = solution_k
                continue
            elif solution_k.num_solutions() == solution_selected.num_solutions():
                if solution_k.count_explicit_solution_total_ops() < solution_selected.count_explicit_solution_total_ops():
                    solution_selected = solution_k
        elif solution_selected.is_polynomial:
            assert solution_k.is_polynomial
            if solution_k.count_polynomial_coefficient_ops() < solution_selected.count_polynomial_coefficient_ops():
                solution_selected = solution_k
        else:
            raise RuntimeError('General_6Dof/LinearType_2 should appears as the only solution if it is used')

    # Maybe None, but checking
    if len(input_solutions) > 0 and degenerate_type is None:
        assert solution_selected is not None
    return solution_selected


def select_solution_in_type_old(
        input_solutions: List[VariableSolution],
        degenerate_type: Optional[str]) -> Optional[VariableSolution]:
    """
    Given a list of input solutions, select the one with the least number of solutions/operations
    :param input_solutions: the input solution to select from
    :param degenerate_type: If not None, then only select solution that matched the type.
                            If None, then ignore it.
    """
    solution_selected: Optional[VariableSolution] = None
    for k in range(len(input_solutions)):
        solution_k = input_solutions[k]
        if (degenerate_type is None) or solution_k.degenerate_record.type == degenerate_type:
            pass
        else:
            # Ignore this one
            continue

        # Init
        if solution_selected is None:
            solution_selected = solution_k
            continue

        # From non-explicit to explicit
        assert solution_selected is not None
        if (not solution_selected.is_explicit_solution) and solution_k.is_explicit_solution:
            solution_selected = solution_k
            continue

        # Never change from explicit to non-explicit
        if solution_selected.is_explicit_solution and (not solution_k.is_explicit_solution):
            continue

        # Select between polynomial solutions
        # Polynomial solution may have very long, just count it (without considering the #)
        if solution_selected.is_polynomial and solution_k.is_polynomial:
            if solution_k.count_polynomial_coefficient_ops() < solution_selected.count_polynomial_coefficient_ops():
                solution_selected = solution_k
            # Other rules are not for polynomial solution, jump to next
            continue

        # General 6-dof case
        if solution_selected.is_general_6dof_solution and (not solution_k.is_general_6dof_solution):
            solution_selected = solution_k
            continue
        if solution_selected.is_general_6dof_solution and solution_k.is_general_6dof_solution:
            raise RuntimeError('General 6Dof should appears as the only solution if it is used')

        # Now both solution should be explicit
        assert solution_k.is_explicit_solution and solution_selected.is_explicit_solution
        if solution_k.num_solutions() < solution_selected.num_solutions():
            solution_selected = solution_k
            continue
        elif solution_k.num_solutions() == solution_selected.num_solutions():
            if solution_k.count_explicit_solution_total_ops() < solution_selected.count_explicit_solution_total_ops():
                solution_selected = solution_k

    # Maybe None, but checking
    if len(input_solutions) > 0 and degenerate_type is None:
        assert solution_selected is not None
    return solution_selected


def select_solution(
        solution_list: List[VariableSolution],
        degenerate_context: DegenerateAnalyserContext(),
        degenerate_option: DegenerateAnalyserOption()) -> Optional[VariableSolution]:
    """
    Give a list of solutions, perform analyse of their degenerate record and select the one with most
    desirable type of degeneration and least number of solutions.
    :param solution_list:
    :param degenerate_context:
    :param degenerate_option:
    :return:
    """
    # Do analyse for each solution
    analyser = DegenerateAnalyser()
    for i in range(len(solution_list)):
        solution_i = solution_list[i]
        degenerate_record_i = solution_i.degenerate_record
        new_record = analyser.perform_analyse(
            degenerate_record_i,
            context=degenerate_context,
            option=degenerate_option)
        if new_record is not None:
            solution_list[i].update_degenerate_record(new_record)

    # According to the preferred order, find the solution
    # If we have always non-degenerate, just select from them
    non_degenerate_solution = select_solution_in_type(solution_list, DegenerateType.AlwaysNonDegenerate.name)
    if non_degenerate_solution is not None:
        return non_degenerate_solution

    # Select from solved-variable degeneration
    variable_degenerate_solution = select_solution_in_type(
        solution_list, DegenerateType.DegenerateOnVariableValue.name)
    if variable_degenerate_solution is not None:
        return variable_degenerate_solution

    # If no numerical solution just select others
    if len(degenerate_context.numerical_test_cases) == 0:
        print('I don not have always non-degenerate/degenerate on variable solution, '
              'but do not have numerical test cases. Thus, just make selection. ')
        selected_sol = select_solution_in_type(solution_list, DegenerateType.DegenerateIfAllEquationsZero.name)
        selected_sol = selected_sol if selected_sol is not None \
            else select_solution_in_type(solution_list, DegenerateType.CannotAnalyse.name)
        return selected_sol

    # Select other solutions using numerical analyze
    print('I don not have always non-degenerate/degenerate on variable solution, thus need numerical analysis')
    numerical_analyzer = NumericalCheckDegenerateAnalyzer(degenerate_context.numerical_test_cases)
    numerical_always_non_degenerate_solution: List[VariableSolution] = list()
    numerical_usually_non_degenerate_solution: List[VariableSolution] = list()
    numerical_degenerate_solution: List[VariableSolution] = list()
    cannot_analysis_solution: List[VariableSolution] = list()
    for i in range(len(solution_list)):
        solution_i = solution_list[i]
        degenerate_record_i = solution_i.degenerate_record
        if degenerate_record_i.type == DegenerateType.CannotAnalyse.name:
            cannot_analysis_solution.append(solution_i)
            continue

        # Perform analysis of equation all zero solution
        assert degenerate_record_i.type == DegenerateType.DegenerateIfAllEquationsZero.name
        analyzed_type, degenerate_ratio = numerical_analyzer.perform_analyse(
            degenerate_record_i,
            degenerate_context.solved_variables,
            degenerate_context.all_parameters,
            degenerate_context.parameter_bounds,
            degenerate_context.parameter_value)
        if analyzed_type == NumericalAnalysedResultType.NumericalAlwaysNonDegenerate.name:
            solution_list[i].update_numerical_analysis_result(
                NumericalAnalysedResultType.NumericalAlwaysNonDegenerate.name)
            numerical_always_non_degenerate_solution.append(solution_i)
        elif analyzed_type == NumericalAnalysedResultType.NumericalUsuallyNonDegenerate.name:
            solution_list[i].update_numerical_analysis_result(
                NumericalAnalysedResultType.NumericalUsuallyNonDegenerate.name)
            numerical_usually_non_degenerate_solution.append(solution_i)
        elif analyzed_type == NumericalAnalysedResultType.NumericalDegenerate.name:
            solution_list[i].update_numerical_analysis_result(
                NumericalAnalysedResultType.NumericalDegenerate.name)
            numerical_degenerate_solution.append(solution_i)
        else:
            raise RuntimeError("Unknown numerical result")

    # Some logging
    n_numerical_non_degenerate = len(numerical_always_non_degenerate_solution)
    n_numerical_usually_non_degnerate = len(numerical_usually_non_degenerate_solution)
    n_numerical_degenerate = len(numerical_degenerate_solution)
    n_cannot_analyze = len(cannot_analysis_solution)
    print('After numerical testing, {n0} always non-degenerate, {n1} usually non-degenerate, {n2} degenerate solutions'.
          format(n0=n_numerical_non_degenerate,
                 n1=n_numerical_usually_non_degnerate,
                 n2=n_numerical_degenerate))

    # Do selection
    solution = select_solution_in_type(numerical_always_non_degenerate_solution, None)
    solution = solution if solution is not None \
        else select_solution_in_type(numerical_usually_non_degenerate_solution, None)
    solution = solution if solution is not None \
        else select_solution_in_type(cannot_analysis_solution, None)

    # A checking
    if n_numerical_non_degenerate + n_numerical_usually_non_degnerate + n_cannot_analyze > 0:
        assert solution is not None

    # OK
    return solution
