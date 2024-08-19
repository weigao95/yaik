from solver.general_6dof.numerical_reduce_closure_equation import NumericalReduceInput, verify_solve
from solver.equation_utils import cast_expr_to_float
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import sympy as sp
import numpy as np


def select_lines_to_reduce(
        symbolic_out: NumericalReduceInput,
        test_cases: List[Dict[sp.Symbol, float]],
        logging: bool = True) -> Optional[Tuple[int]]:
    A_sin = symbolic_out.lhs_A_sin
    A_cos = symbolic_out.lhs_A_cos
    C_const = symbolic_out.lhs_C_const
    matrix_equation = symbolic_out.matrix_equation
    nonlinear_coefficient = symbolic_out.expr_should_be_zero

    # Check test cases
    if logging:
        print('Checking expression that should be zero')
    for expr_should_be_zero in nonlinear_coefficient:
        for i in range(len(test_cases)):
            test_case_i = test_cases[i]
            expr_subst = expr_should_be_zero.subs(test_case_i)
            expr_value = cast_expr_to_float(expr_subst)
            if expr_value is None or abs(expr_value) > 1e-7:
                return None

    if logging:
        print('Checking passed.')

    # Only two lines from
    only_select_2_lines_from: Optional[List[int]] = list()
    for row_idx in [2, 5, 6, 7, 10, 13]:
        n_zeros = 0
        for i in range(matrix_equation.rhs_matrix.shape[1]):
            if matrix_equation.rhs_matrix[row_idx, i] == sp.S.Zero:
                n_zeros += 1
        if n_zeros >= 6:
            only_select_2_lines_from.append(row_idx)
    if len(only_select_2_lines_from) <= 2:
        only_select_2_lines_from = None

    # The threshold
    n_total_try_threshold = 400 if (only_select_2_lines_from is not None) else 2000
    n_break_on_fully_solved_lines = 2
    n_fully_solved_lines = 0
    n_total_solved = 0
    n_total_try = 0

    # Initial select of lines
    n_initial_select = min(len(test_cases), 4)
    lines_to_succeed: List[Tuple[Tuple[int], int]] = list()
    n_equations, n_rhs_vars = matrix_equation.rhs_matrix.shape[0], matrix_equation.rhs_matrix.shape[1]
    for equation_indices in combinations(range(n_equations), n_rhs_vars):
        if only_select_2_lines_from is not None:
            select_from_count = 0
            for row_idx in equation_indices:
                if row_idx in only_select_2_lines_from:
                    select_from_count += 1
            if select_from_count >= 3:
                continue

        # Break condition as we cannot iterate over everything
        if n_total_try > n_total_try_threshold:
            break
        if n_fully_solved_lines >= n_break_on_fully_solved_lines:
            break
        if logging:
            print('Initial try of selected lines ', equation_indices, ' total try count is ', n_total_try)

        # For this case
        n_solved = 0
        for i in range(n_initial_select):
            n_total_try += 1
            # test_idx_i = np.random.randint(0, len(test_cases), 1)
            test_idx_i = i
            test_case_i = test_cases[int(test_idx_i)]
            solved_i = verify_solve(
                A_sin, A_cos, C_const, matrix_equation, test_case_i, equation_indices)

            # Update the counter
            if solved_i:
                n_solved += 1
                n_total_solved += 1

        # Assign back
        if logging:
            print('Test result ', equation_indices, ' n solved equations ', n_solved)
        lines_to_succeed.append((equation_indices, n_solved))
        if n_solved == n_initial_select:
            n_fully_solved_lines += 1

    # Do nothing if not solved
    if n_total_solved == 0:
        return None

    # First select the fully solved ones
    assert len(lines_to_succeed) > 0
    lines_to_succeed.sort(key=(lambda s: s[1]), reverse=True)
    n_max_solved = lines_to_succeed[0][1]
    assert n_max_solved > 0  # else should return above
    print('We found feasible solutions in general solver of variable {var}, now do stage 2 selection'.
          format(var=symbolic_out.var_in_lhs_matrix.variable_symbol.name))

    select_lines_to_reduce_stage_2: List[Tuple[Tuple[int], int]] = list()
    for i in range(min(len(lines_to_succeed), n_break_on_fully_solved_lines)):
        selected_lines_i, initial_solved_i = lines_to_succeed[i]
        solved_by_lines_i = 0
        for j in range(len(test_cases)):
            test_case_j = test_cases[j]
            solved_j = verify_solve(
                A_sin, A_cos, C_const, matrix_equation, test_case_j, selected_lines_i)
            if solved_j:
                solved_by_lines_i += 1

            # Logging
            if j % 30 == 1:
                print('For lines ', selected_lines_i, 'iteration ', j, ' the # of solved is ', solved_by_lines_i)
        select_lines_to_reduce_stage_2.append((selected_lines_i, solved_by_lines_i))

    # Return the one that is most likely
    select_lines_to_reduce_stage_2.sort(key=(lambda s: s[1]), reverse=True)
    selected_lines, n_solved = select_lines_to_reduce_stage_2[0]
    print('Reduce on ', selected_lines, f' the number of solved equations are {n_solved} in {len(test_cases)}')
    if n_solved < int(0.8 * len(test_cases)):
        return None
    return selected_lines


# Debug code
def test_numerical_reduce():
    import fk.robots as robots
    import solver.general_6dof.dh_utils as dh_utils
    from solver.general_6dof.numerical_reduce_closure_equation import \
        build_reduce_inputs, generate_numerical_reduce_input, matrix_to_value, numerical_reduce
    robot = robots.yaskawa_HC10_robot()
    dh_params = dh_utils.modified_dh_to_classic(robot.dh_params)
    revolute_vars = dh_utils.RevoluteVariable.convert_from_robot_unknowns(robot)
    test_cases = dh_utils.generate_classic_dh_numerical_test(
        dh_params, revolute_vars, robot.parameters_value, n_test_cases=50)

    reduce_input_tuple = build_reduce_inputs(robot)
    assert reduce_input_tuple is not None
    reduce_input_list, revolute_vars = reduce_input_tuple
    numerical_reduce_input_list = generate_numerical_reduce_input(reduce_input_list[4], revolute_vars)
    numerical_reduce_input = numerical_reduce_input_list[0]
    # selected_lines = select_lines_to_reduce(numerical_reduce_input, test_cases)
    # if selected_lines is None:
    #     print('There are not solution for this input')
    #     return
    assert numerical_reduce_input is not None
    A_sin = numerical_reduce_input.lhs_A_sin
    A_cos = numerical_reduce_input.lhs_A_cos
    C_const = numerical_reduce_input.lhs_C_const
    matrix_equation = numerical_reduce_input.matrix_equation
    nonlinear_coefficient = numerical_reduce_input.expr_should_be_zero

    # For each test case
    solved_counter = 0
    for i in range(len(test_cases)):
        test_case_i = test_cases[i]
        A_sin_value = matrix_to_value(A_sin, test_case_i)
        A_cos_value = matrix_to_value(A_cos, test_case_i)
        C_const_value = matrix_to_value(C_const, test_case_i)
        N_rhs_value = matrix_to_value(matrix_equation.rhs_matrix, test_case_i)

        # Do numerical reduce
        only_two_lines_from = [2, 5, 6, 7, 10, 13]
        selected_lines = [0, 1, 2, 3, 4, 6, 8, 9]
        numerical_reduce_out = numerical_reduce(A_sin_value, A_cos_value, C_const_value, N_rhs_value, selected_lines)
        if numerical_reduce_out is None:
            print('Entering the loop for ', selected_lines)
            for equation_indices in combinations(range(14), 8):
                numerical_reduce_out = numerical_reduce(
                    A_sin_value, A_cos_value, C_const_value, N_rhs_value, equation_indices)
                if numerical_reduce_out is not None:
                    print('Invertible at', equation_indices)
                    break

        # Get out
        if numerical_reduce_out is None:
            print('Cannot find invertible submatrix')
            continue

        find_in_solution = verify_solve(
                A_sin, A_cos, C_const, matrix_equation, test_case_i, selected_lines)
        if find_in_solution:
            print('Find the solution in test case ', i)
            solved_counter += 1

    # Logging
    print(f'Solve {solved_counter} among {len(test_cases)} test cases')

    # Ensure everything in nonlinear_coefficient are zero
    for test_case_i in test_cases:
        for equation in nonlinear_coefficient:
            equation_subst = equation.subs(test_case_i)
            equation_value = cast_expr_to_float(equation_subst)
            if abs(equation_value) > 1e-10:
                print('A nonlinear term that should be zero but it is not ', equation)


if __name__ == '__main__':
    # np.random.seed(0)
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(precision=30)
    # test_aggregate()
    test_numerical_reduce()
