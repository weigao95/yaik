from solver.equation_types import ScalarEquation, Unknown, NumericalAnalyseContext
from solver.equation_utils import CollectedEquations
from solver.solved_variable import VariableSolution
from solver.solved_variable_impl import LinearSinCosType_2_SolutionImpl
from solver.linear_solver.linear_sin_cos_type2_numerical_solve import \
    try_solve_linear_type2, try_solve_linear_type2_combined
from itertools import combinations
from solver.equation_utils import count_unknowns_expr
from typing import Dict, Optional, List, Tuple, Set
import sympy as sp
import numpy as np
import attr


@attr.s
class LinearCoefficient(object):
    # dot([A, B, C, D], [sp.sin(x), sp.cos(x), sp.sin(y), sp.cos(y)]) = residual
    A: Optional[sp.Symbol] = None
    B: Optional[sp.Symbol] = None
    C: Optional[sp.Symbol] = None
    D: Optional[sp.Symbol] = None
    residual: Optional[sp.Expr] = None


def find_linear_pair_equations_raw(
        equation_to_check: List[ScalarEquation],
        x: sp.Symbol,
        y: sp.Symbol,
        unknowns: List[sp.Symbol]) -> List[LinearCoefficient]:
    """
    For a set of equations, find all of them in the form of
    A sin(x) + B cos(x) + C sin(y) + D cos(y) == residual
    A, B, C, D and residual should not contain anything unknown
    """
    # Iterate over all equations
    coefficient_list: List[LinearCoefficient] = list()
    for this_equation in equation_to_check:
        rhs_minus_lhs: sp.Expr = this_equation.rhs - this_equation.lhs
        rhs_minus_lhs: sp.Expr = rhs_minus_lhs.expand()
        if (not rhs_minus_lhs.has(x)) and (not rhs_minus_lhs.has(y)):
            continue

        # The coefficient
        is_coefficient_valid: bool = True
        coefficient = list()
        for term in [sp.sin(x), sp.cos(x), sp.sin(y), sp.cos(y)]:
            collected_expr: sp.Expr = rhs_minus_lhs.collect(term)
            coefficient_term: sp.Expr = collected_expr.coeff(term)

            # Check the coefficient, should not contain anything in unknowns
            if count_unknowns_expr(coefficient_term, unknowns) > 0:
                is_coefficient_valid = False

            # Check the validity
            if not is_coefficient_valid:
                break
            else:
                coefficient.append(coefficient_term)

        # Check for residual
        if not is_coefficient_valid:
            continue
        assert len(coefficient) == 4
        residual = rhs_minus_lhs - coefficient[0] * sp.sin(x) - coefficient[1] * sp.cos(x) \
                                 - coefficient[2] * sp.sin(y) - coefficient[3] * sp.cos(y)
        residual = sp.simplify(residual)
        if count_unknowns_expr(residual, unknowns) > 0:
            continue

        # Make the result
        this_coefficient = LinearCoefficient()
        this_coefficient.A = coefficient[0]
        this_coefficient.B = coefficient[1]
        this_coefficient.C = coefficient[2]
        this_coefficient.D = coefficient[3]
        this_coefficient.residual = - residual
        coefficient_list.append(this_coefficient)

    # OK
    return coefficient_list


def verify_linear_coefficient_as_solution(
        A: sp.Matrix,
        x: sp.Symbol, y: sp.Symbol,
        test_cases: List[Dict[sp.Symbol, float]],
        pass_solved_ratio: float = 0.9) -> bool:
    """
    For a 3 x 4 matrix A, each row of whom corresponds to a linear equation in the form of
    A sin(x) + B cos(x) + C sin(y) + D cos(y) == 0
    Try all examples in the test_case and count how many of them can be solved, and return
    if the solution count is more than pass_solved_ratio
    """
    solved_count = 0
    for i in range(len(test_cases)):
        test_case_i = test_cases[i]
        assert x in test_case_i
        assert y in test_case_i
        A_value = A.subs(test_case_i)
        y_value = float(test_case_i[y])
        A_np = np.array(A_value).astype(np.float64)
        y_solution_tuple = try_solve_linear_type2(A_np)
        if y_solution_tuple is None:
            continue

        y_solved = False
        tolerance = 1e-3
        for j in range(2):
            if abs(y_solution_tuple[j] - y_value) < tolerance:
                y_solved = True
                break
        if y_solved:
            solved_count += 1

    # Check passing
    return solved_count >= int(pass_solved_ratio * float(len(test_cases)))


def verify_combined_solution(
        solved_symbol: sp.Symbol,
        A_solution: sp.Matrix,
        rows2try: List[Tuple[int, int, int]],
        test_cases: List[Dict[sp.Symbol, float]]):
    """
    For a m x 4 matrix A, each row of whom corresponds to a linear equation in the form of
    A sin(x) + B cos(x) + C sin(y) + D cos(y) == 0
    For 3-tuple of row index in rows2try, gather the three equation and try to solve it.
    Do it for each test case and count how many of them are solved.
    """
    solved_count = 0
    for i in range(len(test_cases)):
        test_case_i = test_cases[i]
        assert solved_symbol in test_case_i
        A_value = A_solution.subs(test_case_i)
        A_np = np.array(A_value).astype(np.float64)
        solution_i = try_solve_linear_type2_combined(A_np, rows2try)
        if solution_i is None:
            continue

        # Check it is solved
        symbol_value = test_case_i[solved_symbol]
        i_solved = False
        tolerance = 1e-3
        for j in range(2):
            if abs(solution_i[j] - symbol_value) < tolerance:
                i_solved = True
                break
        if i_solved:
            solved_count += 1
    print('Solution stat')
    print(solved_count, len(test_cases))


def select_linear_solution_tuples(
        pair_equations: List[LinearCoefficient],
        x: sp.Symbol, y: sp.Symbol, test_cases: List[Dict[sp.Symbol, float]]) -> List[Tuple[int, int, int]]:
    """
    For a m x 4 matrix A, each row of whom corresponds to a linear equation in the form of
    A sin(x) + B cos(x) + C sin(y) + D cos(y) == 0
    We only need three equations to solve y. This function selects 3-tuple that corresponds to the equations
    from which y can be solved
    """
    n_equations = len(pair_equations)
    result_row_tuples: List[Tuple[int, int, int]] = list()
    max_row_tuple_count = 5
    for row_index_tuple in combinations(range(n_equations), 3):
        A = sp.zeros(3, 4)
        for r in range(3):
            A[r, 0] = pair_equations[row_index_tuple[r]].A
            A[r, 1] = pair_equations[row_index_tuple[r]].B
            A[r, 2] = pair_equations[row_index_tuple[r]].C
            A[r, 3] = pair_equations[row_index_tuple[r]].D
        # We can use a low ration here as the final solution would be the merge of many tuples
        pass_solved_ratio = 0.5
        if verify_linear_coefficient_as_solution(A, x, y, test_cases, pass_solved_ratio):
            result_row_tuples.append(row_index_tuple)
        if len(result_row_tuples) >= max_row_tuple_count:
            break
    return result_row_tuples


def find_linear_pair_equations(
        equation_to_check: List[ScalarEquation],
        unknowns: List[Unknown],
        test_cases: List[Dict[sp.Symbol, float]]) -> Optional[VariableSolution]:
    """
    Iterate over all unknown pair (x, y) in unknowns, find one pair with at least three
    equations in the form of
    A sin(x) + B cos(x) + C sin(y) + D cos(y) == 0
    Note that this solver requires ZERO RESIDUAL.
    Verify the solution and return if one if found. Else return None
    """
    # Only use a subset of test cases for initial solve
    sampled_test_cases = None
    initial_solve_n_tests = 50
    if len(test_cases) <= initial_solve_n_tests:
        sampled_test_cases = test_cases
    else:
        sampled_test_cases = list()
        selected_index = np.random.choice(len(test_cases), initial_solve_n_tests)
        for i in range(initial_solve_n_tests):
            sampled_test_cases.append(test_cases[selected_index[i]])

    # Try solve
    n_unknowns = len(unknowns)
    unknown_symbols = [elem.symbol for elem in unknowns]
    for i in range(n_unknowns):
        if not unknowns[i].is_revolute:
            continue
        unknown_i = unknowns[i]
        for j in range(n_unknowns):
            if not unknowns[j].is_revolute:
                continue
            unknown_j = unknowns[j]
            output_ij = find_linear_pair_equations_raw(
                equation_to_check, unknown_i.symbol, unknown_j.symbol, unknown_symbols)

            # Should be at least three equations
            if len(output_ij) < 3:
                continue

            # This version requires no residual
            no_residual = True
            for r in range(len(output_ij)):
                residual_r = output_ij[r].residual
                if residual_r != sp.S.Zero:
                    no_residual = False
                    break
            if not no_residual:
                continue

            # Check the solution
            solved_row_tuples = select_linear_solution_tuples(
                output_ij, unknown_i.symbol, unknown_j.symbol, sampled_test_cases)
            if len(solved_row_tuples) > 0:
                # Convert output_ij to matrix
                n_rows = len(output_ij)
                A_ij = sp.zeros(n_rows, 4)
                for r in range(n_rows):
                    A_ij[r, 0] = output_ij[r].A
                    A_ij[r, 1] = output_ij[r].B
                    A_ij[r, 2] = output_ij[r].C
                    A_ij[r, 3] = output_ij[r].D

                # Make the new solution
                solution_impl = LinearSinCosType_2_SolutionImpl(unknown_j.symbol, A_ij, solved_row_tuples)
                return VariableSolution(solution_impl)

    # Cannot find the solution
    return None


def linear_sin_cos_type2_try_solve(
        collected_equations: CollectedEquations,
        unknowns: List[Unknown],
        all_parameters: Set[sp.Symbol],
        numerical_context: NumericalAnalyseContext) -> Optional[VariableSolution]:
    """
    Simple wrapper in the shape of Unary/Polynomial Solver
    """
    equation_to_check = collected_equations.one_unknown_equations + collected_equations.two_unknown_equations
    numerical_test_cases = numerical_context.numerical_test_cases
    return find_linear_pair_equations(equation_to_check, unknowns, numerical_test_cases)


# Debug code
def test_franka():
    th_0 = sp.Symbol('th_0')
    th_2 = sp.Symbol('th_2')
    th_3, th_4, th_5, th_6 = sp.symbols('th_3 th_4 th_5 th_6')
    r_11, r_12, r_13 = sp.symbols('r_11 r_12 r_13')
    r_21, r_22, r_23 = sp.symbols('r_21 r_22 r_23')
    Px, Py, Pz = sp.symbols('Px Py Pz ')
    a_3, a_5, d_4 = sp.symbols('a_3 a_5 d_4')
    expr_0_lhs = -r_11*sp.sin(th_0) + r_21*sp.cos(th_0)
    expr_0_rhs = ((-sp.sin(th_2)*sp.cos(th_3)*sp.cos(th_4) + sp.sin(th_4)*sp.cos(th_2))*sp.cos(th_5) + sp.sin(th_2)*sp.sin(th_3)*sp.sin(th_5))*sp.cos(th_6) + (sp.sin(th_2)*sp.sin(th_4)*sp.cos(th_3) + sp.cos(th_2)*sp.cos(th_4))*sp.sin(th_6)
    expr_1_lhs = -r_12*sp.sin(th_0) + r_22*sp.cos(th_0)
    expr_1_rhs = -((-sp.sin(th_2)*sp.cos(th_3)*sp.cos(th_4) + sp.sin(th_4)*sp.cos(th_2))*sp.cos(th_5) + sp.sin(th_2)*sp.sin(th_3)*sp.sin(th_5))*sp.sin(th_6) + (sp.sin(th_2)*sp.sin(th_4)*sp.cos(th_3) + sp.cos(th_2)*sp.cos(th_4))*sp.cos(th_6)
    expr_2_lhs = -r_13*sp.sin(th_0) + r_23*sp.cos(th_0)
    expr_2_rhs = (-sp.sin(th_2)*sp.cos(th_3)*sp.cos(th_4) + sp.sin(th_4)*sp.cos(th_2))*sp.sin(th_5) - sp.sin(th_2)*sp.sin(th_3)*sp.cos(th_5)
    expr_3_lhs = -Px*sp.sin(th_0) + Py*sp.cos(th_0)
    expr_3_rhs = -a_3*sp.sin(th_2)*sp.cos(th_3) - a_3*sp.sin(th_2) + a_5*((-sp.sin(th_2)*sp.cos(th_3)*sp.cos(th_4) + sp.sin(th_4)*sp.cos(th_2))*sp.cos(th_5) + sp.sin(th_2)*sp.sin(th_3)*sp.sin(th_5)) - d_4*sp.sin(th_2)*sp.sin(th_3)
    equation_0 = ScalarEquation(expr_0_lhs, expr_0_rhs)
    equation_1 = ScalarEquation(expr_1_lhs, expr_1_rhs)
    equation_2 = ScalarEquation(expr_2_lhs, expr_2_rhs)
    equation_3 = ScalarEquation(expr_3_lhs, expr_3_rhs)

    # Verify with test data
    import solver.degenerate_analyse.numerical_check_data as numerical_check_data
    test_case_path = '../../gallery/test_data/franka_panda_numerical_test.yaml'
    numerical_test_cases = numerical_check_data.load_test_cases(test_case_path)
    for i in range(2):
        pass
        # test_case_i = numerical_test_cases[i]
        # print(expr_3_lhs.subs(test_case_i))
        # print(expr_3_rhs.subs(test_case_i))

    # Try it
    th_0_unknown = Unknown(th_0)
    th_2_unknown = Unknown(th_2)
    solution = find_linear_pair_equations(
        [equation_0, equation_1, equation_2, equation_3],
        [th_0_unknown, th_2_unknown],
        numerical_test_cases)
    linear_solution: LinearSinCosType_2_SolutionImpl = solution.internal_solution
    print(linear_solution.rows_to_try)
    print(linear_solution.A_matrix)

    # Save and load
    solution_dict = solution.to_dict()
    loaded_solution = VariableSolution()
    loaded_solution.from_dict(solution_dict)

    # Overall verification
    verify_combined_solution(
        solution.solved_variable,
        loaded_solution.internal_solution.A_matrix,
        loaded_solution.internal_solution.rows_to_try,
        numerical_test_cases[0:50])


if __name__ == '__main__':
    test_franka()
