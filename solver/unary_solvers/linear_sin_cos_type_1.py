from solver.equation_types import ScalarEquation
from solver.equation_utils import CollectedEquations
from solver.solved_variable import VariableSolution, SolutionMethod, SolutionDegenerateRecord
from solver.unary_solvers.unary_variable_solver import UnaryVariableSolver
from solver.equation_utils import count_unknowns_expr
from typing import Tuple, Optional, List
import sympy as sp


def find_equation_pair(collected_equations: CollectedEquations, var_to_try: sp.Symbol) -> \
        List[Tuple[sp.Expr, sp.Expr]]:
    """
    Find equations in the form of
         C == A sin(x) + B cos(x)
         D == A cos(x) - B sin(x)
    where A, B, C and D doesn't contain x terms
    :param collected_equations:
    :param var_to_try:
    :return: [expr_1, expr_2] tuple or None
    """
    one_unknown_equations = collected_equations.one_unknown_equations
    x = var_to_try
    equation_to_check = list()
    for expr in one_unknown_equations:
        rhs_minus_lhs = expr.rhs - expr.lhs
        rhs_minus_lhs = rhs_minus_lhs.expand()
        if not rhs_minus_lhs.has(x):
            continue
        if rhs_minus_lhs.has(sp.sin(x)) and rhs_minus_lhs.has(sp.cos(x)):
            rhs_minus_lhs = rhs_minus_lhs.collect(sp.sin(x))
            rhs_minus_lhs = rhs_minus_lhs.collect(sp.cos(x))
            if rhs_minus_lhs not in equation_to_check:
                equation_to_check.append(rhs_minus_lhs)

    # Now find the pairs
    expr_ij_list = list()
    for i in range(len(equation_to_check)):
        expr_i = equation_to_check[i]
        sin_coef_i = expr_i.coeff(sp.sin(x))
        cos_coef_i = expr_i.coeff(sp.cos(x))
        for j in range(i + 1, len(equation_to_check)):
            expr_j = equation_to_check[j]
            if expr_i - expr_j == 0 or expr_i + expr_j == 0:
                continue
            sin_coef_j = expr_j.coeff(sp.sin(x))
            cos_coef_j = expr_j.coeff(sp.cos(x))
            if (sin_coef_i == cos_coef_j and cos_coef_i == - sin_coef_j) or (
                    sin_coef_j == - cos_coef_i and cos_coef_i == sin_coef_j):
                expr_ij_list.append((expr_i, expr_j))

    # OK, find all pairs
    return expr_ij_list


def linear_type_1_try_solve(
        collected_equations: CollectedEquations,
        var_to_try: sp.Symbol,
        unknowns: List[sp.Symbol]) -> List[VariableSolution]:
    equation_pair_list = find_equation_pair(collected_equations, var_to_try)
    solution_list = list()
    for equation_pair in equation_pair_list:
        x = var_to_try
        eq1, eq2 = equation_pair
        A = eq1.coeff(sp.sin(x))
        B = eq1.coeff(sp.cos(x))
        C = A * sp.sin(x) + B * sp.cos(x) - eq1
        C = C.simplify()
        D = A * sp.cos(x) - B * sp.sin(x) - eq2
        D = D.simplify()

        if C == 0 and D == 0:
            continue
        # Test that A, B are constants
        if A.has(x) or B.has(x):
            continue
        if count_unknowns_expr(A, unknowns) > 0 or count_unknowns_expr(B, unknowns) > 0:
            continue

        # Now solve it
        solution = sp.atan2(A * C - B * D, A * D + B * C)
        valid_checker = (sp.Abs(A) >= 1e-6) | (sp.Abs(B) >= 1e-6)
        solution_entry = VariableSolution.make_explicit_solution(
            solved_variable=var_to_try,
            solutions=[solution],
            solution_method=SolutionMethod.LinearSinCosType_1.name,
            solve_from_equations=[ScalarEquation(sp.S.Zero, eq1), ScalarEquation(sp.S.Zero, eq2)],
            argument_valid_checkers=[valid_checker],
            # The solution is invalid when both A and B are zero
            degenerate_record=SolutionDegenerateRecord.record_all_equations([A, B])
        )
        solution_list.append(solution_entry)

    # Return the solutions
    return solution_list


class UnaryLinearSolverType_1(UnaryVariableSolver):

    def try_solve(self, collected_equations: CollectedEquations,
                  var_to_try: sp.Symbol,
                  unknowns: List[sp.Symbol]) -> List[VariableSolution]:
        solutions = linear_type_1_try_solve(collected_equations, var_to_try, unknowns)
        return solutions


def test_solver():
    Px, Py, Pz, th_1, th_23, th_3, a_3, a_2, d_4 = sp.symbols('Px Py Pz th_1 th_23 th_3 a_3 a_2 d_4')
    exp1 = Pz * sp.sin(th_23) + a_2 * sp.cos(th_3) + a_3 + (-Px * sp.cos(th_1) - Py * sp.sin(th_1)) * sp.cos(th_23)
    exp2 = Pz * sp.cos(th_23) - a_2 * sp.sin(th_3) + d_4 + (Px * sp.cos(th_1) + Py * sp.sin(th_1)) * sp.sin(th_23)
    e1 = ScalarEquation(sp.S.Zero, exp1)
    e2 = ScalarEquation(sp.S.Zero, exp2)
    collected_equs = CollectedEquations([], [], [])
    collected_equs.one_unknown_equations.append(e1)
    collected_equs.one_unknown_equations.append(e2)

    solver = UnaryLinearSolverType_1()
    sol = solver.try_solve(collected_equs, th_23, [th_23])
    assert len(sol) > 0
    sol = sol[0]
    print(sol.explicit_solutions)
    print(sol.degenerate_record.equations)


if __name__ == '__main__':
    test_solver()
