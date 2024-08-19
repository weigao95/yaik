from solver.equation_utils import CollectedEquations, ScalarEquation
from solver.solved_variable import SolutionMethod, VariableSolution, SolutionDegenerateRecord
from solver.unary_solvers.unary_variable_solver import UnaryVariableSolver
import fk.fk_equations as fk_equations
from solver.equation_utils import count_unknowns_expr
from typing import List, Tuple
import sympy as sp


def find_equation(
        collected_equations: CollectedEquations,
        var_to_try: sp.Symbol,
        unknowns: List[sp.Symbol]) -> List[Tuple[sp.Expr, sp.Expr, sp.Expr]]:
    """
    Find equations in the form of
         A sin(x) + B cos(x) + C == 0
    where A, B and C doesn't contain x terms
    :param collected_equations:
    :param var_to_try:
    :param unknowns: the symbols that cannot be treated as a constant
    :return:
    """
    one_unknown_equations = collected_equations.one_unknown_equations
    x = var_to_try
    abc_list = list()
    for expr_i in one_unknown_equations:
        rhs_minus_lhs = expr_i.rhs - expr_i.lhs
        if not rhs_minus_lhs.has(x):
            continue

        if rhs_minus_lhs.has(sp.sin(x)) and rhs_minus_lhs.has(sp.cos(x)):
            rhs_minus_lhs = rhs_minus_lhs.expand()
            rhs_minus_lhs = rhs_minus_lhs.collect(sp.sin(x))
            rhs_minus_lhs = rhs_minus_lhs.collect(sp.cos(x))

            # Do it
            sin_coef = rhs_minus_lhs.coeff(sp.sin(x))
            cos_coef = rhs_minus_lhs.coeff(sp.cos(x))
            if sin_coef.has(x) or cos_coef.has(x):
                continue
            if count_unknowns_expr(sin_coef, unknowns) > 0 or \
                    count_unknowns_expr(cos_coef, unknowns) > 0:
                continue

            const: sp.Expr = sin_coef * sp.sin(x) + cos_coef * sp.cos(x) - rhs_minus_lhs
            const: sp.Expr = sp.simplify(const)

            # Check if the const is OK
            if const.has(x):
                continue
            if count_unknowns_expr(const, unknowns) > 0:
                continue

            # This is a valid one
            abc = (sin_coef, cos_coef, const)
            abc_list.append(abc)

    # OK
    return abc_list


def sin_and_cos_try_solve(
        collected_equations: CollectedEquations,
        var_to_try: sp.Symbol,
        unknowns: List[sp.Symbol]) -> List[VariableSolution]:
    abc_list = find_equation(collected_equations, var_to_try, unknowns)
    solution_list = list()
    for abc in abc_list:
        A, B, C = abc
        x = var_to_try
        expr_to_solve = A * sp.sin(x) + B * sp.cos(x) + C

        # Make solution
        solutions = list()
        arguments = None
        validity_checker = None
        if not C == 0:
            t = sp.sqrt(A * A + B * B - C * C)
            solutions.append(sp.atan2(A, B) + sp.atan2(t, C))
            solutions.append(sp.atan2(A, B) + sp.atan2(-t, C))
            arguments = [A, B, t, C]
            validity_checker = (sp.Abs(A) >= fk_equations.zero_tolerance) | \
                               (sp.Abs(B) >= fk_equations.zero_tolerance) | \
                               (sp.Abs(C) >= fk_equations.zero_tolerance)
        else:
            solutions.append(sp.atan2(-B, A))
            solutions.append(sp.atan2(-B, A) + sp.pi)
            arguments = [A, B]
            validity_checker = (sp.Abs(A) >= fk_equations.zero_tolerance) | \
                               (sp.Abs(B) >= fk_equations.zero_tolerance)

        # Make the solution, only used the first
        assert len(solutions) > 0
        assert arguments is not None
        solution_entry = VariableSolution.make_explicit_solution(
            solved_variable=var_to_try,
            solutions=solutions,
            solution_method=SolutionMethod.SinAndCos.name,
            solve_from_equations=[ScalarEquation(sp.S.Zero, expr_to_solve)],
            argument_valid_checkers=[validity_checker, validity_checker],
            degenerate_record=SolutionDegenerateRecord.record_all_equations([A, B, C])
        )
        solution_list.append(solution_entry)

    # Ok
    return solution_list


class UnarySinAndCosSolver(UnaryVariableSolver):

    def try_solve(self, collected_equations: CollectedEquations,
                  var_to_try: sp.Symbol,
                  unknowns: List[sp.Symbol]) -> List[VariableSolution]:
        solutions = sin_and_cos_try_solve(
            collected_equations, var_to_try, unknowns)
        return solutions


# Test code
def test_sin_and_cos_solve():
    import solver.equation_utils as equation_utils
    Td = sp.zeros(4)
    Ts = sp.zeros(4)

    # Make the equations
    l_1, l_2, l_3, l_4, l_6 = sp.symbols('l_1 l_2 l_3 l_4 l_6')
    th_1, th_2, th_3, th_4, th_6 = sp.symbols('th_1 th_2 th_3 th_4 th_6')
    d_1 = sp.symbols('d_1')

    Td[1, 1] = l_1
    Ts[1, 1] = sp.sin(th_2)

    Td[1, 2] = l_2
    Ts[1, 2] = sp.cos(th_3)

    Td[2, 0] = l_6
    Ts[2, 0] = l_1 * sp.sin(th_1) + l_2 * sp.cos(th_1)

    Td[2, 1] = l_2 + 5
    Ts[2, 1] = l_3 * sp.cos(th_3) + l_3 * sp.sin(th_3) + l_4  # match!!

    Td[2, 2] = l_2
    Ts[2, 2] = l_3 * sp.cos(th_6) + l_1 * sp.sin(th_6) + l_4 * sp.sin(th_6)

    Td[3, 3] = l_2 + l_1
    Ts[3, 3] = sp.sin(th_3) * sp.sin(th_4) + l_1 * sp.cos(
        th_4)  # should only match if test repeats and th_3 becomes known

    # OK
    unknowns = [th_1, th_2, th_3, th_4, th_6]
    matrix_equation = fk_equations.MatrixEquation(Td, Ts)
    collected_equs = equation_utils.collect_equations([matrix_equation], unknowns)

    # First solve for d_1
    solver = UnarySinAndCosSolver()
    sol = solver.try_solve(collected_equs, th_1, unknowns)
    assert len(sol) > 0

    th_1_sol_0 = sp.atan2(l_1, l_2) + sp.atan2( sp.sqrt(l_1**2 + l_2**2 - l_6**2), l_6)
    th_1_sol_1 = sp.atan2(l_1, l_2) + sp.atan2(-sp.sqrt(l_1**2 + l_2**2 - l_6**2), l_6)
    th_1_sol: VariableSolution = sol[0]
    assert sp.simplify(th_1_sol.explicit_solutions[0] - th_1_sol_0) == 0
    assert sp.simplify(th_1_sol.explicit_solutions[1] - th_1_sol_1) == 0


if __name__ == '__main__':
    test_sin_and_cos_solve()
