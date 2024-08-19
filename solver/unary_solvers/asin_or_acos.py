from solver.equation_types import ScalarEquation
from solver.equation_utils import CollectedEquations, count_unknowns_expr
from solver.solved_variable import VariableSolution, SolutionMethod, SolutionDegenerateRecord
from solver.unary_solvers.unary_variable_solver import UnaryVariableSolver
from typing import List, Dict, Tuple, Optional
import sympy as sp


def find_equation(collected_equations: CollectedEquations, var_to_try: sp.Symbol, unknowns: List[sp.Symbol],
                  is_sin_expr: bool) -> List[Tuple[sp.Expr, sp.Expr]]:
    """
    Find A sin(x) + B = 0 or A cos(x) + B = 0
    Equation pairs.
    :param collected_equations:
    :param var_to_try:
    :param unknowns:
    :param is_sin_expr:
    :return: (A, B) tuple
    """
    one_unknown_equations = collected_equations.one_unknown_equations
    x = var_to_try
    sin_or_cos_equations = list()
    for expr in one_unknown_equations:
        rhs_minus_lhs = expr.rhs - expr.lhs
        if not rhs_minus_lhs.has(x):
            continue

        # Should be captured by sin and cos solver
        if rhs_minus_lhs.has(sp.sin(x)) and rhs_minus_lhs.has(sp.cos(x)):
            continue

        if is_sin_expr and rhs_minus_lhs.has(sp.sin(x)):
            sin_or_cos_equations.append(expr)
        if (not is_sin_expr) and rhs_minus_lhs.has(sp.cos(x)):
            sin_or_cos_equations.append(expr)

    terms = [sp.sin(x), sp.cos(x)]
    Aw = sp.Wild('Aw')
    Bw = sp.Wild('Bw')
    ab_list: List[Tuple[sp.Expr, sp.Expr]] = list()
    for expr in sin_or_cos_equations:
        expr_collected = (expr.rhs - expr.lhs).collect(terms)
        d_map: Optional[Dict[sp.Wild, sp.Expr]] = None
        if is_sin_expr:
            d_map = expr_collected.match(Aw * sp.sin(x) + Bw)
        else:
            d_map = expr_collected.match(Aw * sp.cos(x) + Bw)

        assert d_map is not None
        A = d_map[Aw]
        B = d_map[Bw]
        assert A is not None
        if B is None:
            B = sp.S.Zero
        if count_unknowns_expr(A, unknowns) > 0 or count_unknowns_expr(B, unknowns) > 0:
            continue
        else:
            ab_list.append((A, B))
    # OK, solution finished
    return ab_list


def asin_or_acos_try_solve(collected_equations: CollectedEquations,
                           var_to_try: sp.Symbol,
                           unknowns: List[sp.Symbol],
                           is_sin_solver: bool) -> List[VariableSolution]:
    ab_list = find_equation(collected_equations, var_to_try, unknowns, is_sin_solver)
    solution_list: List[VariableSolution] = list()
    for ab in ab_list:
        A, B = ab
        x = var_to_try
        if is_sin_solver:
            asin_arg = - B / A
            sol_1 = sp.asin(asin_arg)
            sol_2 = sp.pi - sp.asin(asin_arg)
            expr_to_solve = A * sp.sin(x) + B
            validity_checker = sp.Abs(asin_arg) <= sp.S.One
            solution_entry = VariableSolution.make_explicit_solution(
                solved_variable=var_to_try,
                solutions=[sol_1, sol_2],
                solution_method=SolutionMethod.ArcSin.name,
                solve_from_equations=[ScalarEquation(sp.S.Zero, expr_to_solve)],
                argument_valid_checkers=[validity_checker, validity_checker],
                degenerate_record=SolutionDegenerateRecord.record_all_equations([A]))
            solution_list.append(solution_entry)
        else:
            acos_arg = - B / A
            sol_1 = sp.acos(acos_arg)
            sol_2 = - sp.acos(acos_arg)
            expr_to_solve = A * sp.cos(x) + B
            validity_checker = sp.Abs(acos_arg) <= sp.S.One
            solution_entry = VariableSolution.make_explicit_solution(
                solved_variable=var_to_try,
                solutions=[sol_1, sol_2],
                solution_method=SolutionMethod.ArcCos.name,
                solve_from_equations=[ScalarEquation(sp.S.Zero, expr_to_solve)],
                argument_valid_checkers=[validity_checker, validity_checker],
                degenerate_record=SolutionDegenerateRecord.record_all_equations([A]))
            solution_list.append(solution_entry)

    # Ok, finished
    return solution_list


# The solver interface
class UnaryArcSinSolver(UnaryVariableSolver):

    def try_solve(self,
                  collected_equations: CollectedEquations,
                  var_to_try: sp.Symbol,
                  unknowns: List[sp.Symbol]) -> List[VariableSolution]:
        solutions = asin_or_acos_try_solve(collected_equations, var_to_try, unknowns, True)
        return solutions


class UnaryArcCosSolver(UnaryVariableSolver):

    def try_solve(self,
                  collected_equations: CollectedEquations,
                  var_to_try: sp.Symbol,
                  unknowns: List[sp.Symbol]) -> List[VariableSolution]:
        solutions = asin_or_acos_try_solve(collected_equations, var_to_try, unknowns, False)
        return solutions


# Test code
def test_solver():
    th_2, th_3, l_1, l_2, l_3, l_4 = sp.symbols('th_2 th_3 l_1 l_2  l_3 l_4')
    equations = CollectedEquations([], [], [])
    equations.one_unknown_equations.append(ScalarEquation(l_1, l_1 * sp.sin(th_3) + 15))
    equations.one_unknown_equations.append(ScalarEquation(l_1, l_2 * sp.cos(th_2) + (l_4 + l_3) * sp.cos(th_2)))
    equations.one_unknown_equations.append(ScalarEquation(l_1, l_2 * sp.sin(th_3)))
    sol_th_2 = asin_or_acos_try_solve(equations, th_2, [th_2, th_3], False)
    assert len(sol_th_2) > 0
    sol_th_2 = sol_th_2[0]
    print(sol_th_2.explicit_solutions)
    sol_th_3 = asin_or_acos_try_solve(equations, th_3, [th_2, th_3], True)
    assert len(sol_th_3) == 2
    assert sol_th_3 is not None
    sol_th_3 = sol_th_3[0]
    print(sol_th_3.explicit_solutions)


if __name__ == '__main__':
    test_solver()
