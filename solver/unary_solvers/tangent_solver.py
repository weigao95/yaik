from solver.equation_types import ScalarEquation
from solver.equation_utils import CollectedEquations, count_unknowns_expr
from solver.solved_variable import VariableSolution, SolutionMethod, SolutionDegenerateRecord
from solver.unary_solvers.unary_variable_solver import UnaryVariableSolver
import fk.fk_equations as fk_equations
from typing import List, Dict, Tuple, Optional
import sympy as sp


def find_equation(
        collected_equations: CollectedEquations,
        var_to_try: sp.Symbol,
        unknowns: List[sp.Symbol]) -> List[Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]]:
    """
    Find A sin(x) + B = 0
         C cos(x) + D = 0
    Equation pairs.
    :param collected_equations:
    :param var_to_try:
    :param unknowns:
    :return: (A, B, C, D) tuple list
    """
    one_unknown_equations = collected_equations.one_unknown_equations
    two_unknown_equations = collected_equations.two_unknown_equations
    x = var_to_try
    sin_equations: List[sp.Expr] = list()
    cos_equations: List[sp.Expr] = list()
    for expr in one_unknown_equations + two_unknown_equations:
        rhs_minus_lhs = expr.rhs - expr.lhs
        if not rhs_minus_lhs.has(x):
            continue

        # Should be captured by sin and cos solver
        if rhs_minus_lhs.has(sp.sin(x)) and rhs_minus_lhs.has(sp.cos(x)):
            continue

        if rhs_minus_lhs.has(sp.sin(x)):
            sin_equations.append(rhs_minus_lhs)
        if rhs_minus_lhs.has(sp.cos(x)):
            cos_equations.append(rhs_minus_lhs)

    terms = [sp.sin(x), sp.cos(x)]
    Aw = sp.Wild('Aw')
    Bw = sp.Wild('Bw')
    Cw = sp.Wild('Cw')
    Dw = sp.Wild('Dw')
    abcd_list = list()
    for sin_expr in sin_equations:
        sin_expr_collected = sin_expr.collect(terms)
        d_sin = sin_expr_collected.match(Aw * sp.sin(x) + Bw)
        assert d_sin is not None
        for cos_expr in cos_equations:
            cos_expr_collected = cos_expr.collect(terms)
            d_cos = cos_expr_collected.match(Cw * sp.cos(x) + Dw)
            assert d_cos is not None
            ratio = d_sin[Aw] / d_cos[Cw]

            too_many_unknowns = False
            if count_unknowns_expr(ratio, unknowns) > 0 \
                    or count_unknowns_expr(d_sin[Bw], unknowns) > 0 \
                    or count_unknowns_expr(d_cos[Dw], unknowns) > 0:
                too_many_unknowns = True

            if not too_many_unknowns:
                abcd = (d_sin[Aw], d_sin[Bw], d_cos[Cw], d_cos[Dw])
                abcd_list.append(abcd)

    return abcd_list


def tangent_try_solve(
        collected_equations: CollectedEquations,
        var_to_try: sp.Symbol,
        unknowns: List[sp.Symbol]) -> List[VariableSolution]:
    abcd_list = find_equation(collected_equations, var_to_try, unknowns)
    solution_list = list()
    for abcd in abcd_list:
        A, B, C, D = abcd
        x = var_to_try
        sin_expr = A * sp.sin(x) + B
        cos_expr = C * sp.cos(x) + D
        ratio = A / C
        if count_unknowns_expr(A, unknowns) == 0:
            solution = sp.atan2(- B / A, - D / C)
            validity_checker = (sp.Abs(A) >= fk_equations.zero_tolerance) | \
                               (sp.Abs(B) >= fk_equations.zero_tolerance) | \
                               (sp.Abs(C) >= fk_equations.zero_tolerance) | \
                               (sp.Abs(D) >= fk_equations.zero_tolerance)
            solution_entry = VariableSolution.make_explicit_solution(
                solved_variable=x,
                solutions=[solution],
                solution_method=SolutionMethod.Tangent.name,
                solve_from_equations=[ScalarEquation(sp.S.Zero, sin_expr), ScalarEquation(sp.S.Zero, cos_expr)],
                argument_valid_checkers=[validity_checker],
                degenerate_record=SolutionDegenerateRecord.record_all_equations([A * C]))
            solution_list.append(solution_entry)
        else:
            sol1 = sp.atan2(B / ratio, D)
            sol2 = sp.atan2(-B / ratio, -D)
            validity_checker = \
                (sp.Abs(D) >= fk_equations.zero_tolerance) | (sp.Abs(B / ratio) >= fk_equations.zero_tolerance)
            solution_entry = VariableSolution.make_explicit_solution(
                solved_variable=x,
                solutions=[sol1, sol2],
                solution_method=SolutionMethod.Tangent.name,
                solve_from_equations=[ScalarEquation(sp.S.Zero, sin_expr), ScalarEquation(sp.S.Zero, cos_expr)],
                argument_valid_checkers=[validity_checker, validity_checker],
                degenerate_record=SolutionDegenerateRecord.record_all_equations([A * C]))
            solution_list.append(solution_entry)

    # Ok
    return solution_list


class UnaryTangentSolver(UnaryVariableSolver):

    def try_solve(self, collected_equations: CollectedEquations,
                  var_to_try: sp.Symbol,
                  unknowns: List[sp.Symbol]) -> List[VariableSolution]:
        solution_list = tangent_try_solve(collected_equations, var_to_try, unknowns)
        return solution_list


# Test code
def test_solver():
    th_2, th_3, l_1, l_3 = sp.symbols('th_2 th_3 l_1  l_3')
    r_22, r_23 = sp.symbols('r_22 r_23')
    equations = CollectedEquations([], [], [])
    equations.one_unknown_equations.append(ScalarEquation(r_22, l_1 * sp.sin(th_2) + 15))
    equations.one_unknown_equations.append(ScalarEquation(r_23, l_3 * sp.cos(th_2) + 90))
    equations.one_unknown_equations.append(ScalarEquation(r_22, l_1 * sp.sin(th_2) * sp.sin(th_3) + 15))
    equations.one_unknown_equations.append(ScalarEquation(r_23, l_3 * sp.sin(th_2) * sp.cos(th_3) + 90))
    solver = UnaryTangentSolver()
    sol = solver.try_solve(equations, th_3, [th_2, th_3])
    assert len(sol) > 0
    print(sol[0].explicit_solutions)


if __name__ == '__main__':
    test_solver()
