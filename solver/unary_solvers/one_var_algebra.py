from solver.equation_types import ScalarEquation
from solver.equation_utils import CollectedEquations
from solver.solved_variable import VariableSolution, SolutionMethod, SolutionDegenerateRecord
from solver.unary_solvers.unary_variable_solver import UnaryVariableSolver
from solver.equation_utils import count_unknowns_expr
import fk.fk_equations as fk_equations
from typing import List, Optional
import sympy as sp


# Find equation in the form of
# A theta = b
# This is theta, not sin/cos theta
def find_equation(collected_equations: CollectedEquations, var_to_try: sp.Symbol) -> List[ScalarEquation]:
    one_unknown_equations = collected_equations.one_unknown_equations
    x = var_to_try
    result = list()
    for expr_i in one_unknown_equations:
        # Not is expr_i
        if (not expr_i.lhs.has(x)) and (not expr_i.rhs.has(x)):
            continue

        # Should be handled by other unary_solvers
        if (expr_i.lhs.has(sp.sin(x)) or expr_i.lhs.has(sp.cos(x))) or expr_i.rhs.has(sp.sin(x)) or expr_i.rhs.has(
                sp.cos(x)):
            continue

        # One var, no sin/cos
        if expr_i.lhs.has(x) or expr_i.rhs.has(x):
            l_minus_r: sp.Expr = expr_i.lhs - expr_i.rhs
            l_minus_r = l_minus_r.expand()
            l_minus_r = l_minus_r.collect(x)
            result.append(ScalarEquation(sp.S.Zero, l_minus_r, expr_i.equation_type))

    # OK
    return result


def one_variable_algebra_try_solve(
        collected_equations: CollectedEquations,
        var_to_try: sp.Symbol,
        unknowns: List[sp.Symbol]) -> List[VariableSolution]:
    equations = find_equation(collected_equations, var_to_try)
    if len(equations) == 0:
        return list()

    # Try it
    Aw = sp.Wild("Aw")
    Bw = sp.Wild("Bw")
    solution_candidate: List[sp.Expr] = list()
    solution_non_zero_assumption: List[sp.Expr] = list()
    equation_to_solve: List[ScalarEquation] = list()
    equation_A: List[sp.Expr] = list()
    for equ in equations:
        d = equ.rhs.match(Aw * var_to_try + Bw)
        A: sp.Expr = d[Aw]
        B: sp.Expr = d[Bw]
        if B.has(var_to_try) or A.has(var_to_try):
            continue
        if count_unknowns_expr(B, unknowns) > 0 or count_unknowns_expr(A, unknowns) > 0:
            continue
        solution = - B / A
        solution_candidate.append(solution)
        solution_non_zero_assumption.append(A)
        equation_to_solve.append(equ)
        equation_A.append(A)

    # Select one that is easy
    if len(solution_candidate) == 0:
        return list()

    # we might need selection here, too
    solution_list = list()
    for i in range(len(solution_candidate)):
        solution_i = solution_candidate[i]
        equation_to_solve_i = equation_to_solve[i]
        assumption_i = solution_non_zero_assumption[i]
        A = equation_A[i]
        validity_checker = sp.Abs(A) >= fk_equations.zero_tolerance
        solution_entry = VariableSolution.make_explicit_solution(
            solved_variable=var_to_try,
            solutions=[solution_i],
            solution_method=SolutionMethod.OneVariableAlgebra.name,
            solve_from_equations=[equation_to_solve_i],
            argument_valid_checkers=[validity_checker],
            degenerate_record=SolutionDegenerateRecord.record_all_equations([assumption_i])
        )
        solution_list.append(solution_entry)

    # OK
    return solution_list


class UnaryOneVariableAlgebraSolver(UnaryVariableSolver):

    def try_solve(self,
                  collected_equations: CollectedEquations,
                  var_to_try: sp.Symbol,
                  unknowns: List[sp.Symbol]) -> List[VariableSolution]:
        solutions = one_variable_algebra_try_solve(collected_equations, var_to_try, unknowns)
        return solutions


# Test code
def test_algebra_solve():
    import solver.equation_utils as equation_utils
    Td = fk_equations.ik_target_4x4()
    Ts = sp.zeros(4)

    # Make the equations
    l_1, l_2, l_3 = sp.symbols('l_1 l_2 l_3')
    th_1, th_2, th_3, th_5 = sp.symbols('th_1 th_2 th_3 th_5')
    d_1 = sp.symbols('d_1')
    Ts[0, 1] = th_2 + l_1 * l_2
    Ts[0, 2] = d_1 * l_3 + l_1
    Ts[1, 1] = th_5 + th_2 * l_1
    Ts[1, 2] = 0
    Ts[2, 1] = th_2 * th_3 + l_1
    Ts[2, 2] = sp.sin(th_3)
    Ts[3, 1] = sp.sin(th_1 + th_2)
    Ts[3, 2] = sp.sin(th_1 + th_2 + th_3)

    # OK
    unknowns = [d_1, th_1, th_2, th_3, th_5]
    matrix_equation = fk_equations.MatrixEquation(Td, Ts)
    collected_equs = equation_utils.collect_equations([matrix_equation], unknowns)

    # First solve for d_1
    d_1_sol = one_variable_algebra_try_solve(collected_equs, d_1, [d_1, th_2, th_3])
    assert len(d_1_sol) > 0
    d_1_sol = d_1_sol[0]
    assert sp.simplify(d_1_sol.explicit_solutions[0] - (Td[0, 2] - l_1) / l_3) == 0

    # Next is th_2
    th_2_sol = one_variable_algebra_try_solve(collected_equs, th_2, [d_1, th_2, th_3])
    assert len(th_2_sol) > 0
    th_2_sol = th_2_sol[0]
    assert sp.simplify(th_2_sol.explicit_solutions[0] - (Td[0, 1] - l_1 * l_2)) == 0

    # Next is th_2
    th_3_sol = one_variable_algebra_try_solve(collected_equs, th_3, [d_1, th_2, th_3])
    assert len(th_3_sol) == 0


if __name__ == '__main__':
    test_algebra_solve()
