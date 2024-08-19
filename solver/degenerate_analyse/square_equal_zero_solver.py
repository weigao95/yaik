from solver.equation_types import ScalarEquation
from solver.equation_utils import CollectedEquations, count_unknowns_expr, cast_expr_to_float
from solver.solved_variable import VariableSolution, SolutionMethod, SolutionDegenerateRecord
from solver.unary_solvers.unary_variable_solver import UnaryVariableSolver
from typing import List, Dict, Tuple, Optional
import sympy as sp


def find_equation(collected_equations: CollectedEquations, x: sp.Symbol, unknowns: List[sp.Symbol],
                  is_sin_expr: bool) -> bool:
    """
    Find sin(x)^2 = 0 or cos(x)^2 = 0, this only used in degenerate analyze
    Equation pairs.
    :param collected_equations:
    :param x:
    :param unknowns:
    :param is_sin_expr:
    :return: (A, B) tuple
    """
    terms = [sp.sin(x), sp.cos(x)]
    Aw = sp.Wild('Aw')
    for expr in collected_equations.one_unknown_equations:
        rhs_minus_lhs = expr.rhs - expr.lhs
        if not rhs_minus_lhs.has(x):
            continue

        # Should not be that complex
        if rhs_minus_lhs.count_ops() > 7:
            continue

        # Should be captured by sin and cos solver
        if is_sin_expr and (not rhs_minus_lhs.has(sp.sin(x))):
            continue
        if (not is_sin_expr) and (not rhs_minus_lhs.has(sp.cos(x))):
            continue

        expr_collected = rhs_minus_lhs.collect(terms)
        d_map: Optional[Dict[sp.Wild, sp.Expr]] = None
        if is_sin_expr:
            d_map = expr_collected.match(Aw * sp.sin(x) * sp.sin(x))
        else:
            d_map = expr_collected.match(Aw * sp.cos(x) * sp.cos(x))

        # Check d_map
        if d_map is None:
            continue

        assert d_map is not None
        A = d_map[Aw]
        assert A is not None
        A_float = cast_expr_to_float(A)
        if A_float is None or (abs(A_float) < 1e-6):
            continue
        else:
            return True

    # OK, solution finished
    return False


def sin_or_cos_square_equal_zero_try_solve(
        collected_equations: CollectedEquations,
        var_to_try: sp.Symbol,
        unknowns: List[sp.Symbol],
        is_sin_solver: bool) -> List[VariableSolution]:
    find_square_expr = find_equation(collected_equations, var_to_try, unknowns, is_sin_solver)
    if find_square_expr:
        solution_list = list()
        if is_sin_solver:
            solution_entry = VariableSolution.make_explicit_solution(
                solved_variable=var_to_try,
                solutions=[sp.S.Zero, sp.pi],
                solution_method=SolutionMethod.ArcSin.name,
                solve_from_equations=[ScalarEquation(sp.S.Zero, sp.sin(var_to_try) * sp.sin(var_to_try))],
                argument_valid_checkers=[sp.S.BooleanTrue, sp.S.BooleanTrue],
                degenerate_record=SolutionDegenerateRecord.record_always_non_degenerate())
            solution_list.append(solution_entry)
        else:
            solution_entry = VariableSolution.make_explicit_solution(
                solved_variable=var_to_try,
                solutions=[- sp.pi / 2, sp.pi / 2],
                solution_method=SolutionMethod.ArcCos.name,
                solve_from_equations=[ScalarEquation(sp.S.Zero, sp.cos(var_to_try) * sp.cos(var_to_try))],
                argument_valid_checkers=[sp.S.BooleanTrue, sp.S.BooleanTrue],
                degenerate_record=SolutionDegenerateRecord.record_always_non_degenerate())
            solution_list.append(solution_entry)
        return solution_list
    else:
        return list()


# The solver interface
class UnarySinSquareEqualZeroSolver(UnaryVariableSolver):

    def try_solve(self,
                  collected_equations: CollectedEquations,
                  var_to_try: sp.Symbol,
                  unknowns: List[sp.Symbol]) -> List[VariableSolution]:
        solutions = sin_or_cos_square_equal_zero_try_solve(collected_equations, var_to_try, unknowns, True)
        return solutions


class UnaryCosSquareEqualZeroSolver(UnaryVariableSolver):

    def try_solve(self,
                  collected_equations: CollectedEquations,
                  var_to_try: sp.Symbol,
                  unknowns: List[sp.Symbol]) -> List[VariableSolution]:
        solutions = sin_or_cos_square_equal_zero_try_solve(collected_equations, var_to_try, unknowns, False)
        return solutions


# debug code
def test_square_equal_zero():
    x = sp.Symbol('x')
    equ_1 = ScalarEquation(sp.S.Zero, sp.cos(x) * sp.cos(x))
    collected_equs = CollectedEquations([equ_1], [], [])
    solver = UnaryCosSquareEqualZeroSolver()
    solution = solver.try_solve(collected_equs, x, [x])
    print(len(solution))


if __name__ == '__main__':
    test_square_equal_zero()
