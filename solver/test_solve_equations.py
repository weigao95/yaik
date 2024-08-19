from solver.equation_types import ScalarEquation, Unknown, NumericalAnalyseContext
import sympy as sp
import solver.equation_utils as equation_utils
from solver.solved_variable import VariableSolution, SolutionMethod, SolutionDegenerateRecord
from solver.solve_equations import solve_equations, EquationInput
from typing import List
import unittest


class TestSolveEquations(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_asin_or_tangent(self):
        # The symbols
        th_0, th_1th_2_soa, th_3, th_4 = sp.symbols('th_0 th_1th_2_soa th_3 th_4')
        r_13, r_23, r_33 = sp.symbols('r_13 r_23 r_33')

        # Make the equations
        equation_0 = r_13 * sp.sin(th_0) - r_23 * sp.cos(th_0) - sp.sin(th_3) * sp.sin(th_4)
        equation_1 = -r_13 * sp.cos(th_0) * sp.cos(th_1th_2_soa) - r_23 * sp.sin(th_0) * sp.cos(th_1th_2_soa) + \
                     r_33 * sp.sin(th_1th_2_soa) + sp.sin(th_4) * sp.cos(th_3)
        scalar_equation_0 = ScalarEquation(sp.S.Zero, equation_0)
        scalar_equation_1 = ScalarEquation(sp.S.Zero, equation_1)
        equation_input = EquationInput([scalar_equation_0, scalar_equation_1], list())

        # The unknown
        th_3_unknown = Unknown(th_3)
        all_parameters = {th_0, th_1th_2_soa, th_3, th_4, r_13, r_23, r_33}
        solution: List[VariableSolution] = solve_equations(
            equation_input, [th_3_unknown], all_parameters, NumericalAnalyseContext())
        self.assertTrue(len(solution) == 1)
        self.assertTrue(solution[0].is_explicit_solution)
        self.assertTrue(solution[0].solution_method == SolutionMethod.Tangent.name)
        self.assertTrue(solution[0].num_solutions() == 1)

    def test_should_no_analyze(self):
        # The symbols
        th_0, th_1, th_2, th_3, th_4 = sp.symbols('th_0 th_1 th_2 th_3 th_4')
        r_13, r_23, r_33 = sp.symbols('r_13 r_23 r_33')

        # Make the equations
        equation_0 = -r_13*sp.sin(th_0) + r_23*sp.cos(th_0) + sp.sin(th_3)*sp.sin(th_4)
        equation_1 = -r_13*sp.cos(th_0)*sp.cos(th_1) - r_23*sp.sin(th_0)*sp.cos(th_1) + \
                     r_33*sp.sin(th_1) - sp.sin(th_2)*sp.cos(th_4) + sp.sin(th_4)*sp.cos(th_2)*sp.cos(th_3)
        scalar_equation_0 = ScalarEquation(sp.S.Zero, equation_0)
        scalar_equation_1 = ScalarEquation(sp.S.Zero, equation_1)
        equation_input = EquationInput([scalar_equation_0, scalar_equation_1], list())

        # The unknown
        th_3_unknown = Unknown(th_3)
        all_parameters = {th_0, th_1, th_2, th_3, th_4, r_13, r_23, r_33}
        solution: List[VariableSolution] = solve_equations(
            equation_input, [th_3_unknown], all_parameters, NumericalAnalyseContext())
        self.assertTrue(len(solution) == 1)
        self.assertTrue(solution[0].is_explicit_solution)
        self.assertTrue(solution[0].num_solutions() == 2)

    def test_cast_float(self):
        # The pi exprs
        pi_div_2 = 1.234 + sp.pi / 2
        pi_div_2_float = equation_utils.cast_expr_to_float(pi_div_2)
        self.assertTrue(pi_div_2_float is not None)
        print(pi_div_2_float)

        # The symbols
        th_0 = sp.Symbol('th_0')
        casted_float = equation_utils.cast_expr_to_float(th_0)
        self.assertTrue(casted_float is None)


if __name__ == '__main__':
    pass
