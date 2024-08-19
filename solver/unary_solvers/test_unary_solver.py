from solver.solved_variable import VariableSolution
from solver.equation_utils import CollectedEquations, ScalarEquation, cast_expr_to_float
from solver.unary_solvers.tangent_solver import UnaryVariableSolver, UnaryTangentSolver
from solver.unary_solvers.one_var_algebra import UnaryOneVariableAlgebraSolver
from solver.unary_solvers.sin_and_cos import UnarySinAndCosSolver
from solver.unary_solvers.linear_sin_cos_type_1 import UnaryLinearSolverType_1
from solver.unary_solvers.asin_or_acos import UnaryArcSinSolver, UnaryArcCosSolver
import sympy as sp
import numpy as np
from typing import List
import unittest


def solution_match_numerical(
        solver: UnaryVariableSolver,
        equation_to_test: List[ScalarEquation],
        x: sp.Symbol,
        other_parameters: List[sp.Symbol],
        parameter_values: List[List[float]],
        tolerance: float = 1e-10) -> bool:
    collect_equations = CollectedEquations(equation_to_test, [], [])
    solution_list: List[VariableSolution] = solver.try_solve(collect_equations, x, [x])
    lhs_minus_rhs_list = list()
    for i in range(len(equation_to_test)):
        expr_i = equation_to_test[i].lhs - equation_to_test[i].rhs
        lhs_minus_rhs_list.append(expr_i)

    numerical_match = True
    for solution in solution_list:
        for i in range(len(parameter_values)):
            # Init parameter for all other values
            # Build the map
            value_i = parameter_values[i]
            assert len(value_i) >= len(other_parameters)
            subst_map_i = dict()
            for j in range(len(other_parameters)):
                subst_map_i[other_parameters[j]] = value_i[j]

            # For each solution
            explicit_solutions = solution.explicit_solutions
            for j in range(len(explicit_solutions)):
                sol_j = explicit_solutions[j]
                sol_j_with_value = sol_j.subs(subst_map_i)
                for k in range(len(equation_to_test)):
                    equation_k = lhs_minus_rhs_list[k]
                    equation_k_value = equation_k.subs(subst_map_i)
                    numerical_value = equation_k_value.subs(solution.solved_variable, sol_j_with_value)
                    numerical_value = cast_expr_to_float(numerical_value)
                    if numerical_value is None:
                        numerical_match = False
                    else:
                        # Not none, check the value
                        if abs(float(numerical_value)) > tolerance:
                            numerical_match = False
                        pass
    return numerical_match


class TestUnarySolverNumerical(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_algebra_solve(self):
        th_0, th_1, d_0, l_0 = sp.symbols('th_0 th_1 d_0 l_0')
        algebra_solver = UnaryOneVariableAlgebraSolver()
        equation_to_test = ScalarEquation(sp.sin(th_1) * l_0, d_0 * sp.sin(th_1))
        x = d_0
        other_symbols = [th_0, th_1, l_0]

        # Test values
        test_n = 100
        values_to_test = list()
        for i in range(test_n):
            rand_tmp = np.random.random(size=(3, ))
            th_0_v, th_1_v, l_0_v = rand_tmp[0], rand_tmp[1], rand_tmp[2]
            values_i = [th_0_v, th_1_v, l_0_v]
            values_to_test.append(values_i)

        # Test the solver
        ok = solution_match_numerical(algebra_solver, [equation_to_test], x, other_symbols, values_to_test)
        self.assertTrue(ok)

    def test_tangent_solve(self):
        th_2, l_1, l_2, l_3 = sp.symbols('th_2 l_1 l_2 l_3')
        r_22, r_23 = sp.symbols('r_22 r_23')
        equation_0 = ScalarEquation(r_22, l_1 * sp.sin(th_2) + l_3)
        equation_1 = ScalarEquation(r_23, l_2 * sp.cos(th_2) + l_1)
        solver_to_test = UnaryTangentSolver()
        x = th_2
        equations = [equation_0, equation_1]
        other_symbols = [l_1, l_2, l_3, r_22, r_23]
        values_to_test = list()
        test_n = 100
        for i in range(test_n):
            # This is overly constrained, thus we need to be careful with value generator
            # Generate l_1, th_2 and l_3
            r3_tmp = np.random.random(size=(4, ))
            l_1_v, l_2_v, l_3_v, th_2_v = r3_tmp[0], r3_tmp[1], r3_tmp[2], r3_tmp[3]
            r_22_v = l_1_v * np.sin(th_2_v) + l_3_v
            r_23_v = l_2_v * np.cos(th_2_v) + l_1_v
            sin_th_2 = (r_22_v - l_3_v) / l_1_v
            cos_th_2 = (r_23_v - l_1_v) / l_2_v
            value_i = [l_1_v, l_2_v, l_3_v, r_22_v, r_23_v, th_2_v, sin_th_2, cos_th_2]
            values_to_test.append(value_i)

        # Test the solver
        ok = solution_match_numerical(solver_to_test, equations, x, other_symbols, values_to_test)
        self.assertTrue(ok)

    def test_sin_and_cos_solve(self):
        th_2, l_1, l_2 = sp.symbols('th_2 l_1 l_2')
        r_22 = sp.symbols('r_22')
        equation_to_test = ScalarEquation(r_22, l_1 * sp.sin(th_2) + l_2 * sp.cos(th_2))
        solver_to_test = UnarySinAndCosSolver()
        x = th_2
        equations = [equation_to_test]
        other_symbols = [l_1, l_2, r_22]
        values_to_test = list()
        test_n = 100
        for i in range(test_n):
            # This is overly constrained, thus we need to be careful with value generator
            # Generate l_1, th_2 and l_3
            r3_tmp = np.random.random(size=(3, ))
            l_1_v, l_2_v, th_2_v = r3_tmp[0], r3_tmp[1], r3_tmp[2]
            r_22_v = l_1_v * np.sin(th_2_v) + l_2_v * sp.cos(th_2_v)
            value_i = [l_1_v, l_2_v, r_22_v]
            values_to_test.append(value_i)

        # Test the solver
        ok = solution_match_numerical(solver_to_test, equations, x, other_symbols, values_to_test)
        self.assertTrue(ok)

    def test_arc_sin_solve(self):
        th_2, l_1, l_2 = sp.symbols('th_2 l_1 l_2')
        r_22 = sp.symbols('r_22')
        for is_sin in [True, False]:
            if is_sin:
                print('Test sin solver')
                equation = ScalarEquation(r_22, l_1 * sp.sin(th_2) + l_2)
                solver_to_test = UnaryArcSinSolver()
            else:
                print('Test cos solver')
                equation = ScalarEquation(r_22, l_1 * sp.cos(th_2) + l_2)
                solver_to_test = UnaryArcCosSolver()

            x = th_2
            equations = [equation]
            other_symbols = [l_1, l_2, r_22]
            values_to_test = list()
            test_n = 100
            for i in range(test_n):
                # This is overly constrained, thus we need to be careful with value generator
                # Generate l_1, th_2 and l_3
                r3_tmp = np.random.random(size=(3, ))
                l_1_v, l_2_v, th_2_v = r3_tmp[0], r3_tmp[1], r3_tmp[2]
                if is_sin:
                    r_22_v = l_1_v * np.sin(th_2_v) + l_2_v
                else:
                    r_22_v = l_1_v * np.cos(th_2_v) + l_2_v
                value_i = [l_1_v, l_2_v, r_22_v]
                values_to_test.append(value_i)

            # Test the solver
            ok = solution_match_numerical(solver_to_test, equations, x, other_symbols, values_to_test)
            self.assertTrue(ok)

    def test_linear_solve_type_1(self):
        Px, Py, Pz, th_1, th_23, th_3, a_3, a_2, d_4 = sp.symbols('Px Py Pz th_1 th_23 th_3 a_3 a_2 d_4')
        exp1 = Pz * sp.sin(th_23) + a_2 * sp.cos(th_3) + (-Px * sp.cos(th_1) - Py * sp.sin(th_1)) * sp.cos(th_23)
        exp2 = Pz * sp.cos(th_23) - a_2 * sp.sin(th_3) + (Px * sp.cos(th_1) + Py * sp.sin(th_1)) * sp.sin(th_23)
        e1 = ScalarEquation(a_3, exp1)
        e2 = ScalarEquation(d_4, exp2)
        equations = [e1, e2]
        solver_to_test = UnaryLinearSolverType_1()
        x = th_23
        other_symbols = [Px, Py, Pz, th_1, th_3, a_2, a_3, d_4]
        values_to_test = list()
        test_n = 100
        for i in range(test_n):
            rand_tmp = np.random.random(size=(len(other_symbols) - 2, ))
            subst_map_i = dict()
            for i in range(len(other_symbols) - 2):
                subst_map_i[other_symbols[i]] = rand_tmp[i]
            a_3_v = exp1.subs(subst_map_i)
            d_4_v = exp2.subs(subst_map_i)
            value_i = list()
            for i in range(len(rand_tmp)):
                value_i.append(rand_tmp[i])
            value_i.append(a_3_v)
            value_i.append(d_4_v)
            values_to_test.append(value_i)

        # Test the solver
        ok = solution_match_numerical(solver_to_test, equations, x, other_symbols, values_to_test)
        self.assertTrue(ok)


if __name__ == '__main__':
    pass
