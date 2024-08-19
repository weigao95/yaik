import sympy as sp
import numpy as np
from typing import List, Set, Dict
import unittest
import copy
from solver.equation_types import Unknown, UnknownType
from solver.solved_variable import SolutionMethod
from solver.polynomial_solver.groebner_reduce_sympy import groebner_reduce_sympy
from solver.polynomial_solver.verify_polynomial_solution import verify_polynomial_solution


def polynomial_numerical_match(
        polynomials: List[sp.Expr],
        unknowns: List[Unknown],
        parameters: Set[sp.Symbol],
        parameter_values_to_test: List[Dict[sp.Symbol, float]],
        use_sage: bool = True) -> bool:
    from solver.polynomial_solver.groebner_reduce_sage import groebner_reduce_sage
    value_match = True
    for to_cos in [False, True]:
        if use_sage:
            output = groebner_reduce_sage(polynomials, unknowns, parameters, to_cos)
        else:
            output = groebner_reduce_sympy(polynomials, unknowns, parameters, to_cos)
        if output is None:
            value_match = False
        # Send to verification
        all_matched, _ = verify_polynomial_solution(polynomials, output, parameter_values_to_test)
        value_match = value_match and all_matched

    return value_match


class TestPolynomialSolverNumerical(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_polynomial_0(self):
        th_0_unknown = Unknown(sp.Symbol('th_0'))
        th_0 = th_0_unknown.symbol
        l_0 = sp.Symbol('l_0')
        expr_0 = sp.sin(th_0) - sp.cos(th_0) ** 2
        poly_0 = expr_0 - l_0
        parameters = {th_0, l_0}
        values_to_test = list()
        test_n = 100
        for i in range(test_n):
            value_map_i = dict()
            value_map_i[th_0] = np.random.random(size=(1, ))[0]
            expr_0_value = expr_0.subs(value_map_i)
            value_map_i[l_0] = expr_0_value
            values_to_test.append(value_map_i)

        ok_sage = polynomial_numerical_match([poly_0], [th_0_unknown], parameters, values_to_test, True)
        ok_sympy = polynomial_numerical_match([poly_0], [th_0_unknown], parameters, values_to_test, False)
        self.assertTrue(ok_sage)
        self.assertTrue(ok_sympy)

    def test_polynomial_1(self):
        from fk.fk_equations import Px
        th_0_unknown = Unknown(sp.Symbol('th_0'))
        th_0 = th_0_unknown.symbol
        d_6 = sp.Symbol('d_6')
        r_11, r_12, r_13 = sp.symbols('r_11 r_12 r_13')
        l_1, l_3 = sp.symbols('l_1 l_3')
        expr_0 = - d_6 * r_13 - l_3 * (r_11 * sp.sin(th_0) + r_12 * sp.cos(th_0)) - l_1 * sp.cos(th_0)
        poly_0 = expr_0 - Px
        parameter_list = [th_0, d_6, r_11, r_12, r_13, l_1, l_3, Px]
        parameter_set = set()
        for elem in parameter_list:
            parameter_set.add(elem)

        # Build the test values
        values_to_test = list()
        test_n = 100
        for i in range(test_n):
            value_map_i = dict()
            rand_tmp = np.random.random(size=(len(parameter_list) - 1,))
            for k in range(rand_tmp.size):
                value_map_i[parameter_list[k]] = rand_tmp[k]
            Px_value = expr_0.subs(value_map_i)
            assert Px_value.is_Number
            value_map_i[Px] = Px_value
            values_to_test.append(value_map_i)

        ok_sage = polynomial_numerical_match([poly_0], [th_0_unknown], parameter_set, values_to_test, True)
        ok_sympy = polynomial_numerical_match([poly_0], [Unknown(th_0)], parameter_set, values_to_test, False)
        self.assertTrue(ok_sage)
        self.assertTrue(ok_sympy)

    def test_polynomial_2(self):
        from fk.fk_equations import inv_Px, inv_Py
        th_0 = sp.Symbol('th_0')
        r_11, r_12 = sp.symbols('r_11 r_12')
        r_21, r_22 = sp.symbols('r_21 r_22')
        r_31, r_32 = sp.symbols('r_31 r_32')
        l_1, l_2, l_3_square = sp.symbols('l_1 l_2 l_3_square')
        expr_0 = (inv_Px + l_1 * (r_11 * sp.cos(th_0) + r_31 * sp.sin(th_0)) - l_2 * r_21) ** 2 + (
                inv_Py + l_1 * (r_12 * sp.cos(th_0) + r_32 * sp.sin(th_0)) - l_2 * r_22) ** 2
        poly_0 = expr_0 - l_3_square
        parameter_list = [r_11, r_12, r_21, r_22, r_31, r_32, l_1, l_2, th_0, inv_Px, inv_Py, l_3_square]
        parameter_set = set()
        for elem in parameter_list:
            parameter_set.add(elem)

        # Build the test values
        values_to_test = list()
        test_n = 100
        for i in range(test_n):
            value_map_i = dict()
            rand_tmp = np.random.random(size=(len(parameter_list) - 1,))
            for k in range(rand_tmp.size):
                value_map_i[parameter_list[k]] = rand_tmp[k]
            # The last one of l3_square
            l_3_square_value = expr_0.subs(value_map_i)
            assert l_3_square_value.is_Number
            value_map_i[l_3_square] = float(l_3_square_value)
            values_to_test.append(value_map_i)

        ok_sage = polynomial_numerical_match([poly_0], [Unknown(th_0)], parameter_set, values_to_test, True)
        # sympy cannot solve this
        # ok_sympy = polynomial_numerical_match([poly_0], [Unknown(th_0)], parameter_set, values_to_test, False)
        self.assertTrue(ok_sage)
        # self.assertTrue(ok_sympy)

    def test_polynomial_3(self):
        from solver.equation_utils import default_unknowns
        from fk.fk_equations import Px, Py
        unknowns = default_unknowns(2)
        th_0 = unknowns[0].symbol
        th_4 = unknowns[1].symbol
        d_6 = sp.Symbol('d_6')
        r_11, r_12, r_13 = sp.symbols('r_11 r_12 r_13')
        r_31, r_32, r_33 = sp.symbols('r_31 r_32 r_33')
        l_1, l_3 = sp.symbols('l_1 l_3')
        expr_0 = - d_6 * r_13 - l_3 * (r_11 * sp.sin(th_4) + r_12 * sp.cos(th_4)) - l_1 * sp.cos(th_0)
        expr_1 = - d_6 * r_33 - l_3 * (r_31 * sp.sin(th_4) + r_32 * sp.cos(th_4)) - l_1 * sp.sin(th_0)
        poly_0 = expr_0 - Px
        poly_1 = expr_1 - Py
        parameter_list = [th_0, th_4, d_6, r_11, r_12, r_13, r_31, r_32, r_33, l_1, l_3, Px, Py]
        parameter_set = set()
        for elem in parameter_list:
            parameter_set.add(elem)

        # Build the test values
        values_to_test = list()
        test_n = 100
        for i in range(test_n):
            value_map_i = dict()
            rand_tmp = np.random.random(size=(len(parameter_list) - 2,))
            for k in range(rand_tmp.size):
                value_map_i[parameter_list[k]] = rand_tmp[k]
            Px_value = expr_0.subs(value_map_i)
            Py_value = expr_1.subs(value_map_i)
            assert Px_value.is_Number
            assert Py_value.is_Number
            value_map_i[Px] = Px_value
            value_map_i[Py] = Py_value
            values_to_test.append(value_map_i)

        # Run the code
        ok_0 = polynomial_numerical_match([poly_0, poly_1], [unknowns[0], unknowns[1]], parameter_set, values_to_test)
        ok_1 = polynomial_numerical_match([poly_0, poly_1], [unknowns[1], unknowns[0]], parameter_set, values_to_test)
        self.assertTrue(ok_0 and ok_1)

    def test_polynomial_4(self):
        from solver.equation_utils import default_unknowns
        from fk.fk_equations import Px, Py
        unknowns = default_unknowns(4)
        th_0 = unknowns[0].symbol
        th_3 = unknowns[1].symbol
        th_4 = unknowns[2].symbol
        th_5 = unknowns[3].symbol
        l_5 = sp.Symbol('l_5')
        r_11, r_12, r_13 = sp.symbols('r_11 r_12 r_13')
        r_21, r_22, r_23 = sp.symbols('r_21 r_22 r_23')
        expr_0 = Px*sp.sin(th_0) - Py*sp.cos(th_0) + l_5*sp.cos(th_3)
        expr_1 = r_11*sp.sin(th_0) - r_21*sp.cos(th_0) + sp.sin(th_3)*sp.cos(th_4)*sp.cos(th_5) + sp.sin(th_5)*sp.cos(th_3)
        expr_2 = r_12*sp.sin(th_0) - r_22*sp.cos(th_0) - sp.sin(th_3)*sp.sin(th_5)*sp.cos(th_4) + sp.cos(th_3)*sp.cos(th_5)
        expr_3 = r_13*sp.sin(th_0) - r_23*sp.cos(th_0) + sp.sin(th_3)*sp.sin(th_4)
        parameter_set = {Px, Py, th_0, th_3, th_4, th_5, l_5, r_11, r_12, r_13, r_21, r_22, r_23}
        polynomial_numerical_match([expr_0, expr_1, expr_2, expr_3], unknowns, parameter_set, list())

    def test_sympy_on_invalid_input(self):
        from fk.fk_equations import Py
        d_6, th_4 = sp.symbols('d_6 th_4')
        l_2, l_3 = sp.symbols('l_2 l_3')
        r_21, r_22, r_23 = sp.symbols('r_21 r_22 r_23')
        expr_0 = Py - d_6*r_23 + l_2 - l_3*(r_21*sp.sin(th_4) + r_22*sp.cos(th_4))
        expr_1 = -Py + d_6*r_23 - l_2 + l_3*(r_21*sp.sin(th_4) + r_22*sp.cos(th_4))
        expr_2 = Py - d_6 * r_23 + l_2 - l_3 * (r_21 * sp.sin(th_4) + r_22 * sp.cos(th_4))
        expr_3 = -Py + d_6*r_23 - l_2 + l_3*(r_21*sp.sin(th_4) + r_22*sp.cos(th_4))
        parameter_set = {th_4, d_6, l_2, l_3, r_21, r_22, r_23, Py}
        d_6_unknown = Unknown(d_6, UnknownType.Translational.name)
        th_4_unknown = Unknown(th_4, UnknownType.Revolute.name)
        output = groebner_reduce_sympy(
            [expr_0, expr_1, expr_2, expr_3], [d_6_unknown, th_4_unknown], parameter_set, False)
        self.assertTrue(output is None)


if __name__ == '__main__':
    pass
