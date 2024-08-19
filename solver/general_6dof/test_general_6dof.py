import random
import unittest
import numpy as np
import yaml
import os
import sympy as sp
from fk import robots
from fk.fk_equations import DHEntry, ik_target_4x4
from fk.robots import RobotDescription
from solver.equation_types import ScalarEquation
from solver.equation_utils import cast_expr_to_float, default_unknowns_with_offset
import solver.general_6dof.general_closure_equation as closure
import solver.general_6dof.dh_utils as dh_utils
from solver.general_6dof.reduce_input import build_reduce_inputs
from solver.general_6dof.numerical_reduce_closure_equation import factor_lhs_matrix, \
    numerical_reduce, sincos_coefficient_to_tanhalf_coefficient, generate_numerical_reduce_input
from utility import symbolic_utils
from typing import Tuple


def gmf_arc_mate_robot() -> RobotDescription:
    n_dofs = 6
    robot = RobotDescription("GMF_arc_mate")
    robot.unknowns = default_unknowns_with_offset(n_dofs, offset=1)
    d_1, d_3, d_4, d_5 = sp.symbols('d_1 d_3 d_4 d_5')
    a_1, a_2, a_3 = sp.symbols('a_1 a_2 a_3')
    dh_0 = DHEntry(sp.pi / 2, a_1, d_1, robot.unknowns[0].symbol)
    dh_1 = DHEntry(        0, a_2,   0, robot.unknowns[1].symbol)
    dh_2 = DHEntry(sp.pi / 2, a_3, d_3, robot.unknowns[2].symbol)
    dh_3 = DHEntry(sp.pi / 2,   0, d_4, robot.unknowns[3].symbol)
    dh_4 = DHEntry(sp.pi / 2,   0, d_5, robot.unknowns[4].symbol)
    dh_5 = DHEntry(        0,   0,   0, robot.unknowns[5].symbol)
    robot.dh_params = [dh_0, dh_1, dh_2, dh_3, dh_4, dh_5]
    robot.symbolic_parameters = {a_1, a_2, a_3, d_1, d_3, d_4, d_5}
    robot.parameters_value = {
        a_1: 0.2, a_2: 0.6, a_3: 0.13,
        d_1: 0.81, d_3: 0.03, d_4: 0.55, d_5: 0.1}
    return robot


class TestGeneral6DofSolver(unittest.TestCase):

    def setUp(self) -> None:
        self._robot = gmf_arc_mate_robot()
        self._revolute_vars = dh_utils.RevoluteVariable.convert_from_robot_unknowns(self._robot)
        self._test_cases = dh_utils.generate_classic_dh_numerical_test(
            self._robot.dh_params,
            self._revolute_vars,
            self._robot.parameters_value,
            n_test_cases=10)

    def test_solve(self):
        dh_params = self._robot.dh_params
        revolute_vars = self._revolute_vars
        A1v, A1s = dh_utils.reflected_variable_and_structure_transform(dh_params[0])
        A1 = A1v * A1s
        A2v, A2s = dh_utils.reflected_variable_and_structure_transform(dh_params[1])
        A2 = A2v * A2s
        A3v, A3s = dh_utils.reflected_variable_and_structure_transform(dh_params[2])
        A3 = A3v * A3s
        A4v, A4s = dh_utils.reflected_variable_and_structure_transform(dh_params[3])
        A4 = A4v * A4s
        A5v, A5s = dh_utils.reflected_variable_and_structure_transform(dh_params[4])
        A5 = A5v * A5s
        A6v, A6s = dh_utils.reflected_variable_and_structure_transform(dh_params[5])
        A6 = A6v * A6s

        inv_A1 = symbolic_utils.inverse_transform(A1)
        inv_A6 = symbolic_utils.inverse_transform(A6)
        inv_A2v = sp.transpose(A2v)
        lhs_matrix = A2s * A3 * A4 * A5
        rhs_matrix = inv_A2v * inv_A1 * ik_target_4x4() * inv_A6
        var2remove = self._revolute_vars[5]
        rhs_gen = None

        # Step 1: build the closure equation
        raw_closure = closure.closure_equation_from_matrix(
            lhs_matrix, rhs_matrix, revolute_vars, var2remove)
        self.assertTrue(raw_closure is not None)

        # Logging
        print('The equations')
        for equ in raw_closure.scalar_equations:
            print(equ.lhs, ' == ', equ.rhs)
        print([elem.variable_symbol for elem in raw_closure.lhs_unknowns])
        print([elem.variable_symbol for elem in raw_closure.rhs_unknowns])

        # Step 2: build the matrix equation
        matrix_equation = closure.closure_equations_matrix_form(raw_closure, rhs_gen=rhs_gen)

        # Logging
        print('LHS matrix in matrix equation')
        dh_utils.print_sp_matrix_by_row(matrix_equation.lhs_matrix)
        print('LHS terms in matrix equation')
        dh_utils.print_sp_matrix_by_row(matrix_equation.lhs_terms)
        print('RHS matrix in matrix equation')
        dh_utils.print_sp_matrix_by_row(matrix_equation.rhs_matrix)
        print('RHS terms in matrix equation')
        dh_utils.print_sp_matrix_by_row(matrix_equation.rhs_terms)

        # Step 3: reduce the equation
        reduce_output = closure.reduce_matrix_equation(matrix_equation)
        self.assertTrue(reduce_output is not None)
        reduced_scalar_equations, A_to_reduce = reduce_output
        unknown_in_matrix = raw_closure.lhs_unknowns[0]
        coefficient_matrix, coefficient_matrix_tan_half = closure.reduced_expr_as_matrix_form(
            raw_closure, matrix_equation, reduced_scalar_equations, unknown_in_matrix)
        A_sin, A_cos, C_const, nonlinear_coefficient = closure.compute_dialytical_coefficient_matrix(
            coefficient_matrix_tan_half, unknown_in_matrix)

        # Logging
        print('A_sin')
        dh_utils.print_sp_matrix_by_row(A_sin)
        print('A_cos')
        dh_utils.print_sp_matrix_by_row(A_cos)
        print('C_const')
        dh_utils.print_sp_matrix_by_row(C_const)

        # Numerical test
        for test_case_i in self._test_cases:
            lhs_matrix_value = matrix_equation.lhs_matrix.subs(test_case_i)
            rhs_matrix_value = matrix_equation.rhs_matrix.subs(test_case_i)
            lhs_term_value = matrix_equation.lhs_terms.subs(test_case_i)
            rhs_term_value = matrix_equation.rhs_terms.subs(test_case_i)
            residual_value = lhs_matrix_value * lhs_term_value - rhs_matrix_value * rhs_term_value
            for k in range(residual_value.shape[0]):
                residual_value_k = residual_value[k]
                if abs(residual_value_k) > 1e-10:
                    print('Wrong residual', residual_value_k)

            for equation in matrix_equation.ignored_coefficient + nonlinear_coefficient:
                equation_value = equation.subs(test_case_i)
                float_value = cast_expr_to_float(equation_value)
                assert float_value is not None
                if abs(float_value) > 1e-10:
                    print('Falsely ignore a non-zero equation ', equation)

            for equation in reduced_scalar_equations:
                equation_value = equation.subs(test_case_i)
                float_value = cast_expr_to_float(equation_value)
                assert float_value is not None
                if abs(float_value) > 1e-10:
                    print('Wrong equation ', equation)

            # Test the abc matrix
            A_sin_value = A_sin.subs(test_case_i)
            A_cos_value = A_cos.subs(test_case_i)
            C_const_value = C_const.subs(test_case_i)
            var_solution = test_case_i[unknown_in_matrix.variable_symbol]
            coefficient_matrix = A_sin_value * np.sin(var_solution) + \
                                 A_cos_value * np.cos(var_solution) + \
                                 C_const_value
            det_value = coefficient_matrix.det()
            print('Determinant value on solution: ', det_value)
            var_solution += 0.5
            coefficient_matrix = A_sin_value * np.sin(var_solution) + \
                                 A_cos_value * np.cos(var_solution) + \
                                 C_const_value
            det_value = coefficient_matrix.det()
            print('Determinant value on Non-solution: ', det_value)

    def test_numerical_reduce(self):
        revolute_vars = self._revolute_vars
        dh_params = self._robot.dh_params
        test_cases = self._test_cases

        A1v, A1s = dh_utils.reflected_variable_and_structure_transform(dh_params[0])
        A1 = A1v * A1s
        A2v, A2s = dh_utils.reflected_variable_and_structure_transform(dh_params[1])
        A2 = A2v * A2s
        A3v, A3s = dh_utils.reflected_variable_and_structure_transform(dh_params[2])
        A3 = A3v * A3s
        A4v, A4s = dh_utils.reflected_variable_and_structure_transform(dh_params[3])
        A4 = A4v * A4s
        A5v, A5s = dh_utils.reflected_variable_and_structure_transform(dh_params[4])
        A5 = A5v * A5s
        A6v, A6s = dh_utils.reflected_variable_and_structure_transform(dh_params[5])
        A6 = A6v * A6s

        inv_A1 = symbolic_utils.inverse_transform(A1)
        inv_A6 = symbolic_utils.inverse_transform(A6)
        inv_A2v = sp.transpose(A2v)
        lhs_matrix = A2s * A3 * A4 * A5
        rhs_matrix = inv_A2v * inv_A1 * ik_target_4x4() * inv_A6
        var2remove = revolute_vars[5]

        # Step 1: build the closure equation
        raw_closure = closure.closure_equation_from_matrix(
            lhs_matrix, rhs_matrix, revolute_vars, var2remove)
        assert (raw_closure is not None)
        matrix_equation = closure.closure_equations_matrix_form(raw_closure)

        # Factor the lhs
        var_in_lhs_matrix = raw_closure.lhs_unknowns[0]
        A_sin, A_cos, C_const, nonlinear_coefficient = factor_lhs_matrix(matrix_equation, var_in_lhs_matrix)

        def matrix_to_value(sp_matrix: sp.Matrix, test_case_in) -> np.ndarray:
            sp_mat_subst = sp_matrix.subs(test_case_in)
            return np.array(sp_mat_subst).astype(np.float64)

        def compute_det(
                tau_sin_tanhalf_in,
                tau_cos_tanhalf_in,
                tau_const_tanhalf_in,
                var_solution: float):
            coefficient_6x9 = tau_sin_tanhalf_in * np.sin(var_solution) + \
                              tau_cos_tanhalf_in * np.cos(var_solution) + \
                              tau_const_tanhalf_in
            coefficient_12x12 = np.zeros(shape=(12, 12))
            coefficient_12x12[0:6, 0:9] = coefficient_6x9
            coefficient_12x12[6:12, 3:12] = coefficient_6x9
            return np.linalg.det(coefficient_12x12)

        # For each test case
        for test_case_i in test_cases:
            A_sin_value = matrix_to_value(A_sin, test_case_i)
            A_cos_value = matrix_to_value(A_cos, test_case_i)
            C_const_value = matrix_to_value(C_const, test_case_i)
            N_rhs_value = matrix_to_value(matrix_equation.rhs_matrix, test_case_i)

            # Do numerical reduce
            only_two_lines_from = [2, 5, 6, 7, 10, 13]
            selected_lines = [0, 1, 2, 3, 4, 5, 8, 9]
            tau_sin, tau_cos, tau_const = numerical_reduce(
                A_sin_value, A_cos_value, C_const_value,
                N_rhs_value,
                selected_lines
            )

            # Into tanhalf
            tau_sin_tanhalf = sincos_coefficient_to_tanhalf_coefficient(tau_sin)
            tau_cos_tanhalf = sincos_coefficient_to_tanhalf_coefficient(tau_cos)
            tau_const_tanhalf = sincos_coefficient_to_tanhalf_coefficient(tau_const)

            var_in_lhs_matrix_value = float(test_case_i[var_in_lhs_matrix.variable_symbol])
            print('Det value on solution ',
                  compute_det(tau_sin_tanhalf, tau_cos_tanhalf, tau_const_tanhalf, var_in_lhs_matrix_value))
            print('Det value on non-solution ',
                  compute_det(tau_sin_tanhalf, tau_cos_tanhalf, tau_const_tanhalf, var_in_lhs_matrix_value + 0.1))


if __name__ == '__main__':
    pass
