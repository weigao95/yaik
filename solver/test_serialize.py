from fk.fk_equations import Px, Py, Pz, inv_Px, inv_Py, inv_Pz
from fk.robots import RobotDescription, puma_robot
from solver.equation_types import ScalarEquation, ScalarEquationType, Unknown
from solver.solved_variable import SolutionMethod, SolutionDegenerateRecord
from solver.solved_variable_impl import ExplicitVariableSolutionImpl, PolynomialVariableSolutionImpl, \
    solution_impl_to_dict, solution_impl_from_dict
import solver.equation_utils as equation_utils
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import unittest


class TestSerialize(unittest.TestCase):

    def setUp(self) -> None:
        th_0, th_1 = sp.symbols('th_0 th_1')
        r_11, r_12, r_13 = sp.symbols('r_11 r_12 r_13')
        l_1, l_2, l_3 = sp.symbols('l_1 l_2 l_3')
        expr_0 = Px - l_1 * r_13 - l_3 * (r_11 * sp.sin(th_1) + r_12 * sp.cos(th_1)) - l_1 * sp.cos(th_0)
        expr_1 = (inv_Px + l_1 * (r_11 * sp.cos(th_0) + r_12 * sp.sin(th_0)) - l_2 * r_12) ** 2 + inv_Py
        expr_2 = Py - l_2 * r_13 - l_3 * (r_12 * sp.sin(th_1) + r_11 * sp.cos(th_0)) + l_2

        # Collect the input
        self._thetas = [th_0, th_1]
        self._parameters = [r_11, r_12, r_13, l_1, l_2, l_3, Px, Py, Pz, inv_Px, inv_Py, inv_Pz]
        self._expr_to_test = [expr_0, expr_1, expr_2]

    def test_sympy_serialize_simple(self):
        for expr_i in self._expr_to_test:
            expr_i_str = str(expr_i)
            parsed_expr = parse_expr(expr_i_str)
            parsed_diff = expr_i - parsed_expr
            parsed_diff = parsed_diff.simplify()
            self.assertTrue(parsed_diff == sp.S.Zero)

    def test_scalar_equation(self):
        equ_0 = ScalarEquation(self._expr_to_test[0], self._expr_to_test[1], ScalarEquationType.SumOfSquare.name)
        equ_1 = ScalarEquation(self._expr_to_test[1], self._expr_to_test[2], ScalarEquationType.SumOfAngle.name)
        for equ in [equ_0, equ_1]:
            equ_dict = equ.to_dict()
            loaded_equ = ScalarEquation.load_equation_from_dict(equ_dict)
            # Check they are same
            self.assertTrue(loaded_equ.equation_type == equ.equation_type)
            self.assertTrue(loaded_equ.rhs - equ.rhs == sp.S.Zero)
            self.assertTrue(loaded_equ.lhs - equ.lhs == sp.S.Zero)

    def test_unknown(self):
        unknown_0 = Unknown(self._thetas[0])
        unknown_1 = Unknown(self._thetas[1])
        unknowns = [unknown_0, unknown_1]
        for unknown in unknowns:
            unknown_dict = unknown.to_dict()
            loaded_unknown = Unknown.load_unknown_from_dict(unknown_dict)
            self.assertTrue(unknown.symbol == loaded_unknown.symbol)
            self.assertTrue(abs(unknown.lower_bound - loaded_unknown.lower_bound) < 1e-10)
            self.assertTrue(abs(unknown.upper_bound - loaded_unknown.upper_bound) < 1e-10)
            self.assertTrue(len(unknown.degenerate_check_value) == len(loaded_unknown.degenerate_check_value))
            for i in range(len(unknown.degenerate_check_value)):
                unknown_expr_i = unknown.degenerate_check_value[i]
                loaded_expr_i = loaded_unknown.degenerate_check_value[i]
                self.assertTrue(loaded_expr_i - unknown_expr_i == sp.S.Zero)

    def test_solved_variable_explicit(self):
        th_0 = self._thetas[0]
        th_1 = self._thetas[1]
        solutions_to_test = [sp.sin(th_0) + self._parameters[0], sp.cos(th_0) - self._parameters[1]]
        degenerate_record = SolutionDegenerateRecord.record_variable_value({th_0: [sp.S.Zero, sp.pi]})
        solved_var = ExplicitVariableSolutionImpl.make_explicit_solution(
            th_1, solutions_to_test, SolutionMethod.OneVariableAlgebra.name,
            degenerate_record=degenerate_record, argument_valid_checkers=[sp.S.BooleanTrue, sp.S.BooleanTrue])
        solved_var_dict = solution_impl_to_dict(solved_var)
        parsed_var: ExplicitVariableSolutionImpl = solution_impl_from_dict(solved_var_dict)

        # Check the parsed expr
        self.assertTrue(parsed_var.is_explicit_solution)
        self.assertTrue(parsed_var.solved_variable == solved_var.solved_variable)
        self.assertTrue(parsed_var.solution_method == solved_var.solution_method)
        self.assertTrue(len(parsed_var.explicit_solutions) == 2)
        for i in range(len(solutions_to_test)):
            sol_diff = solutions_to_test[i] - parsed_var.explicit_solutions[i]
            self.assertTrue(sol_diff == sp.S.Zero)

        # Check the degenerate record
        self.assertTrue(parsed_var.degenerate_record.type == degenerate_record.type)
        self.assertTrue(len(parsed_var.degenerate_record.equations) == 0)
        self.assertTrue(len(parsed_var.degenerate_record.variable_value_map) == 1)
        for key in parsed_var.degenerate_record.variable_value_map:
            self.assertTrue(key == th_0)
            degenerate_values = parsed_var.degenerate_record.variable_value_map[key]
            for degenerate_value in degenerate_values:
                diff_with_pi = degenerate_value - sp.pi
                self.assertTrue(diff_with_pi == sp.S.Zero or degenerate_value == sp.S.Zero)

    def test_solved_variable_polynomial(self):
        th_0 = self._thetas[0]
        th_1 = self._thetas[1]
        degenerate_record = SolutionDegenerateRecord.record_variable_value({th_0: [sp.S.Zero, sp.pi]})
        poly_dict = {0: (sp.sin(th_0) + self._parameters[0], sp.S.One)}
        solved_var = PolynomialVariableSolutionImpl.make_polynomial_solution(
            th_1, SolutionMethod.PolynomialSin.name, poly_dict, degenerate_record)
        solved_var_dict = solution_impl_to_dict(solved_var)
        parsed_var: PolynomialVariableSolutionImpl = solution_impl_from_dict(solved_var_dict)

        # Check the parsed expr
        self.assertTrue(parsed_var.is_polynomial)
        self.assertTrue(parsed_var.solved_variable == solved_var.solved_variable)
        self.assertTrue(parsed_var.solution_method == solved_var.solution_method)
        for k in poly_dict:
            assert k in parsed_var.polynomial_to_solve
            v = poly_dict[k]
            poly_expr_k_num, poly_expr_k_denum = parsed_var.polynomial_to_solve[k]
            diff_num = poly_expr_k_num - v[0]
            diff_denum = poly_expr_k_denum - v[1]
            self.assertTrue(sp.simplify(diff_num) == sp.S.Zero)
            self.assertTrue(sp.simplify(diff_denum) == sp.S.Zero)

        # Check the degenerate record
        self.assertTrue(parsed_var.degenerate_record.type == degenerate_record.type)
        self.assertTrue(len(parsed_var.degenerate_record.equations) == 0)
        self.assertTrue(len(parsed_var.degenerate_record.variable_value_map) == 1)
        for key in parsed_var.degenerate_record.variable_value_map:
            self.assertTrue(key == th_0)
            degenerate_values = parsed_var.degenerate_record.variable_value_map[key]
            for degenerate_value in degenerate_values:
                diff_with_pi = degenerate_value - sp.pi
                self.assertTrue(diff_with_pi == sp.S.Zero or degenerate_value == sp.S.Zero)

    def test_snapshot(self):
        import solver.build_equations as build_equations
        import solver.solver_snapshot as solver_snapshot
        import os
        robot = puma_robot()

        # Build the snapshot
        fk_out, scalar_equations, sos_hint = build_equations.build_raw_fk_equations(robot)
        all_symbol_set = build_equations.get_all_symbols(robot)
        parameter_bounds = build_equations.get_symbol_bounds(robot)
        snapshot = solver_snapshot.SolverSnapshot(
            scalar_equations, sos_hint, robot.unknowns, all_symbol_set)
        snapshot.parameter_values = robot.parameters_value
        snapshot.parameter_bounds = parameter_bounds

        # Save snapshot and load_it
        save_snapshot_path = 'snapshot_test_tmp.yaml'
        solver_snapshot.save_solver_snapshot(snapshot, save_snapshot_path)
        loaded_snapshot = solver_snapshot.load_solver_snapshot(save_snapshot_path)

        # Compare with original
        self.assertTrue(len(loaded_snapshot.unknowns) == len(snapshot.unknowns))
        for i in range(len(snapshot.unknowns)):
            self.assertTrue(snapshot.unknowns[i].symbol.name == loaded_snapshot.unknowns[i].symbol.name)

        # Utility function
        def expr_pair_equals(equ_0: sp.Expr, equ_1: sp.Expr):
            if equ_1 - equ_0 == sp.S.Zero:
                return True
            simplified_diff = sp.simplify(equ_1 - equ_0)
            return simplified_diff == sp.S.Zero

        def scalar_equation_pair_equals(equ_0: ScalarEquation, equ_1: ScalarEquation):
            return expr_pair_equals(equ_0.lhs, equ_1.lhs) and expr_pair_equals(equ_0.rhs, equ_1.rhs)

        # Compare scalar equations
        self.assertTrue(len(loaded_snapshot.scalar_equations) == len(snapshot.scalar_equations))
        for i in range(len(snapshot.scalar_equations)):
            equ_i = snapshot.scalar_equations[i]
            equ_i_loaded = loaded_snapshot.scalar_equations[i]
            self.assertTrue(scalar_equation_pair_equals(equ_i, equ_i_loaded))

        # Compare the sos hints
        self.assertTrue(len(loaded_snapshot.sos_hints) == len(snapshot.sos_hints))
        for i in range(len(snapshot.sos_hints)):
            hint_i = snapshot.sos_hints[i]
            hint_i_loaded = loaded_snapshot.sos_hints[i]
            self.assertTrue(scalar_equation_pair_equals(hint_i.equ_1, hint_i_loaded.equ_1))
            self.assertTrue(scalar_equation_pair_equals(hint_i.equ_2, hint_i_loaded.equ_2))
            if hint_i.equ_3 is not None:
                self.assertTrue(hint_i_loaded.equ_3 is not None)
                self.assertTrue(scalar_equation_pair_equals(hint_i.equ_3, hint_i_loaded.equ_3))

        # Remove the tmp path
        os.remove(save_snapshot_path)

    def test_sp_matrix(self):
        matrix_0 = sp.zeros(3, 1)
        matrix_1 = sp.zeros(3, 2)
        for i in range(3):
            matrix_0[i] = self._expr_to_test[i]
            matrix_1[i, 0] = self._expr_to_test[i]
            matrix_1[i, 1] = self._expr_to_test[i] + sp.S.One

        # Parse and load
        matrix_0_dict = equation_utils.sp_matrix_to_dict_representation(matrix_0)
        matrix_1_dict = equation_utils.sp_matrix_to_dict_representation(matrix_1)
        loaded_matrix_0 = equation_utils.parse_sp_matrix_from_dict(matrix_0_dict)
        loaded_matrix_1 = equation_utils.parse_sp_matrix_from_dict(matrix_1_dict)

        def sp_matrix_matched(a: sp.Matrix, b: sp.Matrix) -> bool:
            for r in range(a.shape[0]):
                for c in range(a.shape[1]):
                    diff_rc = sp.simplify(a[r, c] - b[r, c])
                    if diff_rc != sp.S.Zero:
                        return False
            return True

        self.assertTrue(sp_matrix_matched(loaded_matrix_0, matrix_0))
        self.assertTrue(sp_matrix_matched(loaded_matrix_1, matrix_1))


if __name__ == '__main__':
    pass
