import fk.kinematics_dh as kinematics_dh
import fk.intersecting_axis as intersecting_axis
from fk.intersecting_axis_equation import intersecting_axis_triplet_equation, intersecting_axis_pair_equation
from utility.symbolic_utils import is_identity_matrix, multiple_list_of_transforms, inverse_transform
from fk.fk_equations import build_fk_matrix_equations, Py, ik_target_4x4, inv_Px, inv_Py, inv_Pz
from fk.robots import puma_robot, arm_robo
import sympy as sp
from typing import List
import numpy as np
import copy
import unittest


class TestSymbolicFK(unittest.TestCase):
    def setUp(self) -> None:
        (th_1, th_2, th_3, th_4) = sp.symbols(('th_1', 'th_2', 'th_3', 'th_4'))
        h = sp.Symbol('h')
        l_3 = sp.Symbol('l_3')
        l_4 = sp.Symbol('l_4')
        dh_0 = kinematics_dh.DHEntry(alpha=sp.pi / 2, a=0, d=h, theta=th_1)
        dh_1 = kinematics_dh.DHEntry(0, 0, 0, th_2)
        dh_2 = kinematics_dh.DHEntry(-sp.pi / 2, 0, l_3, th_3)
        dh_3 = kinematics_dh.DHEntry(0, 0, l_4, th_4)

        # Update
        self._dhs = [dh_0, dh_1, dh_2, dh_3]
        self._thetas = [th_1, th_2, th_3, th_4]
        self._h = h

        # For robot
        self._robot = arm_robo()
        self._robot_unknowns = [elem.symbol for elem in self._robot.unknowns]
        self._robot_fk_out = kinematics_dh.forward_kinematics_dh(
            self._robot.dh_params, self._robot_unknowns)

        # Random test n
        numerical_test_n = 50
        self._numerical_test_subst_maps = list()
        for test_idx in range(numerical_test_n):
            # Do forward kinematics to get the result
            value = np.random.random(len(self._robot_unknowns))
            subst_map = copy.deepcopy(self._robot.parameters_value)
            for j in range(len(self._robot_unknowns)):
                subst_map[self._robot_unknowns[j]] = value[j]
            T_ee = self._robot_fk_out.T_ee()
            ik_target = ik_target_4x4()
            result_map = dict()
            for i in range(3):
                for j in range(4):
                    symbol_ij = ik_target[i, j]
                    value_ij = T_ee[i, j].subs(subst_map)
                    result_map[symbol_ij] = value_ij
            subst_map.update(result_map)

            # Update the inverse map
            inv_target = inverse_transform(ik_target_4x4())
            inv_target_translations = [inv_Px, inv_Py, inv_Pz]
            for i in range(3):
                symbol_i = inv_target_translations[i]
                expr_i = inv_target[i, 3]
                expr_i_value = expr_i.subs(subst_map)
                subst_map[symbol_i] = expr_i_value
            self._numerical_test_subst_maps.append(subst_map)

    def test_rotation_matrix(self):
        # Test of rotation utils
        identity_mat = sp.eye(4)
        self.assertTrue(is_identity_matrix(identity_mat))

        # Test of DH parameters
        for k in range(len(self._dhs)):
            dh_param = self._dhs[k]
            theta = self._thetas[k]
            dh_transform_0 = kinematics_dh.modified_dh_transform(dh_param)
            dh_transform_tuple_1 = kinematics_dh.modified_dh_transform_tuple(dh_param, theta)
            dh_transform_1 = multiple_list_of_transforms(dh_transform_tuple_1)
            for i in range(4):
                for j in range(4):
                    for pair in [(dh_transform_0, dh_transform_1)]:
                        first_transform, second_transform = pair
                        term_0 = first_transform[i, j]
                        term_1 = second_transform[i, j]
                        diff_ij = term_0 - term_1
                        diff_ij = diff_ij.simplify()
                        self.assertTrue(diff_ij == sp.S.Zero, 'Term mismatched in rotation equations.')
                        if diff_ij != sp.S.Zero:
                            print('Term mismatch between')
                            print(term_0)
                            print(term_1)

    def test_fk(self):
        result: kinematics_dh.ForwardKinematicsOutput = kinematics_dh.forward_kinematics_dh(self._dhs, self._thetas)
        m = result.Ts(0) * result.Ts(1)
        self.assertTrue(m[0, 0] == sp.cos(self._thetas[0]) * sp.cos(self._thetas[1]) - sp.sin(self._thetas[0]) * sp.sin(
            self._thetas[1]), 'kinematics class FAIL')
        self.assertTrue(
            m[0, 1] == -sp.cos(self._thetas[0]) * sp.sin(self._thetas[1]) - sp.cos(self._thetas[1]) * sp.sin(
                self._thetas[0]), 'kinematics class FAIL')
        self.assertTrue(m[1, 2] == -1, 'kinematics class FAIL')
        self.assertTrue(m[1, 3] == -self._h, 'kinematics class FAIL')

        # Test equations
        L = build_fk_matrix_equations(result)
        test_idx = 3
        r_21, r_22, r_23 = sp.symbols('r_21 r_22 r_23')
        self.assertTrue(L[test_idx].Td[2, 0] == -r_21, 'intermediate equations FAIL')
        self.assertTrue(L[test_idx].Td[2, 1] == -r_22, 'intermediate equations FAIL')
        self.assertTrue(L[test_idx].Td[2, 2] == -r_23, 'intermediate equations FAIL')
        self.assertTrue(L[test_idx].Td[2, 3] == -Py - self._h, 'intermediate equations FAIL')

    def test_intersecting_axis_pair(self):
        dh_2, dh_3 = self._dhs[-2], self._dhs[-1]
        intersect = intersecting_axis.detect_intersecting_axis_pair(dh_2, dh_3)
        self.assertTrue(intersect is not None, 'Incorrect intersect')
        intersect_in_1_direct, intersect_in_2, intersect_in_3, _ = intersect
        transform_3_to_2 = kinematics_dh.modified_dh_transform(dh_3)
        transform_2_to_1 = kinematics_dh.modified_dh_transform(dh_2)
        transform_3_to_1 = transform_2_to_1 * transform_3_to_2

        # Do multiple
        intersect_in_3 = sp.Matrix([intersect_in_3[0], intersect_in_3[1], intersect_in_3[2], 1])
        intersect_in_2_by_multiple = transform_3_to_2 * intersect_in_3
        intersect_in_1 = transform_3_to_1 * intersect_in_3

        def check_not_contained_var_23(point_expr: sp.Expr):
            point_expr = sp.simplify(point_expr)
            point_expr = sp.expand_trig(point_expr)
            self.assertTrue(self._thetas[3] not in point_expr.free_symbols)
            self.assertTrue(self._thetas[2] not in point_expr.free_symbols)

        # Test the intersect in 2
        for i in range(3):
            point_i: sp.Expr = intersect_in_2[i] - intersect_in_2_by_multiple[i]
            point_i = sp.simplify(point_i)
            self.assertTrue(point_i == sp.S.Zero)

        # Test the intersect in 1
        for i in range(3):
            point_i: sp.Expr = intersect_in_1[i]
            check_not_contained_var_23(point_i)
            point_diff_i = point_i - intersect_in_1_direct[i]
            point_diff_i = sp.simplify(point_diff_i)
            self.assertTrue(point_diff_i == sp.S.Zero)

        # Test of inverse multiplication
        inv_multiple_result = \
            inverse_transform(transform_3_to_2) * inverse_transform(transform_2_to_1) * intersect_in_1
        for i in range(3):
            point_i: sp.Expr = inv_multiple_result[i]
            check_not_contained_var_23(point_i)

    def test_intersecting_axis_equations_variables(self):
        # Unpack the data
        robot = self._robot
        unknowns = self._robot_unknowns
        fk_out = self._robot_fk_out

        # Test pair equations
        def is_symbol_contained(symbol_element: sp.Symbol, name_list: List[str]) -> bool:
            for idx in range(len(name_list)):
                if name_list[idx] == symbol_element.name:
                    return True
            return False

        # Test triplet equations
        equations_dict_triplet = intersecting_axis_triplet_equation(
            fk_out, robot.dh_params, unknowns, True)
        assert len(equations_dict_triplet) == 1
        for k in equations_dict_triplet:
            equations = equations_dict_triplet[k]
            variable_name_list = list(k)
            variable_symbols_to_check = list()
            for i in range(len(unknowns)):
                if not is_symbol_contained(unknowns[i], variable_name_list):
                    variable_symbols_to_check.append(unknowns[i])
            for equation_xyz in equations:
                for equation in [equation_xyz.x, equation_xyz.y, equation_xyz.z]:
                    lhs_minus_rhs = equation.lhs - equation.rhs
                    for variable in variable_symbols_to_check:
                        self.assertTrue(variable not in lhs_minus_rhs.free_symbols, lhs_minus_rhs)
                        if variable in lhs_minus_rhs.free_symbols:
                            print(variable)
                            print(lhs_minus_rhs)

        equation_dict_pair = intersecting_axis_pair_equation(fk_out, robot.dh_params, unknowns)
        for k in equation_dict_pair:
            variable_name_list = list(k)
            translational_equations, cos_equations = equation_dict_pair[k]
            variable_symbols_to_check = list()
            for i in range(len(unknowns)):
                if not is_symbol_contained(unknowns[i], variable_name_list):
                    variable_symbols_to_check.append(unknowns[i])

            # Check translational equation
            for equation_xyz in translational_equations:
                for xyz in [equation_xyz.x, equation_xyz.y, equation_xyz.z]:
                    lhs_minus_rhs = xyz.lhs - xyz.rhs
                    for symbol_i in variable_symbols_to_check:
                        self.assertTrue(symbol_i not in lhs_minus_rhs.free_symbols)
                        if symbol_i in lhs_minus_rhs.free_symbols:
                            print(lhs_minus_rhs)

            # Check cos equations
            for equation in cos_equations:
                lhs_minus_rhs = equation.lhs - equation.rhs
                for symbol_i in variable_symbols_to_check:
                    self.assertTrue(symbol_i not in lhs_minus_rhs.free_symbols)
                    if symbol_i in lhs_minus_rhs.free_symbols:
                        print(lhs_minus_rhs)

    def test_robot_fk_numerical(self):
        fk_out = self._robot_fk_out
        matrix_equations = build_fk_matrix_equations(fk_out)

        # OK, start the test
        for test_idx in range(len(self._numerical_test_subst_maps)):
            # The forward kinematics result
            subst_map = self._numerical_test_subst_maps[test_idx]
            print('Perform numerical checking of matrix equations: ', test_idx)
            for i in range(len(matrix_equations)):
                matrix_equation_i = matrix_equations[i]
                for r in range(3):
                    for c in range(4):
                        lhs: sp.Expr = matrix_equation_i.Td[r, c]
                        rhs: sp.Expr = matrix_equation_i.Ts[r, c]
                        rhs_minus_lhs = rhs - lhs
                        rhs_minus_lhs_value = rhs_minus_lhs.subs(subst_map)
                        self.assertTrue(abs(rhs_minus_lhs_value) < 1e-10)
                        if abs(rhs_minus_lhs_value) > 1e-10:
                            print(rhs_minus_lhs)

    def test_intersecting_axis_equations_numerical(self):
        # First test pair
        robot = self._robot
        unknowns = self._robot_unknowns
        fk_out = self._robot_fk_out
        equation_dict_pair = intersecting_axis_pair_equation(fk_out, robot.dh_params, unknowns)
        equation_dict_triplet = intersecting_axis_triplet_equation(
            fk_out, robot.dh_params, unknowns, True)

        # OK, start the test
        for test_idx in range(len(self._numerical_test_subst_maps)):
            # The forward kinematics result
            subst_map = self._numerical_test_subst_maps[test_idx]

            # Check the equation
            print('Perform numerical checking of intersecting axis equations: ', test_idx)
            for variables in equation_dict_triplet:
                translation_equations = equation_dict_triplet[variables]
                for equation in translation_equations:
                    for component in [equation.x, equation.y, equation.z]:
                        lhs_minus_rhs = component.lhs - component.rhs
                        component_value = lhs_minus_rhs.subs(subst_map)
                        self.assertTrue(abs(component_value) < 1e-10, component_value)
                        if abs(component_value) > 1e-10:
                            print(lhs_minus_rhs)

            # Check the pair equation
            for variables in equation_dict_pair:
                translation_equations, cos_equations = equation_dict_pair[variables]
                self.assertTrue(len(translation_equations) > 0)
                self.assertTrue(len(cos_equations) > 0)
                for equation in translation_equations:
                    for xyz_expr in [equation.x, equation.y, equation.z]:
                        xyz_value = (xyz_expr.lhs - xyz_expr.rhs).subs(subst_map)
                        self.assertTrue(abs(xyz_value) < 1e-10)
                        if abs(xyz_value) > 1e-10:
                            print(xyz_expr)

                # Check the cos equations
                for equation in cos_equations:
                    equation_value = (equation.lhs - equation.rhs).subs(subst_map)
                    self.assertTrue(abs(equation_value) < 1e-10)
                    if abs(equation_value) > 1e-10:
                        print(equation)


if __name__ == '__main__':
    pass
