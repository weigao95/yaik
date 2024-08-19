import copy

import numpy as np
import sympy as sp
import solver.equation_utils as equation_utils
from solver.general_6dof.dh_utils import RevoluteVariable, reflected_variable_and_structure_transform
import solver.general_6dof.dh_utils as dh_utils
from solver.general_6dof.matrix_closure_equation import ReducedRaghavanRothEquationFormatter
from solver.build_equations import ScalarEquation
from typing import List, Set, Dict, Optional
from utility import symbolic_utils
from itertools import combinations
from sympy.solvers.solveset import linsolve
from fk.fk_equations import ik_target_4x4, ik_target_inv_4x4
from solver.equation_utils import cast_expr_to_float
import attr


@attr.s
class RawClosureEquation(object):
    scalar_equations: List[ScalarEquation] = attr.ib()
    lhs_unknowns: List[RevoluteVariable] = attr.ib()
    rhs_unknowns: List[RevoluteVariable] = attr.ib()


@attr.s
class UnReducedMatrixClosureEquation(object):
    lhs_matrix: sp.Matrix = attr.ib()
    rhs_matrix: sp.Matrix = attr.ib()
    lhs_terms: sp.Matrix = attr.ib()
    rhs_terms: sp.Matrix = attr.ib()

    # The variable in lhs gen
    variable_in_lhs_matrix: RevoluteVariable = None
    lhs_gen_vars: List[RevoluteVariable] = list()

    # The coefficient that should be zero, but can be very complex in symbolic processing
    # Thus, check them numerically
    ignored_coefficient: List[sp.Expr] = list()

    def subs(self, subst_map):
        subst_lhs_matrix = self.lhs_matrix.subs(subst_map)
        subst_rhs_matrix = self.rhs_matrix.subs(subst_map)
        subst_lhs_terms = self.lhs_terms.subs(subst_map)
        subst_rhs_terms = self.rhs_terms.subs(subst_map)
        closure = UnReducedMatrixClosureEquation(
            subst_lhs_matrix, subst_rhs_matrix, subst_lhs_terms, subst_rhs_terms)
        closure.variable_in_lhs_matrix = copy.deepcopy(self.variable_in_lhs_matrix)
        closure.lhs_gen_vars = copy.deepcopy(self.lhs_gen_vars)
        return closure

    def to_dict(self):
        datamap = dict()
        datamap['lhs_matrix'] = equation_utils.sp_matrix_to_dict_representation(self.lhs_matrix)
        datamap['rhs_matrix'] = equation_utils.sp_matrix_to_dict_representation(self.rhs_matrix)
        datamap['lhs_terms'] = equation_utils.sp_matrix_to_dict_representation(self.lhs_terms)
        datamap['rhs_terms'] = equation_utils.sp_matrix_to_dict_representation(self.rhs_terms)

        # The lhs variable
        assert self.variable_in_lhs_matrix is not None
        datamap['variable_in_lhs_matrix'] = self.variable_in_lhs_matrix.to_dict()

        # The lhs generator
        assert len(self.lhs_gen_vars) == 2
        flatten_gen = list()
        for i in range(len(self.lhs_gen_vars)):
            flatten_gen.append(self.lhs_gen_vars[i].to_dict())
        datamap['lhs_gen_vars'] = flatten_gen

        # Done
        return datamap

    def from_dict(self, datamap: Dict):
        self.lhs_matrix = equation_utils.parse_sp_matrix_from_dict(datamap['lhs_matrix'])
        self.rhs_matrix = equation_utils.parse_sp_matrix_from_dict(datamap['rhs_matrix'])
        self.lhs_terms = equation_utils.parse_sp_matrix_from_dict(datamap['lhs_terms'])
        self.rhs_terms = equation_utils.parse_sp_matrix_from_dict(datamap['rhs_terms'])

        # Load the variable
        self.variable_in_lhs_matrix = RevoluteVariable.load_from_dict(datamap['variable_in_lhs_matrix'])

        # Load the flatten gen
        flatten_gen = datamap['lhs_gen_vars']
        for i in range(len(flatten_gen)):
            self.lhs_gen_vars.append(RevoluteVariable.load_from_dict(flatten_gen[i]))

    @staticmethod
    def load_from_dict(datamap: Dict):
        dummy = UnReducedMatrixClosureEquation(sp.zeros(1, 1), sp.zeros(1, 1), sp.zeros(1, 1), sp.zeros(1, 1))
        dummy.from_dict(datamap)
        return dummy


def merge_all_sin_cos_sos(expr2merge: sp.Expr, unknowns: List[RevoluteVariable]):
    poly_expr = expr2merge
    for revolute_var in unknowns:
        poly_expr, _ = dh_utils.merge_sin_cos_sos(poly_expr, revolute_var.sin_var, revolute_var.cos_var)
    return poly_expr


def closure_equation_from_matrix(
        lhs_matrix_in: sp.Matrix,
        rhs_matrix_in: sp.Matrix,
        unknowns: List[RevoluteVariable],
        unknown2remove_on_rhs: RevoluteVariable) -> Optional[RawClosureEquation]:
    """
    Build the closure equation from matrix. The input should be matrix
    with variable in sin(theta). In the output, these will be reduced to sin_theta symbol
    :return the 14 raghavan_roth_equations or nothing
    """
    # Make substitute map
    sin_cos_subst_map = dict()
    sos_subst_map = dict()
    for unknown_i in unknowns:
        sin_cos_subst_map[sp.sin(unknown_i.variable_symbol)] = unknown_i.sin_var
        sin_cos_subst_map[sp.cos(unknown_i.variable_symbol)] = unknown_i.cos_var
        sos_subst_map[unknown_i.sin_var ** 2] = 1 - unknown_i.cos_var ** 2

    # Subst the lhs and rhs
    def subst_as_polynomial(expr2subst: sp.Expr) -> sp.Expr:
        poly_expr = expr2subst.subs(sin_cos_subst_map)
        poly_expr = merge_all_sin_cos_sos(poly_expr, unknowns)
        return poly_expr

    # Process the matrix
    lhs_matrix, rhs_matrix = sp.eye(4), sp.eye(4)
    for c in range(4):
        for r in range(3):
            lhs_matrix[r, c] = subst_as_polynomial(lhs_matrix_in[r, c])
            rhs_matrix[r, c] = subst_as_polynomial(rhs_matrix_in[r, c])

    rhs_column_no_removed_var: List[int] = list()
    for c in range(4):
        find_removed: bool = False
        for r in range(3):
            rhs_rc_value: sp.Expr = rhs_matrix[r, c]
            if rhs_rc_value.has(unknown2remove_on_rhs.sin_var) or rhs_rc_value.has(unknown2remove_on_rhs.cos_var):
                find_removed = True
        # Update the column freed of removed symbol
        if not find_removed:
            rhs_column_no_removed_var.append(c)

    # Check result
    if (len(rhs_column_no_removed_var)) < 2 or (3 not in rhs_column_no_removed_var):
        return None
    rhs_column_no_removed_var = sorted(rhs_column_no_removed_var)

    # The column that is part of the rotation matrix
    column_0 = rhs_column_no_removed_var[0] if (2 not in rhs_column_no_removed_var) else 2
    h_lhs_vector = sp.Matrix([lhs_matrix[0, column_0], lhs_matrix[1, column_0], lhs_matrix[2, column_0]])
    i_rhs_vector = sp.Matrix([rhs_matrix[0, column_0], rhs_matrix[1, column_0], rhs_matrix[2, column_0]])

    # The one depends on translation
    column_1 = 3
    f_lhs_vector = sp.Matrix([lhs_matrix[0, column_1], lhs_matrix[1, column_1], lhs_matrix[2, column_1]])
    g_rhs_vector = sp.Matrix([rhs_matrix[0, column_1], rhs_matrix[1, column_1], rhs_matrix[2, column_1]])

    # Build equations
    scalar_equations = raghavan_roth_14_equations(
        f_lhs_vector=f_lhs_vector, g_rhs_vector=g_rhs_vector,
        h_lhs_vector=h_lhs_vector, i_rhs_vector=i_rhs_vector,
        unknowns=unknowns)

    # The map from sin cos to unknowns
    sin_cos_to_revolute_variable_map = dict()
    for unknown_i in unknowns:
        sin_cos_to_revolute_variable_map[unknown_i.sin_var] = unknown_i
        sin_cos_to_revolute_variable_map[unknown_i.cos_var] = unknown_i

    def collect_revolute_variables(expr_list: List[sp.Expr]) -> List[RevoluteVariable]:
        expr_variable_set: Set[sp.Symbol] = set()
        expr_revolute_variables: List[RevoluteVariable] = list()
        for expr_elem in expr_list:
            for expr_symbol in expr_elem.free_symbols:
                if expr_symbol in sin_cos_to_revolute_variable_map:
                    revolute_var = sin_cos_to_revolute_variable_map[expr_symbol]
                    if not (revolute_var.variable_symbol in expr_variable_set):
                        expr_variable_set.add(revolute_var.variable_symbol)
                        expr_revolute_variables.append(revolute_var)
        return expr_revolute_variables
    lhs_revolute_variables: List[RevoluteVariable] = collect_revolute_variables([elem.lhs for elem in scalar_equations])
    rhs_revolute_variables: List[RevoluteVariable] = collect_revolute_variables([elem.rhs for elem in scalar_equations])
    lhs_revolute_variables.sort(key=(lambda elem: elem.variable_symbol.name))
    rhs_revolute_variables.sort(key=(lambda elem: elem.variable_symbol.name))

    # OK
    raw_equations = RawClosureEquation(
        scalar_equations,
        lhs_unknowns=lhs_revolute_variables, rhs_unknowns=rhs_revolute_variables)
    return raw_equations


def raghavan_roth_14_equations(
        f_lhs_vector: sp.Matrix, g_rhs_vector: sp.Matrix,
        h_lhs_vector: sp.Matrix, i_rhs_vector: sp.Matrix,
        unknowns: List[RevoluteVariable]) -> List[ScalarEquation]:
    """
    Build the raghavan_roth 14 equations from the vector.
    The vector should have been substituted out sin/cos as polynomial variable.
    """
    f_vector = f_lhs_vector
    g_vector = g_rhs_vector
    h_vector = h_lhs_vector
    i_vector = i_rhs_vector
    scalar_equations: List[ScalarEquation] = list()

    def append_scalar_equation(lhs: sp.Expr, rhs: sp.Expr):
        # lhs = merge_all_sin_cos_sos(lhs, unknowns)
        # rhs = merge_all_sin_cos_sos(rhs, unknowns)
        scalar_equations.append(ScalarEquation(lhs, rhs))

    # f = g, h = i
    for k in range(3):
        append_scalar_equation(f_vector[k], g_vector[k])
    for k in range(3):
        append_scalar_equation(h_vector[k], i_vector[k])

    # f_dot_f = g_dot_g
    f_dot_f = f_vector.dot(f_vector)
    g_dot_g = g_vector.dot(g_vector)
    append_scalar_equation(f_dot_f, g_dot_g)

    # f_dot_h = g_dot_i
    f_dot_h = f_vector.dot(h_vector)
    g_dot_i = g_vector.dot(i_vector)
    append_scalar_equation(f_dot_h, g_dot_i)

    # f_cross_h = g_cross_i
    f_cross_h = f_vector.cross(h_vector)
    g_cross_i = g_vector.cross(i_vector)
    for i in range(3):
        append_scalar_equation(f_cross_h[i], g_cross_i[i])

    # The reflection
    lhs_reflection = f_dot_f * h_vector - 2 * f_dot_h * f_vector
    rhs_reflection = g_dot_g * i_vector - 2 * g_dot_i * g_vector
    for i in range(3):
        append_scalar_equation(lhs_reflection[i], rhs_reflection[i])
    return scalar_equations


def closure_equations_matrix_form(
        raw_closure: RawClosureEquation,
        lhs_gen=None,
        rhs_gen=None) -> Optional[UnReducedMatrixClosureEquation]:
    """
    Rewrite the raghavan roth 14 equations into matrix form as:
    lhs_matrix.dot(lhs_terms) = rhs_matrix.dot(rhs_terms)
    """
    assert len(raw_closure.lhs_unknowns) == 3
    assert len(raw_closure.rhs_unknowns) == 2
    lhs_unknowns = raw_closure.lhs_unknowns
    rhs_unknowns = raw_closure.rhs_unknowns
    lhs_unknowns.sort(key=(lambda elem: elem.variable_symbol.name))
    rhs_unknowns.sort(key=(lambda elem: elem.variable_symbol.name))

    # Process the generator and the remaining variable
    variable_in_lhs_matrix: RevoluteVariable = None
    if lhs_gen is None:
        lhs_gen = [lhs_unknowns[1].sin_var, lhs_unknowns[1].cos_var, lhs_unknowns[2].sin_var, lhs_unknowns[2].cos_var]
        variable_in_lhs_matrix = lhs_unknowns[0]
    else:
        assert len(lhs_gen) == 4
        for var in lhs_unknowns:
            if var.sin_var in lhs_gen or var.cos_var in lhs_gen:
                continue
            variable_in_lhs_matrix = var
        assert variable_in_lhs_matrix is not None

    # Rhs is simpler
    if rhs_gen is None:
        rhs_gen = [rhs_unknowns[0].sin_var, rhs_unknowns[0].cos_var, rhs_unknowns[1].sin_var, rhs_unknowns[1].cos_var]

    # sos terms
    sos_subst_map = dict()
    for unknown_i in lhs_unknowns + rhs_unknowns:
        sos_subst_map[unknown_i.sin_var ** 2] = 1 - unknown_i.cos_var ** 2
        sos_subst_map[unknown_i.sin_var ** 3] = unknown_i.sin_var * (1 - unknown_i.cos_var ** 2)

    # The map from monomial to the offset
    non_constant_monos = [
        (1, 0, 1, 0),
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (0, 1, 0, 1),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1)
    ]
    mono2offset = dict()
    for i in range(len(non_constant_monos)):
        mono2offset[non_constant_monos[i]] = i

    # First process lhs matrix
    ignored_coefficient: List[sp.Expr] = list()
    rhs_remaining_constant: List[sp.Expr] = list()
    rhs_coefficient_matrix = sp.zeros(len(raw_closure.scalar_equations), len(non_constant_monos))
    for i in range(len(raw_closure.scalar_equations)):
        equation_i_rhs = raw_closure.scalar_equations[i].rhs
        equation_i_rhs = sp.expand(equation_i_rhs, multinomial=True)
        equation_i_rhs = equation_i_rhs.subs(sos_subst_map)
        # equation_i_rhs = merge_all_sin_cos_sos(equation_i_rhs, rhs_unknowns)
        poly_rhs_i = sp.Poly(equation_i_rhs, gens=rhs_gen)
        poly_rhs_constant = sp.S.Zero
        for term in poly_rhs_i.terms():
            mono, coefficient = term
            if mono in mono2offset:
                offset_for_mono = mono2offset[mono]
                rhs_coefficient_matrix[i, offset_for_mono] = coefficient
            elif mono == (0, 0, 0, 0):
                poly_rhs_constant += coefficient
            else:
                ignored_coefficient.append(coefficient)
        rhs_remaining_constant.append(poly_rhs_constant)

    lhs_coefficient_matrix = sp.zeros(len(raw_closure.scalar_equations), len(non_constant_monos) + 1)
    for i in range(len(raw_closure.scalar_equations)):
        equation_i_lhs = raw_closure.scalar_equations[i].lhs
        equation_i_lhs = sp.expand(equation_i_lhs, multinomial=True)
        equation_i_lhs = equation_i_lhs.subs(sos_subst_map)
        poly_lhs_i = sp.Poly(equation_i_lhs, gens=lhs_gen)
        poly_lhs_constant = - rhs_remaining_constant[i]
        for term in poly_lhs_i.terms():
            mono, coefficient = term
            if mono in mono2offset:
                offset_for_mono = mono2offset[mono]
                coefficient = merge_all_sin_cos_sos(coefficient, unknowns=[variable_in_lhs_matrix])
                lhs_coefficient_matrix[i, offset_for_mono] = coefficient
            elif mono == (0, 0, 0, 0):
                poly_lhs_constant += coefficient
            else:
                ignored_coefficient.append(coefficient)
        lhs_coefficient_matrix[i, len(non_constant_monos)] = poly_lhs_constant

    # Build the terms from mono
    lhs_terms = sp.zeros(len(non_constant_monos) + 1, 1)
    rhs_terms = sp.zeros(len(non_constant_monos), 1)
    for i in range(len(non_constant_monos)):
        mono_i = non_constant_monos[i]
        lhs_this_term = sp.S.One
        rhs_this_term = sp.S.One
        for k in range(len(lhs_gen)):
            lhs_this_term = lhs_this_term * (lhs_gen[k] ** mono_i[k])
            rhs_this_term = rhs_this_term * (rhs_gen[k] ** mono_i[k])
        lhs_terms[i] = lhs_this_term
        rhs_terms[i] = rhs_this_term
    lhs_terms[len(lhs_terms) - 1] = sp.S.One

    # Collect the result
    matrix_equation = UnReducedMatrixClosureEquation(
        lhs_matrix=lhs_coefficient_matrix, rhs_matrix=rhs_coefficient_matrix,
        lhs_terms=lhs_terms, rhs_terms=rhs_terms)
    matrix_equation.variable_in_lhs_matrix = variable_in_lhs_matrix

    # Get the variable for lhs gen
    def find_lhs_variable_of_sin_cos_symbol(sin_cos_symbol: sp.Symbol):
        for rev_var in lhs_unknowns:
            if rev_var.sin_var == sin_cos_symbol or rev_var.cos_var == sin_cos_symbol:
                return rev_var
        return None
    matrix_equation.lhs_gen_vars = [
        find_lhs_variable_of_sin_cos_symbol(lhs_gen[0]),
        find_lhs_variable_of_sin_cos_symbol(lhs_gen[2])]
    assert matrix_equation.lhs_gen_vars[0] is not None
    assert matrix_equation.lhs_gen_vars[1] is not None
    matrix_equation.ignored_coefficient = ignored_coefficient
    return matrix_equation


def reduce_matrix_equation(matrix_equation: UnReducedMatrixClosureEquation):
    assert len(matrix_equation.lhs_terms) == 9
    assert len(matrix_equation.rhs_terms) == 8

    # Local vars
    restore_map: Dict[sp.Symbol, sp.Expr] = dict()
    rhs_var_generator = dh_utils.NumberedSymbolGenerator('rhs_substituted_var')
    lhs_temp_generator = dh_utils.NumberedSymbolGenerator('expr_lhs_tmp')

    # Use symbol to represent the lhs of the equations
    # Size should be len(closure_equation.scalar_equations), usually 14
    equation_lhs_terms: List[sp.Symbol] = list()
    for i in range(matrix_equation.lhs_matrix.shape[0]):
        lhs_tmp_i = lhs_temp_generator.next()
        lhs_matrix_row_i = matrix_equation.lhs_matrix[i, :]
        restore_map[lhs_tmp_i] = lhs_matrix_row_i.dot(matrix_equation.lhs_terms)
        equation_lhs_terms.append(lhs_tmp_i)

    # Use symbol to represent the rhs vars (s1c2, s1c1)
    # Size should be len(closure_equation.rhs_terms), usually 8
    rhs_vars: List[sp.Symbol] = list()
    for i in range(len(matrix_equation.rhs_terms)):
        rhs_var_i = rhs_var_generator.next()
        rhs_vars.append(rhs_var_i)
    rhs_vars_vector = sp.Matrix(rhs_vars)

    # Now the equation becomes
    # equation_lhs_terms = R_rhs * rhs_vars
    R_rhs = matrix_equation.rhs_matrix
    R_rhs_nonzero_rows: List[int] = list()
    for i in range(R_rhs.shape[0]):
        all_zero = True
        for j in range(R_rhs.shape[1]):
            if R_rhs[i, j] != sp.S.Zero:
                all_zero = False
                break
        if not all_zero:
            R_rhs_nonzero_rows.append(i)

    # Get the rhs
    R_rhs_nonzero: sp.Matrix = sp.zeros(len(R_rhs_nonzero_rows), R_rhs.shape[1])
    lhs_for_nonzero_R_rhs: sp.Matrix = sp.zeros(len(R_rhs_nonzero_rows), 1)
    nonzero_row2original_row: Dict[int, int] = dict()
    for i in range(len(R_rhs_nonzero_rows)):
        row_idx = R_rhs_nonzero_rows[i]
        R_rhs_nonzero[i, :] = R_rhs[row_idx, :]
        nonzero_row2original_row[i] = row_idx
        lhs_for_nonzero_R_rhs[i] = equation_lhs_terms[row_idx]
    print('lhs ', lhs_for_nonzero_R_rhs)
    print('R_rhs matrix')
    dh_utils.print_sp_matrix_by_row(R_rhs_nonzero)

    # The lines should only contain 2 of them
    only_select_2_lines_from: Optional[List[int]] = list()
    for row_idx in [2, 5, 6, 7, 10, 13]:
        n_zeros = 0
        for i in range(R_rhs.shape[1]):
            if R_rhs[row_idx, i] == sp.S.Zero:
                n_zeros += 1
        if n_zeros >= 6:
            only_select_2_lines_from.append(row_idx)
    if len(only_select_2_lines_from) <= 2:
        only_select_2_lines_from = None
    print('Only select two lines from: ', only_select_2_lines_from)

    # Try solving it
    A_to_reduce: Optional[sp.Matrix] = None
    n_equations = len(lhs_for_nonzero_R_rhs)
    solution_map: Dict[sp.Symbol, sp.Expr] = dict()
    reduced_scalar_equations: List[sp.Expr] = list()
    for equation_indices in combinations(range(n_equations), len(rhs_vars)):
        A = sp.zeros(len(equation_indices), len(equation_indices))
        b = sp.zeros(len(equation_indices), 1)
        used_rows: Set[int] = set()
        for i in range(len(equation_indices)):
            row_idx = int(equation_indices[i])
            A[i, :] = R_rhs_nonzero[row_idx, :]
            b[i] = lhs_for_nonzero_R_rhs[row_idx]
            used_rows.add(nonzero_row2original_row[row_idx])

        # Filter by only_select_2_line_from
        used_row_in_only_2_line = 0
        for row_idx in used_rows:
            if row_idx in only_select_2_lines_from:
                used_row_in_only_2_line += 1
        if used_row_in_only_2_line >= 3:
            continue

        # Try to solve Ax = b
        print('Ax = b try solve with equations in indices ', used_rows)

        solution = linsolve((A, b), rhs_vars)
        if len(solution) == 0:
            continue

        # There it at least/and only one solution
        print('Ax = b solved and find a solution with equations in indices ', used_rows)

        # Update
        A_to_reduce = A
        assert len(solution) == 1
        for this_solution in solution:
            for i in range(len(this_solution)):
                solution_map[rhs_vars[i]] = this_solution[i].subs(restore_map)

        # Substitute the rhs to get the equations
        closure_lhs_expr = matrix_equation.lhs_terms

        rhs_vars_vector_subst = rhs_vars_vector.subs(solution_map)
        for i in range(matrix_equation.lhs_matrix.shape[0]):
            if i in used_rows:
                continue
            rhs_row_i = R_rhs[i, :].dot(rhs_vars_vector_subst)
            lhs_row_i = matrix_equation.lhs_matrix[i, :].dot(closure_lhs_expr)
            new_expr_i = lhs_row_i - rhs_row_i
            if new_expr_i != sp.S.Zero:
                print('Find a new reduced expr of {idx}:'.format(idx=len(reduced_scalar_equations)), new_expr_i)
                reduced_scalar_equations.append(new_expr_i)

        if len(solution_map) > 0:
            break

    if A_to_reduce is None:
        return None

    return reduced_scalar_equations, A_to_reduce


def reduced_expr_as_matrix_form(
        raw_closure: RawClosureEquation,
        matrix_closure: UnReducedMatrixClosureEquation,
        reduced_scalar_equations: List[sp.Expr],
        var_in_matrix: RevoluteVariable):
    unknown_in_reduced = raw_closure.lhs_unknowns
    assert (var_in_matrix.variable_symbol in ([elem.variable_symbol for elem in unknown_in_reduced]))

    # symbol to revolute var
    unknown_symbol_to_revolute = dict()
    for revolute_var in unknown_in_reduced:
        unknown_symbol_to_revolute[revolute_var.sin_var] = revolute_var
        unknown_symbol_to_revolute[revolute_var.cos_var] = revolute_var
        unknown_symbol_to_revolute[revolute_var.variable_symbol] = revolute_var
        unknown_symbol_to_revolute[revolute_var.tan_half_var] = revolute_var

    remaining_unknowns_set: Set[sp.Symbol] = set()
    remaining_unknowns_outside_matrix: List[RevoluteVariable] = list()
    for reduced_expr in reduced_scalar_equations:
        for expr_symbol in reduced_expr.free_symbols:
            if expr_symbol in unknown_symbol_to_revolute:
                corresponded_revolute_var = unknown_symbol_to_revolute[expr_symbol]
                if corresponded_revolute_var.variable_symbol == var_in_matrix.variable_symbol:
                    continue
                if corresponded_revolute_var.variable_symbol not in remaining_unknowns_set:
                    remaining_unknowns_set.add(corresponded_revolute_var.variable_symbol)
                    remaining_unknowns_outside_matrix.append(corresponded_revolute_var)

    remaining_unknowns_outside_matrix.sort(key=(lambda e: e.variable_symbol.name))
    for unknown_i in remaining_unknowns_outside_matrix:
        assert (unknown_i.variable_symbol in ([elem.variable_symbol for elem in unknown_in_reduced]))

    # The generator of the polynomial expr
    # sin/cos of var_in_matrix not included
    poly_gens: List[sp.Symbol] = list()
    tan_half_poly_gen: List[sp.Symbol] = list()
    for var in remaining_unknowns_outside_matrix:
        poly_gens.append(var.cos_var)
        poly_gens.append(var.sin_var)
        tan_half_poly_gen.append(var.tan_half_var)

    # The polynomial expressions that do not include var_in_matrix
    reduced_poly_equations: List[sp.Poly] = list()
    for reduced_expr in reduced_scalar_equations:
        subst_poly = sp.Poly(reduced_expr, gens=poly_gens)
        reduced_poly_equations.append(subst_poly)

    # Get the coefficient matrix representation
    coefficient_matrix, monomial_vector, monomials_list = \
        ReducedRaghavanRothEquationFormatter.rewrite_poly_equations_into_matrix_monomial(
            reduced_poly_equations, poly_gens)

    # Tangent half polynomial
    # sin <- 2 tan_half / (1 + tan_half**2)
    # cos <- 1 - tan_half**2 / (1 + tan_half**2)
    mono_subst_map: Dict[sp.Symbol, sp.Expr] = dict()
    for var in remaining_unknowns_outside_matrix:
        sin_var = var.sin_var
        cos_var = var.cos_var
        tan_half_var = var.tan_half_var
        mono_subst_map[sin_var] = (2 * tan_half_var) / (1 + (tan_half_var ** 2))
        mono_subst_map[cos_var] = (1 - (tan_half_var ** 2)) / (1 + (tan_half_var ** 2))

    # The maximum sin/cos degree for a var
    max_sin_and_cos_order_for_var: [sp.Symbol, int] = dict()
    for var in remaining_unknowns_outside_matrix:
        sin_var = var.sin_var
        cos_var = var.cos_var

        # Find the max order
        max_sin_cos_order = 0
        for mono_i in monomials_list:
            mono_i_order = 0
            for k in range(len(mono_i)):
                order_k = mono_i[k]
                if order_k > 0:
                    poly_gen_k = poly_gens[k]
                    if poly_gen_k == sin_var or poly_gen_k == cos_var:
                        mono_i_order += 1
            if mono_i_order > max_sin_cos_order:
                max_sin_cos_order = mono_i_order

        # Assign the result
        max_sin_and_cos_order_for_var[var.variable_symbol] = max_sin_cos_order

    # The term to clear the denominator
    clear_denominator_multiple = sp.S.One
    for var in max_sin_and_cos_order_for_var:
        requested_order = max_sin_and_cos_order_for_var[var]
        tan_half_var = unknown_symbol_to_revolute[var].tan_half_var
        clear_denominator_multiple = clear_denominator_multiple * ((1 + (tan_half_var ** 2)) ** requested_order)

    tan_half_subst_monomial_vector = sp.zeros(len(monomials_list), 1)
    for i in range(monomial_vector.shape[0]):
        mono_expr_i: sp.Expr = monomial_vector[i]
        subst_i = mono_expr_i.subs(mono_subst_map) * clear_denominator_multiple
        tan_half_subst_monomial_vector[i] = sp.simplify(subst_i)

    # Construct new polynomials
    tan_half_poly: List[sp.Poly] = list()
    for i in range(coefficient_matrix.shape[0]):
        expr_i = coefficient_matrix[i, :].dot(tan_half_subst_monomial_vector)
        poly_i = sp.Poly(expr_i, gens=tan_half_poly_gen)
        tan_half_poly.append(poly_i)

    # Get the coefficient matrix representation
    coefficient_matrix_tan_half, monomial_vector_tan_half, _ = \
        ReducedRaghavanRothEquationFormatter.rewrite_poly_equations_into_matrix_monomial(
            tan_half_poly, tan_half_poly_gen)
    return coefficient_matrix, coefficient_matrix_tan_half


def compute_dialytical_coefficient_matrix(
        coefficient_matrix_tan_half: sp.Matrix,
        unknown_in_matrix: RevoluteVariable):
    assert coefficient_matrix_tan_half.shape[0] == 6
    assert coefficient_matrix_tan_half.shape[1] == 9
    dialytical_matrix = sp.zeros(12, 12)
    dialytical_matrix[0:6, 0:9] = coefficient_matrix_tan_half
    dialytical_matrix[6:12, 3:12] = coefficient_matrix_tan_half

    # Make sin/cos to variable
    return ReducedRaghavanRothEquationFormatter.factor_linear_sin_cos_matrix_sum(
        dialytical_matrix, unknown_in_matrix.sin_var, unknown_in_matrix.cos_var)


# Debug code is merged to test
