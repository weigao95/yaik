from fk.fk_equations import ik_target_4x4
from solver.general_6dof import dh_utils
from solver.general_6dof.general_closure_equation import UnReducedMatrixClosureEquation
import solver.general_6dof.general_closure_equation as closure
from solver.general_6dof.matrix_closure_equation import ReducedRaghavanRothEquationFormatter
from solver.general_6dof.reduce_input import build_reduce_inputs, ReduceInput
import solver.equation_utils as equation_utils
from python_run_import import compute_solution_from_tanhalf_LME
import fk.robots as robots
from itertools import combinations
from utility import symbolic_utils
from solver.equation_utils import cast_expr_to_float
import sympy as sp
import numpy as np
from typing import List, Optional, Dict, Tuple
import attr


@attr.s
class NumericalReduceInput(object):
    matrix_equation: closure.UnReducedMatrixClosureEquation = attr.ib()
    lhs_A_sin: sp.Matrix = attr.ib()
    lhs_A_cos: sp.Matrix = attr.ib()
    lhs_C_const: sp.Matrix = attr.ib()
    expr_should_be_zero: List[sp.Expr] = list()

    def to_dict(self):
        datamap = dict()
        datamap['matrix_equation'] = self.matrix_equation.to_dict()
        datamap['lhs_A_sin'] = equation_utils.sp_matrix_to_dict_representation(self.lhs_A_sin)
        datamap['lhs_A_cos'] = equation_utils.sp_matrix_to_dict_representation(self.lhs_A_cos)
        datamap['lhs_C_const'] = equation_utils.sp_matrix_to_dict_representation(self.lhs_C_const)
        return datamap

    def from_dict(self, datamap):
        self.matrix_equation.from_dict(datamap['matrix_equation'])
        self.lhs_A_sin = equation_utils.parse_sp_matrix_from_dict(datamap['lhs_A_sin'])
        self.lhs_A_cos = equation_utils.parse_sp_matrix_from_dict(datamap['lhs_A_cos'])
        self.lhs_C_const = equation_utils.parse_sp_matrix_from_dict(datamap['lhs_C_const'])

    @property
    def var_in_lhs_matrix(self):
        return self.matrix_equation.variable_in_lhs_matrix

    @staticmethod
    def load_from_dict(datamap):
        matrix_equation = closure.UnReducedMatrixClosureEquation.load_from_dict(datamap['matrix_equation'])
        result = NumericalReduceInput(matrix_equation,
                                      sp.zeros(1, 1),
                                      sp.zeros(1, 1),
                                      sp.zeros(1, 1))
        result.from_dict(datamap)
        return result

    def subs(self, subst_map):
        subst_matrix_equation = self.matrix_equation.subs(subst_map)
        subst_lhs_A_sin: sp.Matrix = self.lhs_A_sin.subs(subst_map)
        subst_lhs_A_cos: sp.Matrix = self.lhs_A_cos.subs(subst_map)
        subst_lhs_C_const: sp.Matrix = self.lhs_C_const.subs(subst_map)
        return NumericalReduceInput(subst_matrix_equation, subst_lhs_A_sin, subst_lhs_A_cos, subst_lhs_C_const)


def factor_lhs_matrix(unreduced_equation: UnReducedMatrixClosureEquation, var_in_matrix: dh_utils.RevoluteVariable):
    sin_symbol = var_in_matrix.sin_var
    cos_symbol = var_in_matrix.cos_var
    return ReducedRaghavanRothEquationFormatter.factor_linear_sin_cos_matrix_sum(
        unreduced_equation.lhs_matrix,
        sin_symbol,
        cos_symbol
    )


def numerical_reduce(
        lhs_A_sin: np.ndarray,
        lhs_A_cos: np.ndarray,
        lhs_C_const: np.ndarray,
        rhs_matrix: np.ndarray,
        lines_to_reduce: List[int],
        print_invert: bool = False):
    """
    Implement Equation 8.29 in the book "Fundamentals of Robotic Mechanical Systems".
    Note that the lhs matrix P is expanded as
    P = lhs_A_sin * sin(var2solve) + lhs_A_cos * cos(var2solve) + lhs_A_const
    :return the reduced equation
    """
    assert lhs_A_sin.shape[0] == lhs_A_cos.shape[0] == lhs_C_const.shape[0] == 14
    assert lhs_A_sin.shape[1] == lhs_A_cos.shape[1] == lhs_C_const.shape[1] == 9
    assert rhs_matrix.shape[0] == 14
    assert rhs_matrix.shape[1] == 8
    assert len(lines_to_reduce) == rhs_matrix.shape[1]
    n_equations = lhs_A_sin.shape[0]
    remaining_rows = n_equations - len(lines_to_reduce)
    R_u = np.zeros(shape=(remaining_rows, rhs_matrix.shape[1]))
    R_l = np.zeros(shape=(len(lines_to_reduce), rhs_matrix.shape[1]))
    P_u_sin = np.zeros(shape=(remaining_rows, lhs_A_sin.shape[1]))
    P_u_cos = np.zeros(shape=(remaining_rows, lhs_A_sin.shape[1]))
    P_u_const = np.zeros(shape=(remaining_rows, lhs_A_sin.shape[1]))
    P_l_sin = np.zeros(shape=(len(lines_to_reduce), lhs_A_sin.shape[1]))
    P_l_cos = np.zeros(shape=(len(lines_to_reduce), lhs_A_sin.shape[1]))
    P_l_const = np.zeros(shape=(len(lines_to_reduce), lhs_A_sin.shape[1]))

    # Fill in the matrix
    reduced_row_idx_counter = 0
    remaining_row_idx_counter = 0
    for row_idx in range(n_equations):
        if row_idx in lines_to_reduce:
            R_l[reduced_row_idx_counter, :] = rhs_matrix[row_idx, :]
            P_l_sin[reduced_row_idx_counter, :] = lhs_A_sin[row_idx, :]
            P_l_cos[reduced_row_idx_counter, :] = lhs_A_cos[row_idx, :]
            P_l_const[reduced_row_idx_counter, :] = lhs_C_const[row_idx, :]
            reduced_row_idx_counter += 1
        else:
            R_u[remaining_row_idx_counter, :] = rhs_matrix[row_idx, :]
            P_u_sin[remaining_row_idx_counter, :] = lhs_A_sin[row_idx, :]
            P_u_cos[remaining_row_idx_counter, :] = lhs_A_cos[row_idx, :]
            P_u_const[remaining_row_idx_counter, :] = lhs_C_const[row_idx, :]
            remaining_row_idx_counter += 1

    # Invert R_l
    R_l_inv = None
    if print_invert:
        print(R_l)
    try:
        R_l_inv = np.linalg.inv(R_l)
    except:
        return None

    assert R_l_inv is not None
    R_u_dot_R_l_inv = R_u.dot(R_l_inv)
    tau_sin = P_u_sin - R_u_dot_R_l_inv.dot(P_l_sin)
    tau_cos = P_u_cos - R_u_dot_R_l_inv.dot(P_l_cos)
    tau_const = P_u_const - R_u_dot_R_l_inv.dot(P_l_const)
    return tau_sin, tau_cos, tau_const


def sincos_coefficient_to_tanhalf_coefficient(A_sincos: np.ndarray) -> np.ndarray:
    """
    Transform
        A_sincos.dot([s4 * s5, s4 * c5, c4 * s5, c4 * c5, s4, c4, s5, c5, 1]) = 0
    into
        A_tanhalf.dot([x4**2*x5**2, x4**2*x5, x4**2, x4*x5**2, x4*x5, x4, x5**2, x5, 1]) = 0
    Given A_sincos, return the A_tanhalf
    """
    assert A_sincos.shape[1] == 9
    A_tanhalf = np.zeros_like(A_sincos)
    A_tanhalf[:, 0] = A_sincos[:, 3] - A_sincos[:, 5] - A_sincos[:, 7] + A_sincos[:, 8]
    A_tanhalf[:, 1] = (- 2) * (A_sincos[:, 2] - A_sincos[:, 6])
    A_tanhalf[:, 2] = - A_sincos[:, 3] - A_sincos[:, 5] + A_sincos[:, 7] + A_sincos[:, 8]
    A_tanhalf[:, 3] = (- 2) * (A_sincos[:, 1] - A_sincos[:, 4])
    A_tanhalf[:, 4] = 4 * A_sincos[:, 0]
    A_tanhalf[:, 5] = 2 * (A_sincos[:, 1] + A_sincos[:, 4])
    A_tanhalf[:, 6] = - A_sincos[:, 3] + A_sincos[:, 5] - A_sincos[:, 7] + A_sincos[:, 8]
    A_tanhalf[:, 7] = 2 * (A_sincos[:, 2] + A_sincos[:, 6])
    A_tanhalf[:, 8] = A_sincos[:, 3] + A_sincos[:, 5] + A_sincos[:, 7] + A_sincos[:, 8]
    return A_tanhalf


def sincos_LME_to_tanhalf_LME(A_sin: np.ndarray, A_cos: np.ndarray, C_const: np.ndarray):
    """
    Converse a linear matrix equality (LME)
        (A_sin * sin(theta) + A_cos * cos(theta) + C_const).dot(
            [x4**2*x5**2, x4**2*x5, x4**2, x4*x5**2, x4*x5, x4, x5**2, x5, 1]) = 0
    Into another LME:
        (A * x**2 + B * x + C).dot(
            [x4**2*x5**2, x4**2*x5, x4**2, x4*x5**2, x4*x5, x4, x5**2, x5, 1]) = 0
    where x = tan(theta/2)
    """
    A = C_const - A_cos
    B = 2 * A_sin
    C = C_const + A_cos
    return A, B, C


def generate_numerical_reduce_input(
        reduce_input: ReduceInput,
        revolute_vars: List[dh_utils.RevoluteVariable]) -> List[NumericalReduceInput]:
    # Generate the lhs
    lhs_matrix = reduce_input.lhs_matrix
    rhs_matrix = reduce_input.rhs_matrix
    var2remove = reduce_input.rhs_var_to_remove
    raw_closure = closure.closure_equation_from_matrix(
        lhs_matrix, rhs_matrix, revolute_vars, var2remove)
    if raw_closure is None:
        return list()

    # Build the matrix equation
    assert len(raw_closure.lhs_unknowns) == 3
    result_list: List[NumericalReduceInput] = list()
    for i in range(len(raw_closure.lhs_unknowns)):
        lhs_unknowns = raw_closure.lhs_unknowns
        var_in_lhs_matrix = lhs_unknowns[i]
        i_p_1_idx = (i + 1) % len(lhs_unknowns)
        i_p_2_idx = (i + 2) % len(lhs_unknowns)
        lhs_poly_gen = [lhs_unknowns[i_p_1_idx].sin_var, lhs_unknowns[i_p_1_idx].cos_var,
                        lhs_unknowns[i_p_2_idx].sin_var, lhs_unknowns[i_p_2_idx].cos_var]
        matrix_equation = closure.closure_equations_matrix_form(raw_closure, lhs_gen=lhs_poly_gen)
        A_sin, A_cos, C_const, nonlinear_coefficient = factor_lhs_matrix(matrix_equation, var_in_lhs_matrix)
        # Make output
        numerical_reduce_input = NumericalReduceInput(
            matrix_equation=matrix_equation,
            lhs_A_sin=A_sin, lhs_A_cos=A_cos, lhs_C_const=C_const)
        numerical_reduce_input.expr_should_be_zero = nonlinear_coefficient
        result_list.append(numerical_reduce_input)

    # Done
    return result_list


def matrix_to_value(sp_matrix: sp.Matrix, test_case_in: Dict[sp.Symbol, float]) -> np.ndarray:
    sp_mat_subst = sp_matrix.subs(test_case_in)
    return np.array(sp_mat_subst).astype(np.float64)


def numerical_solve(
        A_sin: np.ndarray,
        A_cos: np.ndarray,
        C_const: np.ndarray,
        N_rhs_value: np.ndarray,
        lines2reduce: List[int],
        return_coupled_3_solution: bool = False):
    # Try reduce
    numerical_reduce_out = numerical_reduce(
        A_sin, A_cos, C_const, N_rhs_value, lines2reduce)
    if numerical_reduce_out is None:
        return list()
    tau_sin, tau_cos, tau_const = numerical_reduce_out
    tau_sin_tanhalf = sincos_coefficient_to_tanhalf_coefficient(tau_sin)
    tau_cos_tanhalf = sincos_coefficient_to_tanhalf_coefficient(tau_cos)
    tau_const_tanhalf = sincos_coefficient_to_tanhalf_coefficient(tau_const)
    A, B, C = sincos_LME_to_tanhalf_LME(tau_sin_tanhalf, tau_cos_tanhalf, tau_const_tanhalf)
    var_solution = compute_solution_from_tanhalf_LME(A, B, C, return_coupled_3_solution)

    # Convert None encoding of invalid to empty-encoding
    if var_solution is None:
        return list()
    return var_solution


def symbolic_solve(
        A_sin: sp.Matrix, A_cos: sp.Matrix, C_const: sp.Matrix,
        matrix_equation: closure.UnReducedMatrixClosureEquation,
        test_case_to_run: Dict[sp.Symbol, float], lines2reduce: List[int],
        return_coupled_3_solution: bool = False) -> List[float]:
    """
    Solve the equation in the form of
    lhs_matrix * lhs_terms = rhs_matrix = rhs_terms, where
        lhs_matrix = A_sin * sin(x) + A_cos * cos(x) + C_const
    In reducing this equation (or eliminate the rhs_terms), we would use the lines in the input.
    """
    A_sin_value = matrix_to_value(A_sin, test_case_to_run)
    A_cos_value = matrix_to_value(A_cos, test_case_to_run)
    C_const_value = matrix_to_value(C_const, test_case_to_run)
    N_rhs_value = matrix_to_value(matrix_equation.rhs_matrix, test_case_to_run)

    # Given to numerical solver
    return numerical_solve(
        A_sin_value, A_cos_value, C_const_value,
        N_rhs_value, lines2reduce, return_coupled_3_solution)


def verify_solve(
        A_sin: sp.Matrix, A_cos: sp.Matrix, C_const: sp.Matrix,
        matrix_equation: closure.UnReducedMatrixClosureEquation,
        test_case_to_run: Dict[sp.Symbol, float], lines2reduce: List[int],
        try_coupled_3_solution: bool = False):
    # Compute the solution
    var_solution = symbolic_solve(
        A_sin, A_cos, C_const, matrix_equation, test_case_to_run, lines2reduce,
        return_coupled_3_solution=try_coupled_3_solution)
    if var_solution is None:
        return False
    var_in_lhs_matrix_value = float(test_case_to_run[matrix_equation.variable_in_lhs_matrix.variable_symbol])
    lhs_var_0 = float(test_case_to_run[matrix_equation.lhs_gen_vars[0].variable_symbol])
    lhs_var_1 = float(test_case_to_run[matrix_equation.lhs_gen_vars[1].variable_symbol])

    # Only one solution
    if not try_coupled_3_solution:
        for solution_j in var_solution:
            if abs(solution_j - var_in_lhs_matrix_value) < 1e-5:
                return True

        # Cannot find
        return False

    # All three solution
    print('The input solution')
    print(var_in_lhs_matrix_value, lhs_var_0, lhs_var_1)
    for solution_j in var_solution:
        print('Solution j', solution_j)
        if abs(solution_j[0] - var_in_lhs_matrix_value) > 1e-5:
            continue
        if abs(solution_j[1] - lhs_var_0) > 1e-5:
            continue
        if abs(solution_j[2] - lhs_var_1) > 1e-5:
            continue
        # Find solution
        return True

    # Cannot find
    return False


def sin_cos_to_tan_half_draft():
    s4, c4, s5, c5 = sp.symbols('s4 c4 s5 c5')
    x4, x5 = sp.symbols('x4 x5')  # t4_half, t5_half
    subst_map = {
        s4: 2 * x4 / (1 + x4 ** 2),
        c4: (1 - x4 ** 2) / (1 + x4 ** 2),
        s5: 2 * x5 / (1 + x5 ** 2),
        c5: (1 - x5 ** 2) / (1 + x5 ** 2)
    }

    terms = sp.Matrix([s4 * s5, s4 * c5, c4 * s5, c4 * c5, s4, c4, s5, c5, 1])
    coefficient_list = list()
    for i in range(terms.shape[0]):
        coefficient_list.append(sp.Symbol('A_{i}'.format(i=i)))
    coefficient = sp.Matrix(coefficient_list)
    dot_output: sp.Expr = coefficient.dot(terms)
    dot_output_subst = dot_output.subs(subst_map) * (1 + x4 ** 2) * (1 + x5 ** 2)
    dot_output_subst = sp.simplify(dot_output_subst)
    dot_output_subst = sp.expand(dot_output_subst, multinomial=True)
    poly_gen = [x4, x5]
    dot_output_poly: sp.Poly = sp.Poly(dot_output_subst, gens=[x4, x5])
    for mono, mono_coeff in dot_output_poly.terms():
        mono_expr = sp.S.One
        for k in range(len(mono)):
            gen_k = poly_gen[k]
            order_k = mono[k]
            mono_expr = mono_expr * (gen_k ** order_k)
        print('********************')
        print(mono, mono_expr)
        print(mono_coeff)


# Debug code
# Please refer to general_6dof.test_general_6dof and select_lines_to_reduce_numerical
