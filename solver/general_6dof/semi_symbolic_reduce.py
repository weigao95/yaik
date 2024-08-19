from solver.general_6dof.numerical_reduce_closure_equation import NumericalReduceInput, sincos_LME_to_tanhalf_LME
from solver.equation_utils import sp_matrix_to_dict_representation, parse_sp_matrix_from_dict
from typing import List, Tuple
import sympy as sp
import attr


@attr.s
class SemiSymbolicReduceInput(object):
    A: sp.Matrix = attr.ib()
    B: sp.Matrix = attr.ib()
    C: sp.Matrix = attr.ib()
    R_l: sp.Matrix = attr.ib()
    R_l_inv_as_symbols: sp.Matrix = attr.ib()

    def subs(self, subst_map):
        A_subst = self.A.subs(subst_map)
        B_subst = self.B.subs(subst_map)
        C_subst = self.C.subs(subst_map)
        R_l_subst = self.R_l.subs(subst_map)
        R_l_inv_as_symbols_subst = self.R_l_inv_as_symbols.subs(subst_map)
        return SemiSymbolicReduceInput(A_subst, B_subst, C_subst, R_l_subst, R_l_inv_as_symbols_subst)

    def to_dict(self):
        datamap = dict()
        datamap['A'] = sp_matrix_to_dict_representation(self.A)
        datamap['B'] = sp_matrix_to_dict_representation(self.B)
        datamap['C'] = sp_matrix_to_dict_representation(self.C)
        datamap['R_l'] = sp_matrix_to_dict_representation(self.R_l)
        datamap['R_l_inv_as_symbols'] = sp_matrix_to_dict_representation(self.R_l_inv_as_symbols)
        return datamap

    @staticmethod
    def load_from_dict(datamap):
        A = parse_sp_matrix_from_dict(datamap['A'])
        B = parse_sp_matrix_from_dict(datamap['B'])
        C = parse_sp_matrix_from_dict(datamap['C'])
        R_l = parse_sp_matrix_from_dict(datamap['R_l'])
        R_l_inv_as_symbols = parse_sp_matrix_from_dict(datamap['R_l_inv_as_symbols'])
        return SemiSymbolicReduceInput(A, B, C, R_l, R_l_inv_as_symbols)


def sincos_coefficient_to_tanhalf_coefficient_sp(A_sincos: sp.Matrix) -> sp.Matrix:
    """
    Transform
        A_sincos.dot([s4 * s5, s4 * c5, c4 * s5, c4 * c5, s4, c4, s5, c5, 1]) = 0
    into
        A_tanhalf.dot([x4**2*x5**2, x4**2*x5, x4**2, x4*x5**2, x4*x5, x4, x5**2, x5, 1]) = 0
    Given A_sincos, return the A_tanhalf
    """
    assert A_sincos.shape[1] == 9
    A_tanhalf = sp.zeros(A_sincos.shape[0], A_sincos.shape[1])
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


def symbolic_reduce_with_inv_symbol(numerical_reduce: NumericalReduceInput, lines_to_reduce: List[int]):
    lhs_A_sin_full = numerical_reduce.lhs_A_sin
    lhs_A_cos_full = numerical_reduce.lhs_A_cos
    lhs_C_const_full = numerical_reduce.lhs_C_const
    rhs_matrix_full = numerical_reduce.matrix_equation.rhs_matrix
    assert rhs_matrix_full.shape[0] == 14
    assert rhs_matrix_full.shape[1] == 8
    n_equations = lhs_A_sin_full.shape[0]
    remaining_rows = n_equations - len(lines_to_reduce)
    R_u = sp.zeros(remaining_rows, rhs_matrix_full.shape[1])
    R_l: sp.Matrix = sp.zeros(len(lines_to_reduce), rhs_matrix_full.shape[1])
    P_u_sin = sp.zeros(remaining_rows, lhs_A_sin_full.shape[1])
    P_u_cos = sp.zeros(remaining_rows, lhs_A_sin_full.shape[1])
    P_u_const = sp.zeros(remaining_rows, lhs_A_sin_full.shape[1])
    P_l_sin = sp.zeros(len(lines_to_reduce), lhs_A_sin_full.shape[1])
    P_l_cos = sp.zeros(len(lines_to_reduce), lhs_A_sin_full.shape[1])
    P_l_const = sp.zeros(len(lines_to_reduce), lhs_A_sin_full.shape[1])

    # Fill in the matrix
    reduced_row_idx_counter = 0
    remaining_row_idx_counter = 0
    for row_idx in range(n_equations):
        if row_idx in lines_to_reduce:
            R_l[reduced_row_idx_counter, :] = rhs_matrix_full[row_idx, :]
            P_l_sin[reduced_row_idx_counter, :] = lhs_A_sin_full[row_idx, :]
            P_l_cos[reduced_row_idx_counter, :] = lhs_A_cos_full[row_idx, :]
            P_l_const[reduced_row_idx_counter, :] = lhs_C_const_full[row_idx, :]
            reduced_row_idx_counter += 1
        else:
            R_u[remaining_row_idx_counter, :] = rhs_matrix_full[row_idx, :]
            P_u_sin[remaining_row_idx_counter, :] = lhs_A_sin_full[row_idx, :]
            P_u_cos[remaining_row_idx_counter, :] = lhs_A_cos_full[row_idx, :]
            P_u_const[remaining_row_idx_counter, :] = lhs_C_const_full[row_idx, :]
            remaining_row_idx_counter += 1

    # Make the new R_l_inv matrix
    R_l_inv: sp.Matrix = sp.zeros(R_l.shape[0], R_l.shape[1])
    for r in range(R_l_inv.shape[0]):
        for c in range(R_l_inv.shape[1]):
            rc_symbol_name = 'R_l_inv_{r}{c}'.format(r=r, c=c)
            R_l_inv[r, c] = sp.Symbol(rc_symbol_name)

    # Make the matrix
    R_u_dot_R_l_inv = R_u * R_l_inv
    tau_sin = P_u_sin - R_u_dot_R_l_inv * P_l_sin
    tau_cos = P_u_cos - R_u_dot_R_l_inv * P_l_cos
    tau_const = P_u_const - R_u_dot_R_l_inv * P_l_const
    return tau_sin, tau_cos, tau_const, R_l_inv, R_l


def convert_to_semi_symbolic_reduce(numerical_reduce: NumericalReduceInput, lines_to_reduce: List[int]):
    tau_sin, tau_cos, tau_const, R_l_inv, R_l = symbolic_reduce_with_inv_symbol(numerical_reduce, lines_to_reduce)
    tau_sin_tanhalf = sincos_coefficient_to_tanhalf_coefficient_sp(tau_sin)
    tau_cos_tanhalf = sincos_coefficient_to_tanhalf_coefficient_sp(tau_cos)
    tau_const_tanhalf = sincos_coefficient_to_tanhalf_coefficient_sp(tau_const)
    A, B, C = sincos_LME_to_tanhalf_LME(tau_sin_tanhalf, tau_cos_tanhalf, tau_const_tanhalf)
    A = sp.simplify(A)
    B = sp.simplify(B)
    C = sp.simplify(C)

    # Make the output
    return SemiSymbolicReduceInput(A=A, B=B, C=C, R_l=R_l, R_l_inv_as_symbols=R_l_inv)


# Debug code
def test_semi_symbolic_reduce():
    import fk.robots as robots
    import solver.general_6dof.dh_utils as dh_utils
    from solver.general_6dof.numerical_reduce_closure_equation import \
        build_reduce_inputs, generate_numerical_reduce_input, matrix_to_value, numerical_reduce
    from solver.general_6dof.select_lines_to_reduce_numerical import select_lines_to_reduce
    robot = robots.yaskawa_HC10_robot()
    dh_params = dh_utils.modified_dh_to_classic(robot.dh_params)
    revolute_vars = dh_utils.RevoluteVariable.convert_from_robot_unknowns(robot)
    test_cases = dh_utils.generate_classic_dh_numerical_test(
        dh_params, revolute_vars, robot.parameters_value, n_test_cases=50)

    reduce_input_tuple = build_reduce_inputs(robot)
    assert reduce_input_tuple is not None
    reduce_input_list, revolute_vars = reduce_input_tuple
    numerical_reduce_input_list = generate_numerical_reduce_input(reduce_input_list[4], revolute_vars)
    numerical_reduce_input = numerical_reduce_input_list[0]
    selected_lines = select_lines_to_reduce(numerical_reduce_input, test_cases)
    if selected_lines is None:
        print('There are not solution for this input')
        return
    # Try the new reduce
    semi_symbolic_reduce = convert_to_semi_symbolic_reduce(numerical_reduce_input, selected_lines)
    import solver.general_6dof.dh_utils as dh_utils
    print('Rl')
    dh_utils.print_sp_matrix_by_row(semi_symbolic_reduce.R_l)
    print('inv_RL')
    dh_utils.print_sp_matrix_by_row(semi_symbolic_reduce.R_l_inv_as_symbols)
    print('A')
    dh_utils.print_sp_matrix_by_row(semi_symbolic_reduce.A)
    print('B')
    dh_utils.print_sp_matrix_by_row(semi_symbolic_reduce.B)
    print('C')
    dh_utils.print_sp_matrix_by_row(semi_symbolic_reduce.C)


if __name__ == '__main__':
    # np.random.seed(0)
    test_semi_symbolic_reduce()
