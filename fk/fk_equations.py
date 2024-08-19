import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr as parse_sympy_expr
from typing import List, Tuple, Optional, Set, Dict
import attr
from utility.symbolic_utils import inverse_transform, multiple_list_of_transforms
from solver.equation_types import MatrixEquation

# For check zero
zero_tolerance = sp.Symbol('zero_tolerance')  # Usually 1e-6

# The target translation
Px = sp.Symbol('Px')
Py = sp.Symbol('Py')
Pz = sp.Symbol('Pz')

# The translation at the inverse target
# use special symbols instead of direct inverse the ik_target_4x4
inv_Px = sp.Symbol('inv_Px')
inv_Py = sp.Symbol('inv_Py')
inv_Pz = sp.Symbol('inv_Pz')


# The target matrix
def ik_target_4x4() -> sp.Matrix:
    m = sp.zeros(4)
    for i in [1, 2, 3]:
        for j in [1, 2, 3]:
            v = 'r_' + '{:1}{:1}'.format(i, j)
            m[i - 1, j - 1] = sp.var(v)

    m[0, 3] = Px
    m[1, 3] = Py
    m[2, 3] = Pz
    m[3, 0] = 0
    m[3, 1] = 0
    m[3, 2] = 0
    m[3, 3] = 1
    return m


# The inverse of the target
def ik_target_inv_4x4() -> sp.Matrix:
    rotation_mat = ik_target_rotation().T
    m = sp.eye(4)
    for i in range(3):
        for j in range(3):
            m[i, j] = rotation_mat[i, j]

    m[0, 3] = inv_Px
    m[1, 3] = inv_Py
    m[2, 3] = inv_Pz
    return m


def ik_target_rotation() -> sp.Matrix:
    m = sp.zeros(3)
    for i in [1, 2, 3]:
        for j in [1, 2, 3]:
            v = 'r_' + '{:1}{:1}'.format(i, j)
            m[i - 1, j - 1] = sp.var(v)
    return m


def ik_target_symbols() -> Set[sp.Symbol]:
    """
    All possible symbols that can appear in the ik target
    :return:
    """
    symbols_set: Set[sp.Symbol] = {Px, Py, Pz, inv_Px, inv_Py, inv_Pz}
    rotation_mat = ik_target_rotation()
    for i in range(3):
        for j in range(3):
            symbols_set.add(rotation_mat[i, j])
    return symbols_set


def ik_target_parameter_bounds() -> Dict[sp.Symbol, Tuple[float, float]]:
    """
    Get the bounds for rotational target, they are restricted to [-1, 1]
    :return:
    """
    bound_dict: Dict[sp.Symbol, Tuple[float, float]] = dict()
    rotation_mat = ik_target_rotation()
    for i in range(3):
        for j in range(3):
            symbol_ij = rotation_mat[i, j]
            bound_dict[symbol_ij] = (-1.0, 1.0)
    return bound_dict


def ik_target_subst_map(ee_pose: np.ndarray) -> Dict[sp.Symbol, float]:
    """
    Given a computed ee target pose, transform it into a substitute map
    that can be directly used in sympy.subs
    """
    # The forward target
    assert ee_pose.shape[0] == 4 and ee_pose.shape[1] == 4
    subst_map: Dict[sp.Symbol, float] = dict()
    target_4x4 = ik_target_4x4()
    for r in range(3):
        for c in range(4):
            subst_map[target_4x4[r, c]] = ee_pose[r, c]

    # The inverse part
    inv_ee_translation = - ee_pose[0:3, 0:3].T.dot(ee_pose[0:3, 3])
    subst_map[inv_Px] = inv_ee_translation[0]
    subst_map[inv_Py] = inv_ee_translation[1]
    subst_map[inv_Pz] = inv_ee_translation[2]

    # OK
    return subst_map


def expr_include_translation_target(expr_to_test: sp.Expr):
    """
    Whether an expression contains P_xyz or inv_P_xyz
    :param expr_to_test:
    :return:
    """
    symbols_to_test = (Px, Py, Pz, inv_Px, inv_Py, inv_Pz)
    for symbol_i in symbols_to_test:
        if expr_to_test.has(symbol_i):
            return True
    return False


def simplify_with_ik_symbol(expr_to_simplify: sp.Expr) -> Tuple[sp.Expr, bool]:
    """
    Simplify an expression with the following constraints:
    1. ik_rotation[:, i].dot(ik_rotation[:, i]) == 1
    2. ik_rotation[i, :].dot(ik_rotation[i, :]) == 1
    return the processed expression and whether it is simplified
    """
    ik_rotation = ik_target_rotation()
    # for each row
    sos_to_one_terms: List[Tuple[sp.Expr]] = list()
    for i in range(3):
        sos_to_one_terms.append((ik_rotation[i, 0], ik_rotation[i, 1], ik_rotation[i, 2]))
        sos_to_one_terms.append((ik_rotation[0, i], ik_rotation[1, i], ik_rotation[2, i]))

    for i in range(len(sos_to_one_terms)):
        r_i1, r_i2, r_i3 = sos_to_one_terms[i]
        expand_expr: sp.Expr = sp.expand(expr_to_simplify, multinomial=True)
        coefficient_ri1 = expand_expr.coeff(r_i1 ** 2)
        coefficient_ri2 = expand_expr.coeff(r_i2 ** 2)
        coefficient_ri3 = expand_expr.coeff(r_i3 ** 2)
        if coefficient_ri1 != sp.S.Zero \
                and coefficient_ri1 - coefficient_ri2 == sp.S.Zero \
                and coefficient_ri1 - coefficient_ri3 == sp.S.Zero:
            reduced_expr = expand_expr - coefficient_ri1 * (r_i1 * r_i1 + r_i2 * r_i2 + r_i3 * r_i3)
            reduced_expr = reduced_expr + coefficient_ri1
            reduced_expr = sp.simplify(reduced_expr)
            return reduced_expr, True

    # Not changed
    return expr_to_simplify, False


@attr.s
class DHEntry(object):
    alpha: sp.Symbol = attr.ib()
    a: sp.Symbol = attr.ib()
    d: sp.Symbol = attr.ib()
    theta: sp.Symbol = attr.ib()

    def flatten(self) -> List[sp.Symbol]:
        if self.theta is None:
            return [self.alpha, self.a, self.d]
        else:
            return [self.alpha, self.a, self.d, self.theta]

    # to/from dict
    def to_dict(self):
        data_map = dict()

        # alpha, a, d, theta can be expr
        def try_save_symbol(key: str, instance):
            if isinstance(instance, sp.Expr):
                data_map[key] = str(instance)
                data_map[key + '_is_Expr'] = 'True'
            else:
                # Just an int/float
                data_map[key] = float(instance)

        # Save the model
        try_save_symbol('alpha', self.alpha)
        try_save_symbol('a', self.a)
        try_save_symbol('d', self.d)
        try_save_symbol('theta', self.theta)
        return data_map

    def from_dict(self, data_map: Dict):
        # Load the model
        def try_load(key: str):
            is_expr_key = key + '_is_Expr'
            if is_expr_key in data_map:
                str_expr = data_map[key]
                parsed_expr = parse_sympy_expr(str_expr)
                return parsed_expr
            else:
                return float(data_map[key])

        # Load the components
        self.alpha = try_load('alpha')
        self.a = try_load('a')
        self.d = try_load('d')
        self.theta = try_load('theta')

    @staticmethod
    def load_from_dict(data_map: Dict):
        dummy_symbol = sp.Symbol('dummy_symbol_tmp')
        entry = DHEntry(dummy_symbol, dummy_symbol, dummy_symbol, dummy_symbol)
        entry.from_dict(data_map)
        return entry

    def print_to_stdout(self):
        print('alpha ', self.alpha, ' a ', self.a, ' d ', self.d, ' theta ', self.theta)


@attr.s
class ForwardKinematicsOutput(object):
    """
    The result of forward kinematics, each element of the tuple is the output of
    modified_dh_transform_tuple
    """
    tuple_list: List[Tuple[Optional[sp.Matrix], sp.Matrix, Optional[sp.Matrix]]] = attr.ib()

    # For compatibility with old FKResult
    def Ts(self, i: int) -> sp.Matrix:
        tuple_i = self.tuple_list[i]
        return multiple_list_of_transforms(list(tuple_i))

    def T_ee(self) -> sp.Matrix:
        output_T = sp.eye(4)
        for i in range(len(self.tuple_list)):
            output_T = output_T * self.Ts(i)
        return output_T


def split_transformation_into_equation(
        original_rhs_list: List[Optional[sp.Matrix]],
        split_point: int,
        lhs_target: sp.Matrix) -> (sp.Matrix, sp.Matrix):
    """
    Transform the original equation in the form of
    lhs_target = T_0 * T_1 * ... T_split_point * ... T_n
    into a new form:
    inv(T_{split_point-1}) ... * inv(T_0) * lhs_target = T_split_point * ... T_n
    Note that T_split_point is included in the right-hand-side
    :param original_rhs_list: the T list mentioned above
    :param split_point: the point to split the equation
    :param lhs_target: as above
    :return: The first element is inv(T_{split_point-1}) ... * inv(T_0) * lhs_target,
             the second element is T_split_point * ... T_n
    """
    # rhs = T_split_point * ... * T_k
    assert split_point < len(original_rhs_list)
    assert split_point >= 0
    split_rhs_list: List[Optional[sp.Matrix]] = list()
    i = 0
    while split_point + i < len(original_rhs_list):
        split_rhs_list.append(original_rhs_list[split_point + i])
        i += 1
    rhs_transform = multiple_list_of_transforms(split_rhs_list)

    # lhs = inv(T[split_point - 1]) * ... inv(T[0]) * lhs_target
    lhs_list: List[Optional[sp.Matrix]] = list()
    i = split_point - 1
    while i >= 0:
        if original_rhs_list[i] is None:
            lhs_list.append(None)
        else:
            lhs_list.append(inverse_transform(original_rhs_list[i]))
        i -= 1
    lhs_list.append(lhs_target)
    lhs_transform = multiple_list_of_transforms(lhs_list)

    # OK
    return lhs_transform, rhs_transform


def build_fk_matrix_equations(fk_out: ForwardKinematicsOutput) -> List[MatrixEquation]:
    """
    Build the matrix equation from the forward kinematics output. The original equation is
    ik_target_4x4() == T_0 * T_1 * ... * T_n, we can transform it into
    inv(T_0) * ik_target_4x4() == T_1 * ... * T_n, and so on.
    :param fk_out: the output of forward kinematics
    :return: the list of matrix equations, in each equation
             lhs(Td) = inv(T_0) * ik_target_4x4()
             rhs(Ts) = T_1 * ... * T_n, here we use split_point == 0 as an example
    """
    # First flatten the transformations
    flatten_transform_list: List[Optional[sp.Matrix]] = list()
    for i in range(len(fk_out.tuple_list)):
        tuple_i = fk_out.tuple_list[i]
        for j in range(len(tuple_i)):
            flatten_transform_list.append(tuple_i[j])

    # make the split point
    result: List[MatrixEquation] = list()
    split_point = 0
    while split_point < len(flatten_transform_list):
        # This is identity, don't need to process
        if split_point >= 1 and flatten_transform_list[split_point] is None:
            split_point += 1
            continue

        # rhs = multiple_list_transforms[flatten_transform_list[split_point:end]]
        # lhs = inv(T[split_point - 1]) * ... inv(T[0]) * ik_target
        lhs_transform, rhs_transform = split_transformation_into_equation(
            flatten_transform_list, split_point, ik_target_4x4())

        # Update the counter
        result.append(MatrixEquation(lhs_transform, rhs_transform))
        split_point += 1

    # For 6-dof special cases
    if len(fk_out.tuple_list) == 6:
        lhs = inverse_transform(fk_out.Ts(0)) * ik_target_4x4() * inverse_transform(fk_out.Ts(5))
        rhs = fk_out.Ts(1) * fk_out.Ts(2) * fk_out.Ts(3) * fk_out.Ts(4)
        result.append(MatrixEquation(lhs, rhs))

        lhs = inverse_transform(fk_out.Ts(1)) * inverse_transform(
            fk_out.Ts(0)) * ik_target_4x4() * inverse_transform(fk_out.Ts(5)) * inverse_transform(fk_out.Ts(4))
        rhs = fk_out.Ts(2) * fk_out.Ts(3)
        result.append(MatrixEquation(lhs, rhs))

    # Finished
    return result
