import copy
import numpy as np
import sympy as sp
from fk.robots import DHEntry
from fk.robots import RobotDescription
from fk.fk_equations import ik_target_4x4, ik_target_inv_4x4
from solver.equation_utils import cast_expr_to_float
from typing import List, Tuple, Dict, Optional


class RevoluteVariable(object):
    """
    Hold the variable it self and the corresponded sin/cos/tan_half
    """
    def __init__(self, variable_symbol: sp.Symbol):
        self._var_symbol = variable_symbol
        self._cos_of_var_symbol = sp.Symbol('cos_{var}'.format(var=variable_symbol.name))
        self._sin_of_var_symbol = sp.Symbol('sin_{var}'.format(var=variable_symbol.name))
        self._tan_half_of_var_symbol = sp.Symbol('tan_half_{var}'.format(var=variable_symbol.name))

    @property
    def variable_symbol(self):
        return self._var_symbol

    @property
    def cos_var(self):
        return self._cos_of_var_symbol

    @property
    def sin_var(self):
        return self._sin_of_var_symbol

    @property
    def tan_half_var(self):
        return self._tan_half_of_var_symbol

    @staticmethod
    def convert_from_robot_unknowns(robot: RobotDescription):
        """
        Given a robot, assert its unknowns are all revolute and convert them
        into a list of revolute variables.
        """
        revolute_variable_list = list()
        for unknown_i in robot.unknowns:
            if not unknown_i.is_revolute:
                return None
            revolute_var = RevoluteVariable(unknown_i.symbol)
            revolute_variable_list.append(revolute_var)
        return revolute_variable_list

    def to_dict(self):
        datamap = dict()
        datamap['var'] = self.variable_symbol.name
        return datamap

    @staticmethod
    def load_from_dict(datamap: Dict):
        var_name: str = datamap['var']
        return RevoluteVariable(sp.Symbol(var_name))


class NumberedSymbolGenerator(object):

    def __init__(self, prefix: str):
        self._prefix = prefix
        self._idx = 0

    def next(self):
        current_idx = self._idx
        symbol_name = self._prefix + '_{idx}'.format(idx=current_idx)
        current_symbol = sp.Symbol(symbol_name)
        self._idx += 1
        return current_symbol


def modified_dh_to_classic(dh_list: List[DHEntry]) -> Optional[List[DHEntry]]:
    """
    Convert modified dh to classic dh
    The first element of modified dh must be (0, 0, 0, theta)
    """
    if len(dh_list) == 0:
        return None

    # Not empty
    dh_0 = dh_list[0]
    alpha_0_is_zero = dh_0.alpha == 0 or dh_0.alpha == sp.S.Zero
    a_0_is_zero = dh_0.a == 0 or dh_0.a == sp.S.Zero
    if (not alpha_0_is_zero) or (not a_0_is_zero):
        return None

    current_theta = dh_0.theta
    current_d = dh_0.d
    modified_idx = 1
    classic_dh_list: List[DHEntry] = list()
    while modified_idx < len(dh_list):
        current_modified_dh = dh_list[modified_idx]
        classic_dh_term = DHEntry(
            alpha=current_modified_dh.alpha,
            a=current_modified_dh.a,
            d=current_d,
            theta=current_theta
        )
        classic_dh_list.append(classic_dh_term)
        current_d = current_modified_dh.d
        current_theta = current_modified_dh.theta
        modified_idx += 1

    # The last term
    last_dh_term = DHEntry(
        alpha=0,
        a=0,
        d=current_d,
        theta=current_theta
    )
    classic_dh_list.append(last_dh_term)
    return classic_dh_list


def classic_dh_transform(dh_param: DHEntry) -> sp.Matrix:
    """
    For a dh_entry, compute its transform
    """
    return classic_dh_transform_raw(dh_param.alpha, dh_param.a, dh_param.d, dh_param.theta)


def classic_dh_transform_raw(alpha: sp.Symbol, a: sp.Symbol, d: sp.Symbol, theta: sp.Symbol) -> sp.Matrix:
    """
    Compute the forward transform T_{i+1}_to_i given the dh parameters
    :param alpha:
    :param a:
    :param d:
    :param theta:
    :return:
    """
    t = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta) * sp.cos(alpha),  sp.sin(theta) * sp.sin(alpha),  a * sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha),  a * sp.sin(theta)],
        [0,              sp.sin(alpha),                  sp.cos(alpha),                  d],
        [0,              0,                              0,                              1]
    ])
    return t


def reflected_variable_and_structure_transform_raw(
        alpha: sp.Symbol, a: sp.Symbol,
        d: sp.Symbol, theta: sp.Symbol) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Factor the forward transform T_{i+1}_to_i as a variable_transform (depends on variable)
    and a structure one that does not depend on. Both the variable/structure transform are reflected
    :param alpha:
    :param a:
    :param d:
    :param theta:
    :return:
    """
    variable_transform = sp.Matrix([
        [sp.cos(theta),  sp.sin(theta), 0, 0],
        [sp.sin(theta), -sp.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    structure_transform = sp.Matrix([
        [1, 0, 0, a],
        [0, - sp.cos(alpha), sp.sin(alpha), 0],
        [0,   sp.sin(alpha), sp.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    return variable_transform, structure_transform


def reflected_variable_and_structure_transform(dh: DHEntry) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    A handy caller for the method above
    """
    return reflected_variable_and_structure_transform_raw(alpha=dh.alpha, a=dh.a, d=dh.d, theta=dh.theta)


def print_sp_matrix_by_row(mat2print: sp.Matrix):
    for i in range(mat2print.shape[0]):
        print(mat2print[i, :])


def merge_sin_cos_sos(expr_to_merge: sp.Expr, sin_symbol: sp.Symbol, cos_symbol: sp.Symbol) -> Tuple[sp.Expr, bool]:
    """
    Merge sin and cos sum-of-square term by polynomial
    :return the simplified expression and whether it is changed from the input
    """
    if (not expr_to_merge.has(sin_symbol)) and (not expr_to_merge.has(cos_symbol)):
        return expr_to_merge, False
    poly_of_expr: sp.Poly = sp.Poly(expr_to_merge, gens=[sin_symbol, cos_symbol])
    coefficient_of_sin_square = None
    coefficient_of_cos_square = None
    for term in poly_of_expr.terms():
        mono, coefficient = term
        assert len(mono) == 2
        if mono[0] == 2 and mono[1] == 0:
            coefficient_of_sin_square = coefficient
        elif mono[1] == 2 and mono[0] == 0:
            coefficient_of_cos_square = coefficient
    if coefficient_of_sin_square is None or coefficient_of_cos_square is None:
        return expr_to_merge, False
    if sp.simplify(coefficient_of_cos_square - coefficient_of_sin_square) == sp.S.Zero:
        poly_of_expr_subst = poly_of_expr - coefficient_of_sin_square * (sin_symbol**2 + cos_symbol**2)
        return poly_of_expr_subst.as_expr() + coefficient_of_sin_square, True
    else:
        return expr_to_merge, False


def generate_classic_dh_numerical_test(
        dh_params: List[DHEntry],
        unknowns: List[RevoluteVariable],
        constant_map: Dict[sp.Symbol, float],
        n_test_cases: int = 100) -> List[Dict[sp.Symbol, float]]:
    # Compute fk
    ee_pose = sp.eye(4, 4)
    for i in range(len(dh_params)):
        transform_i = classic_dh_transform(dh_params[i])
        ee_pose = ee_pose * transform_i

    test_case_list: List[Dict[sp.Symbol, float]] = list()
    for i in range(n_test_cases):
        case_i_map = copy.deepcopy(constant_map)
        unknown_value = np.random.uniform(-np.pi, np.pi, len(unknowns))
        for j in range(len(unknowns)):
            case_i_map[unknowns[j].variable_symbol] = unknown_value[j]
            case_i_map[unknowns[j].sin_var] = np.sin(unknown_value[j])
            case_i_map[unknowns[j].cos_var] = np.cos(unknown_value[j])
        ee_pose_value = ee_pose.subs(case_i_map)
        ee_pose_np = np.zeros(shape=(4, 4))
        for r in range(4):
            for c in range(4):
                rc_value = cast_expr_to_float(ee_pose_value[r, c])
                ee_pose_np[r, c] = rc_value

        # Assign ik target
        ik_target = ik_target_4x4()
        for r in range(3):
            for c in range(4):
                rc_symbol = ik_target[r, c]
                case_i_map[rc_symbol] = float(ee_pose_np[r, c])

        # Assign inv-ik target
        inv_pose = np.linalg.inv(ee_pose_np)
        inv_ik_target = ik_target_inv_4x4()
        for r in range(3):
            rc_symbol = inv_ik_target[r, 3]
            case_i_map[rc_symbol] = inv_pose[r, 3]
        test_case_list.append(case_i_map)

    # Finished
    return test_case_list
