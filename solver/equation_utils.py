import sympy as sp
from sympy.parsing.sympy_parser import parse_expr as parse_sympy_expr
from solver.equation_types import ScalarEquation, MatrixEquation, CollectedEquations, ScalarEquationType
from solver.equation_types import UnknownType, Unknown, TranslationalEquation, EquationInput, SumOfSquareHint
from typing import List, Dict, Set, Optional


def make_default_unknown(unknown_idx: int, is_revolute: bool = True) -> Unknown:
    """
    Construct the unknown with given name
    """
    if is_revolute:
        unknown_symbol = sp.Symbol('th_' + '{:1}'.format(unknown_idx))
        unknown = Unknown(unknown_symbol, UnknownType.Revolute.name)
        return unknown
    else:
        unknown_symbol = sp.Symbol('d_' + '{:1}'.format(unknown_idx))
        unknown = Unknown(unknown_symbol, UnknownType.Translational.name)
        return unknown


def default_unknowns(num_dofs: int) -> List[Unknown]:
    """
    Generate th_0 ... th_n, each of them are revolute unknown with region [0, 2 pi)
    :param num_dofs:
    :return:
    """
    unknowns = list()
    for i in range(num_dofs):
        unknown_i = make_default_unknown(i)
        unknowns.append(unknown_i)
    return unknowns


def default_unknowns_with_offset(num_dofs: int, offset: int = 1) -> List[Unknown]:
    """
    Generate th_0 ... th_n, each of them are revolute unknown with region [0, 2 pi)
    :param num_dofs:
    :return:
    """
    unknowns = list()
    for i in range(num_dofs):
        unknown_i = make_default_unknown(i + offset)
        unknowns.append(unknown_i)
    return unknowns


def find_unknown(unknowns: List[Unknown], name_to_find: str) -> Optional[Unknown]:
    """
    For a given name :param name_to_find, check whether it exists an unknown in the :param unknowns list.
    If so, return that unknown; else return None.
    :param unknowns:
    :param name_to_find:
    :return:
    """
    # Iterate over the unknowns
    for i in range(len(unknowns)):
        unknown_i = unknowns[i]
        if unknown_i.symbol.name == name_to_find:
            return unknown_i

    # Cannot find
    return None


def find_unknown_idx(unknowns: List[Unknown], name_to_find: str) -> int:
    """
    For a given name :param name_to_find, check whether it exists an unknown in the :param unknowns list.
    If so, return the index corresponded to that unknown; else return -1.
    :param unknowns:
    :param name_to_find:
    :return:
    """
    for i in range(len(unknowns)):
        if unknowns[i].symbol.name == name_to_find:
            return i
    return -1


def all_symbols_in_expr(sp_expr: sp.Expr) -> Set[sp.Symbol]:
    """
    All the symbols in the given expression. In our case sp_expr is a polynomial with sin/cos,
    thus just free symbols.
    :param sp_expr:
    :return:
    """
    return sp_expr.free_symbols


def all_symbols_in_equation(lhs: sp.Expr, rhs: sp.Expr) -> Set[sp.Symbol]:
    """
    Get all the symbols in a scalar equation, note that the return is unique
    :param lhs:
    :param rhs:
    :return:
    """
    lhs_symbols = all_symbols_in_expr(lhs)
    rhs_symbols = all_symbols_in_expr(rhs)
    lhs_symbols.update(rhs_symbols)
    return lhs_symbols


def all_symbols_in_matrix_equations(matrix_equations: List[MatrixEquation]) -> Set[sp.Symbol]:
    """
    Get all the symbols in all the equations, note that the return is unique
    :param matrix_equations:
    :return:
    """
    all_symbols = set()
    for i in range(len(matrix_equations)):
        equ_i = matrix_equations[i]
        for r in range(3):
            for c in range(4):
                lhs_rc: sp.Expr = equ_i.Ts[r, c]
                rhs_rc: sp.Expr = equ_i.Td[r, c]
                all_symbols.update(lhs_rc.free_symbols)
                all_symbols.update(rhs_rc.free_symbols)
    return all_symbols


def count_unknowns(lhs: sp.Expr, rhs: sp.Expr, unknowns: List[sp.Symbol]) -> int:
    """
    Count how many of the unknowns appear in either lhs or rhs. If the return
    is 0, then the lhs and rhs doesn't contains anything in unknowns.
    :param lhs:
    :param rhs:
    :param unknowns:
    :return:
    """
    n = 0
    for i in range(len(unknowns)):
        if lhs.has(unknowns[i]) or rhs.has(unknowns[i]):
            n += 1
    return n


def count_unknowns_expr(expr_to_count: sp.Expr, unknowns: List[sp.Symbol]) -> int:
    """
    Count how many unknowns appear in the given expression. If the return
    is 0, then the expression doesn't contains anything in unknowns.
    :param expr_to_count:
    :param unknowns:
    :return:
    """
    n = 0
    for i in range(len(unknowns)):
        if expr_to_count.has(unknowns[i]):
            n += 1
    return n


def append_intersecting_axis_equations_to_input(
        translational_equations: List[TranslationalEquation],
        equation_input: EquationInput,
        cos_equations: Optional[List[ScalarEquation]] = None):
    """
    Add all the equations from translational equations into the equation input
    :param translational_equations:
    :param equation_input:
    :param cos_equations:
    :return:
    """
    for elem in translational_equations:
        # The flatten equations
        for xyz in [elem.x, elem.y, elem.z]:
            equation_input.scalar_equations.append(xyz)
        # Sum-of-square hint
        sos_hint = SumOfSquareHint(elem.x, elem.y, elem.z)
        equation_input.sum_of_square_hint.append(sos_hint)

    # No cos equations, just return
    if cos_equations is None or len(cos_equations) == 0:
        return

    # Process cos equations
    for elem in cos_equations:
        equation_input.scalar_equations.append(elem)


def sort_equations(equ_list: List[ScalarEquation]) -> List[ScalarEquation]:
    """
    Sort the equation according to the number of operations
    :param equ_list:
    :return:
    """
    ops_count_to_expr: Dict[int, List[ScalarEquation]] = dict()
    for i in range(len(equ_list)):
        e_i: ScalarEquation = equ_list[i]
        count = int(sp.count_ops(e_i.lhs)) + int(sp.count_ops(e_i.rhs))
        if count not in ops_count_to_expr.keys():
            ops_count_to_expr[count] = list()
        ops_count_to_expr[count].append(e_i)

    # Sort according to ops
    keys = ops_count_to_expr.keys()
    keys = sorted(keys, reverse=False)
    sorted_expr: List[ScalarEquation] = list()
    for key in keys:
        sorted_expr.extend(ops_count_to_expr[key])
    return sorted_expr


def collect_scalar_equations(matrix_equations: List[MatrixEquation]) -> List[ScalarEquation]:
    """
    Flatten the matrix equations to get scalar equations.
    Note the last row of the matrix equations are ignored.
    :param matrix_equations:
    :return:
    """
    result = list()
    for mat_equ in matrix_equations:
        mat_lhs = mat_equ.Td
        mat_rhs = mat_equ.Ts
        for i in range(3):  # Don't need to care the last row
            for j in range(4):
                lhs = mat_lhs[i, j]
                rhs = mat_rhs[i, j]
                # Ignore the useless terms
                # if sp.simplify(rhs - lhs) == 0:
                #    continue
                if j != 3:
                    expr_ij = ScalarEquation(lhs, rhs, ScalarEquationType.Rotation.name)
                else:
                    expr_ij = ScalarEquation(lhs, rhs, ScalarEquationType.Translation.name)
                result.append(expr_ij)
    return result


def scalar_equation_by_unknowns(
        scalar_equations: List[ScalarEquation],
        unknowns: List[sp.Symbol]) -> CollectedEquations:
    """
    Classify the equations according to the number of unknowns.
    The equations with 1 or 2 unknowns are seperated out.
    :param scalar_equations:
    :param unknowns:
    :return:
    """
    one_unknown_equations: List[ScalarEquation] = list()
    two_unknown_equations: List[ScalarEquation] = list()
    other_equations: List[ScalarEquation] = list()
    for equ_i in scalar_equations:
        n_unknowns = count_unknowns(equ_i.lhs, equ_i.rhs, unknowns)
        if n_unknowns == 1:
            # Strict de-duplication. Need to consider at the last stage of solving,
            # many equations would have a small number of variables.
            # This is O(n^2), a slow operation.
            if equ_i not in one_unknown_equations:
                one_unknown_equations.append(equ_i)
        elif n_unknowns == 2:
            if equ_i not in two_unknown_equations:
                two_unknown_equations.append(equ_i)
        elif n_unknowns >= 3:
            # This append is not that important
            other_equations.append(equ_i)
        else:
            pass

    # Make the result
    result = CollectedEquations(one_unknown_equations=one_unknown_equations,
                                two_unknown_equations=two_unknown_equations,
                                other_equations=other_equations)
    return result


def collect_equations(matrix_equations: List[MatrixEquation], unknowns: List[sp.Symbol]) -> CollectedEquations:
    scalar_equs = collect_scalar_equations(matrix_equations)
    return scalar_equation_by_unknowns(scalar_equs, unknowns)


def append_equation_no_duplicate(unique_equations: List[ScalarEquation], new_equation: ScalarEquation) -> bool:
    """
    Use sp.simplify to check the equation is unique, this is expensive
    :param unique_equations:
    :param new_equation:
    :return:
    """
    found = False
    for expr in unique_equations:
        lhs = expr.lhs
        rhs = expr.rhs
        # Check minus
        this_lhs_minus_rhs = lhs - rhs
        other_lhs_minus_rhs = new_equation.lhs - new_equation.rhs
        minus_result = this_lhs_minus_rhs - other_lhs_minus_rhs
        if sp.simplify(minus_result) == sp.S.Zero:
            found = True
            break

        # Check negative
        plus_result = this_lhs_minus_rhs + other_lhs_minus_rhs
        if sp.simplify(plus_result) == sp.S.Zero:
            found = True
            break

    # Append if not found
    if not found:
        unique_equations.append(new_equation)
        return True
    return False


def append_expr_no_duplicate(unique_equations: List[sp.Expr], new_expr: sp.Expr) -> bool:
    """
    The same functionality as above, but for expression
    :param unique_equations:
    :param new_expr:
    :return:
    """
    found = False
    for expr in unique_equations:
        minus_result = new_expr - expr
        if sp.simplify(minus_result) == sp.S.Zero:
            found = True
            break

        # Check negative
        plus_result = new_expr + expr
        if sp.simplify(plus_result) == sp.S.Zero:
            found = True
            break

    # Append if not found
    if not found:
        unique_equations.append(new_expr)
        return True
    return False


def expression_list_to_string_representation(expr_list: List[sp.Expr]) -> List[str]:
    """
    Convert list of sp.Expr into list string representation
    :param expr_list: A list of sp.Expr
    :return:
    """
    string_list: List[str] = list()
    for expr in expr_list:
        expr_str = str(expr)
        string_list.append(expr_str)
    return string_list


def parse_string_expressions(string_expr_list: List[str]) -> List[sp.Expr]:
    """
    Given a list of string representations of the expressions, parse them into sp.Expr list
    :param string_expr_list:
    :return:
    """
    expr_list = list()
    for string_i in string_expr_list:
        expr_i = parse_sympy_expr(string_i)
        # Bool constant fix
        if type(expr_i) == bool:
            expr_i = sp.S.BooleanTrue & expr_i
        expr_list.append(expr_i)
    return expr_list


def cast_expr_to_float(expr_to_cast: sp.Expr) -> Optional[float]:
    """
    Try to cast a sp.Expr into float, return None if we cannot cast it.
    Example: cast_expr_to_float(sp.pi/2) = 1.57...
    :param expr_to_cast: the expression to process
    """
    casted_float = None
    try:
        casted_float = float(expr_to_cast)
    except TypeError:
        pass
    return casted_float


def sp_matrix_to_dict_representation(matrix_to_serialize: sp.Matrix) -> Dict:
    rows, cols = matrix_to_serialize.shape
    assert (rows is not None) and rows >= 1
    assert (cols is not None) and cols >= 1
    flatten_mat: List[sp.Expr] = list()
    for r in range(rows):
        for c in range(cols):
            expr_rc = matrix_to_serialize[r, c]
            flatten_mat.append(expr_rc)

    # Into dict
    data_map = dict()
    data_map['rows'] = rows
    data_map['cols'] = cols
    data_map['flatten_data'] = expression_list_to_string_representation(flatten_mat)
    return data_map


def parse_sp_matrix_from_dict(data_map: Dict) -> sp.Matrix:
    assert 'rows' in data_map
    assert 'cols' in data_map
    assert 'flatten_data' in data_map
    rows = int(data_map['rows'])
    cols = int(data_map['cols'])
    matrix = sp.zeros(rows, cols)
    flatten_mat = parse_string_expressions(data_map['flatten_data'])

    # Load it
    offset = 0
    for r in range(rows):
        for c in range(cols):
            rc_expr = flatten_mat[offset]
            matrix[r, c] = rc_expr
            offset += 1
    return matrix
