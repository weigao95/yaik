import sympy as sp
from typing import Tuple, Dict, List


def count_polynomial_operations(poly_dict: Dict[int, Tuple[sp.Expr, sp.Expr]]) -> Tuple[int, int]:
    """
    Count the number of expression is the poly dict
    :param poly_dict:
    :return: the first element is number of operations
             the second element is the number of division
    """
    n_operations = 0
    n_div = 0
    for order in poly_dict:
        numerator, denominator = poly_dict[order]
        n_operations += sp.count_ops(numerator) + sp.count_ops(denominator)
        if denominator.is_Number or denominator.is_NumberSymbol:
            pass
        else:
            n_div += 1
    return n_operations, n_div


def multiply_polynomial_with_expr(
        poly_dict: Dict[int, Tuple[sp.Expr, sp.Expr]],
        expr_to_multiple: sp.Expr) -> Dict[int, Tuple[sp.Expr, sp.Expr]]:
    """
    For a given polynomial represented by :param poly_dict, do multiplication with :param expr_to_multiple
    and simplify the resulting expression.
    We want to compute f(x) == 0, where f is a polynomial. Now we transform it into g f(x) == 0,
    Note that g does NOT depend on x, although it may contains other symbols.
    :param poly_dict:
    :param expr_to_multiple:
    :return:
    """
    new_dict: Dict[int, Tuple[sp.Expr, sp.Expr]] = dict()
    for order in poly_dict:
        numerator, denominator = poly_dict[order]
        multiplied_numerator = expr_to_multiple * numerator
        c, p, q = sp.cancel((multiplied_numerator, denominator))
        new_numerator = c * p.as_expr()
        new_denominator = q.as_expr()
        # Use factor to make the equation shorter
        new_numerator = sp.simplify(new_numerator)
        new_numerator = sp.factor(new_numerator)
        new_denominator = sp.simplify(new_denominator)
        new_denominator = sp.factor(new_denominator)
        new_dict[order] = (new_numerator, new_denominator)
    return new_dict


def select_simplest_polynomial(
        candidate_list: List[Dict[int, Tuple[sp.Expr, sp.Expr]]]) -> Dict[int, Tuple[sp.Expr, sp.Expr]]:
    """
    Given the candidate the their operations, select the simplest polynomial
    :param candidate_list:
    :return:
    """
    assert len(candidate_list) > 0
    candidate_ops_list: List[Tuple[int, int]] = list()
    for i in range(len(candidate_list)):
        candidate_i = candidate_list[i]
        n_ops, n_div = count_polynomial_operations(candidate_i)
        candidate_ops_list.append((n_ops, n_div))

    # The cost is a linear sum of ops and div
    def compute_cost(num_ops: int, num_div: int, div_weight: int = 1000) -> int:
        return num_ops + div_weight * num_div

    # Start the loop
    min_cost_idx = None
    min_cost = None
    for i in range(len(candidate_ops_list)):
        n_ops, n_div = candidate_ops_list[i]
        this_cost = compute_cost(n_ops, n_div)
        if min_cost_idx is None:
            min_cost_idx = i
            min_cost = this_cost
        else:
            if this_cost < min_cost:
                min_cost = this_cost
                min_cost_idx = i

    # Finish the selection
    assert min_cost_idx is not None
    assert 0 <= min_cost_idx < len(candidate_list)
    return candidate_list[min_cost_idx]


def simplify_polynomial_dict(poly_dict: Dict[int, Tuple[sp.Expr, sp.Expr]]) -> Dict[int, Tuple[sp.Expr, sp.Expr]]:
    """
    We want to solve a uni-variable polynomial equation f(x, parameters) == 0 with respect to x. This polynomial is
    represented by a map from order (of x) to coefficients. We can construct a new polynomial in the form of:
       g(parameters) f(x, parameters) == 0
    which might be simpler than the original one.
    This function try some of the g(parameters) and return the simplest polynomial.
    :param poly_dict:
    :return:
    """
    # Multi with
    multiply_with_list: List[sp.Expr] = list()
    for order in poly_dict:
        _, denominator = poly_dict[order]
        if denominator.is_Number or denominator.is_NumberSymbol:
            continue
        multiply_with_list.append(denominator)

    # Build new polynomials and count the operations
    candidate_list: List[Dict] = list()
    candidate_list.append(poly_dict)
    for i in range(len(multiply_with_list)):
        expr_i = multiply_with_list[i]
        poly_i_dict = multiply_polynomial_with_expr(poly_dict, expr_i)
        candidate_list.append(poly_i_dict)

    # Do selection
    return select_simplest_polynomial(candidate_list)


def test_simplify_polynomial_dict():
    from sympy.abc import x, y, z
    poly_dict = {0: (x, y + z), 1: (sp.S.One, sp.S.One)}
    simplified_poly_dict = simplify_polynomial_dict(poly_dict)
    print(simplified_poly_dict)


if __name__ == '__main__':
    test_simplify_polynomial_dict()
