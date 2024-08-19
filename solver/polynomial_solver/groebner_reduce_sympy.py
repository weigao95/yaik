from solver.polynomial_solver.intermediate import IntermediateManager, get_solution_type
from solver.equation_types import Unknown
import sympy as sp
from sympy import Poly
from sympy import groebner
from typing import List, Optional, Dict, Tuple, Set


def groebner_reduce_sympy(polynomials: List[sp.Expr],
                          unknowns: List[Unknown],
                          parameters: Set[sp.Symbol],
                          reduce_to_cosine: bool = True) -> Optional[Tuple[Dict, sp.Expr, Unknown, str]]:
    """
    :param polynomials:
    :param unknowns:
    :param parameters: everything symbol appear in the polynomials, include the unknowns
    :param reduce_to_cosine:
    :return: The first is the uni-variable polynomial dict, from order to coefficient
             The second element is the term that is the intermediate, usually sin(x)/cos(x)/x where x is a unknown
             The third is the unknown itself
    """
    # The # of equations is not sufficient
    if len(polynomials) < len(unknowns):
        return None

    # Replace the old poly with new one
    manager = IntermediateManager(unknowns, reduce_to_cosine)
    substituted_poly_list: List[Poly] = manager.build_polynomials(polynomials)

    # OK, invoke the solver
    basis = groebner(substituted_poly_list, gens=manager.intermediates, order='lex')
    polys_in_basis = basis.polys
    reduced_poly: Poly = polys_in_basis[-1]

    # The reduced_poly must be a non-zero, uni-variable polynomial
    poly_dict = reduced_poly.as_dict()
    if len(poly_dict) == 0:
        return None

    # Process of the dict
    nonzero_exp_idx = len(manager.intermediates) - 1
    unary_poly_dict: Dict[int, Tuple[sp.Expr, sp.Expr]] = dict()
    for term in poly_dict:
        for k in range(len(term)):
            # Should only be non-zero at the last intermediates
            if k == nonzero_exp_idx:
                continue
            # This polynomial has more than one intermediate
            # Thus, not valid
            if term[k] != 0:
                return None
        exponent: int = term[nonzero_exp_idx]
        coefficient: sp.Expr = poly_dict[term]
        unary_poly_dict[exponent] = (coefficient, sp.S.One)

    # OK
    solved_term = manager.restore_map[manager.intermediates[-1]]
    solved_unknown = manager.intermediates_to_unknown[manager.intermediates[-1]]
    solution_method = get_solution_type(solved_term, solved_unknown.symbol)
    return unary_poly_dict, solved_term, solved_unknown, solution_method


# Debug code
# Please refer to test_polynomial_solve.py
