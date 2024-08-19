from solver.polynomial_solver.intermediate import IntermediateManager, get_solution_type
from solver.polynomial_solver.simplify_poly_dict import simplify_polynomial_dict
from solver.equation_types import Unknown
import sympy as sp
from sympy import Poly
from sage.all import PolynomialRing, SR, QQ
from typing import List, Optional, Dict, Tuple, Set


def sage_tmp_symbol(i: int):
    """
    Sage cannot handle sin(x) even through it is a constant, thus we
    must subst them with new symbol
    :param i:
    :return:
    """
    v = 'sage_tmp_' + '{:04}'.format(i)
    v_symbol = sp.Symbol(v)
    return v_symbol


def parameters_not_in_unknown(parameters: Set[sp.Symbol], unknowns: List[Unknown]) -> List[sp.Symbol]:
    """
    Find the set of parameters that are not in the Unknowns. The set minus operation.
    :param parameters:
    :param unknowns:
    :return:
    """
    declared_parameters: List[sp.Symbol] = list()
    for parameter in parameters:
        # Only declare the parameter
        parameter_is_unknown = False
        for unknown in unknowns:
            if unknown.symbol.name == parameter.name:
                parameter_is_unknown = True
                break
        if not parameter_is_unknown:
            declared_parameters.append(parameter)
    return declared_parameters


def find_expr_in_polynomials(polynomials: List[sp.Expr], expr_to_test: sp.Expr) -> bool:
    """
    Found whether the expr appears in the polynomial list
    :param polynomials: the list of polynomial, note that they might not be a polynomial for other parameters
    :param expr_to_test: usually sin(x)/cos(x) for x in parameters
    :return: whether we found the expr
    """
    found_expr = False
    for poly in polynomials:
        if poly.has(expr_to_test):
            found_expr = True
            break
    return found_expr


def substitute_parameters_sin_cos(polynomials: List[sp.Expr],
                                  declared_parameters: List[sp.Symbol]) -> (Dict[sp.Symbol, sp.Expr], List[sp.Symbol]):
    """
    As sage cannot handle sin(x)/cos(x) for x in declared_parameters(not unknown), we must replace them with
    new symbols. This method performs the substitute and return the restore map and all new parameters
    :param polynomials:
    :param declared_parameters:
    :return:
    """
    new_symbol_counter = 0
    substitute_map: Dict[sp.Expr, sp.Symbol] = dict()
    restore_map: Dict[sp.Symbol, sp.Expr] = dict()
    all_parameters = list()
    # Find sin(x) and cos(x) in polynomials
    for x in declared_parameters:
        for expr_to_test in [sp.sin(x), sp.cos(x)]:
            found_expr = find_expr_in_polynomials(polynomials, expr_to_test)
            if found_expr:
                new_symbol = sage_tmp_symbol(new_symbol_counter)
                new_symbol_counter += 1
                all_parameters.append(new_symbol)
                substitute_map[expr_to_test] = new_symbol
                restore_map[new_symbol] = expr_to_test

    # Need to do substitute in the polynomials
    if len(substitute_map) > 0:
        for i in range(len(polynomials)):
            poly_i = polynomials[i]
            new_poly_i = poly_i.subs(substitute_map)
            polynomials[i] = new_poly_i

    # Now check for x themselves
    for x in declared_parameters:
        found_x = find_expr_in_polynomials(polynomials, x)
        if found_x:
            all_parameters.append(x)

    # OK
    return restore_map, all_parameters


def groebner_reduce_sage(polynomials: List[sp.Expr],
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
             The last is a string representation of SolutionMethod
    """
    # The # of equations is not sufficient
    if len(polynomials) < len(unknowns):
        return None

    # Replace the old poly with new one
    manager = IntermediateManager(unknowns, reduce_to_cosine)
    substituted_poly_list = manager.build_polynomials(polynomials)
    substituted_poly_expr = list()
    for i in range(len(substituted_poly_list)):
        substituted_poly_expr.append(substituted_poly_list[i].as_expr())

    # First declare the parameters
    declared_parameters: List[sp.Symbol] = parameters_not_in_unknown(parameters, unknowns)

    # Do substitute
    restore_map, all_parameters = substitute_parameters_sin_cos(substituted_poly_expr, declared_parameters)

    # Make them into the sage format
    F = None
    if len(all_parameters) > 0:
        parameter_list_str = ''
        for i in range(len(all_parameters)):
            name_i = all_parameters[i].name
            parameter_list_str += name_i
            if i != len(all_parameters) - 1:
                parameter_list_str += ','
        parameter_poly_ring = PolynomialRing(QQ, order='lex', names=parameter_list_str)
        F_poly = parameter_poly_ring.fraction_field()
        F_poly.inject_variables()
        F = F_poly
    else:
        F = QQ

    # Make the ring for the polynomials
    variable_list_str = ''
    for i in range(len(manager.intermediates)):
        intermediate_i = manager.intermediates[i]
        variable_list_str += intermediate_i.name
        if i != len(manager.intermediates) - 1:
            variable_list_str += ','
    poly_ring = PolynomialRing(F, order='lex', names=variable_list_str)

    # Construct the polynomial in sage
    sage_polynomials = list()
    for elem in substituted_poly_expr:
        sage_poly = elem.as_expr()._sage_()
        sage_polynomials.append(sage_poly)

    # Compute the ideal
    poly_ideal = poly_ring.ideal(sage_polynomials)
    ideal_basis = poly_ideal.groebner_basis()
    reduced_polynomial_sage = ideal_basis[-1]
    if not reduced_polynomial_sage.is_univariate():
        return None

    poly = reduced_polynomial_sage.univariate_polynomial()
    poly_dict = poly.dict()

    # Constant polynomial
    if len(poly_dict) == 1 and (0 in poly_dict):
        print('Find constant polynomial in an ideal')
        return None

    # OK, now we do process processing
    poly_dict_sp = unary_poly_dict_sage2sympy(poly_dict, all_parameters)
    poly_dict_sp = simplify_polynomial_dict(poly_dict_sp)
    poly_dict_restored = dict()
    for order in poly_dict_sp:
        numerator, denominator = poly_dict_sp[order]
        numerator_restored = numerator.subs(restore_map)
        denominator_restored = denominator.subs(restore_map)
        poly_dict_restored[order] = (numerator_restored, denominator_restored)

    solved_term = manager.restore_map[manager.intermediates[-1]]
    solved_unknown = manager.intermediates_to_unknown[manager.intermediates[-1]]
    solution_method = get_solution_type(solved_term, solved_unknown.symbol)
    return poly_dict_restored, solved_term, solved_unknown, solution_method


def unary_poly_dict_sage2sympy(
        unary_poly_dict: Dict,
        all_parameters: List[sp.Symbol]) -> Dict[int, Tuple[sp.Expr, sp.Expr]]:
    """
    Convert a uni-variable polynomial dict from sage to sympy
    :param unary_poly_dict: map from int (order) to FractionalFieldElement
    :param all_parameters: all parameters for the coefficient of the unary polynomial
    :return: the polynomial dict in sympy
    """
    coefficient_dict: Dict[int, Tuple[sp.Expr, sp.Expr]] = dict()
    for order in unary_poly_dict:
        coefficient = unary_poly_dict[order]
        # These must be multi-variable polynomial in all_parameters
        numerator = coefficient.numerator()
        denominator = coefficient.denominator()

        # Convert to sp
        numerator_sp = poly_sage2sympy(numerator, all_parameters)
        denominator_sp = poly_sage2sympy(denominator, all_parameters)

        # Simplify in sp
        coefficient_dict[order] = (numerator_sp, denominator_sp)

    # OK
    return coefficient_dict


def poly_sage2sympy(poly_sage, all_parameters: List[sp.Symbol]) -> sp.Expr:
    """
    Convert a sage multi-variable polynomial into an sympy expression.
    The gens of the sage polynomial must be included in :param all_parameters
    :param poly_sage: the sage multi-variable polynomial
    :param all_parameters: all parameters that appears in poly_sage
    :return: the sympy expression of this sage polynomial
    """
    gens_name = list()
    for x in poly_sage.args():
        gens_name.append(x._repr_())

    # Check gens are in all_parameters
    for x in gens_name:
        found = False
        for parameter in all_parameters:
            if x == parameter.name:
                found = True
                break
        assert found

    # Gens symbol
    gens_symbol = list()
    for x in gens_name:
        gens_symbol.append(sp.Symbol(x))

    # Convert into a sp.Expr
    poly_dict_sage = poly_sage.dict()
    poly_expr = sp.S.Zero
    for term in poly_dict_sage:
        coefficient = poly_dict_sage[term]
        coefficient_sp = coefficient._sympy_()
        this_monomial = sp.S.One
        if len(gens_symbol) > 1:
            for k in range(len(term)):
                this_monomial = this_monomial * sp.Pow(gens_symbol[k], term[k])
        else:
            this_monomial = this_monomial * sp.Pow(gens_symbol[0], term)
        this_monomial *= coefficient_sp
        poly_expr += this_monomial
    return poly_expr


# Debug code
# Please refer to test_polynomial_solve.py
