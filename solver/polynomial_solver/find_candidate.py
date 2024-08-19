from solver.equation_types import CollectedEquations, ScalarEquation, Unknown, UnknownType
import solver.equation_utils as equation_utils
import sympy as sp
from typing import List, Tuple, Set, Dict


def find_equation_of_variables(variables: Tuple[str],
                               unknowns: List[Unknown],
                               equations: List[sp.Expr]) -> List[sp.Expr]:
    """
    Given a list of variables, find the equations whose contained variables is a subset of :param variables
    Note that many symbols are parameters, not unknown
    :param variables:
    :param unknowns:
    :param equations:
    :return:
    """
    all_unknown_names = [var.symbol.name for var in unknowns]
    equations_with_these_vars: List[sp.Expr] = list()
    for equation in equations:
        contained_symbols = equation.free_symbols
        contained_name = list()
        for symbol in contained_symbols:
            if symbol.name in all_unknown_names:
                contained_name.append(symbol.name)

        # Compare the result
        if len(contained_name) <= len(variables):
            contained = True
            for i in range(len(contained_name)):
                if contained_name[i] not in variables:
                    contained = False
                    break
            if contained:
                # This can be slow, but we should not encounter many equations here
                equations_with_these_vars.append(equation)

    # OK
    return equations_with_these_vars


def pop_variable_tuples(equations: List[sp.Expr],
                        unknowns: List[Unknown],
                        length_limit: int = 3) -> Set[Tuple[str]]:
    variable_tuples: Set[Tuple[str]] = set()
    all_unknown_names = [var.symbol.name for var in unknowns]
    for equation in equations:
        contained_symbols = equation.free_symbols
        contained_unknown: List[str] = list()
        for symbol in contained_symbols:
            if symbol.name in all_unknown_names:
                contained_unknown.append(symbol.name)

        # Ok, pop it
        if len(contained_unknown) <= length_limit:
            sorted_contained_unknown = sorted(contained_unknown)
            contained_unknown_tuple = tuple(sorted_contained_unknown)
            variable_tuples.add(contained_unknown_tuple)

    # OK
    return variable_tuples


def collect_equation_by_variables(collected_equations: CollectedEquations,
                                  unknowns: List[Unknown],
                                  length_limit: int = 3) -> Dict[Tuple[str], List[sp.Expr]]:
    """
    index the equations according to the variables that are contained in them.
    :param collected_equations:
    :param unknowns:
    :param length_limit:
    :return: the map from variable name tuples to the equations whose variables are contained in them
    """
    equations: List[sp.Expr] = list()
    one_unknown_equations = collected_equations.one_unknown_equations
    two_unknown_equations = collected_equations.two_unknown_equations
    other_equations = collected_equations.other_equations
    for equation in one_unknown_equations + two_unknown_equations + other_equations:
        if equation.is_sum_of_angle:
            continue
        lhs_minus_rhs = equation.lhs - equation.rhs
        # Expand the triangle equations to avoid sin(2 * x) or sin(x + y), where x/y might be parameters
        # Sum-of-angle terms should already be replaced by one variable
        # lhs_minus_rhs = sp.expand_trig(lhs_minus_rhs)
        equations.append(lhs_minus_rhs)

    # First pop the variable tuple
    variable_tuples = pop_variable_tuples(equations, unknowns, length_limit)
    variable_to_equations: Dict[Tuple[str], List[sp.Expr]] = dict()
    for variables in variable_tuples:
        equation_of_vars = find_equation_of_variables(variables, unknowns, equations)
        if len(equation_of_vars) >= len(variables):
            variable_to_equations[variables] = equation_of_vars

    # OK
    return variable_to_equations


# debug code
def test_candidate():
    from solver.equation_utils import default_unknowns
    unknowns = default_unknowns(3)
    th_2 = unknowns[0].symbol
    th_4 = unknowns[1].symbol
    r_21, r_22 = sp.symbols('r_21 r_22')
    expr_0 = -r_21 * sp.sin(th_4) - r_22 * sp.cos(th_4)
    expr_1 = (-r_21 * sp.cos(th_4) + r_22 * sp.sin(th_4)) * sp.sin(th_4) - sp.cos(th_2) * sp.cos(th_4) - r_22
    expr_2 = -(-r_21 * sp.cos(th_4) + r_22 * sp.sin(th_4)) * sp.cos(th_4) - sp.sin(th_4) * sp.cos(th_2)
    equ_0 = ScalarEquation(expr_0, sp.S.Zero)
    equ_1 = ScalarEquation(expr_1, sp.S.Zero)
    equ_2 = ScalarEquation(expr_2, sp.S.Zero)
    collected_equations = CollectedEquations([], [equ_0, equ_1, equ_2], [])
    var2equation = collect_equation_by_variables(collected_equations, unknowns)
    print(var2equation)


if __name__ == '__main__':
    test_candidate()
