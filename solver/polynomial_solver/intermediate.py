from solver.equation_types import Unknown, UnknownType
from solver.solved_variable import SolutionMethod
from typing import List, Dict
import sympy as sp
from sympy import Poly


def get_solution_type(solved_term: sp.Expr, solved_symbol: sp.Expr) -> str:
    """
    Depends on whether :param solved_term is sin/cos of x of x itself, where x is :param solved_symbol
    :param solved_term:
    :param solved_symbol:
    :return:
    """
    if solved_term == sp.sin(solved_symbol):
        return SolutionMethod.PolynomialSin.name
    elif solved_term == sp.cos(solved_symbol):
        return SolutionMethod.PolynomialCos.name
    elif solved_term == solved_symbol:
        return SolutionMethod.PolynomialDirect.name
    else:
        raise RuntimeError("Unknown solution type")


def intermediate_symbol(i: int):
    v = 'intermediate_' + '{:03}'.format(i)
    v_symbol = sp.Symbol(v)
    return v_symbol


class IntermediateManager(object):

    def __init__(self, unknowns: List[Unknown], sin_first: bool = True):
        """
        Build the intermediate symbols for all unknowns, which includes sin(x) and cos(x)
        for all symbol x in :param unknowns that are Revolute. For Translational unknown,
        just the unknown itself with a new name for ordering.
        As we will use Groebner Reduction with lex-order to get uni-variable polynomial,
        the reduction order is the same as the order of unknowns, and by default we try sin(x) at first.
        In other word, the final polynomial with be the one with sin(unknown[0]) if sin_first = True
        :param unknowns:
        """
        self._unknowns = unknowns
        assert len(self._unknowns) <= 6
        self._substitute_map: Dict[sp.Expr, sp.Symbol] = dict()
        self._restore_map: Dict[sp.Symbol, sp.Expr] = dict()
        self._intermediates: List[sp.Symbol] = list()
        self._intermediate_to_unknown: Dict[sp.Symbol, Unknown] = dict()

        # Build the map
        intermediate_counter = 0
        for i in range(len(unknowns)):
            unknown_i = unknowns[i]
            if unknown_i.unknown_type == UnknownType.Translational.name:
                intermediate_i = intermediate_symbol(intermediate_counter)
                self._substitute_map[unknown_i.symbol] = intermediate_i
                self._restore_map[intermediate_i] = unknown_i.symbol
                self._intermediates.append(intermediate_i)
                self._intermediate_to_unknown[intermediate_i] = unknown_i
                intermediate_counter += 1
            else:
                # Note the order of sin and cos, self.intermediates should be
                # in the same order as groebner reduction.
                if sin_first:
                    sin_intermediate_i = intermediate_symbol(intermediate_counter + 0)
                    cos_intermediate_i = intermediate_symbol(intermediate_counter + 1)
                    self._intermediates.append(sin_intermediate_i)
                    self._intermediates.append(cos_intermediate_i)
                else:
                    sin_intermediate_i = intermediate_symbol(intermediate_counter + 1)
                    cos_intermediate_i = intermediate_symbol(intermediate_counter + 0)
                    self._intermediates.append(cos_intermediate_i)
                    self._intermediates.append(sin_intermediate_i)
                self._substitute_map[sp.sin(unknown_i.symbol)] = sin_intermediate_i
                self._substitute_map[sp.cos(unknown_i.symbol)] = cos_intermediate_i
                self._restore_map[sin_intermediate_i] = sp.sin(unknown_i.symbol)
                self._restore_map[cos_intermediate_i] = sp.cos(unknown_i.symbol)
                self._intermediate_to_unknown[sin_intermediate_i] = unknown_i
                self._intermediate_to_unknown[cos_intermediate_i] = unknown_i
                intermediate_counter += 2

    def substitute_new_intermediate(self, expr_to_substitute: sp.Expr) -> sp.Expr:
        return expr_to_substitute.subs(self._substitute_map)

    def restore_original_unknowns(self, expr_to_restore: sp.Expr) -> sp.Expr:
        return expr_to_restore.subs(self._restore_map)

    def build_polynomials(self, polys_in: List[sp.Expr]) -> List[Poly]:
        """
        Given the input polynomials, perform substitute to replace sin(x) with intermediate,
        and add sin(x)**2 + cos(x)**2 - 1 terms.
        The result can be directly used in sympy.groebner to obtain the new basis.
        :param polys_in:
        :return:
        """
        # Perform variable substitute
        substituted_poly_list: List[Poly] = list()
        poly_gens = self._intermediates
        for poly in polys_in:
            new_poly = self.substitute_new_intermediate(poly)
            substituted_poly_list.append(new_poly.as_poly(gens=poly_gens))

        # Add sin(x)^2 + cos(x)^2 - 1 == 0
        for i in range(len(self._unknowns)):
            unknown_i = self._unknowns[i]
            if unknown_i.unknown_type == UnknownType.Revolute.name:
                sos_expr = sp.sin(unknown_i.symbol) ** 2 + sp.cos(unknown_i.symbol) ** 2 - 1
                substituted_sos = self.substitute_new_intermediate(sos_expr)
                substituted_poly_list.append(substituted_sos.as_poly(gens=poly_gens))

        # OK
        return substituted_poly_list

    @property
    def substitute_map(self):
        return self._substitute_map

    @property
    def restore_map(self):
        return self._restore_map

    @property
    def intermediates(self):
        return self._intermediates

    @property
    def unknowns(self):
        return self._unknowns

    @property
    def intermediates_to_unknown(self):
        return self._intermediate_to_unknown


# Debug code
def test_intermediate():
    from solver.equation_utils import default_unknowns
    unknowns = default_unknowns(2)
    unknowns.append(Unknown(sp.Symbol('th_6'), UnknownType.Translational.name))
    manager = IntermediateManager(unknowns)
    print(manager.substitute_map)
    print(manager.restore_map)
    expr_to_test = sp.sin(unknowns[0].symbol)**2 + sp.cos(unknowns[1].symbol)
    expr_with_intermediate = manager.substitute_new_intermediate(expr_to_test)
    restored_expr = manager.restore_original_unknowns(expr_with_intermediate)
    print(expr_with_intermediate)
    print(restored_expr)


if __name__ == '__main__':
    test_intermediate()
