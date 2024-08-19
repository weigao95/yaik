import sympy as sp
from typing import Dict, List, Optional, Tuple, Set


class ReducedRaghavanRothEquationFormatter(object):

    def __init__(self):
        pass

    @staticmethod
    def sort_monomial_by_lex_order(mono_tuple_list: List[Tuple[int]]) -> List[Tuple[int]]:
        selected_flag: List[bool] = list()
        for i in range(len(mono_tuple_list)):
            selected_flag.append(False)

        def compare_a_greater_than_b(mono_a: Tuple[int], mono_b: Tuple[int]):
            assert len(mono_a) == len(mono_b)
            for k in range(len(mono_a)):
                if mono_a[k] > mono_b[k]:
                    return True
                elif mono_a[k] < mono_b[k]:
                    return False
            return False

        # The loop
        sorted_mono_list = list()
        while True:
            unselected_mono_idx: Optional[int] = None
            for mono_idx in range(len(mono_tuple_list)):
                if selected_flag[mono_idx]:
                    continue
                else:
                    unselected_mono_idx = mono_idx
                    break

            # Everything selected
            if unselected_mono_idx is None:
                return sorted_mono_list

            # Find the max among remaining
            max_mono_idx = unselected_mono_idx
            for mono_idx in range(len(mono_tuple_list)):
                if selected_flag[mono_idx]:
                    continue
                if compare_a_greater_than_b(mono_tuple_list[mono_idx], mono_tuple_list[max_mono_idx]):
                    max_mono_idx = mono_idx

            # Update
            assert max_mono_idx >= 0
            selected_flag[max_mono_idx] = True
            sorted_mono_list.append(mono_tuple_list[max_mono_idx])

    @staticmethod
    def rewrite_poly_equations_into_matrix_monomial(poly_equations: List[sp.Poly], poly_gens: List[sp.Symbol]):
        monomials: Set[Tuple[int]] = set()
        for poly in poly_equations:
            for term in poly.terms():
                mono, coefficient = term
                assert len(mono) == len(poly_gens)
                monomials.add(mono)

        # Get a flatted list of monomials and sort it
        monomials_list_raw: List[Tuple[int]] = [elem for elem in monomials]
        monomials_list = ReducedRaghavanRothEquationFormatter.sort_monomial_by_lex_order(monomials_list_raw)

        # Functor to find the index
        def find_monomial_idx(mono_to_find, list_of_monomial):
            for mono_list_idx in range(len(list_of_monomial)):
                if monomials_list[mono_list_idx] == mono_to_find:
                    return mono_list_idx
            return -1

        # Construct the matrix
        coefficient_matrix: sp.Matrix = sp.zeros(len(poly_equations), len(monomials_list))
        for i in range(len(poly_equations)):
            poly_i: sp.Poly = poly_equations[i]
            for term in poly_i.terms():
                mono, coefficient = term
                mono_idx = find_monomial_idx(mono_to_find=mono, list_of_monomial=monomials_list)
                assert mono_idx >= 0
                coefficient_matrix[i, mono_idx] = coefficient

        # Construct the monomial vector
        monomial_vector = sp.zeros(len(monomials_list), 1)
        for i in range(len(monomials_list)):
            mono_i = monomials_list[i]
            mono_in_expr = sp.S.One
            for k in range(len(mono_i)):
                order_k = mono_i[k]
                mono_in_expr = mono_in_expr * (poly_gens[k] ** order_k)
            monomial_vector[i] = mono_in_expr
        return coefficient_matrix, monomial_vector, monomials_list

    @staticmethod
    def factor_linear_sin_cos_matrix_sum(
            linear_sum_matrix: sp.Matrix,
            sin_symbol: sp.Symbol,
            cos_symbol: sp.Symbol) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix, List[sp.Expr]]:
        """
        Factor linear_sum_matrix as A_sin * sin_symbol + A_cos * cos_symbol + C_const. Ignore all terms that are
        not linear and append them into a non_linear_coefficient_list.
        :return A_sin, A_cos, C_const, non_linear_coefficient_list
        """
        matrix_rows, matrix_cols = linear_sum_matrix.shape[0], linear_sum_matrix.shape[1]
        poly_gen = [sin_symbol, cos_symbol]
        A_sin = sp.zeros(matrix_rows, matrix_cols)
        A_cos = sp.zeros(matrix_rows, matrix_cols)
        C_const = sp.zeros(matrix_rows, matrix_cols)
        non_linear_coefficient_list: List[sp.Expr] = list()
        for r in range(matrix_rows):
            for c in range(matrix_cols):
                rc_expr: sp.Expr = linear_sum_matrix[r, c]
                rc_poly: sp.Poly = sp.Poly(rc_expr, gens=poly_gen)
                for poly_term in rc_poly.terms():
                    mono, coefficient = poly_term
                    if sum(mono) >= 2:
                        non_linear_coefficient_list.append(coefficient)
                        continue
                    assert sum(mono) <= 1  # Should be linear
                    assert len(mono) == 2
                    if mono[0] == 1:
                        A_sin[r, c] = coefficient
                    elif mono[1] == 1:
                        A_cos[r, c] = coefficient
                    else:
                        C_const[r, c] = coefficient

        # Finished
        return A_sin, A_cos, C_const, non_linear_coefficient_list
