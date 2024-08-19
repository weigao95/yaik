import solver.equation_utils as equation_utils
from solver.solved_variable import VariableSolution, SolutionMethod
from typing import List, Dict, Tuple
import sympy as sp
import numpy as np
import copy


def verify_polynomial_solution(
        polynomials: List[sp.Expr],
        poly_reduce_output,
        parameter_values_to_test: List[Dict[sp.Symbol, float]],
        log_if_not_match: bool = False,
        max_n_test_cases: int = 100) -> Tuple[bool, float]:
    """
    Given a polynomial solution reduction input/output and a set of test data, check whether the reduced
    polynomial is a solution.
    @param polynomials
    @param poly_reduce_output
    @param parameter_values_to_test
    @param log_if_not_match
    @param max_n_test_cases
    @return a tuple, first is a bool indicating whether all value match
                     the second is the ration of passed case (which <= len(parameter_values_to_test))
    """
    # Extract the output
    if poly_reduce_output is None:
        return False, 0
    poly_dict, solved_term, solved_unknown, solution_method = poly_reduce_output

    # Make the coefficient list
    poly_order = -1
    for order in poly_dict:
        if order >= poly_order:
            poly_order = order

    # Iterate through the parameters
    n_matched = 0
    all_value_match = True
    n_test_case = min(len(parameter_values_to_test), max_n_test_cases)
    index2test = np.random.choice(np.arange(len(parameter_values_to_test)), size=n_test_case, replace=False)
    for i in range(n_test_case):
        idx = index2test[i]
        value_map_i = parameter_values_to_test[idx]
        p_coefficients = np.zeros(shape=(poly_order + 1,))
        for order in poly_dict:
            numerator, denominator = poly_dict[order]
            numerator_value = numerator.subs(value_map_i)
            denominator_value = denominator.subs(value_map_i)
            coefficient_value = numerator_value / denominator_value
            assert coefficient_value.is_Number
            # Note that in np.roots, p[0] corresponds to the highest order
            p_coefficients[poly_order - order] = float(coefficient_value)

        # Find the root
        poly_roots = np.roots(p_coefficients)
        solutions = list()
        for root_idx in range(poly_roots.size):
            this_root = poly_roots[root_idx]
            if not np.isreal(this_root):
                continue
            if solution_method == SolutionMethod.PolynomialDirect.name:
                solutions.append(this_root)
            if solution_method == SolutionMethod.PolynomialSin.name and abs(this_root) < 1:
                first_angle = np.arcsin(this_root)
                solutions.append(first_angle)
                solutions.append(np.pi - first_angle)
            if solution_method == SolutionMethod.PolynomialCos.name and abs(this_root) < 1:
                first_angle = np.arccos(this_root)
                solutions.append(first_angle)
                solutions.append(-first_angle)

        # Try the root
        contain_solution = False
        diff_values = list()
        for j in range(len(solutions)):
            subst_map_j = copy.deepcopy(value_map_i)
            subst_map_j[solved_unknown.symbol] = solutions[j]
            for k in range(len(polynomials)):
                poly_k = polynomials[k]
                poly_k_value = poly_k.subs(subst_map_j)
                if not poly_k_value.is_Number:
                    continue
                if abs(poly_k_value) < 1e-10:
                    contain_solution = True
                else:
                    diff_values.append(abs(poly_k_value))

        # Update the statistic
        all_value_match = all_value_match and contain_solution
        if log_if_not_match and (not contain_solution):
            print(diff_values)
        if contain_solution:
            n_matched += 1
    return all_value_match, float(n_matched) / float(n_test_case)
