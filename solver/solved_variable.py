import sympy as sp
from sympy.parsing.sympy_parser import parse_expr as parse_sympy_expr
from solver.solution_degeneration_record import DegenerateType, NumericalAnalysedResultType, SolutionDegenerateRecord
from solver.solved_variable_impl import SolutionMethod, VariableSolutionImplBase, VariableSolutionClassKey
from solver.solved_variable_impl import ExplicitVariableSolutionImpl, PolynomialVariableSolutionImpl
from solver.solved_variable_impl import LinearSinCosType_2_SolutionImpl
import solver.solved_variable_impl as solved_variable_impl
import solver.equation_utils as equation_utils
from solver.equation_types import ScalarEquation
from typing import List, Optional, Dict, Tuple


class VariableSolution(object):

    def __init__(self, solution_impl: Optional[VariableSolutionImplBase] = None):
        self._solution_impl = solution_impl

    def to_dict(self) -> Dict:
        return solved_variable_impl.solution_impl_to_dict(self._solution_impl)

    def from_dict(self, data_map: Dict):
        self._solution_impl = solved_variable_impl.solution_impl_from_dict(data_map)

    @staticmethod
    def make_explicit_solution(
            solved_variable: sp.Symbol,
            solutions: List[sp.Expr],
            solution_method: str,
            solve_from_equations: Optional[List[ScalarEquation]] = None,
            argument_valid_checkers: Optional[List[sp.Expr]] = None,
            degenerate_record: SolutionDegenerateRecord = SolutionDegenerateRecord.record_cannot_analyse()):
        solution_impl = ExplicitVariableSolutionImpl.make_explicit_solution(
            solved_variable=solved_variable,
            solutions=solutions, solution_method=solution_method,
            solve_from_equations=solve_from_equations,
            argument_valid_checkers=argument_valid_checkers,
            degenerate_record=degenerate_record)

        # General info
        solution = VariableSolution(solution_impl)
        return solution

    @staticmethod
    def make_polynomial_solution(
            solved_variable: sp.Symbol,
            solution_method: str,
            polynomial_dict: Dict[int, Tuple[sp.Expr, sp.Expr]],
            degenerate_record: SolutionDegenerateRecord = SolutionDegenerateRecord.record_cannot_analyse()):
        solution_impl = PolynomialVariableSolutionImpl.make_polynomial_solution(
            solved_variable=solved_variable, solution_method=solution_method,
            polynomial_dict=polynomial_dict, degenerate_record=degenerate_record)
        assert solution_method == SolutionMethod.PolynomialDirect.name \
               or solution_method == SolutionMethod.PolynomialSin.name \
               or solution_method == SolutionMethod.PolynomialCos.name
        # General info
        solution = VariableSolution(solution_impl)
        return solution

    # Update the record
    def update_degenerate_record(self, degenerate_record: SolutionDegenerateRecord):
        self._solution_impl.update_degenerate_record(degenerate_record=degenerate_record)

    def update_numerical_analysis_result(self, result_in_str: str):
        self._solution_impl.update_numerical_analysis_result(result_in_str=result_in_str)

    # Access method
    def num_solutions(self):
        """
        Get the number of solutions represented by this struct
        :return:
        """
        return self._solution_impl.num_solutions()

    @property
    def valid(self) -> bool:
        return self._solution_impl is not None

    @property
    def internal_solution(self) -> VariableSolutionImplBase:
        return self._solution_impl

    @property
    def solved_variable(self) -> sp.Symbol:
        assert self.valid
        return self._solution_impl.solved_variable

    @property
    def solution_method(self) -> str:
        assert self.valid
        return self._solution_impl.solution_method

    @property
    def solved_from_scalar_equations(self) -> List[ScalarEquation]:
        assert self.valid
        return self._solution_impl.solved_from_scalar_equations

    @property
    def degenerate_record(self) -> SolutionDegenerateRecord:
        assert self.valid
        return self._solution_impl.degenerate_record

    @property
    def is_explicit_solution(self):
        assert self.valid
        return self._solution_impl.is_explicit_solution

    @property
    def explicit_solutions(self):
        assert self._solution_impl.is_explicit_solution
        return self._solution_impl.explicit_solutions

    @property
    def arguments(self):
        return None

    @property
    def argument_validity_checkers(self):
        if self.is_explicit_solution:
            return self._solution_impl.argument_validity_checkers
        else:
            all_valid = list()
            for i in range(self.num_solutions()):
                all_valid.append(sp.S.BooleanTrue)
            return all_valid

    @property
    def is_polynomial(self):
        assert self.valid
        return self._solution_impl.is_polynomial

    @property
    def impl_class_key(self):
        assert self.valid
        return self._solution_impl.class_key()

    @property
    def polynomial_to_solve(self):
        assert self.is_polynomial
        return self._solution_impl.polynomial_to_solve

    @property
    def is_general_6dof_solution(self):
        assert self.valid
        return self._solution_impl.is_general_6dof

    def count_explicit_solution_total_ops(self):
        """
        For an explicit solution, count the total number of operations in the solutions.
        Note that the result is summed over different solutions.
        """
        assert self.is_explicit_solution
        return self._solution_impl.count_explicit_solution_total_ops()

    def count_polynomial_coefficient_ops(self):
        """
        For a polynomial solution, count the total number of operations for computing
        the polynomial coefficient.
        """
        assert self.is_polynomial
        return self._solution_impl.count_polynomial_coefficient_ops()


def read_solution_list(solution_list_yaml: str) -> List[VariableSolution]:
    """
    Read a list of variable solution from a yaml file
    """
    import yaml
    with open(solution_list_yaml, 'r') as read_stream:
        input_dict = yaml.load(read_stream, Loader=yaml.CLoader)
    assert input_dict is not None
    solution_str_list = input_dict['solutions']
    solution_list: List[VariableSolution] = list()
    for i in range(len(solution_str_list)):
        solution_i = VariableSolution(solution_impl=None)
        solution_i.from_dict(solution_str_list[i])
        solution_list.append(solution_i)
    return solution_list
