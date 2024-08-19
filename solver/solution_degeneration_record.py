import sympy as sp
import solver.equation_utils as equation_utils
from typing import List, Optional, Dict, Tuple
from enum import Enum


class DegenerateType(Enum):
    # The solution is always non-degenerated (after symbolic analysis), most desirable case
    AlwaysNonDegenerate = 1
    # The solution is degenerated if all the equations are zero
    DegenerateIfAllEquationsZero = 2
    # The solution is degenerated if any of the equations is zero
    DegenerateOnVariableValue = 3
    # We don't know even after the analyse, this is the least desirable type
    CannotAnalyse = 4


class NumericalAnalysedResultType(Enum):
    NotAnalyzedYet = 0
    # The solution is non-degenerated after numerical testing of many (interesting) values
    NumericalAlwaysNonDegenerate = 1
    # The solution sometimes degenerated after numerical testing, but not too-often
    NumericalUsuallyNonDegenerate = 2
    # Too many value in numerical analyzer cause degeneration, should NOT USE
    NumericalDegenerate = 6


class SolutionDegenerateRecord(object):

    def __init__(self):
        self._type: str = DegenerateType.CannotAnalyse.name
        self._numerical_analysed_result: str = NumericalAnalysedResultType.NotAnalyzedYet.name
        self._equations: List[sp.Expr] = list()
        self._variable_value: Dict[sp.Symbol, List[sp.Expr]] = dict()

    # Serialize and de-serialize
    def to_dict(self) -> Dict:
        data_map = dict()
        data_map['type'] = self._type
        data_map['numerical_result'] = self._numerical_analysed_result

        # The equations
        data_map['equations'] = equation_utils.expression_list_to_string_representation(self.equations)

        # The value map
        value_map = dict()
        for var_key in self._variable_value:
            var_value = self._variable_value[var_key]
            str_key = var_key.name
            str_list_value = equation_utils.expression_list_to_string_representation(var_value)
            value_map[str_key] = str_list_value
        data_map['variable_value_map'] = value_map

        # OK
        return data_map

    def from_dict(self, data_map: Dict):
        self._type = data_map['type']

        # numerical_result
        if 'numerical_result' in data_map:
            self._numerical_analysed_result = data_map['numerical_result']
        else:
            self._numerical_analysed_result = NumericalAnalysedResultType.NotAnalyzedYet.name

        # Parse expressions
        equation_str_list = data_map['equations']
        self._equations = equation_utils.parse_string_expressions(equation_str_list)

        # Parse the value map
        var_value_map = data_map['variable_value_map']
        self._variable_value.clear()
        for str_key in var_value_map:
            str_value = var_value_map[str_key]
            symbol_key = sp.Symbol(str_key)
            expr_value = equation_utils.parse_string_expressions(str_value)
            self._variable_value[symbol_key] = expr_value

    @staticmethod
    def record_cannot_analyse():
        record = SolutionDegenerateRecord()
        record._type = DegenerateType.CannotAnalyse.name
        return record

    @staticmethod
    def record_always_non_degenerate():
        record = SolutionDegenerateRecord()
        record._type = DegenerateType.AlwaysNonDegenerate.name
        return record

    @staticmethod
    def record_all_equations(equations: List[sp.Expr]):
        record = SolutionDegenerateRecord()
        record._type = DegenerateType.DegenerateIfAllEquationsZero.name
        record._equations = equations
        return record

    @staticmethod
    def record_variable_value(variable_dict: Dict[sp.Symbol, List[sp.Expr]]):
        record = SolutionDegenerateRecord()
        record._type = DegenerateType.DegenerateOnVariableValue.name
        record._equations = list()
        record._variable_value = variable_dict
        return record

    @property
    def type(self):
        return self._type

    @property
    def numerical_analysed_result(self) -> str:
        return self._numerical_analysed_result

    @property
    def is_degenerate_on_variable_values(self):
        return self._type == DegenerateType.DegenerateOnVariableValue.name

    @property
    def equations(self) -> List[sp.Expr]:
        return self._equations

    @property
    def variable_value_map(self):
        return self._variable_value

    def set_numerical_result(self, result_in_str: str):
        if result_in_str != NumericalAnalysedResultType.NotAnalyzedYet.name \
                and result_in_str != NumericalAnalysedResultType.NumericalAlwaysNonDegenerate.name \
                and result_in_str != NumericalAnalysedResultType.NumericalUsuallyNonDegenerate.name \
                and result_in_str != NumericalAnalysedResultType.NumericalDegenerate.name:
            print('Unknown numerical analysis result type of {result_type}'.format(result_type=result_in_str))
            return
        self._numerical_analysed_result = result_in_str

    def count_number_variable_solutions(self) -> int:
        """
        Count how many of solution is given in this degenerate record.
        Only for variable value record.
        """
        assert self.type == DegenerateType.DegenerateOnVariableValue.name
        n_solutions = None
        for var in self._variable_value:
            var_solutions = self._variable_value[var]
            if n_solutions is None:
                n_solutions = len(var_solutions)
            else:
                assert n_solutions == len(var_solutions)
        return n_solutions

    def get_variable_solution(self, solution_idx) -> List[Tuple[sp.Symbol, sp.Expr]]:
        """
        Get the variable solution by index :param solution_idx,
        where :param solution_idx < self.count_number_variable_solutions().
        """
        assert self.type == DegenerateType.DegenerateOnVariableValue.name
        variable_solution_list: List[Tuple[sp.Symbol, sp.Expr]] = list()
        for var in self._variable_value:
            var_solutions = self._variable_value[var]
            assert solution_idx < len(var_solutions)
            var_solution_at_idx = var_solutions[solution_idx]
            variable_solution_list.append((var, var_solution_at_idx))
        return variable_solution_list
