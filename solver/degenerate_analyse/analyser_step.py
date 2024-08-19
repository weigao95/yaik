from solver.solved_variable import DegenerateType, SolutionDegenerateRecord, VariableSolution
from solver.equation_utils import cast_expr_to_float
from typing import Optional, List, Dict, Set, Tuple
import sympy as sp
import attr


@attr.s
class DegenerateAnalyserContext(object):
    # Everything is default constructable
    solved_variables: List[VariableSolution] = list()
    all_unknown_variables: List[sp.Symbol] = list()
    all_parameters: Set[sp.Symbol] = list()
    parameter_bounds: Dict[sp.Symbol, Tuple[float, float]] = dict()
    parameter_value: Dict[sp.Symbol, float] = dict()
    numerical_test_cases: List[Dict[sp.Symbol, float]] = list()


@attr.s
class DegenerateAnalyserOption(object):
    main_branch_solution: bool = False


class DegenerateAnalyserStep(object):

    def perform_analyse(
            self,
            current_record: SolutionDegenerateRecord,
            context: DegenerateAnalyserContext,
            option: DegenerateAnalyserOption) -> Optional[SolutionDegenerateRecord]:
        """
        Perform analysis of the current_record, given :param parameter_value and :param solved_variables.
        If this analyser make some progress, then it returns the updated degenerate record
        :param current_record: the record for analyse
        :param context:
        :param option:
        :return: None if this analyser cannot analyse the record, else return the updated record
        """
        raise NotImplementedError


class CheckConstantAnalyserStep(DegenerateAnalyserStep):

    def perform_analyse(
            self,
            current_record: SolutionDegenerateRecord,
            context: DegenerateAnalyserContext,
            option: DegenerateAnalyserOption) -> Optional[SolutionDegenerateRecord]:
        """
        If any of the equations can be reduced to a non-zero constant, then we're done.
        """
        if current_record.type == DegenerateType.AlwaysNonDegenerate.name:
            return None
        for equation in current_record.equations:
            simplified = equation.subs(context.parameter_value).simplify()
            simplified_value = cast_expr_to_float(simplified)
            if (simplified_value is not None) and simplified != sp.S.Zero:
                return current_record.record_always_non_degenerate()
        return None

# Debug code
# Please refer to test_degenerate.py
