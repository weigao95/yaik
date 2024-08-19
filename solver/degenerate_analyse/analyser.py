from solver.degenerate_analyse.analyser_step import CheckConstantAnalyserStep, DegenerateAnalyserStep
from solver.degenerate_analyse.analyser_step import DegenerateAnalyserContext, DegenerateAnalyserOption
from solver.degenerate_analyse.unary_analyser import AnalyseByUnarySolverSingleVariable
from solver.degenerate_analyse.ignore_degenerate_on_same_variable import IgnoreDegenerateOnSameVariableValue
from solver.solved_variable import DegenerateType, SolutionDegenerateRecord
from typing import Optional, List, Dict, Set, Tuple


class DegenerateAnalyser(DegenerateAnalyserStep):

    def __init__(self):
        self._check_constant = CheckConstantAnalyserStep()
        self._single_variable_solve = AnalyseByUnarySolverSingleVariable()
        self._ignore_degenerate_on_same_variable_value = IgnoreDegenerateOnSameVariableValue()

    def perform_analyse(
            self,
            current_record: SolutionDegenerateRecord,
            context: DegenerateAnalyserContext,
            option: DegenerateAnalyserOption) -> Optional[SolutionDegenerateRecord]:
        # Each iteration only updates once, but we can do many iterations
        iteration_record = self.perform_analyse_iteration(current_record, context, option)
        while iteration_record is not None:
            updated_record = self.perform_analyse_iteration(iteration_record, context, option)
            if updated_record is None:
                return iteration_record
            else:
                iteration_record = updated_record

        # The case when iteration_case is directly None
        return iteration_record

    def perform_analyse_iteration(
        self,
        current_record: SolutionDegenerateRecord,
        context: DegenerateAnalyserContext,
        option: DegenerateAnalyserOption) -> Optional[SolutionDegenerateRecord]:
        """
        Perform analyse of the degenerate record, return a new one if progress made
        """
        # Do nothing if it is already very good/bad
        if current_record.type == DegenerateType.AlwaysNonDegenerate.name \
                or current_record.type == DegenerateType.CannotAnalyse.name:
            return None

        # Check constant
        new_record = self._check_constant.perform_analyse(
            current_record, context, option)
        if new_record is not None:
            return new_record

        # Variable solution
        new_record = self._single_variable_solve.perform_analyse(
            current_record, context, option)
        if new_record is not None:
            return new_record

        # Ignore the degeneration on the same variable value
        new_record = self._ignore_degenerate_on_same_variable_value.perform_analyse(
            current_record, context, option)
        if new_record is not None:
            return new_record

        # Nothing new
        return None
