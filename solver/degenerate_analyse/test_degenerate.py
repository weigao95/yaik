import sympy as sp
import numpy as np
import unittest
from typing import List, Dict
from solver.solved_variable import DegenerateType, SolutionDegenerateRecord, VariableSolution
from solver.degenerate_analyse.analyser_step import \
    CheckConstantAnalyserStep, DegenerateAnalyserContext, DegenerateAnalyserOption
from solver.degenerate_analyse.unary_analyser import AnalyseByUnarySolverSingleVariable
from solver.degenerate_analyse.numerical_analyser import NumericalCheckDegenerateAnalyzer, NumericalAnalysedResultType
from solver.degenerate_analyse.analyser import DegenerateAnalyser
from fk.fk_equations import ik_target_symbols, Px, Py


class TestDegenerateAnalyse(unittest.TestCase):

    def setUp(self) -> None:
        self._th_0 = sp.Symbol('th_0')
        self._l_3 = sp.Symbol('l_3')

    def test_check_constant(self):
        l_3 = sp.Symbol('l_3')
        record = SolutionDegenerateRecord.record_all_equations([Px, Py, l_3])
        parameter_value = {l_3: 0.342}
        analyser_step = CheckConstantAnalyserStep()
        analyser_context = DegenerateAnalyserContext()
        analyser_context.parameter_value = parameter_value
        updated_record = analyser_step.perform_analyse(
            record, context=analyser_context, option=DegenerateAnalyserOption())
        self.assertTrue(updated_record.type == DegenerateType.AlwaysNonDegenerate.name)

    def test_unary_solver(self):
        parameters = ik_target_symbols()
        th_0 = self._th_0
        parameters.add(th_0)
        record = SolutionDegenerateRecord.record_all_equations([0.7 * sp.sin(th_0)])
        analyser_step = AnalyseByUnarySolverSingleVariable()
        analyser_context = DegenerateAnalyserContext()
        th_0_solution = VariableSolution.make_explicit_solution(
            th_0, [sp.S.Zero], 'fake_method_for_test',
            solve_from_equations=None, argument_valid_checkers=[sp.S.BooleanTrue])
        analyser_context.solved_variables = [th_0_solution]
        analyser_context.all_unknown_variables = [th_0]
        analyser_context.all_parameters = parameters
        updated_record = analyser_step.perform_analyse(record, analyser_context, DegenerateAnalyserOption())
        self.assertTrue(updated_record.type == DegenerateType.DegenerateOnVariableValue.name)

        # Another one
        record = SolutionDegenerateRecord.record_all_equations([0.7 * sp.sin(th_0), Px + Py])
        updated_record = analyser_step.perform_analyse(record, analyser_context, DegenerateAnalyserOption())
        self.assertTrue(updated_record is None)

        # Should not depend on un-solved variable
        record = SolutionDegenerateRecord.record_all_equations([0.7 * sp.sin(th_0)])
        context_unsolved = DegenerateAnalyserContext()
        context_unsolved.all_unknown_variables = [th_0]
        context_unsolved.all_parameters = parameters
        updated_record = analyser_step.perform_analyse(record, context_unsolved, DegenerateAnalyserOption())
        self.assertTrue(updated_record is None)

    def test_same_variable_value(self):
        parameters = ik_target_symbols()
        th_0 = self._th_0
        th_1 = sp.Symbol('th_1')
        parameters.add(th_0)

        # Make the variable value dict
        variable_value_dict: Dict[sp.Symbol, List[sp.Expr]] = dict()
        variable_value_dict[th_0] = [sp.S.Zero, sp.pi]
        variable_value_degenerate = SolutionDegenerateRecord.record_variable_value(variable_value_dict)
        th_1_solution = VariableSolution.make_explicit_solution(
            th_1, [sp.S.Zero], 'fake_method_for_test',
            solve_from_equations=None,
            argument_valid_checkers=[sp.S.BooleanTrue],
            degenerate_record=variable_value_degenerate)

        # OK, let's try it
        analyser = DegenerateAnalyser()
        record = SolutionDegenerateRecord.record_all_equations([0.7 * sp.sin(th_0)])
        context = DegenerateAnalyserContext()
        context.all_parameters = parameters
        context.solved_variables = [th_1_solution]
        option = DegenerateAnalyserOption()
        option.main_branch_solution = True
        updated_record = analyser.perform_analyse(record, context, option)
        self.assertTrue(updated_record is not None)
        self.assertTrue(updated_record.type == DegenerateType.AlwaysNonDegenerate.name)

    def test_numerical_analyzer(self):
        # Make the record
        parameters = ik_target_symbols()
        th_0 = self._th_0
        parameters.add(th_0)
        record = SolutionDegenerateRecord.record_all_equations([0.7 * sp.sin(th_0)])

        # Make the data
        numerical_test_cases: List[Dict[sp.Symbol, float]] = []
        test_n = 10000
        for i in range(test_n + 1):
            th_0_value = float(np.pi) * i * (1.0 / float(test_n))
            value_map_i = {th_0: th_0_value}
            numerical_test_cases.append(value_map_i)

        # OK
        analyzer = NumericalCheckDegenerateAnalyzer(numerical_test_cases)
        analyzed_result, degenerate_ratio = analyzer.perform_analyse(record, [], parameters, dict(), dict())
        self.assertTrue(analyzed_result == NumericalAnalysedResultType.NumericalUsuallyNonDegenerate.name)

    def test_integration(self):
        parameters = ik_target_symbols()
        th_0 = self._th_0
        parameters.add(th_0)
        analyser = DegenerateAnalyser()

        l_3 = self._l_3
        record = SolutionDegenerateRecord.record_all_equations([Px, Py, l_3])
        parameter_value_0 = {l_3: 1.2324}
        context = DegenerateAnalyserContext()
        context.parameter_value = parameter_value_0
        updated_record = analyser.perform_analyse(record, context, DegenerateAnalyserOption())
        self.assertTrue(updated_record is not None)
        self.assertTrue(updated_record.type == DegenerateType.AlwaysNonDegenerate.name)

        # Another value
        parameter_value_1 = {l_3: 0}
        context.parameter_value = parameter_value_1
        updated_record = analyser.perform_analyse(record, context, DegenerateAnalyserOption())
        self.assertTrue(updated_record is None)
        # self.assertTrue(updated_record.type == DegenerateType.DegenerateOnVariableValue.name)
        # variable_value = updated_record.variable_value_map
        # self.assertTrue(len(variable_value) == 2)
        # for k in variable_value:
        #     self.assertTrue(k == Px or k == Py)

    def test_square(self):
        th_0 = self._th_0
        analyser = DegenerateAnalyser()
        all_symbol_set = {th_0}
        record_sin = SolutionDegenerateRecord.record_all_equations([0.2 * sp.sin(th_0) * sp.sin(th_0)])
        context = DegenerateAnalyserContext()
        th_0_solution = VariableSolution.make_explicit_solution(
            th_0, [sp.S.Zero], 'fake_method_for_test',
            solve_from_equations=None, argument_valid_checkers=[sp.S.BooleanTrue])
        context.solved_variables = [th_0_solution]
        context.all_unknown_variables = [th_0]
        context.all_parameters = all_symbol_set
        updated_record = analyser.perform_analyse(record_sin, context, DegenerateAnalyserOption())
        self.assertTrue(updated_record is not None)
        self.assertTrue(updated_record.type == DegenerateType.DegenerateOnVariableValue.name)
        record_cos = SolutionDegenerateRecord.record_all_equations([0.2 * sp.cos(th_0) * sp.cos(th_0)])
        updated_record = analyser.perform_analyse(record_cos, context, DegenerateAnalyserOption())
        self.assertTrue(updated_record is not None)
        self.assertTrue(updated_record.type == DegenerateType.DegenerateOnVariableValue.name)


if __name__ == '__main__':
    pass
