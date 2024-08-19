from solver.degenerate_analyse.analyser_step import \
    DegenerateAnalyserStep, DegenerateAnalyserContext, DegenerateAnalyserOption
from solver.solved_variable import DegenerateType, SolutionDegenerateRecord, VariableSolution
from solver.equation_utils import cast_expr_to_float
from typing import Optional, Dict, List
import sympy as sp


class IgnoreDegenerateOnSameVariableValue(DegenerateAnalyserStep):

    def __init__(self, variable_difference_tolerance: float = 1e-6):
        self._check_variable_tolerance: float = variable_difference_tolerance

    def perform_analyse(
            self,
            current_record: SolutionDegenerateRecord,
            context: DegenerateAnalyserContext,
            option: DegenerateAnalyserOption) -> Optional[SolutionDegenerateRecord]:
        # Only handle degenerate on variable value
        if current_record.type != DegenerateType.DegenerateOnVariableValue.name:
            return None

        # Not on main branch, do nothing
        if not option.main_branch_solution:
            return None

        # If current value map is a subset of some parent value map,
        # Then the degeneration should be handled by the parent
        current_variable_value_map = current_record.variable_value_map
        parent_solutions: List[VariableSolution] = context.solved_variables
        for i in range(len(parent_solutions)):
            parent_i = parent_solutions[i]
            parent_degenerate = parent_i.degenerate_record
            if not parent_degenerate.is_degenerate_on_variable_values:
                continue

            # Parent also degenerate on variable value
            parent_variable_value_map = parent_degenerate.variable_value_map
            if self.value_map_is_subset(current_variable_value_map, parent_variable_value_map):
                # Found a parent
                print('The child and parent depends on the same variable value')
                return SolutionDegenerateRecord.record_always_non_degenerate()

        # Not found
        return None

    def value_map_is_subset(
            self,
            child_map: Dict[sp.Symbol, List[sp.Expr]],
            parent_map: Dict[sp.Symbol, List[sp.Expr]]):
        # Only handle 1-variable case, constant solution case
        if len(child_map) != 1 or len(parent_map) != 1:
            return False

        # Get the variable, this is dirty (but other solutions are wired)
        child_symbol: Optional[sp.Symbol] = None
        child_values: Optional[List[sp.Expr]] = None
        parent_symbol: Optional[sp.Symbol] = None
        parent_values: Optional[List[sp.Expr]] = None
        for key in child_map:
            child_symbol = key
            child_values = child_map[key]
        for key in parent_map:
            parent_symbol = key
            parent_values = parent_map[key]
        assert child_symbol is not None and child_values is not None
        assert parent_symbol is not None and parent_values is not None

        # Not on the same variable, do nothing
        if child_symbol.name != parent_symbol.name:
            return False

        # Cast the parent map to numerical value
        parent_numerical_values: List[float] = list()
        for j in range(len(parent_values)):
            parent_value_j = cast_expr_to_float(parent_values[j])
            if parent_value_j is None:
                return False
            parent_numerical_values.append(parent_value_j)

        # Now they degenerate on same variable
        every_child_in_parent = True
        for i in range(len(child_values)):
            child_symbolic_value_i = child_values[i]
            child_value_i = cast_expr_to_float(child_symbolic_value_i)
            # Only do numerical value
            if child_value_i is None:
                return False

            # Now do checking
            found_in_parent = False
            for j in range(len(parent_numerical_values)):
                parent_value_j = parent_numerical_values[j]
                if abs(parent_value_j - child_value_i) < self._check_variable_tolerance:
                    found_in_parent = True
                    break

            # Not find, then not a subset
            if not found_in_parent:
                return False
            every_child_in_parent = every_child_in_parent and found_in_parent

        # The loop passed, thus should be in
        return every_child_in_parent
