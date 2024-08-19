from solver.degenerate_analyse.analyser_step import \
    DegenerateAnalyserStep, DegenerateAnalyserContext, DegenerateAnalyserOption
from solver.unary_solvers.unary_solver_step import UnarySolverStep
from solver.solved_variable import DegenerateType, SolutionDegenerateRecord, VariableSolution
import solver.equation_utils as equation_utils
from solver.equation_types import ScalarEquation
from solver.degenerate_analyse.square_equal_zero_solver import \
    UnarySinSquareEqualZeroSolver, UnaryCosSquareEqualZeroSolver
from typing import Optional, List, Dict, Set, Tuple
import sympy as sp


class AnalyseByUnarySolverSingleVariable(DegenerateAnalyserStep):
    """
    Using rule-based solver to analyze the degenerate record. This analyzer
    only handle the case of ONE variable in record.equations
    """
    def __init__(self):
        self._unary_solver = UnarySolverStep.make_default()
        self._solvers = self._unary_solver.get_solvers()
        self._solvers.append(UnarySinSquareEqualZeroSolver())
        self._solvers.append(UnaryCosSquareEqualZeroSolver())

    def perform_analyse(
            self,
            current_record: SolutionDegenerateRecord,
            context: DegenerateAnalyserContext,
            option: DegenerateAnalyserOption) -> Optional[SolutionDegenerateRecord]:
        if current_record.type == DegenerateType.AlwaysNonDegenerate.name:
            return None
        if len(current_record.equations) == 0:
            return None

        # The set of un-solved variable
        unsolved_variable = set()
        solved_variable_symbol = [elem.solved_variable for elem in context.solved_variables]
        for var_symbol in context.all_unknown_variables:
            if var_symbol in solved_variable_symbol:
                continue
            else:
                unsolved_variable.add(var_symbol)

        # All the variable that we want a try
        variable_to_try_set: Set[sp.Symbol] = set()
        equation_with_value: List[sp.Expr] = list()
        for equation in current_record.equations:
            equation_i = equation.subs(context.parameter_value)
            for symbol in equation_i.free_symbols:
                # We should not depend on variable that has not been solved
                if symbol not in unsolved_variable:
                    variable_to_try_set.add(symbol)
            equation_with_value.append(equation_i)

        # This analyzer can only handle one variable
        # But we don't assume the solution are numerical
        if len(variable_to_try_set) != 1:
            return None
        variable_to_try_list = [elem for elem in variable_to_try_set]
        variable_to_try = variable_to_try_list[0]

        # Try solving the equation
        solved_dict: Dict[sp.Symbol, List[sp.Expr]] = dict()
        scalar_equations: List[ScalarEquation] = list()
        for i in range(len(equation_with_value)):
            scalar_equations.append(ScalarEquation(sp.S.Zero, equation_with_value[i]))
        collected_equations = equation_utils.scalar_equation_by_unknowns(scalar_equations, variable_to_try_list)

        # Let's find the solution
        var_solution: Optional[VariableSolution] = None
        for solver in self._solvers:
            solution_i = solver.try_solve(collected_equations, variable_to_try, variable_to_try_list)
            if len(solution_i) > 0:
                # Just select a random one
                var_solution = solution_i[0]
                break

        # Check the solution
        if var_solution is None:
            return None

        # Now we have a solution
        solved_dict[variable_to_try] = var_solution.explicit_solutions
        return SolutionDegenerateRecord.record_variable_value(solved_dict)
