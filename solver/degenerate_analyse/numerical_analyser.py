from solver.solved_variable import \
    DegenerateType, SolutionDegenerateRecord, NumericalAnalysedResultType, VariableSolution
import solver.equation_utils as equation_utils
import multiprocessing
from typing import Optional, List, Dict, Set, Tuple
import sympy as sp


def check_zero_processor(input_args) -> bool:
    """
    Check whether the given set of input equations are all zero.
    """
    equations, subst_map = input_args
    all_zero: bool = True
    for equation_i in equations:
        equation_i_subst = equation_i.subs(subst_map)
        equation_i_value: Optional[float] = equation_utils.cast_expr_to_float(equation_i_subst)
        if equation_i_value is None:
            print('Incorrect substitution result in numerical check zero!')
            continue

        # Check the value
        if abs(equation_i_value) > 1e-6:
            all_zero = False
            break
    return all_zero


class NumericalCheckDegenerateAnalyzer(object):

    def __init__(self, numerical_test_cases: List[Dict[sp.Symbol, float]]):
        self._test_cases = numerical_test_cases

    def perform_analyse(
            self,
            current_record: SolutionDegenerateRecord,
            solved_variables: List[VariableSolution],
            all_parameters: Set[sp.Symbol],
            parameter_bounds: Dict[sp.Symbol, Tuple[float, float]],
            parameter_value: Dict[sp.Symbol, float]) -> Tuple[str, float]:
        """
        This method perform numerical analyse of the input solution, and return
        the degenerate ratio and type.
        """
        # Only handle current record
        assert current_record.type == DegenerateType.DegenerateIfAllEquationsZero.name
        equations = current_record.equations
        input_args = list()
        for i in range(len(self._test_cases)):
            test_map_i = self._test_cases[i]
            input_args.append((equations, test_map_i))

        # Run processing in parallel
        n_process = min(32, len(input_args))
        n_process = max(n_process, 1)
        with multiprocessing.Pool(n_process) as pool:
            output = pool.map(check_zero_processor, input_args)

        # Count the # of degenerate
        n_degenerate = 0
        for i in range(len(output)):
            if output[i]:
                n_degenerate += 1

        # Check result
        ratio = 0.05
        if n_degenerate == 0:
            return_type = NumericalAnalysedResultType.NumericalAlwaysNonDegenerate.name
        elif float(n_degenerate) < float(ratio * len(self._test_cases)):
            return_type = NumericalAnalysedResultType.NumericalUsuallyNonDegenerate.name
        else:
            return_type = NumericalAnalysedResultType.NumericalDegenerate.name

        # Finished
        return return_type, float(n_degenerate) / float(len(self._test_cases))
