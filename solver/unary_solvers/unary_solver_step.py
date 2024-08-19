from solver.solved_variable import VariableSolution
from solver.equation_types import ScalarEquationType
from solver.equation_utils import CollectedEquations
from solver.unary_solvers.tangent_solver import UnaryVariableSolver, UnaryTangentSolver
from solver.unary_solvers.one_var_algebra import UnaryOneVariableAlgebraSolver
from solver.unary_solvers.sin_and_cos import UnarySinAndCosSolver
from solver.unary_solvers.linear_sin_cos_type_1 import UnaryLinearSolverType_1
from solver.unary_solvers.asin_or_acos import UnaryArcSinSolver, UnaryArcCosSolver
from typing import List, Dict
import sympy as sp
import copy
import multiprocessing


def parallel_solve_processor(dict_in) -> List[VariableSolution]:
    """
    Just a parallel processor that encapsulate the solver.try_solve
    """
    collected_equations = dict_in['collected_equations']
    var_to_try = dict_in['var_to_try']
    solver_in: UnaryVariableSolver = dict_in['solver']
    unknowns = dict_in['unknowns']
    solver_output = solver_in.try_solve(collected_equations, var_to_try, unknowns)
    return solver_output


def parallel_solve_processor_list_output(dict_in, output_list: List[List[VariableSolution]], lock=None):
    this_output = parallel_solve_processor(dict_in)

    # Append a result
    if lock is not None:
        lock.acquire()
    output_list.append(this_output)
    if lock is not None:
        lock.release()


class UnarySolverStep(object):

    def __init__(self):
        self._unary_algebra_solver = UnaryOneVariableAlgebraSolver()
        self._other_solvers: List[UnaryVariableSolver] = list()

    def append_solver(self, solver: UnaryVariableSolver):
        self._other_solvers.append(solver)

    @staticmethod
    def make_default():
        default_solver = UnarySolverStep()
        default_solver.append_solver(UnarySinAndCosSolver())
        default_solver.append_solver(UnaryLinearSolverType_1())
        default_solver.append_solver(UnaryTangentSolver())
        default_solver.append_solver(UnaryArcSinSolver())
        default_solver.append_solver(UnaryArcCosSolver())
        return default_solver

    def _try_algebra_solution(self,
                              collected_equations: CollectedEquations,
                              unknowns: List[sp.Symbol]) -> (List[VariableSolution], bool):
        """
        Try solving the equations using algebra solver.
        :return: the first tuple element is the variable solution,
                 the second tuple element is a bool indicating whether we found sum-of-angle solution
        """
        # The final output
        solutions = list()

        # First try algebra solver, record if we have solutions from sum-of-angle equations
        has_soa_solution: bool = False
        for var_to_try in unknowns:
            algebra_sol = self._unary_algebra_solver.try_solve(
                collected_equations, var_to_try, unknowns)
            for this_sol in algebra_sol:
                scalar_equs = this_sol.solved_from_scalar_equations
                assert len(scalar_equs) == 1
                if scalar_equs[0].equation_type == ScalarEquationType.SumOfAngle.name:
                    has_soa_solution = True
            solutions.extend(algebra_sol)

        # Just return (because SOA equations very good ones)
        return solutions, has_soa_solution

    def solve_step(self, collected_equations: CollectedEquations,
                   unknowns: List[sp.Symbol]) -> List[VariableSolution]:
        # The final output
        solutions, has_soa_solution = self._try_algebra_solution(
            collected_equations, unknowns)

        # Just return (because SOA equations very good ones)
        if has_soa_solution:
            print('We have solutions from sum-of-angle equations, thus do not continue')
            return solutions

        # Get as many solutions as possible
        for var_to_try in unknowns:
            for solver in self._other_solvers:
                sol = solver.try_solve(collected_equations, var_to_try, unknowns)
                if len(sol) > 0:
                    solutions.extend(sol)
        return solutions

    def solve_step_parallel(self,
                            collected_equations: CollectedEquations,
                            unknowns: List[sp.Symbol],
                            with_timeout: bool = False):
        # The final output
        solutions, has_soa_solution = self._try_algebra_solution(
            collected_equations, unknowns)

        # Just return (because SOA equations very good ones)
        if has_soa_solution:
            print('We have solutions from sum-of-angle equations, thus do not continue')
            return solutions

        # Make the parallel input args
        input_args = list()
        for var_to_try in unknowns:
            for this_solver in self._other_solvers:
                dict_in = dict()
                dict_in['collected_equations'] = collected_equations
                dict_in['var_to_try'] = var_to_try
                dict_in['solver'] = this_solver
                dict_in['unknowns'] = unknowns
                input_args.append(dict_in)

        # OK, map it
        if with_timeout:
            output = self.parallel_run(input_args)
        else:
            n_process = min(32, len(input_args))
            n_process = max(n_process, 1)
            print('Try solving the equations using rule solver. The candidate number: ', len(input_args))
            output: List[List[VariableSolution]] = list()
            with multiprocessing.Pool(n_process) as pool:
                output = pool.map(parallel_solve_processor, input_args)

        # Collect the result
        for this_sol in output:
            if len(this_sol) > 0:
                solutions.extend(this_sol)
        return solutions

    def get_solvers(self):
        solver_list = list()
        solver_list.append(self._unary_algebra_solver)
        for solver in self._other_solvers:
            solver_list.append(solver)
        return solver_list

    @staticmethod
    def parallel_run(input_args: List[Dict], timeout_seconds: int = 300) -> List[List[VariableSolution]]:
        n_processor = min(32, int(len(input_args)))
        n_processor = max(n_processor, 1)
        print('Try solving the equations using rule solver. The candidate number: ', len(input_args))
        lock = multiprocessing.Lock()
        manager = multiprocessing.Manager()
        shared_result_list = manager.list()
        processed_offset = 0
        while processed_offset < len(input_args):
            processor_list = list()
            for i in range(n_processor):
                offset_i = processed_offset + i
                if offset_i < len(input_args):
                    # OK, make the processor
                    processor = multiprocessing.Process(
                        target=parallel_solve_processor_list_output,
                        args=(input_args[offset_i], shared_result_list, lock))
                    processor_list.append(processor)

            # Start the processor with timeout
            for p in processor_list:
                p.start()
            for p in processor_list:
                p.join(timeout=timeout_seconds)
                p.terminate()

            # Update the offset
            processed_offset += n_processor

        # To usual list
        output_list = list()
        for elem in shared_result_list:
            output_list.append(copy.deepcopy(elem))
        return output_list
