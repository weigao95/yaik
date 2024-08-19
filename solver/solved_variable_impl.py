import sympy as sp
from sympy.parsing.sympy_parser import parse_expr as parse_sympy_expr
from solver.solution_degeneration_record import DegenerateType, NumericalAnalysedResultType, SolutionDegenerateRecord
import solver.general_6dof.numerical_reduce_closure_equation as numerical_reduce
from solver.general_6dof.semi_symbolic_reduce import SemiSymbolicReduceInput
from solver.general_6dof.dh_utils import RevoluteVariable
import solver.equation_utils as equation_utils
from solver.equation_types import ScalarEquation
from typing import List, Optional, Dict, Tuple
from enum import Enum


class SolutionMethod(Enum):
    DefaultMethod = 0
    Invalid = 1
    InputParameter = 2
    OneVariableAlgebra = 3
    SinAndCos = 4
    Tangent = 5
    ArcSin = 6,
    ArcCos = 7,
    LinearSinCosType_1 = 8
    # Polynomial
    PolynomialDirect = 9
    PolynomialSin = 10
    PolynomialCos = 11
    # General 6-dof
    General6DoFNumericalReduce = 12
    # Linear solver
    LinearSinCosType_2 = 13


class VariableSolutionClassKey(Enum):
    ExplicitSolution = 0
    PolynomialSolution = 1
    General6DoFNumericalReduce = 2
    LinearSinCosType_2 = 3


class VariableSolutionImplBase(object):

    def __init__(self,
                 solved_var: sp.Symbol,
                 solution_method: str = SolutionMethod.Invalid.name,
                 degenerate_record: SolutionDegenerateRecord = SolutionDegenerateRecord(),
                 solved_from_equations: Optional[List[ScalarEquation]] = None):
        self._solved_variable: sp.Symbol = solved_var
        self._solution_method: str = solution_method
        self._degenerate_record: SolutionDegenerateRecord = degenerate_record
        self._solved_from_equations: Optional[List[ScalarEquation]] = solved_from_equations

    # Update the record
    def update_degenerate_record(self, degenerate_record: SolutionDegenerateRecord):
        self._degenerate_record = degenerate_record

    def update_numerical_analysis_result(self, result_in_str: str):
        self._degenerate_record.set_numerical_result(result_in_str)

    @property
    def solved_variable(self) -> sp.Symbol:
        return self._solved_variable

    @property
    def solved_from_scalar_equations(self) -> List[ScalarEquation]:
        return self._solved_from_equations

    @property
    def solution_method(self) -> str:
        return self._solution_method

    @property
    def degenerate_record(self) -> SolutionDegenerateRecord:
        return self._degenerate_record

    @property
    def is_explicit_solution(self):
        return VariableSolutionClassKey.ExplicitSolution.name == self.class_key()

    @property
    def is_polynomial(self):
        return VariableSolutionClassKey.PolynomialSolution.name == self.class_key()

    @property
    def is_general_6dof(self):
        return VariableSolutionClassKey.General6DoFNumericalReduce.name == self.class_key()

    # The abstract method
    def num_solutions(self):
        raise NotImplementedError

    def class_key(self) -> str:
        raise NotImplementedError


class ExplicitVariableSolutionImpl(VariableSolutionImplBase):

    def __init__(self,
                 solved_var: sp.Symbol,
                 solutions: List[sp.Expr],
                 solution_validity_checkers: List[sp.Expr],
                 solution_method: str = SolutionMethod.Invalid.name,
                 degenerate_record: SolutionDegenerateRecord = SolutionDegenerateRecord(),
                 solved_from_equations: Optional[List[ScalarEquation]] = None):
        super().__init__(solved_var, solution_method, degenerate_record, solved_from_equations)
        # One checker per solution
        assert solution_validity_checkers is not None
        assert len(solution_validity_checkers) == len(solutions)
        self._explicit_solutions = solutions
        self._solution_validity_checkers = solution_validity_checkers

    def num_solutions(self):
        return len(self._explicit_solutions)

    def class_key(self) -> str:
        return VariableSolutionClassKey.ExplicitSolution.name

    def count_explicit_solution_total_ops(self):
        """
        For an explicit solution, count the total number of operations in the solutions.
        Note that the result is summed over different solutions.
        """
        total_ops = 0
        for solution_i in self._explicit_solutions:
            ops_i = int(sp.count_ops(solution_i))
            total_ops += ops_i
        return total_ops

    @property
    def explicit_solutions(self):
        return self._explicit_solutions

    @property
    def argument_validity_checkers(self):
        return self._solution_validity_checkers

    @property
    def solution_validity_checkers(self):
        return self._solution_validity_checkers

    @staticmethod
    def make_explicit_solution(
            solved_variable: sp.Symbol,
            solutions: List[sp.Expr],
            solution_method: str,
            solve_from_equations: Optional[List[ScalarEquation]] = None,
            argument_valid_checkers: Optional[List[sp.Expr]] = None,
            degenerate_record: SolutionDegenerateRecord = SolutionDegenerateRecord.record_cannot_analyse()):
        # If validity checker is None, assume all valid
        validity_checker = argument_valid_checkers
        if validity_checker is None:
            validity_checker = list()
            for i in range(len(solutions)):
                validity_checker.append(sp.S.BooleanTrue)

        # General info
        explicit_solution = ExplicitVariableSolutionImpl(
            solved_var=solved_variable,
            solutions=solutions,
            solution_validity_checkers=validity_checker,
            solution_method=solution_method,
            degenerate_record=degenerate_record,
            solved_from_equations=solve_from_equations)
        return explicit_solution


class PolynomialVariableSolutionImpl(VariableSolutionImplBase):

    def __init__(self,
                 solved_var: sp.Symbol,
                 polynomial_dict: Dict[int, Tuple[sp.Expr, sp.Expr]],
                 solution_method: str = SolutionMethod.Invalid.name,
                 degenerate_record: SolutionDegenerateRecord = SolutionDegenerateRecord(),
                 solved_from_equations: Optional[List[ScalarEquation]] = None):
        super().__init__(solved_var, solution_method, degenerate_record, solved_from_equations)
        self._polynomial_to_solve = polynomial_dict

    # The abstract method
    def num_solutions(self):
        n_degree = 0
        for key in self._polynomial_to_solve:
            if key >= n_degree:
                n_degree = key
        if self._solution_method == SolutionMethod.PolynomialDirect.name:
            return n_degree
        else:
            return n_degree * 2

    def class_key(self) -> str:
        return VariableSolutionClassKey.PolynomialSolution.name

    def count_polynomial_coefficient_ops(self):
        """
        For a polynomial solution, count the total number of operations for computing
        the polynomial coefficient.
        """
        total_ops = 0
        for key in self._polynomial_to_solve:
            numerator, denominator = self._polynomial_to_solve[key]
            total_ops += numerator.count_ops()
            total_ops += denominator.count_ops()
        return total_ops

    @property
    def polynomial_to_solve(self):
        return self._polynomial_to_solve

    @staticmethod
    def make_polynomial_solution(
            solved_variable: sp.Symbol,
            solution_method: str,
            polynomial_dict: Dict[int, Tuple[sp.Expr, sp.Expr]],
            degenerate_record: SolutionDegenerateRecord = SolutionDegenerateRecord.record_cannot_analyse()):
        assert solution_method == SolutionMethod.PolynomialDirect.name \
               or solution_method == SolutionMethod.PolynomialSin.name \
               or solution_method == SolutionMethod.PolynomialCos.name
        # General info
        solution = PolynomialVariableSolutionImpl(solved_variable, polynomial_dict, solution_method, degenerate_record)
        return solution


class LinearSinCosType_2_SolutionImpl(VariableSolutionImplBase):

    def __init__(self, solved_var: sp.Symbol, A_matrix: sp.Matrix, rows_to_try: List[Tuple[int, int, int]]):
        super().__init__(solved_var=solved_var,
                         solution_method=SolutionMethod.General6DoFNumericalReduce.name,
                         degenerate_record=SolutionDegenerateRecord.record_cannot_analyse())
        self._A = A_matrix
        self._rows_to_try = rows_to_try

    def class_key(self) -> str:
        return VariableSolutionClassKey.LinearSinCosType_2.name

    def num_solutions(self):
        return 2

    @property
    def A_matrix(self):
        return self._A

    @property
    def rows_to_try(self):
        return self._rows_to_try


class General6DoFNumericalReduceSolutionImpl(VariableSolutionImplBase):

    def __init__(self,
                 reduce_out: numerical_reduce.NumericalReduceInput,
                 select_lines: Tuple[int],
                 semi_symbolic_reduce: Optional[SemiSymbolicReduceInput] = None):
        super().__init__(solved_var=reduce_out.var_in_lhs_matrix.variable_symbol,
                         solution_method=SolutionMethod.General6DoFNumericalReduce.name,
                         degenerate_record=SolutionDegenerateRecord.record_cannot_analyse())
        self._reduce_out = reduce_out
        self._select_lines = select_lines
        self._semi_symbolic_reduce = semi_symbolic_reduce

    def class_key(self) -> str:
        return VariableSolutionClassKey.General6DoFNumericalReduce.name

    def num_solutions(self):
        # No more solution than this, but there would be invalid ones
        return 24

    @property
    def reduce_out(self):
        return self._reduce_out

    @property
    def select_lines(self):
        return self._select_lines

    @property
    def has_semi_symbolic_reduce(self):
        return self._semi_symbolic_reduce is not None

    @property
    def semi_symbolic_reduce_record(self):
        return self._semi_symbolic_reduce

    @property
    def matrix_equation(self):
        return self._reduce_out.matrix_equation

    def lhs_matrices(self):
        reduce_out = self._reduce_out
        return reduce_out.lhs_A_sin, reduce_out.lhs_A_cos, reduce_out.lhs_C_const

    def rhs_matrix(self):
        return self._reduce_out.matrix_equation.rhs_matrix

    def solved_revolute_variable(self) -> RevoluteVariable:
        return self._reduce_out.var_in_lhs_matrix


def solution_impl_to_dict(solution: VariableSolutionImplBase) -> Dict:
    datamap = dict()
    if solution.is_explicit_solution:
        datamap['type'] = 'explicit'
        datamap['explicit_solution'] = explicit_solution_to_dict(solution)
        return datamap
    elif solution.is_polynomial:
        datamap['type'] = 'polynomial'
        datamap['polynomial_solution'] = polynomial_solution_to_dict(solution)
        return datamap
    elif solution.class_key() == VariableSolutionClassKey.General6DoFNumericalReduce.name:
        datamap['type'] = solution.class_key()
        datamap['reduce_solution'] = general_6dof_numerical_reduce_to_dict(solution)
        return datamap
    elif solution.class_key() == VariableSolutionClassKey.LinearSinCosType_2.name:
        datamap['type'] = solution.class_key()
        datamap['linear_type2_solution'] = linear_sin_cos_type2_to_dict(solution)
        return datamap
    else:
        raise NotImplementedError('type not supported yet')


def solution_impl_from_dict(datamap: Dict) -> VariableSolutionImplBase:
    solution_type = datamap['type']
    if solution_type == 'explicit':
        return explicit_solution_from_dict(datamap['explicit_solution'])
    elif solution_type == 'polynomial':
        return polynomial_solution_from_dict(datamap['polynomial_solution'])
    elif solution_type == VariableSolutionClassKey.General6DoFNumericalReduce.name:
        return general_6dof_numerical_reduce_from_dict(datamap['reduce_solution'])
    elif solution_type == VariableSolutionClassKey.LinearSinCosType_2.name:
        return linear_sin_cos_type2_from_dict(datamap['linear_type2_solution'])
    else:
        raise NotImplementedError('type not supported yet')


def solution_meta_to_dict(solution: VariableSolutionImplBase) -> Dict:
    meta_dict = dict()
    meta_dict['solved_variable'] = solution.solved_variable.name
    meta_dict['solution_method'] = solution.solution_method
    if solution.solved_from_scalar_equations is not None:
        solved_from_dict_list = list()
        for equ_i in solution.solved_from_scalar_equations:
            solved_from_dict_list.append(equ_i.to_dict())
        meta_dict['solve_from_equations'] = solved_from_dict_list
    meta_dict['degenerate_record'] = solution.degenerate_record.to_dict()
    return meta_dict


def solution_meta_from_dict(meta_dict: Dict):
    # Variable and solution
    solved_var = sp.Symbol(meta_dict['solved_variable'])
    solution_method = meta_dict['solution_method']

    # The scalar equations
    solved_from_scalar_equations = None
    if 'solve_from_equations' in meta_dict:
        solved_from_dict_list = meta_dict['solve_from_equations']
        solved_from_scalar_equations = list()
        for i in range(len(solved_from_dict_list)):
            equ_dict_i = solved_from_dict_list[i]
            equ_i = ScalarEquation.load_equation_from_dict(equ_dict_i)
            solved_from_scalar_equations.append(equ_i)

    # The degenerate record
    degenerate_record = SolutionDegenerateRecord()
    degenerate_record.from_dict(meta_dict['degenerate_record'])
    return solved_var, solution_method, solved_from_scalar_equations, degenerate_record


def explicit_solution_to_dict(solution: ExplicitVariableSolutionImpl) -> Dict:
    datamap = dict()
    datamap['meta'] = solution_meta_to_dict(solution)
    datamap['explicit_solutions'] = \
        equation_utils.expression_list_to_string_representation(solution.explicit_solutions)
    datamap['argument_checkers'] = \
        equation_utils.expression_list_to_string_representation(solution.solution_validity_checkers)
    return datamap


def explicit_solution_from_dict(datamap: Dict) -> ExplicitVariableSolutionImpl:
    assert 'explicit_solutions' in datamap
    assert 'argument_checkers' in datamap
    solved_var, solution_method, solved_from_scalar_equations, degenerate_record = \
        solution_meta_from_dict(datamap['meta'])
    explicit_solutions = equation_utils.parse_string_expressions(datamap['explicit_solutions'])
    solution_checkers = equation_utils.parse_string_expressions(datamap['argument_checkers'])
    solution = ExplicitVariableSolutionImpl(
        solved_var,
        explicit_solutions, solution_checkers,
        solution_method,
        degenerate_record,
        solved_from_scalar_equations)
    return solution


def polynomial_solution_to_dict(solution: PolynomialVariableSolutionImpl) -> Dict:
    datamap = dict()
    datamap['meta'] = solution_meta_to_dict(solution)

    # Make the dict for polynomial
    numerator_dict: Dict[int, str] = dict()
    denominator_dict: Dict[int, str] = dict()
    for key in solution.polynomial_to_solve:
        numerator, denominator = solution.polynomial_to_solve[key]
        numerator_str_value = str(numerator)
        denominator_str_value = str(denominator)
        numerator_dict[key] = numerator_str_value
        denominator_dict[key] = denominator_str_value

    datamap['numerator_dict'] = numerator_dict
    datamap['denominator_dict'] = denominator_dict
    return datamap


def polynomial_solution_from_dict(datamap: Dict) -> PolynomialVariableSolutionImpl:
    assert 'numerator_dict' in datamap
    assert 'denominator_dict' in datamap
    numerator_dict = datamap['numerator_dict']
    denominator_dict = datamap['denominator_dict']
    polynomial_to_solve = dict()
    for key in numerator_dict:
        str_numerator = numerator_dict[key]
        str_denominator = denominator_dict[key]
        expr_numerator = parse_sympy_expr(str_numerator)
        expr_denominator = parse_sympy_expr(str_denominator)
        polynomial_to_solve[int(key)] = (expr_numerator, expr_denominator)

    solved_var, solution_method, solved_from_scalar_equations, degenerate_record = \
        solution_meta_from_dict(datamap['meta'])
    solution = PolynomialVariableSolutionImpl(
        solved_var, polynomial_to_solve,
        solution_method, degenerate_record, solved_from_scalar_equations)
    return solution


def general_6dof_numerical_reduce_to_dict(solution: General6DoFNumericalReduceSolutionImpl):
    datamap = dict()
    datamap['meta'] = solution_meta_to_dict(solution)
    datamap['reduce_out'] = solution.reduce_out.to_dict()
    datamap['select_lines'] = solution.select_lines
    if solution.has_semi_symbolic_reduce:
        datamap['semi_symbolic_reduce'] = solution.semi_symbolic_reduce_record.to_dict()
    return datamap


def general_6dof_numerical_reduce_from_dict(datamap: Dict) -> General6DoFNumericalReduceSolutionImpl:
    solved_var, solution_method, solved_from_scalar_equations, degenerate_record = \
        solution_meta_from_dict(datamap['meta'])
    reduce_out = numerical_reduce.NumericalReduceInput.load_from_dict(datamap['reduce_out'])
    select_lines = datamap['select_lines']

    # The optional parameter
    semi_symbolic_reduce = None
    if 'semi_symbolic_reduce' in datamap:
        semi_symbolic_reduce = SemiSymbolicReduceInput.load_from_dict(datamap['semi_symbolic_reduce'])

    # Make the struct
    solution = General6DoFNumericalReduceSolutionImpl(reduce_out, select_lines, semi_symbolic_reduce)
    return solution


def linear_sin_cos_type2_to_dict(solution: LinearSinCosType_2_SolutionImpl):
    datamap = dict()
    datamap['meta'] = solution_meta_to_dict(solution)
    datamap['A'] = equation_utils.sp_matrix_to_dict_representation(solution.A_matrix)
    datamap['rows_to_try'] = solution.rows_to_try
    return datamap


def linear_sin_cos_type2_from_dict(datamap: Dict) -> LinearSinCosType_2_SolutionImpl:
    solved_var, _, _, _ = solution_meta_from_dict(datamap['meta'])
    A = equation_utils.parse_sp_matrix_from_dict(datamap['A'])
    rows_to_try = datamap['rows_to_try']
    solution = LinearSinCosType_2_SolutionImpl(solved_var, A, rows_to_try)
    return solution
