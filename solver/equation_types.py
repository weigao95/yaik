from enum import Enum
from typing import List, Optional, Dict, Tuple
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr as parse_sympy_expr
import numpy as np
import attr


# The type of the unknown variable
class UnknownType(Enum):
    Revolute = 1
    Translational = 2


class Unknown(object):
    def __init__(self,
                 symbol: sp.Symbol,
                 unknown_type: str = UnknownType.Revolute.name,
                 lower_bound: float = None,
                 upper_bound: float = None,
                 value_for_degenerate_check: List[sp.Symbol] = list()):
        """
        A very simple record that maintains the unknown symbol and type
        :param symbol:
        :param unknown_type:
        :param lower_bound:
        :param upper_bound:
        :param value_for_degenerate_check: A set of interesting value that may lead to solution degeneration
        """
        self._symbol = symbol
        self._unknown_type = unknown_type

        # The bound
        if lower_bound is None:
            lower_bound = - float(np.pi) if unknown_type == UnknownType.Revolute.name else -1.0
        if upper_bound is None:
            upper_bound =   float(np.pi) if unknown_type == UnknownType.Revolute.name else  1.0
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        # The degenerate check value
        self._degenerate_check_value = value_for_degenerate_check
        if len(self._degenerate_check_value) < 4:
            # Some default value
            self._degenerate_check_value.append(sp.S.Zero)
            self._degenerate_check_value.append(sp.pi / 2)
            self._degenerate_check_value.append(sp.pi)
            self._degenerate_check_value.append(- sp.pi / 2)

    @property
    def symbol(self):
        return self._symbol

    @property
    def unknown_type(self):
        return self._unknown_type

    @property
    def is_revolute(self) -> bool:
        return self._unknown_type == UnknownType.Revolute.name

    @property
    def is_translational(self) -> bool:
        return self._unknown_type == UnknownType.Translational.name

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def degenerate_check_value(self):
        return self._degenerate_check_value

    # Serialize and de-serialize
    def to_dict(self) -> Dict:
        data_map = dict()
        data_map['symbol'] = self._symbol.name
        data_map['unknown_type'] = self._unknown_type

        # Parse of bounds
        data_map['lb'] = float(self.lower_bound)
        data_map['ub'] = float(self.upper_bound)

        # Degenerate check
        degenerate_check_expr: List[str] = list()
        for expr_i in self._degenerate_check_value:
            expr_i_str = str(expr_i)
            degenerate_check_expr.append(expr_i_str)
        data_map['degenerate_check_value'] = degenerate_check_expr

        # OK
        return data_map

    def from_dict(self, data_map: Dict):
        symbol_name = data_map['symbol']
        self._symbol = sp.Symbol(symbol_name)
        self._unknown_type = data_map['unknown_type']

        # Parse of bounds
        self._lower_bound = float(data_map['lb'])
        self._upper_bound = float(data_map['ub'])

        # Parse of check expr
        degenerate_check_expr: List[str] = data_map['degenerate_check_value']
        self._degenerate_check_value.clear()
        for expr_i_str in degenerate_check_expr:
            expr_i = parse_sympy_expr(expr_i_str)
            self._degenerate_check_value.append(expr_i)

    @staticmethod
    def load_unknown_from_dict(data_map):
        tmp_unknown = Unknown(sp.Symbol('__load_tmp__'))
        tmp_unknown.from_dict(data_map)
        return tmp_unknown


class ScalarEquationType(Enum):
    Default = 0  # No-explicit type
    Rotation = 1  # appear in [0:3, 0:3] of the homogenous transformation matrix
    Translation = 2  # appear in [0:3, 3] of the transformation matrix
    SumOfAngle = 3  # New sum-of-angle equation
    SumOfSquare = 4  # New sum-of-square equation


class ScalarEquation(object):

    def __init__(self, lhs: sp.Expr, rhs: sp.Expr, equation_type: str = ScalarEquationType.Default.name):
        self._lhs = lhs
        self._rhs = rhs
        self._equation_type = equation_type

    def __repr__(self):
        return "%s = %s" % (self._lhs, self._rhs)

    def __eq__(self, other):
        if other is None:
            return False

        # This one is somehow less tighter than below
        minus_expr = (self.lhs - self.rhs) - (other.lhs - other.rhs)
        if minus_expr == 0:
            return True

        # Old one
        # if self._lhs - other._lhs == 0 and self._rhs - other._rhs == 0:
        #    return True

        # Not same to the capability of sympy
        return False

    def __ne__(self, other):
        if not self.__eq__(other):
            return True
        else:
            return False

    def __hash__(self):
        return hash(str(self._lhs) + str(self._rhs))

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    @property
    def equation_type(self):
        return self._equation_type

    @property
    def is_sum_of_angle(self):
        return self._equation_type == ScalarEquationType.SumOfAngle.name

    @property
    def is_sum_of_square(self):
        return self._equation_type == ScalarEquationType.SumOfSquare.name

    # Serialize and de-serialize
    def to_dict(self) -> Dict:
        data_map = dict()
        data_map['lhs'] = str(self._lhs)
        data_map['rhs'] = str(self._rhs)
        data_map['type'] = self._equation_type
        return data_map

    def from_dict(self, data_map: Dict):
        self._equation_type = data_map['type']
        lhs_expr = parse_sympy_expr(data_map['lhs'])
        rhs_expr = parse_sympy_expr(data_map['rhs'])
        self._lhs = lhs_expr
        self._rhs = rhs_expr

    @staticmethod
    def load_equation_from_dict(data_map):
        dummy_expr = ScalarEquation(sp.S.Zero, sp.S.Zero)
        dummy_expr.from_dict(data_map)
        return dummy_expr


class TranslationalEquation(object):

    def __init__(self, equation_x: ScalarEquation, equation_y: ScalarEquation, equation_z: Optional[ScalarEquation]):
        self._equation_x = ScalarEquation(equation_x.lhs, equation_x.rhs, ScalarEquationType.Translation.name)
        self._equation_y = ScalarEquation(equation_y.lhs, equation_y.rhs, ScalarEquationType.Translation.name)

        # The last one is optional
        if equation_z is None:
            self._equation_z = None
        else:
            self._equation_z = ScalarEquation(equation_z.lhs, equation_z.rhs, ScalarEquationType.Translation.name)

    @property
    def x(self):
        return self._equation_x

    @property
    def y(self):
        return self._equation_y

    @property
    def z(self):
        return self._equation_z


class MatrixEquation(object):

    def __init__(self, Td: sp.Matrix, Ts: sp.Matrix):
        self._Td = sp.zeros(4)  # LHS (T desired)
        self._Ts = sp.zeros(4)  # RHS (T symbolic)
        for i in range(0, 3):  # just first 3 rows
            for j in range(0, 4):  # all 4 cols
                self._Td[i, j] = Td[i, j]
                self._Ts[i, j] = Ts[i, j]
        self._Td[3, 3] = 1  # handle row 4
        self._Ts[3, 3] = 1

    def build_scalar_equations(self) -> List[ScalarEquation]:
        result = list()
        for i in range(3):
            for j in range(4):
                result.append(ScalarEquation(self._Td[i, j], self._Ts[i, j]))
        return result

    @property
    def Ts(self):
        return self._Ts

    @property
    def Td(self):
        return self._Td


@attr.s
class SumOfSquareHint(object):
    """
    Provide hint for sum-of-square transform
    """
    equ_1: ScalarEquation = attr.ib()
    equ_2: ScalarEquation = attr.ib()
    equ_3: Optional[ScalarEquation] = attr.ib()

    # Serialize and de-serialize
    def to_dict(self) -> Dict:
        data_map = dict()
        data_map['equ_1'] = self.equ_1.to_dict()
        data_map['equ_2'] = self.equ_2.to_dict()
        if self.equ_3 is not None:
            data_map['equ_3'] = self.equ_3.to_dict()
        return data_map

    def from_dict(self, data_map: Dict):
        self.equ_1 = ScalarEquation.load_equation_from_dict(data_map['equ_1'])
        self.equ_2 = ScalarEquation.load_equation_from_dict(data_map['equ_2'])
        if 'equ_3' in data_map:
            self.equ_3 = ScalarEquation.load_equation_from_dict(data_map['equ_3'])
        else:
            self.equ_3 = None

    @staticmethod
    def load_hint_from_dict(data_map):
        dummy_equation_0 = ScalarEquation(sp.S.Zero, sp.S.Zero)
        dummy_equation_1 = ScalarEquation(sp.S.Zero, sp.S.Zero)
        sos_hint = SumOfSquareHint(dummy_equation_0, dummy_equation_1, None)
        sos_hint.from_dict(data_map)
        return sos_hint


@attr.s
class EquationInput(object):
    """
    The raw input to the polynomial solver
    """
    scalar_equations: List[ScalarEquation] = attr.ib()
    sum_of_square_hint: List[SumOfSquareHint] = attr.ib()

    # Serialize and de-serialize
    def to_dict(self):
        data_map = dict()
        equations_data: List[Dict] = list()
        for i in range(len(self.scalar_equations)):
            expr_map_i = self.scalar_equations[i].to_dict()
            equations_data.append(expr_map_i)
        data_map['scalar_equations'] = equations_data

        # Sos hint
        sos_data: List[Dict] = list()
        for i in range(len(self.sum_of_square_hint)):
            sos_data_i = self.sum_of_square_hint[i].to_dict()
            sos_data.append(sos_data_i)
        data_map['sos_hint'] = sos_data
        return data_map

    def from_dict(self, data_map):
        equations_data = data_map['scalar_equations']
        self.scalar_equations.clear()
        for i in range(len(equations_data)):
            expr_map_i = equations_data[i]
            expr_i = ScalarEquation(sp.S.Zero, sp.S.Zero)
            expr_i.from_dict(expr_map_i)
            self.scalar_equations.append(expr_i)

        # Load sos hint
        if 'sos_hint' not in data_map:
            return
        sos_data = data_map['sos_hint']
        self.sum_of_square_hint.clear()
        for i in range(len(sos_data)):
            sos_data_i = sos_data[i]
            sos_hint = SumOfSquareHint.load_hint_from_dict(sos_data_i)
            self.sum_of_square_hint.append(sos_hint)


@attr.s
class CollectedEquations(object):
    one_unknown_equations: List[ScalarEquation] = attr.ib()
    two_unknown_equations: List[ScalarEquation] = attr.ib()
    other_equations: List[ScalarEquation] = attr.ib()

    def print_stdout(self):
        # Logging
        if len(self.one_unknown_equations) > 0:
            print('One unknowns')
            for equation in self.one_unknown_equations:
                print(equation.lhs, ' == ', equation.rhs)

        if len(self.two_unknown_equations) > 0:
            print('Two unknowns')
            for equation in self.two_unknown_equations:
                print(equation.lhs, ' == ', equation.rhs)

        if len(self.other_equations) > 0:
            print('At least three unknowns')
            for equation in self.other_equations:
                print(equation.lhs, ' == ', equation.rhs)


@attr.s
class NumericalAnalyseContext(object):
    # All the parameter should be valued or bounded
    parameter_values: Dict[sp.Symbol, float] = dict()
    parameter_bounds: Dict[sp.Symbol, Tuple[float, float]] = dict()

    # An potentially empty test case lists
    numerical_test_cases: List[Dict[sp.Symbol, float]] = list()
