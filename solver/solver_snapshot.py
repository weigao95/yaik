from solver.equation_types import ScalarEquation, SumOfSquareHint, Unknown
from typing import List, Dict, Set, Tuple
import sympy as sp
import attr
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


@attr.s
class SolverSnapshot(object):
    scalar_equations: List[ScalarEquation] = attr.ib()
    sos_hints: List[SumOfSquareHint] = attr.ib()
    unknowns: List[Unknown] = attr.ib()
    all_parameters: Set[sp.Symbol] = attr.ib()
    parameter_values: Dict[sp.Symbol, float] = dict()
    parameter_bounds: Dict[sp.Symbol, Tuple[float, float]] = dict()
    numerical_test_cases: List[Dict[sp.Symbol, float]] = list()


def solver_snapshot_dict_raw(
        scalar_equations: List[ScalarEquation],
        sos_hints: List[SumOfSquareHint],
        unknowns: List[Unknown],
        all_parameters: Set[sp.Symbol],
        parameter_values: Dict[sp.Symbol, float],
        parameter_bounds: Dict[sp.Symbol, Tuple[float, float]] = dict(),
        numerical_test_cases: List[Dict[sp.Symbol, float]] = list()) -> Dict:
    data_map = dict()

    # For scalar equations
    scalar_equation_map_list = list()
    for i in range(len(scalar_equations)):
        equ_map_i = scalar_equations[i].to_dict()
        scalar_equation_map_list.append(equ_map_i)
    data_map['scalar_equations'] = scalar_equation_map_list

    # For sos hint
    sos_hints_map_list = list()
    for i in range(len(sos_hints)):
        hint_map_i = sos_hints[i].to_dict()
        sos_hints_map_list.append(hint_map_i)
    data_map['sos_hints'] = sos_hints_map_list

    # For unknowns
    unknown_map_list = list()
    for i in range(len(unknowns)):
        unknown_map_i = unknowns[i].to_dict()
        unknown_map_list.append(unknown_map_i)
    data_map['unknowns'] = unknown_map_list

    # For parameters
    parameter_list = list()
    for parameter_i in all_parameters:
        parameter_list.append(parameter_i.name)
    data_map['all_parameters'] = parameter_list

    # Parameter value
    parameter_value_map = dict()
    for k in parameter_values:
        str_key = k.name
        value = parameter_values[k]
        parameter_value_map[str_key] = float(value)
    data_map['parameter_values'] = parameter_value_map

    # Parameter bounds
    parameter_bounds_map = dict()
    for k in parameter_bounds:
        str_key = k.name
        value = parameter_bounds[k]
        parameter_bounds_map[str_key] = value
    data_map['parameter_bounds'] = parameter_bounds_map

    # The numerical examples
    # This can be large
    numerical_example_list: List[Dict[str, float]] = list()
    for i in range(len(numerical_test_cases)):
        test_case_map_i = numerical_test_cases[i]
        string_map_i: Dict[str, float] = dict()
        for var in test_case_map_i:
            value = test_case_map_i[var]
            string_map_i[var.name] = float(value)
        numerical_example_list.append(string_map_i)
    data_map['numerical_test_cases'] = numerical_example_list

    # Ok
    return data_map


def solver_snapshot_dict(snapshot: SolverSnapshot) -> Dict:
    return solver_snapshot_dict_raw(
        snapshot.scalar_equations,
        snapshot.sos_hints,
        snapshot.unknowns,
        snapshot.all_parameters,
        snapshot.parameter_values,
        snapshot.parameter_bounds,
        snapshot.numerical_test_cases
    )


def read_solver_snapshot_dict(data_map: Dict) -> SolverSnapshot:
    # For scalar equations
    scalar_equations: List[ScalarEquation] = list()
    scalar_equation_map_list = data_map['scalar_equations']
    for i in range(len(scalar_equation_map_list)):
        equ_map_i = scalar_equation_map_list[i]
        equ_i = ScalarEquation.load_equation_from_dict(equ_map_i)
        scalar_equations.append(equ_i)

    # For sos hint
    sos_hints_map_list = data_map['sos_hints']
    sos_hints = list()
    for i in range(len(sos_hints_map_list)):
        hint_map_i = sos_hints_map_list[i]
        hint_i = SumOfSquareHint.load_hint_from_dict(hint_map_i)
        sos_hints.append(hint_i)

    # For unknowns
    unknown_map_list = data_map['unknowns']
    unknowns = list()
    for i in range(len(unknown_map_list)):
        unknown_map_i = unknown_map_list[i]
        unknown_i = Unknown.load_unknown_from_dict(unknown_map_i)
        unknowns.append(unknown_i)

    # The parameters
    parameter_list = data_map['all_parameters']
    all_parameters = set()
    for parameter_name in parameter_list:
        all_parameters.add(sp.Symbol(parameter_name))

    # Parameter values
    parameter_value_map = data_map['parameter_values']
    symbol_value_map = dict()
    for k in parameter_value_map:
        symbol_key = sp.Symbol(k)
        value = parameter_value_map[k]
        symbol_value_map[symbol_key] = value

    # Parameter bounds
    parameter_bounds_map = data_map['parameter_bounds']
    symbol_bounds_map = dict()
    for k in parameter_bounds_map:
        symbol_key = sp.Symbol(k)
        value = parameter_bounds_map[k]
        symbol_bounds_map[symbol_key] = value

    # The numerical examples
    numerical_test_cases_list = list()
    if 'numerical_test_cases' in data_map:
        test_example_string_map_list = data_map['numerical_test_cases']
        numerical_test_cases_list: List[Dict[sp.Symbol, float]] = list()
        for i in range(len(test_example_string_map_list)):
            string_map_i = test_example_string_map_list[i]
            symbol_map_i: Dict[sp.Symbol, float] = dict()
            for var_name in string_map_i:
                value = string_map_i[var_name]
                symbol_map_i[sp.Symbol(var_name)] = float(value)
            numerical_test_cases_list.append(symbol_map_i)

    # Make the snapshot
    snapshot = SolverSnapshot(
        scalar_equations,
        sos_hints,
        unknowns,
        all_parameters
    )
    snapshot.parameter_values = symbol_value_map
    snapshot.parameter_bounds = symbol_bounds_map
    snapshot.numerical_test_cases = numerical_test_cases_list
    return snapshot


def save_solver_snapshot(snapshot: SolverSnapshot, save_path: str):
    snapshot_dict = solver_snapshot_dict(snapshot)
    with open(save_path, 'w') as file_stream:
        yaml.dump(snapshot_dict, file_stream, Dumper=Dumper)
    file_stream.close()


def load_solver_snapshot(load_path: str) -> SolverSnapshot:
    with open(load_path, 'r') as file_stream:
        data_map = yaml.load(file_stream, Loader=Loader)
        snapshot = read_solver_snapshot_dict(data_map)
    file_stream.close()
    return snapshot
