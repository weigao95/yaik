import os.path
import yaml
from solver.build_equations import IkEquationCollection
import solver.equation_utils as equation_utils
from fk.robots import Unknown
import fk.fk_equations as fk_equations
import multiprocessing
import sympy as sp
import numpy as np
from typing import Set, List, Dict, Tuple
import copy
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def collect_bounded_parameters(
        ik_input: IkEquationCollection,
        include_unknown: bool = True) -> Tuple[List[sp.Symbol], List[float], List[float]]:
    """
    Collect all the parameter in the ik problem and their bounds
    :param ik_input: the input to ik problem
    :param include_unknown: should we include the unknown and their bounds
    :return: the map from symbol to their bounds
    """
    # The parameter need generate
    parameters_need_generate: List[sp.Symbol] = list()
    parameters_lb: List[float] = list()
    parameters_ub: List[float] = list()
    ik_target_symbols = fk_equations.ik_target_symbols()

    # First iterate the symbolic parameters
    for parameter in ik_input.all_symbol_set:
        if parameter in ik_target_symbols:
            # We don't need to generate this
            continue
        elif parameter in ik_input.parameter_value_dict:
            # This is valued, we can also ignore this
            continue
        elif parameter in ik_input.parameter_bound_dict:
            lb, ub = ik_input.parameter_bound_dict[parameter]
            parameters_need_generate.append(parameter)
            parameters_lb.append(lb)
            parameters_ub.append(ub)
        else:
            # This must be a unknown, first check it
            unknown_idx = equation_utils.find_unknown_idx(ik_input.used_unknowns, parameter.name)
            assert unknown_idx >= 0
            if include_unknown:
                lb = float(ik_input.used_unknowns[unknown_idx].lower_bound)
                ub = float(ik_input.used_unknowns[unknown_idx].upper_bound)
                parameters_need_generate.append(parameter)
                parameters_lb.append(lb)
                parameters_ub.append(ub)

    # Now we are finished
    return parameters_need_generate, parameters_lb, parameters_ub


def compute_fk_processor(tuple_in: Tuple):
    # Gather the info
    ee_pose, subst_map, idx = tuple_in
    ee_pose_value_sp = ee_pose.subs(subst_map)

    # Check and to numpy
    ee_pose_np = np.eye(4)
    for r in range(3):
        for c in range(4):
            rc_sp: sp.Expr = ee_pose_value_sp[r, c]
            ee_pose_np[r, c] = float(rc_sp)

    # Return the np pose
    return ee_pose_np


def generate_test_cases(
        ik_input: IkEquationCollection,
        subset_variable_value: Dict[sp.Symbol, float] = dict(),
        n_test_cases: int = 2000,
        degenerate_ratio: float = 0.3) -> List[Dict[sp.Symbol, float]]:
    """
    Build the numerical test case for later solving use. The output
    is in the form of mapping from symbolic parameter to float, which can
    be directly applied to sympy.subs
    :param ik_input:
    :param subset_variable_value: explicitly request the value of a subset of variables.
                                  Typically, this is used in branched solution
    :param n_test_cases
    :param degenerate_ratio: if subset_variable_value.empty(), then set a subset of variables to some interesting values
                             else, do nothing.
    """
    # Check the size
    n_degenerate = int(n_test_cases * degenerate_ratio)

    # The parameter need generate
    parameters_need_generate, parameters_lb, parameters_ub = collect_bounded_parameters(ik_input, include_unknown=True)

    # Build all random sample at once
    # Note that random sample can also be slow
    parameters_random_values: List[np.ndarray] = list()
    for i in range(len(parameters_need_generate)):
        lb_i = parameters_lb[i]
        ub_i = parameters_ub[i]
        samples_i = np.random.uniform(lb_i, ub_i, n_test_cases)
        parameters_random_values.append(samples_i)

    # First build non-degenerate ones
    dict_with_valued_parameters: Dict[sp.Symbol, float] = copy.deepcopy(ik_input.parameter_value_dict)
    subst_map_list: List[Dict[sp.Symbol, float]] = list()
    for i in range(n_test_cases):
        subst_map = copy.deepcopy(dict_with_valued_parameters)
        for j in range(len(parameters_need_generate)):
            parameter_j = parameters_need_generate[j]
            value_j = parameters_random_values[j][i]
            subst_map[parameter_j] = value_j
        # OK, append to map_list
        subst_map_list.append(subst_map)

    # Next the degenerate, set one value to degenerate
    def set_degenerate_value_for_first_n_element(n_element: int):
        random_idx = np.random.randint(0, high=len(ik_input.used_unknowns), size=(n_element, ))
        for i in range(n_element):
            perturbed_unknown_idx_i = random_idx[i]
            this_unknown: Unknown = ik_input.used_unknowns[perturbed_unknown_idx_i]
            value_to_check = this_unknown.degenerate_check_value
            value_to_check_idx = np.random.randint(low=0, high=len(value_to_check), size=1)[0]

            # Randomly set some unknown to special value
            selected_value = value_to_check[value_to_check_idx]

            # Update the map
            subst_map_list[i][this_unknown.symbol] = selected_value

    # Set some degenerate value
    if len(subset_variable_value) == 0:
        # First set n_degenerate instance to 1-degenerate value
        set_degenerate_value_for_first_n_element(n_element=n_degenerate)

        # Half of them has two degenerate, and so on
        max_n_degenerate = min(len(ik_input.used_unknowns), 4)
        div_by = 2
        for i in range(max_n_degenerate + 1):
            n_element_i = int(n_degenerate / div_by)
            set_degenerate_value_for_first_n_element(n_element=n_element_i)
            div_by *= 2
    else:
        # Update the subst map
        for i in range(len(subst_map_list)):
            for variable in subset_variable_value:
                variable_value = subset_variable_value[variable]
                subst_map_list[i][variable] = variable_value

    # Build fk
    ee_pose = ik_input.ee_pose
    input_args = list()
    for i in range(n_test_cases):
        subst_map_i = subst_map_list[i]
        input_args.append((ee_pose, subst_map_i, i))

    # OK, do the mapping
    n_process = min(32, len(input_args))
    n_process = max(n_process, 1)
    print('Generate {n} numerical test cases. The # of processors is {m}: '.format(n=n_test_cases, m=n_process))
    output: List[np.ndarray] = list()
    with multiprocessing.Pool(n_process) as pool:
        output = pool.map(compute_fk_processor, input_args)

    # Collect the result
    for i in range(n_test_cases):
        ee_pose_np_i = output[i]
        pose_map_i = fk_equations.ik_target_subst_map(ee_pose_np_i)
        subst_map_list[i].update(pose_map_i)

    # Finished
    return subst_map_list


def save_test_cases(numerical_test_cases: List[Dict[sp.Symbol, float]], save_path: str):
    """
    Save a large list of numerical test cases for later use
    """
    # Turn into string dict
    string_map_list = list()
    for map_i in numerical_test_cases:
        elem_str_dict = dict()
        for symbol_j in map_i:
            elem_str_dict[symbol_j.name] = float(map_i[symbol_j])
        string_map_list.append(elem_str_dict)

    # OK, save it
    data_map = dict()
    data_map['n_test_cases'] = len(numerical_test_cases)
    data_map['numerical_test_cases'] = string_map_list
    with open(save_path, 'w') as write_stream:
        yaml.dump(data_map, write_stream, Dumper=Dumper)
    write_stream.close()


def load_test_cases(load_path: str) -> List[Dict[sp.Symbol, float]]:
    assert os.path.exists(load_path) and load_path.endswith('.yaml')
    with open(load_path, 'r') as read_stream:
        data_map = yaml.load(read_stream, Loader=Loader)

    # To symbol map
    numerical_test_cases = list()
    string_map_list = data_map['numerical_test_cases']
    for map_i in string_map_list:
        symbol_map_i = dict()
        for symbol_name_j in map_i:
            symbol_map_i[sp.Symbol(symbol_name_j)] = float(map_i[symbol_name_j])
        numerical_test_cases.append(symbol_map_i)

    # Finished
    return numerical_test_cases
