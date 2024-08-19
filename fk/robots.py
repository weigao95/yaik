import copy
from typing import List, Dict, Set, Tuple, Optional
from fk.fk_equations import DHEntry
from solver.equation_types import Unknown
from solver.equation_utils import sp_matrix_to_dict_representation, parse_sp_matrix_from_dict
import sympy as sp
import attr


@attr.s
class RobotAuxiliaryData(object):
    """
    unknown = raw_unknown + unknown_offset
    ee_pose = pre_transform * FK(dh_params, unknown) * post_transform
    unknown will be solved in the solver, while in generated code from codegen raw_unknown will be returned
    These data will not be involved in solver, only codegen will use them
    """
    # The pre- & post-transform, None implies identity
    pre_transform_sp: Optional[sp.Matrix] = None
    post_transform_sp: Optional[sp.Matrix] = None

    # The offset of unknown, None implies no offset
    # The solver does NOT consider the offset. However, the offset should be handled in codegen
    # Must be in the same size as robot.unknowns
    unknown_offset: Optional[List[float]] = list()

    # Flatten
    def to_dict(self):
        data_map = dict()
        if self.pre_transform_sp is not None:
            data_map['pre_transform_sp'] = sp_matrix_to_dict_representation(self.pre_transform_sp)
        if self.post_transform_sp is not None:
            data_map['post_transform_sp'] = sp_matrix_to_dict_representation(self.post_transform_sp)
        if self.unknown_offset is not None:
            data_map['unknown_offset'] = self.unknown_offset
        return data_map

    # De-flatten
    def from_dict(self, data_map):
        if 'unknown_offset' in data_map:
            self.unknown_offset = data_map['unknown_offset']
        else:
            self.unknown_offset = None

        if 'pre_transform_sp' in data_map:
            self.pre_transform_sp = parse_sp_matrix_from_dict(data_map['pre_transform_sp'])
        else:
            self.pre_transform_sp = None

        if 'post_transform_sp' in data_map:
            self.post_transform_sp = parse_sp_matrix_from_dict(data_map['post_transform_sp'])
        else:
            self.post_transform_sp = None


@attr.s
class RobotDescription(object):
    name: str = attr.ib()

    # One unknown corresponds to one DH entry
    # Thus, these two lists have the same length
    dh_params: List[DHEntry] = list()
    unknowns: List[Unknown] = list()

    # These two may NOT be in equal size
    # For example, in a 7-DOF robot we may select one joint as a parameter,
    # but is value is given at runtime.
    symbolic_parameters: Set[sp.Symbol] = set()
    parameters_value: Dict[sp.Symbol, float] = dict()

    # However, each parameter must be either valued or bounded
    parameters_bound: Dict[sp.Symbol, Tuple[float, float]] = dict()

    # If the robot has more than 6 DoF, then one or more of its unknown
    # should be treated as parameter (else the problem is undefined).
    unknown_as_parameter_more_dof: List[sp.Symbol] = list()

    # The auxiliary data that won't go into the solver
    auxiliary_data: Optional[RobotAuxiliaryData] = None

    # Getter
    def get_pre_transform_sp(self) -> Optional[sp.Matrix]:
        if self.auxiliary_data is not None:
            return self.auxiliary_data.pre_transform_sp
        return None

    def get_post_transform_sp(self) -> Optional[sp.Matrix]:
        if self.auxiliary_data is not None:
            return self.auxiliary_data.post_transform_sp
        return None

    def get_unknown_offset(self) -> Optional[List[float]]:
        if self.auxiliary_data is not None:
            return self.auxiliary_data.unknown_offset
        return None

    @property
    def n_dofs(self):
        return len(self.unknowns)

    # Flatten/deflatten
    def to_dict(self):
        data_map = dict()
        data_map['name'] = self.name

        # For dh parameters
        dh_params_dicts = list()
        for i in range(len(self.dh_params)):
            dh_i = self.dh_params[i]
            dh_map_i = dh_i.to_dict()
            dh_params_dicts.append(dh_map_i)
        data_map['dh_params'] = dh_params_dicts

        # For unknowns
        unknown_dict_list = list()
        for i in range(len(self.unknowns)):
            unknown_dict_i = self.unknowns[i].to_dict()
            unknown_dict_list.append(unknown_dict_i)
        data_map['unknowns'] = unknown_dict_list

        # For symbolic_parameters
        parameters = [elem.name for elem in self.symbolic_parameters]
        data_map['symbolic_parameters'] = parameters

        # For parameter value
        parameter_value_map = dict()
        for key in self.parameters_value:
            parameter_value_map[key.name] = float(self.parameters_value[key])
        data_map['parameters_value'] = parameter_value_map

        # For parameter bound
        parameter_bound_map = dict()
        for key in self.parameters_bound:
            lb_key, ub_key = self.parameters_bound[key]
            parameter_bound_map[key.name] = (lb_key, ub_key)
        data_map['parameters_bound'] = parameter_bound_map

        # All unknowns that can be treated as parameter if more DoF
        unknown_as_parameter_more_dof: List[str] = list()
        for unknown_symbol in self.unknown_as_parameter_more_dof:
            unknown_as_parameter_more_dof.append(unknown_symbol.name)
        data_map['unknown_as_parameter_more_dof'] = unknown_as_parameter_more_dof

        # The auxiliary data that will not get into the solver
        if self.auxiliary_data is not None:
            data_map['auxiliary_data'] = self.auxiliary_data.to_dict()

        # Finished
        return data_map

    def from_dict(self, data_map: Dict):
        self.name = data_map['name']

        # For dh params
        dh_params_dicts = data_map['dh_params']
        self.dh_params.clear()
        for i in range(len(dh_params_dicts)):
            dh_map_i = dh_params_dicts[i]
            entry_i = DHEntry.load_from_dict(dh_map_i)
            self.dh_params.append(entry_i)

        # For unknowns
        unknown_dict_list = data_map['unknowns']
        self.unknowns.clear()
        for i in range(len(unknown_dict_list)):
            unknown_dict_i = unknown_dict_list[i]
            unknown_i = Unknown.load_unknown_from_dict(unknown_dict_i)
            self.unknowns.append(unknown_i)
        assert len(self.unknowns) == len(self.dh_params)

        # For symbolic_parameters
        symbolic_parameter_names = data_map['symbolic_parameters']
        self.symbolic_parameters.clear()
        for parameter_name in symbolic_parameter_names:
            self.symbolic_parameters.add(sp.Symbol(parameter_name))

        # For parameter value
        self.parameters_value.clear()
        parameter_value_map = data_map['parameters_value']
        for key in parameter_value_map:
            variable = sp.Symbol(key)
            variable_value = parameter_value_map[key]
            self.parameters_value[variable] = variable_value

        # For parameter bounds
        self.parameters_bound.clear()
        if 'parameters_bound' in data_map:
            parameter_bound_map = data_map['parameters_bound']
            for key in parameter_bound_map:
                variable = sp.Symbol(key)
                lb, ub = parameter_bound_map[variable]
                self.parameters_bound[variable] = (lb, ub)

        # For unknown_as_parameter_more_dof
        self.unknown_as_parameter_more_dof = list()
        if 'unknown_as_parameter_more_dof' in data_map:
            for unknown_name in data_map['unknown_as_parameter_more_dof']:
                unknown_symbol = sp.Symbol(unknown_name)
                self.unknown_as_parameter_more_dof.append(unknown_symbol)
        else:
            pass  # leave it as empty

        # The auxiliary_data
        if 'auxiliary_data' in data_map:
            self.auxiliary_data = RobotAuxiliaryData()
            self.auxiliary_data.from_dict(data_map['auxiliary_data'])
        else:
            self.auxiliary_data = None

    @staticmethod
    def load_from_dict(data_map: Dict):
        robot = RobotDescription('')
        robot.from_dict(data_map)
        return robot


def robot_parameter_bounds(
        robot: RobotDescription,
        include_parameter_with_value: bool = False) -> Dict[sp.Symbol, Tuple[float, float]]:
    """
    Get the map from variable symbol to their bounds
    :param robot:
    :param include_parameter_with_value:
    :return:
    """
    bound_dict: Dict[sp.Symbol, Tuple[float, float]] = copy.copy(robot.parameters_bound)
    for unknown in robot.unknowns:
        unknown_symbol = unknown.symbol
        lb = unknown.lower_bound
        ub = unknown.upper_bound
        if (lb is not None) and (ub is not None):
            bound_dict[unknown_symbol] = (lb, ub)

    # If a parameter has value, then its lb/ub are both the designated value
    if include_parameter_with_value:
        for key in robot.parameters_value:
            value = robot.parameters_value[key]
            bound_dict[key] = (value, value)
    return bound_dict
