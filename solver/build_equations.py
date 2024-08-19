import copy
from fk.robots import RobotDescription, robot_parameter_bounds
import fk.fk_equations as fk_equations
import fk.kinematics_dh as kinematics_dh
from fk.fk_equations import ForwardKinematicsOutput
from solver.equation_types import \
    ScalarEquation, SumOfSquareHint, TranslationalEquation, Unknown, NumericalAnalyseContext
from fk.intersecting_axis_equation import intersecting_axis_triplet_equation, intersecting_axis_pair_equation
import solver.equation_utils as equation_utils
import solver.sos_transform as sos_transform
from typing import List, Tuple, Set, Dict, Optional
import sympy as sp
import attr


@attr.s
class IkEquationCollection(object):
    # The result that can be directly collected from fk_out
    ee_pose: sp.Matrix = attr.ib()
    scalar_equations: List[ScalarEquation] = list()
    sos_hint: List[SumOfSquareHint] = list()

    # The transform for each unknown
    transform_for_unknown: List[sp.Matrix] = list()

    # The unknowns and symbols
    all_symbol_set: Set[sp.Symbol] = set()
    parameter_value_dict: Dict[sp.Symbol, float] = dict()
    parameter_bound_dict: Dict[sp.Symbol, Tuple[float, float]] = dict()

    # The intersection-axis equations
    sub_equation_dict: Dict[Tuple[str, ...], Tuple[List[TranslationalEquation], List[ScalarEquation]]] = dict()

    # The unknowns, note the different between initial and used
    # As some variable from initial unknowns will be moved to used one
    initial_unknowns: List[Unknown] = list()
    used_unknowns: List[Unknown] = list()
    unknown_as_parameter_more_dof: List[sp.Symbol] = list()

    def make_numerical_context(
            self,
            numerical_test_cases: List[Dict[sp.Symbol, float]] = list()) -> NumericalAnalyseContext:
        """
        Hand method to construct the numerical context from internal valud/bound
        """
        context = NumericalAnalyseContext()
        context.parameter_values = self.parameter_value_dict
        context.parameter_bounds = self.parameter_bound_dict
        context.numerical_test_cases = numerical_test_cases
        return context


def build_raw_fk_equations(robot: RobotDescription) \
        -> Tuple[ForwardKinematicsOutput, List[ScalarEquation], List[SumOfSquareHint]]:
    """
    Build the raw forward kinematics equations of the robot
    :param robot:
    :return:
    """
    unknowns = [elem.symbol for elem in robot.unknowns]
    fk_out = kinematics_dh.forward_kinematics_dh(robot.dh_params, unknowns)
    raw_fk_equations = fk_equations.build_fk_matrix_equations(fk_out)
    scalar_equations = equation_utils.collect_scalar_equations(raw_fk_equations)
    sos_hint = sos_transform.collect_sos_hint_translation(raw_fk_equations)
    return fk_out, scalar_equations, sos_hint


def get_all_symbols(robot: RobotDescription) -> Set[sp.Symbol]:
    """
    Collect all the parameters, which include robot parameters and ik target parameters
    :param robot:
    :return:
    """
    all_symbol_set = fk_equations.ik_target_symbols()
    all_symbol_set.update(robot.symbolic_parameters)
    for unknown in robot.unknowns:
        all_symbol_set.add(unknown.symbol)
    return all_symbol_set


def get_symbol_bounds(robot: RobotDescription) -> Dict[sp.Symbol, Tuple[float, float]]:
    """
    Collect the bounds of some parameters, which include robot and ik target parameters
    :param robot:
    :return:
    """
    parameter_bounds = robot_parameter_bounds(robot)
    parameter_bounds.update(fk_equations.ik_target_parameter_bounds())
    return parameter_bounds


def build_ik_equations(
        robot: RobotDescription,
        use_all_intersection_pair_axis_equation: bool = False) -> IkEquationCollection:
    """
    Build the equations used in ik using robot description.
    Here we assume one-to-one correspondence between robot unknowns and robot dhs
    :param robot
    :param use_all_intersection_pair_axis_equation should we include equations for pairwise intersection axis?
                                                   there can be a lot (thus much longer running time)
    """
    # The raw fk equations
    fk_out, scalar_equations, sos_hint = build_raw_fk_equations(robot)
    ee_pose = fk_out.T_ee()
    ik_equation = IkEquationCollection(ee_pose)
    ik_equation.scalar_equations = scalar_equations
    ik_equation.sos_hint = sos_hint

    # The transform for each unknown
    ik_equation.transform_for_unknown = list()
    for i in range(robot.n_dofs):
        ik_equation.transform_for_unknown.append(fk_out.Ts(i))

    # The value and bound_dict
    ik_equation.all_symbol_set = get_all_symbols(robot)
    ik_equation.parameter_bound_dict = get_symbol_bounds(robot)
    ik_equation.parameter_value_dict = copy.deepcopy(robot.parameters_value)

    # The triplet equation dict
    initial_unknowns = [elem.symbol for elem in robot.unknowns]
    equation_dict_triplet = intersecting_axis_triplet_equation(
        fk_out,
        robot.dh_params, initial_unknowns,
        use_inverse_equations=True
    )

    # The pair-wise equation dict
    equation_dict_pair = intersecting_axis_pair_equation(
        fk_out, robot.dh_params, initial_unknowns, use_inverse_equations=True)

    # Add to ik_equation
    has_three_axis: bool = False
    subset_translation_equations: List[TranslationalEquation] = list()
    for key in equation_dict_triplet:
        has_three_axis = True
        translation_equations: List[TranslationalEquation] = equation_dict_triplet[key]
        ik_equation.sub_equation_dict[key] = (translation_equations, list())
        for equation in translation_equations:
            subset_translation_equations.append(equation)

    # For pair-wise intersection.
    # If we have (better) three-axis equations, then do not add them into main equations
    if not has_three_axis:
        for key in equation_dict_pair:
            ik_equation.sub_equation_dict[key] = equation_dict_pair[key]
            # Should we add this? There can be a lot
            if use_all_intersection_pair_axis_equation:
                translation_equations, _ = equation_dict_pair[key]
                for equation in translation_equations:
                    subset_translation_equations.append(equation)

    # Add translation equations into scalar equations and sos hints
    for translation_equation in subset_translation_equations:
        ik_equation.scalar_equations.append(translation_equation.x)
        ik_equation.scalar_equations.append(translation_equation.y)
        if translation_equation.z is not None:
            ik_equation.scalar_equations.append(translation_equation.z)
            this_sos_hint = SumOfSquareHint(translation_equation.x, translation_equation.y, translation_equation.z)
            ik_equation.sos_hint.append(this_sos_hint)
        else:
            this_sos_hint = SumOfSquareHint(translation_equation.x, translation_equation.y, None)
            ik_equation.sos_hint.append(this_sos_hint)

    # The unknowns
    for unknown in robot.unknowns:
        ik_equation.initial_unknowns.append(copy.deepcopy(unknown))
    ik_equation.unknown_as_parameter_more_dof = copy.deepcopy(robot.unknown_as_parameter_more_dof)

    # Minus the variable in unknown_as_parameter_more_dof
    ik_equation.used_unknowns = list()
    for unknown in ik_equation.initial_unknowns:
        unknown_symbol = unknown.symbol
        if unknown_symbol in ik_equation.unknown_as_parameter_more_dof:
            pass
        else:
            ik_equation.used_unknowns.append(unknown)

    # Update of the equation dict
    removed_unknown_names = [elem.name for elem in ik_equation.unknown_as_parameter_more_dof]
    need_remove_keys = set()
    for key in ik_equation.sub_equation_dict:
        need_remove = False
        for removed_var in removed_unknown_names:
            if removed_var not in key:
                need_remove = True
                break
        if need_remove:
            need_remove_keys.add(key)

    # Do remove
    for key in need_remove_keys:
        print('Removing a sub-equations with variables ', key)
        del ik_equation.sub_equation_dict[key]

    # Ok
    return ik_equation
