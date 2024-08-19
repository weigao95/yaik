import os.path
import solver.build_equations as build_equations
from fk.robots import RobotDescription
import fk.robot_models as robot_models
import solver.degenerate_analyse.numerical_check_data as numerical_check_data
from solver.solve_non_degenerate import solve_main_branch
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.build_variable_branch import build_solved_variable_branches
from codegen.skeleton.tree_serialize import save_ik_tree, read_skeleton_tree
from typing import Optional, Set
import sympy as sp
import numpy as np
import attr


@attr.s
class RunIKOption(object):
    use_all_intersection_pair_axis_equation: bool = False
    try_intersecting_axis_equation: bool = True
    use_polynomial_solver_in_main_branch: bool = True


def check_robot(robot: RobotDescription) -> bool:
    """
    Everything in robot.dh_params should be declared in either unknown or robot.symbolic_parameters,
    and everything in robot.symbolic_parameters must be valued and bounded.
    The Unknown should contain bound by themselves.
    """
    # Verify everything in robot.dh_params are in parameters
    unknown_set: Set[sp.Symbol] = set([elem.symbol for elem in robot.unknowns])
    uncollected_parameters: Set[sp.Symbol] = set()
    for dh_param in robot.dh_params:
        this_param_list = dh_param.flatten()
        for elem in this_param_list:
            if isinstance(elem, sp.Symbol):
                if elem in unknown_set:
                    continue
                if elem in robot.symbolic_parameters:
                    continue
                # Collect it
                uncollected_parameters.add(elem)

    # Do warning
    if len(uncollected_parameters) > 0:
        print('Not all parameters are declared in robot.symbolic_parameters, those are not ', uncollected_parameters)

    # Check all unknowns are bounded
    unbounded_unknown: Set[sp.Symbol] = set()
    for unknown_i in robot.unknowns:
        lb_i = unknown_i.lower_bound
        ub_i = unknown_i.upper_bound
        # Not none
        if lb_i is None or ub_i is None:
            unbounded_unknown.add(unknown_i.symbol)
            continue

        # Not inf
        if np.isinf(lb_i) or np.isinf(ub_i):
            print('Infinite bound for unknown {name}'.format(name=unknown_i.symbol.name))
            unbounded_unknown.add(unknown_i.symbol)
            continue

        # lb < ub
        if ub_i < lb_i:
            print('Incorrect bound for unknown {name}'.format(name=unknown_i.symbol.name))
            unbounded_unknown.add(unknown_i.symbol)
            continue

    # Do warning
    if len(unbounded_unknown) > 0:
        print('Not all unknown are associated with valid bounds, those are not ', unbounded_unknown)

    # Check the element in symbolic parameters
    unbounded_parameters: Set[sp.Symbol] = set()
    for parameter in robot.symbolic_parameters:
        in_bound = parameter in robot.parameters_bound
        in_value = parameter in robot.parameters_value
        if in_value or in_bound:
            continue
        else:
            unbounded_parameters.add(parameter)

    # Do warning
    if len(unbounded_parameters) > 0:
        print('Not all parameters are bounded, those are not ', unbounded_parameters)

    # Check the dof
    if len(robot.unknowns) != robot.n_dofs or len(robot.dh_params) != robot.n_dofs:
        print('The unknowns, DoF number or dh-parameters are not matched. It is required that each dh parameter'
              'corresponds to ONE unknown.')
        return False
    if len(robot.unknowns) >= 7 and len(robot.unknown_as_parameter_more_dof) == 0:
        print('You must specify unknown that can be treated as parameter if DoF >= 7')
        return False

    # Check pass if everything collected and bounded
    ok = len(uncollected_parameters) == 0 and len(unbounded_parameters) == 0 and len(unbounded_unknown) == 0
    return ok


def run_ik(
        robot: RobotDescription,
        numerical_test_cases_path: Optional[str] = None,
        run_ik_option: Optional[RunIKOption] = None):
    # Check the robot
    robot_ok = check_robot(robot)
    if not robot_ok:
        print('Robot input checking failed. Do nothing and return.')
        return

    # Make the option if not available
    if run_ik_option is None:
        run_ik_option = RunIKOption()

    # Build the equations
    ik_equation_input = build_equations.build_ik_equations(robot, run_ik_option.use_all_intersection_pair_axis_equation)

    # Generate numerical test cases
    if numerical_test_cases_path is None:
        numerical_test_cases = numerical_check_data.generate_test_cases(ik_equation_input)
        test_cases_save_path = robot.name + '_numerical_test.yaml'
        numerical_check_data.save_test_cases(numerical_test_cases, test_cases_save_path)
    else:
        numerical_test_cases = numerical_check_data.load_test_cases(numerical_test_cases_path)

    # Logging
    print('We have {n} numerical test cases'.format(n=len(numerical_test_cases)))

    # invoke the solver
    variable_solutions = solve_main_branch(
        robot,
        ik_equation_input,
        numerical_test_cases,
        try_intersecting_axis_equation=run_ik_option.try_intersecting_axis_equation,
        use_polynomial_solver=run_ik_option.use_polynomial_solver_in_main_branch)
    tree = SkeletonTree(variable_solutions, robot, ik_equation_input)

    # Save the non-degenerate branch
    non_degenerate_save_path = robot.name + '_main.yaml'
    save_ik_tree(tree, non_degenerate_save_path)

    # Build the variable branch, and do it again
    build_solved_variable_branches(tree, max_num_nodes=50)
    save_path = robot.name + '_ik.yaml'
    save_ik_tree(tree, save_path)


def solve_from_yaml(robot_yaml_path: str, test_case_path: Optional[str], run_ik_option: Optional[RunIKOption] = None):
    import yaml
    assert os.path.exists(robot_yaml_path)
    with open(robot_yaml_path, 'r') as read_stream:
        data_map = yaml.load(read_stream, Loader=yaml.CLoader)
    loaded_robot = RobotDescription.load_from_dict(data_map)
    run_ik(loaded_robot, test_case_path, run_ik_option)


def run_robot_from_script():
    robot_to_solve = robot_models.ur10_urdf_robot()
    test_case_path = None
    option = RunIKOption()
    option.use_all_intersection_pair_axis_equation = True
    option.try_intersecting_axis_equation = True
    # test_case_path = './gallery/test_data/franka_panda_numerical_test.yaml'
    # test_case_path = test_case_path if os.path.exists(test_case_path) else None
    run_ik(robot_to_solve, test_case_path, option)


def run_robot_ik_from_yaml():
    robot_path = './gallery/spherical_wrist_six_axis/spherical_wrist_six_axis.yaml'
    test_case_path = None
    option = RunIKOption()
    option.use_all_intersection_pair_axis_equation = True
    option.try_intersecting_axis_equation = True
    option.use_polynomial_solver_in_main_branch = False
    solve_from_yaml(robot_path, test_case_path, option)


def extend_main_branch_tree():
    main_branch_tree_path = './gallery/arm_robo/arm_robo_main.yaml'
    tree = read_skeleton_tree(main_branch_tree_path)
    rbt_name = tree.robot.name

    # Build the variable branch, and do it again
    build_solved_variable_branches(tree, max_num_nodes=50)
    save_path = rbt_name + '_ik.yaml'
    save_ik_tree(tree, save_path)


if __name__ == '__main__':
    # run_robot_from_script()
    run_robot_ik_from_yaml()
    # extend_main_branch_tree()
