import copy
from fk.robots import RobotDescription
from solver.equation_utils import cast_expr_to_float
from solver.general_6dof import dh_utils
from solver.general_6dof.reduce_input import build_reduce_inputs, is_solvable, ik_target_4x4_internal
from solver.solved_variable_impl import General6DoFNumericalReduceSolutionImpl as General6DoFNumericalReduceSolution
import solver.general_6dof.numerical_reduce_closure_equation as numerical_reduce
from solver.general_6dof.numerical_reduce_closure_equation import verify_solve
from solver.general_6dof.select_lines_to_reduce_numerical import select_lines_to_reduce
from solver.general_6dof.semi_symbolic_reduce import convert_to_semi_symbolic_reduce
import multiprocessing
import numpy as np
import sympy as sp
from typing import Tuple, Optional, List, Dict


def build_test_cases_with_rhs_target_subst(original_test_case_list: List[Dict[sp.Symbol, float]],
                                           rhs_matrix_subst: sp.Matrix) -> List[Dict[sp.Symbol, float]]:
    """
    In the current implementation, the rhs target of the FK equation will be used as ik_target_4x4_internal,
    instead of the usual ik_target_4x4. ik_target_internal can be computed using ik_target_4x4 with zero or more
    unknown treated as parameters, and the expression is in rhs_matrix_subst.
    """
    subst_test_cases = list()
    for i in range(len(original_test_case_list)):
        test_case_i = original_test_case_list[i]
        rhs_matrix_subst_i = rhs_matrix_subst.subs(test_case_i)
        case_i_map = copy.deepcopy(test_case_i)
        subst_rhs_matrix_np = np.zeros(shape=(4, 4))
        for r in range(3):
            for c in range(4):
                rc_value = cast_expr_to_float(rhs_matrix_subst_i[r, c])
                subst_rhs_matrix_np[r, c] = rc_value

        # Assign ik target
        ik_target_tmp = ik_target_4x4_internal()
        for r in range(3):
            for c in range(4):
                rc_symbol = ik_target_tmp[r, c]
                case_i_map[rc_symbol] = float(subst_rhs_matrix_np[r, c])
        subst_test_cases.append(case_i_map)
    return subst_test_cases


def parallel_solve_processor(input_tuple):
    numerical_reduce_input, subst_test_cases, order_idx, rhs_matrix_subst, original_test_cases, logging = input_tuple
    selected_lines = select_lines_to_reduce(
        numerical_reduce_input, subst_test_cases, logging=logging)
    if selected_lines is None:
        return None

    # Compute the semi-symbolic reduction
    semi_symbolic = convert_to_semi_symbolic_reduce(numerical_reduce_input, selected_lines)

    # Make the subst map
    subst_map = dict()
    ik_target_tmp = ik_target_4x4_internal()
    for r in range(3):
        for c in range(4):
            rc_symbol = ik_target_tmp[r, c]
            subst_map[rc_symbol] = rhs_matrix_subst[r, c]
    numerical_reduce_input_subst = numerical_reduce_input.subs(subst_map)
    semi_symbolic_subst = semi_symbolic.subs(subst_map)

    # Do a verify
    solved_count = 0
    for i in range(len(original_test_cases)):
        test_case_i = original_test_cases[i]
        i_solved = verify_solve(
            numerical_reduce_input_subst.lhs_A_sin,
            numerical_reduce_input_subst.lhs_A_cos,
            numerical_reduce_input_subst.lhs_C_const,
            numerical_reduce_input_subst.matrix_equation,
            test_case_i, selected_lines)
        if i_solved:
            solved_count += 1
    print(f'In verification, the solved ratio is {solved_count}/{len(original_test_cases)} for '
          f'a solution in input order {order_idx}')
    solution = General6DoFNumericalReduceSolution(numerical_reduce_input_subst, selected_lines, semi_symbolic_subst)
    print('Find solution of {var} in input order {i}'.
          format(i=order_idx, var=solution.solved_variable.name))
    return solution


def parallel_solve_processor_v2(input_tuple, lock: multiprocessing.Lock, result_list):
    out = parallel_solve_processor(input_tuple)
    if out is None:
        return
    # Not none, append to result
    if lock is not None:
        lock.acquire()
    result_list.append(out)
    if lock is not None:
        lock.release()


def build_solver_input(robot: RobotDescription, only_reduce_first_on_lhs: bool = True, logging: bool = False):
    # Convert to classic dh
    dh_params = dh_utils.modified_dh_to_classic(robot.dh_params)
    if dh_params is None:
        print('Cannot convert from modified dh to classic dh, maybe because dh_0.alpha/a is not zero. '
              'In that case, please set them as zero and move the term to pre-transform.')
        return None
    revolute_vars = dh_utils.RevoluteVariable.convert_from_robot_unknowns(robot)

    # This solver is very slow, thus do NOT use a large test set
    test_cases = dh_utils.generate_classic_dh_numerical_test(
        dh_params, revolute_vars, robot.parameters_value, n_test_cases=50)

    # Build the reduction inputs
    reduce_input_tuple = build_reduce_inputs(robot)
    if reduce_input_tuple is None:
        return None

    # Make the input and update the test_cases
    reduce_input_list, revolute_vars, rhs_matrix_subst = reduce_input_tuple
    subst_test_cases = build_test_cases_with_rhs_target_subst(test_cases, rhs_matrix_subst)

    # Start making the input tuples
    input_tuple_list = list()
    counter = 0
    for i in range(len(reduce_input_list)):
        if logging:
            print('Building input ordering {i}'.format(i=i))
        reduce_input_i = reduce_input_list[i]
        numerical_reduce_input_list = numerical_reduce.generate_numerical_reduce_input(reduce_input_i, revolute_vars)
        if only_reduce_first_on_lhs:
            input_tuple_j = (
                numerical_reduce_input_list[0],
                subst_test_cases,
                counter,
                rhs_matrix_subst,
                test_cases,
                logging
            )
            input_tuple_list.append(input_tuple_j)
            counter += 1
        else:
            for j in range(len(numerical_reduce_input_list)):
                input_tuple_j = (
                    numerical_reduce_input_list[j],
                    subst_test_cases,
                    counter,
                    rhs_matrix_subst,
                    test_cases,
                    logging
                )
                input_tuple_list.append(input_tuple_j)
                counter += 1
    return input_tuple_list


def try_solve_serial(robot: RobotDescription, logging: bool = True) -> List[General6DoFNumericalReduceSolution]:
    # run mapping
    input_tuple_list = build_solver_input(robot, only_reduce_first_on_lhs=True, logging=logging)

    # Iterate over all order
    for i in range(len(input_tuple_list)):
        input_i = input_tuple_list[i]
        output_i = parallel_solve_processor(input_i)
        if output_i is not None:
            return [output_i]

    # None of them can solve
    return list()


def try_solve_parallel(
        robot: RobotDescription,
        only_reduce_first_on_lhs: bool = False) -> List[General6DoFNumericalReduceSolution]:
    # run mapping
    input_tuple_list = build_solver_input(robot, only_reduce_first_on_lhs=only_reduce_first_on_lhs, logging=False)
    n_process = len(input_tuple_list)
    print(f'Trying solve ik with general solver using {n_process} processors in parallel')
    output: List[Optional[General6DoFNumericalReduceSolution]] = list()
    with multiprocessing.Pool(n_process) as pool:
        output = pool.map(parallel_solve_processor, input_tuple_list)

    # Collect the result
    result_list = list()
    for i in range(len(output)):
        if output[i] is not None:
            result_list.append(output[i])
    return result_list


def try_solve_parallel_v2(
        robot: RobotDescription,
        only_reduce_first_on_lhs: bool = False) -> List[General6DoFNumericalReduceSolution]:
    # run mapping
    input_tuple_list = build_solver_input(robot, only_reduce_first_on_lhs=only_reduce_first_on_lhs, logging=False)
    n_process = len(input_tuple_list)
    print(f'Trying solve ik with general solver using {n_process} processors in parallel')

    # Make the lock and result
    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    result_list = manager.list()
    processor_list = list()
    for i in range(n_process):
        processor = multiprocessing.Process(target=parallel_solve_processor_v2,
                                            args=(input_tuple_list[i], lock, result_list))
        processor_list.append(processor)

    # Start with timeout
    timeout_seconds: int = 4 * 3600  # 4 hours
    for p in processor_list:
        p.start()
    for p in processor_list:
        p.join(timeout=timeout_seconds)
        p.terminate()  # Explicit kill a process if the reduction takes too long

    # Collect the result
    output: List[General6DoFNumericalReduceSolution] = list()
    for elem in result_list:
        if elem is not None:
            output.append(copy.deepcopy(elem))
    return output


# Debug code
def test_general_solve():
    import fk.robot_models as robot_models
    robot = robot_models.yaskawa_HC10_robot()
    print(is_solvable(robot))
    # solver_input = build_solver_input(robot, logging=True)
    # print(solver_input)
    solution_list = try_solve_parallel_v2(robot)
    for sol in solution_list:
        print(sol.solved_variable)

    from solver.solved_variable import VariableSolution
    solution = VariableSolution(solution_impl=solution_list[0])
    solution_dict = solution.to_dict()
    print(solution_dict)
    print('Solution dict')
    for k in solution_dict:
        print(k, solution_dict[k])
    loaded_solution = VariableSolution()
    loaded_solution.from_dict(solution_dict)
    print(loaded_solution.solved_variable)


if __name__ == '__main__':
    test_general_solve()
