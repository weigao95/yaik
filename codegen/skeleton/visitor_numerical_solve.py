from codegen.skeleton.tree_node import SkeletonTreeNode, SolutionNodeBase
from codegen.skeleton.tree_node import SkeletonNodeVisitor, NoBranchDispatcherNode, EquationAllZeroDispatcherNode
from codegen.skeleton.tree_node import SolvedVariableDispatcherNode, ExplicitSolutionNode, PolynomialSolutionNode
from codegen.skeleton.skeleton_tree import SkeletonTree
from fk.fk_equations import zero_tolerance
from solver.solved_variable import VariableSolution, SolutionMethod, VariableSolutionClassKey
from solver.solved_variable_impl import General6DoFNumericalReduceSolutionImpl, LinearSinCosType_2_SolutionImpl
from solver.general_6dof.numerical_reduce_closure_equation import symbolic_solve
from fk.fk_equations import ik_target_subst_map
from solver.equation_utils import find_unknown_idx, cast_expr_to_float
from typing import List, Tuple, Dict, Optional, Set
import sympy as sp
import numpy as np
import multiprocessing
import copy
import attr

# A magic number as hte invalid solution
INVALID_SOLUTION_VALUE = 1e6


def is_valid_solution_number(solution_number: float) -> bool:
    """
    A better solution is to construct another binary list to determine whether
    a number is valid or not, initialized to all false.
    But this somehow work as a hacky solution
    """
    abs_diff = abs(solution_number - INVALID_SOLUTION_VALUE)
    return abs_diff > 1e-10


@attr.s
class SkeletonNumericalSolution(object):
    # The map from symbol to its value
    variable_values: List[float] = attr.ib()
    variable_value_in_solution_idx: List[int] = attr.ib()

    def initialize(self, value_size):
        self.variable_values.clear()
        self.variable_value_in_solution_idx.clear()
        for i in range(value_size):
            self.variable_values.append(INVALID_SOLUTION_VALUE)
            self.variable_value_in_solution_idx.append(-1)


def step_from_explicit_solution(
        current_value: SkeletonNumericalSolution,
        value_subst_map: Dict[sp.Symbol, float],
        next_solution: VariableSolution,
        next_solution_index_in_value: int,
        parameter_values: Dict[sp.Symbol, float]) -> List[SkeletonNumericalSolution]:
    """
    Given solutions of variable 0, 1, ..., i-1, instantiated the values of solution i (there might
    be multiple solutions, thus multiple values) and return them.
    This method assumes explicit solution, and it does NOT check the degenerate condition.
    :param current_value: the value of variables 0, 1, ..., i - 1
    :param value_subst_map: visitor.value_to_substitute_map(current_value)
    :param next_solution: the symbolic solution of variable i
    :param next_solution_index_in_value: save to this index
    :param parameter_values: the numerical solution with variable i
    :return:
    """
    assert next_solution.is_explicit_solution
    solutions = next_solution.explicit_solutions
    argument_checkers = next_solution.argument_validity_checkers
    assert len(solutions) == len(argument_checkers)
    variable_subst_map = value_subst_map
    next_variable_values: List[SkeletonNumericalSolution] = list()
    for i in range(len(solutions)):
        # First perform checking
        checker_i = argument_checkers[i]
        checker_i = checker_i.subs(parameter_values)
        checker_i = checker_i.subs(variable_subst_map)
        if checker_i == sp.S.BooleanFalse:
            continue

        solution_i = solutions[i]
        solution_i = solution_i.subs(parameter_values)
        solution_i = solution_i.subs(variable_subst_map)
        solution_i = cast_expr_to_float(solution_i)
        assert solution_i is not None
        next_value_i = copy.deepcopy(current_value)
        next_value_i.variable_values[next_solution_index_in_value] = solution_i
        next_value_i.variable_value_in_solution_idx[next_solution_index_in_value] = i
        next_variable_values.append(next_value_i)
    return next_variable_values


def step_from_linear_type2_solution(
        current_value: SkeletonNumericalSolution,
        value_subst_map: Dict[sp.Symbol, float],
        next_solution: VariableSolution,
        next_solution_index_in_value: int,
        parameter_values: Dict[sp.Symbol, float]) -> List[SkeletonNumericalSolution]:
    # The internal solution
    from solver.linear_solver.linear_sin_cos_type2_numerical_solve import try_solve_linear_type2_combined
    solution_impl = next_solution.internal_solution
    assert solution_impl.class_key() == VariableSolutionClassKey.LinearSinCosType_2.name

    # Extract the value
    linear_solution: LinearSinCosType_2_SolutionImpl = solution_impl
    A_symbolic = linear_solution.A_matrix
    rows_to_try = linear_solution.rows_to_try
    solved_var = solution_impl.solved_variable

    # To numerical
    A_subst = A_symbolic.subs(value_subst_map)
    A_subst = A_subst.subs(parameter_values)
    A_np = np.array(A_subst).astype(np.float64)
    solutions = try_solve_linear_type2_combined(A_np, rows_to_try)
    if solutions is None:
        return list()

    # Assign the next value
    next_variable_values: List[SkeletonNumericalSolution] = list()
    for i in range(len(solutions)):
        solution_i = solutions[i]
        next_value_i = copy.deepcopy(current_value)
        next_value_i.variable_values[next_solution_index_in_value] = solution_i
        next_value_i.variable_value_in_solution_idx[next_solution_index_in_value] = i
        next_variable_values.append(next_value_i)
    return next_variable_values


def step_from_polynomial_solution(
        current_value: SkeletonNumericalSolution,
        value_subst_map: Dict[sp.Symbol, float],
        next_solution: VariableSolution,
        next_solution_index_in_value: int,
        parameter_values: Dict[sp.Symbol, float]) -> List[SkeletonNumericalSolution]:
    """
    Given solutions of variable 0, 1, ..., i-1, instantiated the values of solution i (there might
    be multiple solutions, thus multiple values) and return them.
    This method assumes polynomial solution, and it does NOT check the degenerate condition.
    :param current_value: the value of variables 0, 1, ..., i - 1
    :param value_subst_map: visitor.value_to_substitute_map(current_value)
    :param next_solution: the symbolic solution of variable i
    :param next_solution_index_in_value: save to this index
    :param parameter_values: the numerical solution with variable i
    :return:
    """
    assert next_solution.is_polynomial
    variable_subst_map = value_subst_map
    poly_dict_input = next_solution.polynomial_to_solve
    poly_dict: Dict[int, float] = dict()
    poly_order = -1
    for order in poly_dict_input:
        # Get the coefficient
        numerator, denominator = poly_dict_input[order]
        numerator = numerator.subs(parameter_values)
        numerator = numerator.subs(variable_subst_map)
        denominator = denominator.subs(parameter_values)
        denominator = denominator.subs(variable_subst_map)
        denominator = cast_expr_to_float(denominator)
        assert denominator is not None
        if abs(denominator) < 1e-6:
            return list()
        poly_dict[order] = float(numerator / denominator)

        # Update the order
        if order >= poly_order:
            poly_order = order

    # Solve the equations
    assert poly_order >= 1
    p_coefficients = np.zeros(shape=(poly_order + 1,))
    for order in poly_dict:
        p_coefficients[poly_order - order] = poly_dict[order]
    poly_roots = np.roots(p_coefficients)

    # Collect the results
    solved_symbol = next_solution.solved_variable
    solution_method = next_solution.solution_method
    next_variable_values: List[SkeletonNumericalSolution] = list()
    for root_idx in range(len(poly_roots)):
        this_root = poly_roots[root_idx]
        if not np.isreal(this_root):
            continue
        if solution_method == SolutionMethod.PolynomialDirect.name:
            next_value = copy.deepcopy(current_value)
            next_value.variable_values[next_solution_index_in_value] = this_root
            next_value.variable_value_in_solution_idx[next_solution_index_in_value] = root_idx
            next_variable_values.append(next_value)
            continue

        # Now we do with sin/cos intermediate
        first_angle: Optional[float] = None
        second_angle: Optional[float] = None
        if solution_method == SolutionMethod.PolynomialSin.name and abs(this_root) < 1:
            first_angle = np.arcsin(this_root)
            second_angle = np.pi - first_angle
        elif solution_method == SolutionMethod.PolynomialCos.name and abs(this_root) < 1:
            first_angle = np.arccos(this_root)
            second_angle = - first_angle

        # Append to result
        if first_angle is not None:
            assert second_angle is not None
            next_value_0 = copy.deepcopy(current_value)
            next_value_1 = copy.deepcopy(current_value)
            next_value_0.variable_values[next_solution_index_in_value] = first_angle
            next_value_1.variable_values[next_solution_index_in_value] = second_angle
            next_value_0.variable_value_in_solution_idx[next_solution_index_in_value] = 2 * root_idx
            next_value_1.variable_value_in_solution_idx[next_solution_index_in_value] = 2 * root_idx + 1
            next_variable_values.append(next_value_0)
            next_variable_values.append(next_value_1)

    # OK
    return next_variable_values


def step_from_general_6dof_numerical_reduce_solution(
        current_value: SkeletonNumericalSolution,
        value_subst_map: Dict[sp.Symbol, float],
        next_solution: VariableSolution,
        next_solution_index_in_value: int,
        parameter_values: Dict[sp.Symbol, float]) -> List[SkeletonNumericalSolution]:
    """
    Given solutions of variable 0, 1, ..., i-1, instantiated the values of solution i (there might
    be multiple solutions, thus multiple values) and return them.
    This method assumes explicit solution, and it does NOT check the degenerate condition.
    :param current_value: the value of variables 0, 1, ..., i - 1
    :param value_subst_map: visitor.value_to_substitute_map(current_value)
    :param next_solution: the symbolic solution of variable i
    :param next_solution_index_in_value: save to this index
    :param parameter_values: the numerical solution with variable i
    :return:
    """
    assert next_solution.is_general_6dof_solution
    solution_impl = next_solution.internal_solution
    assert solution_impl.class_key() == VariableSolutionClassKey.General6DoFNumericalReduce.name
    general_6dof_solution: General6DoFNumericalReduceSolutionImpl = solution_impl
    A_sin, A_cos, C_const = general_6dof_solution.lhs_matrices()
    matrix_equation = general_6dof_solution.matrix_equation
    lines2reduce = general_6dof_solution.select_lines
    var_solutions = symbolic_solve(A_sin, A_cos, C_const, matrix_equation, parameter_values, lines2reduce)

    # Gather the solution
    next_variable_values: List[SkeletonNumericalSolution] = list()
    for root_idx in range(len(var_solutions)):
        this_root = var_solutions[root_idx]
        if not np.isreal(this_root):
            continue

        next_value_i = copy.deepcopy(current_value)
        next_value_i.variable_values[next_solution_index_in_value] = this_root
        next_value_i.variable_value_in_solution_idx[next_solution_index_in_value] = root_idx
        next_variable_values.append(next_value_i)
    return next_variable_values


class VisitorNumericalSolution(SkeletonNodeVisitor):

    def __init__(self, n_nodes: int):
        super(VisitorNumericalSolution, self).__init__()
        self._node_inputs: List[List[SkeletonNumericalSolution]] = list()
        self._input_validity: List[bool] = list()
        for i in range(n_nodes):
            self._node_inputs.append(list())
            self._input_validity.append(False)

        # The parameter value, init to be empty
        self._parameter_values: Dict[sp.Symbol, float] = dict()
        self._ik_output: List[SkeletonNumericalSolution] = list()

        # From symbol to variable index
        self._symbol_to_index: Dict[sp.Symbol, int] = dict()
        self._index_to_symbol: Dict[int, sp.Symbol] = dict()

        # One possible gt
        self._gt_theta_instance: Optional[List[Tuple[sp.Symbol, float]]] = None

    def initialize(
            self, parameter_value: Dict[sp.Symbol, float],
            all_variable_in_tree: Set[sp.Symbol],
            one_gt_theta: Optional[List[Tuple[sp.Symbol, float]]] = None):
        """
        Initialize the visitor with the parameter value and every symbols in the solution tree
        """
        self._parameter_values = parameter_value
        self._parameter_values[zero_tolerance] = 1e-6

        self._ik_output.clear()
        for i in range(len(self._node_inputs)):
            self._node_inputs[i].clear()
            self._input_validity[i] = False

        # The root
        self._input_validity[0] = True

        # Build the index map
        index = 0
        all_variable_in_tree_names = [elem.name for elem in all_variable_in_tree]
        all_variable_in_tree_names = sorted(all_variable_in_tree_names)
        for symbol_i_name in all_variable_in_tree_names:
            symbol_i = sp.Symbol(symbol_i_name)
            self._symbol_to_index[symbol_i] = index
            self._index_to_symbol[index] = symbol_i
            index += 1

        # One possible solution
        self._gt_theta_instance = one_gt_theta

    @property
    def ik_output(self) -> List[SkeletonNumericalSolution]:
        return self._ik_output

    def value_to_substitute_map(self, value: SkeletonNumericalSolution):
        """
        Build the substitute map from this solution.
        :return:
        """
        subst_dict = dict()
        for index_i in self._index_to_symbol:
            symbol_i = self._index_to_symbol[index_i]
            value_i = value.variable_values[index_i]
            if is_valid_solution_number(value_i):
                subst_dict[symbol_i] = value_i
        return subst_dict

    def assign_invalid_to_children(self, node_to_assign: SkeletonTreeNode):
        """
        For a given node, assign all input valid of its child as invalid.
        Essentially stop at this node.
        """
        for i in range(len(node_to_assign.children)):
            child_i = node_to_assign.children[i]
            if child_i is not None:
                child_i_idx = child_i.flatten_idx
                self._input_validity[child_i_idx] = False

    def visit_no_branch_dispatcher(self, dispatcher: NoBranchDispatcherNode):
        """
        For this one, there is no difference between root/non-root, just copy
        parent to child.
        """
        input_idx = dispatcher.flatten_idx
        in_list = self._node_inputs[input_idx]
        assert len(dispatcher.children) == 1
        output_idx = dispatcher.children[0].flatten_idx
        for elem in in_list:
            self._node_inputs[output_idx].append(elem)
        self._input_validity[output_idx] = self._input_validity[input_idx]

    def visit_equation_all_zero_dispatcher_root(self, dispatcher: EquationAllZeroDispatcherNode):
        """
        In root, invalid is different from input empty. This dispatcher can only have one validity
        """
        assert dispatcher.is_root
        flatten_dix = dispatcher.flatten_idx
        assert flatten_dix == 0
        assert len(self._node_inputs[flatten_dix]) == 0
        assert self._input_validity[flatten_dix]
        checker = dispatcher.degenerate_checker
        checker_with_parameter = checker.subs(self._parameter_values)
        if checker_with_parameter == sp.S.BooleanTrue:
            if dispatcher.degenerate_child is not None:
                degenerate_child_idx = dispatcher.degenerate_child.flatten_idx
                self._input_validity[degenerate_child_idx] = True
                self._node_inputs[degenerate_child_idx].clear()
        else:
            assert dispatcher.non_degenerate_child is not None
            non_degenerate_child_idx = dispatcher.non_degenerate_child.flatten_idx
            self._input_validity[non_degenerate_child_idx] = True
            self._node_inputs[non_degenerate_child_idx].clear()

    def visit_equation_all_zero_dispatcher_non_root(self, dispatcher: EquationAllZeroDispatcherNode):
        """
        Non-root dispatcher, thus its parent must be a solution node.
        In this case, dispatcher_input == empty means invalid
        """
        # Fetch the input
        input_idx = dispatcher.flatten_idx_in_tree
        input_validity = self._input_validity[input_idx]
        dispatcher_input = self._node_inputs[input_idx]
        if (not input_validity) or (len(dispatcher_input) == 0):
            self.assign_invalid_to_children(dispatcher)
            return

        # Has at least one input
        checker = dispatcher.degenerate_checker
        checker_with_parameter = checker.subs(self._parameter_values)

        # Do dispatching
        non_degenerate_input: List[SkeletonNumericalSolution] = list()
        degenerate_input: List[SkeletonNumericalSolution] = list()
        for elem in dispatcher_input:
            elem_subst_map = self.value_to_substitute_map(elem)
            checker_for_elem = checker_with_parameter.subs(elem_subst_map)
            if checker_for_elem == sp.S.BooleanTrue:
                degenerate_input.append(elem)
            elif checker_for_elem == sp.S.BooleanFalse:
                non_degenerate_input.append(elem)

        # Assign to non-degenerate output
        non_degenerate_child_idx = dispatcher.non_degenerate_child.flatten_idx
        self._input_validity[non_degenerate_child_idx] = input_validity and (len(non_degenerate_input) > 0)
        for elem in non_degenerate_input:
            self._node_inputs[non_degenerate_child_idx].append(elem)

        # Assign to degenerate output
        if dispatcher.degenerate_child is not None:
            degenerate_child_idx = dispatcher.degenerate_child.flatten_idx
            self._input_validity[degenerate_child_idx] = input_validity and (len(degenerate_input) > 0)
            for elem in degenerate_input:
                self._node_inputs[degenerate_child_idx].append(elem)

    def visit_equation_all_zero_dispatcher(self, dispatcher: EquationAllZeroDispatcherNode):
        """
        Just according to whether dispatcher is root
        """
        assert len(dispatcher.children) == 2
        if dispatcher.is_root:
            self.visit_equation_all_zero_dispatcher_root(dispatcher)
        else:
            self.visit_equation_all_zero_dispatcher_non_root(dispatcher)

    def visit_solved_variable_dispatcher_root(self, dispatcher: SolvedVariableDispatcherNode):
        """
        In this case, the variable solution degenerate record means a PARAMETER (not unknown)
        equals a given value. Only one child can be valid.
        """
        assert dispatcher.is_root
        assert dispatcher.flatten_idx == 0
        taken_by_degenerate = False
        for i in range(len(dispatcher.branch_conditions)):
            condition_i = dispatcher.branch_conditions[i]
            condition_i = condition_i.subs(self._parameter_values)
            if condition_i == sp.S.BooleanTrue:
                # Here the condition for a degenerate child is invoked
                if dispatcher.degenerate_branch_child(i) is not None:
                    degenerate_child_idx = dispatcher.degenerate_branch_child(i).flatten_idx
                    self._input_validity[degenerate_child_idx] = True
                    taken_by_degenerate = True
                    break

        # Goes to non-degenerate
        if not taken_by_degenerate:
            non_degenerate_child_idx = dispatcher.non_degenerate_child.flatten_idx
            self._input_validity[non_degenerate_child_idx] = True

    def visit_solved_variable_dispatcher_non_root(self, dispatcher: SolvedVariableDispatcherNode):
        """
        Non-root dispatcher, thus its parent must be a solution node.
        In this case, dispatcher_input == empty means invalid
        """
        # Gather input
        dispatcher_idx = dispatcher.flatten_idx
        input_validity = self._input_validity[dispatcher_idx]
        dispatcher_input = self._node_inputs[dispatcher_idx]
        if (not input_validity) or (len(dispatcher_input) == 0):
            self.assign_invalid_to_children(dispatcher)
            return

        # Init data for dispatching
        degenerate_input_list: List[List[SkeletonNumericalSolution]] = list()
        non_degenerate_input: List[SkeletonNumericalSolution] = list()
        for i in range(len(dispatcher.branch_conditions)):
            degenerate_input_list.append(list())

        # Do dispatch for all inputs
        for i in range(len(dispatcher_input)):
            solution_i = dispatcher_input[i]
            taken_by_branch = False
            for j in range(len(dispatcher.branch_conditions)):
                condition_j = dispatcher.branch_conditions[j]
                condition_j_with_parameter = condition_j.subs(self._parameter_values)
                solution_i_subst = self.value_to_substitute_map(solution_i)
                condition_j_with_variable = condition_j_with_parameter.subs(solution_i_subst)
                if condition_j_with_variable == sp.S.BooleanTrue:
                    degenerate_input_list[j].append(solution_i)
                    taken_by_branch = True
                    break

            if not taken_by_branch:
                non_degenerate_input.append(solution_i)

        # Append to non-degenerate
        assert dispatcher.non_degenerate_child is not None
        non_degenerate_child_idx = dispatcher.non_degenerate_child.flatten_idx
        self._input_validity[non_degenerate_child_idx] = input_validity and (len(non_degenerate_input) > 0)
        for elem in non_degenerate_input:
            self._node_inputs[non_degenerate_child_idx].append(elem)

        # Append to degenerate
        for i in range(len(dispatcher.branch_conditions)):
            if dispatcher.degenerate_branch_child(i) is not None:
                this_child = dispatcher.degenerate_branch_child(i).flatten_idx
                self._input_validity[this_child] = input_validity and (len(degenerate_input_list[i]) > 0)
                for elem in degenerate_input_list[i]:
                    self._node_inputs[this_child].append(elem)

    def visit_solved_variable_dispatcher(self, dispatcher: SolvedVariableDispatcherNode):
        """
        Just according to whether dispatcher is root
        """
        assert len(dispatcher.children) == len(dispatcher.branch_conditions) + 1
        if dispatcher.is_root:
            self.visit_solved_variable_dispatcher_root(dispatcher)
        else:
            self.visit_solved_variable_dispatcher_non_root(dispatcher)

    def solution_node_assign_solved_values(
            self,
            solution_node: SolutionNodeBase,
            output_values: List[SkeletonNumericalSolution]):
        if solution_node.dispatcher_child is not None:
            child_idx = solution_node.dispatcher_child.flatten_idx
            self._input_validity[child_idx] = (len(output_values) > 0)
            for elem in output_values:
                self._node_inputs[child_idx].append(elem)
        else:
            # Append to output
            for elem in output_values:
                self._ik_output.append(elem)

    def visit_solution_node_root(self, solution_node: SolutionNodeBase, variable_solution: VariableSolution):
        """
        The root node does not take any variable value (only parameter)
        """
        assert solution_node.is_parent_root_dispatcher()
        input_validity = self._input_validity[solution_node.flatten_idx]
        if not input_validity:
            self.assign_invalid_to_children(solution_node)
            return

        # Input is valid
        empty_value = SkeletonNumericalSolution([], [])
        empty_value.initialize(len(self._index_to_symbol))
        index_to_assign = self._symbol_to_index[variable_solution.solved_variable]
        if variable_solution.is_explicit_solution:
            next_values = step_from_explicit_solution(
                empty_value, dict(), variable_solution, index_to_assign, self._parameter_values)
        elif variable_solution.is_polynomial:
            next_values = step_from_polynomial_solution(
                empty_value, dict(), variable_solution, index_to_assign, self._parameter_values)
        elif variable_solution.impl_class_key == VariableSolutionClassKey.General6DoFNumericalReduce.name:
            next_values = step_from_general_6dof_numerical_reduce_solution(
                empty_value, dict(), variable_solution, index_to_assign, self._parameter_values)
        elif variable_solution.impl_class_key == VariableSolutionClassKey.LinearSinCosType_2.name:
            next_values = step_from_linear_type2_solution(
                empty_value, dict(), variable_solution, index_to_assign, self._parameter_values)
        else:
            raise NotImplementedError('The solution type is not support yet')
        self.solution_node_assign_solved_values(solution_node, next_values)

    def visit_solution_node_non_root(self, solution_node: SolutionNodeBase, variable_solution: VariableSolution):
        """
        Again, for non-root node, node_input == empty implies invalid
        """
        # Fetch data
        node_inputs = self._node_inputs[solution_node.flatten_idx]
        input_validity = self._input_validity[solution_node.flatten_idx]
        if (not input_validity) or (len(node_inputs) == 0):
            self.assign_invalid_to_children(solution_node)
            return

        # Do solution
        output_values: List[SkeletonNumericalSolution] = list()
        for current_value in node_inputs:
            value_subst_map = self.value_to_substitute_map(current_value)
            this_assign_index = self._symbol_to_index[variable_solution.solved_variable]
            if variable_solution.is_explicit_solution:
                next_value = step_from_explicit_solution(
                    current_value, value_subst_map, variable_solution, this_assign_index, self._parameter_values)
            elif variable_solution.is_polynomial:
                next_value = step_from_polynomial_solution(
                    current_value, value_subst_map, variable_solution, this_assign_index, self._parameter_values)
            elif variable_solution.impl_class_key == VariableSolutionClassKey.LinearSinCosType_2.name:
                next_value = step_from_linear_type2_solution(
                    current_value, value_subst_map, variable_solution, this_assign_index, self._parameter_values)
            else:
                raise NotImplementedError('Solution type not supported yet')
            output_values.extend(next_value)
        self.solution_node_assign_solved_values(solution_node, output_values)

    def visit_explicit_solution_node(self, solution_node: ExplicitSolutionNode):
        if solution_node.is_parent_root_dispatcher():
            self.visit_solution_node_root(solution_node, solution_node.explicit_solution)
        else:
            self.visit_solution_node_non_root(solution_node, solution_node.explicit_solution)

    def visit_polynomial_solution_node(self, solution_node: PolynomialSolutionNode):
        if solution_node.is_parent_root_dispatcher():
            self.visit_solution_node_root(solution_node, solution_node.polynomial_solution)
        else:
            self.visit_solution_node_non_root(solution_node, solution_node.polynomial_solution)

    def visit_linear_type2_solution_node(self, solution_node: SolutionNodeBase):
        if solution_node.is_parent_root_dispatcher():
            self.visit_solution_node_root(solution_node, solution_node.solution)
        else:
            self.visit_solution_node_non_root(solution_node, solution_node.solution)

    def visit_general_6dof_numerical_reduce_node(self, solution_node: SolutionNodeBase):
        if solution_node.is_parent_root_dispatcher():
            self.visit_solution_node_root(solution_node, solution_node.solution)
        else:
            raise NotImplementedError('General 6-dof solution is only available on root')


def run_visitor_ik_random(
        skeleton_tree: SkeletonTree,
        theta_in: Optional[np.ndarray] = None,
        logging: bool = False) -> int:
    """
    Solve the skeleton_tree with ee_pose from a random joint-space configuration,
    and return the number of valid solution. Optionally, a theta_in can be provided
    instead of the random configuration
    """
    # First do fk
    robot = skeleton_tree.robot
    from fk.kinematics_dh import forward_kinematics_dh
    fk_out = forward_kinematics_dh(
        robot.dh_params,
        [elem.symbol for elem in robot.unknowns])
    ee_symbolic = fk_out.T_ee()
    ee_symbolic = ee_symbolic.subs(robot.parameters_value)

    # Make the theta
    if theta_in is None:
        theta = np.random.random(size=(robot.n_dofs, ))
    else:
        theta = theta_in

    if logging:
        print('The theta input for testing is: ', theta)

    # Run fk
    subst_input = [(robot.unknowns[i].symbol, theta[i]) for i in range(robot.n_dofs)]
    ee_target_sp = ee_symbolic.subs(subst_input)
    ee_target = np.eye(4)
    for i in range(4):
        for j in range(4):
            ee_target[i, j] = float(ee_target_sp[i, j])

    # Make the subst map
    print(ee_target)
    subst_map: Dict[sp.Symbol, float] = ik_target_subst_map(ee_target)
    subst_map.update(robot.parameters_value)
    unknown_as_parameter_subst_map = dict()
    for i in range(len(robot.unknowns)):
        # if not in used unknown
        unknown_i = robot.unknowns[i]
        if find_unknown_idx(skeleton_tree.solved_equation_record.used_unknowns, unknown_i.symbol.name) < 0:
            unknown_as_parameter_subst_map[unknown_i.symbol] = theta[i]
    subst_map.update(unknown_as_parameter_subst_map)

    # Make the visitor
    visitor = VisitorNumericalSolution(len(skeleton_tree.node_list))
    visitor.initialize(subst_map, all_variable_in_tree=skeleton_tree.all_solved_symbol(), one_gt_theta=subst_input)
    skeleton_tree.preorder_visit(visitor)

    # Test of numerical solutions
    ik_output = visitor.ik_output
    if logging:
        print('Found {n} candidate solutions'.format(n=len(ik_output)))

    # Count the number of solution
    n_valid_solution = 0
    for i in range(len(ik_output)):
        output_i = ik_output[i]
        output_i_subst = visitor.value_to_substitute_map(output_i)
        output_i_subst.update(unknown_as_parameter_subst_map)
        ee_numerical = ee_symbolic.subs(output_i_subst)
        contain_symbol = False
        for r in range(3):
            for c in range(4):
                rc_value_expr: sp.Expr = ee_numerical[r, c]
                rc_value = cast_expr_to_float(rc_value_expr)
                if rc_value is not None:
                    pass
                else:
                    contain_symbol = True

        # The printing
        diff_value = None
        if not contain_symbol:
            ee_diff = ee_numerical - ee_target
            ee_diff = np.abs(ee_diff)
            diff_value = np.max(ee_diff)
            if diff_value < 1e-4:
                n_valid_solution += 1

        # Logging
        output_array = list()
        for k in range(len(output_i.variable_values)):
            value_k = output_i.variable_values[k]
            solution_idx_k = output_i.variable_value_in_solution_idx[k]
            symbol_k = visitor._index_to_symbol[k]
            output_array.append((symbol_k, value_k, solution_idx_k))

        if logging:
            print('Difference w/ ee', diff_value)
            print('The solution ', output_array)

    # Finish
    return n_valid_solution


# Debug code
def load_tree(test_data_dir: Optional[str] = None):
    # Make the tree
    import yaml
    from codegen.skeleton.tree_serialize import TreeSerializer
    if test_data_dir is None:
        test_data_dir = '../../gallery/atlas_new/atlas_l_hand_main.yaml'
    with open(test_data_dir, 'r') as read_stream:
        data_map = yaml.load(read_stream, Loader=yaml.CLoader)
    read_stream.close()

    # Save back
    tree = TreeSerializer.load_skeleton_tree(data_map)
    return tree


def test_tree():
    tree = load_tree()
    for i in range(40):
        n_solution = run_visitor_ik_random(tree, logging=True)
        print('Find {n} solution'.format(n=n_solution))


def run_tree():
    tree = load_tree()
    # theta_in = np.array([0.38308188, 0.40290335, 0.12104437, 0.74706485, 0.75383882, 0.76679588, 0.31858327])
    run_visitor_ik_random(tree, theta_in=None, logging=True)


if __name__ == '__main__':
    run_tree()
