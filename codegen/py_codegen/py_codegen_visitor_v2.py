import numpy as np

from codegen.skeleton.codegen_base import indent
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.tree_node import SkeletonTreeNode, NoBranchDispatcherNode, DispatcherNodeBase
from codegen.skeleton.tree_node import EquationAllZeroDispatcherNode, SolvedVariableDispatcherNode
from codegen.skeleton.tree_node import ExplicitSolutionNode, PolynomialSolutionNode
from codegen.skeleton.tree_node import General6DoFNumericalReduceSolutionNode
from codegen.skeleton.tree_node import LinearSinCosType_2_SolverNode
from codegen.py_codegen.py_codegen_visitor_base import IkSolutionGeneratorPythonBase
from solver.solved_variable import SolutionMethod, VariableSolutionClassKey
from solver.solved_variable_impl import General6DoFNumericalReduceSolutionImpl
from typing import List, Optional
import sympy as sp


class ScopeManager(object):

    def __init__(self, code_lines_workspace: List[str]):
        self._code_lines = code_lines_workspace
        self._scope_name_stack: List[str] = list()

    def enter_scope(self, scope_name: str):
        self._scope_name_stack.append(scope_name)

    def exit_scope(self, scope_name: Optional[str]):
        if scope_name is not None:
            assert self._scope_name_stack[-1] == scope_name
        self._scope_name_stack.pop()

    def current_indent(self) -> str:
        return indent(self.indent_level)

    def append_code_line(self, code_line: str):
        self._code_lines.append(self.current_indent() + code_line)

    @property
    def indent_level(self):
        return len(self._scope_name_stack)


class NodeIkSolutionGeneratorPython_v2(IkSolutionGeneratorPythonBase):

    def __init__(self, tree: SkeletonTree, use_safe_operation: bool = True):
        super().__init__(tree, use_safe_operation)
        self._scope_manager = ScopeManager(self._code_lines)

    def add_code_line(self, code_line: str):
        self._scope_manager.append_code_line(code_line)

    def enter_scope(self, scope_name: str):
        self._scope_manager.enter_scope(scope_name)

    def exit_scope(self, scope_name: Optional[str]):
        self._scope_manager.exit_scope(scope_name)

    @property
    def indent_level(self):
        return self._scope_manager.indent_level

    # The init code
    def append_init_lines(self):
        tree = self._tree
        n_nodes = tree.num_nodes
        # Code for the ik solution type
        self.add_code_line('')
        self.add_code_line('# A new ik type. Should be a fixed array in C++')
        self.add_code_line('IkSolution = NewType("IkSolution", List[float])')
        self.add_code_line('def make_ik_solution():')
        self.enter_scope('make_ik_solution')
        self.add_code_line('tmp_sol = IkSolution(list())')
        self.add_code_line(
            'for tmp_sol_idx in range({n_vars}):'.format(n_vars=len(tree.all_solved_symbol())))
        self.enter_scope('for tmp_sol_idx')
        self.add_code_line('tmp_sol.append(100000.0)')
        self.exit_scope('for tmp_sol_idx')
        self.add_code_line('return tmp_sol')
        self.exit_scope('make_ik_solution')

        # Code for the shared queue
        self.add_code_line('')
        self.add_code_line('solution_queue: List[IkSolution] = list()')
        self.add_code_line('queue_element_validity: List[bool] = list()')

        # Helper functor to append solution
        self.add_code_line('def append_solution_to_queue(solution_2_add: IkSolution):')
        self.enter_scope('append_solution')
        self.add_code_line('index_4_appended = len(solution_queue)')
        self.add_code_line('solution_queue.append(solution_2_add)')
        self.add_code_line('queue_element_validity.append(True)')
        self.add_code_line('return index_4_appended')
        self.exit_scope('append_solution')

        # Code for the node input index
        self.add_code_line('')
        self.add_code_line('# Init for workspace as empty list. A list of fixed size array for each node')
        self.add_code_line('max_n_solutions: int = 16')
        self.add_code_line('node_input_index: List[List[int]] = list()')
        self.add_code_line('node_input_validity: List[bool] = list()')
        self.add_code_line('for i in range({num_nodes}):'.format(num_nodes=n_nodes))
        self.enter_scope('for i')
        self.add_code_line('node_input_index.append(list())')
        self.add_code_line('node_input_validity.append(False)')
        self.exit_scope('for i')

        # Helper functor to append input
        self.add_code_line('def add_input_index_to(node_idx: int, solution_idx: int):')
        self.enter_scope('add_input_index')
        self.add_code_line('node_input_index[node_idx].append(solution_idx)')
        self.add_code_line('node_input_validity[node_idx] = True')
        self.exit_scope('add_input_index')
        assert self.indent_level == 0

        # Init for node 0
        self.add_code_line('node_input_validity[0] = True')

    def append_finalize_lines(self):
        # Collect info
        tree = self._tree
        robot = tree.robot
        solved_unknowns = tree.solved_unknowns  # change from robot.unknowns
        solved_symbols = [elem.symbol for elem in solved_unknowns]
        solved_symbol_to_index = self.symbol_to_index
        free_symbols = self.free_symbols()

        # Start writing the code
        assert self.indent_level == 0
        self.add_code_line('')
        self.add_code_line('# Collect the output')
        self.add_code_line('ik_out: List[np.ndarray] = list()'.
                           format(last_idx=tree.num_nodes))
        self.add_code_line('for i in range(len(solution_queue)):')
        self.enter_scope('for_all_queue')
        self.add_code_line('if not queue_element_validity[i]:')
        self.add_code_line(indent() + 'continue')
        self.add_code_line('ik_out_i = solution_queue[i]')
        self.add_code_line('new_ik_i = np.zeros((robot_nq, 1))')

        # Collect the variable corresponds to unknowns
        for i in range(len(robot.unknowns)):
            unknown_i = robot.unknowns[i]
            if unknown_i.symbol in solved_symbols:
                solved_symbol_i = unknown_i.symbol
                self.add_code_line('value_at_{idx} = ik_out_i[{out_symbol_idx}]  # {symbol}'.
                                   format(idx=i,
                                          out_symbol_idx=solved_symbol_to_index[solved_symbol_i],
                                          symbol=solved_symbol_i.name))
            else:
                # This should be a parameter
                assert unknown_i.symbol in free_symbols
                self.add_code_line('value_at_{idx} = {symbol}  # {symbol}'.
                                   format(idx=i, symbol=unknown_i.symbol.name))
            self.add_code_line('new_ik_i[{idx}] = value_at_{idx}'.format(idx=i))
        self.add_code_line('ik_out.append(new_ik_i)')
        self.exit_scope('for_all_queue')
        self.add_code_line('return ik_out')

    def visit_no_branch_dispatcher(self, dispatcher: DispatcherNodeBase):
        # Input/and output in workspace
        input_idx = dispatcher.flatten_idx
        assert dispatcher.non_degenerate_child is not None
        output_idx = dispatcher.non_degenerate_child.flatten_idx

        # Node code
        self.add_code_line('')
        self.add_code_line('# Code for non-branch dispatcher node {idx}'.format(idx=input_idx))
        self.add_code_line('# Actually, there is no code')

    def visit_equation_all_zero_dispatcher(self, dispatcher: EquationAllZeroDispatcherNode):
        # Shared code, extract function headers
        input_idx = dispatcher.flatten_idx
        assert dispatcher.non_degenerate_child is not None

        # The code
        self.add_code_line('')
        self.add_code_line('# Code for equation all-zero dispatcher node {idx}'.format(idx=input_idx))

        # Depends on root or not
        if dispatcher.is_root:
            self._visit_equation_all_zero_dispatcher_root(dispatcher)
        else:
            self._visit_equation_all_zero_dispatcher_non_root(dispatcher)

        # Call the processor and finish
        self.add_code_line('')
        self.add_code_line('# Invoke the processor')
        self.add_code_line(self.node_processor_name(dispatcher) + '()')
        self.add_code_line('# Finish code for equation all-zero dispatcher node {idx}'.format(idx=input_idx))

    def _visit_equation_all_zero_dispatcher_root(self, dispatcher: EquationAllZeroDispatcherNode):
        assert self.indent_level == 0
        checker_i = dispatcher.degenerate_checker
        checker_i_code = self.generate_py_code(checker_i)
        non_degenerate_child_idx = dispatcher.non_degenerate_child.flatten_idx
        degenerate_child_idx = None if dispatcher.degenerate_child is None else dispatcher.degenerate_child.flatten_idx

        # We do not need to expand the variable as this is the root
        self.add_code_line('def ' + self.node_processor_name(dispatcher) + '():')
        self.enter_scope('checker_processor')
        self.add_code_line('checked_result: bool = {check_code}'.format(check_code=checker_i_code))
        self.add_code_line('if not checked_result:  # To non-degenerate node')
        self.add_code_line(indent() + 'node_input_validity[{out_idx}] = True'.
                           format(out_idx=non_degenerate_child_idx))
        if degenerate_child_idx is not None:
            self.add_code_line('else:')
            self.add_code_line(indent() + 'node_input_validity[{out_idx}] = True'.
                               format(out_idx=degenerate_child_idx))
        self.exit_scope('checker_processor')

    def _visit_equation_all_zero_dispatcher_non_root(self, dispatcher: EquationAllZeroDispatcherNode):
        assert self.indent_level == 0
        # Code is in indent 1
        checker_i = dispatcher.degenerate_checker
        checker_i_code = self.generate_py_code(checker_i)
        input_idx = dispatcher.flatten_idx
        non_degenerate_child_idx = dispatcher.non_degenerate_child.flatten_idx
        degenerate_child_idx = None if dispatcher.degenerate_child is None else dispatcher.degenerate_child.flatten_idx

        # Extract the input
        self._generate_processor_function_define(dispatcher, input_idx)
        assert self.indent_level == 1

        # We do NEED to expand the variable as this is not the root
        solution_name = 'this_solution'
        requested_vars_i = self.get_requested_solved_variables(None, checker_i, dispatcher)
        var_extract_lines = self.extract_solved_variable_codes(requested_vars_i, solution_name=solution_name)

        # The codegen
        self.add_code_line('for i in range(len(this_node_input_index)):')
        self.enter_scope('for_each_solution')
        self.add_code_line('node_input_i_idx_in_queue = this_node_input_index[i]')
        # Check the validity of this solution
        self.add_code_line('if not queue_element_validity[node_input_i_idx_in_queue]:')
        self.add_code_line(indent() + 'continue')
        self.add_code_line('{sol_name} = solution_queue[node_input_i_idx_in_queue]'.format(sol_name=solution_name))
        for this_line in var_extract_lines:
            self.add_code_line(this_line)

        self.add_code_line('checked_result: bool = {check_code}'.format(check_code=checker_i_code))
        self.add_code_line('if not checked_result:  # To non-degenerate node')
        self.add_code_line(indent() + 'add_input_index_to({node_idx}, node_input_i_idx_in_queue)'.
                           format(node_idx=non_degenerate_child_idx))

        # Generate degenerate child idx
        if degenerate_child_idx is not None:
            self.add_code_line('else:')
            self.add_code_line(indent() + 'add_input_index_to({node_idx}, node_input_i_idx_in_queue)'.
                               format(node_idx=degenerate_child_idx))

        # Finish
        self.exit_scope('for_each_solution')
        self.exit_scope(None)  # The processor scope

    def visit_solved_variable_dispatcher(self, dispatcher: SolvedVariableDispatcherNode):
        # Shared code, extract function headers
        input_idx = dispatcher.flatten_idx
        assert dispatcher.non_degenerate_child is not None

        # The code
        self.add_code_line('')
        self.add_code_line('# Code for solved_variable dispatcher node {idx}'.format(idx=input_idx))
        self._generate_processor_function_define(dispatcher, input_idx)

        # Depends on root or not
        if dispatcher.is_root:
            self._visit_solved_variable_dispatcher_root(dispatcher)
        else:
            self._visit_solved_variable_dispatcher_non_root(dispatcher)

        # Finish the processor
        self.exit_scope(None)

        # Call the processor and finish
        assert self.indent_level == 0
        self.add_code_line('')
        self.add_code_line('# Invoke the processor')
        self.add_code_line(self.node_processor_name(dispatcher) + '()')
        self.add_code_line('# Finish code for solved_variable dispatcher node {idx}'.format(idx=input_idx))

    def _visit_solved_variable_dispatcher_root(self, dispatcher: SolvedVariableDispatcherNode):
        assert self.indent_level == 1
        raise NotImplementedError

    def _visit_solved_variable_dispatcher_non_root(self, dispatcher: SolvedVariableDispatcherNode):
        assert self.indent_level == 1
        n_child = len(dispatcher.branch_conditions)
        assert dispatcher.non_degenerate_child is not None
        non_degenerate_child_idx = dispatcher.non_degenerate_child.flatten_idx
        self.add_code_line('')

        solution_name = 'this_solution'
        self.add_code_line('for i in range(len(this_node_input_index)):')
        self.enter_scope('for_each_solution')
        self.add_code_line('node_input_i_idx_in_queue = this_node_input_index[i]')

        # Check the validity of this solution
        self.add_code_line('if not queue_element_validity[node_input_i_idx_in_queue]:')
        self.add_code_line(indent() + 'continue')
        self.add_code_line('{sol_name} = solution_queue[node_input_i_idx_in_queue]'.format(sol_name=solution_name))
        self.add_code_line('taken_by_degenerate: bool = False')

        # The case of taken by degenerated child
        for i in range(n_child):
            child_i = dispatcher.degenerate_branch_child(i)
            child_i_idx = None if child_i is None else child_i.flatten_idx
            condition_i = dispatcher.branch_conditions[i]
            condition_i_code = self.generate_py_code(condition_i)
            requested_vars_i = self.get_requested_solved_variables(None, condition_i, dispatcher)
            extract_lines = self.extract_solved_variable_codes(requested_vars_i, solution_name=solution_name)

            # For this solution
            for this_line in extract_lines:
                self.add_code_line(this_line)
            self.add_code_line('degenerate_valid_{idx} = {code}'.format(idx=i, code=condition_i_code))
            self.add_code_line('if degenerate_valid_{idx}:'.format(idx=i))
            self.add_code_line(indent() + 'taken_by_degenerate = True')
            if child_i_idx is not None:
                self.add_code_line(indent() + 'add_input_index_to({node_idx}, node_input_i_idx_in_queue)'.
                                   format(node_idx=child_i_idx))
            self.add_code_line('')

        # The case of non-degenerate
        self.add_code_line('if not taken_by_degenerate:')
        self.add_code_line(indent() + 'add_input_index_to({node_idx}, node_input_i_idx_in_queue)'.
                           format(node_idx=non_degenerate_child_idx))

        self.exit_scope('for_each_solution')

    def visit_explicit_solution_node(self, solution_node: ExplicitSolutionNode):
        # The shared parts
        parent_node = solution_node.parent
        assert parent_node is not None
        if isinstance(parent_node, NoBranchDispatcherNode):
            input_idx = parent_node.flatten_idx
        else:
            input_idx = solution_node.flatten_idx

        self.add_code_line('')
        self.add_code_line('# Code for explicit solution node {idx}, solved variable is {var}'.
                           format(idx=solution_node.flatten_idx, var=solution_node.solution.solved_variable.name))
        self._generate_processor_function_define(solution_node, input_idx)
        assert self.indent_level == 1

        # Whether we need to extract value depends on the root
        if solution_node.is_parent_root_dispatcher():
            self._visit_explicit_solution_node_root(solution_node)
        else:
            self._visit_explicit_solution_node_non_root(solution_node)

        # Generation done for this node
        self.exit_scope(None)

        # Call the processor and finish
        self.add_code_line('# Invoke the processor')
        self.add_code_line(self.node_processor_name(solution_node) + '()')
        self.add_code_line('# Finish code for explicit solution node {idx}'.format(idx=input_idx))
        assert self.indent_level == 0

    def _visit_explicit_solution_node_root(self, solution_node: ExplicitSolutionNode):
        assert self.indent_level == 1
        output_idx = None if solution_node.children[0] is None else solution_node.children[0].flatten_idx
        self.add_code_line('')
        self.add_code_line('# The explicit solution of root node')
        solution = solution_node.solution
        assert solution.is_explicit_solution
        solved_var = solution.solved_variable
        solved_var_idx = self._symbol_to_index[solved_var]

        # The cse
        n_solution = len(solution.explicit_solutions)
        solution_list: List[sp.Expr] = list()
        for i in range(n_solution):
            solution_i = solution.explicit_solutions[i]
            solution_list.append(solution_i)

        # Do cse for solution/checker
        replacements, reduced_solutions = sp.cse(solution_list)

        # The code gen
        for i in range(n_solution):
            # The reduced code
            reduced_solution_i = reduced_solutions[i]
            checker_i = solution.argument_validity_checkers[i]
            solution_i_code = self.generate_py_code(reduced_solution_i)
            checker_i_code = self.generate_py_code(checker_i)
            self.add_code_line('condition_{sol_idx}: bool = {cond_code}'.
                               format(sol_idx=i, cond_code=checker_i_code))
            self.add_code_line('if condition_{sol_idx}:'.format(sol_idx=i))

            # The replacement code
            self.enter_scope('if condition')
            self.add_code_line('# Temp variable for efficiency')
            for k in range(len(replacements)):
                symbol_k, expr_k = replacements[k]
                expr_code = self.generate_py_code(expr_k)
                self.add_code_line('{var} = {expr_code}'.format(var=str(symbol_k), expr_code=expr_code))
            self.add_code_line('# End of temp variables')

            # Actual solution
            self.add_code_line('solution_{sol_idx}: IkSolution = make_ik_solution()'.format(sol_idx=i))
            self.add_code_line('solution_{sol_idx}[{var_idx}] = {sol_code}'.
                               format(sol_idx=i, var_idx=solved_var_idx, sol_code=solution_i_code))
            self.add_code_line('appended_idx = append_solution_to_queue(solution_{sol_idx})'.format(sol_idx=i))
            if output_idx is not None:
                self.add_code_line('add_input_index_to({node_idx}, appended_idx)'.format(node_idx=output_idx))
            self.add_code_line('')
            self.exit_scope('if condition')

    def _visit_explicit_solution_node_non_root(self, solution_node: ExplicitSolutionNode):
        assert self.indent_level == 1
        output_idx = None if solution_node.children[0] is None else solution_node.children[0].flatten_idx
        self.add_code_line('')
        self.add_code_line('# The solution of non-root node {idx}'.format(idx=solution_node.flatten_idx))
        solution = solution_node.solution
        solved_var = solution.solved_variable
        solved_var_idx = self._symbol_to_index[solved_var]

        # Gather the solution and checker
        n_solution = len(solution.explicit_solutions)
        solution_list: List[sp.Expr] = list()
        checker_list: List[sp.Expr] = list()
        for i in range(n_solution):
            solution_i = solution.explicit_solutions[i]
            checker_i = solution.argument_validity_checkers[i]
            solution_list.append(solution_i)
            checker_list.append(checker_i)

        # Do cse for solution/checker
        solution_replacements, reduced_solutions = sp.cse(solution_list)

        # All variables
        solution_name = 'this_solution'
        requested_vars = self.gather_request_solved_variables(solution_list + checker_list, solution_node)
        var_extract_lines = self.extract_solved_variable_codes(requested_vars, solution_name=solution_name)

        # The codegen
        self.add_code_line('for i in range(len(this_node_input_index)):')
        self.enter_scope('for_each_solution')
        self.add_code_line('node_input_i_idx_in_queue = this_node_input_index[i]')
        # Check the validity of this solution
        self.add_code_line('if not queue_element_validity[node_input_i_idx_in_queue]:')
        self.add_code_line(indent() + 'continue')
        self.add_code_line('{sol_name} = solution_queue[node_input_i_idx_in_queue]'.format(sol_name=solution_name))
        for this_line in var_extract_lines:
            self.add_code_line(this_line)

        # For each solution
        for i in range(n_solution):
            # The reduced code
            reduced_solution_i = reduced_solutions[i]
            checker_i = checker_list[i]
            solution_i_code = self.generate_py_code(reduced_solution_i)
            checker_i_code = self.generate_py_code(checker_i)

            # Extraction done, do solving
            self.add_code_line('condition_{sol_idx}: bool = {cond_code}'.
                               format(sol_idx=i, cond_code=checker_i_code))
            self.add_code_line('if condition_{sol_idx}:'.format(sol_idx=i))
            self.enter_scope('if condition')

            # The replacement code, we can only place it here as it may contain dangerous code
            self.add_code_line('# Temp variable for efficiency')
            for k in range(len(solution_replacements)):
                symbol_k, expr_k = solution_replacements[k]
                expr_code = self.generate_py_code(expr_k)
                self.add_code_line('{var} = {expr_code}'.format(var=str(symbol_k), expr_code=expr_code))
            self.add_code_line('# End of temp variables')

            # The actual solve code
            if i != n_solution - 1:
                self.add_code_line('solution_{sol_idx}: IkSolution = copy.copy({sol_name})'.
                                   format(sol_idx=i, sol_name=solution_name))
                self.add_code_line('solution_{sol_idx}[{var_idx}] = {sol_code}'.
                                   format(sol_idx=i, var_idx=solved_var_idx, sol_code=solution_i_code))
                self.add_code_line('appended_idx = append_solution_to_queue(solution_{sol_idx})'.format(sol_idx=i))
                if output_idx is not None:
                    self.add_code_line('add_input_index_to({node_idx}, appended_idx)'.format(node_idx=output_idx))
                self.add_code_line('')
            else:
                # Directly assign to this_solution
                self.add_code_line('{sol_name}[{var_idx}] = {sol_code}'.
                                   format(sol_name=solution_name, var_idx=solved_var_idx, sol_code=solution_i_code))
                self.add_code_line('solution_queue[node_input_i_idx_in_queue] = {sol_name}'.
                                   format(sol_name=solution_name))
                if output_idx is not None:
                    self.add_code_line('add_input_index_to({node_idx}, node_input_i_idx_in_queue)'.format(node_idx=output_idx))

            # Finish for if
            self.exit_scope('if condition')

            # Set validity for the last idx
            if i == n_solution - 1:
                self.add_code_line('else:')
                self.enter_scope('else')
                self.add_code_line('queue_element_validity[node_input_i_idx_in_queue] = False')
                self.exit_scope('else')
                self.add_code_line('')
        self.exit_scope('for_each_solution')

    def visit_polynomial_solution_node(self, solution_node: PolynomialSolutionNode):
        # The shared parts
        parent_node = solution_node.parent
        assert parent_node is not None
        if isinstance(parent_node, NoBranchDispatcherNode):
            input_idx = parent_node.flatten_idx
        else:
            input_idx = solution_node.flatten_idx

        self.add_code_line('')
        self.add_code_line('# Code for polynomial solution node {idx}, solved variable is {var}'.
                           format(idx=input_idx, var=solution_node.solution.solved_variable.name))

        self._generate_processor_function_define(solution_node, input_idx)
        assert self.indent_level == 1

        # Generate the solver body
        if solution_node.is_parent_root_dispatcher():
            self._visit_polynomial_solution_node_root(solution_node)
        else:
            raise NotImplementedError("We should use polynomial only if nothing-else can be found")

        # Call the processor and finish
        assert self.indent_level == 0
        self.add_code_line('')
        self.add_code_line('# Invoke the processor')
        self.add_code_line(self.node_processor_name(solution_node) + '()')
        self.add_code_line('# Finish code for polynomial solution node {idx}'.format(idx=input_idx))

    def _visit_polynomial_solution_node_root(self, solution_node: PolynomialSolutionNode):
        assert self.indent_level == 1
        output_idx = None if solution_node.children[0] is None else solution_node.children[0].flatten_idx
        self.add_code_line('')
        self.add_code_line('# The polynomial solution of root node')
        solution = solution_node.solution
        assert solution.is_polynomial
        solved_var = solution.solved_variable
        solved_var_idx = self._symbol_to_index[solved_var]

        # Build the polynomial
        poly_dict_input = solution.polynomial_to_solve
        poly_order = -1
        for order in poly_dict_input:
            # Get the coefficient
            numerator, denominator = poly_dict_input[order]
            numerator_code = self.generate_py_code(numerator)
            denominator_code = self.generate_py_code(denominator)
            self.add_code_line('poly_coefficient_{order}_num = {code}'.format(order=order, code=numerator_code))
            self.add_code_line('poly_coefficient_{order}_denom = {code}'.format(order=order, code=denominator_code))
            self.add_code_line('poly_coefficient_{order} = poly_'
                               'coefficient_{order}_num / poly_coefficient_{order}_denom'.format(order=order))
            if order > poly_order:
                poly_order = order

        # Make the coefficients
        assert poly_order >= 1
        self.add_code_line('p_coefficients = np.zeros(shape=({poly_order} + 1,))'.format(poly_order=poly_order))
        for order in poly_dict_input:
            self.add_code_line('p_coefficients[{idx}] = poly_coefficient_{order}'.
                               format(order=order, idx=poly_order - order))

        self.add_code_line('')
        self.add_code_line('# Note that in np.roots, p_coefficient[0] is the highest order')
        self.add_code_line('poly_roots = np.roots(p_coefficients)')

        # Collect the result
        self.add_code_line('')
        self.add_code_line('# Result collection')
        self.add_code_line('for root_idx in range(len(poly_roots)):')
        self.enter_scope('for_root')
        self.add_code_line('this_root = poly_roots[root_idx]')
        self.add_code_line('if not np.isreal(this_root):')
        self.add_code_line(indent() + 'continue')

        # Depends on the solution method
        solution_method = solution.solution_method
        if solution_method == SolutionMethod.PolynomialSin.name or solution_method == SolutionMethod.PolynomialCos.name:
            self.add_code_line('if abs(this_root) > 1:')
            self.add_code_line(indent() + 'continue')
            if solution_method == SolutionMethod.PolynomialSin.name:
                self.add_code_line('first_angle = np.arcsin(this_root)')
                self.add_code_line('second_angle = np.pi - np.arcsin(this_root)')
            else:
                self.add_code_line('first_angle = np.arccos(this_root)')
                self.add_code_line('second_angle = - np.arccos(this_root)')

            # The remaining code is the same
            self.add_code_line('solution_{sol_idx}: IkSolution = make_ik_solution()'.format(sol_idx=0))
            self.add_code_line('solution_{sol_idx}[{var_idx}] = first_angle'.
                               format(sol_idx=0, var_idx=solved_var_idx))
            self.add_code_line('solution_{sol_idx}: IkSolution = make_ik_solution()'.format(sol_idx=1))
            self.add_code_line('solution_{sol_idx}[{var_idx}] = second_angle'.
                               format(sol_idx=1, var_idx=solved_var_idx))
            self.add_code_line('appended_idx_{sol_idx} = append_solution_to_queue(solution_{sol_idx})'.
                               format(sol_idx=0))
            self.add_code_line('appended_idx_{sol_idx} = append_solution_to_queue(solution_{sol_idx})'.
                               format(sol_idx=1))
            if output_idx is not None:
                self.add_code_line('add_input_index_to({node_idx}, appended_idx_0)'.format(node_idx=output_idx))
                self.add_code_line('add_input_index_to({node_idx}, appended_idx_1)'.format(node_idx=output_idx))
        else:
            raise NotImplementedError("Polynomial solver codegen with non sin/cos intermediate is not implemented yet")

        # Finish the polynomial solve
        self.exit_scope('for_root')
        self.exit_scope(None)

    def visit_linear_type2_solution_node(self, solution_node: LinearSinCosType_2_SolverNode):
        parent_node = solution_node.parent
        assert parent_node is not None
        if isinstance(parent_node, NoBranchDispatcherNode):
            input_idx = parent_node.flatten_idx
        else:
            input_idx = solution_node.flatten_idx

        self.add_code_line('')
        self.add_code_line('# Code for linear solution type2 node {idx}, solved variable is {var}'.
                           format(idx=solution_node.flatten_idx, var=solution_node.solution.solved_variable.name))
        self._generate_processor_function_define(solution_node, input_idx)
        assert self.indent_level == 1

        if solution_node.is_parent_root_dispatcher():
            pass
        else:
            self._visit_linear_type2_solution_non_root(solution_node)

        # Generation done for this node
        self.exit_scope(None)

        # Call the processor and finish
        self.add_code_line('# Invoke the processor')
        self.add_code_line(self.node_processor_name(solution_node) + '()')
        self.add_code_line('# Finish code for explicit solution node {idx}'.format(idx=input_idx))
        assert self.indent_level == 0

    def _visit_linear_type2_solution_non_root(self, solution_node: LinearSinCosType_2_SolverNode):
        assert self.indent_level == 1
        assert solution_node.solution.impl_class_key == VariableSolutionClassKey.LinearSinCosType_2.name
        output_idx = None if solution_node.children[0] is None else solution_node.children[0].flatten_idx

        # Gather information
        solution = solution_node.solution
        solved_var = solution.solved_variable
        solved_var_idx = self._symbol_to_index[solved_var]
        A_matrix: sp.Matrix = solution_node.A_matrix
        rows_to_try = solution_node.rows_to_try

        # Gather the flat matrix
        A_matrix_flat_expr: List[sp.Expr] = list()
        assert A_matrix.cols == 4
        for r in range(A_matrix.rows):
            for c in range(A_matrix.cols):
                rc_expr: sp.Expr = A_matrix[r, c]
                A_matrix_flat_expr.append(rc_expr)

        # Do cse for solution/checker
        solution_replacements, reduced_A_matrix_flat = sp.cse(A_matrix_flat_expr)
        # All variables
        solution_name = 'this_solution'
        requested_vars = self.gather_request_solved_variables(A_matrix_flat_expr, solution_node)
        var_extract_lines = self.extract_solved_variable_codes(requested_vars, solution_name=solution_name)

        # The codegen
        self.add_code_line('')
        self.add_code_line('for i in range(len(this_node_input_index)):')
        self.enter_scope('for_each_solution')
        self.add_code_line('node_input_i_idx_in_queue = this_node_input_index[i]')
        # Check the validity of this solution
        self.add_code_line('if not queue_element_validity[node_input_i_idx_in_queue]:')
        self.add_code_line(indent() + 'continue')
        self.add_code_line('{sol_name} = solution_queue[node_input_i_idx_in_queue]'.format(sol_name=solution_name))
        for this_line in var_extract_lines:
            self.add_code_line(this_line)

        # The replacement code, we can only place it here as it may contain dangerous code
        self.add_code_line('# Temp variable for efficiency')
        for k in range(len(solution_replacements)):
            symbol_k, expr_k = solution_replacements[k]
            expr_code = self.generate_py_code(expr_k)
            self.add_code_line('{var} = {expr_code}'.format(var=str(symbol_k), expr_code=expr_code))
        self.add_code_line('# End of temp variables')

        # Generate the matrix code
        self.add_code_line('A_matrix = np.zeros(shape=({rows}, 4))'.format(rows=A_matrix.rows))
        offset = 0
        for r in range(A_matrix.rows):
            for c in range(A_matrix.cols):
                rc_expr: sp.Expr = reduced_A_matrix_flat[offset]
                if rc_expr != sp.S.Zero:
                    rc_expr_code = self.generate_py_code(rc_expr)
                    self.add_code_line(f'A_matrix[{r}, {c}] = {rc_expr_code}')
                offset += 1

        # For each rows
        for i in range(len(rows_to_try)):
            rows_i = rows_to_try[i]
            r0, r1, r2 = rows_i
            self.add_code_line(f'solution_tuple_{i} = try_solve_linear_type2_specific_rows(A_matrix, {r0}, {r1}, {r2})')
            self.add_code_line(f'if solution_tuple_{i} is not None:')
            self.enter_scope('if_valid_solution')
            # The first solution
            self.add_code_line('solution_0: IkSolution = copy.copy({sol_name})'.
                               format(sol_name=solution_name))
            self.add_code_line('solution_0[{var_idx}] = solution_tuple_{i}[0]'.
                               format(var_idx=solved_var_idx, i=i))
            self.add_code_line('appended_idx = append_solution_to_queue(solution_0)')
            if output_idx is not None:
                self.add_code_line('add_input_index_to({node_idx}, appended_idx)'.format(node_idx=output_idx))
            self.add_code_line('')
            # The second solution
            self.add_code_line('{sol_name}[{var_idx}] = solution_tuple_{i}[1]'.
                               format(sol_name=solution_name, var_idx=solved_var_idx, i=i))
            self.add_code_line('solution_queue[node_input_i_idx_in_queue] = {sol_name}'.
                               format(sol_name=solution_name))
            if output_idx is not None:
                self.add_code_line(
                    'add_input_index_to({node_idx}, node_input_i_idx_in_queue)'.format(node_idx=output_idx))
            self.add_code_line('continue')
            self.add_code_line('')
            self.exit_scope('if_valid_solution')

        self.exit_scope('for_each_solution')

    def visit_general_6dof_numerical_reduce_node(self, solution_node: General6DoFNumericalReduceSolutionNode):
        # The shared parts
        parent_node = solution_node.parent
        assert parent_node is not None
        if isinstance(parent_node, NoBranchDispatcherNode):
            input_idx = parent_node.flatten_idx
        else:
            input_idx = solution_node.flatten_idx

        self.add_code_line('')
        self.add_code_line('# Code for explicit solution node {idx}, solved variable is {var}'.
                           format(idx=solution_node.flatten_idx, var=solution_node.solution.solved_variable.name))
        self._generate_processor_function_define(solution_node, input_idx)
        assert self.indent_level == 1

        if solution_node.is_parent_root_dispatcher():
            if solution_node.has_semi_symbolic_reduce:
                self._visit_general_6dof_semi_symbolic_reduce_node_root(solution_node)
            else:
                self._visit_general_6dof_numerical_reduce_node_root_cse(solution_node)
        else:
            raise NotImplementedError('General solver can only be the root solution in current implementation')

        # Generation done for this node
        self.exit_scope(None)

        # Call the processor and finish
        self.add_code_line('# Invoke the processor')
        self.add_code_line(self.node_processor_name(solution_node) + '()')
        self.add_code_line('# Finish code for explicit solution node {idx}'.format(idx=input_idx))
        assert self.indent_level == 0

    def _visit_general_6dof_numerical_reduce_node_root(self, solution_node: General6DoFNumericalReduceSolutionNode):
        assert self.indent_level == 1
        self.add_code_line('')
        self.add_code_line('# The general 6-dof solution of root node with numerical reduce')
        raw_solution = solution_node.general_6dof_solution
        solution_impl = raw_solution.internal_solution
        assert solution_impl.is_general_6dof
        assert solution_impl.class_key() == VariableSolutionClassKey.General6DoFNumericalReduce.name
        general_6dof_solution: General6DoFNumericalReduceSolutionImpl = solution_impl

        def generate_symbolic_matrix_code(sp_matrix: sp.Matrix, matrix_name: str):
            self.add_code_line(f'{matrix_name} = np.zeros(shape=({sp_matrix.shape[0]}, {sp_matrix.shape[1]}))')
            for r in range(sp_matrix.shape[0]):
                for c in range(sp_matrix.shape[1]):
                    rc_expr = sp_matrix[r, c]
                    if rc_expr != sp.S.Zero:
                        rc_expr_code = self.generate_py_code(rc_expr)
                        self.add_code_line(f'{matrix_name}[{r}, {c}] = {rc_expr_code}')

        # Generate the required matrix
        A_sin, A_cos, C_const = general_6dof_solution.lhs_matrices()
        N_rhs = general_6dof_solution.rhs_matrix()
        generate_symbolic_matrix_code(A_sin, 'A_sin')
        self.add_code_line('')
        generate_symbolic_matrix_code(A_cos, 'A_cos')
        self.add_code_line('')
        generate_symbolic_matrix_code(C_const, 'C_const')
        self.add_code_line('')
        generate_symbolic_matrix_code(N_rhs, 'N_rhs')

        # Generate lines to reduce
        self._generate_reduced_line_and_invoking_in_general_6dof_code(solution_node)

    def _generate_reduced_line_and_invoking_in_general_6dof_code(
            self, solution_node: General6DoFNumericalReduceSolutionNode):
        raw_solution = solution_node.general_6dof_solution
        solution_impl = raw_solution.internal_solution
        assert solution_impl.is_general_6dof
        assert solution_impl.class_key() == VariableSolutionClassKey.General6DoFNumericalReduce.name
        general_6dof_solution: General6DoFNumericalReduceSolutionImpl = solution_impl
        solved_var = solution_node.solved_variable
        solved_var_idx = self._symbol_to_index[solved_var]

        # Generate lines to reduce
        lines2reduce = general_6dof_solution.select_lines
        lines_to_reduce_code = 'lines2reduce = ('
        for i in range(len(lines2reduce)):
            if i == len(lines2reduce) - 1:
                lines_to_reduce_code += '{line_i}'.format(line_i=lines2reduce[i])
            else:
                lines_to_reduce_code += '{line_i}, '.format(line_i=lines2reduce[i])
        lines_to_reduce_code += ')'
        self.add_code_line('')
        self.add_code_line(lines_to_reduce_code)

        # Solve it
        output_idx = None if solution_node.children[0] is None else solution_node.children[0].flatten_idx
        solution_name = 'local_solutions'
        self.add_code_line('from solver.general_6dof.numerical_reduce_closure_equation import numerical_solve')
        self.add_code_line(f'{solution_name} = numerical_solve(A_sin, A_cos, C_const, N_rhs, lines2reduce)')
        self.add_code_line(f'for {solution_name}_i in {solution_name}:')
        self.enter_scope('for_solution')
        self.add_code_line('solution_i: IkSolution = make_ik_solution()')
        self.add_code_line(f'solution_i[{solved_var_idx}] = {solution_name}_i')
        self.add_code_line('appended_idx = append_solution_to_queue(solution_i)')
        if output_idx is not None:
            self.add_code_line('add_input_index_to({node_idx}, appended_idx)'.format(node_idx=output_idx))
        self.exit_scope('for_solution')

    def _visit_general_6dof_numerical_reduce_node_root_cse(self, solution_node: General6DoFNumericalReduceSolutionNode):
        assert self.indent_level == 1
        self.add_code_line('')
        self.add_code_line('# The general 6-dof solution of root node with numerical reduce')
        raw_solution = solution_node.general_6dof_solution
        solution_impl = raw_solution.internal_solution
        assert solution_impl.is_general_6dof
        assert solution_impl.class_key() == VariableSolutionClassKey.General6DoFNumericalReduce.name
        general_6dof_solution: General6DoFNumericalReduceSolutionImpl = solution_impl

        # Generate the required matrix
        A_sin, A_cos, C_const = general_6dof_solution.lhs_matrices()
        N_rhs = general_6dof_solution.rhs_matrix()
        matrix_expressions: List[sp.Expr] = list()

        # Do accumulation
        def accumulate_matrix_expr(sp_matrix: sp.Matrix):
            for r in range(sp_matrix.shape[0]):
                for c in range(sp_matrix.shape[1]):
                    rc_expr = sp_matrix[r, c]
                    matrix_expressions.append(rc_expr)
        accumulate_matrix_expr(A_sin)
        accumulate_matrix_expr(A_cos)
        accumulate_matrix_expr(C_const)
        accumulate_matrix_expr(N_rhs)

        # Generate temp variable
        replacements, reduced_expressions = sp.cse(matrix_expressions)
        self.add_code_line('')
        self.add_code_line('# Temp variable for efficiency')
        for i in range(len(replacements)):
            symbol_i, expr_i = replacements[i]
            expr_code = self.generate_py_code(expr_i)
            self.add_code_line('{var} = {expr_code}'.format(var=str(symbol_i), expr_code=expr_code))
        self.add_code_line('# End of temp variable')

        # The generator
        def generate_symbolic_matrix_code(sp_matrix: sp.Matrix, matrix_name: str, counter_in: int) -> int:
            self.add_code_line(f'{matrix_name} = np.zeros(shape=({sp_matrix.shape[0]}, {sp_matrix.shape[1]}))')
            local_counter = counter_in
            for r in range(sp_matrix.shape[0]):
                for c in range(sp_matrix.shape[1]):
                    rc_expr = reduced_expressions[local_counter]
                    local_counter += 1
                    if rc_expr != sp.S.Zero:
                        rc_expr_code = self.generate_py_code(rc_expr)
                        self.add_code_line(f'{matrix_name}[{r}, {c}] = {rc_expr_code}')
            return local_counter

        # The counter method
        matrix_expression_reduced_counter = 0
        matrix_expression_reduced_counter = \
            generate_symbolic_matrix_code(A_sin, 'A_sin', matrix_expression_reduced_counter)
        matrix_expression_reduced_counter = \
            generate_symbolic_matrix_code(A_cos, 'A_cos', matrix_expression_reduced_counter)
        matrix_expression_reduced_counter = \
            generate_symbolic_matrix_code(C_const, 'C_const', matrix_expression_reduced_counter)
        matrix_expression_reduced_counter = \
            generate_symbolic_matrix_code(N_rhs, 'N_rhs', matrix_expression_reduced_counter)

        # Generate lines to reduce
        self._generate_reduced_line_and_invoking_in_general_6dof_code(solution_node)

    def _visit_general_6dof_semi_symbolic_reduce_node_root(self, solution_node: General6DoFNumericalReduceSolutionNode):
        assert self.indent_level == 1
        self.add_code_line('')
        self.add_code_line('# The general 6-dof solution of root node with semi-symbolic reduce')
        raw_solution = solution_node.general_6dof_solution
        solution_impl = raw_solution.internal_solution
        assert solution_impl.is_general_6dof
        assert solution_impl.class_key() == VariableSolutionClassKey.General6DoFNumericalReduce.name
        general_6dof_solution: General6DoFNumericalReduceSolutionImpl = solution_impl
        semi_reduce = general_6dof_solution.semi_symbolic_reduce_record

        # Generate R_l and its inverse
        R_l = semi_reduce.R_l
        self.add_code_line(f'R_l = np.zeros(shape=({R_l.shape[0]}, {R_l.shape[1]}))')
        for r in range(R_l.shape[0]):
            for c in range(R_l.shape[1]):
                rc_expr = R_l[r, c]
                if rc_expr != sp.S.Zero:
                    rc_expr_code = self.generate_py_code(rc_expr)
                    self.add_code_line(f'R_l[{r}, {c}] = {rc_expr_code}')

        # Generate matrix inverse
        self.add_code_line('try:')
        self.add_code_line(indent() + 'R_l_mat_inv = np.linalg.inv(R_l)')
        self.add_code_line('except:')
        self.add_code_line(indent() + 'return')

        # Extract the R_l_inv variable
        R_l_inv_symbols = semi_reduce.R_l_inv_as_symbols
        for r in range(R_l.shape[0]):
            for c in range(R_l.shape[1]):
                inv_Rl_symbol_rc = R_l_inv_symbols[r, c]
                self.add_code_line(f'{inv_Rl_symbol_rc.name} = R_l_mat_inv[{r}, {c}]')

        # Generate the required matrix
        A, B, C = semi_reduce.A, semi_reduce.B, semi_reduce.C
        matrix_expressions: List[sp.Expr] = list()

        # Do accumulation
        def accumulate_matrix_expr(sp_matrix: sp.Matrix):
            for r in range(sp_matrix.shape[0]):
                for c in range(sp_matrix.shape[1]):
                    rc_expr = sp_matrix[r, c]
                    matrix_expressions.append(rc_expr)

        accumulate_matrix_expr(A)
        accumulate_matrix_expr(B)
        accumulate_matrix_expr(C)

        # Generate temp variable
        replacements, reduced_expressions = sp.cse(matrix_expressions)
        self.add_code_line('')
        self.add_code_line('# Temp variable for efficiency')
        for i in range(len(replacements)):
            symbol_i, expr_i = replacements[i]
            expr_code = self.generate_py_code(expr_i)
            self.add_code_line('{var} = {expr_code}'.format(var=str(symbol_i), expr_code=expr_code))
        self.add_code_line('# End of temp variable')

        def generate_symbolic_matrix_code(sp_matrix: sp.Matrix, matrix_name: str, counter_in: int) -> int:
            self.add_code_line(f'{matrix_name} = np.zeros(shape=({sp_matrix.shape[0]}, {sp_matrix.shape[1]}))')
            local_counter = counter_in
            for r in range(sp_matrix.shape[0]):
                for c in range(sp_matrix.shape[1]):
                    rc_expr = reduced_expressions[local_counter]
                    local_counter += 1
                    if rc_expr != sp.S.Zero:
                        rc_expr_code = self.generate_py_code(rc_expr)
                        self.add_code_line(f'{matrix_name}[{r}, {c}] = {rc_expr_code}')
            return local_counter

        # The codegen
        matrix_expression_reduced_counter = 0
        matrix_expression_reduced_counter = \
            generate_symbolic_matrix_code(A, 'A', matrix_expression_reduced_counter)
        matrix_expression_reduced_counter = \
            generate_symbolic_matrix_code(B, 'B', matrix_expression_reduced_counter)
        matrix_expression_reduced_counter = \
            generate_symbolic_matrix_code(C, 'C', matrix_expression_reduced_counter)

        # Solve it
        output_idx = None if solution_node.children[0] is None else solution_node.children[0].flatten_idx
        solved_var = solution_node.solved_variable
        solved_var_idx = self._symbol_to_index[solved_var]
        solution_name = 'local_solutions'
        self.add_code_line(f'{solution_name} = '
                           f'compute_solution_from_tanhalf_LME(A, B, C)')
        self.add_code_line(f'for {solution_name}_i in {solution_name}:')
        self.enter_scope('for_solution')
        self.add_code_line('solution_i: IkSolution = make_ik_solution()')
        self.add_code_line(f'solution_i[{solved_var_idx}] = {solution_name}_i')
        self.add_code_line('appended_idx = append_solution_to_queue(solution_i)')
        if output_idx is not None:
            self.add_code_line('add_input_index_to({node_idx}, appended_idx)'.format(node_idx=output_idx))
        self.exit_scope('for_solution')

    # Shared utility
    def _generate_processor_function_define(self, node: SkeletonTreeNode, input_idx: int):
        node_processor_name = self.node_processor_name(node)
        self.add_code_line('def ' + self.node_processor_name(node) + '():')
        self.enter_scope(node_processor_name)
        self.add_code_line('this_node_input_index: List[int] = node_input_index[{in_idx}]'.format(in_idx=input_idx))
        self.add_code_line('this_input_valid: bool = node_input_validity[{in_idx}]'.
                           format(in_idx=input_idx))
        self.add_code_line('if not this_input_valid:')
        self.add_code_line(indent() + 'return')
