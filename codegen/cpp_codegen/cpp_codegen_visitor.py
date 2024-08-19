import copy

from codegen.skeleton.codegen_base import indent
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.tree_node import SkeletonTreeNode, NoBranchDispatcherNode, DispatcherNodeBase
from codegen.skeleton.tree_node import EquationAllZeroDispatcherNode, SolvedVariableDispatcherNode
from codegen.skeleton.tree_node import ExplicitSolutionNode, PolynomialSolutionNode
from codegen.skeleton.tree_node import General6DoFNumericalReduceSolutionNode
from codegen.skeleton.tree_node import LinearSinCosType_2_SolverNode
from codegen.skeleton.tree_node import SolutionNodeBase
from codegen.skeleton.tree_node import SkeletonNodeVisitor
from codegen.cpp_codegen.scope_manager import ScopeManager
from solver.solved_variable import SolutionMethod, VariableSolutionClassKey
from solver.solved_variable_impl import General6DoFNumericalReduceSolutionImpl
from typing import List, Optional, Dict, Set
import sympy as sp


class NodeIkSolutionGeneratorCpp(SkeletonNodeVisitor):

    def __init__(self, tree: SkeletonTree, use_safe_operator: bool = False):
        super().__init__()
        self._use_safe_operator = use_safe_operator
        self._tree = tree
        self._n_nodes = tree.num_nodes

        # From symbol to variable index
        self._symbol_to_index: Dict[sp.Symbol, int] = dict()
        self._index_to_symbol: Dict[int, sp.Symbol] = dict()
        self._all_variables_in_tree = tree.all_solved_symbol()

        # Build the map
        all_solved_symbol_name_list = [symbol_i.name for symbol_i in self._all_variables_in_tree]
        all_solved_symbol_name_list = sorted(all_solved_symbol_name_list)
        for i in range(len(all_solved_symbol_name_list)):
            self._index_to_symbol[i] = sp.Symbol(all_solved_symbol_name_list[i])
            self._symbol_to_index[sp.Symbol(all_solved_symbol_name_list[i])] = i

        self._code_lines: List[str] = list()
        self._scope_manager: ScopeManager = ScopeManager(self._code_lines)

    def add_code_line(self, code_line: str):
        self._scope_manager.append_code_line(code_line)

    def add_empty_line(self):
        self.add_code_line('')

    def enter_scope(self, scope_name: str):
        self._scope_manager.enter_scope(scope_name)

    def exit_scope(self, scope_name: Optional[str], with_semicolon: bool = False):
        self._scope_manager.exit_scope(scope_name, with_semicolon)

    @property
    def indent_level(self) -> int:
        return self._scope_manager.indent_level

    @property
    def generated_code_lines(self):
        return self._code_lines

    @property
    def symbol_to_index(self):
        return self._symbol_to_index

    def append_init_lines(self):
        self.add_empty_line()
        self.add_code_line('solution_queue.reset();')
        self.add_code_line('node_index_workspace.reset(n_tree_nodes);')

        # Functor to make intermediate solution
        self.add_empty_line()
        self.add_code_line('using RawSolution = IntermediateSolution<intermediate_solution_size>;')
        self.add_code_line('auto make_raw_solution = []() -> RawSolution { return {}; };')

        # Functor to add solution to queue
        self.add_empty_line()
        self.add_code_line('auto append_solution_to_queue = [&solution_queue](RawSolution solution_2_add) -> int {')
        self.add_code_line(indent() + 'return solution_queue.push_solution(solution_2_add);')
        self.add_code_line('};')

        # Functor to add index to workspace
        self.add_empty_line()
        self.add_code_line('auto add_input_index_to = [&node_index_workspace](int node_idx, int solution_idx) -> void {')
        self.add_code_line(indent() + 'if (solution_idx < 0) return;')
        self.add_code_line(indent() + 'node_index_workspace.append_index_to_node(node_idx, solution_idx);')
        self.add_code_line('};')
        assert self._scope_manager.indent_level == 0

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
        self.add_empty_line()
        self.add_code_line('// Collect the output')
        self.add_code_line('for(int i = 0; i < solution_queue.size(); i++)')
        self.enter_scope('for_all_queue')
        self.add_code_line('if(!solution_queue.solutions_validity[i])')
        self.add_code_line(indent() + 'continue;')
        self.add_code_line('const auto& raw_ik_out_i = solution_queue.get_solution(i);')
        self.add_code_line('std::array<double, robot_nq> new_ik_i;')
        # Collect the variable corresponds to unknowns
        for i in range(len(robot.unknowns)):
            unknown_i = robot.unknowns[i]
            if unknown_i.symbol in solved_symbols:
                solved_symbol_i = unknown_i.symbol
                self.add_code_line('const double value_at_{idx} = raw_ik_out_i[{out_symbol_idx}];  // {symbol}'.
                                   format(idx=i,
                                          out_symbol_idx=solved_symbol_to_index[solved_symbol_i],
                                          symbol=solved_symbol_i.name))
            else:
                # This should be a parameter
                assert unknown_i.symbol in free_symbols
                self.add_code_line('const double value_at_{idx} = {symbol};  // {symbol}'.
                                   format(idx=i, symbol=unknown_i.symbol.name))
            self.add_code_line('new_ik_i[{idx}] = value_at_{idx};'.format(idx=i))
        self.add_code_line('ik_output.push_back(new_ik_i);')
        self.exit_scope('for_all_queue')

    def visit_no_branch_dispatcher(self, dispatcher: DispatcherNodeBase):
        # Input/and output in workspace
        input_idx = dispatcher.flatten_idx
        assert dispatcher.non_degenerate_child is not None
        output_idx = dispatcher.non_degenerate_child.flatten_idx

        # Node code
        self.add_code_line('')
        self.add_code_line('// Code for non-branch dispatcher node {idx}'.format(idx=input_idx))
        self.add_code_line('// Actually, there is no code')

    def visit_equation_all_zero_dispatcher(self, dispatcher: EquationAllZeroDispatcherNode):
        # Shared code, extract function headers
        input_idx = dispatcher.flatten_idx
        assert dispatcher.non_degenerate_child is not None

        # The code
        self.add_code_line('')
        self.add_code_line('// Code for equation all-zero dispatcher node {idx}'.format(idx=input_idx))

        # Depends on root or not
        if dispatcher.is_root:
            self._visit_equation_all_zero_dispatcher_root(dispatcher)
        else:
            self._visit_equation_all_zero_dispatcher_non_root(dispatcher)

        # Call the processor and finish
        self.add_empty_line()
        self.add_code_line('// Invoke the processor')
        self.add_code_line(self.node_processor_name(dispatcher) + '();')
        self.add_code_line('// Finish code for equation all-zero dispatcher node {idx}'.format(idx=input_idx))

    def _visit_equation_all_zero_dispatcher_root(self, dispatcher: EquationAllZeroDispatcherNode):
        assert self.indent_level == 0
        checker_i = dispatcher.degenerate_checker
        checker_i_code = self.generate_cxx_code(checker_i)
        non_degenerate_child_idx = dispatcher.non_degenerate_child.flatten_idx
        degenerate_child_idx = None if dispatcher.degenerate_child is None else dispatcher.degenerate_child.flatten_idx

        # We do not need to expand the variable as this is the root
        self.add_code_line('auto ' + self.node_processor_name(dispatcher) + '= [&]()')
        self.enter_scope('checker_processor')
        self.add_code_line('const bool checked_result = {check_code};'.format(check_code=checker_i_code))
        self.add_code_line('if (!checked_result)  // To non-degenerate node')
        self.add_code_line(indent() + 'node_index_workspace.node_input_validity_vector[{out_idx}] = true;'.
                           format(out_idx=non_degenerate_child_idx))
        if degenerate_child_idx is not None:
            self.add_code_line('else')
            self.add_code_line(indent() + 'node_index_workspace.node_input_validity_vector[{out_idx}] = true;'.
                               format(out_idx=degenerate_child_idx))
        self.exit_scope('checker_processor', True)

    def _visit_equation_all_zero_dispatcher_non_root(self, dispatcher: EquationAllZeroDispatcherNode):
        assert self.indent_level == 0
        # Code is in indent 1
        checker_i = dispatcher.degenerate_checker
        checker_i_code = self.generate_cxx_code(checker_i)
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
        self.add_code_line('for(int i = 0; i < this_node_input_index.size(); i++)')
        self.enter_scope('for_i')
        self.add_code_line('int node_input_i_idx_in_queue = this_node_input_index[i];')
        self.add_code_line('if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))')
        self.add_code_line(indent() + 'continue;')
        self.add_code_line('const auto& {sol_name} = solution_queue.get_solution(node_input_i_idx_in_queue);'.format(
            sol_name=solution_name))
        for this_line in var_extract_lines:
            self.add_code_line(this_line)

        # Do checking and sent to non-degenerate child
        self.add_code_line('const bool checked_result = {check_code};'.format(check_code=checker_i_code))
        self.add_code_line('if (!checked_result)  // To non-degenerate node')
        self.add_code_line(indent() + 'add_input_index_to({node_idx}, node_input_i_idx_in_queue);'.
                           format(node_idx=non_degenerate_child_idx))

        # Generate degenerate child idx
        if degenerate_child_idx is not None:
            self.add_code_line('else')
            self.add_code_line(indent() + 'add_input_index_to({node_idx}, node_input_i_idx_in_queue);'.
                               format(node_idx=degenerate_child_idx))
        self.exit_scope('for_i')
        self.exit_scope(None, True)  # The processor scope

    def visit_solved_variable_dispatcher(self, dispatcher: SolvedVariableDispatcherNode):
        # Shared code, extract function headers
        input_idx = dispatcher.flatten_idx
        assert dispatcher.non_degenerate_child is not None

        # The code
        self.add_empty_line()
        self.add_code_line('// Code for solved_variable dispatcher node {idx}'.format(idx=input_idx))
        self._generate_processor_function_define(dispatcher, input_idx)

        # Depends on root or not
        if dispatcher.is_root:
            self._visit_solved_variable_dispatcher_root(dispatcher)
        else:
            self._visit_solved_variable_dispatcher_non_root(dispatcher)

        # Exit the checker
        self.exit_scope(None, True)
        assert self.indent_level == 0
        self.add_empty_line()
        self.add_code_line('// Invoke the processor')
        self.add_code_line(self.node_processor_name(dispatcher) + '();')
        self.add_code_line('// Finish code for solved_variable dispatcher node {idx}'.format(idx=input_idx))

    def _visit_solved_variable_dispatcher_root(self, dispatcher: SolvedVariableDispatcherNode):
        assert self.indent_level == 1
        raise NotImplementedError

    def _visit_solved_variable_dispatcher_non_root(self, dispatcher: SolvedVariableDispatcherNode):
        assert self.indent_level == 1
        n_child = len(dispatcher.branch_conditions)
        assert dispatcher.non_degenerate_child is not None
        non_degenerate_child_idx = dispatcher.non_degenerate_child.flatten_idx
        self.add_empty_line()

        solution_name = 'this_solution'
        self.add_code_line('for(int i = 0; i < this_node_input_index.size(); i++)')
        self.enter_scope('for_i')
        self.add_code_line('int node_input_i_idx_in_queue = this_node_input_index[i];')
        self.add_code_line('if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))')
        self.add_code_line(indent() + 'continue;')
        self.add_code_line('const auto& {sol_name} = solution_queue.get_solution(node_input_i_idx_in_queue);'.format(
            sol_name=solution_name))
        self.add_code_line('bool taken_by_degenerate = false;')

        # The case of taken by degenerated child
        for i in range(n_child):
            child_i = dispatcher.degenerate_branch_child(i)
            child_i_idx = None if child_i is None else child_i.flatten_idx
            condition_i = dispatcher.branch_conditions[i]
            condition_i_code = self.generate_cxx_code(condition_i)
            requested_vars_i = self.get_requested_solved_variables(None, condition_i, dispatcher)
            extract_lines = self.extract_solved_variable_codes(requested_vars_i, solution_name=solution_name)

            # Extract the solved variable only once
            if i == 0:
                for this_line in extract_lines:
                    self.add_code_line(this_line)

            # For this solution
            self.add_empty_line()
            self.add_code_line('const bool degenerate_valid_{idx} = {code};'.format(idx=i, code=condition_i_code))
            self.add_code_line('if (degenerate_valid_{idx})'.format(idx=i))
            self.enter_scope('if_degenerate')
            self.add_code_line('taken_by_degenerate = true;')
            if child_i_idx is not None:
                self.add_code_line('add_input_index_to({node_idx}, node_input_i_idx_in_queue);'.
                                   format(node_idx=child_i_idx))
            self.exit_scope('if_degenerate')

        # The case of non-degenerate
        self.add_empty_line()
        self.add_code_line('if (!taken_by_degenerate)')
        self.add_code_line(indent() + 'add_input_index_to({node_idx}, node_input_i_idx_in_queue);'.
                           format(node_idx=non_degenerate_child_idx))

        # For loop finished
        self.exit_scope('for_i')

    def visit_explicit_solution_node(self, solution_node: ExplicitSolutionNode):
        # The shared parts
        parent_node = solution_node.parent
        assert parent_node is not None
        if isinstance(parent_node, NoBranchDispatcherNode):
            input_idx = parent_node.flatten_idx
        else:
            input_idx = solution_node.flatten_idx

        self.add_empty_line()
        self.add_code_line('// Code for explicit solution node {idx}, solved variable is {var}'.
                           format(idx=solution_node.flatten_idx, var=solution_node.solution.solved_variable.name))
        self._generate_processor_function_define(solution_node, input_idx)
        assert self._scope_manager.indent_level == 1

        # Whether we need to extract value depends on the root
        if solution_node.is_parent_root_dispatcher():
            self._visit_explicit_solution_node_root(solution_node)
        else:
            self._visit_explicit_solution_node_non_root(solution_node)

        # Generation done for this node
        self.exit_scope(None, True)

        # Call the processor and finish
        self.add_code_line('// Invoke the processor')
        self.add_code_line(self.node_processor_name(solution_node) + '();')
        self.add_code_line('// Finish code for explicit solution node {idx}'.format(idx=input_idx))
        assert self.indent_level == 0

    def _visit_explicit_solution_node_root(self, solution_node: ExplicitSolutionNode):
        assert self.indent_level == 1
        output_idx = None if solution_node.children[0] is None else solution_node.children[0].flatten_idx
        self.add_code_line('')
        self.add_code_line('// The explicit solution of root node')
        solution = solution_node.solution
        assert solution.is_explicit_solution
        solved_var = solution.solved_variable
        solved_var_idx = self._symbol_to_index[solved_var]

        # The cse
        n_solution = len(solution.explicit_solutions)
        solution_list: List[sp.Expr] = list()
        for i in range(n_solution):
            solution_i = solution.explicit_solutions[i]
            checker_i = solution.argument_validity_checkers[i]
            solution_list.append(solution_i)

        # Do cse for solution/checker
        replacements, reduced_solutions = sp.cse(solution_list)

        # The code gen
        for i in range(n_solution):
            # The reduced code
            reduced_solution_i = reduced_solutions[i]
            checker_i = solution.argument_validity_checkers[i]
            solution_i_code = self.generate_cxx_code(reduced_solution_i)
            checker_i_code = self.generate_cxx_code(checker_i)
            self.add_code_line('const bool condition_{sol_idx} = {cond_code};'.
                               format(sol_idx=i, cond_code=checker_i_code))
            self.add_code_line('if (condition_{sol_idx})'.format(sol_idx=i))
            self.enter_scope('if_condition')
            self.add_code_line('// Temp variable for efficiency')
            for k in range(len(replacements)):
                symbol_k, expr_k = replacements[k]
                expr_code = self.generate_cxx_code(expr_k)
                self.add_code_line('const double {var} = {expr_code};'.format(var=str(symbol_k), expr_code=expr_code))
            self.add_code_line('// End of temp variables')

            # Actual solution
            self.add_empty_line()
            self.add_code_line('auto solution_{sol_idx} = make_raw_solution();'.format(sol_idx=i))
            self.add_code_line('solution_{sol_idx}[{var_idx}] = {sol_code};'.
                               format(sol_idx=i, var_idx=solved_var_idx, sol_code=solution_i_code))
            self.add_code_line('int appended_idx = append_solution_to_queue(solution_{sol_idx});'.format(sol_idx=i))
            if output_idx is not None:
                self.add_code_line('add_input_index_to({node_idx}, appended_idx);'.format(node_idx=output_idx))
            self.exit_scope('if_condition')
            self.add_empty_line()

    def _visit_explicit_solution_node_non_root(self, solution_node: ExplicitSolutionNode):
        assert self.indent_level == 1
        output_idx = None if solution_node.children[0] is None else solution_node.children[0].flatten_idx
        self.add_code_line('')
        self.add_code_line('// The solution of non-root node {idx}'.format(idx=solution_node.flatten_idx))
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
        self.add_code_line('for(int i = 0; i < this_node_input_index.size(); i++)')
        self.enter_scope('for_i')
        self.add_code_line('int node_input_i_idx_in_queue = this_node_input_index[i];')
        self.add_code_line('if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))')
        self.add_code_line(indent() + 'continue;')
        self.add_code_line('const auto& {sol_name} = solution_queue.get_solution(node_input_i_idx_in_queue);'.format(sol_name=solution_name))
        for this_line in var_extract_lines:
            self.add_code_line(this_line)

        # For each solution
        for i in range(n_solution):
            # The reduced code
            reduced_solution_i = reduced_solutions[i]
            checker_i = checker_list[i]
            solution_i_code = self.generate_cxx_code(reduced_solution_i)
            checker_i_code = self.generate_cxx_code(checker_i)

            # Extraction done, do solving
            self.add_code_line('')
            self.add_code_line('const bool condition_{sol_idx} = {cond_code};'.
                               format(sol_idx=i, cond_code=checker_i_code))
            self.add_code_line('if (condition_{sol_idx})'.format(sol_idx=i))
            self.enter_scope('if_condition')

            # The replacement code, we can only place it here as it may contain dangerous code
            self.add_code_line('// Temp variable for efficiency')
            for k in range(len(solution_replacements)):
                symbol_k, expr_k = solution_replacements[k]
                expr_code = self.generate_cxx_code(expr_k)
                self.add_code_line('const double {var} = {expr_code};'.format(var=str(symbol_k), expr_code=expr_code))
            self.add_code_line('// End of temp variables')

            # The actual solve code
            if i != n_solution - 1:
                self.add_code_line('RawSolution solution_{sol_idx}({sol_name});'.
                                   format(sol_idx=i, sol_name=solution_name))
                self.add_code_line('solution_{sol_idx}[{var_idx}] = {sol_code};'.
                                   format(sol_idx=i, var_idx=solved_var_idx, sol_code=solution_i_code))
                self.add_code_line('int appended_idx = append_solution_to_queue(solution_{sol_idx});'.format(sol_idx=i))
                if output_idx is not None:
                    self.add_code_line('add_input_index_to({node_idx}, appended_idx);'.format(node_idx=output_idx))
            else:
                # Directly assign to this_solution
                self.add_code_line('const double tmp_sol_value = {sol_code};'.
                                   format(sol_code=solution_i_code))
                self.add_code_line('solution_queue.get_solution(node_input_i_idx_in_queue)[{var_idx}] = tmp_sol_value;'.
                                   format(var_idx=solved_var_idx))
                if output_idx is not None:
                    self.add_code_line('add_input_index_to({node_idx}, node_input_i_idx_in_queue);'.
                                       format(node_idx=output_idx))
            self.exit_scope('if_condition')

            # Set validity for the last idx
            if i == n_solution - 1:
                self.add_code_line('else')
                self.enter_scope('else')
                self.add_code_line('solution_queue.solutions_validity[node_input_i_idx_in_queue] = false;')
                self.exit_scope('else')

        self.exit_scope('for_i')

    def visit_polynomial_solution_node(self, solution_node: PolynomialSolutionNode):
        # The shared parts
        parent_node = solution_node.parent
        assert parent_node is not None
        if isinstance(parent_node, NoBranchDispatcherNode):
            input_idx = parent_node.flatten_idx
        else:
            input_idx = solution_node.flatten_idx

        self.add_code_line('')
        self.add_code_line('// Code for polynomial solution node {idx}, solved variable is {var}'.
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
        self.add_code_line('// Invoke the processor')
        self.add_code_line(self.node_processor_name(solution_node) + '();')
        self.add_code_line('// Finish code for polynomial solution node {idx}'.format(idx=input_idx))

    def _visit_polynomial_solution_node_root(self, solution_node: PolynomialSolutionNode):
        assert self.indent_level == 1
        output_idx = None if solution_node.children[0] is None else solution_node.children[0].flatten_idx
        self.add_code_line('')
        self.add_code_line('// The polynomial solution of root node')
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
            numerator_code = self.generate_cxx_code(numerator)
            denominator_code = self.generate_cxx_code(denominator)
            self.add_code_line('const double poly_coefficient_{order}_num = {code};'.
                               format(order=order, code=numerator_code))
            self.add_code_line('const double poly_coefficient_{order}_denom = {code};'.format(order=order, code=denominator_code))
            self.add_code_line('const double poly_coefficient_{order} = poly_'
                               'coefficient_{order}_num / poly_coefficient_{order}_denom;'.format(order=order))
            if order > poly_order:
                poly_order = order

        # Make the coefficients
        assert poly_order >= 1
        self.add_code_line('std::array<double, {poly_order} + 1> p_coefficients;'.format(poly_order=poly_order))
        for order in poly_dict_input:
            self.add_code_line('p_coefficients[{idx}] = poly_coefficient_{order};'.
                               format(order=order, idx=poly_order - order))

        self.add_empty_line()
        self.add_code_line('// Invoke the solver. Note that p_coefficient[0] is the highest order')
        self.add_code_line('const auto poly_roots = computePolynomialRealRoots<{poly_order}>(p_coefficients);'.
                           format(poly_order=poly_order))

        # Collect the result
        self.add_empty_line()
        self.add_code_line('// Result collection')
        self.add_code_line('for(int root_idx = 0; root_idx < poly_roots.size(); root_idx++)')
        self.enter_scope('for_root')
        self.add_code_line('const auto& this_root_record = poly_roots[root_idx];')
        self.add_code_line('if(!this_root_record.is_valid)')
        self.add_code_line(indent() + 'continue;')
        self.add_code_line('const double this_root = this_root_record.value;')

        # Depends on the solution method
        solution_method = solution.solution_method
        if solution_method == SolutionMethod.PolynomialSin.name or solution_method == SolutionMethod.PolynomialCos.name:
            self.add_code_line('if (std::abs(this_root) > 1)')
            self.add_code_line(indent() + 'continue;')
            if solution_method == SolutionMethod.PolynomialSin.name:
                self.add_code_line('const double first_angle = std::asin(this_root);')
                self.add_code_line('const double second_angle = M_PI - std::asin(this_root);')
            else:
                self.add_code_line('const double first_angle = std::acos(this_root);')
                self.add_code_line('const double second_angle = - std::acos(this_root);')

            # The remaining code is the same
            self.add_code_line('auto solution_{sol_idx} = make_raw_solution();'.format(sol_idx=0))
            self.add_code_line('solution_{sol_idx}[{var_idx}] = first_angle;'.
                               format(sol_idx=0, var_idx=solved_var_idx))
            self.add_code_line('auto solution_{sol_idx} = make_raw_solution();'.format(sol_idx=1))
            self.add_code_line('solution_{sol_idx}[{var_idx}] = second_angle;'.
                               format(sol_idx=1, var_idx=solved_var_idx))
            self.add_code_line('int appended_idx_{sol_idx} = append_solution_to_queue(solution_{sol_idx});'.
                               format(sol_idx=0))
            self.add_code_line('int appended_idx_{sol_idx} = append_solution_to_queue(solution_{sol_idx});'.
                               format(sol_idx=1))
            if output_idx is not None:
                self.add_code_line('add_input_index_to({node_idx}, appended_idx_0);'.format(node_idx=output_idx))
                self.add_code_line('add_input_index_to({node_idx}, appended_idx_1);'.format(node_idx=output_idx))
        else:
            raise NotImplementedError("Polynomial solver codegen with non sin/cos intermediate is not implemented yet")


        # Finish the polynomial solve
        self.exit_scope('for_root')
        self.exit_scope(None, True)

    def visit_linear_type2_solution_node(self, solution_node: LinearSinCosType_2_SolverNode):
        # The shared parts
        parent_node = solution_node.parent
        assert parent_node is not None
        if isinstance(parent_node, NoBranchDispatcherNode):
            input_idx = parent_node.flatten_idx
        else:
            input_idx = solution_node.flatten_idx

        self.add_empty_line()
        self.add_code_line('// Code for linear type2 solution node {idx}, solved variable is {var}'.
                           format(idx=solution_node.flatten_idx, var=solution_node.solution.solved_variable.name))
        self._generate_processor_function_define(solution_node, input_idx)
        assert self._scope_manager.indent_level == 1

        if solution_node.is_parent_root_dispatcher():
            raise NotImplementedError
        else:
            self._visit_linear_type2_solution_non_root(solution_node)

        # Generation done for this node
        self.exit_scope(None, True)

        # Call the processor and finish
        self.add_code_line('// Invoke the processor')
        self.add_code_line(self.node_processor_name(solution_node) + '();')
        self.add_code_line('// Finish code for general_6dof solution node {idx}'.format(idx=input_idx))
        assert self.indent_level == 0

    def _visit_linear_type2_solution_non_root(self, solution_node: LinearSinCosType_2_SolverNode):
        assert self.indent_level == 1
        output_idx = None if solution_node.children[0] is None else solution_node.children[0].flatten_idx
        self.add_code_line('')
        self.add_code_line('// The solution of non-root node {idx}'.format(idx=solution_node.flatten_idx))
        solution = solution_node.solution
        solved_var = solution.solved_variable
        solved_var_idx = self._symbol_to_index[solved_var]

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
        self.add_code_line('for(int i = 0; i < this_node_input_index.size(); i++)')
        self.enter_scope('for_i')
        self.add_code_line('int node_input_i_idx_in_queue = this_node_input_index[i];')
        self.add_code_line('if (!solution_queue.is_solution_valid(node_input_i_idx_in_queue))')
        self.add_code_line(indent() + 'continue;')
        self.add_code_line('const auto& {sol_name} = solution_queue.get_solution(node_input_i_idx_in_queue);'.format(
            sol_name=solution_name))
        for this_line in var_extract_lines:
            self.add_code_line(this_line)

        # The replacement code, we can only place it here as it may contain dangerous code
        self.add_code_line('// Temp variable for efficiency')
        for k in range(len(solution_replacements)):
            symbol_k, expr_k = solution_replacements[k]
            expr_code = self.generate_cxx_code(expr_k)
            self.add_code_line('const double {var} = {expr_code};'.format(var=str(symbol_k), expr_code=expr_code))
        self.add_code_line('// End of temp variables')

        # The matrix
        self.add_code_line('Eigen::Matrix<double, {rows}, 4> A_matrix;'.format(rows=A_matrix.rows))
        self.add_code_line('A_matrix.setZero();')
        offset = 0
        for r in range(A_matrix.rows):
            for c in range(A_matrix.cols):
                rc_expr: sp.Expr = reduced_A_matrix_flat[offset]
                if rc_expr != sp.S.Zero:
                    rc_expr_code = self.generate_cxx_code(rc_expr)
                    self.add_code_line(f'A_matrix({r}, {c}) = {rc_expr_code};')
                offset += 1

        # The solve process
        self.add_code_line('// Code for solving')
        for j in range(len(rows_to_try)):
            rows_j = rows_to_try[j]
            r0, r1, r2 = rows_j
            self.enter_scope('case_j')
            self.add_code_line('// Local variable for solving')
            self.add_code_line('double var_solution_0 = 0.0;')
            self.add_code_line('double var_solution_1 = 0.0;')
            self.add_code_line(f'bool solved = ::yaik_cpp::linear_solver::trySolveLinearType2SpecificRows<{A_matrix.rows}>(A_matrix, {r0}, {r1}, {r2}, var_solution_0, var_solution_1);')

            # If solution ok
            self.add_code_line('if(solved)')
            self.enter_scope('if_solved')
            # The first candidate solution
            self.add_code_line('RawSolution solution_0({sol_name});'.format(sol_name=solution_name))
            self.add_code_line('solution_0[{var_idx}] = var_solution_0;'.format(var_idx=solved_var_idx))
            self.add_code_line('int appended_idx = append_solution_to_queue(solution_0);')
            if output_idx is not None:
                self.add_code_line('add_input_index_to({node_idx}, appended_idx);'.format(node_idx=output_idx))

            # The second candidate solution
            self.add_code_line('solution_queue.get_solution(node_input_i_idx_in_queue)[{var_idx}] = var_solution_1;'.
                               format(var_idx=solved_var_idx))
            if output_idx is not None:
                self.add_code_line('add_input_index_to({node_idx}, node_input_i_idx_in_queue);'.
                                   format(node_idx=output_idx))

            # Note the 'continue' here
            self.add_code_line('continue;')
            self.exit_scope('if_solved')
            self.exit_scope('case_j')
        self.exit_scope('for_i')

    def visit_general_6dof_numerical_reduce_node(self, solution_node: General6DoFNumericalReduceSolutionNode):
        # The shared parts
        parent_node = solution_node.parent
        assert parent_node is not None
        if isinstance(parent_node, NoBranchDispatcherNode):
            input_idx = parent_node.flatten_idx
        else:
            input_idx = solution_node.flatten_idx

        self.add_empty_line()
        self.add_code_line('// Code for general_6dof solution node {idx}, solved variable is {var}'.
                           format(idx=solution_node.flatten_idx, var=solution_node.solution.solved_variable.name))
        self._generate_processor_function_define(solution_node, input_idx)
        assert self._scope_manager.indent_level == 1

        # Whether we need to extract value depends on the root
        if solution_node.is_parent_root_dispatcher():
            if solution_node.has_semi_symbolic_reduce:
                self._visit_general_6dof_semi_symbolic_reduce_node_root(solution_node)
            else:
                self._visit_general_6dof_numerical_reduce_node_root(solution_node)
        else:
            raise NotImplementedError('General 6-DoF solution is only supported at the root')

        # Generation done for this node
        self.exit_scope(None, True)

        # Call the processor and finish
        self.add_code_line('// Invoke the processor')
        self.add_code_line(self.node_processor_name(solution_node) + '();')
        self.add_code_line('// Finish code for general_6dof solution node {idx}'.format(idx=input_idx))
        assert self.indent_level == 0

    def _visit_general_6dof_semi_symbolic_reduce_node_root(self, solution_node: General6DoFNumericalReduceSolutionNode):
        assert self.indent_level == 1
        assert solution_node.has_semi_symbolic_reduce
        output_idx = None if solution_node.children[0] is None else solution_node.children[0].flatten_idx
        self.add_empty_line()
        self.add_code_line('// The general 6-dof solution of root node')
        raw_solution = solution_node.general_6dof_solution
        solution_impl = raw_solution.internal_solution
        assert solution_impl.is_general_6dof
        assert solution_impl.class_key() == VariableSolutionClassKey.General6DoFNumericalReduce.name
        general_6dof_solution: General6DoFNumericalReduceSolutionImpl = solution_impl
        semi_reduce = general_6dof_solution.semi_symbolic_reduce_record

        # Generate the R_l_matrix
        def generate_symbolic_matrix_code(sp_matrix: sp.Matrix, matrix_name: str):
            mat_rows, mat_cols = sp_matrix.shape[0], sp_matrix.shape[1]
            self.add_code_line(f'Eigen::Matrix<double, {mat_rows}, {mat_cols}> {matrix_name};')
            self.add_code_line(f'{matrix_name}.setZero();')
            for r in range(sp_matrix.shape[0]):
                for c in range(sp_matrix.shape[1]):
                    rc_expr = sp_matrix[r, c]
                    if rc_expr != sp.S.Zero:
                        rc_expr_code = self.generate_cxx_code(rc_expr)
                        self.add_code_line(f'{matrix_name}({r}, {c}) = {rc_expr_code};')

        # The R_l inverse
        R_l = semi_reduce.R_l
        generate_symbolic_matrix_code(R_l, 'R_l')
        self.add_code_line('Eigen::Matrix<double, {rows}, {cols}> R_l_mat_inv = '
                           'R_l.inverse();'.format(rows=R_l.shape[0], cols=R_l.shape[1]))
        self.add_code_line('for(auto r = 0; r < R_l_mat_inv.rows(); r++) {')
        self.add_code_line(indent() + 'for(auto c = 0; c < R_l_mat_inv.cols(); c++) {')
        self.add_code_line(indent(2) + 'if (std::isnan(R_l_mat_inv(r, c)) || (!std::isfinite(R_l_mat_inv(r, c)))) return;')
        self.add_code_line(indent() + '}')
        self.add_code_line('}')

        # Extract the variable
        self.add_empty_line()
        R_l_inv_symbols = semi_reduce.R_l_inv_as_symbols
        for r in range(R_l.shape[0]):
            for c in range(R_l.shape[1]):
                inv_Rl_symbol_rc = R_l_inv_symbols[r, c]
                self.add_code_line(f'const double {inv_Rl_symbol_rc.name} = R_l_mat_inv({r}, {c});')

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
        replacements, reduced_expressions = sp.cse(matrix_expressions)

        self.add_code_line('')
        self.add_code_line('// Temp variable for efficiency')
        for i in range(len(replacements)):
            symbol_i, expr_i = replacements[i]
            expr_code = self.generate_cxx_code(expr_i)
            self.add_code_line('const double {var} = {expr_code};'.format(var=str(symbol_i), expr_code=expr_code))

        def generate_symbolic_matrix_code(sp_matrix: sp.Matrix, matrix_name: str, counter_in: int) -> int:
            self.add_code_line(f'Eigen::Matrix<double, {sp_matrix.shape[0]}, {sp_matrix.shape[1]}> {matrix_name};')
            self.add_code_line(f'{matrix_name}.setZero();')
            local_counter = counter_in
            for r in range(sp_matrix.shape[0]):
                for c in range(sp_matrix.shape[1]):
                    rc_expr = reduced_expressions[local_counter]
                    local_counter += 1
                    if rc_expr != sp.S.Zero:
                        rc_expr_code = self.generate_cxx_code(rc_expr)
                        self.add_code_line(f'{matrix_name}({r}, {c}) = {rc_expr_code};')
            return local_counter

        # The codegen
        matrix_expression_reduced_counter = 0
        self.add_empty_line()
        matrix_expression_reduced_counter = \
            generate_symbolic_matrix_code(A, 'A', matrix_expression_reduced_counter)
        self.add_empty_line()
        matrix_expression_reduced_counter = \
            generate_symbolic_matrix_code(B, 'B', matrix_expression_reduced_counter)
        self.add_empty_line()
        generate_symbolic_matrix_code(C, 'C', matrix_expression_reduced_counter)

        # Invoke the method
        solved_var = solution_node.solved_variable
        solved_var_idx = self._symbol_to_index[solved_var]
        self.add_empty_line()
        self.add_code_line('// Invoke the solver')
        self.add_code_line('std::array<double, 16> solution_buffer;')
        self.add_code_line('int n_solutions = '
                           'yaik_cpp::general_6dof_internal::computeSolutionFromTanhalfLME'
                           '(A, B, C, &solution_buffer);')

        self.add_empty_line()
        self.add_code_line('for(auto i = 0; i < n_solutions; i++)')
        self.enter_scope('for_loop')
        self.add_code_line('auto solution_i = make_raw_solution();')
        self.add_code_line('solution_i[{var_idx}] = solution_buffer[i];'.format(var_idx=solved_var_idx))
        self.add_code_line('int appended_idx = append_solution_to_queue(solution_i);')
        if output_idx is not None:
            self.add_code_line('add_input_index_to({node_idx}, appended_idx);'.format(node_idx=output_idx))
        self.exit_scope('for_loop', True)

    def _visit_general_6dof_numerical_reduce_node_root(self, solution_node: General6DoFNumericalReduceSolutionNode):
        assert self.indent_level == 1
        output_idx = None if solution_node.children[0] is None else solution_node.children[0].flatten_idx
        self.add_empty_line()
        self.add_code_line('// The general 6-dof solution of root node')
        raw_solution = solution_node.general_6dof_solution
        solution_impl = raw_solution.internal_solution
        assert solution_impl.is_general_6dof
        assert solution_impl.class_key() == VariableSolutionClassKey.General6DoFNumericalReduce.name
        general_6dof_solution: General6DoFNumericalReduceSolutionImpl = solution_impl
        solved_var = solution_node.solved_variable
        solved_var_idx = self._symbol_to_index[solved_var]

        def generate_symbolic_matrix_code(sp_matrix: sp.Matrix, matrix_name: str):
            mat_rows, mat_cols = sp_matrix.shape[0], sp_matrix.shape[1]
            self.add_code_line(f'Eigen::Matrix<double, {mat_rows}, {mat_cols}> {matrix_name};')
            self.add_code_line(f'{matrix_name}.setZero();')
            for r in range(sp_matrix.shape[0]):
                for c in range(sp_matrix.shape[1]):
                    rc_expr = sp_matrix[r, c]
                    if rc_expr != sp.S.Zero:
                        rc_expr_code = self.generate_cxx_code(rc_expr)
                        self.add_code_line(f'{matrix_name}({r}, {c}) = {rc_expr_code};')

        # Generate the required matrix
        A_sin, A_cos, C_const = general_6dof_solution.lhs_matrices()
        N_rhs = general_6dof_solution.rhs_matrix()
        generate_symbolic_matrix_code(A_sin, 'A_sin')
        self.add_empty_line()
        generate_symbolic_matrix_code(A_cos, 'A_cos')
        self.add_empty_line()
        generate_symbolic_matrix_code(C_const, 'C_const')
        self.add_empty_line()
        generate_symbolic_matrix_code(N_rhs, 'N_rhs')

        # Generate lines to reduce
        n_total_lines = A_sin.shape[0]
        lines2reduce: List[int] = general_6dof_solution.select_lines
        remaining_lines: List[int] = list()
        for i in range(n_total_lines):
            if i in lines2reduce:
                continue
            else:
                remaining_lines.append(i)

        # Write the info to cpp
        def generate_std_array_of_int(list_of_int: List[int], array_name):
            int_list_code = f'const std::array<int, {len(list_of_int)}> {array_name}'
            int_list_code += '{'
            for i in range(len(list_of_int)):
                if i == len(list_of_int) - 1:
                    int_list_code += '{int_i}'.format(int_i=list_of_int[i])
                else:
                    int_list_code += '{int_i}, '.format(int_i=list_of_int[i])
            int_list_code += '};'
            self.add_code_line(int_list_code)

        self.add_empty_line()
        self.add_code_line('// The lines for reduction')
        generate_std_array_of_int(lines2reduce, 'lines_to_reduce')
        generate_std_array_of_int(remaining_lines, 'remaining_lines')

        # Invoke the method
        self.add_empty_line()
        self.add_code_line('// Invoke the solver')
        self.add_code_line('std::array<double, 16> solution_buffer;')
        self.add_code_line('int n_solutions = '
                           'yaik_cpp::general6DofNumericalReduceSolve<{n_equation}, {n_lhs_unknowns}, {n_rhs_unknowns}>'
                           '(A_sin, A_cos, C_const, N_rhs, lines_to_reduce, remaining_lines, &solution_buffer);'.
                           format(n_equation=A_sin.shape[0], n_lhs_unknowns=A_sin.shape[1], n_rhs_unknowns=N_rhs.shape[1]))

        self.add_empty_line()
        self.add_code_line('for(auto i = 0; i < n_solutions; i++)')
        self.enter_scope('for_loop')
        self.add_code_line('auto solution_i = make_raw_solution();')
        self.add_code_line('solution_i[{var_idx}] = solution_buffer[i];'.format(var_idx=solved_var_idx))
        self.add_code_line('int appended_idx = append_solution_to_queue(solution_i);')
        if output_idx is not None:
            self.add_code_line('add_input_index_to({node_idx}, appended_idx);'.format(node_idx=output_idx))
        self.exit_scope('for_loop', True)

    def _generate_processor_function_define(self, node: SkeletonTreeNode, input_idx: int):
        node_processor_name = self.node_processor_name(node)
        self.add_code_line('auto ' + self.node_processor_name(node) + ' = [&]() -> void')
        self.enter_scope(node_processor_name)
        self.add_code_line('const auto& this_node_input_index = node_index_workspace.get'
                           '_input_indices_for_node({in_idx});'.format(in_idx=input_idx))
        self.add_code_line('const bool this_input_valid = node_index_workspace.is_input'
                           '_indices_valid_for_node({in_idx});'.
                           format(in_idx=input_idx))
        self.add_code_line('if (!this_input_valid)')
        self.add_code_line(indent() + 'return;')

    @staticmethod
    def node_processor_name(node: SkeletonTreeNode) -> str:
        type_name = type(node).__name__
        if node.is_dispatcher_node:
            return '{type_name}_node_{idx}_processor'.format(idx=node.flatten_idx, type_name=type_name)
        # Solution node, also indicate the solved variable
        solution_node: SolutionNodeBase = node
        solved_var_name: str = solution_node.solution.solved_variable.name
        return '{type_name}_node_{idx}_solve_{var_name}_processor'.format(idx=node.flatten_idx, type_name=type_name,
                                                                          var_name=solved_var_name)

    def gather_request_solved_variables(
            self,
            expressions_in: List[sp.Expr],
            node: SkeletonTreeNode) -> List[sp.Symbol]:
        # First get the requested symbols
        requested_symbol_names_set: Set[str] = set()
        for expr_i in expressions_in:
            for symbol_j in expr_i.free_symbols:
                if symbol_j in self._all_variables_in_tree:
                    requested_symbol_names_set.add(symbol_j.name)

        # Sort the name
        requested_symbol_names = [elem for elem in requested_symbol_names_set]
        requested_symbol_names = sorted(requested_symbol_names)

        # Next check they are in parent
        self._verify_requested_variables_solved(requested_symbol_names, node)

        # OK
        return [sp.Symbol(symbol_name) for symbol_name in requested_symbol_names]

    def _verify_requested_variables_solved(
            self,
            requested_symbol_names: List[str],
            node: SkeletonTreeNode) -> bool:
        solutions_from_root = self._tree.variable_solutions_from_root(node, include_current_node=False)
        for symbol_i_name in requested_symbol_names:
            found = False
            for j in range(len(solutions_from_root)):
                solution_j = solutions_from_root[j]
                if solution_j.solved_variable.name == symbol_i_name:
                    found = True
                    break
            if not found:
                return False
        return True

    def extract_solved_variable_codes(
            self,
            requested_variables: List[sp.Symbol],
            solution_name: str = 'this_solution') -> List[str]:
        extraction_lines: List[str] = list()
        for var_i in requested_variables:
            var_i_index = self._symbol_to_index[var_i]
            this_line = 'const double {var_name} = {sol_name}[{idx}];'. \
                format(var_name=str(var_i), sol_name=solution_name, idx=var_i_index)
            extraction_lines.append(this_line)
        return extraction_lines

    def get_requested_solved_variables(
            self,
            solution_i: Optional[sp.Expr],
            checker_i: Optional[sp.Expr],
            node: SkeletonTreeNode) -> List[sp.Symbol]:
        """
        For a given solution, check all variables it requires are already solved, and return them
        """
        # First get the requested symbols
        requested_symbol_names_set: Set[str] = set()
        if solution_i is not None:
            for symbol_i in solution_i.free_symbols:
                if symbol_i in self._all_variables_in_tree:
                    requested_symbol_names_set.add(symbol_i.name)
        if checker_i is not None:
            for symbol_i in checker_i.free_symbols:
                if symbol_i in self._all_variables_in_tree:
                    requested_symbol_names_set.add(symbol_i.name)
        requested_symbol_names = [elem for elem in requested_symbol_names_set]
        requested_symbol_names = sorted(requested_symbol_names)

        # Next check they are in parent
        self._verify_requested_variables_solved(requested_symbol_names, node)

        # OK
        return [sp.Symbol(symbol_name) for symbol_name in requested_symbol_names]

    def free_symbols(self, include_unknown_as_parameter: bool = True) -> List[sp.Symbol]:
        """
        All symbols that must be provided by the caller
        except the ik target
        """
        return self._tree.get_free_symbols(include_unknown_as_parameter)

    def generate_cxx_code(self, expr_to_proc: sp.Expr) -> str:
        """
        Generate the cpp code for a given expression. Should be mostly sp.cxxcode, but we might need some operations
        """
        if not self._use_safe_operator:
            return sp.cxxcode(expr_to_proc)

        # The function should be replaced
        function_replacements = {
            'std::sqrt': 'safe_sqrt',
            'std::asin': 'safe_asin',
            'std::acos': 'safe_acos'
        }
        original_code = sp.cxxcode(expr_to_proc)
        processed_code = original_code
        for func in function_replacements:
            replaced = function_replacements[func]
            processed_code = processed_code.replace(func, replaced)
        return processed_code
