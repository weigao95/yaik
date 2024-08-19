from solver.solved_variable import VariableSolution, SolutionDegenerateRecord, DegenerateType
from codegen.skeleton.tree_node import SkeletonTreeNode, DispatcherNodeBase, SolvedVariableDispatcherNode
from codegen.skeleton.tree_node import SolutionNodeBase
from codegen.skeleton.tree_node import EquationAllZeroDispatcherNode, NoBranchDispatcherNode, SkeletonTreeNodeType
from codegen.skeleton.tree_node import ExplicitSolutionNode, SkeletonNodeVisitor, VariableSolutionClassKey
from codegen.skeleton.tree_node import PolynomialSolutionNode, General6DoFNumericalReduceSolutionNode
from codegen.skeleton.tree_node import LinearSinCosType_2_SolverNode
from solver.equation_utils import find_unknown_idx
from fk.robots import RobotDescription
from solver.build_equations import IkEquationCollection
from typing import Optional, List, Set
import sympy as sp


class SkeletonTree(object):

    def __init__(self,
                 main_branch_solution: List[VariableSolution],
                 robot: RobotDescription,
                 solved_equations: IkEquationCollection):
        # Init the variable
        self._robot = robot
        # TODO(wei): consider do we need to implement a separate serializer for
        #            solved_equations, seems it is one-to-one mapping from robot now
        self._solved_equations = solved_equations
        self._root_dispatcher: Optional[DispatcherNodeBase] = None
        self._node_list: List[SkeletonTreeNode] = list()

        # Build the 'tree' of the main branch
        solution: Optional[SkeletonTreeNode] = None
        for i in range(len(main_branch_solution)):
            new_dispatcher, new_solution = self.append_non_degenerate_solution(main_branch_solution[i], solution)
            self.add_node(new_dispatcher)
            self.add_node(new_solution)

            # Init the root
            if self._root_dispatcher is None:
                self._root_dispatcher = new_dispatcher

            # Update the iteration variable
            solution = new_solution

    @property
    def root(self):
        return self._root_dispatcher

    @property
    def robot(self):
        return self._robot

    @property
    def solved_unknowns(self):
        return self._solved_equations.used_unknowns

    @property
    def solved_equation_record(self):
        return self._solved_equations

    @property
    def node_list(self):
        return self._node_list

    def has_general_6dof_solution_node(self):
        for node in self.node_list:
            if node.is_solution_node:
                solution_node: SolutionNodeBase = node
                if solution_node.solution.is_general_6dof_solution:
                    return True
        return False

    @property
    def num_nodes(self):
        return len(self._node_list)

    def add_node(self, node: SkeletonTreeNode):
        node_idx = len(self._node_list)
        self._node_list.append(node)
        self._node_list[node_idx].set_flatten_idx(node_idx)

    def preorder_visit(self, visitor: SkeletonNodeVisitor):
        """
        Visit the root first, then all its valid children
        """
        node_stack: List[SkeletonTreeNode] = list()
        node_stack.append(self.root)
        while len(node_stack) > 0:
            # Visit the element
            this_node = node_stack.pop()
            this_node.accept(visitor)

            # Append all children to stack
            for i in range(len(this_node.children)):
                child_i = this_node.children[i]
                if child_i is not None:
                    node_stack.append(child_i)

    def postorder_visit(self, visitor: SkeletonNodeVisitor):
        """
        Visit all valid children first, then the root
        """
        self._postorder_recursive(self.root, visitor)

    def _postorder_recursive(self, root_node: SkeletonTreeNode, visitor: SkeletonNodeVisitor):
        """
        The recursive implementation of post-order visit.
        There exists more efficient solutions, but this is not the critical point.
        """
        for i in range(len(root_node.children)):
            child_i = root_node.children[i]
            if child_i is not None:
                self._postorder_recursive(child_i, visitor)

        # The root element
        root_node.accept(visitor)
        return

    def append_non_degenerate_solution(
            self,
            variable_solution: VariableSolution,
            parent: Optional[SolutionNodeBase] = None):
        # Make dispatcher and variable
        dispatcher_node = self.new_dispatcher_node(variable_solution.degenerate_record, parent)
        solution_node = self.new_solution_node(variable_solution)

        # Update the relationship
        dispatcher_node.set_non_degenerate_child(solution_node)
        solution_node.set_parent(dispatcher_node)

        # Return the node
        return dispatcher_node, solution_node

    @staticmethod
    def new_solution_node(variable_solution: VariableSolution):
        """
        Construct a solution node given the input solution
        """
        if variable_solution.is_explicit_solution:
            solution_node = ExplicitSolutionNode(variable_solution)
        elif variable_solution.is_polynomial:
            solution_node = PolynomialSolutionNode(variable_solution)
        elif variable_solution.internal_solution.class_key() == VariableSolutionClassKey.General6DoFNumericalReduce.name:
            solution_node = General6DoFNumericalReduceSolutionNode(variable_solution)
        elif variable_solution.internal_solution.class_key() == VariableSolutionClassKey.LinearSinCosType_2.name:
            solution_node = LinearSinCosType_2_SolverNode(variable_solution)
        else:
            raise NotImplementedError('Solution type not supported yet')

        # Should be ok here
        return solution_node

    def append_branched_solution(
            self,
            variable_solutions: List[VariableSolution],
            parent: DispatcherNodeBase,
            index_in_parent_children_list: int):
        """
        :param variable_solutions
        :param parent
        :param index_in_parent_children_list: note that this starts from 1 for degenerate children
        """
        assert parent is not None
        assert len(variable_solutions) > 0
        current_parent: Optional[SolutionNodeBase] = None
        for i in range(len(variable_solutions)):
            solution_i = variable_solutions[i]
            if i == 0:
                # Here we discard a dispatcher
                solution_node = self.new_solution_node(solution_i)
                solution_node.set_parent(parent)
                parent.set_child_by_index(solution_node, index_in_parent_children_list)
                current_parent = solution_node
                self.add_node(solution_node)
            else:
                assert current_parent is not None
                dispatcher_node = self.new_dispatcher_node(solution_i.degenerate_record, current_parent)
                solution_node = self.new_solution_node(solution_i)

                # Update the relationship
                dispatcher_node.set_non_degenerate_child(solution_node)
                solution_node.set_parent(dispatcher_node)
                self.add_node(dispatcher_node)
                self.add_node(solution_node)

                # Update the parent
                current_parent = solution_node

    def variable_solutions_from_root(
            self,
            node: SkeletonTreeNode,
            include_current_node: bool = True) -> List[VariableSolution]:
        """
        Given a node in the tree, return all variable solutions from root
        node to this node.
        :param node: the node to check
        :param include_current_node:
        """
        assert self.is_node_in_tree(node)
        current_node = node
        variable_solutions: List[VariableSolution] = list()
        counter = 0
        while current_node is not None:
            if isinstance(current_node, SolutionNodeBase):
                solution_node: SolutionNodeBase = current_node
                this_solution = solution_node.solution
                if include_current_node or counter >= 1:
                    variable_solutions.append(this_solution)

            # Update iterator
            counter += 1
            current_node = current_node.parent

        # The order need to be reversed
        variable_solutions.reverse()
        return variable_solutions

    def new_dispatcher_node(
            self,
            degenerate_record: SolutionDegenerateRecord,
            parent: Optional[SolutionNodeBase] = None):
        # Update the parent and return
        def set_parent_and_return(new_node: DispatcherNodeBase) -> DispatcherNodeBase:
            new_node.set_parent(parent)
            if parent is not None:
                assert parent.node_type == SkeletonTreeNodeType.SolutionNode.name
                parent.set_child(new_node)
            return new_node

        if degenerate_record.type == DegenerateType.DegenerateIfAllEquationsZero.name:
            this_node = EquationAllZeroDispatcherNode(degenerate_record)
            return set_parent_and_return(this_node)
        if degenerate_record.type == DegenerateType.DegenerateOnVariableValue.name:
            this_node = self._new_variable_dispatcher_node(degenerate_record, parent)
            return set_parent_and_return(this_node)

        # No-branch
        this_node = NoBranchDispatcherNode()
        return set_parent_and_return(this_node)

    def _new_variable_dispatcher_node(
            self,
            degenerate_record: SolutionDegenerateRecord,
            parent: Optional[SolutionNodeBase] = None):
        """
        If the record requires variable that is not solved, then it is actually an
        UnsolvedVariableDispatcherNode.
        We do NOT support half-solved variable. (eq: record depends on var_0 and var_1,
        var_0 is solved but var_1 is not)
        """
        if parent is None:
            return SolvedVariableDispatcherNode(degenerate_record)

        # Fetch the variables in record
        assert self.is_node_in_tree(parent)
        degenerate_variable: List[sp.Symbol] = list()
        for var in degenerate_record.variable_value_map:
            degenerate_variable.append(var)

        # Check whether a variable is solved before (and include) the parent node
        def is_variable_solved_before_parent(variable: sp.Symbol) -> bool:
            node_to_check = parent
            while node_to_check is not None:
                if node_to_check.is_solution_node:
                    solution_node: SolutionNodeBase = node_to_check
                    if solution_node.solved_variable == variable:
                        return True
                # Update the node
                node_to_check = node_to_check.parent

            # After the loop but find nothing
            return False

        # Get unsolved vars
        unsolved_variables: List[sp.Symbol] = list()
        for var in self._solved_equations.used_unknowns:  # change from robot._unknowns
            if not is_variable_solved_before_parent(var.symbol):
                unsolved_variables.append(var.symbol)

        # Whether the degenerate variable depends on un-solved ones
        dependent_on_unsolved = False
        for var in degenerate_variable:
            if var in unsolved_variables:
                dependent_on_unsolved = True

        # Make different node
        if dependent_on_unsolved:
            # Not implement yet
            raise NotImplementedError
        else:
            return SolvedVariableDispatcherNode(degenerate_record)

    def is_node_in_tree(self, node: SkeletonTreeNode) -> bool:
        """
        Check whether a given node is in tree, do not use index for more generalization
        """
        for i in range(len(self._node_list)):
            node_i = self._node_list[i]
            if node_i is node:
                return True
        return False

    def all_solved_symbol(self) -> Set[sp.Symbol]:
        """
        Return all the solved variable in a tree
        """
        symbol_set: Set[sp.Symbol] = set()
        for node_i in self.node_list:
            if node_i.is_solution_node:
                solution_node: SolutionNodeBase = node_i
                solution = solution_node.solution
                symbol_set.add(solution.solved_variable)
        return symbol_set

    def get_free_symbols(self, include_unknown_as_parameter: bool = True) -> List[sp.Symbol]:
        """
        All symbols that must be provided by the caller
        except the ik target
        """
        robot = self._robot
        all_symbol_list: List[sp.Symbol] = list()
        for symbol_i in robot.symbolic_parameters:
            if symbol_i not in robot.parameters_value:
                all_symbol_list.append(symbol_i)

        # If unknown is not required, we're done
        if not include_unknown_as_parameter:
            return all_symbol_list

        # Also the unknown
        ik_equations = self.solved_equation_record
        for unknown_i in robot.unknowns:
            idx = find_unknown_idx(ik_equations.used_unknowns, unknown_i.symbol.name)
            if idx < 0:
                all_symbol_list.append(unknown_i.symbol)

        # OK
        return all_symbol_list
