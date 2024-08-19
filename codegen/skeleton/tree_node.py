from abc import ABC
from typing import List, Optional, Dict
from enum import Enum
import sympy as sp
from solver.solved_variable import VariableSolution, SolutionDegenerateRecord, DegenerateType, VariableSolutionClassKey


class SkeletonTreeNodeType(Enum):
    DispatcherNode = 0,
    SolutionNode = 1


class SkeletonTreeNode(object):

    def __init__(self, node_type: str):
        # Dispatcher or solution
        self._node_type = node_type

        # The topology, would be updated after construction
        self._parent: Optional[SkeletonTreeNode] = None

        # The children should always be CORRECTLY SIZED after construction
        # Although the element might be None
        self._children: List[Optional[SkeletonTreeNode]] = list()

        # The flatten index in tree
        # Do NOT assume it is mid-order/pre-order/post-order traverse of the tree
        # tree.nodes[node.flatten_idx] == node
        # Again, would be updated after construction
        self._flatten_idx_in_tree = -1

    @property
    def node_type(self):
        return self._node_type

    @property
    def is_dispatcher_node(self):
        return self._node_type == SkeletonTreeNodeType.DispatcherNode.name

    @property
    def is_solution_node(self):
        return self._node_type == SkeletonTreeNodeType.SolutionNode.name

    @property
    def flatten_idx(self):
        return self._flatten_idx_in_tree

    @property
    def flatten_idx_in_tree(self):
        return self._flatten_idx_in_tree

    def set_flatten_idx(self, flatten_idx_in_tree: int):
        self._flatten_idx_in_tree = flatten_idx_in_tree

    @property
    def parent(self):
        return self._parent

    def set_parent(self, parent):
        self._parent = parent

    @property
    def is_root(self):
        return self.parent is None

    @property
    def children(self):
        return self._children

    def set_child_by_index(self, child, index_in_children_list: int):
        """
        Do not confuse this index with branch index in variable dispatcher node
        branch_idx + 1 == index_in_children_list
        """
        assert index_in_children_list < len(self.children)
        self._children[index_in_children_list] = child

    def accept(self, visitor):
        raise NotImplementedError


class SkeletonNodeVisitor(object):

    def __init__(self):
        pass

    def visit_no_branch_dispatcher(self, dispatcher: SkeletonTreeNode):
        raise NotImplementedError

    def visit_equation_all_zero_dispatcher(self, dispatcher: SkeletonTreeNode):
        raise NotImplementedError

    def visit_solved_variable_dispatcher(self, dispatcher: SkeletonTreeNode):
        raise NotImplementedError

    def visit_explicit_solution_node(self, solution_node: SkeletonTreeNode):
        raise NotImplementedError

    def visit_polynomial_solution_node(self, solution_node: SkeletonTreeNode):
        raise NotImplementedError

    def visit_general_6dof_numerical_reduce_node(self, solution_node: SkeletonTreeNode):
        raise NotImplementedError

    def visit_linear_type2_solution_node(self, solution_node: SkeletonTreeNode):
        raise NotImplementedError


class DispatcherNodeBase(SkeletonTreeNode, ABC):

    def __init__(self):
        super().__init__(SkeletonTreeNodeType.DispatcherNode.name)

    def set_non_degenerate_child(self, child: SkeletonTreeNode):
        """
        The first child is always the non-degenerate child
        """
        assert len(self.children) >= 1
        assert child.is_solution_node
        self._children[0] = child
        child.set_parent(self)

    @property
    def non_degenerate_child(self):
        return self.children[0]


class NoBranchDispatcherNode(DispatcherNodeBase):

    def __init__(self):
        super().__init__()

        # Only 1 child
        self._children.clear()
        self._children.append(None)

    def accept(self, visitor: SkeletonNodeVisitor):
        visitor.visit_no_branch_dispatcher(self)

    def set_child(self, child: SkeletonTreeNode):
        # Only one child
        assert len(self.children) == 1
        self.set_non_degenerate_child(child)


class EquationAllZeroDispatcherNode(DispatcherNodeBase):

    def __init__(self, degenerate_record: SolutionDegenerateRecord, check_zero_tolerance: float = 1e-6):
        super(EquationAllZeroDispatcherNode, self).__init__()
        self._degenerate_record = degenerate_record
        assert self._degenerate_record.type == DegenerateType.DegenerateIfAllEquationsZero.name
        self._degenerate_checker: sp.Expr = self._make_degenerate_predicate(
            degenerate_record, check_zero_tolerance
        )

        # Two child
        self._children.clear()
        self._children.append(None)
        self._children.append(None)

    @staticmethod
    def _make_degenerate_predicate(
            degenerate_record: SolutionDegenerateRecord,
            check_zero_tolerance: float) -> sp.Expr:
        """
        Init the checker, checker == True implies degenerate input
        """
        equations_to_check = degenerate_record.equations
        degenerate_checker = sp.S.BooleanTrue
        for equation in equations_to_check:
            this_equation_checker = sp.Abs(equation) <= check_zero_tolerance
            degenerate_checker = degenerate_checker & this_equation_checker
        return degenerate_checker

    @property
    def degenerate_checker(self) -> sp.Expr:
        return self._degenerate_checker

    @property
    def degenerate_record(self):
        return self._degenerate_record

    def set_degenerate_child(self, child: SkeletonTreeNode):
        assert len(self.children) == 2
        assert child.is_solution_node
        self.children[1] = child
        child.set_parent(self)

    @property
    def degenerate_child(self):
        assert len(self.children) == 2
        return self.children[1]

    def accept(self, visitor: SkeletonNodeVisitor):
        visitor.visit_equation_all_zero_dispatcher(self)


class SolvedVariableDispatcherNode(DispatcherNodeBase):

    def __init__(self, degenerate_record: SolutionDegenerateRecord, check_zero_tolerance: float = 1e-6):
        super().__init__()
        self._degenerate_record = degenerate_record
        assert self._degenerate_record.type == DegenerateType.DegenerateOnVariableValue.name

        # One for each degenerate branch
        # If everything is false, then on the main branch
        self._branch_conditions: List[sp.Expr] = list()
        self._branch_variable_values: List[Dict[sp.Symbol, sp.Expr]] = list()
        self._init_branch_conditions(check_zero_tolerance)

        # num_degenerate_branches + 1 children
        self._children.clear()
        for i in range(len(self._branch_conditions) + 1):
            self._children.append(None)

    def _init_branch_conditions(self, check_zero_tolerance: float):
        """
        Create the member self._branch_conditions and self._branch_variable_values.
        Note that we need check-zero tolerance because we cannot check x == 0 for double directly.
        """
        # Find the # of solutions
        variable_map: Dict[sp.Symbol, List[sp.Expr]] = self._degenerate_record.variable_value_map
        n_solutions = self._degenerate_record.count_number_variable_solutions()

        # Zero case
        if n_solutions is None or n_solutions == 0:
            return

        for solution_i in range(n_solutions):
            # Init the condition
            condition = sp.S.BooleanTrue
            for var in variable_map:
                var_solutions = variable_map[var]
                # Note that we use tolerance here instead of directly check ==
                condition_on_var = (sp.Abs(var - var_solutions[solution_i]) <= check_zero_tolerance)
                condition = condition_on_var & condition
            condition = condition.simplify()
            self._branch_conditions.append(condition)

            # Init the variable value
            variable_i: Dict[sp.Symbol, sp.Expr] = dict()
            for var in variable_map:
                var_solutions = variable_map[var]
                variable_i[var] = var_solutions[solution_i]
            self._branch_variable_values.append(variable_i)

    @property
    def degenerate_record(self):
        return self._degenerate_record

    @property
    def branch_conditions(self):
        return self._branch_conditions

    def get_branch_condition(self, branch_idx: int) -> sp.Expr:
        assert branch_idx < len(self._branch_conditions)
        return self._branch_conditions[branch_idx]

    def get_branch_variable_value(self, branch_idx: int) -> Dict[sp.Symbol, sp.Expr]:
        assert branch_idx < len(self._branch_variable_values)
        return self._branch_variable_values[branch_idx]

    @property
    def num_degenerate_branches(self):
        return len(self._branch_conditions)

    def set_branch_degenerate_child(self, branch_idx: int, child: SkeletonTreeNode):
        assert child.is_solution_node
        assert 0 <= branch_idx < len(self.branch_conditions)
        # Note this plus one, 0 is always the non-degenerate case
        self._children[branch_idx + 1] = child
        child.set_parent(self)

    def degenerate_branch_child(self, branch_idx: int) -> SkeletonTreeNode:
        assert 0 <= branch_idx < len(self.branch_conditions)
        return self.children[branch_idx + 1]

    def accept(self, visitor: SkeletonNodeVisitor):
        visitor.visit_solved_variable_dispatcher(self)


class SolutionNodeBase(SkeletonTreeNode, ABC):

    def __init__(self, solution: VariableSolution):
        super(SolutionNodeBase, self).__init__(SkeletonTreeNodeType.SolutionNode.name)

        # Update the variable
        self._solution = solution

        # Only one child, must be dispatcher, can be None
        self._children.clear()
        self._children.append(None)

    @property
    def solved_variable(self) -> sp.Symbol:
        return self._solution.solved_variable

    @property
    def solution(self):
        return self._solution

    def set_child(self, child: SkeletonTreeNode):
        assert child.is_dispatcher_node
        assert len(self._children) == 1
        self._children[0] = child
        child.set_parent(self)

    def get_child(self) -> SkeletonTreeNode:
        return self._children[0]

    @property
    def dispatcher_child(self) -> SkeletonTreeNode:
        return self.get_child()

    def is_parent_root_dispatcher(self) -> bool:
        assert self.parent is not None
        return self.parent.is_root


class ExplicitSolutionNode(SolutionNodeBase):

    def __init__(self, solution: VariableSolution):
        super(ExplicitSolutionNode, self).__init__(solution)
        assert solution.is_explicit_solution

    @property
    def explicit_solution(self):
        return self._solution

    def accept(self, visitor: SkeletonNodeVisitor):
        visitor.visit_explicit_solution_node(self)


class PolynomialSolutionNode(SolutionNodeBase):

    def __init__(self, solution: VariableSolution):
        super(PolynomialSolutionNode, self).__init__(solution)
        assert solution.is_polynomial

    @property
    def polynomial_solution(self):
        return self._solution

    def accept(self, visitor: SkeletonNodeVisitor):
        visitor.visit_polynomial_solution_node(self)


class LinearSinCosType_2_SolverNode(SolutionNodeBase):

    def __init__(self, solution: VariableSolution):
        super(LinearSinCosType_2_SolverNode, self).__init__(solution)
        assert solution.internal_solution.class_key() == VariableSolutionClassKey.LinearSinCosType_2.name

    def accept(self, visitor: SkeletonNodeVisitor):
        return visitor.visit_linear_type2_solution_node(self)

    @property
    def A_matrix(self):
        return self.solution.internal_solution.A_matrix

    @property
    def rows_to_try(self):
        return self.solution.internal_solution.rows_to_try


class General6DoFNumericalReduceSolutionNode(SolutionNodeBase):

    def __init__(self, solution: VariableSolution):
        super(General6DoFNumericalReduceSolutionNode, self).__init__(solution)
        assert solution.is_general_6dof_solution
        assert solution.internal_solution.class_key() == VariableSolutionClassKey.General6DoFNumericalReduce.name

    def accept(self, visitor: SkeletonNodeVisitor):
        visitor.visit_general_6dof_numerical_reduce_node(self)

    @property
    def general_6dof_solution(self):
        return self._solution

    @property
    def reduce_out(self):
        return self._solution.internal_solution.reduce_out

    @property
    def select_lines(self):
        return self._solution.internal_solution.select_lines

    @property
    def has_semi_symbolic_reduce(self):
        raw_solution = self.general_6dof_solution
        solution_impl = raw_solution.internal_solution
        general_6dof_solution = solution_impl
        return general_6dof_solution.has_semi_symbolic_reduce
