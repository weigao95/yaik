from fk.robots import RobotDescription
import solver.build_equations as build_equations
import solver.equation_utils as equation_utils
from solver.solved_variable import SolutionDegenerateRecord, VariableSolution
from codegen.skeleton.tree_node import SkeletonTreeNode, SolutionNodeBase
from codegen.skeleton.tree_node import SkeletonNodeVisitor, NoBranchDispatcherNode, EquationAllZeroDispatcherNode
from codegen.skeleton.tree_node import SolvedVariableDispatcherNode, \
    ExplicitSolutionNode, PolynomialSolutionNode, General6DoFNumericalReduceSolutionNode
from codegen.skeleton.tree_node import LinearSinCosType_2_SolverNode
from codegen.skeleton.skeleton_tree import SkeletonTree
from typing import Dict, List, Optional
import sympy as sp
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class TreeSerializer(SkeletonNodeVisitor):
    """
    Convert a skeleton tree into a yaml save-able dict
    """
    def __init__(self, n_nodes: int, robot: RobotDescription):
        super(TreeSerializer, self).__init__()
        self._robot = robot

        # Element i of this list is the flat info for i'th node,
        # The node is ordered by their flatten_idx
        self._serialize_dict_for_nodes: List[Optional[Dict]] = list()
        for i in range(n_nodes):
            self._serialize_dict_for_nodes.append(None)

    @property
    def n_nodes(self):
        return len(self._serialize_dict_for_nodes)

    def serialize_dict(self):
        """
        Build the dict that can be serialized, only
        called after visiting
        """
        data_map = dict()
        data_map['robot'] = self._robot.to_dict()

        # The dict list
        node_dict_list: List[Dict] = list()
        for i in range(len(self._serialize_dict_for_nodes)):
            node_i_dict = self._serialize_dict_for_nodes[i]
            if node_i_dict is None:
                print('Invalid data at node ', i)
            node_dict_list.append(node_i_dict)

        # OK
        data_map['tree_nodes'] = node_dict_list
        return data_map

    @staticmethod
    def _build_node_connect_info(node: SkeletonTreeNode) -> Dict:
        data_map = dict()
        data_map['node_type'] = node.node_type
        data_map['flatten_idx_in_tree'] = node.flatten_idx_in_tree
        data_map['parent_idx'] = -1 if node.is_root else node.parent.flatten_idx

        # The child index
        child_index_list: List[int] = list()
        for child in node.children:
            child_idx = -1 if (child is None) else child.flatten_idx
            child_index_list.append(child_idx)
        data_map['children_index_list'] = child_index_list

        # OK
        return data_map

    def visit_no_branch_dispatcher(self, dispatcher: NoBranchDispatcherNode):
        data_map = self._build_node_connect_info(dispatcher)
        data_map['class_type'] = 'NoBranchDispatcherNode'
        assert dispatcher.flatten_idx < self.n_nodes
        self._serialize_dict_for_nodes[dispatcher.flatten_idx] = data_map

    def visit_solved_variable_dispatcher(self, dispatcher: SolvedVariableDispatcherNode):
        # Meta info, only class_type is different
        data_map = self._build_node_connect_info(dispatcher)
        data_map['class_type'] = 'SolvedVariableDispatcherNode'

        # Degenerate record
        data_map['degenerate_record'] = dispatcher.degenerate_record.to_dict()

        # Branch condition
        data_map['branch_conditions'] = equation_utils.expression_list_to_string_representation(
            dispatcher.branch_conditions)

        # Variable values
        branch_variable_value_list: List[Dict] = list()
        for i in range(dispatcher.num_degenerate_branches):
            variable_value_i = dispatcher.get_branch_variable_value(i)
            value_dict_i = dict()
            for key in variable_value_i:
                expr_value: sp.Expr = variable_value_i[key]
                value_dict_i[key.name] = str(expr_value)
            branch_variable_value_list.append(value_dict_i)
        data_map['branch_variable_values'] = branch_variable_value_list

        # Now we are done
        assert dispatcher.flatten_idx < self.n_nodes
        self._serialize_dict_for_nodes[dispatcher.flatten_idx] = data_map

    def visit_equation_all_zero_dispatcher(self, dispatcher: EquationAllZeroDispatcherNode):
        # Meta info, only class_type is different
        data_map = self._build_node_connect_info(dispatcher)
        data_map['class_type'] = 'EquationAllZeroDispatcherNode'

        # Degenerate record and checker
        data_map['degenerate_record'] = dispatcher.degenerate_record.to_dict()
        data_map['degenerate_checker'] = str(dispatcher.degenerate_checker)

        # Save back
        assert dispatcher.flatten_idx < self.n_nodes
        self._serialize_dict_for_nodes[dispatcher.flatten_idx] = data_map

    def visit_solution_node(self, solution_node: SolutionNodeBase, class_type: str):
        # Meta info, only class_type is different
        data_map = self._build_node_connect_info(solution_node)
        data_map['class_type'] = class_type
        data_map['solution'] = solution_node.solution.to_dict()
        assert solution_node.flatten_idx < self.n_nodes
        self._serialize_dict_for_nodes[solution_node.flatten_idx] = data_map

    def visit_explicit_solution_node(self, solution_node: ExplicitSolutionNode):
        self.visit_solution_node(solution_node, class_type='ExplicitSolutionNode')

    def visit_polynomial_solution_node(self, solution_node: PolynomialSolutionNode):
        self.visit_solution_node(solution_node, class_type='PolynomialSolutionNode')

    def visit_linear_type2_solution_node(self, solution_node: LinearSinCosType_2_SolverNode):
        self.visit_solution_node(solution_node, class_type='LinearSinCosType_2_SolverNode')

    def visit_general_6dof_numerical_reduce_node(self, solution_node: General6DoFNumericalReduceSolutionNode):
        self.visit_solution_node(solution_node, class_type='General6DoFNumericalReduceSolutionNode')

    @staticmethod
    def load_skeleton_tree(data_map: Dict) -> SkeletonTree:
        # Robot
        robot_map = data_map['robot']
        robot = RobotDescription.load_from_dict(robot_map)

        # Load the tree nodes
        node_dict_list = data_map['tree_nodes']
        tree_node_list: List[SkeletonTreeNode] = list()
        for i in range(len(node_dict_list)):
            node_dict_i = node_dict_list[i]
            node_i = TreeSerializer.load_node_no_connection(node_dict_i)
            tree_node_list.append(node_i)

        # Restore the connection
        for i in range(len(tree_node_list)):
            node_dict_i = node_dict_list[i]
            assert tree_node_list[i] is not None

            node_type = node_dict_i['node_type']
            tree_node_list[i]._node_type = node_type

            flatten_idx = int(node_dict_i['flatten_idx_in_tree'])
            tree_node_list[i]._flatten_idx_in_tree = flatten_idx

            parent_idx = int(node_dict_i['parent_idx'])
            parent = None if (parent_idx < 0) else tree_node_list[parent_idx]
            tree_node_list[i].set_parent(parent)

            # The child
            children_index_list = node_dict_i['children_index_list']
            assert len(tree_node_list[i].children) == len(children_index_list)
            for j in range(len(children_index_list)):
                child_idx_j = int(children_index_list[j])
                if child_idx_j < 0:
                    tree_node_list[i].set_child_by_index(None, j)
                else:
                    tree_node_list[i].set_child_by_index(tree_node_list[child_idx_j], j)

        # Update the tree
        ik_equations = build_equations.build_ik_equations(robot)
        tree = SkeletonTree([], robot, ik_equations)
        tree._node_list = tree_node_list
        tree._root_dispatcher = tree_node_list[0]

        # Finished
        return tree

    @staticmethod
    def load_node_no_connection(data_map: Dict):
        assert 'class_type' in data_map
        class_type = data_map['class_type']
        if class_type == 'NoBranchDispatcherNode':
            return TreeSerializer.load_no_branch_dispatcher(data_map)
        if class_type == 'EquationAllZeroDispatcherNode':
            return TreeSerializer.load_equation_all_zero_dispatcher(data_map)
        if class_type == 'SolvedVariableDispatcherNode':
            return TreeSerializer.load_solved_variable_dispatcher(data_map)
        if class_type == 'ExplicitSolutionNode':
            return TreeSerializer.load_explicit_solution_node(data_map)
        if class_type == 'PolynomialSolutionNode':
            return TreeSerializer.load_polynomial_solution_node(data_map)
        if class_type == 'LinearSinCosType_2_SolverNode':
            return TreeSerializer.load_linear_type2_solution_node(data_map)
        if class_type == 'General6DoFNumericalReduceSolutionNode':
            return TreeSerializer.load_general_6dof_numerical_reduce_solution_node(data_map)

        # Should not go here
        raise RuntimeError('Unknown class type of ', data_map['class_type'])

    @staticmethod
    def load_no_branch_dispatcher(data_map: Dict) -> NoBranchDispatcherNode:
        assert data_map['class_type'] == 'NoBranchDispatcherNode'
        node = NoBranchDispatcherNode()
        return node

    @staticmethod
    def load_equation_all_zero_dispatcher(data_map: Dict) -> EquationAllZeroDispatcherNode:
        assert data_map['class_type'] == 'EquationAllZeroDispatcherNode'
        degenerate_record_map = data_map['degenerate_record']
        degenerate_record = SolutionDegenerateRecord()
        degenerate_record.from_dict(degenerate_record_map)
        node = EquationAllZeroDispatcherNode(degenerate_record)
        return node

    @staticmethod
    def load_solved_variable_dispatcher(data_map: Dict) -> SolvedVariableDispatcherNode:
        assert data_map['class_type'] == 'SolvedVariableDispatcherNode'
        degenerate_record_map = data_map['degenerate_record']
        degenerate_record = SolutionDegenerateRecord()
        degenerate_record.from_dict(degenerate_record_map)
        node = SolvedVariableDispatcherNode(degenerate_record)
        return node

    @staticmethod
    def load_explicit_solution_node(data_map: Dict) -> ExplicitSolutionNode:
        assert data_map['class_type'] == 'ExplicitSolutionNode'
        solution_dict = data_map['solution']
        solution = VariableSolution()
        solution.from_dict(solution_dict)
        solution_node = ExplicitSolutionNode(solution)
        return solution_node

    @staticmethod
    def load_polynomial_solution_node(data_map: Dict) -> PolynomialSolutionNode:
        assert data_map['class_type'] == 'PolynomialSolutionNode'
        solution_dict = data_map['solution']
        solution = VariableSolution()
        solution.from_dict(solution_dict)
        solution_node = PolynomialSolutionNode(solution)
        return solution_node

    @staticmethod
    def load_general_6dof_numerical_reduce_solution_node(data_map: Dict) -> General6DoFNumericalReduceSolutionNode:
        assert data_map['class_type'] == 'General6DoFNumericalReduceSolutionNode'
        solution_dict = data_map['solution']
        solution = VariableSolution()
        solution.from_dict(solution_dict)
        solution_node = General6DoFNumericalReduceSolutionNode(solution)
        return solution_node

    @staticmethod
    def load_linear_type2_solution_node(data_map: Dict) -> LinearSinCosType_2_SolverNode:
        assert data_map['class_type'] == 'LinearSinCosType_2_SolverNode'
        solution_dict = data_map['solution']
        solution = VariableSolution()
        solution.from_dict(solution_dict)
        solution_node = LinearSinCosType_2_SolverNode(solution)
        return solution_node


def save_ik_tree(tree: SkeletonTree, save_path: str):
    """
    We might save ik multiple times during the process
    """
    visitor = TreeSerializer(tree.num_nodes, tree.robot)
    tree.preorder_visit(visitor)
    ik_dict = visitor.serialize_dict()
    with open(save_path, 'w') as file_stream:
        yaml.dump(ik_dict, file_stream, Dumper=Dumper)
    file_stream.close()


def save_ik_list(solution_list: List[VariableSolution], save_path: str):
    dict_list = list()
    for sol in solution_list:
        sol_dict = sol.to_dict()
        dict_list.append(sol_dict)
    ik_dict = dict()
    ik_dict['solution_list'] = dict_list
    with open(save_path, 'w') as file_stream:
        yaml.dump(ik_dict, file_stream, Dumper=Dumper)
    file_stream.close()


def read_ik_list(load_path: str) -> List[VariableSolution]:
    with open(load_path, 'r') as file_stream:
        data_map = yaml.load(file_stream, Loader=Loader)
    file_stream.close()
    dict_list = data_map['solution_list']
    solution_list = list()
    for sol_dict in dict_list:
        this_sol = VariableSolution()
        this_sol.from_dict(sol_dict)
        solution_list.append(this_sol)
    return solution_list


def read_skeleton_tree(save_path: str) -> Optional[SkeletonTree]:
    """
    Read the skeleton tree from a saved path
    """
    with open(save_path, 'r') as read_stream:
        data_map = yaml.load(read_stream, Loader=yaml.CLoader)
    read_stream.close()
    if data_map is None:
        return None
    tree = TreeSerializer.load_skeleton_tree(data_map)
    return tree


# Debug code
def test_tree_serializer():
    # Make the tree
    import yaml
    test_data_dir = '../../gallery/puma/puma_ik.yaml'
    with open(test_data_dir, 'r') as read_stream:
        data_map = yaml.load(read_stream, Loader=yaml.CLoader)
    read_stream.close()

    # Save back
    tree = TreeSerializer.load_skeleton_tree(data_map)
    print(tree.num_nodes)


if __name__ == '__main__':
    test_tree_serializer()
