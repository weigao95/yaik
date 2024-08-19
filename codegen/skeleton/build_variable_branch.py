from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.skeleton.tree_node import SolutionNodeBase, SolvedVariableDispatcherNode
from codegen.skeleton.tree_serialize import save_ik_tree, save_ik_list
from solver.solve_equations import solve_from_snapshot, SolveEquationOption
from solver.solve_branched import branched_equations_at_solved_variable_degeneration
from typing import Optional, Set, Dict, List
import sympy as sp


def find_node_to_visit(tree: SkeletonTree, visited_node: Set[int]) -> Optional[SolutionNodeBase]:
    """
    Find a node whose degenerate record is on variable value, and try to solve from that.
    """
    for i in range(tree.num_nodes):
        if i in visited_node:
            continue
        node_i = tree.node_list[i]
        if isinstance(node_i, SolutionNodeBase):
            degenerate_record = node_i.solution.degenerate_record
            parent = node_i.parent
            assert parent is not None
            assert parent.is_dispatcher_node
            if degenerate_record.is_degenerate_on_variable_values and \
                    isinstance(parent, SolvedVariableDispatcherNode):
                already_visited = False
                for j in range(len(parent.children)):
                    if j != 0 and parent.children[j] is not None:
                        already_visited = True
                        break

                # If not visited, do it
                if not already_visited:
                    return node_i

    # We are done, there is nothing to visit
    return None


def build_solved_variable_branches(
        tree: SkeletonTree,
        max_num_nodes: int = 50,
        numerical_test_cases: List[Dict[sp.Symbol, float]] = list()):
    """
    This method try to build the variable branch given the non-degenerate branch.
    Although this is a visiting to different nodes, we do not implement it as a visitor
    because it alters the tree structure
    """
    # The shared solve option
    solve_option = SolveEquationOption()
    solve_option.on_main_branch = False
    solve_option.disable_polynomial_solution = True

    # Obtain the fk equations
    visited_node: Set[int] = set()
    save_counter = 0
    while True:
        # Exist if tree is too large
        if tree.num_nodes > max_num_nodes:
            break

        # Done if nothing to visit
        current_solution_node = find_node_to_visit(tree, visited_node)
        if current_solution_node is None:
            break

        # Mark visited
        visited_node.add(current_solution_node.flatten_idx)

        # Get all parent solutions
        parent = current_solution_node.parent
        assert parent is not None
        variable_solutions = tree.variable_solutions_from_root(current_solution_node)

        # Find solution, this can be very expensive
        branch_solver_input_list = branched_equations_at_solved_variable_degeneration(
            tree.solved_equation_record, variable_solutions)
        for i in range(len(branch_solver_input_list)):
            solver_input_i, new_solution_i = branch_solver_input_list[i]
            print('Trying to solve variable degenerate branch at node index', current_solution_node.flatten_idx,
                  'branch', i, 'with unknowns', [elem.symbol.name for elem in solver_input_i.unknowns])
            solution_i = solve_from_snapshot(solver_input_i, option=solve_option)
            if len(solution_i) > 0:
                new_solution_i.extend(solution_i)
                tree.append_branched_solution(new_solution_i, parent, i + 1)

                # Save in each update
                save_counter += 1
                save_list_name = 'branch_list_at_{idx}.yaml'.format(idx=save_counter)
                save_tree_name = 'branch_tree_at_{idx}.yaml'.format(idx=save_counter)
                save_ik_list(solution_i, save_list_name)
                save_ik_tree(tree, save_tree_name)


# Debug code
# TODO(wei): revive it
def test_build_variable_branch():
    from fk.robots import puma_robot
    from solver.solved_variable import read_solution_list

    # Make the tree
    test_data_dir = '../test_data/puma_ik.yaml'
    solution_list = read_solution_list(test_data_dir)
    robot = puma_robot()
    skeleton_tree = SkeletonTree(solution_list, robot)

    # Build the branch
    build_solved_variable_branches(skeleton_tree)

    # Save the tree
    import yaml
    from codegen.skeleton.tree_serialize import TreeSerializer
    visitor = TreeSerializer(skeleton_tree.num_nodes, robot)
    skeleton_tree.preorder_visit(visitor)

    # Feed to yaml
    data_map = visitor.serialize_dict()
    save_path = 'puma_ik_tree.yaml'
    with open(save_path, 'w') as write_stream:
        yaml.dump(data_map, write_stream, Dumper=yaml.CDumper)
    write_stream.close()


def test_expanding_tree():
    import yaml
    from codegen.skeleton.tree_serialize import TreeSerializer
    # TODO(wei): revive this test
    load_tree_path = '../test_data/puma_main.yaml'
    with open(load_tree_path, 'r') as read_stream:
        data_map = yaml.load(read_stream, Loader=yaml.CLoader)
    read_stream.close()
    tree = TreeSerializer.load_skeleton_tree(data_map)

    # Build from that tree
    build_solved_variable_branches(tree)


if __name__ == '__main__':
    # test_build_variable_branch()
    test_expanding_tree()
