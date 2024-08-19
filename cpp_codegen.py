from codegen.cpp_codegen.codegen_cpp import generate_code_cpp
from typing import Optional


def run_cpp_codegen(tree_yaml_path: Optional[str] = None, use_safe_operator: bool = True):
    # Make the tree
    import yaml
    from codegen.skeleton.tree_serialize import TreeSerializer
    if tree_yaml_path is None:
        tree_yaml_path = './gallery/spherical_wrist_six_axis/spherical_wrist_six_axis_ik.yaml'
    with open(tree_yaml_path, 'r') as read_stream:
        data_map = yaml.load(read_stream, Loader=yaml.CLoader)
    read_stream.close()

    # Run codegen
    tree = TreeSerializer.load_skeleton_tree(data_map)
    generate_code_cpp(tree, use_safe_operator=use_safe_operator)


if __name__ == '__main__':
    run_cpp_codegen()
