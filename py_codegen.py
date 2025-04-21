from codegen.py_codegen.codegen_py import generate_code_python
from typing import Optional


def run_python_codegen(tree_path: Optional[str] = None):
    # Make the tree
    import yaml
    from codegen.skeleton.tree_serialize import TreeSerializer
    if tree_path is None:
        tree_path = './gallery/rainbow_y1_right_arm/rainbow_y1_r_arm_ik.yaml'
    with open(tree_path, 'r') as read_stream:
        data_map = yaml.load(read_stream, Loader=yaml.CLoader)
    read_stream.close()

    # Run codegen
    tree = TreeSerializer.load_skeleton_tree(data_map)
    generate_code_python(tree)


if __name__ == '__main__':
    run_python_codegen()
