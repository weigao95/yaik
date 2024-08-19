from fk.robots import RobotDescription
from codegen.skeleton.skeleton_tree import SkeletonTree
from typing import List
import sympy as sp
import copy


class CodeGenerationBase(object):

    def __init__(self, tree: SkeletonTree):
        self._tree = tree

    @property
    def skeleton_tree(self) -> SkeletonTree:
        return self._tree

    @property
    def robot(self) -> RobotDescription:
        return self._tree.robot

    # The naming utilities
    def ik_function_name(self, raw_ik: bool) -> str:
        """
        :param raw_ik: whether the function check the solved value with fk
        """
        robot_name: str = self.robot.name
        if raw_ik:
            ik_name = robot_name + '_ik_solve_raw'
        else:
            ik_name = robot_name + '_ik_solve'
        return ik_name

    def fk_function_name(self) -> str:
        robot_name: str = self.robot.name
        fk_name = robot_name + '_fk'
        return fk_name

    def twist_jacobian_func_name(self) -> str:
        robot_name: str = self.robot.name
        func_name = robot_name + '_twist_jacobian'
        return func_name

    def angular_velocity_jacobian_func_name(self) -> str:
        robot_name: str = self.robot.name
        func_name = robot_name + '_angular_velocity_jacobian'
        return func_name

    def transform_point_jacobian_func_name(self) -> str:
        robot_name: str = self.robot.name
        func_name = robot_name + '_transform_point_jacobian'
        return func_name

    def default_file_name(self) -> str:
        return self.robot.name + '_ik'

    # The symbolic utility
    def free_symbols(self, include_unknown_as_parameter: bool = True) -> List[sp.Symbol]:
        """
        All symbols that must be provided by the caller
        except the ik target
        """
        return self._tree.get_free_symbols(include_unknown_as_parameter)

    # The string utility
    @staticmethod
    def new_line():
        return '\n'

    @staticmethod
    def remove_whitespace(string_to_remove: str):
        new_str = copy.copy(string_to_remove)
        new_str.replace(' ', '')
        return new_str


# The indent string
def indent(n_indent: int = 1):
    indent_str = '    '
    all_indent_str = ''
    for i in range(n_indent):
        all_indent_str += indent_str
    return all_indent_str
