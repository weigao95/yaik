import fk.fk_equations as fk_equations
import fk.kinematics_dh as kinematics_dh
from solver.equation_utils import find_unknown_idx
from codegen.skeleton.codegen_base import CodeGenerationBase, indent
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.py_codegen.py_codegen_visitor_v2 import NodeIkSolutionGeneratorPython_v2
from utility.symbolic_utils import inverse_transform
from typing import List, Optional
import sympy as sp


class CodeGenerationPython(CodeGenerationBase):

    def __init__(self, tree: SkeletonTree):
        super().__init__(tree)
        self._all_solved_symbol = tree.all_solved_symbol()

        # Init the code
        self._generate_visitor = NodeIkSolutionGeneratorPython_v2(tree)
        self._code_lines: List[str] = list()

    def _add_code_line(self, code_line: str, indent_level: int = 0):
        if indent_level == 0:
            self._code_lines.append(code_line)
        else:
            self._code_lines.append(indent(indent_level) + code_line)

    def _add_empty_line(self):
        self._add_code_line('')

    def generate_code(self) -> List[str]:
        self._generate_import()
        self._generate_parameters()
        self._generate_ik_target_transform()
        self._generate_fk_method()
        self._generate_jacobian()
        self._generate_function_define()
        self._generate_node_processor()
        self._generate_checked_ik_method()
        self._generate_test_method()
        return self._code_lines

    def _generate_import(self):
        self._add_code_line('import numpy as np')
        self._add_code_line('import copy')
        self._add_code_line('import math')
        self._add_code_line('from typing import List, NewType')
        self._add_code_line('from python_run_import import *')

    def _generate_parameters(self):
        tree = self.skeleton_tree
        robot = self.robot
        self._add_empty_line()
        self._add_code_line('# Constants for solver')
        self._add_code_line('robot_nq: int = {nq}'.format(nq=len(robot.unknowns)))
        self._add_code_line('n_tree_nodes: int = {n_tree_nodes}'.
                            format(n_tree_nodes=self.skeleton_tree.num_nodes))
        # The tolerance
        self._add_code_line('pose_tolerance: float = 1e-4')
        self._add_code_line('{zero_tolerance}: float = 1e-6'.format(zero_tolerance=fk_equations.zero_tolerance))

        # Parameter values
        parameter_values = tree.solved_equation_record.parameter_value_dict
        self._add_empty_line()
        self._add_code_line('# Robot parameters')
        for parameter in parameter_values:
            parameter_value = parameter_values[parameter]
            variable_name = parameter.name
            define_line = "{var_name}: float = {var_value}". \
                format(var_name=variable_name, var_value=parameter_value)
            self._add_code_line(define_line)

        # The unknown offset
        self._add_empty_line()
        self._add_code_line('# Unknown offsets from original unknown value to raw value')
        self._add_code_line('# Original value are the ones corresponded to robot (usually urdf/sdf)')
        self._add_code_line('# Raw value are the ones used in the solver')
        self._add_code_line('# unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw')
        unknown_offset = self.robot.get_unknown_offset()
        if unknown_offset is not None:
            unknowns = self.robot.unknowns
            assert len(unknown_offset) == len(unknowns)
            for i in range(len(unknowns)):
                unknown_i = unknowns[i]
                unknown_offset_i = unknown_offset[i]
                offset_name_i = self.get_unknown_offset_name(unknown_i.symbol.name)
                define_line = "{var_name}: float = {var_value}". \
                    format(var_name=offset_name_i, var_value=unknown_offset_i)
                self._add_code_line(define_line)

    @staticmethod
    def get_unknown_offset_name(unknown_name: str):
        return unknown_name + '_offset_original2raw'

    def get_ik_target_raw_to_original_name(self):
        return self.robot.name + '_ik_target_raw_to_original'

    def get_ik_target_original_to_raw_name(self):
        return self.robot.name + '_ik_target_original_to_raw'

    def _generate_direct_forward_ee_transform(self, func_name: str):
        # Generate the code
        self._add_code_line('def {func_name}(T_ee: np.ndarray):'.format(func_name=func_name))
        self._add_code_line('return T_ee', indent_level=1)

    def _generate_ee_transform_sp(self,
                                  func_name: str,
                                  left_4x4_multiply: Optional[sp.Matrix],
                                  right_4x4_multiply: Optional[sp.Matrix]):
        # The sympy multiple
        left_4x4 = sp.eye(4) if left_4x4_multiply is None else left_4x4_multiply
        right_4x4 = sp.eye(4) if right_4x4_multiply is None else right_4x4_multiply
        ik_target_4x4 = fk_equations.ik_target_4x4()
        left_4x4 = sp.Matrix(left_4x4)
        right_4x4 = sp.Matrix(right_4x4)
        transformed_4x4 = left_4x4 * (ik_target_4x4 * right_4x4)

        # Generate the code
        self._add_code_line(
            'def {func_name}(T_ee: np.ndarray):'.format(func_name=func_name))

        # Extract the original symbol
        for r in range(3):
            for c in range(4):
                rc_symbol: sp.Symbol = ik_target_4x4[r, c]
                this_line = '{symbol_name}: float = T_ee[{r_idx}, {c_idx}]'.format(
                    symbol_name=str(rc_symbol),
                    r_idx=r, c_idx=c)
                self._add_code_line(this_line, indent_level=1)

        # Compute the new target
        self._add_code_line('ee_transformed = np.eye(4)', indent_level=1)
        for r in range(3):
            for c in range(4):
                expr_rc = transformed_4x4[r, c]
                expr_rc = sp.simplify(expr_rc)
                expr_code = sp.cxxcode(expr_rc)
                self._add_code_line('ee_transformed[{r}, {c}] = {expr_code}'.
                                    format(r=r, c=c, expr_code=expr_code), indent_level=1)
        self._add_code_line('return ee_transformed', indent_level=1)

    def _generate_ik_target_transform(self):
        self.append_new_function_two_empty_lines()
        self._add_code_line('# The transformation between raw and original ee target')
        self._add_code_line('# Original value are the ones corresponded to robot (usually urdf/sdf)')
        self._add_code_line('# Raw value are the ones used in the solver')
        self._add_code_line('# ee_original = pre_transform * ee_raw * post_transform')
        self._add_code_line('# ee_raw = dh_forward_transform(theta_raw)')

        # Trivial case
        if self.robot.get_pre_transform_sp() is None and self.robot.get_post_transform_sp() is None:
            self._generate_direct_forward_ee_transform(self.get_ik_target_original_to_raw_name())
            self.append_new_function_two_empty_lines()
            self._generate_direct_forward_ee_transform(self.get_ik_target_raw_to_original_name())
            return

        # Non-trivial codegen
        pre_transform_sp_inv = None if self.robot.get_pre_transform_sp() is None \
            else inverse_transform(self.robot.get_pre_transform_sp())
        post_transform_sp_inv = None if self.robot.get_post_transform_sp() is None \
            else inverse_transform(self.robot.get_post_transform_sp())
        self._generate_ee_transform_sp(self.get_ik_target_original_to_raw_name(),
                                       pre_transform_sp_inv, post_transform_sp_inv)
        self.append_new_function_two_empty_lines()
        self._generate_ee_transform_sp(self.get_ik_target_raw_to_original_name(),
                                       self.robot.get_pre_transform_sp(), self.robot.get_post_transform_sp())

    def _ik_func_args_string(self) -> str:
        free_symbols = self.free_symbols()
        func_arguments = '(T_ee: np.ndarray'
        for symbol_i in free_symbols:
            func_arguments += ', ' + str(symbol_i)
        func_arguments += '):'
        return func_arguments

    def _generate_function_define(self):
        """
        The function definition of the ik method
        """
        self.append_new_function_two_empty_lines()

        # The free symbols
        func_arguments = self._ik_func_args_string()
        self._add_code_line('def ' + self.ik_function_name(True) + func_arguments)

        # Expansion lines
        def expand_ik_target_lines():
            ik_target_lines: List[str] = ['# Extracting the ik target symbols']
            ik_4x4 = fk_equations.ik_target_4x4()
            for r in range(3):
                for c in range(4):
                    rc_symbol: sp.Symbol = ik_4x4[r, c]
                    this_line = '{symbol_name} = T_ee[{r_idx}, {c_idx}]'.format(
                        symbol_name=str(rc_symbol),
                        r_idx=r, c_idx=c)
                    ik_target_lines.append(this_line)

            # The ik inv target
            ik_target_lines.append('inv_ee_translation = - T_ee[0:3, 0:3].T.dot(T_ee[0:3, 3])')
            inv_ik_4x4 = fk_equations.ik_target_inv_4x4()
            for i in range(3):
                this_line = '{symbol_name} = inv_ee_translation[{idx}]'.format(
                    symbol_name=str(inv_ik_4x4[i, 3]),
                    idx=i)
                ik_target_lines.append(this_line)
            return ik_target_lines

        # Ok
        expand_lines = expand_ik_target_lines()
        for line_i in expand_lines:
            self._add_code_line(line_i, indent_level=1)

    def _generate_node_processor(self):
        # Make the visitor and go
        visitor = self._generate_visitor
        assert len(visitor.generated_code_lines) == 0
        visitor.append_init_lines()
        self.skeleton_tree.preorder_visit(visitor)
        visitor.append_finalize_lines()
        code_lines = visitor.generated_code_lines

        # A global 1-indent
        for this_line in code_lines:
            self._add_code_line(this_line, indent_level=1)

    def _generate_fk_method(self):
        # This is a new function
        self.append_new_function_two_empty_lines()

        # Generate the define
        free_symbols = self.free_symbols(include_unknown_as_parameter=False)
        func_arguments = '(theta_input: np.ndarray'
        for symbol_i in free_symbols:
            func_arguments += ',' + str(symbol_i)
        func_arguments += '):'
        self._add_code_line('def ' + self.fk_function_name() + func_arguments)

        # Compute fk
        robot = self.robot
        unknown_symbols = [elem.symbol for elem in robot.unknowns]
        fk_out = kinematics_dh.forward_kinematics_dh(
            robot.dh_params, unknown_symbols)
        ee_pose = fk_out.T_ee()

        # If pre_transform or post_transform is not None
        if robot.get_pre_transform_sp() is not None:
            ee_pose = robot.get_pre_transform_sp() * ee_pose
        if robot.get_post_transform_sp() is not None:
            ee_pose = ee_pose * robot.get_post_transform_sp()

        # Generate the unknown extraction
        unknown_offset = robot.get_unknown_offset()
        for i in range(len(unknown_symbols)):
            symbol_i = unknown_symbols[i]
            if unknown_offset is None:
                self._add_code_line('{var} = theta_input[{idx}]'.format(var=symbol_i.name, idx=i), indent_level=1)
            else:
                self._add_code_line('{var} = theta_input[{idx}] + {offset_i}'.
                                    format(var=symbol_i.name, idx=i,
                                           offset_i=self.get_unknown_offset_name(symbol_i.name)), indent_level=1)

        # Collect the fk code and perform cse
        fk_expressions: List[sp.Expr] = list()
        for i in range(3):
            for j in range(4):
                out_ij_expr = ee_pose[i, j]
                # out_ij_expr = out_ij_expr.subs(robot.parameters_value)
                fk_expressions.append(out_ij_expr)
        replacements, reduced_expressions = sp.cse(fk_expressions)

        # Generate the replacements
        self._add_empty_line()
        self._add_code_line('# Temp variable for efficiency', indent_level=1)
        for i in range(len(replacements)):
            symbol_i, expr_i = replacements[i]
            expr_code = sp.pycode(expr_i)
            self._add_code_line('{var} = {expr_code}'.format(var=str(symbol_i), expr_code=expr_code), indent_level=1)
        self._add_code_line('# End of temp variables', indent_level=1)

        # The computation code
        self._add_code_line('ee_pose = np.eye(4)', indent_level=1)
        for i in range(3):
            for j in range(4):
                out_ij_expr: sp.Expr = reduced_expressions[j + i * 4]
                ij_code = sp.pycode(out_ij_expr)
                self._add_code_line('ee_pose[{i}, {j}] = {ij_code}'.format(i=i, j=j, ij_code=ij_code), indent_level=1)
        # Finish
        self._add_code_line('return ee_pose', indent_level=1)

    def _generate_twist_or_angular_or_point_jacobian_impl(
            self,
            jacobian_matrix: sp.Matrix,
            jacobian_func_name: str,
            point_jacobian_point_on_ee: Optional[List[sp.Symbol]] = None):
        robot = self.robot
        unknown_symbols = [elem.symbol for elem in robot.unknowns]

        # Generate the define
        free_symbols = self.free_symbols(include_unknown_as_parameter=False)
        func_arguments = '(theta_input: np.ndarray'
        for symbol_i in free_symbols:
            func_arguments += ',' + str(symbol_i)

        # If this is point jacobian
        if point_jacobian_point_on_ee is not None:
            assert len(point_jacobian_point_on_ee) == 3
            func_arguments += ', point_on_ee: np.ndarray'

        # Done with declare
        func_arguments += '):'
        self.append_new_function_two_empty_lines()
        self._add_code_line('def ' + jacobian_func_name + func_arguments)

        # Generate the unknown extraction
        unknown_offset = robot.get_unknown_offset()
        for i in range(len(unknown_symbols)):
            symbol_i = unknown_symbols[i]
            if unknown_offset is None:
                self._add_code_line('{var} = theta_input[{idx}]'.format(var=symbol_i.name, idx=i), indent_level=1)
            else:
                self._add_code_line('{var} = theta_input[{idx}] + {offset_i}'.
                                    format(var=symbol_i.name, idx=i,
                                           offset_i=self.get_unknown_offset_name(symbol_i.name)),
                                    indent_level=1)

        # If this is point jacobian
        if point_jacobian_point_on_ee is not None:
            assert len(point_jacobian_point_on_ee) == 3
            for i in range(3):
                self._add_code_line('{var}: float = point_on_ee[{idx}]'.
                                    format(var=point_jacobian_point_on_ee[i], idx=i), indent_level=1)

        # The jacobian matrix
        jacobian_expressions: List[sp.Expr] = list()
        for i in range(jacobian_matrix.shape[0]):
            for j in range(jacobian_matrix.shape[1]):
                out_ij_expr = jacobian_matrix[i, j]
                jacobian_expressions.append(out_ij_expr)
        replacements, reduced_expressions = sp.cse(jacobian_expressions)

        self._add_empty_line()
        self._add_code_line('# Temp variable for efficiency', indent_level=1)
        for i in range(len(replacements)):
            symbol_i, expr_i = replacements[i]
            expr_code = sp.pycode(expr_i)
            self._add_code_line('{var} = {expr_code}'.format(var=str(symbol_i), expr_code=expr_code), indent_level=1)
        self._add_code_line('# End of temp variables', indent_level=1)

        # The computation code
        self._add_code_line('jacobian_output = np.zeros(shape=({r}, {c}))'.
                            format(r=jacobian_matrix.shape[0], c=jacobian_matrix.shape[1]), indent_level=1)

        counter = 0
        for i in range(jacobian_matrix.shape[0]):
            for j in range(jacobian_matrix.shape[1]):
                out_ij_expr: sp.Expr = reduced_expressions[counter]
                counter += 1
                if out_ij_expr == sp.S.Zero:
                    continue
                ij_code = sp.pycode(out_ij_expr)
                self._add_code_line('jacobian_output[{i}, {j}] = {ij_code}'.
                                    format(i=i, j=j, ij_code=ij_code), indent_level=1)

        # Finish
        self._add_code_line('return jacobian_output', indent_level=1)

    def _generate_jacobian(self):
        from fk.kinematics_dh import twist_jacobian_dh
        robot = self.robot
        unknown_symbols = [elem.symbol for elem in robot.unknowns]

        # Twist jacobian
        jacobian_twist = twist_jacobian_dh(robot.dh_params, unknown_symbols, robot.get_pre_transform_sp())
        self._generate_twist_or_angular_or_point_jacobian_impl(jacobian_twist, self.twist_jacobian_func_name(), None)

        # The angular jacobian
        self._generate_twist_or_angular_or_point_jacobian_impl(jacobian_twist[0:3, :],
                                                               self.angular_velocity_jacobian_func_name())

        # The point jacobian
        p_on_ee = sp.zeros(3, 1)
        p_on_ee[0] = sp.Symbol('p_on_ee_x')
        p_on_ee[1] = sp.Symbol('p_on_ee_y')
        p_on_ee[2] = sp.Symbol('p_on_ee_z')
        point_jacobian: sp.Matrix = sp.zeros(3, jacobian_twist.shape[1])
        for c in range(jacobian_twist.shape[1]):
            point_jacobian[:, c] = jacobian_twist[0:3, c].cross(p_on_ee) + jacobian_twist[3:6, c]
        self._generate_twist_or_angular_or_point_jacobian_impl(
            point_jacobian, self.transform_point_jacobian_func_name(), [p_on_ee[0], p_on_ee[1], p_on_ee[2]])

    def _generate_checked_ik_method(self):
        # The function definition
        self.append_new_function_two_empty_lines()
        func_arguments = self._ik_func_args_string()
        self._add_code_line('def ' + self.ik_function_name(False) + func_arguments)

        # The ee target
        if self.robot.get_pre_transform_sp() is None and self.robot.get_post_transform_sp() is None:
            self._add_code_line('T_ee_raw_in = T_ee', indent_level=1)
        else:
            self._add_code_line('T_ee_raw_in = ' + self.get_ik_target_original_to_raw_name() + '(T_ee)', indent_level=1)

        # Call the old method
        ik_raw_func = self.ik_function_name(True)
        free_symbols = self.free_symbols()
        unknown_symbols = [elem.symbol for elem in self.robot.unknowns]

        func_arguments_call = '(T_ee_raw_in'
        for symbol_i in free_symbols:
            if symbol_i in unknown_symbols:
                offset_var_i = self.get_unknown_offset_name(symbol_i.name)
                func_arguments_call += ', ' + str(symbol_i) + ' + ' + offset_var_i
            else:
                func_arguments_call += ', ' + str(symbol_i)
        func_arguments_call += ')'
        self._add_code_line('ik_output_raw = ' + ik_raw_func + func_arguments_call, indent_level=1)
        self._add_code_line('ik_output = list()', indent_level=1)
        self._add_code_line('for i in range(len(ik_output_raw)):', indent_level=1)
        self._add_code_line('ik_out_i = ik_output_raw[i]', indent_level=2)

        # Handle joint offset
        unknown_offset = self.robot.get_unknown_offset()
        if unknown_offset is not None:
            for j in range(len(unknown_offset)):
                unknown_j = self.robot.unknowns[j].symbol
                self._add_code_line('ik_out_i[{j}] -= {offset_j}'.
                                    format(j=j, offset_j=self.get_unknown_offset_name(unknown_j.name)),
                                    indent_level=2)

        # Make the fk args
        unknown_symbols = [elem.symbol for elem in self.skeleton_tree.robot.unknowns]
        fk_func_name = self.fk_function_name()
        fk_args_call = '(ik_out_i'
        for symbol_i in free_symbols:
            if symbol_i not in unknown_symbols:
                fk_args_call += ',' + str(symbol_i)
        fk_args_call += ')'
        self._add_code_line('ee_pose_i = {fk_func}{fk_args}'.
                            format(fk_func=fk_func_name, fk_args=fk_args_call), indent_level=2)
        self._add_code_line('ee_pose_diff = np.max(np.abs(ee_pose_i - T_ee))', indent_level=2)
        self._add_code_line('if ee_pose_diff < pose_tolerance:', indent_level=2)
        self._add_code_line('ik_output.append(ik_out_i)', indent_level=3)

        # Finished
        self._add_code_line('return ik_output', indent_level=1)

    def _generate_test_method(self):
        free_symbols = self.free_symbols()
        unknown_symbols = [elem.symbol for elem in self.robot.unknowns]
        for free_symbol_instance in free_symbols:
            if free_symbol_instance not in unknown_symbols:
                print('Cannot automatically generate tests for problems with free symbols that are not unknown')
                return

        # Necessary information
        ik_test_name = 'test_ik_solve_' + self.robot.name
        n_unknowns = len(self.robot.unknowns)
        fk_func_name = self.fk_function_name()

        # Start coding the test function
        self.append_new_function_two_empty_lines()
        self._add_code_line('def ' + ik_test_name + '():')
        self._add_code_line('theta_in = np.random.random(size=({theta_size}, ))'.
                            format(theta_size=n_unknowns), indent_level=1)
        self._add_code_line('ee_pose = {fk_func}(theta_in)'.format(fk_func=fk_func_name), indent_level=1)

        # The code to invoke ik, note that there might be free symbols other than ee pose
        if len(free_symbols) == 0:
            self._add_code_line('ik_output = ' + self.ik_function_name(raw_ik=False) + '(ee_pose)', indent_level=1)
        else:
            # There are free symbols, but they must all be unknown
            ik_args_str = '(ee_pose'
            for free_symbol_instance in free_symbols:
                idx_in_unknown = find_unknown_idx(self.robot.unknowns, free_symbol_instance.name)
                assert idx_in_unknown >= 0
                ik_args_str += ', {free_symbol_i}=theta_in[{unknown_idx}]'. \
                    format(free_symbol_i=free_symbol_instance.name, unknown_idx=idx_in_unknown)
            ik_args_str += ')'
            self._add_code_line('ik_output = ' + self.ik_function_name(raw_ik=False) + ik_args_str, indent_level=1)

        # OK
        self._add_code_line('for i in range(len(ik_output)):', indent_level=1)
        self._add_code_line('ee_pose_i = {fk_func}(ik_output[i])'.format(fk_func=fk_func_name), indent_level=2)
        self._add_code_line('ee_pose_diff = np.max(np.abs(ee_pose_i - ee_pose))', indent_level=2)
        self._add_code_line('print(\'The pose difference is \', ee_pose_diff)', indent_level=2)

        # The main func
        self.append_new_function_two_empty_lines()
        self._add_code_line('if __name__ == \'__main__\':')
        self._add_code_line(ik_test_name + '()', indent_level=1)

    def append_new_function_two_empty_lines(self):
        self._add_empty_line()
        self._add_empty_line()


def generate_code_python(tree: SkeletonTree, save_path: Optional[str] = None):
    """
    Perform code generation of the given skeleton tree and save it
    """
    codegen_py = CodeGenerationPython(tree)
    code_lines = codegen_py.generate_code()

    # Fill in the path
    if save_path is None:
        save_path = codegen_py.default_file_name() + '_generated.py'

    # Save the output
    with open(save_path, 'w') as write_stream:
        for code_line in code_lines:
            write_stream.writelines(code_line)
            write_stream.write('\n')
    write_stream.close()


def run_python_codegen(tree_path: Optional[str] = None):
    # Make the tree
    import yaml
    from codegen.skeleton.tree_serialize import TreeSerializer
    if tree_path is None:
        tree_path = '../../gallery/rokae_SR4/rokae_SR4_ik.yaml'
    with open(tree_path, 'r') as read_stream:
        data_map = yaml.load(read_stream, Loader=yaml.CLoader)
    read_stream.close()

    # Run codegen
    tree = TreeSerializer.load_skeleton_tree(data_map)
    generate_code_python(tree)


if __name__ == '__main__':
    run_python_codegen()
