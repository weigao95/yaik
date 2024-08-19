import numpy as np
import fk.fk_equations as fk_equations
import fk.kinematics_dh as kinematics_dh
from solver.equation_utils import find_unknown_idx
from codegen.skeleton.codegen_base import CodeGenerationBase, indent
from codegen.skeleton.skeleton_tree import SkeletonTree
from codegen.cpp_codegen.scope_manager import ScopeManager
from codegen.cpp_codegen.cpp_codegen_visitor import NodeIkSolutionGeneratorCpp
from solver.general_6dof.dh_utils import print_sp_matrix_by_row
from utility.symbolic_utils import inverse_transform
from typing import List, Optional
import sympy as sp


class CodeGenerationCpp(CodeGenerationBase):

    def __init__(self, tree: SkeletonTree, use_safe_operator: bool = False):
        super().__init__(tree)
        self._all_solved_symbol = tree.all_solved_symbol()

        # Init the code
        self._code_lines: List[str] = list()
        self._scope_manager: ScopeManager = ScopeManager(self._code_lines)

        # The visitor for ik
        self._generate_visitor = NodeIkSolutionGeneratorCpp(tree, use_safe_operator=use_safe_operator)

    def add_code_line(self, code_line: str):
        self._scope_manager.append_code_line(code_line)

    def add_empty_line(self):
        self.add_code_line('')

    def enter_scope(self, scope_name: str):
        self._scope_manager.enter_scope(scope_name)

    def exit_scope(self, scope_name: Optional[str]):
        self._scope_manager.exit_scope(scope_name)

    def fk_function_name(self) -> str:
        return 'computeFK'

    def twist_jacobian_func_name(self) -> str:
        return 'computeTwistJacobian'

    def angular_velocity_jacobian_func_name(self) -> str:
        return 'computeAngularVelocityJacobian'

    def transform_point_jacobian_func_name(self) -> str:
        return 'computeTransformPointJacobian'

    def ik_function_name(self, raw_ik: bool) -> str:
        if raw_ik:
            return 'computeRawIK'
        else:
            return 'computeIK'

    def generate_code(self) -> List[str]:
        self._generate_include()
        self._generate_class_start()
        self._generate_constexpr_parameters()
        self._generate_ik_target_transform()

        # Start the implementation
        self.add_empty_line()
        self.add_code_line('///************* Below are the actual FK and IK implementations *******************')
        self._generate_fk_method()
        self._generate_jacobian_method()
        self._generate_raw_ik()
        self._generate_raw_ik_with_workspace()
        self._generate_unchecked_ik_method()
        self._generate_warp_angle()
        self._generate_checked_ik_method()
        self._generate_handy_ik()
        self._generate_class_end()
        self._generate_test_method()
        return self._code_lines

    def _generate_include(self):
        self.add_code_line('#include \"yaik_cpp_common.h\"')
        self.add_empty_line()
        self.add_code_line('using namespace yaik_cpp;')
        self.add_empty_line()

    def get_class_name(self):
        tree = self.skeleton_tree
        robot_name = self.remove_whitespace(tree.robot.name)
        return robot_name + "_ik"

    def _generate_class_start(self):
        namespace_start_str = "struct " + self.get_class_name() + " {"
        # We don't treat this as a scope in codegen
        self.add_code_line(namespace_start_str)

    def _generate_class_end(self):
        namespace_end_str = "}; // struct " + self.get_class_name()
        # We don't treat this as a scope in codegen
        self.add_empty_line()
        self.add_code_line(namespace_end_str)

    def _generate_constexpr_parameters(self, generate_parameter: bool = True):
        tree = self.skeleton_tree
        robot = self.robot
        parameter_values = tree.solved_equation_record.parameter_value_dict
        self.add_empty_line()
        self.add_code_line('// Constants for solver')
        self.add_code_line('static constexpr int robot_nq = {nq};'.format(nq=len(robot.unknowns)))

        max_n_solution = 128 if tree.has_general_6dof_solution_node() else 16
        self.add_code_line(f'static constexpr int max_n_solutions = {max_n_solution};')
        self.add_code_line('static constexpr int n_tree_nodes = {n_tree_nodes};'.
                           format(n_tree_nodes=self.skeleton_tree.num_nodes))
        # The size of intermediate solutions
        all_solved_symbol = tree.all_solved_symbol()
        n_temp_sol = len(all_solved_symbol)
        self.add_code_line('static constexpr int intermediate_solution_size = {n_temp_sol};'.
                           format(n_temp_sol=n_temp_sol))
        self.add_code_line('static constexpr double pose_tolerance = 1e-6;')
        self.add_code_line('static constexpr double pose_tolerance_degenerate = 1e-4;')
        self.add_code_line('static constexpr double {zero_tolerance} = 1e-6;'.
                           format(zero_tolerance=fk_equations.zero_tolerance))
        self.add_code_line('using RawIKWorksace = ::yaik_cpp::RawIkWorkspaceTemplate'
                           '<intermediate_solution_size, max_n_solutions, robot_nq>;')
        if not generate_parameter:
            return

        # We also need to generate parameters
        self.add_empty_line()
        self.add_code_line('// Robot parameters')
        for parameter in parameter_values:
            parameter_value = parameter_values[parameter]
            variable_name = parameter.name
            define_line = "static constexpr double {var_name} = {var_value};". \
                format(var_name=variable_name, var_value=parameter_value)
            self.add_code_line(define_line)

        # The unknown offset
        self.add_empty_line()
        self.add_code_line('// Unknown offsets from original unknown value to raw value')
        self.add_code_line('// Original value are the ones corresponded to robot (usually urdf/sdf)')
        self.add_code_line('// Raw value are the ones used in the solver')
        self.add_code_line('// unknown_i_raw = unknown_i_original + unknown_i_offset_original2raw')
        unknown_offset = self.robot.get_unknown_offset()
        if unknown_offset is not None:
            unknowns = self.robot.unknowns
            assert len(unknown_offset) == len(unknowns)
            for i in range(len(unknowns)):
                unknown_i = unknowns[i]
                unknown_offset_i = unknown_offset[i]
                offset_name_i = self.get_unknown_offset_name(unknown_i.symbol.name)
                define_line = "static constexpr double {var_name} = {var_value};". \
                    format(var_name=offset_name_i, var_value=unknown_offset_i)
                self.add_code_line(define_line)

    def _generate_direct_forward_ee_transform(self, func_name: str):
        # Generate the code
        assert self._scope_manager.indent_level == 0
        self.add_code_line(
            'static Eigen::Matrix4d {func_name}(const Eigen::Matrix4d& T_ee)'.format(func_name=func_name))
        self.enter_scope(func_name)
        self.add_code_line('return T_ee;')
        self.exit_scope(func_name)

    def _generate_ee_transform_sp(self,
                                  func_name: str,
                                  left_4x4_multiply: Optional[sp.Matrix],
                                  right_4x4_multiply: Optional[sp.Matrix]):
        left_4x4 = sp.eye(4) if left_4x4_multiply is None else left_4x4_multiply
        right_4x4 = sp.eye(4) if right_4x4_multiply is None else right_4x4_multiply
        ik_target_4x4 = fk_equations.ik_target_4x4()
        left_4x4 = sp.Matrix(left_4x4)
        right_4x4 = sp.Matrix(right_4x4)
        transformed_4x4 = left_4x4 * (ik_target_4x4 * right_4x4)

        # Generate the code
        assert self._scope_manager.indent_level == 0
        self.add_code_line(
            'static Eigen::Matrix4d {func_name}(const Eigen::Matrix4d& T_ee)'.format(func_name=func_name))
        self.enter_scope(func_name)

        # Extract the original symbol
        for r in range(3):
            for c in range(4):
                rc_symbol: sp.Symbol = ik_target_4x4[r, c]
                this_line = 'const double {symbol_name} = T_ee({r_idx}, {c_idx});'.format(
                    symbol_name=str(rc_symbol),
                    r_idx=r, c_idx=c)
                self.add_code_line(this_line)

        # Compute the new target
        self.add_code_line('Eigen::Matrix4d ee_transformed;')
        self.add_code_line('ee_transformed.setIdentity();')
        for r in range(3):
            for c in range(4):
                expr_rc = transformed_4x4[r, c]
                expr_rc = sp.simplify(expr_rc)
                expr_code = sp.cxxcode(expr_rc)
                self.add_code_line('ee_transformed({r}, {c}) = {expr_code};'.format(r=r, c=c, expr_code=expr_code))
        self.add_code_line('return ee_transformed;')
        self.exit_scope(func_name)

    def _generate_ik_target_transform(self):
        self.add_empty_line()
        self.add_code_line('// The transformation between raw and original ee target')
        self.add_code_line('// Original value are the ones corresponded to robot (usually urdf/sdf)')
        self.add_code_line('// Raw value are the ones used in the solver')
        self.add_code_line('// ee_original = pre_transform * ee_raw * post_transform')
        self.add_code_line('// ee_raw = dh_forward_transform(theta_raw)')

        # Trivial case
        if self.robot.get_pre_transform_sp() is None and self.robot.get_post_transform_sp() is None:
            self._generate_direct_forward_ee_transform('endEffectorTargetOriginalToRaw')
            self.add_empty_line()
            self._generate_direct_forward_ee_transform('endEffectorTargetRawToOriginal')
            return

        # Non-trivial codegen
        pre_transform_sp_inv = None if self.robot.get_pre_transform_sp() is None \
            else inverse_transform(self.robot.get_pre_transform_sp())
        post_transform_sp_inv = None if self.robot.get_post_transform_sp() is None \
            else inverse_transform(self.robot.get_post_transform_sp())
        self._generate_ee_transform_sp('endEffectorTargetOriginalToRaw', pre_transform_sp_inv, post_transform_sp_inv)
        self.add_empty_line()
        self._generate_ee_transform_sp(
            'endEffectorTargetRawToOriginal', self.robot.get_pre_transform_sp(), self.robot.get_post_transform_sp())

    @staticmethod
    def get_unknown_offset_name(unknown_name: str):
        return unknown_name + '_offset_original2raw'

    def _generate_fk_method(self):
        # Generate the define
        robot = self.robot
        free_symbols = self.free_symbols(include_unknown_as_parameter=False)
        nq = len(robot.unknowns)
        func_arguments = '(const std::array<double, robot_nq>& theta_input_original'
        for symbol_i in free_symbols:
            func_arguments += ', double' + str(symbol_i)
        func_arguments += ')'

        # Since we are in namespace, we just use a fixed fk name
        fk_function_name = self.fk_function_name()
        self.add_code_line('static Eigen::Matrix4d ' + fk_function_name + func_arguments)
        self.enter_scope(fk_function_name)

        # Compute fk in symbolic form
        unknown_symbols = [elem.symbol for elem in robot.unknowns]
        fk_out = kinematics_dh.forward_kinematics_dh(
            robot.dh_params, unknown_symbols)
        ee_pose = fk_out.T_ee()

        # Generate the unknown extraction
        self.add_code_line('// Extract the variables')
        unknown_offset = robot.get_unknown_offset()
        for i in range(len(unknown_symbols)):
            symbol_i = unknown_symbols[i]
            if unknown_offset is None:
                self.add_code_line('const double {var} = theta_input_original[{idx}];'.format(var=symbol_i.name, idx=i))
            else:
                self.add_code_line('const double {var} = theta_input_original[{idx}] + {offset_value};'.
                                   format(var=symbol_i.name, idx=i,
                                          offset_value=self.get_unknown_offset_name(symbol_i.name)))

        # Collect the fk code and perform cse
        fk_expressions: List[sp.Expr] = list()
        for i in range(3):
            for j in range(4):
                out_ij_expr = ee_pose[i, j]
                fk_expressions.append(out_ij_expr)
        replacements, reduced_expressions = sp.cse(fk_expressions)

        # Generate the replacements
        self.add_empty_line()
        self.add_code_line('// Temp variable for efficiency')
        for i in range(len(replacements)):
            symbol_i, expr_i = replacements[i]
            expr_code = sp.cxxcode(expr_i)
            self.add_code_line('const double {var} = {expr_code};'.format(var=str(symbol_i), expr_code=expr_code))
        self.add_code_line('// End of temp variables')

        # The computation code
        self.add_code_line('Eigen::Matrix4d ee_pose_raw;')
        self.add_code_line('ee_pose_raw.setIdentity();')
        for i in range(3):
            for j in range(4):
                out_ij_expr: sp.Expr = reduced_expressions[j + i * 4]
                ij_code = sp.cxxcode(out_ij_expr)
                self.add_code_line('ee_pose_raw({i}, {j}) = {ij_code};'.format(i=i, j=j, ij_code=ij_code))
        # Finish
        self.add_code_line('return endEffectorTargetRawToOriginal(ee_pose_raw);')

        # End the codegen
        self.exit_scope(fk_function_name)

    def _generate_twist_or_angular_or_point_jacobian_impl(
            self,
            jacobian_matrix: sp.Matrix,
            jacobian_func_name: str,
            point_jacobian_point_on_ee: Optional[List[sp.Symbol]] = None):
        # The function define
        robot = self.robot
        free_symbols = self.free_symbols(include_unknown_as_parameter=False)
        nq = len(robot.unknowns)
        func_arguments = '(const std::array<double, robot_nq>& theta_input_original'
        for symbol_i in free_symbols:
            func_arguments += ', double' + str(symbol_i)

        # If this is point jacobian
        if point_jacobian_point_on_ee is not None:
            assert len(point_jacobian_point_on_ee) == 3
            func_arguments += ', Eigen::Vector3d& point_on_ee'

        # The output
        func_arguments += ', Eigen::Matrix<double, {n_rows}, robot_nq>& jacobian'.format(
            n_rows=jacobian_matrix.shape[0])
        func_arguments += ')'

        # Since we are in namespace, we just use a fixed fk name
        self.add_empty_line()
        self.add_code_line('static void ' + jacobian_func_name + func_arguments)
        self.enter_scope(jacobian_func_name)

        # Generate the unknown extraction
        self.add_code_line('// Extract the variables')
        unknown_symbols = [elem.symbol for elem in robot.unknowns]
        unknown_offset = robot.get_unknown_offset()
        for i in range(len(unknown_symbols)):
            symbol_i = unknown_symbols[i]
            if unknown_offset is None:
                self.add_code_line('const double {var} = theta_input_original[{idx}];'.format(var=symbol_i.name, idx=i))
            else:
                self.add_code_line('const double {var} = theta_input_original[{idx}] + {offset_value};'.
                                   format(var=symbol_i.name, idx=i,
                                          offset_value=self.get_unknown_offset_name(symbol_i.name)))

        # If this is point jacobian
        if point_jacobian_point_on_ee is not None:
            assert len(point_jacobian_point_on_ee) == 3
            for i in range(3):
                self.add_code_line('const double {var} = point_on_ee[{idx}];'.
                                   format(var=point_jacobian_point_on_ee[i], idx=i))

        # Collect the fk code and perform cse
        jacobian_expressions: List[sp.Expr] = list()
        for i in range(jacobian_matrix.shape[0]):
            for j in range(jacobian_matrix.shape[1]):
                out_ij_expr = jacobian_matrix[i, j]
                jacobian_expressions.append(out_ij_expr)
        replacements, reduced_expressions = sp.cse(jacobian_expressions)

        # Generate the replacements
        self.add_empty_line()
        self.add_code_line('// Temp variable for efficiency')
        for i in range(len(replacements)):
            symbol_i, expr_i = replacements[i]
            expr_code = sp.cxxcode(expr_i)
            self.add_code_line('const double {var} = {expr_code};'.format(var=str(symbol_i), expr_code=expr_code))
        self.add_code_line('// End of temp variables')

        # The computation code
        self.add_empty_line()
        self.add_code_line('jacobian.setZero();')
        counter = 0
        for i in range(jacobian_matrix.shape[0]):
            for j in range(jacobian_matrix.shape[1]):
                out_ij_expr: sp.Expr = reduced_expressions[counter]
                counter += 1
                if out_ij_expr != sp.S.Zero:
                    ij_code = sp.cxxcode(out_ij_expr)
                    self.add_code_line('jacobian({i}, {j}) = {ij_code};'.format(i=i, j=j, ij_code=ij_code))
        # Finish
        self.add_code_line('return;')

        # End the codegen
        self.exit_scope(jacobian_func_name)

    def _generate_jacobian_method(self):
        # Generate the define
        robot = self.robot

        # Compute fk in symbolic form
        unknown_symbols = [elem.symbol for elem in robot.unknowns]
        fk_out = kinematics_dh.forward_kinematics_dh(
            robot.dh_params, unknown_symbols)

        # The twist jacobian
        jacobian_twist = kinematics_dh.twist_jacobian_dh(robot.dh_params, unknown_symbols, robot.get_pre_transform_sp())
        self._generate_twist_or_angular_or_point_jacobian_impl(jacobian_twist, self.twist_jacobian_func_name())

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

    def _generate_raw_ik(self):
        self._generate_raw_ik_define()
        self._generate_raw_ik_node_processor()
        self._end_raw_ik_define()

    def _generate_raw_ik_define(self):
        self.add_empty_line()

        # The free symbols
        func_arguments = self._ik_func_args_string()
        self.add_code_line('static void ' + self.ik_function_name(True) + func_arguments)
        self.enter_scope('raw_ik')

        # Expansion lines
        def expand_ik_target_lines():
            ik_target_lines: List[str] = ['// Extracting the ik target symbols']
            ik_4x4 = fk_equations.ik_target_4x4()
            for r in range(3):
                for c in range(4):
                    rc_symbol: sp.Symbol = ik_4x4[r, c]
                    this_line = 'const double {symbol_name} = T_ee({r_idx}, {c_idx});'.format(
                        symbol_name=str(rc_symbol),
                        r_idx=r, c_idx=c)
                    ik_target_lines.append(this_line)

            # The ik inv target
            ik_target_lines.append('const auto& ee_rotation = T_ee.block<3, 3>(0, 0);')
            ik_target_lines.append('const auto& ee_translation = T_ee.block<3, 1>(0, 3);')
            ik_target_lines.append(
                'const Eigen::Vector3d inv_ee_translation = - ee_rotation.transpose() * ee_translation;')
            inv_ik_4x4 = fk_equations.ik_target_inv_4x4()
            for i in range(3):
                this_line = 'const double {symbol_name} = inv_ee_translation({idx});'.format(
                    symbol_name=str(inv_ik_4x4[i, 3]),
                    idx=i)
                ik_target_lines.append(this_line)
            return ik_target_lines

        # Add expand lines to code
        expand_lines = expand_ik_target_lines()
        for line_i in expand_lines:
            self.add_code_line(line_i)

    def _ik_func_args_string(self) -> str:
        free_symbols = self.free_symbols()
        func_arguments = '(const Eigen::Matrix4d& T_ee'
        for symbol_i in free_symbols:
            func_arguments += ', double ' + str(symbol_i)
        func_arguments += ', SolutionQueue<intermediate_solution_size, max_n_solutions>& solution_queue'
        func_arguments += ', NodeIndexWorkspace<max_n_solutions>& node_index_workspace'
        func_arguments += ', std::vector<std::array<double, robot_nq>>& ik_output'
        func_arguments += ')'
        return func_arguments

    def _generate_raw_ik_node_processor(self):
        # Make the visitor and go
        visitor = self._generate_visitor
        assert len(visitor.generated_code_lines) == 0
        visitor.append_init_lines()
        self.skeleton_tree.preorder_visit(visitor)
        visitor.append_finalize_lines()
        code_lines = visitor.generated_code_lines

        # A global 1-indent
        for this_line in code_lines:
            self._code_lines.append(indent() + this_line)

    def _end_raw_ik_define(self):
        self.exit_scope('raw_ik')

    def _generate_raw_ik_with_workspace(self):
        self.add_empty_line()
        free_symbols = self.free_symbols()
        func_arguments = '(const Eigen::Matrix4d& T_ee_raw'
        for symbol_i in free_symbols:
            func_arguments += ', double ' + str(symbol_i)
        func_arguments += ', RawIKWorksace& workspace'
        func_arguments += ')'

        # The function define
        self.add_code_line('static void ' + self.ik_function_name(True) + func_arguments)
        self.enter_scope('workspace_raw_ik')
        self.add_code_line('workspace.raw_ik_out.clear();')
        self.add_code_line('workspace.raw_ik_out.reserve(max_n_solutions);')

        func_arguments_call = '(T_ee_raw'
        for symbol_i in free_symbols:
            # We DO NOT need to add offset here, since the parameter input is already raw value.
            func_arguments_call += ', ' + str(symbol_i)
        func_arguments_call += ', workspace.solution_queue'
        func_arguments_call += ', workspace.node_index_workspace'
        func_arguments_call += ', workspace.raw_ik_out'
        func_arguments_call += ');'
        self.add_code_line(self.ik_function_name(True) + func_arguments_call)
        self.exit_scope('workspace_raw_ik')

    def _generate_unchecked_ik_method(self):
        # The function argument
        self.add_empty_line()
        free_symbols = self.free_symbols()
        func_arguments = '(const Eigen::Matrix4d& T_ee'
        for symbol_i in free_symbols:
            func_arguments += ', double ' + str(symbol_i)
        func_arguments += ', RawIKWorksace& workspace'
        func_arguments += ', std::vector<std::array<double, robot_nq>>& ik_output'
        func_arguments += ')'

        # The ik define
        self.add_code_line('static void ' + self.ik_function_name(False) + 'UnChecked' + func_arguments)
        self.enter_scope('un_checked_ik')

        # To raw target
        if (self.robot.get_pre_transform_sp() is not None) or (self.robot.get_post_transform_sp() is not None):
            self.add_code_line('const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);')
        else:
            self.add_code_line('const Eigen::Matrix4d& T_ee_raw = T_ee;')

        # Call the old method
        ik_raw_func = self.ik_function_name(True)
        unknown = self.robot.unknowns
        unknown_symbols = [elem.symbol for elem in unknown]

        def make_func_call_args(ee_pose_name: str = 'T_ee_raw'):
            func_call_arguments = '(' + ee_pose_name
            for symbol_j in free_symbols:
                # We need to add offset here, since the parameter from unknown used in rawIK is with offset.
                if symbol_j in unknown_symbols:
                    offset_var_i = self.get_unknown_offset_name(symbol_j.name)
                    func_call_arguments += ', ' + str(symbol_j) + ' + ' + offset_var_i
                else:
                    func_call_arguments += ', ' + str(symbol_j)
            func_call_arguments += ', workspace'
            func_call_arguments += ');'
            return func_call_arguments

        func_arguments_call = make_func_call_args('T_ee_raw')
        self.add_code_line(ik_raw_func + func_arguments_call)

        # From raw to original
        self.add_code_line('const auto& raw_ik_out = workspace.raw_ik_out;')
        self.add_code_line('ik_output.clear();')
        self.add_code_line('for(int i = 0; i < raw_ik_out.size(); i++)')
        self.enter_scope('for')
        self.add_code_line('auto ik_out_i = raw_ik_out[i];')

        # Handle offset
        unknown_offset = self.robot.get_unknown_offset()
        if unknown_offset is not None:
            unknowns = self.robot.unknowns
            for j in range(len(unknown_offset)):
                unknown_j = unknowns[j]
                self.add_code_line('ik_out_i[{j}] -= {offset_j};'.
                                   format(j=j, offset_j=self.get_unknown_offset_name(unknown_j.symbol.name)))

        self.add_code_line('ik_output.push_back(ik_out_i);')
        self.exit_scope('for')
        self.exit_scope('un_checked_ik')

    def _generate_warp_angle(self):
        self.add_empty_line()
        self.add_code_line('static void wrapAngleToPi(std::vector<std::array<double, robot_nq>>& ik_output)')
        self.enter_scope('wrap_angle')
        self.add_code_line('for(int i = 0; i < ik_output.size(); i++)')
        self.enter_scope('for_each_solution')
        self.add_code_line('// Wrap angular value to [-pi, pi]')
        self.add_code_line('auto& solution_i = ik_output[i];')
        unknowns = self.robot.unknowns
        for j in range(len(unknowns)):
            unknown_j = unknowns[j]
            if not unknown_j.is_revolute:
                continue
            # Do wrapping
            self.add_code_line('// Revolute unknown {symbol_name}'.format(symbol_name=unknown_j.symbol.name))
            self.add_code_line('while(solution_i[{j}] > M_PI)'.format(j=j))
            self.add_code_line(indent() + 'solution_i[{j}] -= 2 * M_PI;'.format(j=j))
            self.add_code_line('while(solution_i[{j}] < - M_PI)'.format(j=j))
            self.add_code_line(indent() + 'solution_i[{j}] += 2 * M_PI;'.format(j=j))

        # Done with solving
        self.exit_scope('for_each_solution')
        self.exit_scope('wrap_angle')

    def _generate_checked_ik_method(self):
        # The function argument
        self.add_empty_line()
        free_symbols = self.free_symbols()
        func_arguments = '(const Eigen::Matrix4d& T_ee'
        for symbol_i in free_symbols:
            func_arguments += ', double ' + str(symbol_i)
        func_arguments += ', RawIKWorksace& workspace'
        func_arguments += ', std::vector<std::array<double, robot_nq>>& ik_output'
        func_arguments += ')'

        # The ik define
        self.add_code_line('static void ' + self.ik_function_name(False) + func_arguments)
        self.enter_scope('checked_ik')

        # To raw target
        if (self.robot.get_pre_transform_sp() is not None) or (self.robot.get_post_transform_sp() is not None):
            self.add_code_line('const Eigen::Matrix4d& T_ee_raw = endEffectorTargetOriginalToRaw(T_ee);')
        else:
            self.add_code_line('const Eigen::Matrix4d& T_ee_raw = T_ee;')

        # Call the old method
        ik_raw_func = self.ik_function_name(True)
        unknown = self.robot.unknowns
        unknown_symbols = [elem.symbol for elem in unknown]

        def make_func_call_args(ee_pose_name: str = 'T_ee_raw'):
            func_call_arguments = '(' + ee_pose_name
            for symbol_j in free_symbols:
                # We need to add offset here, since the parameter from unknown used in rawIK is with offset.
                if symbol_j in unknown_symbols:
                    offset_var_j = self.get_unknown_offset_name(symbol_j.name)
                    func_call_arguments += ', ' + str(symbol_j) + ' + ' + offset_var_j
                else:
                    func_call_arguments += ', ' + str(symbol_j)
            func_call_arguments += ', workspace'
            func_call_arguments += ');'
            return func_call_arguments

        func_arguments_call = make_func_call_args('T_ee_raw')
        self.add_code_line(ik_raw_func + func_arguments_call)

        # Check them
        self.add_code_line('const auto& raw_ik_out = workspace.raw_ik_out;')
        self.add_code_line('ik_output.clear();')
        self.add_code_line('for(int i = 0; i < raw_ik_out.size(); i++)')
        self.enter_scope('for')
        self.add_code_line('auto ik_out_i = raw_ik_out[i];')

        # Handle offset
        unknown_offset = self.robot.get_unknown_offset()
        if unknown_offset is not None:
            unknowns = self.robot.unknowns
            for j in range(len(unknown_offset)):
                unknown_j = unknowns[j]
                self.add_code_line('ik_out_i[{j}] -= {offset_j};'.
                                   format(j=j, offset_j=self.get_unknown_offset_name(unknown_j.symbol.name)))

        # Make the fk call
        unknown_symbols = [elem.symbol for elem in self.skeleton_tree.robot.unknowns]
        fk_func_name = self.fk_function_name()
        fk_args_call = '(ik_out_i'
        for symbol_i in free_symbols:
            if symbol_i not in unknown_symbols:
                fk_args_call += ',' + str(symbol_i)
        fk_args_call += ')'
        self.add_code_line('const Eigen::Matrix4d ee_pose_i = {fk_func}{fk_args};'.
                           format(fk_func=fk_func_name, fk_args=fk_args_call))
        self.add_code_line('double ee_pose_diff = (ee_pose_i - T_ee).squaredNorm();')
        self.add_code_line('if (ee_pose_diff < pose_tolerance)')
        self.add_code_line(indent() + 'ik_output.push_back(ik_out_i);')
        self.exit_scope('for')

        # We have solutions here
        self.add_empty_line()
        self.add_code_line('if (!ik_output.empty())')
        self.enter_scope('if_valid_solution')
        self.add_code_line('wrapAngleToPi(ik_output);')
        self.add_code_line('removeDuplicate<robot_nq>(ik_output, zero_tolerance);')
        self.add_code_line('return;')
        self.exit_scope('if_valid_solution')

        # Numerical handling of degenerate case
        self.add_empty_line()
        self.add_code_line('// Disturbing method for degenerate handling')
        self.add_code_line('Eigen::Matrix4d T_ee_raw_disturbed = yaik_cpp::disturbTransform(T_ee_raw);')
        self.add_code_line('Eigen::Matrix4d T_ee_disturbed = endEffectorTargetRawToOriginal(T_ee_raw_disturbed);')
        func_arguments_call = make_func_call_args('T_ee_raw_disturbed')
        self.add_code_line(ik_raw_func + func_arguments_call)

        self.add_code_line('const auto& raw_ik_out_disturb = workspace.raw_ik_out;')
        self.add_code_line('for(int i = 0; i < raw_ik_out_disturb.size(); i++)')
        self.enter_scope('for')
        self.add_code_line('auto ik_out_i = raw_ik_out_disturb[i];')

        # Handle offset
        unknown_offset = self.robot.get_unknown_offset()
        if unknown_offset is not None:
            unknowns = self.robot.unknowns
            for j in range(len(unknown_offset)):
                unknown_j = unknowns[j]
                self.add_code_line('ik_out_i[{j}] -= {offset_j};'.
                                   format(j=j, offset_j=self.get_unknown_offset_name(unknown_j.symbol.name)))

        # Compute the pose again, note that we can use same fk_func_args
        self.add_code_line('Eigen::Matrix4d ee_pose_i = {fk_func}{fk_args};'.
                           format(fk_func=fk_func_name, fk_args=fk_args_call))
        self.add_code_line('double ee_pose_diff = (ee_pose_i - T_ee_disturbed).squaredNorm();')
        self.add_code_line('if (ee_pose_diff > pose_tolerance_degenerate)')
        self.add_code_line(indent() + 'continue;')

        # Try numerical
        self.add_empty_line()
        self.add_code_line('// Try numerical refinement')
        self.add_code_line('yaik_cpp::numericalRefinement<robot_nq>(computeFK, computeTwistJacobian, T_ee, ik_out_i);')
        self.add_code_line('ee_pose_i = {fk_func}{fk_args};'.
                           format(fk_func=fk_func_name, fk_args=fk_args_call))
        self.add_code_line('ee_pose_diff = (ee_pose_i - T_ee).squaredNorm();')
        self.add_code_line('if (ee_pose_diff < pose_tolerance_degenerate)')
        self.add_code_line(indent() + 'ik_output.push_back(ik_out_i);')
        self.exit_scope('for')

        # De-duplicate and end the function
        self.add_empty_line()
        self.add_code_line('wrapAngleToPi(ik_output);')
        self.add_code_line('removeDuplicate<robot_nq>(ik_output, zero_tolerance);')
        self.exit_scope('checked_ik')

    def _generate_handy_ik(self):
        # The function definition
        self.add_empty_line()
        free_symbols = self.free_symbols()
        func_arguments = '(const Eigen::Matrix4d& T_ee'
        for symbol_i in free_symbols:
            func_arguments += ', double ' + str(symbol_i)
        func_arguments += ')'
        self.add_code_line('static std::vector<std::array<double, robot_nq>> '
                           + self.ik_function_name(False) + func_arguments)
        self.enter_scope('handy_ik')
        self.add_code_line('std::vector<std::array<double, robot_nq>> ik_output;')
        self.add_code_line('RawIKWorksace raw_ik_workspace;')

        # Call ik
        func_arguments_call = '(T_ee'
        for symbol_i in free_symbols:
            func_arguments_call += ', ' + str(symbol_i)
        func_arguments_call += ', raw_ik_workspace'
        func_arguments_call += ', ik_output'
        func_arguments_call += ');'
        self.add_code_line(self.ik_function_name(False) + func_arguments_call)

        # Return and end the function
        self.add_code_line('return ik_output;')
        self.exit_scope('handy_ik')

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
        assert self._scope_manager.indent_level == 0
        self.add_empty_line()
        self.add_code_line('// Code below for debug')
        self.add_code_line('void ' + ik_test_name + '()')
        self.enter_scope('test_ik')
        class_prefix = self.get_class_name() + '::'
        self.add_code_line('std::array<double, {class_prefix}robot_nq> theta;'.format(class_prefix=class_prefix))
        self.add_code_line('std::random_device rd;')
        self.add_code_line('std::uniform_real_distribution<double> distribution;')
        self.add_code_line('for(auto i = 0; i < theta.size(); i++)')
        self.add_code_line(indent() + 'theta[i] = distribution(rd);')
        self.add_code_line('const Eigen::Matrix4d ee_pose = {class_prefix}{fk_func}(theta);'.
                           format(class_prefix=class_prefix, fk_func=fk_func_name))

        # The code to invoke ik, note that there might be free symbols other than ee pose
        if len(free_symbols) == 0:
            self.add_code_line('auto ik_output = ' + class_prefix + self.ik_function_name(raw_ik=False) + '(ee_pose);')
        else:
            # There are free symbols, but they must all be unknown
            ik_args_str = '(ee_pose'
            for free_symbol_instance in free_symbols:
                idx_in_unknown = find_unknown_idx(self.robot.unknowns, free_symbol_instance.name)
                assert idx_in_unknown >= 0
                ik_args_str += ', theta[{unknown_idx}]'. \
                    format(free_symbol_i=free_symbol_instance.name, unknown_idx=idx_in_unknown)
            ik_args_str += ');'
            self.add_code_line('auto ik_output = ' + class_prefix
                               + self.ik_function_name(raw_ik=False) + ik_args_str)

        # Check the ik result
        self.add_code_line('for(int i = 0; i < ik_output.size(); i++)')
        self.enter_scope('for_i')
        self.add_code_line('Eigen::Matrix4d ee_pose_i = {class_prefix}{fk_func}(ik_output[i]);'.
                           format(class_prefix=class_prefix, fk_func=fk_func_name))
        self.add_code_line('double ee_pose_diff = (ee_pose_i - ee_pose).norm();')
        self.add_code_line('std::cout << \"For solution \" << i << \" Pose different '
                           'with ground-truth \" << ee_pose_diff << std::endl;')
        self.exit_scope('for_i')
        self.exit_scope('test_ik')

        # The main func
        self.add_empty_line()
        self.add_code_line('int main()')
        self.enter_scope('main')
        self.add_code_line(ik_test_name + '();')
        self.exit_scope('main')


def generate_code_cpp(tree: SkeletonTree, save_path: Optional[str] = None, use_safe_operator: bool = False):
    """
    Perform code generation of the given skeleton tree and save it
    """
    codegen_cpp = CodeGenerationCpp(tree, use_safe_operator=use_safe_operator)
    code_lines = codegen_cpp.generate_code()

    # Fill in the path
    if save_path is None:
        save_path = codegen_cpp.default_file_name() + '_generated.cpp'

    # Save the output
    with open(save_path, 'w') as write_stream:
        for code_line in code_lines:
            write_stream.writelines(code_line)
            write_stream.write('\n')
    write_stream.close()


def run_cpp_codegen(tree_yaml_path: Optional[str] = None):
    # Make the tree
    import yaml
    from codegen.skeleton.tree_serialize import TreeSerializer
    if tree_yaml_path is None:
        tree_yaml_path = '../../gallery/atlas_l_hand/atlas_l_hand_ik.yaml'
    with open(tree_yaml_path, 'r') as read_stream:
        data_map = yaml.load(read_stream, Loader=yaml.CLoader)
    read_stream.close()

    # Run codegen
    tree = TreeSerializer.load_skeleton_tree(data_map)
    generate_code_cpp(tree)


if __name__ == '__main__':
    run_cpp_codegen()
