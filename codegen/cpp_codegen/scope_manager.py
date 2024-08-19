from codegen.skeleton.codegen_base import indent
from typing import List, Optional


class ScopeManager(object):

    def __init__(self, code_lines_workspace: List[str]):
        self._code_lines = code_lines_workspace
        self._scope_name_stack: List[str] = list()

    def enter_scope(self, scope_name: str):
        self._code_lines.append(self.current_indent() + '{')
        self._scope_name_stack.append(scope_name)

    def exit_scope(self, scope_name: Optional[str], with_semicolon: bool = False):
        if scope_name is not None:
            assert self._scope_name_stack[-1] == scope_name
        self._scope_name_stack.pop()

        # The last line
        if with_semicolon:
            self._code_lines.append(self.current_indent() + '};')
        else:
            self._code_lines.append(self.current_indent() + '}')

    def current_indent(self) -> str:
        return indent(self.indent_level)

    def append_code_line(self, code_line: str):
        self._code_lines.append(self.current_indent() + code_line)

    @property
    def indent_level(self):
        return len(self._scope_name_stack)
