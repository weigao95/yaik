from solver.equation_utils import CollectedEquations, count_unknowns_expr
from solver.solved_variable import VariableSolution
from typing import List, Dict, Tuple, Optional
import sympy as sp


class UnaryVariableSolver(object):

    def try_solve(self,
                  collected_equations: CollectedEquations,
                  var_to_try: sp.Symbol,
                  unknowns: List[sp.Symbol]) -> List[VariableSolution]:
        """
        Perform a solving step for the given variable :param var_to_try, suppose all
        the unknown variables are collected in the :param unknowns list (every other symbols are known)
        :param collected_equations:
        :param var_to_try:
        :param unknowns:
        :return:
        """
        raise NotImplementedError
