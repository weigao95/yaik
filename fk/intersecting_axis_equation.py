from fk.fk_equations import DHEntry, ForwardKinematicsOutput, ik_target_4x4
from fk.fk_equations import split_transformation_into_equation, ik_target_inv_4x4
from fk.intersecting_axis import detect_intersecting_axis_triplet, detect_intersecting_axis_pair
from utility.symbolic_utils import inverse_transform, multiple_list_of_transforms
from solver.equation_types import ScalarEquation, TranslationalEquation
import sympy as sp
from typing import List, Optional, Tuple, Dict

# The usual equation is
# ik_target_4x4() == T_0 * T_1 * ... * T_i * T_j * T_k * T_k+1 * ... * T_n
# As (T_i * T_j * T_k) * point_k should not depends on theta_i/j/k, we transform the equation as
# ik_target_4x4() * inv(T_n) * inv(T_n-1) * ... * inv(T_k+1) == T_0 * T_1 * ... * T_i * T_j * T_k,
# And multiple point_k in both side. Of course we can move T_0 to the left of the equation.

# The inversed form of the equations
# Suppose dh i, j k intersects, then for equations as
# T_target = T_0 * T_1 * ... * T_i * T_j * T_k * ... * T_n
# we can inverse it as
# inv(T_target) = inv(T_n) * inv(T_{n-1}) * ... * inv(T_k) * inv(T_j) * inv(T_i) * ... * inv(T_0)
# inv(T_target) * T_0 * ... * T_{i-1} = inv(T_n) * inv(T_{n-1}) * ... * inv(T_k) * inv(T_j) * inv(T_i)
# If we multiple both side with point_prev (or point_{i-1}), where point_prev is the intersecting point in the 
# frame before i, then right-hand-side should NOT depends on unknowns_i/j/k, and we get another
# form of reduced equations.
# Of course, we can more inv(T_n) to left by
# T_n * inv(T_target) * T_0 * ... * T_{i-1} = inv(T_{n-1}) * ... * inv(T_k) * inv(T_j) * inv(T_i)


def build_intersecting_axis_point_match_equations(
        lhs_target: sp.Matrix,
        origin_rhs_list: List[Optional[sp.Matrix]],
        rhs_split_end: int,
        point_to_multiple: sp.Matrix) -> List[TranslationalEquation]:
    """
    Build equations in the form of: lhs * point = rhs * point
    :param lhs_target: ik_target_4x4() * inv(T_n) * inv(T_n-1) * ... * inv(T_k+1)
    :param origin_rhs_list: T_0, T_1, ..., T_i, T_j, T_k as the tuple list
    :param rhs_split_end: the first list element that belongs to T_i, we should NOT BREAK T_i * T_j * T_k
    :param point_to_multiple: lhs_target * point_to_multiple = rhs * point_to_multiple
    :return: List of scalar xyz equations
    """
    split_point = 0
    result_list: List[TranslationalEquation] = list()
    while split_point <= rhs_split_end:
        # This is identity (equation same as before, thus before must exist), don't need to process
        if split_point >= 1 and origin_rhs_list[split_point] is None:
            split_point += 1
            continue

        lhs_transform, split_rhs_transform = split_transformation_into_equation(
            origin_rhs_list, split_point, lhs_target)
        rhs_times_point = split_rhs_transform * point_to_multiple
        lhs_times_point = lhs_transform * point_to_multiple

        # Build equations
        equation_x = ScalarEquation(lhs_times_point[0], rhs_times_point[0])
        equation_y = ScalarEquation(lhs_times_point[1], rhs_times_point[1])
        equation_z = ScalarEquation(lhs_times_point[2], rhs_times_point[2])
        equation_xyz = TranslationalEquation(equation_x, equation_y, equation_z)
        result_list.append(equation_xyz)
        split_point += 1

    # OK, return the result
    return result_list


def get_lhs_target_and_rhs_list(
        fk_out: ForwardKinematicsOutput,
        intersect_begin: int,
        lhs_end_link_idx: int,
        rhs_end_link_idx: int,
        lhs_target: Optional[sp.Matrix] = None) -> Tuple[sp.Matrix, List[Optional[sp.Matrix]], int]:
    """
    Let k = :param lhs_end_link_idx, T_left = lhs_target * inv(T_n) * ... * inv(T_{k+1}), Note that T_k is NOT included
    Let i = :param rhs_end_link_idx, T_right = T_0 * T_1 * ... * T_i, Note that T_i is included
    :param fk_out:
    :param intersect_begin: The first link that is included in the intersection.
    :param lhs_end_link_idx:
    :param rhs_end_link_idx:
    :param lhs_target: Please refer to the equation above
    :return: The first element is T_left
             The second element is the list representation of T_right, we get T_right if we multiple them together
             The third element is the index of the first T_right list element that belongs to intersect_begin,
             when we constructing equations, we should stop at this position to avoid separate T_i * T_j * T_k
    """
    if lhs_target is None:
        lhs_target = ik_target_4x4()
    after_k_idx = len(fk_out.tuple_list) - 1
    while after_k_idx >= lhs_end_link_idx + 1:  # Not include the lhs_end_link_idx
        lhs_target = lhs_target * inverse_transform(fk_out.Ts(after_k_idx))
        after_k_idx -= 1

    rhs_list: List[Optional[sp.Matrix]] = list()
    first_tuple_element_in_i = None
    for tuple_idx in range(rhs_end_link_idx + 1):  # Include the rhs_end_link_idx
        tuple_i = fk_out.tuple_list[tuple_idx]
        for elem in tuple_i:
            rhs_list.append(elem)
            if tuple_idx == intersect_begin and first_tuple_element_in_i is None:
                first_tuple_element_in_i = len(rhs_list) - 1

    # OK
    return lhs_target, rhs_list, first_tuple_element_in_i


def build_intersecting_axis_translational_equations(
        fk_out: ForwardKinematicsOutput,
        intersect_in_last: sp.Matrix,
        intersecting_first_link: int,
        intersecting_last_link: int,
        intersect_in_prev: Optional[sp.Matrix] = None) -> List[TranslationalEquation]:
    """
    Build the translation equations supposed that links from :param intersecting_first_link
    to :param intersecting_first_link intersects, there can be at least 2, at most 3 intersecting links.
    :param fk_out: forward kinematics output
    :param intersect_in_last: see the document above
    :param intersecting_first_link:
    :param intersecting_last_link:
    :param intersect_in_prev:
    :return: The list of translational equations
    """
    # Build the basic equation
    intersect_i = intersecting_first_link
    intersect_k = intersecting_last_link

    # Compute T_left = T_ee_to_0 * inv(T_n) * ... * inv(T_{k+1}),
    # T_right = T_0 * T_1 * ... * T_i * T_j * T_k, as the list representation.
    # We must multiple T_i * T_j * T_k with the point to eliminate the variables.
    lhs_target, rhs_list, first_tuple_element_in_i = get_lhs_target_and_rhs_list(
        fk_out, intersect_i, intersect_k, intersect_k)

    # Build the equations from above
    translational_equations = build_intersecting_axis_point_match_equations(
        lhs_target, rhs_list, first_tuple_element_in_i, intersect_in_last)
    if intersect_in_prev is None:
        return translational_equations

    # For equations as
    # T_target = T_0 * T_1 * ... * T_i * T_j * T_k * ... * T_n
    # Inverse it as
    # inv(T_target) = inv(T_n) * inv(T_{n-1}) * ... * inv(T_k) * inv(T_j) * inv(T_i) * ... * inv(T_0)
    # inv(T_target) * T_0 * ... * T_{i-1} = inv(T_n) * inv(T_{n-1}) * ... * inv(T_k) * inv(T_j) * inv(T_i)
    inv_fk_tuple_list = list()
    for tuple_idx in reversed(range(len(fk_out.tuple_list))):
        this_tuple = fk_out.tuple_list[tuple_idx]
        new_tuple_list = list()
        for j in reversed(range(len(this_tuple))):
            elem_j = this_tuple[j]
            if elem_j is None:
                new_tuple_list.append(None)
            else:
                new_tuple_list.append(inverse_transform(elem_j))
        inv_fk_tuple_list.append(tuple(new_tuple_list))
    inv_fk_out = ForwardKinematicsOutput(inv_fk_tuple_list)

    # Note the different in index
    inv_begin = len(fk_out.tuple_list) - intersect_k - 1
    inv_end = len(fk_out.tuple_list) - intersect_i - 1
    inv_lhs_target, inv_rhs_list, inv_first_tuple_element_in_i = get_lhs_target_and_rhs_list(
        inv_fk_out, inv_begin, inv_end, inv_end, ik_target_inv_4x4())
    inv_translational_equations = build_intersecting_axis_point_match_equations(
        inv_lhs_target, inv_rhs_list, inv_first_tuple_element_in_i, intersect_in_prev)
    translational_equations.extend(inv_translational_equations)

    # Return the result
    return translational_equations


def intersecting_axis_triplet_equation(
        fk_out: ForwardKinematicsOutput,
        dh_params: List[DHEntry],
        unknowns: List[sp.Symbol],
        use_inverse_equations: bool = True) -> Dict[Tuple[str], List[TranslationalEquation]]:
    """
    Iterate over the robot parameters and forward kinematics output, detect whether there existing three
    intersecting axis. If so, build the intersecting axis equations from them. The new equations should
    only depends on three variables.
    TODO(wei): this assume each dh parameter corresponds to an unknown. Is this necessary?
    :param fk_out:
    :param dh_params:
    :param unknowns:
    :param use_inverse_equations:
    :return: the map from dependent variable names to the equations of these variables
    """
    # The output
    equation_dict = dict()
    # Find the joint i such that joints (i, i + 1, i + 2) intersect
    intersect_i = int(len(fk_out.tuple_list)) - 3
    while intersect_i >= 0:
        intersect_j = intersect_i + 1
        intersect_k = intersect_i + 2
        link_i = dh_params[intersect_i]
        link_j = dh_params[intersect_j]
        link_k = dh_params[intersect_k]
        # Only support revolute joint for now
        if unknowns[intersect_i] != link_i.theta \
                or unknowns[intersect_j] != link_j.theta \
                or unknowns[intersect_k] != link_k.theta:
            intersect_i -= 1
            continue

        # Check if intersect
        intersecting = detect_intersecting_axis_triplet(link_i, link_j, link_k)
        if intersecting is None:
            intersect_i -= 1
            continue

        # Now we know the point
        point_prev, _, _, point_k = intersecting
        point_k = sp.Matrix([point_k[0], point_k[1], point_k[2], 1])
        point_prev = sp.Matrix([point_prev[0], point_prev[1], point_prev[2], 1])
        point_prev = point_prev if use_inverse_equations else None

        # Build the translational equation
        equations = build_intersecting_axis_translational_equations(
            fk_out, point_k, intersect_i, intersect_k, point_prev)

        # Insert into the dictionary
        dependent_variables: List[str] = list()
        for var_idx in range(len(unknowns)):
            if var_idx != intersect_i and var_idx != intersect_j and var_idx != intersect_k:
                dependent_variables.append(unknowns[var_idx].name)
        dependent_variables = sorted(dependent_variables)
        dependent_variables_key = tuple(dependent_variables)
        equation_dict[dependent_variables_key] = equations

        # Update the iterator
        intersect_i -= 1

    # Find nothing
    return equation_dict


def build_intersecting_axis_cos_equations(
        fk_out: ForwardKinematicsOutput,
        intersecting_first_link: int,
        axis_i: sp.Matrix,
        axis_j: sp.Matrix,
        cos_target: sp.Expr) -> List[ScalarEquation]:
    """
    Let i = :param intersecting_first_link, and j = i + 1.
    The matrix equations can be written as:
    T_target = T_ee_to_0 = T_0 * T_1 ... T_i * T_j * ... * T_n
    Then, T_i_to_0 = T_0 * T_1 ... T_i,
          T_left = T_ee_to_0 * inv(T_n) * ... * inv(T_{j+1})
    where T_left maps vectors in frame j to the world frame (0). Thus, we have
          axis_i_in_0 = T_i_to_0.rotation() * axis_i
          axis_j_in_0 = T_left * axis_j,
          dot(axis_i_in_0, axis_j_in_0) = cos_target.
    The same equation holds for (axis_i_in_1, axis_j_in_1), and so on.
    :param fk_out:
    :param intersecting_first_link:
    :param axis_i:
    :param axis_j:
    :param cos_target:
    :return:
    """
    # Build the basic equation
    intersect_i = intersecting_first_link
    intersect_j = intersect_i + 1
    assert intersect_j <= len(fk_out.tuple_list) - 1

    # Compute T_left = T_ee_to_0 * inv(T_n) * ... * inv(T_{j+1}),
    # T_right = T_0 * T_1 * ... * T_i, as the list representation.
    # Note that T_j is included in T_left.
    lhs_target, original_rhs_list, first_tuple_element_in_i = get_lhs_target_and_rhs_list(
        fk_out, intersect_i, intersect_j, intersect_i)

    # OK, build the equations
    split_point = 0
    result_list: List[ScalarEquation] = list()
    while split_point <= first_tuple_element_in_i:
        # This is identity (equation same as before), don't need to process
        if split_point >= 1 and original_rhs_list[split_point] is None:
            split_point += 1
            continue

        # Build the transform
        lhs_transform, rhs_transform = split_transformation_into_equation(
            original_rhs_list, split_point, lhs_target)
        lhs_times_axis = lhs_transform * axis_i
        rhs_times_axis = rhs_transform * axis_j

        # Do the dot-product
        dot_axis_ij = lhs_times_axis[0] * rhs_times_axis[0] \
            + lhs_times_axis[1] * rhs_times_axis[1] + lhs_times_axis[2] * rhs_times_axis[2]
        cos_equation = ScalarEquation(cos_target, dot_axis_ij)
        result_list.append(cos_equation)
        split_point += 1

    # OK, forward the result
    return result_list


def intersecting_axis_pair_equation(
        fk_out: ForwardKinematicsOutput,
        dh_params: List[DHEntry],
        unknowns: List[sp.Symbol],
        use_inverse_equations: bool = True) -> Dict[Tuple[str, ...], Tuple[List[TranslationalEquation], List[ScalarEquation]]]:
    """
    Iterate over the robot parameters and forward kinematics output, detect whether there existing TWO
    intersecting axis. If so, build the intersecting axis equations from them. The new equations should
    only depends on FOUR variables.
    TODO(wei): this assume each dh parameter corresponds to an unknown. Is this necessary?
    :param fk_out:
    :param dh_params:
    :param unknowns:
    :param use_inverse_equations:
    :return: the map from dependent variable names to the equations of these variables
    """
    # The output
    equation_dict = dict()
    # Find the joint i such that joints (i, i + 1) intersect
    intersect_i = int(len(dh_params)) - 2
    while intersect_i >= 0:
        intersect_j = intersect_i + 1
        link_i = dh_params[intersect_i]
        link_j = dh_params[intersect_j]
        # Only support revolute joint for now
        if unknowns[intersect_i] != link_i.theta \
                or unknowns[intersect_j] != link_j.theta:
            intersect_i -= 1
            continue

        # Check if intersect
        intersecting = detect_intersecting_axis_pair(link_i, link_j)
        if intersecting is None:
            intersect_i -= 1
            continue

        # Now we know the point and angle, first build translational equation
        point_prev, _, point_j, cos_of_angle = intersecting
        point_j = sp.Matrix([point_j[0], point_j[1], point_j[2], 1])
        if use_inverse_equations:
            point_prev = sp.Matrix([point_prev[0], point_prev[1], point_prev[2], 1])
        else:
            point_prev = None

        # Build the translational equation
        translational_equations = build_intersecting_axis_translational_equations(
            fk_out, point_j, intersect_i, intersect_j, point_prev)

        # Build the cosine equations
        axis_to_multiple = sp.Matrix([0, 0, 1, 0])  # z-axis
        cos_equations = build_intersecting_axis_cos_equations(
            fk_out,
            intersect_i,
            axis_to_multiple, axis_to_multiple,
            cos_of_angle)

        # Obtain the variables
        dependent_variables: List[str] = list()
        for var_idx in range(len(unknowns)):
            if var_idx != intersect_i and var_idx != intersect_j:
                dependent_variables.append(unknowns[var_idx].name)
        dependent_variables = sorted(dependent_variables)
        dependent_variables_key = tuple(dependent_variables)
        equation_dict[dependent_variables_key] = (translational_equations, cos_equations)

        # Update the iterator
        intersect_i -= 1

    return equation_dict


# Debug code
# Please refer to fk/test_fk.py
