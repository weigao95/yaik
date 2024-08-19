from fk.fk_equations import DHEntry
import sympy as sp
from typing import Optional, Tuple


def modified_dh_intersecting_axis_pair(dh_0: DHEntry,
                                       dh_1: DHEntry) -> Optional[Tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Expr]]:
    """
    Check whether two axis given above intersect at a common point for modified DH parameters.
    The DH parameter ASSUMES theta is the UNKNOWN.
    Two joints will connect three frames, here we return the intersecting point in all three frames.
    :param dh_0:
    :param dh_1:
    :return: The interesting point in frame_{-1}, frame 0 and frame_1, if intersects. Else return None.
             The forth element is the cosine of the intersecting angle.
    """
    if dh_1.a == sp.S.Zero:
        intersect_in_prev = sp.Matrix([dh_0.a, -sp.sin(dh_0.alpha) * dh_0.d, sp.cos(dh_0.alpha) * dh_0.d])
        intersect_in_0 = sp.Matrix([0, 0,         0])
        intersect_in_1 = sp.Matrix([0, 0, -dh_1.d])
        return intersect_in_prev, intersect_in_0, intersect_in_1, sp.cos(dh_1.alpha)
    return None


def detect_intersecting_axis_pair(
        link_0: DHEntry,
        link_1: DHEntry) -> Optional[Tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Expr]]:
    """
    Check whether two axis given above intersect at a common point for modified DH parameters.
    The DH parameter ASSUMES theta is the UNKNOWN.
    Two joints will connect three frames, here we return the intersecting point in all three frames.
    :param link_0:
    :param link_1:
    :return:
    """
    return modified_dh_intersecting_axis_pair(link_0, link_1)


def detect_intersecting_axis_triplet(
        dh_0: DHEntry,
        dh_1: DHEntry,
        dh_2: DHEntry) -> Optional[Tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]]:
    """
    Check whether three axis given above intersect at a common point for DH parameters.
    The DH parameter ASSUMES theta is the UNKNOWN.
    Three joints will connect four frames, here we return the intersecting point in all four frames.
    :param dh_0:
    :param dh_1:
    :param dh_2:
    :return: the interesting point in link_0/1/2, if intersects. Else return None.
    """
    # Perform analysis first for pair
    intersect_01 = detect_intersecting_axis_pair(dh_0, dh_1)
    if intersect_01 is None:
        return None
    intersect_12 = detect_intersecting_axis_pair(dh_1, dh_2)
    if intersect_12 is None:
        return None

    # Un-pack the data
    intersect_in_prev, intersect_01_in_0, intersect_01_in_1, _ = intersect_01
    _, intersect_12_in_1, intersect_12_in_2, _ = intersect_12

    # Check whether they are the same point
    point_diff = intersect_12_in_1 - intersect_01_in_1
    for i in range(3):
        if sp.simplify(point_diff[i]) != sp.S.Zero:
            return None

    # They are the same point
    return intersect_in_prev, intersect_01_in_0, intersect_01_in_1, intersect_12_in_2


# Debug code
def test_interesting_axis():
    from fk.robots import puma_robot
    robot = puma_robot()
    points = detect_intersecting_axis_triplet(
        robot.dh_params[-3], robot.dh_params[-2], robot.dh_params[-1])
    point_in_2, point_in_3, point_in_4, point_in_5 = points
    print(point_in_2)
    print(point_in_3)
    print(point_in_4)
    print(point_in_5)


if __name__ == '__main__':
    test_interesting_axis()
