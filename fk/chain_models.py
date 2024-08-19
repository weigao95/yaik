from fk.chain_transform import ChainTransform
import utility.rotation_utils as rotation_utils
import numpy as np
import math
from typing import List


def bool2float(bool_value: bool) -> float:
    if bool_value:
        return 1.0
    else:
        return -1.0


def make_ur_robot(dh_length: List[float], shoulder_offset: float, elbow_offset: float):
    d1 = dh_length[0]
    a2 = dh_length[1]
    a3 = dh_length[2]
    d4 = dh_length[3]
    d5 = dh_length[4]
    d6 = dh_length[5]

    # 0, shoulder-pan joint
    robot = ChainTransform()
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, d1], axis=[0, 0, 1], flip_axis=False)

    # 1
    robot.add_chain_element_mmind(
        rpy=[0, - np.pi * 0.5, 0], xyz=[0, -shoulder_offset, 0], axis=[0, 1, 0], flip_axis=True)

    # 2
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, elbow_offset, a2], axis=[0, 1, 0], flip_axis=True)

    # 3
    robot.add_chain_element_mmind(
        rpy=[0, - np.pi * 0.5, 0],
        xyz=[0, -d4 + shoulder_offset - elbow_offset, a3], axis=[0, 1, 0], flip_axis=True)

    # 4
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, d5], axis=[0, 0, 1], flip_axis=False)

    # 5
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, -d6, 0], axis=[0, 1, 0], flip_axis=True)

    # The ee-offset
    ee_offset = rotation_utils.mmind_transform(
        np.array([np.pi * 0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    robot.set_post_transform(ee_offset)
    return robot


def ur5_robot() -> ChainTransform:
    dh_length = [0.089159, 0.42500, 0.39225, 0.10915, 0.09465, 0.0823]
    shoulder_offset = 0.220941
    elbow_offset = 0.1719
    return make_ur_robot(dh_length, shoulder_offset, elbow_offset)


def ur10_robot() -> ChainTransform:
    dh_length = [0.1273, 0.612, 0.5723, 0.163941, 0.1157, 0.0922]
    shoulder_offset = 0.220941
    elbow_offset = 0.1719
    return make_ur_robot(dh_length, shoulder_offset, elbow_offset)


def abb_crb_15000():
    mmind_dh = [0.265, 0.444, 0.11, 0.47, 0.101, 0.08]
    a = mmind_dh[0]
    c = mmind_dh[1]
    d = mmind_dh[2]
    e = mmind_dh[3]
    f = mmind_dh[4]
    j = mmind_dh[5]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, 0], axis=[0, 0, 1])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, a], axis=[0, 1, 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, c], axis=[0, 1, 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, d], axis=[1, 0, 0])

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[e, 0, 0], axis=[0, 1, 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[f, 0, j], axis=[1, 0, 0])
    return robot


def denso_cvr_038():
    mmind_dh = [0.18, 0.165, 0.012, 0.1775, 0.045, 0.02, 0.0445]
    a = mmind_dh[0]
    c = mmind_dh[1]
    d = mmind_dh[2]
    e = mmind_dh[3]
    f = mmind_dh[4]
    g = mmind_dh[5]
    h = mmind_dh[6]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, 0], axis=[0, 0, 1])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, a], axis=[0, 1, 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, g, c], axis=[0, 1, 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, d], axis=[1, 0, 0])

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[e, -h, 0], axis=[0, 1, 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[f, 0, 0], axis=[1, 0, 0])
    return robot


def denso_cvrb():
    mmind_dh = [0.22, 0.71, 0.59, 0.16, -0.05, -0.10]
    a = mmind_dh[0]
    c = mmind_dh[1]
    e = mmind_dh[2]
    f = mmind_dh[3]
    g = mmind_dh[4]
    h = mmind_dh[5]

    # Joint 1 in mmind
    robot = ChainTransform()
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, 0], axis=[0, 0, 1])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, a], axis=[0, 1, 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, g, c], axis=[0, 1, 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, 0], axis=[1, 0, 0])

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[e, -h, 0], axis=[0, 1, 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[f, 0, 0], axis=[1, 0, 0])
    return robot


def yaskawa_HC10():
    mmind_dh = [0.275, 0.7, 0.5, 0.13, 0.162]
    a = mmind_dh[0]
    c = mmind_dh[1]
    e = mmind_dh[2]
    f = mmind_dh[3]
    j = mmind_dh[4]
    mmind_mastering = [0, 0, 0, 0, 0, np.pi]
    mmind_axis_flip = [False, False, True, True, True, True]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mmind_mastering[0]], xyz=[0, 0, 0], axis=[0, 0, bool2float(not mmind_axis_flip[0])])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[1], 0], xyz=[0, 0, a], axis=[0, bool2float(not mmind_axis_flip[1]), 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[2], 0], xyz=[0, 0, c], axis=[0, bool2float(not mmind_axis_flip[2]), 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[3], 0, 0], xyz=[0, 0, 0], axis=[bool2float(not mmind_axis_flip[3]), 0, 0])

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[4], 0], xyz=[e, j, 0], axis=[0, bool2float(not mmind_axis_flip[4]), 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[5], 0, 0], xyz=[f, 0, 0], axis=[bool2float(not mmind_axis_flip[5]), 0, 0])

    # Ignore the post transform
    return robot


def faunc_M900IB_700():
    mmind_dh = [0.94, 0.410, 1.12, 0.250, 1.285, 0.3]
    d1 = mmind_dh[0]
    a1 = mmind_dh[1]
    d2 = mmind_dh[2]
    d3 = mmind_dh[3]
    d4 = mmind_dh[4]
    d5 = mmind_dh[5]
    a2 = 0
    mmind_mastering = [0, 0, 0, 0, 0, np.pi]
    mmind_axis_flip = [False, False, True, True, True, True]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mmind_mastering[0]], xyz=[0, 0, 0], axis=[0, 0, bool2float(not mmind_axis_flip[0])])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[1], 0], xyz=[a1, 0, d1], axis=[0, bool2float(not mmind_axis_flip[1]), 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[2], 0], xyz=[0, a2, d2], axis=[0, bool2float(not mmind_axis_flip[2]), 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[3], 0, 0], xyz=[0, 0, d3], axis=[bool2float(not mmind_axis_flip[3]), 0, 0])

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[4], 0], xyz=[d4, 0, 0], axis=[0, bool2float(not mmind_axis_flip[4]), 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[5], 0, 0], xyz=[d5, 0, 0], axis=[bool2float(not mmind_axis_flip[5]), 0, 0])
    return robot


def kuka_kr6_r900_sixx():
    mmind_dh = [0.400, 0.025, 0.455, 0.035, 0.420, 0.080]
    d1 = mmind_dh[0]
    a1 = mmind_dh[1]
    d2 = mmind_dh[2]
    d3 = mmind_dh[3]
    d4 = mmind_dh[4]
    d5 = mmind_dh[5]
    a2 = 0
    mmind_mastering = [0, - np.pi / 2, np.pi / 2, 0, 0, 0]
    mmind_axis_flip = [True, False, False, True, False, True]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mmind_mastering[0]], xyz=[0, 0, 0], axis=[0, 0, bool2float(not mmind_axis_flip[0])])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[1], 0], xyz=[a1, 0, d1], axis=[0, bool2float(not mmind_axis_flip[1]), 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[2], 0], xyz=[0, a2, d2], axis=[0, bool2float(not mmind_axis_flip[2]), 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[3], 0, 0], xyz=[0, 0, d3], axis=[bool2float(not mmind_axis_flip[3]), 0, 0])

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[4], 0], xyz=[d4, 0, 0], axis=[0, bool2float(not mmind_axis_flip[4]), 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[5], 0, 0], xyz=[d5, 0, 0], axis=[bool2float(not mmind_axis_flip[5]), 0, 0])
    return robot


def kawasaki_BX250L():
    mmind_dh = [0.67, 0.21, 1.1, 0.27, 0.185, 1.35, 0.343]
    d1 = mmind_dh[0]
    a1 = mmind_dh[1]
    l = mmind_dh[2]
    d2 = mmind_dh[3]
    a2 = mmind_dh[4]
    a4 = mmind_dh[5]
    a5 = mmind_dh[6]
    mmind_mastering = [- np.pi / 2, 0, 0, 0, 0, np.pi / 2]
    mmind_axis_flip = [True, False, True, False, True, False]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mmind_mastering[0]], xyz=[0, 0, 0], axis=[0, 0, bool2float(not mmind_axis_flip[0])])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[1], 0], xyz=[a1, 0, d1], axis=[0, bool2float(not mmind_axis_flip[1]), 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[2], 0], xyz=[a2, 0, d2], axis=[0, bool2float(not mmind_axis_flip[2]), 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[3], 0, 0], xyz=[0, 0, 0], axis=[bool2float(not mmind_axis_flip[3]), 0, 0])

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[4], 0], xyz=[a4, 0, 0], axis=[0, bool2float(not mmind_axis_flip[4]), 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[5], 0, 0], xyz=[a5, 0, 0], axis=[bool2float(not mmind_axis_flip[5]), 0, 0])
    return robot


def fanuc_M410IB_140H():
    mmind_dh = [0.72, 0.24, 1.16, 0.15, 1.73, 0.215]
    d1 = mmind_dh[0]
    a1 = mmind_dh[1]
    d2 = mmind_dh[2]
    d3 = mmind_dh[3]
    d4 = mmind_dh[4]
    d5 = mmind_dh[5]
    mmind_mastering = [0, 0, 0, - np.pi / 2, np.pi]
    mmind_axis_flip = [False, False, True, True, True]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mmind_mastering[0]], xyz=[0, 0, 0], axis=[0, 0, bool2float(not mmind_axis_flip[0])])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[1], 0], xyz=[a1, 0, d1], axis=[0, bool2float(not mmind_axis_flip[1]), 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[2], 0], xyz=[0, 0, d2], axis=[0, bool2float(not mmind_axis_flip[2]), 0])

    # Joint 4 in mmind is always zero

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[3], 0], xyz=[d4, 0, d3], axis=[0, bool2float(not mmind_axis_flip[3]), 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[4], 0, 0], xyz=[d5, 0, 0], axis=[bool2float(not mmind_axis_flip[4]), 0, 0])
    return robot


def abb_irb460_110_240():
    mmind_dh = [0.7425, 0.260, 0.945, 1.025, 0.220, 0.2515]
    d1 = mmind_dh[0]
    a1 = mmind_dh[1]
    d2 = mmind_dh[2]
    d3 = mmind_dh[3]
    d4 = mmind_dh[4]
    a4 = mmind_dh[5]
    mmind_mastering = [0, 0, 0, 0]
    mmind_axis_flip = [False, False, False, True]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mmind_mastering[0]], xyz=[0, 0, 0], axis=[0, 0, bool2float(not mmind_axis_flip[0])])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[1], 0], xyz=[a1, 0, d1], axis=[0, bool2float(not mmind_axis_flip[1]), 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[2], 0], xyz=[0, 0, d2], axis=[0, bool2float(not mmind_axis_flip[2]), 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mmind_mastering[3]], xyz=[d4, 0, -a4], axis=[0, 0, bool2float(not mmind_axis_flip[3])])
    return robot


def kawasaki_CP130L():
    mmind_dh = [0.75, 0.255, 1.2, 0.26, 1.55, 0.25, 0.24]
    d1 = mmind_dh[0]
    a1 = mmind_dh[1]
    a2 = mmind_dh[2]
    d2 = mmind_dh[3]
    d3 = mmind_dh[4]
    d4 = mmind_dh[5]
    a4 = mmind_dh[6]
    mmind_mastering = [0, 0, 0, - np.pi / 2]
    mmind_axis_flip = [True, False, True, True]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mmind_mastering[0]], xyz=[0, 0, 0], axis=[0, 0, bool2float(not mmind_axis_flip[0])])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[1], 0], xyz=[a1, 0, d1], axis=[0, bool2float(not mmind_axis_flip[1]), 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[2], 0], xyz=[0, 0, d2], axis=[0, bool2float(not mmind_axis_flip[2]), 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mmind_mastering[3]], xyz=[d4, 0, -a4], axis=[0, 0, bool2float(not mmind_axis_flip[3])])
    return robot


def staubli_tx2_60():
    mmind_dh = [0.375, 0.0, 0.290, 0.0, 0.310, 0.070, 0.02]
    d1 = mmind_dh[0]
    a1 = mmind_dh[1]
    d2 = mmind_dh[2]
    d3 = mmind_dh[3]
    d4 = mmind_dh[4]
    d5 = mmind_dh[5]
    a2 = mmind_dh[6]
    mmind_mastering = [0, 0, np.pi / 2, 0, 0, 0]
    mmind_axis_flip = [False, False, False, False, False, False]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mmind_mastering[0]], xyz=[0, 0, 0], axis=[0, 0, bool2float(not mmind_axis_flip[0])])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[1], 0], xyz=[a1, 0, d1], axis=[0, bool2float(not mmind_axis_flip[1]), 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[2], 0], xyz=[0, a2, d2], axis=[0, bool2float(not mmind_axis_flip[2]), 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[3], 0, 0], xyz=[0, 0, d3], axis=[bool2float(not mmind_axis_flip[3]), 0, 0])

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[4], 0], xyz=[d4, 0, 0], axis=[0, bool2float(not mmind_axis_flip[4]), 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[5], 0, 0], xyz=[d5, 0, 0], axis=[bool2float(not mmind_axis_flip[5]), 0, 0])
    return robot


def rokae_SR4():
    mmind_dh = [0.355, 0.400, 0.050, 0.050, 0.400, 0.1035, -0.136]
    a = mmind_dh[0]
    b = 0
    c = mmind_dh[1]
    d = mmind_dh[2]
    e = mmind_dh[4]
    f = mmind_dh[5]
    h = 0
    l = mmind_dh[6]
    m = 0
    g = mmind_dh[3]
    _q = l - h

    # No flip and offset
    mmind_mastering = [0, 0, 0, 0, 0, 0]
    mmind_axis_flip = [False, False, False, False, False, False]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mmind_mastering[0]], xyz=[0, 0, 0], axis=[0, 0, bool2float(not mmind_axis_flip[0])])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[1], 0], xyz=[b, 0, a], axis=[0, bool2float(not mmind_axis_flip[1]), 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[2], 0], xyz=[g, h, c], axis=[0, bool2float(not mmind_axis_flip[2]), 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[3], 0, 0], xyz=[0, 0, d], axis=[bool2float(not mmind_axis_flip[3]), 0, 0])

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[4], 0], xyz=[e, _q, 0], axis=[0, bool2float(not mmind_axis_flip[4]), 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[5], 0, 0], xyz=[f, 0, m], axis=[bool2float(not mmind_axis_flip[5]), 0, 0])
    return robot


def rokae_SR3():
    mmind_dh = [0.344, 0.290, 0.050, 0.050, 0.290, 0.1035, -0.136]
    a = mmind_dh[0]
    b = 0
    c = mmind_dh[1]
    d = mmind_dh[2]
    e = mmind_dh[4]
    f = mmind_dh[5]
    h = 0
    l = mmind_dh[6]
    m = 0
    g = mmind_dh[3]
    _q = l - h

    # No flip and offset
    mmind_mastering = [0, 0, 0, 0, 0, 0]
    mmind_axis_flip = [False, False, False, False, False, False]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mmind_mastering[0]], xyz=[0, 0, 0], axis=[0, 0, bool2float(not mmind_axis_flip[0])])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[1], 0], xyz=[b, 0, a], axis=[0, bool2float(not mmind_axis_flip[1]), 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[2], 0], xyz=[g, h, c], axis=[0, bool2float(not mmind_axis_flip[2]), 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[3], 0, 0], xyz=[0, 0, d], axis=[bool2float(not mmind_axis_flip[3]), 0, 0])

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[4], 0], xyz=[e, _q, 0], axis=[0, bool2float(not mmind_axis_flip[4]), 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[5], 0, 0], xyz=[f, 0, m], axis=[bool2float(not mmind_axis_flip[5]), 0, 0])
    return robot


def yaskawa_mpx3500():
    mmind_dh = [0.800, 0.000, 1.300, 0.000, 1.400, 0.180, 0.0987, 0.057]
    a = mmind_dh[0]
    b = mmind_dh[1]
    c = mmind_dh[2]
    d = mmind_dh[3]
    e = mmind_dh[4]
    f = mmind_dh[5]
    n = mmind_dh[6]
    o = mmind_dh[7]

    # No flip and offset
    # mmind_mastering = [0, 0, 0, 0, 0, np.pi]
    mmind_mastering = [0, 0, 0, 0, 0, 0]
    mmind_axis_flip = [False, False, True, True, True, True]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mmind_mastering[0]], xyz=[0, 0, 0], axis=[0, 0, bool2float(not mmind_axis_flip[0])])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[1], 0], xyz=[b, 0, a], axis=[0, bool2float(not mmind_axis_flip[1]), 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mmind_mastering[2], 0], xyz=[0, 0, c], axis=[0, bool2float(not mmind_axis_flip[2]), 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[3], 0, 0], xyz=[0, 0, d], axis=[bool2float(not mmind_axis_flip[3]), 0, 0])

    # There is a wired angle here
    theta = math.atan(n / o)

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, theta, 0], xyz=[e, 0, 0], axis=[bool2float(not mmind_axis_flip[4]), 0, 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mmind_mastering[5], -theta, 0],
        xyz=[f * math.cos(theta) + n * math.sin(theta), 0, f * math.sin(theta) - n * math.cos(theta)],
        axis=[bool2float(not mmind_axis_flip[5]), 0, 0])
    return robot


def abb_crb15000_10_1_52():
    a = 0.4
    b = 0.15
    c = 0.707
    d = 0.11
    e = 0.637
    f = 0.101
    h = 0
    l = 0
    m = 0.08
    g = 0
    _q = l - h
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, 0], axis=[0, 0, 1])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[b, 0, a], axis=[0, 1, 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[g, h, c], axis=[0, 1, 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[0, 0, d], axis=[1, 0, 0])

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[e, _q, 0], axis=[0, 1, 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, 0], xyz=[f, 0, m], axis=[1, 0, 0])
    return robot


def spherical_wrist_six_axis(length_parameters: List[float], mastering_joints: List[float],
                             axis_flip: List[bool], a2: float = 0) -> ChainTransform:
    assert len(length_parameters) == 6
    assert len(mastering_joints) == 6
    assert len(axis_flip) == 6
    d1 = length_parameters[0]
    a1 = length_parameters[1]
    d2 = length_parameters[2]
    d3 = length_parameters[3]
    d4 = length_parameters[4]
    d5 = length_parameters[5]
    robot = ChainTransform()

    # Joint 1 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, 0, - mastering_joints[0]], xyz=[0, 0, 0], axis=[0, 0, -1 if axis_flip[0] else 1])

    # Joint 2 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mastering_joints[1], 0], xyz=[a1, 0, d1], axis=[0, -1 if axis_flip[1] else 1, 0])

    # Joint 3 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mastering_joints[2], 0], xyz=[0, a2, d2], axis=[0, -1 if axis_flip[2] else 1, 0])

    # Joint 4 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mastering_joints[3], 0, 0], xyz=[0, 0, d3], axis=[-1 if axis_flip[3] else 1, 0, 0])

    # Joint 5 in mmind
    robot.add_chain_element_mmind(
        rpy=[0, - mastering_joints[4], 0], xyz=[d4, 0, 0], axis=[0, -1 if axis_flip[4] else 1, 0])

    # Joint 6 in mmind
    robot.add_chain_element_mmind(
        rpy=[- mastering_joints[5], 0, 0], xyz=[d5, 0, 0], axis=[-1 if axis_flip[5] else 1, 0, 0])

    # Post transform
    robot.set_post_transform(rotation_utils.mmind_rpy(np.array([0, 0.5 * np.pi, 0])))

    # Done
    return robot


def fanuc_m710_ic70():
    # mmind_dh = [0.565, 0.15, 0.87, 0.17, 1.016, 0.175]
    mmind_dh = [0.0, 0.15, 0.87, 0.17, 1.016, 0.175]
    mastering_joints = [0, 0, 0, 0, 0, np.pi]
    axis_flip = [False, False, True, True, True, True]
    return spherical_wrist_six_axis(mmind_dh, mastering_joints, axis_flip)


def fanuc_m710_ic70_raw():
    # mmind_dh = [0.565, 0.15, 0.87, 0.17, 1.016, 0.175]
    mmind_dh = [0.0, 0.15, 0.87, 0.17, 1.016, 0.175]
    mastering_joints = [0, 0, 0, 0, 0, 0]
    axis_flip = [False, False, False, False, False, False]
    return spherical_wrist_six_axis(mmind_dh, mastering_joints, axis_flip)


def mmind2dh_no_couple(jps_mmind: np.ndarray, master_joint: List[float], axis_flip: List[bool]) -> np.ndarray:
    jps_dh = jps_mmind.copy()
    for i in range(len(master_joint)):
        master_i: float = master_joint[i]
        flip_i: float = -1 if axis_flip[i] else 1
        jps_mmind_i: float = jps_mmind[i]
        # jps_mmind_i = flip_i * (jps_dh_i + master_i)
        # flip_i * jps_mmind_i = jps_dh_i + master_i
        # flip_i * jps_mmind_i - master_i = jps_dh_i
        jps_dh[i] = flip_i * jps_mmind_i - master_i
    return jps_dh


def fanuc_m710_ic70_mmind2dh_decouple(theta_in: np.ndarray) -> np.ndarray:
    theta_out = theta_in.copy()
    theta_out[2] += theta_in[1]
    return theta_out


def fanuc_m710_ic70_type_mmind2dh_decouple(theta_in: np.ndarray, master_joint: List[float],
                                           axis_flip: List[bool]) -> np.ndarray:
    theta_out = theta_in.copy()
    flip_1: float = -1 if axis_flip[1] else 1
    flip_2: float = -1 if axis_flip[2] else 2
    theta_out[2] -= flip_1 * flip_2 * theta_out[1]
    return theta_out


def fanuc_m710_ic70_mmind2dh_full(theta_in: np.ndarray) -> np.ndarray:
    mastering_joints = [0, 0, 0, 0, 0, np.pi]
    axis_flip = [False, False, True, True, True, True]
    # theta_out1 = fanuc_m710_ic70_mmind2dh_decouple(theta_in)
    theta_out1 = fanuc_m710_ic70_type_mmind2dh_decouple(theta_in, mastering_joints, axis_flip)
    theta_out2 = mmind2dh_no_couple(theta_out1, mastering_joints, axis_flip)
    return theta_out2


def abb_irb_6700_205_2_80():
    mmind_dh = [0.780, 0.320, 1.28, 0.200, 1.1825, 0.200]
    mastering_joints = [0, 0, 0, 0, 0, 0]
    axis_flip = [False, False, False, False, False, False]
    return spherical_wrist_six_axis(mmind_dh, mastering_joints, axis_flip)


if __name__ == '__main__':
    from fk.chain_transform import try_convert_to_dh
    import utility.transformations as transformations

    chain_raw = fanuc_m710_ic70_raw()
    theta = np.zeros(shape=(6,))
    theta[0] = 0.0 * np.pi
    theta[1] = 0.25 * np.pi
    theta[2] = 0.0 * np.pi
    theta[3] = 0.25 * np.pi
    theta[4] = 0.25 * np.pi
    theta[5] = 0.25 * np.pi
    theta_inner = fanuc_m710_ic70_mmind2dh_full(theta)
    pose1 = chain_raw.compute_fk(theta_inner)
    quat1 = transformations.quaternion_from_matrix(pose1, True)
    print(pose1)
    print(quat1)

    chain = fanuc_m710_ic70()
    theta_inner = fanuc_m710_ic70_mmind2dh_decouple(theta)
    pose1 = chain.compute_fk(theta_inner)
    quat1 = transformations.quaternion_from_matrix(pose1, True)
    print(pose1)
    print(quat1)

    # Random test
    for i in range(10):
        theta = np.random.rand(6, )
        theta1 = fanuc_m710_ic70_mmind2dh_full(theta)
        pose1 = chain_raw.compute_fk(theta1)

        theta2 = fanuc_m710_ic70_mmind2dh_decouple(theta)
        pose2 = chain.compute_fk(theta2)
        pose_diff: np.ndarray = pose1 - pose2
        pose_diff_scalar = np.absolute(pose_diff).sum()
        assert pose_diff_scalar < 1e-5

    # To dh representation
    try_convert_to_dh(chain_raw)
