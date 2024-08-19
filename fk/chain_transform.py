import numpy as np
import attr
import copy
import math
from typing import List, Optional, Tuple
import utility.rotation_utils as rotation_utils


@attr.s
class ChainTransformElement(object):
    joint_placement: np.ndarray = attr.ib()  # 4x4 transformation matrix
    joint_axis: List[float] = [0, 0, 1]  # Must be unit length
    flip_angle: bool = False

    # Compute the fk
    def compute_fk(self, theta: float) -> np.ndarray:
        theta_i = - theta if self.flip_angle else theta
        joint_transform_i = rotation_utils.angle_axis(theta_i, np.array(self.joint_axis))
        return self.joint_placement.dot(joint_transform_i)


class ChainTransform(object):

    def __init__(self):
        self._chain_elements: List[ChainTransformElement] = list()
        self._pre_transform: np.ndarray = np.eye(4)
        self._post_transform: np.ndarray = np.eye(4)

    @property
    def chain_length(self):
        return len(self._chain_elements)

    @property
    def chain_elements(self):
        return self._chain_elements

    @property
    def pre_transform(self):
        return self._pre_transform

    @property
    def post_transform(self):
        return self._post_transform

    def set_pre_transform(self, pre_transform: np.ndarray):
        assert len(pre_transform.shape) == 2
        assert pre_transform.shape[0] == 4
        assert pre_transform.shape[1] == 4
        self._pre_transform = pre_transform

    def set_post_transform(self, post_transform: np.ndarray):
        assert len(post_transform.shape) == 2
        assert post_transform.shape[0] == 4
        assert post_transform.shape[1] == 4
        self._post_transform = post_transform

    def add_chain_element(self, element: ChainTransformElement):
        self._chain_elements.append(element)

    def add_chain_element_mmind(self, rpy: List[float], xyz: List[float], axis: List[float], flip_axis: bool = False):
        assert len(rpy) == 3
        assert len(xyz) == 3
        assert len(axis) == 3
        joint_placement = rotation_utils.mmind_transform(np.array(rpy), np.array(xyz))
        axis_norm = np.linalg.norm(np.array(axis))
        normalized_axis = [axis[0] / axis_norm, axis[1] / axis_norm, axis[2] / axis_norm]
        element = ChainTransformElement(joint_placement)
        element.joint_axis = normalized_axis
        element.flip_angle = flip_axis
        self.add_chain_element(element)

    def compute_fk(self, theta: List[float]) -> np.ndarray:
        assert len(theta) == self.chain_length
        ee_pose = copy.copy(self._pre_transform)
        for i in range(len(theta)):
            element_i = self.chain_elements[i]
            element_transform_i = element_i.compute_fk(theta[i])
            ee_pose = ee_pose.dot(element_transform_i)
        ee_pose = ee_pose.dot(self._post_transform)
        return ee_pose


@attr.s
class ModifiedDHEntry(object):
    alpha: float = 0.0
    a: float = 0.0
    d: float = 0.0
    theta_offset: float = 0.0

    def compute_dh_transform(self, theta: float):
        # For x, these two can change order
        rotation_on_x = rotation_utils.rotation_x(self.alpha)
        translation_on_x = rotation_utils.translation_x(self.a)
        transform_on_x = rotation_on_x.dot(translation_on_x)

        # For z, these two can change order
        rotation_on_z = rotation_utils.rotation_z(self.theta_offset + theta)
        translation_on_z = rotation_utils.translation_z(self.d)
        transform_on_z = rotation_on_z.dot(translation_on_z)
        return transform_on_x.dot(transform_on_z)


class RobotDH(object):

    def __init__(self):
        self._robot_dh_parameters: List[ModifiedDHEntry] = list()
        self._pre_transform: np.ndarray = np.eye(4)
        self._post_transform: np.ndarray = np.eye(4)

    @property
    def dh_parameters(self):
        return self._robot_dh_parameters

    @property
    def pre_transform(self):
        return self._pre_transform

    @property
    def post_transform(self):
        return self._post_transform

    def set_pre_transform(self, pre_transform: np.ndarray):
        assert len(pre_transform.shape) == 2
        assert pre_transform.shape[0] == 4
        assert pre_transform.shape[1] == 4
        self._pre_transform = pre_transform

    def set_post_transform(self, post_transform: np.ndarray):
        assert len(post_transform.shape) == 2
        assert post_transform.shape[0] == 4
        assert post_transform.shape[1] == 4
        self._post_transform = post_transform

    def add_dh_entry(self, entry: ModifiedDHEntry):
        self._robot_dh_parameters.append(entry)

    def compute_fk(self, theta: List[float]) -> np.ndarray:
        assert len(self._robot_dh_parameters) == len(theta)
        ee_pose = copy.copy(self._pre_transform)
        for i in range(len(theta)):
            theta_i = theta[i]
            transform_i = self._robot_dh_parameters[i].compute_dh_transform(theta_i)
            ee_pose = ee_pose.dot(transform_i)
        ee_pose = ee_pose.dot(self._post_transform)
        return ee_pose


def to_rotation_z_impl(
        joint_placement_in: np.ndarray,
        joint_axis_in: List[float],
        flip_axis: bool) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Express joint_placement_in * calcJointTransform(theta) as
            pre_transform * rot_z(theta) * post_transform
    Return pre/post transform if it is feasible, else return None
    """
    axis = np.array(joint_axis_in)
    if flip_axis:
        axis = axis * -1.0

    # Invalid axis
    zero_threshold = 1e-6
    axis_norm = np.linalg.norm(axis)
    if axis_norm < zero_threshold:
        return None

    # Now it should have unit length
    axis = axis / axis_norm

    # Helpers
    def T_xz_transform() -> np.ndarray:
        # rot_x(theta) = T_xz * rot_z(theta) * T_xz
        T_xz = np.zeros(shape=(4, 4))
        T_xz[0, 2] = 1
        T_xz[1, 1] = -1
        T_xz[2, 0] = 1
        T_xz[3, 3] = 1
        return T_xz

    def T_yz_transform() -> np.ndarray:
        # rot_y(theta) = T_yz * rot_z(theta) * T_yz
        T_yz = np.zeros(shape=(4, 4))
        T_yz[0, 0] = -1
        T_yz[1, 2] = 1
        T_yz[2, 1] = 1
        T_yz[3, 3] = 1
        return T_yz

    # Unit z
    if np.linalg.norm(axis - np.array([0, 0, 1])) < zero_threshold:
        return joint_placement_in, np.eye(4)

    # - Unit z
    if np.linalg.norm(axis + np.array([0, 0, 1])) < zero_threshold:
        z_flip_transform = np.eye(4)
        z_flip_transform[0, 0] = -1
        z_flip_transform[2, 2] = -1
        return joint_placement_in.dot(z_flip_transform), z_flip_transform

    # Unit x
    if np.linalg.norm(axis - np.array([1, 0, 0])) < zero_threshold:
        t_xz = T_xz_transform()
        return joint_placement_in.dot(t_xz), t_xz

    # - Unit x
    if np.linalg.norm(axis + np.array([1, 0, 0])) < zero_threshold:
        t_xz = T_xz_transform()
        x_flip_transform = np.eye(4)
        x_flip_transform[0, 0] = -1
        x_flip_transform[1, 1] = -1
        return joint_placement_in.dot(x_flip_transform.dot(t_xz)), t_xz.dot(x_flip_transform)

    # Unit y
    if np.linalg.norm(axis - np.array([0, 1, 0])) < zero_threshold:
        t_yz = T_yz_transform()
        return joint_placement_in.dot(t_yz), t_yz

    # - Unit y
    if np.linalg.norm(axis + np.array([0, 1, 0])) < zero_threshold:
        t_yz = T_yz_transform()
        y_flip_transform = np.eye(4)
        y_flip_transform[1, 1] = -1
        y_flip_transform[2, 2] = -1
        return joint_placement_in.dot(y_flip_transform.dot(t_yz)), t_yz.dot(y_flip_transform)

    # Not implemented
    return None


def to_rotation_z(element: ChainTransformElement) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    pre_post_transform = to_rotation_z_impl(element.joint_placement, element.joint_axis, element.flip_angle)
    if pre_post_transform is None:
        return None

    # Check the result
    joint_pre, joint_post = pre_post_transform
    random_test_n = 20
    angle2check = np.random.random(size=(random_test_n, ))
    for angle in angle2check:
        fk_original = element.compute_fk(angle)
        fk_new = joint_pre.dot(rotation_utils.rotation_z(angle).dot(joint_post))
        fk_diff = fk_new - fk_original
        max_diff = np.max(np.abs(fk_diff))
        if max_diff > 1e-6:
            print('Max difference in to_rotation_z', max_diff)
            return None
    return pre_post_transform


def canonicalize_chain(chain: ChainTransform) -> Optional[ChainTransform]:
    """
    Convert a kinematic chain into a new chain such that all rotations are about the z axis
    Return None if not feasible
    """
    # Only contains the information about chain.elements
    joint_placement_list: List[np.ndarray] = list()

    # Start the loop
    next_transform_left_product = np.eye(4)
    for i in range(len(chain.chain_elements)):
        element_i = chain.chain_elements[i]
        # pre_post_transform = to_rotation_z_impl(element_i.joint_placement, element_i.joint_axis, element_i.flip_angle)
        pre_post_transform = to_rotation_z(element_i)
        if pre_post_transform is None:
            return None
        joint_i_pre, joint_i_post = pre_post_transform
        assert joint_i_pre is not None
        assert joint_i_post is not None

        # Make the joint placement
        aggregated_joint_pre = next_transform_left_product.dot(joint_i_pre)
        joint_placement_list.append(aggregated_joint_pre)
        next_transform_left_product = joint_i_post

    # Make a new chain
    new_pre_transform = copy.copy(chain.pre_transform)
    new_post_transform = next_transform_left_product.dot(chain.post_transform)
    new_chain = ChainTransform()
    new_chain.set_pre_transform(new_pre_transform)
    new_chain.set_post_transform(new_post_transform)

    # Add rot z joints
    for i in range(len(joint_placement_list)):
        placement_i = joint_placement_list[i]
        element_i = ChainTransformElement(placement_i)
        element_i.joint_axis = [0, 0, 1]
        element_i.flip_angle = False
        new_chain.add_chain_element(element_i)

    # Done
    return new_chain


def min_distance_with_z_axis(point_on_axis: np.ndarray, direction: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute the minimum distance of a line defined by a
    (point_on_axis, direction) pair with the z-axis. Return the point
    on the z-axis and the point on that axis.
    Obviously, these two point can be the same.
    :return a tuple consists of: 1) the point on z axis represented by its z-coordinate
                                 2) the point on the axis represented by its xyz coordinate
    """
    d0: float = direction[0]
    d1: float = direction[1]
    zero_threshold = 1e-6
    if d0 * d0 + d1 * d1 < zero_threshold:
        # This line is parallel with z
        min_distance_p_on_axis = np.array([point_on_axis[0], point_on_axis[1], 0.0])
        min_distance_p_on_z = 0.0
        return min_distance_p_on_z, min_distance_p_on_axis

    # Non-degenerate case, the line is not parallel with z
    # A point on the line is p(s) = point_on_axis + s * direction
    # The s corresponds to the min distance is computed using
    #         d_(distance_2_z_axis)_d_s = 0
    p0 = point_on_axis[0]
    p1 = point_on_axis[1]
    s = -(p0 * d0 + p1 * d1) / (d0 * d0 + d1 * d1)
    min_distance_p_on_axis = point_on_axis + s * direction
    min_distance_p_on_z: float = min_distance_p_on_axis[2]
    return min_distance_p_on_z, min_distance_p_on_axis


def construct_dh(chain_transform: ChainTransform) -> Optional[RobotDH]:
    # This should be canonical
    joint_pre_transform: List[np.ndarray] = list()
    for i in range(chain_transform.chain_length):
        element_i = chain_transform.chain_elements[i]
        axis_i = np.array(element_i.joint_axis)
        joint_pre_transform.append(chain_transform.chain_elements[i].joint_placement)
        assert (not element_i.flip_angle)
        assert (np.linalg.norm(axis_i - np.ndarray([0, 0, 1])) < 1e-6)
    assert len(joint_pre_transform) == chain_transform.chain_length

    # A vector of dh parameters to hold the result
    dh_params_list: List[ModifiedDHEntry] = list()
    for i in range(chain_transform.chain_length):
        dh_params_list.append(ModifiedDHEntry())

    # Make a copy to avoid mutating the original one
    i = chain_transform.chain_length - 1
    while i >= 1:
        pre_joint_i = joint_pre_transform[i]
        pre_joint_rotation = pre_joint_i[0:3, 0:3]
        pre_joint_origin = pre_joint_i[0:3, 3]
        pre_joint_z_direction: np.ndarray = pre_joint_rotation[:, 2]
        min_distance_p_on_z, min_distance_p_on_axis = min_distance_with_z_axis(pre_joint_origin, pre_joint_z_direction)
        p_on_z = np.array([0, 0, min_distance_p_on_z])
        p_on_axis = min_distance_p_on_axis
        new_x_direction = p_on_axis - p_on_z

        # Handle the degenerate case
        zero_threshold_for_axis_length = 1e-6
        if np.linalg.norm(new_x_direction) < zero_threshold_for_axis_length:
            new_x_direction = np.cross(pre_joint_z_direction, np.array([0, 0, 1]))
        new_x_direction = new_x_direction / np.linalg.norm(new_x_direction)

        # The z and y axis
        new_x_axis = new_x_direction
        new_z_axis = np.array([0, 0, 1])
        new_y_axis = np.cross(new_z_axis, new_x_axis)
        new_origin = p_on_z
        new_T_in_old = np.eye(4)
        new_T_in_old[0:3, 0] = new_x_axis
        new_T_in_old[0:3, 1] = new_y_axis
        new_T_in_old[0:3, 2] = new_z_axis
        new_T_in_old[0:3, 3] = new_origin

        # The pre-joint transform to new T
        new_pre_joint = np.linalg.inv(new_T_in_old).dot(pre_joint_i)
        prev_prejoint_right_product = pre_joint_i.dot(np.linalg.inv(new_pre_joint))
        joint_pre_transform[i] = new_pre_joint
        joint_pre_transform[i - 1] = joint_pre_transform[i - 1].dot(prev_prejoint_right_product)

        # Make the dh enetry
        this_entry = ModifiedDHEntry()
        dot_value_line_dir_and_unit_z: float = pre_joint_z_direction[2]  # the line direction dot unit z
        this_entry.alpha = math.acos(dot_value_line_dir_and_unit_z) if np.dot(pre_joint_z_direction, new_y_axis) < 0 \
            else - math.acos(dot_value_line_dir_and_unit_z)

        # Always positive as new axis point to the pre_joint frame
        this_entry.a = np.linalg.norm(p_on_axis - p_on_z)

        # Note the direction (sign) of d
        d_norm = np.linalg.norm(pre_joint_origin - p_on_axis)
        this_entry.d = d_norm if np.dot(pre_joint_origin - p_on_axis, pre_joint_z_direction) > 0 else - d_norm

        # Theta offset
        prejoint_x_axis = pre_joint_rotation[0:3, 0]
        prejoint_y_axis = pre_joint_rotation[0:3, 1]
        dot_value_two_x_axis = np.dot(prejoint_x_axis, new_x_axis)
        this_entry.theta_offset = - math.acos(dot_value_two_x_axis) if np.dot(new_x_axis, prejoint_y_axis) > 0 \
            else math.acos(dot_value_two_x_axis)

        # Debug code
        should_be_transform = this_entry.compute_dh_transform(0.0)
        pose_diff = np.linalg.norm(new_pre_joint - should_be_transform)
        if pose_diff > 1e-5:
            print("The pose difference is too large for joint ", i)
            print("The pre-joint should be ", new_pre_joint)
            print("While the computation from dh is ", should_be_transform)
            return None

        # Update idx
        dh_params_list[i] = this_entry
        i = i - 1

    # Finish
    dh_params_list[0] = ModifiedDHEntry()  # all zero
    robot_dh = RobotDH()
    for i in range(len(dh_params_list)):
        robot_dh.add_dh_entry(dh_params_list[i])
    robot_dh.set_pre_transform(joint_pre_transform[0])
    robot_dh.set_post_transform(chain_transform.post_transform)
    return robot_dh


def compare_fk(agent_0: ChainTransform, agent_1: ChainTransform) -> bool:
    test_n = 1000
    for i in range(test_n):
        q_i = np.random.random(size=(agent_0.chain_length, ))
        pose_0 = agent_0.compute_fk(q_i)
        pose_1 = agent_1.compute_fk(q_i)
        pose_diff = np.linalg.norm(pose_0 - pose_1)
        if pose_diff > 1e-6:
            print(pose_0)
            print(pose_1)
            return False
    return True


def try_convert_to_dh(chain: ChainTransform) -> Optional[RobotDH]:
    new_chain = canonicalize_chain(chain)
    if new_chain is None:
        print('Cannot make the new canonical kinematic chain')
        return None

    fk_match = compare_fk(chain, new_chain)
    if not fk_match:
        print('The canonicalize step failed as fk does not match')
        return None
    robot_dh = construct_dh(new_chain)
    fk_match = compare_fk(chain, robot_dh)
    if not fk_match:
        print('The dh construction step failed as fk does not match')
        return None
    return robot_dh


# Debug function below
def test_to_rot_z():
    output = to_rotation_z_impl(np.eye(4), joint_axis_in=[-1, 0, 0], flip_axis=False)
    pre, post = output
    theta2test = 0.15
    pre_post_fk = pre.dot(rotation_utils.rotation_z(theta2test).dot(post))
    original_fk = rotation_utils.rotation_x(-theta2test)
    print(pre_post_fk)
    print(original_fk)


if __name__ == '__main__':
    test_to_rot_z()
