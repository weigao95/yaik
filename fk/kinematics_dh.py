import sympy as sp
from typing import List, Optional, Tuple
from utility.symbolic_utils import rotation_x, rotation_z, translation_x, translation_z, is_identity_matrix
from fk.fk_equations import DHEntry, ForwardKinematicsOutput


def modified_dh_transform_raw(alpha: sp.Symbol, a: sp.Symbol, d: sp.Symbol, theta: sp.Symbol) -> sp.Matrix:
    """
    Compute the forward transform T_{i+1}_to_i given the dh parameters
    :param alpha:
    :param a:
    :param d:
    :param theta:
    :return:
    """
    t = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta), 0, a],
        [sp.sin(theta) * sp.cos(alpha), sp.cos(theta) * sp.cos(alpha), -sp.sin(alpha), -sp.sin(alpha) * d],
        [sp.sin(theta) * sp.sin(alpha), sp.cos(theta) * sp.sin(alpha),  sp.cos(alpha),  sp.cos(alpha) * d],
        [0, 0, 0, 1]
    ])
    return t


def modified_dh_transform(dh_param: DHEntry) -> sp.Matrix:
    return modified_dh_transform_raw(dh_param.alpha, dh_param.a, dh_param.d, dh_param.theta)


def modified_dh_transform_tuple(dh_param: DHEntry,
                                unknown_symbol: sp.Symbol) -> (Optional[sp.Matrix], sp.Matrix, Optional[sp.Matrix]):
    """
    Return the dh transform as a tuple of (left_T, unknown_T, right_T), only unknown_T depends on the unknown.
    left_T * unknown_T * right_T == modified_dh_transform(dh_param).
    left_T and right_T can be None if they are identity.
    :param dh_param:
    :param unknown_symbol: can only be theta and d
    :return: (left_T, unknown_T, right_T) tuple
    """
    if unknown_symbol == dh_param.theta:
        left_transform = rotation_x(dh_param.alpha) * translation_x(dh_param.a)
        left_transform = left_transform if not is_identity_matrix(left_transform) else None
        theta_transform = rotation_z(dh_param.theta)
        right_transform = translation_z(dh_param.d)
        right_transform = right_transform if not is_identity_matrix(right_transform) else None
        return left_transform, theta_transform, right_transform
    else:
        assert unknown_symbol == dh_param.d
        left_transform = rotation_x(dh_param.alpha) * translation_x(dh_param.a) * rotation_z(dh_param.theta)
        left_transform = left_transform if not is_identity_matrix(left_transform) else None
        unknown_transform = translation_z(dh_param.d)
        return left_transform, unknown_transform, None


def forward_kinematics_dh(dh_params: List[DHEntry],
                          unknowns: List[sp.Symbol]) -> ForwardKinematicsOutput:
    """
    Compute the forward kinematics using the DH parameters. The DH parameters is modified.
    Please refer to the images at docs/classic_dh.png and docs/modified_dh.png.
    :param dh_params: the list of dh parameters
    :param unknowns: the list of unknowns, unknowns[i] must be theta/d in dh_params[i]
    :return:
    """
    assert len(dh_params) == len(unknowns)
    tuple_list: List[Tuple[Optional[sp.Matrix], sp.Matrix, Optional[sp.Matrix]]] = list()
    for i in range(len(unknowns)):
        dh_param_i = dh_params[i]
        unknown_i = unknowns[i]
        tuple_i = modified_dh_transform_tuple(dh_param_i, unknown_i)
        tuple_list.append(tuple_i)

    # OK, make the result
    fk_output = ForwardKinematicsOutput(tuple_list)
    return fk_output


def twist_jacobian_dh(dh_params: List[DHEntry],
                      unknowns: List[sp.Symbol],
                      pre_transform: Optional[sp.Matrix] = None) -> sp.Matrix:
    """
    Compute the jacobian that maps joint space velocity to the twist jacobian of ee
    """
    assert len(dh_params) == len(unknowns)
    pose_to_world = sp.eye(4)
    if pre_transform is not None:
        assert pre_transform.shape[0] == 4
        assert pre_transform.shape[1] == 4
        pose_to_world = pre_transform

    jacobian = sp.zeros(6, len(unknowns))
    for i in range(len(unknowns)):
        # Compute the pose
        transform_i = modified_dh_transform(dh_params[i])
        pose_to_world = pose_to_world * transform_i

        # Compute the twist axis
        unknown_i = unknowns[i]
        twist_axis_i = sp.zeros(6, 1)
        if unknown_i == dh_params[i].theta:
            twist_axis_i[2] = sp.S.One
        elif unknown_i == dh_params[i].d:
            twist_axis_i[5] = sp.S.One
        else:
            raise NotImplementedError('Only d/theta are supported variables')

        # The twist axis in world
        world_twist_axis = transform_twist(pose_to_world, twist_axis_i)
        jacobian[:, i] = world_twist_axis

    # OK
    return jacobian


def transform_twist(transform_a_to_b: sp.Matrix, twist_in_a: sp.Matrix) -> sp.Matrix:
    assert transform_a_to_b.shape[0] == 4 and transform_a_to_b.shape[1] == 4
    assert twist_in_a.shape[0] == 6
    rotation: sp.Matrix = sp.eye(3)
    rotation = transform_a_to_b[0:3, 0:3]
    twist_in_a_omega = sp.zeros(3, 1)
    v_twist_at_a = sp.zeros(3, 1)
    for i in range(3):
        twist_in_a_omega[i] = twist_in_a[i]
        v_twist_at_a[i] = twist_in_a[i + 3]
    omega_in_b: sp.Matrix = rotation * twist_in_a_omega
    # omega_in_b: sp.Matrix = rotation * twist_in_a[0:3]
    # v_twist_at_a = twist_in_a[3:]
    a_origin_in_b = transform_a_to_b[0:3, 3]
    v_twist_in_b = - omega_in_b.cross(a_origin_in_b) + (rotation * v_twist_at_a)

    # Make the output
    twist_in_b = sp.zeros(6, 1)
    for i in range(3):
        twist_in_b[i + 0] = omega_in_b[i]
        twist_in_b[i + 3] = v_twist_in_b[i]
    # twist_in_b[0:3] = omega_in_b
    # twist_in_b[3:] = v_twist_in_b
    return twist_in_b
