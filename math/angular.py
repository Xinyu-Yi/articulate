r"""
    Angular math utils that contain calculations of angles.
"""


__all__ = ['RotationRepresentation', 'to_rotation_matrix', 'radian_to_degree', 'degree_to_radian', 'normalize_angle',
           'angle_difference', 'angle_between', 'angle_from_two_vectors', 'svd_rotate', 'generate_random_rotation_matrix',
           'axis_angle_to_rotation_matrix', 'rotation_matrix_to_axis_angle', 'r6d_to_rotation_matrix',
           'rotation_matrix_to_r6d', 'quaternion_to_axis_angle', 'axis_angle_to_quaternion',
           'quaternion_to_rotation_matrix', 'rotation_matrix_to_euler_angle', 'euler_angle_to_rotation_matrix',
           'rotation_matrix_to_euler_angle_np', 'euler_angle_to_rotation_matrix_np', 'euler_convert_np',
           'quaternion_product', 'quaternion_inverse', 'quaternion_mean', 'generate_random_rotation_matrix_constrained',
           'quaternion_mean_robust', 'normalize_rotation_matrix', 'from_to_rotation_matrix']


from .general import *
import enum
import numpy as np
import torch


class RotationRepresentation(enum.Enum):
    r"""
    Rotation representations. Quaternions are in wxyz. Euler angles are in local XYZ.
    """
    AXIS_ANGLE = 0
    ROTATION_MATRIX = 1
    QUATERNION = 2
    R6D = 3
    EULER_ANGLE = 4


def to_rotation_matrix(r: torch.Tensor, rep: RotationRepresentation):
    r"""
    Convert any rotations into rotation matrices. (torch, batch)

    :param r: Rotation tensor.
    :param rep: The rotation representation used in the input.
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    if rep == RotationRepresentation.AXIS_ANGLE:
        return axis_angle_to_rotation_matrix(r)
    elif rep == RotationRepresentation.QUATERNION:
        return quaternion_to_rotation_matrix(r)
    elif rep == RotationRepresentation.R6D:
        return r6d_to_rotation_matrix(r)
    elif rep == RotationRepresentation.EULER_ANGLE:
        return euler_angle_to_rotation_matrix(r)
    elif rep == RotationRepresentation.ROTATION_MATRIX:
        return r.view(-1, 3, 3)
    else:
        raise Exception('unknown rotation representation')


def normalize_rotation_matrix(R: torch.Tensor, check_det=True):
    r"""
    Normalize rotation matrices using svd. (torch, batch)

    :param R: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :param check_det: Check if the determinant is 1.
    :return: Normalized rotation matrix tensor of shape [batch_size, 3, 3].
    """
    R = R.view(-1, 3, 3)
    U, S, V = torch.svd(R)
    R = U.bmm(V.transpose(1, 2))
    if check_det:
        m = R.det() < 0
        R[m] = U[m].matmul(torch.diag(torch.tensor([1, 1, -1.], device=R.device))).bmm(V[m].transpose(1, 2))
    return R


def radian_to_degree(q):
    r"""
    Convert radians to degrees.
    """
    return q * (180.0 / np.pi)


def degree_to_radian(q):
    r"""
    Convert degrees to radians.
    """
    return q * (np.pi / 180.0)


def quaternion_mean(q):
    r"""
    Calculate the mean quaternion. (torch)

    Warning: Input quaternions should be very close to each other.

    :param q: Tensor [N, 4].
    :return: Tensor [4].
    """
    q = q.view(-1, 4)
    q = q * q[:, int(q.abs().mean(dim=0).argmax())].sign().view(-1, 1).expand(-1, 4)
    return normalize_tensor(q.mean(dim=0))


def quaternion_mean_robust(q, w=None):
    r"""
    Calculate the (weighted) average quaternion. (torch)

    This is more robust than quaternion_mean(), but slower.
    Reference: https://ntrs.nasa.gov/api/citations/20070017872/downloads/20070017872.pdf

    :param q: Tensor [N, 4].
    :param w: Tensor [N] for quaternion weights.
    :return: Tensor [4].
    """
    q = q.view(-1, 4)
    w = torch.ones_like(q[:, 0]) / q.shape[0] if w is None else w.view(-1) / w.sum()
    m = (q.unsqueeze(-1).bmm(q.unsqueeze(-2)) * w.view(-1, 1, 1)).sum(dim=0)
    q = torch.symeig(m, eigenvectors=True).eigenvectors[:, -1]
    return q


def quaternion_product(q1, q2):
    r"""
    Multiply two quaternions. (torch, batch)
    Quaternion in w, x, y, z.

    :param q1: Tensor [N, 4].
    :param q2: Tensor [N, 4].
    :return: Tensor [N, 4].
    """
    w1, xyz1 = q1.view(-1, 4)[:, :1], q1.view(-1, 4)[:, 1:]
    w2, xyz2 = q2.view(-1, 4)[:, :1], q2.view(-1, 4)[:, 1:]
    xyz = torch.cross(xyz1, xyz2, dim=-1) + w1 * xyz2 + w2 * xyz1
    w = w1 * w2 - (xyz1 * xyz2).sum(dim=1, keepdim=True)
    q = torch.cat((w, xyz), dim=1).view_as(q1)
    return q


def quaternion_inverse(q):
    r"""
    Inverse a quaternion. (torch, batch)
    Quaternion in w, x, y, z.

    :param q: Tensor [N, 4].
    :return: Tensor [N, 4].
    """
    invq = q.clone().view(-1, 4)
    invq[:, 1:].neg_()
    return invq.view_as(q)


def normalize_angle(q):
    r"""
    Normalize radians into [-pi, pi). (np/torch, batch)

    :param q: A tensor (np/torch) of angles in radians.
    :return: The normalized tensor where each angle is in [-pi, pi).
    """
    mod = q % (2 * np.pi)
    mod[mod >= np.pi] -= 2 * np.pi
    return mod


def angle_difference(target, source):
    r"""
    Calculate normalized target - source. (np/torch, batch)
    """
    return normalize_angle(target - source)


def angle_between(rot1: torch.Tensor, rot2: torch.Tensor, rep=RotationRepresentation.ROTATION_MATRIX):
    r"""
    Calculate the angle in radians between two rotations. (torch, batch)

    :param rot1: Rotation tensor 1 that can reshape to [batch_size, rep_dim].
    :param rot2: Rotation tensor 2 that can reshape to [batch_size, rep_dim].
    :param rep: The rotation representation used in the input.
    :return: Tensor in shape [batch_size] for angles in radians.
    """
    rot1 = to_rotation_matrix(rot1, rep)
    rot2 = to_rotation_matrix(rot2, rep)
    offsets = rot1.transpose(1, 2).bmm(rot2)
    angles = rotation_matrix_to_axis_angle(offsets).norm(dim=1)
    return angles


def angle_from_two_vectors(v1, v2, signed=False):
    r"""
    Calculate the angle between two vectors. (torch, batch)

    If signed is False, return radians in [0, pi].
    If signed is True, return radians in [-pi, pi].

    :param v1: Tensor that can reshape to [batch_size, 3].
    :param v2: Tensor that can reshape to [batch_size, 3].
    :param signed: If True, return signed angles.
    :return: Angles in shape [batch_size] in radians.
    """
    v1 = normalize_tensor(v1.view(-1, 3))
    v2 = normalize_tensor(v2.view(-1, 3))
    angle = (v1 * v2).sum(dim=-1).clip(-1, 1).acos()
    if signed:
        cross_product = torch.cross(v1, v2, dim=-1)
        angle[cross_product[:, 2] < 0] = -angle[cross_product[:, 2] < 0]
    return angle


def from_to_rotation_matrix(from_vector: torch.Tensor, to_vector: torch.Tensor):
    r"""
    Get the rotation matrix that rotates from one vector to another. R * from_vector = to_vector. (torch, batch)

    :param from_vector: From vector that can reshape to [batch_size, 3].
    :param to_vector: To vector that can reshape to [batch_size, 3].
    :return: Rotation matrix in shape [batch_size, 3, 3].
    """
    from_vector = normalize_tensor(from_vector.view(-1, 3))
    to_vector = normalize_tensor(to_vector.view(-1, 3))
    axis = normalize_tensor(torch.cross(from_vector, to_vector, dim=-1))
    if torch.isnan(axis).any():
        axis_alt = normalize_tensor(torch.cross(torch.randn_like(from_vector), to_vector, dim=-1))
        axis[torch.isnan(axis)] = axis_alt[torch.isnan(axis)]
    angle = (from_vector * to_vector).sum(dim=-1, keepdim=True).clip(-1, 1).acos()
    return axis_angle_to_rotation_matrix(axis * angle)


def svd_rotate(source_points: torch.Tensor, target_points: torch.Tensor, calc_R=True, calc_t=False, calc_s=False):
    r"""
    Get the rotation/translation/scale that transform source points to the corresponding target points. (torch, batch)
    i.e., min || s * R * source_points + t - target_points || ^ 2.
    Note: maybe not exactly the mathematical global minimum. I'm not sure.

    :param source_points: Source points in shape [batch_size, m, n]. m is the number of the points. n is the dim.
    :param target_points: Target points in shape [batch_size, m, n]. m is the number of the points. n is the dim.
    :param calc_R: Calculate rotation. If false, fix R=I.
    :param calc_t: Calculate translation. If false, fix t=0.
    :param calc_s: Calculate scale. If false, fix s=1.
    :return: Rotation R in shape [batch_size, n, n], translation t in shape [batch_size, n],
             and scale s in shape [batch_size] that transform source points to target points,
             and the transformed source points (s * R * source_points + t) in shape [batch_size, m, n].
    """
    source_points_mean = source_points.mean(dim=1, keepdim=True) if calc_t else torch.zeros_like(source_points[:, :1])
    target_points_mean = target_points.mean(dim=1, keepdim=True) if calc_t else torch.zeros_like(target_points[:, :1])

    if calc_s:
        source_points_rms = ((source_points - source_points_mean) * (source_points - source_points_mean)).sum(dim=[1, 2])
        target_points_rms = ((target_points - target_points_mean) * (target_points - target_points_mean)).sum(dim=[1, 2])
        scale = (target_points_rms / source_points_rms).sqrt()
    else:
        scale = torch.ones_like(source_points[:, 0, 0])

    if calc_R:
        usv = [m.svd() for m in (source_points - source_points_mean).transpose(1, 2).bmm(target_points - target_points_mean)]
        u = torch.stack([_[0] for _ in usv])
        v = torch.stack([_[2] for _ in usv])
        vut = v.bmm(u.transpose(1, 2))
        for i in range(vut.shape[0]):
            if vut[i].det() < -0.9:
                v[i, 2].neg_()
                vut[i] = v[i].mm(u[i].t())
        rotation = vut
    else:
        rotation = torch.eye(source_points.shape[2]).repeat(source_points.shape[0], 1, 1).to(source_points.device)

    translation = -scale.view(-1, 1, 1) * rotation.bmm(source_points_mean.transpose(1, 2)) + target_points_mean.transpose(1, 2)
    transformed_source_points = scale.view(-1, 1, 1) * source_points.bmm(rotation.transpose(1, 2)) + translation.transpose(1, 2)
    return rotation, translation.squeeze(2), scale, transformed_source_points


def generate_random_rotation_matrix(n=1):
    r"""
    Generate random rotation matrices. (torch, batch)

    :param n: Number of rotation matrices to generate.
    :return: Random rotation matrices of shape [n, 3, 3].
    """
    q = torch.zeros(n, 4)
    while True:
        n = q.norm(dim=1)
        mask = (n == 0) | (n > 1)
        if q[mask].shape[0] == 0:
            break
        q[mask] = torch.rand_like(q[mask]) * 2 - 1
    q = q / q.norm(dim=1, keepdim=True)
    return quaternion_to_rotation_matrix(q)


def generate_random_rotation_matrix_constrained(n=1, y=(-180, 180), p=(-90, 90), r=(-180, 180)):
    r"""
    Generate random rotation matrices with rpy constraints. Rotation in local Y(y)-X(p)-Z(r) order. (torch, batch)

    :param n: Number of rotation matrices to generate.
    :param y: Yaw range in degrees.
    :param p: Pitch range in degrees.
    :param r: Roll range in degrees.
    :return: Random rotation matrices of shape [n, 3, 3].
    """
    ry = degree_to_radian(lerp(y[0], y[1], torch.rand(n)))
    rp = degree_to_radian(lerp(p[0], p[1], torch.rand(n)))
    rr = degree_to_radian(lerp(r[0], r[1], torch.rand(n)))
    return euler_angle_to_rotation_matrix(torch.stack((ry, rp, rr), dim=1), seq='YXZ')


def axis_angle_to_rotation_matrix(a: torch.Tensor):
    r"""
    Turn axis-angles into rotation matrices. (torch, batch)

    :param a: Axis-angle tensor that can reshape to [batch_size, 3].
    :return: Rotation matrix of shape [batch_size, 3, 3].
    """
    axis, angle = normalize_tensor(a.view(-1, 3), return_norm=True)
    axis[torch.isnan(axis) | torch.isinf(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * hat(axis)
    return r


def rotation_matrix_to_axis_angle(r: torch.Tensor):
    r"""
    Turn rotation matrices into axis-angles. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :return: Axis-angle tensor of shape [batch_size, 3].
    """
    import cv2
    result = [cv2.Rodrigues(_)[0] for _ in r.clone().detach().cpu().view(-1, 3, 3).numpy()]
    result = torch.from_numpy(np.stack(result)).float().squeeze(-1).to(r.device)
    return result


def r6d_to_rotation_matrix(r6d: torch.Tensor):
    r"""
    Turn 6D vectors into rotation matrices. (torch, batch)

    **Warning:** The two 3D vectors of any 6D vector must be linearly independent.

    :param r6d: 6D vector tensor that can reshape to [batch_size, 6].
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    r6d = r6d.view(-1, 6)
    column0 = normalize_tensor(r6d[:, 0:3])
    column1 = normalize_tensor(r6d[:, 3:6] - (column0 * r6d[:, 3:6]).sum(dim=1, keepdim=True) * column0)
    column2 = column0.cross(column1, dim=1)
    r = torch.stack((column0, column1, column2), dim=-1)
    r[torch.isnan(r)] = 0
    return r


def rotation_matrix_to_r6d(r: torch.Tensor):
    r"""
    Turn rotation matrices into 6D vectors. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :return: 6D vector tensor of shape [batch_size, 6].
    """
    return r.view(-1, 3, 3)[:, :, :2].transpose(1, 2).contiguous().clone().view(-1, 6)


def quaternion_to_axis_angle(q: torch.Tensor):
    r"""
    Turn (unnormalized) quaternions wxyz into axis-angles. (torch, batch)

    **Warning**: The returned axis angles may have a rotation larger than 180 degrees (in 180 ~ 360).

    :param q: Quaternion tensor that can reshape to [batch_size, 4].
    :return: Axis-angle tensor of shape [batch_size, 3].
    """
    q = normalize_tensor(q.view(-1, 4))
    q[q[:, 0] < 0] = -q[q[:, 0] < 0]
    theta_half = q[:, 0].clamp(min=-1, max=1).acos()
    a = (q[:, 1:] / theta_half.sin().view(-1, 1) * 2 * theta_half.view(-1, 1)).view(-1, 3)
    a[theta_half < 1e-6] = q[theta_half < 1e-6][:, 1:] * 2
    return a


def axis_angle_to_quaternion(a: torch.Tensor):
    r"""
    Turn axis-angles into quaternions. (torch, batch)

    :param a: Axis-angle tensor that can reshape to [batch_size, 3].
    :return: Quaternion wxyz tensor of shape [batch_size, 4].
    """
    axes, angles = normalize_tensor(a.view(-1, 3), return_norm=True)
    axes[torch.isnan(axes)] = 0
    q = torch.cat(((angles / 2).cos(), (angles / 2).sin() * axes), dim=1)
    return q


def quaternion_to_rotation_matrix(q: torch.Tensor):
    r"""
    Turn (unnormalized) quaternions wxyz into rotation matrices. (torch, batch)

    :param q: Quaternion tensor that can reshape to [batch_size, 4].
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    q = normalize_tensor(q.view(-1, 4))
    a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    r = torch.cat((- 2 * c * c - 2 * d * d + 1, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                   2 * b * c + 2 * a * d, - 2 * b * b - 2 * d * d + 1, 2 * c * d - 2 * a * b,
                   2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, - 2 * b * b - 2 * c * c + 1), dim=1)
    return r.view(-1, 3, 3)


def rotation_matrix_to_euler_angle(r: torch.Tensor, seq='XYZ'):
    r"""
    Turn rotation matrices into euler angles. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :param seq: 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
                rotations, or {'x', 'y', 'z'} for extrinsic rotations (radians).
                See scipy for details.
    :return: Euler angle tensor of shape [batch_size, 3].
    """
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(r.clone().detach().cpu().view(-1, 3, 3).numpy())
    ret = torch.from_numpy(rot.as_euler(seq)).float().to(r.device)
    return ret


def euler_angle_to_rotation_matrix(q: torch.Tensor, seq='XYZ'):
    r"""
    Turn euler angles into rotation matrices. (torch, batch)

    :param q: Euler angle tensor that can reshape to [batch_size, 3].
    :param seq: 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
                rotations, or {'x', 'y', 'z'} for extrinsic rotations (radians).
                See scipy for details.
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler(seq, q.clone().detach().cpu().view(-1, 3).numpy())
    ret = torch.from_numpy(rot.as_matrix()).float().to(q.device)
    return ret


def rotation_matrix_to_euler_angle_np(r, seq='XYZ'):
    r"""
    Turn rotation matrices into euler angles. (numpy, batch)

    :param r: Rotation matrix (np/torch) that can reshape to [batch_size, 3, 3].
    :param seq: 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
                rotations, or {'x', 'y', 'z'} for extrinsic rotations (radians).
                See scipy for details.
    :return: Euler angle ndarray of shape [batch_size, 3].
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.array(r).reshape(-1, 3, 3)).as_euler(seq)


def euler_angle_to_rotation_matrix_np(q, seq='XYZ'):
    r"""
    Turn euler angles into rotation matrices. (numpy, batch)

    :param q: Euler angle (np/torch) that can reshape to [batch_size, 3].
    :param seq: 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
                rotations, or {'x', 'y', 'z'} for extrinsic rotations (radians).
                See scipy for details.
    :return: Rotation matrix ndarray of shape [batch_size, 3, 3].
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler(seq, np.array(q).reshape(-1, 3)).as_matrix()


def euler_convert_np(q, from_seq='XYZ', to_seq='XYZ'):
    r"""
    Convert euler angles into different axis orders. (numpy, single/batch)

    :param q: An ndarray of euler angles (radians) in from_seq order. Shape [3] or [N, 3].
    :param from_seq: The source(input) axis order. See scipy for details.
    :param to_seq: The target(output) axis order. See scipy for details.
    :return: An ndarray with the same size but in to_seq order.
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler(from_seq, q).as_euler(to_seq)
