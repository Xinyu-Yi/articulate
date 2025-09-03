r"""
    Utils for optimization.
"""


__all__ = ['SO3', 'cholesky_solve']


import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor, cho_solve


class SO3:
    r"""
    SO(3) utils.
    """
    @staticmethod
    @jax.jit
    def Exp(theta):
        r"""
        SO(3) exponential map.

        :param theta: axis-angle representation in shape [..., 3].
        :return: rotation matrix in shape [..., 3, 3].
        """
        angles = jnp.linalg.norm(theta, axis=-1, keepdims=True)[..., None]
        cross_product_matrix = SO3.hat(theta)
        cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix
        angles_sqrd = angles * angles
        angles_sqrd = jnp.where(angles_sqrd == 0, 1, angles_sqrd)
        R = np.eye(3) + jnp.sinc(angles / np.pi) * cross_product_matrix + ((1 - jnp.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
        return R

    @staticmethod
    @jax.jit
    def Log(R):
        r"""
        SO(3) logarithmic map.

        :param R: rotation matrix in shape [..., 3, 3].
        :return: axis-angle representation in shape [..., 3].
        """
        omegas = jnp.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], axis=-1)
        norms = jnp.linalg.norm(omegas, axis=-1, keepdims=True)
        traces = jnp.diagonal(R, axis1=-2, axis2=-1).sum(axis=-1)[..., None]
        angles = jnp.atan2(norms, traces - 1)
        omegas = jnp.where(jnp.isclose(angles, 0), np.zeros(3), omegas)
        near_pi = jnp.squeeze(jnp.isclose(angles, np.pi), axis=-1)
        n = 0.5 * (R[..., 0, :] + np.array([1, 0, 0.]))
        axis_angles = jnp.where(~near_pi[..., None], 0.5 * omegas / jnp.sinc(angles / np.pi), angles * n / jnp.linalg.norm(n, axis=-1, keepdims=True))
        return axis_angles
    
    @staticmethod
    @jax.jit
    def hat(theta):
        r"""
        SO(3) hat map.

        :param theta: vectors in shape [..., 3].
        :return: skew-symmetric matrix in shape [..., 3, 3].
        """
        rx, ry, rz, zeros = theta[..., 0], theta[..., 1], theta[..., 2], np.zeros(theta.shape[:-1])
        return jnp.stack([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=-1).reshape(theta.shape + (3,))
    
    @staticmethod
    @jax.jit
    def project_gradient(dfdR, R):
        r"""
        Project gradient to SO(3) manifold.

        :param dfdR: gradient w.r.t. rotation matrix in shape [...(nf), ...(nR), 3, 3].
        :param R: rotation matrix in shape [...(nR), 3, 3] (can broadcast to dfdR).
        :return: gradient w.r.t. right perturbation in shape [...(nf), ...(nR), 3].
        """
        dfdtheta = jnp.stack([(dfdR[..., 1] * R[..., 2]).sum(axis=-1) - (dfdR[..., 2] * R[..., 1]).sum(axis=-1),
                              (dfdR[..., 2] * R[..., 0]).sum(axis=-1) - (dfdR[..., 0] * R[..., 2]).sum(axis=-1),
                              (dfdR[..., 0] * R[..., 1]).sum(axis=-1) - (dfdR[..., 1] * R[..., 0]).sum(axis=-1)], axis=-1)
        return dfdtheta

    @staticmethod
    @jax.jit
    def oplus(R, delta):
        r"""
        SO(3) oplus.

        :param R: rotation matrix in shape [..., 3, 3].
        :param delta: right pertubation in shape [..., 3].
        :return: updated rotation matrix in shape [..., 3, 3].
        """
        return R @ SO3.Exp(delta)


@jax.jit
def cholesky_solve(J, e, damp):
    r"""
    Solve linear system (J.T @ J + damp * I) @ x = J.T @ e with Cholesky decomposition.

    :param J: Jacobian matrix in shape [m, n].
    :param e: error vector in shape [m].
    :param damp: damping factor.
    :return: solution vector in shape [n].
    """
    A = J.T @ J + damp * np.eye(J.shape[1])
    c, low = cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
    x = cho_solve((c, low), J.T @ e, overwrite_b=True, check_finite=False)
    return x
