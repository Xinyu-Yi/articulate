r"""
    Fit SMPL/MANO/SMPLH parametric model to target vertex position.
"""


__all__ = ['fit_vertex']


from .model import *
from .utils import *
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnums=(0,))
def error(model, pose, shape, tran, vert_gt, reg_weight=0.01):
    vert_error = (model.forward_kinematics(pose, shape, tran)[2] - vert_gt).ravel()
    shape_error = (shape - np.zeros(shape.shape)).ravel()   # regularization
    return jnp.concatenate((vert_error, shape_error * reg_weight))

@partial(jax.jit, static_argnums=(0, 6, 7, 8))
def jac(model, pose, shape, tran, vert_gt, reg_weight=0.01, optimize_pose=True, optimize_shape=True, optimize_tran=True):
    J_pose, J_shape, J_tran = jax.jacfwd(error, argnums=(1, 2, 3))(model, pose, shape, tran, vert_gt, reg_weight)
    J_pose = SO3.project_gradient(J_pose, pose).reshape((J_pose.shape[0], -1))
    return jnp.hstack((J_pose * optimize_pose, J_shape * optimize_shape, J_tran * optimize_tran))

@jax.jit
def update(pose, shape, tran, delta):
    pose = SO3.oplus(pose, delta[:-shape.shape[0]-tran.shape[0]].reshape(pose.shape[0], 3))
    shape = shape + delta[-shape.shape[0]-tran.shape[0]:-tran.shape[0]]
    tran = tran + delta[-tran.shape[0]:]
    return pose, shape, tran

@jax.jit
def mpvpe(e):
    return jnp.linalg.norm(e[:-10].reshape((-1, 3)), axis=-1).mean() * 1000  # millimeters


def fit_vertex(model: ParametricModelJax, target_vertex, 
               init_pose=None, init_shape=None, init_tran=None, 
               optimize_pose=True, optimize_shape=True, optimize_tran=True,
               reg_weight=1e-2, damp=1e-3, max_iterations=100, mpvpe_millimeter_threshold=1, verbose=True):
    r"""
    Fit SMPL/MANO/SMPLH parametric model to target vertex position.

    :param model: Parametric model to be fitted.
    :param target_vertex: Target vertex position, shape (n_vertices, 3).
    :param init_pose: Initial pose, shape (n_joints, 3, 3).
    :param init_shape: Initial shape, shape (10,).
    :param init_tran: Initial translation, shape (3,).
    :param optimize_pose: Whether to optimize pose.
    :param optimize_shape: Whether to optimize shape.
    :param optimize_tran: Whether to optimize translation.
    :param reg_weight: Regularization weight on zero shape.
    :param max_iterations: Maximum number of iterations.
    :param mpvpe_millimeter_threshold: MPVPE threshold in millimeters.
    :param verbose: Whether to print verbose information.
    :return: Optimized pose, shape, and translation.
    """
    n_joints = len(model._J)
    n_vertices = len(model._v_template)
    target_vertex = jnp.array(target_vertex).reshape((n_vertices, 3))
    pose = jnp.array(init_pose).reshape((n_joints, 3, 3)) if init_pose is not None else jnp.eye(3).reshape(1, 3, 3).repeat(n_joints, axis=0)
    shape = jnp.array(init_shape).reshape(10) if init_shape is not None else jnp.zeros(10)
    tran = jnp.array(init_tran).reshape(3) if init_tran is not None else jnp.zeros(3)

    for i in range(max_iterations):
        e = error(model, pose, shape, tran, target_vertex, reg_weight)
        J = jac(model, pose, shape, tran, target_vertex, reg_weight, optimize_pose, optimize_shape, optimize_tran)
        delta = cholesky_solve(J, e, damp)
        pose, shape, tran = update(pose, shape, tran, -delta)
        mpvpe_value = mpvpe(e)
        if verbose:
            print(f'Iter {i:4d}   mpvpe[mm]: {mpvpe_value}')
        if mpvpe_value < mpvpe_millimeter_threshold:
            break

    return np.array(pose), np.array(shape), np.array(tran)


if __name__ == '__main__':
    # copy the codes out to test the function
    import numpy as np
    from articulate.utils.jax import *

    # must use float64 type in jax
    pose = np.array(SO3.Exp(np.random.randn(24, 3)))
    shape = np.random.randn(10)
    tran = np.random.randn(3)

    # compute target vertex and init params
    model = ParametricModelJax('SMPL_male.pkl')
    target_vertex = model.forward_kinematics(pose, shape, tran)[2]
    init_pose = SO3.oplus(pose, np.random.randn(24, 3) * 0.3)
    init_shape = shape + np.random.randn(10) * 0.3
    init_tran = tran + np.random.randn(3) * 0.3

    # model fitting
    pose_est, shape_est, tran_est = fit_vertex(model, target_vertex, init_pose, init_shape, init_tran, reg_weight=0.01, mpvpe_millimeter_threshold=0.1)
    print(tran, tran_est)