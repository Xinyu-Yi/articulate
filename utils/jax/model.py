r"""
    SMPL/MANO/SMPLH parametric model implemented in JAX. Used for model fitting.
"""


__all__ = ['ParametricModelJax']


import pickle
import numpy as np
import jax 
import jax.numpy as jnp
from functools import partial


class ParametricModelJax:
    r"""
    SMPL/MANO/SMPLH parametric model implemented in JAX.
    """
    def __init__(self, official_model_file: str):
        r"""
        Init an SMPL/MANO/SMPLH parametric model.

        :param official_model_file: Path to the official model to be loaded.
        """
        with open(official_model_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self._J_regressor = data['J_regressor'].toarray()
        self._skinning_weights = data['weights']
        self._posedirs = data['posedirs']
        self._shapedirs = np.array(data['shapedirs'])
        self._v_template = data['v_template']
        self._J = data['J']
        self.face = data['f']
        self.parent = np.array(data['kintree_table'][0].tolist())  # warning: parent[0] is arbitrary

    @staticmethod
    @jax.jit
    def _get_T(R, p):
        r"""
        return [[ R  p ]
                [ 0  1 ]]
        """
        T34 = jnp.concatenate((R.reshape((-1, 3, 3)), p.reshape((-1, 3, 1))), axis=-1)
        T44 = jnp.concatenate((T34, np.tile(np.array([0, 0, 0, 1.]), (R.shape[0], 1, 1))), axis=1)
        return T44
     
    @staticmethod
    @jax.jit
    def _get_p(p):
        r"""
        return [ p  1 ]
        """
        p3 = p.reshape((-1, 3))
        p4 = jnp.concatenate((p3, np.ones((p3.shape[0], 1))), axis=-1)
        return p4
       
    @partial(jax.jit, static_argnums=(0,))
    def forward_kinematics(self, pose: jax.Array, shape: jax.Array, tran: jax.Array):
        r"""
        Forward kinematics that computes the global joint rotation, joint position, and
        mesh vertex position from pose, shape, and translation.

        :param pose: Joint local rotation in shape [num_joint, 3, 3].
        :param shape: Shape parameter in shape [10].
        :param tran: Root position in shape [3].
        :return: Joint global rotation in [num_joint, 3, 3],
                 joint position in [num_joint, 3],
                 and mesh vertex position in [num_vertex, 3].
        """
        v = self._v_template + jnp.einsum('k, ijk -> ij', shape, self._shapedirs)
        j = self._J_regressor @ v
        T_local = self._get_T(pose, jnp.concatenate((tran + j[:1] * 0, j[1:] - j[self.parent[1:]]), axis=0))  # remove *0 if root joint position normaization is not needed
        T_global = [T_local[0]]
        for i in range(1, len(self.parent)):
            T_global.append(T_global[self.parent[i]] @ T_local[i])
        T_global = jnp.stack(T_global)
        pose_global, joint_global = T_global[:, :3, :3], T_global[:, :3, 3]
        v = v + jax.lax.stop_gradient(jnp.einsum('k, ijk -> ij', (pose[1:] - np.eye(3)).ravel(), self._posedirs))  # disable gradients on pose blendshapes
        T_temp = self._get_T(pose_global, joint_global - jnp.einsum('ijk, ik -> ij', pose_global, j))
        T_vertex = jnp.einsum('ijk, ni -> njk', T_temp, self._skinning_weights)
        vertex_global = jnp.einsum('ijk, ik -> ij', T_vertex, self._get_p(v))[:, :3]
        return pose_global, joint_global, vertex_global

    def save_ply_mesh(self, verts, fname='a.ply'):
        r"""
        Export mesh using the input vertex position.

        :param verts: Vertex position in shape [num_vertex, 3].
        :param fname: Output file name.
        """
        import trimesh
        mesh = trimesh.Trimesh(verts, self.face)
        mesh.export(fname)


