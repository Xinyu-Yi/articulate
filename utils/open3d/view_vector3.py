r"""
    View 3D vectors in real-time using Open3D.
"""


__all__ = ['Vector3Viewer']


import numpy as np
import open3d as o3d


class Vector3Viewer:
    r"""
    View 3d vectors in real-time.
    """
    width = 1920
    height = 1080

    def __init__(self, line_scale=1.):
        r"""
        :param line_scale: Scale of the line length.
        """
        self.vis = None
        self.line_scale = line_scale
        self.first_update = True

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        r"""
        Connect to the viewer.
        """
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name='Vector3 Viewer', width=self.width, height=self.height)

    def disconnect(self):
        r"""
        Disconnect to the viewer.
        """
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis.close()
        self.vis = None
        self.first_update = True

    def _create_arrow(self, start, vec, color=(0.4, 0.4, 0.4)):
        r"""
        Create a 3D arrow. If length is zero, create a 3D sphere instead.
        :param start: (x, y, z).
        :param vec: (x, y, z).
        """
        start = np.array(start)
        vec = np.array(vec)
        length = np.linalg.norm(vec) * self.line_scale
        if length > 0:
            mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.04,
                cone_radius=0.06,
                cylinder_height=length - 0.16 if length > 0.32 else length * 0.5,
                cone_height=0.16 if length > 0.32 else length * 0.5
            )
            angle = np.arccos(vec[2] / np.linalg.norm(vec))
            axis = np.array([-vec[1], vec[0], 0]) if not vec[0] == vec[1] == 0 else np.array([1, 0, 0.])
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * angle)
            mesh_arrow.rotate(R, center=np.zeros(3))
            mesh_arrow.translate(start)
        else:
            mesh_arrow = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            mesh_arrow.translate(start)
        mesh_arrow.compute_vertex_normals()
        mesh_arrow.paint_uniform_color(color)
        return mesh_arrow

    def update(self, vectors, colors=None):
        r"""
        Update the viewer.

        :param vectors: 3D vectors in shape [N, 3].
        :param colors: Vector colors in shape [N, 3].
        """
        if self.vis is None:
            print('[Error] Vector3Viewer is not connected.')
            return
        self.vis.clear_geometries()
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
        for i in range(vectors.shape[0]):
            self.vis.add_geometry(self._create_arrow([0, 0, 0], vectors[i], (0.4, 0.4, 0.4) if colors is None else colors[i]), reset_bounding_box=self.first_update)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.first_update = False

    def pause(self):
        r"""
        Pause the viewer. You can control the camera during pausing. Use `q` to continue.
        """
        paused = [True]

        def key_callback(vis):
            nonlocal paused
            paused[0] = False
            return True

        self.vis.register_key_callback(81, key_callback)
        print('Vector3Viewer is paused. Press Q to continue.')
        while paused[0]:
            self.vis.poll_events()


# example
if __name__ == '__main__':
    with Vector3Viewer() as viewer:
        viewer.update(np.random.randn(5, 3))
        viewer.pause()
