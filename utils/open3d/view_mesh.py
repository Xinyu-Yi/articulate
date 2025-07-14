r"""
    View 3D mesh in real-time using Open3D.
"""


__all__ = ['MeshViewer']


import numpy as np
import open3d as o3d


class MeshViewer:
    r"""
    View 3d mesh in real-time.
    """
    width = 1920
    height = 1080
    
    def __init__(self):
        r"""
        Initialize the MeshViewer.
        """
        self.vis = None
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
        self.vis.create_window(window_name='Mesh Viewer', width=self.width, height=self.height)

    def disconnect(self):
        r"""
        Disconnect to the viewer.
        """
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis.close()
        self.vis = None
        self.first_update = True

    def update(self, vertices_list, faces_list, colors=None):
        r"""
        Update the viewer.

        :param vertices: List of numpy arrays in shape [N, 3] for vertex coordinates of multiple meshes.
        :param faces: List of numpy arrays in shape [M, 3] for face indices of multiple meshes.
        :param colors: List of RGB float colors of multiple meshes, in range (0, 1).
        """
        if self.vis is None:
            print('[Error] MeshViewer is not connected.')
            return
        
        if len(vertices_list) != len(faces_list):
            print('[Error] vertices_list and faces_list lengths are not equal. Did you forget the batch dimension?')
            return

        if colors is None:
            colors = [np.array([0.5, 0.5, 0.5])] * len(vertices_list)

        self.vis.clear_geometries()
        for v, f, c in zip(vertices_list, faces_list, colors):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.array(v).reshape(-1, 3))
            mesh.triangles = o3d.utility.Vector3iVector(np.array(f).reshape(-1, 3))
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(c)
            self.vis.add_geometry(mesh, reset_bounding_box=self.first_update)

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
        print('MeshViewer is paused. Press Q to continue.')
        while paused[0]:
            self.vis.poll_events()


# example
if __name__ == "__main__":
    import time
    vertice = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    face = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ], dtype=np.int32)

    vertices = [vertice, vertice + (1, 0, 0)]
    faces = [face, face]
    colors = [np.array([0.5, 0, 0]), np.array([0, 0, 0.5])]


    with MeshViewer() as viewer:
        for _ in range(10):
            vertices[0] = vertices[0] + np.random.rand(*vertices[0].shape) * 0.1
            viewer.update(vertices, faces, colors)
            time.sleep(0.5)
        viewer.pause()
