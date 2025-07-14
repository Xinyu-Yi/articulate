r"""
    View 3D point cloud in real-time using Open3D.
"""


__all__ = ['PointCloud3DViewer']


import open3d as o3d


class PointCloud3DViewer:
    r"""
    View 3d point cloud in real-time.
    """
    width = 1920
    height = 1080

    def __init__(self, point_scale=5.):
        r"""
        :param point_scale: Scale of the points.
        """
        self.vis = None
        self.pc = None
        self.point_scale = point_scale

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        r"""
        Connect to the viewer.
        """
        self.pc = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name='PointCloud3D Viewer', width=self.width, height=self.height)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
        self.vis.add_geometry(self.pc)
        self.vis.get_render_option().point_size = self.point_scale

    def disconnect(self):
        r"""
        Disconnect to the viewer.
        """
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis.close()
        self.vis = None
        self.pc = None

    def update(self, points, colors=None, reset_view_point=False):
        r"""
        Update the viewer.

        :param points: Point cloud in shape [N, 3].
        :param colors: Point color in shape [N, 3].
        :param reset_view_point: Whether to reset the camera view to capture all the points.
        """
        if self.vis is None:
            print('[Error] PointCloud3DViewer is not connected.')
            return
        self.pc.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            self.pc.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.pc)
        self.vis.reset_view_point(reset_view_point)
        self.vis.poll_events()
        self.vis.update_renderer()

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
        print('PointCloud3DViewer is paused. Press Q to continue.')
        while paused[0]:
            self.vis.poll_events()


# example
if __name__ == '__main__':
    import numpy as np
    with PointCloud3DViewer() as viewer:
        while True:
            viewer.update(np.random.randn(1000, 3), reset_view_point=True)
            viewer.pause()
