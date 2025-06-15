from dataclasses import dataclass
from Pixels import PixelList
import cv2
import numpy as np
import open3d as o3d
from utils import generic_o3d_vis
from utils import calculate_xyz_from_pixel
from utils import fit_circle_to_pcd

@dataclass
class Circle:
    center_3d: tuple[float, float]
    radius_3d: float
    mean_depth: float
    pixel_list: 'PixelList'

    def __post_init__(self):
        self.color = self.pixel_list.create_color_from_pixels()
        self.depth = self.pixel_list.create_depth_from_pixels()
        self.pcd = self.pixel_list.create_pcd_from_pixels()
        self.line_set = self.create_circle_line_set()

    def create_circle_line_set(self) -> o3d.geometry.LineSet:
        num_points = 100
        angles = np.linspace(0, 2 * np.pi, num_points)
        x = self.center_3d[0] + self.radius_3d * np.cos(angles)
        y = self.center_3d[1] + self.radius_3d * np.sin(angles)
        z = np.full_like(x, self.mean_depth)
        points = np.vstack((x, y, z)).T
        lines = [[i, (i + 1) % num_points] for i in range(num_points)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1, 0, 0])
        return line_set
    
    def visualize_o3d(self):
        geometries = [self.pcd, self.line_set]
        generic_o3d_vis(geometries)

    def get_geometries(self):
        return [self.pcd, self.line_set]

    
