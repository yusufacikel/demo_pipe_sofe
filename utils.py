import logging
from pathlib import Path
import re
from datetime import timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

INPUT_FILENAME_REGEX = re.compile(r"_(?:Color|Depth)_(\d+\.\d+)\.png")

@dataclass
class ImagePair:
    depth_path: Path
    color_path: Path
    raw_timestamp: int
    recording_time_str: str

class FileManager:
    def __init__(self, run_dir_path):
        self.run_dir_path = Path(run_dir_path)
        self.paired_files = self._load_and_pair_files()

    def _load_and_pair_files(self):
        depth_dir, color_dir = self.run_dir_path / "depth", self.run_dir_path / "color"
        if not depth_dir.exists() or not color_dir.exists():
            logger.error(f"Missing directories: {depth_dir}, {color_dir}")
            return []

        depth_paths = sorted(depth_dir.glob("*.png"), key=lambda p: self._extract_timestamp(p.name))
        color_paths = sorted(color_dir.glob("*.png"), key=lambda p: self._extract_timestamp(p.name))

        if not depth_paths or not color_paths:
            logger.warning("No images found.")
            return []

        if len(depth_paths) != len(color_paths):
            logger.error("Depth and color file counts do not match.")
            exit(1)

        start_ts = self._extract_timestamp(depth_paths[0].name)
        pairs = []

        for i, (d_path, c_path) in enumerate(zip(depth_paths, color_paths)):
            ts = self._extract_timestamp(d_path.name)
            elapsed = timedelta(milliseconds=ts - start_ts)
            recording_time_str = f"{str(elapsed)[:-3] if '.' in str(elapsed) else str(elapsed) + '.000'}"
            pair = ImagePair(
                depth_path=d_path,
                color_path=c_path,
                raw_timestamp=int(ts),
                recording_time_str=recording_time_str
            )
            pairs.append(pair)
            logger.debug(f"Pair #{i}: {pair}")

        logger.info(f"Loaded {len(pairs)} image pairs.")
        return pairs

    def _extract_timestamp(self, filename):
        match = INPUT_FILENAME_REGEX.search(filename)
        if not match:
            logger.warning(f"No timestamp in filename: {filename}")
            return 0.0
        return float(match.group(1))

    def get_pair(self, index):
        if not (0 <= index < len(self.paired_files)):
            raise IndexError("Index out of range")
        pair = self.paired_files[index]
        logger.debug(f"Retrieved pair at index {index}: {pair.depth_path}, {pair.color_path}")
        return pair

""""
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"""

WIDTH = 848
HEIGHT = 480
FOCAL_LENGTH_X = 422.068
FOCAL_LENGTH_Y = 424.824
PRINCIPAL_POINT_X = 404.892
PRINCIPAL_POINT_Y = 260.621

def calculate_xyz_from_pixel(u,v, depth):
    x = (u - PRINCIPAL_POINT_X) * depth / FOCAL_LENGTH_X
    y = (v - PRINCIPAL_POINT_Y) * depth / FOCAL_LENGTH_Y
    z = depth
    return (x, y, z)

""""
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np

class ImageVisualizer:
    def __init__(self):
        pass

    def depth(self, depth_image):
        global_min = 0.0
        global_max = 225.0

        depth_clipped = np.clip(depth_image, global_min, global_max)
        normalized = ((depth_clipped - global_min) / (global_max - global_min + 1e-8)) * 255
        img_normalized = normalized.astype(np.uint8)

        img_colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
        plt.imshow(img_colored)
        plt.axis('off')
        plt.show()

    def color(self, color_image):
        plt.imshow(color_image)
        plt.axis('off')
        plt.show()

    def depth_color_side_by_side(self, depth_image, color_image):
        global_min = 0.0
        global_max = 225.0

        depth_clipped = np.clip(depth_image, global_min, global_max)
        normalized = ((depth_clipped - global_min) / (global_max - global_min + 1e-8)) * 255
        img_normalized = normalized.astype(np.uint8)

        img_colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img_colored)
        ax[0].set_title('Depth Image')
        ax[0].axis('off')

        ax[1].imshow(color_image)
        ax[1].set_title('Color Image')
        ax[1].axis('off')

        plt.show()

    def depth_depth_side_by_side(self, depth_image_1, depth_image_2):
        global_min = 0.0
        global_max = 225.0

        depth_clipped_1 = np.clip(depth_image_1, global_min, global_max)
        normalized_1 = ((depth_clipped_1 - global_min) / (global_max - global_min + 1e-8)) * 255
        img_normalized_1 = normalized_1.astype(np.uint8)

        img_colored_1 = cv2.applyColorMap(img_normalized_1, cv2.COLORMAP_JET)

        depth_clipped_2 = np.clip(depth_image_2, global_min, global_max)
        normalized_2 = ((depth_clipped_2 - global_min) / (global_max - global_min + 1e-8)) * 255
        img_normalized_2 = normalized_2.astype(np.uint8)

        img_colored_2 = cv2.applyColorMap(img_normalized_2, cv2.COLORMAP_JET)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img_colored_1)
        ax[0].set_title('Depth Image 1')
        ax[0].axis('off')

        ax[1].imshow(img_colored_2)
        ax[1].set_title('Depth Image 2')
        ax[1].axis('off')

        plt.show()

    def color_color_side_by_side(self, color_image_1, color_image_2):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(color_image_1)
        ax[0].set_title('Color Image 1')
        ax[0].axis('off')

        ax[1].imshow(color_image_2)
        ax[1].set_title('Color Image 2')
        ax[1].axis('off')

        plt.show()

 
""""
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"""

import open3d as o3d

def generic_o3d_vis(geometries):
    if not isinstance(geometries, (list, tuple)):
        geometries = [geometries]
    vis = o3d.visualization.Visualizer() # type: ignore
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.get_render_option().background_color = np.array([0.0, 0.0, 0.0])

    view_ctl = vis.get_view_control()
    cam_params = view_ctl.convert_to_pinhole_camera_parameters()
    cam_params.extrinsic = np.eye(4)
    view_ctl.convert_from_pinhole_camera_parameters(cam_params)

    vis.run()
    vis.destroy_window()


""""
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"""

from scipy.optimize import least_squares

def fit_circle_to_depth_image(depth):

    depth_norm = cv2.normalize(depth, np.zeros_like(depth, dtype=np.uint8), 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_uint8 = depth_norm.astype(np.uint8)

    _, binary_img = cv2.threshold(depth_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_1 = np.ones((5, 5), np.uint8)
    kernel_2 = np.ones((13, 13), np.uint8)
    dilated = cv2.dilate(binary_img, kernel_2, iterations=1)
    eroded = cv2.erode(dilated, kernel_1, iterations=3)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_points = np.vstack([cnt.squeeze() for cnt in contours if cnt.shape[0] > 4])

    if all_points.shape[0] < 5:
        return None

    X, Y = all_points[:, 0], all_points[:, 1]

    def calc_R(xc, yc):
        return np.sqrt((X - xc)**2 + (Y - yc)**2)

    def cost(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    x0 = [np.mean(X), np.mean(Y)]
    res = least_squares(cost, x0=x0)

    center_2d = tuple(res.x)
    radius_2d = calc_R(*center_2d).mean()

    valid_depth_mask = depth > 0
    mean_depth = np.mean(depth[valid_depth_mask])

    return center_2d, radius_2d, mean_depth


def fit_circle_to_pcd(pcd):
    if not pcd.has_points():
        return None

    points = np.asarray(pcd.points)
    X, Y = points[:, 0], points[:, 1]

    def calc_R(xc, yc):
        return np.sqrt((X - xc)**2 + (Y - yc)**2)

    def cost(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    x0 = [np.mean(X), np.mean(Y)]
    res = least_squares(cost, x0=x0)

    center_3d = tuple(res.x)
    radius_3d = calc_R(*center_3d).mean()

    mean_depth = np.mean(points[:, 2])

    return center_3d, radius_3d, mean_depth