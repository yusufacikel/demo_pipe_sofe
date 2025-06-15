from dataclasses import dataclass
from utils import calculate_xyz_from_pixel
import numpy as np
import open3d as o3d


HEIGHT = 480
WIDTH = 848

@dataclass
class PixelList:
    def __post_init__(self):
        self.pixels: list[Pixel] = []

    def append(self, pixel: 'Pixel'):
        self.pixels.append(pixel)

    def get_pixels_in_range(self, min_depth: float, max_depth: float) -> 'PixelList':
        filtered = PixelList()
        for pixel in self.pixels:
            if min_depth < pixel.raw_depth < max_depth:
                filtered.append(pixel)
        return filtered

    def get_background_pixels(self) -> 'PixelList':
        filtered = PixelList()
        for pixel in self.pixels:
            if pixel.is_bg == True:
                filtered.append(pixel)
        return filtered

    def get_foreground_pixels(self) -> 'PixelList':
        filtered = PixelList()
        for pixel in self.pixels:
            if pixel.is_bg == False:
                filtered.append(pixel)
        return filtered
    
    def create_color_from_pixels(self) -> np.ndarray:
        color_image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        for pixel in self.pixels:
            if 0 <= pixel.v < HEIGHT and 0 <= pixel.u < WIDTH:
                color_image[pixel.v, pixel.u] = pixel.color
        return color_image
    
    def create_depth_from_pixels(self) -> np.ndarray:
        depth_image = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        for pixel in self.pixels:
            if 0 <= pixel.v < HEIGHT and 0 <= pixel.u < WIDTH:
                depth_image[pixel.v, pixel.u] = pixel.raw_depth
        return depth_image
    
    def create_pcd_from_pixels(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        
        points = []
        colors = []
        for pixel in self.pixels:
            points.append(pixel.raw_position)
            colors.append(np.array(pixel.color) / 255.0)
        
        if points:
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return pcd

    
    


@dataclass
class Pixel:
    raw_timestamp: int
    recording_time_str: str
    u: int
    v: int
    color: tuple[int, int, int]
    raw_depth: float
    
    def __post_init__(self):
        self.raw_position = calculate_xyz_from_pixel(self.u, self.v, self.raw_depth)

    def set_background_flag(self, is_bg: bool) -> None:
        self.is_bg = is_bg
