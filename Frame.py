import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import open3d as o3d
from Pixels import Pixel, PixelList
from utils import fit_circle_to_pcd
from utils import fit_circle_to_depth_image
from Circles import Circle




class FeatureExtractor:
    def __init__(self, frame):
        self.frame = frame

    def extract_orb(self):
        orb = cv2.ORB.create()
        mask = np.ones(self.frame.gray.shape, dtype=np.uint8) * 255
        keypoints, descriptors = orb.detectAndCompute(self.frame.gray, mask)
        return {
            "type": "orb",
            "features": list(zip(keypoints, descriptors)),
            "frame": self.frame,
        }
    
    def extract_sift(self):
        sift = cv2.SIFT.create()
        mask = np.ones(self.frame.gray.shape, dtype=np.uint8) * 255
        keypoints, descriptors = sift.detectAndCompute(self.frame.gray, mask)
        return {
            "type": "sift",
            "features": list(zip(keypoints, descriptors)),
            "frame": self.frame,
        }
    
def visualize_features(feature_dict, frame):
    features = feature_dict["features"]
    img_with_features = cv2.drawKeypoints(
        frame.color,
        [kp for kp, _ in features],
        frame.color.copy(),
        (0, 255, 0),
        cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )
    return img_with_features

@dataclass
class Frame:
    depth_path: Path
    color_path: Path
    raw_timestamp: int
    recording_time_str: str

    def __post_init__(self):
        self.gray: np.ndarray = cv2.cvtColor(cv2.imread(str(self.color_path)), cv2.COLOR_BGR2GRAY)
        self.color: np.ndarray = cv2.cvtColor(cv2.imread(str(self.color_path)), cv2.COLOR_BGR2RGB)
        self.depth: np.ndarray = cv2.imread(str(self.depth_path), cv2.IMREAD_ANYDEPTH)
        
        feature_extractor = FeatureExtractor(self)
        self.orb = feature_extractor.extract_orb()
        self.sift = feature_extractor.extract_sift()

        self.raw_pixels: 'PixelList' = self.extract_pixels()
        self.fg_circles, self.bg_circles = self.extract_circles()


    def extract_pixels(self) -> 'PixelList':
        pixels_list = PixelList()
        max_depth = np.max(self.depth)
        min_depth = np.min(self.depth)

        mask = (self.depth > min_depth) & (self.depth < max_depth)
        v_coords, u_coords = np.where(mask)

        for u, v in zip(u_coords, v_coords):
            pixels_list.append(Pixel(self.raw_timestamp, self.recording_time_str, int(u), int(v), self.color[v, u], self.depth[v, u]))

        ref_range_pixels = pixels_list.get_pixels_in_range(min_depth=200.0, max_depth=225.0)
        depth = ref_range_pixels.create_depth_from_pixels()
        ref_center_2d, ref_radius_2d, ref_mean_depth = fit_circle_to_depth_image(depth) # type: ignore

        for pixel in pixels_list.pixels:
            if pixel.raw_depth > 200.0 and pixel.raw_depth < 225.0:
                pixel.set_background_flag(False)

            else:
                distance = np.sqrt((pixel.u - ref_center_2d[0])**2 + (pixel.v - ref_center_2d[1])**2)

                if distance < ref_radius_2d:
                    pixel.set_background_flag(True)
                else:
                    pixel.set_background_flag(False)

        return pixels_list
    
    def extract_circles(self) -> tuple[list[Circle], list[Circle]]:
        bin_size = 25
        min_depth = np.min(self.depth)
        max_depth = np.max(self.depth)
        bin_edges = np.arange(min_depth, max_depth + bin_size, bin_size)
        fg_circles = []
        bg_circles = []
        for i in range(3, len(bin_edges) - 1):
            bin_min = bin_edges[i]
            bin_max = bin_edges[i + 1]

            bin_pixels = self.raw_pixels.get_pixels_in_range(bin_min, bin_max)
            bg_pixels = bin_pixels.get_background_pixels()
            fg_pixels = bin_pixels.get_foreground_pixels()

            bg_pcd = bg_pixels.create_pcd_from_pixels()
            if len(bg_pixels.pixels) > 0 and len(np.asarray(bg_pcd.points)) > 0:
                bg_center_3d, bg_radius_3d, bg_mean_depth = fit_circle_to_pcd(bg_pcd) # type: ignore
                bg_circle = Circle(bg_center_3d, bg_radius_3d, bg_mean_depth, bg_pixels) # type: ignore
                bg_circles.append(bg_circle)
            else:
                pass

            fg_pcd = fg_pixels.create_pcd_from_pixels()
            if len(fg_pixels.pixels) > 0 and len(np.asarray(fg_pcd.points)) > 0:
                fg_center_3d, fg_radius_3d, fg_mean_depth = fit_circle_to_pcd(fg_pcd) # type: ignore
                fg_circle = Circle(fg_center_3d, fg_radius_3d, fg_mean_depth, fg_pixels) # type: ignore
                fg_circles.append(fg_circle)
            else:
                pass

        return fg_circles, bg_circles


        




