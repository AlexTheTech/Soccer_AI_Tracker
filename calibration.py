from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class PitchCalibration:
    pixel_points: Dict[str, Tuple[float, float]]
    homography: np.ndarray


def compute_homography(pixel_points: Dict[str, Tuple[float, float]],
                       pitch_length_m: float,
                       pitch_width_m: float) -> np.ndarray:
    required = {"top_left", "top_right", "bottom_right", "bottom_left"}
    if set(pixel_points.keys()) != required:
        missing = required - set(pixel_points.keys())
        raise ValueError(f"Missing calibration points: {missing}")

    src = np.array([
        pixel_points["top_left"],
        pixel_points["top_right"],
        pixel_points["bottom_right"],
        pixel_points["bottom_left"],
    ], dtype=np.float32)

    dst = np.array([
        [0.0, 0.0],
        [pitch_length_m, 0.0],
        [pitch_length_m, pitch_width_m],
        [0.0, pitch_width_m],
    ], dtype=np.float32)

    homography, _ = cv2.findHomography(src, dst)
    if homography is None:
        raise RuntimeError("Failed to compute homography")
    return homography


def pixel_to_world(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    pts = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), homography)
    return pts.reshape(-1, 2)
