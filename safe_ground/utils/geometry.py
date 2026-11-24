"""Geometry, coordinate transforms, and spatial reasoning utilities."""

import math
from typing import Tuple
from safe_ground.utils.constants import (
    BEARING_SECTORS,
    DISTANCE_BINS,
    PATH_IN_PATH_LATERAL_THRESHOLD,
    PATH_ADJACENT_LATERAL_THRESHOLD,
    PATH_WEIGHT_IN_PATH,
    PATH_WEIGHT_ADJACENT,
    PATH_WEIGHT_OFF_PATH,
)


def compute_distance_3d(x: float, y: float, z: float = 0.0) -> float:
    """Computes Euclidean distance from ego vehicle origin (0, 0, 0)."""
    return math.sqrt(x * x + y * y + z * z)


def get_bearing_sector(x: float, y: float) -> str:
    """Computes bearing sector in ego coordinates (x: forward, y: left).
    
    Angle phi = atan2(y, x):
      0 rad: directly ahead
      pi/2 rad: directly left
      -pi/2 rad: directly right
      pi or -pi: directly behind
    """
    angle = math.atan2(y, x)  # Range [-pi, pi]
    deg = math.degrees(angle)

    if -22.5 <= deg <= 22.5:
        return "front"
    elif 22.5 < deg <= 67.5:
        return "front_left"
    elif 67.5 < deg <= 112.5:
        return "left"
    elif 112.5 < deg <= 157.5:
        return "back_left"
    elif deg > 157.5 or deg < -157.5:
        return "back"
    elif -157.5 <= deg < -112.5:
        return "back_right"
    elif -112.5 <= deg < -67.5:
        return "right"
    else:  # -67.5 <= deg < -22.5
        return "front_right"


def get_distance_bin(distance: float) -> str:
    """Maps distance to a coarse distance bin ('near', 'medium', 'far')."""
    if distance < 15.0:
        return "near"
    elif distance < 35.0:
        return "medium"
    else:
        return "far"


def get_path_relevance(y_lateral: float) -> Tuple[str, float]:
    """Computes path relevance and weighting from ego lateral offset (y)."""
    abs_y = abs(y_lateral)
    if abs_y <= PATH_IN_PATH_LATERAL_THRESHOLD:
        return "in_path", PATH_WEIGHT_IN_PATH
    elif abs_y <= PATH_ADJACENT_LATERAL_THRESHOLD:
        return "adjacent", PATH_WEIGHT_ADJACENT
    else:
        return "off_path", PATH_WEIGHT_OFF_PATH


def sector_and_dist_to_coords(sector: str, dist_bin: str) -> Tuple[float, float, float]:
    """Converts a textual sector and distance bin to approximate (x, y, z) ego coords.
    
    Used for Hungarian matching cost matrix when matching textual claims to 3D boxes.
    """
    # Nominal distance in meters
    dist_map = {
        "near": 8.0,
        "medium": 25.0,
        "far": 50.0,
    }
    r = dist_map.get(dist_bin.lower(), 20.0)

    # Angle in radians for sector center
    angle_map = {
        "front": 0.0,
        "front_left": math.radians(45.0),
        "left": math.radians(90.0),
        "back_left": math.radians(135.0),
        "back": math.radians(180.0),
        "back_right": math.radians(-135.0),
        "right": math.radians(-90.0),
        "front_right": math.radians(-45.0),
    }
    theta = angle_map.get(sector.lower(), 0.0)

    # In ego frame: x = r * cos(theta) (forward), y = r * sin(theta) (left), z = 0
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = 0.0
    return x, y, z

