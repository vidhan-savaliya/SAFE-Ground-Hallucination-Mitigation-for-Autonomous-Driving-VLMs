"""Constants and configuration defaults for SAFE-Ground."""

from typing import Dict, List

# Class safety-criticality weights lambda_cls
CLASS_SEVERITY_WEIGHTS: Dict[str, float] = {
    # Vulnerable Road Users (VRUs) - highest priority
    "pedestrian": 3.0,
    "cyclist": 3.0,
    "bicycle": 3.0,
    "motorcycle": 2.5,
    "vru": 3.0,
    
    # Dynamic Vehicles
    "car": 2.0,
    "truck": 2.0,
    "bus": 2.0,
    "trailer": 2.0,
    "construction_vehicle": 2.0,
    "vehicle": 2.0,
    
    # Dynamic / static road safety obstacles
    "traffic_cone": 1.5,
    "barrier": 1.5,
    "traffic_light": 1.5,
    "traffic_sign": 1.2,
    "debris": 1.5,
    
    # Static background / infrastructure
    "vegetation": 0.5,
    "building": 0.5,
    "static_object": 0.8,
    "other": 1.0,
}

CLASS_NAME_TO_ID: Dict[str, int] = {
    "pedestrian": 0,
    "cyclist": 1,
    "bicycle": 1,
    "motorcycle": 2,
    "car": 3,
    "truck": 4,
    "bus": 5,
    "traffic_cone": 6,
    "barrier": 7,
    "traffic_sign": 8,
    "other": 9,
}

ID_TO_CLASS_NAME: Dict[int, str] = {v: k for k, v in CLASS_NAME_TO_ID.items()}

# Distance decay parameters: f_d(d) = min(d_max / (d + eps), w_max)
DISTANCE_MAX: float = 50.0
DISTANCE_EPS: float = 1.0
DISTANCE_WEIGHT_MAX: float = 10.0

# Path relevance factors lambda_path
PATH_IN_PATH_LATERAL_THRESHOLD: float = 1.75   # meters (standard lane half-width)
PATH_ADJACENT_LATERAL_THRESHOLD: float = 5.25  # meters (adjacent lane)

PATH_WEIGHT_IN_PATH: float = 2.0
PATH_WEIGHT_ADJACENT: float = 1.5
PATH_WEIGHT_OFF_PATH: float = 1.0

# Relative velocity factor f_v(v) = 1 + beta * ||v_rel||
VELOCITY_BETA: float = 0.1

# Head A: Claim Verification Labels
CLAIM_LABEL_SUPPORTED: int = 0
CLAIM_LABEL_CONTRADICTED: int = 1
CLAIM_LABEL_UNVERIFIABLE: int = 2

LABEL_TO_CLAIM_STATUS: Dict[int, str] = {
    CLAIM_LABEL_SUPPORTED: "supported",
    CLAIM_LABEL_CONTRADICTED: "contradicted",
    CLAIM_LABEL_UNVERIFIABLE: "unverifiable",
}
CLAIM_STATUS_TO_LABEL: Dict[str, int] = {
    v: k for k, v in LABEL_TO_CLAIM_STATUS.items()
}

# Head B: Omission Labels
OMISSION_LABEL_MENTIONED: int = 0
OMISSION_LABEL_OMITTED: int = 1

# Spatial bearing sectors
BEARING_SECTORS: List[str] = [
    "front",
    "front_left",
    "front_right",
    "left",
    "right",
    "back",
    "back_left",
    "back_right",
]

# Distance bin thresholds
DISTANCE_BINS = {
    "near": (0.0, 15.0),
    "medium": (15.0, 35.0),
    "far": (35.0, 100.0),
}

# nuScenes 6 camera views
NUSCENES_CAMERAS: List[str] = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Special tokens for Claim span marking in VLM vocabulary
CLAIM_START_TOKEN: str = "<|claim_start|>"
CLAIM_END_TOKEN: str = "<|claim_end|>"

