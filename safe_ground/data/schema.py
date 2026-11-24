"""Data models and schemas for SAFE-Ground claims, evidence, and samples."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Claim:
    """An atomic claim extracted from the VLM's response."""
    object_class: str
    bearing_sector: str
    distance_bin: str
    attribute: str = "normal"
    camera_view: str = "CAM_FRONT"
    claim_id: Optional[str] = None
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    ground_truth_status: Optional[str] = None  # "supported", "contradicted", "unverifiable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_class": self.object_class,
            "bearing_sector": self.bearing_sector,
            "distance_bin": self.distance_bin,
            "attribute": self.attribute,
            "camera_view": self.camera_view,
            "claim_id": self.claim_id,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "ground_truth_status": self.ground_truth_status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        return cls(
            object_class=str(data.get("object_class", "other")).lower(),
            bearing_sector=str(data.get("bearing_sector", "front")).lower(),
            distance_bin=str(data.get("distance_bin", "medium")).lower(),
            attribute=str(data.get("attribute", "normal")).lower(),
            camera_view=str(data.get("camera_view", "CAM_FRONT")),
            claim_id=data.get("claim_id"),
            span_start=data.get("span_start"),
            span_end=data.get("span_end"),
            ground_truth_status=data.get("ground_truth_status"),
        )


@dataclass
class Detection3D:
    """A single predicted 3D detection bounding box from BEVFusion/CenterPoint."""
    class_name: str
    class_id: int
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    confidence: float = 1.0
    cam_visible: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dt: float = 0.0
    is_omitted: Optional[bool] = None  # True if missing from VLM claims

    def to_feature_vector(self) -> List[float]:
        """Returns 17-dimensional continuous feature vector."""
        vec = [
            self.x, self.y, self.z,
            self.l, self.w, self.h,
            self.vx, self.vy, self.vz,
            self.confidence,
        ]
        # Ensure 6 camera visibility flags
        vis = self.cam_visible if len(self.cam_visible) == 6 else [1.0] * 6
        vec.extend(vis)
        vec.append(self.dt)
        return vec


@dataclass
class SampleData:
    """Complete annotated sample for training or evaluation."""
    sample_id: str
    scene_token: str
    image_paths: Dict[str, str]  # camera_name -> image path
    question: str
    answer: str
    claims: List[Claim] = field(default_factory=list)
    detections: List[Detection3D] = field(default_factory=list)
    ground_truth_boxes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

