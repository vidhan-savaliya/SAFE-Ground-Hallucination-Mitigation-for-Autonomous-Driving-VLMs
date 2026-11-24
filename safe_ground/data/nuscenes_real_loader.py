"""Real nuScenes dataset loader for SAFE-Ground.

Loads genuine multi-camera samples, ego poses, and 3D bounding boxes from
the nuScenes dataset stored at /scratch/m25csa031/nuscenes (or nuscenes_vlm_phase1).
"""

import os
import math
import numpy as np
from typing import List, Dict, Any, Optional
from pyquaternion import Quaternion

from safe_ground.data.schema import SampleData, Claim, Detection3D
from safe_ground.utils.constants import (
    CLASS_NAME_TO_ID,
    NUSCENES_CAMERAS,
)
from safe_ground.utils.geometry import get_bearing_sector, get_distance_bin, compute_distance_3d


# Mapping from nuScenes fine-grained categories to SAFE-Ground canonical classes
NUSCENES_CAT_MAP = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "vehicle.bicycle": "cyclist",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "truck",
    "vehicle.construction": "truck",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
    "movable_object.debris": "barrier",
}


class RealNuScenesLoader:
    """Loads genuine nuScenes keyframes, 3D boxes, and transforms them into SAFE-Ground samples."""

    def __init__(
        self,
        dataroot: str = "/scratch/m25csa031/nuscenes",
        version: str = "v1.0-mini",
        max_samples: Optional[int] = None,
        max_distance: float = 60.0,
        scenes_filter: Optional[set] = None,
    ):
        from nuscenes.nuscenes import NuScenes

        self.dataroot = dataroot
        self.version = version
        self.max_distance = max_distance
        self.scenes_filter = scenes_filter
        print(f"[*] Initializing NuScenes API from {dataroot} (version: {version})...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        
        if self.scenes_filter is not None:
            # Filter samples by scene name
            filtered = []
            for s in self.nusc.sample:
                sc_name = self.nusc.get("scene", s["scene_token"])["name"]
                if sc_name in self.scenes_filter:
                    filtered.append(s)
            self.samples_records = filtered if max_samples is None else filtered[:max_samples]
            print(f"[✓] Filtered {len(self.samples_records)} keyframes across {len(self.scenes_filter)} target scenes.")
        else:
            self.samples_records = self.nusc.sample if max_samples is None else self.nusc.sample[:max_samples]
            print(f"[✓] Loaded {len(self.samples_records)} real nuScenes keyframes.")

    def load_samples(self) -> List[SampleData]:
        """Parses all keyframes into SAFE-Ground SampleData instances."""
        parsed_samples = []
        total = len(self.samples_records)

        for idx, sample in enumerate(self.samples_records):
            if idx > 0 and idx % 2500 == 0:
                print(f"    - Parsed {idx}/{total} keyframes ({(idx/total)*100:.1f}%)...")
            scene = self.nusc.get("scene", sample["scene_token"])
            cam_front_token = sample["data"].get("CAM_FRONT")
            if not cam_front_token:
                continue

            cam_front_data = self.nusc.get("sample_data", cam_front_token)
            ego_pose = self.nusc.get("ego_pose", cam_front_data["ego_pose_token"])
            ego_trans = np.array(ego_pose["translation"])
            ego_rot = Quaternion(ego_pose["rotation"]).inverse

            # Collect image paths across all 6 surround cameras
            image_paths = {}
            for cam in NUSCENES_CAMERAS:
                if cam in sample["data"]:
                    c_data = self.nusc.get("sample_data", sample["data"][cam])
                    image_paths[cam] = os.path.join(self.dataroot, c_data["filename"])

            # Extract and transform 3D boxes
            detections = []
            gt_boxes = []

            for ann_token in sample["anns"]:
                ann = self.nusc.get("sample_annotation", ann_token)
                cat_name = ann["category_name"]
                canonical_cls = NUSCENES_CAT_MAP.get(cat_name)
                if not canonical_cls:
                    continue  # Skip unmapped non-obstacle categories

                box = self.nusc.get_box(ann_token)
                # Transform from global coordinates to ego-vehicle coordinates
                box.translate(-ego_trans)
                box.rotate(ego_rot)

                x, y, z = float(box.center[0]), float(box.center[1]), float(box.center[2])
                dist = compute_distance_3d(x, y, z)
                if dist > self.max_distance:
                    continue  # Beyond sensory evaluation horizon

                w, l, h = float(box.wlh[0]), float(box.wlh[1]), float(box.wlh[2])
                
                # nuScenes velocities
                vx, vy = 0.0, 0.0
                try:
                    vel = self.nusc.box_velocity(box.token)
                    if not np.isnan(vel[0]):
                        vx, vy = float(vel[0]), float(vel[1])
                except Exception:
                    pass

                cls_id = CLASS_NAME_TO_ID.get(canonical_cls, 9)

                detection = Detection3D(
                    class_name=canonical_cls,
                    class_id=cls_id,
                    x=x,
                    y=y,
                    z=z,
                    l=l,
                    w=w,
                    h=h,
                    vx=vx,
                    vy=vy,
                    confidence=0.95,  # Real annotated box
                    cam_visible=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                )
                detections.append(detection)
                gt_boxes.append({
                    "class_name": canonical_cls,
                    "x": x,
                    "y": y,
                    "z": z,
                    "vx": vx,
                    "vy": vy,
                    "vz": 0.0,
                    "is_omitted": False,
                })

            # Create baseline claims based on closest critical foreground detections
            claims = []
            # Sort detections by distance to ego vehicle
            sorted_dets = sorted(detections, key=lambda d: math.sqrt(d.x**2 + d.y**2))
            # Take top 1-4 nearest real obstacles
            for d in sorted_dets[:4]:
                dist_val = math.sqrt(d.x**2 + d.y**2)
                sector = get_bearing_sector(d.x, d.y)
                dist_bin = get_distance_bin(dist_val)
                claims.append(
                    Claim(
                        object_class=d.class_name,
                        bearing_sector=sector,
                        distance_bin=dist_bin,
                        attribute="moving" if (abs(d.vx) > 0.5 or abs(d.vy) > 0.5) else "stationary",
                        camera_view="CAM_FRONT",
                        ground_truth_status="supported",
                    )
                )

            question = f"Describe the traffic scene in scene {scene['name']} and identify critical road users."
            answer = f"The ego vehicle is driving in scene {scene['name']}. "
            if claims:
                claim_texts = [f"a {c.object_class} located {c.bearing_sector} at {c.distance_bin} distance" for c in claims]
                answer += "We observe " + ", ".join(claim_texts) + "."
            else:
                answer += "The forward roadway appears clear of immediate obstacles."

            sample_data = SampleData(
                sample_id=sample["token"][:8],
                scene_token=sample["scene_token"][:8],
                image_paths=image_paths,
                question=question,
                answer=answer,
                claims=claims,
                detections=detections,
                ground_truth_boxes=gt_boxes,
                metadata={"scene_name": scene["name"], "timestamp": sample["timestamp"]},
            )
            parsed_samples.append(sample_data)

        return parsed_samples

