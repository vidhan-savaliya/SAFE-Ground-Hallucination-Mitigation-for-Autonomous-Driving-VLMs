"""Pre-caching script for nuScenes v1.0-trainval full dataset.

Extracts all 3D annotations, ego transforms, and ground truth claims for
both official 'train' (700 scenes, ~28k keyframes) and 'val' (150 scenes, ~6k keyframes)
splits and serializes them into fast binary torch archives on /scratch.
"""

import os
import sys
import time
import json
import argparse
import torch
from collections import Counter

# Ensure package root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from safe_ground.data.nuscenes_real_loader import RealNuScenesLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-cache full nuScenes trainval dataset")
    parser.add_argument("--dataroot", type=str, default="/scratch/m25csa031/nuscenes", help="nuScenes dataset root")
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="nuScenes version (v1.0-trainval or v1.0-mini)")
    parser.add_argument("--output_dir", type=str, default="/scratch/m25csa031/safe_ground_cache", help="Directory to save cached datasets")
    parser.add_argument("--max_distance", type=float, default=60.0, help="Maximum radial distance for 3D boxes (meters)")
    parser.add_argument("--force", action="store_true", help="Force re-generation even if cache exists")
    return parser.parse_args()


def compute_statistics(samples):
    class_counts = Counter()
    total_detections = 0
    total_claims = 0
    scenes = set()

    for s in samples:
        scenes.add(s.scene_token)
        total_claims += len(s.claims)
        total_detections += len(s.detections)
        for d in s.detections:
            class_counts[d.class_name] += 1

    return {
        "num_scenes": len(scenes),
        "num_keyframes": len(samples),
        "total_claims": total_claims,
        "total_3d_detections": total_detections,
        "avg_detections_per_frame": round(total_detections / max(len(samples), 1), 2),
        "class_breakdown": dict(class_counts.most_common()),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_cache_path = os.path.join(args.output_dir, f"train_samples_{args.version}.pt")
    val_cache_path = os.path.join(args.output_dir, f"val_samples_{args.version}.pt")
    manifest_path = os.path.join(args.output_dir, f"manifest_{args.version}.json")

    if not args.force and os.path.exists(train_cache_path) and os.path.exists(val_cache_path) and os.path.exists(manifest_path):
        print(f"[✓] Full dataset cache already exists in {args.output_dir}!")
        print(f"    - Train cache: {train_cache_path}")
        print(f"    - Val cache:   {val_cache_path}")
        print(f"    - Manifest:    {manifest_path}")
        with open(manifest_path, "r") as f:
            print(json.dumps(json.load(f), indent=2))
        return

    print("=" * 75)
    print(f"[*] Starting SAFE-Ground Full Dataset Pre-Caching: {args.version}")
    print(f"    - Source Dataroot: {args.dataroot}")
    print(f"    - Target Cache Dir: {args.output_dir}")
    print(f"    - Detection Horizon: {args.max_distance}m")
    print("=" * 75)

    from nuscenes.utils.splits import create_splits_scenes
    splits = create_splits_scenes()
    
    if args.version == "v1.0-mini":
        train_split_name = "mini_train"
        val_split_name = "mini_val"
    else:
        train_split_name = "train"
        val_split_name = "val"

    train_scenes = set(splits.get(train_split_name, []))
    val_scenes = set(splits.get(val_split_name, []))
    print(f"[*] Official Split Configuration:")
    print(f"    - Train Split ('{train_split_name}'): {len(train_scenes)} scenes")
    print(f"    - Val Split   ('{val_split_name}'):   {len(val_scenes)} scenes")

    t0 = time.time()

    # 1. Cache Training Set
    print("\n[Phase 1/2] Processing and Caching TRAINING Split...")
    train_loader = RealNuScenesLoader(
        dataroot=args.dataroot,
        version=args.version,
        max_distance=args.max_distance,
        scenes_filter=train_scenes,
    )
    t_start_train = time.time()
    train_samples = train_loader.load_samples()
    print(f"[✓] Extracted {len(train_samples)} training samples in {time.time() - t_start_train:.1f}s")
    
    print(f"[*] Serializing training samples to {train_cache_path}...")
    torch.save(train_samples, train_cache_path)
    train_size_mb = os.path.getsize(train_cache_path) / (1024 * 1024)
    print(f"[✓] Saved training cache ({train_size_mb:.2f} MB)")

    # 2. Cache Validation Set (using the already initialized nusc to avoid re-reading 2.5GB JSON)
    print("\n[Phase 2/2] Processing and Caching VALIDATION Split...")
    val_records = [
        s for s in train_loader.nusc.sample
        if train_loader.nusc.get("scene", s["scene_token"])["name"] in val_scenes
    ]
    train_loader.samples_records = val_records
    print(f"[✓] Filtered {len(val_records)} validation keyframes across {len(val_scenes)} scenes.")

    t_start_val = time.time()
    val_samples = train_loader.load_samples()
    print(f"[✓] Extracted {len(val_samples)} validation samples in {time.time() - t_start_val:.1f}s")

    print(f"[*] Serializing validation samples to {val_cache_path}...")
    torch.save(val_samples, val_cache_path)
    val_size_mb = os.path.getsize(val_cache_path) / (1024 * 1024)
    print(f"[✓] Saved validation cache ({val_size_mb:.2f} MB)")

    # 3. Generate Statistics & Manifest
    train_stats = compute_statistics(train_samples)
    val_stats = compute_statistics(val_samples)

    manifest = {
        "version": args.version,
        "dataroot": args.dataroot,
        "cache_dir": args.output_dir,
        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_sec": round(time.time() - t0, 2),
        "train_cache_file": os.path.basename(train_cache_path),
        "val_cache_file": os.path.basename(val_cache_path),
        "train_statistics": train_stats,
        "val_statistics": val_stats,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 75)
    print(f"[✓] Pre-Caching Complete in {time.time() - t0:.1f} seconds!")
    print(f"    - Train Samples: {len(train_samples)} across {train_stats['num_scenes']} scenes")
    print(f"    - Val Samples:   {len(val_samples)} across {val_stats['num_scenes']} scenes")
    print(f"    - Total 3D Boxes: {train_stats['total_3d_detections'] + val_stats['total_3d_detections']:,}")
    print(f"    - Manifest Saved: {manifest_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()

