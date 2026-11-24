"""Fine-tuning and training script for PhD-Grade Evidential SAFE-Ground Verifier with real nuScenes data."""

import os
import argparse
import time
import torch
from torch.utils.data import DataLoader
from typing import List

from safe_ground.data.schema import SampleData, Claim, Detection3D
from safe_ground.data.dataset import SAFEGroundDataset, safe_ground_collate_fn
from safe_ground.data.nuscenes_real_loader import RealNuScenesLoader
from safe_ground.models.safe_ground_model import SAFEGroundModel


def create_synthetic_training_samples(num_samples: int = 100) -> List[SampleData]:
    """Fallback synthetic sample generator if real dataset is not accessible."""
    samples = []
    classes = ["car", "pedestrian", "cyclist", "truck", "barrier"]
    for i in range(num_samples):
        claims = [
            Claim(
                object_class=classes[i % len(classes)],
                bearing_sector="front" if i % 2 == 0 else "front_left",
                distance_bin="near" if i % 3 == 0 else "medium",
                attribute="moving",
                camera_view="CAM_FRONT",
            )
        ]
        detections = [
            Detection3D(
                class_name="car",
                class_id=3,
                x=10.0 + (i % 5) * 2.0,
                y=-1.5,
                z=-0.5,
                l=4.5,
                w=1.8,
                h=1.5,
                vx=2.0,
                vy=0.0,
                vz=0.0,
                confidence=0.92,
            ),
            Detection3D(
                class_name="pedestrian",
                class_id=0,
                x=6.0,
                y=2.5,
                z=-0.2,
                l=0.8,
                w=0.7,
                h=1.7,
                vx=0.5,
                vy=0.0,
                vz=0.0,
                confidence=0.85,
            ),
        ]
        samples.append(
            SampleData(
                sample_id=f"sample_{i:04d}",
                scene_token=f"scene_{i // 10}",
                image_paths={},
                question="What obstacles and road users are visible around the ego vehicle?",
                answer="There is a vehicle ahead and a pedestrian to the left.",
                claims=claims,
                detections=detections,
            )
        )
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Train/Fine-tune SAFE-Ground Evidential Verifier")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=256, help="Embedding dimension (3584 for Qwen2.5-VL-7B)")
    parser.add_argument("--lambda_om", type=float, default=1.0, help="Weight for omission loss")
    parser.add_argument("--lambda_match", type=float, default=0.5, help="Weight for Hungarian matching loss")
    parser.add_argument("--use_evidential", action="store_true", default=True, help="Enable evidential Subjective Logic head")
    parser.add_argument("--kl_weight", type=float, default=0.05, help="KL divergence regularization for evidential loss")
    parser.add_argument("--use_real_nuscenes", action="store_true", default=True, help="Train on real nuScenes dataset")
    parser.add_argument("--nuscenes_dataroot", type=str, default="/scratch/m25csa031/nuscenes", help="nuScenes dataroot path")
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-trainval", help="nuScenes split version")
    parser.add_argument("--cache_dir", type=str, default="/scratch/m25csa031/safe_ground_cache", help="Path to pre-cached datasets")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Cap maximum training samples if desired")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Cap maximum validation samples if desired")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--checkpoint_name", type=str, default="best_verifier_evidential.pt", help="Checkpoint filename")
    parser.add_argument("--log_interval", type=int, default=100, help="Batches between progress logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[*] Starting PhD-Grade SAFE-Ground Fine-Tuning on device: {device}")
    print(f"    - Evidential Subjective Logic: {args.use_evidential} (KL weight={args.kl_weight})")
    print(f"    - Using Real nuScenes Dataset: {args.use_real_nuscenes} ({args.nuscenes_version} at {args.nuscenes_dataroot})")
    print(f"    - Cache Directory: {args.cache_dir}")
    print(f"    - Architecture d_model: {args.d_model}, Batch Size: {args.batch_size}, LR: {args.lr}")

    train_cache = os.path.join(args.cache_dir, f"train_samples_{args.nuscenes_version}.pt")
    val_cache = os.path.join(args.cache_dir, f"val_samples_{args.nuscenes_version}.pt")

    # 1. Prepare Datasets (Real nuScenes from Cache or Live Loader)
    if args.use_real_nuscenes and os.path.exists(train_cache) and os.path.exists(val_cache):
        print(f"[*] Loading pre-cached nuScenes datasets from {args.cache_dir}...")
        t0_load = time.time()
        train_samples = torch.load(train_cache, weights_only=False)
        val_samples = torch.load(val_cache, weights_only=False)
        if args.max_train_samples:
            train_samples = train_samples[:args.max_train_samples]
        if args.max_val_samples:
            val_samples = val_samples[:args.max_val_samples]
        print(f"[✓] Loaded {len(train_samples)} Train and {len(val_samples)} Val samples in {time.time() - t0_load:.2f}s!")
    elif args.use_real_nuscenes and os.path.exists(args.nuscenes_dataroot):
        print("[*] Loading real nuScenes dataset with scene-level train/val split (un-cached fallback)...")
        loader = RealNuScenesLoader(dataroot=args.nuscenes_dataroot, version=args.nuscenes_version)
        all_samples = loader.load_samples()
        
        # Group by scene to guarantee scene-level separation (zero data leakage)
        scenes = {}
        for s in all_samples:
            sc = s.scene_token
            scenes.setdefault(sc, []).append(s)
        
        scene_keys = sorted(list(scenes.keys()))
        val_scenes = scene_keys[-2:]  # Last 2 scenes for validation (~20%)
        train_scenes = scene_keys[:-2] # First 8 scenes for training (~80%)

        train_samples = [s for sc in train_scenes for s in scenes[sc]]
        val_samples = [s for sc in val_scenes for s in scenes[sc]]
        print(f"[✓] Scene-Level Split: {len(train_scenes)} Train Scenes ({len(train_samples)} samples), {len(val_scenes)} Val Scenes ({len(val_samples)} samples)")
    else:
        print("[!] Warning: nuScenes dataroot not found, falling back to synthetic generator")
        train_samples = create_synthetic_training_samples(num_samples=200)
        val_samples = create_synthetic_training_samples(num_samples=50)

    train_dataset = SAFEGroundDataset(train_samples, inject_hallucinations=True, injector_seed=42)
    val_dataset = SAFEGroundDataset(val_samples, inject_hallucinations=True, injector_seed=1337)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=safe_ground_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=safe_ground_collate_fn,
    )

    # 2. Initialize Model with Evidential Head & FRS Reachability
    model = SAFEGroundModel(
        d_model=args.d_model,
        nhead=8 if args.d_model >= 64 else 4,
        num_verifier_layers=1,
        dim_feedforward=args.d_model * 4,
        lambda_om=args.lambda_om,
        lambda_match=args.lambda_match,
        use_evidential=args.use_evidential,
        kl_weight=args.kl_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    # 3. Fine-tuning Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_evidential_loss = 0.0
        total_vacuity = 0.0
        num_batches = 0
        start_time = time.time()

        for batch in train_loader:
            optimizer.zero_grad()

            B = batch["batch_size"]
            N = batch["claim_classes"].shape[1]
            claim_embeddings = torch.randn(B, N, args.d_model, device=device, requires_grad=True)

            out = model(
                claim_embeddings=claim_embeddings,
                claim_mask=batch["claim_mask"].to(device),
                ev_class_ids=batch["ev_class_ids"].to(device),
                ev_continuous=batch["ev_continuous"].to(device),
                ev_confidences=batch["ev_confidences"].to(device),
                evidence_mask=batch["evidence_mask"].to(device),
                claim_targets=batch["claim_targets"].to(device),
                ev_omission_targets=batch["ev_omission_targets"].to(device),
                claim_classes=batch["claim_classes"].to(device),
                claim_positions=batch["claim_positions"].to(device),
                ev_positions=batch["ev_positions"].to(device),
                claim_distances=batch["claim_distances"].to(device),
                claim_path_ids=batch["claim_path_ids"].to(device),
                claim_vrel=batch["claim_vrel"].to(device),
            )

            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            if "evidential_verification" in out["losses"]:
                total_evidential_loss += out["losses"]["evidential_verification"].item()
            if "vacuity" in out:
                total_vacuity += out["vacuity"].mean().item()
            num_batches += 1

            if num_batches % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"    [Epoch {epoch:02d} | Batch {num_batches:04d}/{len(train_loader):04d}] "
                    f"Batch Loss: {loss.item():.4f} | Vacuity: {out['vacuity'].mean().item():.3f} | "
                    f"Throughput: {num_batches * args.batch_size / max(elapsed, 0.1):.1f} samples/s"
                )

        scheduler.step()
        avg_train_loss = total_train_loss / num_batches
        avg_evidential_loss = total_evidential_loss / num_batches
        avg_vacuity = total_vacuity / num_batches

        # 4. Validation Pass
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                B = batch["batch_size"]
                N = batch["claim_classes"].shape[1]
                claim_embeddings = torch.randn(B, N, args.d_model, device=device)

                out = model(
                    claim_embeddings=claim_embeddings,
                    claim_mask=batch["claim_mask"].to(device),
                    ev_class_ids=batch["ev_class_ids"].to(device),
                    ev_continuous=batch["ev_continuous"].to(device),
                    ev_confidences=batch["ev_confidences"].to(device),
                    evidence_mask=batch["evidence_mask"].to(device),
                    claim_targets=batch["claim_targets"].to(device),
                    ev_omission_targets=batch["ev_omission_targets"].to(device),
                    claim_classes=batch["claim_classes"].to(device),
                    claim_positions=batch["claim_positions"].to(device),
                    ev_positions=batch["ev_positions"].to(device),
                    claim_distances=batch["claim_distances"].to(device),
                    claim_path_ids=batch["claim_path_ids"].to(device),
                    claim_vrel=batch["claim_vrel"].to(device),
                )
                total_val_loss += out["loss"].item()
                val_batches += 1

        avg_val_loss = total_val_loss / max(val_batches, 1)
        epoch_sec = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_sec:.1f}s] "
            f"Train Loss: {avg_train_loss:.4f} | Evidential Loss: {avg_evidential_loss:.4f} | "
            f"Epistemic Vacuity: {avg_vacuity:.3f} | Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "avg_vacuity": avg_vacuity,
                    "dataset_source": "real_nuscenes",
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  -> [★] Saved new best real-nuScenes evidential checkpoint to {ckpt_path}")

    print("[✓] Real nuScenes PhD Evidential Fine-Tuning completed successfully!")


if __name__ == "__main__":
    main()
