#!/bin/bash
#SBATCH --job-name=safe_ground_full
#SBATCH --partition=mtech
#SBATCH --qos=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/safe_ground_full_%j.out
#SBATCH --error=logs/safe_ground_full_%j.err
#SBATCH --exclude=cn07
# ==============================================================================
#  SAFE-Ground: Full-Scale Training and Evaluation on Real nuScenes v1.0-trainval
#  - Train: 700 scenes, 28,130 keyframes, 844,200 real 3D annotated boxes
#  - Val:   150 scenes,  6,019 keyframes, 164,636 real 3D annotated boxes
# ==============================================================================

echo "=============================================================================="
echo "  SAFE-Ground Full-Scale nuScenes Run"
echo "  Slurm Job ID: $SLURM_JOB_ID"
echo "  Node Name:    $SLURM_NODELIST"
echo "  GPU Devices:  $CUDA_VISIBLE_DEVICES"
echo "  Start Time:   $(date)"
echo "=============================================================================="

# Navigate to project root
cd /csehome/m25csa031/SAFE-Ground

# Environment setup
mkdir -p logs checkpoints
export PATH=/csehome/m25csa031/miniconda3/envs/vit/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/csehome/m25csa031/SAFE-Ground:$PYTHONPATH

# GPU and Environment diagnostics
echo ""
echo "[*] GPU & Hardware Diagnostics:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi
fi

# Step 1: Run All 12 Unit Tests (Foundations + PhD Novelty Pillars)
echo ""
echo "=============================================================================="
echo "  Step 1: Running All Unit Tests (including PhD Pillars)"
echo "=============================================================================="
python -m unittest discover -s tests -p "test_*.py"

# Step 2: Ensure Full Dataset Pre-Cache is Present
echo ""
echo "=============================================================================="
echo "  Step 2: Checking Full Dataset Pre-Cache on /scratch"
echo "=============================================================================="
python -u scripts/cache_nuscenes_trainval.py \
    --dataroot /scratch/m25csa031/nuscenes \
    --version v1.0-trainval \
    --output_dir /scratch/m25csa031/safe_ground_cache

# Step 3: Run Full-Scale Evidential Verifier Training (28,130 Train Samples)
echo ""
echo "=============================================================================="
echo "  Step 3: Evidential Verifier Training on Full nuScenes v1.0-trainval"
echo "=============================================================================="
DEVICE="cuda"
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "Warning: CUDA not available on this node, falling back to CPU"
    DEVICE="cpu"
fi

python -u scripts/train.py \
    --epochs 5 \
    --batch_size 64 \
    --d_model 256 \
    --lr 3e-4 \
    --use_evidential \
    --kl_weight 0.05 \
    --use_real_nuscenes \
    --nuscenes_version v1.0-trainval \
    --cache_dir /scratch/m25csa031/safe_ground_cache \
    --checkpoint_dir checkpoints \
    --checkpoint_name best_verifier_full_trainval.pt \
    --log_interval 50 \
    --device $DEVICE

# Step 4: Run Full Comparative Benchmark on All 6,019 Real nuScenes Val Keyframes
echo ""
echo "=============================================================================="
echo "  Step 4: Running Comparative Benchmark on 6,019 Real Val Keyframes (150 Scenes)"
echo "=============================================================================="
python -u scripts/evaluate.py \
    --use_real_nuscenes \
    --nuscenes_version v1.0-trainval \
    --cache_dir /scratch/m25csa031/safe_ground_cache \
    --num_samples 0 \
    --hallucination_rate 0.45 \
    --output_file evaluation_comparison_full_trainval.json

# Step 5: Run Interactive End-to-End Demo (Evidential + Reachability + CBF Shield)
echo ""
echo "=============================================================================="
echo "  Step 5: Running PhD-Grade End-to-End Demonstration"
echo "=============================================================================="
python -u scripts/demo.py

echo ""
echo "=============================================================================="
echo "  SAFE-Ground Full-Scale Run Completed Successfully at $(date)"
echo "=============================================================================="

