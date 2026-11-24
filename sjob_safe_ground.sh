#!/bin/bash
#SBATCH --job-name=safe_ground_phd
#SBATCH --partition=mtech
#SBATCH --qos=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/safe_ground_phd_%j.out
#SBATCH --error=logs/safe_ground_phd_%j.err
#SBATCH --exclude=cn07
# ============================================================
#  SAFE-Ground 2.0: PhD Evidential Fine-Tuning & Trajectory Shield
# ============================================================

echo "============================================================"
echo "  Slurm Job ID: $SLURM_JOB_ID"
echo "  Node Name:    $SLURM_NODELIST"
echo "  GPU Devices:  $CUDA_VISIBLE_DEVICES"
echo "  Start Time:   $(date)"
echo "============================================================"

# Navigate to project root
cd /csehome/m25csa031/SAFE-Ground

# Environment setup
mkdir -p logs checkpoints
source ~/miniconda3/bin/activate vit
export PYTHONNOUSERSITE=1
export PYTHONPATH=/csehome/m25csa031/SAFE-Ground:$PYTHONPATH

# GPU and Environment diagnostics
echo ""
echo "[*] GPU Diagnostics:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi
fi

# Step 1: Run All 12 Unit Tests (Foundations + PhD Novelty Pillars)
echo ""
echo "============================================================"
echo "  Step 1: Running All Unit Tests (including PhD Pillars)"
echo "============================================================"
python -m unittest discover -s tests -p "test_*.py"

# Step 2: Run Interactive End-to-End Demo (Evidential + Reachability + CBF Shield)
echo ""
echo "============================================================"
echo "  Step 2: Running PhD-Grade End-to-End Demonstration"
echo "============================================================"
python -u scripts/demo.py

# Step 3: Run Evidential Fine-Tuning on GPU
echo ""
echo "============================================================"
echo "  Step 3: Evidential Verifier Fine-Tuning on GPU"
echo "============================================================"
DEVICE="cuda"
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "Warning: CUDA not available on this node, falling back to CPU"
    DEVICE="cpu"
fi

python -u scripts/train.py \
    --epochs 10 \
    --batch_size 16 \
    --d_model 256 \
    --lr 3e-4 \
    --use_evidential \
    --kl_weight 0.05 \
    --use_real_nuscenes \
    --nuscenes_dataroot /scratch/m25csa031/nuscenes \
    --nuscenes_version v1.0-mini \
    --checkpoint_dir checkpoints \
    --device $DEVICE

# Step 4: Run Comparative Benchmark Evaluation (Raw vs Prompt-Inj vs SAFE-Ground on Real nuScenes)
echo ""
echo "============================================================"
echo "  Step 4: Running Comparative Benchmark on Real nuScenes"
echo "============================================================"
python -u scripts/evaluate.py \
    --use_real_nuscenes \
    --nuscenes_dataroot /scratch/m25csa031/nuscenes \
    --nuscenes_version v1.0-mini \
    --hallucination_rate 0.45 \
    --output_file evaluation_comparison_real_nuscenes.json

echo ""
echo "============================================================"
echo "  SAFE-Ground PhD Full Run Completed at $(date)"
echo "============================================================"
