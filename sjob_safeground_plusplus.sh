#!/bin/bash
#SBATCH --job-name=safeground_pp
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=cn07
#SBATCH --output=/csehome/m25csa031/SAFE-Ground/logs/slurm-sgpp-%j.out
#SBATCH --error=/csehome/m25csa031/SAFE-Ground/logs/slurm-sgpp-%j.err

set -e

source /csehome/m25csa031/miniconda3/etc/profile.d/conda.sh
conda activate vit

cd /csehome/m25csa031/SAFE-Ground
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

nvidia-smi
python scripts/run_safeground_plusplus_20model_benchmark.py --epochs 8 --batch-size 32 --lr 5e-4 --d-model 256
