#!/bin/bash
#SBATCH --job-name=shakespeare_gpt
#SBATCH -A MLMI-HRAH2-SL2-GPU          # ← your GPU project
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2              # ≤3 per GPU is the site rule
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
mkdir -p logs

source "$HOME/miniforge/etc/profile.d/conda.sh"
conda activate audiogpt

# ---------- compiler for Triton (conda-clang) ---------------------
export CC=$(which clang)
export CXX=$(which clang++)
export TRITON_CACHE_DIR=$HOME/rds/hpc-work/audioGPT/wandb/triton_cache
mkdir -p "$TRITON_CACHE_DIR"
# ------------------------------------------------------------------

export WANDB_PROJECT=shakespeare-char
export WANDB_DIR=$HOME/rds/hpc-work/audioGPT/wandb
export WANDB_CACHE_DIR=$WANDB_DIR
export WANDB_MODE=online
mkdir -p "$WANDB_DIR"

echo "[$(date)] job $SLURM_JOB_ID on $(hostname) GPU:$CUDA_VISIBLE_DEVICES"

cd ~/rds/hpc-work/audioGPT
python train.py config/examples/train_shakespeare_char.py --wandb_log=True --wandb_run_name=test-gcc1

echo "[$(date)] finished"
