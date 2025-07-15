#!/bin/bash
#SBATCH --job-name=camstories_10k_shadow_audio
#SBATCH -A MLMI-HRAH2-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
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

export WANDB_DIR=$HOME/rds/hpc-work/audioGPT/wandb
export WANDB_CACHE_DIR=$WANDB_DIR
export WANDB_MODE=online
mkdir -p "$WANDB_DIR"

echo "[$(date)] job $SLURM_JOB_ID on $(hostname) GPU:$CUDA_VISIBLE_DEVICES"

cd ~/rds/hpc-work/audioGPT
python train.py config/camstories_10k_shadow_audio_auxloss.py --wandb_log=True --wandb_run_name=camstories-10k-shadow-audio-auxloss-run2

echo "[$(date)] finished" 