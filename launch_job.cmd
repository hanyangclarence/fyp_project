#!/bin/bash
#SBATCH --job-name=yh
#SBATCH -o output_logs/tr_%j.out
#SBATCH -e output_logs/tr_%j.err
#SBATCH --gres=gpu:1

source /research/d2/fyp24/hyang2/anaconda3/etc/profile.d/conda.sh

conda activate fyp

xvfb-run python train.py --stage train_vqvae --base configs/train_motion_vqvae_20.yaml
@REM xvfb-run python train.py --base configs/train_policy.yaml