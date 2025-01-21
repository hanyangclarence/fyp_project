#!/bin/bash
#SBATCH --job-name=yh
#SBATCH -o output_logs/tr_%j.out
#SBATCH -e output_logs/tr_%j.err
#SBATCH --gres=gpu:1

conda activate 3d_diffuser_actor

xvfb-run python train.py --stage train_vqvae --base configs/train_motion_vqvae_v2.yaml