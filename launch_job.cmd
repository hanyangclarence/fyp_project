#!/bin/bash
#SBATCH --job-name=yh
#SBATCH --nodes=4                 # total number of nodes
#SBATCH --ntasks-per-node=1       # 1 task per node
#SBATCH --gres=gpu:1              # 1 GPU per node
#SBATCH -o output_logs/tr_%j.out
#SBATCH -e output_logs/tr_%j.err
#SBATCH --nodelist=gpu[39-51]

source /research/d2/fyp24/hyang2/anaconda3/etc/profile.d/conda.sh
conda activate 3d_diffuser_actor

xvfb-run python train.py --stage train_vqvae --base configs/train_motion_vqvae_no_chunk.yaml