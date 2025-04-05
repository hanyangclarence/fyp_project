import os
import argparse
from omegaconf import OmegaConf
import torch
import numpy as np

from collections import Counter
import matplotlib.pyplot as plt

from unimumo.util import instantiate_from_config, load_model_from_config
from unimumo.data.motion_vqvae_dataset_v4 import MotionVQVAEDataset

vis_name = "visualize_codebook_similarity"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--save_dir", type=str, default="visualization_logs")
    args = parser.parse_args()
    
    # load config
    config = OmegaConf.load(args.config)
    codebook_size = config["model"]["params"]["quantizer_config"]["bins"]

    # set save dir
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, vis_name), exist_ok=True)
    save_dir = os.path.join(args.save_dir, vis_name, f"{os.path.basename(args.config).replace('.yaml', '')}_{os.path.basename(args.ckpt).replace('.ckpt', '')}")
    os.makedirs(save_dir, exist_ok=True)

    # load model
    model = load_model_from_config(config, args.ckpt, verbose=True)
    model.cuda()
    
    codebook = model.quantizer.vq.layers[0]._codebook.embed  # (N, D)
    codebook = codebook.cpu().numpy()
    
    N, D = codebook.shape
    sim_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i, N):
            if np.linalg.norm(codebook[i]) < 1e-8 or np.linalg.norm(codebook[j]) < 1e-8:
                sim = 0.0
            else:
                sim = np.dot(codebook[i], codebook[j]) / (np.linalg.norm(codebook[i]) * np.linalg.norm(codebook[j]))
            sim_mat[i, j] = sim
            sim_mat[j, i] = sim
    # save sim_mat as image
    plt.figure(figsize=(50, 50))
    plt.imshow(sim_mat, cmap='hot', interpolation='nearest')
    
    # add a colorbar with large text
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=50)
    cbar.ax.set_ylabel('Similarity', fontsize=50)
    
    plt.title("Codebook Similarity Matrix", fontsize=60)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.xlabel("Codebook Index", fontsize=60)
    plt.ylabel("Codebook Index", fontsize=60)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "codebook_similarity.png"))
    plt.close()
    print("Codebook similarity matrix saved to", os.path.join(save_dir, "codebook_similarity.png"))