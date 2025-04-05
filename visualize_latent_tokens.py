import os
import argparse
from omegaconf import OmegaConf
import torch
import numpy as np

from collections import Counter
import matplotlib.pyplot as plt

from unimumo.util import instantiate_from_config, load_model_from_config
from unimumo.data.motion_vqvae_dataset_v4 import MotionVQVAEDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--save_dir", type=str, default="visualization_logs")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_path", type=str, default="/research/d2/fyp24/hyang2/fyp/code/3d_diffuser_actor/data/peract/raw")
    parser.add_argument("--max_eps_per_task", type=int, default=1000)
    args = parser.parse_args()

    # load config
    config = OmegaConf.load(args.config)
    codebook_size = config["model"]["params"]["quantizer_config"]["bins"]

    # set save dir
    save_dir = os.path.join(args.save_dir, f"{os.path.basename(args.config).replace('.yaml', '')}_{os.path.basename(args.ckpt).replace('.ckpt', '')}")
    os.makedirs(save_dir, exist_ok=True)

    # load model
    model = load_model_from_config(config, args.ckpt, verbose=True)
    model.cuda()

    # load dataset
    chunk_size = config["data"]["params"]["validation"]["params"]["chunk_size"]
    n_chunk_per_traj = config["data"]["params"]["validation"]["params"]["n_chunk_per_traj"]
    dataset = MotionVQVAEDataset(
        args.split,
        args.dataset_path,
        preload_data=False,
        load_observations=config["data"]["params"]["validation"]["params"]["load_observations"],
        load_proprioception=config["data"]["params"]["validation"]["params"]["load_proprioception"],
        use_chunk=config["data"]["params"]["validation"]["params"]["use_chunk"],
        chunk_size=chunk_size,
        n_chunk_per_traj=n_chunk_per_traj,
        load_sparce=config["data"]["params"]["validation"]["params"]["load_sparce"],
        load_full_traj=True,
    )

    stats_results = {}
    eps_count = {}
    for i in range(len(dataset)):
        batch = dataset[i]
        gt_traj = batch["trajectory"]  # (T, D)
        task_str = batch["task_str"]
        if task_str not in stats_results:
            stats_results[task_str] = np.array([], dtype=np.int64)
        if task_str not in eps_count:
            eps_count[task_str] = 0
        if eps_count[task_str] >= args.max_eps_per_task:
            continue
        eps_count[task_str] += 1

        # reconstruct
        gt_traj = gt_traj.unsqueeze(0).cuda()  # (1, T, D)
        traj_length = chunk_size * n_chunk_per_traj
        with torch.no_grad():
            all_codes = []
            for sec_start in range(0, gt_traj.shape[1], traj_length):
                traj_chunk = gt_traj[:, sec_start:sec_start + traj_length]
                code = model.encode(traj_chunk)  # (1, N_q, n_chunk_per_traj)
                recon_traj = model.decode(code)

                all_codes.append(code)

            code = torch.cat(all_codes, dim=-1)  # (1, N_q, T')
            assert code.shape[2] == gt_traj.shape[1] // chunk_size, f"Code shape {code.shape} does not match gt_traj shape {gt_traj.shape}, {chunk_size}"

            # decode the code
            all_recon_trajectories = []
            for sec_idx in range(code.shape[2] // n_chunk_per_traj):
                code_chunk = code[:, :, sec_idx * n_chunk_per_traj:(sec_idx + 1) * n_chunk_per_traj]
                recon_traj = model.decode(code_chunk)  # (1, T, D)

                all_recon_trajectories.append(recon_traj)
            recon_traj = torch.cat(all_recon_trajectories, dim=1)
            print(f"{i}/{len(dataset)}  GT shape: {gt_traj.shape}, recon shape: {recon_traj.shape}")

        code = code.flatten().cpu().numpy().astype(np.int64)
        stats_results[task_str] = np.concatenate([stats_results[task_str], code], axis=0)

    for task_str, all_token_ids in stats_results.items():
        counts = Counter(all_token_ids)
        frequencies = np.zeros(codebook_size)
        for token_id, count in counts.items():
            frequencies[token_id] = count

        heatmap = frequencies.reshape(16, -1)
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap, cmap='viridis')
        plt.colorbar(label="Frequency")
        plt.title(f"Token Frequencies for {task_str}")
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{task_str}.png"), bbox_inches='tight')
    
    all_token_ids = np.concatenate(list(stats_results.values()), axis=0)
    counts = Counter(all_token_ids)
    frequencies = np.zeros(codebook_size)
    for token_id, count in counts.items():
        frequencies[token_id] = count
        
    heatmap = frequencies.reshape(16, -1)
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap, cmap='viridis')
    plt.colorbar(label="Frequency")
    plt.title(f"Token Frequencies for all tasks")
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"all_tasks.png"), bbox_inches='tight')
    plt.close()


