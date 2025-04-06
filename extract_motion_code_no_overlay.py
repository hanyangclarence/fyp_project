import argparse

from omegaconf import OmegaConf
import os
import torch
import numpy as np


from unimumo.util import instantiate_from_config, load_model_from_config
from unimumo.data.motion_vqvae_dataset_v4 import MotionVQVAEDataset
from unimumo.rlbench.utils_with_rlbench import RLBenchEnv


# this is for VQVAE version 21
# there is no overlay in encoding and decoding

START_TOKEN = 512
END_TOKEN = 513

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/vqvae.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="/research/d2/fyp24/hyang2/fyp/code/3d_diffuser_actor/data/peract/raw")
    parser.add_argument("--save_dir", type=str, default="data")
    args = parser.parse_args()

    # load config
    config = OmegaConf.load(args.config)

    # set save dir
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = os.path.join(args.save_dir, "motion_code_v26")
    os.makedirs(save_dir, exist_ok=True)

    # load model
    model = load_model_from_config(config, args.ckpt, verbose=True)
    model.cuda()

    pose_recon_losses = []
    gripper_classification_losses = []
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(save_dir, split), exist_ok=True)
        # load dataset
        chunk_size = config["data"]["params"]["validation"]["params"]["chunk_size"]
        n_chunk_per_traj = config["data"]["params"]["validation"]["params"]["n_chunk_per_traj"]
        dataset = MotionVQVAEDataset(
            split,
            args.dataset_path,
            preload_data=False,
            load_observations=config["data"]["params"]["validation"]["params"]["load_observations"],
            load_proprioception=config["data"]["params"]["validation"]["params"]["load_proprioception"],
            use_chunk=config["data"]["params"]["validation"]["params"]["use_chunk"],
            chunk_size=chunk_size,
            n_chunk_per_traj=n_chunk_per_traj,
            load_sparce=config["data"]["params"]["validation"]["params"]["load_sparce"],
            load_full_traj=True,
            load_traj_index=True
        )

        # init simulation env
        rlbench_env = RLBenchEnv(
            data_path=os.path.join(args.dataset_path, split),
            image_size=[256, 256],
            apply_rgb=False,
            apply_pc=False,
            headless=True,
            apply_cameras=("front",),
            collision_checking=False,
        )
        rlbench_env.env.launch()

        # run inference
        for i in range(len(dataset)):
            batch = dataset[i]
            gt_traj = batch["trajectory"]  # (T, D)
            traj_index = batch["trajectory_index"]  # (T, )
            description = batch["description"]
            task_str = batch["task_str"]
            variation = batch["variation"]
            episode = batch["episode"]

            assert gt_traj.shape[0] % (chunk_size * n_chunk_per_traj) == 0, f"Trajectory length {gt_traj.shape[0]} is not divisible by chunk_size {chunk_size} and n_chunk_per_traj {n_chunk_per_traj}"

            # reconstruct
            gt_traj = gt_traj.unsqueeze(0).cuda()  # (1, T, D)
            traj_length = chunk_size * n_chunk_per_traj
            with torch.no_grad():
                all_codes = []
                all_indices = []
                for sec_start in range(0, gt_traj.shape[1], traj_length):
                    traj_chunk = gt_traj[:, sec_start:sec_start+traj_length]
                    code = model.encode(traj_chunk)  # (1, N_q, n_chunk_per_traj)
                    recon_traj = model.decode(code)

                    all_codes.append(code)
                    all_indices.append(traj_index[sec_start])

                code = torch.cat(all_codes, dim=-1)  # (1, N_q, T')
                assert code.shape[2] == gt_traj.shape[1] // chunk_size, f"Code shape {code.shape} does not match gt_traj shape {gt_traj.shape}, {chunk_size}"

                # decode the code
                all_recon_trajectories = []
                for sec_idx in range(len(all_indices)):
                    code_chunk = code[:, :, sec_idx * n_chunk_per_traj:(sec_idx + 1) * n_chunk_per_traj]
                    recon_traj = model.decode(code_chunk)  # (1, T, D)

                    all_recon_trajectories.append(recon_traj)
                recon_traj = torch.cat(all_recon_trajectories, dim=1)

                print(f"GT shape: {gt_traj.shape}, recon shape: {recon_traj.shape}, code shape: {code.shape}")

                # append end tokens
                code = torch.cat(
                    [code, torch.tensor([[[END_TOKEN] * n_chunk_per_traj]]).cuda()],
                    dim=-1
                )
                all_indices.append(traj_index[-1])

            if model.motion_mode == "proprior":
                gt_traj = gt_traj[:, :, :8]

            # calculate loss
            gt_traj = gt_traj.squeeze(0).cpu()
            recon_traj = recon_traj.squeeze(0).cpu()
            pose_recon_loss = torch.nn.functional.smooth_l1_loss(gt_traj[:, :7], recon_traj[:, :7], reduction="mean")
            gripper_classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(gt_traj[:, 7:8], recon_traj[:, 7:8], reduction="mean")
            pose_recon_losses.append(pose_recon_loss.item())
            gripper_classification_losses.append(gripper_classification_loss.item())
            print(f"Split {split}, {i}/{len(dataset)}: pose_recon_loss={pose_recon_loss: .4f}, gripper_classification_loss={gripper_classification_loss: .4f} ")

            os.makedirs(os.path.join(save_dir, split, task_str), exist_ok=True)
            episode_dir = os.path.join(save_dir, split, task_str, f"var_{variation}_eps_{episode}")
            os.makedirs(episode_dir, exist_ok=True)
            # save code
            np.save(os.path.join(episode_dir, "code.npy"), code.cpu().numpy())
            np.save(os.path.join(episode_dir, "indices.npy"), np.array(all_indices))

        print(f"Average pose_recon_loss={np.mean(pose_recon_losses): .4f}, average gripper_classification_loss={np.mean(gripper_classification_losses): .4f}")
        rlbench_env.env.shutdown()
        # write losses to file
        with open(os.path.join(save_dir, "losses.txt"), "w") as f:
            content = f"Average pose_recon_loss={np.mean(pose_recon_losses): .4f}\nAverage gripper_classification_loss={np.mean(gripper_classification_losses): .4f}\n"
            f.write(content)





















