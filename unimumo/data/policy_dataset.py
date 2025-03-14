import pickle

import torch
import os
from torch.utils.data import Dataset
from os.path import join as pjoin
import random
from typing import Tuple, Dict
from tqdm import tqdm
import quaternion
import numpy as np
from torchvision import transforms
from scipy.spatial.transform import Rotation as R


Instructions = Dict[str, Dict[int, torch.Tensor]]


class PolicyDataset(Dataset):
    def __init__(
            self,
            split: str,
            motion_code_dir: str,
            visual_data_dir: str,
            instruction_path: str,
            traj_length: int,
            start_idx: int = 512,
            end_idx: int = 513,
            pad_idx: int = 514,
    ):
        self.split = split
        self.motion_code_dir = motion_code_dir
        self.visual_data_dir = visual_data_dir
        self.traj_length = traj_length

        self.start_idx = start_idx
        self.end_idx = end_idx
        self.pad_idx = pad_idx

        # load instruction
        self.instructions: Instructions = pickle.load(open(instruction_path, "rb"))

        # load data
        self.tasks = os.listdir(pjoin(motion_code_dir, split))
        self.all_demos_ids = []
        for task in self.tasks:
            assert task in self.instructions, f"task {task} not found in instruction"
            all_var_eps = os.listdir(pjoin(motion_code_dir, split, task))
            for var_eps in all_var_eps:
                if not os.path.exists(pjoin(motion_code_dir, split, task, var_eps, "code.npy")):
                    continue
                if not os.path.exists(pjoin(motion_code_dir, split, task, var_eps, "indices.npy")):
                    continue

                var = int(var_eps.split("_eps")[0].replace("var_", ""))
                eps = int(var_eps.split("_eps_")[1])

                if var not in self.instructions[task]:
                    continue

                self.all_demos_ids.append((task, var, eps))
        print(f"{split} data loaded, total number of demos: {len(self.all_demos_ids)}")

    def __len__(self):
        return len(self.all_demos_ids)

    def __getitem__(self, idx):
        task, var, eps = self.all_demos_ids[idx]

        # randomly sample an instruction
        num_instr = len(self.instructions[task][var])
        instr_idx = random.randint(0, num_instr - 1)
        instruction = self.instructions[task][var][instr_idx]  # (53, 512)

        # load the corresponding visual data
        full_obs = torch.load(pjoin(self.visual_data_dir, self.split, f"{task}_var_{var}_eps_{eps}.pt"))
        _, n_cameras, _, H, W = full_obs.shape

        # load motion code
        full_traj_code = torch.tensor(
            np.load(pjoin(self.motion_code_dir, self.split, task, f"var_{var}_eps_{eps}", "code.npy"))
        ).squeeze()  # (T * 4)  where each timestep is represented by 4 codes
        code_indices = np.load(pjoin(self.motion_code_dir, self.split, task, f"var_{var}_eps_{eps}", "indices.npy"))  # (T,)

        # randomly sample a trajectory
        if len(code_indices) - self.traj_length >= -1:
            input_mask = torch.ones((self.traj_length - 1) * 4, dtype=torch.bool)  # ((T'-1) * 4)
            start_idx = random.randint(-1, len(code_indices) - self.traj_length)
            end_idx = start_idx + self.traj_length
            if start_idx != -1:
                traj_code = full_traj_code[start_idx * 4:end_idx * 4]  # (T' * 4)
                traj_indices = code_indices[start_idx:end_idx]  # (T', )
            else:
                # add start token
                traj_code = torch.cat([
                    torch.tensor([self.start_idx] * 4), full_traj_code[:end_idx * 4]
                ])
            # some sanity check
            if end_idx == len(code_indices):
                assert torch.all(traj_code[-4:] == self.end_idx), f"last code is not end token: {traj_code[-4:]}"

            rgb_pcd = full_obs[start_idx + 1:end_idx]  # (T'-1, n_cameras, 6, H, W)
            rgb = rgb_pcd[:, :, :3]  # (T'-1, n_cameras, 3, H, W)
            pcd = rgb_pcd[:, :, 3:]  # (T'-1, n_cameras, 3, H, W)
            context_mask = torch.ones(self.traj_length - 1, dtype=torch.bool)  # (T'-1, )
        else:
            # pad the trajectory
            pad_length = self.traj_length - len(code_indices) - 1
            traj_code = torch.cat([
                torch.tensor([self.start_idx] * 4), full_traj_code, torch.tensor([self.pad_idx] * 4 * pad_length)
            ])
            input_mask = torch.ones((self.traj_length - 1) * 4, dtype=torch.bool)
            input_mask[-4 * pad_length:] = False

            rgb_pcd = torch.cat([full_obs, torch.zeros((pad_length, n_cameras, 6, H, W))], dim=0)  # (T'-1, n_cameras, 6, H, W)
            rgb = rgb_pcd[:, :, :3]
            pcd = rgb_pcd[:, :, 3:]
            context_mask = torch.ones(self.traj_length - 1, dtype=torch.bool)
            context_mask[-pad_length:] = False  # (T'-1, )

        return {
            "trajectory": traj_code,  # (T' * 4)
            "instruction": instruction,  # (54, 512)
            "rgb": rgb,  # (T'-1, n_cameras, 3, H, W)
            "pcd": pcd,  # (T'-1, n_cameras, 3, H, W)
            "input_mask": input_mask,  # ((T'-1) * 4)
            "context_mask": context_mask,  # (T'-1)
        }















