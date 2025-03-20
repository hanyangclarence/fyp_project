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

# this model generate the tokens sequentially, not generating all tokens at once


Instructions = Dict[str, Dict[int, torch.Tensor]]


class PolicyDataset(Dataset):
    def __init__(
            self,
            split: str,
            motion_code_dir: str,
            visual_data_dir: str,
            instruction_path: str,
            traj_length: int,
            chunk_size: int,
            start_idx: int = 512,
            end_idx: int = 513,
            pad_idx: int = 514,
    ):
        self.split = split
        self.motion_code_dir = motion_code_dir
        self.visual_data_dir = visual_data_dir
        self.traj_length = traj_length
        self.chunk_size = chunk_size

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

        assert self.chunk_size == len(full_traj_code) / len(code_indices), f"chunk size {self.chunk_size} does not match the length of full_traj_code {len(full_traj_code)} and code_indices {len(code_indices)}"
        assert full_obs.shape[0] == len(code_indices), f"full_obs shape {full_obs.shape} does not match code_indices shape {code_indices.shape}"

        # randomly sample a trajectory
        if len(code_indices) - self.traj_length >= 0:
            input_mask = torch.ones(self.traj_length * self.chunk_size, dtype=torch.bool)  # (T' * 4)
            start_idx = random.randint(0, len(code_indices) - self.traj_length)
            end_idx = start_idx + self.traj_length

            traj_code_target = full_traj_code[start_idx * self.chunk_size:end_idx * self.chunk_size]  # (T' * 4)
            if start_idx == 0:
                traj_code_input = torch.cat([
                    torch.tensor([self.start_idx]), traj_code_target[:-1]
                ])
            else:
                traj_code_input = full_traj_code[start_idx * self.chunk_size - 1:end_idx * self.chunk_size - 1]  # (T' * 4)

            # some sanity check
            if end_idx == len(code_indices):
                assert torch.all(traj_code_target[-self.chunk_size:] == self.end_idx), f"last code is not end token: {traj_code_target[-self.chunk_size:]}"

            rgb_pcd = full_obs[start_idx:end_idx]  # (T', n_cameras, 6, H, W)
            rgb = rgb_pcd[:, :, :3]  # (T', n_cameras, 3, H, W)
            pcd = rgb_pcd[:, :, 3:]  # (T', n_cameras, 3, H, W)
            context_mask = torch.ones(self.traj_length, dtype=torch.bool)  # (T', )
        else:
            # pad the trajectory
            pad_length = self.traj_length - len(code_indices)
            traj_code_target = torch.cat([
                full_traj_code, torch.tensor([self.pad_idx] * self.chunk_size * pad_length)
            ])  # (T' * 4)
            traj_code_input = torch.cat([
                torch.tensor([self.start_idx]), full_traj_code[:-1], torch.tensor([self.pad_idx] * self.chunk_size * pad_length)
            ])  # (T' * 4)
            input_mask = torch.ones(self.traj_length * self.chunk_size, dtype=torch.bool)  # (T' * 4)
            input_mask[-self.chunk_size * pad_length:] = False

            rgb_pcd = torch.cat([full_obs, torch.zeros((pad_length, n_cameras, 6, H, W))], dim=0)  # (T', n_cameras, 6, H, W)
            rgb = rgb_pcd[:, :, :3]  # (T', n_cameras, 3, H, W)
            pcd = rgb_pcd[:, :, 3:]  # (T', n_cameras, 3, H, W)
            context_mask = torch.ones(self.traj_length, dtype=torch.bool)
            context_mask[-pad_length:] = False  # (T', )

        assert rgb.shape[0] == self.traj_length, f"rgb shape {rgb.shape} does not match traj_length {self.traj_length}"

        return {
            "trajectory": traj_code_target,  # (T' * 4)
            "traj_input": traj_code_input,  # (T' * 4)
            "instruction": instruction,  # (54, 512)
            "rgb": rgb,  # (T', n_cameras, 3, H, W)
            "pcd": pcd,  # (T', n_cameras, 3, H, W)
            "input_mask": input_mask,  # (T' * 4)
            "context_mask": context_mask,  # (T', )
            "task_str": task,  # str
            "variation": var,  # int
            "episode": eps,  # int
        }















