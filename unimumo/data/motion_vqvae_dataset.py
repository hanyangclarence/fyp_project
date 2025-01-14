import torch
import os
import numpy as np
import codecs as cs
from torch.utils.data import Dataset
from os.path import join as pjoin
import random
import pickle
from typing import Tuple
from einops import rearrange

from rlbench.demo import Demo

from unimumo.rlbench.utils_with_rlbench import RLBenchEnv, keypoint_discovery, interpolate_trajectory


class MotionVQVAEDataset(Dataset):
    def __init__(
            self,
            split: str,
            data_dir: str,
            cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist", "front"),
            image_size: str = "256,256",
            load_observations: bool = False,
            chunk_size: int = 4,  # number of frames in a chunk
            n_chunk_per_traj: int = 2,  # number of chunks in a trajectory
    ):
        # load RLBench environment
        self.env = RLBenchEnv(
            data_path=pjoin(data_dir, split),
            image_size=[int(x) for x in image_size.split(",")],
            apply_rgb=True,
            apply_pc=True,
            apply_cameras=cameras,
        )

        self.chunk_size = chunk_size
        self.n_chunk_per_traj = n_chunk_per_traj

        # load data
        self.data = []
        self.tasks = os.listdir(pjoin(data_dir, split))
        for task in self.tasks:
            for var in range(1):  # seems that there is only one variation for each task
                num_episode = len(os.listdir(pjoin(data_dir, split, task, "all_variations", "episodes")))
                for eps in range(num_episode):
                    action_traj, descriptions = self.load_obs_traj(task, var, eps, load_observations)
                    self.data.append((action_traj, descriptions))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        action_traj, descriptions = self.data[idx]
        len_traj = len(action_traj)

        # sample a random chunk
        start_idx = random.randint(0, len_traj // self.chunk_size - self.n_chunk_per_traj) * self.chunk_size
        end_idx = start_idx + self.chunk_size * self.n_chunk_per_traj
        traj = action_traj[start_idx:end_idx]  # (T, 8)

        # TODO: data augmentation?

        # sample a random description
        desc = random.choice(descriptions)

        return  {
            "trajectory": traj,  # (T, 8)
            "description": desc  # str
        }


    def load_obs_traj(self, task: str, variation: int, episode: int, load_observations: bool = False):
        # load stored demo
        demo: Demo = self.env.get_demo(task, variation, episode, image_paths=load_observations)[0]

        # get keypoints
        key_frame_ids = keypoint_discovery(demo)  # List[int]
        key_frame_ids.insert(0, 0)

        action_traj = []

        # process the segment between each pair of key frames: interpolate their length to a multiple of chunk_size
        for i in range(len(key_frame_ids) - 1):
            traj_segment = []
            start_frame = key_frame_ids[i]
            end_frame = key_frame_ids[i + 1]

            for j in range(start_frame, end_frame):
                _, action = self.env.get_obs_action(demo[j])  # action: (8)
                traj_segment.append(action.unsqueeze(0))
            traj_segment = torch.cat(traj_segment, dim=0)  # (n_frames, 8)

            len_segment = len(traj_segment)
            target_interp_length = len_segment + self.chunk_size - len_segment % self.chunk_size
            traj_segment = interpolate_trajectory(traj_segment, target_interp_length)

            action_traj.append(traj_segment)

        action_traj = torch.cat(action_traj, dim=0)  # (n_frames, 8)

        # TODO: code for loading observations images/point clouds is not implemented yet

        descriptions = demo[0].misc['descriptions']

        return action_traj, descriptions


if __name__ == "__main__":
    data_dir = "/research/d2/fyp24/hyang2/fyp/code/3d_diffuser_actor/data/peract/raw"
    dataset = MotionVQVAEDataset("val", data_dir)

    for i in range(len(dataset)):
        sample = dataset.__getitem__(i)
        print(sample)


























