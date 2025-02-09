import torch
import os
from torch.utils.data import Dataset
from os.path import join as pjoin
import random
from typing import Tuple
from tqdm import tqdm
import quaternion
import numpy as np
from torchvision import transforms
from PIL import Image

from rlbench.demo import Demo

from unimumo.rlbench.utils_with_rlbench import RLBenchEnv, keypoint_discovery, interpolate_trajectory


class MotionVQVAEDataset(Dataset):
    def __init__(
            self,
            split: str,
            data_dir: str,
            preload_data: bool = False,
            apply_rgb: bool = True,
            apply_depth: bool = False,
            apply_pc: bool = False,
            cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist", "front"),
            image_size: str = "256,256",
            load_proprioception: bool = False,  # Whether to load proprioception data, concatenated with the gripper pose
            use_chunk: bool = False,  # Whether to load trajectory in chunks (segmented by key frames)
            chunk_size: int = 4,  # number of frames in a chunk
            n_chunk_per_traj: int = 2,  # number of chunks in a trajectory
            compression_rate: int = 4,  # compression rate in time dimension
    ):
        # load RLBench environment
        self.env = RLBenchEnv(
            data_path=pjoin(data_dir, split),
            image_size=[int(x) for x in image_size.split(",")],
            apply_rgb=apply_rgb,
            apply_pc=apply_pc,
            apply_depth=apply_depth,
            apply_cameras=cameras,
        )

        self.use_chunk = use_chunk  # TODO: when use_chunk is true, the interpolation for images are not implemented
        self.chunk_size = chunk_size
        self.n_chunk_per_traj = n_chunk_per_traj
        self.compression_rate = compression_rate

        # about load content settings
        self.load_proprioception = load_proprioception
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),  # converts to float32 and scales to [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # load data
        self.tasks = os.listdir(pjoin(data_dir, split))
        self.all_demos_ids = []
        for task in self.tasks:
            # get all the variations
            variation_folders = os.listdir(pjoin(data_dir, split, task))
            variation_folders = [x for x in variation_folders if x.startswith("variation")]
            for var_folder in variation_folders:
                var = int(var_folder.replace("variation", ""))
                num_episode = len(os.listdir(pjoin(data_dir, split, task, var_folder, "episodes")))
                for eps in range(num_episode):
                    self.all_demos_ids.append((task, var, eps))

        self.data = []
        self.preload_data = preload_data
        if preload_data:
            for task, var, eps in tqdm(self.all_demos_ids, desc=f"Loading {split} data"):
                action_traj, descriptions, observations = self.load_obs_traj(task, var, eps)
                self.data.append((action_traj, descriptions, observations, task, var, eps))
        print(f"{split} data loaded, total number of demos: {len(self.all_demos_ids)}")

    def __len__(self):
        return len(self.all_demos_ids)

    def __getitem__(self, idx):
        if self.preload_data:
            action_traj, descriptions, observations, task, var, eps = self.data[idx]
        else:
            task, var, eps = self.all_demos_ids[idx]
            action_traj, descriptions, observations = self.load_obs_traj(task, var, eps)
        len_traj = len(action_traj)

        if self.use_chunk:
            # sample a random chunk
            start_idx = random.randint(0, len_traj // self.chunk_size - self.n_chunk_per_traj) * self.chunk_size
            end_idx = start_idx + self.chunk_size * self.n_chunk_per_traj
            traj = action_traj[start_idx:end_idx].float()  # (T, 8), also convert to float32 tensor
        else:
            # directly sample a trajectory
            start_idx = random.randint(0, len_traj - self.n_chunk_per_traj * self.chunk_size)
            end_idx = start_idx + self.n_chunk_per_traj * self.chunk_size
            traj = action_traj[start_idx:end_idx].float()  # (T, 8)


        # sample a random description
        desc = random.choice(descriptions)

        data_dict = {
            "trajectory": traj,  # (T, 8)
            "description": desc,  # str
            "task_str": task,  # str
            "variation": var,  # int
            "episode": eps,  # int
        }

        if self.apply_rgb:
            data_dict["rgb"] = torch.stack(
                observations["rgb"][start_idx:end_idx:self.compression_rate]
            )  # (T', N, 3, H, W)
        if self.apply_depth:
            data_dict["depth"] = torch.stack(
                observations["depth"][start_idx:end_idx:self.compression_rate]
            )  # (T', N, 1, H, W)
        if self.apply_pc:
            data_dict["pc"] = torch.stack(
                observations["pc"][start_idx:end_idx:self.compression_rate]
            )  # (T', N, 3, H, W)

        return data_dict


    def load_obs_traj(self, task: str, variation: int, episode: int):
        # load stored demo
        demo: Demo = self.env.get_demo(task, variation, episode)[0]

        # get keypoints
        key_frame_ids = keypoint_discovery(demo)  # List[int]
        key_frame_ids.insert(0, 0)

        action_traj = []
        observations = {"rgb": [], "depth": [], "pc": []}

        # process the segment between each pair of key frames: interpolate their length to a multiple of chunk_size
        for i in range(len(key_frame_ids) - 1):
            traj_segment = []
            start_frame = key_frame_ids[i]
            end_frame = key_frame_ids[i + 1]

            for j in range(start_frame, end_frame):
                obs_dict, action, proprioception = self.env.get_obs_action(demo[j])  # action: (8), proprioception: (16)

                if not self.load_proprioception:
                    traj_segment.append(action.unsqueeze(0))
                else:
                    traj_segment.append(torch.cat([action.unsqueeze(0), proprioception.unsqueeze(0)], dim=1))

                # preprocess observation data
                if self.apply_rgb:
                    observations["rgb"].append(
                        torch.stack([self.preprocess_rgb(rgb) for rgb in obs_dict["rgb"]])  # (N, 3, H, W)
                    )
                if self.apply_depth:
                    observations["depth"].append(
                        torch.stack([self.preprocess_depth(depth) for depth in obs_dict["depth"]])  # (N, 1, H, W)
                    )
                if self.apply_pc:
                    observations["pc"].append(
                        torch.stack([self.preprocess_pc(pc) for pc in obs_dict["pc"]])  # (N, 3, H, W)
                    )

            traj_segment = torch.cat(traj_segment, dim=0)  # (n_frames, 8)

            if self.use_chunk:
                # if load trajectory in chunks, interpolate the length of each segment to a multiple of chunk_size
                len_segment = len(traj_segment)
                target_interp_length = len_segment + self.chunk_size - len_segment % self.chunk_size
                traj_segment = interpolate_trajectory(traj_segment, target_interp_length)

            action_traj.append(traj_segment)

        action_traj = torch.cat(action_traj, dim=0)  # (n_frames, 8)

        # TODO: code for loading observations images/point clouds is not implemented yet

        descriptions = demo[0].misc['descriptions']

        return action_traj, descriptions, observations

    def preprocess_rgb(self, rgb: np.ndarray) -> torch.Tensor:
        return self.rgb_transform(rgb)  # (3, H, W)

    def preprocess_depth(self, depth: np.ndarray) -> torch.Tensor:
        # Convert to float32
        depth = depth.astype(np.float32)

        # TODO: normalize

        depth = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
        return depth

    def preprocess_pc(self, pc: np.ndarray) -> torch.Tensor:
        # Convert to float32
        pc = pc.astype(np.float32)

        # TODO: normalize

        pc = torch.from_numpy(pc).permute(2, 0, 1)  # (3, H, W)
        return pc




























