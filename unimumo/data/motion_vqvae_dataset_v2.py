import torch
import os
from torch.utils.data import Dataset
from os.path import join as pjoin
import random
from typing import Tuple
from tqdm import tqdm
import quaternion
import numpy as np
from scipy.spatial.transform import Rotation as R
from rlbench.demo import Demo

from unimumo.rlbench.utils_with_rlbench import RLBenchEnv, keypoint_discovery, interpolate_trajectory

MIN_DELTA_TRANS = 0.03
MIN_DELTA_ROT = 0.2


class MotionVQVAEDataset(Dataset):
    def __init__(
            self,
            split: str,
            data_dir: str,
            preload_data: bool = False,
            cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist", "front"),
            image_size: str = "256,256",
            load_observations: bool = False,  # Whether to load rgb/depth/pc for each timestep
            load_proprioception: bool = False,  # Whether to load proprioception data, concatenated with the gripper pose
            use_chunk: bool = True,  # Whether to load trajectory in chunks (segmented by key frames)
            chunk_size: int = 4,  # number of frames in a chunk
            n_chunk_per_traj: int = 2,  # number of chunks in a trajectory
            load_sparce: bool = False,  # Whether to load sparse trajectory
    ):
        # load RLBench environment
        self.env = RLBenchEnv(
            data_path=pjoin(data_dir, split),
            image_size=[int(x) for x in image_size.split(",")],
            apply_rgb=True,
            apply_pc=True,
            apply_cameras=cameras,
        )

        self.use_chunk = use_chunk
        self.chunk_size = chunk_size
        self.n_chunk_per_traj = n_chunk_per_traj

        # about load content settings
        self.load_proprioception = load_proprioception
        self.load_sparce = load_sparce

        # load data
        self.split = split
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

        if split == "train":
            assert preload_data == True, "preload_data must be True for training data"
            # also load validation data
            val_dataset = MotionVQVAEDataset(
                "val",
                data_dir,
                preload_data=True,
                cameras=cameras,
                image_size=image_size,
                load_observations=load_observations,
                load_proprioception=load_proprioception,
                use_chunk=use_chunk,
                chunk_size=chunk_size,
                n_chunk_per_traj=n_chunk_per_traj,
                load_sparce=load_sparce
            )
            self.data.extend(val_dataset.data)

        self.preload_data = preload_data
        self.load_observations = load_observations
        if preload_data:
            for task, var, eps in tqdm(self.all_demos_ids, desc=f"Loading {split} data"):
                action_traj, descriptions = self.load_obs_traj(task, var, eps, load_observations)
                self.data.append((action_traj, descriptions, task, var, eps))
        print(f"{split} data loaded, total number of demos: {len(self.all_demos_ids)}")

    def __len__(self):
        return len(self.all_demos_ids)

    def __getitem__(self, idx):
        if self.preload_data:
            action_traj, descriptions, task, var, eps = self.data[idx]
        else:
            task, var, eps = self.all_demos_ids[idx]
            action_traj, descriptions = self.load_obs_traj(task, var, eps, self.load_observations)
        len_traj = len(action_traj)

        if self.split == "test":
            # for test set, directly return the trajectory and description
            return {
                "trajectory": action_traj.float(),  # (T, 8)
                "description": descriptions[0],  # str
                "task_str": task,  # str
                "variation": var,  # int
                "episode": eps,  # int
            }
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

        return  {
            "trajectory": traj,  # (T, 8)
            "description": desc,  # str
            "task_str": task,  # str
            "variation": var,  # int
            "episode": eps,  # int
        }


    def load_obs_traj(self, task: str, variation: int, episode: int, load_observations: bool = False):
        # load stored demo
        demo: Demo = self.env.get_demo(task, variation, episode, image_paths= not load_observations)[0]

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
                _, action, proprioception = self.env.get_obs_action(demo[j])  # action: (8), proprioception: (16)

                if len(traj_segment) == 0 or i == len(key_frame_ids) - 2:
                    # always keep the first frame and the last frame
                    delta_trans = 999
                    delta_rot = 999
                else:
                    delta_trans = torch.norm(action[:3] - traj_segment[-1][0, :3])
                    r1 = R.from_quat(traj_segment[-1][0, 3:7])
                    r2 = R.from_quat(action[3:7])
                    relative_rot = r1.inv() * r2
                    delta_rot = relative_rot.magnitude()
                if self.load_sparce and delta_trans < MIN_DELTA_TRANS and delta_rot < MIN_DELTA_ROT:
                    if j == end_frame - 1 and len(traj_segment) == 1 and self.use_chunk:
                        # keep the last frame if the segment is too short after filtering, which could cause error in interpolation
                        pass
                    else:
                        # skip the frame if the action is too small
                        continue

                if not self.load_proprioception:
                    traj_segment.append(action.unsqueeze(0))
                else:
                    traj_segment.append(torch.cat([action.unsqueeze(0), proprioception.unsqueeze(0)], dim=1))

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

        return action_traj, descriptions


if __name__ == "__main__":
    data_dir = "/research/d2/fyp24/hyang2/fyp/code/3d_diffuser_actor/data/peract/raw"
    dataset = MotionVQVAEDataset("val", data_dir)

    for i in range(len(dataset)):
        sample = dataset.__getitem__(i)
        print(sample)


























