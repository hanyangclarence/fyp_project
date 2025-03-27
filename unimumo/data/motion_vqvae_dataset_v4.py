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


# in v3, we always load trajectory in chunks

TASK_LIST = [
    'close_jar', 'light_bulb_in', 'meat_off_grill', 'open_drawer', 'push_buttons', 'put_groceries_in_cupboard',  'put_item_in_drawer', 'put_money_in_safe', 'slide_block_to_color_target'
]
VARIATIONS = 3

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
            load_full_traj: bool = False,  # Whether to load full trajectory
            load_traj_index: bool = False
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

        # about load content settings
        self.load_proprioception = load_proprioception
        self.load_full_traj = load_full_traj
        self.load_traj_index = load_traj_index

        # load data
        self.split = split
        self.tasks = os.listdir(pjoin(data_dir, split))
        self.all_demos_ids = []
        for task in self.tasks:
            if task not in TASK_LIST:
                continue

            # get all the variations
            variation_folders = os.listdir(pjoin(data_dir, split, task))
            variation_folders = [x for x in variation_folders if x.startswith("variation")]
            for var_folder in variation_folders:
                var = int(var_folder.replace("variation", ""))
                if var >= VARIATIONS:
                    continue
                num_episode = len(os.listdir(pjoin(data_dir, split, task, var_folder, "episodes")))
                for eps in range(num_episode):
                    self.all_demos_ids.append((task, var, eps))

        self.data = []
        self.preload_data = preload_data
        self.load_observations = load_observations
        if preload_data:
            for task, var, eps in tqdm(self.all_demos_ids, desc=f"Loading {split} data"):
                action_traj, descriptions, traj_index = self.load_obs_traj(task, var, eps, load_observations)
                self.data.append((action_traj, descriptions, traj_index, task, var, eps))
        print(f"{split} data loaded, total number of demos: {len(self.all_demos_ids)}")

    def __len__(self):
        return len(self.all_demos_ids)

    def __getitem__(self, idx):
        if self.preload_data:
            action_traj, descriptions, traj_index, task, var, eps = self.data[idx]
        else:
            task, var, eps = self.all_demos_ids[idx]
            action_traj, descriptions, traj_index = self.load_obs_traj(task, var, eps, self.load_observations)
        len_traj = len(action_traj)

        if self.load_full_traj:
            # directly return the trajectory and description
            data_dict = {
                "trajectory": action_traj.float(),  # (T, 8)
                "description": descriptions[0],  # str
                "task_str": task,  # str
                "variation": var,  # int
                "episode": eps,  # int
            }
            if self.load_traj_index:
                data_dict["trajectory_index"] = traj_index
            return data_dict

        assert len_traj % (self.chunk_size * self.n_chunk_per_traj) == 0, f"Trajectory length must be a multiple of chunk_size * n_chunk_per_traj, but got {len_traj}"
        # directly sample a trajectory
        start_idx = random.randint(0, len_traj // (self.chunk_size * self.n_chunk_per_traj) - 1) * self.chunk_size * self.n_chunk_per_traj
        end_idx = start_idx + self.n_chunk_per_traj * self.chunk_size
        traj = action_traj[start_idx:end_idx].float()  # (T, 8)
        sliced_traj_index = traj_index[start_idx:end_idx]  # (T,)


        # sample a random description
        desc = random.choice(descriptions)

        data_dict = {
            "trajectory": traj,  # (T, 8)
            "description": desc,  # str
            "task_str": task,  # str
            "variation": var,  # int
            "episode": eps,  # int
        }
        if self.load_traj_index:
            data_dict["trajectory_index"] = sliced_traj_index
        return data_dict


    def load_obs_traj(self, task: str, variation: int, episode: int, load_observations: bool = False):
        # load stored demo
        demo: Demo = self.env.get_demo(task, variation, episode, image_paths= not load_observations)[0]

        # get keypoints
        key_frame_ids = keypoint_discovery(demo)  # List[int]
        key_frame_ids.insert(0, 0)

        action_traj = []
        traj_index = []

        # process the segment between each pair of key frames: interpolate their length to a multiple of chunk_size
        for i in range(len(key_frame_ids) - 1):
            traj_segment = []
            start_frame = key_frame_ids[i]
            end_frame = key_frame_ids[i + 1]

            # find the indices of the sparse action frames

            if (end_frame - start_frame) == 1:
                # if only one frame, directly add it
                final_indices = [start_frame] * (self.chunk_size * self.n_chunk_per_traj)
            else:
                len_sparce = self.chunk_size * self.n_chunk_per_traj
                delta = (end_frame - start_frame) / (len_sparce - 1)

                final_indices = []
                for num_delta in range(len_sparce):
                    final_indices.append(int(start_frame + num_delta * delta))

            for j in final_indices:
                _, action, proprioception = self.env.get_obs_action(demo[j])  # action: (8), proprioception: (16)

                if not self.load_proprioception:
                    traj_segment.append(action.unsqueeze(0))
                else:
                    traj_segment.append(torch.cat([action.unsqueeze(0), proprioception.unsqueeze(0)], dim=1))

                traj_index.append(j)

            traj_segment = torch.cat(traj_segment, dim=0)  # (n_frames, 8)
            action_traj.append(traj_segment)

        action_traj = torch.cat(action_traj, dim=0)  # (n_frames, 8)
        traj_index = torch.tensor(traj_index, dtype=torch.long)

        # TODO: code for loading observations images/point clouds is not implemented yet

        descriptions = demo[0].misc['descriptions']

        return action_traj, descriptions, traj_index


























