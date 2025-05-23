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

MIN_DELTA_TRANS = 0.05
MIN_DELTA_ROT = 0.4


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

        self.use_chunk = use_chunk
        self.chunk_size = chunk_size
        self.n_chunk_per_traj = n_chunk_per_traj

        # about load content settings
        self.load_proprioception = load_proprioception
        self.load_sparce = load_sparce
        self.load_full_traj = load_full_traj
        self.load_traj_index = load_traj_index

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

        # directly sample a trajectory
        start_idx = random.randint(0, len_traj - self.n_chunk_per_traj * self.chunk_size)
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

            if self.load_sparce:
                if (end_frame - start_frame) == 1:
                    # if only one frame, directly add it
                    final_indices = [start_frame] * self.chunk_size
                else:
                    selected_indices = []
                    tempt_selected_traj = []
                    for j in range(start_frame, end_frame):
                        _, action, _ = self.env.get_obs_action(demo[j])
                        if len(selected_indices) == 0 or (i == len(key_frame_ids) - 2 and j == end_frame - 1):
                            # always keep the first frame and the last frame
                            delta_trans = 999
                            delta_rot = 999
                        else:
                            delta_trans = torch.norm(action[:3] - tempt_selected_traj[-1][0, :3])
                            r1 = R.from_quat(tempt_selected_traj[-1][0, 3:7])
                            r2 = R.from_quat(action[3:7])
                            relative_rot = r1.inv() * r2
                            delta_rot = relative_rot.magnitude()
                        if delta_trans < MIN_DELTA_TRANS and delta_rot < MIN_DELTA_ROT:
                            # skip the frame if the action is too small
                            continue
                        selected_indices.append(j)
                        tempt_selected_traj.append(action.unsqueeze(0))
                    len_sparce = len(selected_indices)
                    # make it a multiple of chunk_size
                    extra_length = (self.chunk_size - len_sparce % self.chunk_size) % self.chunk_size
                    len_sparce += extra_length
                    delta = (end_frame - 1 - start_frame) / (len_sparce - 1)

                    final_indices = []
                    for num_delta in range(len_sparce):
                        final_indices.append(int(start_frame + num_delta * delta))
            else:
                final_indices = range(start_frame, end_frame)

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


























