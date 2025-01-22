import torch
import os
from torch.utils.data import Dataset
from os.path import join as pjoin
import random
from typing import Tuple
from tqdm import tqdm
import quaternion
import numpy as np

from rlbench.demo import Demo

from unimumo.rlbench.utils_with_rlbench import RLBenchEnv, keypoint_discovery, interpolate_trajectory


class MotionVQVAEDataset(Dataset):
    def __init__(
            self,
            split: str,
            data_dir: str,
            preload_data: bool = False,
            cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist", "front"),
            image_size: str = "256,256",
            load_observations: bool = False,
            load_quaternion: bool = True,
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
                action_traj, descriptions = self.load_obs_traj(task, var, eps, load_observations)
                self.data.append((action_traj, descriptions, task, var, eps))
        print(f"{split} data loaded, total number of demos: {len(self.all_demos_ids)}")

        self.load_quaternion = load_quaternion

    def __len__(self):
        return len(self.all_demos_ids)

    def __getitem__(self, idx):
        if self.preload_data:
            action_traj, descriptions, task, var, eps = self.data[idx]
        else:
            task, var, eps = self.all_demos_ids[idx]
            action_traj, descriptions = self.load_obs_traj(task, var, eps, self.load_observations)
        len_traj = len(action_traj)

        # sample a random chunk
        start_idx = random.randint(0, len_traj // self.chunk_size - self.n_chunk_per_traj) * self.chunk_size
        end_idx = start_idx + self.chunk_size * self.n_chunk_per_traj
        traj = action_traj[start_idx:end_idx].float()  # (T, 8), also convert to float32 tensor
        print(start_idx, end_idx, traj.shape, "herehere")

        # TODO: data augmentation?

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
                _, action = self.env.get_obs_action(demo[j])  # action: (8)

                if not self.load_quaternion:
                    trans = action[:3]
                    rot_quat = action[3:7]
                    gripper = action[7:]
                    # convert quaternion to euler angles
                    rot_angle = quaternion.as_euler_angles(quaternion.quaternion(rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]))
                    action = np.concatenate([trans, rot_angle, gripper], axis=0)
                    action = torch.tensor(action, dtype=torch.float32)

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


























