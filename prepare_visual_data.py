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
from scipy.spatial.transform import Rotation as R

from rlbench.demo import Demo

from unimumo.rlbench.utils_with_rlbench import RLBenchEnv, keypoint_discovery, interpolate_trajectory, load_depth, load_rgb, get_point_cloud

MIN_DELTA_TRANS = 0.03
MIN_DELTA_ROT = 0.2


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
            load_sparce: bool = False,  # Whether to load sparse trajectory
            load_full_traj: bool = False,  # Whether to load the complete trajectory for testing
    ):
        # load RLBench environment
        self.split = split
        self.env = RLBenchEnv(
            data_path=pjoin(data_dir, split),
            image_size=[int(x) for x in image_size.split(",")],
            apply_rgb=apply_rgb,
            apply_pc=apply_pc,
            apply_depth=apply_depth,
            apply_cameras=cameras,
        )

        self.use_chunk = use_chunk
        self.chunk_size = chunk_size
        self.n_chunk_per_traj = n_chunk_per_traj
        self.compression_rate = compression_rate

        # about load content settings
        self.load_proprioception = load_proprioception
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.load_sparce = load_sparce
        self.load_full_traj = load_full_traj

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),  # converts to float32 and scales to [0,1]
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
                action_traj, observations, misc = self.load_obs_traj(task, var, eps)
                self.data.append((action_traj, observations, misc, task, var, eps))
        print(f"{split} data loaded, total number of demos: {len(self.all_demos_ids)}")

    def __len__(self):
        return len(self.all_demos_ids)

    def __getitem__(self, idx):
        if self.preload_data:
            action_traj, observations, misc, task, var, eps = self.data[idx]
        else:
            task, var, eps = self.all_demos_ids[idx]
            action_traj, observations, misc = self.load_obs_traj(task, var, eps)
        len_traj = len(action_traj)

        # directly sample a trajectory
        if self.load_full_traj:
            start_idx = 0
            end_idx = len_traj
        else:
            start_idx = random.randint(0, len_traj - self.n_chunk_per_traj * self.chunk_size)
            end_idx = start_idx + self.n_chunk_per_traj * self.chunk_size
        traj = action_traj[start_idx:end_idx].float()  # (T, 8)

        # sample a random description
        desc = random.choice(misc["descriptions"])

        data_dict = {
            "trajectory": traj,  # (T, 8)
            "description": desc,  # str
            "task_str": task,  # str
            "variation": var,  # int
            "episode": eps,  # int
        }

        obs_ids_path = f"data/motion_code/{self.split}/{task}/var_{var}_eps_{eps}/indices.npy"
        obs_ids = np.load(obs_ids_path)

        if self.apply_rgb:
            all_frame_rgb_paths = [observations["rgb"][i] for i in obs_ids]
            data_dict["rgb"] = []
            for rgb_paths_per_frame in all_frame_rgb_paths:
                rgbs = []
                for rgb_path in rgb_paths_per_frame:
                    cam_name = rgb_path.split("/")[-2][:-4]  # remove the "_rgb" suffix
                    rgbs.append(
                        load_rgb(
                            rgb_path=rgb_path,
                            cam_config=getattr(self.env.env._obs_config, f"{cam_name}_camera")
                        )
                    )
                rgbs = torch.stack(
                    [self.preprocess_rgb(rgb) for rgb in rgbs]
                )  # (N, 3, H, W)
                data_dict["rgb"].append(rgbs)
            data_dict["rgb"] = torch.stack(data_dict["rgb"])  # (T', N, 3, H, W)

        if self.apply_depth or self.apply_pc:
            # calculating pc also needs depth
            all_frame_depth_paths = [observations["depth"][i] for i in obs_ids]
            data_dict["depth"] = []
            all_frame_depth_m = []
            for frame_id, depth_paths_per_frame in zip(obs_ids, all_frame_depth_paths):
                depths = []
                depths_m = {}
                for depth_path in depth_paths_per_frame:
                    cam_name = depth_path.split("/")[-2][:-6]  # remove the "_depth" suffix
                    # load unscaled depth
                    depth = load_depth(
                        depth_path=depth_path,
                        cam_config=getattr(self.env.env._obs_config, f"{cam_name}_camera")
                    )
                    # scale depth if loading pc or specified by config
                    if getattr(self.env.env._obs_config, f"{cam_name}_camera").depth_in_meters or self.apply_pc:
                        near = misc[f"{cam_name}_camera_near"][frame_id]
                        far = misc[f"{cam_name}_camera_far"][frame_id]
                        depth_m = near + (far - near) * depth
                    if getattr(self.env.env._obs_config, f"{cam_name}_camera").depth_in_meters:
                        depths.append(depth_m)
                    else:
                        depths.append(depth)
                    if self.apply_pc:
                        depths_m[cam_name] = depth_m

                depths = torch.stack(
                    [self.preprocess_depth(depth) for depth in depths]
                )  # (N, 1, H, W)
                data_dict["depth"].append(depths)

                all_frame_depth_m.append(depths_m)
            data_dict["depth"] = torch.stack(data_dict["depth"])  # (T', N, 1, H, W)

        if self.apply_pc:
            data_dict["pc"] = []
            for frame_id, depths_m in zip(obs_ids, all_frame_depth_m):
                pcs = []
                for cam_name, depth_m in depths_m.items():
                    pc = get_point_cloud(
                        depth_m=depth_m,
                        extrinsic=misc[f"{cam_name}_camera_extrinsics"][frame_id],
                        intrinsic=misc[f"{cam_name}_camera_intrinsics"][frame_id],
                    )
                    pcs.append(pc)
                data_dict["pc"].append(torch.stack(
                    [self.preprocess_pc(pc) for pc in pcs]
                ))
            data_dict["pc"] = torch.stack(data_dict["pc"])  # (T', N, 3, H, W)

        data_dict["trajectory_index"] = obs_ids.tolist()

        return data_dict


    def load_obs_traj(self, task: str, variation: int, episode: int):
        # load stored demo
        demo: Demo = self.env.get_demo(task, variation, episode, image_paths=True)[0]

        # get keypoints
        key_frame_ids = keypoint_discovery(demo)  # List[int]
        key_frame_ids.insert(0, 0)

        action_traj = []
        observations = {"rgb": [], "depth": [], "pc": []}
        misc = {k: [] for k in demo[0].misc.keys()}
        misc["descriptions"] = demo[0].misc["descriptions"]

        # process the segment between each pair of key frames: interpolate their length to a multiple of chunk_size
        for i in range(len(key_frame_ids) - 1):
            traj_segment = []
            start_frame = key_frame_ids[i]
            end_frame = key_frame_ids[i + 1]

            for j in range(start_frame, end_frame):
                obs_dict, action, proprioception = self.env.get_obs_action(demo[j])  # action: (8), proprioception: (16)

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
                    # skip the frame if the action is too small
                    continue

                if not self.load_proprioception:
                    traj_segment.append(action.unsqueeze(0))
                else:
                    traj_segment.append(torch.cat([action.unsqueeze(0), proprioception.unsqueeze(0)], dim=1))

                # add observation data
                if self.apply_rgb:
                    observations["rgb"].append(obs_dict["rgb"])  # list of str
                if self.apply_depth:
                    observations["depth"].append(obs_dict["depth"])
                if self.apply_pc:
                    observations["pc"].append(obs_dict["pc"])

                # add misc data
                for k, v in demo[j].misc.items():
                    if k != "descriptions":
                        misc[k].append(v)

            traj_segment = torch.cat(traj_segment, dim=0)  # (n_frames, 8)

            if self.use_chunk:
                # if load trajectory in chunks, repeat the last frame to make the length a multiple of chunk_size
                len_segment = len(traj_segment)
                extra_length = self.chunk_size - len_segment % self.chunk_size
                if extra_length != self.chunk_size:
                    traj_segment = torch.cat([traj_segment, traj_segment[-1].unsqueeze(0).repeat(extra_length, 1)], dim=0)
                    for _ in range(extra_length):
                        if self.apply_rgb:
                            observations["rgb"].append(observations["rgb"][-1])
                        if self.apply_depth:
                            observations["depth"].append(observations["depth"][-1])
                        if self.apply_pc:
                            observations["pc"].append(observations["pc"][-1])
                        for k in misc.keys():
                            if k != "descriptions":
                                misc[k].append(misc[k][-1])


            action_traj.append(traj_segment)

        action_traj = torch.cat(action_traj, dim=0)  # (n_frames, 8)

        return action_traj, observations, misc

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



if __name__ == "__main__":

    save_dir = "data/observations"
    os.makedirs(save_dir, exist_ok=True)

    for split in ["train", "test", "val"]:
        dataset = MotionVQVAEDataset(
            split=split,
            data_dir="/research/d2/fyp24/hyang2/fyp/code/3d_diffuser_actor/data/peract/raw",
            preload_data=False,
            apply_rgb=True,
            apply_pc=True,
            apply_depth=True,
            cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
            image_size="256,256",
            load_proprioception=True,
            use_chunk=False,
            chunk_size=4,
            n_chunk_per_traj=4,
            compression_rate=4,
            load_sparce=False,
            load_full_traj=True
        )

        os.makedirs(pjoin(save_dir, split), exist_ok=True)

        for i in range(len(dataset)):
            print(f"Processing {split} data {i}/{len(dataset)}")
            data = dataset[i]

            task_str = data["task_str"]
            variation = data["variation"]
            episode = data["episode"]

            rgb_list = data["rgb"]  # (T', N, 3, H, W)
            pcd_list = data["pc"]   # (T', N, 3, H, W)

            rgb_pcd = torch.cat([rgb_list, pcd_list], dim=2) # (T', N, 6, H, W)
            torch.save(rgb_pcd, pjoin(save_dir, split, f"{task_str}_var_{variation}_eps_{episode}.pt"))




























