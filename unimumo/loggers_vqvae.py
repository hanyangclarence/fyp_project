import os
from os.path import join as pjoin
import random
import typing as tp
from typing import Any

import torch
import numpy as np
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning import LightningModule, Trainer

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from unimumo.rlbench.utils_with_rlbench import RLBenchEnv, Mover, task_file_to_task_class
from unimumo.rlbench.utils_with_recorder import TaskRecorder, StaticCameraMotion, CircleCameraMotion, AttachedCameraMotion


class TrajectoryLogger(Callback):
    def __init__(
            self,
            save_video: bool = True,
            num_videos: int = 1,
            rlb_config: dict = None,
    ):
        super().__init__()
        self.save_video = save_video
        self.num_videos = num_videos
        self.rlb_config = rlb_config

        self.env = None

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.save_video and trainer.global_rank == 0:
            batch = pl_module.last_train_batch
            self.visualize_save_trajectories(batch, pl_module, "train")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.save_video and trainer.global_rank == 0:
            batch = pl_module.last_val_batch
            self.visualize_save_trajectories(batch, pl_module, "val")

    @rank_zero_only
    def visualize_save_trajectories(self, batch: Any, pl_module: LightningModule, split: str) -> None:
        print(f"Visualizing {split} trajectory on epoch {pl_module.current_epoch}")

        is_training = pl_module.training
        if is_training:
            pl_module.eval()

        trajectory = batch["trajectory"]  # (B, T, 8)
        description = batch["description"]  # (B,)
        task_strs = batch["task_str"]
        variations = batch["variation"]
        episodes = batch["episode"]

        with torch.no_grad():
            code = pl_module.encode(trajectory)
            trajectory_recon = pl_module.decode(code)

        # initialize env
        self.env = RLBenchEnv(
            data_path=pjoin(self.rlb_config["data_path"], "val"),
            image_size=[int(x) for x in self.rlb_config["image_size"].split(",")],
            apply_rgb=self.rlb_config["apply_rgb"],
            apply_pc=self.rlb_config["apply_pc"],
            headless=True,
            apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
            collision_checking=False
        )
        self.env.env.launch()

        for b in range(min(len(description), self.num_videos)):
            task_str = task_strs[b]
            var = variations[b]
            eps = episodes[b]

            gt_traj = trajectory[b].cpu().numpy()
            recon_traj = trajectory_recon[b].cpu().numpy()
            desc = description[b]

            # process recon trajectory to make it feasible
            recon_traj[:, -1] = (recon_traj[:, -1] > 0.5).astype(np.float32)
            # ensure recon_traj[:, 3:7] is unit quaternion
            recon_traj[:, 3:7] /= np.linalg.norm(recon_traj[:, 3:7], axis=1, keepdims=True)

            task = self.env.env.get_task(task_file_to_task_class(task_str))
            task.set_variation(var)

            # setup video recorder
            # Add a global camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_resolution = [480, 480]
            cam = VisionSensor.create(cam_resolution)
            cam.set_pose(cam_placeholder.get_pose())
            cam.set_parent(cam_placeholder)

            # cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.005)
            global_cam_motion = StaticCameraMotion(cam)
            cams_motion = {"global": global_cam_motion}
            tr = TaskRecorder(cams_motion, fps=15)

            task._scene.register_step_callback(tr.take_snap)
            print(f"Finished setting up video recorder for task {task_str}")

            # run trajectories
            save_dir = pjoin(pl_module.logger.log_dir, split)
            os.makedirs(save_dir, exist_ok=True)

            self.run_single_trajectory(gt_traj, task, task_str, var, eps)
            tr.save(pjoin(save_dir, f"e{pl_module.current_epoch}_b{b}_var{var}_eps{eps}_gt.mp4"))

            self.run_single_trajectory(recon_traj, task, task_str, var, eps)
            tr.save(pjoin(save_dir, f"e{pl_module.current_epoch}_b{b}_var{var}_eps{eps}_recon.mp4"))

        # clean up
        self.env.env.shutdown()
        self.env = None

        if is_training:
            pl_module.train()


    def run_single_trajectory(
            self,
            trajectory: np.ndarray,
            task,
            task_str: str,
            var: int,
            eps: int,
    ):
        try:
            gt_demo = self.env.get_demo(task_str, var, episode_index=eps, image_paths=True)[0]
        except Exception as e:
            print(f"Error loading demo: {e}")
            return

        # reset scene
        _ = task.reset_to_demo(gt_demo)
        move = Mover(task, max_tries=1)

        trajectory[:, -1] = trajectory[:, -1].round()  # the last dim is the gripper state
        for action in tqdm(trajectory, desc="Executing and visualizing trajectory"):
            _ = move(action, collision_checking=False)












