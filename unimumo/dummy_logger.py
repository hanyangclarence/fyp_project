from typing import Any
from omegaconf import OmegaConf
import torch
from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from unimumo.util import load_model_from_config
from unimumo.rlbench.utils_with_rlbench import RLBenchEnv, Mover, task_file_to_task_class
from unimumo.rlbench.utils_with_recorder import TaskRecorder, StaticCameraMotion, CircleCameraMotion, AttachedCameraMotion


class Logger(Callback):
    def __init__(
            self, rlb_config, save_video, visualize_data, save_num, train_save_freq, val_save_freq,
            vqvae_config = None, vqvae_ckpt_path = None
    ):
        super().__init__()
        self.rlb_config = rlb_config
        self.save_video = save_video
        self.visualize_data = visualize_data
        self.save_num = save_num
        self.train_save_freq = train_save_freq
        self.val_save_freq = val_save_freq

        if self.save_video or self.visualize_data:
            vqvae_config = OmegaConf.load(vqvae_config)
            self.vqvae = load_model_from_config(
                vqvae_config, vqvae_ckpt_path, verbose=True
            )
            self.vqvae.cuda()
            self.codebook_size = vqvae_config["model"]["params"]["quantizer_config"]["bins"]
        else:
            self.vqvae = None
            self.codebook_size = -1

    def visualize_data(self, batch, global_step, split: str, logger_dir: str):
        env = RLBenchEnv(
            data_path=pjoin(self.rlb_config["data_path"], split),
            image_size=[int(x) for x in self.rlb_config["image_size"].split(",")],
            apply_rgb=self.rlb_config["apply_rgb"],
            apply_pc=self.rlb_config["apply_pc"],
            headless=True,
            apply_cameras=self.rlb_config["apply_cameras"],
            collision_checking=False
        )
        env.env.launch()

        traj_code = batch["trajectory"]  # (B, T * 4)
        rgb = batch["rgb"]  # (B, T-1, ncam, 3, H, W)
        traj_code = traj_code[:self.save_num, 4:]  # (B, (T-1) * 4)
        rgb = rgb[:self.save_num, :, -1]  # (B, T-1, 3, H, W)

        for b in range(traj_code.shape[0]):
            task_str = batch["task_str"][b]
            var = batch["variation"][b]
            eps = batch["episode"][b]

            # setup the simulation environment
            task = env.env.get_task(task_file_to_task_class(task_str))
            task.set_variation(var)

            # setup video recorder
            # Add a global camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_resolution = [240, 240]
            cam = VisionSensor.create(cam_resolution)
            cam.set_pose(cam_placeholder.get_pose())
            cam.set_parent(cam_placeholder)

            cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.005)
            cams_motion = {"global": cam_motion}
            tr = TaskRecorder(cams_motion, fps=20)

            task._scene.register_step_callback(tr.take_snap)

            # run each segment of the trajectory
            for t in range(rgb.shape[1]):
                code = rgb[b:b+1, t*4:(t+1)*4]  # (1, 4)
                if torch.any(code >= self.codebook_size):
                    print(f"Invalid codebook index in trajectory code: {code}")
                traj_recon = self.vqvae.decode(code[None, ...])  # (1, 16, 8)
                traj_recon = traj_recon.squeeze(0).cpu().numpy()  # (16, 8)

                try:
                    self.run_single_trajectory(env, traj_recon, task, task_str, var, eps, tr)
                    save_path = pjoin(logger_dir, "visualize_data", f"{split}_{global_step}_{b}_{t}")
                    tr.save(save_path)
                except Exception as e:
                    print(f"Error running recon trajectory: {e}")
                    tr._snaps = {cam_name: [] for cam_name in tr._cams_motion.keys()}

                # save the ground truth image
                observation = rgb[b, t] # (3, H, W)
                observation = observation.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                observation = observation * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                plt.imsave(pjoin(logger_dir, "visualize_data", f"{split}_{global_step}_{b}_{t}_gt.png"), observation)


    def run_single_trajectory(
            self,
            env: RLBenchEnv,
            trajectory: np.ndarray,
            task,
            task_str: str,
            var: int,
            eps: int,
            recorder: TaskRecorder
    ):
        assert len(trajectory.shape) == 2 and trajectory.shape[1] == 8, f"Invalid trajectory shape: {trajectory.shape}"
        trajectory[:, -1] = (trajectory[:, -1] > 0.5).astype(np.float32)
        trajectory[:, 3:7] /= np.linalg.norm(trajectory[:, 3:7], axis=1, keepdims=True)

        try:
            gt_demo = env.get_demo(task_str, var, episode_index=eps, image_paths=True)[0]
        except Exception as e:
            print(f"Error loading demo: {e}")
            return

        # reset scene
        _ = task.reset_to_demo(gt_demo)
        move = Mover(task, max_tries=1)

        trajectory[:, -1] = trajectory[:, -1].round()  # the last dim is the gripper state
        for action in tqdm(trajectory, desc="Executing and visualizing trajectory"):
            _ = move(action, collision_checking=False)
            recorder.save_blank_frame()


    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.visualize_data and batch_idx % self.train_save_freq == 0:
            self.visualize_data(batch, trainer.global_step, "train", pl_module.logger.log_dir)

