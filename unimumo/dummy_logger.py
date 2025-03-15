import os
from typing import Any
from omegaconf import OmegaConf
import torch
from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.backend.observation import Observation

from unimumo.util import load_model_from_config
from unimumo.rlbench.utils_with_rlbench import RLBenchEnv, Mover, task_file_to_task_class
from unimumo.rlbench.utils_with_recorder import TaskRecorder, StaticCameraMotion, CircleCameraMotion, AttachedCameraMotion


class Logger(Callback):
    def __init__(
            self, rlb_config, save_rollout, visualize_data, save_num, train_save_freq, val_save_freq,
            vqvae_config = None, vqvae_ckpt_path = None
    ):
        super().__init__()
        self.rlb_config = rlb_config
        self.save_rollout = save_rollout
        self.visualize_data = visualize_data
        self.save_num = save_num
        self.train_save_freq = train_save_freq
        self.val_save_freq = val_save_freq

        if self.save_rollout or self.visualize_data:
            vqvae_config = OmegaConf.load(vqvae_config)
            self.vqvae = load_model_from_config(
                vqvae_config, vqvae_ckpt_path, verbose=True
            )
            self.vqvae.cuda()
            self.codebook_size = vqvae_config["model"]["params"]["quantizer_config"]["bins"]
        else:
            self.vqvae = None
            self.codebook_size = -1

        self.env: RLBenchEnv = None

    def visualize_dataset(self, batch, global_step, split: str, logger_dir: str):
        self.env = RLBenchEnv(
            data_path=pjoin(self.rlb_config["data_path"], split),
            image_size=[int(x) for x in self.rlb_config["image_size"].split(",")],
            apply_rgb=self.rlb_config["apply_rgb"],
            apply_pc=self.rlb_config["apply_pc"],
            headless=True,
            apply_cameras=self.rlb_config["apply_cameras"],
            collision_checking=False
        )
        self.env.env.launch()

        traj_code = batch["trajectory"]  # (B, T * 4)
        rgb = batch["rgb"]  # (B, T-1, ncam, 3, H, W)
        traj_code = traj_code[:self.save_num, 4:]  # (B, (T-1) * 4)
        rgb = rgb[:self.save_num, :, -1]  # (B, T-1, 3, H, W)

        for b in range(traj_code.shape[0]):
            task_str = batch["task_str"][b]
            var = batch["variation"][b]
            eps = batch["episode"][b]

            # setup the simulation environment
            task = self.env.env.get_task(task_file_to_task_class(task_str))
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
            os.makedirs(pjoin(logger_dir, "visualize_data"), exist_ok=True)
            os.makedirs(pjoin(logger_dir, "visualize_data", f"{split}_{global_step}_{b}"), exist_ok=True)
            for t in range(rgb.shape[1]):
                code = traj_code[b:b+1, t*4:(t+1)*4]  # (1, 4)
                if torch.any(code >= self.codebook_size):
                    print(f"Invalid codebook index in trajectory code: {code}")
                else:
                    with torch.no_grad():
                        traj_recon = self.vqvae.decode(code[None, ...])  # (1, 16, 8)
                        traj_recon = traj_recon.squeeze(0).detach().cpu().numpy()  # (16, 8)

                    try:
                        self.run_single_trajectory(traj_recon, task, task_str, var, eps, tr)
                        save_path = pjoin(logger_dir, "visualize_data", f"{split}_{global_step}_{b}", f"{t}")
                        print(f"Saving video to {save_path}")
                        tr.save(save_path)
                    except Exception as e:
                        print(f"Error running recon trajectory: {e}")
                        tr._snaps = {cam_name: [] for cam_name in tr._cams_motion.keys()}

                # save the ground truth image
                observation = rgb[b, t] # (3, H, W)
                observation = observation.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                plt.imsave(pjoin(logger_dir, "visualize_data", f"{split}_{global_step}_{b}", f"{t}.png"), observation)
                print(f"Saved image to {pjoin(logger_dir, 'visualize_data', f'{split}_{global_step}_{b}', f'{t}.png')}")

        self.env.env.shutdown()
        self.env = None


    def run_rollout(self, batch, global_step, split: str, pl_module: "pl.LightningModule"):
        self.env = RLBenchEnv(
            data_path=pjoin(self.rlb_config["data_path"], split),
            image_size=[int(x) for x in self.rlb_config["image_size"].split(",")],
            apply_rgb=self.rlb_config["apply_rgb"],
            apply_pc=self.rlb_config["apply_pc"],
            headless=True,
            apply_cameras=self.rlb_config["apply_cameras"],
        )
        self.env.env.launch()

        instructions = batch["instruction"]  # (B, 53, 512)

        for b in range(self.save_num):
            instruction = instructions[b:b+1]  # (1, 53, 512)
            task_str = batch["task_str"][b]
            var = batch["variation"][b]
            eps = batch["episode"][b]

            # setup the simulation environment
            task = self.env.env.get_task(task_file_to_task_class(task_str))
            task.set_variation(var)

            obs_list = self.test_policy(task, task_str, var, eps, pl_module, instruction)
            save_dir = pjoin(pl_module.logger.log_dir, "rollout")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(pjoin(save_dir, f"{split}_{global_step}_{b}"), exist_ok=True)
            for t, obs in enumerate(obs_list):
                rgb: np.ndarray = obs[-1]  # the last camera view, (H, W, 3)
                plt.imsave(pjoin(save_dir, f"{split}_{global_step}_{b}", f"{t}.png"), rgb)

        self.env.env.shutdown()
        self.env = None


    def run_single_trajectory(
            self,
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
            recorder.save_blank_frame()


    def test_policy(self, task, task_str, var, eps,
                    pl_module: "pl.LightningModule", instruction: torch.Tensor):
        assert instruction.shape == (1, 53, 512), f"Invalid instruction shape: {instruction.shape}"
        try:
            gt_demo = self.env.get_demo(task_str, var, episode_index=eps, image_paths=True)[0]
        except Exception as e:
            print(f"Error loading demo: {e}")
            return

        # reset scene
        obs: Observation = task.reset_to_demo(gt_demo)[1]
        move = Mover(task, max_tries=1)
        rgb, pcd = self.obs_to_rgb_pcd(obs)

        exe_fn = lambda code, run_partial: self.execute_function(move, code, run_partial)

        obs_list = pl_module.generate(
            instruction=instruction, rgb=rgb, pcd=pcd, execute_function=exe_fn
        )  # ndarray of (T, ncam, 3, H, W)

        return obs_list


    def execute_function(self, move: Mover, code, run_partial=True):
        # code: (1, 4)
        assert code.shape == (1, 4), f"Invalid code shape: {code.shape}"

        if torch.any(code >= self.codebook_size):
            print(f"Invalid codebook index in trajectory code: {code}")
            return None

        with torch.no_grad():
            traj_recon = self.vqvae.decode(code[None, ...])  # (1, 16, 8)
            traj_recon = traj_recon.squeeze(0).detach().cpu().numpy()  # (16, 8)

        if run_partial:
            # run the first few steps
            traj_recon = traj_recon[:4]
        else:
            # run the remaining steps
            traj_recon = traj_recon[4:]

        traj_recon[:, -1] = (traj_recon[:, -1] > 0.5).astype(np.float32)
        traj_recon[:, 3:7] /= np.linalg.norm(traj_recon[:, 3:7], axis=1, keepdims=True)

        try:
            obs = None
            for action in traj_recon:
                obs, _, _, _ = move(action, collision_checking=False)
        except Exception as e:
            print(f"Error executing trajectory: {e}")
            return None

        return self.obs_to_rgb_pcd(obs)

    def obs_to_rgb_pcd(self, obs: Observation):
        obs_dict = self.env.get_obs_action(obs)[0]

        # TODO: check the format here...
        rgb = obs_dict["rgb"]  # list of (H, W, 3), uint8
        rgb = [transforms.ToTensor()(x).unsqueeze(0) for x in rgb]
        rgb = torch.cat(rgb, dim=0).unsqueeze(0).unsqueeze(0)  # (1, 1, ncam, 3, H, W)

        pcd = obs_dict["pc"]
        pcd = [torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0) for x in pcd]
        pcd = [x.to(torch.float32) for x in pcd]
        pcd = torch.cat(pcd, dim=0).unsqueeze(0).unsqueeze(0)  # (1, 1, ncam, 3, H, W)

        return rgb.cuda(), pcd.cuda()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.visualize_data and trainer.global_step % self.train_save_freq == 0:
            self.visualize_dataset(batch, trainer.global_step, "train", pl_module.logger.log_dir)
        if self.save_rollout and trainer.global_step % self.train_save_freq == 0:
            self.run_rollout(batch, trainer.global_step, "train", pl_module)