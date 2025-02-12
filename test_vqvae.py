import pytorch_lightning as pl
import argparse

import torch
from omegaconf import OmegaConf
import os
import torch
import numpy as np
from tqdm import tqdm

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from unimumo.util import instantiate_from_config
from unimumo.data.motion_vqvae_dataset import MotionVQVAEDataset
from unimumo.rlbench.utils_with_rlbench import RLBenchEnv, Mover, task_file_to_task_class, traj_euler_to_quat
from unimumo.rlbench.utils_with_recorder import TaskRecorder, StaticCameraMotion, CircleCameraMotion, AttachedCameraMotion


def run_single_trajectory(
    rlbench_env: RLBenchEnv,
    trajectory: np.ndarray,
    task,
    task_str: str,
    var: int,
    eps: int,
    recorder: TaskRecorder
):
    gt_demo = rlbench_env.get_demo(task_str, var, episode_index=eps, image_paths=True)[0]

    # reset scene
    _ = task.reset_to_demo(gt_demo)
    move = Mover(task, max_tries=1)

    trajectory[:, -1] = trajectory[:, -1].round()  # the last dim is the gripper state
    for action in tqdm(trajectory, desc="Executing and visualizing trajectory"):
        _ = move(action, collision_checking=False)
        recorder.save_blank_frame()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/vqvae.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="/research/d2/fyp24/hyang2/fyp/code/3d_diffuser_actor/data/peract/raw")
    parser.add_argument("--preload_data", type=bool, default=True)
    parser.add_argument("--vis_freq", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="test_logs")
    args = parser.parse_args()

    # load config
    config = OmegaConf.load(args.config)

    # set save dir
    save_dir = os.path.join(args.save_dir, f"{os.path.basename(args.config).replace('.yaml', '')}_{os.path.basename(args.ckpt).replace('.ckpt', '')}" )
    os.makedirs(save_dir, exist_ok=True)

    # load model
    model: pl.LightningModule = instantiate_from_config(config["model"])
    assert os.path.exists(args.ckpt), f"ckpt file not found: {args.ckpt}"
    model.load_from_checkpoint(args.ckpt)
    model = model.cuda().eval()

    # load dataset
    dataset = MotionVQVAEDataset(
        "test",
        args.dataset_path,
        preload_data=args.preload_data,
        load_observations=config["data"]["params"]["validation"]["load_observations"],
        load_proprioception=config["data"]["params"]["validation"]["load_proprioception"],
    )

    # init simulation env
    rlbench_env = RLBenchEnv(
        data_path=os.path.join(args.dataset_path, "test"),
        image_size=[256, 256],
        apply_rgb=False,
        apply_pc=False,
        headless=True,
        apply_cameras=("front",),
        collision_checking=False
    )
    rlbench_env.env.launch()

    # run inference
    pose_recon_losses = []
    gripper_classification_losses = []
    for i in range(len(dataset)):
        batch = dataset[i]
        gt_traj = batch["trajectory"]  # (T, D)
        description = batch["description"]
        task_str = batch["task_str"]
        variation = batch["variation"]
        episode = batch["episode"]

        # reconstruct
        gt_traj = gt_traj.unsqueeze(0).cuda()  # (1, T, D)
        with torch.no_grad():
            code = model.encode(gt_traj)
            recon_traj = model.decode(code)

        if model.motion_mode == "proprior":
            gt_traj = gt_traj[:, :, :8]

        # calculate loss
        gt_traj = gt_traj.squeeze(0).cpu()
        recon_traj = recon_traj.squeeze(0).cpu()
        pose_recon_loss = torch.nn.functional.l1_loss(gt_traj[:, :7], recon_traj[:, :7])
        gripper_classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(gt_traj[:, 7:8], recon_traj[:, 7:8])
        pose_recon_losses.append(pose_recon_loss.item())
        gripper_classification_losses.append(gripper_classification_loss.item())
        print(f"Eval {i}/{len(dataset)}: pose_recon_loss={pose_recon_loss: .4f}, gripper_classification_loss={gripper_classification_loss: .4f}")

        # visualize
        if i % args.vis_freq == 0:
            gt_traj = gt_traj.numpy()
            recon_traj = recon_traj.numpy()
            # process recon trajectory to make it feasible
            recon_traj[:, -1] = (recon_traj[:, -1] > 0.5).astype(np.float32)
            # ensure recon_traj[:, 3:7] is unit quaternion
            recon_traj[:, 3:7] /= np.linalg.norm(recon_traj[:, 3:7], axis=1, keepdims=True)

            task = rlbench_env.env.get_task(task_file_to_task_class(task_str))
            task.set_variation(variation)

            # setup video recorder
            # Add a global camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_resolution = [480, 480]
            cam = VisionSensor.create(cam_resolution)
            cam.set_pose(cam_placeholder.get_pose())
            cam.set_parent(cam_placeholder)

            cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.005)
            cams_motion = {"global": cam_motion}
            tr = TaskRecorder(cams_motion, fps=15)

            task._scene.register_step_callback(tr.take_snap)

            try:
                run_single_trajectory(
                    rlbench_env,
                    gt_traj,
                    task,
                    task_str,
                    variation,
                    episode,
                    tr
                )
                tr.save(os.path.join(
                    save_dir, f"{i}_task_{task_str}_var_{variation}_eps_{episode}_gt.mp4"
                ))
            except Exception as e:
                print(f"Error running GT trajectory: {e}")

            try:
                run_single_trajectory(
                    rlbench_env,
                    recon_traj,
                    task,
                    task_str,
                    variation,
                    episode,
                    tr
                )
                tr.save(os.path.join(
                    save_dir, f"{i}_task_{task_str}_var_{variation}_eps_{episode}_recon.mp4"
                ))
            except Exception as e:
                print(f"Error running recon trajectory: {e}")





















