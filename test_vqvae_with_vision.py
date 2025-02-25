import pytorch_lightning as pl
import argparse

import torch
from omegaconf import OmegaConf
import os
import torch
import numpy as np
from tqdm import tqdm
import typing as tp

from rlbench.backend.observation import Observation
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from unimumo.util import instantiate_from_config, load_model_from_config
from unimumo.data.motion_vqvae_dataset_with_vision import MotionVQVAEDataset
from unimumo.rlbench.utils_with_rlbench import RLBenchEnv, Mover, task_file_to_task_class, traj_euler_to_quat
from unimumo.rlbench.utils_with_recorder import TaskRecorder, StaticCameraMotion, CircleCameraMotion, AttachedCameraMotion


def run_step(
    trajectory,
    move: Mover,
    recorder: TaskRecorder = None,
) -> tp.Tuple[Observation, float]:
    trajectory = trajectory.squeeze(0).cpu().numpy()
    trajectory[:, -1] = (trajectory[:, -1] > 0.5).astype(np.float32)
    trajectory[:, 3:7] /= np.linalg.norm(trajectory[:, 3:7], axis=1, keepdims=True)

    obs = None
    reward = 0.0
    for action in tqdm(trajectory, desc="Executing Trajectory Section"):
        obs, reward, _, _ = move(action, collision_checking=False)
        if recorder is not None:
            recorder.take_snap()
        if reward > 0.0:
            break

    return obs, reward



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/vqvae.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="/research/d2/fyp24/hyang2/fyp/code/3d_diffuser_actor/data/peract/raw")
    parser.add_argument("--vis_freq", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="test_logs")
    parser.add_argument("--cal_reward", action="store_true")
    args = parser.parse_args()

    # load config
    config = OmegaConf.load(args.config)

    # set save dir
    save_dir = os.path.join(args.save_dir, f"{os.path.basename(args.config).replace('.yaml', '')}_{os.path.basename(args.ckpt).replace('.ckpt', '')}" )
    os.makedirs(save_dir, exist_ok=True)

    # load model
    model = load_model_from_config(config, args.ckpt, verbose=True)
    model.cuda()

    # load dataset
    chunk_size = config["data"]["params"]["validation"]["params"]["chunk_size"]
    n_chunk_per_traj = config["data"]["params"]["validation"]["params"]["n_chunk_per_traj"]
    dataset = MotionVQVAEDataset(
        "test",
        args.dataset_path,
        preload_data=False,
        apply_rgb=config["data"]["params"]["validation"]["params"]["apply_rgb"],
        apply_depth=config["data"]["params"]["validation"]["params"]["apply_depth"],
        apply_pc=config["data"]["params"]["validation"]["params"]["apply_pc"],
        cameras=config["data"]["params"]["validation"]["params"]["cameras"],
        image_size=config["data"]["params"]["validation"]["params"]["image_size"],
        load_proprioception=config["data"]["params"]["validation"]["params"]["load_proprioception"],
        use_chunk=config["data"]["params"]["validation"]["params"]["use_chunk"],
        chunk_size=chunk_size,
        n_chunk_per_traj=n_chunk_per_traj,
        compression_rate=config["data"]["params"]["validation"]["params"]["compression_rate"],
        load_sparce=config["data"]["params"]["validation"]["params"]["load_sparce"],
        load_full_traj=True
    )

    # init simulation env
    rlbench_env = RLBenchEnv(
        data_path=os.path.join(args.dataset_path, "test"),
        image_size=[int(x) for x in config["data"]["params"]["validation"]["params"]["image_size"].split(",")],
        apply_rgb=config["data"]["params"]["validation"]["params"]["apply_rgb"],
        apply_depth=config["data"]["params"]["validation"]["params"]["apply_depth"],
        apply_pc=config["data"]["params"]["validation"]["params"]["apply_pc"],
        headless=True,
        apply_cameras=config["data"]["params"]["validation"]["params"]["cameras"],
        collision_checking=False,
    )
    rlbench_env.env.launch()

    # run inference
    pose_recon_losses = []
    gripper_classification_losses = []
    all_rewards = []
    for i in range(len(dataset)):
        batch = dataset[i]
        gt_traj = batch["trajectory"]  # (T, D)
        description = batch["description"]
        task_str = batch["task_str"]
        variation = batch["variation"]
        episode = batch["episode"]
        obs_rgb = batch["rgb"]  # (T', N, 3, H, W)

        # pad the gt_traj to make it length divisible by 4
        gt_traj_padded = torch.zeros((gt_traj.shape[0] + (chunk_size - gt_traj.shape[0] % chunk_size) % chunk_size, gt_traj.shape[1]))
        gt_traj_padded[:gt_traj.shape[0], :] = gt_traj
        gt_traj_padded[gt_traj.shape[0]:, :] = gt_traj[-1]
        gt_traj = gt_traj_padded

        task = rlbench_env.env.get_task(task_file_to_task_class(task_str))
        task.set_variation(variation)

        if (i + 1) % args.vis_freq == 0:
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
        else:
            tr = None

        gt_demo = rlbench_env.get_demo(task_str, variation, episode_index=episode, image_paths=True)[0]

        # reset scene
        _ = task.reset_to_demo(gt_demo)
        move = Mover(task, max_tries=1)

        # reconstruct
        gt_traj = gt_traj.unsqueeze(0).cuda()  # (1, T, D)
        traj_length = chunk_size * n_chunk_per_traj
        obs_id_counter = traj_length // config["data"]["params"]["validation"]["params"]["compression_rate"]
        rgb = obs_rgb[:obs_id_counter].unsqueeze(0).cuda()  # (1, , N, 3, H, W)
        with torch.no_grad():
            all_recon_traj = []
            for sec_start in range(0, gt_traj.shape[1] - traj_length + chunk_size, chunk_size):
                traj_chunk = gt_traj[:, sec_start:sec_start+traj_length]
                code = model.encode(traj_chunk)
                recon_traj = model.decode(code, rgb)
                if sec_start == 0:
                    all_recon_traj.append(recon_traj)
                    obs, _ = run_step(recon_traj, move, tr)
                else:
                    all_recon_traj.append(recon_traj[:, -chunk_size:, :])
                    obs, _ = run_step(recon_traj[:, -chunk_size:, :], move, tr)

                # Choose either use the gt observation, or observation from simulator rollout

                new_rgbs = []
                for cam_name in config["data"]["params"]["validation"]["params"]["cameras"]:
                    new_rgbs.append(
                        getattr(obs, f"{cam_name}_rgb")  # shape (H, W, 3)
                    )
                new_rgbs = torch.stack(
                    [dataset.preprocess_rgb(new_rgb) for new_rgb in new_rgbs]
                )[None, None, ...].cuda()  # (1, 1, N, 3, H, W)

                # new_rgbs = obs_rgb[obs_id_counter:obs_id_counter+1].unsqueeze(0).cuda()  # (1, 1, N, 3, H, W)
                # obs_id_counter += 1

                rgb = torch.cat(
                    [rgb[:, 1:], new_rgbs], dim=1
                )

                # tempt save image for visualization and debug
                # import matplotlib.pyplot as plt
                # plt.imsave(os.path.join(save_dir, f"{i}_task_{task_str}_var_{variation}_eps_{episode}_sec_{sec_start}.png"), obs.front_rgb)

            recon_traj = torch.cat(all_recon_traj, dim=1)
            print(f"GT shape: {gt_traj.shape}, recon shape: {recon_traj.shape}")

        if model.motion_mode == "proprior":
            gt_traj = gt_traj[:, :, :8]

        # calculate loss
        gt_traj = gt_traj.squeeze(0).cpu()
        recon_traj = recon_traj.squeeze(0).cpu()
        pose_recon_loss = torch.nn.functional.smooth_l1_loss(gt_traj[:, :7], recon_traj[:, :7], reduction="mean")
        gripper_classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(gt_traj[:, 7:8], recon_traj[:, 7:8], reduction="mean")
        pose_recon_losses.append(pose_recon_loss.item())
        gripper_classification_losses.append(gripper_classification_loss.item())
        print(f"Eval {i}/{len(dataset)}: pose_recon_loss={pose_recon_loss: .4f}, gripper_classification_loss={gripper_classification_loss: .4f} ")

        # save video
        if (i + 1) % args.vis_freq == 0 and tr is not None:
            tr.save(os.path.join(
                save_dir, f"{i}_task_{task_str}_var_{variation}_eps_{episode}_recon.mp4"
            ))

    print(f"Average pose_recon_loss={np.mean(pose_recon_losses): .4f}, average gripper_classification_loss={np.mean(gripper_classification_losses): .4f}")
    if len(all_rewards) > 0:
        print(f"Average reward={np.mean(all_rewards): .4f}")
    rlbench_env.env.shutdown()
    # write losses to file
    with open(os.path.join(save_dir, "losses.txt"), "w") as f:
        content = f"Average pose_recon_loss={np.mean(pose_recon_losses): .4f}\nAverage gripper_classification_loss={np.mean(gripper_classification_losses): .4f}\n"
        if len(all_rewards) > 0:
            content += f"Average reward={np.mean(all_rewards): .4f}\n"
        f.write(content)





















