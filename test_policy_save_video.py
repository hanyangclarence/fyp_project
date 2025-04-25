import os
from typing import Any
from omegaconf import OmegaConf
import torch
from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.backend.observation import Observation

from unimumo.util import load_model_from_config, instantiate_from_config
from unimumo.rlbench.utils_with_rlbench import RLBenchEnv, Mover, task_file_to_task_class
from unimumo.rlbench.utils_with_recorder import TaskRecorder, StaticCameraMotion, CircleCameraMotion, AttachedCameraMotion


data_path = "/research/d2/fyp24/hyang2/fyp/code/3d_diffuser_actor/data/peract/raw"
CODEBOOK_SIZE = 512

vqvae = policy_model = None
env: RLBenchEnv = None

reward = 0.0

def obs_to_rgb_pcd(obs: Observation):
    obs_dict = env.get_obs_action(obs)[0]

    # TODO: check the format here...
    rgb = obs_dict["rgb"]  # list of (H, W, 3), uint8
    rgb = [transforms.ToTensor()(x).unsqueeze(0) for x in rgb]
    rgb = torch.cat(rgb, dim=0).unsqueeze(0).unsqueeze(0)  # (1, 1, ncam, 3, H, W)

    pcd = obs_dict["pc"]
    pcd = [torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0) for x in pcd]
    pcd = [x.to(torch.float32) for x in pcd]
    pcd = torch.cat(pcd, dim=0).unsqueeze(0).unsqueeze(0)  # (1, 1, ncam, 3, H, W)

    return rgb.cuda(), pcd.cuda()


def execute_function(move: Mover, code, start_idx, end_idx):
    global reward

    # code: (1, 4)
    assert len(code.shape) == 2 and code.shape[0] == 1, f"Invalid code shape: {code.shape}"

    if torch.any(code >= CODEBOOK_SIZE):
        print(f"Invalid codebook index in trajectory code: {code}")
        return None

    with torch.no_grad():
        traj_recon = vqvae.decode(code[None, ...])  # (1, 16, 8)
        traj_recon = traj_recon.squeeze(0).detach().cpu().numpy()  # (16, 8)

    traj_recon = traj_recon[start_idx:end_idx]

    traj_recon[:, -1] = (traj_recon[:, -1] > 0.5).astype(np.float32)
    traj_recon[:, 3:7] /= np.linalg.norm(traj_recon[:, 3:7], axis=1, keepdims=True)

    try:
        obs = None
        for action in traj_recon:
            obs, reward, _, _ = move(action, collision_checking=False)
    except Exception as e:
        print(f"Error executing trajectory: {e}")
        return None

    if reward > 0.0:
        print("Execution Success!")
        rgb, pcd = obs_to_rgb_pcd(obs)
        return rgb, pcd, reward

    return obs_to_rgb_pcd(obs)


def test_policy(task, task_str, var, eps, instruction: torch.Tensor):
    global policy_model, reward
    assert instruction.shape == (1, 53, 512), f"Invalid instruction shape: {instruction.shape}"
    try:
        gt_demo = env.get_demo(task_str, var, episode_index=eps, image_paths=True)[0]
    except Exception as e:
        print(f"Error loading demo: {e}")
        return

    # reset scene
    obs: Observation = task.reset_to_demo(gt_demo)[1]
    move = Mover(task, max_tries=1)
    rgb, pcd = obs_to_rgb_pcd(obs)

    exe_fn = lambda code, start_idx, end_idx: execute_function(move, code, start_idx, end_idx)

    # reset the reward
    reward = 0.0

    obs_list = policy_model.generate(
        instruction=instruction, rgb=rgb, pcd=pcd, execute_function=exe_fn
    )  # ndarray of (T, ncam, 3, H, W)

    return obs_list, reward


def main(config):
    global vqvae, policy_model, env, CODEBOOK_SIZE

    # Load the VQVAE and policy model
    config_vq = OmegaConf.load(config.cfg_vq)
    vqvae = load_model_from_config(config_vq, config.pth_vq, verbose=True)
    vqvae.eval().cuda()
    CODEBOOK_SIZE = config_vq["model"]["params"]["quantizer_config"]["bins"]

    config_policy = OmegaConf.load(config.cfg_policy)
    policy_model = load_model_from_config(config_policy, config.pth_policy, verbose=True)
    policy_model.eval().cuda()

    # Load the RLBench environment
    env = RLBenchEnv(
        data_path=pjoin(data_path, config.split),
        image_size=[256, 256],
        apply_rgb=True,
        apply_pc=True,
        headless=True,
        apply_cameras=["left_shoulder", "right_shoulder", "wrist", "front"],
    )

    # Load dataset
    dataset_config = config_policy["data"]["params"]["train"]
    dataset_config["params"]["split"] = config.split
    dataset = instantiate_from_config(dataset_config)

    save_dir = os.path.join(config.save_dir, f"{os.path.basename(config.cfg_policy).replace('.yaml', '')}_{os.path.basename(config.pth_policy).replace('.ckpt', '')}" )
    os.makedirs(save_dir, exist_ok=True)

    # Iterate through the dataset

    results = {}
    for idx, (task_str, var, eps) in enumerate(dataset.all_demos_ids):
        if config.max_eps_per_task > 0 and task_str in results and len(results[task_str]) >= config.max_eps_per_task:
            continue

        instruction = dataset.instructions[task_str][var][0].unsqueeze(0)
        instruction = torch.tensor(instruction).cuda()  # (1, 53, 512)

        # setup the simulation environment
        task = env.env.get_task(task_file_to_task_class(task_str))
        task.set_variation(var)
        
        # set up camera
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        cam_resolution = [480, 480]
        cam = VisionSensor.create(cam_resolution)
        cam.set_pose(cam_placeholder.get_pose())
        cam.set_parent(cam_placeholder)
        
        cam_motion = StaticCameraMotion(cam)
        cams_motion = {"global": cam_motion}
        tr = TaskRecorder(cams_motion, fps=40)
        
        task._scene.register_step_callback(tr.take_snap)

        obs_list, success = test_policy(task, task_str, var, eps, instruction)

        if task_str not in results:
            results[task_str] = []
        results[task_str].append(success)
        print(f"Task: {task_str}, Var: {var}, eps: {eps}, Success: {success}")

        if (idx + 1) % config.save_freq == 0:
            os.makedirs(pjoin(save_dir, f"{idx}_{task_str}_{var}_{eps}_{success}"), exist_ok=True)
            tr.save(pjoin(save_dir, f"{idx}_{task_str}_{var}_{eps}_{success}"))
            
            tr._snaps = {cam_name: [] for cam_name in tr._cams_motion.keys()}

    # print the results
    for task_str, successes in results.items():
        success_rate = sum(successes) / len(successes)
        print(f"Task: {task_str}, Success Rate: {success_rate:.4f}")

    # save the results
    result_file = os.path.join(save_dir, "results.txt")
    with open(result_file, "w") as f:
        for task_str, successes in results.items():
            success_rate = sum(successes) / len(successes)
            f.write(f"Task: {task_str}, Success Rate: {success_rate:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_vq", type=str)
    parser.add_argument("--pth_vq", type=str)
    parser.add_argument("--cfg_policy", type=str)
    parser.add_argument("--pth_policy", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--save_dir", type=str, default="test_logs")
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--max_eps_per_task", type=int, default=-1)

    conf = parser.parse_args()
    main(conf)

