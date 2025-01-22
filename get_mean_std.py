from unimumo.data.motion_vqvae_dataset import MotionVQVAEDataset
import numpy as np
import quaternion
import torch

sum_traj = np.zeros(7)
sum_traj2 = np.zeros(7)
sum_length = 0

data_dir = "/research/d2/fyp24/hyang2/fyp/code/3d_diffuser_actor/data/peract/raw"

for split in ("val", "test", "train"):
    dataset = MotionVQVAEDataset(split, data_dir, preload_data=True)

    for i in range(len(dataset.data)):
        traj = dataset.data[i][0].numpy()

        new_traj = []
        for action in traj:
            trans = action[:3]
            rot_quat = action[3:7]
            gripper = action[7:]
            rot_angle = quaternion.as_euler_angles(
                quaternion.quaternion(rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]))
            # rot_angle = torch.tensor(rot_angle)
            # quat_recon = quaternion.from_euler_angles(*rot_angle)
            # quat_recon = quaternion.as_float_array(quat_recon)
            new_action = np.concatenate([trans, rot_angle, gripper], axis=0)
            new_traj.append(new_action[None, ...])

        new_traj = np.concatenate(new_traj, axis=0)
        print(new_traj.shape)
        traj = new_traj

        sum_traj += np.sum(traj, axis=0)
        sum_traj2 += np.sum(traj ** 2, axis=0)
        sum_length += traj.shape[0]

mean = sum_traj / sum_length
std = np.sqrt((sum_traj2 / sum_length) - (mean ** 2))

# np.save("statistics.npy", results)
np.save('mean.npy', mean)
np.save('std.npy', std)

