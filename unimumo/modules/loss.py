import torch
import torch.nn as nn


class MotionVqVaeLoss(nn.Module):
        def __init__(
            self,
            lambda_recon: float = 1.0,
            lambda_commit: float = 0.02,
        ):
            super().__init__()

            self.recon_loss = nn.SmoothL1Loss(reduction="mean")

            self.lambda_recon = lambda_recon
            self.lambda_commit = lambda_commit

        def forward(self, traj_ref, traj_recon, loss_commit, split):
            loss_recon = self.recon_loss(traj_recon, traj_ref)

            loss = self.lambda_recon * loss_recon + self.lambda_commit * loss_commit
            loss_dict = {
                f"{split}/loss_recon": loss_recon,
                f"{split}/loss_commit": loss_commit,
                f"{split}/loss": loss,
            }

            return loss, loss_dict


class LossWithCrossEntropy(nn.Module):
    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_commit: float = 0.02,
        gripper_weight: float = 0.1,
    ):
        super().__init__()

        self.recon_loss = nn.SmoothL1Loss(reduction="mean")
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.lambda_recon = lambda_recon
        self.lambda_commit = lambda_commit
        self.gripper_weight = gripper_weight

    def forward(self, traj_ref, traj_recon, loss_commit, split):
        pose_recon_loss = self.recon_loss(traj_recon[:, :, :7], traj_ref[:, :, :7])
        gripper_recon_loss = self.cross_entropy_loss(traj_recon[:, :, 7:8], traj_ref[:, :, 7:8])
        loss_recon = pose_recon_loss * (1 - self.gripper_weight) + gripper_recon_loss * self.gripper_weight

        loss = self.lambda_recon * loss_recon + self.lambda_commit * loss_commit
        loss_dict = {
            f"{split}/loss_recon": loss_recon,
            f"{split}/loss_commit": loss_commit,
            f"{split}/loss": loss,
            f"{split}/loss_pose": pose_recon_loss,
            f"{split}/loss_gripper": gripper_recon_loss,
        }

        return loss, loss_dict



















