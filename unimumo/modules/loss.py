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