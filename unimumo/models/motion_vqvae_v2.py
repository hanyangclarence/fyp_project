from os.path import join as pjoin
import os

import numpy as np
import torch
from torch import Tensor
import pytorch_lightning as pl
from einops import rearrange
import typing as tp

from unimumo.util import instantiate_from_config, get_obj_from_str
from unimumo.audio.audiocraft_.quantization.vq import ResidualVectorQuantizer


class MotionVQVAE(pl.LightningModule):
    def __init__(
        self,
        encoder_config: dict,
        decoder_config: dict,
        quantizer_config: dict,
        loss_config: dict,
        optimizer_config: dict,
        mean_dir: str,
        std_dir: str,
        monitor: tp.Optional[str] = None,
        normalize_motion: bool = True,
        motion_mode: str = None,
    ):
        super().__init__()

        self.motion_encoder = instantiate_from_config(encoder_config)
        self.motion_decoder = instantiate_from_config(decoder_config)

        self.quantizer = ResidualVectorQuantizer(**quantizer_config)

        self.loss = instantiate_from_config(loss_config)

        self.optimizer_config = optimizer_config

        # load mean and std
        self.input_dim = encoder_config['params']['input_dim']
        if normalize_motion:
            self.mean = torch.from_numpy(np.load(mean_dir)).float()
            self.std = torch.from_numpy(np.load(std_dir)).float()
            assert self.mean.shape == self.std.shape == (self.input_dim,), f"Expected shape {(self.input_dim,)}, got {self.mean.shape} or {self.std.shape}"

            self.mean[3:8] = 0
            self.std[3:8] = 1
        else:
            self.mean = torch.zeros(self.input_dim)
            self.std = torch.ones(self.input_dim)

        # store the last batch for visualization
        self.last_train_batch = None
        self.last_val_batch = None

        self.monitor = monitor
        self.motion_mode = motion_mode

        self.codebook_usage = np.zeros(self.quantizer.bins, dtype=np.int64)

    def normalize(self, x: Tensor) -> Tensor:
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)

        dim = x.shape[-1]
        return (x - self.mean[:dim]) / self.std[:dim]

    def denormalize(self, x: Tensor) -> Tensor:
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)

        dim = x.shape[-1]
        return x * self.std[:dim] + self.mean[:dim]

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int):
        self.last_train_batch = batch

        trajectory = batch["trajectory"]  # (B, T, 8)
        description = batch["description"]  # (B,)

        trajectory = self.normalize(trajectory)

        t = trajectory.shape[1]
        gripper_start = trajectory[:, 0:1, 7:8]  # (B, 1, 1)
        gripper_end = trajectory[:, -1:, 7:8]  # (B, 1, 1)
        trajectory[:, :t//2, 7:8] = gripper_start
        trajectory[:, t//2:, 7:8] = gripper_end

        traj_recon, commitment_loss = self.forward(trajectory, description)
        loss, loss_dict = self.loss(trajectory[:, :, :8], traj_recon, commitment_loss, split="train")  # only reconstruct the motion part

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        # also log the learning rate
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, logger=True, on_step=True, on_epoch=False)

        # calculate the mean value of the model parameter and log the value
        enc_param_mean = torch.mean(torch.stack([torch.mean(param) for param in self.motion_encoder.parameters()]))
        dec_param_mean = torch.mean(torch.stack([torch.mean(param) for param in self.motion_decoder.parameters()]))
        quant_param_mean = torch.mean(torch.stack([torch.mean(param) for param in self.quantizer.state_dict().values()]))
        self.log("enc_param_mean", enc_param_mean, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("dec_param_mean", dec_param_mean, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("quant_param_mean", quant_param_mean, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int):
        self.last_val_batch = batch

        trajectory = batch["trajectory"]
        description = batch["description"]

        code = self.encode(trajectory)
        traj_recon = self.decode(code)

        loss, loss_dict = self.loss(self.normalize(trajectory[:, :, :8]), self.normalize(traj_recon), 0, split="val")

        if batch_idx == 0:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!")
            print(trajectory[0, :10, :3])
            print(traj_recon[0, :10, :3])

        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        # log perplexity
        with torch.no_grad():
            code_flat = code.view(-1)
            unique_codes, counts = torch.unique(code_flat, return_counts=True)
            usage = torch.zeros(self.quantizer.bins, dtype=torch.long, device=code.device)
            usage[unique_codes] = counts
            usage = usage.cpu().numpy()

            # empirical probabilities
            p = usage / usage.sum()  # shape (codebook_size,)

            # perplexity
            eps = 1e-10
            perplexity = np.exp(-np.sum(p * np.log(p + eps)))

            # Number of unique codes used
            unique_codes_used = float((usage > 0).sum())

            self.log("code_usage/unique_codes_used", unique_codes_used, on_step=True, on_epoch=False)
            self.log("code_usage/perplexity", perplexity, on_step=True, on_epoch=False)

            # add the usage to the codebook_usage statistics
            self.codebook_usage += usage

        return loss

    def on_validation_epoch_end(self):
        # log the codebook usage statistics as histogram
        if self.logger is not None:
            writer = self.logger.experiment
            writer.add_histogram("code_usage/usage_hist", self.codebook_usage, global_step=self.current_epoch)
            print(f"Codebook usage histogram logged at epoch {self.current_epoch}")

            # reset the codebook usage statistics
            self.codebook_usage = np.zeros(self.quantizer.bins, dtype=np.int64)

    def configure_optimizers(self):
        # optimizer
        optim_target = "torch.optim." + self.optimizer_config["optimizer"]["target"]
        optimizer = get_obj_from_str(optim_target)(
            self.parameters(), **self.optimizer_config["optimizer"]["params"]
        )

        # scheduler
        scheduler_target = "torch.optim.lr_scheduler." + self.optimizer_config["lr_scheduler"]["target"]
        scheduler = get_obj_from_str(scheduler_target)(
            optimizer, **self.optimizer_config["lr_scheduler"]["params"]
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def embed_motion(self, trajectory: torch.Tensor):
        # trajectory: (B, T, 8)
        trajectory = rearrange(trajectory, 'b t d -> b d t')
        motion_emb = self.motion_encoder(trajectory)  # [B, 128, T']

        return motion_emb

    def decode_motion_embed(self, motion_emb: torch.Tensor) -> torch.Tensor:
        # motion_emb: [B, 128, T']
        motion_recon = self.motion_decoder(motion_emb)
        motion_recon = rearrange(motion_recon, 'b d t -> b t d')  # [B, T, 8]

        return motion_recon

    def forward(self, trajectory: Tensor, description: tp.List[str]):
        motion_emb = self.embed_motion(trajectory)

        q_res_motion = self.quantizer(motion_emb, 50)

        motion_recon = self.decode_motion_embed(q_res_motion.x)

        return motion_recon, q_res_motion.penalty  # penalty is the commitment loss

    def encode(self, trajectory: Tensor):
        N, T, C = trajectory.shape
        assert C == self.input_dim, f"Expected {self.input_dim} channels, got {C}"

        trajectory = trajectory.clone()
        t = trajectory.shape[1]
        gripper_start = trajectory[:, 0:1, 7:8]  # (B, 1, 1)
        gripper_end = trajectory[:, -1:, 7:8]  # (B, 1, 1)
        trajectory[:, :t//2, 7:8] = gripper_start
        trajectory[:, t//2:, 7:8] = gripper_end

        trajectory = self.normalize(trajectory)

        motion_emb = self.embed_motion(trajectory)
        motion_code = self.quantizer.encode(motion_emb).contiguous()

        return motion_code  # [B, N_q, T']

    def decode(self, motion_code: Tensor, description: tp.List[str] = None):
        assert len(motion_code.shape) == 3, f"Expected 2D tensor, got {len(motion_code.shape)}"

        motion_emb = self.quantizer.decode(motion_code)
        motion_recon = self.decode_motion_embed(motion_emb)

        motion_recon = self.denormalize(motion_recon)

        gripper_start = motion_recon[:, 0:1, 7:8]
        motion_recon[:, :-1, 7:8] = gripper_start

        return motion_recon


