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
from unimumo.modules.motion_vqvae_module import Encoder, Decoder


class MotionVQVAE(pl.LightningModule):
    def __init__(
        self,
        encoder_decoder_config: dict,
        quantizer_config: dict,
        loss_config: dict,
        optimizer_config: dict,
        mean_std_dir: str,
        monitor: tp.Optional[str] = None,
        normalize_motion: bool = True,
    ):
        super().__init__()

        self.motion_encoder = Encoder(**encoder_decoder_config)
        self.motion_decoder = Decoder(**encoder_decoder_config)

        self.quantizer = ResidualVectorQuantizer(**quantizer_config)

        self.loss = instantiate_from_config(loss_config)

        self.optimizer_config = optimizer_config

        # load mean and std
        if normalize_motion:
            assert os.path.exists(pjoin(mean_std_dir, "mean.npy")), f"mean.npy not found in {mean_std_dir}"
            assert os.path.exists(pjoin(mean_std_dir, "std.npy")), f"std.npy not found in {mean_std_dir}"
            self.mean = torch.from_numpy(np.load(pjoin(mean_std_dir, "mean.npy"))).float()
            self.std = torch.from_numpy(np.load(pjoin(mean_std_dir, "std.npy"))).float()
        else:
            self.mean = torch.zeros(8)
            self.std = torch.ones(8)

        # store the last batch for visualization
        self.last_train_batch = None
        self.last_val_batch = None

        self.monitor = monitor

    def normalize(self, x: Tensor) -> Tensor:
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)

        return (x - self.mean) / self.std

    def denormalize(self, x: Tensor) -> Tensor:
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)

        return x * self.std + self.mean

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int):
        self.last_train_batch = batch

        trajectory = batch["trajectory"]  # (B, T, 8)
        description = batch["description"]  # (B,)

        trajectory = self.normalize(trajectory)

        traj_recon, commitment_loss = self.forward(trajectory, description)
        loss, loss_dict = self.loss(trajectory, traj_recon, commitment_loss, split="train")

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

        trajectory = self.normalize(trajectory)

        traj_recon, commitment_loss = self.forward(trajectory, description)
        loss, loss_dict = self.loss(trajectory, traj_recon, commitment_loss, split="val")

        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

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
        assert C == 8, f"Expected 8 channels, got {C}"

        trajectory = self.normalize(trajectory)

        motion_emb = self.embed_motion(trajectory)
        motion_code = self.quantizer.encode(motion_emb).contiguous()

        motion_code = motion_code[:, 0]  # (B, 1, T') -> (B, T')

        return motion_code

    def decode(self, motion_code: Tensor):
        assert len(motion_code.shape) == 2, f"Expected 2D tensor, got {len(motion_code.shape)}"
        motion_code = motion_code[:, None]  # (B, T') -> (B, 1, T')

        motion_emb = self.quantizer.decode(motion_code)
        motion_recon = self.decode_motion_embed(motion_emb)

        motion_recon = self.denormalize(motion_recon)

        return motion_recon


