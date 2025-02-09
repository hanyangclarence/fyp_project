from os.path import join as pjoin
import os

import numpy as np
import torch
from torch import Tensor
import pytorch_lightning as pl
from einops import rearrange
import typing as tp
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict

from unimumo.util import instantiate_from_config, get_obj_from_str
from unimumo.audio.audiocraft_.quantization.vq import ResidualVectorQuantizer


class MotionVQVAE(pl.LightningModule):
    def __init__(
        self,
        encoder_config: dict,
        decoder_config: dict,
        quantizer_config: dict,
        vision_encoder_config: dict,
        fusor_config: dict,
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

        if vision_encoder_config["name"] == 'resnet18':
            encoder = models.resnet18(pretrained=True)
            self.vision_encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]))  # remove the last classification layer
        else:
            raise NotImplementedError(f"Vision encoder {vision_encoder_config['name']} is not implemented")
        self.vision_encoder_out_layer = nn.Linear(
            vision_encoder_config["n_cameras"] * vision_encoder_config["encoder_dim"], vision_encoder_config["encoder_dim"]
        )

        self.encoder_fusor = instantiate_from_config(fusor_config)
        self.decoder_fusor = instantiate_from_config(fusor_config)

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

        self.print_trainable_parameters()

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
        rgb = batch["rgb"]  # (B, T', N, 3, H, W)

        trajectory = self.normalize(trajectory)

        traj_recon, commitment_loss = self.forward(trajectory, description, rgb)
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
        rgb = batch["rgb"]

        code = self.encode(trajectory, rgb)
        traj_recon = self.decode(code, rgb)

        loss, loss_dict = self.loss(self.normalize(trajectory[:, :, :8]), self.normalize(traj_recon), 0, split="val")

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
            try:
                writer = self.logger.experiment
                writer.add_histogram("code_usage/usage_hist", self.codebook_usage, global_step=self.current_epoch)
                print(f"Codebook usage histogram logged at epoch {self.current_epoch}")
            except Exception as e:
                print(f"Failed to log codebook usage histogram: {e}")

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

    def encode_rgb(self, rgb: Tensor) -> Tensor:
        assert len(rgb.shape) == 6, f"Expected 6D tensor, got {rgb.shape}"
        B, T, N, C, H, W = rgb.shape
        rgb = rearrange(rgb, 'b t n c h w -> (b t n) c h w')  # (B*T*N, C, H, W)
        rgb_features = self.vision_encoder(rgb)  # (B*T*N, 512, 1, 1)
        rgb_features = rgb_features.squeeze(-1).squeeze(-1)  # (B*T*N, 512)
        rgb_features = rearrange(rgb_features, '(b t n) c -> b t (n c)', b=B, t=T, n=N)  # (B, T, N*512)
        rgb_features = self.vision_encoder_out_layer(rgb_features)  # (B, T, D)
        rgb_features = rearrange(rgb_features, 'b t c -> b c t')  # (B, D, T)
        return rgb_features

    def embed_motion(self, trajectory: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        # trajectory: (B, T, 8), visual_features: (B, 512, T')
        trajectory = rearrange(trajectory, 'b t d -> b d t')
        motion_emb = self.motion_encoder(trajectory)  # [B, D, T']

        motion_emb = self.fuse_vision_in_encoder(motion_emb, visual_features)  # (B, D, T')

        return motion_emb

    def decode_motion_embed(self, motion_emb: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        # motion_emb: [B, D, T'], visual_features: [B, 512, T']
        motion_emb = self.fuse_vision_in_decoder(motion_emb, visual_features)  # (B, D, T')

        motion_recon = self.motion_decoder(motion_emb)
        motion_recon = rearrange(motion_recon, 'b d t -> b t d')  # [B, T, 8]

        return motion_recon

    def fuse_vision_in_encoder(self, motion_emb: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        # motion_emb: (B, D, T'), visual_features: (B, 512, T')
        assert motion_emb.shape[-1] == visual_features.shape[-1], f"Expected the same length, got {motion_emb.shape[-1]} and {visual_features.shape[-1]}"
        fused_features = self.encoder_fusor(
            torch.cat([motion_emb, visual_features], dim=1)
        )  # (B, D, T')
        return fused_features

    def fuse_vision_in_decoder(self, motion_emb: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        # motion_emb: (B, D, T'), visual_features: (B, 512, T')
        assert motion_emb.shape[-1] == visual_features.shape[-1], f"Expected the same length, got {motion_emb.shape[-1]} and {visual_features.shape[-1]}"
        fused_features = self.decoder_fusor(
            torch.cat([motion_emb, visual_features], dim=1)
        )
        return fused_features

    def forward(self, trajectory: Tensor, description: tp.List[str], rgb: Tensor):
        visual_features = self.encode_rgb(rgb)  # (B, 512, T')

        motion_emb = self.embed_motion(trajectory, visual_features)  # (B, D, T')

        q_res_motion = self.quantizer(motion_emb, 50)

        motion_recon = self.decode_motion_embed(q_res_motion.x, visual_features)

        return motion_recon, q_res_motion.penalty  # penalty is the commitment loss

    def encode(self, trajectory: Tensor, rgb: Tensor):
        N, T, C = trajectory.shape
        assert C == self.input_dim, f"Expected {self.input_dim} channels, got {C}"

        trajectory = self.normalize(trajectory)
        visual_features = self.encode_rgb(rgb)

        motion_emb = self.embed_motion(trajectory, visual_features)
        motion_code = self.quantizer.encode(motion_emb).contiguous()

        motion_code = motion_code[:, 0]  # (B, 1, T') -> (B, T')

        return motion_code

    def decode(self, motion_code: Tensor, rgb: Tensor):
        assert len(motion_code.shape) == 2, f"Expected 2D tensor, got {len(motion_code.shape)}"
        motion_code = motion_code[:, None]  # (B, T') -> (B, 1, T')

        motion_emb = self.quantizer.decode(motion_code)
        visual_features = self.encode_rgb(rgb)
        motion_recon = self.decode_motion_embed(motion_emb, visual_features)

        motion_recon = self.denormalize(motion_recon)

        return motion_recon

    def print_trainable_parameters(self):
        trainable_name_list = []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                trainable_name_list.append(name)
        # remove repetitive names
        filtered_name = []
        for name in trainable_name_list:
            name = name.split('.')
            name = [s for s in name if not s.isdigit()]
            name = '.'.join(name)
            filtered_name.append(name)
        name_set = list(OrderedDict.fromkeys(filtered_name))
        name_count = {}
        for name in name_set:
            name_count[name] = sum([s == name for s in filtered_name])
        print('All trainable parameters:')
        for name, count in name_count.items():
            print(f'\t[{name}] x {count}')

        frozen_name_list = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                frozen_name_list.append(name)
        # remove repetitive names
        filtered_name = []
        for name in frozen_name_list:
            name = name.split('.')
            name = [s for s in name if not s.isdigit()]
            name = '.'.join(name)
            filtered_name.append(name)
        name_set = list(OrderedDict.fromkeys(filtered_name))
        name_count = {}
        for name in name_set:
            name_count[name] = sum([s == name for s in filtered_name])
        print('\nAll frozen parameters:')
        for name, count in name_count.items():
            print(f'\t[{name}] x {count}')
