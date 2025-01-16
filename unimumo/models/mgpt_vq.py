# Partially from https://github.com/Mael-zys/T2M-GPT

from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from .tools.resnet import Resnet1D
from .tools.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from collections import OrderedDict
import pytorch_lightning as pl
import typing as tp
import numpy as np
import os
from os.path import join as pjoin

from unimumo.util import instantiate_from_config, get_obj_from_str


class VQVae(pl.LightningModule):

    def __init__(
        self,
        encoder_decoder_config: dict,
        quantizer_config: dict,
        loss_config: dict,
        optimizer_config: dict,
    ):

        super().__init__()

        self.encoder = Encoder(**encoder_decoder_config)
        self.decoder = Decoder(**encoder_decoder_config)

        if quantizer_config["quantizer"] == "ema_reset":
            self.quantizer = QuantizeEMAReset(quantizer_config["code_num"], quantizer_config["code_dim"], mu=0.99)
        elif quantizer_config["quantizer"] == "orig":
            self.quantizer = Quantizer(quantizer_config["code_num"], quantizer_config["code_dim"], beta=1.0)
        elif quantizer_config["quantizer"] == "ema":
            self.quantizer = QuantizeEMA(quantizer_config["code_num"], quantizer_config["code_dim"], mu=0.99)
        elif quantizer_config["quantizer"] == "reset":
            self.quantizer = QuantizeReset(quantizer_config["code_num"], quantizer_config["code_dim"])

        self.loss = instantiate_from_config(loss_config)

        self.optimizer_config = optimizer_config

    def preprocess(self, x: Tensor) -> Tensor:
        # (B, T, 8) -> (B, 8, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x: Tensor) -> Tensor:
        # (B, 8, T) ->  (B, T, 8)
        x = x.permute(0, 2, 1)
        return x

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int):
        trajectory = batch["trajectory"]  # (B, T, 8)
        description = batch["description"]  # (B,)

        traj_recon, loss_commit, perplexity = self.forward(trajectory, description)

        loss, loss_dict = self.loss(trajectory, traj_recon, loss_commit, split="train")

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        # also log the learning rate
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int):
        trajectory = batch["trajectory"]
        description = batch["description"]

        traj_recon, loss_commit, perplexity = self.forward(trajectory, description)

        loss, loss_dict = self.loss(trajectory, traj_recon, loss_commit, split="val")

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        # optimizer
        optim_target = "torch.optim" + self.optimizer_config["optimizer"]["target"]
        optimizer = get_obj_from_str(optim_target)(
            self.parameters(), **self.optimizer_config["optimizer"]["params"]
        )

        # scheduler
        scheduler_target = "torch.optim.lr_scheduler" + self.optimizer_config["lr_scheduler"]["target"]
        scheduler = get_obj_from_str(scheduler_target)(
            optimizer, **self.optimizer_config["lr_scheduler"]["params"]
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def forward(self, trajectory: Tensor, description: tp.List[str]):
        # Preprocess
        x_in = self.preprocess(trajectory)  # (B, 8, T)

        # Encode
        x_encoder = self.encoder(x_in)

        # quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)

        # decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)

        return x_out, loss, perplexity

    def encode(
        self,
        features: Tensor,
    ) -> Tensor:

        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)  # (N, C, T)
        x_encoder = self.postprocess(x_encoder)  # (N, T, C)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)  # (NT,)
        code_idx = code_idx.view(N, -1)  # (N, T)

        # latent code
        return code_idx

    def decode(self, z: Tensor):
        # z: (N, T)

        x_d = self.quantizer.dequantize(z)  # (N, T, C)
        x_d = self.preprocess(x_d).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
