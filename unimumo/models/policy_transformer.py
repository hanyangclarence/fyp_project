from typing import Any

import torch
import numpy as np
import einops
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from unimumo.util import instantiate_from_config, get_obj_from_str
from unimumo.x_transformers import TransformerWrapper, Decoder
from unimumo.models.feature_encoder import Encoder


# transformer_config = {
#     "dim": 512,
#     "depth": 12,
#     "heads": 8,
#     "cross_attend": True,
#     "dim_lang_feature": 192,
#     "dim_rgb_feature": 192 * 4,
#     "dim_pcd_feature": 192 * 4,
# }


class PolicyTransformer(pl.LightningModule):
    def __init__(
            self,
            num_tokens: int,
            max_traj_length: int,
            start_idx: int,
            end_idx: int,
            encoder_config: dict,
            transformer_config: dict,
            optimizer_config: dict,
            monitor: str = None
    ):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.feature_encoder = Encoder(**encoder_config)

        transformer_config["dim_lang_feature"] = encoder_config["embedding_dim"]
        transformer_config["dim_rgb_feature"] = encoder_config["embedding_dim"] * 4
        transformer_config["dim_pcd_feature"] = encoder_config["embedding_dim"] * 4
        self.transformer_model = TransformerWrapper(
            num_tokens=num_tokens,
            max_seq_len=max_traj_length * 4,
            attn_layers=Decoder(**transformer_config)
        )

        self.optimizer_config = optimizer_config

        self.monitor = monitor


    def training_step(self, batch, batch_idx):
        traj_code = batch["trajectory"]  # (B, T * 4)
        instruction = batch["instruction"]  # (B, 53, 512)
        rgb = batch["rgb"]  # (B, T-1, ncam, 3, H, W)
        pcd = batch["pcd"]  # (B, T-1, ncam, 3, H, W)

        loss = self.forward(traj_code, instruction, rgb, pcd)

        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        traj_code = batch["trajectory"]
        instruction = batch["instruction"]
        rgb = batch["rgb"]
        pcd = batch["pcd"]

        loss = self.forward(traj_code, instruction, rgb, pcd)

        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss

    def encode_inputs(self, rgb: torch.Tensor, pcd: torch.Tensor, instruction: torch.Tensor):
        # rgb: (B, T-1, ncam, 3, H, W)  pcd: (B, T-1, ncam, 3, H, W)  instruction: (B, 53, 512)
        rgb_feature, pcd_feature = self.feature_encoder.encode_rgb_pcd(rgb, pcd)  # (B, T-1, ncam * D)  (B, T-1, ncam * D)

        instruction_feature = self.feature_encoder.encode_instruction(instruction)  # (B, 53, D)

        return rgb_feature, pcd_feature, instruction_feature

    def get_visual_cross_attn_mask(self, length: int):
        mask = torch.zeros(length * 4, length, dtype=torch.bool)  # (T * 4, T)
        for i in range(length):
            mask[i * 4: (i + 1) * 4, :i + 1] = True
        return mask.to(self.device)

    def get_self_attn_mask(self, length: int):
        mask = torch.zeros(length * 4, length * 4, dtype=torch.bool)  # (T * 4, T * 4)
        for i in range(length):
            mask[i * 4: (i + 1) * 4, :(i + 1) * 4] = True
        return mask.to(self.device)

    def forward(self, traj_code: torch.Tensor, instruction: torch.Tensor, rgb: torch.Tensor, pcd: torch.Tensor):
        # traj_code: (B, T * 4)  instruction: (B, 53, 512)  rgb: (B, T-1, ncam, 3, H, W)  pcd: (B, T-1, ncam, 3, H, W)

        rgb_feature, pcd_feature, instruction_feature = self.encode_inputs(rgb, pcd, instruction)
        # rgb_feature: (B, T-1, ncam * D)  pcd_feature: (B, T-1, ncam * D)  instruction_feature: (B, 53, D)
        visual_cross_attn_mask = self.get_visual_cross_attn_mask(rgb_feature.shape[1])  # ((T-1)*4, T-1)
        self_attn_mask = self.get_self_attn_mask(rgb_feature.shape[1])  # ((T-1)*4, (T-1)*4)

        inp, target = traj_code[:, :-4], traj_code[:, 4:]  # (B, (T-1) * 4)

        logits = self.transformer_model.forward(
            inp, rgb_feature=rgb_feature, pcd_feature=pcd_feature, instruction_feature=instruction_feature,
            visual_cross_attn_mask=visual_cross_attn_mask, attn_mask=self_attn_mask
        )

        loss_fn = F.cross_entropy if not self.transformer_model.output_is_log_prob else F.nll_loss

        loss = loss_fn(
            einops.rearrange(logits, 'b n c -> b c n'),
            target
        )

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



















