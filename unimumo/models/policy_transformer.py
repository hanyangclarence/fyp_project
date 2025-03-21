from typing import Any

import torch
import numpy as np
import einops
import torch.nn.functional as F
import typing as tp
from tqdm import tqdm

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
            input_traj_length: int,
            max_traj_length: int,
            start_idx: int,
            end_idx: int,
            pad_idx: int,
            encoder_config: dict,
            transformer_config: dict,
            optimizer_config: dict,
            monitor: str = None
    ):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.pad_idx = pad_idx

        self.feature_encoder = Encoder(**encoder_config)

        transformer_config["dim_lang_feature"] = encoder_config["embedding_dim"]
        transformer_config["dim_rgb_feature"] = encoder_config["embedding_dim"] * 4
        transformer_config["dim_pcd_feature"] = encoder_config["embedding_dim"] * 4
        self.transformer_model = TransformerWrapper(
            num_tokens=num_tokens,
            max_seq_len=max_traj_length * 4,
            attn_layers=Decoder(**transformer_config)
        )
        self.input_traj_length = input_traj_length
        self.max_traj_length = max_traj_length

        self.optimizer_config = optimizer_config

        self.monitor = monitor


    def training_step(self, batch, batch_idx):
        traj_code = batch["trajectory"]  # (B, T * 4)
        instruction = batch["instruction"]  # (B, 53, 512)
        rgb = batch["rgb"]  # (B, T-1, ncam, 3, H, W)
        pcd = batch["pcd"]  # (B, T-1, ncam, 3, H, W)
        input_mask = batch["input_mask"]  # (B, (T-1) * 4)
        context_mask = batch["context_mask"]  # (B, T-1)

        loss = self.forward(traj_code, instruction, rgb, pcd, input_mask, context_mask)

        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        traj_code = batch["trajectory"]
        instruction = batch["instruction"]
        rgb = batch["rgb"]
        pcd = batch["pcd"]
        input_mask = batch["input_mask"]
        context_mask = batch["context_mask"]

        loss = self.forward(traj_code, instruction, rgb, pcd, input_mask, context_mask)

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

    def forward(self, traj_code: torch.Tensor, instruction: torch.Tensor, rgb: torch.Tensor, pcd: torch.Tensor,
                input_mask: torch.Tensor, visual_context_mask: torch.Tensor):
        # traj_code: (B, T * 4)  instruction: (B, 53, 512)  rgb: (B, T-1, ncam, 3, H, W)  pcd: (B, T-1, ncam, 3, H, W)
        # input_mask: (B, (T-1) * 4)  context_mask: (B, T-1)

        rgb_feature, pcd_feature, instruction_feature = self.encode_inputs(rgb, pcd, instruction)
        # rgb_feature: (B, T-1, ncam * D)  pcd_feature: (B, T-1, ncam * D)  instruction_feature: (B, 53, D)
        visual_cross_attn_mask = self.get_visual_cross_attn_mask(rgb_feature.shape[1])  # ((T-1)*4, T-1)
        self_attn_mask = self.get_self_attn_mask(rgb_feature.shape[1])  # ((T-1)*4, (T-1)*4)

        inp, target = traj_code[:, :-4], traj_code[:, 4:]  # (B, (T-1) * 4)

        logits = self.transformer_model.forward(
            inp, rgb_feature=rgb_feature, pcd_feature=pcd_feature, instruction_feature=instruction_feature,
            visual_cross_attn_mask=visual_cross_attn_mask, attn_mask=self_attn_mask, mask=input_mask,
            visual_context_mask=visual_context_mask
        )

        loss_fn = F.cross_entropy if not self.transformer_model.output_is_log_prob else F.nll_loss

        loss = loss_fn(
            einops.rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index=self.pad_idx
        )

        return loss

    @torch.no_grad()
    def generate(
        self,
        instruction: torch.Tensor, rgb: torch.Tensor, pcd: torch.Tensor,
        execute_function: tp.Callable,
        temperature=1.,
    ):
        # instruction: (1, 53, 512)  rgb: (1, 1, ncam, 3, H, W)  pcd: (1, 1, ncam, 3, H, W)
        assert instruction.shape[0] == 1, f"Batch size must be 1, got {instruction.shape[0]}"
        assert instruction.shape[0] == rgb.shape[0] == pcd.shape[0], f"Batch size mismatch: {instruction.shape[0]} {rgb.shape[0]} {pcd.shape[0]}"

        out = torch.ones((1, 4), dtype=torch.long, device=self.device) * self.start_idx
        for _ in tqdm(range(self.max_traj_length), desc="Generating"):
            x = out[:, -4 * (self.input_traj_length - 1):]  # (1, 4 * T)
            rgb_context = rgb[:, -(self.input_traj_length - 1):]  # (1, T, ncam, 3, H, W)
            pcd_context = pcd[:, -(self.input_traj_length - 1):]  # (1, T, ncam, 3, H, W)

            rgb_feature, pcd_feature, instruction_feature = self.encode_inputs(rgb_context, pcd_context, instruction)
            visual_cross_attn_mask = self.get_visual_cross_attn_mask(rgb_feature.shape[1])
            self_attn_mask = self.get_self_attn_mask(rgb_feature.shape[1])

            logits = self.transformer_model.forward(
                x, rgb_feature=rgb_feature, pcd_feature=pcd_feature, instruction_feature=instruction_feature,
                visual_cross_attn_mask=visual_cross_attn_mask, attn_mask=self_attn_mask,
                visual_context_mask=torch.ones((1, rgb_context.shape[1]), dtype=torch.bool, device=self.device)
            )  # (1, 4 * T, C)

            logits = logits[:, -4:]  # (1, 4, C)
            probs = F.softmax(logits / temperature, dim=-1)  # (1, 4, C)
            sample = [torch.multinomial(probs[0, i], 1).item() for i in range(4)]
            sample = torch.tensor(sample, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, 4)

            if torch.all(sample == self.end_idx):
                ret_value = execute_function(out[:, -8:-4], False)
                if ret_value is None:
                    pass
                else:
                    new_rgb, _ = ret_value
                    rgb = torch.cat([rgb, new_rgb], dim=1)
                print("End token reached!")
                break

            # run the predicted trajectory to get the next observation
            ret_value = execute_function(sample, True)  # (1, 1, ncam, 3, H, W)  (1, 1, ncam, 3, H, W)
            if ret_value is None:
                # skip the current step
                continue

            # update the trajectory
            new_rgb, new_pcd = ret_value
            rgb = torch.cat([rgb, new_rgb], dim=1)  # (1, T+1, ncam, 3, H, W)
            pcd = torch.cat([pcd, new_pcd], dim=1)
            out = torch.cat([out, sample], dim=1)  # (1, 4 * (T+1))

        rgb = einops.rearrange(rgb[0], "t ncam c h w -> t ncam h w c").cpu().numpy()
        return rgb

    @torch.no_grad()
    def rollout_gt(self, traj_code: torch.Tensor, rgb: torch.Tensor, pcd: torch.Tensor,
                   execute_function: tp.Callable):
        assert traj_code.shape[0] == 1, f"Batch size must be 1, got {traj_code.shape[0]}"
        assert traj_code.shape[0] == rgb.shape[0] == pcd.shape[0], f"Batch size mismatch: {traj_code.shape[0]} {rgb.shape[0]} {pcd.shape[0]}"

        out = torch.ones((1, 4), dtype=torch.long, device=self.device) * self.start_idx
        for i in tqdm(range(traj_code.shape[1] // 4), desc="Rollout Ground Truth"):
            sample = traj_code[:, 4 * i: 4 * (i + 1)]

            if torch.all(sample == self.end_idx):
                ret_value = execute_function(out[:, -8:-4], False)
                if ret_value is None:
                    pass
                else:
                    new_rgb, _ = ret_value
                    rgb = torch.cat([rgb, new_rgb], dim=1)
                print("End token reached!")
                break

            ret_value = execute_function(sample, True)
            if ret_value is None:
                continue

            new_rgb, new_pcd = ret_value
            rgb = torch.cat([rgb, new_rgb], dim=1)
            pcd = torch.cat([pcd, new_pcd], dim=1)
            out = torch.cat([out, sample], dim=1)

        rgb = einops.rearrange(rgb[0], "t ncam c h w -> t ncam h w c").cpu().numpy()
        return rgb




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



















