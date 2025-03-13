import einops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork
from torchvision import transforms

from unimumo.modules.resnet import ResnetEncoder
from unimumo.modules.pointnet import PointNetfeat



class Encoder(nn.Module):

    def __init__(self,
                 image_size=(3, 256, 256),
                 embedding_dim=60,
                 n_cameras=4,
                 rgb_encoder: str = "resnet18",):
        super().__init__()

        self.pcd_encoder = nn.Sequential(
            PointNetfeat(),
            nn.Mish(),
            nn.Linear(1024, 256),
            nn.Mish(),
            nn.Linear(256, embedding_dim),
        )

        resnet = ResnetEncoder(rgb=True, freeze=False, pretrained=True, model=rgb_encoder, input_shape=image_size)
        self.image_encoder = nn.Sequential(
            resnet,
            nn.Mish(),
            nn.Linear(resnet.n_channel, embedding_dim),
        )
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Instruction encoder
        self.instruction_encoder = nn.Linear(512, embedding_dim)

    def encode_rgb_pcd(self, rgb, pcd):
        # rgb: (B, T, ncam, 3, H, W)  pcd: (B, T, ncam, 3, H, W)
        B, T, ncam = rgb.shape[:3]

        rgb = einops.rearrange(rgb, "b t ncam c h w -> (b t ncam) c h w")
        rgb = self.normalize(rgb)
        rgb_feature = self.image_encoder(rgb)
        rgb_feature = einops.rearrange(rgb_feature, "(b t ncam) d -> b t (ncam d)", b=B, t=T, ncam=ncam)

        pcd = einops.rearrange(pcd, "b t ncam c h w -> (b t ncam) c h w")
        pcd = F.interpolate(pcd, (64, 64), mode='bilinear')
        pcd_feature = self.pcd_encoder(pcd)
        pcd_feature = einops.rearrange(pcd_feature, "(b t ncam) d -> b t (ncam d)", b=B, t=T, ncam=ncam)

        return rgb_feature, pcd_feature

    def encode_instruction(self, instruction):
        """
        Compute language features/pos embeddings on top of CLIP features.

        Args:
            - instruction: (B, max_instruction_length, 512)

        Returns:
            - instr_feats: (B, 53, F)
            - instr_dummy_pos: (B, 53, F, 2)
        """
        instr_feats = self.instruction_encoder(instruction)
        return instr_feats