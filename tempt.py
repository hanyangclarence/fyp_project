from unimumo.models.mgpt_vq import VQVae
import torch

model_config = {
    "input_emb_width": 8,
    "output_emb_width": 128,
    "down_t": 3,
    "stride_t": 2,
}
quantizer_config = {
    "quantizer": "ema_reset",
    "code_num": 512,
    "code_dim": 128,
}
loss_config = {
    "target": "unimumo.modules.loss.MotionVqVaeLoss",
    "params": {
        "lambda_recon": 1.0,
        "lambda_commit": 0.02,
    }
}
optimizer_config = None

model = VQVae(model_config, quantizer_config, loss_config, optimizer_config)
model = model.cuda()

trajectory = torch.rand(2, 64, 8).cuda()
description = ["test"] * 2

# traj_recon, loss_commit, perplexity = model(trajectory, description)
# loss, loss_dict = model.loss(trajectory, traj_recon, loss_commit, split="train")

code = model.encode(trajectory)
recon = model.decode(code)

print('here')