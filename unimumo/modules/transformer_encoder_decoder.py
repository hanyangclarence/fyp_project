import torch
import torch.nn as nn
import math
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x


class ConvTransformerEncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, nhead):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.pos_encoder = PositionalEncoding(out_channels)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=out_channels, nhead=nhead, batch_first=True
        )

    def forward(self, x):
        print(f"{x.shape} => ", end="")
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = rearrange(x, 'b t c -> b c t')

        print(x.shape)
        return x


class ConvTransformerDecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, nhead, is_transposed=False):
        super().__init__()
        self.pos_encoder = PositionalEncoding(in_channels)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=in_channels, nhead=nhead, batch_first=True
        )
        if is_transposed:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        print(f"{x.shape} => ", end="")
        x = rearrange(x, 'b c t -> b t c')
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        print(x.shape)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
            input_dim=24,
            output_dim=576,
            emb_dim_encoder=(24, 48, 96, 192, 384, 576),
            nhead=(3, 6, 6, 12, 12, 18),
            downsample=(0, 1, 0, 1, 0),
    ):
        super().__init__()
        assert len(downsample) == len(emb_dim_encoder) - 1

        self.init_conv = nn.Sequential(
            nn.Conv1d(input_dim, emb_dim_encoder[0], 3, 1, 1),
            nn.ReLU()
        )

        blocks = []
        for i in range(len(emb_dim_encoder) - 1):
            in_channel = emb_dim_encoder[i]
            out_channel = emb_dim_encoder[i + 1]
            stride = 2 if downsample[i] else 1

            block = ConvTransformerEncoderStage(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=stride,
                nhead=nhead[i]
            )
            blocks.append(block)

        self.encoder = nn.Sequential(*blocks)

        self.post_conv = nn.Conv1d(emb_dim_encoder[-1], output_dim, 3, 1, 1)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.encoder(x)
        x = self.post_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim=576,
        output_dim=8,
        emb_dim_decoder=(576, 384, 192, 96, 48, 24),
        nhead=(18, 12, 12, 6, 6, 3),
        upsample=(0, 1, 0, 1, 0),
    ):
        super().__init__()
        assert len(upsample) == len(emb_dim_decoder) - 1

        self.init_conv = nn.Sequential(
            nn.Conv1d(input_dim, emb_dim_decoder[0], 3, 1, 1),
            nn.ReLU()
        )

        blocks = []
        for i in range(len(emb_dim_decoder) - 1):
            in_channel = emb_dim_decoder[i]
            out_channel = emb_dim_decoder[i + 1]
            stride = 2 if upsample[i] else 1
            kernel_size = 4 if upsample[i] else 3

            block = ConvTransformerDecoderStage(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                nhead=nhead[i],
                is_transposed=upsample[i]
            )
            blocks.append(block)

        self.decoder = nn.Sequential(*blocks)

        self.post_conv = nn.Conv1d(emb_dim_decoder[-1], output_dim, 3, 1, 1)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.decoder(x)
        x = self.post_conv(x)
        return x


if __name__ == "__main__":
    encoder = Encoder()
    decoder = Decoder()

    print(sum(p.numel() for p in encoder.parameters()))
    print(sum(p.numel() for p in decoder.parameters()))

    x = torch.randn(2, 24, 4)
    y = encoder(x)
    z = decoder(y)

    print(x.shape, y.shape, z.shape)



















