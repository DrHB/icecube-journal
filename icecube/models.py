# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_models.ipynb.

# %% auto 0
__all__ = ['LogCoshLoss', 'MeanPoolingWithMask', 'FeedForward', 'IceCubeModelEncoderV0', 'IceCubeModelEncoderV1']

# %% ../nbs/01_models.ipynb 1
import torch
from x_transformers import ContinuousTransformerWrapper, Encoder, Decoder
from torch import nn

# %% ../nbs/01_models.ipynb 2
class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))



class MeanPoolingWithMask(nn.Module):
    def __init__(self):
        super(MeanPoolingWithMask, self).__init__()

    def forward(self, x, mask):
        # Multiply the mask with the input tensor to zero out the padded values
        x = x * mask.unsqueeze(-1)

        # Sum the values along the sequence dimension
        x = torch.sum(x, dim=1)

        # Divide the sum by the number of non-padded values (i.e. the sum of the mask)
        x = x / torch.sum(mask, dim=1, keepdim=True)

        return x

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out = None, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class IceCubeModelEncoderV0(nn.Module):
    def __init__(slf):
        super().__init__()
        self.encoder = ContinuousTransformerWrapper(
            dim_in=6,
            dim_out=128,
            max_seq_len=150,
            attn_layers=Encoder(dim=128,
                        depth=3, 
                        heads=8),
        )

        #self.pool = MeanPoolingWithMask()
        self.head = FeedForward(128, 2)

    def forward(self, x, mask):
        x = self.encoder(x, mask = mask)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


class IceCubeModelEncoderV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ContinuousTransformerWrapper(
            dim_in=6,
            dim_out=128,
            max_seq_len=150,
            attn_layers=Encoder(dim=128,
                        depth=3, 
                        heads=8),
        )

        self.pool = MeanPoolingWithMask()
        self.head = FeedForward(128, 2)

    def forward(self, x, mask):
        x = self.encoder(x, mask = mask)
        x = self.pool(x, mask)
        x = self.head(x)
        return x
