# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_models.ipynb.

# %% auto 0
__all__ = ['DIST_KERNELS', 'LogCoshLoss', 'MeanPoolingWithMask', 'FeedForward', 'IceCubeModelEncoderV0', 'IceCubeModelEncoderV1',
           'always', 'l2norm', 'TokenEmbedding', 'IceCubeModelEncoderSensorEmbeddinng',
           'IceCubeModelEncoderSensorEmbeddinngV1', 'TokenEmbeddingV2', 'IceCubeModelEncoderSensorEmbeddinngV2',
           'exists', 'default', 'Residual', 'PreNorm', 'FeedForwardV1', 'Attention', 'MAT', 'MATMaskedPool',
           'IceCubeModelEncoderMAT', 'IceCubeModelEncoderMATMasked']

# %% ../nbs/01_models.ipynb 1
import torch
from x_transformers import ContinuousTransformerWrapper, Encoder, Decoder
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F



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
    def __init__(self, dim, dim_out=None, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class IceCubeModelEncoderV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ContinuousTransformerWrapper(
            dim_in=6,
            dim_out=128,
            max_seq_len=150,
            attn_layers=Encoder(dim=128, depth=6, heads=8),
        )

        # self.pool = MeanPoolingWithMask()
        self.head = FeedForward(128, 2)

    def forward(self, x, mask):
        x = self.encoder(x, mask=mask)
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
            attn_layers=Encoder(dim=128, depth=6, heads=8),
        )

        self.pool = MeanPoolingWithMask()
        self.head = FeedForward(128, 2)

    def forward(self, x, mask):
        x = self.encoder(x, mask=mask)
        x = self.pool(x, mask)
        x = self.head(x)
        return x


class always:
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


def l2norm(t, groups=1):
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")


class TokenEmbedding(nn.Module):
    def __init__(self, dim, num_tokens, l2norm_embed=False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim, padding_idx=0)

    def forward(self, x):
        token_emb = self.emb(x)
        return l2norm(token_emb) if self.l2norm_embed else token_emb

    def init_(self):
        nn.init.kaiming_normal_(self.emb.weight)


class IceCubeModelEncoderSensorEmbeddinng(nn.Module):
    def __init__(self, dim=128, in_features=14):
        super().__init__()
        self.token_emb = TokenEmbedding(dim, num_tokens=5161)
        self.post_norma = nn.LayerNorm(dim)
        self.token_emb.init_()
        self.encoder = ContinuousTransformerWrapper(
            dim_in=in_features + dim,
            dim_out=256,
            max_seq_len=150,
            attn_layers=Encoder(dim=256, depth=6, heads=8),
        )

        self.pool = MeanPoolingWithMask()
        self.head = FeedForward(256, 2)

    def forward(self, batch):
        x, mask, sensor_id = batch['event'], batch['mask'], batch['sensor_id']
        embed = self.token_emb(sensor_id)
        embed = self.post_norma(embed)
        x = torch.cat([x, embed], dim=-1)
        x = self.encoder(x, mask=mask)
        x = self.pool(x, mask)
        x = self.head(x)
        return x

class IceCubeModelEncoderSensorEmbeddinngV1(nn.Module):
    def __init__(self, dim=128, in_features=6):
        super().__init__()
        self.token_emb = TokenEmbedding(dim, num_tokens=5161)
        self.post_norma = nn.LayerNorm(dim)
        self.token_emb.init_()
        self.encoder = ContinuousTransformerWrapper(
            dim_in=in_features + dim,
            dim_out=256,
            max_seq_len=150,
            attn_layers=Encoder(dim=256, depth=6, heads=8),
        )

        self.pool = MeanPoolingWithMask()
        self.head = FeedForward(256, 2)

    def forward(self, batch):
        x, mask, sensor_id = batch['event'], batch['mask'], batch['sensor_id']
        embed = self.token_emb(sensor_id)
        embed = self.post_norma(embed)
        x = torch.cat([x, embed], dim=-1)
        x = self.encoder(x, mask=mask)
        x = self.pool(x, mask)
        x = self.head(x)
        return x


class TokenEmbeddingV2(nn.Module):
    def __init__(self, dim, num_tokens, l2norm_embed=False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x)
        return l2norm(token_emb) if self.l2norm_embed else token_emb

    def init_(self):
        nn.init.kaiming_normal_(self.emb.weight)

class IceCubeModelEncoderSensorEmbeddinngV2(nn.Module):
    def __init__(self, dim=128, in_features=6):
        super().__init__()
        self.token_emb = TokenEmbeddingV2(dim, num_tokens=5161)
        self.post_norma = nn.LayerNorm(dim)
        self.token_emb.init_()
        self.encoder = ContinuousTransformerWrapper(
            dim_in=in_features + dim,
            dim_out=256,
            max_seq_len=150,
            attn_layers=Encoder(dim=256, depth=6, heads=8),
        )

        self.pool = MeanPoolingWithMask()
        self.head = FeedForward(256, 2)

    def forward(self, batch):
        x, mask, sensor_id = batch['event'], batch['mask'], batch['sensor_id']
        embed = self.token_emb(sensor_id)
        x = torch.cat([x, embed], dim=-1)
        x = self.encoder(x, mask=mask)
        x = self.pool(x, mask)
        x = self.head(x)
        return x


# %% ../nbs/01_models.ipynb 5
# MOLECULAR TRANFORMER
DIST_KERNELS = {
    "exp": {
        "fn": lambda t: torch.exp(-t),
        "mask_value_fn": lambda t: torch.finfo(t.dtype).max,
    },
    "softmax": {
        "fn": lambda t: torch.softmax(t, dim=-1),
        "mask_value_fn": lambda t: -torch.finfo(t.dtype).max,
    },
}

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return d if not exists(val) else val


# helper classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForwardV1(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, dim, heads=8, dim_head=64, Lg=0.5, Ld=0.5, La=1, dist_kernel_fn="exp"
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        # hyperparameters controlling the weighted linear combination from
        # self-attention (La)
        # adjacency graph (Lg)
        # pair-wise distance matrix (Ld)

        self.La = La
        self.Ld = Ld
        self.Lg = Lg

        self.dist_kernel_fn = dist_kernel_fn

    def forward(self, x, mask=None, adjacency_mat=None, distance_mat=None):
        h, La, Ld, Lg, dist_kernel_fn = (
            self.heads,
            self.La,
            self.Ld,
            self.Lg,
            self.dist_kernel_fn,
        )

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b n (h qkv d) -> b h n qkv d", h=h, qkv=3).unbind(
            dim=-2
        )
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        assert (
            dist_kernel_fn in DIST_KERNELS
        ), f"distance kernel function needs to be one of {DIST_KERNELS.keys()}"
        dist_kernel_config = DIST_KERNELS[dist_kernel_fn]

        if exists(distance_mat):
            distance_mat = rearrange(distance_mat, "b i j -> b () i j")

        if exists(adjacency_mat):
            adjacency_mat = rearrange(adjacency_mat, "b i j -> b () i j")

        if exists(mask):
            mask_value = torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * mask[:, None, None, :]

            # mask attention
            dots.masked_fill_(~mask, -mask_value)

            if exists(distance_mat):
                # mask distance to infinity
                # todo - make sure for softmax distance kernel, use -infinity
                dist_mask_value = dist_kernel_config["mask_value_fn"](dots)
                distance_mat.masked_fill_(~mask, dist_mask_value)

            if exists(adjacency_mat):
                adjacency_mat.masked_fill_(~mask, 0.0)

        attn = dots.softmax(dim=-1)

        # sum contributions from adjacency and distance tensors
        attn = attn * La

        if exists(adjacency_mat):
            attn = attn + Lg * adjacency_mat

        if exists(distance_mat):
            distance_mat = dist_kernel_config["fn"](distance_mat)
            attn = attn + Ld * distance_mat

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# main class


class MAT(nn.Module):
    def __init__(
        self,
        *,
        dim_in,
        model_dim,
        dim_out,
        depth,
        heads=8,
        Lg=0.5,
        Ld=0.5,
        La=1,
        dist_kernel_fn="exp",
    ):
        super().__init__()

        self.embed_to_model = nn.Linear(dim_in, model_dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            layer = nn.ModuleList(
                [
                    Residual(
                        PreNorm(
                            model_dim,
                            Attention(
                                model_dim,
                                heads=heads,
                                Lg=Lg,
                                Ld=Ld,
                                La=La,
                                dist_kernel_fn=dist_kernel_fn,
                            ),
                        )
                    ),
                    Residual(PreNorm(model_dim, FeedForwardV1(model_dim))),
                ]
            )
            self.layers.append(layer)

        self.norm_out = nn.LayerNorm(model_dim)
        self.ff_out = FeedForward(model_dim, dim_out)


    def forward(self, batch):

        x = batch["event"]
        mask = batch["mask"]
        adjacency_mat = batch["adjecent_matrix"]
        distance_mat = batch["distance_matrix"]

        x = self.embed_to_model(x)

        for (attn, ff) in self.layers:
            x = attn(
                x, mask=mask, adjacency_mat=adjacency_mat, distance_mat=distance_mat
            )
            x = ff(x)

        x = self.norm_out(x)
        x = x.mean(dim=-2)
        x = self.ff_out(x)
        return x


class MATMaskedPool(nn.Module):
    def __init__(
        self,
        *,
        dim_in,
        model_dim,
        dim_out,
        depth,
        heads=8,
        Lg=0.5,
        Ld=0.5,
        La=1,
        dist_kernel_fn="exp",
    ):
        super().__init__()

        self.embed_to_model = nn.Linear(dim_in, model_dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            layer = nn.ModuleList(
                [
                    Residual(
                        PreNorm(
                            model_dim,
                            Attention(
                                model_dim,
                                heads=heads,
                                Lg=Lg,
                                Ld=Ld,
                                La=La,
                                dist_kernel_fn=dist_kernel_fn,
                            ),
                        )
                    ),
                    Residual(PreNorm(model_dim, FeedForwardV1(model_dim))),
                ]
            )
            self.layers.append(layer)

        self.norm_out = nn.LayerNorm(model_dim)
        self.pool = MeanPoolingWithMask()
        self.ff_out = FeedForward(model_dim, dim_out)


    def forward(self, batch):

        x = batch["event"]
        mask = batch["mask"]
        adjacency_mat = batch["adjecent_matrix"]
        distance_mat = batch["distance_matrix"]

        x = self.embed_to_model(x)

        for (attn, ff) in self.layers:
            x = attn(
                x, mask=mask, adjacency_mat=adjacency_mat, distance_mat=distance_mat
            )
            x = ff(x)

        x = self.norm_out(x)
        x = self.pool(x, mask=mask)
        x = self.ff_out(x)
        return x


class IceCubeModelEncoderMAT(nn.Module):
    def __init__(self, dim=128, in_features=6):
        super().__init__()
        self.md = MAT(
            dim_in=6,
            model_dim=128,
            dim_out=2,
            depth=6,
            Lg=0.5,  # lambda (g)raph - weight for adjacency matrix
            Ld=0.5,  # lambda (d)istance - weight for distance matrix
            La=1,  # lambda (a)ttention - weight for usual self-attention
            dist_kernel_fn="exp",  # distance kernel fn - either 'exp' or 'softmax'
        )

    def forward(self, batch):
        return self.md(batch)


class IceCubeModelEncoderMATMasked(nn.Module):
    def __init__(self, dim=128, in_features=6):
        super().__init__()
        self.md = MATMaskedPool(
            dim_in=6,
            model_dim=128,
            dim_out=2,
            depth=6,
            Lg=0.5,  # lambda (g)raph - weight for adjacency matrix
            Ld=0.5,  # lambda (d)istance - weight for distance matrix
            La=1,  # lambda (a)ttention - weight for usual self-attention
            dist_kernel_fn="exp",  # distance kernel fn - either 'exp' or 'softmax'
        )

    def forward(self, batch):
        return self.md(batch)


