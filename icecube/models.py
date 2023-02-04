# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_models.ipynb.

# %% auto 0
__all__ = ['DIST_KERNELS', 'LogCoshLoss', 'SigmoidRange', 'Adjustoutput', 'MeanPoolingWithMask', 'FeedForward',
           'IceCubeModelEncoderV0', 'IceCubeModelEncoderV1', 'always', 'l2norm', 'TokenEmbedding',
           'IceCubeModelEncoderSensorEmbeddinng', 'IceCubeModelEncoderSensorEmbeddinngV1', 'TokenEmbeddingV2',
           'IceCubeModelEncoderSensorEmbeddinngV2', 'IceCubeModelEncoderSensorEmbeddinngV3', 'LogCMK', 'LossFunction',
           'VonMisesFisherLoss', 'VonMisesFisher2DLoss', 'VonMisesFisher3DLoss', 'eps_like',
           'AzimuthReconstructionWithKappa', 'DirectionReconstructionWithKappa', 'ZenithReconstruction',
           'IceCubeModelEncoderV2', 'exists', 'default', 'Residual', 'PreNorm', 'FeedForwardV1', 'Attention', 'MAT',
           'MATMaskedPool', 'IceCubeModelEncoderMAT', 'IceCubeModelEncoderMATMasked']

# %% ../nbs/01_models.ipynb 1
import torch
from x_transformers import ContinuousTransformerWrapper, Encoder, Decoder
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F
from datasets import load_from_disk
from abc import abstractmethod
from torch import Tensor
from typing import Optional, Any
import scipy
import numpy as np


# %% ../nbs/01_models.ipynb 2
class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x):
        return torch.sigmoid(x) * (self.high - self.low) + self.low

class Adjustoutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.az = SigmoidRange(6.436839548775502e-08, 6.2891)
        self.zn = SigmoidRange(8.631674577710722e-05, 3.1417)

    def forward(self, x):
        x[:, 0] = self.az(x[:, 0])
        x[:, 1] = self.zn(x[:, 1])
        return x


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

    def forward(self, batch):
        x, mask = batch['event'], batch['mask']
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

    def forward(self, batch):
        x, mask = batch['event'], batch['mask']
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
            max_seq_len=196,
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

class IceCubeModelEncoderSensorEmbeddinngV3(nn.Module):
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
        self.sigmout = Adjustoutput()

    def forward(self, batch):
        x, mask, sensor_id = batch['event'], batch['mask'], batch['sensor_id']
        embed = self.token_emb(sensor_id)
        x = torch.cat([x, embed], dim=-1)
        x = self.encoder(x, mask=mask)
        x = self.pool(x, mask)
        x = self.head(x)
        s = self.sigmout(x)
        return x


# %% ../nbs/01_models.ipynb 4
class LogCMK(torch.autograd.Function):
    """MIT License.
    Copyright (c) 2019 Max Ryabinin
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    _____________________
    From [https://github.com/mryab/vmf_loss/blob/master/losses.py]
    Modified to use modified Bessel function instead of exponentially scaled ditto
    (i.e. `.ive` -> `.iv`) as indiciated in [1812.04616] in spite of suggestion in
    Sec. 8.2 of this paper. The change has been validated through comparison with
    exact calculations for `m=2` and `m=3` and found to yield the correct results.
    """

    @staticmethod
    def forward(
        ctx: Any, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name,arguments-differ
        """Forward pass."""
        dtype = kappa.dtype
        ctx.save_for_backward(kappa)
        ctx.m = m
        ctx.dtype = dtype
        kappa = kappa.double()
        iv = torch.from_numpy(
            scipy.special.iv(m / 2.0 - 1, kappa.cpu().numpy())
        ).to(kappa.device)
        return (
            (m / 2.0 - 1) * torch.log(kappa)
            - torch.log(iv)
            - (m / 2) * np.log(2 * np.pi)
        ).type(dtype)

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name,arguments-differ
        """Backward pass."""
        kappa = ctx.saved_tensors[0]
        m = ctx.m
        dtype = ctx.dtype
        kappa = kappa.double().cpu().numpy()
        grads = -(
            (scipy.special.iv(m / 2.0, kappa))
            / (scipy.special.iv(m / 2.0 - 1, kappa))
        )
        return (
            None,
            grad_output
            * torch.from_numpy(grads).to(grad_output.device).type(dtype),
        )



class LossFunction(nn.Module):
    """Base class for loss functions in `graphnet`."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(  # type: ignore[override]
        self,
        prediction: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
        return_elements: bool = False,
    ) -> Tensor:
        """Forward pass for all loss functions.
        Args:
            prediction: Tensor containing predictions. Shape [N,P]
            target: Tensor containing targets. Shape [N,T]
            return_elements: Whether elementwise loss terms should be returned.
                The alternative is to return the averaged loss across examples.
        Returns:
            Loss, either averaged to a scalar (if `return_elements = False`) or
            elementwise terms with shape [N,] (if `return_elements = True`).
        """
        elements = self._forward(prediction, target)
        if weights is not None:
            elements = elements * weights
        assert elements.size(dim=0) == target.size(
            dim=0
        ), "`_forward` should return elementwise loss terms."

        return elements if return_elements else torch.mean(elements)

    @abstractmethod
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Syntax like `.forward`, for implentation in inheriting classes."""

class VonMisesFisherLoss(LossFunction):
    """General class for calculating von Mises-Fisher loss.
    Requires implementation for specific dimension `m` in which the target and
    prediction vectors need to be prepared.
    """

    @classmethod
    def log_cmk_exact(
        cls, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss exactly."""
        return LogCMK.apply(m, kappa)

    @classmethod
    def log_cmk_approx(
        cls, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss approx.
        [https://arxiv.org/abs/1812.04616] Sec. 8.2 with additional minus sign.
        """
        v = m / 2.0 - 0.5
        a = torch.sqrt((v + 1) ** 2 + kappa**2)
        b = v - 1
        return -a + b * torch.log(b + a)

    @classmethod
    def log_cmk(
        cls, m: int, kappa: Tensor, kappa_switch: float = 100.0
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss.
        Since `log_cmk_exact` is diverges for `kappa` >~ 700 (using float64
        precision), and since `log_cmk_approx` is unaccurate for small `kappa`,
        this method automatically switches between the two at `kappa_switch`,
        ensuring continuity at this point.
        """
        kappa_switch = torch.tensor([kappa_switch]).to(kappa.device)
        mask_exact = kappa < kappa_switch

        # Ensure continuity at `kappa_switch`
        offset = cls.log_cmk_approx(m, kappa_switch) - cls.log_cmk_exact(
            m, kappa_switch
        )
        ret = cls.log_cmk_approx(m, kappa) - offset
        ret[mask_exact] = cls.log_cmk_exact(m, kappa[mask_exact])
        return ret

    def _evaluate(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for a vector in D dimensons.
        This loss utilises the von Mises-Fisher distribution, which is a
        probability distribution on the (D - 1) sphere in D-dimensional space.
        Args:
            prediction: Predicted vector, of shape [batch_size, D].
            target: Target unit vector, of shape [batch_size, D].
        Returns:
            Elementwise von Mises-Fisher loss terms.
        """
        # Check(s)
        assert prediction.dim() == 2
        assert target.dim() == 2
        assert prediction.size() == target.size()

        # Computing loss
        m = target.size()[1]
        k = torch.norm(prediction, dim=1)
        dotprod = torch.sum(prediction * target, dim=1)
        elements = -self.log_cmk(m, k) - dotprod
        return elements

    @abstractmethod
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError

class VonMisesFisher2DLoss(VonMisesFisherLoss):
    """von Mises-Fisher loss function vectors in the 2D plane."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for an angle in the 2D plane.
        Args:
            prediction: Output of the model. Must have shape [N, 2] where 0th
                column is a prediction of `angle` and 1st column is an estimate
                of `kappa`.
            target: Target tensor, extracted from graph object.
        Returns:
            loss: Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 2
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        # Formatting target
        angle_true = target[:, 0]
        t = torch.stack(
            [
                torch.cos(angle_true),
                torch.sin(angle_true),
            ],
            dim=1,
        )

        # Formatting prediction
        angle_pred = prediction[:, 0]
        kappa = prediction[:, 1]
        p = kappa.unsqueeze(1) * torch.stack(
            [
                torch.cos(angle_pred),
                torch.sin(angle_pred),
            ],
            dim=1,
        )

        return self._evaluate(p, t)

class VonMisesFisher3DLoss(VonMisesFisherLoss):
    """von Mises-Fisher loss function vectors in the 3D plane."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for a direction in the 3D.
        Args:
            prediction: Output of the model. Must have shape [N, 4] where
                columns 0, 1, 2 are predictions of `direction` and last column
                is an estimate of `kappa`.
            target: Target tensor, extracted from graph object.
        Returns:
            Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        target = target.reshape(-1, 3)
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 4
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        kappa = prediction[:, 3]
        p = kappa.unsqueeze(1) * prediction[:, [0, 1, 2]]
        return self._evaluate(p, target)

def eps_like(tensor: torch.Tensor) -> torch.Tensor:
    """Return `eps` matching `tensor`'s dtype."""
    return torch.finfo(tensor.dtype).eps

class AzimuthReconstructionWithKappa(nn.Module):
    """Module for predicting azimuth angle and kappa."""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        kappa = torch.linalg.vector_norm(x, dim=1) + eps_like(x)
        angle = torch.atan2(x[:, 1], x[:, 0])
        angle = torch.where(
            angle < 0, angle + 2 * np.pi, angle
        )  
        return torch.stack((angle, kappa), dim=1)
        
class DirectionReconstructionWithKappa(nn.Module):
    """Module for predicting direction and kappa."""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 3)

    def forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        x = self.linear(x)
        kappa = torch.linalg.vector_norm(x, dim=1) + eps_like(x)
        vec_x = x[:, 0] / kappa
        vec_y = x[:, 1] / kappa
        vec_z = x[:, 2] / kappa
        return torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)

class ZenithReconstruction(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: Tensor):
        x = self.linear(x)
        return torch.sigmoid(x[:, :1]) * np.pi


class IceCubeModelEncoderV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ContinuousTransformerWrapper(
            dim_in=8,
            dim_out=128,
            max_seq_len=150,
            attn_layers=Encoder(dim=128, depth=6, heads=8),
        )

        self.pool = MeanPoolingWithMask()
        self.out = DirectionReconstructionWithKappa(128)

    def forward(self, batch):
        x, mask = batch['event'], batch['mask']
        x = self.encoder(x, mask=mask)
        x = self.pool(x, mask)
        return self.out(x)

# %% ../nbs/01_models.ipynb 7
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


