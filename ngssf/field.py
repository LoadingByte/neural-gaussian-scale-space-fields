from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Optional, Union

import torch
import torch.nn.functional as F
from jaxtyping import Float
from pysdf import SDF
from torch import Tensor, nn
from torch.distributions import MultivariateNormal
from trimesh import Trimesh

from ngssf.util import FloatScalar, IntScalar, IntSeq


class Field(nn.Module, ABC):

    def __init__(self, coords: IntScalar, channels: IntScalar) -> None:
        super().__init__()
        self.coords = int(coords)
        self.channels = int(channels)

    @abstractmethod
    def forward(self, X: Float[Tensor, "batch coords"]) -> Float[Tensor, "batch channels"]:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self):
        raise NotImplementedError


class GridField(Field):

    def __init__(
            self, grid: Float[Tensor, "channels *dims"], bounds: FloatScalar = 1.0, padding_mode: str = "border"
    ) -> None:
        super().__init__(grid.ndim - 1, grid.shape[0])
        if padding_mode not in ("zeros", "border", "reflection"):
            raise ValueError(f"Illegal padding mode: {padding_mode}")
        self.bounds = bounds
        self.padding_mode = padding_mode
        # Due to PyTorch limitations, interpret 1D grids as being 2D with a height of one.
        if self.coords == 1:
            grid = grid[:, None, :]
        # Add a batch dimension.
        self.register_buffer("input", grid[None])

    def forward(self, X: Float[Tensor, "batch coords"]) -> Float[Tensor, "batch channels"]:
        if self.bounds != 1.0:
            X = X / self.bounds
        # Once again, interpret 1D grids as 2D.
        if self.coords == 1:
            X = torch.hstack((X, torch.zeros_like(X)))
        X = X[(None,) * (self.input.ndim - 2)]
        Y = F.grid_sample(self.input, X, padding_mode=self.padding_mode, align_corners=True)
        return Y[(0, slice(None)) + (0,) * (self.input.ndim - 3)].T

    @Field.device.getter
    def device(self):
        return self.input.device


class SDFField(Field):

    def __init__(self, mesh: Trimesh) -> None:
        super().__init__(3, 1)
        self.sdf = SDF(mesh.vertices, mesh.faces)

    def forward(self, X: Float[Tensor, "batch 3"]) -> Float[Tensor, "batch 1"]:
        return torch.as_tensor(-self.sdf(X.detach().numpy()))[:, None]

    @Field.device.getter
    def device(self):
        return "cpu"


class LightStageField(Field):

    def __init__(
            self,
            light_shots: Float[Tensor, "lights channels height width"],
            light_positions: Float[Tensor, "lights 2"]
    ) -> None:
        super().__init__(4, 3)
        self.light_shots_field = GridField(light_shots.permute(1, 0, 2, 3))
        self.register_buffer("light_positions", light_positions)

    def forward(self, X: Float[Tensor, "batch 4"]) -> Float[Tensor, "batch 3"]:
        light_pos = self.light_positions
        light_ind = (X[:, None, 2:] - light_pos).square().sum(dim=2).topk(3, dim=1, largest=False, sorted=True)[1]
        light_corners = light_pos[light_ind]
        light_weights = self._barycentric_coordinates(light_corners, X[:, 2:])
        light_colors = self.light_shots_field(torch.column_stack([
            X[:, :2].repeat_interleave(3, dim=0),
            light_ind.flatten() / ((light_pos.shape[0] - 1) / 2) - 1
        ])).reshape(-1, 3, 3)
        return (light_weights[:, :, None] * light_colors).sum(dim=1)

    @staticmethod
    def _barycentric_coordinates(corners, points):
        v0 = corners[:, 1] - corners[:, 0]
        v1 = corners[:, 2] - corners[:, 0]
        v2 = points - corners[:, 0]
        d00 = v0.square().sum(dim=1)
        d11 = v1.square().sum(dim=1)
        d01 = (v0 * v1).sum(dim=1)
        d20 = (v2 * v0).sum(dim=1)
        d21 = (v2 * v1).sum(dim=1)
        denom = d00 * d11 - d01.square()
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w
        uvw = torch.column_stack([u, v, w])
        outside = ((uvw < 0) | uvw.isnan()).any(dim=1)
        if outside.any():
            t = (d20[outside] / d00[outside]).clamp(0, 1)
            uvw[outside, 0] = 1 - t
            uvw[outside, 1] = t
            uvw[outside, 2] = 0
        return uvw

    @Field.device.getter
    def device(self):
        return self.light_shots_field.device


class AckleyField(Field):

    def __init__(self) -> None:
        super().__init__(2, 1)
        self.register_buffer("device_indicator", torch.empty(0), persistent=False)

    def forward(self, X: Float[Tensor, "batch 2"]) -> Float[Tensor, "batch 1"]:
        X = X * 4
        Y = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * X.square().sum(dim=1, keepdim=True))) - \
            torch.exp(0.5 * torch.cos((2 * torch.pi) * X).sum(dim=1, keepdim=True)) + torch.e + 20
        return Y / 10

    @Field.device.getter
    def device(self):
        return self.device_indicator.device


Scales = Union[
    FloatScalar,
    Float[Tensor, "batch"],
    Float[Tensor, "smoothable_coords smoothable_coords"],
    Float[Tensor, "batch smoothable_coords smoothable_coords"]
]


class SmoothableField(Field):

    def __init__(
            self, coords: IntScalar, channels: IntScalar, anisotropic: bool,
            smoothable_comps: Optional[Collection[IntScalar]]
    ) -> None:
        super().__init__(coords, channels)
        self.anisotropic = anisotropic
        self.smoothable_comps = set(map(int, smoothable_comps)) if smoothable_comps is not None else set(range(coords))
        if len(self.smoothable_comps) == 0:
            raise ValueError("Smoothable field without smoothable components.")
        if any(c < 0 or c >= coords for c in self.smoothable_comps):
            raise ValueError("Some smoothable components are not even part of the input coordinates.")
        if not anisotropic and len(self.smoothable_comps) < coords:
            raise ValueError("Anisotropic smoothable field can't restrict smoothable components.")

    @abstractmethod
    def forward(
            self, X: Float[Tensor, "batch coords"], scales: Optional[Scales] = None
    ) -> Float[Tensor, "batch channels"]:
        raise NotImplementedError

    def _prepare_scales(
            self, X: Float[Tensor, "batch coords"], scales: Optional[Scales]
    ) -> tuple[Optional[Float[Tensor, "..."]], bool, bool]:
        if scales is None:
            return None, False, False
        if not isinstance(scales, Tensor):
            scales = torch.as_tensor(scales, dtype=X.dtype, device=X.device)
        if not scales.any():
            return None, False, False
        # If "scales" is just a single number, be merciful and fix device mismatches for the user.
        if scales.shape == () and scales.device != X.device:
            scales = scales.to(X.device)
        batch_shape = X.shape[:1]
        aniso_shape = (len(self.smoothable_comps),) * 2
        if scales.shape == () or scales.shape == batch_shape:
            if len(self.smoothable_comps) == self.coords:
                return scales, False, scales.ndim == 1
            scales = torch.diag_embed(scales[..., None].tile(aniso_shape[0]))
        if scales.shape == aniso_shape or scales.shape == batch_shape + aniso_shape:
            if not self.anisotropic:
                raise ValueError("This field does not support anisotropic smoothing.")
            scales = self._supermatrix(scales, list(self.smoothable_comps), self.coords)
            return scales, True, scales.ndim == 3
        raise ValueError(f"Scales shape {scales.shape} clashes with field type or X shape {X.shape}.")

    @staticmethod
    def _supermatrix(
            M: Float[Tensor, "*batch sub_coords sub_coords"], sub_comps: Union[slice, IntSeq], sup_coords: IntScalar
    ) -> Float[Tensor, "*batch sup_coords sup_coords"]:
        batch_shape = M.shape[:-2]
        sub_coords = M.shape[-1]
        N = torch.zeros((*batch_shape, sup_coords, sub_coords), dtype=M.dtype, device=M.device)
        O = torch.zeros((*batch_shape, sup_coords, sup_coords), dtype=M.dtype, device=M.device)
        N[..., sub_comps, :] = M
        O[..., :, sub_comps] = N
        return O


class GaussianMonteCarloSmoothableField(SmoothableField):

    def __init__(self, source_field: Field, smoothable_comps: Optional[Collection[IntScalar]] = None):
        super().__init__(source_field.coords, source_field.channels, True, smoothable_comps)
        self.source_field = source_field

    def forward(
            self, X: Float[Tensor, "batch coords"], scales: Optional[Scales] = None
    ) -> Float[Tensor, "batch channels"]:
        scales, anisotropic, individual = self._prepare_scales(X, scales)
        if scales is None:
            return self.source_field(X)
        if not anisotropic:
            q = (1e5 * scales.max()).clamp(10, 1000).pow(self.coords)
        else:
            q = (1e5 * torch.linalg.eigvalsh(scales)).clamp(10, 1000).prod(dim=-1).max()
        n_samples = round(20 * q.sqrt().item())
        accumulator = torch.zeros((X.shape[0], self.channels), device=self.device)
        if not anisotropic:
            std = scales.sqrt()
            if individual:
                std = std[:, None]
            with torch.no_grad():
                for _ in range(n_samples):
                    accumulator += self.source_field(torch.normal(X, std))
        else:
            scales = scales.clone()
            scales.diagonal().clamp_(1e-10)
            dist = MultivariateNormal(X, scales)
            with torch.no_grad():
                for _ in range(n_samples):
                    accumulator += self.source_field(dist.sample())
        return accumulator / n_samples

    @Field.device.getter
    def device(self):
        return self.source_field.device
