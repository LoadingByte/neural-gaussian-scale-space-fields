from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from jaxtyping import Float
from pysdf import SDF
from scipy.stats.qmc import Sobol
from torch import Tensor, nn
from trimesh import Trimesh

from ngssf.field import Field, LightStageField
from ngssf.util import FloatScalar, FloatSeq, IntScalar


class Sampler(nn.Module, ABC):

    @abstractmethod
    def forward(self, n: IntScalar) -> tuple[Float[Tensor, "batch coords"], Float[Tensor, "batch channels"]]:
        raise NotImplementedError


class FieldSampler(Sampler):

    def __init__(self, field: Field, bounds: FloatScalar = 1.2) -> None:
        super().__init__()
        self.field = field
        self.bounds = bounds

    def forward(self, n: IntScalar) -> tuple[Float[Tensor, "batch coords"], Float[Tensor, "batch channels"]]:
        X = (torch.rand(n, self.field.coords, device=self.field.device) - 0.5) * (2 * self.bounds)
        return X, self.field(X)


class SDFSampler(Sampler):

    def __init__(self, mesh: Trimesh, bounds: FloatScalar = 1.2, uniform_ratio: FloatScalar = 0.25) -> None:
        super().__init__()
        self.sdf = SDF(mesh.vertices, mesh.faces)
        self.bounds = bounds
        self.uniform_ratio = uniform_ratio
        self.register_buffer("device_indicator", torch.empty(0), persistent=False)

    def forward(self, n: IntScalar) -> tuple[Float[Tensor, "batch 3"], Float[Tensor, "batch 1"]]:
        dev = self.device_indicator.device
        n1 = round(n * self.uniform_ratio)
        n2 = n - n1
        n21 = n2 // 2
        n22 = n2 - n21
        uniform_coords = (Sobol(3).random(n1).astype(np.float32) - 0.5) * (2 * self.bounds)
        surface_coords = self.sdf.sample_surface(n2)
        surface_coords[:n21] += 0.1 * np.random.standard_normal((n21, 3)).astype(np.float32)
        samples = -self.sdf(np.concatenate([uniform_coords, surface_coords[:n21]]))
        X = torch.as_tensor(np.concatenate([uniform_coords, surface_coords]), device=dev)
        Y = torch.as_tensor(np.concatenate([samples, np.zeros(n22, dtype=np.float32)]), device=dev)[:, None]
        return X, Y


class LightStageFieldSampler(Sampler):

    def __init__(
            self,
            field: LightStageField,
            pixel_bounds: Union[FloatScalar, FloatSeq],
            light_center: FloatSeq,
            light_radius: FloatScalar
    ) -> None:
        super().__init__()
        self.field = field
        self.register_buffer("pixel_bounds", torch.asarray(pixel_bounds))
        self.light_center = light_center
        self.light_radius = light_radius

    def forward(self, n: IntScalar) -> tuple[Float[Tensor, "batch 4"], Float[Tensor, "batch 3"]]:
        radii = torch.rand(n, device=self.field.device) * self.light_radius
        angles = torch.rand(n, device=self.field.device) * (2 * torch.pi)
        X = torch.column_stack([
            (torch.rand(n, 2, device=self.field.device) - 0.5) * (2 * self.pixel_bounds),
            self.light_center[0] + radii * angles.cos(),
            self.light_center[1] + radii * angles.sin()
        ])
        return X, self.field(X)


class MinibatchSampler(Sampler):

    def __init__(self, source_size: IntScalar, source_sampler: Sampler, source_lifetime: IntScalar = -1) -> None:
        super().__init__()
        self.source_sampler = source_sampler
        self.source_size = source_size
        self.source_lifetime = source_lifetime
        self.remaining_source_lifetime = 0
        self.register_buffer("source_X", None)
        self.register_buffer("source_Y", None)

    def forward(self, n: IntScalar) -> tuple[Float[Tensor, "batch coords"], Float[Tensor, "batch channels"]]:
        if self.remaining_source_lifetime == 0:
            self.source_X, self.source_Y = self.source_sampler(self.source_size)
            self.remaining_source_lifetime = self.source_lifetime
        if self.remaining_source_lifetime >= 0:
            self.remaining_source_lifetime -= 1
        # Note: Currently, the only way to sample without replacement is via torch.randperm(), which becomes slow for
        # very large source size. So we just sample with replacement and accept the occasional redundant sample.
        # In our preliminary experiments, this strategy wasn't worse than truly sampling without replacement.
        indices = torch.randint(self.source_X.shape[0], (n,), device=self.source_X.device)
        return self.source_X[indices], self.source_Y[indices]
