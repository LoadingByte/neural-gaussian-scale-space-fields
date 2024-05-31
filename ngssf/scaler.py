from abc import ABC, abstractmethod

import torch
from scipy.stats import ortho_group
from torch import nn

from ngssf.field import Scales
from ngssf.util import FloatSeq, IntScalar


class Scaler(nn.Module, ABC):

    @abstractmethod
    def forward(self, n: IntScalar) -> Scales:
        raise NotImplementedError


class RandomScaler(Scaler):

    def __init__(
            self,
            anisotropic: bool,
            coords: IntScalar = -1,
            log_range: FloatSeq = (-12.0, 2.0)
    ) -> None:
        super().__init__()
        if anisotropic and coords < 0:
            raise ValueError("For an anisotropic random scaler, 'coords' must be specified.")
        self.anisotropic = anisotropic
        self.coords = coords
        self.log_range = log_range
        self.register_buffer("device_indicator", torch.empty(0), persistent=False)

    def forward(self, n: IntScalar) -> Scales:
        dev = self.device_indicator.device
        if self.anisotropic:
            return self._rand_covariance_matrices_with_logrand_eigenvalues(*self.log_range, self.coords, n, device=dev)
        else:
            return self._logrand(*self.log_range, n, device=dev)

    @staticmethod
    def _logrand(log_start, log_end, size, **kwargs):
        return torch.pow(10, torch.rand(size, **kwargs) * (log_end - log_start) + log_start)

    @staticmethod
    def _rand_covariance_matrices_with_logrand_eigenvalues(
            log_start_eigenvalue, log_end_eigenvalue, coords, batch, **kwargs
    ):
        eigenvalues = RandomScaler._logrand(log_start_eigenvalue, log_end_eigenvalue, (batch, coords), **kwargs)
        # Note: https://github.com/JesseLivezey/pytorch_group_sampler would provide a pure PyTorch version of this.
        eigenvectors = torch.tensor(ortho_group.rvs(coords, batch), dtype=eigenvalues.dtype, device=eigenvalues.device)
        return eigenvectors @ (eigenvalues[..., None] * eigenvectors.mT)


class MinibatchScaler(Scaler):

    def __init__(self, source_size: IntScalar, source_scaler: Scaler) -> None:
        super().__init__()
        self.source_size = source_size
        self.source_scaler = source_scaler
        self.register_buffer("source_scales", None)

    def forward(self, n: IntScalar) -> Scales:
        if self.source_scales is None:
            self.source_scales = self.source_scaler(self.source_size)
        return self.source_scales[torch.randint(self.source_scales.shape[0], (n,), device=self.source_scales.device)]
