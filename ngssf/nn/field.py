from collections.abc import Collection
from typing import Optional, Sequence, Union

import torch
from jaxtyping import Float
from torch import Tensor, nn

from ngssf.field import Field, Scales, SmoothableField
from ngssf.nn.encoding import FourierEncoding
from ngssf.util import FloatSeq, IntScalar


class SmoothableNeuralField(SmoothableField):

    def __init__(
            self,
            channels: IntScalar,
            anisotropic: bool,
            smoothable_comps: Optional[Collection[IntScalar]],
            encodings: Union[FourierEncoding, Sequence[FourierEncoding]],
            network: nn.Module,
            concat_encodings: bool = False
    ) -> None:
        if isinstance(encodings, FourierEncoding):
            encodings = [encodings]
        coords = sum(encoding.A.shape[1] for encoding in encodings)

        super().__init__(coords, channels, anisotropic, smoothable_comps)
        self.encodings = nn.ModuleList(encodings)
        self.concat_encodings = concat_encodings
        self.network = network
        self.register_buffer("calibration_factors", torch.ones(len(encodings)))

        self.slices = []
        c = 0
        for encoding in encodings:
            enc_coords = encoding.A.shape[1]
            self.slices.append(slice(c, c + enc_coords))
            c += enc_coords

    def forward(
            self, X: Float[Tensor, "batch coords"], scales: Optional[Scales] = None
    ) -> Float[Tensor, "batch channels"]:
        scales, anisotropic, _ = self._prepare_scales(X, scales)
        if len(self.encodings) == 1:
            return self.network(self._encode(self.encodings[0], self.calibration_factors[0], X, scales, anisotropic))
        else:
            sub_Xs = []
            for encoding, calibration_factor, sub_slice in zip(self.encodings, self.calibration_factors, self.slices):
                sub_scales = scales if scales is None or not anisotropic else scales[..., sub_slice, sub_slice]
                sub_Xs.append(self._encode(encoding, calibration_factor, X[:, sub_slice], sub_scales, anisotropic))
            if self.concat_encodings:
                return self.network(torch.cat(sub_Xs, dim=-1))
            else:
                return self.network(*sub_Xs)

    @staticmethod
    def _encode(encoding, calibration_factor, X, scales, anisotropic):
        X = encoding(X)
        if scales is not None:
            scales = scales * calibration_factor
            if not anisotropic:
                exponent = (scales[..., None] * encoding.A.square().sum(dim=1)).sqrt()
            else:
                exponent = (encoding.A.T * (scales @ encoding.A.T)).sum(dim=-2).clamp(0).sqrt()
            X = X * (-exponent).exp().tile(2)
        return X

    def normalization_step(self) -> None:
        for module in self.modules():
            if module is not self:
                func = getattr(module, "normalization_step", None)
                if callable(func):
                    func()

    # TODO: Calibration fails if anisotropic=False even though it only passes in isotropic covariance matrices.
    def calibrate(
            self,
            n_variances: IntScalar = 16,
            log_variance_range: FloatSeq = (-4.0, -1.0),
            n_mc_samples: IntScalar = 2000,
            n_uncalibrated_scales: IntScalar = 256,
            log_uncalibrated_scale_range: FloatSeq = (-5.0, 2.0),
            n_points: IntScalar = 64,
            point_distribution: str = "uniform",
            n_candidate_points: IntScalar = 10_000
    ) -> tuple[Float[Tensor, "variances"], Float[Tensor, "encodings scales"]]:
        self.calibration_factors.fill_(1)

        dev = self.device
        variances = torch.logspace(*log_variance_range, n_variances, device=dev)
        scales = torch.logspace(*log_uncalibrated_scale_range, n_uncalibrated_scales, device=dev)

        if point_distribution == "uniform":
            X = torch.rand(n_points, self.coords, device=dev) * 2 - 1
        elif point_distribution == "near_zero_level_set":
            X = torch.rand(n_candidate_points, self.coords, device=dev) * 2 - 1
            with torch.no_grad():
                X = X[self(X).norm(dim=1).argsort()[:n_points]]
        else:
            raise ValueError(f"Unknown point distribution: {point_distribution}")

        matching_scales = []
        sc_ctr = 0
        for e, (encoding, sub_slice) in enumerate(zip(self.encodings, self.slices)):
            sub_smoothable_comps = list(self.smoothable_comps & set(range(sub_slice.start, sub_slice.stop)))
            if len(sub_smoothable_comps) == 0:
                continue

            # Create a batch of "covariance matrices" (to be used as scale input). Each matrix is diagonal
            # and only enables the smoothable components associated with the current encoding.
            scale_mat_diags = torch.zeros((n_uncalibrated_scales, len(self.smoothable_comps)), device=dev)
            scale_mat_diags[:, sc_ctr:sc_ctr + len(sub_smoothable_comps)] = scales[:, None]
            sc_ctr += len(sub_smoothable_comps)
            scale_mats = torch.diag_embed(scale_mat_diags)

            # Prepare the standard deviations argument to be used when generating Gaussian noise.
            gauss_stds = variances.sqrt()[:, None].tile(n_mc_samples, len(sub_smoothable_comps))

            Y_ours = torch.empty((n_points, n_uncalibrated_scales, self.channels), device=dev)
            Y_gauss = torch.empty((n_points, n_variances, self.channels), device=dev)
            with torch.no_grad():
                for i, x in enumerate(X):
                    Y_ours[i] = self(x.tile(n_uncalibrated_scales, 1), scale_mats)
                    pert = x.repeat(n_variances * n_mc_samples, 1)
                    pert[:, sub_smoothable_comps] += torch.normal(0, gauss_stds)
                    Y_gauss[i] = self(pert).reshape(n_mc_samples, n_variances, self.channels).mean(dim=0)

            # For each variance, select the scale that best matches it.
            m_scales = scales[(Y_ours[:, None, :, :] - Y_gauss[:, :, None, :]).square().mean(dim=(0, 3)).argmin(dim=1)]
            self.calibration_factors[e] = (m_scales / variances).pow(1 / n_variances).prod()
            matching_scales.append(m_scales)

        return variances, torch.stack(matching_scales)

    @Field.device.getter
    def device(self):
        return self.calibration_factors.device

    def extra_repr(self) -> str:
        return (f"coords={self.coords}, channels={self.channels}, anisotropic={self.anisotropic}, "
                f"smoothable_comps={self.smoothable_comps}")
