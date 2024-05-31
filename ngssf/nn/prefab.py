from collections.abc import Collection
from functools import partial
from typing import Any, Callable, Iterable, Optional

from torch import nn

from ngssf.nn.encoding import FourierEncoding
from ngssf.nn.field import SmoothableNeuralField
from ngssf.nn.linear_norm import MatrixExpOptimizer, MatrixExpSpectralLimLinear, NormalizedDotProducts, \
    PowerIterSpectralLimLinear
from ngssf.util import IntScalar


class Smoothable4x1024NeuralField(SmoothableNeuralField):

    def __init__(
            self, coords: IntScalar, channels: IntScalar, anisotropic: bool,
            smoothable_comps: Optional[Collection[IntScalar]],
            fourier_encoding_kwargs: dict[str, Any] = None
    ) -> None:
        super().__init__(
            channels, anisotropic, smoothable_comps,
            FourierEncoding(coords, 1024, **(fourier_encoding_kwargs or {})),
            MatrixExpOptimizer(
                *_fc(3, 1024, linear=MatrixExpSpectralLimLinear),
                nn.ReLU(),
                NormalizedDotProducts(1024, channels)
            )
        )


class SpectralNormalizationSmoothable4x1024NeuralField(SmoothableNeuralField):

    def __init__(
            self, coords: IntScalar, channels: IntScalar, anisotropic: bool,
            smoothable_comps: Optional[Collection[IntScalar]],
            fourier_encoding_kwargs: dict[str, Any] = None
    ) -> None:
        super().__init__(
            channels, anisotropic, smoothable_comps,
            FourierEncoding(coords, 1024, **(fourier_encoding_kwargs or {})),
            nn.Sequential(
                *_fc(3, 1024, linear=partial(PowerIterSpectralLimLinear, divide_by_smax=True)),
                nn.ReLU(),
                NormalizedDotProducts(1024, channels)
            )
        )


class UnconstrainedSmoothable4x1024NeuralField(SmoothableNeuralField):

    def __init__(
            self, coords: IntScalar, channels: IntScalar, anisotropic: bool,
            smoothable_comps: Optional[Collection[IntScalar]],
            fourier_encoding_kwargs: dict[str, Any] = None
    ) -> None:
        super().__init__(
            channels, anisotropic, smoothable_comps,
            FourierEncoding(coords, 1024, **(fourier_encoding_kwargs or {})),
            nn.Sequential(*_fc(4, 1024, 1024, channels))
        )


class Smoothable4x1024LightStageNeuralField(SmoothableNeuralField):

    def __init__(
            self,
            img_pos_fourier_encoding_kwargs: dict[str, Any] = None,
            light_pos_fourier_encoding_kwargs: dict[str, Any] = None
    ) -> None:
        super().__init__(
            3, True, {2, 3},
            [
                FourierEncoding(2, 512, **(img_pos_fourier_encoding_kwargs or {})),
                FourierEncoding(2, 512, **(light_pos_fourier_encoding_kwargs or {}))
            ],
            MatrixExpOptimizer(
                *_fc(3, 1024, linear=MatrixExpSpectralLimLinear),
                nn.ReLU(),
                NormalizedDotProducts(1024, 3)
            ),
            concat_encodings=True
        )


def _fc(
        depth: IntScalar,
        in_features: IntScalar,
        hidden_features: Optional[IntScalar] = None,
        out_features: Optional[IntScalar] = None,
        linear: Callable[..., nn.Module] = nn.Linear,
        activation: Callable[..., nn.Module] = nn.ReLU
) -> Iterable[nn.Module]:
    if hidden_features is None:
        hidden_features = in_features
    if out_features is None:
        out_features = hidden_features
    if depth == 1:
        yield linear(in_features, out_features)
    elif depth > 1:
        yield linear(in_features, hidden_features)
        yield activation()
        for _ in range(depth - 2):
            yield linear(hidden_features, hidden_features)
            yield activation()
        yield linear(hidden_features, out_features)
    else:
        raise ValueError("Depth must be > 0.")
