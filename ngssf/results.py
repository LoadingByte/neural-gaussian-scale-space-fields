from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from ngssf.data import signature
from ngssf.nn.field import SmoothableNeuralField
from ngssf.nn.prefab import Smoothable4x1024LightStageNeuralField, Smoothable4x1024NeuralField, \
    SpectralNormalizationSmoothable4x1024NeuralField, UnconstrainedSmoothable4x1024NeuralField
from ngssf.util import IntScalar

_results_dir = Path(__file__).parent.parent / "results"


def load_benchmark(
        field_type: str, category: str, name: str, scale_set: str, index: IntScalar
) -> Float[Tensor, "channels *dims"]:
    return torch.load(_benchmark_file(field_type, category, name, scale_set, index))


def store_benchmark(
        field_type: str, category: str, name: str, scale_set: str, index: IntScalar,
        benchmark: Float[Tensor, "channels *dims"]
) -> None:
    torch.save(benchmark.cpu().clone(), _mk_parent(_benchmark_file(field_type, category, name, scale_set, index)))


def _benchmark_file(field_type, category, name, scale_set, index):
    filename = f"{index:02}.pt" if scale_set == "covariance_matrix_benchmark" else f"{index}.pt"
    return case_dir(field_type, category, name) / scale_set / filename


def load_benchmark_metrics(
        field_type: str, category: str, name: str, scale_set: str
) -> tuple[list[str], Float[Tensor, "specimens metrics"]]:
    file = _benchmark_metrics_file(field_type, category, name, scale_set)
    with open(file) as f:
        header = f.readline().strip().split(",")
    return header, torch.as_tensor(np.loadtxt(file, delimiter=",", skiprows=1))


def store_benchmark_metrics(
        field_type: str, category: str, name: str, scale_set: str, header: Sequence[str],
        metrics: Float[Tensor, "scales metrics"]
) -> None:
    _store_metrics(_mk_parent(_benchmark_metrics_file(field_type, category, name, scale_set)), header, metrics)


def _benchmark_metrics_file(field_type, category, name, scale_set):
    return case_dir(field_type, category, name) / f"metrics_{scale_set}.csv"


def store_benchmark_metrics_summary(
        field_type: str, category: str, scale_set: str, header: Sequence[str], metrics: Float[Tensor, "scales metrics"]
) -> None:
    _store_metrics(_mk_parent(_results_dir / "metrics" / f"{field_type}_{category}_{scale_set}.csv"), header, metrics)


def _store_metrics(file, header, metrics):
    np.savetxt(file, metrics, fmt="%.10e", delimiter=",", header=",".join(header), comments="")


def case_dir(field_type: str, category: str, name: str) -> Path:
    return _results_dir / field_type / category / name


def load_neural_field(field_type: str, category: str, name: str) -> SmoothableNeuralField:
    sig = signature(category)
    if category == "lightstage":
        field = Smoothable4x1024LightStageNeuralField()
    else:
        if field_type == "neural_ablation_spectralnormalization":
            field_class = SpectralNormalizationSmoothable4x1024NeuralField
        elif field_type in ("neural_ablation_nolipschitz", "neural_ablation_nolipschitznoscaling"):
            field_class = UnconstrainedSmoothable4x1024NeuralField
        else:
            field_class = Smoothable4x1024NeuralField
        field = field_class(sig.coords, sig.channels, sig.anisotropic, sig.smoothable_comps)
    field.load_state_dict(torch.load(_neural_field_file(field_type, category, name, field)), strict=False)
    return field


def store_neural_field(field_type: str, category: str, name: str, field: SmoothableNeuralField):
    torch.save(field.state_dict(), _mk_parent(_neural_field_file(field_type, category, name, field)))


def _neural_field_file(field_type, category, name, field):
    return case_dir(field_type, category, name) / f"{type(field).__name__}.pt"


def performance_dir() -> Path:
    return _results_dir / "performance"


def visualizations_dir() -> Path:
    return _results_dir / "visualizations"


def _mk_parent(file):
    file.parent.mkdir(parents=True, exist_ok=True)
    return file
