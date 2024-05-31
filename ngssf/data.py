from dataclasses import dataclass
from pathlib import Path
from typing import Union

import imageio.v3 as iio
import numpy as np
import torch
import trimesh
from jaxtyping import Float
from skimage.transform import resize
from torch import Tensor
from trimesh import Scene, Trimesh

from ngssf.util import IntScalar

_data_dir = Path(__file__).parent.parent / "data"

_Specimen = Union[Float[Tensor, "channels *dims"], Float[Tensor, "batch channels *dims"], Trimesh]


@dataclass
class Signature:
    coords: int
    channels: int
    anisotropic: bool
    smoothable_comps: set[int]


def signature(category: str) -> Signature:
    return _signatures[category]


_signatures = {
    "picture": Signature(coords=2, channels=3, anisotropic=True, smoothable_comps={0, 1}),
    "textured": Signature(coords=2, channels=3, anisotropic=True, smoothable_comps={0, 1}),
    "mesh": Signature(coords=3, channels=1, anisotropic=True, smoothable_comps={0, 1, 2}),
    "lightstage": Signature(coords=4, channels=3, anisotropic=True, smoothable_comps={2, 3}),
    "testfunc": Signature(coords=2, channels=1, anisotropic=False, smoothable_comps={0, 1})
}


def names(category: str) -> list[str]:
    if category == "testfunc":
        return ["ackley"]
    return sorted(f.stem for f in (_data_dir / category).iterdir() if f.name != "light_positions.pt")


def load(category: str, name: str) -> _Specimen:
    d = _data_dir / category
    if category == "picture":
        return torch.as_tensor(iio.imread(d / f"{name}.tiff").astype(np.float32)).permute(2, 0, 1) / (65535 / 2) - 1
    elif category == "textured":
        return torch.as_tensor(iio.imread(d / name / "texture.png").astype(np.float32)).permute(2, 0, 1) / (255 / 2) - 1
    elif category == "mesh":
        mesh = trimesh.load_mesh(d / f"{name}.ply")
        if isinstance(mesh, Scene):
            mesh = mesh.dump().sum()
        normalized_vertices = (mesh.vertices - mesh.bounding_box.centroid) * (2 / np.max(mesh.bounding_box.extents))
        return Trimesh(normalized_vertices, mesh.faces)
    elif category == "lightstage":
        return (_sRGB_oetf(_sRGB_eotf(torch.stack([
            torch.as_tensor(resize(iio.imread(f).astype(np.float32) / 255, (512, 512))).permute(2, 0, 1)
            for f in sorted((_data_dir / category / name).iterdir())
        ]) * 2) * 10) - 1).clamp(-1, 1)


def _sRGB_oetf(X):
    return torch.where(X <= 0.0031308, X * 12.92, X.pow(1 / 2.4) * 1.055 - 0.055)


def _sRGB_eotf(X):
    return torch.where(X <= 0.04045, X / 12.92, ((X + 0.055) / 1.055).pow(2.4))


def benchmark_variances() -> Float[Tensor, "batch"]:
    return torch.tensor([0.0, 0.0001, 0.001, 0.01, 0.1])


def benchmark_covariance_matrices(coords: IntScalar) -> Float[Tensor, "batch coords coords"]:
    return torch.load(_data_dir / "covariance_matrices" / f"{coords}d.pt")


def lightstage_light_positions():
    return torch.load(_data_dir / "lightstage" / "light_positions.pt")
