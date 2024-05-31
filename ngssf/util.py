import math
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence, Union

import imageio.v3 as iio
import numpy as np
import scipy
import torch
from jaxtyping import Float, Int
from skimage.measure import marching_cubes
from torch import Tensor
from trimesh import Trimesh

IntScalar = Union[int, Int[np.ndarray, ""], Int[Tensor, ""]]
FloatScalar = Union[float, Float[np.ndarray, ""], Float[Tensor, ""], IntScalar]

IntSeq = Union[Sequence[IntScalar], Int[np.ndarray, "_"], Int[Tensor, "_"]]
FloatSeq = Union[Sequence[FloatScalar], Float[np.ndarray, "_"], Float[Tensor, "_"], IntSeq]


def grid_coords(
        resolution: IntScalar, coords: IntScalar, bounds: FloatScalar = 1.0, **kwargs
) -> Float[Tensor, "batch coords"]:
    axis = torch.linspace(-bounds, bounds, resolution, **kwargs)
    return axis[:, None] if coords == 1 else torch.cartesian_prod(*[axis] * coords).flip(1)


def eval_grid(
        resolution: IntScalar, field, *args, bounds: FloatScalar = 1.0, batch_size: IntScalar = -1
) -> Float[Tensor, "channels *dims"]:
    coord_list = grid_coords(resolution, field.coords, bounds, device=field.device)
    with torch.no_grad():
        if batch_size <= 0 or resolution ** field.coords <= batch_size:
            out_list = field(coord_list, *args)
        else:
            out_list = torch.cat([field(coord_chunk, *args) for coord_chunk in coord_list.split(batch_size)])
    return out_list.T.reshape(out_list.shape[1], *[resolution] * field.coords)


def gaussian_blur(
        grid: Float[Tensor, "channels *dims"], variance: FloatScalar, truncate: FloatScalar = 10
) -> Float[Tensor, "channels *dims"]:
    if variance <= 0:
        return grid
    # By multiplying sigma with half the grid's shape, the Gaussian kernel will be defined on [-1, 1].
    # The high truncation parameter is required to avoid artifacts when later computing a blurred image's spectrum.
    return torch.as_tensor(scipy.ndimage.gaussian_filter(
        grid.detach().numpy(), sigma=math.sqrt(variance) * (np.array(grid.shape[1:]) - 1) / 2,
        axes=range(1, grid.ndim), mode="reflect", truncate=truncate
    ))


def image_spectrum(image: Float[Tensor, "1 height width"]) -> Float[Tensor, "1 height width"]:
    img = image[0].numpy()
    v = np.zeros_like(img)
    v[0, :] = img[-1, :] - img[0, :]
    v[-1, :] = img[0, :] - img[-1, :]
    v[:, 0] += img[:, -1] - img[:, 0]
    v[:, -1] += img[:, 0] - img[:, -1]
    v_hat = np.fft.fftn(v)
    M, N = v_hat.shape
    q = np.arange(M).reshape(M, 1).astype(v_hat.dtype)
    r = np.arange(N).reshape(1, N).astype(v_hat.dtype)
    den = (2 * np.cos(np.divide((2 * np.pi * q), M)) + 2 * np.cos(np.divide((2 * np.pi * r), N)) - 4)
    s = np.divide(v_hat, den, out=np.zeros_like(v_hat), where=den != 0)
    s[0, 0] = 0
    smooth_component = np.real(np.fft.ifftn(s))
    magnitudes = np.abs(scipy.fft.fftshift(scipy.fft.fft2(img - smooth_component)))
    return torch.as_tensor(magnitudes[None])


def mesh_from_grid(grid: Float[Tensor, "1 depth height width"], level: FloatScalar = 0.0) -> Trimesh:
    volume = grid.detach().numpy()[0].T
    spacing = 2 / (torch.tensor(grid.shape[1:]) - 1)
    vertices, faces, normals, _ = marching_cubes(volume, level=level, spacing=spacing, allow_degenerate=False)
    return Trimesh(vertices - 1, faces, vertex_normals=-normals)


def convert_color_space(grid: Float[Tensor, "channels *dims"], src: str, dst: str) -> Float[Tensor, "channels *dims"]:
    icc_dir = Path(__file__).parent.parent / "data" / "icc_profiles"
    with TemporaryDirectory() as d:
        fi = Path(d) / "in.tiff"
        fo = Path(d) / "out.tiff"
        iio.imwrite(fi, (((grid + 1) / 2).clamp(0, 1).permute(1, 2, 0) * 65535).numpy(force=True).astype(np.uint16))
        subprocess.run([
            "convert", fi, "-profile", icc_dir / f"{src}.icc", "-profile", icc_dir / f"{dst}.icc", fo
        ], check=True)
        return torch.as_tensor(iio.imread(fo).astype(np.float32)).permute(2, 0, 1) / (65535 / 2) - 1
