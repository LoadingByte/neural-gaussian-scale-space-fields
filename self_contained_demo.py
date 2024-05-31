"""
This is simplified standalone script that overfits a Neural Gaussian Scale-Space Field on an image, and then produces
both an isotropically and an anisotropically smoothed version of the image.

Attention! This script serves solely as a compact implementation of the core concept, applied to dense n-D fields like
images. It differs from the full codebase used for the paper in the following ways:
- The number of training iterations is smaller (see scripts/train.py).
- Some minor tweaks specific to SDFs and light stage data are missing.
- The matrix exponential is not optimized, thus this script is slower.
"""

import math
from itertools import product
from pathlib import Path

import imageio.v3 as iio
import torch
import torch.nn.functional as F
from scipy.stats import ortho_group
from torch import nn
from torch.distributions import HalfNormal
from torch.quasirandom import SobolEngine
from tqdm import trange

device = "cuda"
coords = 2
channels = 3


def main():
    field = Smoothable4x1024NeuralField().to(device)

    original_grid = load_original_grid()
    rand_cov_matrices = random_covariance_matrices_with_lograndom_eigenvalues(-12, 2, 10_000_000)
    optim = torch.optim.Adam(field.parameters(), lr=5e-4)
    for _ in trange(2000, leave=False):
        X = (torch.rand(100_000, coords, device=device) - 0.5) * (2 * 1.2)
        Y = interpolate(original_grid, X)
        scales = rand_cov_matrices[torch.randint(rand_cov_matrices.shape[0], (X.shape[0],), device=device)]
        loss = (field(X, scales) - Y).square().sum() / X.shape[0]
        optim.zero_grad()
        loss.backward()
        optim.step()

    field.calibrate()

    store_grid("filt_iso.jpg", evaluate_grid(field, 1e-3, resolution=512))
    store_grid("filt_aniso.jpg", evaluate_grid(field, torch.tensor([[1e-2, 0], [0, 0]], device=device), resolution=512))


def load_original_grid():
    file = Path(__file__).parent / "self_contained_demo_bbq.jpg"
    return torch.as_tensor(iio.imread(file), dtype=torch.float32, device=device).permute(2, 0, 1) / (255 / 2) - 1


def random_covariance_matrices_with_lograndom_eigenvalues(log_eigval_0, log_eigval_1, batch):
    eigenvalues = 10 ** (torch.rand(batch, coords, device=device) * (log_eigval_1 - log_eigval_0) + log_eigval_0)
    eigenvectors = torch.as_tensor(ortho_group.rvs(coords, batch), dtype=eigenvalues.dtype, device=eigenvalues.device)
    return eigenvectors @ (eigenvalues[..., None] * eigenvectors.mT)


def interpolate(grid, X):
    if coords == 1:
        grid = grid[:, None, :]
        X = torch.hstack((X, torch.zeros_like(X)))
    grid = grid[None]
    Y = F.grid_sample(grid, X[(None,) * (grid.ndim - 2)], padding_mode="reflection", align_corners=True)
    return Y[(0, slice(None)) + (0,) * (grid.ndim - 3)].T


def evaluate_grid(field, scale, resolution):
    axis = torch.linspace(-1, 1, resolution, device=device)
    coord_list = axis[:, None] if coords == 1 else torch.cartesian_prod(*[axis] * coords).flip(1)
    with torch.no_grad():
        out_list = field(coord_list, scale)
    return out_list.T.reshape(out_list.shape[1], *[resolution] * coords)


def store_grid(file, grid):
    iio.imwrite(file, (((grid + 1) / 2).clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).numpy(force=True))


class Smoothable4x1024NeuralField(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoding = FourierEncoding(1024, length_distribution_variance=2000)
        self.network = nn.Sequential(
            MatrixExpSpectralLimLinear(1024, 1024),
            nn.ReLU(),
            MatrixExpSpectralLimLinear(1024, 1024),
            nn.ReLU(),
            MatrixExpSpectralLimLinear(1024, 1024),
            nn.ReLU(),
            NormalizedDotProducts(1024, channels)
        )
        self.register_buffer("calibration_factor", torch.ones(()))

    def forward(self, X, scales=None):
        scales, anisotropic = self._prepare_scales(X, scales)
        X = self.encoding(X)
        if scales is not None:
            scales = scales * self.calibration_factor
            if not anisotropic:
                exponent = (scales[..., None] * self.encoding.A.square().sum(dim=1)).sqrt()
            else:
                exponent = (self.encoding.A.T * (scales @ self.encoding.A.T)).sum(dim=-2).clamp(0).sqrt()
            X = X * (-exponent).exp().tile(2)
        return self.network(X)

    @staticmethod
    def _prepare_scales(X, scales):
        if scales is None:
            return None, False
        if not isinstance(scales, torch.Tensor):
            scales = torch.as_tensor(scales, dtype=X.dtype, device=X.device)
        if not scales.any():
            return None, False
        batch_shape = X.shape[:1]
        aniso_shape = (coords,) * 2
        if scales.shape == () or scales.shape == batch_shape:
            return scales, False
        if scales.shape == aniso_shape or scales.shape == batch_shape + aniso_shape:
            return scales, True
        raise ValueError(f"Scales shape {scales.shape} clashes with field type or X shape {X.shape}.")

    def calibrate(
            self,
            n_variances=16,
            log_variance_range=(-4.0, -1.0),
            n_mc_samples=2000,
            n_uncalibrated_scales=256,
            log_uncalibrated_scale_range=(-5.0, 2.0),
            n_points=64
    ):
        self.calibration_factor.fill_(1)

        variances = torch.logspace(*log_variance_range, n_variances, device=device)
        scales = torch.logspace(*log_uncalibrated_scale_range, n_uncalibrated_scales, device=device)
        X = torch.rand(n_points, coords, device=device) * 2 - 1

        # Create a batch of "covariance matrices" (to be used as scale input). Each matrix is diagonal.
        scale_matrices = torch.diag_embed(scales[:, None].tile(1, coords))
        # Prepare the standard deviations argument to be used when generating Gaussian noise.
        gauss_stds = variances.sqrt()[:, None].tile(n_mc_samples, coords)

        Y_ours = torch.empty((n_points, n_uncalibrated_scales, channels), device=device)
        Y_gauss = torch.empty((n_points, n_variances, channels), device=device)
        with torch.no_grad():
            for i, x in enumerate(X):
                Y_ours[i] = self(x.tile(n_uncalibrated_scales, 1), scale_matrices)
                pert = x.repeat(n_variances * n_mc_samples, 1) + torch.normal(0, gauss_stds)
                Y_gauss[i] = self(pert).reshape(n_mc_samples, n_variances, channels).mean(dim=0)

        # For each variance, select the scale that best matches it.
        m_scales = scales[(Y_ours[:, None, :, :] - Y_gauss[:, :, None, :]).square().mean(dim=(0, 3)).argmin(dim=1)]
        self.calibration_factor.fill_((m_scales / variances).pow(1 / n_variances).prod())


class FourierEncoding(nn.Module):
    """
    Cube-to-ball mapping from:
    J. A. Griepentrog, W. Höppner, H. C. Kaiser, J. Rehberg.
    "A bi-Lipschitz continuous, volume preserving map from the unit ball onto a cube." Note di Matematica, 28(1). 2008.
    """

    def __init__(self, embed, length_distribution_variance):
        super().__init__()
        while True:
            directions = SobolEngine(coords, scramble=True).draw(embed // 2) * 2 - 1
            if (directions != 0).any(dim=1).all():
                break
        self._cube_to_ball(directions)
        lengths_01 = directions.norm(dim=1, keepdim=True)
        directions /= lengths_01
        # This squaring makes lengths_01 uniformly distributed (instead of "triangularly").
        lengths_01.square_()
        self.register_buffer("A", directions * HalfNormal(math.sqrt(length_distribution_variance)).icdf(lengths_01))

    @staticmethod
    def _cube_to_ball(X):
        # Notice that we can ignore the special 0-point case because that's already avoided by the sampling code.
        N, D = X.shape
        for i, d in product(range(N), range(1, D)):
            xi_len_sq = X[i, :d].square().sum().item()
            eta = X[i, d].item()
            eta_sq = eta * eta
            not_in_cone = eta_sq <= xi_len_sq
            if d == 1:
                xi = X[i, 0].item()
                if not_in_cone:
                    a = (math.pi * eta) / (4 * xi)
                    X[i, 0] = xi * math.cos(a)
                    X[i, 1] = xi * math.sin(a)
                else:
                    a = (math.pi * xi) / (4 * eta)
                    X[i, 0] = eta * math.sin(a)
                    X[i, 1] = eta * math.cos(a)
            elif d == 2:
                if not_in_cone:
                    X[i, :2] *= math.sqrt(1 - (4 * eta_sq) / (9 * xi_len_sq))
                    X[i, 2] = (2 / 3) * eta
                else:
                    X[i, :2] *= math.sqrt(2 / 3 - xi_len_sq / (9 * eta_sq))
                    X[i, 2] = eta - xi_len_sq / (3 * eta)
            else:
                raise ValueError(f"The cube to ball mapping does not support {D}-d yet.")
        return X

    def forward(self, X):
        M = (X @ self.A.T) * (2 * torch.pi)
        return torch.cat([M.cos(), M.sin()], dim=-1)


class MatrixExpSpectralLimLinear(nn.Module):
    """
    Orthogonal matrix parameterization via matrix exponential from:
    M. Lezcano-Casado, D. Martınez-Rubio. "Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization
    of the orthogonal and unitary group." International Conference on Machine Learning (ICML). 2019.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.starting_left_singular_vectors = nn.Parameter(torch.empty((out_features, out_features)))
        self.starting_right_singular_vectors = nn.Parameter(torch.empty((in_features, in_features)))
        self.starting_singular_values = nn.Parameter(torch.empty(min(in_features, out_features)))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        self._initialize_matrix_for_exponential_orthogonalization(self.starting_left_singular_vectors)
        self._initialize_matrix_for_exponential_orthogonalization(self.starting_right_singular_vectors)
        nn.init.uniform_(self.starting_singular_values, -5, 5)
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _initialize_matrix_for_exponential_orthogonalization(M):
        with torch.no_grad():
            diag = torch.zeros(M.shape[0] - 1)
            diag[::2] = M.new(M.shape[0] // 2).uniform_(-torch.pi, torch.pi)
            M.set_(torch.diag(diag, diagonal=1))

    @property
    def weight(self):
        U = self._exponential_orthogonalization(self.starting_left_singular_vectors)[:, :self.in_features]
        V = self._exponential_orthogonalization(self.starting_right_singular_vectors)[:, :self.out_features]
        S = self.starting_singular_values.sigmoid()
        return (U * S) @ V.T

    @staticmethod
    def _exponential_orthogonalization(M):
        triu = M.triu(diagonal=1)
        return torch.matrix_exp(triu - triu.mT)

    def forward(self, X):
        return F.linear(X, self.weight, self.bias)


class NormalizedDotProducts(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.starting_weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.starting_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self):
        W = self.starting_weight
        return W * (1 / W.norm(dim=1, keepdim=True)).minimum(torch.ones((), dtype=W.dtype, device=W.device))

    def forward(self, X):
        return F.linear(X, self.weight, self.bias)


if __name__ == "__main__":
    main()
