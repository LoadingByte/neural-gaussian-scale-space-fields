import math
from itertools import product

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions import HalfNormal
from torch.quasirandom import SobolEngine

from ngssf.util import FloatScalar, IntScalar


class FourierEncoding(nn.Module):
    """
    Random Fourier Features encoding from:
    M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. T. Barron,
    R. Ng. "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains."
    Advances in Neural Information Processing Systems (NeurIPS). 2020.

    Cube-to-ball mapping from:
    J. A. Griepentrog, W. Höppner, H. C. Kaiser, J. Rehberg.
    "A bi-Lipschitz continuous, volume preserving map from the unit ball onto a cube." Note di Matematica, 28(1). 2008.
    """

    def __init__(
            self,
            coords: IntScalar,
            embed: IntScalar,
            noise: str = "sobol",
            length_distribution: str = "folded_normal",
            length_distribution_param: FloatScalar = 20,
            amplitude: FloatScalar = 1
    ) -> None:
        super().__init__()

        half_embed = embed // 2
        if noise == "white":
            directions = F.normalize(torch.randn(half_embed, coords))
            lengths_01 = torch.rand(half_embed, 1)
        elif noise == "sobol":
            drawn = half_embed
            if coords > 3:
                # Just a heuristic on how many more samples we need for our rejection sampling approach.
                drawn *= 2 ** (coords - 1)
            while True:
                directions = SobolEngine(coords, scramble=True).draw(drawn) * 2 - 1
                if (directions != 0).any(dim=1).all():
                    break
            if coords <= 3:
                self._cube_to_ball(directions)
            else:
                directions = directions[directions.square().sum(dim=1) <= 1][:half_embed]
            lengths_01 = directions.norm(dim=1, keepdim=True)
            directions /= lengths_01
            # This squaring makes lengths_01 uniformly distributed (instead of "triangularly").
            lengths_01.square_()
        else:
            raise ValueError(f"Unknown noise: {noise}")

        if length_distribution == "uniform":
            A = directions * (lengths_01 * length_distribution_param)
        elif length_distribution == "folded_normal":
            A = directions * HalfNormal(math.sqrt(length_distribution_param)).icdf(lengths_01)
        else:
            raise ValueError(f"Unknown length distribution: {length_distribution}")

        self.register_buffer("A", A)
        self.register_buffer("amplitude", torch.tensor(amplitude))

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

    def forward(self, X: Float[Tensor, "*batch coords"]) -> Float[Tensor, "*batch embedding"]:
        M = (X @ self.A.T) * (2 * torch.pi)
        return torch.cat([M.cos(), M.sin()], dim=-1) * self.amplitude

    def extra_repr(self) -> str:
        return f"coords={self.A.shape[1]}, embed={2 * self.A.shape[0]}, amplitude={self.amplitude.item()}"
