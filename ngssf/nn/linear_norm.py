import math

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from ngssf.util import IntScalar


class PowerIterSpectralLimLinear(nn.Linear):
    """
    Power iteration and divide_by_smax formulation from:
    T. Miyato, T. Kataoka, M. Koyama, Y. Yoshida. "Spectral Normalization for Generative Adversarial Networks."
    International Conference on Learning Representations (ICLR). 2018.
    """

    def __init__(self, *args, divide_by_smax: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.divide_by_smax = divide_by_smax
        self.register_buffer("u", torch.normal(torch.zeros(self.weight.shape[0], device=self.weight.device), 1))

    def normalization_step(self) -> None:
        W = self.weight
        u = self.u
        with torch.no_grad():
            v = F.normalize(W.T @ u, dim=0)
            u.set_(F.normalize(W @ v, dim=0))
            smax = torch.dot(u, W @ v)
            if smax > 1:
                if self.divide_by_smax:
                    W /= smax
                else:
                    W -= torch.outer((smax - 1) * u, v)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, divide_by_smax={self.divide_by_smax}"


class InlineNormalizedLinear(nn.Module):

    def __init__(self, in_features: IntScalar, out_features: IntScalar, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._register_weight_parameters()
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def _register_weight_parameters(self):
        self.starting_weight = nn.Parameter(torch.empty((self.out_features, self.in_features)))

    def reset_parameters(self) -> None:
        self._reset_weight_parameters()
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X: Float[Tensor, "*batch in"]) -> Float[Tensor, "*batch out"]:
        return F.linear(X, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class NormalizedDotProducts(InlineNormalizedLinear):

    def _reset_weight_parameters(self):
        nn.init.kaiming_uniform_(self.starting_weight, a=math.sqrt(5))

    @property
    def weight(self) -> Float[Tensor, "out in"]:
        W = self.starting_weight
        return W * (1 / W.norm(dim=1, keepdim=True)).minimum(torch.ones((), dtype=W.dtype, device=W.device))


class MatrixExpOrthogonalLinear(InlineNormalizedLinear):
    """
    Orthogonal matrix parameterization via matrix exponential from:
    M. Lezcano-Casado, D. Martınez-Rubio. "Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization
    of the orthogonal and unitary group." International Conference on Machine Learning (ICML). 2019.
    """

    def _register_weight_parameters(self):
        d = max(self.in_features, self.out_features)
        self.starting_weight = nn.Parameter(torch.empty((d, d)))

    def _reset_weight_parameters(self):
        _initialize_matrix_for_exponential_orthogonalization(self.starting_weight)

    @property
    def weight(self) -> Float[Tensor, "out in"]:
        weight_override = getattr(self, "_weight_override", None)
        if weight_override is not None:
            return weight_override
        return _exponential_orthogonalization(self.starting_weight)[:self.out_features, :self.in_features]


class MatrixExpSpectralLimLinear(InlineNormalizedLinear):
    """
    Orthogonal matrix parameterization via matrix exponential from:
    M. Lezcano-Casado, D. Martınez-Rubio. "Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization
    of the orthogonal and unitary group." International Conference on Machine Learning (ICML). 2019.
    """

    def _register_weight_parameters(self):
        self.starting_left_singular_vectors = nn.Parameter(torch.empty((self.out_features, self.out_features)))
        self.starting_right_singular_vectors = nn.Parameter(torch.empty((self.in_features, self.in_features)))
        self.starting_singular_values = nn.Parameter(torch.empty(min(self.in_features, self.out_features)))

    def _reset_weight_parameters(self):
        _initialize_matrix_for_exponential_orthogonalization(self.starting_left_singular_vectors)
        _initialize_matrix_for_exponential_orthogonalization(self.starting_right_singular_vectors)
        nn.init.uniform_(self.starting_singular_values, -5, 5)

    @property
    def weight(self) -> Float[Tensor, "out in"]:
        weight_override = getattr(self, "_weight_override", None)
        if weight_override is not None:
            return weight_override
        U = _exponential_orthogonalization(self.starting_left_singular_vectors)[:, :self.in_features]
        V = _exponential_orthogonalization(self.starting_right_singular_vectors)[:, :self.out_features]
        S = self.starting_singular_values.sigmoid()
        return (U * S) @ V.T


class MatrixExpOptimizer(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)

        modules_by_width = {}
        for module in self.modules():
            if isinstance(module, MatrixExpOptimizer) and module is not self:
                raise ValueError("MatrixExpOptimizer modules must not be nested.")
            elif (isinstance(module, (MatrixExpOrthogonalLinear, MatrixExpSpectralLimLinear)) and
                  module.in_features == module.out_features):
                modules_by_width.setdefault(module.in_features, []).append(module)
        if len(modules_by_width) == 0:
            raise ValueError("MatrixExpOptimizer does not contain any MatrixExp*Linear module.")

        self.orthogonal_modules = []
        self.spectral_lim_modules = []
        starting_Qs = []
        starting_Us = []
        starting_Vs = []
        starting_Ss = []
        for module in max(modules_by_width.values(), key=len):
            if isinstance(module, MatrixExpOrthogonalLinear):
                self.orthogonal_modules.append(module)
                starting_Qs.append(module.starting_weight)
                del module.starting_weight
            elif isinstance(module, MatrixExpSpectralLimLinear):
                self.spectral_lim_modules.append(module)
                starting_Us.append(module.starting_left_singular_vectors)
                starting_Vs.append(module.starting_right_singular_vectors)
                starting_Ss.append(module.starting_singular_values)
                del module.starting_left_singular_vectors
                del module.starting_right_singular_vectors
                del module.starting_singular_values
        self.Q_slice = slice(0, len(starting_Qs))
        self.U_slice = slice(self.Q_slice.stop, self.Q_slice.stop + len(starting_Us))
        self.V_slice = slice(self.U_slice.stop, self.U_slice.stop + len(starting_Vs))
        self.starting_orthogonal_matrices = nn.Parameter(torch.stack(starting_Qs + starting_Us + starting_Vs))
        if len(starting_Ss) != 0:
            self.starting_singular_values = nn.Parameter(torch.stack(starting_Ss))

        self.push_weights()

    def push_weights(self):
        orthogonal_matrices = _exponential_orthogonalization(self.starting_orthogonal_matrices)

        if len(self.orthogonal_modules) != 0:
            Qs = orthogonal_matrices[self.Q_slice]
            for module, Q in zip(self.orthogonal_modules, Qs):
                module._weight_override = Q

        if len(self.spectral_lim_modules) != 0:
            Us = orthogonal_matrices[self.U_slice]
            Vs = orthogonal_matrices[self.V_slice]
            Ss = self.starting_singular_values.sigmoid()
            Ws = (Us * Ss[:, None, :]) @ Vs.mT
            for module, W in zip(self.spectral_lim_modules, Ws):
                module._weight_override = W

    def forward(self, X: Float[Tensor, "*batch in"]) -> Float[Tensor, "*batch out"]:
        self.push_weights()
        return super().forward(X)


def _initialize_matrix_for_exponential_orthogonalization(M):
    with torch.no_grad():
        diag = torch.zeros(M.shape[0] - 1)
        diag[::2] = M.new(M.shape[0] // 2).uniform_(-torch.pi, torch.pi)
        M.set_(torch.diag(diag, diagonal=1))


def _exponential_orthogonalization(M):
    triu = M.triu(diagonal=1)
    return torch.matrix_exp(triu - triu.mT)
