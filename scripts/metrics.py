import math
import sys
from functools import cache

import numpy as np
import torch
from lpips import LPIPS
from scipy.spatial import KDTree
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

import ngssf


def process(field_type, category, name):
    process_scale_set(field_type, category, name, "variance_benchmark", ngssf.data.benchmark_variances())

    cov_max_vars = [
        torch.linalg.eigvalsh(M)[-1]
        for M in ngssf.data.benchmark_covariance_matrices(ngssf.data.signature(category).coords)
    ]
    process_scale_set(field_type, category, name, "covariance_matrix_benchmark", cov_max_vars)


def process_scale_set(field_type, category, name, scale_set, vars):
    if category == "picture":
        header = "psnr", "ssim", "lpips"
    elif category == "mesh":
        header = "mse", "iou", "chamfer"

    rows = []
    for i, var in enumerate(tqdm(vars, desc=scale_set, leave=False)):
        try:
            rows.append(process_specimen(field_type, category, name, scale_set, i, var))
        except FileNotFoundError:
            break

    if len(rows) != 0:
        ngssf.results.store_benchmark_metrics(field_type, category, name, scale_set, header, torch.tensor(rows))


def process_specimen(field_type, category, name, scale_set, index, var):
    candidate_grid = ngssf.results.load_benchmark(field_type, category, name, scale_set, index)
    gauss_grid = ngssf.results.load_benchmark("gauss", category, name, scale_set, index)

    crop = math.ceil(math.sqrt(var) * 3 * candidate_grid.shape[1] / 2)
    if crop > 0:
        crop_slicing = [slice(crop, -crop)] * ngssf.data.signature(category).coords
        candidate_grid = candidate_grid[:, *crop_slicing]
        gauss_grid = gauss_grid[:, *crop_slicing]

    if category == "picture":
        candidate_grid.clamp_(-1, 1)
        gauss_grid.clamp_(-1, 1)
        psnr = peak_signal_noise_ratio(gauss_grid.numpy(), candidate_grid.numpy(), data_range=2)
        ssim = structural_similarity(gauss_grid.numpy(), candidate_grid.numpy(), data_range=2, channel_axis=0)
        lpips = lpips_net()(candidate_grid.cuda(), gauss_grid.cuda()).item()
        return psnr, ssim, lpips
    elif category == "mesh":
        mse = (candidate_grid - gauss_grid).square().mean()
        candidate_bin = candidate_grid <= 0
        gauss_bin = gauss_grid <= 0
        iou = (candidate_bin & gauss_bin).count_nonzero() / ((candidate_bin | gauss_bin).count_nonzero() + 1e-8)
        try:
            c_vert = ngssf.util.mesh_from_grid(candidate_grid).vertices
            g_vert = ngssf.util.mesh_from_grid(gauss_grid).vertices
            chamfer = 0.5 * (np.mean(KDTree(g_vert).query(c_vert)[0]) + np.mean(KDTree(c_vert).query(g_vert)[0]))
        except ValueError as e:
            if str(e) != "Surface level must be within volume data range.":
                raise e
            chamfer = np.nan
        except RuntimeError as e:
            if str(e) != "No surface found at the given iso value.":
                raise e
            chamfer = np.nan
        return mse, iou, chamfer


@cache
def lpips_net():
    return LPIPS().cuda()


if __name__ == "__main__":
    field_type, category = sys.argv[1:3]
    if len(sys.argv) >= 4:
        process(field_type, category, sys.argv[3])
    else:
        for name in tqdm(ngssf.data.names(category), desc=category, leave=False):
            process(field_type, category, name)
