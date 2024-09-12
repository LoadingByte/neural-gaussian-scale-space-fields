import sys
from itertools import count
from time import time

import numpy as np
import torch
from scipy.spatial import KDTree
from tqdm import tqdm, trange

import ngssf


def main(field_type, category, stop_metric):
    stop_metric = float(stop_metric)

    # Warm-up
    time_method(field_type, category, perf_names(category)[0], 0, stop_metric)

    rows = [("var_bench_idx", "field_type", "training_time", "inference_time")]
    for var_bench_idx in tqdm([0, 1], desc="var_bench_idx", leave=False):
        tts = []
        its = []
        for name in tqdm(perf_names(category), desc="name", leave=False):
            for _ in range(1) if category == "picture" else trange(5, desc="repetition", leave=False):
                while (result := time_method(field_type, category, name, var_bench_idx, stop_metric)) is None:
                    pass
                tt, it = result
                tts.append(tt)
                its.append(it)
        rows.append((f"{var_bench_idx}", field_type, f"{np.mean(tts):.4f}", f"{np.mean(its):.4f}"))

    with open(ngssf.results.performance_dir() / f"timings_{field_type}_{category}_{stop_metric}.csv", "w") as f:
        f.write("\n".join(",".join(row) for row in rows))


def perf_names(category):
    if category == "picture":
        return ["apples", "bergsee", "cliffs", "lonelyroad", "mutter",
                "perlenkette", "relief", "rohre", "talkessel", "tunnel"]
    else:
        return ["armadillo", "dragon"]


def time_method(field_type, category, name, var_bench_idx, stop_metric):
    true_grid = None
    true_mesh = None
    if category == "picture":
        true_grid = ngssf.results.load_benchmark("gauss", category, name, "variance_benchmark", var_bench_idx).cuda()
        res = true_grid.shape[1]
    elif category == "mesh":
        if var_bench_idx == 0:
            true_mesh = ngssf.data.load(category, name)
        else:
            true_mesh = ngssf.util.mesh_from_grid(
                ngssf.results.load_benchmark("gauss", category, name, "variance_benchmark", var_bench_idx)
            )
        true_kdtree = KDTree(true_mesh.vertices)
        res = 256

    start_time = None
    inference_durations = []
    result = None

    def start_fn():
        nonlocal start_time
        start_time = cur_time()

    def loop_fn(itr, pred_fn):
        if (itr + 1) % 50 != 0:
            return False
        inference_start_time = cur_time()
        with torch.no_grad():
            pred_grid = pred_fn()
        inference_durations.append(cur_time() - inference_start_time)
        if category == "picture":
            metric = 10 * (4 / (pred_grid - true_grid).square().mean()).log10().item()
            done = metric >= stop_metric
        else:
            done = False
            try:
                pred_vertices = ngssf.util.mesh_from_grid(pred_grid.cpu()).vertices
                metric = 0.5 * np.mean(true_kdtree.query(pred_vertices)[0]) + \
                         0.5 * np.mean(KDTree(pred_vertices).query(true_mesh.vertices)[0])
                done = metric <= stop_metric
            except ValueError as e:
                if str(e) != "Surface level must be within volume data range.":
                    raise e
            except RuntimeError as e:
                if str(e) != "No surface found at the given iso value.":
                    raise e
        nonlocal start_time, result
        if done:
            result = inference_start_time - start_time, np.mean(inference_durations)
        else:
            start_time += cur_time() - inference_start_time
        return done

    train(field_type, category, true_grid, true_mesh, res, start_fn, loop_fn)

    return result


def train(field_type, category, true_grid, true_mesh, res, start_fn, loop_fn):
    sig = ngssf.data.signature(category)
    enc_kw = {}
    scaler = None
    if category == "picture":
        sampler = ngssf.FieldSampler(ngssf.GridField(true_grid, padding_mode="reflection"))
        enc_kw["length_distribution_param"] = 2000
        n_samples = 100_000
    elif category == "mesh":
        sampler = ngssf.MinibatchSampler(2 ** 24, ngssf.SDFSampler(true_mesh)).cuda()
        enc_kw["length_distribution_param"] = 100
        n_samples = 200_000
    if field_type == "neural":
        scaler = ngssf.MinibatchScaler(10_000_000, ngssf.RandomScaler(True, sig.coords)).cuda()
        field = ngssf.nn.prefab.Smoothable4x1024NeuralField(sig.coords, sig.channels, True, None, enc_kw)
    elif field_type == "vanilla":
        field = ngssf.nn.prefab.UnconstrainedSmoothable4x1024NeuralField(sig.coords, sig.channels, True, None, enc_kw)
    field.cuda()
    optim = torch.optim.Adam(field.parameters(), lr=5e-4)
    start_fn()
    for itr in tqdm(count(), leave=False):
        X, Y = sampler(n_samples)
        scales = None if scaler is None else scaler(n_samples)
        loss = (field(X, scales) - Y).square().sum() / n_samples
        if loop_fn(itr, lambda: ngssf.util.eval_grid(res, field, batch_size=2 ** 20)):
            break
        optim.zero_grad()
        loss.backward()
        optim.step()
        del X, Y, scales, loss


def cur_time():
    torch.cuda.synchronize()
    return time()


if __name__ == "__main__":
    main(*sys.argv[1:])
