import sys

import torch
from tqdm import tqdm

import ngssf


def process(field_type, category, scale_set):
    all_metrics = []
    for name in tqdm(ngssf.data.names(category), desc=category, leave=False):
        try:
            header, metrics = ngssf.results.load_benchmark_metrics(field_type, category, name, scale_set)
            all_metrics.append(metrics)
        except FileNotFoundError:
            break
    if len(all_metrics) == 0:
        return
    all_metrics = torch.stack(all_metrics)

    if scale_set == "covariance_matrix_benchmark":
        metrics_summary = all_metrics.nanmean(dim=(0, 1))[None]
    else:
        metrics_summary = all_metrics.mean(dim=0)
    ngssf.results.store_benchmark_metrics_summary(field_type, category, scale_set, header, metrics_summary)


if __name__ == "__main__":
    process(*sys.argv[1:])
