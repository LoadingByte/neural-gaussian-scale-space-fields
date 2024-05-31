import sys

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import ngssf


def process(field_type, category, name):
    field = ngssf.results.load_neural_field(field_type, category, name).cuda()

    if category != "mesh":
        variances, scales = field.calibrate()
    else:
        variances, scales = field.calibrate(log_variance_range=(-2.0, -1.0), point_distribution="near_zero_level_set")

    ngssf.results.store_neural_field(field_type, category, name, field)

    out_dir = ngssf.results.case_dir(field_type, category, name)
    torch.save({"variances": variances, "scales": scales}, out_dir / "calibration_correspondences.pt")

    for i, sub_scales in enumerate(scales):
        plt.figure(figsize=(6, 3))
        plt.loglog()
        plt.scatter(variances.cpu(), sub_scales.cpu(), marker=".", color="black", zorder=3)
        plt.plot([0, field.calibration_factors[i].item()])
        plt.xlabel("Variance")
        plt.ylabel("Uncalibrated Scale")
        plt.savefig(out_dir / f"calibration_correspondences_{i}.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    field_type, category = sys.argv[1:3]
    if len(sys.argv) >= 4:
        process(field_type, category, sys.argv[3])
    else:
        for name in tqdm(ngssf.data.names(category), desc=category, leave=False):
            process(field_type, category, name)
