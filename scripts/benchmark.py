import sys

from tqdm import tqdm

import ngssf


def process(field_type, category, name):
    if category == "picture":
        resolution = 2048
    elif category == "mesh":
        resolution = 256

    if field_type.startswith("neural"):
        batch_size = 2 ** 18
        field = ngssf.results.load_neural_field(field_type, category, name).cuda()
    elif field_type == "gauss":
        batch_size = -1
        specimen = ngssf.data.load(category, name)
        if category == "picture":
            field = ngssf.GridField(specimen.cuda(), padding_mode="reflection")
        elif category == "mesh":
            # Resolution here is chosen s.t. -1 and 1 are part of the sampled coordinates.
            field = ngssf.GridField(ngssf.util.eval_grid(1021, ngssf.SDFField(specimen), bounds=2).cuda(), bounds=2)
        field = ngssf.GaussianMonteCarloSmoothableField(field)

    for i, var in enumerate(tqdm(ngssf.data.benchmark_variances(), desc="variance", leave=False)):
        benchmark = ngssf.util.eval_grid(resolution, field, var, batch_size=batch_size).cpu()
        ngssf.results.store_benchmark(field_type, category, name, "variance_benchmark", i, benchmark)
    if field.anisotropic:
        Ms = ngssf.data.benchmark_covariance_matrices(field.coords)
        for i, M in enumerate(tqdm(Ms.cuda(), desc="covariance_matrix", leave=False)):
            benchmark = ngssf.util.eval_grid(resolution, field, M, batch_size=batch_size).cpu()
            ngssf.results.store_benchmark(field_type, category, name, "covariance_matrix_benchmark", i, benchmark)


if __name__ == "__main__":
    field_type, category = sys.argv[1:3]
    if len(sys.argv) >= 4:
        process(field_type, category, sys.argv[3])
    else:
        for name in tqdm(ngssf.data.names(category), desc=category, leave=False):
            process(field_type, category, name)
