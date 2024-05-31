import math
import sys

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange

import ngssf


def process(field_type, category, name):
    field_kwargs = dict(fourier_encoding_kwargs={})
    loss_func = torch.square
    call_normalization_step = False

    sig = ngssf.data.signature(category)

    if sig.anisotropic:
        scaler = ngssf.MinibatchScaler(10_000_000, ngssf.RandomScaler(True, len(sig.smoothable_comps))).cuda()
    else:
        scaler = ngssf.RandomScaler(False).cuda()

    if category in ("picture", "textured"):
        field_class = ngssf.nn.prefab.Smoothable4x1024NeuralField
        sampler = ngssf.FieldSampler(ngssf.GridField(ngssf.data.load(category, name), padding_mode="reflection")).cuda()
        n_iters = 100_000
        n_samples = 100_000
        lr = 5e-4
        field_kwargs["fourier_encoding_kwargs"]["length_distribution_param"] = 2000
    elif category == "mesh":
        field_class = ngssf.nn.prefab.Smoothable4x1024NeuralField
        sampler = ngssf.MinibatchSampler(2 ** 30, ngssf.SDFSampler(ngssf.data.load(category, name))).cuda()
        n_iters = 50_000
        n_samples = 200_000
        lr = 5e-4
        field_kwargs["fourier_encoding_kwargs"]["length_distribution_param"] = 100
    elif category == "lightstage":
        field_class = ngssf.nn.prefab.Smoothable4x1024LightStageNeuralField
        n_iters = 100_000
        n_samples = 200_000
        lr = 1e-4
        light_positions = ngssf.data.lightstage_light_positions()
        sampler = ngssf.LightStageFieldSampler(
            ngssf.LightStageField(ngssf.data.load("lightstage", name), light_positions),
            [1, 0.75], light_positions[22], 0.15
        ).cuda()
        field_kwargs = dict(
            img_pos_fourier_encoding_kwargs=dict(length_distribution_param=500),
            light_pos_fourier_encoding_kwargs=dict(length_distribution_param=500)
        )
    elif category == "testfunc":
        if name != "ackley":
            raise ValueError(f"Unknown test function: {name}")
        field_class = ngssf.nn.prefab.Smoothable4x1024NeuralField
        sampler = ngssf.FieldSampler(ngssf.AckleyField()).cuda()
        field_kwargs["fourier_encoding_kwargs"]["length_distribution_param"] = 50
        n_iters = 50_000
        n_samples = 100_000
        lr = 5e-4

    # ABLATIONS START
    if field_type == "neural_ablation_maeloss":
        loss_func = torch.abs
    elif field_type == "neural_ablation_spectralnormalization":
        field_class = ngssf.nn.prefab.SpectralNormalizationSmoothable4x1024NeuralField
        call_normalization_step = True
    elif field_type == "neural_ablation_looselipschitz":
        field_kwargs["fourier_encoding_kwargs"]["amplitude"] = 10
    elif field_type == "neural_ablation_nolipschitz":
        field_class = ngssf.nn.prefab.UnconstrainedSmoothable4x1024NeuralField
    elif field_type == "neural_ablation_nolipschitznoscaling":
        field_class = ngssf.nn.prefab.UnconstrainedSmoothable4x1024NeuralField
        scaler = lambda _: None
    elif field_type == "neural_ablation_whitenoisefreqs":
        field_kwargs["fourier_encoding_kwargs"]["noise"] = "white"
    elif field_type == "neural_ablation_uniformfreqlens":
        field_kwargs["fourier_encoding_kwargs"]["length_distribution"] = "uniform"
        field_kwargs["fourier_encoding_kwargs"]["length_distribution_param"] = \
            2 * math.sqrt(field_kwargs["fourier_encoding_kwargs"]["length_distribution_param"])
    # ABLATIONS END

    field = field_class(sig.coords, sig.channels, sig.anisotropic, sig.smoothable_comps, **field_kwargs).cuda()

    optim = torch.optim.Adam(field.parameters(), lr=lr)
    losses = []
    for _ in trange(n_iters, desc="train", leave=False):
        X, Y = sampler(n_samples)
        scales = scaler(n_samples)
        loss = loss_func(field(X, scales) - Y).sum() / n_samples
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
        if call_normalization_step:
            field.normalization_step()
        del X, Y, scales, loss
    del optim

    ngssf.results.store_neural_field(field_type, category, name, field)
    out_dir = ngssf.results.case_dir(field_type, category, name)

    torch.save(torch.tensor(losses), out_dir / "losses.pt")

    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax2 = ax1.twinx()
    ax2.semilogy()
    ax1.plot(losses, color="C0")
    ax2.plot(losses, color="C1")
    ax1.tick_params("y", labelcolor="C0")
    ax2.tick_params("y", labelcolor="C1")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(out_dir / "losses.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    field_type, category = sys.argv[1:3]
    if len(sys.argv) >= 4:
        process(field_type, category, sys.argv[3])
    else:
        for name in tqdm(ngssf.data.names(category), desc=category, leave=False):
            process(field_type, category, name)
