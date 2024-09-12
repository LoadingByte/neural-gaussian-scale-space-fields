import sys
from io import BytesIO

from cv2 import VideoWriter
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import multivariate_normal
from skimage.transform import resize
from tqdm import tqdm, trange
from matplotlib.patheffects import withStroke

import ngssf


def picture_isotropic():
    _picture(
        ngssf.results.visualizations_dir() / "picture_isotropic",
        "variance_benchmark",
        list(range(len(ngssf.data.benchmark_variances())))
    )


def picture_anisotropic():
    _picture(
        ngssf.results.visualizations_dir() / "picture_anisotropic",
        "covariance_matrix_benchmark",
        [87, 99]
    )


def _picture(base_dir, scale_set, indices):
    for name in tqdm(args_or([
        "bbq", "cliffs", "colosseo", "crystals", "firenze", "firewood", "mutter", "peak", "portal", "rue",
        "schaumbrunnen", "steepshore", "toomuchbleach", "tunnelrampe", "zebras"
    ]), desc="name", leave=False):
        for index in tqdm(indices, desc="scale", leave=False):
            gauss_img = _prepare_picture_image(ngssf.results.load_benchmark("gauss", "picture", name, scale_set, index))
            pred_img = _prepare_picture_image(ngssf.results.load_benchmark("neural", "picture", name, scale_set, index))
            error_img = plt.cm.hot(np.mean(np.abs(pred_img - gauss_img), axis=2))[:, :, :3]
            ones = np.ones(gauss_img.shape[:2])
            comp_img = np.rot90(np.triu(ones))[:, :, None] * pred_img + np.rot90(np.tril(ones))[:, :, None] * error_img
            _write_image(base_dir / name / f"{index}.jpg", comp_img)


def picture_foveation():
    name = arg_or("squirrel")
    field = ngssf.results.load_neural_field("neural", "picture", name).cuda()
    res = 512
    X = ngssf.util.grid_coords(res, 2, device="cuda")
    with torch.no_grad():
        pred_img = field(X, (X.norm(dim=1) - 0.35).clamp(0) ** 3 / 200).T.reshape(3, res, res)
    img = _prepare_picture_image(pred_img)
    _write_image(ngssf.results.visualizations_dir() / "picture_foveation" / f"{name}.jpg", img)


def _prepare_picture_image(img):
    return _prepare_image(ngssf.util.convert_color_space(img, "ProPhotoRGB", "sRGB"))


def mesh_isotropic():
    scale_set = "variance_benchmark"
    for name in tqdm(args_or(ngssf.data.names("mesh")), desc="name", leave=False):
        d = ngssf.results.visualizations_dir() / "mesh_isotropic"
        dg = d / name / "gauss"
        dn = d / name / "neural"
        dg.mkdir(parents=True, exist_ok=True)
        dn.mkdir(parents=True, exist_ok=True)
        for index in trange(4, desc="scale", leave=False):
            gauss_m = ngssf.util.mesh_from_grid(ngssf.results.load_benchmark("gauss", "mesh", name, scale_set, index))
            pred_m = ngssf.util.mesh_from_grid(ngssf.results.load_benchmark("neural", "mesh", name, scale_set, index))
            gauss_m.export(dg / f"{index}.ply")
            pred_m.export(dn / f"{index}.ply")


def mesh_anisotropic():
    name = arg_or("thai")
    d = ngssf.results.visualizations_dir() / "mesh_anisotropic" / name
    d.mkdir(parents=True, exist_ok=True)
    field = ngssf.results.load_neural_field("neural", "mesh", name).cuda()
    for label, variances in [
        ("isotropic", [1e-2, 1e-2, 1e-2]),
        ("anisotropic_horizontal", [1e-2, 1e-8, 1e-2]),
        ("anisotropic_vertical", [1e-8, 1e-2, 1e-8])
    ]:
        scale = torch.diag(torch.tensor(variances))
        with torch.no_grad():
            grid = ngssf.util.eval_grid(256, field, scale.cuda(), batch_size=2 ** 18).cpu()
        ngssf.util.mesh_from_grid(grid).export(d / f"{label}.ply")


def lightstage():
    name = arg_or("cute")
    light_positions = ngssf.data.lightstage_light_positions()
    field = ngssf.results.load_neural_field("neural", "lightstage", name).cuda()
    w, h = 512, 384
    X = torch.cat([
        torch.cartesian_prod(torch.linspace(-0.75, 0.75, h), torch.linspace(-1, 1, w)).flip(1),
        light_positions[22].tile(w * h, 1)
    ], dim=1)
    for i, scale in enumerate([0, 1]):
        with torch.no_grad():
            Y = field(X.cuda(), scale)
        img = _prepare_image(Y.T.reshape(3, h, w))
        _write_image(ngssf.results.visualizations_dir() / "lightstage" / name / f"{i}.jpg", img)


def picture_video():
    d = ngssf.results.visualizations_dir() / "picture_video"
    d.mkdir(parents=True, exist_ok=True)

    cov_mats = _interpolate_2d_covariance_matrices(
        np.array([0, 240, 300, 420, 540, 660, 780]),
        np.array([[0, -7, -7], [0, -1, -1], [0, -4, -1], [0.25, -4, -1], [0.5, -4, -2], [0.75, -7, -2], [1, -7, -7]])
    )

    spectrum_label = _label("Spectrum", 1300, (128, 32), overlay=True)
    cov_label = _label("Covariance", 1300, (128, 32))

    for name in tqdm(args_or(["bbq", "firewood", "schaumbrunnen", "tunnelrampe"]), leave=False):
        orig_picture = torch.as_tensor(resize(ngssf.data.load("picture", name).numpy(), (3, 512, 512)), device="cuda")
        neural_field = ngssf.results.load_neural_field("neural", "picture", name).cuda()
        gauss_field = ngssf.GaussianMonteCarloSmoothableField(ngssf.GridField(orig_picture, padding_mode="reflection"))

        video = VideoWriter(str(d / f"{name}.mp4"), VideoWriter.fourcc('a', 'v', 'c', '1'), 60, (1152, 512))
        for cov_mat in tqdm(cov_mats, leave=False):
            with torch.no_grad():
                neural_picture = ngssf.util.eval_grid(512, neural_field, cov_mat).cpu()
            gauss_picture = ngssf.util.eval_grid(512, gauss_field, cov_mat).cpu()
            frame = torch.ones(3, 512, 1152)
            frame[:, :, :512] = ngssf.util.convert_color_space(neural_picture, "ProPhotoRGB", "sRGB")
            frame[:, :, -512:] = ngssf.util.convert_color_space(gauss_picture, "ProPhotoRGB", "sRGB")
            frame[:, -128:, :128] = _spectrum(neural_picture)
            frame[:, -128:, -128:] = _spectrum(gauss_picture)
            frame[:, 128:-128, 512:-512] = _isolines(cov_mat, 0.5, (128, 256))
            _alpha_blend(frame[:, -160:-128, :128], spectrum_label)
            _alpha_blend(frame[:, -160:-128, -128:], spectrum_label)
            frame[:, 96:128, 512:-512] = cov_label
            video.write(np.flip((_prepare_image(frame) * 255).astype(np.uint8), axis=2))
        video.release()


def _spectrum(picture):
    spectrum = ngssf.util.image_spectrum(picture.mean(dim=0, keepdim=True))
    crop = (picture.shape[1] - 128) // 2
    return (spectrum[0, crop:-crop, crop:-crop].clamp(1e-5).log10() - 1.5).clamp(0) / 3.5 * 2 - 1


def mesh_video_objects():
    cov_mats = torch.tensor(_mesh_video_covariance_matrices(), dtype=torch.float32, device="cuda")

    for name in tqdm(args_or(ngssf.data.names("mesh")), leave=False):
        orig_mesh = ngssf.data.load("mesh", name)
        neural_field = ngssf.results.load_neural_field("neural", "mesh", name).cuda()
        gauss_field = ngssf.GaussianMonteCarloSmoothableField(
            ngssf.GridField(ngssf.util.eval_grid(1021, ngssf.SDFField(orig_mesh), bounds=2), bounds=2)
        ).cuda()

        d = ngssf.results.visualizations_dir() / "mesh_video" / name
        dn = d / "meshes_neural"
        dg = d / "meshes_gauss"
        dn.mkdir(parents=True, exist_ok=True)
        dg.mkdir(parents=True, exist_ok=True)

        for i, cov_mat in enumerate(tqdm(cov_mats, leave=False)):
            with torch.no_grad():
                neural_grid = ngssf.util.eval_grid(256, neural_field, cov_mat, batch_size=2 ** 18).cpu()
            neural_mesh = ngssf.util.mesh_from_grid(neural_grid)
            gauss_mesh = ngssf.util.mesh_from_grid(ngssf.util.eval_grid(256, gauss_field, cov_mat).cpu())
            neural_mesh.export(dn / f"{i:05d}.ply")
            gauss_mesh.export(dg / f"{i:05d}.ply")


def mesh_video_ellipsoids():
    d = ngssf.results.visualizations_dir() / "mesh_video" / "ellipsoids"
    d.mkdir(parents=True, exist_ok=True)

    X = np.linspace(-0.5, 0.5, 512)
    for i, cov_mat in enumerate(tqdm(_mesh_video_covariance_matrices(), leave=False)):
        Y = multivariate_normal(cov=cov_mat).pdf(np.stack(np.meshgrid(X, X, X), axis=-1))
        mesh = ngssf.util.mesh_from_grid(torch.as_tensor(-Y)[None], level=-0.5)
        mesh.export(d / f"{i:05d}.ply")


def _mesh_video_covariance_matrices():
    times = np.array([0, 180, 300, 420, 540])
    logvars = np.array([[-7, -7, -7], [-3, -3, -3], [-3, -7, -3], [-7, -3, -7], [-7, -7, -7]])
    frames = np.arange(times.max() + 1)
    cov_mats = np.zeros((len(frames), 3, 3))
    cov_mats[:, 0, 0] = 10 ** np.interp(frames, times, logvars[:, 0])
    cov_mats[:, 1, 1] = 10 ** np.interp(frames, times, logvars[:, 1])
    cov_mats[:, 2, 2] = 10 ** np.interp(frames, times, logvars[:, 2])
    return cov_mats


def lightstage_video():
    name = arg_or("cute")
    d = ngssf.results.visualizations_dir() / "lightstage_video"
    d.mkdir(parents=True, exist_ok=True)

    light_shots = ngssf.data.load("lightstage", name)
    light_pos = ngssf.data.lightstage_light_positions()
    neural_field = ngssf.results.load_neural_field("neural", "lightstage", name).cuda()
    neural_field.calibration_factors[1] = 500
    gauss_field = ngssf.GaussianMonteCarloSmoothableField(ngssf.LightStageField(light_shots, light_pos), {2, 3}).cuda()

    times = np.array([0, 180, 480, 660, 960])
    frames = np.arange(times.max() + 1)
    xs_light = torch.tensor(np.array([
        np.interp(frames, times, coord)
        for coord in np.array([light_pos[2], light_pos[22], light_pos[22], light_pos[2], light_pos[2]]).T
    ]).T, dtype=torch.float32, device="cuda")
    cov_mats = _interpolate_2d_covariance_matrices(
        times,
        np.array([[0, -6, -6], [0, -6, -6], [0, -2, -2], [0, -2, -2], [0, -6, -6]])
    )

    w, h = 512, 384
    X_pixel = torch.cartesian_prod(torch.linspace(-0.75, 0.75, h), torch.linspace(-1, 1, w)).flip(1).cuda()

    light_label = _label("Light Dir.", 1300, (128, 32))
    cov_label = _label("Covariance", 1300, (128, 32))
    plotted_pos = light_pos[(light_pos - (light_pos[2] + light_pos[22]) / 2).norm(dim=1) < 0.15]

    video = VideoWriter(str(d / f"{name}.mp4"), VideoWriter.fourcc('a', 'v', 'c', '1'), 60, (1152, 384))
    for x_light, cov_mat in tqdm(list(zip(xs_light, cov_mats)), leave=False):
        X = torch.cat([X_pixel, x_light.tile(w * h, 1)], dim=1).cuda()
        with torch.no_grad():
            neural_image = neural_field(X, cov_mat).T.reshape(3, h, w).cpu()
        gauss_image = gauss_field(X, cov_mat).T.reshape(3, h, w).cpu()
        frame = torch.ones(3, 384, 1152)
        frame[:, :, :512] = neural_image
        frame[:, :, -512:] = gauss_image
        frame[:, 48:176, 512:-512] = _light_position(plotted_pos, x_light, (128, 128))
        frame[:, 240:368, 512:-512] = _isolines(cov_mat, 0.75, (128, 128))
        frame[:, 16:48, 512:-512] = light_label
        frame[:, 208:240, 512:-512] = cov_label
        video.write(np.flip((_prepare_image(frame) * 255).astype(np.uint8), axis=2))
    video.release()


def _interpolate_2d_covariance_matrices(times, angles_and_logvars):
    frames = np.arange(times.max() + 1)
    angles = np.interp(frames, times, angles_and_logvars[:, 0])
    lv1 = np.interp(frames, times, angles_and_logvars[:, 1])
    lv2 = np.interp(frames, times, angles_and_logvars[:, 2])
    angles = angles * (2 * np.pi)
    rot_mats = np.moveaxis(np.array([[np.cos(angles), -np.sin(angles)], [np.sin(angles), np.cos(angles)]]), 2, 0)
    var_mats = np.moveaxis(np.array([[10 ** lv1, np.zeros(len(angles))], [np.zeros(len(angles)), 10 ** lv2]]), 2, 0)
    return torch.as_tensor(rot_mats @ var_mats @ np.swapaxes(rot_mats, 1, 2), dtype=torch.float32, device="cuda")


def _isolines(cov_mat, min_range, size):
    x_range = min_range * size[0] / min(size)
    y_range = min_range * size[1] / min(size)
    X = np.linspace(-x_range, x_range, 1024)
    Y = np.linspace(-y_range, y_range, 1024)
    Z = multivariate_normal(cov=cov_mat.numpy(force=True)).pdf(np.stack(np.meshgrid(X, Y), axis=-1))
    dpi = 50
    fig, ax = plt.subplots(figsize=np.array(size) / dpi, dpi=dpi)
    ax.contour(X, Y, Z, levels=np.linspace(0.5, Z.max(), 6))
    ax.invert_yaxis()
    ax.axis("equal")
    return _plot_to_tensor(fig, ax, dpi, False)


def _light_position(all_pos, cur_pos, size):
    dpi = 50
    fig, ax = plt.subplots(figsize=np.array(size) / dpi, dpi=dpi)
    plt.scatter(*all_pos.numpy(force=True).T, color="lightgray")
    plt.scatter(*cur_pos.numpy(force=True)[:, None], color="C0", s=dpi * 2)
    ax.axis("equal")
    return _plot_to_tensor(fig, ax, dpi, False)


def _label(text, fontsize, size, overlay=False):
    fig, ax = plt.subplots(figsize=size, dpi=1)
    ax.text(
        0.5, 0.5, text, fontsize=fontsize, horizontalalignment="center", verticalalignment="center_baseline",
        color="white" if overlay else "black",
        path_effects=[withStroke(linewidth=fontsize * 0.15, foreground="black")] if overlay else None
    )
    return _plot_to_tensor(fig, ax, 1, overlay)


def _plot_to_tensor(fig, ax, dpi, transparent):
    ax.axis("off")
    fig.tight_layout()
    with BytesIO() as buf:
        fig.savefig(buf, format="raw", dpi=dpi, transparent=transparent)
        plt.close()
        arr = np.frombuffer(buf.getvalue(), np.uint8).reshape((int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        tensor = torch.tensor(arr).permute(2, 0, 1) / 255
        tensor[:3] = tensor[:3] * 2 - 1
        return tensor if transparent else tensor[:3]


def _alpha_blend(surface, overlay):
    surface[:] = (1 - overlay[3]) * surface + overlay[3] * overlay[:3]


def _prepare_image(img):
    return ((img + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy(force=True)


def _write_image(file, img):
    file.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(file, (img * 255).astype(np.uint8), quality=90)


def arg_or(default):
    return sys.argv[2] if len(sys.argv) > 2 else default


def args_or(default):
    return sys.argv[2:] if len(sys.argv) > 2 else default


if __name__ == "__main__":
    globals()[sys.argv[1]]()
