import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import ngssf


def main(sequence_name, downsampling):
    downsampling = int(downsampling)

    base_dir = Path(__file__).parent.parent / "results" / "textured" / "fish" / sequence_name
    in_dir = base_dir / "uv"
    out_dir_singlesampled = base_dir / "rgb_singlesampled"
    out_dir_multisampled = base_dir / "rgb_multisampled"
    out_dir_ours = base_dir / "rgb_ours"
    out_dir_singlesampled.mkdir(parents=True, exist_ok=True)
    out_dir_multisampled.mkdir(parents=True, exist_ok=True)
    out_dir_ours.mkdir(parents=True, exist_ok=True)

    orig_field = ngssf.GridField(ngssf.util.gaussian_blur(ngssf.data.load("textured", "fish"), 2e-5, 4))
    ours_field = ngssf.results.load_neural_field("neural", "textured", "fish").cuda()

    for f in tqdm(list(in_dir.iterdir()), leave=False):
        in_img_high = cv2.cvtColor(cv2.imread(str(f), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        in_img_high = torch.as_tensor(in_img_high.astype(np.float32))
        in_img_high[:, :, :2] = in_img_high[:, :, :2] / (65535 / 2) - 1
        in_img_high[:, :, 1] *= -1
        in_img_high[:, :, 3] /= 65535

        in_img_low = in_img_high[downsampling // 2::downsampling, downsampling // 2::downsampling]

        out_img_singlesampled = lookup(in_img_low, orig_field)
        out_img_multisampled = downsample(lookup(in_img_high, orig_field), downsampling)

        in_img_hpad = F.pad(in_img_low, (0, 0, 1, 1, 0, 0))
        in_img_vpad = F.pad(in_img_low, (0, 0, 0, 0, 1, 1))
        delta = 2
        h1 = (in_img_hpad[:, 1:] - in_img_hpad[:, :-1]) / delta
        h2 = (in_img_hpad[:, 2:] - in_img_hpad[:, :-2]) / (2 * delta)
        v1 = (in_img_vpad[1:, :] - in_img_vpad[:-1, :]) / delta
        v2 = (in_img_vpad[2:, :] - in_img_vpad[:-2, :]) / (2 * delta)
        Jh = torch.where(h2[:, :, 3:] > 0, h1[:, 1:], torch.where(h2[:, :, 3:] < 0, h1[:, :-1], h2))[:, :, :2]
        Jv = torch.where(v2[:, :, 3:] > 0, v1[1:, :], torch.where(v2[:, :, 3:] < 0, v1[:-1, :], v2))[:, :, :2]
        J = torch.stack([Jh, Jv], dim=3)
        scales = J.mT @ J
        out_img_ours = lookup(in_img_low, ours_field, scales)

        write(out_dir_singlesampled / f.name, out_img_singlesampled)
        write(out_dir_multisampled / f.name, out_img_multisampled)
        write(out_dir_ours / f.name, out_img_ours)


def lookup(in_img, field, scales=None):
    mask = in_img[:, :, 3] != 0
    extra_args = [] if scales is None else [scales[mask].to(field.device)]
    with torch.no_grad():
        rgb = field(in_img[mask][:, :2].to(field.device), *extra_args).cpu()
    rgb = ((rgb + 1) * (255 / 2))
    out_img = torch.zeros((*in_img.shape[:2], 4))
    out_img[mask] = torch.column_stack([rgb, torch.full(rgb.shape[:1], 255)])
    return out_img


def downsample(img, downsampling):
    img = img.permute(2, 0, 1)
    img[:3] *= img[3]
    img = F.avg_pool2d(img, downsampling)
    img[:3] /= img[3].clamp(1e-8)
    return img.permute(1, 2, 0)


def write(f, out_img):
    out_img = F.interpolate(out_img.permute(2, 0, 1)[None], scale_factor=4, mode="nearest")[0].permute(1, 2, 0)
    iio.imwrite(str(f), out_img.clamp(0, 255).to(torch.uint8))


if __name__ == "__main__":
    main(*sys.argv[1:])
