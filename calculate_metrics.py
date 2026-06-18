import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from PIL import Image
from torch.nn import functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms.functional import pil_to_tensor


def load_image(path):
    image = Image.open(path).convert("RGB")
    return pil_to_tensor(image).float().unsqueeze(0) / 255.0


def image_index(path):
    return {
        file.stem: file
        for file in sorted(path.iterdir())
        if file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gt_files = image_index(Path(args.gt_dir))
    pred_files = image_index(Path(args.pred_dir))
    ids = sorted(gt_files.keys() & pred_files.keys())
    if not ids:
        raise ValueError("No matching image ids.")

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = None
    if args.lpips:
        import lpips

        lpips_metric = lpips.LPIPS(net="vgg").to(device)

    values = {"MSE": [], "PSNR": [], "SSIM": [], "LPIPS": []}
    for image_id in ids:
        gt = load_image(gt_files[image_id]).to(device)
        pred = load_image(pred_files[image_id]).to(device)
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear")

        values["MSE"].append(F.mse_loss(pred, gt).item())
        values["PSNR"].append(psnr(pred, gt).item())
        values["SSIM"].append(ssim(pred, gt).item())
        if lpips_metric is not None:
            values["LPIPS"].append(lpips_metric(pred * 2 - 1, gt * 2 - 1).mean().item())

    for name, metric_values in values.items():
        if metric_values:
            print(f"{name}: {sum(metric_values) / len(metric_values):.6f}")
    print(f"Images: {len(ids)}")


if __name__ == "__main__":
    main()
