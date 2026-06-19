from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_to_tensor(image):
    image = load_image(image)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_image(image):
    if isinstance(image, (str, Path)):
        return Image.open(image)
    return image


def hwc_to_chw(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    return tensor.float()


class PSFMixin:
    def __init__(self, use_preprocessor=True, cache_psf=True, psf_mode="simulate"):
        self.use_preprocessor = use_preprocessor
        self.cache_psf = cache_psf
        self.psf_mode = psf_mode
        self._psf_cache = {}

    def mask_to_psf(self, mask_key, mask_vals):
        if self.cache_psf and mask_key in self._psf_cache:
            return self._psf_cache[mask_key].clone()

        if self.psf_mode == "mask":
            psf = torch.from_numpy(mask_vals).float()
            if psf.ndim == 2:
                psf = psf.unsqueeze(0).repeat(3, 1, 1)
            return hwc_to_chw(psf)

        try:
            from src.lensless_helpers.psf import simulate_psf_from_mask
        except ModuleNotFoundError as error:
            if error.name == "waveprop":
                raise ModuleNotFoundError(
                    "Install waveprop or use psf_mode=mask."
                ) from error
            raise

        psf = hwc_to_chw(simulate_psf_from_mask(mask_vals))
        if self.cache_psf:
            self._psf_cache[mask_key] = psf
        return psf.clone()

    def build_sample(self, lensless, lensed, mask_vals, image_id, mask_key):
        if self.use_preprocessor and lensed is not None and self.psf_mode == "simulate":
            from src.lensless_helpers.preprocessor import get_dataset_object

            lensless = load_image(lensless)
            lensed = load_image(lensed)
            lensed, lensless, psf, roi = get_dataset_object(lensed, lensless, mask_vals)
            sample = {
                "lensless": hwc_to_chw(lensless),
                "lensed": hwc_to_chw(lensed),
                "psf": hwc_to_chw(psf),
                "roi": roi,
                "image_id": image_id,
            }
        else:
            sample = {
                "lensless": image_to_tensor(lensless),
                "psf": self.mask_to_psf(mask_key, mask_vals),
                "image_id": image_id,
            }
            if lensed is not None:
                sample["lensed"] = image_to_tensor(lensed)
        return sample


class DigiCamMirflickrDataset(PSFMixin, Dataset):
    def __init__(
        self,
        split,
        repo_id="bezzam/DigiCam-Mirflickr-MultiMask-10K",
        limit=None,
        use_preprocessor=True,
        cache_psf=True,
        psf_mode="simulate",
        *args,
        **kwargs,
    ):
        PSFMixin.__init__(self, use_preprocessor, cache_psf, psf_mode)

        from datasets import load_dataset

        self.repo_id = repo_id
        self.dataset = load_dataset(repo_id, split=split)
        if limit is not None:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        mask_label = row["mask_label"]
        mask_vals = np.load(self._download_mask(mask_label))
        return self.build_sample(
            lensless=row["lensless"],
            lensed=row.get("lensed"),
            mask_vals=mask_vals,
            image_id=str(row.get("image_id", f"{index:06d}")),
            mask_key=str(mask_label),
        )

    def _download_mask(self, mask_label):
        from huggingface_hub import hf_hub_download

        mask_name = str(mask_label)
        if not mask_name.endswith(".npy"):
            mask_name = (
                mask_name if mask_name.startswith("mask_") else f"mask_{mask_name}"
            )
            mask_name = f"{mask_name}.npy"
        return hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=f"masks/{mask_name}",
        )


class CustomDirDataset(PSFMixin, Dataset):
    def __init__(
        self,
        data_dir,
        use_preprocessor=True,
        cache_psf=True,
        psf_mode="simulate",
        limit=None,
        *args,
        **kwargs,
    ):
        PSFMixin.__init__(self, use_preprocessor, cache_psf, psf_mode)
        self.data_dir = Path(data_dir)
        self.lensless_dir = self.data_dir / "lensless"
        self.lensed_dir = self.data_dir / "lensed"
        self.masks_dir = self.data_dir / "masks"
        self.items = self._build_index()
        if limit is not None:
            self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        mask_vals = np.load(item["mask"])
        return self.build_sample(
            lensless=item["lensless"],
            lensed=item.get("lensed"),
            mask_vals=mask_vals,
            image_id=item["image_id"],
            mask_key=item["image_id"],
        )

    def _build_index(self):
        if not self.lensless_dir.is_dir():
            raise FileNotFoundError(self.lensless_dir)
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(self.masks_dir)

        items = []
        for lensless_path in sorted(self.lensless_dir.iterdir()):
            if lensless_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_id = lensless_path.stem
            mask_path = self.masks_dir / f"{image_id}.npy"
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)

            item = {
                "image_id": image_id,
                "lensless": lensless_path,
                "mask": mask_path,
            }
            lensed_path = self._find_lensed(image_id)
            if lensed_path is not None:
                item["lensed"] = lensed_path
            items.append(item)
        return items

    def _find_lensed(self, image_id):
        if not self.lensed_dir.is_dir():
            return None
        for extension in IMAGE_EXTENSIONS:
            path = self.lensed_dir / f"{image_id}{extension}"
            if path.exists():
                return path
        return None
