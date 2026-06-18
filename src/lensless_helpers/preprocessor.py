import torch
import warnings
import numpy as np
import cv2

from src.lensless_helpers.psf import simulate_psf_from_mask
from src.lensless_helpers.utils import resize

ALIGNMENT = {}
ALIGNMENT["top_left"] = (80, 100)
ALIGNMENT["height"] = 200
DISPLAY_RES = [900, 1200]
ORIGINAL_ASPECT_RATIO = DISPLAY_RES[1] / DISPLAY_RES[0]
ALIGNMENT["width"] = int(ALIGNMENT["height"] * ORIGINAL_ASPECT_RATIO)
CROPED_LENSED_SHAPE = (ALIGNMENT["height"], ALIGNMENT["width"], 3)


def force_rgb(image):
    if len(image.shape) == 2:
        warnings.warn(f"Converting image to RGB")
        image = np.stack([image] * 3, axis=2)
    elif len(image.shape) == 3:
        pass
    else:
        raise ValueError(f"Image should be 2D or 3D")
    return image


def convert_image_to_float(image):
    # convert to float
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255
    else:
        # 16 bit
        image = image.astype(np.float32) / 65535
    return image


def get_cropped_lensed(lensed):
    return resize(
        lensed, shape=CROPED_LENSED_SHAPE, interpolation=cv2.INTER_NEAREST
    )


def get_roi(image):
    return image[
        ALIGNMENT["top_left"][0] : ALIGNMENT["top_left"][0] + ALIGNMENT["height"],
        ALIGNMENT["top_left"][1] : ALIGNMENT["top_left"][1] + ALIGNMENT["width"],
    ]


def get_dataset_object(lensed, lensless, mask_vals):
    lensed = convert_image_to_float(force_rgb(np.array(lensed)))
    lensless = convert_image_to_float(force_rgb(np.array(lensless)))

    # lensless image is upside-down
    lensless = torch.rot90(torch.from_numpy(lensless), dims=(-3, -2), k=2)

    lensed = get_cropped_lensed(lensed)
    lensed = torch.from_numpy(lensed)

    psf = simulate_psf_from_mask(mask_vals)
    roi = torch.tensor(
        [
            ALIGNMENT["top_left"][0],
            ALIGNMENT["top_left"][1],
            ALIGNMENT["height"],
            ALIGNMENT["width"],
        ]
    )
    return lensed, lensless, psf, roi
