import time
import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.utils.hydra_compat import patch_hydra_argparse
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)
patch_hydra_argparse()


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    set_random_seed(config.inferencer.seed)

    device = config.inferencer.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloaders, batch_transforms = get_dataloaders(config, device)
    model = instantiate(config.model).to(device).eval()

    checkpoint_path = config.inferencer.get("from_pretrained")
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    warmup = config.get("benchmark", {}).get("warmup", 2)
    max_batches = config.get("benchmark", {}).get("max_batches", 20)

    total_time = 0.0
    total_images = 0
    with torch.no_grad():
        for _, dataloader in dataloaders.items():
            for batch_idx, batch in enumerate(dataloader):
                for key in config.inferencer.device_tensors:
                    if key in batch:
                        batch[key] = batch[key].to(device)

                transforms = batch_transforms.get("inference")
                if transforms is not None:
                    for key, transform in transforms.items():
                        batch[key] = transform(batch[key])

                sync(device)
                start = time.perf_counter()
                model(**batch)
                sync(device)

                if batch_idx >= warmup:
                    total_time += time.perf_counter() - start
                    total_images += batch["lensless"].shape[0]
                if batch_idx + 1 >= warmup + max_batches:
                    break
            break

    print(f"images: {total_images}")
    print(f"time_sec: {total_time:.6f}")
    print(f"ms_per_image: {1000 * total_time / total_images:.3f}")
    print(f"images_per_sec: {total_images / total_time:.3f}")


if __name__ == "__main__":
    main()
