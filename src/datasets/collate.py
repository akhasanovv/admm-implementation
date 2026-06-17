import torch


def collate_fn(dataset_items: list[dict]):
    """
    items to batch
    """

    result_batch = {}

    tensor_keys = ("lensless", "lensed", "psf")
    for key in tensor_keys:
        values = [elem.get(key) for elem in dataset_items]
        if all(value is not None for value in values):
            result_batch[key] = torch.stack(values)

    result_batch["image_id"] = [
        elem.get("image_id", str(index)) for index, elem in enumerate(dataset_items)
    ]

    return result_batch
