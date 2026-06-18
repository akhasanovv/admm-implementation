def crop_prediction(prediction, target, roi=None):
    if prediction.shape[-2:] == target.shape[-2:]:
        return prediction

    h, w = target.shape[-2:]
    if roi is not None:
        values = roi[:2] if roi.ndim == 1 else roi[0, :2]
        top, left = [int(v.item()) for v in values]
    else:
        top = (prediction.shape[-2] - h) // 2
        left = (prediction.shape[-1] - w) // 2

    return prediction[..., top : top + h, left : left + w]
