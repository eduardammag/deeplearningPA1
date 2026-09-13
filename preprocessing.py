import numpy as np
import torch
from scipy import ndimage


def normalize_image(image):

    image = image.float()

    if image.max() > 1.0:
        image = image / 255.0

    return image


def create_boundary_mask(instance_mask, boundary_width=1):

    if isinstance(instance_mask, torch.Tensor):
        instance_mask = instance_mask.detach().cpu().numpy()

    instance_mask = np.asarray(instance_mask)

    if instance_mask.ndim != 2:
        raise ValueError(
            "instance_mask deve possuir dimensão [H, W]."
        )

    if boundary_width < 1:
        raise ValueError(
            "boundary_width deve ser maior ou igual a 1."
        )

    boundary_mask = np.zeros(
        instance_mask.shape,
        dtype=np.uint8
    )

    instance_ids = np.unique(instance_mask)

    instance_ids = instance_ids[instance_ids != 0]

    for instance_id in instance_ids:

        instance = instance_mask == instance_id

        interior = ndimage.binary_erosion(
            instance,
            iterations=boundary_width,
            border_value=0
        )

        boundary = instance & ~interior

        boundary_mask[interior] = 1
        boundary_mask[boundary] = 2

    return boundary_mask