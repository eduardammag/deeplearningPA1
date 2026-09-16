import torch
import torch.nn as nn


def binary_cross_entropy_loss():

    return nn.BCEWithLogitsLoss()


def boundary_cross_entropy_loss(class_weights=None, device=None):

    if class_weights is not None:

        class_weights = torch.tensor(
            class_weights,
            dtype=torch.float32
        )

        if device is not None:
            class_weights = class_weights.to(device)

    return nn.CrossEntropyLoss(weight=class_weights)