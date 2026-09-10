import torch.nn as nn


def binary_cross_entropy_loss():
    """
    BCEWithLogitsLoss para segmentação binária.

    A função já combina:
    sigmoid + binary cross entropy.
    """
    return nn.BCEWithLogitsLoss()