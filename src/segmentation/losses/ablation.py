import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(
        self,
        class_weights=None,
        gamma=2.0,
        reduction="mean"
    ):

        super().__init__()

        self.gamma = gamma
        self.reduction = reduction

        if class_weights is not None:

            self.register_buffer(
                "class_weights",
                torch.tensor(
                    class_weights,
                    dtype=torch.float32
                )
            )

        else:

            self.class_weights = None

    def forward(
        self,
        logits,
        target
    ):

        # ----------------------------------------------------
        # log p_t
        # ----------------------------------------------------

        log_probs = F.log_softmax(
            logits,
            dim=1
        )

        log_pt = log_probs.gather(
            1,
            target.unsqueeze(1)
        ).squeeze(1)

        pt = log_pt.exp()

        # ----------------------------------------------------
        # Cross-entropy por pixel
        # ----------------------------------------------------

        ce = -log_pt

        # ----------------------------------------------------
        # Focal factor
        # ----------------------------------------------------

        focal_factor = (
            1.0 - pt
        ).pow(self.gamma)

        loss = (
            focal_factor
            *
            ce
        )

        # ----------------------------------------------------
        # Class weighting
        # ----------------------------------------------------

        if self.class_weights is not None:

            weights = self.class_weights.to(
                logits.device
            )

            pixel_weights = weights[target]

            loss = (
                pixel_weights
                *
                loss
            )

        else:

            pixel_weights = None

        # ----------------------------------------------------
        # Reduction
        # ----------------------------------------------------

        if self.reduction == "none":

            return loss

        if self.reduction == "sum":

            return loss.sum()

        if self.reduction == "mean":

            # Para CE balanceada, o PyTorch normaliza
            # pela soma dos pesos dos pixels.
            if pixel_weights is not None:

                return (
                    loss.sum()
                    /
                    pixel_weights.sum().clamp_min(
                        1e-12
                    )
                )

            return loss.mean()

        raise ValueError(
            f"Unsupported reduction: {self.reduction}"
        )


def create_loss(
    loss_name,
    class_weights=None,
    gamma=2.0,
    device=None
):

    if loss_name == "ce":

        loss = nn.CrossEntropyLoss()

    elif loss_name == "balanced_ce":

        weights = torch.tensor(
            class_weights,
            dtype=torch.float32
        )

        if device is not None:

            weights = weights.to(device)

        loss = nn.CrossEntropyLoss(
            weight=weights
        )

    elif loss_name == "focal":

        loss = FocalLoss(
            class_weights=None,
            gamma=gamma
        )

    elif loss_name == "balanced_focal":

        loss = FocalLoss(
            class_weights=class_weights,
            gamma=gamma
        )

    else:

        raise ValueError(
            f"Unknown loss: {loss_name}"
        )

    if device is not None:

        loss = loss.to(device)

    return loss