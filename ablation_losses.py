import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()

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

    def forward(self, logits, target):
        return F.cross_entropy(
            logits,
            target,
            weight=self.class_weights
        )


class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0):
        super().__init__()

        self.gamma = gamma

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

    def forward(self, logits, target):

        log_prob = F.log_softmax(
            logits,
            dim=1
        )

        prob = torch.exp(log_prob)

        target_log_prob = log_prob.gather(
            1,
            target.unsqueeze(1)
        ).squeeze(1)

        target_prob = prob.gather(
            1,
            target.unsqueeze(1)
        ).squeeze(1)

        focal_factor = (
            1.0 - target_prob
        ).pow(self.gamma)

        loss = (
            -focal_factor *
            target_log_prob
        )

        if self.class_weights is not None:
            weights = self.class_weights[
                target
            ]
            loss = loss * weights

        return loss.mean()


def create_loss(loss_name, class_weights=None, gamma=0.0):

    if loss_name == "ce":

        return CrossEntropyLoss(
            class_weights=None
        )

    if loss_name == "balanced_ce":

        return CrossEntropyLoss(
            class_weights=class_weights
        )

    if loss_name == "focal":

        return FocalLoss(
            class_weights=None,
            gamma=gamma
        )

    if loss_name == "balanced_focal":

        return FocalLoss(
            class_weights=class_weights,
            gamma=gamma
        )

    raise ValueError(
        f"Loss desconhecida: {loss_name}"
    )
