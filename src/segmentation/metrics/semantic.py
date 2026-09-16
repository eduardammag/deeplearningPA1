import torch


def dice_score(prediction, target, threshold=0.5):
    """
    Calcula o Dice Score para segmentação binária.

    prediction: logits da rede [B, 1, H, W]
    target: máscara verdadeira [B, 1, H, W]
    """

    # Transformamos logits em probabilidades
    probability = torch.sigmoid(prediction)

    # Transformamos probabilidade em máscara binária
    prediction_binary = probability >= threshold

    target_binary = target >= 0.5

    # Convertemos para float
    prediction_binary = prediction_binary.float()
    target_binary = target_binary.float()

    # Interseção
    intersection = (
        prediction_binary * target_binary
    ).sum()

    # Dice
    dice = (
        2 * intersection
        / (
            prediction_binary.sum()
            + target_binary.sum()
            + 1e-8
        )
    )

    return dice.item()


def iou_score(prediction, target, threshold=0.5):
    """
    Calcula IoU (Intersection over Union)
    para segmentação binária.
    """

    probability = torch.sigmoid(prediction)

    prediction_binary = (
        probability >= threshold
    ).float()

    target_binary = (
        target >= 0.5
    ).float()

    intersection = (
        prediction_binary * target_binary
    ).sum()

    union = (
        prediction_binary
        + target_binary
        - prediction_binary * target_binary
    ).sum()

    iou = (
        intersection
        / (union + 1e-8)
    )

    return iou.item()