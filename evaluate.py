import torch
from torch.utils.data import DataLoader

from dataset_pytorch import SyntheticSegmentationDataset
from unet import UNet

from semantic_metrics import (
    dice_score,
    iou_score
)

from connected_components import connected_components

from instance_metrics import (
    mean_average_precision,
    count_error
)


DATA_DIR = "data/synthetic"
CHECKPOINT = "checkpoints/unet_baseline.pth"

BATCH_SIZE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Dataset
# ============================================================

dataset = SyntheticSegmentationDataset(
    DATA_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Modelo
# ============================================================

model = UNet(
    in_channels=1,
    num_classes=1
)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# Métricas
# ============================================================

dice_values = []
iou_values = []

all_predictions = []
all_ground_truths = []
all_probabilities = []

count_errors = []


# ============================================================
# Avaliação
# ============================================================

with torch.no_grad():

    for batch in loader:

        # ----------------------------------------------------
        # Imagem
        # ----------------------------------------------------

        images = batch["image"].to(DEVICE)

        # ----------------------------------------------------
        # Ground truth semântico
        # ----------------------------------------------------

        masks = batch["semantic_mask"].to(DEVICE)

        # [B, H, W] -> [B, 1, H, W]
        masks = masks.unsqueeze(1).float()

        # ----------------------------------------------------
        # Predição da rede
        # ----------------------------------------------------

        predictions = model(images)

        # ----------------------------------------------------
        # Métricas semânticas
        # ----------------------------------------------------

        dice = dice_score(
            predictions,
            masks
        )

        iou = iou_score(
            predictions,
            masks
        )

        dice_values.append(dice)
        iou_values.append(iou)

        # ----------------------------------------------------
        # Probabilidade de foreground
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            predictions
        )

        # ----------------------------------------------------
        # Transformar probabilidade em máscara binária
        # ----------------------------------------------------

        binary_predictions = (
            probabilities >= 0.5
        )

        # ----------------------------------------------------
        # Processar cada imagem do batch
        # ----------------------------------------------------

        for i in range(images.shape[0]):

            # Máscara binária da imagem i
            binary_mask = (
                binary_predictions[i, 0]
                .cpu()
                .numpy()
            )

            # Probabilidade de foreground da imagem i
            foreground_probability = (
                probabilities[i, 0]
                .cpu()
                .numpy()
            )

            # Ground truth de instâncias
            gt_instance_mask = (
                batch["instance_mask"][i]
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Connected Components
            # ------------------------------------------------

            pred_instance_mask, num_instances = (
                connected_components(
                    binary_mask
                )
            )

            # ------------------------------------------------
            # Guardar resultados
            # ------------------------------------------------

            all_predictions.append(
                pred_instance_mask
            )

            all_ground_truths.append(
                gt_instance_mask
            )

            all_probabilities.append(
                foreground_probability
            )

            # ------------------------------------------------
            # Erro absoluto na contagem
            # ------------------------------------------------

            error = count_error(
                pred_instance_mask,
                gt_instance_mask
            )

            count_errors.append(error)


# ============================================================
# mAP
# ============================================================

map_results = []

for (
    pred_instance_mask,
    gt_instance_mask,
    probability
) in zip(
    all_predictions,
    all_ground_truths,
    all_probabilities
):

    _, map_value = mean_average_precision(
        pred_instance_mask,
        gt_instance_mask,
        probability
    )

    map_results.append(
        map_value
    )


mean_map = (
    sum(map_results)
    / len(map_results)
)


# ============================================================
# Médias finais
# ============================================================

mean_dice = (
    sum(dice_values)
    / len(dice_values)
)

mean_iou = (
    sum(iou_values)
    / len(iou_values)
)

mean_count_error = (
    sum(count_errors)
    / len(count_errors)
)


# ============================================================
# Resultados
# ============================================================

print(
    f"Dice: {mean_dice:.4f}"
)

print(
    f"IoU:  {mean_iou:.4f}"
)

print(
    f"mAP:  {mean_map:.4f}"
)

print(
    f"Mean count error: "
    f"{mean_count_error:.4f}"
)
