import torch
from torch.utils.data import DataLoader

from dataset_pytorch import SyntheticSegmentationDataset
from unet import UNet
from semantic_metrics import (
    dice_score,
    iou_score
)


DATA_DIR = "data/synthetic"
CHECKPOINT = "checkpoints/unet_baseline.pth"

BATCH_SIZE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Dataset
dataset = SyntheticSegmentationDataset(
    DATA_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Modelo
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


# Métricas
dice_values = []
iou_values = []


with torch.no_grad():

    for batch in loader:

        images = batch["image"].to(DEVICE)

        masks = batch["semantic_mask"].to(DEVICE)

        masks = masks.unsqueeze(1).float()

        predictions = model(images)

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


# Média
mean_dice = sum(dice_values) / len(dice_values)
mean_iou = sum(iou_values) / len(iou_values)


print(f"Dice: {mean_dice:.4f}")
print(f"IoU:  {mean_iou:.4f}")