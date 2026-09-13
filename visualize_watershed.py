import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_pytorch import SyntheticSegmentationDataset
from unet import UNet
from watershed import (
    watershed_from_logits,
)


# --------------------------------------------------
# Configurações
# --------------------------------------------------

DATA_DIR = "data/synthetic"

CHECKPOINT_PATH = (
    "checkpoints/unet_boundary.pth"
)

OUTPUT_DIR = (
    "output_dir/watershed_examples"
)

SAMPLE_INDEX = 0

BOUNDARY_WIDTH = 1

INTERIOR_THRESHOLD = 0.5
FOREGROUND_THRESHOLD = 0.5
MIN_MARKER_SIZE = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = SyntheticSegmentationDataset(
    root_dir=DATA_DIR,
    boundary_width=BOUNDARY_WIDTH
)

sample = dataset[SAMPLE_INDEX]

image = sample["image"]
instance_mask = sample["instance_mask"]
boundary_mask = sample["boundary_mask"]


image_np = image.squeeze(0).numpy()
instance_np = instance_mask.numpy()
boundary_np = boundary_mask.numpy()


# --------------------------------------------------
# Modelo
# --------------------------------------------------

model = UNet(
    in_channels=1,
    num_classes=3
)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# --------------------------------------------------
# Predição
# --------------------------------------------------

with torch.no_grad():

    input_tensor = (
        image
        .unsqueeze(0)
        .to(DEVICE)
    )

    logits = model(
        input_tensor
    )

    logits = logits.squeeze(0)

    probabilities = torch.softmax(
        logits,
        dim=0
    )

    prediction = torch.argmax(
        probabilities,
        dim=0
    )


prediction_np = (
    prediction
    .cpu()
    .numpy()
)

probabilities_np = (
    probabilities
    .cpu()
    .numpy()
)


# --------------------------------------------------
# Watershed
# --------------------------------------------------

(
    watershed_mask,
    markers,
    foreground_mask
) = watershed_from_logits(
    logits,
    interior_threshold=INTERIOR_THRESHOLD,
    foreground_threshold=FOREGROUND_THRESHOLD,
    min_marker_size=MIN_MARKER_SIZE
)


# --------------------------------------------------
# Informações
# --------------------------------------------------

number_gt_instances = len(
    np.unique(instance_np)
) - 1

number_markers = len(
    np.unique(markers)
) - 1

number_pred_instances = len(
    np.unique(watershed_mask)
) - 1


print()
print("========================================")
print("Watershed")
print("========================================")
print(
    f"Imagem: {SAMPLE_INDEX}"
)
print(
    f"Instâncias no ground truth: "
    f"{number_gt_instances}"
)
print(
    f"Marcadores encontrados: "
    f"{number_markers}"
)
print(
    f"Instâncias previstas: "
    f"{number_pred_instances}"
)
print(
    f"Interior threshold: "
    f"{INTERIOR_THRESHOLD}"
)
print(
    f"Foreground threshold: "
    f"{FOREGROUND_THRESHOLD}"
)
print(
    f"Minimum marker size: "
    f"{MIN_MARKER_SIZE}"
)
print("========================================")
print()


# --------------------------------------------------
# Visualização
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    4,
    figsize=(16, 8)
)


# Imagem
axes[0, 0].imshow(
    image_np,
    cmap="gray"
)

axes[0, 0].set_title(
    "Imagem"
)

axes[0, 0].axis("off")


# Ground truth
axes[0, 1].imshow(
    instance_np
)

axes[0, 1].set_title(
    "Ground truth — instâncias"
)

axes[0, 1].axis("off")


# Boundary GT
axes[0, 2].imshow(
    boundary_np,
    cmap="viridis",
    vmin=0,
    vmax=2
)

axes[0, 2].set_title(
    "Ground truth — boundary"
)

axes[0, 2].axis("off")


# Predição das classes
axes[0, 3].imshow(
    prediction_np,
    cmap="viridis",
    vmin=0,
    vmax=2
)

axes[0, 3].set_title(
    "Predição — 3 classes"
)

axes[0, 3].axis("off")


# Probabilidade do interior
axes[1, 0].imshow(
    probabilities_np[1],
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1, 0].set_title(
    "Probabilidade — interior"
)

axes[1, 0].axis("off")


# Marcadores
axes[1, 1].imshow(
    markers
)

axes[1, 1].set_title(
    f"Markers ({number_markers})"
)

axes[1, 1].axis("off")


# Foreground
axes[1, 2].imshow(
    foreground_mask,
    cmap="gray"
)

axes[1, 2].set_title(
    "Foreground"
)

axes[1, 2].axis("off")


# Watershed
axes[1, 3].imshow(
    watershed_mask
)

axes[1, 3].set_title(
    f"Watershed ({number_pred_instances})"
)

axes[1, 3].axis("off")


fig.suptitle(
    "Trilha A — Decodificação por Watershed",
    fontsize=16
)

plt.tight_layout()


# --------------------------------------------------
# Salvar
# --------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

output_path = os.path.join(
    OUTPUT_DIR,
    f"sample_{SAMPLE_INDEX}.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print(
    f"Figura salva em: {output_path}"
)