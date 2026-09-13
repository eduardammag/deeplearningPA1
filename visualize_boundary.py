import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_pytorch import SyntheticSegmentationDataset
from unet import UNet


DATA_DIR = "data/synthetic"

CHECKPOINT_PATH = "checkpoints/unet_boundary.pth"

OUTPUT_DIR = "output_dir/boundary_examples"

SAMPLE_INDEX = 0

BOUNDARY_WIDTH = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = SyntheticSegmentationDataset(
    root_dir=DATA_DIR,
    boundary_width=BOUNDARY_WIDTH
)

sample = dataset[SAMPLE_INDEX]

image = sample["image"]

semantic_mask = sample["semantic_mask"]

instance_mask = sample["instance_mask"]

boundary_mask = sample["boundary_mask"]

image_np = image.squeeze(0).numpy()

semantic_np = semantic_mask.numpy()

instance_np = instance_mask.numpy()

boundary_np = boundary_mask.numpy()

unique_classes, counts = np.unique(
    boundary_np,
    return_counts=True
)

print("\nDistribuição das classes:")

for class_id, count in zip(
    unique_classes,
    counts
):

    if class_id == 0:
        class_name = "background"

    elif class_id == 1:
        class_name = "interior"

    elif class_id == 2:
        class_name = "boundary"

    else:
        class_name = "desconhecida"

    print(
        f"Classe {class_id} "
        f"({class_name}): "
        f"{count} pixels"
    )

model = UNet(
    in_channels=1,
    num_classes=3
)

prediction_np = None


if os.path.exists(CHECKPOINT_PATH):

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    with torch.no_grad():

        input_tensor = image.unsqueeze(0).to(DEVICE)

        logits = model(input_tensor)

        prediction = torch.argmax(
            logits,
            dim=1
        )

        prediction_np = (
            prediction
            .squeeze(0)
            .cpu()
            .numpy()
        )

    print(
        f"\nCheckpoint carregado: "
        f"{CHECKPOINT_PATH}"
    )

else:

    print(
        "\nCheckpoint não encontrado."
    )

    print(
        "Será exibido apenas o ground truth."
    )

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

if prediction_np is not None:

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4)
    )

else:

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

axes[0].imshow(
    image_np,
    cmap="gray"
)

axes[0].set_title(
    "Imagem"
)

axes[0].axis("off")

axes[1].imshow(
    instance_np
)

axes[1].set_title(
    "Instance mask"
)

axes[1].axis("off")

axes[2].imshow(
    semantic_np,
    cmap="gray"
)

axes[2].set_title(
    "Semantic mask"
)

axes[2].axis("off")

axes[3].imshow(
    boundary_np,
    cmap="viridis",
    vmin=0,
    vmax=2
)

axes[3].set_title(
    "Boundary GT"
)

axes[3].axis("off")

if prediction_np is not None:

    axes[4].imshow(
        prediction_np,
        cmap="viridis",
        vmin=0,
        vmax=2
    )

    axes[4].set_title(
        "Boundary prediction"
    )

    axes[4].axis("off")

fig.suptitle(
    "Trilha A — Interior + Boundary",
    fontsize=14
)


plt.tight_layout()

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

print(f"\nFigura salva em: {output_path}")