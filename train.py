import os

import torch
from torch.utils.data import DataLoader

from dataset_pytorch import SyntheticSegmentationDataset
from unet import UNet
from cross_entropy import boundary_cross_entropy_loss

DATA_DIR = "data/synthetic"

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-3

BOUNDARY_WIDTH = 1

NUM_CLASSES = 3

CLASS_WEIGHTS = [
    0.5,  # background
    1.0,  # interior
    2.0,  # boundary
]

CHECKPOINT_PATH = "checkpoints/unet_boundary.pth"


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")
print(f"Dataset: {DATA_DIR}")
print(f"Boundary width: {BOUNDARY_WIDTH}")
print(f"Classes: {NUM_CLASSES}")
print(f"Class weights: {CLASS_WEIGHTS}")

dataset = SyntheticSegmentationDataset(
    root_dir=DATA_DIR,
    boundary_width=BOUNDARY_WIDTH
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


print(
    f"Quantidade de amostras: {len(dataset)}"
)

model = UNet(
    in_channels=1,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)

criterion = boundary_cross_entropy_loss(
    class_weights=CLASS_WEIGHTS,
    device=DEVICE
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch in loader:

        images = batch["image"].to(DEVICE)

        masks = batch["boundary_mask"].to(DEVICE)

        predictions = model(images)

        loss = criterion(
            predictions,
            masks
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )

os.makedirs(
    os.path.dirname(CHECKPOINT_PATH),
    exist_ok=True
)

torch.save(
    model.state_dict(),
    CHECKPOINT_PATH
)

print(f"Modelo salvo em {CHECKPOINT_PATH}")