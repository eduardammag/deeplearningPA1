import torch
from torch.utils.data import DataLoader

from dataset_pytorch import SyntheticSegmentationDataset
from unet import UNet
from cross_entropy import binary_cross_entropy_loss


# --------------------------------------------------
# Configurações
# --------------------------------------------------

DATA_DIR = "data/synthetic"

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = SyntheticSegmentationDataset(DATA_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# --------------------------------------------------
# Modelo
# --------------------------------------------------

model = UNet(
    in_channels=1,
    num_classes=1
)

model = model.to(DEVICE)


# --------------------------------------------------
# Loss
# --------------------------------------------------

criterion = binary_cross_entropy_loss()


# --------------------------------------------------
# Otimizador
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Treinamento
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch in loader:

        images = batch["image"].to(DEVICE)

        masks = batch["semantic_mask"].to(DEVICE)

        # BCE espera:
        # images -> [B, 1, H, W]
        # masks  -> [B, 1, H, W]

        masks = masks.unsqueeze(1).float()

        # Forward pass
        predictions = model(images)

        # Calcula a loss
        loss = criterion(
            predictions,
            masks
        )

        # Zera gradientes anteriores
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Atualiza os pesos
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )


# --------------------------------------------------
# Salvar modelo
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "checkpoints/unet_baseline.pth"
)

print("Modelo salvo em checkpoints/unet_baseline.pth")