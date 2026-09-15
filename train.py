# train.py

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_bbbc038 import BBBC038Dataset
from unet import UNet
from cross_entropy import binary_cross_entropy_loss


DATA_ROOT = "data/BBBC038"

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-3

CHECKPOINT_PATH = (
    "checkpoints/unet_baseline_bbbc038.pth"
)

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    set_seed(SEED)

    print("=" * 70)
    print("BBBC038 - BASELINE SEMÂNTICO")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    train_dataset = BBBC038Dataset(
        root_dir=DATA_ROOT,
        split="train"
    )

    val_dataset = BBBC038Dataset(
        root_dir=DATA_ROOT,
        split="validation"
    )

    print(
        f"\nTreino: {len(train_dataset)} imagens"
    )

    print(
        f"Validação: {len(val_dataset)} imagens"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # ---------------------------------------------------------
    # Modelo
    # ---------------------------------------------------------

    model = UNet(
        in_channels=1,
        num_classes=1
    )

    model = model.to(
        DEVICE
    )

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    criterion = binary_cross_entropy_loss()

    # ---------------------------------------------------------
    # Otimizador
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ---------------------------------------------------------
    # Treinamento
    # ---------------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0

        for batch in train_loader:

            images = batch[
                "image"
            ].to(
                DEVICE
            )

            masks = batch[
                "semantic_mask"
            ].to(
                DEVICE
            )

            masks = masks.unsqueeze(
                1
            )

            predictions = model(
                images
            )

            loss = criterion(
                predictions,
                masks
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(
            train_loader
        )

        # -----------------------------------------------------
        # Validação
        # -----------------------------------------------------

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for batch in val_loader:

                images = batch[
                    "image"
                ].to(
                    DEVICE
                )

                masks = batch[
                    "semantic_mask"
                ].to(
                    DEVICE
                )

                masks = masks.unsqueeze(
                    1
                )

                predictions = model(
                    images
                )

                loss = criterion(
                    predictions,
                    masks
                )

                val_loss += loss.item()

        val_loss /= len(
            val_loader
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS} "
            f"- "
            f"Train Loss: {train_loss:.4f} "
            f"- "
            f"Val Loss: {val_loss:.4f}"
        )

        # -----------------------------------------------------
        # Salva melhor modelo
        # -----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            os.makedirs(
                os.path.dirname(
                    CHECKPOINT_PATH
                ),
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH
            )

            print(
                f"  Melhor modelo salvo em "
                f"{CHECKPOINT_PATH}"
            )

    print(
        "\nTreinamento concluído."
    )

    print(
        f"Melhor validation loss: "
        f"{best_val_loss:.4f}"
    )


if __name__ == "__main__":
    main()