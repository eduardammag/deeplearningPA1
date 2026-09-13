import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_pytorch import SyntheticSegmentationDataset
from unet import UNet
from segnet import SegNet
from ablation_losses import create_loss


# ============================================================
# Configurações
# ============================================================

DATA_DIR = "data/synthetic"

OUTPUT_DIR = Path("output_dir/ablations")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-3

NUM_CLASSES = 3

SEEDS = [42, 123]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Configuração das losses
# ============================================================

LOSS_CONFIGS = [
    {
        "name": "ce",
        "gamma": 0.0
    },
    {
        "name": "balanced_ce",
        "gamma": 0.0
    },
    {
        "name": "focal",
        "gamma": 0.0
    },
    {
        "name": "focal",
        "gamma": 1.0
    },
    {
        "name": "focal",
        "gamma": 2.0
    },
    {
        "name": "focal",
        "gamma": 5.0
    },
    {
        "name": "balanced_focal",
        "gamma": 0.0
    },
    {
        "name": "balanced_focal",
        "gamma": 1.0
    },
    {
        "name": "balanced_focal",
        "gamma": 2.0
    },
    {
        "name": "balanced_focal",
        "gamma": 5.0
    }
]


# ============================================================
# Pesos das classes
# ============================================================

# As classes são:
#
# 0 = background
# 1 = interior
# 2 = boundary
#
# A fronteira é minoritária, portanto recebe maior peso.

CLASS_WEIGHTS = [
    1.0,
    1.0,
    3.0
]


# ============================================================
# Reprodutibilidade
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset
# ============================================================

def create_loader():

    dataset = SyntheticSegmentationDataset(
        DATA_DIR
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return dataset, loader


# ============================================================
# Modelo
# ============================================================

def create_model(architecture):

    if architecture == "unet":

        model = UNet(
            in_channels=1,
            num_classes=NUM_CLASSES
        )

    elif architecture == "segnet":

        model = SegNet(
            in_channels=1,
            num_classes=NUM_CLASSES
        )

    else:

        raise ValueError(
            f"Arquitetura desconhecida: {architecture}"
        )

    return model.to(DEVICE)


# ============================================================
# Nome da configuração
# ============================================================

def configuration_name(
    axis,
    architecture,
    loss_name,
    gamma,
    seed
):

    if axis == "resolution":

        return (
            f"resolution_"
            f"{architecture}_"
            f"seed_{seed}"
        )

    if axis == "loss":

        return (
            f"loss_"
            f"{loss_name}_"
            f"gamma_{gamma:g}_"
            f"seed_{seed}"
        )

    raise ValueError(
        f"Eixo desconhecido: {axis}"
    )


# ============================================================
# Treinamento
# ============================================================

def train_model(
    model,
    loader,
    criterion
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for batch in loader:

            images = batch["image"].to(
                DEVICE
            )

            boundary_mask = batch[
                "boundary_mask"
            ].to(DEVICE)

            # ------------------------------------------------
            # O boundary_mask já contém:
            #
            # 0 = background
            # 1 = interior
            # 2 = boundary
            # ------------------------------------------------

            target = boundary_mask.long()

            predictions = model(images)

            loss = criterion(
                predictions,
                target
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(loader)
        )

        history.append(
            {
                "epoch": epoch + 1,
                "loss": average_loss
            }
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {average_loss:.4f}"
        )

    return history


# ============================================================
# Configuração de uma execução
# ============================================================

def run_configuration(
    axis,
    architecture,
    loss_name,
    gamma,
    seed,
    loader
):

    set_seed(seed)

    name = configuration_name(
        axis=axis,
        architecture=architecture,
        loss_name=loss_name,
        gamma=gamma,
        seed=seed
    )

    print()
    print("=" * 60)
    print(f"Configuração: {name}")
    print("=" * 60)

    model = create_model(
        architecture
    )

    criterion = create_loss(
        loss_name=loss_name,
        class_weights=CLASS_WEIGHTS,
        gamma=gamma,
        device=DEVICE
    )

    history = train_model(
        model=model,
        loader=loader,
        criterion=criterion
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        CHECKPOINT_DIR /
        f"{name}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    final_loss = history[-1]["loss"]

    result = {
        "axis": axis,
        "architecture": architecture,
        "loss": loss_name,
        "gamma": gamma,
        "seed": seed,
        "final_loss": final_loss,
        "checkpoint": str(
            checkpoint_path
        ),
        "history": history
    }

    print(
        f"Checkpoint salvo em: "
        f"{checkpoint_path}"
    )

    return result


# ============================================================
# Eixo 1
# ============================================================

def run_resolution_ablation(loader):

    results = []

    # Mantemos a loss balanceada fixa.
    #
    # O objetivo aqui é comparar somente
    # o mecanismo de recuperação de resolução.

    for architecture in [
        "unet",
        "segnet"
    ]:

        for seed in SEEDS:

            result = run_configuration(
                axis="resolution",
                architecture=architecture,
                loss_name="balanced_ce",
                gamma=0.0,
                seed=seed,
                loader=loader
            )

            results.append(result)

    return results


# ============================================================
# Eixo 2
# ============================================================

def run_loss_ablation(loader):

    results = []

    # Mantemos U-Net fixa.
    #
    # O objetivo aqui é estudar somente
    # o efeito da função de perda.

    for config in LOSS_CONFIGS:

        for seed in SEEDS:

            result = run_configuration(
                axis="loss",
                architecture="unet",
                loss_name=config["name"],
                gamma=config["gamma"],
                seed=seed,
                loader=loader
            )

            results.append(result)

    return results


# ============================================================
# Main
# ============================================================

def main():

    print(
        f"Device: {DEVICE}"
    )

    dataset, loader = create_loader()

    print(
        f"Dataset: {len(dataset)} imagens"
    )

    print()
    print(
        "Iniciando ablação do Eixo 1..."
    )

    resolution_results = (
        run_resolution_ablation(
            loader
        )
    )

    print()
    print(
        "Iniciando ablação do Eixo 2..."
    )

    loss_results = (
        run_loss_ablation(
            loader
        )
    )

    all_results = (
        resolution_results +
        loss_results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = (
        OUTPUT_DIR /
        "training_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("Treinamento das ablações concluído.")
    print("=" * 60)

    print(
        f"Resultados salvos em: "
        f"{results_path}"
    )

    print(
        f"Checkpoints salvos em: "
        f"{CHECKPOINT_DIR}"
    )


if __name__ == "__main__":

    main()