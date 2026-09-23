import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_bbbc038 import BBBC038Dataset
from unet import UNet
from segnet import SegNet
from ablation_losses import create_loss

DATA_ROOT = "data/BBBC038"

SPLIT_FILE = "data/BBBC038/splits.json"

OUTPUT_DIR = Path(
    "output_dir/ablations_bbbc038"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR / "checkpoints"
)

HISTORY_DIR = (
    OUTPUT_DIR / "histories"
)

BATCH_SIZE = 4

# O baseline foi treinado com 20 épocas.
# Para as ablações usamos um orçamento menor e igual para todas as configurações.
ABLATION_EPOCHS = 5

LEARNING_RATE = 1e-3

NUM_CLASSES = 3

TARGET_SIZE = 256

SEEDS = [42, 123]

# Se a validação não melhorar por esse número de épocas, o treinamento da configuração é interrompido.
EARLY_STOPPING_PATIENCE = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# 0 = background
# 1 = interior
# 2 = boundary

# A classe de fronteira é minoritária. Por isso recebe peso maior na CE balanceada.
CLASS_WEIGHTS = [1.0, 1.0, 3.0]

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

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


def instance_mask_to_boundary_mask(instance_mask, boundary_thickness=1):
    if instance_mask.ndim == 2:

        instance_mask = (
            instance_mask.unsqueeze(0)
        )

    if instance_mask.ndim != 3:

        raise ValueError(
            "instance_mask deve possuir shape "
            "[H, W] ou [B, H, W]."
        )

    boundary = torch.zeros_like(
        instance_mask,
        dtype=torch.bool
    )

    center = instance_mask

    different_up = (
        center[:, 1:, :] > 0
    ) & (
        center[:, :-1, :] > 0
    ) & (
        center[:, 1:, :]
        != center[:, :-1, :]
    )

    boundary[:, 1:, :] |= different_up

    different_down = (
        center[:, :-1, :] > 0
    ) & (
        center[:, 1:, :] > 0
    ) & (
        center[:, :-1, :]
        != center[:, 1:, :]
    )

    boundary[:, :-1, :] |= different_down

    different_left = (
        center[:, :, 1:] > 0
    ) & (
        center[:, :, :-1] > 0
    ) & (
        center[:, :, 1:]
        != center[:, :, :-1]
    )

    boundary[:, :, 1:] |= different_left

    different_right = (
        center[:, :, :-1] > 0
    ) & (
        center[:, :, 1:] > 0
    ) & (
        center[:, :, :-1]
        != center[:, :, 1:]
    )

    boundary[:, :, :-1] |= different_right

    if boundary_thickness > 1:

        for _ in range(
            boundary_thickness - 1
        ):

            expanded = boundary.clone()

            expanded[:, 1:, :] |= (
                boundary[:, :-1, :]
            )

            expanded[:, :-1, :] |= (
                boundary[:, 1:, :]
            )

            expanded[:, :, 1:] |= (
                boundary[:, :, :-1]
            )

            expanded[:, :, :-1] |= (
                boundary[:, :, 1:]
            )

            boundary = expanded

    target = torch.zeros_like(
        instance_mask,
        dtype=torch.long
    )

    target[
        instance_mask > 0
    ] = 1

    target[
        boundary
    ] = 2

    return target


def batch_to_targets(batch):

    if "instance_mask" not in batch:

        raise KeyError(
            "O dataset não fornece 'instance_mask'."
        )

    instance_mask = (
        batch["instance_mask"]
    )

    targets = instance_mask_to_boundary_mask(
        instance_mask,
        boundary_thickness=1
    )

    return targets.to(
        DEVICE
    ).long()


def validate_target_generation(dataset):

    print()
    print("=" * 70)
    print("VERIFICAÇÃO DO TARGET DE 3 CLASSES")
    print("=" * 70)

    sample = dataset[0]

    required_keys = {
        "image",
        "semantic_mask",
        "instance_mask",
        "image_id"
    }

    missing = (
        required_keys
        - set(sample.keys())
    )

    if missing:

        raise KeyError(
            "Chaves ausentes no dataset: "
            f"{missing}"
        )

    instance_mask = (
        sample["instance_mask"]
        .unsqueeze(0)
    )

    target = (
        instance_mask_to_boundary_mask(
            instance_mask,
            boundary_thickness=1
        )
    )

    unique_classes = torch.unique(
        target
    ).tolist()

    print(
        f"Image ID: "
        f"{sample['image_id']}"
    )

    print(
        f"Image shape: "
        f"{tuple(sample['image'].shape)}"
    )

    print(
        f"Instance mask shape: "
        f"{tuple(sample['instance_mask'].shape)}"
    )

    print(
        f"Target shape: "
        f"{tuple(target.shape)}"
    )

    print(
        f"Classes presentes: "
        f"{unique_classes}"
    )

    valid_classes = all(
        cls in [0, 1, 2]
        for cls in unique_classes
    )

    if not valid_classes:

        raise ValueError(
            "O target contém classes inválidas."
        )

    print("Alvo de 3 classes: OK")


def create_datasets():

    train_dataset = BBBC038Dataset(
        root_dir=DATA_ROOT,
        split="train",
        split_file=SPLIT_FILE,
        resize=TARGET_SIZE
    )

    val_dataset = BBBC038Dataset(
        root_dir=DATA_ROOT,
        split="validation",
        split_file=SPLIT_FILE,
        resize=TARGET_SIZE
    )

    return (
        train_dataset,
        val_dataset
    )

def create_loaders(train_dataset, val_dataset, seed):

    generator = torch.Generator()

    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return (
        train_loader,
        val_loader
    )


def create_model(
    architecture
):

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
            f"Arquitetura desconhecida: "
            f"{architecture}"
        )

    return model.to(
        DEVICE
    )

def configuration_name(axis, architecture, loss_name, gamma, seed):

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


# MÉTRICAS SEMÂNTICAS

def calculate_segmentation_metrics(model, loader):

    model.eval()

    intersection = 0.0
    prediction_area = 0.0
    target_area = 0.0

    with torch.no_grad():

        for batch in loader:

            images = (
                batch["image"]
                .to(DEVICE)
            )

            semantic_target = (
                batch["semantic_mask"]
                .to(DEVICE)
            )

            logits = model(
                images
            )

            prediction = (
                torch.argmax(
                    logits,
                    dim=1
                ) > 0
            )

            target = (
                semantic_target > 0
            )

            intersection += (
                prediction
                & target
            ).sum().item()

            prediction_area += (
                prediction.sum().item()
            )

            target_area += (
                target.sum().item()
            )

    union = (
        prediction_area
        + target_area
        - intersection
    )

    dice_denominator = (
        prediction_area
        + target_area
    )

    if dice_denominator == 0:

        dice = 1.0

    else:

        dice = (
            2.0 * intersection
            / dice_denominator
        )

    if union == 0:

        iou = 1.0

    else:

        iou = (
            intersection
            / union
        )

    return (
        float(dice),
        float(iou)
    )

def calculate_loss(model, loader, criterion):

    model.eval()

    total_loss = 0.0

    total_samples = 0

    with torch.no_grad():

        for batch in loader:

            images = (
                batch["image"]
                .to(DEVICE)
            )

            targets = (
                batch_to_targets(
                    batch
                )
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                targets
            )

            batch_size = (
                images.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

    return (
        total_loss
        / max(total_samples, 1)
    )

def train_model(model, train_loader, val_loader, criterion):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []

    best_val_loss = float(
        "inf"
    )

    best_state = None

    epochs_without_improvement = 0

    for epoch in range(
        ABLATION_EPOCHS
    ):

        epoch_start = time.time()

        model.train()

        running_loss = 0.0

        total_samples = 0

        for batch in train_loader:

            images = (
                batch["image"]
                .to(DEVICE)
            )

            targets = (
                batch_to_targets(
                    batch
                )
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                targets
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            batch_size = (
                images.shape[0]
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

        train_loss = (
            running_loss
            / max(total_samples, 1)
        )

        val_loss = calculate_loss(
            model,
            val_loader,
            criterion
        )

        val_dice, val_iou = (
            calculate_segmentation_metrics(
                model,
                val_loader
            )
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(
                    train_loss
                ),
                "val_loss": float(
                    val_loss
                ),
                "val_dice": float(
                    val_dice
                ),
                "val_iou": float(
                    val_iou
                ),
                "epoch_time_seconds": float(
                    epoch_time
                )
            }
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/"
            f"{ABLATION_EPOCHS} "
            f"- Train Loss: "
            f"{train_loss:.4f} "
            f"- Val Loss: "
            f"{val_loss:.4f} "
            f"- Dice: "
            f"{val_dice:.4f} "
            f"- IoU: "
            f"{val_iou:.4f} "
            f"- Tempo: "
            f"{epoch_time / 60:.1f} min"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "Early stopping: "
                "validação não melhorou."
            )

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    best_epoch = min(
        history,
        key=lambda item: item[
            "val_loss"
        ]
    )

    return (
        history,
        best_val_loss,
        best_epoch
    )

def configuration_already_completed(name):

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{name}.pth"
    )

    history_path = (
        HISTORY_DIR
        / f"{name}.json"
    )

    return (
        checkpoint_path.exists()
        and history_path.exists()
    )

def run_configuration(axis, architecture, loss_name, gamma, seed, train_dataset, val_dataset):

    name = configuration_name(
        axis=axis,
        architecture=architecture,
        loss_name=loss_name,
        gamma=gamma,
        seed=seed
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{name}.pth"
    )

    history_path = (
        HISTORY_DIR
        / f"{name}.json"
    )

    if configuration_already_completed(
        name
    ):

        print()
        print("=" * 70)

        print(
            f"Configuração já concluída: "
            f"{name}"
        )

        print("Pulando treinamento.")

        print("=" * 70)

        with open(
            history_path,
            "r",
            encoding="utf-8"
        ) as file:

            saved_result = json.load(
                file
            )

        return saved_result


    set_seed(seed)

    train_loader, val_loader = (
        create_loaders(
            train_dataset,
            val_dataset,
            seed
        )
    )

    print()
    print("=" * 70)

    print(f"Configuração: {name}")

    print("=" * 70)

    print(f"Arquitetura: {architecture}")

    print(f"Loss: {loss_name}")

    print(f"Gamma: {gamma}")

    print(f"Seed: {seed}")

    print(f"Épocas máximas: ", f"{ABLATION_EPOCHS}")

    print(f"Device: {DEVICE}")

    model = create_model(
        architecture
    )

    criterion = create_loss(
        loss_name=loss_name,
        class_weights=CLASS_WEIGHTS,
        gamma=gamma
    )

    start_time = time.time()

    (
        history,
        best_val_loss,
        best_epoch
    ) = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion
    )

    total_time = (
        time.time()
        - start_time
    )

    best_val_dice = (
        best_epoch["val_dice"]
    )

    best_val_iou = (
        best_epoch["val_iou"]
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    result = {

        "axis": axis,

        "architecture":
            architecture,

        "loss":
            loss_name,

        "gamma":
            gamma,

        "seed":
            seed,

        "batch_size":
            BATCH_SIZE,

        "epochs_max":
            ABLATION_EPOCHS,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "learning_rate":
            LEARNING_RATE,

        "target_size":
            TARGET_SIZE,

        "class_weights":
            CLASS_WEIGHTS,

        "boundary_thickness":
            1,

        "best_epoch":
            int(
                best_epoch["epoch"]
            ),

        "best_val_loss":
            float(
                best_val_loss
            ),

        "best_val_dice":
            float(
                best_val_dice
            ),

        "best_val_iou":
            float(
                best_val_iou
            ),

        "total_time_seconds":
            float(
                total_time
            ),

        "total_time_minutes":
            float(
                total_time / 60
            ),

        "checkpoint":
            str(
                checkpoint_path
            ),

        "history":
            history
    }

    with open(history_path, "w", encoding="utf-8") as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Melhor epoch: "
        f"{best_epoch['epoch']}"
    )

    print(
        f"Melhor Val Loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Val Dice: "
        f"{best_val_dice:.4f}"
    )

    print(
        f"Val IoU: "
        f"{best_val_iou:.4f}"
    )

    print(
        f"Tempo total: "
        f"{total_time / 60:.1f} minutos"
    )

    print(
        f"Checkpoint salvo em: "
        f"{checkpoint_path}"
    )

    return result


# EIXO 1 — RECUPERAÇÃO DE RESOLUÇÃO

def run_resolution_ablation(train_dataset, val_dataset):

    results = []

    print()
    print("=" * 70)

    print("EIXO 1 — RECUPERAÇÃO DE RESOLUÇÃO")

    print("=" * 70)

    print("Comparação:")

    print("  U-Net  → skip connections")

    print("  SegNet → max-unpooling")

    print("Loss fixa: balanced CE")

    print()

    for architecture in ["unet","segnet"]:

        for seed in SEEDS:

            result = run_configuration(

                axis="resolution",

                architecture=architecture,

                loss_name="balanced_ce",

                gamma=0.0,

                seed=seed,

                train_dataset=train_dataset,

                val_dataset=val_dataset
            )

            results.append(
                result
            )

    return results


# EIXO 2 — FUNÇÃO DE PERDA

def run_loss_ablation(train_dataset, val_dataset):

    results = []

    print()
    print("=" * 70)

    print("EIXO 2 — FUNÇÃO DE PERDA")

    print("=" * 70)

    print("Arquitetura fixa: U-Net")

    print()

    for config in LOSS_CONFIGS:

        for seed in SEEDS:

            result = run_configuration(

                axis="loss",

                architecture="unet",

                loss_name=config["name"],

                gamma=config["gamma"],

                seed=seed,

                train_dataset=train_dataset,

                val_dataset=val_dataset
            )

            results.append(
                result
            )

    return results

def save_summary(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = (
        OUTPUT_DIR
        / "training_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    groups = {}

    for result in results:

        if result.get(
            "axis"
        ) == "resolution":

            key = (
                "resolution",
                result["architecture"]
            )

        elif result.get(
            "axis"
        ) == "loss":

            key = (
                "loss",
                result["loss"],
                result["gamma"]
            )

        else:

            continue

        if key not in groups:

            groups[key] = []

        groups[key].append(
            result
        )

    summary = []

    for key, values in groups.items():

        dice_values = [
            item["best_val_dice"]
            for item in values
        ]

        iou_values = [
            item["best_val_iou"]
            for item in values
        ]

        loss_values = [
            item["best_val_loss"]
            for item in values
        ]

        summary_item = {

            "axis":
                key[0],

            "configuration":
                list(key),

            "n_seeds":
                len(values),

            "dice_mean":
                float(
                    np.mean(
                        dice_values
                    )
                ),

            "dice_std":
                float(
                    np.std(
                        dice_values,
                        ddof=1
                    )
                ) if len(
                    dice_values
                ) > 1 else 0.0,

            "iou_mean":
                float(
                    np.mean(
                        iou_values
                    )
                ),

            "iou_std":
                float(
                    np.std(
                        iou_values,
                        ddof=1
                    )
                ) if len(
                    iou_values
                ) > 1 else 0.0,

            "val_loss_mean":
                float(
                    np.mean(
                        loss_values
                    )
                ),

            "val_loss_std":
                float(
                    np.std(
                        loss_values,
                        ddof=1
                    )
                ) if len(
                    loss_values
                ) > 1 else 0.0
        }

        summary.append(
            summary_item
        )

    summary_path = (
        OUTPUT_DIR
        / "ablation_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False
        )

    return (
        results_path,
        summary_path,
        summary
    )

def print_summary(summary):

    print()
    print("=" * 70)

    print("RESUMO DAS ABLAÇÕES")

    print("=" * 70)

    for item in summary:

        configuration = (
            item["configuration"]
        )

        print()

        print(
            f"Configuração: "
            f"{configuration}"
        )

        print(
            f"Dice: "
            f"{item['dice_mean']:.4f} "
            f"± "
            f"{item['dice_std']:.4f}"
        )

        print(
            f"IoU:  "
            f"{item['iou_mean']:.4f} "
            f"± "
            f"{item['iou_std']:.4f}"
        )

        print(
            f"Val Loss: "
            f"{item['val_loss_mean']:.4f} "
            f"± "
            f"{item['val_loss_std']:.4f}"
        )

def main():

    global_start = time.time()

    print("=" * 70)

    print("PARTE 3 — ABLAÇÕES BBBC038")

    print("=" * 70)

    print(f"Device: {DEVICE}")

    print(f"Batch size: {BATCH_SIZE}")

    print(
        f"Épocas máximas por configuração: "
        f"{ABLATION_EPOCHS}"
    )

    print(
        f"Early stopping patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Target size: "
        f"{TARGET_SIZE}"
    )

    print(
        f"Seeds: "
        f"{SEEDS}"
    )

    print()

    (
        train_dataset,
        val_dataset
    ) = create_datasets()

    print(
        f"Treino: "
        f"{len(train_dataset)} imagens"
    )

    print(
        f"Validação: "
        f"{len(val_dataset)} imagens"
    )

    validate_target_generation(
        train_dataset
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # EIXO 1

    resolution_results = (
        run_resolution_ablation(
            train_dataset,
            val_dataset
        )
    )

    # EIXO 2

    loss_results = (
        run_loss_ablation(
            train_dataset,
            val_dataset
        )
    )

    all_results = (
        resolution_results
        + loss_results
    )

    (
        results_path,
        summary_path,
        summary
    ) = save_summary(
        all_results
    )

    print_summary(
        summary
    )

    # Tempo total

    total_time = (
        time.time()
        - global_start
    )

    print()
    print("=" * 70)

    print("PARTE 3 CONCLUÍDA")

    print("=" * 70)

    print(
        f"Tempo total: "
        f"{total_time / 3600:.2f} horas"
    )

    print(
        f"Resultados individuais: "
        f"{results_path}"
    )

    print(
        f"Resumo: "
        f"{summary_path}"
    )

    print(
        f"Checkpoints: "
        f"{CHECKPOINT_DIR}"
    )

    print("=" * 70)

if __name__ == "__main__":

    main()