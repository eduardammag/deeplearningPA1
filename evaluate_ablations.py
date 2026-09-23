import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_bbbc038 import BBBC038Dataset

from unet import UNet
from segnet import SegNet

from watershed import watershed_from_logits

from semantic_metrics import dice_score, iou_score

from instance_metrics import mean_average_precision, count_error

DATA_ROOT = "data/BBBC038"

SPLIT_FILE = "data/BBBC038/splits.json"

CHECKPOINT_DIR = Path(
    "output_dir/ablations_bbbc038/checkpoints"
)

OUTPUT_DIR = Path(
    "output_dir/ablations_bbbc038/evaluation"
)

FIGURES_DIR = (
    OUTPUT_DIR / "figures"
)

BATCH_SIZE = 4

TARGET_SIZE = 256

NUM_CLASSES = 3

SEEDS = [42, 123]

INTERIOR_THRESHOLD = 0.5

FOREGROUND_THRESHOLD = 0.5

MIN_MARKER_SIZE = 10

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def create_test_loader():

    dataset = BBBC038Dataset(
        root_dir=DATA_ROOT,
        split="test",
        split_file=SPLIT_FILE,
        resize=TARGET_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return dataset, loader


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
            f"Arquitetura desconhecida: "
            f"{architecture}"
        )

    return model.to(DEVICE)


def load_model(
    architecture,
    checkpoint_path
):

    model = create_model(
        architecture
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    return model


def foreground_probability_from_logits(logits):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    # Classe 0 = background
    # foreground = interior + boundary

    foreground_probability = (
        1.0 - probabilities[:, 0:1]
    )

    return foreground_probability


def evaluate_model(
    model,
    loader
):

    dice_values = []

    iou_values = []

    map_values = []

    count_errors = []

    with torch.no_grad():

        for batch in loader:

            images = batch[
                "image"
            ].to(DEVICE)

            semantic_masks = (
                batch[
                    "semantic_mask"
                ]
                .to(DEVICE)
                .unsqueeze(1)
                .float()
            )

            instance_masks = (
                batch[
                    "instance_mask"
                ]
                .cpu()
                .numpy()
            )

            logits = model(
                images
            )

            foreground_probability = (
                foreground_probability_from_logits(
                    logits
                )
            )

            foreground_prob = (
                foreground_probability
                .clamp(
                    min=1e-7,
                    max=1.0 - 1e-7
                )
            )

            foreground_logits = torch.log(
                foreground_prob /
                (1.0 - foreground_prob)
            )

            dice = dice_score(
                foreground_logits,
                semantic_masks
            )

            iou = iou_score(
                foreground_logits,
                semantic_masks
            )

            dice_values.append(
                float(dice)
            )

            iou_values.append(
                float(iou)
            )

            for index in range(
                images.shape[0]
            ):

                image_logits = (
                    logits[index]
                )

                gt_instance_mask = (
                    instance_masks[index]
                )

                predicted_instances, _, _ = (
                    watershed_from_logits(
                        image_logits,
                        interior_threshold=(
                            INTERIOR_THRESHOLD
                        ),
                        foreground_threshold=(
                            FOREGROUND_THRESHOLD
                        ),
                        min_marker_size=(
                            MIN_MARKER_SIZE
                        )
                    )
                )

                predicted_instances = (
                    np.asarray(
                        predicted_instances
                    )
                )

                gt_instance_mask = (
                    np.asarray(
                        gt_instance_mask
                    )
                )

                probability = (
                    foreground_probability[
                        index,
                        0
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                assert (
                    predicted_instances.ndim
                    == 2
                )

                assert (
                    gt_instance_mask.ndim
                    == 2
                )

                assert (
                    probability.ndim
                    == 2
                )

                assert (
                    predicted_instances.shape
                    ==
                    gt_instance_mask.shape
                )

                assert (
                    predicted_instances.shape
                    ==
                    probability.shape
                )

                _, map_value = (
                    mean_average_precision(
                        predicted_instances,
                        gt_instance_mask,
                        probability
                    )
                )

                map_values.append(
                    float(map_value)
                )

                error = count_error(
                    predicted_instances,
                    gt_instance_mask
                )

                count_errors.append(
                    float(error)
                )

    return {

        "dice": float(
            np.mean(dice_values)
        ),

        "iou": float(
            np.mean(iou_values)
        ),

        "map": float(
            np.mean(map_values)
        ),

        "count_error": float(
            np.mean(count_errors)
        )
    }

def resolution_checkpoint_name(
    architecture,
    seed
):

    return (
        CHECKPOINT_DIR /
        f"resolution_"
        f"{architecture}_"
        f"seed_{seed}.pth"
    )


def loss_checkpoint_name(
    loss_name,
    gamma,
    seed
):

    return (
        CHECKPOINT_DIR /
        f"loss_"
        f"{loss_name}_"
        f"gamma_{gamma:g}_"
        f"seed_{seed}.pth"
    )


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


def evaluate_resolution(loader):

    results = []

    for architecture in [
        "unet",
        "segnet"
    ]:

        for seed in SEEDS:

            checkpoint = (
                resolution_checkpoint_name(
                    architecture,
                    seed
                )
            )

            if not checkpoint.exists():

                print(
                    f"[AVISO] "
                    f"Checkpoint não encontrado: "
                    f"{checkpoint}"
                )

                continue

            print()
            print(
                f"Avaliando "
                f"{architecture} "
                f"seed={seed}"
            )

            model = load_model(
                architecture,
                checkpoint
            )

            metrics = evaluate_model(
                model,
                loader
            )

            results.append({

                "axis":
                    "resolution",

                "architecture":
                    architecture,

                "loss":
                    "balanced_ce",

                "gamma":
                    0.0,

                "seed":
                    seed,

                **metrics
            })

            print(
                f"mAP: "
                f"{metrics['map']:.4f}"
            )

            print(
                f"Dice: "
                f"{metrics['dice']:.4f}"
            )

            print(
                f"IoU: "
                f"{metrics['iou']:.4f}"
            )

            print(
                f"Count error: "
                f"{metrics['count_error']:.4f}"
            )

    return results


def evaluate_loss(loader):

    results = []

    for config in LOSS_CONFIGS:

        for seed in SEEDS:

            checkpoint = (
                loss_checkpoint_name(
                    config["name"],
                    config["gamma"],
                    seed
                )
            )

            if not checkpoint.exists():

                print(
                    f"[AVISO] "
                    f"Checkpoint não encontrado: "
                    f"{checkpoint}"
                )

                continue

            print()
            print(
                f"Avaliando "
                f"{config['name']} "
                f"gamma={config['gamma']} "
                f"seed={seed}"
            )

            model = load_model(
                "unet",
                checkpoint
            )

            metrics = evaluate_model(
                model,
                loader
            )

            results.append({

                "axis":
                    "loss",

                "architecture":
                    "unet",

                "loss":
                    config["name"],

                "gamma":
                    config["gamma"],

                "seed":
                    seed,

                **metrics
            })

            print(
                f"mAP: "
                f"{metrics['map']:.4f}"
            )

            print(
                f"Dice: "
                f"{metrics['dice']:.4f}"
            )

            print(
                f"IoU: "
                f"{metrics['iou']:.4f}"
            )

            print(
                f"Count error: "
                f"{metrics['count_error']:.4f}"
            )

    return results


def mean_std(values):

    values = np.asarray(
        values,
        dtype=float
    )

    return {

        "mean": float(
            np.mean(values)
        ),

        "std": float(
            np.std(
                values,
                ddof=1
            )
        )
    }


def summarize_results(
    results
):

    groups = {}

    for result in results:

        if result["axis"] == "resolution":

            key = (
                result["axis"],
                result["architecture"]
            )

        else:

            key = (
                result["axis"],
                result["loss"],
                result["gamma"]
            )

        if key not in groups:

            groups[key] = []

        groups[key].append(
            result
        )

    summaries = []

    for key, group in groups.items():

        first = group[0]

        summary = {

            "axis":
                first["axis"],

            "architecture":
                first["architecture"],

            "loss":
                first["loss"],

            "gamma":
                first["gamma"],

            "seeds": [
                item["seed"]
                for item in group
            ],

            "dice":
                mean_std([
                    item["dice"]
                    for item in group
                ]),

            "iou":
                mean_std([
                    item["iou"]
                    for item in group
                ]),

            "map":
                mean_std([
                    item["map"]
                    for item in group
                ]),

            "count_error":
                mean_std([
                    item["count_error"]
                    for item in group
                ])
        }

        summaries.append(
            summary
        )

    return summaries


def plot_resolution(
    summaries
):

    data = [
        item
        for item in summaries
        if item["axis"] == "resolution"
    ]

    if not data:
        return

    labels = [
        item["architecture"]
        for item in data
    ]

    means = [
        item["map"]["mean"]
        for item in data
    ]

    stds = [
        item["map"]["std"]
        for item in data
    ]

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    x = np.arange(
        len(labels)
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=5
    )

    plt.xticks(
        x,
        [
            "U-Net\nSkip connections",
            "SegNet\nMax-unpooling"
        ]
    )

    plt.ylabel(
        "mAP@[0.50:0.95]"
    )

    plt.title(
        "Eixo 1 — Recuperação de resolução"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "resolution_map.png",
        dpi=200
    )

    plt.close()


def plot_loss(
    summaries
):

    data = [
        item
        for item in summaries
        if item["axis"] == "loss"
    ]

    if not data:
        return

    labels = []

    means = []

    stds = []

    for item in data:

        if item["loss"] in [
            "ce",
            "balanced_ce"
        ]:

            label = item["loss"]

        else:

            label = (
                f"{item['loss']}\n"
                f"gamma={item['gamma']:g}"
            )

        labels.append(label)

        means.append(
            item["map"]["mean"]
        )

        stds.append(
            item["map"]["std"]
        )

    x = np.arange(
        len(labels)
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=4
    )

    plt.xticks(
        x,
        labels,
        rotation=45,
        ha="right"
    )

    plt.ylabel(
        "mAP@[0.50:0.95]"
    )

    plt.title(
        "Eixo 2 — Função de perda"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "loss_map.png",
        dpi=200
    )

    plt.close()


def print_summary(
    summaries
):

    print()
    print("=" * 80)
    print(
        "RESULTADOS — PARTE 3"
    )
    print("=" * 80)

    for item in summaries:

        print()

        if item["axis"] == "resolution":

            name = (
                f"{item['architecture']}"
            )

        else:

            name = (
                f"{item['loss']}"
            )

            if item["loss"] not in [
                "ce",
                "balanced_ce"
            ]:

                name += (
                    f" (gamma="
                    f"{item['gamma']:g})"
                )

        print(
            f"{name}"
        )

        print(
            f"  Dice: "
            f"{item['dice']['mean']:.4f} "
            f"± "
            f"{item['dice']['std']:.4f}"
        )

        print(
            f"  IoU:  "
            f"{item['iou']['mean']:.4f} "
            f"± "
            f"{item['iou']['std']:.4f}"
        )

        print(
            f"  mAP:  "
            f"{item['map']['mean']:.4f} "
            f"± "
            f"{item['map']['std']:.4f}"
        )

        print(
            f"  Count error: "
            f"{item['count_error']['mean']:.4f} "
            f"± "
            f"{item['count_error']['std']:.4f}"
        )


def main():

    print("=" * 80)
    print("PARTE 3 — AVALIAÇÃO DAS ABLAÇÕES")
    print("=" * 80)

    print(f"Device: {DEVICE}")

    dataset, loader = (create_test_loader())

    print(
        f"Test images: "
        f"{len(dataset)}"
    )

    resolution_results = (evaluate_resolution(loader))

    loss_results = (evaluate_loss(loader))

    all_results = (
        resolution_results +
        loss_results
    )

    summaries = summarize_results(all_results)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = (
        OUTPUT_DIR /
        "ablation_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "per_seed":
                    all_results,

                "mean_std":
                    summaries
            },
            file,
            indent=2,
            ensure_ascii=False
        )

    plot_resolution(summaries)

    plot_loss(summaries)

    print_summary(summaries)

    print()
    print("=" * 80)

    print(
        f"Resultados salvos em: "
        f"{results_path}"
    )

    print(
        f"Figuras salvas em: "
        f"{FIGURES_DIR}"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()