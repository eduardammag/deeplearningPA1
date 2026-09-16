import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.segmentation.postprocessing.connected_components import connected_components
from src.segmentation.data.dataset import SyntheticSegmentationDataset
from src.segmentation.metrics.instance import (
    count_error,
    mean_average_precision,
)
from src.segmentation.metrics.semantic import (
    dice_score,
    iou_score,
)
from src.segmentation.models.unet import UNet
from src.segmentation.postprocessing.watershed import watershed_from_logits

DATA_DIR = "data/synthetic"

BASELINE_CHECKPOINT = "checkpoints/unet_baseline.pth"
BOUNDARY_CHECKPOINT = "checkpoints/unet_boundary.pth"

OUTPUT_DIR = Path("output_dir/evaluation")

BATCH_SIZE = 4

INTERIOR_THRESHOLD = 0.5
FOREGROUND_THRESHOLD = 0.5
MIN_MARKER_SIZE = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = SyntheticSegmentationDataset(
    DATA_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

def load_model(checkpoint_path, num_classes):

    model = UNet(
        in_channels=1,
        num_classes=num_classes
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    return model


def probability_to_logit(probability):

    probability = probability.clamp(
        min=1e-7,
        max=1.0 - 1e-7
    )

    return torch.log(
        probability / (1.0 - probability)
    )


def empty_results():

    return {
        "dice": [],
        "iou": [],
        "map": [],
        "count_error": [],
        "density": [],
        "map_by_threshold": {},
    }


def update_threshold_results(results, ap_results):

    for threshold, ap in ap_results.items():

        threshold = str(threshold)

        if threshold not in results["map_by_threshold"]:
            results["map_by_threshold"][threshold] = []

        results["map_by_threshold"][threshold].append(
            float(ap)
        )


def summarize_results(results):

    mean_map_by_threshold = {}

    for threshold, values in results["map_by_threshold"].items():

        mean_map_by_threshold[threshold] = float(
            np.mean(values)
        )

    return {
        "dice": float(np.mean(results["dice"])),
        "iou": float(np.mean(results["iou"])),
        "map": float(np.mean(results["map"])),
        "count_error": float(
            np.mean(results["count_error"])
        ),
        "map_by_threshold": mean_map_by_threshold,
    }

def evaluate_baseline(model):

    results = empty_results()

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(DEVICE)

            semantic_masks = (
                batch["semantic_mask"]
                .to(DEVICE)
                .unsqueeze(1)
                .float()
            )

            gt_instance_masks = (
                batch["instance_mask"]
                .cpu()
                .numpy()
            )

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            binary_predictions = (
                probabilities >= 0.5
            )

            dice = dice_score(
                logits,
                semantic_masks
            )

            iou = iou_score(
                logits,
                semantic_masks
            )

            results["dice"].append(float(dice))
            results["iou"].append(float(iou))

            for i in range(images.shape[0]):

                binary_mask = (
                    binary_predictions[i, 0]
                    .cpu()
                    .numpy()
                )

                foreground_probability = (
                    probabilities[i, 0]
                    .cpu()
                    .numpy()
                )

                gt_instance_mask = (
                    gt_instance_masks[i]
                )

                pred_instance_mask, _ = (
                    connected_components(
                        binary_mask
                    )
                )

                ap_results, map_value = (
                    mean_average_precision(
                        pred_instance_mask,
                        gt_instance_mask,
                        foreground_probability
                    )
                )

                results["map"].append(
                    float(map_value)
                )

                update_threshold_results(
                    results,
                    ap_results
                )

                error = count_error(
                    pred_instance_mask,
                    gt_instance_mask
                )

                results["count_error"].append(
                    float(error)
                )

                number_of_objects = len(
                    np.unique(
                        gt_instance_mask
                    )
                ) - 1

                results["density"].append(
                    {
                        "num_objects": int(
                            number_of_objects
                        ),
                        "map": float(map_value),
                        "count_error": float(error),
                    }
                )

    return summarize_results(results), results["density"]

def evaluate_boundary(model):

    results = empty_results()

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(DEVICE)

            semantic_masks = (
                batch["semantic_mask"]
                .to(DEVICE)
                .unsqueeze(1)
                .float()
            )

            gt_instance_masks = (
                batch["instance_mask"]
                .cpu()
                .numpy()
            )

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            # Background = classe 0
            # Foreground = 1 - background
            foreground_probability = (
                1.0 - probabilities[:, 0:1]
            )

            foreground_logits = (
                probability_to_logit(
                    foreground_probability
                )
            )

            dice = dice_score(
                foreground_logits,
                semantic_masks
            )

            iou = iou_score(
                foreground_logits,
                semantic_masks
            )

            results["dice"].append(float(dice))
            results["iou"].append(float(iou))

            for i in range(images.shape[0]):

                image_logits = logits[i]

                gt_instance_mask = (
                    gt_instance_masks[i]
                )

                (
                    pred_instance_mask,
                    markers,
                    foreground_mask
                ) = watershed_from_logits(
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

                foreground_probability_image = (
                    foreground_probability[i, 0]
                    .cpu()
                    .numpy()
                )

                ap_results, map_value = (
                    mean_average_precision(
                        pred_instance_mask,
                        gt_instance_mask,
                        foreground_probability_image
                    )
                )

                results["map"].append(
                    float(map_value)
                )

                update_threshold_results(
                    results,
                    ap_results
                )

                error = count_error(
                    pred_instance_mask,
                    gt_instance_mask
                )

                results["count_error"].append(
                    float(error)
                )

                number_of_objects = len(
                    np.unique(
                        gt_instance_mask
                    )
                ) - 1

                results["density"].append(
                    {
                        "num_objects": int(
                            number_of_objects
                        ),
                        "map": float(map_value),
                        "count_error": float(error),
                    }
                )

    return summarize_results(results), results["density"]

def aggregate_by_density(
    baseline_density,
    boundary_density
):

    density_values = sorted(
        set(
            item["num_objects"]
            for item in baseline_density
        )
    )

    comparison = []

    for density in density_values:

        baseline_items = [
            item
            for item in baseline_density
            if item["num_objects"] == density
        ]

        boundary_items = [
            item
            for item in boundary_density
            if item["num_objects"] == density
        ]

        if len(baseline_items) == 0:
            continue

        if len(boundary_items) == 0:
            continue

        comparison.append(
            {
                "num_objects": density,
                "baseline_map": float(
                    np.mean(
                        [
                            item["map"]
                            for item in baseline_items
                        ]
                    )
                ),
                "boundary_map": float(
                    np.mean(
                        [
                            item["map"]
                            for item in boundary_items
                        ]
                    )
                ),
                "baseline_count_error": float(
                    np.mean(
                        [
                            item["count_error"]
                            for item in baseline_items
                        ]
                    )
                ),
                "boundary_count_error": float(
                    np.mean(
                        [
                            item["count_error"]
                            for item in boundary_items
                        ]
                    )
                ),
            }
        )

    return comparison


def save_density_plot(
    density_comparison,
    output_path
):

    if len(density_comparison) == 0:
        return

    density = [
        item["num_objects"]
        for item in density_comparison
    ]

    baseline_map = [
        item["baseline_map"]
        for item in density_comparison
    ]

    boundary_map = [
        item["boundary_map"]
        for item in density_comparison
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        density,
        baseline_map,
        marker="o",
        label="Baseline"
    )

    plt.plot(
        density,
        boundary_map,
        marker="o",
        label="Boundary + Watershed"
    )

    plt.xlabel(
        "Número de objetos na imagem"
    )

    plt.ylabel("mAP")

    plt.title(
        "mAP em função da densidade de objetos"
    )

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


def print_results(
    baseline_results,
    boundary_results
):

    print()
    print("=" * 60)
    print("BASELINE")
    print("=" * 60)

    print(
        f"Dice:              "
        f"{baseline_results['dice']:.4f}"
    )

    print(
        f"IoU:               "
        f"{baseline_results['iou']:.4f}"
    )

    print(
        f"mAP@[0.50:0.95]:   "
        f"{baseline_results['map']:.4f}"
    )

    print(
        f"Mean count error:   "
        f"{baseline_results['count_error']:.4f}"
    )

    print()
    print("=" * 60)
    print("BOUNDARY + WATERSHED")
    print("=" * 60)

    print(
        f"Dice:              "
        f"{boundary_results['dice']:.4f}"
    )

    print(
        f"IoU:               "
        f"{boundary_results['iou']:.4f}"
    )

    print(
        f"mAP@[0.50:0.95]:   "
        f"{boundary_results['map']:.4f}"
    )

    print(
        f"Mean count error:   "
        f"{boundary_results['count_error']:.4f}"
    )

    print()
    print("=" * 60)
    print("COMPARAÇÃO")
    print("=" * 60)

    print(
        f"{'Métrica':<25}"
        f"{'Baseline':>15}"
        f"{'Boundary + WS':>20}"
    )

    print("-" * 60)

    print(
        f"{'Dice':<25}"
        f"{baseline_results['dice']:>15.4f}"
        f"{boundary_results['dice']:>20.4f}"
    )

    print(
        f"{'IoU':<25}"
        f"{baseline_results['iou']:>15.4f}"
        f"{boundary_results['iou']:>20.4f}"
    )

    print(
        f"{'mAP':<25}"
        f"{baseline_results['map']:>15.4f}"
        f"{boundary_results['map']:>20.4f}"
    )

    print(
        f"{'Erro de contagem':<25}"
        f"{baseline_results['count_error']:>15.4f}"
        f"{boundary_results['count_error']:>20.4f}"
    )

    print("=" * 60)

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Dataset: {len(dataset)} imagens"
    )

    print()
    print("Carregando baseline...")

    baseline_model = load_model(
        BASELINE_CHECKPOINT,
        num_classes=1
    )

    print("Carregando Boundary + Watershed...")

    boundary_model = load_model(
        BOUNDARY_CHECKPOINT,
        num_classes=3
    )

    print()
    print("Avaliando baseline...")

    baseline_results, baseline_density = (
        evaluate_baseline(
            baseline_model
        )
    )

    print(
        "Avaliando Boundary + Watershed..."
    )

    boundary_results, boundary_density = (
        evaluate_boundary(
            boundary_model
        )
    )

    density_comparison = (
        aggregate_by_density(
            baseline_density,
            boundary_density
        )
    )

    results = {
        "baseline": baseline_results,
        "boundary_watershed": boundary_results,
        "density_comparison": density_comparison,
        "configuration": {
            "interior_threshold": (
                INTERIOR_THRESHOLD
            ),
            "foreground_threshold": (
                FOREGROUND_THRESHOLD
            ),
            "min_marker_size": (
                MIN_MARKER_SIZE
            ),
            "matching": (
                "greedy IoU-descending"
            ),
            "iou_thresholds": [
                round(
                    float(threshold),
                    2
                )
                for threshold in np.arange(
                    0.50,
                    0.951,
                    0.05
                )
            ],
        },
    }

    results_path = (
        OUTPUT_DIR / "results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    density_plot_path = (
        OUTPUT_DIR /
        "map_vs_object_density.png"
    )

    save_density_plot(
        density_comparison,
        density_plot_path
    )

    print_results(
        baseline_results,
        boundary_results
    )

    print()
    print(
        f"Resultados salvos em: "
        f"{results_path}"
    )

    print(
        f"Gráfico salvo em: "
        f"{density_plot_path}"
    )

if __name__ == "__main__":
    main()
