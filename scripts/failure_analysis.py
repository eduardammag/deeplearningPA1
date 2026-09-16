import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.segmentation.data.dataset import SyntheticSegmentationDataset
from src.segmentation.metrics.instance import (
    count_error,
    mean_average_precision
)
from src.segmentation.inference.receptive_field import (
    object_sizes,
    unet_receptive_field
)
from src.segmentation.models.unet import UNet
from src.segmentation.postprocessing.watershed import watershed_from_logits


DATA_DIR = Path("data/synthetic")

CHECKPOINT = (
    "checkpoints/unet_boundary.pth"
)

OUTPUT_DIR = Path(
    "output_dir/failure_analysis"
)

GALLERY_DIR = (
    OUTPUT_DIR / "gallery"
)

NUM_CLASSES = 3

INTERIOR_THRESHOLD = 0.5
FOREGROUND_THRESHOLD = 0.5

BASELINE_MIN_MARKER_SIZE = 10
CORRECTED_MIN_MARKER_SIZE = 3

NUM_FAILURES = 5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def load_model():

    model = UNet(
        in_channels=1,
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def instance_colors(instance_mask):

    height, width = (
        instance_mask.shape
    )

    output = np.zeros(
        (height, width, 3),
        dtype=float
    )

    ids = np.unique(
        instance_mask
    )

    ids = ids[
        ids != 0
    ]

    for instance_id in ids:

        rng = np.random.default_rng(
            int(instance_id)
        )

        color = rng.uniform(
            0.2,
            1.0,
            size=3
        )

        output[
            instance_mask == instance_id
        ] = color

    return output


def predict_sample(
    model,
    image,
    min_marker_size
):

    image = image.unsqueeze(0).to(
        DEVICE
    )

    with torch.no_grad():

        logits = model(
            image
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    (
        instance_mask,
        markers,
        foreground_mask
    ) = watershed_from_logits(
        logits[0],
        interior_threshold=(
            INTERIOR_THRESHOLD
        ),
        foreground_threshold=(
            FOREGROUND_THRESHOLD
        ),
        min_marker_size=(
            min_marker_size
        )
    )

    return (
        logits[0].cpu(),
        probabilities.cpu().numpy(),
        instance_mask,
        markers,
        foreground_mask
    )


def evaluate_sample(
    probabilities,
    instance_mask,
    ground_truth
):

    foreground_probability = (
        1.0 -
        probabilities[0]
    )

    _, map_value = (
        mean_average_precision(
            instance_mask,
            ground_truth,
            foreground_probability
        )
    )

    error = count_error(
        instance_mask,
        ground_truth
    )

    return (
        float(map_value),
        float(error)
    )


def create_failure_gallery(
    dataset,
    model
):

    failures = []

    for index in range(
        len(dataset)
    ):

        sample = dataset[index]

        image = sample["image"]

        ground_truth = (
            sample["instance_mask"]
            .numpy()
        )

        (
            logits,
            probabilities,
            instance_mask,
            markers,
            foreground_mask
        ) = predict_sample(
            model,
            image,
            BASELINE_MIN_MARKER_SIZE
        )

        map_value, error = (
            evaluate_sample(
                probabilities,
                instance_mask,
                ground_truth
            )
        )

        gt_count = len(
            np.unique(
                ground_truth
            )
        ) - 1

        pred_count = len(
            np.unique(
                instance_mask
            )
        ) - 1

        failures.append(
            {
                "index": index,
                "map": map_value,
                "count_error": error,
                "gt_count": int(gt_count),
                "pred_count": int(pred_count),
                "marker_count": int(
                    markers.max()
                )
            }
        )

    failures.sort(
        key=lambda item: item["map"]
    )

    selected = failures[
        :NUM_FAILURES
    ]

    GALLERY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for failure in selected:

        index = failure["index"]

        sample = dataset[index]

        image = (
            sample["image"]
            .squeeze(0)
            .numpy()
        )

        ground_truth = (
            sample["instance_mask"]
            .numpy()
        )

        (
            logits,
            probabilities,
            instance_mask,
            markers,
            foreground_mask
        ) = predict_sample(
            model,
            sample["image"],
            BASELINE_MIN_MARKER_SIZE
        )

        interior_probability = (
            probabilities[1]
        )

        boundary_probability = (
            probabilities[2]
        )

        figure, axes = plt.subplots(
            2,
            4,
            figsize=(14, 7)
        )

        axes[0, 0].imshow(
            image,
            cmap="gray"
        )
        axes[0, 0].set_title(
            "Imagem"
        )

        axes[0, 1].imshow(
            instance_colors(
                ground_truth
            )
        )
        axes[0, 1].set_title(
            f"GT — {failure['gt_count']} instâncias"
        )

        axes[0, 2].imshow(
            instance_colors(
                instance_mask
            )
        )
        axes[0, 2].set_title(
            f"Predição — {failure['pred_count']}"
        )

        axes[0, 3].imshow(
            boundary_probability
        )
        axes[0, 3].set_title(
            "Probabilidade de boundary"
        )

        axes[1, 0].imshow(
            interior_probability
        )
        axes[1, 0].set_title(
            "Probabilidade de interior"
        )

        axes[1, 1].imshow(
            markers,
            cmap="nipy_spectral"
        )
        axes[1, 1].set_title(
            f"Markers — {failure['marker_count']}"
        )

        axes[1, 2].imshow(
            foreground_mask,
            cmap="gray"
        )
        axes[1, 2].set_title(
            "Foreground"
        )

        axes[1, 3].axis(
            "off"
        )

        figure.suptitle(
            "Falha "
            f"{index:02d} — "
            f"mAP={failure['map']:.4f}, "
            f"erro de contagem="
            f"{failure['count_error']:.0f}"
        )

        for axis in axes.flat:
            axis.axis("off")

        plt.tight_layout()

        output_path = (
            GALLERY_DIR /
            f"failure_{index:02d}.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

    return failures, selected


def evaluate_correction(
    dataset,
    model
):

    results = []

    for index in range(
        len(dataset)
    ):

        sample = dataset[index]

        ground_truth = (
            sample["instance_mask"]
            .numpy()
        )

        (
            _,
            probabilities_before,
            mask_before,
            markers_before,
            _
        ) = predict_sample(
            model,
            sample["image"],
            BASELINE_MIN_MARKER_SIZE
        )

        (
            _,
            probabilities_after,
            mask_after,
            markers_after,
            _
        ) = predict_sample(
            model,
            sample["image"],
            CORRECTED_MIN_MARKER_SIZE
        )

        before_map, before_error = (
            evaluate_sample(
                probabilities_before,
                mask_before,
                ground_truth
            )
        )

        after_map, after_error = (
            evaluate_sample(
                probabilities_after,
                mask_after,
                ground_truth
            )
        )

        results.append(
            {
                "index": index,
                "map_before": before_map,
                "map_after": after_map,
                "count_error_before":
                    before_error,
                "count_error_after":
                    after_error,
                "markers_before":
                    int(markers_before.max()),
                "markers_after":
                    int(markers_after.max())
            }
        )

    return results


def summarize_correction(results):

    map_before = np.array(
        [
            item["map_before"]
            for item in results
        ]
    )

    map_after = np.array(
        [
            item["map_after"]
            for item in results
        ]
    )

    error_before = np.array(
        [
            item["count_error_before"]
            for item in results
        ]
    )

    error_after = np.array(
        [
            item["count_error_after"]
            for item in results
        ]
    )

    return {
        "map_before": {
            "mean": float(
                map_before.mean()
            ),
            "std": float(
                map_before.std(ddof=1)
            )
        },
        "map_after": {
            "mean": float(
                map_after.mean()
            ),
            "std": float(
                map_after.std(ddof=1)
            )
        },
        "count_error_before": {
            "mean": float(
                error_before.mean()
            ),
            "std": float(
                error_before.std(ddof=1)
            )
        },
        "count_error_after": {
            "mean": float(
                error_after.mean()
            ),
            "std": float(
                error_after.std(ddof=1)
            )
        },
        "delta_map": float(
            map_after.mean()
            -
            map_before.mean()
        )
    }


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset = (
        SyntheticSegmentationDataset(
            DATA_DIR
        )
    )

    model = load_model()

    print("=" * 70)
    print("PARTE 5 — GALERIA DE FALHAS")
    print("=" * 70)

    print(
        f"Campo receptivo teórico: "
        f"{unet_receptive_field()} x "
        f"{unet_receptive_field()}"
    )

    failures, selected = (
        create_failure_gallery(
            dataset,
            model
        )
    )

    print()
    print("Cinco piores imagens:")

    for item in selected:

        print(
            f"Imagem {item['index']:02d} | "
            f"mAP={item['map']:.4f} | "
            f"GT={item['gt_count']} | "
            f"pred={item['pred_count']} | "
            f"markers={item['marker_count']}"
        )

    print()
    print("=" * 70)
    print("CORREÇÃO — MINIMUM MARKER SIZE")
    print("=" * 70)

    print(
        f"Antes: {BASELINE_MIN_MARKER_SIZE}"
    )

    print(
        f"Depois: {CORRECTED_MIN_MARKER_SIZE}"
    )

    correction_results = (
        evaluate_correction(
            dataset,
            model
        )
    )

    summary = (
        summarize_correction(
            correction_results
        )
    )

    print()
    print(
        f"mAP antes: "
        f"{summary['map_before']['mean']:.4f} "
        f"± "
        f"{summary['map_before']['std']:.4f}"
    )

    print(
        f"mAP depois: "
        f"{summary['map_after']['mean']:.4f} "
        f"± "
        f"{summary['map_after']['std']:.4f}"
    )

    print(
        f"Δ mAP: "
        f"{summary['delta_map']:+.4f}"
    )

    print()

    print(
        f"Erro de contagem antes: "
        f"{summary['count_error_before']['mean']:.4f} "
        f"± "
        f"{summary['count_error_before']['std']:.4f}"
    )

    print(
        f"Erro de contagem depois: "
        f"{summary['count_error_after']['mean']:.4f} "
        f"± "
        f"{summary['count_error_after']['std']:.4f}"
    )

    output = {
        "receptive_field": (
            unet_receptive_field()
        ),
        "worst_cases": failures,
        "selected_failures": selected,
        "correction": {
            "before_min_marker_size":
                BASELINE_MIN_MARKER_SIZE,
            "after_min_marker_size":
                CORRECTED_MIN_MARKER_SIZE,
            "summary": summary,
            "per_image":
                correction_results
        }
    }

    results_path = (
        OUTPUT_DIR /
        "failure_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Resultados salvos em: "
        f"{results_path}"
    )

    print(
        f"Galeria salva em: "
        f"{GALLERY_DIR}"
    )


if __name__ == "__main__":
    main()
