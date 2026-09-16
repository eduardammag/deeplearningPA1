import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from src.segmentation.data.dataset import SyntheticSegmentationDataset
from src.segmentation.metrics.instance import mean_average_precision
from src.segmentation.models.unet import UNet
from src.segmentation.postprocessing.watershed import watershed_from_logits


DATA_DIR = Path("data/synthetic")

CHECKPOINT = (
    "checkpoints/unet_boundary.pth"
)

OUTPUT_DIR = Path(
    "output_dir/stress_test"
)

NUM_CLASSES = 3

INTERIOR_THRESHOLD = 0.5
FOREGROUND_THRESHOLD = 0.5
MIN_MARKER_SIZE = 10

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

    state_dict = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def normalize_image(image):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    if image.max() > 1.0:
        image = image / 255.0

    return np.clip(
        image,
        0.0,
        1.0
    )


def corrupt_blur(
    image,
    level
):

    sigma_values = {
        1: 0.8,
        2: 1.5,
        3: 2.5
    }

    sigma = sigma_values[level]

    return gaussian_filter(
        image,
        sigma=sigma
    )


def corrupt_noise(
    image,
    level
):

    standard_deviations = {
        1: 0.05,
        2: 0.10,
        3: 0.20
    }

    std = standard_deviations[level]

    rng = np.random.default_rng(
        1000 + level
    )

    noise = rng.normal(
        0.0,
        std,
        size=image.shape
    )

    return np.clip(
        image + noise,
        0.0,
        1.0
    )


def corrupt_brightness_contrast(
    image,
    level
):

    brightness = {
        1: 0.10,
        2: 0.20,
        3: 0.30
    }

    contrast = {
        1: 0.80,
        2: 0.60,
        3: 0.40
    }

    delta = brightness[level]
    factor = contrast[level]

    corrupted = (
        (image - 0.5) * factor
        + 0.5
        + delta
    )

    return np.clip(
        corrupted,
        0.0,
        1.0
    )


def predict(
    model,
    image
):

    tensor = torch.from_numpy(
        image
    ).float()

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        logits = model(
            tensor
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
            MIN_MARKER_SIZE
        )
    )

    foreground_probability = (
        1.0 -
        probabilities[0]
        .cpu()
        .numpy()
    )

    return (
        instance_mask,
        foreground_probability
    )


def evaluate_image(
    model,
    image,
    ground_truth
):

    (
        instance_mask,
        foreground_probability
    ) = predict(
        model,
        image
    )

    _, map_value = (
        mean_average_precision(
            instance_mask,
            ground_truth,
            foreground_probability
        )
    )

    return float(map_value)


def create_corruptions():

    return {
        "blur": corrupt_blur,
        "noise": corrupt_noise,
        "brightness_contrast":
            corrupt_brightness_contrast
    }


def evaluate():

    dataset = (
        SyntheticSegmentationDataset(
            DATA_DIR
        )
    )

    model = load_model()

    corruption_functions = (
        create_corruptions()
    )

    results = {}

    # ----------------------------------------------------------
    # Clean baseline
    # ----------------------------------------------------------

    clean_values = []

    for index in range(
        len(dataset)
    ):

        sample = dataset[index]

        image = normalize_image(
            sample["image"]
            .squeeze(0)
            .numpy()
        )

        ground_truth = (
            sample["instance_mask"]
            .numpy()
        )

        clean_values.append(
            evaluate_image(
                model,
                image,
                ground_truth
            )
        )

    results["clean"] = {
        "mean": float(
            np.mean(clean_values)
        ),
        "std": float(
            np.std(
                clean_values,
                ddof=1
            )
        )
    }

    # ----------------------------------------------------------
    # Corrupted images
    # ----------------------------------------------------------

    for corruption_name, function in (
        corruption_functions.items()
    ):

        results[corruption_name] = {}

        for level in [1, 2, 3]:

            values = []

            for index in range(
                len(dataset)
            ):

                sample = dataset[index]

                image = normalize_image(
                    sample["image"]
                    .squeeze(0)
                    .numpy()
                )

                ground_truth = (
                    sample["instance_mask"]
                    .numpy()
                )

                corrupted = function(
                    image,
                    level
                )

                value = evaluate_image(
                    model,
                    corrupted,
                    ground_truth
                )

                values.append(
                    value
                )

            results[
                corruption_name
            ][str(level)] = {
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

    return results


def plot_results(
    results
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    clean_map = (
        results["clean"]["mean"]
    )

    for corruption_name in [
        "blur",
        "noise",
        "brightness_contrast"
    ]:

        levels = [1, 2, 3]

        means = [
            results[
                corruption_name
            ][str(level)]["mean"]
            for level in levels
        ]

        stds = [
            results[
                corruption_name
            ][str(level)]["std"]
            for level in levels
        ]

        plt.figure(
            figsize=(7, 5)
        )

        plt.axhline(
            clean_map,
            linestyle="--",
            label="Clean"
        )

        plt.errorbar(
            levels,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=corruption_name
        )

        plt.xticks(
            levels,
            [
                "Intensidade 1",
                "Intensidade 2",
                "Intensidade 3"
            ]
        )

        plt.xlabel(
            "Nível de corrupção"
        )

        plt.ylabel(
            "mAP@[0.50:0.95]"
        )

        plt.title(
            f"Teste de estresse — "
            f"{corruption_name}"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR /
            f"{corruption_name}.png",
            dpi=200
        )

        plt.close()


def main():

    print("=" * 70)
    print("PARTE 6 — TESTE DE ESTRESSE")
    print("=" * 70)

    results = evaluate()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = (
        OUTPUT_DIR /
        "stress_results.json"
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

    plot_results(
        results
    )

    print()
    print(
        f"mAP clean: "
        f"{results['clean']['mean']:.4f}"
    )

    for corruption in [
        "blur",
        "noise",
        "brightness_contrast"
    ]:

        print()
        print(
            corruption.upper()
        )

        for level in [1, 2, 3]:

            value = results[
                corruption
            ][str(level)]

            degradation = (
                value["mean"]
                -
                results["clean"]["mean"]
            )

            print(
                f"  nível {level}: "
                f"mAP={value['mean']:.4f} "
                f"± {value['std']:.4f} | "
                f"Δ={degradation:+.4f}"
            )

    print()
    print(
        f"Resultados: {results_path}"
    )


if __name__ == "__main__":
    main()
