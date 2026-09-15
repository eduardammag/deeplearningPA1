# evaluate.py

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from connected_components import connected_components
from dataset_bbbc038 import BBBC038Dataset
from instance_metrics import count_error, mean_average_precision
from semantic_metrics import dice_score, iou_score
from unet import UNet


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DATA_ROOT = "data/BBBC038"

CHECKPOINT_PATH = "checkpoints/unet_baseline_bbbc038.pth"

SPLIT_FILE = "data/BBBC038/splits.json"

OUTPUT_DIR = Path("output_dir/evaluation_bbbc038")

BATCH_SIZE = 4

FOREGROUND_THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

dataset = BBBC038Dataset(
    root_dir=DATA_ROOT,
    split="test",
    split_file=SPLIT_FILE,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# MODELO
# ============================================================

def load_model():
    """
    Cria a mesma arquitetura utilizada no treinamento
    e carrega o melhor checkpoint salvo.
    """

    model = UNet(
        in_channels=1,
        num_classes=1,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# MÉTRICAS SEMÂNTICAS
# ============================================================

def binary_dice(prediction, target):
    """
    Calcula Dice para máscaras binárias.
    """

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    denominator = prediction.sum() + target.sum()

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection / denominator
    )


def binary_iou(prediction, target):
    """
    Calcula IoU para máscaras binárias.
    """

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    union = np.logical_or(
        prediction,
        target,
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


# ============================================================
# RESULTADOS
# ============================================================

def create_results():
    return {
        "dice": [],
        "iou": [],
        "map": [],
        "count_error": [],
        "density": [],
        "map_by_threshold": {},
    }


def update_threshold_results(
    results,
    ap_results,
):
    """
    Armazena o AP de cada limiar de IoU.

    Os limiares esperados são:
    0.50, 0.55, ..., 0.95
    """

    for threshold, ap in ap_results.items():

        threshold = str(threshold)

        if threshold not in results["map_by_threshold"]:
            results["map_by_threshold"][threshold] = []

        results["map_by_threshold"][threshold].append(
            float(ap)
        )


# ============================================================
# RESUMO
# ============================================================

def summarize_results(results):

    mean_map_by_threshold = {}

    for threshold, values in results[
        "map_by_threshold"
    ].items():

        mean_map_by_threshold[threshold] = float(
            np.mean(values)
        )

    return {
        "dice": float(
            np.mean(results["dice"])
        ),
        "iou": float(
            np.mean(results["iou"])
        ),
        "map": float(
            np.mean(results["map"])
        ),
        "count_error": float(
            np.mean(results["count_error"])
        ),
        "map_by_threshold": mean_map_by_threshold,
    }


# ============================================================
# AVALIAÇÃO
# ============================================================

def evaluate(model):

    results = create_results()

    with torch.no_grad():

        for batch_index, batch in enumerate(loader):

            images = batch["image"].to(DEVICE)

            semantic_masks = (
                batch["semantic_mask"]
                .cpu()
                .numpy()
            )

            gt_instance_masks = (
                batch["instance_mask"]
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Predição semântica
            # ------------------------------------------------

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            binary_predictions = (
                probabilities
                >= FOREGROUND_THRESHOLD
            )

            # ------------------------------------------------
            # Avaliação de cada imagem
            # ------------------------------------------------

            for i in range(images.shape[0]):

                prediction = (
                    binary_predictions[i, 0]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                probability = (
                    probabilities[i, 0]
                    .cpu()
                    .numpy()
                )

                gt_semantic = (
                    semantic_masks[i]
                    .astype(np.uint8)
                )

                gt_instance = (
                    gt_instance_masks[i]
                    .astype(np.int32)
                )

                # ====================================================
                # Dice
                # ====================================================

                dice = binary_dice(
                    prediction,
                    gt_semantic,
                )

                # ====================================================
                # IoU
                # ====================================================

                iou = binary_iou(
                    prediction,
                    gt_semantic,
                )

                results["dice"].append(
                    float(dice)
                )

                results["iou"].append(
                    float(iou)
                )

                # ====================================================
                # Connected Components
                # ====================================================

                pred_instance, num_instances = (
                    connected_components(
                        prediction
                    )
                )

                # ====================================================
                # Instance AP / mAP
                # ====================================================

                ap_results, map_value = (
                    mean_average_precision(
                        pred_instance,
                        gt_instance,
                        probability,
                    )
                )

                results["map"].append(
                    float(map_value)
                )

                update_threshold_results(
                    results,
                    ap_results,
                )

                # ====================================================
                # Erro absoluto de contagem
                # ====================================================

                error = count_error(
                    pred_instance,
                    gt_instance,
                )

                results["count_error"].append(
                    float(error)
                )

                # ====================================================
                # Densidade de objetos
                # ====================================================

                gt_object_ids = np.unique(
                    gt_instance
                )

                gt_object_ids = gt_object_ids[
                    gt_object_ids != 0
                ]

                number_of_objects = len(
                    gt_object_ids
                )

                image_area = (
                    gt_instance.shape[0]
                    * gt_instance.shape[1]
                )

                object_density = (
                    number_of_objects
                    / image_area
                )

                results["density"].append(
                    {
                        "num_objects": int(
                            number_of_objects
                        ),
                        "density": float(
                            object_density
                        ),
                        "map": float(map_value),
                        "count_error": float(error),
                    }
                )

            print(
                f"Avaliadas "
                f"{min((batch_index + 1) * BATCH_SIZE, len(dataset))}"
                f"/{len(dataset)} imagens",
                end="\r",
            )

    print()

    summary = summarize_results(
        results
    )

    return summary, results["density"]


# ============================================================
# AGREGAÇÃO POR DENSIDADE
# ============================================================

def aggregate_by_density(
    density_results
):
    """
    Agrupa as imagens pelo número de objetos
    presentes no ground truth.

    Como todas as imagens foram normalizadas
    para 256x256, número de objetos por imagem
    é proporcional à densidade de objetos.
    """

    density_values = sorted(
        set(
            item["num_objects"]
            for item in density_results
        )
    )

    comparison = []

    for density in density_values:

        items = [
            item
            for item in density_results
            if item["num_objects"] == density
        ]

        if len(items) == 0:
            continue

        comparison.append(
            {
                "num_objects": int(
                    density
                ),
                "density": float(
                    items[0]["density"]
                ),
                "map": float(
                    np.mean(
                        [
                            item["map"]
                            for item in items
                        ]
                    )
                ),
                "count_error": float(
                    np.mean(
                        [
                            item["count_error"]
                            for item in items
                        ]
                    )
                ),
                "num_images": len(items),
            }
        )

    return comparison


# ============================================================
# GRÁFICO: mAP × DENSIDADE
# ============================================================

def save_density_plot(
    density_comparison,
    output_path,
):

    if len(density_comparison) == 0:
        return

    density = [
        item["num_objects"]
        for item in density_comparison
    ]

    map_values = [
        item["map"]
        for item in density_comparison
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        density,
        map_values,
        marker="o",
    )

    plt.xlabel(
        "Número de objetos na imagem"
    )

    plt.ylabel(
        "mAP@[0.50:0.95]"
    )

    plt.title(
        "mAP em função da densidade de objetos"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# GRÁFICO: ERRO DE CONTAGEM × DENSIDADE
# ============================================================

def save_count_error_plot(
    density_comparison,
    output_path,
):

    if len(density_comparison) == 0:
        return

    density = [
        item["num_objects"]
        for item in density_comparison
    ]

    count_error_values = [
        item["count_error"]
        for item in density_comparison
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        density,
        count_error_values,
        marker="o",
    )

    plt.xlabel(
        "Número de objetos na imagem"
    )

    plt.ylabel(
        "Erro absoluto médio de contagem"
    )

    plt.title(
        "Erro de contagem em função da densidade de objetos"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# IMPRESSÃO DOS RESULTADOS
# ============================================================

def print_results(
    results,
):

    print()
    print("=" * 60)
    print("BBBC038 - BASELINE SEMÂNTICO")
    print("=" * 60)

    print(
        f"Dice:              "
        f"{results['dice']:.4f}"
    )

    print(
        f"IoU:               "
        f"{results['iou']:.4f}"
    )

    print(
        f"mAP@[0.50:0.95]:   "
        f"{results['map']:.4f}"
    )

    print(
        f"Mean count error:   "
        f"{results['count_error']:.4f}"
    )

    print()
    print("AP POR LIMIAR DE IoU")
    print("-" * 60)

    for threshold, ap in sorted(
        results["map_by_threshold"].items(),
        key=lambda x: float(x[0]),
    ):

        print(
            f"AP@{float(threshold):.2f}: "
            f"{ap:.4f}"
        )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("AVALIAÇÃO - BBBC038")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Test images: {len(dataset)}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print()

    # --------------------------------------------------------
    # Carrega modelo
    # --------------------------------------------------------

    print(
        "Carregando modelo..."
    )

    model = load_model()

    print(
        "Modelo carregado."
    )

    print()

    # --------------------------------------------------------
    # Avaliação
    # --------------------------------------------------------

    print(
        "Avaliando baseline..."
    )

    summary, density_results = evaluate(
        model
    )

    # --------------------------------------------------------
    # Agregação por densidade
    # --------------------------------------------------------

    density_comparison = (
        aggregate_by_density(
            density_results
        )
    )

    # --------------------------------------------------------
    # Resultados finais
    # --------------------------------------------------------

    results = {
        "dataset": "BBBC038",
        "task": "Part 1 - Semantic baseline",
        "model": "U-Net",
        "checkpoint": CHECKPOINT_PATH,
        "threshold": FOREGROUND_THRESHOLD,
        "matching": "greedy IoU-descending",
        "iou_thresholds": [
            round(
                float(threshold),
                2,
            )
            for threshold in np.arange(
                0.50,
                0.951,
                0.05,
            )
        ],
        "metrics": summary,
        "density_analysis": density_comparison,
    }

    # --------------------------------------------------------
    # Salva JSON
    # --------------------------------------------------------

    results_path = (
        OUTPUT_DIR
        / "baseline_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Salva gráfico mAP × densidade
    # --------------------------------------------------------

    density_plot_path = (
        OUTPUT_DIR
        / "map_vs_object_density.png"
    )

    save_density_plot(
        density_comparison,
        density_plot_path,
    )

    # --------------------------------------------------------
    # Salva gráfico erro × densidade
    # --------------------------------------------------------

    count_error_plot_path = (
        OUTPUT_DIR
        / "count_error_vs_object_density.png"
    )

    save_count_error_plot(
        density_comparison,
        count_error_plot_path,
    )

    # --------------------------------------------------------
    # Imprime resultados
    # --------------------------------------------------------

    print_results(
        summary
    )

    print()

    print(
        f"Resultados salvos em: "
        f"{results_path}"
    )

    print(
        f"Gráfico mAP × densidade salvo em: "
        f"{density_plot_path}"
    )

    print(
        f"Gráfico erro × densidade salvo em: "
        f"{count_error_plot_path}"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()