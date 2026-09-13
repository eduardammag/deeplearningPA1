import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_pytorch import SyntheticSegmentationDataset
from unet import UNet
from segnet import SegNet
from watershed import watershed_from_logits

from semantic_metrics import (
    dice_score,
    iou_score
)

from instance_metrics import (
    count_error,
    mean_average_precision
)


# ============================================================
# Configurações
# ============================================================

DATA_DIR = "data/synthetic"

ABLATION_DIR = Path(
    "output_dir/ablations"
)

CHECKPOINT_DIR = (
    ABLATION_DIR /
    "checkpoints"
)

RESULTS_PATH = (
    ABLATION_DIR /
    "training_results.json"
)

EVALUATION_PATH = (
    ABLATION_DIR /
    "evaluation_results.json"
)

FIGURES_DIR = (
    ABLATION_DIR /
    "figures"
)

BATCH_SIZE = 4

NUM_CLASSES = 3

INTERIOR_THRESHOLD = 0.5
FOREGROUND_THRESHOLD = 0.5
MIN_MARKER_SIZE = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
        shuffle=False
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
            f"Arquitetura desconhecida: "
            f"{architecture}"
        )

    return model.to(DEVICE)


# ============================================================
# Carregar modelo
# ============================================================

def load_model(
    architecture,
    checkpoint
):

    model = create_model(
        architecture
    )

    state_dict = torch.load(
        checkpoint,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# Avaliação de uma configuração
# ============================================================

def evaluate_configuration(
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

            semantic_masks = batch[
                "semantic_mask"
            ].cpu().numpy()

            instance_masks = batch[
                "instance_mask"
            ].cpu().numpy()

            # ------------------------------------------------
            # Predição do modelo
            # ------------------------------------------------

            logits = model(images)

            # logits:
            # [B, 3, H, W]

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            # probabilities:
            # [B, 3, H, W]

            # ------------------------------------------------
            # Foreground
            # ------------------------------------------------

            foreground_probability = (
                1.0 -
                probabilities[:, 0:1]
            )

            # Shape:
            # [B, 1, H, W]

            # ------------------------------------------------
            # Logit binário para Dice/IoU
            # ------------------------------------------------
            #
            # semantic_metrics.py espera logits
            # binários e aplica sigmoid internamente.
            #
            # Portanto, convertemos a probabilidade
            # de foreground em logit.
            # ------------------------------------------------

            foreground_probability_clamped = (
                foreground_probability.clamp(
                    1e-6,
                    1.0 - 1e-6
                )
            )

            foreground_logit = torch.log(
                foreground_probability_clamped
                /
                (
                    1.0 -
                    foreground_probability_clamped
                )
            )

            # ------------------------------------------------
            # Predição semântica
            # ------------------------------------------------

            foreground_prediction = (
                foreground_probability >=
                FOREGROUND_THRESHOLD
            ).long()

            foreground_target = (
                torch.from_numpy(
                    semantic_masks
                )
                .unsqueeze(1)
                .to(DEVICE)
            )

            # ------------------------------------------------
            # Dice
            # ------------------------------------------------

            dice = dice_score(
                foreground_logit,
                foreground_target.float(),
                threshold=0.5
            )

            # ------------------------------------------------
            # IoU
            # ------------------------------------------------

            iou = iou_score(
                foreground_logit,
                foreground_target.float(),
                threshold=0.5
            )

            # ------------------------------------------------
            # Avaliação por imagem
            # ------------------------------------------------

            for index in range(
                images.shape[0]
            ):

                # --------------------------------------------
                # Logits de uma única imagem
                # --------------------------------------------

                sample_logits = logits[index]

                # Esperado:
                # [3, H, W]

                assert sample_logits.ndim == 3, (
                    "sample_logits deve possuir "
                    f"shape [3,H,W], mas recebeu "
                    f"{sample_logits.shape}"
                )

                assert sample_logits.shape[0] == NUM_CLASSES, (
                    "sample_logits deve possuir "
                    f"{NUM_CLASSES} classes, mas recebeu "
                    f"{sample_logits.shape[0]}"
                )

                # --------------------------------------------
                # Watershed
                # --------------------------------------------
                #
                # watershed_from_logits retorna:
                #
                # predicted_instances
                # markers
                # foreground_mask
                #
                # Todos com shape [H, W].
                # --------------------------------------------

                (
                    predicted_instances,
                    markers,
                    watershed_foreground
                ) = watershed_from_logits(
                    sample_logits,
                    interior_threshold=
                    INTERIOR_THRESHOLD,
                    foreground_threshold=
                    FOREGROUND_THRESHOLD,
                    min_marker_size=
                    MIN_MARKER_SIZE
                )

                # --------------------------------------------
                # Ground truth
                # --------------------------------------------

                ground_truth_instances = (
                    instance_masks[index]
                )

                # --------------------------------------------
                # Probabilidade de foreground
                # --------------------------------------------
                #
                # foreground_probability possui shape:
                #
                # [B, 1, H, W]
                #
                # Para uma imagem:
                #
                # foreground_probability[index, 0]
                #
                # resulta em:
                #
                # [H, W]
                #
                # Isso é exatamente o formato esperado
                # por instance_metrics.py.
                # --------------------------------------------

                sample_foreground_probability = (
                    foreground_probability[
                        index,
                        0
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                # --------------------------------------------
                # Verificações de shape
                # --------------------------------------------

                assert predicted_instances.ndim == 2, (
                    "predicted_instances deve possuir "
                    f"shape [H,W], mas recebeu "
                    f"{predicted_instances.shape}"
                )

                assert ground_truth_instances.ndim == 2, (
                    "ground_truth_instances deve possuir "
                    f"shape [H,W], mas recebeu "
                    f"{ground_truth_instances.shape}"
                )

                assert sample_foreground_probability.ndim == 2, (
                    "sample_foreground_probability deve possuir "
                    f"shape [H,W], mas recebeu "
                    f"{sample_foreground_probability.shape}"
                )

                assert (
                    predicted_instances.shape ==
                    ground_truth_instances.shape
                ), (
                    "predicted_instances e "
                    "ground_truth_instances possuem "
                    f"shapes diferentes: "
                    f"{predicted_instances.shape} vs "
                    f"{ground_truth_instances.shape}"
                )

                assert (
                    predicted_instances.shape ==
                    sample_foreground_probability.shape
                ), (
                    "predicted_instances e "
                    "sample_foreground_probability possuem "
                    f"shapes diferentes: "
                    f"{predicted_instances.shape} vs "
                    f"{sample_foreground_probability.shape}"
                )

                # --------------------------------------------
                # mAP
                # --------------------------------------------

                _, sample_map_value = mean_average_precision(
                    predicted_instances,
                    ground_truth_instances,
                    sample_foreground_probability
                )
                
                sample_count_error = (
                    count_error(
                        predicted_instances,
                        ground_truth_instances
                    )
                )

                map_values.append(
                    sample_map_value
                )

                count_errors.append(
                    sample_count_error
                )


            # ------------------------------------------------
            # Métricas semânticas do batch
            # ------------------------------------------------

            dice_values.append(
                float(dice)
            )

            iou_values.append(
                float(iou)
            )

    # ========================================================
    # Médias finais
    # ========================================================

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


# ============================================================
# Estatísticas
# ============================================================

def mean_std(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) == 1:

        return {
            "mean": float(values[0]),
            "std": 0.0
        }

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


# ============================================================
# Agrupamento
# ============================================================

def summarize_results(results):

    groups = {}

    for result in results:

        key = (
            result["axis"],
            result["architecture"],
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

        axis = key[0]
        architecture = key[1]
        loss_name = key[2]
        gamma = key[3]

        summary = {
            "axis": axis,

            "architecture": architecture,

            "loss": loss_name,

            "gamma": gamma,

            "seeds": [
                item["seed"]
                for item in group
            ],

            "dice": mean_std(
                [
                    item["dice"]
                    for item in group
                ]
            ),

            "iou": mean_std(
                [
                    item["iou"]
                    for item in group
                ]
            ),

            "map": mean_std(
                [
                    item["map"]
                    for item in group
                ]
            ),

            "count_error": mean_std(
                [
                    item["count_error"]
                    for item in group
                ]
            )
        }

        summaries.append(
            summary
        )

    return summaries


# ============================================================
# Gráfico do Eixo 1
# ============================================================

def plot_resolution_ablation(
    summaries
):

    data = [
        item
        for item in summaries
        if item["axis"] ==
        "resolution"
    ]

    if not data:

        return

    labels = []
    means = []
    stds = []

    for item in data:

        labels.append(
            item["architecture"]
        )

        means.append(
            item["map"]["mean"]
        )

        stds.append(
            item["map"]["std"]
        )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(8, 5)
    )

    x = np.arange(
        len(labels)
    )

    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=5
    )

    plt.xticks(
        x,
        labels
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


# ============================================================
# Gráfico do Eixo 2
# ============================================================

def plot_loss_ablation(
    summaries
):

    data = [
        item
        for item in summaries
        if item["axis"] ==
        "loss"
    ]

    if not data:

        return

    labels = []
    means = []
    stds = []

    for item in data:

        label = (
            f"{item['loss']}\n"
            f"γ={item['gamma']:g}"
        )

        labels.append(
            label
        )

        means.append(
            item["map"]["mean"]
        )

        stds.append(
            item["map"]["std"]
        )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(12, 6)
    )

    x = np.arange(
        len(labels)
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


# ============================================================
# Impressão
# ============================================================

def print_results(
    summaries
):

    print()
    print("=" * 80)
    print("RESULTADOS DAS ABLAÇÕES")
    print("=" * 80)

    for item in summaries:

        print()

        print(
            f"Eixo: "
            f"{item['axis']}"
        )

        print(
            f"Arquitetura: "
            f"{item['architecture']}"
        )

        print(
            f"Loss: "
            f"{item['loss']}"
        )

        print(
            f"Gamma: "
            f"{item['gamma']}"
        )

        print(
            f"Seeds: "
            f"{item['seeds']}"
        )

        print(
            f"Dice: "
            f"{item['dice']['mean']:.4f} "
            f"± "
            f"{item['dice']['std']:.4f}"
        )

        print(
            f"IoU: "
            f"{item['iou']['mean']:.4f} "
            f"± "
            f"{item['iou']['std']:.4f}"
        )

        print(
            f"mAP: "
            f"{item['map']['mean']:.4f} "
            f"± "
            f"{item['map']['std']:.4f}"
        )

        print(
            f"Erro de contagem: "
            f"{item['count_error']['mean']:.4f} "
            f"± "
            f"{item['count_error']['std']:.4f}"
        )


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

    # --------------------------------------------------------
    # Verificar resultados do treinamento
    # --------------------------------------------------------

    if not RESULTS_PATH.exists():

        raise FileNotFoundError(
            f"Arquivo de resultados do treinamento "
            f"não encontrado: {RESULTS_PATH}"
        )

    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        training_results = json.load(
            file
        )

    evaluated_results = []

    print()
    print(
        "Avaliando checkpoints..."
    )

    # --------------------------------------------------------
    # Avaliar cada checkpoint
    # --------------------------------------------------------

    for training_result in (
        training_results
    ):

        checkpoint = (
            training_result["checkpoint"]
        )

        architecture = (
            training_result["architecture"]
        )

        print(
            f"\nAvaliando: "
            f"{checkpoint}"
        )

        model = load_model(
            architecture,
            checkpoint
        )

        metrics = evaluate_configuration(
            model,
            loader
        )

        result = {
            **training_result,
            **metrics
        }

        evaluated_results.append(
            result
        )

    # --------------------------------------------------------
    # Resumo
    # --------------------------------------------------------

    summaries = summarize_results(
        evaluated_results
    )

    ABLATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Salvar resultados
    # --------------------------------------------------------

    with open(
        EVALUATION_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "runs": evaluated_results,
                "summary": summaries
            },
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Gráficos
    # --------------------------------------------------------

    plot_resolution_ablation(
        summaries
    )

    plot_loss_ablation(
        summaries
    )

    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print_results(
        summaries
    )

    print()
    print("=" * 80)

    print(
        f"Resultados salvos em: "
        f"{EVALUATION_PATH}"
    )

    print(
        f"Gráficos salvos em: "
        f"{FIGURES_DIR}"
    )


if __name__ == "__main__":

    main()