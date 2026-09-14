import json
from pathlib import Path

import numpy as np
import torch

from dataset_pytorch import SyntheticSegmentationDataset
from instance_metrics import (
    count_error,
    mean_average_precision
)

from inferencia import (
    load_model,
    tiled_inference,
    DEFAULT_TILE_SIZE,
    DEFAULT_OVERLAP
)

from instance_fusion import (
    fuse_instances,
    count_instances
)

DATA_DIR = "data/synthetic"

CHECKPOINT = (
    "checkpoints/unet_boundary.pth"
)

OUTPUT_DIR = Path(
    "output_dir/tiled"
)

RESULTS_PATH = (
    OUTPUT_DIR /
    "tiled_results.json"
)

TILE_SIZE = DEFAULT_TILE_SIZE
OVERLAP = DEFAULT_OVERLAP

FUSION_IOU_THRESHOLD = 0.40

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

def create_dataset():

    return SyntheticSegmentationDataset(
        DATA_DIR
    )

def evaluate_sample(model, image, ground_truth):
    """
    Executa tiled inference e calcula:

        mAP antes da fusão
        mAP depois da fusão
        erro de contagem antes
        erro de contagem depois
    """

    tiled_result = tiled_inference(
        model,
        image,
        tile_size=TILE_SIZE,
        overlap=OVERLAP
    )

    naive_mask = tiled_result[
        "naive_instance_mask"
    ]

    foreground_probability = tiled_result[
        "foreground_probability"
    ]

    (
        _,
        naive_map
    ) = mean_average_precision(
        naive_mask,
        ground_truth,
        foreground_probability
    )

    naive_count_error = count_error(
        naive_mask,
        ground_truth
    )

    (
        fused_mask,
        merge_candidates,
        groups
    ) = fuse_instances(
        tiled_result,
        iou_threshold=FUSION_IOU_THRESHOLD
    )

    (
        _,
        fused_map
    ) = mean_average_precision(
        fused_mask,
        ground_truth,
        foreground_probability
    )

    fused_count_error = count_error(
        fused_mask,
        ground_truth
    )

    return {
        "naive_map":
            float(naive_map),

        "fused_map":
            float(fused_map),

        "naive_count":
            count_instances(
                naive_mask
            ),

        "fused_count":
            count_instances(
                fused_mask
            ),

        "ground_truth_count":
            count_instances(
                ground_truth
            ),

        "naive_count_error":
            int(naive_count_error),

        "fused_count_error":
            int(fused_count_error),

        "num_merges":
            len(merge_candidates),

        "merge_candidates":
            merge_candidates,

        "num_groups":
            len(groups)
    }

def mean_std(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) == 0:

        return {
            "mean": 0.0,
            "std": 0.0
        }

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

def evaluate_dataset():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset = create_dataset()

    model = load_model(
        CHECKPOINT
    )

    print(
        "=" * 70
    )

    print(
        "AVALIAÇÃO — INFERÊNCIA EM MOSAICO"
    )

    print(
        "=" * 70
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Dataset: {len(dataset)} imagens"
    )

    print(
        f"Tile: {TILE_SIZE} x {TILE_SIZE}"
    )

    print(
        f"Overlap: {OVERLAP}"
    )

    print(
        f"IoU de fusão: "
        f"{FUSION_IOU_THRESHOLD}"
    )

    print()

    results = []

    for index in range(
        len(dataset)
    ):

        sample = dataset[index]

        image = sample[
            "image"
        ]

        ground_truth = sample[
            "instance_mask"
        ].numpy()

        result = evaluate_sample(
            model,
            image,
            ground_truth
        )

        result[
            "image_index"
        ] = index

        results.append(
            result
        )

        print(
            f"Imagem {index:02d} | "
            f"GT={result['ground_truth_count']:2d} | "
            f"naive={result['naive_count']:2d} | "
            f"fused={result['fused_count']:2d} | "
            f"mAP naive="
            f"{result['naive_map']:.4f} | "
            f"mAP fused="
            f"{result['fused_map']:.4f}"
        )

    naive_maps = [
        item["naive_map"]
        for item in results
    ]

    fused_maps = [
        item["fused_map"]
        for item in results
    ]

    naive_count_errors = [
        item["naive_count_error"]
        for item in results
    ]

    fused_count_errors = [
        item["fused_count_error"]
        for item in results
    ]

    summary = {
        "tile_size":
            TILE_SIZE,

        "overlap":
            OVERLAP,

        "fusion_iou_threshold":
            FUSION_IOU_THRESHOLD,

        "naive_map":
            mean_std(
                naive_maps
            ),

        "fused_map":
            mean_std(
                fused_maps
            ),

        "naive_count_error":
            mean_std(
                naive_count_errors
            ),

        "fused_count_error":
            mean_std(
                fused_count_errors
            )
    }

    output = {
        "summary":
            summary,

        "images":
            results
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()

    print(
        "=" * 70
    )

    print(
        "RESULTADO FINAL"
    )

    print(
        "=" * 70
    )

    print(
        "mAP antes da fusão: "
        f"{summary['naive_map']['mean']:.4f} "
        f"± "
        f"{summary['naive_map']['std']:.4f}"
    )

    print(
        "mAP depois da fusão: "
        f"{summary['fused_map']['mean']:.4f} "
        f"± "
        f"{summary['fused_map']['std']:.4f}"
    )

    improvement = (
        summary["fused_map"]["mean"]
        -
        summary["naive_map"]["mean"]
    )

    print(
        "Variação de mAP: "
        f"{improvement:+.4f}"
    )

    print()

    print(
        "Erro de contagem antes: "
        f"{summary['naive_count_error']['mean']:.4f} "
        f"± "
        f"{summary['naive_count_error']['std']:.4f}"
    )

    print(
        "Erro de contagem depois: "
        f"{summary['fused_count_error']['mean']:.4f} "
        f"± "
        f"{summary['fused_count_error']['std']:.4f}"
    )

    print()

    print(
        f"Resultados salvos em: "
        f"{RESULTS_PATH}"
    )

    return output

def main():

    evaluate_dataset()


if __name__ == "__main__":

    main()