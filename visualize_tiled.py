from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset_pytorch import SyntheticSegmentationDataset

from inferencia import (
    load_model,
    tiled_inference,
    DEFAULT_TILE_SIZE,
    DEFAULT_OVERLAP
)

from instance_fusion import (
    fuse_instances
)

DATA_DIR = "data/synthetic"

CHECKPOINT = (
    "checkpoints/unet_boundary.pth"
)

OUTPUT_DIR = Path(
    "output_dir/tiled"
)

FIGURE_PATH = (
    OUTPUT_DIR /
    "tiled_example.png"
)

TILE_SIZE = DEFAULT_TILE_SIZE
OVERLAP = DEFAULT_OVERLAP

FUSION_IOU_THRESHOLD = 0.40

SAMPLE_INDEX = 0

def instance_colors(instance_mask):
    """
    Cria uma imagem colorida para visualização das instâncias.

    Cada ID recebe uma cor determinística.
    """

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

def draw_tile_boundaries(
    axis,
    positions
):
    """
    Desenha as fronteiras dos tiles.
    """

    for index, position in enumerate(
        positions
    ):

        (
            y_start,
            y_end,
            x_start,
            x_end
        ) = position

        rectangle_x = [
            x_start,
            x_end,
            x_end,
            x_start,
            x_start
        ]

        rectangle_y = [
            y_start,
            y_start,
            y_end,
            y_end,
            y_start
        ]

        axis.plot(
            rectangle_x,
            rectangle_y,
            linewidth=1.5
        )

        axis.text(
            x_start + 2,
            y_start + 10,
            str(index),
            fontsize=8
        )

def find_boundary_instances(
    ground_truth,
    tile_size
):
    """
    Procura uma instância que cruza uma das fronteiras
    internas do grid de tiles.

    Retorna o primeiro caso encontrado.
    """

    height, width = (
        ground_truth.shape
    )

    ids = np.unique(
        ground_truth
    )

    ids = ids[
        ids != 0
    ]

    vertical_boundaries = list(
        range(
            tile_size,
            width,
            tile_size
        )
    )

    horizontal_boundaries = list(
        range(
            tile_size,
            height,
            tile_size
        )
    )

    for instance_id in ids:

        mask = (
            ground_truth ==
            instance_id
        )

        ys, xs = np.where(
            mask
        )

        if len(xs) == 0:
            continue

        min_x = xs.min()
        max_x = xs.max()

        min_y = ys.min()
        max_y = ys.max()

        crosses_vertical = any(
            min_x < boundary < max_x
            for boundary
            in vertical_boundaries
        )

        crosses_horizontal = any(
            min_y < boundary < max_y
            for boundary
            in horizontal_boundaries
        )

        if (
            crosses_vertical
            or
            crosses_horizontal
        ):

            return int(
                instance_id
            )

    return None

def create_figure(image, ground_truth, tiled_result, fused_mask):

    naive_mask = tiled_result[
        "naive_instance_mask"
    ]

    positions = tiled_result[
        "tile_positions"
    ]

    image = image.squeeze()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    axes[0, 0].imshow(
        image,
        cmap="gray"
    )

    axes[0, 0].set_title(
        "Mosaico"
    )

    draw_tile_boundaries(
        axes[0, 0],
        positions
    )

    axes[0, 1].imshow(
        instance_colors(
            ground_truth
        )
    )

    axes[0, 1].set_title(
        "Ground truth"
    )

    axes[0, 2].imshow(
        image,
        cmap="gray"
    )

    draw_tile_boundaries(
        axes[0, 2],
        positions
    )

    axes[0, 2].set_title(
        "Tiles sobrepostos"
    )

    axes[1, 0].imshow(
        instance_colors(
            naive_mask
        )
    )

    axes[1, 0].set_title(
        "Predição ingênua"
    )

    axes[1, 1].imshow(
        instance_colors(
            fused_mask
        )
    )

    axes[1, 1].set_title(
        "Predição após fusão"
    )

    difference = (
        naive_mask != fused_mask
    )

    axes[1, 2].imshow(
        difference,
        cmap="gray"
    )

    axes[1, 2].set_title(
        "Pixels alterados pela fusão"
    )

    for axis in axes.ravel():

        axis.axis(
            "off"
        )

    fig.suptitle(
        "Inferência em mosaico — "
        "Boundary + Watershed",
        fontsize=16
    )

    fig.tight_layout()

    return fig

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset = SyntheticSegmentationDataset(
        DATA_DIR
    )

    model = load_model(
        CHECKPOINT
    )

    sample = dataset[
        SAMPLE_INDEX
    ]

    image = sample[
        "image"
    ]

    ground_truth = sample[
        "instance_mask"
    ].numpy()

    print(
        "=" * 70
    )

    print(
        "VISUALIZAÇÃO — INFERÊNCIA EM MOSAICO"
    )

    print(
        "=" * 70
    )

    print(
        f"Imagem: {SAMPLE_INDEX}"
    )

    print(
        f"Tile size: {TILE_SIZE}"
    )

    print(
        f"Overlap: {OVERLAP}"
    )

    tiled_result = tiled_inference(
        model,
        image,
        tile_size=TILE_SIZE,
        overlap=OVERLAP
    )

    naive_mask = tiled_result[
        "naive_instance_mask"
    ]

    (
        fused_mask,
        merge_candidates,
        groups
    ) = fuse_instances(
        tiled_result,
        iou_threshold=FUSION_IOU_THRESHOLD
    )

    gt_count = len(
        np.unique(
            ground_truth
        )
    ) - 1

    naive_count = len(
        np.unique(
            naive_mask
        )
    ) - 1

    fused_count = len(
        np.unique(
            fused_mask
        )
    ) - 1

    print()

    print(
        f"GT: {gt_count} instâncias"
    )

    print(
        f"Predição ingênua: "
        f"{naive_count} instâncias"
    )

    print(
        f"Após fusão: "
        f"{fused_count} instâncias"
    )

    print(
        f"Fusões realizadas: "
        f"{len(merge_candidates)}"
    )

    boundary_instance = (
        find_boundary_instances(
            ground_truth,
            TILE_SIZE
        )
    )

    if boundary_instance is not None:

        print(
            f"Instância do GT que cruza "
            f"fronteira de tile: "
            f"{boundary_instance}"
        )

    else:

        print(
            "Nenhuma instância do GT "
            "foi encontrada cruzando "
            "uma fronteira exata do grid."
        )

    figure = create_figure(
        image,
        ground_truth,
        tiled_result,
        fused_mask
    )

    figure.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print()

    print(
        f"Figura salva em: "
        f"{FIGURE_PATH}"
    )


if __name__ == "__main__":

    main()