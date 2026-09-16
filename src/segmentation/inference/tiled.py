from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..models.unet import UNet
from ..postprocessing.watershed import watershed_from_logits

NUM_CLASSES = 3

INTERIOR_THRESHOLD = 0.5
FOREGROUND_THRESHOLD = 0.5
MIN_MARKER_SIZE = 10

DEFAULT_TILE_SIZE = 64
DEFAULT_OVERLAP = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def create_model():

    model = UNet(
        in_channels=1,
        num_classes=NUM_CLASSES
    )

    return model.to(DEVICE)


def load_model(checkpoint):
    """
    Carrega o checkpoint do U-Net Boundary + Watershed.
    """

    model = create_model()

    state_dict = torch.load(
        checkpoint,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model

def prepare_image(image):
    """
    Converte a imagem para tensor [1, 1, H, W].

    Aceita:
        - numpy [H, W]
        - torch [H, W]
        - torch [1, H, W]
        - torch [1, 1, H, W]
    """

    if isinstance(image, np.ndarray):

        tensor = torch.from_numpy(
            image
        ).float()

    elif isinstance(image, torch.Tensor):

        tensor = image.float()

    else:

        raise TypeError(
            "A imagem deve ser numpy.ndarray "
            "ou torch.Tensor."
        )

    if tensor.ndim == 2:

        tensor = tensor.unsqueeze(0)

    if tensor.ndim == 3:

        if tensor.shape[0] != 1:
            raise ValueError(
                "A imagem deve possuir um único canal."
            )

        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 4:

        raise ValueError(
            "Formato esperado: [H,W], "
            "[1,H,W] ou [1,1,H,W]."
        )

    return tensor

def generate_tile_positions(height, width, tile_size, overlap):
    """
    Gera as coordenadas dos tiles.

    Retorna tuplas:

        (y_start, y_end, x_start, x_end)

    Os tiles possuem sobreposição.
    """

    if overlap >= tile_size:

        raise ValueError(
            "overlap deve ser menor que tile_size."
        )

    stride = tile_size - overlap

    y_positions = list(
        range(
            0,
            max(height - tile_size, 0) + 1,
            stride
        )
    )

    x_positions = list(
        range(
            0,
            max(width - tile_size, 0) + 1,
            stride
        )
    )

    if len(y_positions) == 0:

        y_positions = [0]

    if len(x_positions) == 0:

        x_positions = [0]

    last_y = max(height - tile_size, 0)
    last_x = max(width - tile_size, 0)

    if y_positions[-1] != last_y:

        y_positions.append(last_y)

    if x_positions[-1] != last_x:

        x_positions.append(last_x)

    positions = []

    for y_start in y_positions:

        for x_start in x_positions:

            y_end = min(
                y_start + tile_size,
                height
            )

            x_end = min(
                x_start + tile_size,
                width
            )

            positions.append(
                (
                    y_start,
                    y_end,
                    x_start,
                    x_end
                )
            )

    return positions

def pad_tile(tile, tile_size):

    _, _, height, width = tile.shape

    pad_bottom = max(
        tile_size - height,
        0
    )

    pad_right = max(
        tile_size - width,
        0
    )

    if pad_bottom > 0 or pad_right > 0:

        tile = F.pad(
            tile,
            (
                0,
                pad_right,
                0,
                pad_bottom
            ),
            mode="reflect"
        )

    return (
        tile,
        height,
        width
    )

def predict_tile(model, tile):

    tile = prepare_image(tile)

    _, _, height, width = tile.shape

    padded_tile, original_height, original_width = (
        pad_tile(
            tile,
            max(height, width)
        )
    )

    with torch.no_grad():

        logits = model(
            padded_tile.to(DEVICE)
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    logits = logits[0]

    probabilities = probabilities[0]

    instance_mask, markers, foreground_mask = (
        watershed_from_logits(
            logits,
            interior_threshold=INTERIOR_THRESHOLD,
            foreground_threshold=FOREGROUND_THRESHOLD,
            min_marker_size=MIN_MARKER_SIZE
        )
    )

    foreground_probability = (
        1.0 -
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )

    instance_mask = instance_mask[
        :original_height,
        :original_width
    ]

    foreground_probability = foreground_probability[
        :original_height,
        :original_width
    ]

    probabilities = probabilities[
        :,
        :original_height,
        :original_width
    ].detach().cpu().numpy()

    return (instance_mask, foreground_probability, probabilities)

def tiled_inference(model, image, tile_size=DEFAULT_TILE_SIZE, overlap=DEFAULT_OVERLAP):
    """
    Executa inferência em uma imagem grande usando tiles sobrepostos.

    Cada instância de cada tile recebe inicialmente um ID próprio.

    Essa é a representação "ingênua" usada antes da fusão.

    Retorna um dicionário contendo:

        naive_instance_mask
        tile_predictions
        tile_positions
        foreground_probability
    """

    image = prepare_image(image)

    height = image.shape[-2]
    width = image.shape[-1]

    positions = generate_tile_positions(
        height,
        width,
        tile_size,
        overlap
    )

    naive_instance_mask = np.zeros(
        (height, width),
        dtype=np.int32
    )

    foreground_probability = np.zeros(
        (height, width),
        dtype=np.float32
    )

    tile_predictions = []

    next_instance_id = 1

    for tile_index, position in enumerate(
        positions
    ):

        y_start, y_end, x_start, x_end = position

        tile = image[
            :,
            :,
            y_start:y_end,
            x_start:x_end
        ]

        (
            instance_mask,
            tile_foreground,
            probabilities
        ) = predict_tile(
            model,
            tile
        )

        local_ids = np.unique(
            instance_mask
        )

        local_ids = local_ids[
            local_ids != 0
        ]

        local_to_global = {}

        for local_id in local_ids:

            global_id = next_instance_id

            next_instance_id += 1

            local_to_global[
                int(local_id)
            ] = global_id

            local_mask = (
                instance_mask ==
                local_id
            )

            global_y = (
                slice(
                    y_start,
                    y_end
                )
            )

            global_x = (
                slice(
                    x_start,
                    x_end
                )
            )

            region = naive_instance_mask[
                global_y,
                global_x
            ]

            region[local_mask] = global_id

            naive_instance_mask[
                global_y,
                global_x
            ] = region

        global_y = slice(
            y_start,
            y_end
        )

        global_x = slice(
            x_start,
            x_end
        )

        foreground_probability[
            global_y,
            global_x
        ] = np.maximum(
            foreground_probability[
                global_y,
                global_x
            ],
            tile_foreground
        )

        tile_predictions.append(
            {
                "tile_index": tile_index,
                "position": position,
                "instance_mask": instance_mask,
                "foreground_probability": tile_foreground,
                "probabilities": probabilities,
                "local_ids": [
                    int(value)
                    for value in local_ids
                ],
                "local_to_global": local_to_global
            }
        )

    return {
        "naive_instance_mask":
            naive_instance_mask,

        "foreground_probability":
            foreground_probability,

        "tile_predictions":
            tile_predictions,

        "tile_positions":
            positions
    }

def main():

    checkpoint = (
        "checkpoints/unet_boundary.pth"
    )

    model = load_model(
        checkpoint
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Modelo carregado."
    )

    print(
        "Use tiled_inference() "
        "para executar a inferência."
    )


if __name__ == "__main__":

    main()
