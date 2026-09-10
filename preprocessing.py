import torch


def normalize_image(image):
    """
    Normaliza uma imagem para o intervalo [0, 1].
    """

    image = image.float()

    if image.max() > 1.0:
        image = image / 255.0

    return image