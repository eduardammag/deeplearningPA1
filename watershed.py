import numpy as np
import torch

from scipy import ndimage
from skimage.segmentation import watershed


# --------------------------------------------------
# Classes da saída da U-Net
# --------------------------------------------------

BACKGROUND = 0
INTERIOR = 1
BOUNDARY = 2


# --------------------------------------------------
# Componentes auxiliares
# --------------------------------------------------

def remove_small_components(
    mask,
    min_size=10
):
    """
    Remove componentes conexos menores que min_size.

    É utilizada para evitar que pequenos ruídos na
    previsão do interior sejam interpretados como
    novos marcadores pelo watershed.

    Parâmetros
    ----------
    mask : numpy.ndarray
        Máscara binária.

    min_size : int
        Tamanho mínimo, em pixels, para manter uma
        componente conexa.

    Retorno
    -------
    numpy.ndarray
        Máscara binária sem componentes pequenas.
    """

    mask = np.asarray(mask).astype(bool)

    if min_size < 1:
        raise ValueError(
            "min_size deve ser maior ou igual a 1."
        )

    labels, number_of_labels = ndimage.label(mask)

    if number_of_labels == 0:
        return np.zeros_like(mask, dtype=bool)

    component_sizes = np.bincount(
        labels.ravel()
    )

    keep = component_sizes >= min_size

    keep[0] = False

    cleaned_mask = keep[labels]

    return cleaned_mask.astype(bool)


def create_markers(
    interior_probability,
    threshold=0.5,
    min_size=10
):
    """
    Cria marcadores para o watershed a partir da
    probabilidade prevista para a classe interior.

    Cada componente conexa do interior corresponde
    a um possível objeto.

    Parâmetros
    ----------
    interior_probability : numpy.ndarray
        Probabilidade da classe interior.

    threshold : float
        Limiar utilizado para selecionar pixels
        considerados interior.

    min_size : int
        Tamanho mínimo de um marcador.

    Retorno
    -------
    markers : numpy.ndarray
        Máscara inteira em que:
            0 = sem marcador
            1, 2, ... = marcadores individuais

    number_of_markers : int
        Número de marcadores encontrados.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold deve estar entre 0 e 1."
        )

    interior_mask = (
        interior_probability >= threshold
    )

    interior_mask = remove_small_components(
        interior_mask,
        min_size=min_size
    )

    markers, number_of_markers = ndimage.label(
        interior_mask
    )

    return (
        markers.astype(np.int32),
        number_of_markers
    )


# --------------------------------------------------
# Watershed
# --------------------------------------------------

def watershed_from_probabilities(
    probabilities,
    interior_threshold=0.5,
    foreground_threshold=0.5,
    min_marker_size=10
):
    """
    Decodifica as probabilidades das três classes
    em uma máscara de instâncias utilizando watershed.

    Classes:
        0 = background
        1 = interior
        2 = boundary

    O interior previsto pela rede é utilizado como
    marcador para cada possível instância.

    A probabilidade de boundary é utilizada como
    mapa de elevação do watershed.

    Parâmetros
    ----------
    probabilities : numpy.ndarray
        Array com shape [3, H, W], contendo as
        probabilidades das três classes.

    interior_threshold : float
        Limiar para selecionar o interior.

    foreground_threshold : float
        Limiar para selecionar o foreground.

    min_marker_size : int
        Tamanho mínimo de um marcador.

    Retorno
    -------
    instance_mask : numpy.ndarray
        Máscara de instâncias com shape [H, W].
        0 representa background e cada número positivo
        representa uma instância.

    markers : numpy.ndarray
        Marcadores utilizados pelo watershed.

    foreground_mask : numpy.ndarray
        Máscara do foreground utilizada pelo watershed.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    if probabilities.ndim != 3:
        raise ValueError(
            "probabilities deve possuir shape [3, H, W]."
        )

    if probabilities.shape[0] != 3:
        raise ValueError(
            "A saída deve possuir exatamente 3 classes."
        )

    if not 0.0 <= interior_threshold <= 1.0:
        raise ValueError(
            "interior_threshold deve estar entre 0 e 1."
        )

    if not 0.0 <= foreground_threshold <= 1.0:
        raise ValueError(
            "foreground_threshold deve estar entre 0 e 1."
        )

    background_probability = probabilities[
        BACKGROUND
    ]

    interior_probability = probabilities[
        INTERIOR
    ]

    boundary_probability = probabilities[
        BOUNDARY
    ]

    # ----------------------------------------------
    # Foreground
    # ----------------------------------------------

    foreground_probability = (
        1.0 - background_probability
    )

    foreground_mask = (
        foreground_probability >= foreground_threshold
    )

    # ----------------------------------------------
    # Marcadores
    # ----------------------------------------------

    markers, number_of_markers = create_markers(
        interior_probability,
        threshold=interior_threshold,
        min_size=min_marker_size
    )

    # Se não houver marcadores, não é possível
    # realizar o watershed.
    if number_of_markers == 0:

        instance_mask = np.zeros(
            foreground_mask.shape,
            dtype=np.int32
        )

        return (
            instance_mask,
            markers,
            foreground_mask
        )

    # ----------------------------------------------
    # Mapa de elevação
    # ----------------------------------------------

    elevation = boundary_probability

    # ----------------------------------------------
    # Watershed
    # ----------------------------------------------

    instance_mask = watershed(
        elevation,
        markers=markers,
        mask=foreground_mask
    )

    instance_mask = instance_mask.astype(
        np.int32
    )

    return (
        instance_mask,
        markers,
        foreground_mask
    )


def watershed_from_logits(
    logits,
    interior_threshold=0.5,
    foreground_threshold=0.5,
    min_marker_size=10
):
    """
    Converte diretamente os logits da U-Net em uma
    máscara de instâncias utilizando watershed.

    Parâmetros
    ----------
    logits : torch.Tensor
        Tensor com shape [3, H, W].

    interior_threshold : float
        Limiar para selecionar o interior.

    foreground_threshold : float
        Limiar para selecionar o foreground.

    min_marker_size : int
        Tamanho mínimo de um marcador.

    Retorno
    -------
    instance_mask : numpy.ndarray
        Máscara de instâncias.

    markers : numpy.ndarray
        Marcadores utilizados.

    foreground_mask : numpy.ndarray
        Máscara de foreground.
    """

    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            "logits deve ser um torch.Tensor."
        )

    if logits.ndim != 3:
        raise ValueError(
            "logits deve possuir shape [3, H, W]."
        )

    if logits.shape[0] != 3:
        raise ValueError(
            "A U-Net deve produzir exatamente 3 classes."
        )

    probabilities = torch.softmax(
        logits,
        dim=0
    )

    probabilities = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    return watershed_from_probabilities(
        probabilities,
        interior_threshold=interior_threshold,
        foreground_threshold=foreground_threshold,
        min_marker_size=min_marker_size
    )