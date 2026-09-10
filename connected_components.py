import numpy as np
from scipy import ndimage


def connected_components(binary_mask):
    """
    Recebe uma máscara binária [H, W].

    Retorna:
        instance_mask: máscara de instâncias [H, W]
        num_instances: número de instâncias encontradas

    Na máscara de saída:
        0 = background
        1 = primeira instância
        2 = segunda instância
        3 = terceira instância
        ...
    """

    # Garante que a entrada seja booleana
    binary_mask = binary_mask.astype(bool)

    # Define conectividade 8
    structure = np.ones((3, 3), dtype=np.uint8)

    # Identifica os componentes conectados
    instance_mask, num_instances = ndimage.label(
        binary_mask,
        structure=structure
    )

    return instance_mask.astype(np.int32), num_instances