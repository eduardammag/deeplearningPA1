from pathlib import Path

import numpy as np
from PIL import Image


DATA_DIR = Path("data/synthetic")

IMAGE_SIZE = 128


def unet_receptive_field():
    """
    Calcula o campo receptivo teórico do caminho mais profundo
    da U-Net utilizada no trabalho.

    Arquitetura:
        4 blocos encoder
        4 MaxPool 2x2
        1 bottleneck
        2 convoluções 3x3 por bloco
    """

    receptive_field = 1
    jump = 1

    # Cada DoubleConv possui duas convoluções 3x3.
    for level in range(4):

        receptive_field += 2 * jump

        # MaxPool 2x2, stride 2.
        receptive_field += (2 - 1) * jump
        jump *= 2

    # Bottleneck.
    receptive_field += 2 * jump

    return receptive_field


def object_sizes(instance_mask):
    """
    Retorna medidas aproximadas de tamanho dos objetos.

    Para cada instância:
        - área
        - largura
        - altura
        - diâmetro equivalente
    """

    sizes = []

    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids != 0]

    for instance_id in instance_ids:

        ys, xs = np.where(
            instance_mask == instance_id
        )

        if len(xs) == 0:
            continue

        width = xs.max() - xs.min() + 1
        height = ys.max() - ys.min() + 1
        area = len(xs)

        equivalent_diameter = (
            2.0 * np.sqrt(area / np.pi)
        )

        sizes.append(
            {
                "area": int(area),
                "width": int(width),
                "height": int(height),
                "equivalent_diameter": float(
                    equivalent_diameter
                )
            }
        )

    return sizes


def analyze_dataset():

    all_sizes = []

    sample_count = 0

    for sample_dir in sorted(DATA_DIR.iterdir()):

        if not sample_dir.is_dir():
            continue

        instance_path = (
            sample_dir / "instance_mask.png"
        )

        if not instance_path.exists():
            continue

        instance_mask = np.array(
            Image.open(instance_path)
        )

        sizes = object_sizes(
            instance_mask
        )

        all_sizes.extend(sizes)
        sample_count += 1

    if not all_sizes:
        raise RuntimeError(
            "Nenhuma instância encontrada."
        )

    areas = np.array(
        [item["area"] for item in all_sizes],
        dtype=float
    )

    widths = np.array(
        [item["width"] for item in all_sizes],
        dtype=float
    )

    heights = np.array(
        [item["height"] for item in all_sizes],
        dtype=float
    )

    diameters = np.array(
        [
            item["equivalent_diameter"]
            for item in all_sizes
        ],
        dtype=float
    )

    print("=" * 70)
    print("ANÁLISE DO CAMPO RECEPTIVO E TAMANHO DOS OBJETOS")
    print("=" * 70)

    print()
    print(f"Imagens analisadas: {sample_count}")
    print(f"Instâncias analisadas: {len(all_sizes)}")

    print()
    print("CAMPO RECEPTIVO")
    print("-" * 70)

    receptive_field = (
        unet_receptive_field()
    )

    print(
        f"Campo receptivo teórico: "
        f"{receptive_field} x {receptive_field} pixels"
    )

    print(
        f"Tamanho da imagem: "
        f"{IMAGE_SIZE} x {IMAGE_SIZE} pixels"
    )

    print()
    print("TAMANHO DOS OBJETOS")
    print("-" * 70)

    print(
        f"Área média: {areas.mean():.2f}"
    )
    print(
        f"Área mediana: {np.median(areas):.2f}"
    )
    print(
        f"Área mínima: {areas.min():.2f}"
    )
    print(
        f"Área máxima: {areas.max():.2f}"
    )

    print()

    print(
        f"Largura média: {widths.mean():.2f}"
    )
    print(
        f"Largura máxima: {widths.max():.2f}"
    )

    print(
        f"Altura média: {heights.mean():.2f}"
    )
    print(
        f"Altura máxima: {heights.max():.2f}"
    )

    print()

    print(
        "Diâmetro equivalente médio: "
        f"{diameters.mean():.2f}"
    )

    print(
        "Diâmetro equivalente mediano: "
        f"{np.median(diameters):.2f}"
    )

    print(
        "Diâmetro equivalente máximo: "
        f"{diameters.max():.2f}"
    )

    print()
    print("INTERPRETAÇÃO")
    print("-" * 70)

    if receptive_field >= IMAGE_SIZE:
        print(
            "O campo receptivo teórico cobre a imagem inteira."
        )

    if receptive_field > diameters.max():
        print(
            "O campo receptivo é maior que o maior "
            "diâmetro equivalente observado."
        )

        print(
            "Portanto, o campo receptivo não parece "
            "ser o principal gargalo para este dataset."
        )

    print(
        "Os erros devem ser investigados principalmente "
        "na representação de interior/fronteira e "
        "na geração dos marcadores do watershed."
    )

    print()


if __name__ == "__main__":
    analyze_dataset()