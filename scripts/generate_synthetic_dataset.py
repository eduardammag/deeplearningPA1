"""Gera imagens sintéticas com elipses e suas máscaras de segmentação."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


SIZE = 128


def add_ellipse(mask: np.ndarray, label: int, rng: np.random.Generator) -> None:
    """Desenha uma elipse rotacionada na máscara, usando `label` como seu ID."""
    height, width = mask.shape
    center_x = rng.integers(5, width - 5)
    center_y = rng.integers(5, height - 5)
    radius_x = rng.integers(4, 16)
    radius_y = rng.integers(4, 16)
    angle = rng.uniform(0, 2 * np.pi)

    y, x = np.ogrid[:height, :width]
    dx, dy = x - center_x, y - center_y
    rotated_x = np.cos(angle) * dx + np.sin(angle) * dy
    rotated_y = -np.sin(angle) * dx + np.cos(angle) * dy
    ellipse = (rotated_x / radius_x) ** 2 + (rotated_y / radius_y) ** 2 <= 1
    mask[ellipse] = label


def create_example(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cria imagem em tons de cinza, máscara de instâncias e máscara semântica."""
    instances = np.zeros((SIZE, SIZE), dtype=np.uint16)
    number_of_objects = rng.integers(5, 21)

    for label in range(1, number_of_objects + 1):
        add_ellipse(instances, label, rng)

    # Fundo e objetos recebem intensidades e ruídos diferentes.
    image = rng.normal(rng.uniform(35, 80), rng.uniform(5, 15), instances.shape)
    for label in range(1, number_of_objects + 1):
        image[instances == label] = rng.uniform(130, 230)
    image += rng.normal(0, rng.uniform(2, 15), instances.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)

    semantic = (instances > 0).astype(np.uint8)
    return image, instances, semantic


def save_example(folder: Path, index: int, rng: np.random.Generator) -> None:
    """Gera e salva um exemplo em `folder/0000`, `folder/0001` etc."""
    image, instances, semantic = create_example(rng)
    example_folder = folder / f"{index:04d}"
    example_folder.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image).save(example_folder / "image.png")
    Image.fromarray(instances).save(example_folder / "instance_mask.png")
    Image.fromarray(semantic).save(example_folder / "semantic_mask.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera um dataset de elipses.")
    parser.add_argument("--output", default="data/synthetic", help="Pasta de saída.")
    parser.add_argument("--num-images", type=int, default=20, help="Quantidade de imagens.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    args = parser.parse_args()

    output = Path(args.output)
    rng = np.random.default_rng(args.seed)
    for index in range(args.num_images):
        save_example(output, index, rng)

    print(f"Dataset criado em: {output} ({args.num_images} imagens)")


if __name__ == "__main__":
    main()
