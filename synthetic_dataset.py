"""
Gera o dataset sintético. Cada imagem possui:
- tamanho 128 x 128;
- entre 5 e 20 elipses;
- elipses de tamanhos e orientações variadas;
- ruído e contraste variáveis;
- máscara de instância, na qual cada objeto possui um ID diferente;
- máscara semântica, na qual qualquer objeto recebe o valor 1.
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

IMAGE_SIZE = 128

def draw_random_ellipse(
    instance_mask: np.ndarray,
    instance_id: int,
    rng: np.random.Generator,
) -> None:
    """Desenha uma elipse preenchida na máscara de instâncias.
    A equação usada é:
        ((x - cx) / rx)^2 + ((y - cy) / ry)^2 <= 1
    onde (cx, cy) é o centro e rx/ry são os semi-eixos.
    """

    h, w = instance_mask.shape

    cx = rng.integers(5, w - 5)
    cy = rng.integers(5, h - 5)

    rx = rng.integers(4, 16)
    ry = rng.integers(4, 16)

    theta = rng.uniform(0, 2 * np.pi)

    # Grade de coordenadas.
    y, x = np.ogrid[:h, :w]

    # Translação para o centro da elipse.
    x_shifted = x - cx
    y_shifted = y - cy

    # Rotação das coordenadas.
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_rot = cos_theta * x_shifted + sin_theta * y_shifted
    y_rot = -sin_theta * x_shifted + cos_theta * y_shifted

    ellipse = (x_rot / rx) ** 2 + (y_rot / ry) ** 2 <= 1.0

    # Se houver sobreposição, a instância mais recente sobrescreve a anterior.
    # Isso é aceitável nesta primeira versão porque o objetivo é testar o pipeline.
    instance_mask[ellipse] = instance_id


def generate_example(rng: np.random.Generator):
    """Gera uma imagem e suas duas máscaras."""

    instance_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16)

    # O enunciado pede 5–20 objetos.
    num_instances = int(rng.integers(5, 21))

    for instance_id in range(1, num_instances + 1):
        draw_random_ellipse(instance_mask, instance_id, rng)

    # Máscara semântica: qualquer pixel diferente de zero é objeto.
    semantic_mask = (instance_mask > 0).astype(np.uint8)

    # Gera intensidade de fundo.
    background = rng.normal(
        loc=rng.uniform(35, 80),
        scale=rng.uniform(5, 15),
        size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    # Cada instância recebe uma intensidade ligeiramente diferente.
    image = background.copy()

    for instance_id in range(1, num_instances + 1):
        value = rng.uniform(130, 230)
        image[instance_mask == instance_id] = value

    # Ruído de aquisição.
    noise_sigma = rng.uniform(2, 15)
    noise = rng.normal(0, noise_sigma, size=image.shape)
    image = image + noise

    # Contraste variável.
    contrast = rng.uniform(0.75, 1.25)
    image = 128 + contrast * (image - 128)

    image = np.clip(image, 0, 255).astype(np.uint8)

    return image, instance_mask, semantic_mask


def save_example(
    output_dir: Path,
    index: int,
    image: np.ndarray,
    instance_mask: np.ndarray,
    semantic_mask: np.ndarray,
):
    """Salva um exemplo em uma pasta própria."""

    example_dir = output_dir / f"{index:04d}"
    example_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image).save(example_dir / "image.png")

    # PNG suporta uint16, preservando os IDs das instâncias.
    Image.fromarray(instance_mask).save(example_dir / "instance_mask.png")

    Image.fromarray(semantic_mask).save(example_dir / "semantic_mask.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/synthetic")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    first_example = None

    for index in range(args.num_images):
        image, instance_mask, semantic_mask = generate_example(rng)

        save_example(
            output_dir,
            index,
            image,
            instance_mask,
            semantic_mask,
        )

        if first_example is None:
            first_example = (image, instance_mask, semantic_mask)

    print(f"Dataset criado em: {output_dir}")
    print(f"Número de imagens: {args.num_images}")
    print(f"Seed: {args.seed}")

    if args.show:
        import matplotlib.pyplot as plt

        image, instance_mask, semantic_mask = first_example

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Imagem")

        axes[1].imshow(instance_mask)
        axes[1].set_title("Máscara de instância")

        axes[2].imshow(semantic_mask, cmap="gray")
        axes[2].set_title("Máscara semântica")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

