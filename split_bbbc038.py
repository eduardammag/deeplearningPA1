# split_bbbc038.py

from pathlib import Path
import json
import random

import numpy as np
from PIL import Image


DATA_ROOT = Path("data/BBBC038")
DATASET_DIR = DATA_ROOT / "stage1_train"
SPLIT_PATH = DATA_ROOT / "splits.json"

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def set_seed(seed: int) -> None:
    """
    Define a seed para tornar o split reprodutível.
    """
    random.seed(seed)
    np.random.seed(seed)


def find_image_path(image_dir: Path) -> Path:
    """
    Encontra a imagem dentro da pasta images/.

    O BBBC038 normalmente possui uma única imagem por pasta.
    """
    images_dir = image_dir / "images"

    image_files = sorted(
        list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
        + list(images_dir.glob("*.tif"))
        + list(images_dir.glob("*.tiff"))
    )

    if len(image_files) == 0:
        raise FileNotFoundError(
            f"Nenhuma imagem encontrada em {images_dir}"
        )

    if len(image_files) > 1:
        print(
            f"ATENÇÃO: mais de uma imagem em {images_dir}. "
            f"Usando a primeira."
        )

    return image_files[0]


def get_image_properties(image_path: Path) -> dict:
    """
    Obtém características simples da imagem que podem ser usadas
    para caracterizar o conjunto e verificar o split.

    Não usamos o conteúdo da máscara para definir o split.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        width, height = image.size

        array = np.asarray(image)

    is_grayscale = bool(
        np.array_equal(
            array[:, :, 0],
            array[:, :, 1]
        )
        and np.array_equal(
            array[:, :, 1],
            array[:, :, 2]
        )
    )

    if is_grayscale:
        modality_proxy = "grayscale"
    else:
        modality_proxy = "color"

    return {
        "width": int(width),
        "height": int(height),
        "is_grayscale": is_grayscale,
        "modality_proxy": modality_proxy,
    }


def collect_images() -> list[dict]:
    """
    Coleta todas as imagens válidas do stage1_train.
    """
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {DATASET_DIR}.\n"
            "Execute primeiro:\n"
            "python download_bbbc038.py"
        )

    samples = []

    for image_dir in sorted(DATASET_DIR.iterdir()):

        if not image_dir.is_dir():
            continue

        images_dir = image_dir / "images"
        masks_dir = image_dir / "masks"

        if not images_dir.is_dir():
            continue

        if not masks_dir.is_dir():
            continue

        image_path = find_image_path(
            image_dir
        )

        properties = get_image_properties(
            image_path
        )

        mask_files = sorted(
            masks_dir.glob("*.png")
        )

        if len(mask_files) == 0:
            print(
                f"ATENÇÃO: nenhuma máscara em "
                f"{image_dir.name}. Ignorando."
            )
            continue

        samples.append(
            {
                "image_id": image_dir.name,
                "image_path": str(
                    image_path.relative_to(DATA_ROOT)
                ),
                "num_masks": len(mask_files),
                **properties,
            }
        )

    return samples


def stratified_split(
    samples: list[dict],
    seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Faz um split estratificado por uma característica observável
    da imagem.

    Usamos como proxy:
        grayscale vs color

    Isso não afirma que grayscale/color seja uma classificação
    oficial de modalidade do BBBC038. É apenas uma variável
    observável usada para manter a composição dos splits.

    Quando o metadata oficial puder ser associado diretamente
    a cada ImageId, essa função pode ser substituída por uma
    estratificação por experimento/modalidade.
    """

    rng = random.Random(seed)

    groups = {}

    for sample in samples:
        key = sample["modality_proxy"]

        if key not in groups:
            groups[key] = []

        groups[key].append(sample)

    train = []
    val = []
    test = []

    for key, group in groups.items():

        group = list(group)

        rng.shuffle(group)

        n = len(group)

        n_train = int(
            round(n * TRAIN_RATIO)
        )

        n_val = int(
            round(n * VAL_RATIO)
        )

        # Garante que todos os elementos restantes
        # vão para teste.
        n_test = n - n_train - n_val

        if n_test < 0:
            raise RuntimeError(
                f"Split inválido para grupo {key}."
            )

        train.extend(
            group[:n_train]
        )

        val.extend(
            group[
                n_train:
                n_train + n_val
            ]
        )

        test.extend(
            group[
                n_train + n_val:
            ]
        )

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def summarize_split(
    name: str,
    samples: list[dict]
) -> None:

    grayscale = sum(
        sample["is_grayscale"]
        for sample in samples
    )

    color = len(samples) - grayscale

    total_masks = sum(
        sample["num_masks"]
        for sample in samples
    )

    print(
        f"\n{name}:"
    )

    print(
        f"  imagens: {len(samples)}"
    )

    print(
        f"  grayscale: {grayscale}"
    )

    print(
        f"  coloridas: {color}"
    )

    print(
        f"  núcleos anotados: {total_masks}"
    )


def main() -> None:

    set_seed(SEED)

    print("=" * 70)
    print("BBBC038 - SPLIT")
    print("=" * 70)

    samples = collect_images()

    if len(samples) == 0:
        raise RuntimeError(
            "Nenhuma imagem válida encontrada."
        )

    print(
        f"\nTotal de imagens válidas: {len(samples)}"
    )

    train, val, test = stratified_split(
        samples,
        seed=SEED
    )

    summarize_split(
        "TRAIN",
        train
    )

    summarize_split(
        "VALIDATION",
        val
    )

    summarize_split(
        "TEST",
        test
    )

    total = (
        len(train)
        + len(val)
        + len(test)
    )

    if total != len(samples):
        raise RuntimeError(
            "Algumas imagens desapareceram durante o split."
        )

    train_ids = {
        sample["image_id"]
        for sample in train
    }

    val_ids = {
        sample["image_id"]
        for sample in val
    }

    test_ids = {
        sample["image_id"]
        for sample in test
    }

    if train_ids & val_ids:
        raise RuntimeError(
            "Há imagens compartilhadas entre treino e validação."
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "Há imagens compartilhadas entre treino e teste."
        )

    if val_ids & test_ids:
        raise RuntimeError(
            "Há imagens compartilhadas entre validação e teste."
        )

    split_data = {
        "dataset": "BBBC038v1",
        "source": "stage1_train",
        "seed": SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "stratification": {
            "method": "modality_proxy",
            "variable": "grayscale_vs_color",
            "note": (
                "Proxy observável usado para manter a composição "
                "dos splits. Não representa uma classificação "
                "oficial de modalidade do BBBC038."
            ),
        },
        "counts": {
            "total": len(samples),
            "train": len(train),
            "validation": len(val),
            "test": len(test),
        },
        "train": [
            sample["image_id"]
            for sample in train
        ],
        "validation": [
            sample["image_id"]
            for sample in val
        ],
        "test": [
            sample["image_id"]
            for sample in test
        ],
        "metadata": {
            sample["image_id"]: {
                "image_path": sample["image_path"],
                "num_masks": sample["num_masks"],
                "width": sample["width"],
                "height": sample["height"],
                "is_grayscale": sample["is_grayscale"],
                "modality_proxy": sample["modality_proxy"],
            }
            for sample in samples
        },
    }

    DATA_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        SPLIT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            split_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\nSplit salvo em:"
    )

    print(
        f"  {SPLIT_PATH}"
    )

    print("\nConcluído.")


if __name__ == "__main__":
    main()