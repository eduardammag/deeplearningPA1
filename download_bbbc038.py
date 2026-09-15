# download_bbbc038.py

from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import argparse


DATA_ROOT = Path("data/BBBC038")

TRAIN_URL = (
    "https://data.broadinstitute.org/bbbc/BBBC038/"
    "stage1_train.zip"
)

METADATA_URL = (
    "https://data.broadinstitute.org/bbbc/BBBC038/"
    "metadata.xlsx"
)

ZIP_PATH = DATA_ROOT / "stage1_train.zip"
METADATA_PATH = DATA_ROOT / "metadata.xlsx"
EXTRACT_DIR = DATA_ROOT / "stage1_train"


def download_file(url: str, destination: Path) -> None:
    """
    Baixa um arquivo caso ele ainda não exista.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nBaixando:")
    print(f"  {url}")
    print(f"Para:")
    print(f"  {destination}")

    urlretrieve(url, destination)

    print("Download concluído.")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """
    Extrai o arquivo ZIP para o diretório especificado.
    """
    extract_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nExtraindo:")
    print(f"  {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_dir)

    print("Extração concluída.")


def count_image_directories(root_dir: Path) -> int:
    """
    Conta quantas pastas possuem a estrutura esperada:
        ImageId/
            images/
            masks/
    """
    if not root_dir.exists():
        return 0

    count = 0

    for image_dir in root_dir.iterdir():
        if not image_dir.is_dir():
            continue

        images_dir = image_dir / "images"
        masks_dir = image_dir / "masks"

        if images_dir.is_dir() and masks_dir.is_dir():
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download e preparação inicial do BBBC038."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Refaz o download/extração mesmo se os arquivos já existirem."
    )

    args = parser.parse_args()

    DATA_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("BBBC038 - DOWNLOAD")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Download do stage1_train.zip
    # ---------------------------------------------------------

    if ZIP_PATH.exists() and not args.force:
        print(
            f"\nZIP já existe. Pulando download:\n"
            f"  {ZIP_PATH}"
        )
    else:
        download_file(
            TRAIN_URL,
            ZIP_PATH
        )

    # ---------------------------------------------------------
    # 2. Download do metadata.xlsx
    # ---------------------------------------------------------

    if METADATA_PATH.exists() and not args.force:
        print(
            f"\nMetadata já existe. Pulando download:\n"
            f"  {METADATA_PATH}"
        )
    else:
        download_file(
            METADATA_URL,
            METADATA_PATH
        )

    # ---------------------------------------------------------
    # 3. Extração
    # ---------------------------------------------------------

    expected_images = count_image_directories(
        EXTRACT_DIR
    )

    if expected_images > 0 and not args.force:
        print(
            f"\nDataset já parece estar extraído."
        )
        print(
            f"Imagens encontradas: {expected_images}"
        )
    else:
        if args.force and EXTRACT_DIR.exists():
            import shutil

            print(
                "\nRemovendo extração anterior..."
            )

            shutil.rmtree(EXTRACT_DIR)

        extract_zip(
            ZIP_PATH,
            EXTRACT_DIR
        )

    # ---------------------------------------------------------
    # 4. Verificação
    # ---------------------------------------------------------

    number_of_images = count_image_directories(
        EXTRACT_DIR
    )

    print("\n" + "=" * 70)
    print("VERIFICAÇÃO")
    print("=" * 70)

    print(
        f"Diretório do dataset: {EXTRACT_DIR}"
    )

    print(
        f"Imagens com images/ + masks/: {number_of_images}"
    )

    if number_of_images == 0:
        raise RuntimeError(
            "Nenhuma imagem válida foi encontrada. "
            "Verifique a estrutura do ZIP."
        )

    if number_of_images != 670:
        print(
            "\nATENÇÃO:"
        )
        print(
            "O BBBC038 stage1_train normalmente possui "
            "cerca de 670 imagens."
        )
        print(
            f"Foram encontradas {number_of_images}."
        )

    print(
        "\nDownload e preparação inicial concluídos."
    )


if __name__ == "__main__":
    main()