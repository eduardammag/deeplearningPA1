# dataset_bbbc038.py

from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BBBC038Dataset(Dataset):
    """
    Dataset PyTorch para o BBBC038 stage1_train.

    Cada imagem possui:
        - uma imagem PNG;
        - várias máscaras PNG, uma por núcleo.

    O dataset retorna:

        image:
            Tensor [C, H, W], float32, normalizado para [0, 1]

        semantic_mask:
            Tensor [H, W], float32
            0 = background
            1 = nucleus

        instance_mask:
            Tensor [H, W], int64
            0 = background
            1..N = instâncias individuais

        image_id:
            identificador original da imagem
    """

    def __init__(
        self,
        root_dir="data/BBBC038",
        split="train",
        split_file=None,
        resize=256,
    ):
        super().__init__()

        self.root_dir = Path(
            root_dir
        )

        self.dataset_dir = (
            self.root_dir / "stage1_train"
        )

        if split_file is None:
            split_file = (
                self.root_dir / "splits.json"
            )

        self.split_file = Path(
            split_file
        )

        self.split = split

        self.resize = resize

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado em "
                f"{self.dataset_dir}"
            )

        if not self.split_file.exists():
            raise FileNotFoundError(
                f"Arquivo de split não encontrado em "
                f"{self.split_file}\n"
                "Execute primeiro:\n"
                "python split_bbbc038.py"
            )

        with open(
            self.split_file,
            "r",
            encoding="utf-8"
        ) as file:

            split_data = json.load(file)

        if split not in {
            "train",
            "validation",
            "test",
        }:

            raise ValueError(
                "split deve ser 'train', "
                "'validation' ou 'test'."
            )

        self.image_ids = split_data[
            split
        ]

    def __len__(self):
        return len(
            self.image_ids
        )

    def _find_image(
        self,
        image_id: str
    ) -> Path:

        image_dir = (
            self.dataset_dir / image_id
        )

        images_dir = (
            image_dir / "images"
        )

        image_files = sorted(
            list(images_dir.glob("*.png"))
            + list(images_dir.glob("*.jpg"))
            + list(images_dir.glob("*.jpeg"))
            + list(images_dir.glob("*.tif"))
            + list(images_dir.glob("*.tiff"))
        )

        if len(image_files) == 0:
            raise FileNotFoundError(
                f"Nenhuma imagem encontrada em "
                f"{images_dir}"
            )

        return image_files[0]

    def _find_masks(
        self,
        image_id: str
    ) -> list[Path]:

        masks_dir = (
            self.dataset_dir
            / image_id
            / "masks"
        )

        mask_files = sorted(
            masks_dir.glob("*.png")
        )

        if len(mask_files) == 0:
            raise FileNotFoundError(
                f"Nenhuma máscara encontrada em "
                f"{masks_dir}"
            )

        return mask_files

    @staticmethod
    def _load_image(
        image_path: Path
    ) -> np.ndarray:

        image = Image.open(
            image_path
        )

        image = np.asarray(
            image
        )

        # Remove alpha se existir.
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        # Se RGB, convertemos para grayscale.
        #
        # O baseline atual usa U-Net com in_channels=1.
        # A conversão é feita com média simples dos canais,
        # sem introduzir uma biblioteca externa.
        if image.ndim == 3:

            image = image.astype(
                np.float32
            )

            image = np.mean(
                image,
                axis=2
            )

        image = image.astype(
            np.float32
        )

        # Normalização para [0, 1].
        if image.max() > 1.0:
            image /= 255.0

        image = np.clip(
            image,
            0.0,
            1.0
        )

        return image

    @staticmethod
    def _build_instance_mask(
        mask_paths: list[Path],
        shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Combina as máscaras individuais em uma única
        máscara de instâncias.

        Cada máscara recebe um ID diferente.
        """

        height, width = shape

        instance_mask = np.zeros(
            (height, width),
            dtype=np.int64
        )

        for instance_id, mask_path in enumerate(
            mask_paths,
            start=1
        ):

            mask = Image.open(
                mask_path
            )

            mask = np.asarray(
                mask
            )

            if mask.ndim == 3:
                mask = mask[:, :, 0]

            mask = mask > 0

            if mask.shape != (
                height,
                width
            ):

                raise ValueError(
                    f"A máscara {mask_path.name} "
                    f"possui shape {mask.shape}, "
                    f"mas a imagem possui "
                    f"{(height, width)}."
                )

            # As máscaras oficiais não devem se sobrepor.
            overlap = (
                mask
                & (instance_mask > 0)
            )

            if np.any(overlap):
                raise ValueError(
                    f"Máscaras sobrepostas encontradas "
                    f"em {mask_path.parent.parent.name}."
                )

            instance_mask[mask] = instance_id

        return instance_mask

    def _resize(
        self,
        image: np.ndarray,
        instance_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Redimensiona mantendo a proporção e adiciona padding
        para produzir uma imagem quadrada de tamanho TARGET_SIZE.

        A imagem usa interpolação bilinear.
        A máscara de instâncias usa nearest neighbor para
        preservar os IDs das instâncias.
        """

        if self.resize is None:
            return image, instance_mask

        if isinstance(self.resize, int):
            target_height = self.resize
            target_width = self.resize

        elif (
            isinstance(self.resize, tuple)
            and len(self.resize) == 2
        ):
            target_height = self.resize[0]
            target_width = self.resize[1]

        else:
            raise ValueError(
                "resize deve ser None, um inteiro "
                "ou uma tupla (height, width)."
            )

        original_height, original_width = image.shape

        # ---------------------------------------------------------
        # 1. Calcula escala mantendo proporção
        # ---------------------------------------------------------

        scale = min(
            target_height / original_height,
            target_width / original_width
        )

        new_height = max(
            1,
            int(round(original_height * scale))
        )

        new_width = max(
            1,
            int(round(original_width * scale))
        )

        # ---------------------------------------------------------
        # 2. Resize da imagem
        # ---------------------------------------------------------

        image_pil = Image.fromarray(
            (
                image * 255.0
            ).astype(
                np.uint8
            )
        )

        image_pil = image_pil.resize(
            (
                new_width,
                new_height
            ),
            resample=Image.Resampling.BILINEAR
        )

        image_resized = (
            np.asarray(
                image_pil
            ).astype(
                np.float32
            )
            / 255.0
        )

        # ---------------------------------------------------------
        # 3. Resize da máscara
        # ---------------------------------------------------------

        instance_pil = Image.fromarray(
            instance_mask.astype(
                np.int32
            )
        )

        instance_pil = instance_pil.resize(
            (
                new_width,
                new_height
            ),
            resample=Image.Resampling.NEAREST
        )

        instance_resized = np.asarray(
            instance_pil
        ).astype(
            np.int64
        )

        # ---------------------------------------------------------
        # 4. Padding
        # ---------------------------------------------------------

        padded_image = np.zeros(
            (
                target_height,
                target_width
            ),
            dtype=np.float32
        )

        padded_instance = np.zeros(
            (
                target_height,
                target_width
            ),
            dtype=np.int64
        )

        # Centraliza a imagem.
        top = (
            target_height - new_height
        ) // 2

        left = (
            target_width - new_width
        ) // 2

        padded_image[
            top:top + new_height,
            left:left + new_width
        ] = image_resized

        padded_instance[
            top:top + new_height,
            left:left + new_width
        ] = instance_resized

        return (
            padded_image,
            padded_instance
        )

    def __getitem__(
        self,
        index: int
    ) -> dict:

        image_id = self.image_ids[
            index
        ]

        image_path = self._find_image(
            image_id
        )

        mask_paths = self._find_masks(
            image_id
        )

        image = self._load_image(
            image_path
        )

        instance_mask = self._build_instance_mask(
            mask_paths,
            image.shape
        )

        image, instance_mask = self._resize(
            image,
            instance_mask
        )

        semantic_mask = (
            instance_mask > 0
        ).astype(
            np.float32
        )

        image = torch.from_numpy(
            image
        ).unsqueeze(
            0
        )

        semantic_mask = torch.from_numpy(
            semantic_mask
        )

        instance_mask = torch.from_numpy(
            instance_mask
        )

        return {
            "image": image.float(),
            "semantic_mask": semantic_mask.float(),
            "instance_mask": instance_mask.long(),
            "image_id": image_id,
        }


if __name__ == "__main__":

    dataset = BBBC038Dataset(
        root_dir="data/BBBC038",
        split="train"
    )

    print(
        f"Dataset: {len(dataset)} imagens"
    )

    sample = dataset[0]

    print(
        f"\nImage ID: {sample['image_id']}"
    )

    print(
        f"Image shape: "
        f"{tuple(sample['image'].shape)}"
    )

    print(
        f"Semantic mask shape: "
        f"{tuple(sample['semantic_mask'].shape)}"
    )

    print(
        f"Instance mask shape: "
        f"{tuple(sample['instance_mask'].shape)}"
    )

    unique_instances = torch.unique(
        sample["instance_mask"]
    )

    print(
        f"Número de instâncias: "
        f"{len(unique_instances) - 1}"
    )