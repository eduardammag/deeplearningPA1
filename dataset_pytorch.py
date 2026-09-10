from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SyntheticSegmentationDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        self.samples = sorted(
            [
                path
                for path in self.root_dir.iterdir()
                if path.is_dir()
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample_dir = self.samples[index]

        image_path = sample_dir / "image.png"
        semantic_path = sample_dir / "semantic_mask.png"
        instance_path = sample_dir / "instance_mask.png"

        image = np.array(Image.open(image_path))
        semantic_mask = np.array(Image.open(semantic_path))
        instance_mask = np.array(Image.open(instance_path))

        image = torch.from_numpy(image).float()
        semantic_mask = torch.from_numpy(semantic_mask).long()
        instance_mask = torch.from_numpy(instance_mask).long()

        if image.ndim == 2:
            image = image.unsqueeze(0)
        else:
            image = image.permute(2, 0, 1)

        return {
            "image": image,
            "semantic_mask": semantic_mask,
            "instance_mask": instance_mask,
        }