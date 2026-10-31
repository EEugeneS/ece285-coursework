"""Audited manifests and identical letterbox preprocessing for every split."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def manifest_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preprocess(image, size):
    image = ImageOps.exif_transpose(image).convert("L")
    image = ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR,
                         color=0, centering=(0.5, 0.5))
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).unsqueeze(0) / 255


class XRayDataset(Dataset):
    def __init__(self, root, manifest, split, size=128):
        self.root = Path(root).expanduser().resolve()
        self.manifest = json.loads(Path(manifest).read_text())
        self.records = [r for r in self.manifest["records"] if r["split"] == split]
        self.size = size
        if not self.records:
            raise ValueError(f"No images for split {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = (self.root / record["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Manifest path escapes dataset root")
        with Image.open(path) as image:
            return preprocess(image, self.size)
