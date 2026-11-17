"""WikiArt image cache and reproducible manifest access."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_manifest(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def manifest_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class WikiArtDataset(Dataset):
    def __init__(self, root, split, normalized=True):
        root = Path(root)
        self.cache_path = root / "images.npy"
        self.manifest = read_manifest(root / "manifest.json.gz")
        self.records = [r for r in self.manifest["records"] if r["split"] == split]
        self.normalized = normalized
        self.images = None
        if not self.records:
            raise ValueError(f"Empty split: {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.images is None:
            self.images = np.load(self.cache_path, mmap_mode="r", allow_pickle=False)
        record = self.records[index]
        image = torch.from_numpy(self.images[record["index"]].copy()).permute(2, 0, 1).float() / 255
        return image * 2 - 1 if self.normalized else image
