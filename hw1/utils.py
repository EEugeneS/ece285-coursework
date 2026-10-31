import importlib.metadata
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torchvision.utils import save_image


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def device_for(name="auto"):
    name = ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Select DataHub's 1 GPU environment.")
    return torch.device(name)


def environment():
    packages = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "scipy", "pytorch-ignite"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(packages=packages, cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_model(checkpoint, device):
    from vae import ConvVAE
    # Only load checkpoints produced by this project: torch checkpoint files are executable pickle.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ConvVAE(**state["model_config"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


@torch.no_grad()
def save_examples(model, fixed_z, real, folder, tag):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    previous = model.training
    model.eval()
    save_image(model.decode(fixed_z).sigmoid().cpu(), folder / f"samples_{tag}.png", nrow=8)
    reconstruction, _, _ = model(real, sample=False)
    # First row: real. Second row: reconstruction from posterior mean.
    save_image(torch.cat([real, reconstruction.sigmoid()]).cpu(),
               folder / f"reconstructions_{tag}.png", nrow=len(real))
    model.train(previous)
