import importlib.metadata
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


def environment():
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "numpy", "Pillow", "scipy", "pytorch-ignite")}
    return dict(packages=packages, cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_generator(path, device):
    from models import Generator
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = Generator(state["config"]["latent_dim"], state["config"]["width"]).to(device)
    model.load_state_dict(state["generator"])
    model.eval()
    return model, state
