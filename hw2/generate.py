"""Generate fixed-seed independent samples from a trained GAN."""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from utils import load_generator, write_json


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    if args.count < 1 or (out.exists() and any(out.iterdir())):
        raise ValueError("count must be positive and output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    model, state = load_generator(args.checkpoint, args.device)
    latent = torch.randn(args.count, model.latent_dim, 1, 1,
                         generator=torch.Generator().manual_seed(args.seed))
    samples = []
    with torch.no_grad():
        for start in range(0, args.count, 32):
            images = (model(latent[start:start+32].to(args.device)).cpu() + 1) / 2
            for index, image in enumerate(images):
                save_image(image, out / f"{start+index:05d}.png")
            samples.append(images)
    save_image(torch.cat(samples), out / "gallery.png", nrow=8)
    write_json(out / "generation.json", dict(seed=args.seed, count=args.count,
               epoch=state["epoch"], checkpoint=str(Path(args.checkpoint).resolve()),
               source="independent standard-normal latent samples"))
