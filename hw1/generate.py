"""Generate unconditional samples from z ~ N(0,I), without encoding real images."""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from utils import device_for, load_model, seed_everything, write_json


def main(args):
    if args.count < 1 or args.batch_size < 1:
        raise ValueError("count and batch-size must be positive")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = device_for(args.device)
    model, state = load_model(args.checkpoint, device)
    generator, gallery = torch.Generator().manual_seed(args.seed), []
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            n = min(args.batch_size, args.count - start)
            z = torch.randn(n, model.config["latent_dim"], generator=generator).to(device)
            images = model.decode(z).sigmoid().cpu()
            for i, image in enumerate(images):
                save_image(image, out / f"{start+i:05d}.png")
                if len(gallery) < 64:
                    gallery.append(image)
    save_image(torch.stack(gallery), out / "gallery.png", nrow=8)
    write_json(out / "generation.json", dict(count=args.count, seed=args.seed,
               checkpoint=str(Path(args.checkpoint).resolve()), epoch=state["epoch"],
               source="independent standard-normal latent samples"))
    print(out.resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    main(p.parse_args())
