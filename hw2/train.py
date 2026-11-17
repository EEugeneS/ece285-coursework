"""Train DCGAN and select checkpoints using held-out validation FID."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import WikiArtDataset, manifest_digest
from evaluate import Evaluator
from models import Discriminator, Generator, initialize, train_step
from utils import environment, save_checkpoint, seed_everything, write_json


def main(args):
    if min(args.epochs, args.batch_size, args.eval_every) < 1 or min(args.lr_g, args.lr_d) <= 0:
        raise ValueError("Invalid training configuration")
    torch.set_num_threads(2)
    seed_everything(args.seed)
    device, out = torch.device(args.device), Path(args.out)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dataset = WikiArtDataset(args.data_root, "train")
    signature = manifest_digest(Path(args.data_root) / "manifest.json.gz")
    loader_rng = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, generator=loader_rng,
                        pin_memory=device.type == "cuda", drop_last=True)
    generator = Generator(args.latent_dim, args.width).to(device)
    discriminator = Discriminator(args.width).to(device)
    generator.apply(initialize)
    discriminator.apply(initialize)
    optim_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    evaluator = Evaluator(args.data_root, device)
    fixed_z = torch.randn(64, args.latent_dim, 1, 1,
                          generator=torch.Generator().manual_seed(1234)).to(device)
    history, start, best = [], 1, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["manifest_sha256"] != signature:
            raise ValueError("Resume data manifest differs")
        for key in ("latent_dim", "width", "batch_size", "lr_g", "lr_d", "seed", "eval_every"):
            if state["config"][key] != vars(args)[key]:
                raise ValueError(f"Resume configuration differs: {key}")
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        optim_g.load_state_dict(state["optim_g"])
        optim_d.load_state_dict(state["optim_d"])
        history, start, best = state["history"], state["epoch"] + 1, state["best_fid"]
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        loader_rng.set_state(state["rng_loader"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {out}")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "config.json", dict(vars(args), manifest_sha256=signature,
                   environment=environment(), train_images=len(dataset),
                   parameters_g=sum(p.numel() for p in generator.parameters()),
                   parameters_d=sum(p.numel() for p in discriminator.parameters())))
    (out / "samples").mkdir(exist_ok=True)
    (out / "epochs").mkdir(exist_ok=True)
    print(json.dumps({"train_images": len(dataset), "steps_per_epoch": len(loader),
                      "configuration": vars(args)}), flush=True)
    for epoch in range(start, args.epochs + 1):
        generator.train()
        discriminator.train()
        totals, count, started = {}, 0, time.monotonic()
        for real in loader:
            real = real.to(device, non_blocking=True)
            values = train_step(generator, discriminator, optim_g, optim_d, real)
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value * len(real)
            count += len(real)
        row = dict(epoch=epoch, images_seen=count, seconds=time.monotonic()-started,
                   **{key: value/count for key, value in totals.items()})
        improved = False
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            metric = evaluator.compute(generator, "val", seed=2026)
            row.update(val_fid=metric["fid"], val_is=metric["inception_score"])
            improved = metric["fid"] < best
            best = min(best, metric["fid"])
            write_json(out / f"validation-{epoch:03d}.json", metric)
        history.append(row)
        generator.eval()
        with torch.no_grad():
            save_image((generator(fixed_z).cpu()+1)/2,
                       out / "samples" / f"epoch_{epoch:03d}.png", nrow=8)
        state = dict(generator=generator.state_dict(), discriminator=discriminator.state_dict(),
                     optim_g=optim_g.state_dict(), optim_d=optim_d.state_dict(),
                     epoch=epoch, best_fid=best, config=vars(args), history=history,
                     manifest_sha256=signature, rng_python=random.getstate(),
                     rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
                     rng_loader=loader_rng.get_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
        save_checkpoint(out / "latest.pt", state)
        if improved:
            save_checkpoint(out / "best.pt", state)
        if "val_fid" in row:
            save_checkpoint(out / "epochs" / f"epoch_{epoch:03d}.pt", state)
        write_json(out / "history.json", history)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/wikiart")
    p.add_argument("--out", required=True)
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr-g", type=float, default=0.0002)
    p.add_argument("--lr-d", type=float, default=0.0002)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=285)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume")
    main(p.parse_args())
