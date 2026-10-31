"""Train a VAE with deterministic data order and resumable epoch checkpoints."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import XRayDataset, manifest_digest
from utils import (device_for, environment, save_checkpoint, save_examples,
                   seed_everything, write_json)
from vae import ConvVAE, loss_terms


def worker_seed(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def epoch_pass(model, loader, device, beta, optimizer=None):
    model.train(optimizer is not None)
    totals, count = np.zeros(3), 0
    with torch.set_grad_enabled(optimizer is not None):
        for image in loader:
            image = image.to(device, non_blocking=True)
            logits, mu, logvar = model(image, sample=optimizer is not None)
            terms = loss_terms(logits, image, mu, logvar, beta)
            if not all(torch.isfinite(t).item() for t in terms):
                raise FloatingPointError("Non-finite VAE loss; inspect learning rate and log variance")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                terms[0].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0,
                                              error_if_nonfinite=True)
                optimizer.step()
            totals += np.array([t.item() for t in terms]) * len(image)
            count += len(image)
    return dict(zip(("total", "reconstruction", "kl"), (totals / count).tolist()))


def main(args):
    if args.epochs < 1 or args.batch_size < 1 or args.beta < 0 or args.kl_warmup < 0:
        raise ValueError("Invalid epochs/batch size/beta/warmup")
    if args.patience < 0 or args.workers < 0 or args.lr <= 0:
        raise ValueError("Invalid patience/workers/learning rate")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device, out = device_for(args.device), Path(args.out)
    config = vars(args).copy()
    signature = manifest_digest(args.manifest)
    model = ConvVAE(args.latent_dim, args.image_size, args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train = XRayDataset(args.data_root, args.manifest, "train", args.image_size)
    val = XRayDataset(args.data_root, args.manifest, "val", args.image_size)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=device.type == "cuda", worker_init_fn=worker_seed)
    train_loader = DataLoader(train, shuffle=True, generator=generator, **loader_options)
    # A dedicated generator keeps validation iteration from consuming training RNG.
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    val_loader = DataLoader(val, shuffle=False, generator=val_generator, **loader_options)
    fixed_z = torch.randn(32, args.latent_dim, generator=torch.Generator().manual_seed(1234)).to(device)
    fixed_real = torch.stack([val[i] for i in range(min(8, len(val)))]).to(device)
    history, start, best, stale = [], 1, float("inf"), 0
    if args.resume:
        if not (out / "config.json").is_file():
            raise ValueError("Resume requires the original output directory")
        if Path(args.resume).resolve() != (out / "latest.pt").resolve():
            raise ValueError("Resume from this output directory's latest.pt to preserve run history")
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["manifest_sha256"] != signature:
            raise ValueError("Resume manifest differs from checkpoint")
        for key in ("latent_dim", "image_size", "width", "batch_size", "lr", "seed", "beta", "kl_warmup"):
            if state["config"][key] != config[key]:
                raise ValueError(f"Resume config differs: {key}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best, stale = state["epoch"] + 1, state["best_selection_loss"], state["stale"]
        history = state["history"]
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        generator.set_state(state["rng_loader"])
        val_generator.set_state(state["rng_val_loader"])
        if device.type == "cuda" and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        write_json(out / "resume_config.json", config)
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {out}; use --resume or a new name")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "config.json", dict(**config, manifest_sha256=signature,
                                           environment=environment(),
                                           parameter_count=sum(p.numel() for p in model.parameters())))
        (out / "manifest.json").write_bytes(Path(args.manifest).read_bytes())
    print(json.dumps(dict(device=str(device), train=len(train), val=len(val),
                          parameters=sum(p.numel() for p in model.parameters()))), flush=True)
    for epoch in range(start, args.epochs + 1):
        t0 = time.monotonic()
        beta = args.beta * min(1.0, epoch / args.kl_warmup) if args.kl_warmup else args.beta
        training = epoch_pass(model, train_loader, device, beta, optimizer)
        validation = epoch_pass(model, val_loader, device, beta)
        # Fixed final beta for comparable model selection, even during KL warm-up.
        selection = validation["reconstruction"] + args.beta * validation["kl"]
        row = dict(epoch=epoch, beta=beta, train=training, val=validation,
                   selection_loss=selection, seconds=time.monotonic() - t0)
        history.append(row)
        improved = selection < best
        best, stale = (selection, 0) if improved else (best, stale + 1)
        state = dict(model=model.state_dict(), model_config=model.config,
                     optimizer=optimizer.state_dict(), epoch=epoch, config=config,
                     manifest_sha256=signature, best_selection_loss=best, stale=stale,
                     history=history, rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                     rng_torch=torch.get_rng_state(), rng_loader=generator.get_state(),
                     rng_val_loader=val_generator.get_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
        save_checkpoint(out / "latest.pt", state)
        if improved:
            save_checkpoint(out / "best.pt", state)
        write_json(out / "history.json", history)
        save_examples(model, fixed_z, fixed_real, out / "figures", "latest")
        if improved:
            save_examples(model, fixed_z, fixed_real, out / "figures", "best")
        print(json.dumps(row), flush=True)
        if args.patience and stale >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break
    print(f"Saved run to {out.resolve()}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--kl-warmup", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=285)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--resume")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
