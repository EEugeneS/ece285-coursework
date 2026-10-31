"""Evaluate independent prior samples using the assignment's Ignite metrics."""
import argparse
from pathlib import Path
import time

from ignite.metrics import FID, InceptionScore
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from data import XRayDataset, manifest_digest
from utils import device_for, environment, load_model, seed_everything, write_json


def inception_input(images):
    """Follow the linked tutorial: PIL uint8 bilinear resize, RGB, tensor [0,1]."""
    return torch.stack([TF.to_tensor(TF.to_pil_image(image.cpu()).convert("RGB").resize(
        (299, 299), resample=Image.Resampling.BILINEAR)) for image in images])


def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Will not overwrite metric results: {output}")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid batch size/workers")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = device_for(args.device)
    model, state = load_model(args.checkpoint, device)
    if state["manifest_sha256"] != manifest_digest(args.manifest):
        raise ValueError("Evaluation manifest differs from the training manifest")
    dataset = XRayDataset(args.data_root, args.manifest, args.split, model.config["image_size"])
    if len(dataset) < 2:
        raise ValueError("FID requires at least two samples")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    # Keep the assignment tutorial's default extractors, explicitly label the variant.
    # These scores must not be compared to standard pytorch-fid / CleanFID pool3 scores.
    fid, inception = FID(device=device), InceptionScore(device=device)
    fid.reset()
    inception.reset()
    generator, gallery, count = torch.Generator().manual_seed(args.seed), [], 0
    start = time.monotonic()
    with torch.no_grad():
        for real in loader:
            z = torch.randn(len(real), model.config["latent_dim"], generator=generator).to(device)
            fake = model.decode(z).sigmoid().cpu()
            if sum(len(g) for g in gallery) < 64:
                gallery.append(fake[:64 - sum(len(g) for g in gallery)])
            real_input = inception_input(real).to(device)
            fake_input = inception_input(fake).to(device)
            fid.update((fake_input, real_input))
            inception.update(fake_input)
            count += len(real)
            print(f"Evaluated {count}/{len(dataset)}", flush=True)
    fid_value, is_value = float(fid.compute()), float(inception.compute())
    if not (torch.isfinite(torch.tensor(fid_value)) and torch.isfinite(torch.tensor(is_value))):
        raise FloatingPointError("Non-finite metric; inspect sample count and feature covariance")
    # Floating-point error can cause tiny negative FID, but a material negative is a bug.
    if fid_value < -1e-5:
        raise FloatingPointError(f"Unexpected negative FID: {fid_value}")
    result = dict(fid=max(0.0, fid_value), inception_score=is_value,
                  fid_variant="ignite-0.5.3-default-inception-v3-1000D-softmax (assignment tutorial)",
                  num_real=count, num_generated=count, split=args.split, seed=args.seed,
                  preprocessing="128/64/32px model images; grayscale->RGB; PIL uint8 bilinear 299px; [0,1]",
                  checkpoint=str(Path(args.checkpoint).resolve()), epoch=state["epoch"],
                  model_config=state["model_config"], manifest_sha256=state["manifest_sha256"],
                  seconds=time.monotonic()-start, environment=environment(),
                  limitations=["ImageNet feature space, not a medical fidelity metric",
                               "Finite reference set biases FID; do not compare across sample counts",
                               "Deterministic mean reconstruction is not used as generated data"])
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    save_image(torch.cat(gallery), output.with_suffix(".samples.png"), nrow=8)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    main(p.parse_args())
