"""Ignite tutorial FID and IS on independent generated WikiArt samples."""
import argparse
import json
from pathlib import Path
import time

from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from data import WikiArtDataset, manifest_digest
from utils import environment, load_generator, seed_everything, write_json


VARIANT = "ignite-0.5.3-tutorial-1000D-softmax"
PREPROCESSING = "RGB [0,1] -> PIL uint8 -> bilinear 299x299 -> tensor [0,1]"


def inception_input(images):
    return torch.stack([TF.to_tensor(TF.to_pil_image(image.cpu().clamp(0, 1)).resize(
        (299, 299), resample=Image.Resampling.BILINEAR)) for image in images])


class Evaluator:
    def __init__(self, root, device, batch_size=32):
        self.root, self.device, self.batch_size = Path(root), device, batch_size
        self.digest = manifest_digest(self.root / "manifest.json.gz")
        self.extractor = InceptionModel(return_features=False, device=device)
        self.real_features = {}

    @torch.no_grad()
    def features(self, images):
        output = self.extractor(inception_input(images).to(self.device)).cpu()
        if output.shape[1] != 1000 or not torch.allclose(
                output.sum(1), torch.ones(len(output)), atol=1e-4):
            raise RuntimeError("Installed Inception extractor differs from the declared metric variant")
        return output

    @torch.no_grad()
    def reference(self, split):
        if split in self.real_features:
            return self.real_features[split]
        path = self.root / f"inception-{split}.npz"
        signature = dict(manifest_sha256=self.digest, variant=VARIANT,
                         preprocessing=PREPROCESSING, environment=environment(),
                         precision=dict(cudnn_tf32=torch.backends.cudnn.allow_tf32,
                                        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                        cudnn_deterministic=torch.backends.cudnn.deterministic))
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                if json.loads(str(cached["signature"])) == signature:
                    self.real_features[split] = torch.from_numpy(cached["features"].copy())
                    return self.real_features[split]
        loader = DataLoader(WikiArtDataset(self.root, split, normalized=False),
                            batch_size=self.batch_size, shuffle=False, num_workers=0)
        features = torch.cat([self.features(images) for images in loader])
        np.savez(path, features=features.numpy(), signature=json.dumps(signature, sort_keys=True))
        self.real_features[split] = features
        return features

    @torch.no_grad()
    def compute(self, generator, split, seed=2026, gallery=None):
        was_training = generator.training
        generator.eval()
        try:
            real = self.reference(split)
            latent = torch.randn(len(real), generator.latent_dim, 1, 1,
                                 generator=torch.Generator().manual_seed(seed))
            fid = FID(num_features=1000, feature_extractor=nn.Identity(), device="cpu")
            score = InceptionScore(num_features=1000, feature_extractor=nn.Identity(), device="cpu")
            samples = []
            for start in range(0, len(real), self.batch_size):
                fake = (generator(latent[start:start+self.batch_size].to(self.device)).cpu() + 1) / 2
                features = self.features(fake)
                fid.update((features, real[start:start+len(fake)]))
                score.update(features)
                if start < 64:
                    samples.append(fake[:min(len(fake), 64-start)])
            fid_value, is_value = float(fid.compute()), float(score.compute())
            if not np.isfinite([fid_value, is_value]).all() or fid_value < -1e-5:
                raise FloatingPointError("Invalid GAN metric")
            if gallery:
                save_image(torch.cat(samples), gallery, nrow=8)
            return dict(fid=max(0.0, fid_value), inception_score=is_value, split=split,
                        num_real=len(real), num_generated=len(real), seed=seed,
                        fid_variant=VARIANT, preprocessing=PREPROCESSING,
                        manifest_sha256=self.digest)
        finally:
            generator.train(was_training)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/wikiart")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(2)
    seed_everything(args.seed)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, state = load_generator(args.checkpoint, args.device)
    evaluator = Evaluator(args.data_root, args.device, args.batch_size)
    if state["manifest_sha256"] != evaluator.digest:
        raise ValueError("Checkpoint and data manifest differ")
    started = time.monotonic()
    result = evaluator.compute(model, args.split, args.seed, output.with_suffix(".png"))
    result.update(epoch=state["epoch"], checkpoint=str(Path(args.checkpoint).resolve()),
                  environment=environment(), seconds=time.monotonic()-started)
    write_json(output, result)
    print(json.dumps(result), flush=True)
