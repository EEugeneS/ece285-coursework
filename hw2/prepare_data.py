"""Download Kaggle WikiArt v1 and cache audited 64x64 RGB images."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import time
import zipfile

import numpy as np
from PIL import Image, ImageOps
import requests


URL = "https://www.kaggle.com/api/v1/datasets/download/steubk/wikiart"


def download(path):
    if path.exists() and zipfile.is_zipfile(path):
        return
    partial = path.with_suffix(".part")
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(URL, params={"datasetVersionNumber": 1}, headers=headers,
                              stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Unexpected download range")
                count = offset if append else 0
                report_at = count
                with partial.open("ab" if append else "wb") as output:
                    for block in response.iter_content(8 * 1024**2):
                        output.write(block)
                        count += len(block)
                        if count - report_at >= 512 * 1024**2:
                            print(json.dumps({"downloaded_gib": round(count / 1024**3, 2)}), flush=True)
                            report_at = count
            if not zipfile.is_zipfile(partial):
                raise RuntimeError("Kaggle response is not a complete ZIP archive")
            partial.replace(path)
            return
        except (requests.RequestException, OSError) as error:
            print(f"Download attempt {attempt + 1}: {error}", flush=True)
            if attempt == 3:
                raise
            time.sleep(3)


def preprocess(image):
    """Bilinear short-edge resize followed by a centered 64x64 RGB crop."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width <= height:
        size = (64, int(height * 64 / width))
    else:
        size = (int(width * 64 / height), 64)
    image = image.resize(size, Image.Resampling.BILINEAR)
    left, top = round((size[0] - 64) / 2), round((size[1] - 64) / 2)
    return np.asarray(image.crop((left, top, left + 64, top + 64)), dtype=np.uint8).copy()


def allocation(counts, total):
    raw = {s: total * n / sum(counts.values()) for s, n in counts.items()}
    result = {s: int(x) for s, x in raw.items()}
    for style in sorted(counts, key=lambda s: (-(raw[s] - result[s]), s))[:total-sum(result.values())]:
        result[style] += 1
    return result


def assign_splits(records, seed=285, validation=1000, test=2000):
    if len(records) <= validation + test:
        raise ValueError("Not enough unique images for the requested split")
    groups = defaultdict(list)
    for record in records:
        groups[record["style"]].append(record)
    counts = {style: len(items) for style, items in groups.items()}
    nval, ntest = allocation(counts, validation), allocation(counts, test)
    rng = random.Random(seed)
    for style in sorted(groups):
        items = groups[style]
        rng.shuffle(items)
        for j, record in enumerate(items):
            record["split"] = ("test" if j < ntest[style] else
                               "val" if j < ntest[style] + nval[style] else "train")
    return records


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json.gz"
    if manifest_path.exists():
        if not (root / "images.npy").exists():
            raise FileNotFoundError("Manifest exists but image cache is missing")
        print("Existing cache and manifest retained", flush=True)
        return
    archive = Path(args.archive) if args.archive else root / "wikiart-v1.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() and shutil.disk_usage(archive.parent).free < 40 * 1024**3:
        raise RuntimeError("At least 40 GiB free space is required for the original archive")
    download(archive)
    records, excluded, corrupt, seen = [], [], [], {}
    with zipfile.ZipFile(archive) as source:
        names = sorted(n for n in source.namelist()
                       if Path(n).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        print(json.dumps({"archive_images": len(names)}), flush=True)
        cache_path = root / "images.npy"
        cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8,
                                         shape=(len(names), 64, 64, 3))

        def read(name):
            try:
                payload = source.read(name)
                with Image.open(io.BytesIO(payload)) as image:
                    pixels = preprocess(image)
                return name, pixels, hashlib.sha256(payload).hexdigest(), None
            except Exception as error:
                return name, None, None, str(error)

        # map() preserves sorted input order, so deduplication and split are deterministic.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # Bounded batches prevent eagerly retaining the 34 GB source in futures.
            for start in range(0, len(names), 256):
                for name, pixels, source_hash, error in pool.map(read, names[start:start+256]):
                    if error:
                        corrupt.append({"path": name, "error": error})
                        continue
                    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                    if digest in seen:
                        excluded.append({"path": name, "duplicate_of": seen[digest]})
                        continue
                    seen[digest] = name
                    index = len(records)
                    cache[index] = pixels
                    records.append(dict(index=index, path=name, style=Path(name).parent.name,
                                        sha256=digest, source_sha256=source_hash))
                if start % 4096 == 0:
                    print(json.dumps({"processed": min(start+256, len(names)),
                                      "unique": len(records)}), flush=True)
        cache.flush()
        if len(records) != len(names):
            compact = np.lib.format.open_memmap(root / "compact.npy", mode="w+", dtype=np.uint8,
                                               shape=(len(records), 64, 64, 3))
            for start in range(0, len(records), 2048):
                compact[start:start+2048] = cache[start:min(start+2048, len(records))]
            compact.flush()
            del compact, cache
            (root / "compact.npy").replace(cache_path)
        else:
            del cache
    assign_splits(records, args.seed, args.validation, args.test)
    summary = dict(archive_images=len(names), unique_images=len(records),
                   duplicates=len(excluded), corrupt=len(corrupt),
                   split_counts=dict(Counter(r["split"] for r in records)),
                   style_counts=dict(Counter(r["style"] for r in records)))
    manifest = dict(dataset="steubk/wikiart", version=1, seed=args.seed,
                    preprocessing="EXIF transpose; RGB; bilinear short-edge resize; 64px center crop",
                    deduplication="SHA256 of preprocessed RGB pixel bytes", summary=summary,
                    records=records, excluded=excluded, corrupt=corrupt)
    payload = (json.dumps(manifest, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    temporary = root / "manifest.tmp.gz"
    temporary.write_bytes(gzip.compress(payload, mtime=0))
    temporary.replace(manifest_path)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if args.remove_archive:
        archive.unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/wikiart")
    p.add_argument("--archive")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=285)
    p.add_argument("--validation", type=int, default=1000)
    p.add_argument("--test", type=int, default=2000)
    p.add_argument("--remove-archive", action="store_true")
    main(p.parse_args())
