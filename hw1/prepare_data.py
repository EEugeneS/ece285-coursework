"""Audit the official dataset, de-duplicate, and split training patients only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from PIL import Image, ImageOps


def patient_id(stem):
    for pattern in (r"(person\d+)_", r"((?:NORMAL2-)?IM-\d+)-"):
        match = re.match(pattern, stem, flags=re.I)
        if match:
            return match.group(1).lower(), True
    return stem.lower(), False


def prepare(root, output, seed=285, fraction=0.1):
    root, output = Path(root).expanduser().resolve(), Path(output)
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    for folder in ("train", "test"):
        if not (root / folder).is_dir():
            raise ValueError(f"Expected {root / folder}; pass the inner chest_xray directory")
        for label in ("NORMAL", "PNEUMONIA"):
            if not (root / folder / label).is_dir():
                raise ValueError(f"Missing assigned class directory: {root / folder / label}")
    records, bad, unknown = [], [], []
    for source in ("train", "val", "test"):
        for label in ("NORMAL", "PNEUMONIA"):
            for path in sorted((root / source / label).glob("*")):
                if not path.is_file() or path.name.startswith("."):
                    continue
                relative = path.relative_to(root).as_posix()
                try:
                    with Image.open(path) as image:
                        image = ImageOps.exif_transpose(image).convert("L")
                        image.load()
                        digest = hashlib.sha256(
                            f"{image.size}".encode() + image.tobytes()).hexdigest()
                        size = list(image.size)
                except Exception as error:
                    bad.append(dict(path=relative, error=str(error)))
                    continue
                patient, recognized = patient_id(path.stem)
                if not recognized:
                    unknown.append(relative)
                records.append(dict(path=relative, source=source, label=label,
                                    patient=patient, sha256=digest, size=size))
    # Official validation images are audited but not used for training or selection.
    # Test has priority: remove exact decoded-pixel duplicates, then remove training
    # images from any patient or duplicate-content component touching the test set.
    excluded, retained, seen = [], [], {}
    for record in sorted(records, key=lambda r: (r["source"] != "test", r["path"])):
        if record["source"] == "val":
            excluded.append(dict(**record, reason="official_val_unused"))
        elif record["sha256"] in seen:
            excluded.append(dict(**record, reason="duplicate_pixels",
                                 duplicate_of=seen[record["sha256"]]))
        else:
            seen[record["sha256"]] = record["path"]
            retained.append(record)
    test_patients = {r["patient"] for r in records if r["source"] == "test"}
    clean = []
    for r in retained:
        if r["source"] == "train" and r["patient"] in test_patients:
            excluded.append(dict(**r, reason="patient_overlaps_test"))
        else:
            clean.append(r)
    # Link patient IDs sharing duplicate image content before train/val splitting.
    parent = {r["patient"]: r["patient"] for r in records}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    by_hash = defaultdict(list)
    for r in records:
        if r["source"] != "val":
            by_hash[r["sha256"]].append(r["patient"])
    for group in by_hash.values():
        for patient in group[1:]:
            parent[find(patient)] = find(group[0])
    test_components = {find(p) for p in test_patients}
    eligible = []
    for r in clean:
        if r["source"] == "train" and find(r["patient"]) in test_components:
            excluded.append(dict(**r, reason="duplicate_component_overlaps_test"))
        else:
            eligible.append(r)
    groups = defaultdict(list)
    for r in eligible:
        if r["source"] == "train":
            groups[find(r["patient"])].append(r)
    strata = defaultdict(list)
    for group, items in groups.items():
        strata[tuple(sorted({r["label"] for r in items}))].append(group)
    rng, validation_groups = random.Random(seed), set()
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        if len(keys) < 2:
            raise ValueError("Need at least two patient groups in each stratum")
        rng.shuffle(keys)
        n = max(1, min(len(keys) - 1, round(len(keys) * fraction)))
        validation_groups.update(keys[:n])
    for r in eligible:
        r["split"] = ("test" if r["source"] == "test" else
                      "val" if find(r["patient"]) in validation_groups else "train")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for field in ("patient", "sha256"):
            assert not ({r[field] for r in eligible if r["split"] == left} &
                        {r[field] for r in eligible if r["split"] == right})
    summary = dict(source_counts=dict(Counter(r["source"] for r in records)),
                   split_counts=dict(Counter(r["split"] for r in eligible)),
                   split_label_counts=dict(Counter(f'{r["split"]}/{r["label"]}' for r in eligible)),
                   excluded_reasons=dict(Counter(r["reason"] for r in excluded)),
                   corrupt_count=len(bad), unrecognized_patient_count=len(unknown))
    if unknown:
        raise ValueError(f"Unrecognized patient names; inspect before splitting: {unknown[:10]}")
    for split in ("train", "val", "test"):
        if summary["split_counts"].get(split, 0) < 2:
            raise ValueError(f"Too few images in {split}")
    manifest = dict(schema=1, seed=seed, validation_group_fraction=fraction,
                    patient_id_note="Inferred from filenames, not verified clinical identifiers",
                    preprocessing="grayscale; aspect-preserving bilinear letterbox; [0,1]",
                    summary=summary, records=sorted(eligible, key=lambda r: r["path"]),
                    excluded=excluded, corrupt=bad)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Will not overwrite existing split: {output}")
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default="data/manifest.json")
    p.add_argument("--seed", type=int, default=285)
    p.add_argument("--val-fraction", type=float, default=0.1)
    args = p.parse_args()
    prepare(args.data_root, args.output, args.seed, args.val_fraction)
