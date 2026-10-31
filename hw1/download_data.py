"""Download only the assignment's public Kaggle dataset; never prints credentials."""
import argparse
import json
import os
from pathlib import Path
import shutil


def main(args):
    target = Path(args.cache).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target).free < 6 * 1024**3:
        raise RuntimeError("Less than 6 GiB free. Check course shared datasets/storage before downloading.")
    os.environ["KAGGLEHUB_CACHE"] = str(target)
    import kagglehub
    handle = "paultimothymooney/chest-xray-pneumonia/versions/2"
    downloaded = Path(kagglehub.dataset_download(handle))
    candidates = sorted({p.parent.parent for p in downloaded.rglob("NORMAL")
                         if p.is_dir() and p.parent.name == "train"
                         and (p.parent.parent / "test" / "NORMAL").is_dir()},
                        key=lambda p: (len(p.parts), str(p)))
    if not candidates:
        raise RuntimeError(f"Could not locate official train/test structure under {downloaded}")
    root = candidates[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(kaggle_handle=handle, version=2,
                    downloaded_version_path=str(downloaded), data_root=str(root),
                    note="Uses the shallowest official train/test directory; nested copies are not merged")
    output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Dataset root: {root}\nMetadata: {output.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="data/kaggle-cache")
    p.add_argument("--output", default="data/download.json")
    main(p.parse_args())
