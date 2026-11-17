"""Plot saved GAN histories and fixed-seed samples."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main(runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.85), constrained_layout=True)
    rows = []
    canvas = Image.new("RGB", (8 * 67 + 3, len(runs) * 67 + 3), "white")
    have_samples = True
    for index, path in enumerate(map(Path, runs)):
        config = json.loads((path / "config.json").read_text())
        history = json.loads((path / "history.json").read_text())
        label = f'D lr={config["lr_d"]:g}'
        color = ("#245eaa", "#d06b28")[index % 2]
        for axis, key, title in zip(axes[:2], ("loss_g", "loss_d"), ("Generator", "Discriminator")):
            axis.plot([r["epoch"] for r in history], [r[key] for r in history],
                      label=label, color=color, linewidth=1.1)
            axis.set(title=title, xlabel="Epoch", ylabel="BCE loss")
        evaluated = [r for r in history if "val_fid" in r]
        axes[2].plot([r["epoch"] for r in evaluated], [r["val_fid"] for r in evaluated],
                     "o-", color=color, markersize=3, linewidth=1.1)
        axes[2].set(title="Validation tutorial FID", xlabel="Epoch", ylabel="FID (lower is better)")
        metric_path = path / "metrics-test.json"
        metric = json.loads(metric_path.read_text()) if metric_path.exists() else {}
        best = min(evaluated, key=lambda r: r["val_fid"]) if evaluated else {}
        rows.append(dict(run=path.name, lr_d=config["lr_d"], epochs=len(history),
                         best_epoch=best.get("epoch", ""), val_fid=best.get("val_fid", ""),
                         fid=metric.get("fid", ""), inception_score=metric.get("inception_score", ""),
                         minutes=sum(r["seconds"] for r in history)/60))
        for column in range(8):
            image_path = path / "generated" / f"{column:05d}.png"
            if image_path.exists():
                with Image.open(image_path) as image:
                    canvas.paste(image.convert("RGB"), (3 + column * 67, 3 + index * 67))
            else:
                have_samples = False
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"loss_curves.{suffix}", dpi=220)
    plt.close(fig)
    if have_samples:
        canvas.save(output / "prior_samples.png")
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--output", default="figures")
    args = p.parse_args()
    main(args.runs, args.output)
