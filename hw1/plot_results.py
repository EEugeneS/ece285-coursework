"""Create report-ready loss curves and a CSV summary from real run artifacts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in map(Path, runs):
        history = json.loads((path / "history.json").read_text())
        config = json.loads((path / "config.json").read_text())
        x = [r["epoch"] for r in history]
        fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), constrained_layout=True)
        for ax, term in zip(axes, ("total", "reconstruction", "kl")):
            for split, label in (("train", "Train (sampled z)"), ("val", "Val (posterior mean)")):
                ax.plot(x, [r[split][term] for r in history], label=label, linewidth=1.4)
            ax.set(xlabel="Epoch", ylabel="Loss per image", title=term.capitalize())
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=7)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{path.name}_loss.{extension}", dpi=200)
        plt.close(fig)
        best = min(history, key=lambda r: r["selection_loss"])
        metrics = sorted(path.glob("metrics-test*.json"))
        if not metrics:
            metrics = [None]
        for metric in metrics:
            values = json.loads(metric.read_text()) if metric else {}
            rows.append(dict(run=path.name, latent_dim=config["latent_dim"],
                        image_size=config["image_size"], best_epoch=best["epoch"],
                        val_selection_loss=best["selection_loss"],
                        fid=values.get("fid", ""), inception_score=values.get("inception_score", ""),
                        n_real=values.get("num_real", ""), seed=values.get("seed", "")))
    if not rows:
        raise ValueError("No runs")
    with (output / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output.resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--output", default="runs/report_figures")
    args = p.parse_args()
    main(args.runs, args.output)
