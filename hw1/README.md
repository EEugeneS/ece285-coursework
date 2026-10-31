# ECE285 Homework 1

Convolutional VAE for the Kaggle chest X-ray dataset (version 2). Model details,
hyperparameters, results, and limitations are described in the separate report.

## Setup

Run in the ECE285 DSMLP GPU environment. The recorded experiments used PyTorch
2.2.1+cu121 and torchvision 0.17.1+cu121; package versions are saved in
`recorded_results/environment.json`.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt kagglehub
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Reproduce

Download the dataset and create the audited train/validation/test split:

```bash
python download_data.py
dataset_root="$(python -c 'import json; print(json.load(open("data/download.json"))["data_root"])')"
python prepare_data.py --data-root "$dataset_root" --output data/manifest.json
```

If the dataset is already available, set `dataset_root` to the directory containing
`train/`, `val/`, and `test/` and skip the download. The saved reference split is
`recorded_results/manifest.json` (4,257 train / 480 validation / 618 test images).

Train both configurations, evaluate the selected checkpoints, and generate images:

```bash
for latent in 128 64; do
    run="runs/z${latent}_seed285"
    python train.py --data-root "$dataset_root" --manifest data/manifest.json \
        --out "$run" --latent-dim "$latent" --epochs 50 --seed 285 --device cuda
    python evaluate.py --data-root "$dataset_root" --manifest data/manifest.json \
        --checkpoint "$run/best.pt" --output "$run/metrics-test.json" \
        --seed 2026 --batch-size 16 --device cuda
    python generate.py --checkpoint "$run/best.pt" --out "$run/generated" \
        --seed 2026 --device cuda
done
python plot_results.py runs/z128_seed285 runs/z64_seed285
```

Use new output directories for a fresh run. To continue an interrupted run, add
`--resume "$run/latest.pt"` to the matching training command. GPU/library changes
can affect exact numerical reproducibility.

`recorded_results/` contains the original configs, loss histories, metrics,
environment record, and split manifest. Datasets and checkpoints are not included;
checkpoints are produced by training. Saved results can be plotted without retraining:

```bash
python plot_results.py recorded_results/z128_seed285 recorded_results/z64_seed285
```

Evaluation follows the assigned Ignite tutorial with Ignite 0.5.3's default
1,000-dimensional Inception softmax features. These FID values are not comparable
with standard 2,048-dimensional pool3 FID.
