# Homework 2: WikiArt DCGAN

Train an unconditional 64x64 RGB GAN on Kaggle `steubk/wikiart`, version 1.
Architecture, hyperparameters, results, and limitations are described in the report.

## Run on DSMLP

Use the ECE285 GPU environment. Keep the 34 GB download on the container's temporary
disk; the processed image cache needs approximately 1 GB of persistent storage.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python prepare_data.py --archive /tmp/ece285-wikiart-v1.zip --remove-archive

python train.py --out runs/balanced_seed285 --lr-d 0.0002
python train.py --out runs/slower_d_seed285 --lr-d 0.0001

for setting in balanced slower_d; do
    run="runs/${setting}_seed285"
    python evaluate.py --checkpoint "$run/best.pt" --output "$run/metrics-test.json"
    python generate.py --checkpoint "$run/best.pt" --out "$run/generated"
done
python plot_results.py runs/balanced_seed285 runs/slower_d_seed285
```

Training defaults: 25 epochs, batch 128, latent size 100, Adam (0.5, 0.999),
generator learning rate 0.0002, seed 285. Validation FID selects the checkpoint;
test images are used only for final evaluation. Sampling seed is 2026.
Use new output directories for fresh runs, or pass `--resume runs/NAME/latest.pt`
to continue an interrupted run with the same configuration.
GPU and library changes can affect exact numerical reproducibility.

The split removes exact duplicates after RGB resizing/cropping and reserves
1,000 validation and 2,000 test images, stratified by style. Dataset images,
cached arrays, and checkpoints are excluded from Git. Recorded numeric results
and the split manifest are in `recorded_results/`; versions and hardware are
recorded alongside the experiment configurations.

Evaluation uses Ignite 0.5.3's tutorial FID with 1,000-dimensional Inception
softmax features, not standard pool3 FID. Images are mapped to [0,1] before PIL
uint8 conversion and bilinear 299px resizing.

Implementation follows the [PyTorch DCGAN example](https://github.com/pytorch/examples/tree/main/dcgan)
and the assigned [Ignite evaluation tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).
