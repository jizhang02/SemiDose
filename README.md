# SemiDose

**Semi-supervised dose prediction for targeted radionuclide therapy with synthetic phantoms**

Patient-specific absorbed-dose estimation is important in targeted radionuclide therapy, but labeled post-therapy data are costly and scarce. SemiDose studies how labeled and unlabeled pre-therapy synthetic phantom images can be combined for organ-level dose regression. The repository provides the model, data-loading code, five semi-supervised training variants, and a data-free smoke test for reproducibility.

[Paper](https://doi.org/10.1088/1361-6560/ae36df) · [arXiv](https://arxiv.org/abs/2503.05367) · [MIT License](LICENSE)

![SemiDose workflow](src/workflow.png)

## Overview

The code accompanies the *Physics in Medicine & Biology* paper **“Semi-supervised learning for dose prediction in targeted radionuclide therapy: a synthetic data study.”** The study uses anatomically and physiologically informed synthetic phantoms to investigate dose regression when only part of the training set is labeled.

The supported organ targets in the current data pipeline are bladder, kidneys, liver, spleen, pancreas, prostate, rectum, and salivary glands.

## Method

Each model receives a three-channel 2D input composed of CT, PET, and an organ mask, and predicts one organ-level dose value. The repository includes:

- `train_SimRegMatch.py`: similarity-calibrated pseudo-label regression;
- `train_regfixmatch.py`: regression FixMatch;
- `train_regmeanteacher.py`: regression Mean Teacher;
- `train_regict.py`: regression interpolation consistency training;
- `train_reggan.py`: adversarial semi-supervised regression.

The experiments use five-fold cross-validation. Weak and strong views of unlabeled images provide consistency and pseudo-label objectives alongside the supervised regression loss.

## Quick smoke test

This example uses random tensors only. It does not require patient or phantom data and does not download pretrained weights.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-smoke.txt
python -m examples.smoke_forward
```

Expected shapes:

```text
input:      (2, 3, 256, 256)
prediction: (2, 1)
encoding:   (2, 2048)
```

Run the automated check with:

```bash
python -m unittest tests.test_smoke -v
```

## Installation

The reference environment uses Python 3.10. For the complete training pipeline:

```bash
python -m venv .venv
# activate the environment as shown above
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The pinned files provide a reproducible CPU baseline. For a CUDA build, install the appropriate PyTorch build for your driver first, then install the remaining packages.

## Data preparation

The image data are not distributed in this repository. Prepare the following structure:

```text
wholebody_2D/
├── ct/
│   └── <phantom_id>.png
├── pet/
│   └── <phantom_id>.mhd
├── mask/
│   └── <organ>/
│       └── <phantom_id>.png
└── dose_organs.csv
```

`dose_organs.csv` must contain a `phantoms` column, a `fold` column with fold IDs 0–9, and target columns named `dose_<organ>`. Image identifiers in `phantoms` are stored without file extensions.

## Training

Set the dataset path, organ, label count, batch size, and other experiment parameters in [`src/hyper_parameter.py`](src/hyper_parameter.py). Then run a method from the repository root, for example:

```bash
cd src
python train_SimRegMatch.py
```

The scripts currently assume a CUDA-capable device and save fold checkpoints plus `train_log.txt` in the working directory. Other methods can be launched by replacing the script name with one of the variants listed under **Method**.

## Evaluation and results

Each training script selects the checkpoint with the best validation R² and reports test R², Pearson correlation, MAE, and MAPE for every fold, followed by the five-fold mean and standard deviation. Because the dataset is not bundled, this README does not present unverifiable rerun numbers; the complete experimental tables and comparisons are available in the [published paper](https://doi.org/10.1088/1361-6560/ae36df).

## Repository layout

```text
src/data_load.py          dataset split, loading, and augmentation
src/model_load.py         regression and adversarial model definitions
src/hyper_parameter.py    experiment paths and hyperparameters
src/train_*.py            training, validation, and test workflows
examples/smoke_forward.py data-free forward-pass example
tests/test_smoke.py       minimal model regression test
```

## Citation

```bibtex
@article{Zhang2026SemiDose,
  author  = {Zhang, Jing and Bousse, Alexandre and Pham, Chi-Hieu and Shi, Kuangyu and Bert, Julien},
  title   = {Semi-supervised learning for dose prediction in targeted radionuclide therapy: a synthetic data study},
  journal = {Physics in Medicine & Biology},
  year    = {2026},
  volume  = {71},
  number  = {2},
  pages   = {025005},
  doi     = {10.1088/1361-6560/ae36df}
}
```

## Contact

Questions and reproducibility reports are welcome through [GitHub Issues](https://github.com/jizhang02/SemiDose/issues).
