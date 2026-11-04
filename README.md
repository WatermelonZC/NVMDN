# NVMDN

This repository provides the PyTorch code and experimental data for
*Deep Reinforcement Learning for the Nested-Vehicle Scheduling Problem in Industrial Mother–Daughter Transport Systems*.

## Install

Requirements: Python 3.10 or newer and PyTorch 2.8 or newer.

```bash
pip install -r requirements.txt
```

Gurobi requires a license from the [official website](https://www.gurobi.com/product/download-center).

## Run

### One instance

```bash
python scripts/predict.py --instance instances/main/n20/01.json
```

| Parameter | Values |
| --- | --- |
| `--instance` | Instance JSON path, required |
| `--method` | `nvmdn` (default), `am`, `hadrl`, `tdrl`, `edd`, `ga`, `alns`, `diwo`, `gurobi` |
| `--decode` | `greedy` (default), `sample1024` |
| `--checkpoint` | Optional checkpoint path |
| `--device` | `cpu` (default), `cuda` |
| `--output` | Optional output JSON path |
| `--timeout` | Search budget in seconds, default `3600` |

### Instance directory

```bash
python scripts/run.py evaluate --instances instances/main/n20 --method nvmdn --output runs/n20
```

## Train

```bash
python scripts/train.py --tasks 20 --device cuda --output runs/nvmdn/n20
```

| Parameter | Values |
| --- | --- |
| `--tasks` | `10`, `20`, `50`, `100`, required |
| `--method` | `nvmdn` (default), `am`, `hadrl`, `tdrl` |
| `--device` | `cuda` (default), `cpu` |
| `--output` | Training directory, required |
| `--resume` | Full training checkpoint to resume |
| `--epochs` | Total training epochs, default `100` |

Training settings are in `configs/nvmdn.json`. Epoch checkpoints and TensorBoard logs are saved in the output directory.

## Instances and models

- `instances/` contains the test instances for all experiments.
- `pretrained/` contains NVMDN, AM, HA-DRL, and TDRL weights for N=10, 20, 50, and 100.

## Results

- `results/results.csv` contains 6,495 experiment records.
- `results/results.xlsx` contains summaries, instance records, and training curves.

## Attribution

The attention implementation is adapted from [attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) under the [MIT license](ATTENTION_LICENSE.txt).

The TDRL encoder modules are adapted from [MaRS-Net](https://github.com/Kemingjun/MaRS-Net). Their [MIT license](src/nets/tdrl_encoders/LICENSE) is retained. The learning baselines are adapted to NVSP.
