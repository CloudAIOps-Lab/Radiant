# Radiant

Implementation of Radiant: online multimodal fault diagnosis for cloud microservices.

## Requirements

```bash
pip install -r requirements.txt
```

## Data Format

Input data is expected in `.npz` format containing `metrics`, `logs`, `traces`, and `window_labels`.

## Quick Start

Train and evaluate the model:

```bash
python train.py --train_file data/train.npz --val_file data/val.npz --test_file data/test.npz
```
