# UniSRec

This repository contains a PyTorch implementation of a sequential recommendation pipeline based on UniSRec, with a SASRec baseline for comparison.

The code supports:

- preparing Amazon Review and Online Retail datasets;
- encoding item text with a Hugging Face BERT model;
- pretraining UniSRec with text-aware sequence objectives;
- fine-tuning UniSRec for next-item recommendation;
- evaluating Recall@K and NDCG@K;
- training and evaluating a SASRec baseline.

## Project Structure

```text
.
├── run.py              # Main CLI entry point
├── prepare_data.py     # Dataset download, preprocessing, k-core filtering, text encoding
├── datasets.py         # PyTorch datasets and collate functions
├── model.py            # UniSRec model
├── pretrain.py         # UniSRec pretraining loop
├── finetune.py         # UniSRec fine-tuning loop
├── evaluate.py         # Full-sort evaluation utilities
├── model_sasrec.py     # SASRec baseline model
├── train_sasrec.py     # SASRec training and evaluation
├── loss.py             # Training objectives
├── utils.py            # Checkpoint and tensor helpers
└── requirements.txt    # Python dependencies
```

## Requirements

- Python 3.10 or newer is recommended.
- PyTorch 2.1 or newer.
- A CUDA-capable GPU is recommended for BERT encoding and model training.
- Kaggle CLI credentials are required only when downloading the Online Retail dataset automatically.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

All commands are executed through `run.py`.

The preparation step writes a serialized `data.pt` file containing indexed users/items, train/validation/test samples, item texts, and optionally BERT item text embeddings.

### Amazon Review Datasets

Supported short dataset names:

- `Food`
- `Home`
- `CDs`
- `Kindle`
- `Movies`
- `Pantry`
- `Scientific`
- `Instruments`
- `Arts`
- `Office`

Prepare an Amazon dataset for downstream training:

```bash
python run.py prepare \
  --kind amazon \
  --datasets Food \
  --out data/amazon_food \
  --download \
  --split_mode downstream \
  --device cuda
```

Prepare multiple Amazon domains for UniSRec pretraining:

```bash
python run.py prepare \
  --kind amazon \
  --datasets Food Home Movies \
  --out data/amazon_pretrain \
  --download \
  --split_mode pretrain \
  --device cuda
```

### Online Retail Dataset

To download through Kaggle and prepare Online Retail:

```bash
python run.py prepare \
  --kind or \
  --out data/online_retail \
  --download \
  --split_mode downstream \
  --device cuda
```

If the CSV already exists locally:

```bash
python run.py prepare \
  --kind or \
  --out data/online_retail \
  --or_csv data/raw/or/data-utf8.csv \
  --split_mode downstream \
  --device cuda
```

### Preparation Options

Useful options:

- `--max_len`: maximum interaction sequence length, default `50`.
- `--user_k`: minimum user interaction count for k-core filtering, default `5`.
- `--item_k`: minimum item interaction count for k-core filtering, default `5`.
- `--bert`: Hugging Face encoder used for item text, default `bert-base-uncased`.
- `--bert_batch_size`: text encoding batch size, default `32`.
- `--skip_bert`: skip text embedding generation.

Note: UniSRec commands require `item_text_embs` in `data.pt`. Do not use `--skip_bert` if you plan to run UniSRec pretraining, fine-tuning, or evaluation. SASRec can run without text embeddings.

## UniSRec Workflow

### 1. Pretrain

Use a dataset prepared with `--split_mode pretrain`:

```bash
python run.py pretrain \
  --data data/amazon_pretrain/data.pt \
  --save checkpoints/unisrec_pretrained.pt \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --device cuda
```

### 2. Fine-tune

Use a dataset prepared with `--split_mode downstream`:

```bash
python run.py finetune \
  --data data/amazon_food/data.pt \
  --pretrained checkpoints/unisrec_pretrained.pt \
  --save checkpoints/unisrec_finetuned.pt \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --device cuda
```

Fine-tuning from scratch is also supported by omitting `--pretrained`.

### 3. Evaluate

```bash
python run.py eval \
  --data data/amazon_food/data.pt \
  --ckpt checkpoints/unisrec_finetuned.pt \
  --split test \
  --top_k 10 50 \
  --batch_size 128 \
  --device cuda
```

Evaluation prints metrics to stdout and saves JSON files in `results/`.

## SASRec Baseline

Train SASRec:

```bash
python run.py sasrec \
  --mode train \
  --data data/amazon_food/data.pt \
  --save checkpoints/sasrec.pt \
  --epochs 10 \
  --batch_size 256 \
  --device cuda
```

Evaluate SASRec:

```bash
python run.py sasrec \
  --mode eval \
  --data data/amazon_food/data.pt \
  --ckpt checkpoints/sasrec.pt \
  --split test \
  --top_k 10 50 \
  --batch_size 128 \
  --device cuda
```

## Common Model Options

UniSRec options:

- `--hidden_size`, default `64`
- `--num_layers`, default `2`
- `--num_heads`, default `2`
- `--num_experts`, default `8`
- `--dropout`, default `0.2`
- `--use_id_embedding`, disabled by default

SASRec options:

- `--hidden_size`, default `64`
- `--num_layers`, default `2`
- `--num_heads`, default `2`
- `--dropout`, default `0.2`

Training options:

- `--epochs`
- `--batch_size`
- `--lr`
- `--num_workers`
- `--grad_clip`
- `--device`

## Outputs

Typical generated files:

- `data/<dataset>/data.pt`: preprocessed dataset pack.
- `checkpoints/*.pt`: model checkpoints.
- `results/*.json`: timestamped evaluation results.
- `results/latest_<model>_<dataset>_<split>.json`: latest evaluation result for a model/dataset/split combination.

## Example End-to-End Run

```bash
mkdir -p data checkpoints results

python run.py prepare \
  --kind amazon \
  --datasets Food Home Movies \
  --out data/amazon_pretrain \
  --download \
  --split_mode pretrain \
  --device cuda

python run.py prepare \
  --kind amazon \
  --datasets Food \
  --out data/amazon_food \
  --download \
  --split_mode downstream \
  --device cuda

python run.py pretrain \
  --data data/amazon_pretrain/data.pt \
  --save checkpoints/unisrec_pretrained.pt \
  --device cuda

python run.py finetune \
  --data data/amazon_food/data.pt \
  --pretrained checkpoints/unisrec_pretrained.pt \
  --save checkpoints/unisrec_finetuned.pt \
  --device cuda

python run.py eval \
  --data data/amazon_food/data.pt \
  --ckpt checkpoints/unisrec_finetuned.pt \
  --split test \
  --top_k 10 50 \
  --device cuda
```
