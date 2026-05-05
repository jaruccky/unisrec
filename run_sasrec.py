import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_sasrec import SASRec


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_last_hidden(seq_output, item_seq_ids):
    """
    seq_output:    [B, L, H]
    item_seq_ids:  [B, L]

    Ожидается right padding:
      [10, 25, 31, 0, 0]
    Берём hidden последнего настоящего item-а.
    """
    lengths = (item_seq_ids != 0).long().sum(dim=1) - 1
    lengths = lengths.clamp_min(0)

    batch_idx = torch.arange(item_seq_ids.size(0), device=item_seq_ids.device)
    return seq_output[batch_idx, lengths]


class SASRecTrainDataset(Dataset):
    """
    Dataset под data.pt из prepare_unisrec_data.py.

    samples:
      list[(item_seq_ids, target_id)]

    item_seq_ids уже right-padded до max_len:
      [i1, i2, i3, 0, 0]

    target_id:
      следующий item, который надо предсказать.
    """

    def __init__(self, samples, num_items):
        self.samples = samples
        self.num_items = int(num_items)

    def __len__(self):
        return len(self.samples)

    def _sample_negative(self, seq_ids, pos_id):
        """
        Сэмплим negative item, которого нет в истории и который не равен positive.
        Для больших каталогов этот while обычно быстро заканчивается.
        """
        seen = set(int(x) for x in seq_ids if int(x) != 0)
        seen.add(int(pos_id))

        while True:
            neg_id = random.randint(1, self.num_items)
            if neg_id not in seen:
                return neg_id

    def __getitem__(self, idx):
        seq_ids, pos_id = self.samples[idx]

        neg_id = self._sample_negative(seq_ids, pos_id)

        return {
            "item_seq_ids": torch.tensor(seq_ids, dtype=torch.long),
            "pos_ids": torch.tensor(pos_id, dtype=torch.long),
            "neg_ids": torch.tensor(neg_id, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch):
        item_seq_ids = torch.stack([x["item_seq_ids"] for x in batch])
        pos_ids = torch.stack([x["pos_ids"] for x in batch])
        neg_ids = torch.stack([x["neg_ids"] for x in batch])

        return {
            "item_seq_ids": item_seq_ids,
            "pos_ids": pos_ids,
            "neg_ids": neg_ids,
        }


@torch.no_grad()
def evaluate_full_ranking(model, samples, num_items, device, batch_size=256, top_ks=(10, 50)):
    """
    Full-ranking evaluation на valid/test samples.

    scores shape:
      [B, num_items + 1]

    item 0 — padding, его зануляем через -inf.
    """
    if not samples:
        return {}

    model.eval()

    metrics = {}
    for k in top_ks:
        metrics[f"Recall@{k}"] = 0.0
        metrics[f"NDCG@{k}"] = 0.0

    total = 0

    for start in tqdm(range(0, len(samples), batch_size), desc="eval", dynamic_ncols=True):
        batch_samples = samples[start:start + batch_size]

        item_seq_ids = torch.tensor(
            [x[0] for x in batch_samples],
            dtype=torch.long,
            device=device,
        )
        target_ids = torch.tensor(
            [x[1] for x in batch_samples],
            dtype=torch.long,
            device=device,
        )

        seq_output = model(item_seq_ids)
        last_hidden = get_last_hidden(seq_output, item_seq_ids)

        scores = last_hidden @ model.item_embedding.weight.T
        scores[:, 0] = -1e9

        target_scores = scores[torch.arange(scores.size(0), device=device), target_ids]
        ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1

        for k in top_ks:
            hit = (ranks <= k).float()
            ndcg = hit / torch.log2(ranks.float() + 1.0)

            metrics[f"Recall@{k}"] += hit.sum().item()
            metrics[f"NDCG@{k}"] += ndcg.sum().item()

        total += item_seq_ids.size(0)

    for key in metrics:
        metrics[key] /= max(total, 1)

    return metrics


def print_metrics(title, metrics):
    if not metrics:
        return

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


def train_one_epoch(model, dataloader, optimizer, device, epoch, epochs, grad_clip=None):
    model.train()

    total_loss = 0.0
    steps = 0

    progress = tqdm(
        dataloader,
        desc=f"epoch {epoch}/{epochs}",
        dynamic_ncols=True,
    )

    for batch in progress:
        item_seq_ids = batch["item_seq_ids"].to(device, non_blocking=True)
        pos_ids = batch["pos_ids"].to(device, non_blocking=True)
        neg_ids = batch["neg_ids"].to(device, non_blocking=True)

        seq_output = model(item_seq_ids)
        last_hidden = get_last_hidden(seq_output, item_seq_ids)

        pos_emb = model.item_embedding(pos_ids)
        neg_emb = model.item_embedding(neg_ids)

        pos_logits = (last_hidden * pos_emb).sum(dim=-1)
        neg_logits = (last_hidden * neg_emb).sum(dim=-1)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits,
            torch.ones_like(pos_logits),
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits,
            torch.zeros_like(neg_logits),
        )
        loss = pos_loss + neg_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        steps += 1
        total_loss += float(loss.item())
        avg_loss = total_loss / steps

        postfix = {
            "loss": f"{loss.item():.4f}",
            "avg": f"{avg_loss:.4f}",
        }

        if device.startswith("cuda") and torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            postfix["gpu_mem"] = f"{mem_gb:.2f}G"

        progress.set_postfix(postfix)

    return total_loss / max(steps, 1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save", type=str, default="checkpoints/sasrec.pt")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=None)

    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 0 = не валидировать во время обучения.
    # 1 = валидировать после каждой эпохи.
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=256)

    args = parser.parse_args()

    set_seed(args.seed)

    device = args.device
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pack = safe_torch_load(args.data, map_location="cpu")

    train_samples = pack["train_samples"]
    valid_samples = pack.get("valid_samples", [])
    num_items = int(pack["num_items"])
    max_len = int(pack["max_len"])

    dataset = SASRecTrainDataset(
        samples=train_samples,
        num_items=num_items,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=(device.startswith("cuda")),
    )

    model = SASRec(
        num_items=num_items,
        hidden_size=args.hidden_size,
        max_seq_len=max_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_params, trainable_params = count_parameters(model)

    print()
    print("=" * 60)
    print("SASRec train")
    print("=" * 60)
    print(f"data: {args.data}")
    print(f"save: {args.save}")
    print(f"device: {device}")
    print(f"num_items: {num_items}")
    print(f"max_len: {max_len}")
    print(f"train samples: {len(train_samples)}")
    print(f"valid samples: {len(valid_samples)}")
    print(f"batch_size: {args.batch_size}")
    print(f"steps per epoch: {math.ceil(len(dataset) / args.batch_size)}")
    print(f"epochs: {args.epochs}")
    print(f"lr: {args.lr}")
    print(f"hidden_size: {args.hidden_size}")
    print(f"num_layers: {args.num_layers}")
    print(f"num_heads: {args.num_heads}")
    print(f"dropout: {args.dropout}")
    print(f"model params: {total_params:,}")
    print(f"trainable params: {trainable_params:,}")
    print("=" * 60)
    print()

    best_metric = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
            grad_clip=args.grad_clip,
        )

        print(f"epoch {epoch}/{args.epochs}, train loss = {avg_loss:.4f}")

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            metrics = evaluate_full_ranking(
                model=model,
                samples=valid_samples,
                num_items=num_items,
                device=device,
                batch_size=args.eval_batch_size,
                top_ks=(10, 50),
            )
            print_metrics(f"Valid after epoch {epoch}", metrics)

            # Обычно выбираем по NDCG@10.
            current = metrics.get("NDCG@10")
            if current is not None and (best_metric is None or current > best_metric):
                best_metric = current
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                print(f"[best] epoch={epoch}, NDCG@10={best_metric:.6f}")

    if best_state is not None:
        print(f"[load best] best valid NDCG@10 = {best_metric:.6f}")
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "num_items": num_items,
        "max_len": max_len,
    }

    torch.save(checkpoint, save_path)

    print()
    print(f"[ok] saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()
