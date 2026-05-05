import argparse
import math

import torch
from torch.utils.data import DataLoader

from model import UniSRec
from unisrec_dataset import UniSRecEvalDataset


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_last_hidden(seq_output, item_seq_ids):
    """
    seq_output: [B, L, H]
    item_seq_ids: [B, L]

    Ожидаем right padding:
      [10, 25, 31, 0, 0]
    """
    lengths = (item_seq_ids != 0).long().sum(dim=1) - 1
    lengths = lengths.clamp_min(0)

    batch_idx = torch.arange(item_seq_ids.size(0), device=item_seq_ids.device)

    return seq_output[batch_idx, lengths]


def update_metrics_from_scores(scores, target_ids, metrics, top_ks):
    """
    scores: [B, num_items]
        scores[:, 0] соответствует item_id = 1
        scores[:, 1] соответствует item_id = 2
        ...
    target_ids: [B]
        реальные item_id, начиная с 1
    """
    B = scores.size(0)
    device = scores.device

    target_pos = target_ids - 1

    target_scores = scores[torch.arange(B, device=device), target_pos]

    # rank = 1 + сколько товаров получили score строго больше target
    ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1

    for k in top_ks:
        hit = (ranks <= k).float()
        ndcg = hit / torch.log2(ranks.float() + 1.0)

        metrics[f"Recall@{k}"] += hit.sum().item()
        metrics[f"NDCG@{k}"] += ndcg.sum().item()

    return B


@torch.no_grad()
def encode_all_unisrec_items(model, item_text_embs, device, batch_size=4096):
    """
    Кодируем все item text embeddings через UniSRec MoE adaptor.

    item_text_embs: [num_items + 1, D]
      строка 0 — padding.

    return:
      all_item_embs: [num_items, H]
      соответствует item_id = 1..num_items
    """
    model.eval()

    item_text_embs = item_text_embs.to(device)

    outs = []

    for start in range(1, item_text_embs.size(0), batch_size):
        end = min(start + batch_size, item_text_embs.size(0))

        batch_text = item_text_embs[start:end]  # [chunk, D]
        batch_emb = model.encode_text_items(batch_text)  # [chunk, H]

        if model.use_id_embedding:
            ids = torch.arange(start, end, device=device)
            batch_emb = batch_emb + model.item_embedding(ids)

        outs.append(batch_emb.detach())

    return torch.cat(outs, dim=0)  # [num_items, H]


@torch.no_grad()
def evaluate_unisrec_full_ranking(
    model,
    eval_loader,
    item_text_embs,
    device,
    top_ks=(10, 50),
    item_encode_batch_size=4096,
):
    model.to(device)
    model.eval()

    candidate_embs = encode_all_unisrec_items(
        model=model,
        item_text_embs=item_text_embs,
        device=device,
        batch_size=item_encode_batch_size,
    )  # [num_items, H]

    metrics = {}

    for k in top_ks:
        metrics[f"Recall@{k}"] = 0.0
        metrics[f"NDCG@{k}"] = 0.0

    total = 0

    for batch in eval_loader:
        item_seq_ids = batch["item_seq_ids"].to(device)  # [B, L]
        item_seq_text_embs = batch["item_seq_text_embs"].to(device)  # [B, L, D]
        target_ids = batch["target_ids"].to(device)  # [B]

        seq_output = model(
            item_seq_text_embs=item_seq_text_embs,
            item_seq_ids=item_seq_ids,
        )  # [B, L, H]

        last_hidden = get_last_hidden(seq_output, item_seq_ids)  # [B, H]

        scores = last_hidden @ candidate_embs.T  # [B, num_items]

        total += update_metrics_from_scores(
            scores=scores,
            target_ids=target_ids,
            metrics=metrics,
            top_ks=top_ks,
        )

    for key in metrics:
        metrics[key] /= total

    return metrics


def get_sasrec_item_embedding_weight(model):
    """
    Поддерживает самые частые имена embedding-слоя.
    При необходимости поменяй под свой SASRec.
    """
    possible_names = [
        "item_embedding",
        "item_emb",
        "item_embeddings",
    ]

    for name in possible_names:
        if hasattr(model, name):
            emb = getattr(model, name)

            if hasattr(emb, "weight"):
                return emb.weight

    raise AttributeError(
        "Не нашёл item embedding в SASRec. "
        "Переименуй в коде get_sasrec_item_embedding_weight под свою модель."
    )


@torch.no_grad()
def sasrec_full_scores(model, item_seq_ids, num_items):
    """
    Универсальная обёртка под SASRec.

    Поддерживает 2 варианта:
      1) model(item_seq_ids) -> logits [B, num_items + 1]
      2) model(item_seq_ids) -> seq_output [B, L, H],
         тогда scores считаются как last_hidden @ item_embedding.weight.T
    """
    out = model(item_seq_ids)

    # Вариант 1: модель сразу вернула logits по всем item.
    if out.dim() == 2:
        logits = out  # [B, num_items + 1] или [B, num_items]

        if logits.size(1) == num_items + 1:
            logits = logits[:, 1:]  # убираем padding item 0

        return logits  # [B, num_items]

    # Вариант 2: модель вернула hidden states.
    if out.dim() == 3:
        seq_output = out  # [B, L, H]
        last_hidden = get_last_hidden(seq_output, item_seq_ids)

        item_weight = get_sasrec_item_embedding_weight(model)  # [num_items + 1, H]
        item_weight = item_weight.to(last_hidden.device)

        scores = last_hidden @ item_weight.T  # [B, num_items + 1]
        scores = scores[:, 1:]  # убираем padding item 0

        return scores  # [B, num_items]

    raise RuntimeError(
        f"Неожиданный output SASRec: shape={tuple(out.shape)}. "
        "Нужно либо [B, num_items + 1], либо [B, L, H]."
    )


@torch.no_grad()
def evaluate_sasrec_full_ranking(
    model,
    eval_loader,
    num_items,
    device,
    top_ks=(10, 50),
):
    model.to(device)
    model.eval()

    metrics = {}

    for k in top_ks:
        metrics[f"Recall@{k}"] = 0.0
        metrics[f"NDCG@{k}"] = 0.0

    total = 0

    for batch in eval_loader:
        item_seq_ids = batch["item_seq_ids"].to(device)  # [B, L]
        target_ids = batch["target_ids"].to(device)  # [B]

        scores = sasrec_full_scores(
            model=model,
            item_seq_ids=item_seq_ids,
            num_items=num_items,
        )  # [B, num_items]

        total += update_metrics_from_scores(
            scores=scores,
            target_ids=target_ids,
            metrics=metrics,
            top_ks=top_ks,
        )

    for key in metrics:
        metrics[key] /= total

    return metrics


def print_metrics(name, metrics):
    print()
    print("=" * 50)
    print(name)
    print("=" * 50)

    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--sasrec_ckpt", type=str, required=True)
    parser.add_argument("--unisrec_ckpt", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # параметры UniSRec должны совпадать с тем, как ты создавал модель
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--unisrec_use_id_embedding", action="store_true")

    args = parser.parse_args()

    device = args.device

    pack = safe_torch_load(args.data, map_location="cpu")

    item_text_embs = pack["item_text_embs"]
    num_items = pack["num_items"]

    test_dataset = UniSRecEvalDataset(
        samples=pack["test_samples"],
        item_text_embs=item_text_embs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
    )

    # ------------------------------------------------------------------
    # 1. Загружаем SASRec
    # ------------------------------------------------------------------
    # ВАЖНО:
    # Тут нужно импортировать ТВОЙ SASRec и создать его с теми же параметрами,
    # с которыми ты его обучал.
    #
    # Например:
    # from model_sasrec import SASRec
    #
    # sasrec = SASRec(
    #     num_items=num_items,
    #     hidden_size=args.hidden_size,
    #     max_seq_len=pack["max_len"],
    #     num_layers=args.num_layers,
    #     num_heads=args.num_heads,
    #     dropout=args.dropout,
    # )
    #
    # Ниже специально оставлен raise, чтобы ты не забыл подставить свой класс.
    # ------------------------------------------------------------------

    from model_sasrec import SASRec

    sasrec = SASRec(
        num_items=num_items,
        hidden_size=args.hidden_size,
        max_seq_len=pack["max_len"],
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )

    sasrec_ckpt = safe_torch_load(args.sasrec_ckpt, map_location="cpu")

    if "model_state_dict" in sasrec_ckpt:
        sasrec.load_state_dict(sasrec_ckpt["model_state_dict"])
    else:
        sasrec.load_state_dict(sasrec_ckpt)

    # ------------------------------------------------------------------
    # 2. Загружаем UniSRec
    # ------------------------------------------------------------------

    unisrec = UniSRec(
        num_items=num_items,
        text_emb_dim=item_text_embs.shape[1],
        hidden_size=args.hidden_size,
        max_seq_len=pack["max_len"],
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        dropout=args.dropout,
        use_id_embedding=args.unisrec_use_id_embedding,
    )

    unisrec_ckpt = safe_torch_load(args.unisrec_ckpt, map_location="cpu")

    if "model_state_dict" in unisrec_ckpt:
        unisrec.load_state_dict(unisrec_ckpt["model_state_dict"])
    else:
        unisrec.load_state_dict(unisrec_ckpt)

    # ------------------------------------------------------------------
    # 3. Сравниваем на одном и том же test set
    # ------------------------------------------------------------------

    sasrec_metrics = evaluate_sasrec_full_ranking(
        model=sasrec,
        eval_loader=test_loader,
        num_items=num_items,
        device=device,
        top_ks=(10, 50),
    )

    unisrec_metrics = evaluate_unisrec_full_ranking(
        model=unisrec,
        eval_loader=test_loader,
        item_text_embs=item_text_embs,
        device=device,
        top_ks=(10, 50),
    )

    print_metrics("SASRec", sasrec_metrics)
    print_metrics("UniSRec", unisrec_metrics)

    print()
    print("=" * 50)
    print("Difference: UniSRec - SASRec")
    print("=" * 50)

    for key in sasrec_metrics:
        diff = unisrec_metrics[key] - sasrec_metrics[key]
        print(f"{key}: {diff:+.6f}")


if __name__ == "__main__":
    main()
