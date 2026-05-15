import json

import torch
from torch.utils.data import DataLoader

from datasets import FinetuneDataset, EvalDataset
from loss import sampled_ce_loss
from model_sasrec import SASRec
from utils import safe_load, load_state, save_checkpoint, checkpoint_config, last_hidden


def run_sasrec(args):
    pack = safe_load(args.data)

    item_text_embs = pack.get("item_text_embs")
    if item_text_embs is None:
        item_text_embs = torch.zeros(pack["num_items"] + 1, 1)

    default_config = {
        "num_items": pack["num_items"],
        "hidden_size": args.hidden_size,
        "max_seq_len": pack["max_len"],
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
    }

    ckpt = None
    if args.ckpt is not None:
        ckpt = safe_load(args.ckpt)
        config = checkpoint_config(ckpt, default_config)
    else:
        config = default_config

    model = SASRec(
        num_items=config["num_items"],
        hidden_size=config["hidden_size"],
        max_seq_len=config["max_seq_len"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    )

    if ckpt is not None:
        load_state(model, ckpt, strict=False)

    device = args.device
    model.to(device)

    if args.mode == "train":
        train_dataset = FinetuneDataset(
            samples=pack["train_samples"],
            item_text_embs=item_text_embs,
            num_items=pack["num_items"],
            num_negatives=args.num_negatives,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=train_dataset.collate_fn,
            pin_memory=device.startswith("cuda"),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        losses = []

        for epoch in range(1, args.epochs + 1):
            model.train()

            total_loss = 0.0
            steps = 0

            for batch in train_loader:
                seq_ids = batch["item_seq_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                neg_ids = batch["neg_ids"].to(device)

                seq_output = model(seq_ids)
                seq_emb = last_hidden(seq_output, seq_ids)

                pos_emb = model.item_embedding(target_ids)
                neg_emb = model.item_embedding(neg_ids)

                loss = sampled_ce_loss(seq_emb, pos_emb, neg_emb)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()

                total_loss += loss.item()
                steps += 1

            avg_loss = total_loss / max(steps, 1)
            losses.append(avg_loss)

            print(f"epoch={epoch} loss={avg_loss:.6f}")

        save_checkpoint(args.save, model, config)

        print(args.save)
        return losses

    if args.mode == "eval":
        if args.ckpt is None:
            raise RuntimeError("Для eval нужно указать --ckpt")

        eval_dataset = EvalDataset(
            samples=pack[f"{args.split}_samples"],
            item_text_embs=item_text_embs,
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=eval_dataset.collate_fn,
            pin_memory=device.startswith("cuda"),
        )

        model.eval()

        metrics = {}
        for k in args.top_k:
            metrics[f"Recall@{k}"] = 0.0
            metrics[f"NDCG@{k}"] = 0.0

        total = 0

        with torch.no_grad():
            for batch in eval_loader:
                seq_ids = batch["item_seq_ids"].to(device)
                target_ids = batch["target_ids"].to(device)

                scores = model.full_sort_scores(seq_ids)

                batch_size = scores.size(0)
                target_pos = target_ids - 1

                target_scores = scores[
                    torch.arange(batch_size, device=device),
                    target_pos,
                ]

                ranks = scores.gt(target_scores.unsqueeze(1)).sum(dim=1) + 1

                for k in args.top_k:
                    hit = ranks.le(k).float()
                    metrics[f"Recall@{k}"] += hit.sum().item()
                    metrics[f"NDCG@{k}"] += (
                        hit / torch.log2(ranks.float() + 1)
                    ).sum().item()

                total += batch_size

        metrics = {k: v / max(total, 1) for k, v in metrics.items()}

        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics

    raise RuntimeError(f"Неизвестный mode: {args.mode}")