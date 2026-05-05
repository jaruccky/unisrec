import argparse

import torch
from torch.utils.data import DataLoader

from model import UniSRec
from train import train_unisrec_pretrain
from unisrec_dataset import UniSRecPretrainDataset


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save", type=str, default="unisrec_pretrained.pt")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--lambda_ss", type=float, default=1e-3)
    parser.add_argument("--item_drop_ratio", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=None)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    pack = safe_torch_load(args.data, map_location="cpu")

    if "item_text_embs" not in pack:
        raise RuntimeError(
            "В data.pt нет item_text_embs. "
            "Запусти prepare_unisrec_data.py без --skip_bert."
        )

    item_text_embs = pack["item_text_embs"]

    dataset = UniSRecPretrainDataset(
        samples=pack["train_samples"],
        item_text_embs=item_text_embs,
        max_len=pack["max_len"],
        item_drop_ratio=args.item_drop_ratio,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=(args.device.startswith("cuda")),
    )

    model = UniSRec(
        num_items=pack["num_items"],
        text_emb_dim=item_text_embs.shape[1],
        hidden_size=args.hidden_size,
        max_seq_len=pack["max_len"],
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        dropout=args.dropout,
        use_id_embedding=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_unisrec_pretrain(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=args.device,
        epochs=args.epochs,
        tau=args.tau,
        lambda_ss=args.lambda_ss,
        grad_clip=args.grad_clip,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "num_items": pack["num_items"],
        "max_len": pack["max_len"],
        "text_emb_dim": item_text_embs.shape[1],
    }

    torch.save(checkpoint, args.save)
    print(f"[ok] saved checkpoint: {args.save}")


if __name__ == "__main__":
    main()
