import argparse
import json

import torch
from torch.utils.data import DataLoader

from datasets import EvalDataset, FinetuneDataset, PretrainDataset
from evaluate import evaluate
from finetune import configure_finetune, train_finetune, trainable_parameters
from model import UniSRec
from prepare_data import prepare_amazon, prepare_or
from pretrain import train_pretrain
from utils import checkpoint_config, load_state, safe_load, save_checkpoint


def add_model_args(p):
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--num_experts", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--use_id_embedding", action="store_true")


def model_config(args, pack):
    return {
        "num_items": pack["num_items"],
        "text_emb_dim": pack["item_text_embs"].shape[1],
        "hidden_size": args.hidden_size,
        "max_seq_len": pack["max_len"],
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "num_experts": args.num_experts,
        "dropout": args.dropout,
        "use_id_embedding": args.use_id_embedding,
    }


def build_model(config):
    keys = [
        "num_items",
        "text_emb_dim",
        "hidden_size",
        "max_seq_len",
        "num_layers",
        "num_heads",
        "num_experts",
        "dropout",
        "use_id_embedding",
    ]
    return UniSRec(**{k: config[k] for k in keys})


def load_pack(path):
    pack = safe_load(path)
    if "item_text_embs" not in pack:
        raise RuntimeError("data.pt должен содержать item_text_embs")
    return pack


def cmd_prepare(args):
    if args.kind == "amazon":
        path = prepare_amazon(
            datasets=args.datasets,
            out_dir=args.out,
            raw_root=args.amazon_raw,
            download=args.download,
            force_download=args.force_download,
            split_mode=args.split_mode,
            max_len=args.max_len,
            user_k=args.user_k,
            item_k=args.item_k,
            bert=args.bert,
            bert_batch_size=args.bert_batch_size,
            device=args.device,
            skip_bert=args.skip_bert,
        )
    else:
        path = prepare_or(
            out_dir=args.out,
            raw_dir=args.or_raw_dir,
            csv_path=args.or_csv,
            download=args.download,
            force_download=args.force_download,
            split_mode=args.split_mode,
            max_len=args.max_len,
            user_k=args.user_k,
            item_k=args.item_k,
            bert=args.bert,
            bert_batch_size=args.bert_batch_size,
            device=args.device,
            skip_bert=args.skip_bert,
        )
    print(path)


def cmd_pretrain(args):
    pack = load_pack(args.data)
    dataset = PretrainDataset(pack["train_samples"], pack["item_text_embs"], pack["max_len"], args.item_drop_ratio)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate_fn, pin_memory=args.device.startswith("cuda"))
    config = model_config(args, pack)
    config["use_id_embedding"] = False
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_pretrain(model, loader, optimizer, args.device, args.epochs, args.tau, args.lambda_ss, args.grad_clip)
    save_checkpoint(args.save, model, config)
    print(args.save)


def cmd_finetune(args):
    pack = load_pack(args.data)
    config = model_config(args, pack)
    model = build_model(config)
    if args.pretrained:
        load_state(model, safe_load(args.pretrained), strict=False)
    configure_finetune(model, train_all=args.train_all)
    dataset = FinetuneDataset(pack["train_samples"], pack["item_text_embs"], pack["num_items"], args.num_negatives)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate_fn, pin_memory=args.device.startswith("cuda"))
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=args.lr)
    train_finetune(model, loader, optimizer, args.device, args.epochs, args.grad_clip)
    save_checkpoint(args.save, model, config)
    print(args.save)


def cmd_eval(args):
    pack = load_pack(args.data)
    ckpt = safe_load(args.ckpt)
    config = checkpoint_config(ckpt, model_config(args, pack))
    model = build_model(config)
    load_state(model, ckpt, strict=False)
    dataset = EvalDataset(pack[f"{args.split}_samples"], pack["item_text_embs"])
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=dataset.collate_fn, pin_memory=args.device.startswith("cuda"))
    metrics = evaluate(model, loader, pack["item_text_embs"], args.device, tuple(args.top_k), args.item_batch_size)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--kind", choices=["amazon", "or"], required=True)
    p.add_argument("--datasets", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--force_download", action="store_true")
    p.add_argument("--amazon_raw", default="data/raw/amazon")
    p.add_argument("--or_raw_dir", default="data/raw/or")
    p.add_argument("--or_csv", default="data/raw/or/data-utf8.csv")
    p.add_argument("--split_mode", choices=["pretrain", "downstream"], required=True)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--user_k", type=int, default=5)
    p.add_argument("--item_k", type=int, default=5)
    p.add_argument("--bert", default="bert-base-uncased")
    p.add_argument("--bert_batch_size", type=int, default=32)
    p.add_argument("--skip_bert", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("pretrain")
    p.add_argument("--data", required=True)
    p.add_argument("--save", default="unisrec_pretrained.pt")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--lambda_ss", type=float, default=1e-3)
    p.add_argument("--item_drop_ratio", type=float, default=0.2)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_model_args(p)
    p.set_defaults(func=cmd_pretrain)

    p = sub.add_parser("finetune")
    p.add_argument("--data", required=True)
    p.add_argument("--pretrained", default=None)
    p.add_argument("--save", default="unisrec_finetuned.pt")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_negatives", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--train_all", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_model_args(p)
    p.set_defaults(func=cmd_finetune)

    p = sub.add_parser("eval")
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--top_k", type=int, nargs="+", default=[10, 50])
    p.add_argument("--item_batch_size", type=int, default=4096)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_model_args(p)
    p.set_defaults(func=cmd_eval)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
