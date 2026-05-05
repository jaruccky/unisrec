import argparse
import csv
import gzip
import html
import json
import os
import re
import shutil
import subprocess
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

AMAZON_NAME = {
    # Pretrain datasets from UniSRec paper
    "Food": "Grocery_and_Gourmet_Food",
    "Home": "Home_and_Kitchen",
    "CDs": "CDs_and_Vinyl",
    "Kindle": "Kindle_Store",
    "Movies": "Movies_and_TV",
    # Downstream cross-domain datasets from UniSRec paper
    "Pantry": "Prime_Pantry",
    "Scientific": "Industrial_and_Scientific",
    "Instruments": "Musical_Instruments",
    "Arts": "Arts_Crafts_and_Sewing",
    "Office": "Office_Products",
}

AMAZON_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"


def is_downloaded(path, min_bytes=1024):
    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def download_file(url, dst, force=False):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if is_downloaded(dst) and not force:
        print(f"[skip] already downloaded: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")

    if tmp.exists():
        tmp.unlink()

    print(f"[download] {url}")
    print(f"       -> {dst}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with urllib.request.urlopen(req) as response:
            with open(tmp, "wb") as fout:
                shutil.copyfileobj(response, fout)

        if not is_downloaded(tmp):
            raise RuntimeError(f"Downloaded file is too small: {tmp}")

        tmp.rename(dst)

    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    print(f"[ok] saved: {dst}")


def ensure_amazon_dataset_downloaded(short_name, raw_root, force=False):
    if short_name not in AMAZON_NAME:
        raise ValueError(
            f"Unknown Amazon dataset short name: {short_name}. "
            f"Available: {list(AMAZON_NAME.keys())}"
        )

    raw_root = Path(raw_root)
    full_name = AMAZON_NAME[short_name]

    ratings_path = raw_root / "Ratings" / f"{full_name}.csv"
    meta_path = raw_root / "Metadata" / f"meta_{full_name}.json.gz"

    ratings_url = f"{AMAZON_BASE_URL}/categoryFilesSmall/{full_name}.csv"
    meta_url = f"{AMAZON_BASE_URL}/metaFiles2/meta_{full_name}.json.gz"

    download_file(ratings_url, ratings_path, force=force)
    download_file(meta_url, meta_path, force=force)

    return ratings_path, meta_path


def ensure_online_retail_downloaded(or_dir, force=False):
    """
    Скачивает Online Retail через Kaggle CLI.

    Перед этим надо:
      pip install kaggle

    И положить Kaggle token сюда:
      ~/.kaggle/kaggle.json
    """
    or_dir = Path(or_dir)
    or_dir.mkdir(parents=True, exist_ok=True)

    utf8_csv = or_dir / "data-utf8.csv"
    latin1_csv = or_dir / "data.csv"

    if is_downloaded(utf8_csv) and not force:
        print(f"[skip] already prepared: {utf8_csv}")
        return utf8_csv

    if not is_downloaded(latin1_csv) or force:
        if shutil.which("kaggle") is None:
            raise RuntimeError(
                "Не найден kaggle CLI.\n"
                "Установи:\n"
                "  pip install kaggle\n\n"
                "И настрой Kaggle API token:\n"
                "  ~/.kaggle/kaggle.json"
            )

        print("[download] Kaggle dataset: carrie1/ecommerce-data")

        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "carrie1/ecommerce-data",
                "-p",
                str(or_dir),
                "--unzip",
            ],
            check=True,
        )

    if not latin1_csv.exists():
        raise FileNotFoundError(
            f"Kaggle скачался, но {latin1_csv} не найден. "
            f"Проверь содержимое папки: {or_dir}"
        )

    print(f"[convert] latin1 -> utf-8: {latin1_csv} -> {utf8_csv}")

    with open(latin1_csv, "r", encoding="latin1", errors="ignore") as fin:
        with open(utf8_csv, "w", encoding="utf-8") as fout:
            for line in fin:
                fout.write(line)

    print(f"[ok] saved: {utf8_csv}")
    return utf8_csv


def flatten_text(x):
    if x is None:
        return ""

    if isinstance(x, str):
        return x

    if isinstance(x, dict):
        return " ".join(flatten_text(v) for v in x.values())

    if isinstance(x, list):
        return " ".join(flatten_text(v) for v in x)

    return str(x)


def clean_text(x):
    text = flatten_text(x)
    text = html.unescape(text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    if len(text) >= 4000:
        text = text[:4000]

    return text


def k_core_filter(interactions, user_k=5, item_k=5):
    """
    interactions: list[(user_raw, item_raw, timestamp)]
    Итеративно оставляем users/items с >= k взаимодействий.
    """
    interactions = list(interactions)

    while True:
        user_cnt = Counter(u for u, _, _ in interactions)
        item_cnt = Counter(i for _, i, _ in interactions)

        new_interactions = [
            (u, i, t)
            for u, i, t in interactions
            if user_cnt[u] >= user_k and item_cnt[i] >= item_k
        ]

        if len(new_interactions) == len(interactions):
            return new_interactions

        interactions = new_interactions


def build_sequences(interactions):
    user2events = defaultdict(list)

    for user, item, timestamp in interactions:
        user2events[user].append((timestamp, item))

    user2seq = {}

    for user, events in user2events.items():
        events.sort(key=lambda x: x[0])
        user2seq[user] = [item for _, item in events]

    return user2seq


def pad_right(seq, max_len):
    seq = seq[-max_len:]
    return seq + [0] * (max_len - len(seq))


def build_samples(user2seq_idx, max_len, split_mode):
    """
    split_mode = pretrain:
      Для каждой последовательности делаем prefix -> next item.

    split_mode = downstream:
      Leave-one-out:
        last item      -> test target
        item before it -> valid target
        previous items -> train.
    """
    train_samples = []
    valid_samples = []
    test_samples = []

    for _, seq in user2seq_idx.items():
        if len(seq) < 3:
            continue

        if split_mode == "pretrain":
            for t in range(1, len(seq)):
                hist = seq[:t]
                target = seq[t]
                train_samples.append((pad_right(hist, max_len), target))

        elif split_mode == "downstream":
            train_seq = seq[:-2]
            valid_target = seq[-2]
            test_target = seq[-1]

            for t in range(1, len(train_seq)):
                hist = train_seq[:t]
                target = train_seq[t]
                train_samples.append((pad_right(hist, max_len), target))

            valid_samples.append((pad_right(train_seq, max_len), valid_target))
            test_samples.append(
                (pad_right(train_seq + [valid_target], max_len), test_target)
            )

        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

    return train_samples, valid_samples, test_samples


def read_amazon_ratings(path):
    """
    Amazon categoryFilesSmall CSV format:
      item,user,rating,timestamp
    """
    interactions = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)

        for row in tqdm(reader, desc=f"read ratings {path.name}"):
            if len(row) != 4:
                continue

            item, user, _, timestamp = row

            try:
                timestamp = int(timestamp)
            except ValueError:
                continue

            interactions.append((user, item, timestamp))

    return interactions


def read_amazon_meta(path):
    """
    Возвращает:
      item_raw_id -> item_text

    Для Amazon берём:
      title + category/categories + brand
    """
    item2text = {}

    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc=f"read meta {path.name}"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = obj.get("asin")

            if asin is None:
                continue

            title = clean_text(obj.get("title", ""))
            category = clean_text(obj.get("category", obj.get("categories", "")))
            brand = clean_text(obj.get("brand", ""))

            parts = [title, category, brand]
            text = " ".join(x for x in parts if x).strip()

            item2text[asin] = text if text else "."

    return item2text


def load_one_amazon_dataset(short_name, raw_root, user_k, item_k):
    full_name = AMAZON_NAME[short_name]

    rating_path = raw_root / "Ratings" / f"{full_name}.csv"
    meta_path = raw_root / "Metadata" / f"meta_{full_name}.json.gz"

    if not rating_path.exists():
        raise FileNotFoundError(f"Нет ratings файла: {rating_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Нет metadata файла: {meta_path}")

    item2text_raw = read_amazon_meta(meta_path)
    meta_items = set(item2text_raw)

    interactions = read_amazon_ratings(rating_path)
    interactions = [(u, i, t) for u, i, t in interactions if i in meta_items]
    interactions = k_core_filter(interactions, user_k=user_k, item_k=item_k)

    # Префиксы нужны, чтобы при смешивании доменов не было случайных коллизий.
    prefixed = [
        (f"{short_name}::{u}", f"{short_name}::{i}", t) for u, i, t in interactions
    ]

    remained_raw_items = {i.split("::", 1)[1] for _, i, _ in prefixed}

    item2text = {
        f"{short_name}::{item}": text
        for item, text in item2text_raw.items()
        if item in remained_raw_items
    }

    print(
        f"[{short_name}] users={len(set(u for u, _, _ in prefixed))}, "
        f"items={len(set(i for _, i, _ in prefixed))}, "
        f"inters={len(prefixed)}"
    )

    return prefixed, item2text


def parse_online_retail_timestamp(date_text):
    import datetime

    date_text = str(date_text).strip()

    for fmt in ("%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return int(datetime.datetime.strptime(date_text, fmt).timestamp())
        except ValueError:
            pass

    return hash(date_text) & 0xFFFFFFFF


def load_online_retail(csv_path, user_k, item_k):
    """
    Online Retail:
      user/session = InvoiceNo
      item         = StockCode
      text         = Description
    """
    interactions = []
    item2text = {}

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        for row in tqdm(reader, desc="read Online Retail"):
            invoice = row.get("InvoiceNo", "")
            stock = row.get("StockCode", "")
            desc = row.get("Description", "")
            date = row.get("InvoiceDate", "")

            if not invoice or not stock or not desc:
                continue

            # Обычно отменённые инвойсы начинаются с C.
            if str(invoice).startswith("C"):
                continue

            user = f"OR::{invoice}"
            item = f"OR::{stock}"
            timestamp = parse_online_retail_timestamp(date)

            interactions.append((user, item, timestamp))

            if item not in item2text:
                item2text[item] = clean_text(desc.lower())

    interactions = k_core_filter(interactions, user_k=user_k, item_k=item_k)

    remained_items = {i for _, i, _ in interactions}
    item2text = {i: t for i, t in item2text.items() if i in remained_items}

    print(
        f"[OnlineRetail] users={len(set(u for u, _, _ in interactions))}, "
        f"items={len(set(i for _, i, _ in interactions))}, "
        f"inters={len(interactions)}"
    )

    return interactions, item2text


@torch.no_grad()
def encode_texts_with_bert(texts_by_index, model_name, batch_size, device):
    """
    texts_by_index[0] = padding text.
    На выходе:
      item_text_embs: Tensor[num_items + 1, bert_hidden]
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    hidden = model.config.hidden_size
    embs = torch.zeros(len(texts_by_index), hidden, dtype=torch.float32)

    for start in tqdm(
        range(1, len(texts_by_index), batch_size), desc="encode BERT CLS"
    ):
        end = min(start + batch_size, len(texts_by_index))
        batch_texts = texts_by_index[start:end]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        out = model(**encoded).last_hidden_state[:, 0, :].detach().cpu()
        embs[start:end] = out

    return embs


def build_indexed_pack(interactions, item2text, max_len, split_mode):
    user2seq_raw = build_sequences(interactions)

    item2idx = {}
    user2idx = {}

    for user in user2seq_raw:
        if user not in user2idx:
            user2idx[user] = len(user2idx) + 1

        for item in user2seq_raw[user]:
            if item not in item2idx:
                item2idx[item] = len(item2idx) + 1

    user2seq_idx = {}

    for user, seq in user2seq_raw.items():
        user2seq_idx[user2idx[user]] = [item2idx[item] for item in seq]

    idx2item = {idx: item for item, idx in item2idx.items()}

    texts_by_index = ["" for _ in range(len(item2idx) + 1)]
    texts_by_index[0] = ""

    for idx in range(1, len(texts_by_index)):
        raw_item = idx2item[idx]
        texts_by_index[idx] = item2text.get(raw_item, ".")

    train_samples, valid_samples, test_samples = build_samples(
        user2seq_idx=user2seq_idx,
        max_len=max_len,
        split_mode=split_mode,
    )

    return {
        "num_items": len(item2idx),
        "num_users": len(user2idx),
        "item2idx": item2idx,
        "user2idx": user2idx,
        "texts_by_index": texts_by_index,
        "train_samples": train_samples,
        "valid_samples": valid_samples,
        "test_samples": test_samples,
        "max_len": max_len,
        "split_mode": split_mode,
    }


def save_stats(pack, out_dir):
    stats = {
        "num_users": pack["num_users"],
        "num_items": pack["num_items"],
        "num_train_samples": len(pack["train_samples"]),
        "num_valid_samples": len(pack["valid_samples"]),
        "num_test_samples": len(pack["test_samples"]),
        "max_len": pack["max_len"],
        "split_mode": pack["split_mode"],
        "has_item_text_embs": "item_text_embs" in pack,
        "item_text_emb_dim": (
            int(pack["item_text_embs"].shape[1]) if "item_text_embs" in pack else None
        ),
    }

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--kind", choices=["amazon", "or"], required=True)
    parser.add_argument("--datasets", nargs="*", default=[])

    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force_download", action="store_true")

    parser.add_argument("--amazon_raw", type=str, default="data/raw/amazon")
    parser.add_argument("--or_raw_dir", type=str, default="data/raw/or")
    parser.add_argument("--or_csv", type=str, default="data/raw/or/data-utf8.csv")

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--split_mode", choices=["pretrain", "downstream"], required=True
    )
    parser.add_argument("--max_len", type=int, default=50)

    parser.add_argument("--user_k", type=int, default=5)
    parser.add_argument("--item_k", type=int, default=5)

    parser.add_argument("--bert", type=str, default="bert-base-uncased")
    parser.add_argument("--bert_batch_size", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--skip_bert", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "amazon":
        if not args.datasets:
            raise ValueError("Для --kind amazon надо указать --datasets Food Home ...")

        all_interactions = []
        all_item2text = {}

        raw_root = Path(args.amazon_raw)

        for ds in args.datasets:
            if args.download:
                ensure_amazon_dataset_downloaded(
                    short_name=ds,
                    raw_root=raw_root,
                    force=args.force_download,
                )

            interactions, item2text = load_one_amazon_dataset(
                short_name=ds,
                raw_root=raw_root,
                user_k=args.user_k,
                item_k=args.item_k,
            )

            all_interactions.extend(interactions)
            all_item2text.update(item2text)

    else:
        if args.download:
            or_csv = ensure_online_retail_downloaded(
                or_dir=args.or_raw_dir,
                force=args.force_download,
            )
        else:
            or_csv = Path(args.or_csv)

        all_interactions, all_item2text = load_online_retail(
            csv_path=or_csv,
            user_k=args.user_k,
            item_k=args.item_k,
        )

    pack = build_indexed_pack(
        interactions=all_interactions,
        item2text=all_item2text,
        max_len=args.max_len,
        split_mode=args.split_mode,
    )

    print("======== FINAL STATS ========")
    print("users:", pack["num_users"])
    print("items:", pack["num_items"])
    print("train samples:", len(pack["train_samples"]))
    print("valid samples:", len(pack["valid_samples"]))
    print("test samples:", len(pack["test_samples"]))

    if not args.skip_bert:
        item_text_embs = encode_texts_with_bert(
            texts_by_index=pack["texts_by_index"],
            model_name=args.bert,
            batch_size=args.bert_batch_size,
            device=args.device,
        )

        pack["item_text_embs"] = item_text_embs
    else:
        print("[warn] --skip_bert enabled: item_text_embs will not be saved")

    torch.save(pack, out_dir / "data.pt")
    save_stats(pack, out_dir)

    print(f"[ok] saved: {out_dir / 'data.pt'}")
    print(f"[ok] saved: {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
