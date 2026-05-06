import csv
import gzip
import html
import json
import re
import shutil
import subprocess
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import torch

AMAZON_NAME = {
    "Food": "Grocery_and_Gourmet_Food",
    "Home": "Home_and_Kitchen",
    "CDs": "CDs_and_Vinyl",
    "Kindle": "Kindle_Store",
    "Movies": "Movies_and_TV",
    "Pantry": "Prime_Pantry",
    "Scientific": "Industrial_and_Scientific",
    "Instruments": "Musical_Instruments",
    "Arts": "Arts_Crafts_and_Sewing",
    "Office": "Office_Products",
}
AMAZON_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"


def is_downloaded(path, min_bytes=1024):
    path = Path(path)
    return path.is_file() and path.stat().st_size >= min_bytes


def download_file(url, dst, force=False):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if is_downloaded(dst) and not force:
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(tmp, "wb") as fout:
        shutil.copyfileobj(response, fout)
    tmp.rename(dst)
    return dst


def download_amazon(short_name, raw_root, force=False):
    full = AMAZON_NAME[short_name]
    raw_root = Path(raw_root)
    rating = raw_root / "Ratings" / f"{full}.csv"
    meta = raw_root / "Metadata" / f"meta_{full}.json.gz"
    download_file(f"{AMAZON_BASE_URL}/categoryFilesSmall/{full}.csv", rating, force)
    download_file(f"{AMAZON_BASE_URL}/metaFiles2/meta_{full}.json.gz", meta, force)
    return rating, meta


def download_online_retail(out_dir, force=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_latin1 = out_dir / "data.csv"
    csv_utf8 = out_dir / "data-utf8.csv"
    if is_downloaded(csv_utf8) and not force:
        return csv_utf8
    if not is_downloaded(csv_latin1) or force:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "carrie1/ecommerce-data", "-p", str(out_dir), "--unzip"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    with open(csv_latin1, "r", encoding="latin1", errors="ignore") as fin, open(csv_utf8, "w", encoding="utf-8") as fout:
        fout.writelines(fin)
    return csv_utf8


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


def clean_text(x, max_chars=4000):
    x = html.unescape(flatten_text(x))
    x = re.sub(r"[\n\r\t]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x[:max_chars] if x else "."


def k_core(interactions, user_k=5, item_k=5):
    interactions = list(interactions)
    while True:
        uc = Counter(u for u, _, _ in interactions)
        ic = Counter(i for _, i, _ in interactions)
        new = [(u, i, t) for u, i, t in interactions if uc[u] >= user_k and ic[i] >= item_k]
        if len(new) == len(interactions):
            return new
        interactions = new


def read_amazon_ratings(path):
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if len(row) == 4:
                item, user, _, ts = row
                try:
                    out.append((user, item, int(ts)))
                except ValueError:
                    pass
    return out


def read_amazon_meta(path):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = obj.get("asin")
            if asin:
                parts = [obj.get("title", ""), obj.get("category", obj.get("categories", "")), obj.get("brand", "")]
                out[asin] = clean_text(" ".join(clean_text(p) for p in parts))
    return out


def load_amazon_dataset(short_name, raw_root, user_k=5, item_k=5):
    full = AMAZON_NAME[short_name]
    raw_root = Path(raw_root)
    ratings = raw_root / "Ratings" / f"{full}.csv"
    meta = raw_root / "Metadata" / f"meta_{full}.json.gz"
    item2text_raw = read_amazon_meta(meta)
    interactions = [(u, i, t) for u, i, t in read_amazon_ratings(ratings) if i in item2text_raw]
    interactions = k_core(interactions, user_k, item_k)
    interactions = [(f"{short_name}::{u}", f"{short_name}::{i}", t) for u, i, t in interactions]
    kept = {i.split("::", 1)[1] for _, i, _ in interactions}
    item2text = {f"{short_name}::{i}": text for i, text in item2text_raw.items() if i in kept}
    return interactions, item2text


def parse_online_retail_ts(text):
    import datetime
    for fmt in ("%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return int(datetime.datetime.strptime(str(text).strip(), fmt).timestamp())
        except ValueError:
            pass
    return hash(text) & 0xFFFFFFFF


def load_online_retail(csv_path, user_k=5, item_k=5):
    interactions = []
    item2text = {}
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            invoice = row.get("InvoiceNo", "")
            stock = row.get("StockCode", "")
            desc = row.get("Description", "")
            if not invoice or not stock or not desc or str(invoice).startswith("C"):
                continue
            user = f"OR::{invoice}"
            item = f"OR::{stock}"
            interactions.append((user, item, parse_online_retail_ts(row.get("InvoiceDate", ""))))
            item2text.setdefault(item, clean_text(desc.lower()))
    interactions = k_core(interactions, user_k, item_k)
    kept = {i for _, i, _ in interactions}
    return interactions, {i: t for i, t in item2text.items() if i in kept}


def build_sequences(interactions):
    data = defaultdict(list)
    for user, item, ts in interactions:
        data[user].append((ts, item))
    return {user: [item for _, item in sorted(events)] for user, events in data.items()}


def pad_right(seq, max_len):
    seq = seq[-max_len:]
    return seq + [0] * (max_len - len(seq))


def build_samples(user2seq, max_len, split_mode):
    train, valid, test = [], [], []
    for seq in user2seq.values():
        if len(seq) < 3:
            continue
        if split_mode == "pretrain":
            train.extend((pad_right(seq[:t], max_len), seq[t]) for t in range(1, len(seq)))
        else:
            train_seq = seq[:-2]
            train.extend((pad_right(train_seq[:t], max_len), train_seq[t]) for t in range(1, len(train_seq)))
            valid.append((pad_right(train_seq, max_len), seq[-2]))
            test.append((pad_right(train_seq + [seq[-2]], max_len), seq[-1]))
    return train, valid, test


def build_pack(interactions, item2text, max_len=50, split_mode="downstream"):
    raw = build_sequences(interactions)
    user2idx = {u: idx for idx, u in enumerate(raw, start=1)}
    item2idx = {}
    for seq in raw.values():
        for item in seq:
            item2idx.setdefault(item, len(item2idx) + 1)
    user2seq = {user2idx[u]: [item2idx[i] for i in seq] for u, seq in raw.items()}
    idx2item = {idx: item for item, idx in item2idx.items()}
    texts = [""] + [item2text.get(idx2item[i], ".") for i in range(1, len(item2idx) + 1)]
    train, valid, test = build_samples(user2seq, max_len, split_mode)
    return {
        "num_items": len(item2idx),
        "num_users": len(user2idx),
        "item2idx": item2idx,
        "user2idx": user2idx,
        "texts_by_index": texts,
        "train_samples": train,
        "valid_samples": valid,
        "test_samples": test,
        "max_len": max_len,
        "split_mode": split_mode,
    }


@torch.no_grad()
def encode_texts(texts, model_name="bert-base-uncased", batch_size=32, device="cpu"):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embs = torch.zeros(len(texts), model.config.hidden_size)
    for start in range(1, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = tokenizer(texts[start:end], padding=True, truncation=True, max_length=512, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        embs[start:end] = model(**batch).last_hidden_state[:, 0].cpu()
    return embs


def save_pack(pack, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pack, out_dir / "data.pt")
    return out_dir / "data.pt"


def prepare_amazon(datasets, out_dir, raw_root="data/raw/amazon", download=False, force_download=False, split_mode="downstream", max_len=50, user_k=5, item_k=5, bert="bert-base-uncased", bert_batch_size=32, device="cpu", skip_bert=False):
    interactions = []
    item2text = {}
    for ds in datasets:
        if download:
            download_amazon(ds, raw_root, force_download)
        part_interactions, part_text = load_amazon_dataset(ds, raw_root, user_k, item_k)
        interactions.extend(part_interactions)
        item2text.update(part_text)
    pack = build_pack(interactions, item2text, max_len, split_mode)
    if not skip_bert:
        pack["item_text_embs"] = encode_texts(pack["texts_by_index"], bert, bert_batch_size, device)
    return save_pack(pack, out_dir)


def prepare_or(out_dir, raw_dir="data/raw/or", csv_path="data/raw/or/data-utf8.csv", download=False, force_download=False, split_mode="downstream", max_len=50, user_k=5, item_k=5, bert="bert-base-uncased", bert_batch_size=32, device="cpu", skip_bert=False):
    csv_path = download_online_retail(raw_dir, force_download) if download else Path(csv_path)
    interactions, item2text = load_online_retail(csv_path, user_k, item_k)
    pack = build_pack(interactions, item2text, max_len, split_mode)
    if not skip_bert:
        pack["item_text_embs"] = encode_texts(pack["texts_by_index"], bert, bert_batch_size, device)
    return save_pack(pack, out_dir)
