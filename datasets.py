import random

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, samples, item_text_embs, max_len, item_drop_ratio=0.2):
        self.samples = samples
        self.item_text_embs = item_text_embs.float()
        self.max_len = max_len
        self.item_drop_ratio = item_drop_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, pos = self.samples[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(pos, dtype=torch.long)

    def augment(self, seq):
        items = [int(x) for x in seq if int(x) != 0]
        if len(items) > 1:
            items = [x for x in items if random.random() > self.item_drop_ratio] or [items[-1]]
        items = items[-self.max_len:]
        return items + [0] * (self.max_len - len(items))

    def collate_fn(self, batch):
        seq_ids = torch.stack([x[0] for x in batch])
        pos_ids = torch.stack([x[1] for x in batch])
        aug_ids = torch.tensor([self.augment(x[0].tolist()) for x in batch], dtype=torch.long)
        return {
            "item_seq_ids": seq_ids,
            "pos_ids": pos_ids,
            "aug_seq_ids": aug_ids,
            "item_seq_text_embs": self.item_text_embs[seq_ids],
            "pos_text_embs": self.item_text_embs[pos_ids],
            "aug_seq_text_embs": self.item_text_embs[aug_ids],
        }


class FinetuneDataset(Dataset):
    def __init__(self, samples, item_text_embs, num_items, num_negatives=100):
        self.samples = samples
        self.item_text_embs = item_text_embs.float()
        self.num_items = num_items
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

    def sample_negatives(self, seq, target):
        banned = set(int(x) for x in seq if int(x) != 0)
        banned.add(int(target))
        out = []
        while len(out) < self.num_negatives:
            x = random.randint(1, self.num_items)
            if x not in banned:
                out.append(x)
        return out

    def collate_fn(self, batch):
        seq_ids = torch.stack([x[0] for x in batch])
        target_ids = torch.stack([x[1] for x in batch])
        neg_ids = torch.tensor(
            [self.sample_negatives(seq.tolist(), target.item()) for seq, target in batch],
            dtype=torch.long,
        )
        return {
            "item_seq_ids": seq_ids,
            "target_ids": target_ids,
            "neg_ids": neg_ids,
            "item_seq_text_embs": self.item_text_embs[seq_ids],
            "target_text_embs": self.item_text_embs[target_ids],
            "neg_text_embs": self.item_text_embs[neg_ids],
        }


class EvalDataset(Dataset):
    def __init__(self, samples, item_text_embs):
        self.samples = samples
        self.item_text_embs = item_text_embs.float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

    def collate_fn(self, batch):
        seq_ids = torch.stack([x[0] for x in batch])
        target_ids = torch.stack([x[1] for x in batch])
        return {
            "item_seq_ids": seq_ids,
            "target_ids": target_ids,
            "item_seq_text_embs": self.item_text_embs[seq_ids],
        }
