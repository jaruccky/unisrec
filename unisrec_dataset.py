import random

import torch
from torch.utils.data import Dataset


class UniSRecPretrainDataset(Dataset):
    """
    Dataset под твой train_unisrec_pretrain.

    На вход:
      samples: list[(item_seq_ids, pos_id)]
        item_seq_ids уже right-padded: [L]
        pos_id: int

      item_text_embs: Tensor[num_items + 1, D]
        item_text_embs[0] = zeros для padding.
    """

    def __init__(
        self,
        samples,
        item_text_embs,
        max_len=50,
        item_drop_ratio=0.2,
    ):
        self.samples = samples
        self.item_text_embs = item_text_embs.float()
        self.max_len = max_len
        self.item_drop_ratio = item_drop_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_ids, pos_id = self.samples[idx]

        return {
            "item_seq_ids": torch.tensor(seq_ids, dtype=torch.long),
            "pos_ids": torch.tensor(pos_id, dtype=torch.long),
        }

    def _augment_item_drop(self, seq_ids):
        valid = [int(x) for x in seq_ids if int(x) != 0]

        if len(valid) <= 1:
            aug = valid
        else:
            aug = [x for x in valid if random.random() > self.item_drop_ratio]

            if len(aug) == 0:
                aug = [valid[-1]]

        aug = aug[-self.max_len :]
        aug = aug + [0] * (self.max_len - len(aug))

        return aug

    def collate_fn(self, batch):
        item_seq_ids = torch.stack([x["item_seq_ids"] for x in batch])
        pos_ids = torch.stack([x["pos_ids"] for x in batch])

        aug_seq_ids = torch.tensor(
            [self._augment_item_drop(x["item_seq_ids"].tolist()) for x in batch],
            dtype=torch.long,
        )

        item_seq_text_embs = self.item_text_embs[item_seq_ids]
        pos_text_embs = self.item_text_embs[pos_ids]
        aug_seq_text_embs = self.item_text_embs[aug_seq_ids]

        return {
            "item_seq_text_embs": item_seq_text_embs,  # [B, L, D]
            "item_seq_ids": item_seq_ids,  # [B, L]
            "pos_text_embs": pos_text_embs,  # [B, D]
            "pos_ids": pos_ids,  # [B]
            "aug_seq_text_embs": aug_seq_text_embs,  # [B, L, D]
            "aug_seq_ids": aug_seq_ids,  # [B, L]
        }


class UniSRecEvalDataset(Dataset):
    """
    Для valid/test.

    Возвращает:
      item_seq_text_embs: [B, L, D]
      item_seq_ids:       [B, L]
      target_ids:         [B]
      target_text_embs:   [B, D]
    """

    def __init__(self, samples, item_text_embs):
        self.samples = samples
        self.item_text_embs = item_text_embs.float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_ids, target_id = self.samples[idx]

        return {
            "item_seq_ids": torch.tensor(seq_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_id, dtype=torch.long),
        }

    def collate_fn(self, batch):
        item_seq_ids = torch.stack([x["item_seq_ids"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])

        item_seq_text_embs = self.item_text_embs[item_seq_ids]
        target_text_embs = self.item_text_embs[target_ids]

        return {
            "item_seq_ids": item_seq_ids,
            "item_seq_text_embs": item_seq_text_embs,
            "target_ids": target_ids,
            "target_text_embs": target_text_embs,
        }
