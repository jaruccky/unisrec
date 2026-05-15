import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, num_items, hidden_size=64, max_seq_len=50, num_layers=2, num_heads=2, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

    def encode_sequence(self, item_embs, item_seq_ids=None):
        bsz, seq_len, _ = item_embs.shape
        device = item_embs.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        x = item_embs + self.position_embedding(pos)
        x = self.input_norm(x)
        x = self.dropout(x)
        if item_seq_ids is None:
            key_padding_mask = None
            keep_mask = None
        else:
            key_padding_mask = item_seq_ids.eq(0)
            keep_mask = item_seq_ids.ne(0).unsqueeze(-1)
            x = x * keep_mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.final_norm(x)
        return x if keep_mask is None else x * keep_mask

    def forward(self, item_seq_ids):
        return self.encode_sequence(self.item_embedding(item_seq_ids), item_seq_ids)

    def full_sort_scores(self, item_seq_ids):
        seq_output = self.forward(item_seq_ids)
        pos = item_seq_ids.ne(0).long().sum(dim=1).sub(1).clamp_min(0)
        batch = torch.arange(item_seq_ids.size(0), device=item_seq_ids.device)
        scores = seq_output[batch, pos] @ self.item_embedding.weight.T
        return scores[:, 1:]
