import torch
import torch.nn as nn
import torch.nn.functional as F

from model_sasrec import SASRec


class ParametricWhiteningExpert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x - self.bias)


class MoEAdaptor(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([ParametricWhiteningExpert(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        experts = torch.stack([expert(x) for expert in self.experts], dim=-2)
        weights = F.softmax(self.gate(x), dim=-1).unsqueeze(-1)
        return (experts * weights).sum(dim=-2)


class UniSRec(SASRec):
    def __init__(
        self,
        num_items,
        text_emb_dim=768,
        hidden_size=64,
        max_seq_len=50,
        num_layers=2,
        num_heads=2,
        num_experts=8,
        dropout=0.2,
        use_id_embedding=False,
    ):
        super().__init__(num_items, hidden_size, max_seq_len, num_layers, num_heads, dropout)
        self.use_id_embedding = use_id_embedding
        self.text_item_encoder = MoEAdaptor(text_emb_dim, hidden_size, num_experts)
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0) if use_id_embedding else None

    def encode_text_items(self, text_embs):
        if text_embs.dim() == 2:
            return self.text_item_encoder(text_embs)
        bsz, length, dim = text_embs.shape
        return self.text_item_encoder(text_embs.reshape(bsz * length, dim)).reshape(bsz, length, self.hidden_size)

    def item_representations(self, text_embs, item_ids=None):
        out = self.encode_text_items(text_embs)
        if self.use_id_embedding:
            out = out + self.item_embedding(item_ids)
        return out

    def forward(self, item_seq_text_embs, item_seq_ids=None):
        return self.encode_sequence(self.item_representations(item_seq_text_embs, item_seq_ids), item_seq_ids)
