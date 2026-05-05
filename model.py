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
        # x: [*, D]
        # схема: text_emb -> (x - b) -> Linear -> item_emb
        return self.linear(x - self.bias)


class MoEAdaptor(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super().__init__()

        self.experts = nn.ModuleList(
            [
                ParametricWhiteningExpert(input_dim, output_dim)
                for _ in range(num_experts)
            ]
        )

        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # x: [*, D]

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # [*, G, H]

        gate_weights = F.softmax(self.gate(x), dim=-1)
        # [*, G]

        gate_weights = gate_weights.unsqueeze(-1)
        # [*, G, 1]

        out = (expert_outputs * gate_weights).sum(dim=1)
        # [*, H]

        return out


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
        super().__init__(
            num_items=num_items,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.use_id_embedding = use_id_embedding

        self.text_item_encoder = MoEAdaptor(
            input_dim=text_emb_dim,
            output_dim=hidden_size,
            num_experts=num_experts,
        )

        if not use_id_embedding:
            self.item_embedding = None
        else:
            self.item_embedding = nn.Embedding(
                num_items + 1,
                hidden_size,
                padding_idx=0,
            )

    def encode_text_items(self, text_embs):
        # text_embs: [B,L,D] или [B,N,D] или [B,D]

        if text_embs.dim() == 2:
            return self.text_item_encoder(text_embs)
            # [B,H]

        B, T, D = text_embs.shape

        flat = text_embs.reshape(B * T, D)
        item_emb = self.text_item_encoder(flat)
        item_emb = item_emb.reshape(B, T, self.hidden_size)

        return item_emb
        # [B,T,H]

    def forward(self, item_seq_text_embs, item_seq_ids=None):
        # item_seq_text_embs: [B,L,D]
        # item_seq_ids:       [B,L], нужен только для transductive режима

        B, L, _ = item_seq_text_embs.shape

        positions = torch.arange(L, device=item_seq_text_embs.device)
        positions = positions.unsqueeze(0).expand(B, L)
        # [B,L]

        text_item_emb = self.encode_text_items(item_seq_text_embs)
        # [B,L,H]

        if self.use_id_embedding:
            id_item_emb = self.item_embedding(item_seq_ids)
            # [B,L,H]

            x = text_item_emb + id_item_emb
        else:
            x = text_item_emb

        pos_emb = self.position_embedding(positions)
        # [B,L,H]

        x = x + pos_emb
        x = self.embedding_dropout(x)

        if item_seq_ids is not None:
            padding_mask = (item_seq_ids != 0).unsqueeze(-1)
            x = x * padding_mask
        else:
            padding_mask = None

        for block in self.blocks:
            x = block(x, item_seq_ids)

        x = self.final_norm(x)

        if padding_mask is not None:
            x = x * padding_mask

        return x
        # [B,L,H]

    def calculate_loss(
        self,
        item_seq_text_embs,
        pos_text_embs,
        neg_text_embs,
        item_seq_ids=None,
        pos_ids=None,
        neg_ids=None,
    ):
        # item_seq_text_embs: [B,L,D]
        # pos_text_embs:      [B,L,D]
        # neg_text_embs:      [B,L,D]

        seq_output = self.forward(item_seq_text_embs, item_seq_ids)
        # [B,L,H]

        pos_emb = self.encode_text_items(pos_text_embs)
        neg_emb = self.encode_text_items(neg_text_embs)

        if self.use_id_embedding:
            pos_emb = pos_emb + self.item_embedding(pos_ids)
            neg_emb = neg_emb + self.item_embedding(neg_ids)

        pos_logits = (seq_output * pos_emb).sum(dim=-1)
        neg_logits = (seq_output * neg_emb).sum(dim=-1)

        if pos_ids is not None:
            valid_mask = pos_ids != 0
        else:
            valid_mask = torch.ones_like(pos_logits, dtype=torch.bool)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits[valid_mask],
            torch.ones_like(pos_logits[valid_mask]),
        )

        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits[valid_mask],
            torch.zeros_like(neg_logits[valid_mask]),
        )

        return pos_loss + neg_loss

    def predict(
        self,
        item_seq_text_embs,
        candidate_text_embs,
        item_seq_ids=None,
        candidate_ids=None,
    ):
        # item_seq_text_embs:  [B,L,D]
        # candidate_text_embs: [B,N,D]

        seq_output = self.forward(item_seq_text_embs, item_seq_ids)
        # [B,L,H]

        if item_seq_ids is not None:
            lengths = (item_seq_ids != 0).long().sum(dim=1) - 1
            batch_idx = torch.arange(item_seq_ids.size(0), device=item_seq_ids.device)
            last_hidden = seq_output[batch_idx, lengths]
        else:
            last_hidden = seq_output[:, -1, :]

        candidate_emb = self.encode_text_items(candidate_text_embs)
        # [B,N,H]

        if self.use_id_embedding:
            candidate_emb = candidate_emb + self.item_embedding(candidate_ids)

        scores = torch.matmul(
            candidate_emb,
            last_hidden.unsqueeze(-1),
        ).squeeze(-1)

        return scores
        # [B,N]
