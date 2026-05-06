import torch
import torch.nn.functional as F


def contrastive_loss(x, y, tau=0.07):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = x @ y.T / tau
    labels = torch.arange(x.size(0), device=x.device)
    return F.cross_entropy(logits, labels)


def pretrain_loss(seq_emb, pos_item_emb, aug_seq_emb, tau=0.07, lambda_ss=1e-3):
    return contrastive_loss(seq_emb, pos_item_emb, tau) + lambda_ss * contrastive_loss(seq_emb, aug_seq_emb, tau)


def sampled_ce_loss(seq_emb, pos_emb, neg_emb):
    pos_logits = (seq_emb * pos_emb).sum(dim=-1, keepdim=True)
    neg_logits = torch.matmul(neg_emb, seq_emb.unsqueeze(-1)).squeeze(-1)
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    labels = torch.zeros(seq_emb.size(0), dtype=torch.long, device=seq_emb.device)
    return F.cross_entropy(logits, labels)
