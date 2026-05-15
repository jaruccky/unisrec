import torch
import torch.nn.functional as F


def same_positive_mask(pos_ids):
    same = pos_ids.unsqueeze(1).eq(pos_ids.unsqueeze(0))
    eye = torch.eye(pos_ids.size(0), dtype=torch.bool, device=pos_ids.device)
    return same & ~eye


def contrastive_loss(x, y, tau=0.07, false_negative_mask=None):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = x @ y.T / tau
    if false_negative_mask is not None:
        logits = logits.masked_fill(false_negative_mask, torch.finfo(logits.dtype).min)
    labels = torch.arange(x.size(0), device=x.device)
    return F.cross_entropy(logits, labels)


def pretrain_loss(seq_emb, pos_item_emb, aug_seq_emb, pos_ids=None, tau=0.07, lambda_ss=1e-3):
    false_negative_mask = same_positive_mask(pos_ids) if pos_ids is not None else None
    loss_seq_item = contrastive_loss(seq_emb, pos_item_emb, tau, false_negative_mask)
    loss_seq_seq = contrastive_loss(seq_emb, aug_seq_emb, tau, false_negative_mask)
    return loss_seq_item + lambda_ss * loss_seq_seq


def sampled_ce_loss(seq_emb, pos_emb, neg_emb, tau=0.07):
    seq_emb = F.normalize(seq_emb, dim=-1)
    pos_emb = F.normalize(pos_emb, dim=-1)
    neg_emb = F.normalize(neg_emb, dim=-1)
    pos_logits = (seq_emb * pos_emb).sum(dim=-1, keepdim=True) / tau
    neg_logits = torch.matmul(neg_emb, seq_emb.unsqueeze(-1)).squeeze(-1) / tau
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    labels = torch.zeros(seq_emb.size(0), dtype=torch.long, device=seq_emb.device)
    return F.cross_entropy(logits, labels)


def full_sort_ce_loss(seq_emb, item_embs, target_ids, tau=0.07):
    seq_emb = F.normalize(seq_emb, dim=-1)
    item_embs = F.normalize(item_embs, dim=-1)
    logits = seq_emb @ item_embs.T / tau
    labels = target_ids - 1
    return F.cross_entropy(logits, labels)
