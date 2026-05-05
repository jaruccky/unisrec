import torch
import torch.nn.functional as F


def seq_item_loss(seq_emb, pos_item_emb, tau=0.07):
    # L_SI = -sum_j log exp(s_j·v_j/tau) / sum_k exp(s_j·v_k/tau)
    seq_emb = F.normalize(seq_emb, dim=-1)  # s_j
    pos_item_emb = F.normalize(pos_item_emb, dim=-1)  # v_j

    logits = seq_emb @ pos_item_emb.T  # [B,B], s_j·v_k
    logits = logits / tau  # делим на температуру

    labels = torch.arange(
        seq_emb.size(0), device=seq_emb.device
    )  # правильный item на диагонали

    return F.cross_entropy(logits, labels)


def seq_seq_loss(seq_emb, aug_seq_emb, tau=0.07):
    # L_SS = -sum_j log exp(s_j·s~_j/tau) / sum_k exp(s_j·s~_k/tau)
    seq_emb = F.normalize(seq_emb, dim=-1)  # s_j
    aug_seq_emb = F.normalize(aug_seq_emb, dim=-1)  # s~_j

    logits = seq_emb @ aug_seq_emb.T  # [B,B], s_j·s~_k
    logits = logits / tau  # делим на температуру

    labels = torch.arange(
        seq_emb.size(0), device=seq_emb.device
    )  # своя аугментация на диагонали

    return F.cross_entropy(logits, labels)


def unisrec_pretrain_loss(seq_emb, pos_item_emb, aug_seq_emb, tau=0.07, lambda_ss=1e-3):
    # L_PT = L_SI + lambda * L_SS
    loss_si = seq_item_loss(seq_emb, pos_item_emb, tau)
    loss_ss = seq_seq_loss(seq_emb, aug_seq_emb, tau)

    return loss_si + lambda_ss * loss_ss
