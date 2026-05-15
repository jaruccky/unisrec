import torch
import torch.nn.functional as F

from utils import last_hidden


@torch.no_grad()
def encode_all_items(model, item_text_embs, device, batch_size=4096):
    model.to(device)
    model.eval()
    item_text_embs = item_text_embs.to(device)
    out = []
    for start in range(1, item_text_embs.size(0), batch_size):
        end = min(start + batch_size, item_text_embs.size(0))
        ids = torch.arange(start, end, device=device)
        out.append(model.item_representations(item_text_embs[start:end], ids).detach())
    return torch.cat(out, dim=0)


def update_metrics(scores, target_ids, metrics, top_ks):
    bsz = scores.size(0)
    device = scores.device
    target_pos = target_ids - 1
    target_scores = scores[torch.arange(bsz, device=device), target_pos]
    ranks = scores.gt(target_scores.unsqueeze(1)).sum(dim=1) + 1
    for k in top_ks:
        hit = ranks.le(k).float()
        metrics[f"Recall@{k}"] += hit.sum().item()
        metrics[f"NDCG@{k}"] += (hit / torch.log2(ranks.float() + 1)).sum().item()
    return bsz


@torch.no_grad()
def evaluate(model, dataloader, item_text_embs, device, top_ks=(10, 50), item_batch_size=4096, tau=0.07):
    model.to(device)
    model.eval()
    item_embs = F.normalize(encode_all_items(model, item_text_embs, device, item_batch_size), dim=-1)
    metrics = {name: 0.0 for k in top_ks for name in (f"Recall@{k}", f"NDCG@{k}")}
    total = 0
    for batch in dataloader:
        seq_ids = batch["item_seq_ids"].to(device)
        seq_text = batch["item_seq_text_embs"].to(device)
        target_ids = batch["target_ids"].to(device)
        seq_emb = F.normalize(last_hidden(model(seq_text, seq_ids), seq_ids), dim=-1)
        scores = seq_emb @ item_embs.T / tau
        total += update_metrics(scores, target_ids, metrics, top_ks)
    return {k: v / max(total, 1) for k, v in metrics.items()}
