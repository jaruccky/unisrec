import torch

from loss import full_sort_ce_loss, sampled_ce_loss
from utils import last_hidden


def configure_finetune(model, train_all=False):
    if train_all:
        for p in model.parameters():
            p.requires_grad = True
        return model
    for p in model.parameters():
        p.requires_grad = False
    for p in model.text_item_encoder.parameters():
        p.requires_grad = True
    if model.item_embedding is not None:
        for p in model.item_embedding.parameters():
            p.requires_grad = True
    return model


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def encode_train_items(model, item_text_embs, device):
    item_text_embs = item_text_embs.to(device)
    item_ids = torch.arange(1, item_text_embs.size(0), device=device)
    return model.item_representations(item_text_embs[1:], item_ids)


def finetune_step(model, batch, device, item_text_embs=None, tau=0.07, full_sort=True):
    seq_ids = batch["item_seq_ids"].to(device)
    target_ids = batch["target_ids"].to(device)
    seq_text = batch["item_seq_text_embs"].to(device)
    seq_emb = last_hidden(model(seq_text, seq_ids), seq_ids)

    if full_sort:
        item_embs = encode_train_items(model, item_text_embs, device)
        return full_sort_ce_loss(seq_emb, item_embs, target_ids, tau)

    neg_ids = batch["neg_ids"].to(device)
    target_text = batch["target_text_embs"].to(device)
    neg_text = batch["neg_text_embs"].to(device)
    pos_emb = model.item_representations(target_text, target_ids)
    neg_emb = model.item_representations(neg_text, neg_ids)
    return sampled_ce_loss(seq_emb, pos_emb, neg_emb, tau)


def train_finetune(model, dataloader, optimizer, device, epochs=10, tau=0.07, item_text_embs=None, full_sort=True, grad_clip=None):
    if full_sort and item_text_embs is None:
        raise ValueError("item_text_embs is required for full-sort fine-tuning")
    model.to(device)
    losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        steps = 0
        for batch in dataloader:
            loss = finetune_step(model, batch, device, item_text_embs, tau, full_sort)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), grad_clip)
            optimizer.step()
            total += loss.item()
            steps += 1
        losses.append(total / max(steps, 1))
    return losses
