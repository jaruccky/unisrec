import torch

from loss import pretrain_loss
from utils import last_hidden


def train_pretrain(model, dataloader, optimizer, device, epochs=10, tau=0.07, lambda_ss=1e-3, grad_clip=None):
    model.to(device)
    losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        steps = 0
        for batch in dataloader:
            seq_ids = batch["item_seq_ids"].to(device)
            aug_ids = batch["aug_seq_ids"].to(device)
            seq_text = batch["item_seq_text_embs"].to(device)
            pos_text = batch["pos_text_embs"].to(device)
            aug_text = batch["aug_seq_text_embs"].to(device)

            seq_emb = last_hidden(model(seq_text, seq_ids), seq_ids)
            aug_emb = last_hidden(model(aug_text, aug_ids), aug_ids)
            pos_emb = model.encode_text_items(pos_text)
            loss = pretrain_loss(seq_emb, pos_emb, aug_emb, tau, lambda_ss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total += loss.item()
            steps += 1
        losses.append(total / max(steps, 1))
    return losses
