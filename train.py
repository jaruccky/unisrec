import torch

from loss import unisrec_pretrain_loss


def get_last_hidden(seq_output, item_seq_ids=None):
    """
    seq_output:   [B, L, H]
    item_seq_ids: [B, L] или None

    Важно: ожидается right padding:
      [10, 25, 31, 0, 0]
    """
    if item_seq_ids is not None:
        lengths = (item_seq_ids != 0).long().sum(dim=1) - 1
        lengths = lengths.clamp_min(0)

        batch_idx = torch.arange(
            item_seq_ids.size(0),
            device=item_seq_ids.device,
        )

        return seq_output[batch_idx, lengths]

    return seq_output[:, -1, :]


def train_unisrec_pretrain(
    model,
    dataloader,
    optimizer,
    device,
    epochs=10,
    tau=0.07,
    lambda_ss=1e-3,
    grad_clip=None,
):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in dataloader:
            item_seq_text_embs = batch["item_seq_text_embs"].to(device)
            pos_text_embs = batch["pos_text_embs"].to(device)
            aug_seq_text_embs = batch["aug_seq_text_embs"].to(device)

            item_seq_ids = batch.get("item_seq_ids")
            aug_seq_ids = batch.get("aug_seq_ids")

            if item_seq_ids is not None:
                item_seq_ids = item_seq_ids.to(device)

            if aug_seq_ids is not None:
                aug_seq_ids = aug_seq_ids.to(device)

            seq_output = model(
                item_seq_text_embs=item_seq_text_embs,
                item_seq_ids=item_seq_ids,
            )

            seq_emb = get_last_hidden(seq_output, item_seq_ids)

            pos_item_emb = model.encode_text_items(pos_text_embs)

            aug_seq_output = model(
                item_seq_text_embs=aug_seq_text_embs,
                item_seq_ids=aug_seq_ids,
            )

            aug_seq_emb = get_last_hidden(aug_seq_output, aug_seq_ids)

            loss = unisrec_pretrain_loss(
                seq_emb=seq_emb,
                pos_item_emb=pos_item_emb,
                aug_seq_emb=aug_seq_emb,
                tau=tau,
                lambda_ss=lambda_ss,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"epoch {epoch + 1}/{epochs}, pretrain loss = {avg_loss:.4f}")
