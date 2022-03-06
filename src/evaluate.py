import torch
from torch.cuda import amp


def evaluate(args, model, test_loader):
    # model to eval mode
    model.eval()

    # compute accuracy and loss
    total_num, total_loss, total_correct = 0, 0, 0
    with torch.no_grad():
        for (input_ids, attention_mask, labels) in test_loader:
            # forward with mixed precision
            with amp.autocast(dtype=args.dtype, enabled=args.use_amp):
                losses, logits, _ = model(
                    input_ids=input_ids.to(args.device),
                    attention_mask=attention_mask.to(args.device),
                    labels=labels.to(args.device),
                )
                loss = losses.mean()

            total_num += len(input_ids)
            total_loss += loss.item() * len(input_ids)
            total_correct += logits.cpu().argmax(1).eq(labels).sum().item()

    # model to train mode
    model.train()

    return total_correct / total_num, total_loss / total_num
