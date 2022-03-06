import logging

from torch import nn
from torch.cuda import amp
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_full_data(args, model, train_loader, optimizer, scheduler, epoch):
    # parallel model
    parallel_model = nn.DataParallel(model, device_ids=args.device_ids)

    # scaler for automatic mixed precision
    scaler = amp.GradScaler(enabled=args.use_amp)

    # training loop
    model.train()
    cur_num, cur_loss, cur_correct = 0, 0, 0
    with tqdm(
        train_loader, ncols=120, desc=f"Epoch[{epoch+1}/{args.n_train_epochs}]"
    ) as pbar:
        for step, (input_ids, attention_mask, labels) in enumerate(pbar):
            # initialize all grads to 0
            optimizer.zero_grad()

            # forward with mixed precision (fp16/bf16)
            with amp.autocast(dtype=args.dtype, enabled=args.use_amp):
                losses, logits, _ = parallel_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = losses.mean()

            # backward grads of scaled loss
            scaler.scale(loss).backward()
            # `optimizer.step()` with unscaled grads or skip if Inf/NaN in grads
            scaler.step(optimizer)
            # update scaling factor
            scaler.update()

            scheduler.step()

            cur_num += len(input_ids)
            cur_loss += loss.item() * len(input_ids)
            cur_correct += logits.cpu().argmax(1).eq(labels).sum().item()

            # update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                }
            )

            # logging
            if (step + 1) % args.logging_steps == 0:
                logger.info(
                    "Epoch {:.2f}, Step {} | Running Loss = {:.3f}, "
                    "Running Acc={:.2%}, lr={:.2e}".format(
                        epoch + (step + 1) / len(pbar),
                        step + 1 + len(pbar) * epoch,
                        cur_loss / cur_num,
                        cur_correct / cur_num,
                        scheduler.get_last_lr()[0],
                    )
                )
                cur_num, cur_loss, cur_correct = 0, 0, 0
