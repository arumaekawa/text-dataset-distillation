import logging
import time

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from evaluate import evaluate

logger = logging.getLogger(__name__)


def train_random_data(args, model, train_loader, init_model=False):
    # initialize model parameters (known initial parameters)
    if init_model:
        model.load_state_dict(torch.load(args.initial_model_path))

    # optimizer
    model_opt = SGD(model.parameters(), lr=args.distill_model_lr)
    scheduler = StepLR(model_opt, step_size=1, gamma=args.distill_step_lr_gamma)

    assert len(train_loader) == 1
    input_ids, attention_mask, labels = next(iter(train_loader))

    # gradient steps
    start_time = time.time()
    model.train()
    for inner_step in range(args.n_inner_steps):
        # forward
        losses = model(
            input_ids=input_ids.to(args.device),
            attention_mask=attention_mask.to(args.device),
            labels=labels.to(args.device),
        )[0]
        loss = losses.mean()
        # backward
        model_opt.zero_grad()
        loss.backward()
        model_opt.step()
        scheduler.step()

    logger.info(f"Time for model traning : {time.time() - start_time:.2f}s")


def test_random_data(
    args,
    model,
    random_train_loader,
    full_train_loader=None,
    test_loader=None,
    init_model=False,
):
    assert full_train_loader is not None or test_loader is not None

    # initialize model
    if init_model:
        model.load_state_dict(torch.load(args.initial_model_path))

    # test initial model on test dataset
    if full_train_loader is not None:
        before_train_acc, before_train_loss = evaluate(args, model, full_train_loader)
    # test initial model on test dataset
    if test_loader is not None:
        before_test_acc, before_test_loss = evaluate(args, model, test_loader)

    # train model with distilled data
    train_random_data(args, model, random_train_loader)

    # test trained model on train dataset
    if full_train_loader is not None:
        after_train_acc, after_train_loss = evaluate(args, model, full_train_loader)
        logger.info(
            "Evaluate on Train dataset | (before) loss: {:>6.4f}, acc: {:5.2%}"
            " -> (after) loss: {:>6.4f}, acc: {:5.2%}".format(
                before_train_loss,
                before_train_acc,
                after_train_loss,
                after_train_acc,
            )
        )

    # test trained model on test dataset
    if test_loader is not None:
        after_test_acc, after_test_loss = evaluate(args, model, test_loader)
        logger.info(
            "Evaluate on Test dataset  | (before) loss: {:>6.4f}, acc: {:5.2%}"
            " -> (after) loss: {:>6.4f}, acc: {:5.2%}".format(
                before_test_loss,
                before_test_acc,
                after_test_loss,
                after_test_acc,
            )
        )

    return after_test_acc
