import argparse
import logging
import os

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, get_linear_schedule_with_warmup

from distill import DistilledData, test_distilled_data
from evaluate import evaluate
from full_data import train_full_data
from model import BertClassifier
from random_data import test_random_data
from settings import init_logging, settings
from utils import create_dataloader, get_full_dataset, get_random_dataset, make_dir

logger = logging.getLogger(__name__)


def run_full_data(
    args: argparse.Namespace,
    model: BertClassifier,
    tokenizer: PreTrainedTokenizer,
    test_loader: DataLoader,
    dataset_attrs: dict,
) -> None:

    # load train dataset
    train_data_path = os.path.join(args.data_dir, "train_full_data")
    train_data = get_full_dataset(
        dataset_attrs,
        tokenizer,
        train_data_path,
        data_type="train",
        preprocess=args.preprocess,
    )
    logger.info(f"Train data size: {len(train_data)}")

    # train dataloader
    train_loader = create_dataloader(
        train_data,
        batch_size=args.train_batch_size,
        data_type="train",
        fix_batch_seq_len=args.fix_batch_seq_len,
        num_workers=args.num_workers,
    )

    # logging steps
    if args.logging_steps == 0:
        args.logging_steps = max(len(train_loader) // 10, 1)

    # number of training steps
    num_train_steps = len(train_loader) * args.n_train_epochs
    logger.info(f"Total optimization steps: {num_train_steps}")

    # optimizer
    optimizer = Adam(model.parameters(), lr=args.model_lr)
    # learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_train_steps * args.warmup_ratio, num_train_steps
    )

    # training loop
    logger.info("Start training!!")
    best_test_acc, best_epoch = 0, 0
    for epoch in range(args.n_train_epochs):
        # train one epoch
        train_full_data(args, model, train_loader, optimizer, scheduler, epoch)

        # evaluate on train dataset
        train_acc, train_loss = evaluate(args, model, train_loader)
        logger.info(
            "Epoch[{}] Train Accuracy: {:.2%}, Train Loss: {:.3f}".format(
                epoch + 1, train_acc, train_loss
            )
        )
        # evaluate on test dataset
        test_acc, test_loss = evaluate(args, model, test_loader)
        logger.info(
            "Epoch[{}] Test Accuracy: {:.2%}, Test Loss: {:.3f}".format(
                epoch + 1, test_acc, test_loss
            )
        )

        # update best score
        if test_acc > best_test_acc:
            state_dict = model.state_dict()
            best_test_acc = test_acc
            best_epoch = epoch + 1

        # save model parameters
        torch.save(state_dict, os.path.join(args.model_dir, f"model-{epoch+1}"))

    logger.info("Finish training!!")
    logger.info(f"Best Test Accuracy: {best_test_acc:.2%} (Epoch {best_epoch})")


def run_random_data(
    args: argparse.Namespace,
    model: BertClassifier,
    tokenizer: PreTrainedTokenizer,
    test_loader: DataLoader,
    dataset_attrs: dict,
) -> None:

    # load random dataset list
    full_dataset_path = os.path.join(args.data_dir, "train_full_data")
    random_dataset_path = os.path.join(
        args.data_dir, f"train_random_data_{args.data_size}"
    )
    random_dataset_list = get_random_dataset(
        dataset_attrs,
        tokenizer,
        full_dataset_path=full_dataset_path,
        random_dataset_path=random_dataset_path,
        data_size=args.data_size,
        num_dataset=args.num_dataset,
        preprocess=args.preprocess,
    )
    logger.info(
        "Number of datasets: {}, Each dataset size: {}".format(
            len(random_dataset_list), len(random_dataset_list[0])
        )
    )

    # list of random dataloader
    train_loader_list = [
        create_dataloader(
            random_dataset,
            batch_size=max(len(random_dataset), args.train_batch_size),
            data_type="train",
            fix_batch_seq_len=args.fix_batch_seq_len,
            num_workers=args.num_workers,
        )
        for random_dataset in random_dataset_list
    ]

    # logging steps
    if args.logging_steps == 0:
        args.logging_steps = max(len(train_loader_list[0]) // 10, 1)

    # initial model parameters
    initial_state_dict = torch.load(args.initial_model_path)

    # test all random datasets on fixed initialized model
    logger.info("Start training!!")
    all_acc_with_fixed_init = []
    for dataset_id, train_loader in enumerate(train_loader_list):
        dataset_name = f"random_data_{dataset_id}"
        logger.info(f"Test `{dataset_name}` on fixed initialized model" + "-" * 50)

        # initialize model parameters
        model.load_state_dict(initial_state_dict)
        # test dataset
        acc = test_random_data(args, model, train_loader, test_loader=test_loader)

        all_acc_with_fixed_init.append(acc)

    # test top k datasets on randomly initialized model
    top_k_dataset_ids = np.argsort(all_acc_with_fixed_init)[-args.test_top_k :]
    all_acc_with_random_init = []
    for dataset_id in top_k_dataset_ids:
        dataset_name = f"random_data_{dataset_id}"
        logger.info(f"Test `{dataset_name}` on random initialized model" + "-" * 50)
        all_model_acc = []
        for _ in range(args.n_test_models):
            # randomly initialize parameters
            model.load_state_dict(initial_state_dict)
            model.reset_additional_parameters()
            # run test
            acc = test_random_data(args, model, train_loader, test_loader=test_loader)
            all_model_acc.append(acc)
        all_acc_with_random_init.append(all_model_acc)
        logger.info(
            f"Result: {np.mean(all_model_acc)*100:.1f}±{np.std(all_model_acc)*100:.1f}"
        )

    logger.info(
        "Best results on fixed init.: {:.1%} (random_data_{})".format(
            np.max(all_acc_with_fixed_init), np.argmax(all_acc_with_fixed_init)
        )
    )
    top_dataset_id = np.argmax(np.mean(all_acc_with_random_init, axis=1))

    logger.info(
        "Best result on random init.: {:.1f}±{:.1f} (random_data_{})".format(
            np.mean(all_acc_with_random_init[top_dataset_id]) * 100,
            np.std(all_acc_with_random_init[top_dataset_id]) * 100,
            top_k_dataset_ids[top_dataset_id],
        )
    )


def run_distilled_data(
    args: argparse.Namespace,
    model: BertClassifier,
    tokenizer: PreTrainedTokenizer,
    test_loader: DataLoader,
    dataset_attrs: dict,
) -> None:

    # load train dataset
    train_data_path = os.path.join(args.data_dir, "train_full_data")
    train_data = get_full_dataset(
        dataset_attrs,
        tokenizer,
        train_data_path,
        data_type="train",
        preprocess=args.preprocess,
    )
    logger.info(f"Train data size: {len(train_data)}")

    # train dataloader
    train_loader = create_dataloader(
        train_data,
        batch_size=args.train_batch_size,
        data_type="train",
        fix_batch_seq_len=args.fix_batch_seq_len,
        num_workers=args.num_workers,
    )

    # logging steps
    if args.logging_steps == 0:
        args.logging_steps = max(len(train_loader) // 20, 1)

    # distilled data
    data_shape = (model.bert_config.max_position_embeddings, model.bert_config.dim)

    if args.pretrained_distilled_data:
        # Test pretrained distilled data
        logger.info(f"Test `{args.pretrained_distilled_data}`!!")

        distilled_data = DistilledData.load_distilled_data(
            args.pretrained_distilled_data
        )
        distilled_data.init_trainer(args, test_loader)

        if not args.random_init and args.n_test_models != 1:
            logger.warning("Test distilled data with known initial parameters only!!")
            args.n_test_models = 1

        all_acc = []
        for i in range(args.n_test_models):
            logger.info(f"Test distilled data on model[{i}]")

            model.load_state_dict(torch.load(args.initial_model_path))
            if args.random_init:
                model.reset_additional_parameters()

            # test distilled data
            acc = test_distilled_data(
                args, model, distilled_data, test_loader=test_loader
            )
            all_acc.append(acc)

        logger.info(f"Result: {np.mean(all_acc)*100:.1f}±{np.std(all_acc)*100:.1f}")

    else:
        # Train Distilled Data
        logger.info("Train Distilled Data!!")

        # initialize distilled data
        distilled_data = DistilledData(
            data_shape,
            dataset_attrs["num_classes"],
            data_size=args.data_size,
            label_type=args.label_type,
        )

        # initialize distilled data trainer
        distilled_data.init_trainer(args, train_loader)

        for epoch in range(args.n_distill_epochs):
            # train distilled data
            logger.info(f"Epoch[{epoch+1}]: Train distilled data " + "-" * 70)
            distilled_data.train_distilled_data(model, epoch)

            # test distilled data
            logger.info(f"Epoch[{epoch+1}]: Test distilled data" + "-" * 70)
            test_distilled_data(
                args, model, distilled_data, train_loader, test_loader, init_model=True
            )

            # save distilled data
            distilled_data_path = os.path.join(
                args.data_dir, f"distilled_data_{epoch+1}"
            )
            distilled_data.save_distilled_data(distilled_data_path)
            logger.info(f"Save distilled data in `{distilled_data_path}`")


def main():

    # settings
    args, bert_model, tokenizer, dataset_attrs = settings()

    # make dir
    make_dir(args.model_dir)
    make_dir(args.data_dir)

    # initialize logger
    if args.pretrained_distilled_data:
        init_logging(os.path.join(args.model_dir, "log_test.txt"))
    else:
        init_logging(os.path.join(args.model_dir, "log_train.txt"))

    if args.comment:
        logger.info(f"Comment: {args.comment}")
    logger.info("Args = {}".format(str(args)))

    # load test dataset
    test_data_path = os.path.join(args.data_dir, "test_dataset")
    test_data = get_full_dataset(
        dataset_attrs,
        tokenizer,
        test_data_path,
        data_type="test",
        preprocess=args.preprocess,
    )
    logger.info(f"Test data size: {len(test_data)}")

    # build classification model
    model = BertClassifier(
        bert_model, dataset_attrs["num_classes"], drop_p=args.drop_p
    ).to(args.device)

    # save or load fixed initial model parameters
    if os.path.exists(args.initial_model_path):
        model.load_state_dict(torch.load(args.initial_model_path))
    else:
        torch.save(model.state_dict(), args.initial_model_path)

    # test dataloader
    test_loader = create_dataloader(
        test_data,
        batch_size=args.test_batch_size,
        data_type="test",
        num_workers=args.num_workers,
    )

    # run training
    if args.mode == "full":
        run_full_data(args, model, tokenizer, test_loader, dataset_attrs)
    elif args.mode == "random":
        run_random_data(args, model, tokenizer, test_loader, dataset_attrs)
    elif args.mode == "distill":
        run_distilled_data(args, model, tokenizer, test_loader, dataset_attrs)


if __name__ == "__main__":
    main()
