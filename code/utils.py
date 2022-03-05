import logging
import os
import pathlib

import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from datasets.features.features import ClassLabel
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def make_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def get_full_dataset(
    dataset_attrs,
    tokenizer,
    dataset_path,
    data_type="train",
    preprocess=False,
):
    # load dataset from file if it already exists
    if os.path.exists(dataset_path) and not preprocess:
        logger.info(f"Load {data_type} data from `{dataset_path}`")
        dataset = load_from_disk(dataset_path)
        return dataset

    # load dataset
    dataset = load_dataset(dataset_attrs["name"], split=data_type)

    # rename columns
    assert isinstance(dataset.features[dataset_attrs["label_keys"]], ClassLabel)
    dataset = dataset.map(
        lambda ex: {
            "text": ex[dataset_attrs["text_key"]],
            "labels": ex[dataset_attrs["label_key"]],
        },
        batched=True,
    )

    # tokenize text to input ids
    def tokenize(example):
        # input token ids (tensor[int])
        return tokenizer(example["text"], truncation=True, padding="max_length")

    logger.info(f"Tokenize all samples of {data_type} data ...")
    dataset = dataset.map(tokenize, batched=True, num_proc=10)

    # save preprocessed dataset to file
    dataset.save_to_disk(dataset_path)
    logger.info(f"Save {data_type} dataset in `{dataset_path}`")

    return dataset


def get_random_dataset(
    dataset_attrs,
    tokenizer,
    full_dataset_path,
    random_dataset_path,
    data_size=1,
    num_dataset=1,
    preprocess=False,
):

    # load dataset from file if it already exists
    if os.path.exists(random_dataset_path) and not preprocess:
        logger.info(f"Load random dataset list from `{random_dataset_path}`")
        random_dataset_list = [
            load_from_disk(os.path.join(random_dataset_path, f"random_data_{i}"))
            for i in range(num_dataset)
        ]
        return random_dataset_list

    # get full version dataset
    full_dataset = get_full_dataset(
        dataset_attrs, tokenizer, full_dataset_path, preprocess=preprocess
    )

    random_dataset_list = []
    # split by label
    each_class_dataset = [
        full_dataset.select(
            torch.nonzero(torch.tensor(full_dataset["labels"]) == label)
        )
        for label in sorted(full_dataset.unique("label"))
    ]
    # Create as many datasets as `num_dataset`
    for _ in range(num_dataset):
        single_random_dataset = []
        # retrieve randomly from each class
        for class_dataset in each_class_dataset:
            single_random_dataset.append(
                class_dataset.select(torch.randint(len(class_dataset), (data_size,)))
            )
        single_random_dataset = concatenate_datasets(single_random_dataset)
        random_dataset_list.append(single_random_dataset)

    # save preprocessed datasets list to each file
    for i, dataset in enumerate(random_dataset_list):
        dataset.save_to_disk(os.path.join(random_dataset_path, f"random_data_{i}"))

    logger.info(f"Save random dataset list in `{random_dataset_path}`")

    return random_dataset_list


def create_dataloader(
    dataset, batch_size=1, data_type="train", num_workers=1, fix_batch_seq_len=False
):
    def collate_fn(data):
        # max length of sequences in batch
        if fix_batch_seq_len:
            max_seq_len = len(data[0]["input_ids"])
        else:
            max_seq_len = max([d["attention_mask"].sum().item() for d in data])
        # input_ids
        input_ids = torch.stack([d["input_ids"][:max_seq_len] for d in data])
        # attention mask
        attention_mask = torch.stack([d["attention_mask"][:max_seq_len] for d in data])
        # labels
        labels = torch.stack([d["labels"] for d in data])
        return input_ids, attention_mask, labels

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=data_type == "train",
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return dataloader
