import argparse
import datetime
import json
import logging
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import DistilBertTokenizer

from transformers_models import DistilBertModel

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    # "bert": (BertModel, BertTokenizer),
    "distilbert": (DistilBertModel, DistilBertTokenizer),
    # "roberta": (RobertaModel, RobertaTokenizer),
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

all_dataset_attrs_path = os.path.join(BASE_DIR, "all_dataset_attrs.json")
assert os.path.exists(all_dataset_attrs_path)
with open(all_dataset_attrs_path, "r") as f:
    all_dataset_attrs = json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument(
        "--mode", type=str, default="distill", choices=["full", "random", "distill"]
    )
    # output directory
    parser.add_argument("--model_root_dir", type=str, required=True)
    parser.add_argument("--data_root_dir", type=str, required=True)
    # data
    parser.add_argument(
        "--dataset", type=str, default="ag_news", choices=all_dataset_attrs.keys()
    )
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--max_seq_len", type=str, default=None)
    parser.add_argument("--fix_batch_seq_len", action="store_true")
    # model
    parser.add_argument(
        "--model_type", type=str, default="distilbert", choices=MODEL_CLASSES.keys()
    )
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--drop_p", type=float, default=0.1)
    # device
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    # distillation
    parser.add_argument("--data_size", type=int, default=1)
    parser.add_argument(
        "--label_type",
        type=str,
        default="hard",
        choices=["hard", "soft", "unrestricted"],
    )
    parser.add_argument("--accum_loss", action="store_true")
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--optimize_lr", action="store_true")
    parser.add_argument("--n_distill_epochs", type=int, default=10)
    parser.add_argument("--distill_lr", type=float, default=2e-3)
    parser.add_argument("--distill_warmup_ratio", type=float, default=0.1)
    parser.add_argument("--distill_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--n_inner_steps", type=int, default=3)
    parser.add_argument("--distill_model_lr", type=float, default=0.5)
    parser.add_argument("--distill_step_lr_gamma", type=float, default=0.2)
    parser.add_argument("--pretrained_distilled_data", type=str, default="")
    parser.add_argument("--n_test_models", type=int, default=10)
    # full
    parser.add_argument("--model_lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--n_train_epochs", type=int, default=3)
    # random
    parser.add_argument("--num_dataset", type=int, default=100)
    parser.add_argument("--test_top_k", type=int, default=1)
    # others
    parser.add_argument("--train_batch_size", type=int, default=48)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--logging_steps", type=int, default=0)
    parser.add_argument("--initial_model_path", type=str, default="")

    args = parser.parse_args()

    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model_dir_path(args):
    model_dir_name = args.mode

    if args.mode != "full":
        model_dir_name = f"{model_dir_name}_{args.data_size}"
        if args.mode == "distill":
            if args.random_init:
                model_dir_name = f"{model_dir_name}_random_init"
            else:
                model_dir_name = f"{model_dir_name}_fix_init"
            model_dir_name = f"{model_dir_name}_{args.label_type}"

    if args.comment:
        model_dir_name = f"{model_dir_name}_{args.comment.replace(' ', '_')}"

    return os.path.join(
        args.model_root_dir,
        args.model_type,
        args.dataset,
        args.mode,
        model_dir_name,
    )


def settings():
    # parse arguments
    args = parse_args()

    # set random seed for reproducibility
    set_random_seed(args.seed)

    # directory path to save model
    args.model_dir = get_model_dir_path(args)

    # initail model parameters
    if args.initial_model_path == "":
        args.initial_model_path = os.path.join(
            args.model_root_dir, args.model_type, args.dataset, "initial_model.pt"
        )

    # directory path to save dataset
    args.data_dir = os.path.join(args.data_root_dir, args.model_type, args.dataset)

    # model & tokenizer class
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # load pretrained model
    bert_model = model_class.from_pretrained(args.model_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_name)

    # dataset attributions
    dataset_attrs = all_dataset_attrs[args.dataset]

    # gpu device
    assert (
        args.n_gpus <= torch.cuda.device_count()
    ), f"Available number of GPUs({torch.cuda.device_count()}) < n_gpus({args.n_gpus})"
    assert (
        args.mode != "distill" or args.n_gpus <= 1
    ), "Sorry, mode `distill` does not support Multi GPUs"
    args.device_ids = list(range(args.n_gpus))
    args.device = torch.device("cuda" if args.n_gpus > 0 else "cpu")

    # num workers
    if args.num_workers == 0:
        args.num_workers = os.cpu_count()

    # fp16
    args.use_amp = not args.fp32
    args.dtype = torch.bfloat16 if args.bf16 else torch.float16

    # max sequence length
    if args.max_seq_len is None:
        args.max_seq_len = bert_model.config.max_position_embeddings

    return args, bert_model, tokenizer, dataset_attrs


class LoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
            last = self.last
        except AttributeError:
            last = record.relativeCreated

        delta = record.relativeCreated / 1000 - last / 1000
        record.relative = "{:.1f}".format(delta)
        record.uptime = str(datetime.timedelta(seconds=record.relativeCreated // 1000))
        self.last = record.relativeCreated
        return True


def init_logging(filename):
    logging_format = (
        "[%(asctime)s] - %(uptime)s"
        # " (%(relative)ss)"
        # " - %(levelname)s '%(name)s'"
        " - %(message)s"
    )
    logging.basicConfig(
        format=logging_format, filename=filename, filemode="a", level=logging.INFO
    )
    console_handler = LoggingHandler()
    console_handler.setFormatter(
        logging.Formatter(fmt=logging_format, datefmt="%Y/%m/%d %H:%M:%S")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())
