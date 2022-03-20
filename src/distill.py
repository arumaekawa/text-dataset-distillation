import argparse
import logging
import time
from collections import OrderedDict

import torch
from torch.cuda import amp
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from evaluate import evaluate
from model import BertClassifier

logger = logging.getLogger(__name__)


class DistilledData:
    def __init__(
        self,
        input_embeds_shape,
        num_classes,
        data_size=1,
        label_type="hard",
        attention_label_type="none",
        attention_label_shape=None,
    ):
        # config
        self.num_classes = num_classes
        self.data_size_per_class = data_size
        self.label_type = label_type
        self.attention_label_type = attention_label_type

        self.init_distilled_data(
            input_embeds_shape, attention_label_shape=attention_label_shape
        )

    def init_distilled_data(
        self,
        input_embeds_shape,
        attention_label_shape=None,
    ):
        # initialize inputs_embeds (M * S * E)
        self.inputs_embeds = torch.randn(
            self.num_classes * self.data_size_per_class, *input_embeds_shape
        )
        # initialize labels (M * C)
        label_classes = torch.tensor(
            [[c] * self.data_size_per_class for c in range(self.num_classes)]
        ).view(-1)
        self._labels = torch.eye(self.num_classes)[label_classes]
        # initialize attention labels (M * L * H * S * S)
        if self.attention_label_type != "none":
            assert attention_label_shape is not None
            self._attention_labels = torch.randn(
                self.num_classes * self.data_size_per_class, *attention_label_shape
            )
        # initialize learning rate and decay factor
        self.model_lr, self.step_lr_gamma = None, None

    @property
    def labels(self):
        if self.label_type == "soft":
            return F.softmax(self._labels, dim=-1)
        else:
            return self._labels

    @property
    def attention_labels(self):
        if self.attention_label_type:
            return F.softmax(self._attention_labels, dim=-1)
        else:
            return None

    def init_trainer(self, args, train_loader):
        # training settings
        self.n_distill_epochs = args.n_distill_epochs
        self.distill_lr = args.distill_lr
        self.max_grad_norm = args.distill_max_grad_norm
        self.n_inner_steps = args.n_inner_steps
        self.optimize_lr = args.optimize_lr
        self.accum_loss = args.accum_loss
        self.random_init = args.random_init
        self.device_ids = args.device_ids
        self.logging_steps = args.logging_steps
        self.device = args.device
        self.use_amp = args.use_amp
        self.dtype = args.dtype
        self.attention_kl_lambda = args.attention_kl_lambda

        if self.model_lr is None:
            self.model_lr = torch.tensor(args.distill_model_lr)
        if self.step_lr_gamma is None:
            self.step_lr_gamma = torch.tensor(args.distill_step_lr_gamma)

        # set on device
        self.inputs_embeds = self.inputs_embeds.to(self.device)
        self._labels = self._labels.to(self.device)
        if self.attention_label_type != "none":
            self._attention_labels = self._attention_labels.to(self.device)
        self.model_lr = self.model_lr.to(self.device)
        self.step_lr_gamma = self.step_lr_gamma.to(self.device)

        # train data loader
        self.train_loader = train_loader
        # number of traning steps
        num_tot_train_steps = len(train_loader) * self.n_distill_epochs
        # set optimizer of distilled data
        self.optimize_param_list = [self.inputs_embeds]
        if self.label_type != "hard":
            self.optimize_param_list += [self._labels]
        if self.attention_label_type != "none":
            self.optimize_param_list += [self._attention_labels]
        if self.optimize_lr:
            self.optimize_param_list += [self.model_lr, self.step_lr_gamma]
        for param in self.optimize_param_list:
            param.requires_grad = True
        self.d_optimizer = Adam(self.optimize_param_list, lr=args.distill_lr)
        # scheduler (linear decay with linear warmup)
        self.d_scheduler = get_linear_schedule_with_warmup(
            self.d_optimizer,
            int(num_tot_train_steps * args.distill_warmup_ratio),
            num_tot_train_steps,
        )
        # gradient scaler for mixed precision
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        # initial model parameters
        self.initial_state_dict = torch.load(args.initial_model_path)

    def train_distilled_data(self, model: BertClassifier, epoch: int):
        # set model on device
        model = model.to(self.device)
        # initialize model
        model.load_state_dict(self.initial_state_dict)
        # training loop
        cur_num, cur_before_loss, cur_after_loss = 0, 0, 0
        cur_before_correct, cur_after_correct = 0, 0
        with tqdm(self.train_loader, ncols=140, desc=f"Epoch[{epoch+1}]") as pbar:
            for outer_step, (input_ids, attention_mask, labels) in enumerate(pbar):
                # initialize model parameters
                if self.random_init:
                    model.reset_additional_parameters()
                # model parameters
                weights = OrderedDict(model.named_parameters())

                batch_size = len(input_ids)
                cur_num += batch_size

                # acc & loss of initial parameters (before updating with distilled data)
                with torch.no_grad():
                    with amp.autocast(dtype=self.dtype, enabled=self.use_amp):
                        before_losses, before_logits, _ = model.forward_with_params(
                            input_ids=input_ids.to(self.device),
                            attention_mask=attention_mask.to(self.device),
                            labels=labels.to(self.device),
                            weights=weights,
                        )
                        cur_before_loss += before_losses.mean().item() * batch_size
                        cur_before_correct += (
                            before_logits.cpu().argmax(1).eq(labels).sum().item()
                        )

                # update model parameters with distilled data
                loss = 0
                for inner_step in range(self.n_inner_steps):
                    # forward
                    d_losses, _, bert_outputs = model.forward_with_params(
                        inputs_embeds=self.inputs_embeds,
                        labels=self.labels,
                        weights=weights,
                        output_attentions=True,
                    )
                    d_loss = d_losses.mean()
                    if self.attention_label_type != "none":
                        attn_weights = torch.stack(bert_outputs["attentions"], dim=1)
                        if self.attention_label_type == "cls":
                            attn_weights = attn_weights[..., 0, :]
                        assert attn_weights.shape == self.attention_labels.shape
                        d_attn_kl = F.kl_div(
                            torch.log(attn_weights + 1e-12), self.attention_labels
                        )
                        d_loss = d_loss + d_attn_kl * self.attention_kl_lambda
                    d_loss = d_loss * self.model_lr * (self.step_lr_gamma**inner_step)

                    # backward
                    grads = torch.autograd.grad(
                        d_loss, weights.values(), create_graph=True, allow_unused=True
                    )
                    # update parameters (SGD)
                    weights = OrderedDict(
                        (name, param - grad) if grad is not None else (name, param)
                        for ((name, param), grad) in zip(weights.items(), grads)
                    )

                    if self.accum_loss or (inner_step + 1) == self.n_inner_steps:
                        # loss of updated parameters (after each gradient step)
                        with amp.autocast(dtype=self.dtype, enabled=self.use_amp):
                            after_losses, after_logits, _ = model.forward_with_params(
                                input_ids=input_ids.to(self.device),
                                attention_mask=attention_mask.to(self.device),
                                labels=labels.to(self.device),
                                weights=weights,
                            )
                            after_loss = after_losses.mean()
                            loss += after_loss

                cur_after_loss += after_loss.item() * batch_size
                cur_after_correct += (
                    after_logits.cpu().argmax(1).eq(labels).sum().item()
                )

                self.d_optimizer.zero_grad()
                # backward
                self.scaler.scale(loss).backward()
                # unscale gradients (for gradient clipping)
                self.scaler.unscale_(self.d_optimizer)
                # gradient cliping
                torch.nn.utils.clip_grad_norm_(
                    self.optimize_param_list, self.max_grad_norm
                )
                self.scaler.step(self.d_optimizer)
                self.scaler.update()
                self.d_scheduler.step()

                # logging
                if (outer_step + 1) % self.logging_steps == 0:
                    logger.info(
                        "Epoch[{:.2f}] | (before) loss: {:>6.4f}, acc: {:5.2%}"
                        " -> (after) loss: {:>6.4f}, acc: {:5.2%}"
                        " | lr={:.2E}, gamma={:.2f}".format(
                            epoch + (outer_step + 1) / len(pbar),
                            cur_before_loss / cur_num,
                            cur_before_correct / cur_num,
                            cur_after_loss / cur_num,
                            cur_after_correct / cur_num,
                            self.model_lr.item(),
                            self.step_lr_gamma.item(),
                        )
                    )
                    cur_num, cur_before_loss, cur_after_loss = 0, 0, 0
                    cur_before_correct, cur_after_correct = 0, 0

                # update infomation of progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{after_loss.item():.4}",
                        "lr": f"{self.d_scheduler.get_last_lr()[0]:.1E}",
                        "gd_scale": f"{self.scaler.get_scale()}",
                    }
                )

    def train_model_on_distilled_data(
        self, model: BertClassifier, init_model: bool = False
    ):
        if init_model:
            # initialize model parameters (fixed initial parameters)
            model.load_state_dict(self.initial_state_dict)
        # optimizer
        model_opt = SGD(model.parameters(), lr=1.0)
        # gradient updating with distilled data
        start_time = time.time()
        for inner_step in range(self.n_inner_steps):
            # forward
            losses, _, bert_outputs = model(
                inputs_embeds=self.inputs_embeds,
                labels=self.labels,
                output_attentions=True,
            )
            loss = losses.mean()
            if self.attention_label_type != "none":
                attn_weights = torch.stack(bert_outputs["attentions"], dim=1)
                if self.attention_label_type == "cls":
                    attn_weights = attn_weights[..., 0, :]
                assert attn_weights.shape == self.attention_labels.shape
                attn_kl = F.kl_div(
                    torch.log(attn_weights + 1e-12), self.attention_labels
                )
                loss = loss + attn_kl * self.attention_kl_lambda
            loss = loss * self.model_lr * (self.step_lr_gamma**inner_step)
            # backward
            model_opt.zero_grad()
            loss.backward()
            model_opt.step()
        end_time = time.time() - start_time
        logger.info(f"Time for model traning : {end_time:.2f}s")

    @property
    def data_dict(self):
        return {
            "config": {
                "input_embeds_shape": self.inputs_embeds.shape[1:],
                "num_classes": self.num_classes,
                "data_size": self.data_size_per_class,
                "label_type": self.label_type,
                "attention_label_type": self.attention_label_type,
                "attention_label_shape": self._attention_labels.shape[1:]
                if self.attention_label_type != "none"
                else None,
            },
            "inputs_embeds": self.inputs_embeds.cpu().data,
            "labels": self._labels.cpu().data,
            "attention_labels": self._attention_labels.cpu().data
            if self.attention_label_type != "none"
            else None,
            "lr": self.model_lr.cpu().data,
            "gamma": self.step_lr_gamma.cpu().data,
        }

    def save_distilled_data(self, path):
        # save data as dict
        torch.save(self.data_dict, path)

    @classmethod
    def load_distilled_data(cls, path):
        """
        examples:
            distilled_data = DistilledData.load_distilled_data(path)
            distilled_data.train_model_on_distilled_data
        """
        # load data from path
        data_dict = torch.load(path)
        # make new instance
        distilled_data = cls(**data_dict["config"])
        # set pretrained distilled data and learning rate
        distilled_data.inputs_embeds = data_dict["inputs_embeds"]
        distilled_data._labels = data_dict["labels"]
        distilled_data.model_lr = data_dict["lr"]
        distilled_data.step_lr_gamma = data_dict["gamma"]
        distilled_data._attention_labels = data_dict["attention_labels"]

        return distilled_data


def test_distilled_data(
    args: argparse.Namespace,
    model: BertClassifier,
    distilled_data: DistilledData,
    train_loader: DataLoader = None,
    test_loader: DataLoader = None,
    init_model: bool = False,
):
    assert train_loader is not None or test_loader is not None

    # initialize model
    if init_model:
        model.load_state_dict(torch.load(args.initial_model_path))

    # test initial model on test dataset
    if train_loader is not None:
        before_train_acc, before_train_loss = evaluate(args, model, train_loader)
    # test initial model on test dataset
    if test_loader is not None:
        before_test_acc, before_test_loss = evaluate(args, model, test_loader)

    # train model with distilled data
    distilled_data.train_model_on_distilled_data(model)

    # test trained model on train dataset
    if train_loader is not None:
        after_train_acc, after_train_loss = evaluate(args, model, train_loader)
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
