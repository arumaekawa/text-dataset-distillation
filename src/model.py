from collections import OrderedDict

from torch import nn
from torch.nn import functional as F


def cross_entropy_with_soft_labels(logits, labels):
    return (-labels * F.log_softmax(logits, dim=-1)).sum(-1)


class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, drop_p=0.1):
        super().__init__()

        # config
        self.bert_config = bert_model.config
        self.num_classes = num_classes

        # pretrained bert model
        self.bert_model = bert_model

        # pooler
        if not hasattr(bert_model, "pooler"):
            self.pooler = nn.Sequential(
                nn.Linear(bert_model.config.dim, bert_model.config.dim), nn.Tanh()
            )
        else:
            self.pooler is None

        # dropout layer
        self.dropout = nn.Dropout(p=drop_p)
        # single linear layer for classification
        self.classifier = nn.Linear(self.bert_config.hidden_size, num_classes)

        # loss function
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        output_attentions=False,
    ):
        # encode input sequences with bert model
        bert_outputs = self.bert_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        # hidden state of [CLS] token
        if "pooler_output" in bert_outputs.keys():
            pooler_output = bert_outputs["pooler_output"]
        else:
            pooler_output = self.pooler(bert_outputs["last_hidden_state"][:, 0])

        # dropout
        pooler_output = self.dropout(pooler_output)

        # classifier layer
        logits = self.classifier(pooler_output)

        # calculate losses
        if labels is not None:
            if logits.shape == labels.shape:
                losses = cross_entropy_with_soft_labels(logits, labels)
            else:
                losses = self.cross_entropy(logits, labels)
            return losses, logits, bert_outputs

        return logits, bert_outputs

    def forward_with_params(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        weights=None,
        output_attentions=False,
    ):
        assert weights is not None
        module_name_list = ["bert_model", "classifier"]
        if self.pooler is not None:
            module_name_list.append("pooler")
        weights_dict = {module_name: OrderedDict() for module_name in module_name_list}
        for name, param in weights.items():
            module_name, param_name = name.split(".", maxsplit=1)
            weights_dict[module_name][param_name] = param

        # encode input sequences with bert model
        bert_outputs = self.bert_model.forward_with_params(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            weights=weights_dict["bert_model"],
            output_attentions=output_attentions,
        )

        # hidden state of [CLS] token
        if "pooler_output" in bert_outputs.keys():
            pooler_output = bert_outputs["pooler_output"]
        else:
            pooler_output = F.linear(
                bert_outputs["last_hidden_state"][:, 0],
                weight=weights_dict["pooler"]["0.weight"],
                bias=weights_dict["pooler"]["0.bias"],
            )
            pooler_output = self.pooler[1](pooler_output)

        # dropout
        pooler_output = self.dropout(pooler_output)

        # classifier layer
        logits = F.linear(
            pooler_output,
            weight=weights_dict["classifier"]["weight"],
            bias=weights_dict["classifier"]["bias"],
        )

        # calculate losses
        if labels is not None:
            if logits.shape == labels.shape:
                losses = cross_entropy_with_soft_labels(logits, labels)
            else:
                losses = self.cross_entropy(logits, labels)
            return losses, logits, bert_outputs

        return logits, bert_outputs

    def reset_additional_parameters(self):
        if self.pooler is not None:
            self.pooler[0].reset_parameters()
        self.classifier.reset_parameters()
