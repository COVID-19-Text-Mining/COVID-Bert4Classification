"""
Model
"""

from transformers import BertModel, BertPreTrainedModel, AdamW
import torch
import torch.nn as nn

import os.path
from collections import OrderedDict


class Net(BertPreTrainedModel):
    """
    BERT followed by a linear classifier
    """
    def __init__(self, config, project_config):
        super(Net, self).__init__(config)
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(
            project_config.NetWork.dropout_prob)

        self.classifier = nn.Linear(
            config.hidden_size, len(project_config.Dataset.cats)
        )
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[0][:, 0, :]  # the hidden state of [CLS]

        output = self.dropout(output)
        logits = self.classifier(output)
        logits = torch.sigmoid(logits)

        return logits


def backup(model: nn.Module, optimizer: nn.Module, label, config):
    """
    save the model
    """
    model_state = model.state_dict()
    new_model_state = OrderedDict()
    for k, v in model_state.items():
        namekey = k.lstrip("module.")  # remove `module.`
        new_model_state[namekey] = v

    torch.save(
        {"model": new_model_state, "optimizer": optimizer.state_dict()},
        f"{config.IO.model_dir}model_{label}.pth"
    )


def load(config, device, load_old=True, no_file_warning=False):
    """
    load the model

    if load_model is True, the script will try to load the old model "model/model_best.pth"

    if no_file_warning is set to True,
    it will raise a FileNotFoundError when
     no model found in the target dir
    """
    path = None
    if load_old:
        path = config.IO.model_dir + "model_best.pth"
        if not os.path.isfile(path):
            if no_file_warning:
                raise FileNotFoundError(f"Cannot find {path}, please train the model before using it.")
            path = None

    if path is not None:
        state_dict = torch.load(path, map_location=device)
        new_model = Net.from_pretrained(
            config.NetWork.pretrained_model,
            state_dict=state_dict["model"],
            project_config=config
        )
        new_optimizer = AdamW(new_model.parameters(), lr=config.HyperParam.lr)
        new_optimizer.load_state_dict(
            state_dict["optimizer"]
        )
    else:
        new_model = Net.from_pretrained(
            config.NetWork.pretrained_model,
            project_config=config
        )
        new_optimizer = AdamW(new_model.parameters(), lr=config.HyperParam.lr)

    return new_model, new_optimizer
