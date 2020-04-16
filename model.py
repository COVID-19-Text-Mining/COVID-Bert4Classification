"""
Model
"""

from transformers import BertModel
import torch
import torch.nn as nn

from torch.optim import Adam
import os.path


class Net(nn.Module):
    """
    BERT followed by a linear classifier
    """
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.NetWork.pretrained_model)
        self.dropout = nn.Dropout(config.NetWork.dropout_prob)
        self.classifier = nn.Linear(
            config.NetWork.hidden_size, config.NetWork.label_num
        )

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
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
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
    new_model = Net(config)
    new_optimizer = Adam(new_model.parameters(), lr=config.HyperParam.lr)

    path = None
    if load_old:
        path = config.IO.model_dir + "model_best.pth"
        if not os.path.isfile(path):
            if no_file_warning:
                raise FileNotFoundError(f"Cannot find {path}, please train the model before using it.")
            path = None

    if path is not None:
        state_dict = torch.load(path, map_location=device)
        new_model.load_state_dict(state_dict["model"])
        new_optimizer.load_state_dict(state_dict["optimizer"])
    return new_model, new_optimizer
