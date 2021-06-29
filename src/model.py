"""
Multi-label model
"""

import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel


class RobertaMultiLabelModel(RobertaPreTrainedModel):
    """
    Roberta followed by a linear classifier
    """

    def __init__(self, config, dropout_prob):
        super(RobertaMultiLabelModel, self).__init__(config)

        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(
            dropout_prob
        )
        self.classifier = nn.Linear(
            config.hidden_size, self.num_labels
        )

        self.loss_ftc = nn.BCEWithLogitsLoss()

    def forward(
            self,
            input_ids,
            labels=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        output = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[0][:, 0, :]  # the hidden state of <s>

        output = self.dropout(output)
        logits = self.classifier(output)

        if labels is not None:
            loss = self.loss_ftc(logits, labels)
        else:
            loss = None

        return dict(
            loss=loss,
            logits=logits,
        )
