"""
Multi-label model
"""
import torch
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

        self.loss_ftc = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [1.4067919254302979, 1.2223843336105347, 1.0514204502105713, 0.6244935393333435,
                 1.0210953950881958, 0.8654600381851196, 1.2361549139022827]
            ),
            weight=torch.tensor(
                [1.0690486431121826, 0.2334759533405304, 0.18737506866455078, 0.33329978585243225,
                 2.0128180980682373, 2.6837575435638428, 0.48022496700286865]
            )
        )

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
