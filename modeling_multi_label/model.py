"""
Multi-label model
"""
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:

            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)

            loss *= one_sided_w

        return - loss.mean()


class MultiLabelModel(RobertaPreTrainedModel):
    """
    Roberta followed by a linear classifier
    """

    def __init__(self, config):
        super(MultiLabelModel, self).__init__(config)

        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
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
            ),
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
            return dict(
                loss=loss,
                logits=logits,
            )
        else:
            return logits
