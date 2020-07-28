"""
Train the model
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import CONFIG
from bp_mll import BPMLLLoss, hamming_loss
from dataset import PaperDataset, generate_train_set
from model import load, backup
from evaluate import evaluate

import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logger.log", level=logging.INFO,
    format="%(asctime)-15s - %(levelname)s: %(message)s"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(training_set: PaperDataset, test_set: PaperDataset,
          model: nn.Module, loss_ftc: nn.Module, optimizer, config):
    """
    train the model
    """
    model.train()
    data = DataLoader(training_set, batch_size=config.HyperParam.batch_size, shuffle=True)
    min_hloss = float("inf")
    i = 0
    accomulation_step = config.HyperParam.accumulation_step
    optimizer.zero_grad()
    for epoch in range(1, config.HyperParam.epoch+1):
        for j, (seq, mask, target, _) in enumerate(data):
            output = model(
                seq,
                attention_mask=mask
            )
            loss = loss_ftc(output, target.view_as(output)) / accomulation_step
            loss.backward()
            i += 1
            if i % accomulation_step == 0:
                i = 0
                optimizer.step()
                optimizer.zero_grad()
                logger.info(f"[Epoch {epoch}] loss = {loss:.8f}")
        if epoch % 1 == 0:
            hloss = test(test_set, training_set, model, config)
            evaluate(
                model,
                test_set=test_set,
                training_set=training_set,
                tag=f"h_{hloss:.8f}_{datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')}",
                hloss=hloss
            )
            if hloss < min_hloss:
                min_hloss = hloss
                backup(model, optimizer, "best", config)


def test(test_set: PaperDataset, training_set: PaperDataset, model: nn.Module, config):
    """
    compute the hamming loss
    """
    model.eval()
    with torch.no_grad():
        output = model(training_set.x, training_set.mask)
        hloss_training = hamming_loss(
            output,
            training_set.y.view_as(output),
            threshold=config.Predict.positive_threshold
        )
        logger.info(f"H_Loss (training_set): {hloss_training}")

        output = model(test_set.x, test_set.mask)
        hloss_test = hamming_loss(
            output,
            test_set.y.view_as(output),
            threshold=config.Predict.positive_threshold
        )
        logger.info(f"H_Loss (test_set): {hloss_test}")
        return hloss_test


if __name__ == "__main__":

    config = CONFIG  # get the config file, type: utils.ConfigDict

    model, optimizer = load(config, device)
    model.to(device)

    if torch.cuda.device_count() > 1:
        logger.info(f"Use {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    training_set, test_set = \
        generate_train_set(
            config.Dataset.dataset_path, device
        )  # get training set and test set

    loss_ftc = BPMLLLoss(config.Loss.bias)  # use BP-MLL as loss function
    loss_ftc.to(device)

    train(
        training_set=training_set,
        test_set=test_set,
        model=model,
        loss_ftc=loss_ftc,
        optimizer=optimizer,
        config=config
    )  # train the model

    evaluate(
        model=model,
        tag=f"final_{datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')}",
        training_set=training_set,
        test_set=test_set
    )  # evaluate the result
