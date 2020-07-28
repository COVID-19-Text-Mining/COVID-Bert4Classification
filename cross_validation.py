"""
Train the model with cross validation
"""

import torch
import torch.nn as nn

from utils import CONFIG
from bp_mll import BPMLLLoss
from dataset import generate_cross_validation_sets
from model import load
from evaluate import evaluate
from train import train

import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logger.log", level=logging.INFO,
    format="%(asctime)-15s - %(levelname)s: %(message)s"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    config = CONFIG  # get the config file, type: utils.ConfigDict
    round = 1
    loss_ftc = BPMLLLoss(config.Loss.bias)  # use BP-MLL as loss function
    loss_ftc.to(device)

    for training_set, test_set in generate_cross_validation_sets(config.Dataset.dataset_path, device):
        # get training set and test set
        if round != 1:
            torch.cuda.empty_cache()
        model, optimizer = load(config, device, load_old=False)
        model.to(device)
        if torch.cuda.device_count() > 1:
            if round == 1:
                logger.info(f"Use {torch.cuda.device_count()} GPUs.")
            model = nn.DataParallel(model)
        logger.info(f"The {round} round cross validation")
        round += 1

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
            tag=f"round_{round}_final_{datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')}",
            training_set=training_set,
            test_set=test_set
        )  # evaluate the result
