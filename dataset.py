"""
Load the training data
"""

from torch.utils.data import Dataset
from transformers import BertTokenizer

import torch
import json
from random import shuffle
import utils
import warnings

try:
    from sklearn.model_selection import KFold
except ModuleNotFoundError as e:
    warnings.warn(f"Cannot import {e.name}. "
                  f"Please install the sklearn package"
                  f" if you want to use cross validation model")
    KFold = None


class PaperDataset(Dataset):
    cats = utils.cats
    indexes = utils.indexes

    config = utils.load_config()

    bert_tokenizer = BertTokenizer.from_pretrained(config.Dataset.tokenizer_path)

    def __init__(self, papers, device):
        x = []
        y = []
        text = []
        mask = []

        for paper in papers:
            abstract_text = paper[self.config.Dataset.text_key]
            # make sure the order is right
            label = [paper[self.config.Dataset.label_key][label_name] for label_name in self.cats]
            if abstract_text.strip() and any(label):
                ids = self.bert_tokenizer.encode_plus(
                    abstract_text,
                    add_special_tokens=True,
                    max_length=512,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_overflowing_tokens=False,
                    return_special_tokens_mask=False,
                    return_offsets_mapping=False

                )
                x.append(ids["input_ids"])
                mask.append(ids["attention_mask"])
                y.append(label)
                text.append(abstract_text)

        self.x = torch.tensor(x, dtype=torch.long).to(device)
        self.mask = torch.tensor(mask, dtype=torch.long).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)
        self.text = text
        assert self.x.size()[0] == self.mask.size()[0] == self.y.size()[0] == len(self.text)

    def __iter__(self):
        for each in zip(self.x, self.mask, self.y, self.text):
            yield each

    def __getitem__(self, index):
        return self.x[index], self.mask[index], self.y[index], self.text[index]

    def __len__(self):
        return len(self.text)


def generate_train_set(path, device, test_portion=0.1):
    with open(path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    shuffle(papers)

    training_set = []
    test_set = []
    if 0 < test_portion <= 1:
        test_portion = int(1 / test_portion)
        for i, paper in enumerate(papers):
            if i % test_portion:
                training_set.append(paper)
            else:
                test_set.append(paper)
        return PaperDataset(training_set, device), PaperDataset(test_set, device)
    else:
        return PaperDataset(papers, device)


def generate_cross_validation_sets(path, device, n_splits=10):
    with open(path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    kf = KFold(n_splits, shuffle=True)
    for train_index, test_index in kf.split(papers):
        print("TRAIN: ", train_index)
        print("TEST: ", test_index)

        training_set = PaperDataset([papers[i] for i in train_index], device)
        test_set = PaperDataset([papers[i] for i in test_index], device)

        yield training_set, test_set
