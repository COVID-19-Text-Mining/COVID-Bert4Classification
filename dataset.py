"""
Load the training data
"""

from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
import json
from random import shuffle
import utils


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
            text.append(paper[self.config.Dataset.text_key])
            ids = self.bert_tokenizer.encode(paper[self.config.Dataset.text_key],
                                             add_special_tokens=True,
                                             max_length=512,
                                             pad_to_max_length=True)
            x.append(ids)
            mask.append([int(i > 0) for i in ids])
            y.append([paper[self.config.Dataset.label_key][i] for i in self.cats])

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
