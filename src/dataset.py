"""
Dataset class to provide training data
"""

import json

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import RobertaTokenizer

from config import PRETRAINED_MODEL


class PaperDataset(Dataset):
    def __init__(self, papers, cats):
        x = []
        y = []
        text = []
        mask = []

        roberta_tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL)

        for paper in tqdm(papers, desc="Preparing dataset"):
            abstract_text = paper["abstract"]
            # make sure the order is right
            label = [paper["label"][label_name] for label_name in cats]
            if abstract_text.strip() and any(label):
                ids = roberta_tokenizer(
                    abstract_text,
                    add_special_tokens=True,
                    padding="max_length",
                    truncation="longest_first",
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_overflowing_tokens=False,
                    return_special_tokens_mask=False,
                    return_offsets_mapping=False,
                )
                x.append(ids["input_ids"])
                mask.append(ids["attention_mask"])
                y.append(label)
                text.append(abstract_text)

        self.x = torch.tensor(x, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.text = text
        assert self.x.size()[0] == self.mask.size()[0] == self.y.size()[0] == len(self.text)

    def __getitem__(self, index):
        return {
            "input_ids": self.x[index],
            "attention_mask": self.mask[index],
            "labels": self.y[index],
            "text": self.text[index],
        }

    def __len__(self):
        return len(self.text)

    @classmethod
    def from_file(cls, path, cats):
        with open(path, "r", encoding="utf-8") as f:
            papers = json.load(f)
        dataset = cls(papers, cats)
        return dataset
