"""
Dataset class to provide training data
"""

import json
import re
from typing import Optional, Dict, Any, List, Union, Iterable

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import RobertaTokenizer

from .config import PRETRAINED_MODEL, CATS


class BasePaperDataset:
    def __init__(
            self,
            papers,
            text_key: str = "abstract",
            label_key: Optional[str] = None,
    ):
        """
        Args:
            papers: Collection of paper w/o labels
            text_key: Corresponding key to get text in :obj:`papers`, default to "abstract"
            label_key: Corresponding key to get labels in :obj:`papers`,
                default to :obj:`None` to indicate no label available (useful for model evaluation)
        """
        self.papers = papers  # keeps all the original data
        self.text_key = text_key
        self.label_key = label_key

        self.tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL, mirror="tuna")

    def _process(self, paper: Dict[str, Any]) -> Dict[str, Union[str, torch.Tensor]]:
        text = paper[self.text_key]

        if self.label_key is None:
            # when no label available
            label = None
        else:
            # make sure the order is right
            label = torch.tensor([
                paper[self.label_key][label_name] for label_name in CATS
            ], dtype=torch.float32)

        output = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_offsets_mapping=False,
        )

        return {
            "input_ids": output["input_ids"],
            "attention_mask": output["attention_mask"],
            "text": paper[self.text_key],
            "label_ids": label,
        }


class InMemoryPaperDataset(Dataset, BasePaperDataset):
    def __init__(
            self,
            papers: List[Dict[str, Any]],
            text_key: str = "abstract",
            label_key: Optional[str] = None,
    ):
        super(InMemoryPaperDataset, self).__init__(papers=papers, text_key=text_key, label_key=label_key)
        self.dataset: List[Optional[Dict[str, Any]]] = [None] * len(self.papers)  # placeholder for processed data

    def __getitem__(self, index):
        if self.dataset[index] is None:
            self.dataset[index] = self._process(self.papers[index])
        return self.dataset[index]

    def __len__(self):
        return len(self.papers)

    @classmethod
    def from_file(cls, path, text_key, label_key):
        """
        Load dataset from json file

        Args:
            path: the path to the json file
            text_key: see `~PaperDataset.__init__`
            label_key: see `~PaperDataset.__init__`
        """
        with open(path, "r", encoding="utf-8") as f:
            papers = json.load(f)

        return cls(
            papers=papers,
            text_key=text_key,
            label_key=label_key,
        )


class IterablePaperDataset(IterableDataset, BasePaperDataset):
    papers: Iterable[Dict[str, Any]]

    def __iter__(self):
        for paper in self.papers:
            if re.sub(r"\W", "", paper[self.text_key]):
                yield self._process(paper)
