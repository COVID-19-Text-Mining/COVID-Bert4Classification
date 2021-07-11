"""
Dataset class to provide training data
"""

import json
import random
import re
from typing import Dict, Any, List, Union, Iterable, Optional

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import RobertaTokenizer

from .config import PRETRAINED_MODEL, CATS, USE_MIRROR


class BasePaperDataset:
    def __init__(self, papers):
        """
        Args:
            papers: Collection of paper w/o labels
        """
        self.papers = papers  # keeps all the original data

        self.tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL, mirror=USE_MIRROR)

    def _process(self, paper: Dict[str, Any]) -> Dict[str, Union[str, torch.Tensor]]:
        title = paper["title"]
        abstract = paper["abstract"]

        if "label" not in paper:
            # when no label available
            label = None
        else:
            # make sure the order is right
            label = torch.tensor([
                paper["label"][label_name] for label_name in CATS
            ], dtype=torch.float32)

        output = self.tokenizer(
            abstract,
            text_pair=title,
            add_special_tokens=True,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_offsets_mapping=False,
        )

        return {
            "input_ids": output["input_ids"].squeeze(0),
            "attention_mask": output["attention_mask"].squeeze(0),
            "token_type_ids": output["token_type_ids"].squeeze(0),
            "text": paper["abstract"],
            "label_ids": label,
        }


class InMemoryPaperDataset(BasePaperDataset, Dataset):
    """
    Used for training and test
    """
    papers: List[Dict[str, Any]]

    def __init__(self, papers, drop_abstract_prob: Optional[float] = None):
        super(InMemoryPaperDataset, self).__init__(papers=papers)
        if not 0. <= drop_abstract_prob <= 1.:
            raise ValueError("drop_abstract_prob should be 0 ~ 1.")
        self.drop_abstract_prob = drop_abstract_prob

    def __getitem__(self, index):
        if self.drop_abstract_prob is not None and \
                random.random() <= self.drop_abstract_prob:
            paper = {**self.papers[index], "abstract": ""}
            return self._process(paper)
        return self._process(self.papers[index])

    def __len__(self):
        return len(self.papers)

    @classmethod
    def from_file(cls, path, drop_abstract_prob: Optional[float] = None):
        """
        Load dataset from json file

        Args:
            path: the path to the json file
            drop_abstract_prob: The prob of randomly setting abstract string to empty string
        """
        with open(path, "r", encoding="utf-8") as f:
            papers = json.load(f)

        return cls(papers=papers, drop_abstract_prob=drop_abstract_prob)


class IterablePaperDataset(BasePaperDataset, IterableDataset):
    """
    Used for handling MongoDB cursor

    use internal ObjectId (_id) to identify all the papers

    If the paper contains no valid characters (\\w in regexp),
        it will be skipped automatically
    """
    papers: Iterable[Dict[str, Any]]

    def __iter__(self):
        for paper in self.papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")

            if (title or abstract) and re.sub(r"\W", "", paper["abstract"]):
                data = self._process(paper)
                if "_id" in paper:
                    data["_id"] = paper["_id"]
                yield data
