"""
Dataset class to provide training data
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Union, Iterable, Optional

import numpy as np
import torch
from bson import ObjectId
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from .config import CATS


class BasePaperDataset:
    def __init__(self, papers):
        """
        Args:
            papers: Collection of paper w/o labels
        """
        self.papers = papers  # keeps all the original data

    @staticmethod
    def _process(paper: Dict[str, Any]) -> Dict[str, Union[str, List[int]]]:
        if "label" not in paper:
            # when no label available
            label = None
        else:
            # make sure the order is right
            label = [int(paper["label"][label_name]) for label_name in CATS]

        output = {
            "abstract": paper["abstract"],
            "title": paper["title"],
        }

        if label is not None:
            output["label_ids"] = label

        if "_id" in paper:  # used to identify each document
            output["_id"] = paper["_id"]

        return output


class InMemoryPaperDataset(BasePaperDataset, Dataset):
    """
    Used for training and test
    """
    papers: List[Dict[str, Any]]

    def __getitem__(self, index):
        return self._process(self.papers[index])

    def __len__(self):
        return len(self.papers)

    @classmethod
    def from_file(cls, path):
        """
        Load dataset from json file

        Args:
            path: the path to the json file
        """
        with open(path, "r", encoding="utf-8") as f:
            papers = json.load(f)

        return cls(papers=papers)


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
            paper["title"] = paper.get("title", "") or ""
            paper["abstract"] = paper.get("abstract", "") or ""

            if len(re.split(r"\W+", paper["abstract"])) >= 8:
                data = self._process(paper)
                yield data


@dataclass
class MultiLabelDataCollator:
    """
    Combine data into batch

    When calling this method,
        1. tokenize `title` + `abstract` fields into `input_ids`, `attention_mask`, `token_type_ids`
        2. if data contains `_id` field, combine `_id` of all the data into field `_ids`
        3. if data contains `label_ids` field, combine `label_ids` into `labels`
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = field(default=PaddingStrategy.LONGEST, repr=False)
    return_tensors: str = field(default="pt", repr=False)
    return_attention_mask: bool = field(default=True, repr=False)
    return_token_type_ids: bool = field(default=True, repr=False)
    return_overflowing_tokens: bool = field(default=False, repr=False)
    return_special_tokens_mask: bool = field(default=False, repr=False)
    return_offsets_mapping: bool = field(default=False, repr=False)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Union[np.ndarray, torch.Tensor, List[ObjectId]]]:
        """
        Preserve the papers' ObjectId during batching
        """
        batch = {}
        first = features[0]

        if "_id" in first:
            batch["_ids"] = [f["_id"] for f in features]

        if "label_ids" in first:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        output = self.tokenizer(
            [f.pop("abstract") for f in features],
            text_pair=[f.pop("title") for f in features],
            add_special_tokens=True,
            padding=self.padding,
            truncation="longest_first",
            return_tensors=self.return_tensors,
            return_attention_mask=self.return_attention_mask,
            return_token_type_ids=self.return_token_type_ids,
            return_overflowing_tokens=self.return_overflowing_tokens,
            return_special_tokens_mask=self.return_special_tokens_mask,
            return_offsets_mapping=self.return_offsets_mapping,
        )

        if self.return_tensors == "np":
            output = {k: v.astype(np.int64) for k, v in output.items()}

        batch.update(output)

        return batch
