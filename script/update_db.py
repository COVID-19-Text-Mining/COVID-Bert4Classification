import os
from typing import List, Any, Dict, Union, Optional, Tuple

import pymongo
import torch
import transformers
from bson import ObjectId
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator

from modeling_multi_label.dataset import IterablePaperDataset
from modeling_multi_label.model import MultiLabelModel
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("db_host", default="127.0.0.1")
parser.add_argument("db_user", default="")
parser.add_argument("db_password", default="")
parser.add_argument("db_name", default="LitCOVID")

client = pymongo.MongoClient(
    os.getenv("COVID_HOST"),
    username=os.getenv("COVID_USER"),
    password=os.getenv("COVID_PASS"),
    authSource=os.getenv("COVID_DB")
)
db = client[os.getenv("COVID_DB")]

collection = db["entries2"]
output_collection = db["entries_categories_ml"]

papers = collection.aggregate([
    {"$project": {"_id": 1, "abstract": 1}},
    {"$lookup": {"from": output_collection.name, "localField": "_id",
                 "foreignField": "_id", "as": "predictions"}},
    {"$match": {"predictions": []}},
    {"$project": {"predictions": 0}},
])


def _data_collator(features: List[Dict[str, Any]]) -> Dict[str, Union[torch.Tensor, List[ObjectId]]]:
    """
    Preserve the papers' ObjectId during batching
    """
    first = features[0]
    if "_id" in first:
        ids = [f.pop("_id") for f in features]
    else:
        ids = None

    batch = default_data_collator(features=features)
    batch["_ids"] = ids
    return batch


class DBTrainer(Trainer):
    """
    Use #db_id to update the categories in the MongoDB cluster
    """

    @staticmethod
    def write_to_db(ids: List[ObjectId],
                    logits: torch.Tensor,
                    id2label: Dict[int, str]):
        for _id, logit in zip(ids, logits):
            prob = torch.sigmoid(logit)
            categories = {}
            for label_id, label in id2label.items():
                categories[label] = ((logit[label_id] > 0.).item(), prob[label_id].item())
            output_collection.update_one({"_id": _id}, {"$set": {"categories": categories}}, upsert=True)

    def prediction_step(
            self,
            model: transformers.PreTrainedModel,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        ids = inputs.pop("_ids")
        outputs = super(DBTrainer, self).prediction_step(model=model, inputs=inputs,
                                                         prediction_loss_only=prediction_loss_only,
                                                         ignore_keys=ignore_keys)
        self.write_to_db(ids=ids, logits=outputs[1], id2label=model.config.id2label)
        return outputs


if __name__ == '__main__':
    _model = MultiLabelModel.from_pretrained(
        "../bst_model",
    )
    training_args = TrainingArguments(
        output_dir="../checkpoints/",
        overwrite_output_dir=False,
        do_train=False,
        do_eval=False,
        do_predict=True,
        per_device_eval_batch_size=8,
        no_cuda=False,
    )
    trainer = DBTrainer(model=_model, args=training_args, data_collator=_data_collator)

    pred = trainer.predict(IterablePaperDataset(papers, text_key="abstract", label_key=None))
