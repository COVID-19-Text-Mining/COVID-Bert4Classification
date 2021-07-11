import argparse
import datetime
import hashlib
import logging
from typing import List, Any, Dict, Union, Optional, Tuple

import pymongo
import torch
import transformers
from bson import ObjectId
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator

from modeling_multi_label.dataset import IterablePaperDataset
from modeling_multi_label.model import MultiLabelModel
from modeling_multi_label.utils import root_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s -- %(message)s")
logger = logging.getLogger(__name__)

# handle cli options
parser = argparse.ArgumentParser()
parser.add_argument("--mongo-url", help="The URL of the target db.",
                    default=argparse.SUPPRESS, type=str)
parser.add_argument("--mongo-db", help="The name of the database where all the papers are stored.",
                    required=True, type=str)
args = parser.parse_args()

# initialize MongoDB connection
client = pymongo.MongoClient(
    args.mongo_url,
)
db = client[args.mongo_db]
collection = db["entries2"]
output_collection = db["entries_categories_ml"]

# use md5sum to identify a checkpoint
model_hash = hashlib.md5(
    open(root_dir("bst_model", "pytorch_model.bin"), "rb").read()
).hexdigest()


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

            output_collection.update_one({"_id": _id}, {"$set": {
                "categories": categories,
                "last_updated": datetime.datetime.now(),
                "model_hash": model_hash,
            }}, upsert=True)

            logger.info(
                msg="{{#DB_ID: {_id}, LABELS: {labels}}}".format(
                    _id=_id,
                    labels=[label for label_id, label in id2label.items() if logit[label_id] > 0.])
            )

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
        root_dir("bst_model"),
    )

    papers = collection.aggregate([
        {"$project": {"_id": 1, "abstract": 1}},
        {"$lookup": {"from": output_collection.name, "localField": "_id",
                     "foreignField": "_id", "as": "predictions"}},
        {"$match": {"predictions": []}},
        {"$project": {"predictions": 0}},
    ])

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