import argparse
import datetime
import hashlib
import logging
import os
from typing import List, Any, Dict, Union, Optional, Tuple

import pymongo
import torch
import transformers
from bson import ObjectId
from transformers import Trainer, TrainingArguments, RobertaTokenizerFast

from modeling_multi_label.config import PRETRAINED_MODEL
from modeling_multi_label.dataset import IterablePaperDataset, MultiLabelDataCollator
from modeling_multi_label.model import MultiLabelModelWithLossFn
from modeling_multi_label.utils import root_dir, timer, nop, sigmoid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s -- %(message)s")
logger = logging.getLogger(__name__)


class UpdateArgumentParser(argparse.ArgumentParser):
    def __init__(self, prog, default_path):
        super().__init__(prog)
        self.add_argument("--debug", action="store_true",
                          help="Debug mode that only runs on 100 papers and "
                               "print additional information (time, true labels, ...)")
        self.add_argument("--model-dir", default=default_path, type=str,
                          help="The directory where the pytorch model is saved.")
        self.add_argument("--batch-size", default=1, type=int, help="The batch size of inputs")
        self.add_argument("--collection", default="entries2", type=str,
                          help="The collection where papers are stored.")
        self.add_argument("--output-collection", default="entries_categories_ml", type=str,
                          help="The output collection where the predicted labels are stored.")


def get_collections(collection_name, output_collection_name, debug):
    try:
        client = pymongo.MongoClient(
            os.getenv("COVID_HOST"),
            username=os.getenv("COVID_USER"),
            password=os.getenv("COVID_PASS"),
            authSource=os.getenv("COVID_DB")
        )
        db = client[os.getenv("COVID_DB")]
    except TypeError as e:
        if debug:
            client = pymongo.MongoClient()
            db = client["LitCOVID"]
        else:
            e.args = (e.args[0] + ". Hint: maybe you forget to set all of the following environment variables: "
                                  "['COVID_HOST', 'COVID_USER', 'COVID_PASS', 'COVID_DB']?",) + e.args[1:]
            raise

    collection = db[collection_name]
    output_collection = db[output_collection_name]

    return collection, output_collection


def compute_model_hash(model_path):
    # use md5sum to identify a checkpoint
    _model_hash = hashlib.md5(open(model_path, "rb").read()).hexdigest()
    return _model_hash


def get_iterable_dataset(collection, output_collection, debug):
    cmd = [
        {"$project": {"_id": 1, "abstract": 1, "title": 1}},
        {"$lookup": {"from": output_collection.name, "localField": "_id",
                     "foreignField": "_id", "as": "predictions"}},
        {"$match": {"predictions": []}},
        {"$project": {"predictions": 0}},
        {"$addFields": {"length": {"$sum": [{"$strLenCP": "$title"}, {"$strLenCP": "$abstract"}]}}},
        {"$sort": {"length": 1}}
    ]
    if debug:
        cmd.insert(3, {"$limit": 100})
    papers = collection.aggregate(cmd)

    # def papers():
    #     i = 0
    #     for paper in collection.find({}, projection=["doi", "abstract", "title"]):
    #         if output_collection.find_one({"_id": paper["_id"]}):
    #             continue
    #         if args.debug:
    #             if i == 100:
    #                 break
    #             i += 1
    #         yield paper

    return IterablePaperDataset(papers)


def write_to_db(ids: List[ObjectId],
                logits: torch.Tensor,
                id2label: Dict[int, str],
                model_hash: str,
                collection,
                output_collection,
                debug: bool):
    for _id, logit in zip(ids, logits):
        if isinstance(logit, torch.Tensor):
            prob = torch.sigmoid(logit)
        else:
            prob = sigmoid(logit)  # np version

        categories = {}
        for label_id, label in id2label.items():
            categories[label] = ((logit[label_id] > 0.).item(), prob[label_id].item())

        output_collection.update_one({"_id": _id}, {"$set": {
            "categories": categories,
            "last_updated": datetime.datetime.now(),
            "model_hash": model_hash,
        }}, upsert=True)
        msg = {"_id": _id, "predicted_labels": [
            label for label_id, label in id2label.items() if logit[label_id] > 0.
        ]}
        if debug:
            msg["true_labels"] = [k for k, v in collection.find_one({"_id": _id}).get("label", {}).items() if v] or None

        logger.info(
            msg=str(msg)
        )


class DBTrainer(Trainer):
    """
    Use #db_id to update the categories in the MongoDB cluster
    """

    def __init__(self, model_hash, collection, output_collection, debug, *args, **kwargs):
        super(DBTrainer, self).__init__(*args, **kwargs)
        self.collection = collection
        self.output_collection = output_collection
        self.model_hash = model_hash
        self.debug = debug

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
        write_to_db(
            ids=ids,
            logits=outputs[1],
            model_hash=self.model_hash,
            id2label=model.config.id2label,
            collection=self.collection,
            output_collection=self.output_collection,
            debug=self.debug,
        )
        return outputs


if __name__ == '__main__':
    cli_args = UpdateArgumentParser(prog="Update DB", default_path=root_dir("bst_model")).parse_args()

    collection_, output_collection_ = get_collections(
        collection_name=cli_args.collection,
        output_collection_name=cli_args.output_collection,
        debug=cli_args.debug,
    )
    _model_hash = compute_model_hash(os.path.join(cli_args.model_dir, "pytorch_model.bin"))

    _model = MultiLabelModelWithLossFn.from_pretrained(
        cli_args.model_dir,
    )

    training_args = TrainingArguments(
        output_dir="../checkpoints/",
        overwrite_output_dir=False,
        do_train=False,
        do_eval=False,
        do_predict=True,
        per_device_eval_batch_size=cli_args.batch_size,
        no_cuda=False,
    )

    dataset = get_iterable_dataset(
        collection=collection_,
        output_collection=output_collection_,
        debug=cli_args.debug,
    )

    data_collator = MultiLabelDataCollator(
        tokenizer=RobertaTokenizerFast.from_pretrained(PRETRAINED_MODEL)
    )

    trainer = DBTrainer(
        collection=collection_,
        output_collection=output_collection_,
        debug=cli_args.debug,
        model=_model,
        model_hash=_model_hash,
        data_collator=data_collator,
        args=training_args
    )

    with timer("DEBUG:") if cli_args.debug else nop():
        pred = trainer.predict(dataset)
