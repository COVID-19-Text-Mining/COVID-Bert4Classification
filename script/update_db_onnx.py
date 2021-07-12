import datetime
import hashlib
import logging
import os
from typing import List, Dict, Any, Union

import pymongo
import numpy as np
import torch
from bson import ObjectId
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from torch.utils.data import DataLoader
from transformers import default_data_collator, RobertaConfig

from modeling_multi_label.dataset import IterablePaperDataset
from modeling_multi_label.utils import root_dir, sigmoid

DEBUG_MODE = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s -- %(message)s")
logger = logging.getLogger(__name__)


# use md5sum to identify a checkpoint
model_hash = hashlib.md5(
    open(root_dir("bst_model_onnx", "model-optimized-quantized.onnx"), "rb").read()
).hexdigest()

try:
    client = pymongo.MongoClient(
        os.getenv("COVID_HOST"),
        username=os.getenv("COVID_USER"),
        password=os.getenv("COVID_PASS"),
        authSource=os.getenv("COVID_DB")
    )
    db = client[os.getenv("COVID_DB")]
except TypeError as e:
    if DEBUG_MODE:
        client = pymongo.MongoClient()
        db = client["LitCOVID"]
    else:
        e.args = (e.args[0] + ". Hint: maybe you forget to set all of the following environment variables: "
                              "['COVID_HOST', 'COVID_USER', 'COVID_PASS', 'COVID_DB']?",) + e.args[1:]
        raise

collection = db["entries2"]
output_collection = db["entries_categories_ml"]


def create_model_for_provider(model_path: str, provider: str = "CPUExecutionProvider") -> InferenceSession:
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session


def _data_collator(features: List[Dict[str, Any]]) -> Dict[str, Union[np.ndarray, List[ObjectId]]]:
    """
    Preserve the papers' ObjectId during batching
    """
    first = features[0]
    if "_id" in first:
        ids = [f.pop("_id") for f in features]
    else:
        ids = None

    batch = default_data_collator(features=features)
    batch = {k: v.cpu().detach().numpy() for k, v in batch.items()}
    batch["_ids"] = ids
    return batch


def get_dataloader():
    def papers():
        for paper in collection.find({}, projection=["doi", "abstract", "title"]):
            if output_collection.find_one({"_id": paper["_id"]}):
                continue
            yield paper

    dataset = IterablePaperDataset(papers())

    return DataLoader(dataset=dataset, batch_size=8, collate_fn=_data_collator)


def write_to_db(ids: List[ObjectId],
                logits: torch.Tensor,
                id2label: Dict[int, str]):
    for _id, logit in zip(ids, logits):
        prob = sigmoid(logit)
        categories = {}
        for label_id, label in id2label.items():
            categories[label] = ((logit[label_id] > 0.).item(), prob[label_id].item())

        output_collection.update_one({"_id": _id}, {"$set": {
            "categories": categories,
            "last_updated": datetime.datetime.now(),
            "model_hash": model_hash,
        }}, upsert=True)

        msg = {
            "_id": _id,
            "predicted_labels": [label for label_id, label in id2label.items() if logit[label_id] > 0.],
        }

        if DEBUG_MODE:
            msg["true_labels"] = [k for k, v in collection.find_one({"_id": _id})["label"].items() if v]

        logger.info(
            msg=str(msg)
        )


if __name__ == '__main__':
    model = create_model_for_provider(
        model_path=root_dir("bst_model_onnx", "model-optimized.onnx"),
        provider="CPUExecutionProvider"
    )

    for batch in get_dataloader():
        ids = batch.pop("_ids")
        logits = model.run(None, batch)[0]
        write_to_db(ids=ids, logits=logits,
                    id2label=RobertaConfig.from_pretrained(root_dir("bst_model", "config.json")).id2label)
