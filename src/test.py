"""
Predicting scripts
"""

import json

from transformers import Trainer

from config import CATS, PRETRAINED_MODEL
from dataset import PaperDataset
from model import RobertaMultiLabelModel
from utils import results2html

model = RobertaMultiLabelModel.from_pretrained(
    "../checkpoints/bst_model",
    dropout_prob=0.1,
)
test_set = PaperDataset.from_file("../rsc/test_set.json", cats=CATS)

trainer = Trainer(model=model)

if __name__ == '__main__':
    pred = trainer.predict(test_set)
    logits = pred.predictions.tolist()
    results = []
    for i, (logit, data) in enumerate(zip(logits, test_set)):
        results.append({
            "logits": logit,
            "labels": data["labels"].tolist(),
            "text": data["text"],
            **test_set.metadata[i],
        })

    output = {
        "cats": CATS,
        "model": PRETRAINED_MODEL,
        "results": results,
    }

    output_html = results2html(output=output)

    with open(r"../results/test_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=1)

    with open(r"../results/test_result.html", "w", encoding="utf-8") as f:
        f.write(output_html)
