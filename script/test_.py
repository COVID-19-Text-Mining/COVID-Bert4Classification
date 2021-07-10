"""
Predicting scripts
"""

import json

from transformers import Trainer

from modeling_multi_label.config import CATS, PRETRAINED_MODEL
from modeling_multi_label.dataset import InMemoryPaperDataset
from modeling_multi_label.model import MultiLabelModel
from modeling_multi_label.utils import results2html


def test(trainer: Trainer, test_set):
    pred = trainer.predict(test_set)
    logits = pred.predictions.tolist()
    results = []
    for i, (logit, data) in enumerate(zip(logits, test_set)):
        results.append({
            "logits": logit,
            "labels": data["labels"].tolist(),
            "text": data["text"],
            **data["misc"],
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


if __name__ == '__main__':
    model = MultiLabelModel.from_pretrained(
        "../bst_model",
    )
    _test_set = InMemoryPaperDataset.from_file("../rsc/test_set.json", text_key="abstract", label_key="label")

    _trainer = Trainer(model=model)
    test(trainer=_trainer, test_set=_test_set)
    exec(open(r"./analysis.py", encoding="utf-8").read())