"""
Predicting scripts
"""

import json

from transformers import Trainer, TrainingArguments

from modeling_multi_label.config import CATS, PRETRAINED_MODEL
from modeling_multi_label.dataset import InMemoryPaperDataset
from modeling_multi_label.model import MultiLabelModel
from modeling_multi_label.utils import results2html, root_dir


def test(trainer: Trainer, test_set):
    pred = trainer.predict(test_set)
    logits = pred.predictions.tolist()
    results = []
    for i, (logit, data) in enumerate(zip(logits, test_set)):
        results.append({
            "logits": logit,
            "labels": data["label_ids"].tolist(),
            "text": data["text"],
            **test_set.papers[i],
        })

    output = {
        "cats": CATS,
        "model": PRETRAINED_MODEL,
        "results": results,
    }

    output_html = results2html(output=output)

    with open(root_dir("results", "test_result.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=1)

    with open(root_dir("results", "test_result.html"), "w", encoding="utf-8") as f:
        f.write(output_html)


if __name__ == '__main__':
    model = MultiLabelModel.from_pretrained(
        root_dir("bst_model"),
    )
    _test_set = InMemoryPaperDataset.from_file(
        root_dir(r"rsc", "test_set.json"),
    )
    training_args = TrainingArguments(
        output_dir=root_dir("checkpoints"),
        overwrite_output_dir=False,
        do_train=False,
        do_eval=False,
        do_predict=True,
        per_device_eval_batch_size=8,
        no_cuda=False,
    )

    _trainer = Trainer(model=model, args=training_args)
    test(trainer=_trainer, test_set=_test_set)
    exec(open(root_dir("script", "analysis.py"), encoding="utf-8").read())
