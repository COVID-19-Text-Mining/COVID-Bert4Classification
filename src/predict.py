"""
Predicting scripts
"""

import json

from sklearn.metrics import hamming_loss, label_ranking_loss, label_ranking_average_precision_score
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from transformers import TrainingArguments, IntervalStrategy, SchedulerType, EvalPrediction, Trainer

from config import CATS, PRETRAINED_MODEL
from dataset import PaperDataset
from model import RobertaMultiLabelModel
from utils import sigmoid, results2html

model = RobertaMultiLabelModel.from_pretrained(
    "../checkpoints/checkpoint-4680",
    dropout_prob=0.1,
)
training_args = TrainingArguments(
    output_dir="../checkpoints/",
    overwrite_output_dir=True,
    do_train=False,
    do_eval=False,
    do_predict=True,
    evaluation_strategy=IntervalStrategy.EPOCH,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=8,
    lr_scheduler_type=SchedulerType.LINEAR,
    warmup_steps=2000,
    save_strategy=IntervalStrategy.EPOCH,
    no_cuda=False,
    fp16=False,
    adafactor=True,
    logging_strategy=IntervalStrategy.STEPS,
    logging_steps=10,
    report_to=["tensorboard"],
    save_total_limit=5,
)
test_set = PaperDataset.from_file("../rsc/test_set.json", cats=CATS)


def compute_metrics(eval_prediction: EvalPrediction) -> dict:
    logits = eval_prediction.predictions
    probs = sigmoid(logits)
    predictions = (logits > 0.).astype(int)
    targets = eval_prediction.label_ids
    return {
        "hamming_loss": hamming_loss(targets, predictions),
        "ranking_loss": label_ranking_loss(targets, probs),
        "ranking_average_precision": label_ranking_average_precision_score(targets, probs),
    }


optimizer = Adam(model.parameters(), training_args.learning_rate)
scheduler = MultiStepLR(optimizer, [3000, 7000], gamma=0.1)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
)

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

    output_html = results2html(output=output, cats=CATS)

    with open(r"../results/test_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=1)

    with open(r"../results/test_result.html", "w", encoding="utf-8") as f:
        f.write(output_html)
