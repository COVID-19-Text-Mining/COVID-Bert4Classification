"""
Training scripts
"""

from sklearn.metrics import hamming_loss, label_ranking_loss, label_ranking_average_precision_score
from transformers import TrainingArguments, IntervalStrategy, EvalPrediction, Trainer, AdamW, \
    get_cosine_schedule_with_warmup

from modeling_multi_label.config import CATS, PRETRAINED_MODEL
from modeling_multi_label.dataset import InMemoryPaperDataset
from modeling_multi_label.model import MultiLabelModel
from modeling_multi_label.utils import sigmoid
from test_ import test

training_args = TrainingArguments(
    output_dir="../checkpoints/",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=False,
    evaluation_strategy=IntervalStrategy.EPOCH,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=12,
    warmup_steps=1000,
    metric_for_best_model="hamming_loss",
    greater_is_better=False,
    save_strategy=IntervalStrategy.EPOCH,
    no_cuda=False,
    fp16=False,
    adafactor=False,
    logging_strategy=IntervalStrategy.STEPS,
    logging_steps=10,
    report_to=["tensorboard"],
    load_best_model_at_end=True
)

model = MultiLabelModel.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=len(CATS),
    id2label={i: name for i, name in enumerate(CATS)},
    label2id={name: i for i, name in enumerate(CATS)},
    mirror="tuna",
)

training_set = InMemoryPaperDataset.from_file("../rsc/training_set.json", text_key="abstract", label_key="label")
eval_set = InMemoryPaperDataset.from_file("../rsc/test_set.json", text_key="abstract", label_key="label")


# training_set, eval_set = random_split(
#     _dataset,
#     [len(_dataset) - int(len(_dataset) * 0.25), int(len(_dataset) * 0.25)]
# )


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


optimizer = AdamW(
    model.parameters(),
    lr=training_args.learning_rate,
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=(len(training_set) * training_args.num_train_epochs) //
                       (training_args.per_device_train_batch_size * training_args.n_gpu *
                        training_args.gradient_accumulation_steps),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_set,
    eval_dataset=eval_set,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model("../bst_model")
    test(trainer=trainer, test_set=eval_set)