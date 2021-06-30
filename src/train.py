"""
Training scripts
"""

from sklearn.metrics import hamming_loss, label_ranking_loss, label_ranking_average_precision_score
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import random_split
from transformers import TrainingArguments, IntervalStrategy, EvalPrediction, Trainer, AdamW

from config import CATS, PRETRAINED_MODEL
from dataset import PaperDataset
from model import RobertaMultiLabelModel
from utils import sigmoid

model = RobertaMultiLabelModel.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=len(CATS),
    dropout_prob=0.1,
    id2label={i: name for i, name in enumerate(CATS)},
    mirror="tuna",
)
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
    num_train_epochs=8,
    warmup_steps=2000,
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
_dataset = PaperDataset.from_file("../rsc/training_set.json", cats=CATS)
training_set, eval_set = random_split(
    _dataset,
    [len(_dataset) - int(len(_dataset) * 0.25), int(len(_dataset) * 0.25)]
)


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
scheduler = MultiStepLR(optimizer, milestones=[3000, 5000], gamma=0.1)

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
    trainer.save_model("../checkpoints/bst_model")
