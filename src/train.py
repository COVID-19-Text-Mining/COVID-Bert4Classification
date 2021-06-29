"""
Training scripts
"""

from sklearn.metrics import hamming_loss, label_ranking_loss, label_ranking_average_precision_score
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import random_split
from transformers import TrainingArguments, IntervalStrategy, SchedulerType, EvalPrediction, Trainer

from dataset import PaperDataset
from model import RobertaMultiLabelModel
from config import CATS, PRETRAINED_MODEL
from utils import sigmoid

if __name__ == '__main__':
    model = RobertaMultiLabelModel.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=len(CATS),
        dropout_prob=0.1,
    )
    training_args = TrainingArguments(
        output_dir="../checkpoints/",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=32,
        learning_rate=5e-5,
        num_train_epochs=8,
        lr_scheduler_type=SchedulerType.LINEAR,
        warmup_ratio=0.1,
        save_strategy=IntervalStrategy.EPOCH,
        no_cuda=False,
        fp16=False,
        adafactor=True,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=10,
        report_to=["tensorboard"],
        save_total_limit=5,
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


    optimizer = Adam(model.parameters(), training_args.learning_rate)
    scheduler = MultiStepLR(optimizer, [500, 2000], gamma=0.1)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=eval_set,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()
