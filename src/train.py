from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import random_split
from transformers import TrainingArguments, IntervalStrategy, SchedulerType, EvalPrediction, Trainer
import numpy as np
from dataset import PaperDataset
from model import RobertaMultiLabelModel

cats = (
    "Case_Report",
    "Treatment",
    "Prevention",
    "Diagnosis",
    "Transmission",
    "Epidemic_Forecasting",
    "Mechanism"
)

if __name__ == '__main__':
    model = RobertaMultiLabelModel.from_pretrained(
        "allenai/biomed_roberta_base",
        num_labels=len(cats),
        dropout_prob=0.1,
    )
    training_args = TrainingArguments(
        output_dir="../checkpoints/",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=8,
        per_gpu_eval_batch_size=8,
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
        report_to=["tensorboard"],
    )
    _dataset = PaperDataset.from_file("../rsc/training_set.json", cats=cats)
    training_set, eval_set = random_split(
        _dataset,
        [len(_dataset) - int(len(_dataset) * 0.25), int(len(_dataset) * 0.25)]
    )


    def compute_metrics(eval_prediction: EvalPrediction) -> dict:
        predictions = eval_prediction.predictions
        targets = eval_prediction.label_ids
        p, q = targets.shape
        hamming_loss = 1. / (p * q) * \
            (predictions.astype(np.bool_) ^ targets.astype(np.bool_)).astype(np.float32).sum()
        return {
            "hamming_loss": hamming_loss,
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
