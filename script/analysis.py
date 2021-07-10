import json

import numpy as np
from sklearn.metrics import hamming_loss, label_ranking_loss, label_ranking_average_precision_score

from modeling_multi_label.utils import sigmoid

with open(r"../results/biomed_roberta-bce_loss_with_weight-adamw-30_Jun/test_result.json", "r", encoding="utf-8") as f:
    output = json.load(f)

results = output["results"]
logits = []
labels = []

for result in results:
    logits.append(result["logits"])
    labels.append(result["labels"])

logits = np.array(logits)
labels = np.array(labels).astype(int)
probs = sigmoid(logits)
predictions = (logits > 0.).astype(int)

difference = labels - predictions
missing = (difference > 0).astype(np.float32).sum(axis=0)
extra = (difference < 0).astype(np.float32).sum(axis=0)
total = abs(difference).astype(np.float32).sum(axis=0)
label_count = labels.astype(np.float32).sum(axis=0)

missing /= missing.sum()
extra /= extra.sum()
# total /= total.sum()
label_count /= label_count.sum()
ratio = 1 / label_count
ratio /= ratio.sum()

print(np.count_nonzero(difference.sum(axis=1)))
print((ratio * 7.).tolist())
print((missing / extra).tolist())

cats = output["cats"]
print("-" * 72)
print(f"{'Name': <24} {'missing': <8} {'extra': <8} {'total': <8} {'label_count': <16} {'ratio': <8}")
print("-" * 72)
for i in range(7):
    print(
        f"{cats[i]: <24} {missing[i]: <8.2f} {extra[i]: <8.2f} {total[i]: <8.2f} {label_count[i]: <16.2f} {ratio[i]: <8.2f}")
print("-" * 72)
print(
    f"{'All': <24} {missing.sum(): <8.2f} {extra.sum(): <8.2f} {total.sum(): <8.2f} {label_count.sum(): <16.2f} {ratio.sum(): <8.2f}")
print("-" * 72)
print("\nMetrics:")
print(f"{'Accuracy:': <32} {np.count_nonzero(difference.sum(axis=1)) / len(difference): <12.8f}")
print(f"{'Hamming Loss:': <32} {hamming_loss(labels, predictions): <12.8f}")
print(f"{'Ranking Loss:': <32} {label_ranking_loss(labels, probs): <12.8f}")
print(f"{'Ranking Average Precision:': <32} {label_ranking_average_precision_score(labels, probs): <12.8f}")
