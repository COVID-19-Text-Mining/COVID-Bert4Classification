import json
import numpy as np

with open(r"../results/biomed_roberta-bce_loss-29_Jun/test_result.json", "r", encoding="utf-8") as f:
    output = json.load(f)

results = output["results"]
logits = []
labels = []

for result in results:
    logits.append(result["logits"])
    labels.append(result["labels"])

logits = np.array(logits)
labels = np.array(labels).astype(int)
predictions = (logits > 0.).astype(int)

difference = labels - predictions
missing = (difference > 0).astype(np.float32).sum(axis=0)
extra = (difference < 0).astype(np.float32).sum(axis=0)
total = abs(difference).astype(np.float32).sum(axis=0)
label_count = labels.astype(np.float32).sum(axis=0)

missing /= missing.sum()
extra /= extra.sum()
total /= total.sum()
label_count /= label_count.sum()
ratio = 1 / label_count
ratio /= ratio.sum()

cats = output["cats"]
print("-" * 72)
print(f"{'Name': <24} {'missing': <8} {'extra': <8} {'total': <8} {'label_count': <16} {'ratio': <8}")
print("-" * 72)
for i in range(7):
    print(f"{cats[i]: <24} {missing[i]: <8.2f} {extra[i]: <8.2f} {total[i]: <8.2f} {label_count[i]: <16.2f} {ratio[i]: <8.2f}")
print("-" * 72)
print(f"{'All': <24} {missing.sum(): <8.2f} {extra.sum(): <8.2f} {total.sum(): <8.2f} {label_count.sum(): <16.2f} {ratio.sum(): <8.2f}")
print("-" * 72)
print((ratio * 7.).tolist())
print((missing/extra).tolist())