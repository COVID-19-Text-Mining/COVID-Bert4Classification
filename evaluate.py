"""
Evaluate the model
"""

import torch
from model import load
from utils import CONFIG, generate_html, indexes
from dataset import generate_train_set


def evaluate(model=None, tag="", hloss=None, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = CONFIG

    if model is None:
        model, _ = load(config, device, no_file_warning=True)
        del _  # no need for optimizer
        model.to(device)
    if not kwargs:
        papers = {"Papers": generate_train_set(config.Dataset.dataset_path, device, test_portion=-1)}
    else:
        papers = kwargs

    model.eval()
    with torch.no_grad():
        results = {}
        for name, paper_set in papers.items():
            result = []
            for x, mask, y, text in paper_set:
                output: torch.Tensor = model(
                    x.unsqueeze(0),
                    attention_mask=mask.unsqueeze(0)
                ).view(-1)

                prediction = []
                for i, label in enumerate(output > config.Predict.positive_threshold):
                    if label:
                        prediction.append(str((indexes[i], float(output[i]))))

                target = [indexes[i] for i, label in enumerate(y) if label]

                result.append({"text": text, "cats_manual": target, "cats_ML": prediction})
            results[name] = result

    generate_html(results, f"result/evaluation_{tag}.html", hloss)


if __name__ == "__main__":
    evaluate()
