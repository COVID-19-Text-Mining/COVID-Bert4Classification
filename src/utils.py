"""
Some useful functions
"""

import jinja2
import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def results2html(output):
    """
    >>> import json
    >>> f = json.load(open(r"../results/biomed_roberta-bce_loss-29_Jun/test_result.json", "r", encoding="utf-8"))
    >>> f["results"] = f["results"][:100]
    >>> html_ = results2html(f)
    >>> g = open(r"../results/biomed_roberta-bce_loss-29_Jun/test_result.html", "w", encoding="utf-8")
    >>> g.write(html_) and 1
    1
    >>> g.close()
    """
    def is_correct(logits, labels):
        for logit, label in zip(logits, labels):
            if (logit > 0.) ^ (label == 1.):
                return False
        return True

    file_loader = jinja2.FileSystemLoader("../html_template/")
    env = jinja2.Environment(autoescape=True, loader=file_loader)
    env.filters["is_correct"] = is_correct
    cats = output["cats"]

    template = env.get_template("result.html")
    html = template.render(output=output, index2label={i + 1: name for i, name in enumerate(cats)})
    return html
