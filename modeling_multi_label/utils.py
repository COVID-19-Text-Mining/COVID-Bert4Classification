"""
Some useful functions
"""
import os

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


def get_abs_path(file: str, *rel_paths) -> str:
    """
    Helper function to locate a path

    Args:
        file: reference file path
        rel_paths: relative path which refers to `file`
    """
    return os.path.join(
        os.path.dirname(os.path.abspath(file)), *rel_paths
    )


class DirObj:
    """
    Manage some special dir path

    Examples:
        >>> data_dir = DirObj("path", "to", "data")
        >>> data_dir("some", "dataset")
        'path/to/data/some/dataset'
        >>> data_dir.update_dir("another", "path")
        >>> data_dir("another", "dataset")
        'another/path/another/dataset'
    """

    def __init__(self, *default_dir: str):
        self._dir = os.path.join(*default_dir)

    def __call__(self, *file_name: str):
        if not os.path.exists(self._dir):
            os.mkdir(self._dir, mode=0o775)
        return os.path.join(self._dir, *file_name)

    def update_dir(self, *_dir: str):
        self._dir = os.path.join(*_dir)


script_dir = DirObj(get_abs_path(__file__, "..", "script"))
data_dir = DirObj(get_abs_path(__file__, "..", "rsc"))
cpt_dir = DirObj(get_abs_path(__file__, "..", "checkpoints"))
root_dir = DirObj(get_abs_path(__file__, ".."))
