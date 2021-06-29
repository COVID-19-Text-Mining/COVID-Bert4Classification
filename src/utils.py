"""
Some useful functions
"""
import json

import jinja2
import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def results2html(output, cats):
    template = jinja2.Template(open("../html_template/result.html", "r", encoding="utf-8").read())
    html = template.render(output=output, index2label={i+1: name for i, name in enumerate(cats)})
    return html
