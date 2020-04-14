"""
collections of some useful functions and constants
"""

import json
import warnings

# eight labels
cats = {
    "Treatment": 0,
    "Prevention": 1,
    "Mechanism": 2,
    "Diagnosis": 3,
    "General_Info": 4,
    "Transmission": 5,
    "Epidemic_Forecasting": 6,
    "Case_Report": 7
}
indexes = {v: k for k, v in cats.items()}

# default configure
default_config = \
    {
        "IO": {
            "model_dir": "model/"
        },
        "HyperParam": {
            "batch_size": 8,
            "lr": 5e-05,
            "epoch": 50,
            "accumulation_step": 4
        },
        "Loss": {
            "bias": [1, 2]
        },
        "NetWork": {
            "pretrained_model": "sci-bert/",
            "hidden_size": 768,
            "dropout_prob": 0.1,
            "label_num": 8
        },
        "Dataset": {
            "tokenizer_path": "sci-bert/",
            "dataset_path": "rsc/already_annotated.json",
            "text_key": "text",
            "label_key": "cats"
        },
        "Predict": {
            "positive_threshold": 0.8
        }
    }


class ConfigDict(dict):
    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        super(ConfigDict, self).__setitem__(key, value)

    __setattr__ = __setitem__

    def __getattr__(self, item):
        return self[item]

    def save(self, path=r"config/config.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self, f, indent=1)


def load_config(path=r"config/config.json"):
    config = ConfigDict()
    no_config = False
    try:
        with open(path, "r", encoding="utf-8") as f:
            tmp = json.load(f)
    except FileNotFoundError:
        warnings.warn(f"Not file found in {path}! Use default configure instead. The configure file will be created.")
        tmp = default_config.copy()
        no_config = True
    for key, value in tmp.items():
        if isinstance(value, dict):
            value = ConfigDict(value)
        config[key] = value
    if no_config:
        config.save(path)
    return config


def generate_html(datasets, path, hloss=None):
    ht = "<a name=top></a>"

    for i, name in enumerate(datasets):
        ht += f"<a href=\"#{name}\">{name}</a><hr />"
    if hloss is not None:
        ht += f"&nbsp;&nbsp;&nbsp;&nbsp;hamming loss (test set) = {hloss}<br />"
    for i, (name, papers) in enumerate(datasets.items()):
        ht += f"<hr /><a name={name}><h2>{name}</h2></a>"
        for j, each in enumerate(papers):
            ht += f"<p><strong>Text</strong>: {each['text']}</p>"
            if each["cats_manual"]:
                ht += f"<p><strong>Cats_Manual</strong>: {', '.join(each['cats_manual'])}</p>"
            ht += f"<p><strong>Cats_ML</strong>: {', '.join(each['cats_ML'])}</p>"
            ht += "<a href=\"#top\">back to the top</a><br />"
            if j != len(papers) - 1:
                ht += "<hr />"

    with open(path, "w", encoding="utf-8") as f:
        f.write(ht)
