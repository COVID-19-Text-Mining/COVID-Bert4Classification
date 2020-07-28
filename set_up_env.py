"""
Automatically download the models and HuggingFace's transformers package
"""

import os
import subprocess
import warnings
from importlib.util import find_spec

pretrained_model = "scibert_scivocab_uncased"
model = "model_best.pth"


if find_spec("transformers") is None:
    warnings.warn("Not Transformer module found, download and install it automatically")
    subprocess.run(["make", "install"], check=True)

if not os.path.isdir(pretrained_model):
    warnings.warn("Not scibert pretrained model found, download it automatically")
    subprocess.run(["make", "scibert_scivocab_uncased"], check=True)

print("Done!")
