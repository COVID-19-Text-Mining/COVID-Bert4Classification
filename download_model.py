"""
Automatically download the model
"""

import os
import subprocess
import warnings
from importlib.util import find_spec

model_name = "scibert_scivocab_uncased"

if find_spec("transformers") is None:
    warnings.warn("Not Transformer module found, download and install it automatically")
    subprocess.run(["make", "install"], check=True)

if not os.path.isdir(model_name):
    warnings.warn("Not scibert pretrained model found, download and install it automatically")
    subprocess.run(["make", "scibert_scivocab_uncased"], check=True)

print("Done!")
