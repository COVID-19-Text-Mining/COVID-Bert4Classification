"""
Automatically download the model
"""

import requests
import os
from tqdm import tqdm
import tarfile

model_name = "scibert_scivocab_uncased"
url = r"https://s3-us-west-2.amazonaws.com/" \
      r"ai2-s2-research/scibert/huggingface_pytorch/" \
      r"scibert_scivocab_uncased.tar"

if not os.path.isdir(model_name):
    if not os.path.isfile(model_name+".tar"):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        t = tqdm(total=total_size, unit='B', unit_scale=True)
        with open(model_name+".tar", 'wb') as f:
            for data in r.iter_content(1024):
                t.update(len(data))
                f.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            raise Exception("ERROR, fail to download ")
        tar = tarfile.open(model_name+".tar", mode="r|*")
        tar.extractall(".")
        os.remove(model_name+".tar")
