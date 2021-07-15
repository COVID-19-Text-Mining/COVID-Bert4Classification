# Multi-label Text Classification

Default Pretrained LM: [`allenai/biomed_roberta_base`](https://huggingface.co/allenai/biomed_roberta_base)

Default Loss Function: `BCELoss`

Default Optimizer: `AdamW` 

Default Training Set: 24960 manually reviewed abstracts from LitCOVID

## Result on Test Set
All the models are tested on the same [dev set](https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/BC7-LitCovid-Dev.csv) provided by LitCOVID.

| Trial Name | Hamming Loss | Ranking Loss | Label Ranking Average Precision | Error Rate<sup>*</sup> |Result Links|
| :-------  |      :------:     |     :-----:      |    :------:           | :---:| :----:|
|Old Version (SciBert, 200+ annotated abstracts as training set)|0.114|0.118|0.818|0.410|[1](results/scibert-bce_loss-adamw-small_dataset-2_July)|
|[BP MLL Loss](https://ieeexplore.ieee.org/document/1683770)|0.104|0.0203|0.962|0.553|[2](results/biomed_roberta-bp_mll_loss-adamw-1_July)|
|[Asymmetric Loss](https://arxiv.org/abs/2009.14119)|0.0461|0.0227|0.959|0.210|[3](results/biomed_roberta-as_loss-adamw_1_3_1e-1_-1_July)|
|Default Setting|0.0441|0.0206|0.961|0.195|[4](results/biomed_roberta-bce_loss-29_Jun)|
| Weighted BCE Loss | 0.0428 | 0.0232 | 0.958 |0.188|[5](results/biomed_roberta-bce_loss_with_weight-adamw-30_Jun)|
| Abstract + Title | **0.0386** | **0.0190** | **0.966** | **0.172** | [6](results/biomed_roberta-bce_loss_with_weight-adamw-full-has_title-11_July) |
---
\* Error rate is the ratio of wrongly-predicted entries (the predicted labels are not exactly the same as the true labels) against the whole test set.

## Inference Acceleration
All the test are run on the same PC (No CUDA).

| Optimization Method | Time (sec, 100 papers) |
| :---- | :---: |
| Unoptimized | 113.5 |
| ONNX optimized + Int8 Quantized | 91.6 |
| Dynamic Padding | 52.3 |
| ONNX optimized + Int8 Quantized + Dynamic Padding | **45.1** |
## Deployment
### Deploy with docker
We use docker to deploy our classification model. To use our model to make predictions, run
```shell
## This script will read new documents from MongoDB, 
## make predictions and write the predicted categories
## to database

# CPU version
docker run --rm \
  -n multilabel-classifier-cpu \
  -e COVID_HOST=$COVID_HOST \
  -e COVID_USER=$COVID_USER \
  -e COVID_PASS=$COVID_PASS \
  -e COVID_DB=$COVID_DB \
  idocx/multilabel-classifier:cpu \
  --batch-size 1
```

```shell
# GPU version
docker run --rm --gpus all \
  -n multilabel-classifier-gpu \
  -e COVID_HOST=$COVID_HOST \
  -e COVID_USER=$COVID_USER \
  -e COVID_PASS=$COVID_PASS \
  -e COVID_DB=$COVID_DB \
  idocx/multilabel-classifier:gpu \
  --batch-size 16
```

#### Build (Optional)

```shell
# Download model checkpoint
wget https://www.ocf.berkeley.edu/~yuxingfei/models/model.tar.gz \
  && tar -zxvf model.tar.gz && rm model.tar.gz

# Build docker container (CPU version)
docker build . -t idocx/multilabel-classifier:cpu \
  --build-arg DEVICE=cpu

# Build docker container (GPU version)
docker build . -t idocx/multilabel-classifier:gpu \
  --build-arg DEVICE=gpu
```

### Deploy directly
```shell
pip install -r requirements.txt

# Download model checkpoint
wget https://www.ocf.berkeley.edu/~yuxingfei/models/model.tar.gz \
  && tar -zxvf model.tar.gz && rm model.tar.gz

# remember to set DB env variable before runing the script
export PYTHONPATH=$PYTHONPATH:$(pwd)
python script/upadte_db.py --batch-size 1
```