# COVID-Bert4Classification
Use BERT to solve a multi-label text classification task.

## Architecture
The model is simply a BERT model followed by a linear classifier, using BP-MLL function as loss function.

## Train the model
Simply run the command as you have downloaded the pretrained BERT.
```sh
python train.py &
tail -f logger.log  # read the log file
```

## Evaluate the model
```sh
python evaluate.py
cd result  # the html form result are put in result/
```

## Predict labels of the texts
```sh
python predict.py  # for test
```
To use in a Python script, call
```python
>>> from .predict import Prediction

>>> text = "Some text here"
>>> Prediction.predict(text)
{'Treatment': Label('has_label': True, 'prob': 0.98), ...}

>>> texts = ["many", "texts", "here"]
>>> Prediction.predict(text)
[{'Treatment': Label('has_label': True, 'prob': 0.98), ...}, ...]
```

## Explanation to the parameters in `config/config.json`
### IO
Save and load the trained model, used in `load` function in `model.py`
- `model_dir`: point to the directory where the model (not pretrained model) is saved

### HyperParam
Hyper parameters that control the whole training process, used in `train.py`
- `batch_size`: the number of samples that feed into the model at a time (should not be too big, or the model will need too much memory)
- `lr`: learning rate (5e-5, 3e-5, 1e-5 or something like that)
- `epoch`: number of iteration on the whole training set (depend on the size of the training set)
- `accumulation_step`: Here we use gradient accumulation technique to get the performance of a larger batch size (the actual batch size is approximately `batch_size * accumulation_step`).

### Loss
Parameters for the loss function (here we use BP-MLL loss function), used for initilizing bp_mll loss function in `bp_mll.py`
- `bias`: the weight of positive label and negative label (refer to [this paper](https://ieeexplore.ieee.org/document/1683770))

### Network
Parameters used for initializing the neural network, used in `model.py`
- `pretrained_model`: point to the directory containing the pretained bert model (here we use [SciBert](https://github.com/allenai/scibert) by Allen AI, need to be downloaded MANUALLY)
- `hidden_size`: the size of the pretrained model's hidden size (768 here)
- `dropout_prob`: the probability for dropout layer to drop a element in the input tensor
- `label_num`: the total number of labels

### Dataset
Used in `dataset.py` for loading the dataset from file
- `tokenizer_path`: For pretrained model, keep it the same as `IO.model_dir`
- `dataset_path`: point to the `json` file where annotated data is stored
- `text_key`: key of the text for each entry
- `label_key`: key of the labels for each entry

### Predict
Used for evaluation and prediction
- `position_threshold`: minumum output for an output to be considered as a positive output (`predicted_label = output_probility > position_threshold ? 1 : 0`)

## TODO
- Can only reload the model on a single GPU server.
- Unable to load the optimizer for continuing training.
- Better performance for training on multiple GPUs.
