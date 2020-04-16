from model import load
from utils import load_config, indexes

import torch
from transformers import BertTokenizer
from typing import Collection, Union, NamedTuple, List, Dict


class Prediction:
    config = load_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    label_tuple = NamedTuple("Label", [("has_label", bool), ("prob", float)])

    model_status: bool = False  # False means unloaded
    bert_tokenizer = None
    model = None

    @classmethod
    def predict(cls, text: Union[str, Collection[str]]) \
            -> Union[Dict[str, label_tuple], List[Dict[str, label_tuple]]]:
        r"""
        predict the label of input text(s)

        :param text: str or collection of str
        :return: if text is a single str, get
                {"label_name": Label("is_True": bool, "probability": float (\in [0, 1]))}

                if text is a collection of str, return
                List({"label_name": Label("has_label": bool, "probability": float (\in [0, 1]))}, ...)
                Same order as input text
        """
        if not cls.model_status:
            cls.bert_tokenizer = BertTokenizer.from_pretrained(cls.config.Dataset.tokenizer_path)

            cls.model, _ = load(cls.config, cls.device, no_file_warning=True)

            cls.model.to(cls.device)
            cls.model.eval()  # set the model to eval mode
            torch.set_grad_enabled(False)  # don't record gradient
            cls.model_status = True

        if isinstance(text, str):
            return cls.predict_one(text)
        else:
            return cls.predict_many(text)

    @classmethod
    def predict_one(cls, text: str) -> Dict[str, label_tuple]:
        return cls.predict_many([text])[0]

    @classmethod
    def predict_many(cls, texts: Collection[str]) -> List[Dict[str, label_tuple]]:
        ids = torch.tensor([
            cls.bert_tokenizer.encode(
                text.strip(),
                add_special_tokens=True,
                max_length=512,
                pad_to_max_length=True
            ) for text in texts]
        ).to(cls.device)  # tokenize
        masks = (ids > 0).int()

        outputs: torch.Tensor = cls.model(
            ids,
            attention_mask=masks
        )

        predictions = []
        for output in outputs:
            prediction = dict()
            for i, label in enumerate(output > cls.config.Predict.positive_threshold):
                prediction[indexes[i]] = cls.label_tuple(bool(label), float(output[i]))
            predictions.append(prediction)

        return predictions


if __name__ == "__main__":
    print(Prediction.predict("Hello, world."))
