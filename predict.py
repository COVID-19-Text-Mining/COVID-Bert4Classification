from model import load
from utils import load_config, indexes

import torch
from transformers import BertTokenizer
from typing import Collection, Union, NamedTuple, List, Dict

__all__ = ["label_tuple", "Prediction"]

label_tuple = NamedTuple("Label", [("has_label", bool), ("prob", float)])


class Prediction:
    config = load_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    empty_string_result = {cat_name: label_tuple(False, 0.0) for cat_name in indexes.values()}  # no cats

    model_status: bool = False  # False means unloaded
    bert_tokenizer = None
    model = None

    @classmethod
    def predict(cls, text: Union[str, Collection[str]]) \
            -> Union[Dict[str, tuple], List[Dict[str, tuple]]]:
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
    def predict_one(cls, text: str) -> Dict[str, tuple]:
        if text.strip():
            return cls.predict_many([text])[0]
        return cls.empty_string_result.copy()

    @classmethod
    def predict_many(cls, texts: Collection[str]) -> List[Dict[str, tuple]]:
        texts = [text.strip() for text in texts]
        empty_string_masks = [not bool(text) for text in texts]
        ids = torch.tensor([
            cls.bert_tokenizer.encode(
                text,
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
        for output, is_empty_string in zip(outputs, empty_string_masks):

            if is_empty_string:  # deal with empty string
                predictions.append(cls.empty_string_result.copy())
                continue

            prediction = dict()

            for i, label in enumerate(output > cls.config.Predict.positive_threshold):
                prediction[indexes[i]] = tuple(label_tuple(bool(label), float(output[i])))

            predictions.append(prediction)

        return predictions


if __name__ == "__main__":
    print(Prediction.predict("Since the sudden outbreak of coronavirus"
                             " disease 2019 (COVID-19), it has rapidly"
                             " evolved into a momentous global health"
                             " concern. Due to the lack of constructive"
                             " information on the pathogenesis of COVID-19"
                             " and specific treatment, it highlights the"
                             " importance of early diagnosis and timely"
                             " treatment. In this study, 11 key blood indices"
                             " were extracted through random forest algorithm"
                             " to build the final assistant discrimination tool"
                             " from 49 clinical available blood test data"
                             " which were derived by commercial blood test"
                             " equipments. The method presented robust outcome"
                             " to accurately identify COVID-19 from a variety"
                             " of suspected patients with similar CT information"
                             " or similar symptoms, with accuracy of 0.9795 and"
                             " 0.9697 for the cross-validation set and test set,"
                             " respectively. The tool also demonstrated its outstanding"
                             " performance on an external validation set that was"
                             " completely independent of the modeling process, with"
                             " sensitivity, specificity, and overall accuracy of"
                             " 0.9512, 0.9697, and 0.9595, respectively. Besides,"
                             " 24 samples from overseas infected patients with"
                             " COVID-19 were used to make an in-depth clinical"
                             " assessment with accuracy of 0.9167. After multiple"
                             " verification, the reliability and repeatability"
                             " of the tool has been fully evaluated, and it has"
                             " the potential to develop into an emerging technology"
                             " to identify COVID-19 and lower the burden of global"
                             " public health. The proposed tool is well-suited to"
                             " carry out preliminary assessment of suspected patients"
                             " and help them to get timely treatment and quarantine "
                             "suggestion."))
