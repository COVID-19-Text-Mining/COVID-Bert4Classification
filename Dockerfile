ARG DEVICE=cpu
FROM huggingface/transformers-pytorch-${DEVICE}

ENV PYTHONPATH=$PYTHONPATH:/multilabel_classifier
WORKDIR /multilabel_classifier
COPY . .

RUN pip install pymongo dnspython dataclass

CMD "--batch-size 1 --collection entries2 --output-collection entries_categories_ml"
ENTRYPOINT ["python3", "/multilabel_classifier/script/update_db.py"]
