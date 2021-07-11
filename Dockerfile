ARG DEVICE=cpu
FROM huggingface/transformers-pytorch-${DEVICE}

ENV PYTHONPATH=$PYTHONPATH:/multilabel_classifier
WORKDIR /multilabel_classifier
COPY . .

RUN pip install pymongo dnspython

ENTRYPOINT ["python3", "/multilabel_classifier/script/update_db.py"]
