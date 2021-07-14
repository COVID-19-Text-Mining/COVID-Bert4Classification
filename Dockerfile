ARG DEVICE=cpu
FROM huggingface/transformers-pytorch-${DEVICE}

ENV PYTHONPATH=$PYTHONPATH:/multilabel_classifier
WORKDIR /multilabel_classifier
COPY . .
ADD https://www.ocf.berkeley.edu/~yuxingfei/models/model.tar.gz .

RUN pip3 install -r requirements.txt && rm model.tar.gz

ENTRYPOINT ["python3", "/multilabel_classifier/script/update_db.py"]
