FROM python:3.8.11

WORKDIR /multilabel_classifier
COPY . .
RUN pip install transformers[torch] pymongo

CMD ["python", "/multilabel_classifier/script/update_db.py"]