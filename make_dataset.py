import pymongo
import json
from bson.json_util import default

db = pymongo.MongoClient()["LitCOVID"]
collection = db["dev"]

all_labels = set()

for paper in collection.find():
    for label in paper["label"].split(";"):
        all_labels.add(label)

all_labels = list(all_labels)
print([label.replace(" ", "_") for label in all_labels])

all_papers = []

for paper in collection.find():
    labels = paper["label"].split(";")
    paper["label"] = {label.replace(" ", "_"): int(label in labels) for label in all_labels}
    all_papers.append(paper)

with open("rsc/test_set.json", "w", encoding="utf-8") as f:
    json.dump(all_papers, f, indent=1, default=default)
