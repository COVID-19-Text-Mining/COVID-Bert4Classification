import pymongo
from predict import Prediction
import os


if __name__ == '__main__':

    client = pymongo.MongoClient(
        os.getenv("COVID_HOST"),
        username=os.getenv("COVID_USER"),
        password=os.getenv("COVID_PASS"),
        authSource=os.getenv("COVID_DB")
    )
    db = client[os.getenv("COVID_DB")]

    collection = db["entries"]
    output_collection = db["entries_categories_ml"]

    for paper in collection.find({}, projection=["doi", "abstract"]):
        _id = paper["_id"]

        if output_collection.find_one({"_id": _id}) is not None:
            continue

        abstract = paper["abstract"].strip()

        categories = Prediction.predict(abstract)

        output_collection.insert_one(
            {
                "_id": _id,
                "doi": paper["doi"],
                "categories": categories
            }
        )
