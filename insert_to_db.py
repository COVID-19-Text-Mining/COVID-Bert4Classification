import pymongo
import datetime


def insert_classified_paper(paper, collection):
    paper['Last_Updated'] = datetime.datetime.now()
    collection.update(
        {'doi': paper['doi']},
        paper,
        upsert=True
    )


if __name__ == '__main__':

    client = pymongo.MongoClient(
        os.getenv("COVID_HOST"),
        username=os.getenv("COVID_USER"),
        password=os.getenv("COVID_PASS"),
        authSource=os.getenv("COVID_DB")
    )
    db = client[os.getenv("COVID_DB")]

    collection = db["entries"]

    # TODO: change this target collection
    target_collection = collection

    for paper in collection.find():
        exists = output_collection.find_one({'doi': paper["doi"]}) is not None

        abstract = paper["abstract"].strip()

        if exists:  # skip when paper have no abstract OR already in the database
            continue

        paper_info = {
            "doi": paper["doi"],
            "abstract": paper["abstract"],
            "categories": Prediction.predict(paper["abstract"])  # NamedTuple
        }

        insert_classified_paper(paper_info, output_collection)
