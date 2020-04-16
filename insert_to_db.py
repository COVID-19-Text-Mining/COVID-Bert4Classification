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

    db_host = "dbhost"
    db = "db"
    user = "user_name"
    password = "password"

    client = pymongo.MongoClient(
        db_host,
        username=user,

        password=password,
        authSource=db
    )
    db = client[db]
    input_collection_name = "input_collection_name"
    input_collection = db[input_collection_name]
    output_collection_name = "output_collection_name"
    output_collection = db[output_collection_name]
    output_collection.create_index([('doi', HASHED)])
    cursor = input_collection.find({})
    for paper in cursor:
        exists = output_collection.find_one({'doi': paper["doi"]}) is not None
        if exists:
            continue
        paper_info = {
            "doi": paper["doi"],
            "abstract": paper["abstract"],
            "category": Prediction.predict(paper["abstract"])
        }
        insert_classified_paper(paper_info, output_collection)
