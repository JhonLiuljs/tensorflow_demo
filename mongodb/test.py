# coding:utf-8
from bson import json_util as jsonb
import pymongo

client = pymongo.MongoClient("127.0.0.1", 27017)
db = client.stock
Stock = db.Stock
results = list(Stock.find({"type": "sz", "code": "000001"}))
for result in results:
    print(result)

