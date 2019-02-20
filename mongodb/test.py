# coding:utf-8
from bson import json_util as jsonb
import pymongo
client= pymongo.MongoClient("127.0.0.1",27017)
db=client.myinfo
print(jsonb.dumps(list(db.user.find({"name":"wu"}))))