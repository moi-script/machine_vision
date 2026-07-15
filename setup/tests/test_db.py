import os
import pytest
from app import db


def test_ping_returns_true_when_mongo_up():
    assert db.ping() is True


def test_collections_target_aerosense_db():
    assert db.players().database.name == "aerosense"
    assert db.sessions().database.name == "aerosense"


def test_list_docs_normalizes_id():
    col = db.get_db()["_plan_tmp_test"]
    col.delete_many({})
    col.insert_one({"_id": "x1", "name": "n"})
    out = db.list_docs(col)
    assert out == [{"id": "x1", "name": "n"}]
    col.drop()


def test_list_docs_stringifies_objectid():
    # Docs inserted without an explicit _id get a Mongo ObjectId; list_docs
    # must surface it as a JSON-serializable string (regression: attendance
    # 500'd because a raw ObjectId reached FastAPI's serializer).
    col = db.get_db()["_plan_tmp_oid_test"]
    col.delete_many({})
    col.insert_one({"name": "no-explicit-id"})
    out = db.list_docs(col)
    assert len(out) == 1
    assert isinstance(out[0]["id"], str)
    assert out[0]["name"] == "no-explicit-id"
    col.drop()
