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
