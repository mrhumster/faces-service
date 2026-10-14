"""Person -> videos search: endpoint wiring and threshold logic, stubs only.

No database is touched: the endpoint is exercised against a monkeypatched
``db.get_streams_for_clusters`` and the query builder against a fake cursor that
records the SQL parameters.
"""
import contextlib
import os
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db  # noqa: E402
from app.routes import reader  # noqa: E402


class _Auth:
    def __init__(self, user_id="owner-1", role="user"):
        self.user_id = user_id
        self.role = role


class _Cursor:
    def __init__(self, rows, captured):
        self._rows = rows
        self._captured = captured

    def execute(self, sql, params):
        self._captured["sql"] = sql
        self._captured["params"] = params

    def fetchall(self):
        return self._rows


@contextlib.contextmanager
def _fake_conn(rows, captured):
    yield _Cursor(rows, captured)


class TestSearchVideosEndpoint:
    def test_dedupes_and_scopes_to_owner(self, monkeypatch):
        calls = {}

        def fake(owner_id, cluster_ids, mode):
            calls["owner_id"] = owner_id
            calls["cluster_ids"] = cluster_ids
            calls["mode"] = mode
            return [{"stream_id": "s1", "persons": 2, "frames": 3}]

        monkeypatch.setattr(reader.db, "get_streams_for_clusters", fake)
        body = reader.SearchVideosBody(cluster_ids=["a", "b", "a"], mode="all")
        res = reader.search_videos_by_faces(body, _Auth())

        assert calls == {
            "owner_id": "owner-1",
            "cluster_ids": ["a", "b"],
            "mode": "all",
        }
        assert res == {"streams": [{"stream_id": "s1", "persons": 2, "frames": 3}], "total": 1, "mode": "all"}

    def test_empty_is_rejected(self):
        with pytest.raises(ValidationError):
            reader.SearchVideosBody(cluster_ids=[], mode="all")

    def test_too_many_is_rejected(self):
        with pytest.raises(ValidationError):
            reader.SearchVideosBody(
                cluster_ids=[str(i) for i in range(101)], mode="all"
            )

    def test_bad_mode_is_rejected(self):
        with pytest.raises(ValidationError):
            reader.SearchVideosBody(cluster_ids=["a"], mode="both")


class TestGetStreamsForClusters:
    def test_all_uses_full_threshold(self, monkeypatch):
        captured = {}
        rows = [{"stream_id": "s1", "persons": 2, "frames": 4}]
        monkeypatch.setattr(db, "conn", lambda: _fake_conn(rows, captured))

        out = db.get_streams_for_clusters("o", ["a", "b"], "all")

        assert captured["params"] == ("o", ["a", "b"], 2)
        assert out == [{"stream_id": "s1", "persons": 2, "frames": 4}]

    def test_any_uses_threshold_one(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(db, "conn", lambda: _fake_conn([], captured))

        db.get_streams_for_clusters("o", ["a", "b"], "any")

        assert captured["params"][2] == 1

    def test_empty_ids_skips_query(self, monkeypatch):
        def boom():
            raise AssertionError("should not query the database")

        monkeypatch.setattr(db, "conn", boom)
        assert db.get_streams_for_clusters("o", [], "all") == []
