import sqlite3
import json
import os
import time

from pathlib import Path

CACHE_PATH = Path('data/offline_cache.db')

def _init():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CACHE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                id   TEXT PRIMARY KEY,
                type TEXT,
                data TEXT,
                ts   REAL
            )
            """
        )

def store(item_id: str, data: dict, type_: str = "event"):
    _init()
    with sqlite3.connect(CACHE_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (id, type, data, ts) VALUES (?, ?, ?, ?)",
            (item_id, type_, json.dumps(data), time.time()),
        )

def retrieve_all(type_: str = None):
    _init()
    with sqlite3.connect(CACHE_PATH) as conn:
        cur = conn.cursor()
        if type_:
            cur.execute("SELECT id, data, ts FROM cache WHERE type = ?", (type_,))
        else:
            cur.execute("SELECT id, data, ts FROM cache")
        for _id, data, ts in cur.fetchall():
            yield {"id": _id, "data": json.loads(data), "ts": ts}