"""
cache_manager.py — Graph extraction cache only.

QA answer caching has been removed entirely — it caused different questions
to return the same answer when semantic similarity was too aggressive.

This module is now ONLY used by graph_builder.py to cache LLM triple
extraction results (which are deterministic per text chunk and expensive).

Usage:
    from cache_manager import get_graph_cache
    cache = get_graph_cache()
    hit = cache.get(prompt)
    cache.set(prompt, result)
"""
import os
import json
import hashlib
import sqlite3
from datetime import datetime, timedelta

CACHE_DIR      = "./cache/graph"
CACHE_DB       = os.path.join(CACHE_DIR, "graph_cache.db")
MAX_ENTRIES    = 2000
TTL_DAYS       = 60    # graph triples are stable, keep longer


class GraphCache:
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._init_db()

    def get(self, prompt: str) -> str | None:
        key  = self._hash(prompt)
        conn = sqlite3.connect(CACHE_DB)
        row  = conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if row:
            conn.execute(
                "UPDATE cache SET hits=hits+1, last_used=? WHERE key=?",
                (datetime.utcnow().isoformat(), key)
            )
            conn.commit()
            print("[GraphCache] ✓ hit")
        conn.close()
        return row[0] if row else None

    def set(self, prompt: str, value: str) -> None:
        if not value or len(value.strip()) < 2:
            return
        key = self._hash(prompt)
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(CACHE_DB)
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, prompt, value, hits, created, last_used) "
            "VALUES (?,?,?,0,?,?)",
            (key, prompt[:500], value, now, now)
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        if count > MAX_ENTRIES:
            conn.execute(
                "DELETE FROM cache WHERE key IN "
                "(SELECT key FROM cache ORDER BY last_used ASC LIMIT ?)",
                (count - MAX_ENTRIES,)
            )
            conn.commit()
        conn.close()

    def clear(self) -> None:
        conn = sqlite3.connect(CACHE_DB)
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()

    def stats(self) -> dict:
        conn = sqlite3.connect(CACHE_DB)
        n = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        conn.close()
        return {"entries": n, "db": CACHE_DB}

    def _init_db(self):
        conn = sqlite3.connect(CACHE_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                prompt TEXT,
                value TEXT,
                hits INTEGER DEFAULT 0,
                created TEXT,
                last_used TEXT
            )
        """)
        conn.commit()
        # Prune old entries
        cutoff = (datetime.utcnow() - timedelta(days=TTL_DAYS)).isoformat()
        conn.execute("DELETE FROM cache WHERE last_used < ?", (cutoff,))
        conn.commit()
        conn.close()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()


_instance: GraphCache | None = None

def get_graph_cache() -> GraphCache:
    global _instance
    if _instance is None:
        _instance = GraphCache()
    return _instance