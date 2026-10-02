"""
jobs_db.py
----------
A small SQLite-backed store for summarization jobs. Used for three things:

  1. Background processing: a job's status/progress is written here from a
     background thread and read by the browser's polling requests.
  2. History: every finished job stays listed here (optionally capped by
     HISTORY_RETENTION_DAYS) so past summaries can be revisited.
  3. Durable downloads: the summary/transcript TEXT is stored directly in
     the row, so a .docx/.pdf/.txt can be regenerated on demand even after
     the original generated files were cleaned up by the upload/result
     retention sweep in app.py.

No extra dependency: sqlite3 and threading are both in the Python standard
library. Each function opens and closes its own connection, which is the
simplest safe way to use sqlite3 across the multiple threads a background
job + concurrent page requests create.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "jobs.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    filename        TEXT,
    status          TEXT NOT NULL,      -- queued | processing | done | error
    progress        TEXT,               -- short human-readable current step
    lang_code       TEXT,
    lang_prob       TEXT,
    whisper_model   TEXT,
    engine          TEXT,               -- summarizer backend actually used
    fallback_note   TEXT,
    summary         TEXT,
    transcript      TEXT,
    action_items    TEXT,               -- JSON-encoded list, or null
    action_items_note TEXT,             -- e.g. why extraction was skipped
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs (created_at DESC);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # lets readers and the writer overlap
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def create_job(job_id: str, filename: str) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, created_at, updated_at, filename, status, progress) "
            "VALUES (?, ?, ?, ?, 'queued', 'Waiting to start...')",
            (job_id, now, now, filename),
        )


def update_job(job_id: str, **fields: Any) -> None:
    """Updates only the given columns. action_items (a list) is JSON-encoded
    automatically if present."""
    if "action_items" in fields and fields["action_items"] is not None:
        fields["action_items"] = json.dumps(fields["action_items"])
    fields["updated_at"] = time.time()
    columns = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [job_id]
    with _connect() as conn:
        conn.execute(f"UPDATE jobs SET {columns} WHERE id = ?", values)


def get_job(job_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    job = dict(row)
    if job.get("action_items"):
        try:
            job["action_items"] = json.loads(job["action_items"])
        except (TypeError, ValueError):
            job["action_items"] = None
    return job


def list_jobs(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, created_at, filename, status, lang_code, engine, whisper_model "
            "FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_old_jobs(retention_days: float) -> None:
    """Optional housekeeping: trims history beyond retention_days. Not called
    automatically unless HISTORY_RETENTION_DAYS is set, since history (unlike
    the raw audio/results files) is meant to persist by default."""
    cutoff = time.time() - (retention_days * 86400)
    with _connect() as conn:
        conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
