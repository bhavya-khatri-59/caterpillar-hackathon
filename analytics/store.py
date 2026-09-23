"""Append-only, SHA-256 hash-chained incident log on SQLite. Implements `EventSink`.

    h_i = SHA256(h_{i-1} || canonical_json(seq, event, outcome, note))

There is no update or delete path in the code, and SQLite triggers reject UPDATE/DELETE.
Anyone who edits the file directly (bypassing the triggers) breaks the chain, and
`verify()` names the first broken row.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from shared.contracts import AlertEvent, IncidentRecord

GENESIS = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    seq       INTEGER PRIMARY KEY,
    alert_id  TEXT NOT NULL,
    operator_id TEXT NOT NULL,
    body      TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash      TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS incidents_no_update BEFORE UPDATE ON incidents
BEGIN SELECT RAISE(ABORT, 'incident log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS incidents_no_delete BEFORE DELETE ON incidents
BEGIN SELECT RAISE(ABORT, 'incident log is append-only'); END;
"""


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def chain_hash(prev_hash: str, body: str) -> str:
    return hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()


def _body(seq: int, event: AlertEvent, outcome: Optional[str], note: Optional[str]) -> str:
    return canonical({"seq": seq, "event": event.model_dump(mode="json"), "outcome": outcome, "note": note})


class IncidentStore:
    """Thread-safe; one connection per call so FastAPI worker threads can share an instance."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._lock = threading.Lock()
        self._mem = sqlite3.connect(":memory:", check_same_thread=False) if self.path == ":memory:" else None
        if self._mem is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return self._mem if self._mem is not None else sqlite3.connect(self.path)

    # -- EventSink ---------------------------------------------------------------------------
    def emit(self, event: AlertEvent) -> None:
        self.append(event)

    # -- writes --------------------------------------------------------------------------------
    def append(self, event: AlertEvent, outcome: Optional[str] = None, note: Optional[str] = None) -> IncidentRecord:
        with self._lock:
            c = self._conn()
            try:
                row = c.execute("SELECT seq, hash FROM incidents ORDER BY seq DESC LIMIT 1").fetchone()
                seq, prev = (row[0] + 1, row[1]) if row else (1, GENESIS)
                body = _body(seq, event, outcome, note)
                h = chain_hash(prev, body)
                c.execute("INSERT INTO incidents (seq, alert_id, operator_id, body, prev_hash, hash) VALUES (?,?,?,?,?,?)",
                          (seq, event.alert_id, event.operator_id, body, prev, h))
                c.commit()
            finally:
                if self._mem is None:
                    c.close()
        return IncidentRecord(seq=seq, event=event, outcome=outcome, note=note, prev_hash=prev, hash=h)

    # -- reads ---------------------------------------------------------------------------------
    def _rows(self) -> list[tuple]:
        c = self._conn()
        try:
            return c.execute("SELECT seq, body, prev_hash, hash FROM incidents ORDER BY seq").fetchall()
        finally:
            if self._mem is None:
                c.close()

    def records(self, operator_id: Optional[str] = None) -> list[IncidentRecord]:
        out = []
        for seq, body, prev, h in self._rows():
            b = json.loads(body)
            rec = IncidentRecord(seq=seq, event=AlertEvent(**b["event"]), outcome=b.get("outcome"),
                                 note=b.get("note"), prev_hash=prev, hash=h)
            if operator_id is None or rec.event.operator_id == operator_id:
                out.append(rec)
        return out

    def __len__(self) -> int:
        return len(self._rows())

    def head_hash(self) -> str:
        rows = self._rows()
        return rows[-1][3] if rows else GENESIS

    def verify(self) -> dict:
        """Recompute the chain. Returns {"ok", "count", "head_hash", "first_broken_seq", "reason"}."""
        prev, expected_seq = GENESIS, 1
        rows = self._rows()
        for seq, body, stored_prev, stored_hash in rows:
            reason = None
            if seq != expected_seq:
                reason = f"row {expected_seq} is missing"
            elif stored_prev != prev:
                reason = "prev_hash does not match the previous row"
            else:
                try:
                    embedded_seq = json.loads(body).get("seq")
                except ValueError:
                    embedded_seq = None
                if embedded_seq != seq:
                    reason = "row number was changed"
                elif chain_hash(prev, body) != stored_hash:
                    reason = "row content does not match its hash"
            if reason:
                return {"ok": False, "count": len(rows), "head_hash": rows[-1][3],
                        "first_broken_seq": min(seq, expected_seq), "reason": reason}
            prev, expected_seq = stored_hash, seq + 1
        return {"ok": True, "count": len(rows), "head_hash": prev, "first_broken_seq": None, "reason": None}
