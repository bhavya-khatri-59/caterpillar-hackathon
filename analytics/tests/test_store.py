import json
import sqlite3

import pytest

from analytics.settings import ROOT
from analytics.store import GENESIS, IncidentStore
from shared.contracts import AlertEvent

EVENTS = [AlertEvent(**e) for e in json.loads((ROOT / "shared/fixtures/alerts.json").read_text())]


def _filled(path) -> IncidentStore:
    st = IncidentStore(path)
    for e in EVENTS:
        st.emit(e)
    st.append(EVENTS[0], outcome="near_miss", note="spotter confirmed")
    return st


def test_chain_links_and_verifies(tmp_path):
    st = _filled(tmp_path / "log.db")
    recs = st.records()
    assert [r.seq for r in recs] == [1, 2, 3, 4, 5]
    assert recs[0].prev_hash == GENESIS
    assert all(b.prev_hash == a.hash for a, b in zip(recs, recs[1:]))
    assert st.verify() == {"ok": True, "count": 5, "head_hash": recs[-1].hash, "first_broken_seq": None, "reason": None}


def test_update_and_delete_are_rejected(tmp_path):
    path = tmp_path / "log.db"
    _filled(path)
    c = sqlite3.connect(path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        c.execute("UPDATE incidents SET body = 'x' WHERE seq = 2")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        c.execute("DELETE FROM incidents WHERE seq = 2")


@pytest.mark.parametrize("seq", [1, 3, 5])
def test_editing_a_row_breaks_verify_at_that_row(tmp_path, seq):
    path = tmp_path / "log.db"
    st = _filled(path)
    c = sqlite3.connect(path)
    c.execute("DROP TRIGGER incidents_no_update")              # someone with direct file access
    body = json.loads(c.execute("SELECT body FROM incidents WHERE seq = ?", (seq,)).fetchone()[0])
    body["event"]["tier"] = "log"
    c.execute("UPDATE incidents SET body = ? WHERE seq = ?", (json.dumps(body, sort_keys=True, separators=(",", ":")), seq))
    c.commit()
    v = st.verify()
    assert not v["ok"] and v["first_broken_seq"] == seq


def test_deleting_a_row_is_detected(tmp_path):
    path = tmp_path / "log.db"
    st = _filled(path)
    c = sqlite3.connect(path)
    c.execute("DROP TRIGGER incidents_no_delete")
    c.execute("DELETE FROM incidents WHERE seq = 3")
    c.commit()
    v = st.verify()
    assert not v["ok"] and v["first_broken_seq"] == 3


def test_same_events_give_same_hash(tmp_path):
    assert _filled(tmp_path / "a.db").head_hash() == _filled(tmp_path / "b.db").head_hash()
    assert _filled(":memory:").head_hash() == _filled(tmp_path / "c.db").head_hash()


def test_store_is_an_event_sink():
    from shared.contracts import EventSink

    sink: EventSink = IncidentStore(":memory:")
    sink.emit(EVENTS[0])
    assert len(sink) == 1
