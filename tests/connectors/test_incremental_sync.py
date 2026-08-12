"""Tests for incremental sync via the `since` parameter in SyncEngine."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

import pytest

from openjarvis.connectors._stubs import BaseConnector, Document, SyncStatus
from openjarvis.connectors.pipeline import IngestionPipeline
from openjarvis.connectors.store import KnowledgeStore
from openjarvis.connectors.sync_engine import SyncEngine

# ---------------------------------------------------------------------------
# TimestampConnector — records the `since` value it receives
# ---------------------------------------------------------------------------


class TimestampConnector(BaseConnector):
    """Connector that records the `since` argument passed to each sync call."""

    connector_id = "timestamp_connector"
    display_name = "Timestamp"
    auth_type = "filesystem"

    def __init__(self, docs: List[Document]) -> None:
        self._docs = docs
        self.received_since: List[Optional[datetime]] = []

    # BaseConnector abstract methods

    def is_connected(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def sync(
        self,
        *,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
    ) -> Iterator[Document]:
        self.received_since.append(since)
        yield from self._docs

    def sync_status(self) -> SyncStatus:
        return SyncStatus(state="idle", items_synced=len(self._docs))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(doc_id: str, content: str = "Test content.") -> Document:
    return Document(
        doc_id=doc_id,
        source="timestamp_connector",
        doc_type="note",
        content=content,
        title=f"Doc {doc_id}",
        author="tester@example.com",
        timestamp=datetime(2025, 3, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(db_path=tmp_path / "knowledge.db")


@pytest.fixture()
def pipeline(store: KnowledgeStore) -> IngestionPipeline:
    return IngestionPipeline(store)


@pytest.fixture()
def engine(pipeline: IngestionPipeline, tmp_path: Path) -> SyncEngine:
    return SyncEngine(pipeline, state_db=str(tmp_path / "sync_state.db"))


# ---------------------------------------------------------------------------
# Test 1: first sync has no checkpoint — since should be None
# ---------------------------------------------------------------------------


def test_first_sync_passes_no_since(engine: SyncEngine) -> None:
    """On the very first sync there is no checkpoint, so since must be None."""
    docs = [_make_doc("first:doc:0", content="First sync document")]
    connector = TimestampConnector(docs)

    engine.sync(connector)

    assert len(connector.received_since) == 1
    assert connector.received_since[0] is None


# ---------------------------------------------------------------------------
# Test 2: second sync should receive a non-None since datetime
# ---------------------------------------------------------------------------


def test_second_sync_passes_since(engine: SyncEngine) -> None:
    """After first sync, checkpoint is written; second sync gets since."""
    docs = [_make_doc("second:doc:0", content="Document for incremental sync")]
    connector = TimestampConnector(docs)

    # First sync — establishes the checkpoint
    engine.sync(connector)
    assert connector.received_since[0] is None

    # Second sync — should receive a datetime parsed from last_sync
    engine.sync(connector)
    assert len(connector.received_since) == 2
    since_value = connector.received_since[1]
    assert since_value is not None
    assert isinstance(since_value, datetime)


def test_legacy_chunks_force_one_full_refresh(
    store: KnowledgeStore, tmp_path: Path
) -> None:
    """A checkpoint cannot prevent pre-fingerprint content from being migrated."""
    state_db = str(tmp_path / "legacy_refresh_state.db")
    original = _make_doc("timestamp_connector:legacy", "Legacy original content")
    first_connector = TimestampConnector([original])
    first_pipeline = IngestionPipeline(store)
    first_engine = SyncEngine(first_pipeline, state_db=state_db)
    first_engine.sync(first_connector)

    # Simulate an index created by the previous pipeline version while keeping
    # the newer sync checkpoint that caused modified files to be skipped.
    rows = store._conn.execute(
        "SELECT id, metadata FROM knowledge_chunks WHERE doc_id = ?",
        (original.doc_id,),
    ).fetchall()
    for row in rows:
        metadata = json.loads(row["metadata"])
        metadata.pop("document_fingerprint", None)
        store._conn.execute(
            "UPDATE knowledge_chunks SET metadata = ? WHERE id = ?",
            (json.dumps(metadata), row["id"]),
        )
    store._conn.commit()

    updated = _make_doc("timestamp_connector:legacy", "Current replacement content")
    second_connector = TimestampConnector([updated])
    second_pipeline = IngestionPipeline(store)
    second_engine = SyncEngine(second_pipeline, state_db=state_db)

    written = second_engine.sync(second_connector)

    assert second_connector.received_since == [None]
    assert written == 1
    assert store.count() == 1
    assert store.retrieve("Legacy original content") == []
    assert len(store.retrieve("Current replacement content")) == 1

    # Once migrated, ordinary incremental behaviour resumes.
    third_connector = TimestampConnector([updated])
    second_engine.sync(third_connector)
    assert third_connector.received_since[0] is not None


# ---------------------------------------------------------------------------
# Test 3: incremental sync only adds newly returned items
# ---------------------------------------------------------------------------


def test_incremental_only_adds_new_items(
    pipeline: IngestionPipeline, store: KnowledgeStore, tmp_path: Path
) -> None:
    """Connector returns only new docs on the second call; store grows by that count."""

    class SelectiveTimestampConnector(TimestampConnector):
        """Returns a different doc list on each sync call."""

        def __init__(
            self, first_docs: List[Document], second_docs: List[Document]
        ) -> None:
            super().__init__(first_docs)
            self._first_docs = first_docs
            self._second_docs = second_docs
            self._call_count = 0

        def sync(
            self,
            *,
            since: Optional[datetime] = None,
            cursor: Optional[str] = None,
        ) -> Iterator[Document]:
            self.received_since.append(since)
            self._call_count += 1
            if self._call_count == 1:
                yield from self._first_docs
            else:
                yield from self._second_docs

    old_docs = [
        _make_doc("incr:old:0", content="Old document alpha"),
        _make_doc("incr:old:1", content="Old document beta"),
    ]
    new_docs = [
        _make_doc("incr:new:0", content="New document gamma"),
    ]

    engine = SyncEngine(pipeline, state_db=str(tmp_path / "incr_sync_state.db"))
    connector = SelectiveTimestampConnector(old_docs, new_docs)

    # First sync — 2 old docs ingested
    items_first = engine.sync(connector)
    assert items_first == 2
    count_after_first = store.count()
    assert count_after_first == 2

    # Second sync — only 1 new doc returned by the connector
    items_second = engine.sync(connector)
    assert items_second == 1
    assert store.count() == 3

    # The second call received a valid since datetime
    assert connector.received_since[1] is not None
    assert isinstance(connector.received_since[1], datetime)
