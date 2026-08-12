"""Completeness signalling for read-only knowledge SQL."""

from __future__ import annotations

from openjarvis.connectors.store import KnowledgeStore
from openjarvis.tools.knowledge_sql import KnowledgeSQLTool


def test_fifty_rows_are_marked_potentially_incomplete(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    for index in range(50):
        store.store(f"row {index}", source="test", chunk_index=index)
    try:
        result = KnowledgeSQLTool(store).execute(
            query="SELECT content FROM knowledge_chunks ORDER BY chunk_index LIMIT 50"
        )
    finally:
        store.close()

    assert result.success
    assert result.metadata["complete"] is False
    assert "INCOMPLETE RESULT" in result.content
