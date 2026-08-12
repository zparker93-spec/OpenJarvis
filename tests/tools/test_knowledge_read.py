"""Tests for complete-document knowledge retrieval."""

from __future__ import annotations

from openjarvis.connectors.store import KnowledgeStore
from openjarvis.tools.knowledge_read import KnowledgeReadTool


def _store_note(store: KnowledgeStore, *, path: str, title: str) -> None:
    parts = [
        "# Growth Dashboard",
        "# Growth Dashboard\n\nPrimary objective: create scalable wealth.",
        (
            "Primary objective: create scalable wealth.\n\n"
            "Decision filter: build an asset."
        ),
        (
            "Decision filter: build an asset.\n\n"
            "Jarvis must separate facts from assumptions."
        ),
    ]
    for index, content in enumerate(parts):
        store.store(
            content,
            source="obsidian",
            source_id=path,
            doc_id=f"obsidian:{path}",
            doc_type="note",
            title=title,
            url=f"obsidian://open?vault=Zenen_OS&file={path}",
            chunk_index=index,
        )


def test_reads_complete_note_with_forward_slash_path(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    path = r"03_ZFS\Growth\Growth_Dashboard.md"
    _store_note(store, path=path, title="Growth_Dashboard")
    try:
        result = KnowledgeReadTool(store).execute(
            path="03_ZFS/Growth/Growth_Dashboard.md",
            source="obsidian",
        )
    finally:
        store.close()

    assert result.success
    assert result.metadata["complete"] is True
    assert result.metadata["num_chunks"] == 4
    assert f"Path: {path}" in result.content
    assert result.content.count("Primary objective: create scalable wealth.") == 1
    assert "Jarvis must separate facts from assumptions." in result.content


def test_collapses_doubled_backslashes(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    path = r"03_ZFS\Growth\Growth_Dashboard.md"
    _store_note(store, path=path, title="Growth_Dashboard")
    try:
        result = KnowledgeReadTool(store).execute(
            path=r"03_ZFS\\Growth\\Growth_Dashboard.md"
        )
    finally:
        store.close()

    assert result.success
    assert result.metadata["doc_id"] == f"obsidian:{path}"


def test_ambiguous_title_lists_paths_instead_of_guessing(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    _store_note(store, path=r"A\Dashboard.md", title="Dashboard")
    _store_note(store, path=r"B\Dashboard.md", title="Dashboard")
    try:
        result = KnowledgeReadTool(store).execute(title="Dashboard")
    finally:
        store.close()

    assert result.success
    assert result.metadata["complete"] is False
    assert result.metadata["num_matches"] == 2
    assert "Multiple documents matched" in result.content


def test_marks_character_limited_result_incomplete(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    path = r"Long\Note.md"
    _store_note(store, path=path, title="Note")
    try:
        result = KnowledgeReadTool(store).execute(path=path, max_chars=20)
    finally:
        store.close()

    assert result.success
    assert result.metadata["complete"] is False
    assert "INCOMPLETE" in result.content
