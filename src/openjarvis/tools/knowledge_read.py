"""Read one complete document from the personal knowledge store."""

from __future__ import annotations

from pathlib import PureWindowsPath
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from openjarvis.connectors.store import KnowledgeStore
from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_DEFAULT_MAX_CHARS = 100_000
_MAX_MAX_CHARS = 500_000


def _normalise_path(value: str) -> str:
    """Return a decoded, single-backslash knowledge-store path."""
    raw = unquote((value or "").strip())
    if raw.startswith("obsidian://"):
        raw = parse_qs(urlparse(raw).query).get("file", [raw])[0]
    if raw.lower().startswith("obsidian:"):
        raw = raw.split(":", 1)[1]
    raw = raw.replace("/", "\\")
    while "\\\\" in raw:
        raw = raw.replace("\\\\", "\\")
    return raw.lstrip("\\")


def _merge_overlapping_chunks(parts: list[str]) -> str:
    """Reassemble legacy overlapping chunks without repeating their tails."""
    merged = ""
    for raw in parts:
        part = raw.strip()
        if not part:
            continue
        if not merged:
            merged = part
            continue
        if part in merged:
            continue
        if merged in part:
            merged = part
            continue

        max_overlap = min(len(merged), len(part), 4096)
        overlap = 0
        for size in range(max_overlap, 7, -1):
            if merged[-size:] == part[:size]:
                overlap = size
                break
        merged += ("" if overlap else "\n\n") + part[overlap:]
    return merged


@ToolRegistry.register("knowledge_read")
class KnowledgeReadTool(BaseTool):
    """Resolve and read a complete indexed document in chunk order."""

    tool_id = "knowledge_read"

    def __init__(self, store: Optional[KnowledgeStore] = None) -> None:
        self._store = store

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge_read",
            description=(
                "Read one complete indexed document or Obsidian note. Prefer this "
                "over knowledge_sql after knowledge_search identifies a path. "
                "Accepts forward slashes, backslashes, doubled backslashes, an "
                "obsidian:// URL, or an exact title. Returns an explicit complete "
                "or incomplete status and the canonical source path."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Document path, doc_id, or Obsidian URL.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Exact title when a path is unavailable.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional source filter, such as obsidian.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum returned characters (default 100000).",
                    },
                },
                "anyOf": [{"required": ["path"]}, {"required": ["title"]}],
            },
            category="knowledge",
        )

    def execute(self, **params: Any) -> ToolResult:
        if self._store is None:
            return ToolResult(
                tool_name="knowledge_read",
                content="No knowledge store configured.",
                success=False,
            )

        raw_path = str(params.get("path", "") or "")
        title = str(params.get("title", "") or "").strip()
        source = str(params.get("source", "") or "").strip()
        path = _normalise_path(raw_path)
        if not path and not title:
            return ToolResult(
                tool_name="knowledge_read",
                content="Provide a document path or exact title.",
                success=False,
            )

        clauses = ["deleted_at IS NULL"]
        values: list[Any] = []
        if source:
            clauses.append("source = ?")
            values.append(source)

        identity_clause = ""
        if path:
            expected_doc_id = path if ":" in path else f"{source or 'obsidian'}:{path}"
            identity_clause = (
                "(LOWER(doc_id) = LOWER(?) OR LOWER(source_id) = LOWER(?))"
            )
            values.extend([expected_doc_id, path])
        elif title:
            identity_clause = "LOWER(title) = LOWER(?)"
            values.append(title)
        clauses.append(identity_clause)

        matches = self._store._conn.execute(
            "SELECT DISTINCT doc_id, source_id, title, source, url "
            "FROM knowledge_chunks WHERE " + " AND ".join(clauses),
            values,
        ).fetchall()

        # A path copied from a UI can be incomplete. Fall back to the exact
        # filename only when that produces a single unambiguous document.
        if not matches and path:
            fallback_title = PureWindowsPath(path).stem
            fallback_clauses = ["deleted_at IS NULL", "LOWER(title) = LOWER(?)"]
            fallback_values: list[Any] = [fallback_title]
            if source:
                fallback_clauses.insert(1, "source = ?")
                fallback_values.insert(0, source)
            matches = self._store._conn.execute(
                "SELECT DISTINCT doc_id, source_id, title, source, url "
                "FROM knowledge_chunks WHERE " + " AND ".join(fallback_clauses),
                fallback_values,
            ).fetchall()

        if not matches:
            requested = path or title
            return ToolResult(
                tool_name="knowledge_read",
                content=f"No active document matched: {requested}",
                success=True,
                metadata={"complete": False, "num_matches": 0},
            )

        if len(matches) > 1:
            choices = "\n".join(f"- {row['doc_id']}" for row in matches[:20])
            return ToolResult(
                tool_name="knowledge_read",
                content=(
                    "Multiple documents matched. Call knowledge_read again with one "
                    f"exact path:\n{choices}"
                ),
                success=True,
                metadata={"complete": False, "num_matches": len(matches)},
            )

        match = matches[0]
        rows = self._store._conn.execute(
            "SELECT content, chunk_index FROM knowledge_chunks "
            "WHERE doc_id = ? AND deleted_at IS NULL ORDER BY chunk_index",
            (match["doc_id"],),
        ).fetchall()
        text = _merge_overlapping_chunks([row["content"] for row in rows])
        requested_max = int(params.get("max_chars", _DEFAULT_MAX_CHARS))
        max_chars = max(1, min(requested_max, _MAX_MAX_CHARS))
        complete = len(text) <= max_chars
        visible = text if complete else text[:max_chars]
        status = "complete" if complete else "INCOMPLETE (character limit reached)"
        canonical_path = match["source_id"] or match["doc_id"]
        content = (
            f"Title: {match['title']}\n"
            f"Path: {canonical_path}\n"
            f"Source: {match['source']}\n"
            f"Retrieval status: {status}\n"
            f"Chunks read: {len(rows)}\n\n{visible}"
        )
        if not complete:
            content += "\n\n[INCOMPLETE: do not infer anything from omitted text.]"
        return ToolResult(
            tool_name="knowledge_read",
            content=content,
            success=True,
            metadata={
                "complete": complete,
                "doc_id": match["doc_id"],
                "path": canonical_path,
                "num_chunks": len(rows),
                "num_matches": 1,
            },
        )


__all__ = ["KnowledgeReadTool"]
