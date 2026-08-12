"""KnowledgeSearchTool — filtered BM25 retrieval with source attribution.

Wraps ``KnowledgeStore`` so agents can search ingested documents by text query
and optional provenance filters (source, doc_type, author, date range).
Optionally delegates to a ``TwoStageRetriever`` for BM25 + reranking.
"""

from __future__ import annotations

import re
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import unquote

from openjarvis.connectors.store import KnowledgeStore
from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

if TYPE_CHECKING:
    from openjarvis.connectors.retriever import TwoStageRetriever


@ToolRegistry.register("knowledge_search")
class KnowledgeSearchTool(BaseTool):
    """Search the knowledge store using filtered BM25 retrieval.

    Results include source attribution so agents can cite provenance.
    When a ``TwoStageRetriever`` is supplied it is used in place of the
    store's direct ``retrieve`` method, enabling optional semantic reranking.
    """

    tool_id = "knowledge_search"

    @staticmethod
    def _normalise_query(query: str) -> str:
        """Turn copied paths and filenames into safe FTS search terms."""
        decoded = unquote(query.strip())
        is_path = "/" in decoded or "\\" in decoded
        if is_path:
            decoded = PureWindowsPath(decoded.replace("/", "\\")).name
        if decoded.lower().endswith((".md", ".markdown", ".txt")):
            decoded = decoded.rsplit(".", 1)[0]
            is_path = True
        if is_path:
            decoded = re.sub(r"[^\w\s]", " ", decoded)
        return " ".join(decoded.replace("_", " ").split())

    def __init__(
        self,
        store: Optional[KnowledgeStore] = None,
        retriever: Optional["TwoStageRetriever"] = None,
    ) -> None:
        self._store = store
        self._retriever = retriever

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge_search",
            description=(
                "Search ingested personal knowledge (emails, Slack messages,"
                " documents) using full-text BM25 retrieval with optional"
                " filters for source, type, author, and date range."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Full-text search query.",
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Filter by source connector"
                            " (e.g. 'gmail', 'slack', 'obsidian')."
                        ),
                    },
                    "doc_type": {
                        "type": "string",
                        "description": (
                            "Filter by document type"
                            " (e.g. 'email', 'message', 'document')."
                        ),
                    },
                    "author": {
                        "type": "string",
                        "description": "Filter by author.",
                    },
                    "since": {
                        "type": "string",
                        "description": (
                            "Exclude documents before this ISO 8601 timestamp."
                        ),
                    },
                    "until": {
                        "type": "string",
                        "description": (
                            "Exclude documents after this ISO 8601 timestamp."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results (default 10).",
                    },
                },
                "required": ["query"],
            },
            category="knowledge",
        )

    def execute(self, **params: Any) -> ToolResult:
        if self._store is None and self._retriever is None:
            return ToolResult(
                tool_name="knowledge_search",
                content="No knowledge store configured.",
                success=False,
            )

        original_query: str = params.get("query", "")
        if not original_query:
            return ToolResult(
                tool_name="knowledge_search",
                content="No query provided.",
                success=False,
            )

        query = self._normalise_query(original_query)
        if not query:
            return ToolResult(
                tool_name="knowledge_search",
                content="No searchable terms remained after path normalisation.",
                success=False,
            )

        top_k: int = int(params.get("top_k", 10))
        candidate_k = max(top_k * 4, top_k)
        source: Optional[str] = params.get("source")
        doc_type: Optional[str] = params.get("doc_type")
        author: Optional[str] = params.get("author")
        since: Optional[str] = params.get("since")
        until: Optional[str] = params.get("until")

        if self._retriever is not None:
            results = self._retriever.retrieve(
                query,
                top_k=candidate_k,
                source=source or "",
                doc_type=doc_type or "",
                author=author or "",
                since=since or "",
                until=until or "",
            )
        else:
            results = self._store.retrieve(  # type: ignore[union-attr]
                query,
                top_k=candidate_k,
                source=source,
                doc_type=doc_type,
                author=author,
                since=since,
                until=until,
            )

        if not results:
            return ToolResult(
                tool_name="knowledge_search",
                content="No relevant results found.",
                success=True,
                metadata={"num_results": 0},
            )

        # A search result is a stored chunk, not a note. Group matching chunks
        # by their provenance so the model sees one result per document and
        # does not describe sections from the same note as duplicates.
        grouped: dict[str, list[Any]] = {}
        for result in results:
            meta = result.metadata
            key = str(
                meta.get("doc_id")
                or meta.get("source_id")
                or meta.get("url")
                or f"{result.source}:{meta.get('title', '')}"
            )
            grouped.setdefault(key, []).append(result)

        lines: list[str] = []
        document_groups = list(grouped.values())[:top_k]
        for i, group in enumerate(document_groups, start=1):
            result = group[0]
            meta = result.metadata
            src_label = result.source or meta.get("source", "")
            title = meta.get("title", "")
            result_author = meta.get("author", "")
            url = meta.get("url", "")

            # Build header line
            header_parts: list[str] = []
            if src_label:
                header_parts.append(f"[{src_label}]")
            if title:
                header_parts.append(title)
            if result_author:
                header_parts.append(f"by {result_author}")
            if url:
                header_parts.append(f"({url})")

            header = " ".join(header_parts) if header_parts else "(unknown source)"
            source_path = meta.get("source_id") or meta.get("doc_id", "")
            lines.append(f"**Result {i}:** {header}")
            if source_path:
                lines.append(f"Path: {source_path}")
            for snippet in group[:3]:
                lines.append(snippet.content)
            if len(group) > 3:
                lines.append(
                    f"[{len(group) - 3} additional matching chunks omitted; "
                    "use knowledge_read with the path above for the complete document.]"
                )
            lines.append("")

        formatted = "\n".join(lines).rstrip()

        return ToolResult(
            tool_name="knowledge_search",
            content=formatted,
            success=True,
            metadata={
                "num_results": len(document_groups),
                "num_matching_chunks": len(results),
                "query_used": query,
                "grouped_by_document": True,
            },
        )


__all__ = ["KnowledgeSearchTool"]
