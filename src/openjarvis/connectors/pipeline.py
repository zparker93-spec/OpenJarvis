"""IngestionPipeline — deduplicate, chunk, and store Documents.

Takes ``Document`` objects from connectors, deduplicates by ``doc_id``,
splits content using ``SemanticChunker``, and persists chunks to a
``KnowledgeStore``.

Typical usage::

    store = KnowledgeStore(db_path=":memory:")
    pipeline = IngestionPipeline(store)
    n_chunks = pipeline.ingest(connector.sync())
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Iterable, Optional

from openjarvis.connectors._stubs import Attachment, Document
from openjarvis.connectors.chunker import SemanticChunker
from openjarvis.connectors.embeddings import OllamaEmbedder
from openjarvis.connectors.store import KnowledgeStore


def _namespace_thread_id(source: str, thread_id: Optional[str]) -> Optional[str]:
    """Prefix ``thread_id`` with ``{source}:`` so it can't collide across sources.

    Idempotent: if the input already starts with ``{source}:`` it is returned
    unchanged. Centralised here (rather than per-connector) so a new connector
    author can't forget to namespace.
    """
    if not thread_id:
        return None
    prefix = f"{source}:"
    if thread_id.startswith(prefix):
        return thread_id
    return f"{prefix}{thread_id}"


def _derive_source_id(doc: Document) -> str:
    """Return the connector-set ``source_id`` or extract it from ``doc_id``.

    Many existing connectors compose ``doc_id = f"{source}:{native_id}"``;
    this strips the prefix so storage can index the native ID directly.
    """
    if doc.source_id:
        return doc.source_id
    prefix = f"{doc.source}:"
    if doc.doc_id.startswith(prefix):
        return doc.doc_id[len(prefix) :]
    return doc.doc_id


def _content_hash(text: str) -> str:
    """SHA-256 hex digest of UTF-8-encoded chunk content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _document_fingerprint(doc: Document) -> str:
    """Return a stable digest of every indexed part of *doc*.

    Connectors use a stable ``doc_id`` for the lifetime of a source item.  A
    repeated ID therefore does not necessarily mean a duplicate: local files,
    calendar events, and other mutable records can legitimately change.  The
    fingerprint lets the pipeline skip byte-for-byte repeats while replacing
    stale indexed chunks when any searchable content or provenance changes.
    """
    attachments = [
        {
            "filename": att.filename,
            "mime_type": att.mime_type,
            "size_bytes": att.size_bytes,
            "sha256": att.sha256 or hashlib.sha256(att.content).hexdigest(),
        }
        for att in doc.attachments
    ]
    payload = {
        "content": doc.content,
        "source": doc.source,
        "source_id": _derive_source_id(doc),
        "doc_type": doc.doc_type,
        "title": doc.title,
        "author": doc.author,
        "participants": doc.participants,
        "participants_raw": doc.participants_raw,
        "thread_id": doc.thread_id,
        "channel": doc.channel,
        "url": doc.url,
        "metadata": doc.metadata,
        "attachments": attachments,
    }
    serialised = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


if TYPE_CHECKING:
    from openjarvis.connectors.attachment_store import AttachmentStore


class IngestionPipeline:
    """Deduplicate, chunk, and index documents into a KnowledgeStore.

    Parameters
    ----------
    store:
        The ``KnowledgeStore`` instance to write chunks into.
    max_tokens:
        Soft upper-limit on chunk size passed to ``SemanticChunker``.
    attachment_store:
        Optional ``AttachmentStore`` for persisting attachment blobs and
        extracting text from supported MIME types (PDF, plain text, etc.).
        When ``None`` (default) attachments are silently ignored.
    embedder:
        Optional embedding client (e.g. ``OllamaEmbedder``). When provided,
        every chunk is embedded at ingest time and the resulting float32
        vector is written to the ``embedding`` BLOB column alongside
        ``embedding_model_version``. ``None`` (default) skips embedding so
        in-memory tests and offline runs don't depend on a sidecar daemon.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        max_tokens: int = 512,
        attachment_store: Optional[AttachmentStore] = None,
        embedder: Optional[OllamaEmbedder] = None,
    ) -> None:
        self._store = store
        self._chunker = SemanticChunker(max_tokens=max_tokens)
        self._attachment_store = attachment_store
        self._embedder = embedder
        self._seen_doc_ids: set[str] = set()
        self._document_fingerprints: dict[str, str] = {}
        self._load_existing_doc_ids()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_existing_doc_ids(self) -> None:
        """Populate ``_seen_doc_ids`` from rows already in the store."""
        rows = self._store._conn.execute(
            "SELECT doc_id, metadata FROM knowledge_chunks "
            "WHERE deleted_at IS NULL ORDER BY chunk_index"
        ).fetchall()
        self._seen_doc_ids = {r["doc_id"] for r in rows}
        for row in rows:
            if row["doc_id"] in self._document_fingerprints:
                continue
            try:
                metadata = json.loads(row["metadata"] or "{}")
            except (TypeError, ValueError):
                continue
            fingerprint = metadata.get("document_fingerprint")
            if isinstance(fingerprint, str) and fingerprint:
                self._document_fingerprints[row["doc_id"]] = fingerprint

    def _embed_chunk(self, content: str) -> tuple[Optional[bytes], str]:
        """Return ``(embedding_bytes, model_version)`` for a chunk.

        Returns ``(None, "")`` when no embedder is configured or the embedder
        fails — ingestion continues with the lexical-only row, so a flaky
        local daemon never blocks a sync.
        """
        if self._embedder is None:
            return None, ""
        emb = self._embedder.embed(content)
        if emb is None:
            return None, ""
        return emb, self._embedder.model_version

    def _extract_attachment_text(self, att: Attachment) -> str:
        """Extract text from an attachment.

        Returns the extracted text, or an empty string if the MIME type is
        unsupported or extraction fails.
        """
        if att.mime_type == "application/pdf":
            try:
                import io

                import pdfplumber

                with pdfplumber.open(io.BytesIO(att.content)) as pdf:
                    return "\n".join(page.extract_text() or "" for page in pdf.pages)
            except Exception:  # noqa: BLE001
                return ""
        if att.mime_type in ("text/plain", "text/markdown", "text/csv"):
            return att.content.decode("utf-8", errors="replace")
        return ""

    def needs_full_refresh(self, source: str) -> bool:
        """Return whether *source* still has pre-fingerprint active chunks.

        Incremental connectors normally emit only records changed since their
        last checkpoint.  After this replacement-aware pipeline is installed,
        legacy chunks need one complete pass even when their source files were
        edited before the newest checkpoint was written.
        """
        row = self._store._conn.execute(
            "SELECT 1 FROM knowledge_chunks "
            "WHERE source = ? AND deleted_at IS NULL "
            "AND metadata NOT LIKE ? LIMIT 1",
            (source, '%"document_fingerprint"%'),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, documents: Iterable[Document]) -> int:
        """Ingest an iterable of documents into the knowledge store.

        Duplicate documents are skipped when both ``doc_id`` and content
        fingerprint match.  When an existing ``doc_id`` is emitted with a new
        fingerprint, its old chunks are replaced.  Repeated IDs within one
        input batch are still skipped so the first connector result wins.

        Parameters
        ----------
        documents:
            An iterable of ``Document`` objects (e.g. from a connector's
            ``sync()`` method).

        Returns
        -------
        int
            The total number of chunks written to the store in this call.
        """
        chunks_stored = 0
        seen_this_call: set[str] = set()

        for doc in documents:
            if doc.doc_id in seen_this_call:
                continue
            seen_this_call.add(doc.doc_id)

            document_fingerprint = _document_fingerprint(doc)
            if (
                doc.doc_id in self._seen_doc_ids
                and self._document_fingerprints.get(doc.doc_id) == document_fingerprint
            ):
                continue

            # Compute v1 provenance fields once per document.
            namespaced_thread = _namespace_thread_id(doc.source, doc.thread_id)
            source_id = _derive_source_id(doc)
            ingest_epoch = time.time()

            # Build the parent metadata dict that will be inherited by every
            # chunk produced from this document.
            parent_meta = {
                "title": doc.title,
                "author": doc.author,
                "source": doc.source,
                "source_id": source_id,
                "doc_type": doc.doc_type,
                "url": doc.url or "",
                "thread_id": namespaced_thread or "",
                "channel": doc.channel or "",
            }
            # Merge any extra connector-level metadata (without overwriting
            # the standard provenance fields set above).
            parent_meta.update(doc.metadata)
            parent_meta["document_fingerprint"] = document_fingerprint

            # Normalise the timestamp to a string once.
            if hasattr(doc.timestamp, "isoformat"):
                timestamp_str = doc.timestamp.isoformat()
            else:
                timestamp_str = str(doc.timestamp)

            # Chunk the document content using the type-aware strategy.
            chunks = self._chunker.chunk(
                doc.content,
                doc_type=doc.doc_type,
                metadata=parent_meta,
            )

            # All expensive/fallible document preparation happens before the
            # old rows are removed.  Legacy rows do not have a document
            # fingerprint, so the first time a connector re-emits them they
            # are safely refreshed into the new format.
            if doc.doc_id in self._seen_doc_ids:
                self._store.delete(doc.doc_id)

            for chunk in chunks:
                embedding_bytes, embedding_version = self._embed_chunk(chunk.content)
                self._store.store(
                    content=chunk.content,
                    source=doc.source,
                    source_id=source_id,
                    doc_type=doc.doc_type,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    author=doc.author,
                    participants=doc.participants,
                    participants_raw=doc.participants_raw,
                    timestamp=timestamp_str,
                    thread_id=namespaced_thread,
                    channel=doc.channel,
                    url=doc.url,
                    metadata=chunk.metadata,
                    chunk_index=chunk.index,
                    content_hash=_content_hash(chunk.content),
                    embedding=embedding_bytes,
                    embedding_model_version=embedding_version,
                    last_synced=ingest_epoch,
                )
                chunks_stored += 1

            # Process attachments when an attachment store is configured.
            if self._attachment_store and doc.attachments:
                for att in doc.attachments:
                    if not att.content:
                        continue

                    # Persist the raw blob and obtain its SHA-256.
                    sha = self._attachment_store.store(
                        content=att.content,
                        filename=att.filename,
                        mime_type=att.mime_type,
                        source_doc_id=doc.doc_id,
                    )

                    # Extract searchable text and index it as additional chunks.
                    extracted = self._extract_attachment_text(att)
                    if extracted:
                        att_chunks = self._chunker.chunk(
                            extracted,
                            doc_type=doc.doc_type,
                            metadata={
                                **parent_meta,
                                "attachment": att.filename,
                                "sha256": sha,
                            },
                        )
                        # Synthetic source_id keeps attachment chunks distinct
                        # from body chunks under the UNIQUE(source, source_id,
                        # chunk_index) constraint while still letting them share
                        # a parent doc_id for dedup and blob linkage.
                        att_source_id = f"{source_id}#{att.filename}"
                        for chunk in att_chunks:
                            embedding_bytes, embedding_version = self._embed_chunk(
                                chunk.content
                            )
                            self._store.store(
                                content=chunk.content,
                                source=doc.source,
                                source_id=att_source_id,
                                doc_type=doc.doc_type,
                                doc_id=doc.doc_id,
                                title=f"{doc.title} [{att.filename}]",
                                author=doc.author,
                                participants=doc.participants,
                                participants_raw=doc.participants_raw,
                                timestamp=timestamp_str,
                                thread_id=namespaced_thread,
                                channel=doc.channel,
                                url=doc.url,
                                metadata=chunk.metadata,
                                chunk_index=chunk.index,
                                content_hash=_content_hash(chunk.content),
                                embedding=embedding_bytes,
                                embedding_model_version=embedding_version,
                                last_synced=ingest_epoch,
                            )
                            chunks_stored += 1

            self._seen_doc_ids.add(doc.doc_id)
            self._document_fingerprints[doc.doc_id] = document_fingerprint

        return chunks_stored


__all__ = ["IngestionPipeline"]
