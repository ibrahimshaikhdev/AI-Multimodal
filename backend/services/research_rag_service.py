from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.services.ocr_service import extract_document_text


logger = logging.getLogger(__name__)


class ResearchIndexError(RuntimeError):
    pass


class ResearchRAGService:
    CHUNK_CHARACTERS = 1200
    CHUNK_OVERLAP = 180
    MAX_TOP_K = 10

    def __init__(
        self,
        index_root: str | Path,
        embedding_model: str,
        upload_root: str | Path,
    ):
        self.index_root = Path(index_root)
        self.embedding_model_name = embedding_model
        self.upload_root = Path(upload_root)
        self._model: SentenceTransformer | None = None
        self._model_lock = threading.Lock()
        self._user_locks: dict[int, threading.RLock] = {}
        self._user_locks_guard = threading.Lock()

    def search(
        self,
        user_id: int,
        papers: list[Any],
        query: str,
        *,
        top_k: int = 5,
        paper_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= self.MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {self.MAX_TOP_K}")

        index, chunks, _ = self._load_or_build(user_id, papers)
        if index.ntotal == 0:
            return []

        query_vector = self._encode([query.strip()])
        candidate_count = (
            index.ntotal
            if paper_ids is not None
            else min(index.ntotal, top_k * 4)
        )
        scores, indexes = index.search(query_vector, candidate_count)
        matches = []
        for score, chunk_index in zip(scores[0], indexes[0]):
            if chunk_index < 0:
                continue
            chunk = chunks[int(chunk_index)]
            if paper_ids is not None and chunk["paper_id"] not in paper_ids:
                continue
            matches.append({**chunk, "score": float(score)})
            if len(matches) == top_k:
                break
        return matches

    def paper_text(self, user_id: int, papers: list[Any], paper_id: int) -> str:
        _, _, texts = self._load_or_build(user_id, papers)
        text = texts.get(str(paper_id))
        if not isinstance(text, str) or not text.strip():
            raise ResearchIndexError(f"Research paper {paper_id} has no readable text.")
        return text

    def similarity_matrix(
        self,
        first_texts: list[str],
        second_texts: list[str],
    ) -> list[list[float]]:
        if not first_texts or not second_texts:
            return []
        vectors = self._encode(first_texts + second_texts)
        first_vectors = vectors[:len(first_texts)]
        second_vectors = vectors[len(first_texts):]
        first_norms = np.linalg.norm(first_vectors, axis=1, keepdims=True)
        second_norms = np.linalg.norm(second_vectors, axis=1, keepdims=True)
        first_vectors = first_vectors / np.maximum(first_norms, 1e-12)
        second_vectors = second_vectors / np.maximum(second_norms, 1e-12)
        scores = np.clip(first_vectors @ second_vectors.T, -1.0, 1.0)
        return scores.astype(float).tolist()

    def _load_or_build(self, user_id: int, papers: list[Any]):
        lock = self._lock_for_user(user_id)
        with lock:
            source_files = []
            manifest = []
            for paper in papers:
                path = self._paper_path(user_id, paper)
                try:
                    content = path.read_bytes()
                except OSError as exc:
                    raise ResearchIndexError(
                        f"The source file for paper {paper.id} is unavailable."
                    ) from exc
                manifest.append(
                    {
                        "id": paper.id,
                        "stored_filename": paper.stored_filename,
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
                source_files.append((paper, path, content))

            cache_directory = self.index_root / str(user_id)
            index_path = cache_directory / "index.faiss"
            metadata_path = cache_directory / "metadata.json"
            if index_path.is_file() and metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if not isinstance(metadata, dict):
                        raise ValueError("Cached research metadata is not an object.")
                    if (
                        metadata.get("manifest") == manifest
                        and metadata.get("embedding_model") == self.embedding_model_name
                    ):
                        cached_chunks = metadata.get("chunks")
                        cached_texts = metadata.get("texts")
                        if not isinstance(cached_chunks, list) or not isinstance(cached_texts, dict):
                            raise ValueError("Cached research metadata has an invalid shape.")
                        cached_index = faiss.read_index(str(index_path))
                        if cached_index.ntotal != len(cached_chunks):
                            raise ValueError("Cached index and chunk metadata are inconsistent.")
                        return (
                            cached_index,
                            cached_chunks,
                            cached_texts,
                        )
                except (OSError, ValueError, KeyError, RuntimeError):
                    logger.warning("Rebuilding unreadable research index for user %s", user_id)

            chunks = []
            texts: dict[str, str] = {}
            for paper, path, content in source_files:
                try:
                    text = extract_document_text(content, path.name).strip()
                except (ValueError, RuntimeError) as exc:
                    raise ResearchIndexError(
                        f"Could not extract text from paper {paper.id}: {exc}"
                    ) from exc
                if not text:
                    raise ResearchIndexError(f"Research paper {paper.id} has no readable text.")
                texts[str(paper.id)] = text
                for chunk_number, passage in enumerate(self._chunk_text(text), start=1):
                    chunks.append(
                        {
                            "citation": f"[P{paper.id}-C{chunk_number}]",
                            "paper_id": paper.id,
                            "paper_title": paper.title,
                            "topic": paper.topic,
                            "paper_date": (
                                paper.paper_date.isoformat() if paper.paper_date else None
                            ),
                            "chunk_number": chunk_number,
                            "text": passage,
                        }
                    )

            vectors = self._encode([chunk["text"] for chunk in chunks])
            if not len(chunks):
                raise ResearchIndexError("The research library contains no readable text.")
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)

            cache_directory.mkdir(parents=True, exist_ok=True)
            temporary_index = index_path.with_suffix(".faiss.tmp")
            temporary_metadata = metadata_path.with_suffix(".json.tmp")
            faiss.write_index(index, str(temporary_index))
            temporary_metadata.write_text(
                json.dumps(
                    {
                        "manifest": manifest,
                        "embedding_model": self.embedding_model_name,
                        "chunks": chunks,
                        "texts": texts,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(temporary_index, index_path)
            os.replace(temporary_metadata, metadata_path)
            return index, chunks, texts

    def _paper_path(self, user_id: int, paper: Any) -> Path:
        filename = Path(paper.stored_filename)
        if filename.name != paper.stored_filename or filename.suffix.lower() not in {".pdf", ".docx"}:
            raise ResearchIndexError(f"Research paper {paper.id} has an invalid stored file reference.")
        research_directory = (self.upload_root / "research_papers").resolve()
        user_directory = research_directory / str(user_id)
        resolved_user_directory = user_directory.resolve()
        path = user_directory / paper.stored_filename
        if (
            resolved_user_directory.parent != research_directory
            or resolved_user_directory.name != str(user_id)
            or path.resolve().parent != resolved_user_directory
        ):
            raise ResearchIndexError("Research paper file path is invalid.")
        return path

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        model = self._get_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _get_model(self) -> SentenceTransformer:
        with self._model_lock:
            if self._model is None:
                self._model = SentenceTransformer(
                    self.embedding_model_name,
                    device="cpu",
                )
            return self._model

    def _lock_for_user(self, user_id: int) -> threading.RLock:
        with self._user_locks_guard:
            return self._user_locks.setdefault(user_id, threading.RLock())

    @classmethod
    def _chunk_text(cls, text: str) -> list[str]:
        normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        chunks = []
        start = 0
        while start < len(normalized):
            end = min(start + cls.CHUNK_CHARACTERS, len(normalized))
            if end < len(normalized):
                boundary = normalized.rfind(" ", start + cls.CHUNK_CHARACTERS // 2, end)
                if boundary > start:
                    end = boundary
            passage = normalized[start:end].strip()
            if passage:
                chunks.append(passage)
            if end >= len(normalized):
                break
            start = max(end - cls.CHUNK_OVERLAP, start + 1)
        return chunks


__all__ = ["ResearchIndexError", "ResearchRAGService"]
