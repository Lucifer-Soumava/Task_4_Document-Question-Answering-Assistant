"""Local TF-IDF passage retrieval and extractive, source-grounded QA."""
from __future__ import annotations
from typing import Any
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

FALLBACK_MESSAGE = "Information not found in provided document"


def _location(chunk: dict[str, Any]) -> str:
    if chunk.get("page") is not None:
        return f"page {chunk['page']}"
    paragraphs = chunk.get("paragraphs", [])
    if len(paragraphs) == 1:
        return f"paragraph {paragraphs[0]}"
    if paragraphs:
        return f"paragraphs {paragraphs[0]}–{paragraphs[-1]}"
    return "location unavailable"


class DocumentQA:
    """TF-IDF retrieval plus extractive sentence selection; no generative API."""
    def __init__(self, chunks: list[dict[str, Any]], min_relevance: float = 0.10, top_k: int = 4):
        self.chunks = chunks
        self.min_relevance = min_relevance
        self.top_k = top_k
        self.vectorizer = None
        self.chunk_matrix = None
        if chunks:
            self.vectorizer = TfidfVectorizer(
                lowercase=True, stop_words="english", ngram_range=(1, 2),
                sublinear_tf=True, max_features=50000,
            )
            try:
                self.chunk_matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])
            except ValueError:
                self.vectorizer = None
                self.chunk_matrix = None

    @classmethod
    def from_segments(cls, segments: list[dict[str, Any]], **kwargs: Any) -> "DocumentQA":
        from src.chunking import semantic_chunk_segments
        return cls(semantic_chunk_segments(segments), **kwargs)

    def _analyze(self, text: str) -> set[str]:
        # Use unigram coverage for the support gate; TF-IDF itself still uses bigrams.
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {token for token in tokens if token not in ENGLISH_STOP_WORDS and len(token) > 1}

    def answer(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question or self.vectorizer is None or self.chunk_matrix is None:
            return self._fallback(0.0)

        query_vector = self.vectorizer.transform([question])
        if query_vector.nnz == 0:
            return self._fallback(0.0)

        query_terms = self._analyze(question)
        chunk_terms = [self._analyze(chunk["text"]) for chunk in self.chunks]
        similarities = cosine_similarity(query_vector, self.chunk_matrix).ravel()
        ranked: list[tuple[float, float, float, int]] = []
        for index, similarity in enumerate(similarities):
            coverage = (len(query_terms & chunk_terms[index]) / len(query_terms)) if query_terms else 0.0
            relevance = 0.70 * float(similarity) + 0.30 * coverage
            ranked.append((relevance, float(similarity), coverage, index))
        ranked.sort(reverse=True)

        best_relevance, _, best_coverage, best_index = ranked[0]
        if best_relevance < self.min_relevance or best_coverage < 0.15:
            return self._fallback(best_relevance)

        candidates: list[dict[str, Any]] = []
        for relevance, _, _, index in ranked[:self.top_k]:
            chunk = self.chunks[index]
            sentences = self._split_sentences(chunk["text"]) or [chunk["text"]]
            for sentence in sentences:
                sentence_vector = self.vectorizer.transform([sentence])
                sim = float(cosine_similarity(query_vector, sentence_vector)[0, 0])
                sentence_terms = self._analyze(sentence)
                coverage = (len(query_terms & sentence_terms) / len(query_terms)) if query_terms else 0.0
                score = 0.65 * sim + 0.35 * coverage
                # Require meaningful overlap at sentence level as well as chunk level.
                # This helps reject questions that share only broad terms such as "college" or "fee".
                minimum_coverage = 0.45 if len(query_terms) > 1 else 1.0
                if score > 0 and coverage >= minimum_coverage:
                    candidates.append({
                        "text": sentence, "score": score,
                        "chunk": chunk, "chunk_index": index,
                    })
        candidates.sort(key=lambda item: item["score"], reverse=True)

        selected: list[dict[str, Any]] = []
        seen_text: set[str] = set()
        threshold = max(0.08, candidates[0]["score"] * 0.80) if candidates else 1.0
        for candidate in candidates:
            normalized = candidate["text"].casefold()
            if normalized in seen_text or candidate["score"] < threshold:
                continue
            selected.append(candidate)
            seen_text.add(normalized)
            if len(selected) == 3:
                break

        if not selected:
            return self._fallback(best_relevance)

        answer_lines, citations, passages = [], [], []
        seen_passages: set[tuple[str, int | None, int]] = set()
        for item in selected:
            chunk = item["chunk"]
            citation = f"{chunk['source']} | {_location(chunk)} | chunk {chunk['chunk_id']}"
            answer_lines.append(f"{item['text']} [{citation}]")
            if citation not in citations:
                citations.append(citation)
            passage_key = (chunk["source"], chunk.get("page"), chunk["chunk_id"])
            if passage_key not in seen_passages:
                passages.append({
                    "citation": citation, "text": chunk["text"],
                    "score": round(float(item["score"]), 3),
                })
                seen_passages.add(passage_key)

        return {
            "supported": True,
            "answer": "\n\n".join(answer_lines),
            "citations": citations,
            "passages": passages,
            "relevance": round(best_relevance, 3),
        }

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        from src.chunking import split_sentences
        return split_sentences(text)

    @staticmethod
    def _fallback(relevance: float) -> dict[str, Any]:
        return {
            "supported": False, "answer": FALLBACK_MESSAGE,
            "citations": [], "passages": [], "relevance": round(float(relevance), 3),
        }
