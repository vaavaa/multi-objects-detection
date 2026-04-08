from __future__ import annotations

import re
from dataclasses import asdict

from app.models import FAQEntry


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text)}


def score_entry(question: str, entry: FAQEntry) -> int:
    q_tokens = tokenize(question)
    if not q_tokens:
        return 0

    score = 0

    for variant in entry.question_variants:
        v_tokens = tokenize(variant)
        overlap = len(q_tokens & v_tokens)
        score = max(score, overlap)

        if question.strip().lower() == variant.strip().lower():
            score += 100

    tag_overlap = len(q_tokens & {t.lower() for t in entry.tags})
    score += tag_overlap

    return score


def search_faqs(question: str, entries: list[FAQEntry], min_score: int = 1) -> list[dict]:
    ranked: list[tuple[int, FAQEntry]] = []

    for entry in entries:
        score = score_entry(question, entry)
        if score >= min_score:
            ranked.append((score, entry))

    ranked.sort(key=lambda item: item[0], reverse=True)

    results: list[dict] = []
    for score, entry in ranked:
        item = asdict(entry)
        item["score"] = score
        results.append(item)

    return results
