from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.models import FAQEntry


class FAQFormatError(ValueError):
    pass


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        raise FAQFormatError("FAQ markdown must start with YAML frontmatter")

    parts = text.split("---\n", 2)
    if len(parts) < 3:
        raise FAQFormatError("Invalid frontmatter format")

    _, raw_yaml, body = parts
    meta = yaml.safe_load(raw_yaml) or {}
    return meta, body.strip()


def load_faqs(faq_dir: str) -> list[FAQEntry]:
    base = Path(faq_dir)
    if not base.exists():
        raise FileNotFoundError(f"FAQ directory not found: {faq_dir}")

    entries: list[FAQEntry] = []

    for path in sorted(base.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta, answer = _parse_frontmatter(text)

        faq_id = meta.get("id") or path.stem
        question_variants = meta.get("question_variants") or []
        if not question_variants:
            raise FAQFormatError(f"question_variants is required in {path.name}")

        entry = FAQEntry(
            faq_id=str(faq_id),
            question_variants=[str(x).strip() for x in question_variants if str(x).strip()],
            answer=answer,
            tags=[str(x).strip() for x in meta.get("tags", [])],
            confirmed_by_human=bool(meta.get("confirmed_by_human", False)),
            confirmed_at=meta.get("confirmed_at"),
            freshness_hours=meta.get("freshness_hours"),
            escalate_if_stale=bool(meta.get("escalate_if_stale", False)),
            source_file=path.name,
            metadata=meta,
        )
        entries.append(entry)

    return entries
