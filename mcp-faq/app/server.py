from __future__ import annotations

from fastmcp import FastMCP

from app.config import settings
from app.faq_loader import load_faqs
from app.retrieval import search_faqs

mcp = FastMCP(settings.server_name)
FAQS = load_faqs(settings.faq_dir)


@mcp.tool()
def faq_health() -> dict:
    """Return FAQ service status and basic counters."""
    return {
        "status": "ok",
        "server": settings.server_name,
        "faq_count": len(FAQS),
        "faq_dir": settings.faq_dir,
    }


@mcp.tool()
def faq_search(question: str, limit: int = 3) -> dict:
    """Search FAQ candidates for a user question."""
    results = search_faqs(question=question, entries=FAQS, min_score=settings.faq_min_score)
    return {
        "question": question,
        "matches": results[:limit],
        "total": len(results),
    }


@mcp.tool()
def faq_answer(question: str) -> dict:
    """Return the best FAQ answer for a user question."""
    results = search_faqs(question=question, entries=FAQS, min_score=settings.faq_min_score)
    if not results:
        return {
            "found": False,
            "question": question,
            "message": "No FAQ answer found. Escalate to human or ask for clarification.",
        }

    best = results[0]
    return {
        "found": True,
        "question": question,
        "answer": best["answer"],
        "faq_id": best["faq_id"],
        "score": best["score"],
        "confirmed_by_human": best["confirmed_by_human"],
        "confirmed_at": best["confirmed_at"],
        "source_file": best["source_file"],
        "tags": best["tags"],
    }


if __name__ == "__main__":
    mcp.run()
