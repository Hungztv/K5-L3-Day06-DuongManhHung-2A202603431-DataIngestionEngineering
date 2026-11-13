"""Evaluate retrieved evidence and preserve per-query traces."""
from typing import Any


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check source/page and enough gold evidence; keep hit@1 and hit@3 trace.

    A hit is counted when ANY hit in top-k contains the gold page number in the
    ``page`` field AND the gold source_id in the ``source`` field.
    Multiple accepted gold pages are supported via ``gold_pages`` list.

    Args:
        question: A benchmark question dict with keys:
            - id, split, question, expected_answer
            - gold_source_id, gold_page (int or list[int])
            - evidence_quote (str)
            - filter (optional dict)
        hits: Ordered list of search results (top-3), each with keys:
            chunk_id, text, source, page, metadata, score

    Returns:
        A trace dict with hit@1, evidence_hit@3, top-3 results and labels.
    """
    # Normalize accepted gold pages (support multiple gold pages)
    gold_source = question.get("gold_source_id", "")
    raw_gold_page = question.get("gold_page")
    if isinstance(raw_gold_page, list):
        gold_pages = set(raw_gold_page)
    elif raw_gold_page is not None:
        gold_pages = {int(raw_gold_page)}
    else:
        gold_pages = set()

    evidence_quote = question.get("evidence_quote", "").strip().lower()

    def _is_hit(hit: dict[str, Any]) -> bool:
        """Return True if the hit matches the gold source and page."""
        source_match = hit.get("source", "") == gold_source
        page_match = hit.get("page") in gold_pages if gold_pages else False

        # Secondary check: evidence quote in chunk text (normalized)
        quote_match = False
        if evidence_quote:
            chunk_text = hit.get("text", "").strip().lower()
            # Accept partial match of first 60 chars of evidence quote
            snippet = evidence_quote[:60]
            quote_match = snippet in chunk_text

        return (source_match and page_match) or (source_match and quote_match)

    # Evaluate hit@1 (only the first result)
    hit_at_1 = False
    if hits:
        hit_at_1 = _is_hit(hits[0])

    # Evaluate evidence hit@3 (any of top-3)
    evidence_hit_at_3 = any(_is_hit(h) for h in hits[:3])

    # Build per-hit labels
    hit_labels = []
    for rank, h in enumerate(hits[:3], start=1):
        is_hit = _is_hit(h)
        hit_labels.append({
            "rank": rank,
            "chunk_id": h.get("chunk_id", ""),
            "source": h.get("source", ""),
            "page": h.get("page"),
            "score": h.get("score"),
            "is_hit": is_hit,
            "text_snippet": h.get("text", "")[:200],
        })

    return {
        "question_id": question.get("id", ""),
        "split": question.get("split", "dev"),
        "question": question.get("question", ""),
        "gold_source_id": gold_source,
        "gold_page": list(gold_pages),
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "top3": hit_labels,
        "filter_used": question.get("filter"),
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split.

    Args:
        traces: List of trace dicts, each with 'strategy' key added by run.py,
                plus the fields from evaluate_query().

    Returns:
        Summary dict with per-strategy, per-split metrics and totals.
    """
    if not traces:
        return {"error": "No traces to summarize", "strategies": {}}

    # Group by strategy then split
    from collections import defaultdict
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for trace in traces:
        strategy = trace.get("strategy", "unknown")
        split = trace.get("split", "dev")
        groups[strategy][split].append(trace)

    summary: dict[str, Any] = {"strategies": {}}

    for strategy, splits in groups.items():
        strategy_summary: dict[str, Any] = {}

        all_traces_for_strategy = []
        for split, split_traces in splits.items():
            n = len(split_traces)
            h1 = sum(1 for t in split_traces if t.get("hit_at_1"))
            h3 = sum(1 for t in split_traces if t.get("evidence_hit_at_3"))
            strategy_summary[split] = {
                "n": n,
                "hit_at_1": h1,
                "hit_at_1_rate": round(h1 / n, 4) if n else 0.0,
                "evidence_hit_at_3": h3,
                "evidence_hit_at_3_rate": round(h3 / n, 4) if n else 0.0,
            }
            all_traces_for_strategy.extend(split_traces)

        # Overall (dev + test combined)
        total_n = len(all_traces_for_strategy)
        total_h1 = sum(1 for t in all_traces_for_strategy if t.get("hit_at_1"))
        total_h3 = sum(1 for t in all_traces_for_strategy if t.get("evidence_hit_at_3"))
        strategy_summary["overall"] = {
            "n": total_n,
            "hit_at_1": total_h1,
            "hit_at_1_rate": round(total_h1 / total_n, 4) if total_n else 0.0,
            "evidence_hit_at_3": total_h3,
            "evidence_hit_at_3_rate": round(total_h3 / total_n, 4) if total_n else 0.0,
        }

        summary["strategies"][strategy] = strategy_summary

    return summary
