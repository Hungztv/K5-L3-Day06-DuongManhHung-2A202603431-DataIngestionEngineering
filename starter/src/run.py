"""Pipeline entry point: ingest corpus, embed, store, benchmark three strategies.

Vector store: Qdrant local (QdrantClient with path=, no server/Docker required).
Embedding:    sentence-transformers (local, no API key required).
Strategies:
    A — fixed-size chunks (500 chars, overlap 50)
    B — paragraph-based chunks (max 800 chars)
    C — same as B + enriched chapter/document_type/language metadata (enables filter)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STARTER_DIR = Path(__file__).parent.parent
DATA_DIR = STARTER_DIR / "data"
RESULTS_DIR = STARTER_DIR / "results"
BENCHMARK_FILE = DATA_DIR / "benchmark.json"
CORPUS_PDF = DATA_DIR / "openstax-introduction-sociology-3e.pdf"
DOCUMENT_ID = "openstax-introduction-sociology-3e"
SOURCE_ID = "openstax-introduction-sociology-3e"


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _load_env() -> tuple[str, str]:
    """Load .env, return (qdrant_dir, embedding_model)."""
    env_path = STARTER_DIR / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
    qdrant_dir = os.environ.get("QDRANT_DIR", "qdrant_db")
    model_name = os.environ.get(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    # Resolve relative paths from STARTER_DIR
    qdrant_path = Path(qdrant_dir)
    if not qdrant_path.is_absolute():
        qdrant_path = STARTER_DIR / qdrant_path
    return str(qdrant_path), model_name


def _load_benchmark() -> list[dict]:
    if not BENCHMARK_FILE.is_file():
        raise FileNotFoundError(
            f"Benchmark file not found: {BENCHMARK_FILE}\n"
            "Create data/benchmark.json with at least 20 questions."
        )
    with open(BENCHMARK_FILE, encoding="utf-8") as f:
        questions = json.load(f)
    dev = [q for q in questions if q.get("split") == "dev"]
    test = [q for q in questions if q.get("split") == "test"]
    print(f"[benchmark] {len(questions)} questions loaded: {len(dev)} dev, {len(test)} test")
    if len(questions) < 20:
        raise ValueError(f"Need ≥20 benchmark questions, found {len(questions)}")
    if len(dev) < 15:
        raise ValueError(f"Need ≥15 dev questions, found {len(dev)}")
    if len(test) < 5:
        raise ValueError(f"Need ≥5 test questions, found {len(test)}")
    return questions


def _ingest_pages() -> list:
    from .ingestion import load_pdf

    if not CORPUS_PDF.is_file():
        raise FileNotFoundError(f"PDF not found: {CORPUS_PDF}")

    print(f"[ingestion] Reading {CORPUS_PDF.name} ...")
    pages = load_pdf(CORPUS_PDF, document_id=DOCUMENT_ID, source=SOURCE_ID)
    content_pages = [p for p in pages if p.text.strip()]
    print(f"[ingestion] {len(pages)} physical pages, {len(content_pages)} with extractable text")
    if len(content_pages) < 30:
        raise ValueError(
            f"Need ≥30 content pages, found {len(content_pages)}. Check PDF extraction."
        )
    return pages


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def _build_strategy_a(pages: list) -> list:
    """Strategy A: fixed-size windows (500 chars, 50 overlap)."""
    from .chunking import fixed_size

    chunks = []
    for page in pages:
        if page.text.strip():
            chunks.extend(fixed_size(page, size=500, overlap=50))
    print(f"[strategy-A] fixed_size(500, overlap=50) → {len(chunks)} chunks")
    return chunks


def _build_strategy_b(pages: list) -> list:
    """Strategy B: paragraph grouping (max 800 chars per chunk)."""
    from .chunking import by_paragraph

    chunks = []
    for page in pages:
        if page.text.strip():
            chunks.extend(by_paragraph(page, max_size=800))
    print(f"[strategy-B] by_paragraph(max=800) → {len(chunks)} chunks")
    return chunks


def _build_strategy_c(pages: list) -> list:
    """Strategy C: same paragraph chunking + chapter/type/language metadata.

    Extra metadata fields allow metadata-filtered search, tested by benchmark
    questions that carry a 'filter' key.

    Chapter ranges are approximate for OpenStax Introduction to Sociology 3e.
    """
    from .chunking import by_paragraph

    CHAPTER_RANGES = [
        (1,   30,  "front_matter"),
        (31,  90,  "chapter_1"),
        (91,  150, "chapter_2"),
        (151, 210, "chapter_3"),
        (211, 270, "chapter_4"),
        (271, 330, "chapter_5"),
        (331, 390, "chapter_6"),
        (391, 450, "chapter_7"),
        (451, 510, "chapter_8"),
        (511, 570, "chapter_9"),
        (571, 630, "chapter_10"),
        (631, 664, "back_matter"),
    ]

    def _chapter(page_num: int) -> str:
        for start, end, name in CHAPTER_RANGES:
            if start <= page_num <= end:
                return name
        return "unknown"

    chunks = []
    for page in pages:
        if not page.text.strip():
            continue
        raw = by_paragraph(page, max_size=800)
        for c in raw:
            c.metadata["chapter"] = _chapter(page.page)
            c.metadata["document_type"] = "textbook"
            c.metadata["language"] = "en"
        chunks.extend(raw)
    print(f"[strategy-C] by_paragraph(max=800)+metadata → {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# Embed + store
# ---------------------------------------------------------------------------

def _embed_and_store(
    strategy: str,
    chunks: list,
    embedder,
    store,
) -> None:
    """Embed chunks in batches and upsert into Qdrant (idempotent)."""
    # Check if already ingested (skip re-embedding if data exists)
    existing = store.collection_count(strategy)
    if existing > 0:
        print(f"[{strategy}] Already has {existing} chunks — skipping re-ingest. "
              "Delete qdrant_db/ to force re-ingest.")
        return

    BATCH = 64
    all_vectors: list[list[float]] = []
    total = len(chunks)

    t0 = time.time()
    print(f"[{strategy}] Embedding {total} chunks (batch={BATCH})...")
    for i in range(0, total, BATCH):
        batch = [c.text for c in chunks[i : i + BATCH]]
        all_vectors.extend(embedder.embed_documents(batch))
        print(f"  {min(i + BATCH, total)}/{total}", end="\r", flush=True)

    elapsed = time.time() - t0
    print(f"\n[{strategy}] Embedded in {elapsed:.1f}s — upserting to Qdrant...")
    store.upsert(strategy, chunks, all_vectors)
    count = store.collection_count(strategy)
    print(f"[{strategy}] Collection now has {count} chunks ✓")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _run_benchmark(
    strategy: str,
    questions: list[dict],
    embedder,
    store,
) -> list[dict]:
    from .evaluation import evaluate_query

    traces = []
    for q in questions:
        query_vec = embedder.embed_query(q["question"])
        # Only strategy C supports metadata filtering
        filters = q.get("filter") if strategy == "C" else None
        hits = store.search(strategy, query_vec, top_k=3, filters=filters)
        trace = evaluate_query(q, hits)
        trace["strategy"] = strategy
        traces.append(trace)
    return traces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    1. Load corpus (≥30 content pages) and benchmark (≥20 questions).
    2. Build 3 strategies: A fixed-size, B paragraph, C paragraph+metadata.
    3. Embed with one shared model; upsert into Qdrant local (idempotent).
    4. Run dev + test benchmark on all strategies; save traces.
    5. Report metrics table and chunk statistics.
    """
    print("=" * 60)
    print("Day 06 — Data Ingestion Pipeline (Qdrant local)")
    print("=" * 60)

    qdrant_dir, model_name = _load_env()
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Load benchmark BEFORE any strategy decisions ---
    questions = _load_benchmark()
    dev_qs  = [q for q in questions if q.get("split") == "dev"]
    test_qs = [q for q in questions if q.get("split") == "test"]

    # --- Ingest corpus ---
    pages = _ingest_pages()

    # --- Embedder (shared across all strategies) ---
    print(f"\n[embedding] Loading model: {model_name}")
    from .embeddings import MyEmbedder
    embedder = MyEmbedder(model_name)
    print(f"[embedding] Dimensions: {embedder.dimensions}")

    # --- Vector store ---
    from .store import QdrantVectorStore
    store = QdrantVectorStore(qdrant_dir=qdrant_dir, dimensions=embedder.dimensions)
    print(f"[store] Qdrant persisted at: {qdrant_dir}")

    # --- Build, embed, store ---
    print("\n--- Strategy A: Fixed-size chunking ---")
    chunks_a = _build_strategy_a(pages)
    _embed_and_store("A", chunks_a, embedder, store)

    print("\n--- Strategy B: Paragraph chunking ---")
    chunks_b = _build_strategy_b(pages)
    _embed_and_store("B", chunks_b, embedder, store)

    print("\n--- Strategy C: Paragraph + metadata filter ---")
    chunks_c = _build_strategy_c(pages)
    _embed_and_store("C", chunks_c, embedder, store)

    # --- Benchmark: dev (use to pick best strategy) ---
    print("\n--- Benchmark: DEV split ---")
    all_traces: list[dict] = []
    for s in ("A", "B", "C"):
        t = _run_benchmark(s, dev_qs, embedder, store)
        all_traces.extend(t)
        h1 = sum(1 for x in t if x["hit_at_1"])
        h3 = sum(1 for x in t if x["evidence_hit_at_3"])
        n  = len(t)
        print(f"  {s} dev  Hit@1={h1}/{n}  EvidHit@3={h3}/{n}")

    # --- Benchmark: test (held-out, evaluate AFTER strategy fixed) ---
    print("\n--- Benchmark: TEST split ---")
    for s in ("A", "B", "C"):
        t = _run_benchmark(s, test_qs, embedder, store)
        all_traces.extend(t)
        h1 = sum(1 for x in t if x["hit_at_1"])
        h3 = sum(1 for x in t if x["evidence_hit_at_3"])
        n  = len(t)
        print(f"  {s} test Hit@1={h1}/{n}  EvidHit@3={h3}/{n}")

    # --- Save traces ---
    traces_path = RESULTS_DIR / "traces.json"
    with open(traces_path, "w", encoding="utf-8") as f:
        json.dump(all_traces, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[results] {len(all_traces)} traces → {traces_path}")

    # --- Summary ---
    from .evaluation import summarize
    summary = summarize(all_traces)
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[results] Summary → {summary_path}")

    # --- Print table ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for strat, splits in summary.get("strategies", {}).items():
        print(f"\nStrategy {strat}:")
        for split_name, m in splits.items():
            n  = m.get("n", 0)
            h1 = m.get("hit_at_1", 0)
            h3 = m.get("evidence_hit_at_3", 0)
            r1 = m.get("hit_at_1_rate", 0)
            r3 = m.get("evidence_hit_at_3_rate", 0)
            print(
                f"  [{split_name:7s}] "
                f"Hit@1={h1}/{n} ({r1:.1%})  "
                f"EvidHit@3={h3}/{n} ({r3:.1%})"
            )

    # --- Chunk statistics ---
    chunk_stats = {
        "A": {
            "strategy": "fixed_size(size=500, overlap=50)",
            "count": len(chunks_a),
            "avg_chars": round(sum(len(c.text) for c in chunks_a) / max(len(chunks_a), 1)),
        },
        "B": {
            "strategy": "by_paragraph(max_size=800)",
            "count": len(chunks_b),
            "avg_chars": round(sum(len(c.text) for c in chunks_b) / max(len(chunks_b), 1)),
        },
        "C": {
            "strategy": "by_paragraph(max_size=800) + chapter/document_type/language metadata",
            "count": len(chunks_c),
            "avg_chars": round(sum(len(c.text) for c in chunks_c) / max(len(chunks_c), 1)),
        },
    }
    stats_path = RESULTS_DIR / "chunk_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(chunk_stats, f, ensure_ascii=False, indent=2)

    print(f"\n[results] Chunk stats → {stats_path}")
    for s, st in chunk_stats.items():
        print(f"  {s}: {st['count']} chunks, avg {st['avg_chars']} chars — {st['strategy']}")

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
