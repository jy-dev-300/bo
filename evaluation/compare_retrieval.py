"""Run opt-in retrieval-stage comparisons against the labeled gold queries."""

import argparse
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from evaluation.datasets import load_retrieval_gold
from evaluation.models import RetrievalCaseResult
from evaluation.retrieval_metrics import evaluate_retrieval
from retrieval.corpus import load_chunks_jsonl
from retrieval.stages import STAGES, RetrievalStage, build_retriever


def compare_stage(
    stage: RetrievalStage,
    *,
    chunks_path: Path,
    gold_path: Path,
    k: int,
) -> tuple[list[RetrievalCaseResult], list[float]]:
    """Return per-query quality and latency; first latency includes lazy model load."""
    chunks = load_chunks_jsonl(chunks_path)
    cases = load_retrieval_gold(gold_path)
    searcher = build_retriever(chunks, stage)
    latencies_ms: list[float] = []

    def timed_search(query: str, limit: int):
        start = perf_counter()
        candidates = searcher.search(query, limit=limit)
        latencies_ms.append((perf_counter() - start) * 1000)
        return candidates

    return evaluate_retrieval(cases, timed_search, k=k), latencies_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=["bm25"],
        help="Model-backed stages download weights on first use; default runs BM25 only.",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--chunks", type=Path, default=Path("sample_corpus/chunks.jsonl")
    )
    parser.add_argument(
        "--gold", type=Path, default=Path("sample_corpus/retrieval_gold.jsonl")
    )
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")

    print(f"stage\tRecall@{args.k}\tMRR@{args.k}\tfirst_query_ms\twarm_median_ms")
    for stage in args.stages:
        results, latencies = compare_stage(
            stage,
            chunks_path=args.chunks,
            gold_path=args.gold,
            k=args.k,
        )
        if not results:
            parser.error("gold dataset must contain at least one case")

        warm_median = median(latencies[1:]) if len(latencies) > 1 else 0.0
        print(
            f"{stage}\t{mean(item.recall_at_k for item in results):.3f}"
            f"\t{mean(item.reciprocal_rank for item in results):.3f}"
            f"\t{latencies[0]:.3f}\t{warm_median:.3f}"
        )
        for item in results:
            print(
                f"  {item.query_id}: recall={item.recall_at_k:.3f}, "
                f"RR={item.reciprocal_rank:.3f}, "
                f"hits={','.join(item.retrieved_chunk_ids)}"
            )
        if all(item.recall_at_k == 1.0 and item.reciprocal_rank == 1.0 for item in results):
            print("  No headroom on this gold set; add harder queries before judging new stages.")


if __name__ == "__main__":
    main()
