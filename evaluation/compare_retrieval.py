"""Compare search methods using questions whose correct chunks are already known."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Literal

from evaluation.datasets import load_retrieval_gold
from evaluation.models import RetrievalCaseResult
from evaluation.retrieval_metrics import evaluate_retrieval
from retrieval.corpus import load_chunks_jsonl
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.models import RetrievalCandidate
from retrieval.query_processing import MetadataFilter, process_query
from retrieval.stages import STAGES, RetrievalStage, Searcher, build_retriever


def compare_stage(
    stage: RetrievalStage,
    *,
    chunks_path: Path,
    gold_path: Path,
    k: int,
    query_mode: Literal["raw", "processed"] = "raw",
) -> tuple[list[RetrievalCaseResult], list[float]]:
    """Test one search method and return its results and search times.

    The first search time can be much longer because it may include loading the model.
    """
    chunks = load_chunks_jsonl(chunks_path)
    cases = load_retrieval_gold(gold_path)
    searchers: dict[tuple[tuple[str, str, str, str], ...], Searcher] = {
        (): build_retriever(chunks, stage)
    }
    latencies_ms: list[float] = []

    def timed_search(query: str, limit: int) -> Sequence[RetrievalCandidate]:
        """Run one search and record how many milliseconds it takes."""
        start = perf_counter()
        plan = process_query(query)
        active_filters: list[MetadataFilter] = (
            plan.active_filters() if query_mode == "processed" else []
        )
        filter_key = tuple(
            (item.field, item.operator, item.value, item.source)
            for item in active_filters
        )
        searcher = searchers.get(filter_key)
        if searcher is None:
            searcher = build_retriever(chunks, stage, filters=active_filters)
            searchers[filter_key] = searcher
        search_queries = plan.search_queries if query_mode == "processed" else [query]
        rankings = {
            f"query_{index}": searcher.search(search_query, limit=limit)
            for index, search_query in enumerate(search_queries)
        }
        candidates = (
            next(iter(rankings.values()), [])
            if len(rankings) <= 1
            else reciprocal_rank_fusion(rankings, limit=limit)
        )
        latencies_ms.append((perf_counter() - start) * 1000)
        return candidates

    return evaluate_retrieval(cases, timed_search, k=k), latencies_ms


def main() -> None:
    """Read the command options, test each chosen search method, and print the results."""
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
        "--query-modes",
        nargs="+",
        choices=("raw", "processed"),
        default=["raw"],
        help="Pass both raw and processed to expose per-case gains and regressions.",
    )
    parser.add_argument(
        "--chunks", type=Path, default=Path("sample_corpus/chunks.jsonl")
    )
    parser.add_argument(
        "--gold", type=Path, default=Path("sample_corpus/retrieval_gold.jsonl")
    )
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")

    print(f"stage\tquery_mode\tRecall@{args.k}\tMRR@{args.k}\tfirst_query_ms\twarm_median_ms")
    for stage in args.stages:
        for query_mode in args.query_modes:
            results, latencies = compare_stage(
                stage,
                chunks_path=args.chunks,
                gold_path=args.gold,
                k=args.k,
                query_mode=query_mode,
            )
            if not results:
                parser.error("gold dataset must contain at least one case")

            warm_median = median(latencies[1:]) if len(latencies) > 1 else 0.0
            # Formula: Average Recall@k = sum of every question's Recall@k / number of questions
            # Formula: MRR@k = sum of every question's Reciprocal Rank / number of questions
            print(
                f"{stage}\t{query_mode}\t{mean(item.recall_at_k for item in results):.3f}"
                f"\t{mean(item.reciprocal_rank for item in results):.3f}"
                f"\t{latencies[0]:.3f}\t{warm_median:.3f}"
            )
            cases_by_id = {case.id: case for case in load_retrieval_gold(args.gold)}
            for item in results:
                print(
                    f"  {item.query_id}: recall={item.recall_at_k:.3f}, "
                    f"RR={item.reciprocal_rank:.3f}, "
                    f"hits={','.join(item.retrieved_chunk_ids)}"
                )
                if query_mode == "processed":
                    plan = process_query(cases_by_id[item.query_id].query)
                    print(
                        "    query_variants="
                        f"{plan.search_queries}; filters="
                        f"{[filter_.model_dump() for filter_ in plan.filters]}"
                    )
            if all(
                item.recall_at_k == 1.0 and item.reciprocal_rank == 1.0
                for item in results
            ):
                print(
                    "  No headroom on this gold set; add harder queries before judging new stages."
                )


if __name__ == "__main__":
    main()
