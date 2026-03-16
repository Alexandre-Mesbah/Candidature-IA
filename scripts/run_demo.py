from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from study_genie_extract.pipeline import build_index, load_documents
from study_genie_extract.retrieval import retrieve_top_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public Study Genie extract end to end.")
    parser.add_argument(
        "--documents",
        default=str(ROOT / "data" / "sample_documents.jsonl"),
        help="Path to a JSONL file with id, title, and text fields.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Question used for retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to return.",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="Embedding model name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(args.documents)
    index = build_index(documents, model=args.model)
    hits = retrieve_top_k(args.query, index, top_k=args.top_k, model=args.model)

    print(f"Documents loaded: {len(documents)}")
    print(f"Indexed chunks: {len(index)}")
    print(f"Query: {args.query}")
    print("")

    for rank, hit in enumerate(hits, start=1):
        print(
            f"{rank}. {hit.chunk.document_id} | "
            f"chunk={hit.chunk.chunk_index} | "
            f"similarity={hit.similarity:.4f}"
        )
        print(f"   title: {hit.chunk.document_title}")
        print(f"   text: {hit.chunk.content}")
        print("")


if __name__ == "__main__":
    main()
