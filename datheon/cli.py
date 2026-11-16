from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from datheon.ingestion import chunk_documents, load_documents, load_manifest
from datheon.vector_store import ChromaVectorStore, LocalSentenceEmbedder


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest client documents into ChromaDB")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--data-dir", type=Path, required=True)
    ingest_parser.add_argument("--manifest", type=Path, required=True)
    ingest_parser.add_argument("--batch-size", type=int, default=64)
    ingest_parser.add_argument("--chunk-size", type=int, default=900)
    ingest_parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = load_manifest(args.manifest)
    documents = load_documents(args.data_dir, manifest)
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.overlap)
    store = ChromaVectorStore(
        os.getenv("CHROMA_PATH", "./.chroma"),
        os.getenv("CHROMA_COLLECTION", "datheon_documents"),
        LocalSentenceEmbedder(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")),
    )
    store.upsert(chunks, batch_size=args.batch_size)
    elapsed = time.perf_counter() - started
    print(f"Documents: {len(manifest)} source files, {len(documents)} extracted records")
    print(f"Chunks indexed: {len(chunks)} in {elapsed:.2f}s")
    print(f"Throughput: {len(manifest) / elapsed:.2f} documents/s, {len(chunks) / elapsed:.2f} chunks/s")


if __name__ == "__main__":
    main()