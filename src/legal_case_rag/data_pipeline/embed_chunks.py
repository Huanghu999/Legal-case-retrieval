#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_INPUT = "benchmark_dataset/caselaw_benchmark_chunks_v1.jsonl"
DEFAULT_OUTPUT = "benchmark_dataset/caselaw_benchmark_chunks_v1_embedded.jsonl"
DEFAULT_API_BASE = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_API_KEY_ENV = "SILICONFLOW_API_KEY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings for legal RAG chunks via SiliconFlow.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input chunk JSONL.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output chunk JSONL with embedding field.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="OpenAI-compatible API base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name.")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="Environment variable that stores API key.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N input chunks.")
    parser.add_argument("--resume", action="store_true", help="Append and skip chunk_ids already present in output.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output file if it exists.")
    parser.add_argument("--max-retries", type=int, default=4, help="Retry attempts per batch.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds after each successful batch.")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and payload shape without calling API.")
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_jsonl(path: Path, limit: int | None = None) -> Any:
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if limit is not None and index >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def read_done_chunk_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            chunk_id = payload.get("chunk_id")
            if chunk_id:
                done.add(chunk_id)
    return done


def ensure_output_path(path: Path, resume: bool, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not resume and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --resume or --overwrite.")
    if path.exists() and overwrite and not resume:
        path.unlink()


def batched(items: list[dict[str, Any]], batch_size: int) -> Any:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def embedding_request(
    api_base: str,
    api_key: str,
    model: str,
    inputs: list[str],
    timeout: int = 120,
) -> list[list[float]]:
    url = api_base.rstrip("/") + "/embeddings"
    body = json.dumps({"model": model, "input": inputs}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected embedding response: {payload}")
    embeddings: list[list[float]] = []
    for item in sorted(data, key=lambda value: value.get("index", 0)):
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Missing embedding in response item: {item}")
        embeddings.append(embedding)
    if len(embeddings) != len(inputs):
        raise RuntimeError(f"Expected {len(inputs)} embeddings, got {len(embeddings)}")
    return embeddings


def embed_with_retries(args: argparse.Namespace, api_key: str, inputs: list[str]) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(args.max_retries):
        try:
            return embedding_request(args.api_base, api_key, args.model, inputs)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            wait = min(2 ** attempt, 20)
            time.sleep(wait)
    raise RuntimeError(f"Embedding batch failed after {args.max_retries} attempts: {last_error}")


def main() -> int:
    import sys
    _root = str(Path(__file__).resolve().parents[3])
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from src.legal_case_rag.runtime.env import load_project_env
    load_project_env()

    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    ensure_output_path(output_path, args.resume, args.overwrite)

    done_ids = read_done_chunk_ids(output_path) if args.resume else set()
    chunks: list[dict[str, Any]] = []
    skipped = 0
    for chunk in iter_jsonl(input_path, args.limit):
        if chunk.get("chunk_id") in done_ids:
            skipped += 1
            continue
        chunks.append(chunk)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "model": args.model,
        "api_base": args.api_base,
        "chunks_to_embed": len(chunks),
        "skipped_existing": skipped,
        "batch_size": args.batch_size,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        preview = [
            {
                "chunk_id": item.get("chunk_id"),
                "embedding_text_preview": (item.get("embedding_text") or item.get("chunk_text") or "")[:160],
            }
            for item in chunks[:3]
        ]
        print(json.dumps({"preview": preview}, ensure_ascii=False, indent=2))
        return 0

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Missing API key environment variable: {args.api_key_env}")

    progress = tqdm(total=len(chunks), desc="Embedding chunks", unit="chunk") if tqdm else None
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as output_file:
        for batch in batched(chunks, args.batch_size):
            inputs = [item.get("embedding_text") or item.get("chunk_text") or "" for item in batch]
            embeddings = embed_with_retries(args, api_key, inputs)
            for item, embedding in zip(batch, embeddings):
                item["embedding"] = embedding
                item["embedding_model"] = args.model
                item["embedding_dim"] = len(embedding)
                output_file.write(json_dumps(item) + "\n")
            output_file.flush()
            if progress:
                progress.update(len(batch))
                progress.set_postfix(dim=len(embeddings[0]) if embeddings else 0)
            if args.sleep:
                time.sleep(args.sleep)
    if progress:
        progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
