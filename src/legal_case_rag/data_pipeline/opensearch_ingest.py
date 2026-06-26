#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_HOST = "https://localhost:9200"
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD_ENV = "OPENSEARCH_PASSWORD"
DEFAULT_CASE_INDEX = "caselaw_benchmark_cases_v1"
DEFAULT_CHUNK_INDEX = "caselaw_benchmark_chunks_v1"
DEFAULT_CASES = "benchmark_dataset/caselaw_benchmark_cases_v1.jsonl"
DEFAULT_CHUNKS = "benchmark_dataset/caselaw_benchmark_chunks_v1_embedded.jsonl"
DEFAULT_CHUNKS_FALLBACK = "benchmark_dataset/caselaw_benchmark_chunks_v1.jsonl"
DEFAULT_CASE_MAPPING = "benchmark_dataset/caselaw_benchmark_cases_v1_mapping.json"
DEFAULT_CHUNK_MAPPING = "benchmark_dataset/caselaw_benchmark_chunks_v1_mapping.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create OpenSearch indices and bulk ingest legal RAG JSONL files.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="OpenSearch host.")
    parser.add_argument("--username", default=DEFAULT_USERNAME, help="OpenSearch username.")
    parser.add_argument("--password-env", default=DEFAULT_PASSWORD_ENV, help="Environment variable for password.")
    parser.add_argument("--case-index", default=DEFAULT_CASE_INDEX, help="Case index name.")
    parser.add_argument("--chunk-index", default=DEFAULT_CHUNK_INDEX, help="Chunk index name.")
    parser.add_argument("--cases", default=DEFAULT_CASES, help="Case JSONL.")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS, help="Chunk JSONL, preferably with embeddings.")
    parser.add_argument("--case-mapping", default=DEFAULT_CASE_MAPPING, help="Case mapping JSON.")
    parser.add_argument("--chunk-mapping", default=DEFAULT_CHUNK_MAPPING, help="Chunk mapping JSON.")
    parser.add_argument("--batch-size", type=int, default=500, help="Bulk batch size.")
    parser.add_argument("--delete-existing", action="store_true", help="Delete indices before creating them.")
    parser.add_argument("--skip-create", action="store_true", help="Skip index creation.")
    parser.add_argument("--verify-tls", action="store_false", dest="insecure", help="Enable TLS certificate verification.")
    parser.set_defaults(insecure=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate files and counts without connecting.")
    return parser.parse_args()


class OpenSearchClient:
    def __init__(self, host: str, username: str, password: str, insecure: bool = True) -> None:
        self.host = host.rstrip("/")
        auth = f"{username}:{password}".encode("utf-8")
        self.auth_header = "Basic " + base64.b64encode(auth).decode("ascii")
        self.context = ssl._create_unverified_context() if insecure else None

    def request(
        self,
        method: str,
        path: str,
        body: Any | None = None,
        content_type: str = "application/json",
        expected: tuple[int, ...] = (200, 201),
    ) -> Any:
        data = None
        if body is not None:
            if isinstance(body, bytes):
                data = body
            elif isinstance(body, str):
                data = body.encode("utf-8")
            else:
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.host + path,
            data=data,
            method=method,
            headers={"Authorization": self.auth_header, "Content-Type": content_type},
        )
        try:
            with urllib.request.urlopen(request, context=self.context, timeout=120) as response:
                raw = response.read()
                if response.status not in expected:
                    raise RuntimeError(f"Unexpected status {response.status}: {raw[:500]!r}")
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            if exc.code in expected:
                return json.loads(raw) if raw else {}
            raise RuntimeError(f"{method} {path} failed: HTTP {exc.code}: {raw[:1200]}") from exc

    def exists(self, index: str) -> bool:
        request = urllib.request.Request(
            self.host + f"/{index}",
            method="HEAD",
            headers={"Authorization": self.auth_header},
        )
        try:
            with urllib.request.urlopen(request, context=self.context, timeout=30):
                return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise


def iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def choose_chunk_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = Path(DEFAULT_CHUNKS_FALLBACK)
    if fallback.exists():
        return fallback
    return path


def create_index(client: OpenSearchClient, index: str, mapping: dict[str, Any], delete_existing: bool) -> None:
    exists = client.exists(index)
    if exists and delete_existing:
        client.request("DELETE", f"/{index}", expected=(200,))
        exists = False
    if exists:
        print(json.dumps({"index": index, "status": "exists"}, ensure_ascii=False))
        return
    client.request("PUT", f"/{index}", mapping, expected=(200,))
    print(json.dumps({"index": index, "status": "created"}, ensure_ascii=False))


def bulk_ingest(client: OpenSearchClient, index: str, path: Path, id_field: str, batch_size: int) -> dict[str, int]:
    total = count_jsonl(path)
    progress = tqdm(total=total, desc=f"Bulk ingest {index}", unit="doc") if tqdm else None
    sent = 0
    errors = 0
    batch: list[str] = []

    def flush() -> None:
        nonlocal sent, errors, batch
        if not batch:
            return
        payload = "\n".join(batch) + "\n"
        response = client.request(
            "POST",
            "/_bulk",
            payload,
            content_type="application/x-ndjson",
            expected=(200,),
        )
        if response.get("errors"):
            for item in response.get("items", [])[:5]:
                action = item.get("index", {})
                if "error" in action:
                    print(json.dumps({"bulk_error": action}, ensure_ascii=False))
                    errors += 1
        sent += len(batch) // 2
        if progress:
            progress.update(len(batch) // 2)
        batch = []

    for doc in iter_jsonl(path):
        doc_id = doc.get(id_field)
        action = {"index": {"_index": index, "_id": doc_id}}
        batch.append(json.dumps(action, ensure_ascii=False, separators=(",", ":")))
        batch.append(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
        if len(batch) // 2 >= batch_size:
            flush()
    flush()
    if progress:
        progress.close()
    return {"sent": sent, "errors": errors}


def main() -> int:
    import sys
    _root = str(Path(__file__).resolve().parents[3])
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from src.legal_case_rag.runtime.env import load_project_env
    load_project_env()

    args = parse_args()
    cases_path = Path(args.cases)
    chunks_path = choose_chunk_path(Path(args.chunks))
    case_mapping_path = Path(args.case_mapping)
    chunk_mapping_path = Path(args.chunk_mapping)

    summary = {
        "host": args.host,
        "case_index": args.case_index,
        "chunk_index": args.chunk_index,
        "cases": str(cases_path),
        "chunks": str(chunks_path),
        "case_count": count_jsonl(cases_path) if cases_path.exists() else 0,
        "chunk_count": count_jsonl(chunks_path) if chunks_path.exists() else 0,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    load_json(case_mapping_path)
    load_json(chunk_mapping_path)

    if args.dry_run:
        return 0

    password = os.environ.get(args.password_env)
    if not password:
        raise EnvironmentError(f"Missing OpenSearch password environment variable: {args.password_env}")
    client = OpenSearchClient(args.host, args.username, password, insecure=args.insecure)

    if not args.skip_create:
        create_index(client, args.case_index, load_json(case_mapping_path), args.delete_existing)
        create_index(client, args.chunk_index, load_json(chunk_mapping_path), args.delete_existing)

    case_result = bulk_ingest(client, args.case_index, cases_path, "doc_id", args.batch_size)
    chunk_result = bulk_ingest(client, args.chunk_index, chunks_path, "chunk_id", args.batch_size)
    print(json.dumps({"cases": case_result, "chunks": chunk_result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
