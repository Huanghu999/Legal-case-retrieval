from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("benchmark_dataset")
DEFAULT_CASE_INDEX = "caselaw_benchmark_cases_v1"
DEFAULT_CHUNK_INDEX = "caselaw_benchmark_chunks_v1"
DEFAULT_EMBEDDING_KEY_ENV = "SILICONFLOW_API_KEY"
DEFAULT_OPENSEARCH_PASSWORD_ENV = "OPENSEARCH_PASSWORD"


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def run_step(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild CaseLaw-Bench RAG dataset, embeddings, and OpenSearch indices.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-index", default=DEFAULT_CASE_INDEX)
    parser.add_argument("--chunk-index", default=DEFAULT_CHUNK_INDEX)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-key-env", default=DEFAULT_EMBEDDING_KEY_ENV)
    parser.add_argument("--opensearch-password-env", default=DEFAULT_OPENSEARCH_PASSWORD_ENV)
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    missing_env: list[str] = []
    if not args.skip_embedding and not os.getenv(args.embedding_key_env):
        missing_env.append(args.embedding_key_env)
    if not args.skip_ingest and not os.getenv(args.opensearch_password_env):
        missing_env.append(args.opensearch_password_env)
    if missing_env:
        names = ", ".join(missing_env)
        print(
            f"Missing required environment variable(s): {names}\n"
            "Set them in this PowerShell session, or create a .env file in the project root.\n"
            "Example:\n"
            '$env:SILICONFLOW_API_KEY="your SiliconFlow API Key"\n'
            '$env:OPENSEARCH_PASSWORD="your OpenSearch admin password"',
            file=sys.stderr,
        )
        return 2

    build_cmd = [
        sys.executable,
        "-m",
        "src.legal_case_rag.data_pipeline.benchmark_dataset_builder",
        "--output-dir",
        str(args.output_dir),
        "--case-index-name",
        args.case_index,
        "--chunk-index-name",
        args.chunk_index,
        "--overwrite",
    ]
    if args.limit is not None:
        build_cmd.extend(["--limit", str(args.limit)])
    run_step(build_cmd)

    chunk_jsonl = args.output_dir / f"{args.chunk_index}.jsonl"
    embedded_jsonl = args.output_dir / f"{args.chunk_index}_embedded.jsonl"
    if not args.skip_embedding:
        run_step(
            [
                sys.executable,
                "-m",
                "src.legal_case_rag.data_pipeline.embed_chunks",
                "--input",
                str(chunk_jsonl),
                "--output",
                str(embedded_jsonl),
                "--overwrite",
                "--batch-size",
                str(args.batch_size),
                "--api-key-env",
                args.embedding_key_env,
            ]
        )

    if not args.skip_ingest:
        run_step(
            [
                sys.executable,
                "-m",
                "src.legal_case_rag.data_pipeline.opensearch_ingest",
                "--case-index",
                args.case_index,
                "--chunk-index",
                args.chunk_index,
                "--cases",
                str(args.output_dir / f"{args.case_index}.jsonl"),
                "--chunks",
                str(embedded_jsonl if embedded_jsonl.exists() else chunk_jsonl),
                "--case-mapping",
                str(args.output_dir / f"{args.case_index}_mapping.json"),
                "--chunk-mapping",
                str(args.output_dir / f"{args.chunk_index}_mapping.json"),
                "--password-env",
                args.opensearch_password_env,
                "--delete-existing",
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
