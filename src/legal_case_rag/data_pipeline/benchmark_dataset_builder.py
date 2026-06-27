#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_INPUT = "data/caselaw-benchmark release/data/corpus.jsonl"
DEFAULT_OUTPUT_DIR = "benchmark_dataset"
DEFAULT_CASE_INDEX = "caselaw_benchmark_cases_v1"
DEFAULT_CHUNK_INDEX = "caselaw_benchmark_chunks_v1"
DEFAULT_SCHEMA_VERSION = "caselaw_benchmark_rag_v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DIM = 1024


SECTION_TITLES = {
    "case_profile": "案件画像",
    "fine_issue": "细争点",
    "focus": "争议焦点",
    "claims": "诉称",
    "defense": "辩称",
    "facts": "查明事实",
    "reasoning": "本院认为",
    "judgment": "裁判结果",
    "statutes": "引用法条",
}

SECTION_WEIGHTS = {
    "fine_issue": 1.55,
    "focus": 1.45,
    "reasoning": 1.20,
    "facts": 1.0,
    "claims": 0.70,
    "judgment": 0.75,
    "case_profile": 1.20,
    "statutes": 0.45,
    "defense": 0.85,
}

PLEADING_TAIL_RE = re.compile(r"(当事人围绕诉讼请求依法提交了证据|本院组织当事人进行了证据交换|对当事人无异议的证据)")
PLEADING_MARKER_RE = re.compile(
    r"(?=(?:原告|上诉人|申请人|再审申请人|反诉原告)[^。；\n\r]{0,30}(?:诉称|补充陈述|上诉请求|请求)|"
    r"(?:被告|被上诉人|被申请人|第三人|反诉被告)[^。；\n\r]{0,30}(?:辩称|述称|答辩称))"
)
DEFENSE_MARKER_RE = re.compile(r"^(?:被告|被上诉人|被申请人|第三人|反诉被告)[^。；\n\r]{0,30}(?:辩称|述称|答辩称)")
REASONING_BOUNDARY_RE = re.compile(r"(?=(?:本案争议焦点为|本院认为|首先|其次|再次|最后|关于|对于|故|综上))")
FACT_DATE_BOUNDARY_RE = re.compile(r"(?=(?:19|20)\d{2}年\d{1,2}月\d{1,2}日)")


@dataclass
class BuildStats:
    cases: int = 0
    chunks: int = 0
    section_counts: dict[str, int] = field(default_factory=dict)

    def add_section(self, section_type: str) -> None:
        self.section_counts[section_type] = self.section_counts.get(section_type, 0) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG JSONL files from CaseLaw-Bench corpus.jsonl.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-index-name", default=DEFAULT_CASE_INDEX)
    parser.add_argument("--chunk-index-name", default=DEFAULT_CHUNK_INDEX)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-version", default="")
    parser.add_argument("--schema-version", default=DEFAULT_SCHEMA_VERSION)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "；".join(as_text(item) for item in value if as_text(item))
    if isinstance(value, dict):
        return "；".join(as_text(item) for item in value.values() if as_text(item))
    return str(value)


def clean_text(text: str) -> str:
    return "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())


def trim_pleading_tail(text: str) -> str:
    match = PLEADING_TAIL_RE.search(text or "")
    if match:
        return text[: match.start()]
    return text or ""


def split_pleadings_sections(text: str) -> list[tuple[str, str]]:
    cleaned = clean_text(trim_pleading_tail(text))
    if not cleaned:
        return []

    starts = [match.start() for match in PLEADING_MARKER_RE.finditer(cleaned)]
    if not starts:
        return [("claims", cleaned)]

    sections: list[tuple[str, str]] = []
    if starts[0] > 0:
        sections.append(("claims", clean_text(cleaned[: starts[0]])))

    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(cleaned)
        segment = clean_text(cleaned[start:end])
        if not segment:
            continue
        section_type = "defense" if DEFENSE_MARKER_RE.match(segment) else "claims"
        sections.append((section_type, segment))

    return [(section_type, section_text) for section_type, section_text in sections if section_text]


def court_region_from_court(court_name: str) -> str:
    court_name = court_name or ""
    if "上海" in court_name:
        return "上海"
    return ""


def statute_text(statutes: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in statutes or []:
        if not isinstance(item, dict):
            parts.append(as_text(item))
            continue
        law = item.get("法律") or ""
        article = item.get("条")
        original = item.get("原文") or ""
        parts.append(f"{law} 第{article}条 {original}".strip())
    return "\n".join(part for part in parts if part)


def build_full_text(row: dict[str, Any]) -> str:
    focus = row.get("争议焦点") or {}
    analysis = focus.get("焦点评析") or {}
    fine = row.get("细争点") or {}
    paragraphs = row.get("段落") or {}
    parts = [
        f"案号：{row.get('案号') or ''}",
        f"法院：{row.get('法院') or ''}",
        f"案由：{row.get('案由') or ''}",
        f"法律关系：{row.get('法律关系') or ''}",
        f"标的物类型：{row.get('标的物类型') or ''}",
        f"焦点标签：{as_text(focus.get('焦点标签'))}",
        f"案情核心：{analysis.get('案情核心') or ''}",
        f"法律争点：{analysis.get('法律争点') or ''}",
        f"裁判要旨：{analysis.get('裁判要旨') or ''}",
        f"主叶子：{fine.get('主叶子') or ''}",
        f"细争点：{as_text(fine.get('细争点'))}",
        f"裁判规则争点：{fine.get('裁判规则争点') or ''}",
        f"诉称：{paragraphs.get('诉称') or ''}",
        f"查明事实：{paragraphs.get('查明事实') or ''}",
        f"本院认为：{paragraphs.get('本院认为') or ''}",
        f"裁判结果：{paragraphs.get('裁判结果') or ''}",
        f"引用法条：{statute_text(row.get('引用法条') or [])}",
    ]
    return clean_text("\n\n".join(part for part in parts if part and not part.endswith("：")))


def build_case_doc(row: dict[str, Any], source_file: str, source_row_id: int, schema_version: str) -> dict[str, Any]:
    doc_id = row["doc_id"]
    full_text = build_full_text(row)
    parties = row.get("当事人") or []
    return {
        "doc_id": doc_id,
        "case_code": row.get("案号") or doc_id.split("#", 1)[0],
        "case_name": row.get("案号") or doc_id,
        "court_name": row.get("法院") or "",
        "court_region": court_region_from_court(row.get("法院") or ""),
        "case_type": "民事",
        "trial_level": row.get("审级") or "",
        "reason": row.get("案由") or "买卖合同纠纷",
        "judge_date": row.get("裁判日期") or None,
        "publish_date": None,
        "litigants": parties,
        "statutes": [as_text(item) for item in (row.get("引用法条") or [])],
        "full_text": full_text,
        "full_text_hash": sha256_text(full_text),
        "source_file": source_file,
        "source_row_id": source_row_id,
        "schema_version": schema_version,
    }


def core_issue_text(row: dict[str, Any]) -> str:
    fine = row.get("细争点") or {}
    focus = row.get("争议焦点") or {}
    parts = [
        fine.get("主叶子") or "",
        as_text(fine.get("细争点")),
        as_text(focus.get("焦点标签")),
    ]
    return "；".join(part for part in parts if part)


def build_sections(row: dict[str, Any], case_doc: dict[str, Any]) -> list[tuple[str, str]]:
    focus = row.get("争议焦点") or {}
    analysis = focus.get("焦点评析") or {}
    fine = row.get("细争点") or {}
    paragraphs = row.get("段落") or {}
    profile = "\n".join(
        [
            f"案由：{case_doc['reason']}",
            f"审级：{case_doc['trial_level']}",
            f"法院：{case_doc['court_name']}",
            f"法律关系：{row.get('法律关系') or ''}",
            f"标的物类型：{row.get('标的物类型') or ''}",
            f"裁判结果标签：{row.get('裁判结果_标签') or ''}",
        ]
    )
    fine_issue = "\n".join(
        [
            f"主叶子：{fine.get('主叶子') or ''}",
            f"细争点：{as_text(fine.get('细争点'))}",
            f"裁判规则争点：{fine.get('裁判规则争点') or ''}",
        ]
    )
    focus_text = "\n".join(
        [
            f"焦点标签：{as_text(focus.get('焦点标签'))}",
            f"案情核心：{analysis.get('案情核心') or ''}",
            f"法律争点：{analysis.get('法律争点') or ''}",
            f"裁判要旨：{analysis.get('裁判要旨') or ''}",
            f"焦点原文：{focus.get('焦点原文') or ''}",
        ]
    )
    sections = [
        ("case_profile", profile),
        ("fine_issue", fine_issue),
        ("focus", focus_text),
    ]
    sections.extend(split_pleadings_sections(paragraphs.get("诉称") or ""))
    sections.extend(
        [
            ("facts", paragraphs.get("查明事实") or ""),
            ("reasoning", paragraphs.get("本院认为") or ""),
            ("judgment", paragraphs.get("裁判结果") or ""),
            ("statutes", statute_text(row.get("引用法条") or [])),
        ]
    )
    return sections


def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        end = min(len(text), pos + max_chars)
        if end < len(text):
            split_at = max(text.rfind("。", pos + max_chars // 2, end), text.rfind("\n", pos + max_chars // 2, end))
            if split_at > pos:
                end = split_at + 1
        chunk = clean_text(text[pos:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        pos = max(pos + 1, end - overlap)
    return chunks


def split_marker_units(text: str, marker_re: re.Pattern[str]) -> list[str]:
    units: list[str] = []
    for paragraph in clean_text(text).split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        starts = [match.start() for match in marker_re.finditer(paragraph)]
        if not starts:
            units.append(paragraph)
            continue
        if starts[0] > 0:
            units.append(paragraph[: starts[0]].strip())
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(paragraph)
            units.append(paragraph[start:end].strip())
    return [unit for unit in units if unit]


def pack_units(units: list[str], max_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > max_chars:
            if current:
                chunks.append(clean_text(current))
                current = ""
            chunks.extend(split_text(unit, max_chars, overlap))
            continue
        candidate = f"{current}\n{unit}" if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(clean_text(current))
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(clean_text(current))
    return chunks


def merge_short_leading_units(units: list[str], min_chars: int = 12) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(units):
        unit = units[index]
        if len(unit) < min_chars and index + 1 < len(units):
            merged.append(unit + units[index + 1])
            index += 2
            continue
        merged.append(unit)
        index += 1
    return merged


def split_section_text(section_type: str, text: str, max_chars: int, overlap: int) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if section_type == "facts":
        return pack_units(split_marker_units(text, FACT_DATE_BOUNDARY_RE), max_chars, overlap)
    if section_type == "reasoning":
        chunks: list[str] = []
        for unit in merge_short_leading_units(split_marker_units(text, REASONING_BOUNDARY_RE)):
            chunks.extend(split_text(unit, max_chars, overlap) if len(unit) > max_chars else [unit])
        return chunks
    return split_text(text, max_chars, overlap)


def make_embedding_text(case_doc: dict[str, Any], section_title: str, chunk_text: str, core_issue: str = "") -> str:
    parts = [
        f"案由：{case_doc['reason']}",
        f"审级：{case_doc['trial_level']}",
        f"法院：{case_doc['court_name']}",
    ]
    if core_issue:
        parts.append(f"核心争点：{core_issue}")
    parts.extend(
        [
            f"章节：{section_title}",
            f"正文：{chunk_text}",
        ]
    )
    return "\n".join(parts)


def locate_chunk(full_text: str, chunk_text: str, start_hint: int) -> tuple[int, int, int]:
    start = full_text.find(chunk_text, start_hint)
    if start < 0:
        start = full_text.find(chunk_text)
    if start < 0:
        lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
        if lines:
            first = lines[0]
            start = full_text.find(first, start_hint)
            if start < 0:
                start = full_text.find(first)
            if start >= 0:
                end = start + len(first)
                cursor = end
                for line in lines[1:]:
                    line_start = full_text.find(line, cursor)
                    if line_start < 0:
                        continue
                    end = line_start + len(line)
                    cursor = end
                return start, end, end
        return 0, len(chunk_text), start_hint
    end = start + len(chunk_text)
    return start, end, end


def line_number_at(text: str, position: int) -> int:
    if position <= 0:
        return 1
    return text.count("\n", 0, position) + 1


def chunk_size_for_section(section_type: str) -> tuple[int, int]:
    if section_type in {"claims", "defense"}:
        return 760, 0
    if section_type in {"facts", "reasoning"}:
        return 900, 120
    return 1600, 0


def build_chunks(row: dict[str, Any], case_doc: dict[str, Any], embedding_model: str, embedding_version: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    section_index = 0
    chunk_index_in_case = 0
    core_issue = core_issue_text(row)
    full_text = case_doc.get("full_text", "")
    search_start = 0
    for section_type, section_text in build_sections(row, case_doc):
        max_chars, overlap = chunk_size_for_section(section_type)
        for chunk_index_in_section, chunk_text in enumerate(split_section_text(section_type, section_text, max_chars, overlap)):
            chunk_id = f"{case_doc['doc_id']}#{section_type}#{section_index:02d}#{chunk_index_in_section:03d}"
            char_start, char_end, search_start = locate_chunk(full_text, chunk_text, search_start)
            chunk = {
                "chunk_id": chunk_id,
                "doc_id": case_doc["doc_id"],
                "chunk_text": chunk_text,
                "embedding_text": make_embedding_text(case_doc, SECTION_TITLES[section_type], chunk_text, core_issue),
                "section_type": section_type,
                "section_title": SECTION_TITLES[section_type],
                "section_index": section_index,
                "chunk_index_in_case": chunk_index_in_case,
                "chunk_index_in_section": chunk_index_in_section,
                "char_start": char_start,
                "char_end": char_end,
                "line_start": line_number_at(full_text, char_start),
                "line_end": line_number_at(full_text, char_end),
                "prev_chunk_id": "",
                "next_chunk_id": "",
                "chunk_char_len": len(chunk_text),
                "chunk_hash": sha256_text(chunk_text),
                "full_text_hash": case_doc["full_text_hash"],
                "case_name": case_doc["case_name"],
                "case_code": case_doc["case_code"],
                "court_name": case_doc["court_name"],
                "court_region": case_doc["court_region"],
                "reason": case_doc["reason"],
                "trial_level": case_doc["trial_level"],
                "judge_date": case_doc["judge_date"],
                "publish_date": case_doc["publish_date"],
                "case_type": case_doc["case_type"],
                "statutes": case_doc["statutes"],
                "embedding_model": embedding_model,
                "embedding_version": embedding_version,
                "embedding_dim": DEFAULT_EMBEDDING_DIM,
                "section_weight": SECTION_WEIGHTS.get(section_type, 1.0),
                "quality_flags": ["caselaw_benchmark_structured"],
                "schema_version": case_doc["schema_version"],
            }
            chunks.append(chunk)
            chunk_index_in_case += 1
        section_index += 1
    for index, chunk in enumerate(chunks):
        if index > 0:
            chunk["prev_chunk_id"] = chunks[index - 1]["chunk_id"]
        if index + 1 < len(chunks):
            chunk["next_chunk_id"] = chunks[index + 1]["chunk_id"]
    return chunks


def make_case_mapping() -> dict[str, Any]:
    return {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "dynamic": False,
            "properties": {
                "doc_id": {"type": "keyword"},
                "case_code": {"type": "keyword"},
                "case_name": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
                "court_name": {"type": "keyword"},
                "court_region": {"type": "keyword"},
                "case_type": {"type": "keyword"},
                "trial_level": {"type": "keyword"},
                "reason": {"type": "keyword", "fields": {"text": {"type": "text"}}},
                "judge_date": {"type": "date", "format": "yyyy-MM-dd||strict_date_optional_time||epoch_millis"},
                "publish_date": {"type": "date", "format": "yyyy-MM-dd||strict_date_optional_time||epoch_millis"},
                "litigants": {"type": "object", "enabled": False},
                "statutes": {"type": "keyword"},
                "full_text": {"type": "text"},
                "full_text_hash": {"type": "keyword"},
                "source_file": {"type": "keyword"},
                "source_row_id": {"type": "integer"},
                "schema_version": {"type": "keyword"},
            },
        },
    }


def make_chunk_mapping() -> dict[str, Any]:
    return {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0, "knn": True}},
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "chunk_text": {"type": "text"},
                "embedding_text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": DEFAULT_EMBEDDING_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {"ef_construction": 128, "m": 24},
                    },
                },
                "section_type": {"type": "keyword"},
                "section_title": {"type": "keyword"},
                "section_index": {"type": "integer"},
                "chunk_index_in_case": {"type": "integer"},
                "chunk_index_in_section": {"type": "integer"},
                "char_start": {"type": "integer"},
                "char_end": {"type": "integer"},
                "line_start": {"type": "integer"},
                "line_end": {"type": "integer"},
                "prev_chunk_id": {"type": "keyword"},
                "next_chunk_id": {"type": "keyword"},
                "chunk_char_len": {"type": "integer"},
                "chunk_hash": {"type": "keyword"},
                "full_text_hash": {"type": "keyword"},
                "case_name": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
                "case_code": {"type": "keyword"},
                "court_name": {"type": "keyword"},
                "court_region": {"type": "keyword"},
                "reason": {"type": "keyword", "fields": {"text": {"type": "text"}}},
                "trial_level": {"type": "keyword"},
                "judge_date": {"type": "date", "format": "yyyy-MM-dd||strict_date_optional_time||epoch_millis"},
                "publish_date": {"type": "date", "format": "yyyy-MM-dd||strict_date_optional_time||epoch_millis"},
                "case_type": {"type": "keyword"},
                "statutes": {"type": "keyword"},
                "embedding_model": {"type": "keyword"},
                "embedding_version": {"type": "keyword"},
                "embedding_dim": {"type": "integer"},
                "section_weight": {"type": "float"},
                "quality_flags": {"type": "keyword"},
                "schema_version": {"type": "keyword"},
            },
        },
    }


def output_paths(output_dir: Path, case_index: str, chunk_index: str) -> dict[str, Path]:
    return {
        "cases": output_dir / f"{case_index}.jsonl",
        "chunks": output_dir / f"{chunk_index}.jsonl",
        "case_mapping": output_dir / f"{case_index}_mapping.json",
        "chunk_mapping": output_dir / f"{chunk_index}_mapping.json",
        "stats": output_dir / "dataset_stats.json",
        "sample_chunks": output_dir / "sample_chunks.json",
    }


def ensure_output_paths(paths: dict[str, Path], overwrite: bool) -> None:
    paths["cases"].parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist: {names}. Use --overwrite to replace them.")


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    paths = output_paths(output_dir, args.case_index_name, args.chunk_index_name)
    ensure_output_paths(paths, args.overwrite)
    rows = read_jsonl(input_path, args.limit)
    stats = BuildStats()
    sample_chunks: list[dict[str, Any]] = []

    progress = tqdm(total=len(rows), desc="Build benchmark RAG dataset", unit="case") if tqdm else None
    with paths["cases"].open("w", encoding="utf-8") as cases_file, paths["chunks"].open("w", encoding="utf-8") as chunks_file:
        for row_id, row in enumerate(rows):
            case_doc = build_case_doc(row, input_path.name, row_id, args.schema_version)
            chunks = build_chunks(row, case_doc, args.embedding_model, args.embedding_version)
            cases_file.write(json_dumps(case_doc) + "\n")
            stats.cases += 1
            for chunk in chunks:
                chunks_file.write(json_dumps(chunk) + "\n")
                stats.chunks += 1
                stats.add_section(chunk["section_type"])
                if len(sample_chunks) < 12:
                    sample_chunks.append(
                        {
                            "chunk_id": chunk["chunk_id"],
                            "case_name": chunk["case_name"],
                            "section_type": chunk["section_type"],
                            "chunk_text": chunk["chunk_text"][:360],
                        }
                    )
            if progress:
                progress.update(1)
                progress.set_postfix(chunks=stats.chunks)
    if progress:
        progress.close()

    write_json(paths["case_mapping"], make_case_mapping())
    write_json(paths["chunk_mapping"], make_chunk_mapping())
    write_json(paths["sample_chunks"], sample_chunks)
    summary = {
        "source_file": str(input_path),
        "cases": stats.cases,
        "chunks": stats.chunks,
        "section_counts": stats.section_counts,
        "outputs": {name: str(path) for name, path in paths.items()},
        "case_index": args.case_index_name,
        "chunk_index": args.chunk_index_name,
        "embedding_model": args.embedding_model,
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "schema_version": args.schema_version,
    }
    write_json(paths["stats"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
