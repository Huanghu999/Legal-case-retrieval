from __future__ import annotations

import json
import os
import time
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from src.legal_case_rag.app import advisor_service as advisor


app = Flask(__name__)
STATIC_VERSION = str(int(time.time()))


@app.get("/")
def index() -> str:
    return render_template(
        "legal_case_advisor_index.html",
        defaults=advisor.DEFAULT_APP_CONFIG,
        model_name=advisor.MIMO_MODEL,
        static_version=STATIC_VERSION,
    )


@app.get("/api/health")
def health() -> Any:
    return jsonify(
        {
            "ok": True,
            "has_opensearch_password": bool(
                os.getenv(advisor.retrieval.DEFAULT_OPENSEARCH_PASSWORD_ENV)
            ),
            "has_siliconflow_key": bool(
                os.getenv(advisor.retrieval.DEFAULT_EMBEDDING_KEY_ENV)
            ),
            "has_mimo_key": bool(os.getenv(advisor.MIMO_API_KEY_ENV)),
            "opensearch_url": os.getenv(
                "OPENSEARCH_URL",
                advisor.retrieval.DEFAULT_OPENSEARCH_URL,
            ),
            "retrieval_mode": advisor.DEFAULT_APP_CONFIG["mode"],
            "answer_model": advisor.MIMO_MODEL,
        }
    )


@app.post("/api/analyze")
def analyze() -> Any:
    payload = request.get_json(silent=True) or {}
    started = time.perf_counter()
    try:
        result = advisor.run_advisor_answer(payload)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    result["ok"] = True
    result["duration_ms"] = int((time.perf_counter() - started) * 1000)
    return jsonify(result)


@app.post("/api/analyze/stream")
def analyze_stream() -> Any:
    payload = request.get_json(silent=True) or {}

    def generate():
        started = time.perf_counter()
        try:
            args, results = advisor.prepare_advisor_context(payload)
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)}, ensure_ascii=False)}\n\n"
            return

        if not results:
            empty_payload = advisor.build_empty_answer_payload(args)
            yield f"data: {json.dumps({'type': 'result', 'payload': empty_payload, 'duration_ms': int((time.perf_counter() - started) * 1000)}, ensure_ascii=False)}\n\n"
            return

        meta_event = {
            "type": "retrieval",
            "result_count": len(results),
            "mode": args.mode,
            "rerank_enabled": args.rerank,
        }
        yield f"data: {json.dumps(meta_event, ensure_ascii=False)}\n\n"

        preview_prompt = advisor.build_stream_preview_prompt(args.query, results)
        try:
            for piece in advisor.stream_mimo_preview_text(preview_prompt):
                yield f"data: {json.dumps({'type': 'token', 'text': piece}, ensure_ascii=False)}\n\n"

            prompt = advisor.build_prompt(args.query, results)
            raw_answer = advisor.call_mimo(prompt)
            answer = advisor.normalize_answer(raw_answer, results, args.query)
        except Exception:
            answer = advisor.make_fallback_answer(args.query, results)

        final_payload = advisor.build_advisor_payload(
            args=args,
            results=results,
            answer=answer,
        )
        yield f"data: {json.dumps({'type': 'result', 'payload': final_payload, 'duration_ms': int((time.perf_counter() - started) * 1000)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/cases/<path:doc_id>")
def case_detail(doc_id: str) -> Any:
    try:
        case_detail_payload = advisor.build_case_detail(doc_id)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    if not case_detail_payload:
        return jsonify({"ok": False, "error": "未找到对应文书。"}), 404

    return jsonify({"ok": True, "case": case_detail_payload})


if __name__ == "__main__":
    host = os.getenv("LEGAL_CASE_ADVISOR_HOST", "127.0.0.1")
    port = int(os.getenv("LEGAL_CASE_ADVISOR_PORT", "7870"))
    app.run(host=host, port=port, debug=False)
