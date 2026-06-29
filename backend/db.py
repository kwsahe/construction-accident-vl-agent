from __future__ import annotations

import json
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "vl_agent.sqlite3"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS analysis_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          video_id TEXT NOT NULL,
          filename TEXT NOT NULL,
          provider TEXT NOT NULL,
          model_key TEXT NOT NULL,
          model_name TEXT NOT NULL,
          fast_mode INTEGER NOT NULL,
          run_yolo INTEGER NOT NULL,
          accident_type TEXT,
          injured_count INTEGER,
          cause TEXT,
          confidence REAL,
          latency_seconds REAL,
          payload_json TEXT NOT NULL,
          raw_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS eval_cases (
          video_id TEXT PRIMARY KEY,
          accident_detected INTEGER NOT NULL,
          accident_type TEXT NOT NULL,
          injured_count INTEGER NOT NULL,
          cause_keywords_json TEXT NOT NULL,
          accident_time_start REAL NOT NULL,
          accident_time_end REAL NOT NULL,
          required_evidence_json TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS eval_results (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL,
          video_id TEXT NOT NULL,
          model_name TEXT NOT NULL,
          prompt TEXT NOT NULL,
          accident_detected_score REAL NOT NULL,
          type_score REAL NOT NULL,
          injured_count_score REAL NOT NULL,
          cause_recall REAL NOT NULL,
          time_iou REAL NOT NULL,
          json_valid REAL NOT NULL,
          report_completeness REAL NOT NULL,
          semantic_score REAL NOT NULL DEFAULT 0,
          semantic_reason TEXT NOT NULL DEFAULT '',
          latency_score REAL NOT NULL,
          total_score REAL NOT NULL,
          created_at TEXT NOT NULL,
          UNIQUE(run_id),
          FOREIGN KEY(run_id) REFERENCES analysis_runs(id)
        );
        """
    )
    _ensure_column(conn, "eval_results", "semantic_score", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "eval_results", "semantic_reason", "TEXT NOT NULL DEFAULT ''")
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def insert_analysis_run(
    *,
    video_id: str,
    filename: str,
    provider: str,
    model_key: str,
    model_name: str,
    fast_mode: bool,
    run_yolo: bool,
    summary: dict[str, Any],
    payload: dict[str, Any],
    raw: dict[str, Any],
    latency_seconds: float,
) -> int:
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO analysis_runs (
              video_id, filename, provider, model_key, model_name, fast_mode, run_yolo,
              accident_type, injured_count, cause, confidence, latency_seconds,
              payload_json, raw_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                filename,
                provider,
                model_key,
                model_name,
                int(fast_mode),
                int(run_yolo),
                summary.get("accident_type_ko") or summary.get("accident_type"),
                summary.get("injured_count"),
                summary.get("cause"),
                summary.get("confidence"),
                latency_seconds,
                json.dumps(payload, ensure_ascii=False),
                json.dumps(raw, ensure_ascii=False),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def upsert_eval_case(case: dict[str, Any]) -> dict[str, Any]:
    video_id = str(case["video_id"]).strip()
    start, end = case.get("accident_time_range") or [0, 0]
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO eval_cases (
              video_id, accident_detected, accident_type, injured_count,
              cause_keywords_json, accident_time_start, accident_time_end,
              required_evidence_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
              accident_detected=excluded.accident_detected,
              accident_type=excluded.accident_type,
              injured_count=excluded.injured_count,
              cause_keywords_json=excluded.cause_keywords_json,
              accident_time_start=excluded.accident_time_start,
              accident_time_end=excluded.accident_time_end,
              required_evidence_json=excluded.required_evidence_json,
              updated_at=excluded.updated_at
            """,
            (
                video_id,
                int(bool(case.get("accident_detected"))),
                str(case.get("accident_type") or "기타"),
                int(case.get("injured_count") or 0),
                json.dumps(case.get("cause_keywords") or [], ensure_ascii=False),
                float(start),
                float(end),
                json.dumps(case.get("required_evidence") or [], ensure_ascii=False),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    return get_eval_case(video_id) or case


def get_eval_case(video_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM eval_cases WHERE video_id = ?", (video_id,)).fetchone()
    return _case_from_row(row) if row else None


def list_eval_cases() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM eval_cases ORDER BY updated_at DESC").fetchall()
    return [_case_from_row(row) for row in rows]


def score_latest_run_for_case(video_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        run = conn.execute(
            "SELECT * FROM analysis_runs WHERE video_id = ? ORDER BY id DESC LIMIT 1",
            (video_id,),
        ).fetchone()
    if not run:
        return None
    return score_run_if_case_exists(int(run["id"]))


def score_run_if_case_exists(run_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        run = conn.execute("SELECT * FROM analysis_runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            return None
        case = conn.execute("SELECT * FROM eval_cases WHERE video_id = ?", (run["video_id"],)).fetchone()
        if not case:
            return None
        result = _score_run(dict(run), _case_from_row(case))
        conn.execute(
            """
            INSERT INTO eval_results (
              run_id, video_id, model_name, prompt, accident_detected_score, type_score,
              injured_count_score, cause_recall, time_iou, json_valid,
              report_completeness, semantic_score, semantic_reason, latency_score, total_score, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
              accident_detected_score=excluded.accident_detected_score,
              type_score=excluded.type_score,
              injured_count_score=excluded.injured_count_score,
              cause_recall=excluded.cause_recall,
              time_iou=excluded.time_iou,
              json_valid=excluded.json_valid,
              report_completeness=excluded.report_completeness,
              semantic_score=excluded.semantic_score,
              semantic_reason=excluded.semantic_reason,
              latency_score=excluded.latency_score,
              total_score=excluded.total_score,
              created_at=excluded.created_at
            """,
            (
                run_id,
                result["video_id"],
                result["model_name"],
                result["prompt"],
                result["accident_detected_score"],
                result["type_score"],
                result["injured_count_score"],
                result["cause_recall"],
                result["time_iou"],
                result["json_valid"],
                result["report_completeness"],
                result["semantic_score"],
                result["semantic_reason"],
                result["latency_score"],
                result["total_score"],
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    return result


def evaluation_summary_from_db() -> dict[str, Any] | None:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT er.*, ar.latency_seconds
            FROM eval_results er
            JOIN analysis_runs ar ON ar.id = er.run_id
            ORDER BY er.created_at DESC
            """
        ).fetchall()
    if not rows:
        return None

    grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault((row["model_name"], row["prompt"]), []).append(row)

    scores = []
    for (model_name, prompt), items in grouped.items():
        scores.append({
            "name": model_name,
            "prompt": prompt,
            "total": _avg(row["total_score"] for row in items),
            "typeAccuracy": _avg(row["type_score"] for row in items),
            "injuredCountAccuracy": _avg(row["injured_count_score"] for row in items),
            "causeRecall": _avg(row["cause_recall"] for row in items),
            "semanticScore": _avg(row["semantic_score"] for row in items),
            "jsonValid": _avg(row["json_valid"] for row in items),
            "latency": _avg(row["latency_seconds"] for row in items),
        })
    scores.sort(key=lambda item: item["total"], reverse=True)
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "sqlite_eval_cases",
        "best_model": scores[0]["name"] if scores else "",
        "scores": scores,
        "charts": [
            "eval/output/model_score_bar.png",
            "eval/output/confusion_matrix.png",
            "eval/output/cause_recall_by_prompt.png",
            "eval/output/latency_boxplot.png",
        ],
    }


def _score_run(run: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(run["payload_json"])
    raw = json.loads(run["raw_json"])
    text = json.dumps(payload, ensure_ascii=False) + json.dumps(raw, ensure_ascii=False)
    expected_detected = bool(case.get("accident_detected"))
    predicted_detected = str(run.get("accident_type") or "기타") != "기타"
    accident_detected_score = 1.0 if expected_detected == predicted_detected else 0.0
    type_score = 1.0 if str(run.get("accident_type") or "") == str(case.get("accident_type")) else 0.0
    injured_count_score = _injured_score(int(run.get("injured_count") or 0), int(case.get("injured_count") or 0))
    cause_recall = _keyword_recall(case.get("cause_keywords") or [], text)
    time_iou = _time_iou(payload, case.get("accident_time_range") or [0, 0])
    json_valid = 1.0 if payload.get("judgment") and payload.get("video_part_tables") else 0.0
    report_completeness = _report_completeness(raw)
    latency_score = _latency_score(float(run.get("latency_seconds") or 0))
    semantic = _semantic_judge(case, raw)
    semantic_score = float(semantic.get("score") or 0.0)
    total = (
        0.15 * accident_detected_score
        + 0.20 * type_score
        + 0.10 * injured_count_score
        + 0.15 * cause_recall
        + 0.10 * semantic_score
        + 0.10 * time_iou
        + 0.10 * json_valid
        + 0.05 * report_completeness
        + 0.05 * latency_score
    )
    return {
        "video_id": run["video_id"],
        "model_name": run["model_name"],
        "prompt": "fast_yolo" if run["fast_mode"] else "full",
        "accident_detected_score": round(accident_detected_score, 4),
        "type_score": round(type_score, 4),
        "injured_count_score": round(injured_count_score, 4),
        "cause_recall": round(cause_recall, 4),
        "time_iou": round(time_iou, 4),
        "json_valid": round(json_valid, 4),
        "report_completeness": round(report_completeness, 4),
        "semantic_score": round(semantic_score, 4),
        "semantic_reason": str(semantic.get("reason") or ""),
        "latency_score": round(latency_score, 4),
        "total_score": round(total, 4),
    }


def _case_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "video_id": row["video_id"],
        "accident_detected": bool(row["accident_detected"]),
        "accident_type": row["accident_type"],
        "injured_count": row["injured_count"],
        "cause_keywords": json.loads(row["cause_keywords_json"]),
        "accident_time_range": [row["accident_time_start"], row["accident_time_end"]],
        "required_evidence": json.loads(row["required_evidence_json"]),
        "updated_at": row["updated_at"],
    }


def _keyword_recall(keywords: list[str], text: str) -> float:
    if not keywords:
        return 1.0
    return sum(1 for keyword in keywords if keyword and keyword in text) / len(keywords)


def _semantic_judge(case: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    fallback_score = _keyword_recall(
        list(case.get("cause_keywords") or []) + list(case.get("required_evidence") or []),
        json.dumps(raw, ensure_ascii=False),
    )
    prompt = {
        "instruction": (
            "너는 건설현장 사고 분석 결과를 채점하는 평가자다. "
            "정답 JSON과 예측 JSON을 비교해서 원인 설명과 근거가 의미적으로 얼마나 일치하는지 0~1로 채점한다. "
            "사고 유형/부상자 수는 이미 별도 채점되므로, cause/evidence/details의 의미 일치만 평가한다. "
            "반드시 JSON만 출력한다."
        ),
        "output_schema": {"score": 0.0, "reason": "한국어 한 문장"},
        "gold": {
            "accident_type": case.get("accident_type"),
            "cause_keywords": case.get("cause_keywords") or [],
            "required_evidence": case.get("required_evidence") or [],
        },
        "prediction": {
            "cause": raw.get("cause"),
            "details": raw.get("details"),
            "evidence": raw.get("evidence") or raw.get("visible_clues") or [],
            "timeline": raw.get("timeline") or [],
        },
    }
    try:
        data = _ollama_json("qwen2.5:3b", json.dumps(prompt, ensure_ascii=False), timeout=45)
        score = max(0.0, min(1.0, float(data.get("score", fallback_score))))
        return {"score": score, "reason": str(data.get("reason") or "qwen2.5:3b 의미 채점 완료")}
    except Exception as exc:
        return {
            "score": fallback_score,
            "reason": f"qwen2.5:3b 의미 채점 실패, 키워드 recall fallback 사용: {exc}",
        }


def _ollama_json(model: str, prompt: str, timeout: int = 45) -> dict[str, Any]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 240},
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    text = ((data.get("message") or {}).get("content") or "").strip()
    match = __import__("re").search(r"\{.*\}", text, __import__("re").S)
    if not match:
        raise RuntimeError(f"Ollama judge returned non-JSON: {text[:300]}")
    return json.loads(match.group(0))


def _injured_score(predicted: int, expected: int) -> float:
    if predicted == expected:
        return 1.0
    return 0.5 if abs(predicted - expected) == 1 else 0.0


def _time_iou(payload: dict[str, Any], expected: list[float]) -> float:
    event = ((payload.get("video_part_tables") or {}).get("cctv_events") or [{}])[0]
    pred_start = float(event.get("clip_start_offset") or 0)
    pred_end = float(event.get("clip_end_offset") or 0)
    exp_start, exp_end = float(expected[0]), float(expected[1])
    intersection = max(0.0, min(pred_end, exp_end) - max(pred_start, exp_start))
    union = max(pred_end, exp_end) - min(pred_start, exp_start)
    return intersection / union if union > 0 else 0.0


def _report_completeness(raw: dict[str, Any]) -> float:
    required = ["primary_type", "injured_count", "cause", "details"]
    return sum(1 for key in required if raw.get(key)) / len(required)


def _latency_score(seconds: float) -> float:
    if seconds <= 30:
        return 1.0
    if seconds >= 300:
        return 0.0
    return max(0.0, 1.0 - ((seconds - 30) / 270))


def _avg(values) -> float:
    items = list(values)
    return round(sum(float(item) for item in items) / len(items), 4) if items else 0.0
