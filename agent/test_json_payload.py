"""Validate generated Judgement Agent JSON files.

This script is for local JSON-only testing. It does not call Colab and does
not insert rows into the Django database.

Examples:
    python -m agent.test_json_payload
    python -m agent.test_json_payload --raw agent/output/accident_judgment_result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .save_judgment import validate_colab_judgment
except ImportError:  # direct script execution
    from save_judgment import validate_colab_judgment


AGENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = AGENT_DIR / "output"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Judgement Agent JSON outputs.")
    parser.add_argument(
        "--raw",
        default=str(DEFAULT_OUTPUT_DIR / "accident_judgment_result.json"),
        help="Raw Qwen judgement JSON path",
    )
    parser.add_argument(
        "--payload",
        default=str(DEFAULT_OUTPUT_DIR / "accident_analysis_payload.json"),
        help="Schema payload JSON path",
    )
    parser.add_argument(
        "--pt",
        default=str(DEFAULT_OUTPUT_DIR / "pt_detection_result.json"),
        help="PT detection result JSON path",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    payload_path = Path(args.payload)
    pt_path = Path(args.pt)

    failed = False

    raw = _read_json(raw_path, "raw judgement")
    validation = validate_colab_judgment(raw)
    print("[raw judgement]")
    print(f"- path: {raw_path}")
    print(f"- primary_type: {raw.get('primary_type')}")
    print(f"- confidence: {raw.get('confidence')}")
    print(f"- details: {'ok' if raw.get('details') else 'missing'}")
    if validation["errors"]:
        failed = True
        print("- errors:")
        for error in validation["errors"]:
            print(f"  - {error}")
    if validation["warnings"]:
        print("- warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    payload = _read_json(payload_path, "schema payload")
    payload_errors = _validate_payload(payload)
    print("\n[schema payload]")
    print(f"- path: {payload_path}")
    print(f"- accident_type: {payload.get('judgment', {}).get('accident_type')}")
    print(f"- event_logs: {len(payload.get('event_logs') or [])}")
    tables = payload.get("video_part_tables") or {}
    print(f"- cctv_events: {len(tables.get('cctv_events') or [])}")
    print(f"- evidence_photos: {len(tables.get('evidence_photos') or [])}")
    print(f"- tts_alert_logs: {len(tables.get('tts_alert_logs') or [])}")
    if payload_errors:
        failed = True
        print("- errors:")
        for error in payload_errors:
            print(f"  - {error}")

    if pt_path.exists():
        pt = _read_json(pt_path, "pt detection")
        print("\n[pt detection]")
        print(f"- path: {pt_path}")
        print(f"- status: {pt.get('status')}")
        print(f"- model_path: {pt.get('model_path')}")
        print(f"- detections: {len(pt.get('detections') or [])}")
    else:
        print("\n[pt detection]")
        print(f"- path: {pt_path}")
        print("- status: missing file")

    if failed:
        print("\nJSON test failed.")
        return 2

    print("\nJSON test passed.")
    return 0


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    if label == "raw judgement" and isinstance(data.get("judgment"), dict):
        return data["judgment"]
    return data


def _validate_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ["judgment", "event_logs", "video_part_tables", "pt_status"]:
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    pt_status = payload.get("pt_status")
    if not isinstance(pt_status, dict):
        errors.append("pt_status must be an object")
    elif "status" not in pt_status:
        errors.append("pt_status missing key: status")

    judgment = payload.get("judgment")
    if not isinstance(judgment, dict):
        errors.append("judgment must be an object")
    else:
        for key in ["accident_type", "accident_type_ko", "agent_verdict", "confidence", "details"]:
            if key not in judgment:
                errors.append(f"judgment missing key: {key}")

    event_logs = payload.get("event_logs")
    if not isinstance(event_logs, list):
        errors.append("event_logs must be a list")

    tables = payload.get("video_part_tables")
    if not isinstance(tables, dict):
        errors.append("video_part_tables must be an object")
        return errors

    for key in ["cctv_events", "evidence_photos", "tts_alert_logs"]:
        rows = tables.get(key)
        if not isinstance(rows, list):
            errors.append(f"video_part_tables.{key} must be a list")
        elif not rows:
            errors.append(f"video_part_tables.{key} must have at least one row")

    errors.extend(_validate_erd_rows(tables))
    return errors


def _validate_erd_rows(tables: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "cctv_events": [
            "camera_id",
            "zone_id",
            "detected_at",
            "label",
            "confidence",
            "clip_path",
            "clip_start_offset",
            "clip_end_offset",
            "agent_verdict",
            "agent_summary",
            "is_incident_trigger",
            "bbox_json",
            "snapshot_path",
            "is_simulated",
        ],
        "evidence_photos": [
            "event_id",
            "incident_id",
            "photo_url",
            "detected_label",
            "confidence",
            "taken_at",
            "source_type",
            "is_representative",
        ],
        "tts_alert_logs": [
            "event_id",
            "camera_id",
            "zone_id",
            "language",
            "message",
            "play_order",
            "audio_url",
            "played_at",
            "status",
        ],
    }

    for table_name, keys in required.items():
        rows = tables.get(table_name) or []
        if not rows:
            continue
        row = rows[0]
        if not isinstance(row, dict):
            errors.append(f"video_part_tables.{table_name}[0] must be an object")
            continue
        for key in keys:
            if key not in row:
                errors.append(f"video_part_tables.{table_name}[0] missing ERD key: {key}")

    cctv_rows = tables.get("cctv_events") or []
    if cctv_rows and isinstance(cctv_rows[0], dict):
        verdict = cctv_rows[0].get("agent_verdict")
        if verdict not in {"accident", "near_miss", "normal"}:
            errors.append(f"cctv_events.agent_verdict invalid: {verdict!r}")

    photo_rows = tables.get("evidence_photos") or []
    if photo_rows and isinstance(photo_rows[0], dict):
        source_type = photo_rows[0].get("source_type")
        if source_type not in {"cctv_capture", "upload", "system"}:
            errors.append(f"evidence_photos.source_type invalid: {source_type!r}")

    tts_rows = tables.get("tts_alert_logs") or []
    if tts_rows and isinstance(tts_rows[0], dict):
        status = tts_rows[0].get("status")
        if status not in {"success", "failed", "skipped"}:
            errors.append(f"tts_alert_logs.status invalid: {status!r}")

    return errors


if __name__ == "__main__":
    raise SystemExit(main())
