"""Save a Colab/Qwen judgment result in the construction accident analysis schema.

Examples:
    python -m agent.save_judgment --input agent/output/colab_judgment.json
    python -m agent.save_judgment --input agent/output/colab_judgment.json --insert-db
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .schemas import AccidentJudgment, AgentOutput, EventLog, VisualObservation, now_iso
except ImportError:  # direct script execution
    from schemas import AccidentJudgment, AgentOutput, EventLog, VisualObservation, now_iso


TYPE_MAP = {
    "추락": ("fall_from_height", "추락"),
    "낙상": ("slip_and_fall", "낙상"),
    "화재": ("fire_explosion", "화재"),
    "기타": ("other", "기타"),
}

AGENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = AGENT_DIR.parent
DEFAULT_OUTPUT_DIR = AGENT_DIR / "output"


def main() -> int:
    parser = argparse.ArgumentParser(description="Save Qwen judgment output in the accident analysis schema.")
    parser.add_argument("--input", "-i", required=True, help="Colab judgement JSON file path")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT_DIR / "accident_analysis_payload.json"))
    parser.add_argument("--insert-db", action="store_true", help="Insert video-part rows into Django DB (direct ORM)")
    parser.add_argument("--api-url", default="", help="Django API base URL (e.g. http://127.0.0.1:8000). POST /api/agent/ingest 로 전송합니다.")
    parser.add_argument("--camera-id", default="Camera 15")
    parser.add_argument("--zone-id", default="construction_site")
    parser.add_argument("--zone-name", default="건설현장 사고 구역")
    parser.add_argument("--detected-at", default="")
    parser.add_argument("--clip-path", default="video/accident_video.mp4")
    parser.add_argument("--snapshot-path", default="output/evidence/judgement_snapshot.jpg")
    parser.add_argument("--pt-status", default="", help="Optional PT detection result JSON path")
    args = parser.parse_args()

    raw = _load_json(Path(args.input))
    pt_status = _load_optional_json(Path(args.pt_status)) if args.pt_status else None
    validation = validate_colab_judgment(raw)
    if validation["errors"]:
        print("[validation failed]")
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 2
    if validation["warnings"]:
        print("[validation warnings]")
        print(json.dumps(validation, ensure_ascii=False, indent=2))

    detected_at = args.detected_at or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    output = build_agent_output(
        raw,
        VisualObservation(
            camera_id=args.camera_id,
            zone_id=args.zone_id,
            zone_name=args.zone_name,
            detected_at=detected_at,
            image_url=args.snapshot_path,
            clip_path=args.clip_path,
            confidence=float(raw.get("confidence") or 0.0) or 0.85,
        ),
        pt_status=pt_status,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(output), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved schema payload: {out_path}")

    if args.insert_db and args.api_url:
        print("[error] --insert-db 와 --api-url 은 동시에 사용할 수 없습니다.")
        return 1

    if args.insert_db:
        # 기존 방식: Django ORM 직접 접근 (django.setup() 필요)
        ids = insert_video_part_tables(output.video_part_tables)
        print(json.dumps(ids, ensure_ascii=False, indent=2))

    elif args.api_url:
        # 신규 방식: Django REST API 호출
        result = post_to_django(args.api_url, asdict(output))
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0


def validate_colab_judgment(raw: dict[str, Any]) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    primary_type = raw.get("primary_type")
    if primary_type not in TYPE_MAP:
        errors.append(
            "primary_type must be one of: "
            + ", ".join(TYPE_MAP.keys())
            + f" (got {primary_type!r})"
        )

    confidence = raw.get("confidence")
    try:
        confidence_float = float(confidence)
        if not 0.0 <= confidence_float <= 1.0:
            errors.append(f"confidence must be between 0 and 1 (got {confidence!r})")
    except (TypeError, ValueError):
        errors.append(f"confidence must be numeric (got {confidence!r})")

    if not raw.get("details"):
        warnings.append("details is missing or empty; summary will be used as fallback.")

    if not raw.get("timeline"):
        warnings.append("timeline is missing or empty.")

    if not raw.get("evidence") and not raw.get("visible_clues"):
        warnings.append("evidence and visible_clues are both missing or empty.")

    workers = raw.get("workers")
    if workers is not None and not isinstance(workers, list):
        errors.append("workers must be a list when provided.")

    return {"errors": errors, "warnings": warnings}


def build_agent_output(
    raw: dict[str, Any],
    observation: VisualObservation,
    pt_status: dict[str, Any] | None = None,
) -> AgentOutput:
    primary_type = str(raw.get("primary_type") or "기타")
    accident_type, accident_type_ko = TYPE_MAP.get(primary_type, ("other", "기타"))
    confidence = _safe_float(raw.get("confidence"), observation.confidence)
    summary = str(raw.get("photo_summary") or raw.get("summary") or _summary_for(accident_type))
    details = str(raw.get("details") or summary)
    evidence = raw.get("evidence") or raw.get("visible_clues") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)]

    judgment = AccidentJudgment(
        accident_type=accident_type,  # type: ignore[arg-type]
        accident_type_ko=accident_type_ko,
        agent_verdict="accident" if accident_type != "other" else "near_miss",
        confidence=confidence,
        summary=summary,
        details=details,
        evidence=[str(item) for item in evidence],
        model="Qwen/Qwen2.5-VL-32B-Instruct",
        raw=raw,
    )
    event_logs = _event_logs_from_judgment(raw, observation, judgment)
    return AgentOutput(
        judgment=judgment,
        event_logs=event_logs,
        video_part_tables=_video_part_tables(observation, judgment, event_logs, pt_status),
        pt_status=pt_status or {
            "status": "qwen_saved",
            "message": "Qwen2.5-VL 판단 결과를 사고 분석 payload로 저장했습니다.",
            "saved_at": now_iso(),
        },
    )


def insert_video_part_tables(tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    sys.path.insert(0, str(BACKEND_DIR))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag.settings")

    import django
    django.setup()

    from django.utils.dateparse import parse_datetime
    from django.utils import timezone
    from core.models import CctvEvent, EvidencePhoto, Incident, IncidentDetail, Report, TtsAlertLog, Workplace

    def _aware(dt_str: str | None):
        dt = parse_datetime(dt_str or "") if dt_str else None
        if dt is None:
            return timezone.now()
        return timezone.make_aware(dt) if timezone.is_naive(dt) else dt

    wp, _ = Workplace.objects.get_or_create(
        business_reg_no="000-00-00000",
        defaults={
            "company_name": "기본 사업장",
            "total_workers": 0,
            "direct_workers": 0,
            "contracted_workers": 0,
        },
    )

    cctv_ids = []
    event_map: dict[int, Any] = {}
    for row in tables.get("cctv_events", []):
        event = CctvEvent.objects.create(
            workplace=wp,
            camera_id=row.get("camera_id") or "Camera 15",
            zone_id=row.get("zone_id") or "unknown",
            detected_at=_aware(row.get("detected_at")),
            label=row.get("label") or "unknown",
            confidence=float(row.get("confidence") or 0.0),
            clip_path=row.get("clip_path"),
            clip_start_offset=row.get("clip_start_offset"),
            clip_end_offset=row.get("clip_end_offset"),
            agent_verdict=row.get("agent_verdict"),
            agent_summary=row.get("agent_summary"),
            is_incident_trigger=bool(row.get("is_incident_trigger")),
            bbox_json=row.get("bbox_json"),
            snapshot_path=row.get("snapshot_path"),
            is_simulated=bool(row.get("is_simulated", True)),
        )
        event_map[int(row.get("id") or len(event_map) + 1)] = event
        cctv_ids.append(event.id)

    photo_ids = []
    for row in tables.get("evidence_photos", []):
        event = event_map.get(int(row.get("event_id") or 1))
        photo = EvidencePhoto.objects.create(
            event=event,
            incident=None,
            photo_url=row.get("photo_url") or "",
            detected_label=row.get("detected_label"),
            confidence=row.get("confidence"),
            taken_at=_aware(row.get("taken_at")),
            source_type=row.get("source_type") or "cctv_capture",
            is_representative=bool(row.get("is_representative")),
        )
        photo_ids.append(photo.id)

    tts_ids = []
    for row in tables.get("tts_alert_logs", []):
        event = event_map.get(int(row.get("event_id") or 1))
        if event is None:
            continue
        log = TtsAlertLog.objects.create(
            event=event,
            camera_id=row.get("camera_id") or event.camera_id,
            zone_id=row.get("zone_id") or event.zone_id,
            language=row.get("language") or "ko",
            message=row.get("message") or "",
            play_order=int(row.get("play_order") or 1),
            audio_url=row.get("audio_url"),
            played_at=_aware(row.get("played_at")),
            status=row.get("status") or "success",
        )
        tts_ids.append(log.id)

    incident_ids = []
    report_ids = []
    for event in event_map.values():
        if not event.is_incident_trigger:
            continue
        incident = Incident.objects.create(
            workplace=event.workplace,
            cctv_event=event,
            occurred_at=event.detected_at,
            location=event.zone_id,
            accident_type=_incident_type_from_label(event.label),
            status="open",
            description=event.agent_summary or f"{event.label} 감지 사고",
        )
        EvidencePhoto.objects.filter(event=event).update(incident=incident)
        IncidentDetail.objects.create(
            incident=incident,
            timelines=[{
                "seq": 1,
                "time": event.detected_at.isoformat(),
                "title": event.agent_summary or f"{event.label} 감지",
                "phase": "accident",
                "tag": event.label,
                "is_key": True,
            }],
            special_notes={
                "source": "judgement_agent",
                "cctv_event_id": event.id,
            },
        )
        report = Report.objects.create(
            incident=incident,
            doc_id=f"SP-{incident.id}-{timezone.now():%Y%m%d%H%M%S}",
            status="draft",
            accident_summary=event.agent_summary,
            ai_recommendation="사고 원인 분석 및 법령 RAG 판단이 필요합니다.",
            why_items=[],
            cause_items=[],
            legal_violations=[],
            compliance_levels={},
            final_opinion="",
        )
        incident_ids.append(incident.id)
        report_ids.append(report.id)

    return {
        "inserted": {
            "cctv_events": cctv_ids,
            "evidence_photos": photo_ids,
            "tts_alert_logs": tts_ids,
        },
        "incidents": incident_ids,
        "reports": report_ids,
    }


def _incident_type_from_label(label: str | None) -> str:
    value = (label or "").lower()
    if "fall_from_height" in value:
        return "fall_from_height"
    if "slip" in value or "fall" in value:
        return "slip_and_fall"
    if "fire" in value or "smoke" in value:
        return "fire_explosion"
    if "collapse" in value:
        return "collapse"
    return "other"


def post_to_django(api_base: str, payload: dict[str, Any]) -> dict[str, Any]:
    """payload를 /api/agent/ingest 엔드포인트로 POST 전송 (stdlib만 사용)."""
    import urllib.request
    import urllib.error

    url = api_base.rstrip("/") + "/api/agent/ingest"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {e.code}", "detail": error_body}
    except urllib.error.URLError as e:
        return {"error": "URLError", "detail": str(e.reason)}


def _load_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        data = json.loads(match.group(0))

    if "judgment" in data and isinstance(data["judgment"], dict):
        return data["judgment"]
    return data


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _event_logs_from_judgment(
    raw: dict[str, Any],
    observation: VisualObservation,
    judgment: AccidentJudgment,
) -> list[EventLog]:
    """Create a generic accident alert; cause analysis is handled by the VL judgment."""
    if judgment.agent_verdict != "accident":
        return []

    cause = str(raw.get("cause") or "").strip()
    message = cause or judgment.summary or "사고 의심 상황이 감지되었습니다. 즉시 현장을 확인하십시오."
    return [EventLog(
        event_type="ACCIDENT_ALERT",
        label="accident_detected",
        message=message,
        confidence=judgment.confidence,
        severity="high",
    )]

def _video_part_tables(
    observation: VisualObservation,
    judgment: AccidentJudgment,
    event_logs: list[EventLog],
    pt_status: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    event_label = {
        "fall_from_height": "worker_fall_from_height",
        "slip_and_fall": "worker_slip_and_fall",
        "fire_explosion": "fire_or_smoke",
    }.get(judgment.accident_type, event_logs[0].label if event_logs else "unknown")
    created_at = now_iso()
    bbox_json = "[]"
    if pt_status and isinstance(pt_status.get("detections"), list):
        bbox_json = json.dumps(pt_status["detections"], ensure_ascii=False)

    return {
        "cctv_events": [{
            "id": 1,
            "workplace_id": 1,
            "camera_id": observation.camera_id,
            "zone_id": observation.zone_id,
            "detected_at": observation.detected_at,
            "label": event_label,
            "confidence": round(judgment.confidence, 4),
            "clip_path": observation.clip_path,
            "clip_start_offset": observation.clip_start_offset,
            "clip_end_offset": observation.clip_end_offset,
            "agent_verdict": judgment.agent_verdict,
            "agent_summary": judgment.details,
            "is_incident_trigger": judgment.agent_verdict == "accident",
            "bbox_json": bbox_json,
            "snapshot_path": observation.image_url,
            "is_simulated": True,
            "created_at": created_at,
        }],
        "evidence_photos": [{
            "id": 1,
            "event_id": 1,
            "incident_id": None,
            "photo_url": observation.image_url or "output/evidence/judgement_snapshot.jpg",
            "detected_label": event_label,
            "confidence": round(judgment.confidence, 4),
            "taken_at": observation.detected_at,
            "source_type": "cctv_capture",
            "is_representative": True,
            "created_at": created_at,
        }],
        "tts_alert_logs": _tts_rows(observation, judgment, event_logs, created_at),
    }


def _tts_rows(
    observation: VisualObservation,
    judgment: AccidentJudgment,
    event_logs: list[EventLog],
    created_at: str,
) -> list[dict[str, Any]]:
    if not event_logs:
        message = "추락 사고 의심 상황이 감지되었습니다. 즉시 작업을 중지하고 현장을 통제하십시오."
        if judgment.accident_type == "slip_and_fall":
            message = "낙상 사고 의심 상황이 감지되었습니다. 즉시 작업을 중지하고 현장을 확인하십시오."
        if judgment.accident_type == "fire_explosion":
            message = "화재 또는 폭발 위험이 감지되었습니다. 즉시 대피하고 초기 대응을 준비하십시오."
        event_logs = [EventLog(
            event_type="ACCIDENT_ALERT",
            label="accident_detected",
            message=message,
            confidence=judgment.confidence,
            severity="high",
        )]

    return [{
        "id": idx,
        "event_id": 1,
        "camera_id": observation.camera_id,
        "zone_id": observation.zone_id,
        "language": "ko",
        "message": log.message,
        "play_order": idx,
        "audio_url": None,
        "played_at": created_at,
        "status": "success",
        "created_at": created_at,
    } for idx, log in enumerate(event_logs, start=1)]


def _summary_for(accident_type: str) -> str:
    return {
        "fall_from_height": "고소작업 또는 높은 위치에서 추락 사고가 의심됩니다.",
        "slip_and_fall": "동일 평면에서 미끄러짐 또는 넘어짐 사고가 의심됩니다.",
        "fire_explosion": "화재 또는 폭발 사고가 의심됩니다.",
    }.get(accident_type, "사고 유형 판단이 불확실합니다.")

def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
