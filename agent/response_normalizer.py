"""Normalize heterogeneous VL model responses into one evaluation schema."""

from __future__ import annotations

import json
import re
from typing import Any


VALID_PRIMARY_TYPES = {"낙상", "추락", "화재", "기타"}


def parse_model_response(text: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(text, dict):
        parsed = text
    else:
        parsed = _parse_json_object(text)
    return normalize_accident_response(parsed)


def normalize_accident_response(raw: dict[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    primary_type = str(result.get("primary_type") or result.get("accident_type_ko") or "기타")
    if primary_type not in VALID_PRIMARY_TYPES:
        primary_type = _infer_primary_type(json.dumps(result, ensure_ascii=False))

    result["primary_type"] = primary_type
    result["secondary_type"] = str(result.get("secondary_type") or "기타")
    result["injured_count"] = _safe_int(result.get("injured_count", result.get("injured_workers_count", 0)))
    result["confidence"] = _safe_float(result.get("confidence"), 0.0)
    result["cause"] = str(result.get("cause") or result.get("cause_summary") or "원인 불확실")
    result["cause_confidence"] = _safe_float(result.get("cause_confidence"), result["confidence"])
    result["timeline"] = _ensure_list(result.get("timeline"))
    result["workers"] = _ensure_list(result.get("workers"))
    result["evidence"] = _ensure_list(result.get("evidence") or result.get("visible_clues"))
    result["uncertain_points"] = _ensure_list(result.get("uncertain_points"))
    result["details"] = str(result.get("details") or result.get("summary") or result["cause"])
    result["report_draft"] = _ensure_report(result.get("report_draft"), result)
    result["prevention_actions"] = _ensure_prevention_actions(result.get("prevention_actions"))
    return result


def _parse_json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    value = re.sub(r"^```(?:json)?", "", value).strip()
    value = re.sub(r"```$", "", value).strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", value, re.S)
    if not match:
        return {"primary_type": "기타", "confidence": 0.0, "details": value}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError:
        return {"primary_type": "기타", "confidence": 0.0, "details": value}


def _infer_primary_type(text: str) -> str:
    if any(token in text for token in ("추락", "떨어", "고소", "높은 위치")):
        return "추락"
    if any(token in text for token in ("낙상", "넘어", "미끄러")):
        return "낙상"
    if any(token in text for token in ("화재", "연기", "폭발", "불꽃")):
        return "화재"
    return "기타"


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _ensure_report(value: Any, result: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        report = dict(value)
    else:
        report = {}
    report.setdefault("title", f"건설현장 {result.get('primary_type', '기타')} 사고 분석 보고서 초안")
    report.setdefault("overview", result.get("details") or result.get("cause") or "")
    report.setdefault("cause_analysis", result.get("cause") or "원인 불확실")
    report.setdefault("damage_summary", f"부상자 {result.get('injured_count', 0)}명 추정")
    return report


def _ensure_prevention_actions(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list) and value:
        normalized = []
        for item in value:
            if isinstance(item, dict):
                normalized.append({
                    "priority": str(item.get("priority") or "medium"),
                    "action": str(item.get("action") or item.get("text") or ""),
                    "reason": str(item.get("reason") or ""),
                })
            else:
                normalized.append({"priority": "medium", "action": str(item), "reason": ""})
        return normalized
    return [
        {
            "priority": "medium",
            "action": "사고 원인이 불확실한 경우 현장 관리자가 원본 영상을 재검토합니다.",
            "reason": "VL 모델 판단만으로 단정하기 어려운 항목을 분리하기 위함입니다.",
        }
    ]
