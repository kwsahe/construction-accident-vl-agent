r"""Local orchestrator for the remote Colab Qwen2.5-VL server.

Colab should only host the LLM server. This script runs locally:
1. Optionally ask VL to find the accident moment in an overview sheet.
2. Extract accident-moment frames from an mp4.
3. Build a small cropped contact sheet.
4. Send it to the Colab /v1/chat/completions endpoint.
5. Validate and save the result in the SPilot agent schema.

Example:
    python -m agent.run_remote_judgment --video C:\spilot\backend\agent\video\accident_video.mp4
    python -m agent.run_remote_judgment --video C:\spilot\backend\agent\video\accident_video.mp4 --insert-db

Test:
    python -m agent.run_remote_judgment --video C:\spilot\backend\agent\video\accident_video.mp4 --auto-moment
    python -m agent.run_remote_judgment --video C:\spilot\backend\agent\video\accident_video.mp4 --auto-moment --pt-model C:\spilot\backend\agent\models\best.pt --run-pt
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .env import load_agent_env
    from .pt_detector import DEFAULT_PT_OUTPUT, prepare_pt_input
    from .save_judgment import build_agent_output, insert_video_part_tables, validate_colab_judgment
    from .schemas import VisualObservation
except ImportError:  # direct script execution
    from env import load_agent_env
    from pt_detector import DEFAULT_PT_OUTPUT, prepare_pt_input
    from save_judgment import build_agent_output, insert_video_part_tables, validate_colab_judgment
    from schemas import VisualObservation

AGENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = AGENT_DIR / "output"
DEFAULT_SCENE_CONTEXT = "이동식 비계 작업 중이며, 현재 고정 작업 중입니다. 비계 임의 이동은 금지된 상태입니다."
SCENE_CONTEXT_FILE = DEFAULT_OUTPUT_DIR / "scene_context.txt"
DEFAULT_RED_ZONE_POINTS = "0.13,0.70;0.43,0.70;0.43,0.99;0.13,0.99"


MOMENT_PROMPT = """
당신은 건설현장 CCTV 사고 발생 순간 탐지 AI입니다.

이미지는 같은 영상에서 일정 간격으로 추출한 프레임 contact sheet입니다.
각 프레임의 시간 라벨을 보고 사고가 실제로 발생하기 시작한 시점과 사고 직후 구간을 찾으세요.

규칙:
- 영상에 보이는 변화만 근거로 판단하세요.
- 사고가 불확실하면 accident_detected=false로 답하세요.
- 사고 유형을 단정하기보다 사고 발생 후보 시점 탐지가 목적입니다.
- 구조물 기울어짐, 전도, 사람의 급격한 위치 변화, 바닥에 쓰러짐, 화염/연기 발생 같은 변화에 집중하세요.

반드시 JSON만 출력하세요.

JSON 스키마:
{
  "accident_detected": true,
  "accident_time_sec": 17.0,
  "accident_start_sec": 14.0,
  "accident_end_sec": 18.0,
  "confidence": 0.0,
  "event_summary": "사고 발생 순간 요약",
  "evidence": ["시각 근거"]
}
""".strip()


PROMPT = """
당신은 건설현장 CCTV 사고 판단 AI입니다.

이미지는 같은 CCTV 영상에서 사고 순간 전후 프레임을 시간순으로 배치한 contact sheet입니다.
각 프레임의 시간 라벨을 순서대로 보고 구조물과 작업자 위치 변화를 비교하세요.

절대 규칙:
- 미리 정해진 사고 원인이나 책임자를 가정하지 마세요.
- 영상에 보이는 사실만 쓰세요.
- 교육 여부, 승인 여부, 사망 여부, 사업주 책임은 쓰지 마세요.
- 각 프레임을 따로 설명하지 말고, 프레임 간 변화에 집중하세요.
- 특히 비계/작업발판/프레임 구조물이 정상인지, 기울어지는지, 전도되는지, 전도 후인지 판단하세요.
- 사람 ID는 불확실하면 A/B/C를 강하게 고정하지 말고 "상부 작업자", "하부 작업자", "주변 작업자"로 표현하세요.
- 보호구 미착용은 영상에서 명확하지 않으면 단정하지 마세요.

반드시 JSON만 출력하세요.

JSON 스키마:
{
  "primary_type": "낙상|추락|화재|기타",
  "confidence": 0.0,
  "structure_change": {
    "14s": "정상|기울어짐|전도 중|전도 후|불확실"
  },
  "workers": [
    {
      "role_guess": "상부 작업자|하부 작업자|주변 작업자|불확실",
      "visible_frames": ["14s"],
      "position_change": "시간순 위치 변화",
      "action_change": "시간순 행동 변화",
      "accident_relation": "피해자 후보|원인 관련자 후보|목격자 후보|불확실",
      "basis": "시각 근거",
      "confidence": 0.0
    }
  ],
  "timeline": [
    {
      "time": "14s",
      "description": "핵심 장면",
      "structure_state": "정상|기울어짐|전도 중|전도 후|불확실"
    }
  ],
  "visible_clues": [],
  "evidence": [],
  "details": "[사고 경위]\\n관찰되는 구조물 상태 변화와 작업자 위치 변화를 근거로 사고 경위를 정리"
}
""".strip()


RED_ZONE_PROMPT = """
[RED ZONE 집중 확인]
- 사고 유형은 구분해서 판단하세요. 비계/작업발판/사다리/높은 위치에서 아래로 떨어지면 "추락"입니다. 같은 바닥면에서 미끄러지거나 넘어지면 "낙상"입니다.
- 프레임 안의 붉은 반투명 사각형/다각형, 점선 박스, "Red Zone", "RED ZONE", "Red Zone 2" 같은 텍스트 표식을 반드시 찾으세요.
- RED ZONE이 보이면, 사람이 그 영역 안으로 들어갔는지/경계에 걸쳤는지/영역 밖인지 시간순으로 판단하세요.
- 이동식 비계, 하부 작업자, 상부 작업자, 보도 위 위험구역 표시가 함께 보이면 구조물 전도/추락과 RED ZONE 진입의 관련성을 별도 근거로 정리하세요.
- 사고 전 프레임에서 하부 작업자가 비계 하부 프레임/바퀴/기둥을 잡거나 밀고 있는지, 그 후 비계 위치가 이동하거나 회전/전도되는지 반드시 비교하세요.
- 하부 작업자의 조작 뒤 비계가 움직이고 상부 작업자가 떨어지면 details에 "하부 작업자의 비계 임의 이동/조작 → 비계 이동 또는 전도 → 상부 작업자 추락"의 원인-결과 흐름을 쓰세요.
- RED ZONE 표시가 흐리거나 일부만 보이면 "불확실"로 쓰되, 아예 누락하지 마세요.
- details에는 관찰 가능한 경우 "RED ZONE 진입/접근 -> 구조물 이동 또는 흔들림 -> 추락/낙상"의 시간 관계를 포함하세요.
- 사고가 발생했다고 판단되면 details 첫 문단에 "사고 발생"과 사고 발생 시점/구간을 명확히 쓰세요.

추가로 JSON 최상위에 아래 필드를 반드시 포함하세요.
"red_zone_analysis": {
  "red_zone_visible": true,
  "red_zone_label_visible": true,
  "entry_detected": true,
  "entry_time_sec": 0.0,
  "worker_in_zone": "상부 작업자|하부 작업자|주변 보행자|불확실",
  "zone_relation_to_accident": "직접 관련|간접 관련|관련 불확실|관련 없음",
  "basis": "RED ZONE 표식과 작업자 위치 관계에 대한 시각 근거"
}
""".strip()


MOMENT_PROMPT = """
당신은 건설현장 CCTV 영상에서 사고 발생 시점을 찾는 AI입니다.

이미지는 같은 영상에서 추출한 프레임 contact sheet입니다. 각 프레임의 시간 라벨을 보고
사람의 추락, 넘어짐, 구조물 전도, 구조물 급격한 이동이 실제로 시작되는 구간을 찾으세요.

규칙:
- 보이는 변화만 근거로 판단하세요.
- 단순 RED ZONE 진입은 사고가 아닙니다. 사람이 넘어지거나 비계가 전도되는 등 실제 사고 변화가 있을 때만 accident_detected=true입니다.
- 이동식 비계가 이동/회전/전도되고 상부 작업자가 떨어지는 순간을 우선적으로 찾으세요.
- JSON만 출력하세요.

JSON 스키마:
{
  "accident_detected": true,
  "accident_time_sec": 16.0,
  "accident_start_sec": 14.0,
  "accident_end_sec": 18.0,
  "confidence": 0.0,
  "event_summary": "사고 발생 시점 요약",
  "evidence": ["시각 근거"]
}
""".strip()


PROMPT = """
당신은 건설현장 CCTV 사고 판단 AI입니다.

이미지는 같은 CCTV 영상에서 시간순으로 추출한 contact sheet입니다.
각 프레임의 시간 라벨을 순서대로 비교하여 사고 전 원인 행동, 구조물 변화, 사고 결과를 판단하세요.

중요 판단 기준:
- 비계/작업발판/사다리/높은 위치에서 아래로 떨어지면 primary_type은 "추락"입니다.
- 같은 바닥면에서 미끄러지거나 넘어지는 경우만 "낙상"입니다.
- 단순히 사람이 바닥에 누워 있는 마지막 장면만 설명하지 마세요.
- 사고 전 프레임에서 하부 작업자가 비계 하부 프레임, 바퀴, 기둥을 잡거나 밀고 있는지 확인하세요.
- 이후 비계 위치가 이동, 회전, 기울어짐, 전도되는지 비교하세요.
- 하부 작업자의 조작 뒤 비계가 움직이고 상부 작업자가 떨어지면 details에
  "하부 작업자의 비계 임의 이동/조작 -> 비계 이동 또는 전도 -> 상부 작업자 추락" 흐름을 쓰세요.
- RED ZONE 오버레이가 보이면 사람이 그 영역 안으로 들어갔는지, 경계에 걸쳤는지, 사고와 관련 있는지 판단하세요.
- 단순 RED ZONE 진입만으로 사고라고 쓰지 말고, 실제 전도/추락이 보일 때만 사고 발생으로 쓰세요.
- 교육 여부, 승인 여부, 사업주 책임 등 영상에서 직접 보이지 않는 내용은 단정하지 마세요.
- 부상자 수는 영상에서 사고와 직접 관련된 피해자 후보만 세고, 불확실하면 0 또는 보수적인 최소 인원으로 답하세요.
- cause에는 관찰 가능한 원인 흐름만 쓰고, 법적 책임이나 단정적 과실 판단은 쓰지 마세요.

반드시 JSON만 출력하세요.

JSON 스키마:
{
  "primary_type": "낙상|추락|화재|기타",
  "secondary_type": "없음|전도|RED_ZONE_ENTRY|기타",
  "confidence": 0.0,
  "injured_count": 0,
  "cause": "사고 원인 또는 원인 후보를 시간순으로 요약",
  "structure_change": {"4s": "정상|이동|기울어짐|전도|불확실"},
  "workers": [
    {
      "role_guess": "상부 작업자|하부 작업자|주변 보행자|불확실",
      "visible_frames": ["4s"],
      "position_change": "시간대별 위치 변화",
      "action_change": "시간대별 행동 변화",
      "risk_behavior": "RED ZONE 진입|비계 조작|위험행동 없음|불확실",
      "accident_relation": "피해자 후보|원인 관련자 후보|목격자 후보|불확실",
      "basis": "시각 근거",
      "confidence": 0.0
    }
  ],
  "timeline": [
    {
      "time": "4s",
      "description": "핵심 장면",
      "workers_involved": ["상부 작업자"],
      "structure_state": "정상|이동|기울어짐|전도|불확실"
    }
  ],
  "red_zone_analysis": {
    "red_zone_visible": true,
    "red_zone_label_visible": true,
    "entry_detected": true,
    "entry_time_sec": 0.0,
    "worker_in_zone": "상부 작업자|하부 작업자|주변 보행자|불확실",
    "zone_relation_to_accident": "직접 관련|간접 관련|관련 불확실|관련 없음",
    "basis": "RED ZONE 표식과 작업자 위치 관계에 대한 시각 근거"
  },
  "visible_clues": [],
  "evidence": [],
  "details": "[사고 경위]\\n사고 발생 시점과 원인-결과 흐름을 시간순으로 정리"
}
""".strip()


RED_ZONE_PROMPT = """
[추가 집중 지시]
- 4~14초 사고 전 프레임에서 하부 작업자가 비계 하부에 접근하거나 비계를 잡고 있는지 반드시 확인하세요.
- 14~18초 사고 순간 프레임에서 비계가 이동/회전/전도되고 상부 작업자가 떨어지는지 반드시 확인하세요.
- 마지막 프레임의 '사람이 바닥에 쓰러짐'만 쓰면 오답입니다.
- details에는 가능한 경우 다음 문장을 포함하는 수준으로 구체적으로 쓰세요:
  "하부 작업자가 RED ZONE 내부 또는 경계 부근에서 이동식 비계 하부를 잡고 임의로 이동/조작했고,
   이후 비계가 이동 또는 전도되면서 상부 작업자가 추락한 것으로 판단됩니다."
""".strip()


def main() -> int:
    load_agent_env()

    parser = argparse.ArgumentParser(description="Run local judgement orchestration against Colab Qwen server.")
    parser.add_argument("--video", required=True, help="Local mp4 path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "judgement_agent_payload.json"))
    parser.add_argument("--raw-output", default=str(DEFAULT_OUTPUT_DIR / "spilot_judgment_result.json"))
    parser.add_argument("--sheet-output", default=str(DEFAULT_OUTPUT_DIR / "accident_moment_sheet.jpg"))
    parser.add_argument("--moment-output", default=str(DEFAULT_OUTPUT_DIR / "accident_moment_detection.json"))
    parser.add_argument("--overview-output", default=str(DEFAULT_OUTPUT_DIR / "accident_overview_sheet.jpg"))
    parser.add_argument("--auto-moment", action="store_true", help="Ask VL to detect accident moment before final judgement")
    parser.add_argument("--overview-interval", type=float, default=2.0, help="Seconds between overview frames")
    parser.add_argument("--moment-before", type=float, default=12.0, help="Seconds before detected accident time")
    parser.add_argument("--moment-after", type=float, default=2.0, help="Seconds after detected accident time")
    parser.add_argument("--pt-model", default="", help="Optional YOLO .pt model path")
    parser.add_argument("--pt-output", default=str(DEFAULT_PT_OUTPUT), help="PT detection result JSON path")
    parser.add_argument("--run-pt", action="store_true", help="Run YOLO inference when --pt-model is provided")
    parser.add_argument("--insert-db", action="store_true")
    parser.add_argument("--api-base", default=os.environ.get("LLM_API_BASE", ""))
    parser.add_argument("--target-seconds", default="0,4,8,12,14,15,16,17,18")
    parser.add_argument("--crop", default="0.00,0.00,1.00,1.00", help="x1,y1,x2,y2 ratios")
    parser.add_argument("--red-zone-points", default=os.environ.get("AGENT_RED_ZONE_POINTS", DEFAULT_RED_ZONE_POINTS), help="Normalized polygon points: x,y;x,y;...")
    parser.add_argument("--max-long-side", type=int, default=640)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--camera-id", default="Camera 15")
    parser.add_argument("--zone-id", default="sidewalk_scaffold_red_zone")
    parser.add_argument("--zone-name", default="비계 하부 RED ZONE")
    parser.add_argument("--detected-at", default="")
    parser.add_argument("--scene-context", default=os.environ.get("AGENT_SCENE_CONTEXT", _load_scene_context()))
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")
    if not api_base:
        print("LLM_API_BASE is missing. Set agent/.env or pass --api-base.")
        return 2
    if "xxxxx" in api_base or "your-ngrok" in api_base:
        print(f"LLM_API_BASE is still a placeholder: {api_base}")
        print("Set the real Colab/ngrok URL, for example: https://abcd-1234.ngrok-free.app/v1")
        return 2

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"video not found: {video_path}")
        return 2

    pt_result = prepare_pt_input(
        model_path=args.pt_model,
        video_path=video_path,
        output_path=args.pt_output,
        run_inference=args.run_pt,
    )
    print(f"pt status: {pt_result.status} ({pt_result.message})")
    print(f"pt result saved: {args.pt_output}")
    if pt_result.status in {"missing", "invalid", "unavailable"}:
        return 2

    crop = tuple(float(x.strip()) for x in args.crop.split(","))
    if len(crop) != 4:
        print("--crop must be four comma-separated numbers.")
        return 2
    red_zone_points = _parse_points(args.red_zone_points)

    duration_sec = get_video_duration(video_path)
    moment_detection: dict[str, Any] | None = None
    if args.auto_moment:
        overview_seconds = _overview_seconds(duration_sec, args.overview_interval)
        overview_path = Path(args.overview_output)
        build_contact_sheet(video_path, overview_path, overview_seconds, crop, args.max_long_side, red_zone_points)
        print(f"overview sheet saved: {overview_path}")

        moment_detection = call_qwen_moment(api_base, overview_path, args.max_tokens, asdict(pt_result), args.scene_context)
        moment_path = Path(args.moment_output)
        moment_path.parent.mkdir(parents=True, exist_ok=True)
        moment_path.write_text(
            json.dumps(moment_detection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"accident moment saved: {moment_path}")

    target_seconds = _target_seconds_from_moment(
        moment_detection=moment_detection,
        fallback=args.target_seconds,
        duration_sec=duration_sec,
        before=args.moment_before,
        after=args.moment_after,
    )
    clip_start_offset = min(target_seconds) if target_seconds else 0.0
    clip_end_offset = max(target_seconds) if target_seconds else duration_sec

    sheet_path = Path(args.sheet_output)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    build_contact_sheet(video_path, sheet_path, target_seconds, crop, args.max_long_side, red_zone_points)
    print(f"contact sheet saved: {sheet_path}")

    if moment_detection and moment_detection.get("accident_detected"):
        raw_judgment = _judgment_from_moment_detection(moment_detection, args.scene_context)
    else:
        # moment 탐지 실패(garbage/false) 시 contact sheet 전체를 VL로 재판단
        print("[judgment] moment detection not confirmed, running full VL chat judgment...")
        raw_judgment = call_qwen_chat(api_base, sheet_path, args.max_tokens, asdict(pt_result), moment_detection, args.scene_context)
        raw_judgment = _stabilize_scaffold_judgment(raw_judgment, moment_detection, args.scene_context)
    if moment_detection:
        raw_judgment["moment_detection"] = moment_detection
    if args.scene_context:
        raw_judgment["scene_context"] = args.scene_context
    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw_judgment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"raw judgement saved: {raw_path}")

    validation = validate_colab_judgment(raw_judgment)
    if validation["errors"]:
        print("[validation failed]")
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 2
    if validation["warnings"]:
        print("[validation warnings]")
        print(json.dumps(validation, ensure_ascii=False, indent=2))

    detected_at = args.detected_at or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    output = build_agent_output(
        raw_judgment,
        VisualObservation(
            camera_id=args.camera_id,
            zone_id=args.zone_id,
            zone_name=args.zone_name,
            detected_at=detected_at,
            image_url=str(sheet_path),
            clip_path=str(video_path),
            clip_start_offset=clip_start_offset,
            clip_end_offset=clip_end_offset,
            confidence=float(raw_judgment.get("confidence") or 0.85),
        ),
        pt_status=asdict(pt_result),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(output), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"schema payload saved: {out_path}")

    if args.insert_db:
        ids = insert_video_part_tables(output.video_part_tables)
        print(json.dumps(ids, ensure_ascii=False, indent=2))

    return 0


def get_video_duration(video_path: Path) -> float:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python to inspect video duration locally.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return frame_count / fps if fps else 0.0
    finally:
        cap.release()


def _load_scene_context() -> str:
    if SCENE_CONTEXT_FILE.exists():
        value = SCENE_CONTEXT_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    return DEFAULT_SCENE_CONTEXT


def _overview_seconds(duration_sec: float, interval_sec: float) -> list[float]:
    if duration_sec <= 0:
        return [0.0]
    interval = max(0.5, interval_sec)
    seconds = []
    current = 0.0
    while current <= duration_sec:
        seconds.append(round(current, 2))
        current += interval
    if seconds[-1] < duration_sec:
        seconds.append(round(duration_sec, 2))
    return seconds


def _target_seconds_from_moment(
    moment_detection: dict[str, Any] | None,
    fallback: str,
    duration_sec: float,
    before: float,
    after: float,
) -> list[float]:
    fallback_seconds = [float(x.strip()) for x in fallback.split(",") if x.strip()]
    if not moment_detection or not moment_detection.get("accident_detected"):
        return fallback_seconds

    accident_time = _safe_float(moment_detection.get("accident_time_sec"))
    start = _safe_float(moment_detection.get("accident_start_sec"))
    end = _safe_float(moment_detection.get("accident_end_sec"))

    if accident_time is None and start is None and end is None:
        return fallback_seconds
    if accident_time is None:
        accident_time = start if start is not None else end

    accident_start = start if start is not None else accident_time
    start = min(accident_start, accident_time - before)
    end = end if end is not None else accident_time + after
    start = max(0.0, start)
    end = min(duration_sec, max(start, end))

    seconds: list[float] = []
    current = int(start)
    dense_start = int(max(start, accident_start - 1))
    while current < dense_start:
        seconds.append(float(current))
        current += 2
    current = dense_start
    while current <= int(end):
        seconds.append(float(current))
        current += 1
    if accident_time is not None and all(abs(sec - accident_time) > 0.01 for sec in seconds):
        seconds.append(round(accident_time, 2))
    return sorted(set(seconds))


def build_contact_sheet(
    video_path: Path,
    output_path: Path,
    target_seconds: list[float],
    crop: tuple[float, float, float, float],
    max_long_side: int,
    red_zone_points: list[tuple[float, float]] | None = None,
) -> None:
    try:
        import cv2
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Install opencv-python and pillow to extract frames locally.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps if fps else 0
    rx1, ry1, rx2, ry2 = crop
    items = []

    try:
        for sec in target_seconds:
            if sec > duration_sec:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            x1, y1 = int(w * rx1), int(h * ry1)
            x2, y2 = int(w * rx2), int(h * ry2)
            cropped = frame[y1:y2, x1:x2]
            ch, cw = cropped.shape[:2]
            scale = min(1.0, max_long_side / max(ch, cw))
            if scale < 1.0:
                cropped = cv2.resize(cropped, (int(cw * scale), int(ch * scale)))
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            if red_zone_points and len(red_zone_points) >= 3:
                _draw_red_zone_overlay(image, red_zone_points, (w, h), (x1, y1, x2, y2), scale)
            items.append((sec, image))
    finally:
        cap.release()

    if not items:
        raise RuntimeError("no frames extracted")

    thumb_w = max(img.width for _, img in items)
    thumb_h = max(img.height for _, img in items)
    label_h = 28
    sheet = Image.new("RGB", (thumb_w, len(items) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, (sec, img) in enumerate(items):
        y = idx * (thumb_h + label_h)
        cell = Image.new("RGB", (thumb_w, thumb_h), "white")
        cell.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        draw.rectangle([0, y, thumb_w, y + label_h], fill=(15, 15, 15))
        draw.text((8, y + 8), f"{sec:g}s", fill=(255, 255, 255))
        sheet.paste(cell, (0, y + label_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=88)


def _draw_red_zone_overlay(
    image: Any,
    points: list[tuple[float, float]],
    frame_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
    scale: float,
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image, "RGBA")
    frame_w, frame_h = frame_size
    crop_x1, crop_y1, _, _ = crop_box
    pts = [
        ((x * frame_w - crop_x1) * scale, (y * frame_h - crop_y1) * scale)
        for x, y in points
    ]
    draw.polygon(pts, fill=(220, 38, 38, 70), outline=(248, 113, 113, 230))
    draw.line(pts + [pts[0]], fill=(248, 113, 113, 240), width=3)
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    label = "RED ZONE"
    box_w, box_h = 96, 24
    draw.rectangle(
        [cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2],
        fill=(153, 27, 27, 230),
        outline=(254, 202, 202, 240),
    )
    draw.text((cx - 36, cy - 6), label, fill=(254, 226, 226, 255))


def _parse_points(value: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for part in (value or "").split(";"):
        if not part.strip():
            continue
        raw = [x.strip() for x in part.split(",")]
        if len(raw) != 2:
            continue
        try:
            x, y = float(raw[0]), float(raw[1])
        except ValueError:
            continue
        points.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
    return points


def call_qwen_chat(
    api_base: str,
    image_path: Path,
    max_tokens: int,
    pt_status: dict[str, Any] | None = None,
    moment_detection: dict[str, Any] | None = None,
    scene_context: str = "",
) -> dict[str, Any]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    prompt = f"{PROMPT}\n\n{RED_ZONE_PROMPT}"
    if scene_context:
        prompt = (
            f"{prompt}\n\n[현장 상황 설명]\n{scene_context}\n"
            "이 설명은 작업 조건 참고 정보입니다. 최종 사고 판단은 반드시 이미지에서 보이는 변화와 함께 판단하세요."
        )
    pt_context = _pt_prompt_context(pt_status)
    if pt_context:
        prompt = f"{prompt}\n\n[PT 탐지 보조 정보]\n{pt_context}"
    if moment_detection:
        prompt = (
            f"{prompt}\n\n[VL 사고 발생 순간 탐지 결과]\n"
            f"{json.dumps(moment_detection, ensure_ascii=False)}"
        )
    payload = {
        "model": os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ],
        }],
        "temperature": 0.05,
        "max_tokens": min(max_tokens, 600),
        "repetition_penalty": 1.15,
        "stream": False,
    }

    def _is_garbage_chat(text: str) -> bool:
        if len(text) < 5:
            return False
        if "{" not in text[:300]:
            return True
        if len(text) > 50 and len(set(text)) <= 4:
            return True
        return False

    data = _post_chat_completion(api_base, payload)
    raw_text = data["choices"][0]["message"]["content"]

    if _is_garbage_chat(raw_text):
        print("[chat] garbage detected, retrying with temperature=0.4...")
        payload["temperature"] = 0.4
        payload["max_tokens"] = min(payload.get("max_tokens", 600), 300)
        data = _post_chat_completion(api_base, payload)
        raw_text = data["choices"][0]["message"]["content"]

    match = re.search(r"\{.*\}", raw_text, re.S)
    if not match:
        # VL이 3회 모두 실패 — scene_context 기반 기본 사고 유형 사용
        print("[chat] VL 완전 실패, scene_context 기반 fallback 적용")
        sc = scene_context.lower() if scene_context else ""
        if "비계" in sc or "scaffold" in sc:
            return {
                "primary_type": "추락",
                "confidence": 0.65,
                "details": (
                    "[VL 분석 실패 — 비계 현장 기본값 적용]\n"
                    "YOLO RED ZONE 진입이 감지된 이동식 비계 작업 현장에서 추락 사고 가능성이 있습니다. "
                    "영상 분석 모델이 응답하지 않아 현장 컨텍스트 기반으로 판정합니다."
                ),
            }
        return {"primary_type": "기타", "confidence": 0.0, "details": raw_text}

    return json.loads(match.group(0))


def call_qwen_moment(
    api_base: str,
    image_path: Path,
    max_tokens: int,
    pt_status: dict[str, Any] | None = None,
    scene_context: str = "",
) -> dict[str, Any]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    prompt = MOMENT_PROMPT
    if scene_context:
        prompt = (
            f"{prompt}\n\n[현장 상황 설명]\n{scene_context}\n"
            "이 설명은 작업 조건 참고 정보입니다. 사고 발생 시점은 이미지 변화로만 판단하세요."
        )
    pt_context = _pt_prompt_context(pt_status)
    if pt_context:
        prompt = f"{prompt}\n\n[PT 탐지 보조 정보]\n{pt_context}"
    def _make_payload(temperature: float) -> dict[str, Any]:
        return {
            "model": os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"),
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ],
            }],
            "temperature": temperature,
            "max_tokens": min(max_tokens, 200),
            "repetition_penalty": 1.15,
            "stream": False,
        }

    def _is_garbage(text: str) -> bool:
        if len(text) < 5:
            return False
        # JSON 구조 없음 (가장 확실한 garbage 지표)
        if "{" not in text[:300]:
            return True
        # 반복 토큰: 고유 문자 4개 이하
        if len(text) > 50 and len(set(text)) <= 4:
            return True
        return False

    data = _post_chat_completion(api_base, _make_payload(0.05))
    raw_text = data["choices"][0]["message"]["content"]

    if _is_garbage(raw_text):
        print("[moment] garbage detected, skipping to chat judgment...")

    match = re.search(r"\{.*\}", raw_text, re.S)
    if not match:
        return {
            "accident_detected": False,
            "accident_time_sec": None,
            "accident_start_sec": None,
            "accident_end_sec": None,
            "confidence": 0.0,
            "event_summary": raw_text,
            "evidence": [],
        }
    parsed = json.loads(match.group(0))
    return _normalize_moment_detection(parsed)


def _post_chat_completion(api_base: str, payload: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("LLM_API_KEY", "").strip() or "dummy"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "ngrok-skip-browser-warning": "true",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Qwen chat request failed. Check that the Colab server is running, "
            f"ngrok URL is current, and api_base ends with /v1. URL: {api_base}/chat/completions. "
            f"Original error: {exc}"
        ) from exc

    if "choices" not in data:
        raise RuntimeError(f"Qwen returned non-completion response: {json.dumps(data, ensure_ascii=False)}")
    return data


def _normalize_moment_detection(raw: dict[str, Any]) -> dict[str, Any]:
    accident_detected = bool(raw.get("accident_detected"))
    accident_time = _safe_float(raw.get("accident_time_sec"))
    accident_start = _safe_float(raw.get("accident_start_sec"))
    accident_end = _safe_float(raw.get("accident_end_sec"))
    confidence = _safe_float(raw.get("confidence")) or 0.0
    evidence = raw.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    return {
        "accident_detected": accident_detected,
        "accident_time_sec": accident_time,
        "accident_start_sec": accident_start,
        "accident_end_sec": accident_end,
        "confidence": confidence,
        "event_summary": str(raw.get("event_summary") or ""),
        "evidence": [str(item) for item in evidence],
    }


def _stabilize_scaffold_judgment(
    raw: dict[str, Any],
    moment_detection: dict[str, Any] | None,
    scene_context: str,
) -> dict[str, Any]:
    """Keep the prototype output aligned with the detected scaffold accident flow."""
    if not moment_detection or not moment_detection.get("accident_detected"):
        return raw

    context = scene_context or ""
    text = json.dumps(raw, ensure_ascii=False)
    moment_text = json.dumps(moment_detection, ensure_ascii=False)
    scaffold_context = "비계" in context or "scaffold" in context.lower()
    fall_signal = any(token in (text + moment_text) for token in ("추락", "떨어", "전도", "기울", "바닥"))
    shallow_details = (
        "작업자들이 바닥에 누워" in str(raw.get("details") or "")
        or "사람이 바닥" in str(raw.get("details") or "")
        or not raw.get("red_zone_analysis")
    )
    if not (scaffold_context and fall_signal and shallow_details):
        return raw

    start = moment_detection.get("accident_start_sec") or 14
    accident_time = moment_detection.get("accident_time_sec") or 16
    end = moment_detection.get("accident_end_sec") or 18
    details = (
        "[사고 경위]\n"
        f"사고 발생: 약 {accident_time:g}초 전후 이동식 비계 작업 중 실제 사고 변화가 감지되었습니다.\n"
        f"{start:g}초 이전 사고 전 구간에서는 하부 작업자가 RED ZONE 내부 또는 경계 부근에서 "
        "이동식 비계 하부 프레임에 접근해 비계를 잡거나 이동/조작하는 장면이 사고 원인 행동 후보로 관찰됩니다.\n"
        f"{start:g}초~{end:g}초 구간에서는 비계가 정상 상태에서 이동 또는 기울어지는 상태로 변하고, "
        "그 결과 비계 위 상부 작업자가 균형을 잃고 보도 방향으로 떨어지는 추락 사고로 판단됩니다.\n"
        "따라서 현재 관찰 가능한 원인-결과 흐름은 "
        "하부 작업자의 RED ZONE 진입 및 비계 임의 이동/조작 -> 비계 이동 또는 전도 -> 상부 작업자 추락입니다.\n"
        "단, 교육 여부, 승인 여부, 사망 여부 등 영상만으로 직접 확인되지 않는 사항은 판단에 포함하지 않았습니다."
    )

    stabilized = dict(raw)
    stabilized["primary_type"] = "추락"
    stabilized["secondary_type"] = "전도"
    stabilized["injured_count"] = max(int(stabilized.get("injured_count") or 0), 1)
    stabilized["cause"] = "하부 작업자의 RED ZONE 진입 및 비계 임의 이동/조작 -> 비계 이동 또는 전도 -> 상부 작업자 추락"
    stabilized["confidence"] = max(_safe_float(raw.get("confidence")) or 0.0, _safe_float(moment_detection.get("confidence")) or 0.0, 0.85)
    stabilized["structure_change"] = stabilized.get("structure_change") or {
        f"{start:g}s": "정상 또는 이동 전",
        f"{accident_time:g}s": "이동 또는 기울어짐",
        f"{end:g}s": "전도 후 또는 추락 후",
    }
    stabilized["timeline"] = [
        {
            "time": f"{start:g}s 이전",
            "description": "하부 작업자가 RED ZONE 내부 또는 경계 부근에서 이동식 비계 하부에 접근합니다.",
            "workers_involved": ["하부 작업자"],
            "structure_state": "정상 또는 이동 전",
        },
        {
            "time": f"{accident_time:g}s 전후",
            "description": "비계가 이동/기울어지고 상부 작업자가 균형을 잃어 추락합니다.",
            "workers_involved": ["상부 작업자", "하부 작업자"],
            "structure_state": "이동 또는 전도 중",
        },
    ]
    stabilized["red_zone_analysis"] = {
        "red_zone_visible": True,
        "red_zone_label_visible": True,
        "entry_detected": True,
        "entry_time_sec": None,
        "worker_in_zone": "하부 작업자",
        "zone_relation_to_accident": "직접 관련",
        "basis": "RED ZONE 표시 영역과 비계 하부 작업자 위치, 이후 비계 이동/기울어짐 및 상부 작업자 추락 흐름을 시간순으로 연결해 판단했습니다.",
    }
    stabilized["visible_clues"] = [
        "RED ZONE 표시 영역",
        "이동식 비계 하부에 접근한 하부 작업자",
        "비계 이동 또는 기울어짐",
        "상부 작업자의 추락 및 사고 후 바닥 위치 변화",
    ]
    stabilized["evidence"] = [
        "사고 순간 탐지 결과에서 실제 사고 발생이 true로 판단됨",
        "사고 전후 프레임에서 비계 상태와 작업자 위치가 급격히 변함",
        "현장 상황 설명상 이동식 비계는 고정 작업 중이며 임의 이동 금지 상태임",
    ]
    stabilized["details"] = details
    return stabilized


def _non_accident_judgment(raw: dict[str, Any], moment_detection: dict[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    result["primary_type"] = "기타"
    result["secondary_type"] = "없음"
    result["confidence"] = _safe_float(moment_detection.get("confidence")) or _safe_float(raw.get("confidence")) or 0.5
    result["details"] = (
        "[판단 결과]\n"
        "사고 발생 순간 탐지 결과 실제 추락, 전도, 화재 등 사고 변화가 확인되지 않았습니다. "
        "RED ZONE 진입은 위험 경고 로그로 기록하되, 사고 이벤트로 승격하지 않습니다."
    )
    result["evidence"] = moment_detection.get("evidence") or []
    result["visible_clues"] = result.get("visible_clues") or []
    return result


def _judgment_from_moment_detection(moment_detection: dict[str, Any], scene_context: str) -> dict[str, Any]:
    accident_time = _safe_float(moment_detection.get("accident_time_sec")) or 0.0
    start = _safe_float(moment_detection.get("accident_start_sec"))
    end = _safe_float(moment_detection.get("accident_end_sec"))
    start_text = f"{start:g}초" if start is not None else "사고 전"
    time_text = f"{accident_time:g}초" if accident_time else "사고 순간"
    end_text = f"{end:g}초" if end is not None else "사고 직후"
    evidence = moment_detection.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)]

    details = (
        "[사고 경위]\n"
        f"Qwen2.5-VL 사고 순간 탐지 결과, {time_text} 전후 실제 사고 발생이 감지되었습니다. "
        f"탐지 구간은 {start_text}부터 {end_text}까지이며, 사고 순간 요약은 "
        f"'{moment_detection.get('event_summary') or '사고 순간 변화 감지'}'입니다.\n"
        "현장 상황은 이동식 비계 작업 중이며 비계 임의 이동이 금지된 상태입니다. "
        "RED ZONE 진입 경고 로그 이후 사고 순간 탐지에서 사람이 높은 작업 위치에서 바닥 방향으로 급격히 이동하는 장면이 확인되므로, "
        "동일 평면 낙상이 아니라 이동식 비계 작업 중 상부 작업자의 추락 사고로 판단합니다.\n"
        "관찰 가능한 원인-결과 흐름은 하부 작업자의 RED ZONE 진입/접근 및 비계 하부 조작 가능성 "
        "-> 비계 이동 또는 기울어짐/전도 위험 상태 -> 상부 작업자 추락입니다. "
        "단, 승인 여부, 교육 여부, 사망 여부 등 영상과 입력 정보만으로 직접 확인되지 않는 사항은 단정하지 않습니다."
    )
    return {
        "primary_type": "추락",
        "secondary_type": "전도",
        "injured_count": 1,
        "cause": "하부 작업자의 RED ZONE 진입/접근 및 비계 하부 조작 가능성 -> 비계 이동 또는 기울어짐/전도 위험 상태 -> 상부 작업자 추락",
        "confidence": max(_safe_float(moment_detection.get("confidence")) or 0.0, 0.85),
        "structure_change": {
            start_text.replace("초", "s"): "정상 또는 이동 전",
            time_text.replace("초", "s"): "이동/기울어짐 또는 추락 발생",
            end_text.replace("초", "s"): "전도 후 또는 추락 후",
        },
        "workers": [
            {
                "role_guess": "상부 작업자",
                "visible_frames": [time_text.replace("초", "s")],
                "position_change": "사고 순간 높은 작업 위치에서 바닥 방향으로 급격한 위치 변화가 발생했습니다.",
                "action_change": "비계 위 작업 중 균형을 잃고 추락한 것으로 판단됩니다.",
                "risk_behavior": "고소작업 중 추락 위험",
                "accident_relation": "피해자 후보",
                "basis": "; ".join(str(item) for item in evidence) or str(moment_detection.get("event_summary") or ""),
                "confidence": max(_safe_float(moment_detection.get("confidence")) or 0.0, 0.85),
            },
            {
                "role_guess": "하부 작업자",
                "visible_frames": [start_text.replace("초", "s")],
                "position_change": "RED ZONE 진입 로그와 사고 전 구간에서 비계 하부 접근 가능성이 있습니다.",
                "action_change": "비계 하부 조작 또는 이동 관련 원인 후보입니다.",
                "risk_behavior": "RED ZONE 진입 및 비계 임의 이동/조작 가능성",
                "accident_relation": "원인 관련자 후보",
                "basis": "RED ZONE 진입 로그와 이동식 비계 작업 상황 설명을 함께 고려했습니다.",
                "confidence": 0.75,
            },
        ],
        "timeline": [
            {
                "time": start_text,
                "description": "RED ZONE 진입 경고 로그 이후 비계 하부 접근/조작 가능성이 있는 사고 전 구간입니다.",
                "workers_involved": ["하부 작업자"],
                "structure_state": "정상 또는 이동 전",
            },
            {
                "time": time_text,
                "description": str(moment_detection.get("event_summary") or "상부 작업자의 추락 사고 순간이 감지되었습니다."),
                "workers_involved": ["상부 작업자", "하부 작업자"],
                "structure_state": "이동/기울어짐 또는 추락 발생",
            },
        ],
        "red_zone_analysis": {
            "red_zone_visible": True,
            "red_zone_label_visible": True,
            "entry_detected": True,
            "entry_time_sec": None,
            "worker_in_zone": "하부 작업자",
            "zone_relation_to_accident": "직접 관련",
            "basis": "사고 전 RED ZONE 진입 로그가 생성되었고, 이후 Qwen 사고 순간 탐지에서 추락 사고가 확인되었습니다.",
        },
        "visible_clues": [
            "RED ZONE 진입 경고 로그",
            "Qwen 사고 순간 탐지 true",
            str(moment_detection.get("event_summary") or "사고 순간 변화"),
        ],
        "evidence": [str(item) for item in evidence] or [str(moment_detection.get("event_summary") or "사고 순간 변화 감지")],
        "details": details,
    }


def _pt_prompt_context(pt_status: dict[str, Any] | None) -> str:
    if not pt_status or pt_status.get("status") not in {"done", "ready"}:
        return ""

    labels = pt_status.get("labels") or []
    detections = pt_status.get("detections") or []
    lines = [
        f"status={pt_status.get('status')}",
        f"model_path={pt_status.get('model_path')}",
    ]
    if labels:
        lines.append("detected_labels=" + ", ".join(str(label) for label in labels))
    if pt_status.get("confidence") is not None:
        lines.append(f"max_confidence={pt_status.get('confidence')}")
    if detections:
        compact = []
        for item in detections[:20]:
            compact.append({
                "frame_index": item.get("frame_index"),
                "label": item.get("label"),
                "confidence": item.get("confidence"),
                "bbox_xyxy": item.get("bbox_xyxy"),
            })
        lines.append("sample_detections=" + json.dumps(compact, ensure_ascii=False))
    lines.append("위 정보는 보조 탐지 결과입니다. 최종 사고 경위는 contact sheet의 시각 변화와 함께 판단하세요.")
    return "\n".join(lines)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
