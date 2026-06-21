from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT_DIR / "video"
AGENT_OUTPUT_DIR = ROOT_DIR / "agent" / "output"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

app = FastAPI(title="Construction Accident VL Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/videos")
def list_videos() -> dict[str, list[dict[str, Any]]]:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    videos = []
    for path in sorted(VIDEO_DIR.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            videos.append(_video_meta(path))
    return {"videos": videos}


@app.post("/api/videos")
def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="mp4, mov, avi, mkv 영상만 업로드할 수 있습니다.")

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = _safe_stem(Path(file.filename or "upload").stem)
    saved_name = f"{datetime.now():%Y%m%d_%H%M%S}_{safe_stem}{suffix}"
    saved_path = VIDEO_DIR / saved_name

    with saved_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    return {"video": _video_meta(saved_path)}


@app.get("/api/videos/{filename}")
def get_video(filename: str) -> FileResponse:
    path = (VIDEO_DIR / filename).resolve()
    if not _is_inside(path, VIDEO_DIR) or not path.exists():
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")
    return FileResponse(path)


@app.post("/api/analyze")
def analyze_video(
    filename: str = Form(...),
    api_base: str = Form(""),
    camera_id: str = Form("Camera 15"),
    zone_id: str = Form("construction_site"),
    zone_name: str = Form("건설현장 사고 구역"),
    scene_context: str = Form("건설현장 CCTV 사고 영상입니다. 영상에 보이는 행동, 구조물 변화, 사람의 위치 변화를 근거로 사고 원인을 판단합니다."),
) -> dict[str, Any]:
    video_path = (VIDEO_DIR / filename).resolve()
    if not _is_inside(video_path, VIDEO_DIR) or not video_path.exists():
        raise HTTPException(status_code=404, detail="분석할 mp4 파일을 찾을 수 없습니다.")

    AGENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AGENT_OUTPUT_DIR / "accident_analysis_payload.json"
    raw_output_path = AGENT_OUTPUT_DIR / "accident_judgment_result.json"
    env = os.environ.copy()
    if api_base.strip():
        env["LLM_API_BASE"] = api_base.strip()
    if scene_context.strip():
        env["AGENT_SCENE_CONTEXT"] = scene_context.strip()

    command = [
        sys.executable,
        "-m",
        "agent.run_remote_judgment",
        "--video",
        str(video_path),
        "--auto-moment",
        "--camera-id",
        camera_id,
        "--zone-id",
        zone_id,
        "--zone-name",
        zone_name,
        "--output",
        str(output_path),
        "--raw-output",
        str(raw_output_path),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=env,
            text=True,
            capture_output=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"분석 시간이 초과되었습니다. {exc}") from exc

    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "VL Agent 분석에 실패했습니다.",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )

    payload = _read_json(output_path)
    raw = _read_json(raw_output_path)
    return {
        "video": _video_meta(video_path),
        "analysis": _summary_from_payload(payload, raw),
        "payload": payload,
        "raw_judgment": raw,
        "logs": completed.stdout,
    }


def _video_meta(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "filename": path.name,
        "path": str(path),
        "url": f"/api/videos/{path.name}",
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def _summary_from_payload(payload: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    judgment = payload.get("judgment") or {}
    first_event = ((payload.get("video_part_tables") or {}).get("cctv_events") or [{}])[0]
    injured_count = raw.get("injured_count") or raw.get("injured_workers_count")
    if injured_count is None:
        injured_count = _estimate_injured_count(raw, judgment)

    return {
        "accident_type": judgment.get("accident_type"),
        "accident_type_ko": judgment.get("accident_type_ko"),
        "agent_verdict": judgment.get("agent_verdict"),
        "confidence": judgment.get("confidence"),
        "injured_count": injured_count,
        "cause": raw.get("cause") or raw.get("cause_summary") or _extract_cause(judgment.get("details")),
        "details": judgment.get("details"),
        "clip_start_offset": first_event.get("clip_start_offset"),
        "clip_end_offset": first_event.get("clip_end_offset"),
    }


def _estimate_injured_count(raw: dict[str, Any], judgment: dict[str, Any]) -> int:
    workers = raw.get("workers")
    if isinstance(workers, list):
        injured = [
            item for item in workers
            if isinstance(item, dict) and "피해" in str(item.get("accident_relation") or "")
        ]
        if injured:
            return len(injured)
    text = json.dumps(raw, ensure_ascii=False) + json.dumps(judgment, ensure_ascii=False)
    return 1 if any(token in text for token in ("추락", "낙상", "넘어", "부상", "피해")) else 0


def _extract_cause(details: Any) -> str:
    text = str(details or "").strip()
    if "원인-결과 흐름" in text:
        return text.split("원인-결과 흐름", 1)[1].split("\n", 1)[0].strip(" .:")
    if "->" in text:
        return text.split("\n")[-1].strip()
    return text[:180]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_stem(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in value).strip("_")
    return safe[:80] or "video"


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent.resolve())
    except ValueError:
        return False
    return True
