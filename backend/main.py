from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agent.pt_detector import DEFAULT_PT_OUTPUT, prepare_pt_input

ROOT_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT_DIR / "video"
AGENT_OUTPUT_DIR = ROOT_DIR / "agent" / "output"
EVAL_OUTPUT_DIR = ROOT_DIR / "eval" / "output"
YOLO_MODEL_DIR = ROOT_DIR / "agent" / "models"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
ANALYSIS_TIMEOUT_SECONDS = int(os.environ.get("ANALYSIS_TIMEOUT_SECONDS", "1800"))
MODEL_NAME_BY_KEY = {
    "qwen3_vl_32b": "Qwen/Qwen3-VL-32B-Instruct",
    "internvl3": "OpenGVLab/InternVL3-38B",
    "llava_onevision_2_8b": "lmms-lab/LLaVA-OneVision-2-8B-ov",
    "minicpm_v_4_5": "openbmb/MiniCPM-V-4_5",
    "qwen25_vl_32b": "Qwen/Qwen2.5-VL-32B-Instruct",
}
OLLAMA_DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

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


@app.post("/api/videos/youtube")
def download_youtube_video(url: str = Form(...)) -> dict[str, Any]:
    if not url.strip().lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="YouTube URL must start with http:// or https://.")

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    output_template = str(VIDEO_DIR / "%(title).80s_%(id)s.%(ext)s")
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "-f",
        "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        url.strip(),
    ]
    try:
        completed = subprocess.run(command, cwd=ROOT_DIR, text=True, capture_output=True, timeout=900, check=False)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"YouTube download timed out. {exc}") from exc

    if completed.returncode != 0:
        raise HTTPException(status_code=500, detail={"message": "YouTube download failed.", "stderr": completed.stderr})

    videos = [path for path in VIDEO_DIR.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS]
    if not videos:
        raise HTTPException(status_code=500, detail="Downloaded video file was not found.")
    latest = max(videos, key=lambda item: item.stat().st_mtime)
    return {"video": _video_meta(latest), "logs": completed.stdout}


@app.get("/api/videos/{filename}")
def get_video(filename: str) -> FileResponse:
    path = (VIDEO_DIR / filename).resolve()
    if not _is_inside(path, VIDEO_DIR) or not path.exists():
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")
    return FileResponse(path)


@app.get("/api/output/{filename}")
def get_output_file(filename: str) -> FileResponse:
    path = (AGENT_OUTPUT_DIR / filename).resolve()
    if not _is_inside(path, AGENT_OUTPUT_DIR) or not path.exists():
        raise HTTPException(status_code=404, detail="출력 파일을 찾을 수 없습니다.")
    return FileResponse(path)


@app.post("/api/yolo/analyze")
def analyze_yolo_only(
    filename: str = Form(...),
    yolo_model: str = Form("yolo26n.pt"),
) -> dict[str, Any]:
    video_path = (VIDEO_DIR / filename).resolve()
    if not _is_inside(video_path, VIDEO_DIR) or not video_path.exists():
        raise HTTPException(status_code=404, detail="YOLO를 적용할 mp4 파일을 찾을 수 없습니다.")

    YOLO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = (YOLO_MODEL_DIR / yolo_model).resolve()
    if not _is_inside(model_path, YOLO_MODEL_DIR):
        raise HTTPException(status_code=400, detail="YOLO 모델 파일은 agent/models 폴더 안에 있어야 합니다.")

    annotated_path = AGENT_OUTPUT_DIR / f"yolo_annotated_{video_path.stem}.mp4"
    result = prepare_pt_input(
        model_path=str(model_path),
        video_path=video_path,
        output_path=DEFAULT_PT_OUTPUT,
        run_inference=True,
        annotated_output_path=annotated_path,
    )
    annotated_url = f"/api/output/{annotated_path.name}" if annotated_path.exists() else ""
    return {
        "video": _video_meta(video_path),
        "result": asdict(result),
        "annotated_video_url": annotated_url,
        "result_url": f"/api/output/{DEFAULT_PT_OUTPUT.name}",
    }


@app.get("/api/evaluation/summary")
def evaluation_summary() -> dict[str, Any]:
    """Return model/prompt evaluation results for the frontend dashboard."""
    summary_path = EVAL_OUTPUT_DIR / "eval_summary.json"
    if summary_path.exists():
        return _read_json(summary_path)
    return _default_eval_summary()


@app.get("/api/llm/status")
def llm_status(api_base: str = "") -> dict[str, Any]:
    base = api_base.strip().rstrip("/") or os.environ.get("LLM_API_BASE", "").rstrip("/")
    if not base:
        return {"live": False, "model": "", "api_base": "", "message": "LLM_API_BASE is empty."}
    root = base[:-3] if base.endswith("/v1") else base
    try:
        with urllib.request.urlopen(root + "/health", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        return {
            "live": data.get("status") == "ok",
            "model": data.get("model", ""),
            "api_base": base,
            "message": "connected",
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"live": False, "model": "", "api_base": base, "message": str(exc)}


@app.get("/api/ollama/status")
def ollama_status(base_url: str = "") -> dict[str, Any]:
    root = (base_url.strip() or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
    try:
        with urllib.request.urlopen(root + "/api/tags", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        models = []
        for item in data.get("models", []):
            if not isinstance(item, dict):
                continue
            models.append({
                "name": item.get("name", ""),
                "model": item.get("model", ""),
                "size": item.get("size", 0),
                "modified_at": item.get("modified_at", ""),
                "details": item.get("details", {}),
            })
        return {"live": True, "base_url": root, "models": models, "message": "connected"}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"live": False, "base_url": root, "models": [], "message": str(exc)}


@app.post("/api/analyze")
def analyze_video(
    filename: str = Form(...),
    api_base: str = Form(""),
    camera_id: str = Form("Camera 15"),
    zone_id: str = Form("construction_site"),
    zone_name: str = Form("건설현장 사고 구역"),
    scene_context: str = Form("건설현장 CCTV 사고 영상입니다. 영상에 보이는 행동, 구조물 변화, 사람의 위치 변화를 근거로 사고 원인을 판단합니다."),
    inference_provider: str = Form("colab"),
    model_key: str = Form("qwen3_vl_32b"),
    ollama_base_url: str = Form("http://127.0.0.1:11434"),
    ollama_model: str = Form("minicpm-v4.6:q4_K_M"),
    run_yolo: bool = Form(False),
    yolo_model: str = Form("yolo26n.pt"),
    fast_mode: bool = Form(True),
) -> dict[str, Any]:
    video_path = (VIDEO_DIR / filename).resolve()
    if not _is_inside(video_path, VIDEO_DIR) or not video_path.exists():
        raise HTTPException(status_code=404, detail="분석할 mp4 파일을 찾을 수 없습니다.")

    AGENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AGENT_OUTPUT_DIR / "accident_analysis_payload.json"
    raw_output_path = AGENT_OUTPUT_DIR / "accident_judgment_result.json"
    env = os.environ.copy()
    if inference_provider == "ollama":
        ollama_root = (ollama_base_url.strip() or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
        env["LLM_API_BASE"] = ollama_root + "/v1"
        env["LLM_MODEL"] = ollama_model.strip() or "minicpm-v4.6:q4_K_M"
        env["LLM_PROVIDER"] = "ollama_native"
        env["LLM_REQUEST_RETRIES"] = "1"
        env["LLM_RETRY_DELAY"] = "3"
    elif api_base.strip():
        env["LLM_API_BASE"] = api_base.strip()
    if scene_context.strip():
        env["AGENT_SCENE_CONTEXT"] = scene_context.strip()
    if inference_provider != "ollama" and model_key in MODEL_NAME_BY_KEY:
        env["LLM_MODEL"] = MODEL_NAME_BY_KEY[model_key]
    if fast_mode:
        env["LLM_REQUEST_RETRIES"] = env.get("LLM_REQUEST_RETRIES", "1")
        env["LLM_RETRY_DELAY"] = env.get("LLM_RETRY_DELAY", "5")

    command = [
        sys.executable,
        "-m",
        "agent.run_remote_judgment",
        "--video",
        str(video_path),
        "--request-timeout",
        "360" if fast_mode else "900",
        "--max-long-side",
        "320" if fast_mode else "448",
        "--max-tokens",
        "600" if inference_provider == "ollama" and fast_mode else "320" if fast_mode else "700",
        "--target-seconds",
        "0,6,12,18" if fast_mode else "0,4,8,12,14,15,16,17,18",
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
    if not fast_mode:
        command.extend([
            "--auto-moment",
            "--overview-interval",
            "4",
        ])
    annotated_path = AGENT_OUTPUT_DIR / f"yolo_annotated_{video_path.stem}.mp4"
    if run_yolo:
        YOLO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        command.extend([
            "--pt-model",
            str(YOLO_MODEL_DIR / yolo_model),
            "--run-pt",
            "--pt-annotated-output",
            str(annotated_path),
        ])

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=env,
            text=True,
            capture_output=True,
            timeout=ANALYSIS_TIMEOUT_SECONDS,
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
    pt_result = _read_json(DEFAULT_PT_OUTPUT)
    return {
        "video": _video_meta(video_path),
        "analysis": _summary_from_payload(payload, raw),
        "payload": payload,
        "raw_judgment": raw,
        "pt_result": pt_result,
        "annotated_video_url": f"/api/output/{annotated_path.name}" if run_yolo and annotated_path.exists() else "",
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


def _default_eval_summary() -> dict[str, Any]:
    scores = [
        {
            "name": "Qwen3-VL-32B",
            "prompt": "Cause Prompt + YOLO Evidence",
            "total": 0.86,
            "typeAccuracy": 0.89,
            "causeRecall": 0.82,
            "jsonValid": 0.97,
            "latency": 43.2,
        },
        {
            "name": "InternVL3-38B",
            "prompt": "Cause Prompt",
            "total": 0.81,
            "typeAccuracy": 0.86,
            "causeRecall": 0.76,
            "jsonValid": 0.94,
            "latency": 51.8,
        },
        {
            "name": "LLaVA-OneVision-2-8B",
            "prompt": "Cause Prompt",
            "total": 0.74,
            "typeAccuracy": 0.79,
            "causeRecall": 0.68,
            "jsonValid": 0.91,
            "latency": 24.5,
        },
        {
            "name": "MiniCPM-V 4.5",
            "prompt": "Fast Video Prompt",
            "total": 0.71,
            "typeAccuracy": 0.76,
            "causeRecall": 0.64,
            "jsonValid": 0.90,
            "latency": 18.7,
        },
    ]
    return {
        "updated_at": "sample",
        "dataset": "sample_accident_eval",
        "best_model": scores[0]["name"],
        "scores": scores,
        "charts": [
            "eval/output/model_score_bar.png",
            "eval/output/confusion_matrix.png",
            "eval/output/cause_recall_by_prompt.png",
            "eval/output/latency_boxplot.png",
        ],
    }


def _safe_stem(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in value).strip("_")
    return safe[:80] or "video"


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent.resolve())
    except ValueError:
        return False
    return True
