"""YOLO .pt input adapter for the Judgement Agent.

The trained PT model is not required for the current VL-only flow. This module
keeps the file input contract ready so the runtime can accept a model path now
and switch to real YOLO inference when the model arrives.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


AGENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = AGENT_DIR.parent
DEFAULT_MODELS_DIR = AGENT_DIR / "models"
DEFAULT_PT_OUTPUT = AGENT_DIR / "output" / "pt_detection_result.json"


@dataclass
class PtDetectionResult:
    status: str
    model_path: str | None = None
    video_path: str | None = None
    annotated_video_path: str | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    confidence: float | None = None
    message: str = ""


def prepare_pt_input(
    model_path: str | None,
    video_path: str | Path,
    output_path: str | Path = DEFAULT_PT_OUTPUT,
    run_inference: bool = False,
    annotated_output_path: str | Path | None = None,
) -> PtDetectionResult:
    """Validate or run a future YOLO .pt model against the input video."""

    if not model_path:
        result = PtDetectionResult(
            status="waiting",
            video_path=str(video_path),
            message="PT 모델 미입력. 현재는 Qwen2.5-VL 판단 결과만 사용합니다.",
        )
        _write_result(result, output_path)
        return result

    pt_path = Path(model_path)
    if not pt_path.exists() and not run_inference:
        result = PtDetectionResult(
            status="missing",
            model_path=str(pt_path),
            video_path=str(video_path),
            message=f"PT 모델 파일을 찾을 수 없습니다: {pt_path}",
        )
        _write_result(result, output_path)
        return result

    if pt_path.suffix.lower() != ".pt":
        result = PtDetectionResult(
            status="invalid",
            model_path=str(pt_path),
            video_path=str(video_path),
            message=f"PT 모델 파일 확장자는 .pt 여야 합니다: {pt_path.name}",
        )
        _write_result(result, output_path)
        return result

    if not run_inference:
        result = PtDetectionResult(
            status="ready",
            model_path=str(pt_path),
            video_path=str(video_path),
            message="PT 모델 입력 확인 완료. --run-pt 옵션을 주면 YOLO 추론을 실행합니다.",
        )
        _write_result(result, output_path)
        return result

    result = _run_yolo(pt_path, Path(video_path), Path(annotated_output_path) if annotated_output_path else None)
    _write_result(result, output_path)
    return result


def _run_yolo(model_path: Path, video_path: Path, annotated_output_path: Path | None = None) -> PtDetectionResult:
    try:
        from ultralytics import YOLO
    except ImportError:
        return PtDetectionResult(
            status="unavailable",
            model_path=str(model_path),
            video_path=str(video_path),
            message="ultralytics가 설치되어 있지 않아 PT 추론을 실행할 수 없습니다.",
        )

    try:
        import cv2
    except ImportError:
        cv2 = None

    model_source = str(model_path) if model_path.exists() else model_path.name
    model = YOLO(model_source)
    detections: list[dict[str, Any]] = []
    labels: list[str] = []
    max_confidence = 0.0
    writer = None
    fps = 30.0

    if annotated_output_path and cv2 is not None:
        annotated_output_path.parent.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        captured_fps = capture.get(cv2.CAP_PROP_FPS)
        if captured_fps and captured_fps > 1:
            fps = float(captured_fps)
        capture.release()

    for frame_index, result in enumerate(model.predict(source=str(video_path), stream=True, verbose=False)):
        names = result.names or {}
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = str(names.get(cls_id, cls_id))
                confidence = float(box.conf[0].item())
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                labels.append(label)
                max_confidence = max(max_confidence, confidence)
                detections.append({
                    "frame_index": frame_index,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": xyxy,
                })

        if annotated_output_path and cv2 is not None:
            frame = result.plot()
            height, width = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(
                    str(annotated_output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            writer.write(frame)

    if writer is not None:
        writer.release()

    return PtDetectionResult(
        status="done",
        model_path=str(model_path),
        video_path=str(video_path),
        annotated_video_path=str(annotated_output_path) if annotated_output_path and annotated_output_path.exists() else None,
        detections=detections,
        labels=sorted(set(labels)),
        confidence=round(max_confidence, 4) if detections else None,
        message="PT 추론 완료",
    )


def _write_result(result: PtDetectionResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
