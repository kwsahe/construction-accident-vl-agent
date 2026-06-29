"""YOLO .pt input adapter for the Judgement Agent.

The trained PT model is not required for the current VL-only flow. This module
keeps the file input contract ready so the runtime can accept a model path now
and switch to real YOLO inference when the model arrives.
"""

from __future__ import annotations

import json
import shutil
import subprocess
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
    annotated_sheet_path: str | None = None
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
    annotated_sheet_output_path: str | Path | None = None,
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

    result = _run_yolo(
        pt_path,
        Path(video_path),
        Path(annotated_output_path) if annotated_output_path else None,
        Path(annotated_sheet_output_path) if annotated_sheet_output_path else None,
    )
    _write_result(result, output_path)
    return result


def build_filtered_evidence_sheet(
    video_path: str | Path,
    pt_result_path: str | Path = DEFAULT_PT_OUTPUT,
    output_path: str | Path = AGENT_DIR / "output" / "yolo_filtered_evidence_sheet.jpg",
    selected_labels: list[str] | None = None,
    sample_count: int = 6,
) -> PtDetectionResult:
    result_data = json.loads(Path(pt_result_path).read_text(encoding="utf-8"))
    detections = result_data.get("detections") or []
    selected = set(selected_labels or result_data.get("labels") or [])
    filtered = [item for item in detections if str(item.get("label")) in selected]

    try:
        import cv2
    except ImportError:
        rebuilt = PtDetectionResult(
            status="unavailable",
            model_path=result_data.get("model_path"),
            video_path=str(video_path),
            detections=filtered,
            labels=sorted(selected),
            confidence=result_data.get("confidence"),
            message="opencv가 설치되어 있지 않아 filtered evidence sheet를 생성할 수 없습니다.",
        )
        _write_result(rebuilt, pt_result_path)
        return rebuilt

    frame_indices = _representative_detection_frames(filtered, sample_count)
    if not frame_indices:
        frame_indices = _sample_frame_indices_from_video(video_path, sample_count)

    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    samples = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue
        frame_detections = [item for item in filtered if int(item.get("frame_index") or -1) == int(frame_index)]
        annotated = _draw_filtered_boxes(frame, frame_detections)
        samples.append((int(frame_index), annotated))
    capture.release()

    out = Path(output_path)
    if samples:
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_annotated_sheet(out, samples, fps)

    confidences = [float(item.get("confidence") or 0.0) for item in filtered]
    rebuilt = PtDetectionResult(
        status="done",
        model_path=result_data.get("model_path"),
        video_path=str(video_path),
        annotated_video_path=result_data.get("annotated_video_path"),
        annotated_sheet_path=str(out) if out.exists() else result_data.get("annotated_sheet_path"),
        detections=filtered,
        labels=sorted(selected),
        confidence=round(max(confidences), 4) if confidences else None,
        message=f"선택 라벨 evidence 생성 완료: {', '.join(sorted(selected)) or '없음'}",
    )
    _write_result(rebuilt, pt_result_path)
    return rebuilt


def _run_yolo(
    model_path: Path,
    video_path: Path,
    annotated_output_path: Path | None = None,
    annotated_sheet_output_path: Path | None = None,
) -> PtDetectionResult:
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
    frame_count = 0
    sheet_samples: list[tuple[int, Any]] = []
    sample_indices: set[int] = set()

    if cv2 is not None:
        capture = cv2.VideoCapture(str(video_path))
        captured_fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if captured_fps and captured_fps > 1:
            fps = float(captured_fps)
        capture.release()
        if frame_count > 0:
            sample_indices = _sample_frame_indices(frame_count, 6)
    if annotated_output_path and cv2 is not None:
        annotated_output_path.parent.mkdir(parents=True, exist_ok=True)
    if annotated_sheet_output_path:
        annotated_sheet_output_path.parent.mkdir(parents=True, exist_ok=True)

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
            if annotated_sheet_output_path and (frame_index in sample_indices or not sample_indices and len(sheet_samples) < 6):
                sheet_samples.append((frame_index, frame.copy()))
        elif annotated_sheet_output_path and cv2 is not None and (frame_index in sample_indices or not sample_indices and len(sheet_samples) < 6):
            sheet_samples.append((frame_index, result.plot()))

    if writer is not None:
        writer.release()
        _make_browser_playable_mp4(annotated_output_path)
    if annotated_sheet_output_path and sheet_samples:
        _write_annotated_sheet(annotated_sheet_output_path, sheet_samples, fps)

    return PtDetectionResult(
        status="done",
        model_path=str(model_path),
        video_path=str(video_path),
        annotated_video_path=str(annotated_output_path) if annotated_output_path and annotated_output_path.exists() else None,
        annotated_sheet_path=str(annotated_sheet_output_path) if annotated_sheet_output_path and annotated_sheet_output_path.exists() else None,
        detections=detections,
        labels=sorted(set(labels)),
        confidence=round(max_confidence, 4) if detections else None,
        message="PT 추론 완료",
    )


def _sample_frame_indices(frame_count: int, sample_count: int) -> set[int]:
    if frame_count <= sample_count:
        return set(range(frame_count))
    return {
        round(index * (frame_count - 1) / (sample_count - 1))
        for index in range(sample_count)
    }


def _sample_frame_indices_from_video(video_path: str | Path, sample_count: int) -> list[int]:
    try:
        import cv2
    except ImportError:
        return []
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return sorted(_sample_frame_indices(frame_count, sample_count)) if frame_count else []


def _representative_detection_frames(detections: list[dict[str, Any]], sample_count: int) -> list[int]:
    if not detections:
        return []
    by_frame: dict[int, float] = {}
    for item in detections:
        frame_index = int(item.get("frame_index") or 0)
        by_frame[frame_index] = max(by_frame.get(frame_index, 0.0), float(item.get("confidence") or 0.0))
    ordered = sorted(by_frame, key=lambda frame: (by_frame[frame], -frame), reverse=True)
    return sorted(ordered[:sample_count])


def _draw_filtered_boxes(frame: Any, detections: list[dict[str, Any]]) -> Any:
    import cv2

    output = frame.copy()
    for item in detections:
        xyxy = item.get("bbox_xyxy") or []
        if len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [int(float(value)) for value in xyxy]
        label = str(item.get("label") or "object")
        confidence = float(item.get("confidence") or 0.0)
        cv2.rectangle(output, (x1, y1), (x2, y2), (20, 180, 110), 2)
        text = f"{label} {confidence:.2f}"
        cv2.rectangle(output, (x1, max(0, y1 - 22)), (x1 + min(220, 8 * len(text) + 16), y1), (20, 180, 110), -1)
        cv2.putText(output, text, (x1 + 5, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _write_annotated_sheet(path: Path, samples: list[tuple[int, Any]], fps: float) -> None:
    from PIL import Image, ImageDraw

    cells: list[Image.Image] = []
    cell_width = 320
    label_height = 26
    for frame_index, frame in samples[:6]:
        rgb = frame[:, :, ::-1]
        image = Image.fromarray(rgb)
        image.thumbnail((cell_width, 220))
        canvas = Image.new("RGB", (cell_width, image.height + label_height), (24, 23, 20))
        canvas.paste(image, ((cell_width - image.width) // 2, label_height))
        draw = ImageDraw.Draw(canvas)
        second = frame_index / fps if fps > 0 else frame_index
        draw.text((8, 6), f"{second:.1f}s / frame {frame_index}", fill=(255, 227, 145))
        cells.append(canvas)

    if not cells:
        return
    columns = 3 if len(cells) > 2 else len(cells)
    rows = (len(cells) + columns - 1) // columns
    cell_height = max(cell.height for cell in cells)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (24, 23, 20))
    for index, cell in enumerate(cells):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(cell, (x, y))
    sheet.save(path, quality=92)


def _make_browser_playable_mp4(path: Path | None) -> None:
    if not path or not path.exists() or path.suffix.lower() != ".mp4":
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return
    temp_path = path.with_name(path.stem + "_h264.mp4")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(temp_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode == 0 and temp_path.exists():
        temp_path.replace(path)
    elif temp_path.exists():
        temp_path.unlink()


def _write_result(result: PtDetectionResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
