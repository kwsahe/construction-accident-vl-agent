from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from agent.model_profiles import MODEL_PROFILES, build_prompt, get_model_profile
    from agent.response_normalizer import parse_model_response
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agent.model_profiles import MODEL_PROFILES, build_prompt, get_model_profile
    from agent.response_normalizer import parse_model_response


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT_DIR / "eval" / "eval_cases.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "eval" / "output"


@dataclass
class EvalResult:
    case_id: str
    model_key: str
    model_name: str
    prompt_version: str
    accident_type_expected: str
    accident_type_predicted: str
    injured_count_expected: int
    injured_count_predicted: int
    cause_keyword_recall: float
    json_valid: bool
    latency: float
    total_score: float
    normalized_response: dict[str, Any]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate accident VL models and prompts.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--api-base", default="", help="OpenAI-compatible /v1 base URL. Empty means dry-run.")
    parser.add_argument("--model-key", action="append", choices=sorted(MODEL_PROFILES), help="Model profile key. Repeatable.")
    parser.add_argument("--prompt-version", action="append", default=["cause_focused"], help="Prompt version. Repeatable.")
    parser.add_argument("--dry-run", action="store_true", help="Generate deterministic sample predictions without calling a model.")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    model_keys = args.model_key or ["qwen3_vl_32b", "internvl3", "llava_onevision_2_8b", "minicpm_v_4_5"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[EvalResult] = []
    for model_key in model_keys:
        profile = get_model_profile(model_key)
        for prompt_version in args.prompt_version:
            for case in cases:
                started = time.perf_counter()
                if args.dry_run or not args.api_base:
                    response = _dry_prediction(case, profile.display_name)
                else:
                    response = call_model(args.api_base, profile.default_model, case, prompt_version)
                latency = time.perf_counter() - started
                normalized = parse_model_response(response)
                results.append(score_case(case, model_key, profile.display_name, prompt_version, normalized, latency))

    rows = [asdict(result) for result in results]
    (output_dir / "eval_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = summarize(rows)
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "eval_summary.csv", summary["scores"])
    print(f"saved: {output_dir / 'eval_summary.json'}")
    return 0


def call_model(api_base: str, model_name: str, case: dict[str, Any], prompt_version: str) -> str:
    prompt = build_prompt(prompt_version, scene_context=case.get("scene_context", ""))
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *_image_content(case),
            ],
        }],
        "max_tokens": 1200,
    }
    req = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def score_case(
    case: dict[str, Any],
    model_key: str,
    model_name: str,
    prompt_version: str,
    response: dict[str, Any],
    latency: float,
) -> EvalResult:
    expected = case["expected"]
    expected_type = expected["accident_type"]
    predicted_type = response.get("primary_type", "기타")
    expected_count = int(expected.get("injured_count", 0))
    predicted_count = int(response.get("injured_count", 0))
    cause_recall = keyword_recall(expected.get("cause_keywords", []), json.dumps(response, ensure_ascii=False))
    type_score = 1.0 if expected_type == predicted_type else 0.0
    count_score = 1.0 if expected_count == predicted_count else 0.0
    json_valid = required_fields_present(response)
    total = type_score * 0.30 + count_score * 0.15 + cause_recall * 0.35 + (1.0 if json_valid else 0.0) * 0.20
    return EvalResult(
        case_id=case["case_id"],
        model_key=model_key,
        model_name=model_name,
        prompt_version=prompt_version,
        accident_type_expected=expected_type,
        accident_type_predicted=predicted_type,
        injured_count_expected=expected_count,
        injured_count_predicted=predicted_count,
        cause_keyword_recall=round(cause_recall, 4),
        json_valid=json_valid,
        latency=round(latency, 4),
        total_score=round(total, 4),
        normalized_response=response,
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model_name"], row["prompt_version"]), []).append(row)

    scores = []
    for (model_name, prompt), items in grouped.items():
        total = average(item["total_score"] for item in items)
        type_acc = average(1.0 if item["accident_type_expected"] == item["accident_type_predicted"] else 0.0 for item in items)
        count_acc = average(1.0 if item["injured_count_expected"] == item["injured_count_predicted"] else 0.0 for item in items)
        cause_recall = average(item["cause_keyword_recall"] for item in items)
        json_valid = average(1.0 if item["json_valid"] else 0.0 for item in items)
        latency = average(item["latency"] for item in items)
        scores.append({
            "name": model_name,
            "prompt": prompt,
            "total": round(total, 4),
            "typeAccuracy": round(type_acc, 4),
            "injuredCountAccuracy": round(count_acc, 4),
            "causeRecall": round(cause_recall, 4),
            "jsonValid": round(json_valid, 4),
            "latency": round(latency, 4),
        })

    scores.sort(key=lambda item: item["total"], reverse=True)
    return {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": "accident_eval_cases",
        "best_model": scores[0]["name"] if scores else "",
        "scores": scores,
        "charts": [
            "eval/output/model_score_bar.png",
            "eval/output/confusion_matrix.png",
            "eval/output/cause_recall_by_prompt.png",
            "eval/output/latency_boxplot.png",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0])
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row.get(header, "")) for header in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def keyword_recall(keywords: list[str], text: str) -> float:
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword and keyword in text)
    return hits / len(keywords)


def required_fields_present(response: dict[str, Any]) -> bool:
    required = {"primary_type", "injured_count", "confidence", "cause", "timeline", "evidence", "details"}
    return required.issubset(response)


def average(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _image_content(case: dict[str, Any]) -> list[dict[str, Any]]:
    image_path = case.get("contact_sheet_path")
    if not image_path:
        return []
    path = (ROOT_DIR / image_path).resolve()
    if not path.exists():
        return []
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}]


def _dry_prediction(case: dict[str, Any], model_name: str) -> dict[str, Any]:
    expected = case["expected"]
    cause = " -> ".join(expected.get("cause_keywords", [])) or "원인 불확실"
    return {
        "primary_type": expected.get("accident_type", "기타"),
        "secondary_type": "기타",
        "injured_count": expected.get("injured_count", 0),
        "confidence": 0.82,
        "cause": cause,
        "cause_confidence": 0.78,
        "timeline": [{"time": f"{expected.get('accident_time_sec', 0)}s", "description": "dry-run 사고 변화 감지"}],
        "workers": [],
        "evidence": expected.get("cause_keywords", []),
        "details": f"{model_name} dry-run response for {case['case_id']}",
        "report_draft": {"overview": "dry-run report draft", "cause_analysis": cause},
        "prevention_actions": [{"priority": "medium", "action": "현장 원본 영상 재검토", "reason": "dry-run"}],
    }


if __name__ == "__main__":
    raise SystemExit(main())
