"""Model routing and prompt profiles for accident-video evaluation."""

from __future__ import annotations

from dataclasses import dataclass


ACCIDENT_JSON_SCHEMA = """
Return JSON only. Do not wrap the answer in markdown.

Required JSON schema:
{
  "primary_type": "낙상|추락|화재|기타",
  "secondary_type": "없음|전도|충돌|끼임|붕괴|기타",
  "injured_count": 0,
  "confidence": 0.0,
  "cause": "visible accident cause or cause candidate in time order",
  "cause_confidence": 0.0,
  "timeline": [
    {
      "time": "16s",
      "description": "visible change",
      "observed_change": "pre-accident action -> visual change -> accident result"
    }
  ],
  "workers": [
    {
      "role_guess": "피해자 후보|원인 관련자 후보|목격자 후보|불확실",
      "visible_frames": ["16s"],
      "position_change": "visible position change",
      "action_change": "visible action change",
      "accident_relation": "피해자 후보|원인 관련자 후보|목격자 후보|불확실",
      "basis": "visual evidence",
      "confidence": 0.0
    }
  ],
  "evidence": ["visual evidence"],
  "uncertain_points": ["what cannot be concluded from the video"],
  "details": "[사고 경위] time-ordered accident narrative",
  "report_draft": {
    "title": "accident report draft title",
    "overview": "accident overview",
    "cause_analysis": "visible cause analysis",
    "damage_summary": "injured count or uncertainty"
  },
  "prevention_actions": [
    {
      "priority": "high|medium|low",
      "action": "recommended prevention action",
      "reason": "reason based on visual evidence"
    }
  ]
}
""".strip()


BASE_SAFETY_RULES = """
You are a construction-site accident analysis vision-language model.
Use only visible evidence from the video/contact sheet.
Do not infer legal liability, permission, training status, death, or fault.
If the cause is uncertain, write "원인 불확실" and separate possible candidates in evidence or uncertain_points.
Count only people directly related to the accident as injured candidates.
Focus on cause flow: pre-accident action -> visual change -> accident result.
""".strip()


PROMPT_VARIANTS = {
    "standard": """
Analyze the contact sheet in time order.
Classify accident type, injured count, visual cause, evidence, and report draft.
""".strip(),
    "cause_focused": """
Analyze why the accident happened.
Do not stop at accident type classification.
Compare the frames before, during, and after the accident.
Explain the most likely visible cause flow and list uncertainty separately.
""".strip(),
    "yolo_evidence": """
Use the provided YOLO evidence only as auxiliary observations.
Do not treat YOLO labels as final truth.
Combine bbox/person movement hints with visual reasoning from the contact sheet.
""".strip(),
}


@dataclass(frozen=True)
class ModelProfile:
    key: str
    display_name: str
    default_model: str
    notebook: str
    default_prompt_version: str
    role: str
    json_repair_hint: str


MODEL_PROFILES: dict[str, ModelProfile] = {
    "qwen3_vl_32b": ModelProfile(
        key="qwen3_vl_32b",
        display_name="Qwen3-VL-32B",
        default_model="Qwen/Qwen3-VL-32B-Instruct",
        notebook="server/Qwen3_VL_32B_colab_server.ipynb",
        default_prompt_version="cause_focused",
        role="main reasoning model",
        json_repair_hint="Qwen usually follows JSON well; strip markdown fences and parse the first JSON object.",
    ),
    "internvl3": ModelProfile(
        key="internvl3",
        display_name="InternVL3",
        default_model="OpenGVLab/InternVL3-38B",
        notebook="server/InternVL3_colab_server.ipynb",
        default_prompt_version="cause_focused",
        role="perception/reasoning comparison model",
        json_repair_hint="InternVL may add natural-language prefaces; extract the first JSON object.",
    ),
    "llava_onevision_2_8b": ModelProfile(
        key="llava_onevision_2_8b",
        display_name="LLaVA-OneVision-2-8B",
        default_model="lmms-lab/LLaVA-OneVision-2-8B-ov",
        notebook="server/LLaVA_OneVision_2_8B_colab_server.ipynb",
        default_prompt_version="standard",
        role="lightweight video baseline",
        json_repair_hint="LLaVA may omit fields; fill missing optional fields after JSON extraction.",
    ),
    "minicpm_v_4_5": ModelProfile(
        key="minicpm_v_4_5",
        display_name="MiniCPM-V 4.5",
        default_model="openbmb/MiniCPM-V-4_5",
        notebook="server/MiniCPM_V_4_5_colab_server.ipynb",
        default_prompt_version="standard",
        role="efficient fallback model",
        json_repair_hint="MiniCPM may be concise; normalize missing arrays to empty lists.",
    ),
}


def build_prompt(prompt_version: str, scene_context: str = "", yolo_evidence: str = "") -> str:
    variant = PROMPT_VARIANTS.get(prompt_version, PROMPT_VARIANTS["standard"])
    parts = [BASE_SAFETY_RULES, variant, ACCIDENT_JSON_SCHEMA]
    if scene_context.strip():
        parts.append("[Scene context]\n" + scene_context.strip())
    if yolo_evidence.strip():
        parts.append("[YOLO auxiliary evidence]\n" + yolo_evidence.strip())
    return "\n\n".join(parts)


def get_model_profile(key: str) -> ModelProfile:
    if key not in MODEL_PROFILES:
        raise KeyError(f"Unknown model profile: {key}")
    return MODEL_PROFILES[key]
