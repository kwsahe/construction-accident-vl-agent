# Construction Accident VL Agent 포트폴리오

## 프로젝트 개요

Construction Accident VL Agent는 건설현장 CCTV 사고 영상을 분석해 사고 유형, 부상자 수, 사고 원인을 구조화하는 Vision-Language Agent 프로젝트입니다.

단순히 VL 모델을 호출하는 데서 끝내지 않고, 프론트엔드 영상 업로드부터 백엔드 저장, contact sheet 생성, Colab Qwen 서버 호출, JSON 검증, 분석 payload 변환까지 연결했습니다.

## 문제 정의

건설현장 사고 영상은 사람이 직접 전체 영상을 확인해야 사고 발생 구간과 경위를 판단할 수 있습니다. 특히 사고 전 행동, 구조물 이동, 작업자 위치 변화가 시간순으로 이어지는 경우 마지막 장면만 보고는 원인을 잘못 판단하기 쉽습니다.

이 프로젝트는 영상 전체를 모델에 그대로 넣지 않고, 핵심 프레임을 contact sheet로 압축해 VL 모델이 사고 전후 변화를 비교하도록 설계했습니다.

```text
mp4 영상
-> video 폴더 저장
-> 프레임 추출
-> 사고 전후 contact sheet 생성
-> Qwen2.5-VL 판단
-> 사고 유형 / 부상자 수 / 원인 JSON 생성
-> 분석 payload 변환
```

## 내 역할

- React + Vite + TypeScript 기반 프론트엔드 구성
- mp4 업로드 및 `video/` 폴더 저장 API 구현
- 저장된 영상 목록 조회 및 재분석 UI 구현
- FastAPI 분석 API 구현
- VL 프롬프트 설계
- contact sheet 생성 흐름 구현
- Colab/ngrok Qwen 서버 연동
- Qwen 판단 JSON 검증 및 fallback 처리
- 사고 분석 결과를 서비스 전달용 payload로 변환

## 핵심 구현

### 1. 영상 업로드와 분석 실행 분리

프론트엔드는 사용자가 선택한 mp4를 백엔드로 전송합니다. 백엔드는 파일을 루트 `video/` 폴더에 저장하고, 저장된 파일명을 기준으로 분석을 실행합니다.

이 구조 덕분에 같은 영상을 반복 분석하거나, 이미 저장된 영상을 목록에서 선택해 재분석할 수 있습니다.

### 2. Contact Sheet 기반 VL 입력

VL 모델에 긴 영상을 직접 전달하지 않고, OpenCV로 주요 프레임을 추출해 한 장의 contact sheet를 생성합니다.

이 방식은 입력 비용을 줄이면서도 다음 정보를 한 화면에서 비교할 수 있게 합니다.

- 사고 전 작업자 위치
- 사고 직전 구조물 변화
- 사고 순간 사람의 이동
- 사고 후 결과 장면

### 3. 사고 원인 중심 프롬프트

프롬프트는 사고 유형 분류보다 원인 판단을 더 중요하게 다룹니다.

모델에게 다음 기준을 명시했습니다.

- 마지막 장면만 보고 결론 내리지 말 것
- 사고 전 행동과 사고 순간 변화를 연결할 것
- 영상에서 보이지 않는 법적 책임, 과실, 교육 여부는 단정하지 말 것
- 원인이 불확실하면 가능한 후보와 근거를 분리할 것
- 부상자 수는 사고와 직접 관련된 사람만 보수적으로 산정할 것

### 4. JSON 검증과 payload 변환

Qwen 응답은 바로 저장하지 않습니다. `agent/save_judgment.py`에서 다음 과정을 거칩니다.

- `primary_type`, `confidence`, `details`, `timeline`, `evidence` 검증
- 사고 유형을 내부 라벨로 매핑
- 사고 경위와 원인을 요약
- `cctv_events`, `evidence_photos`, `tts_alert_logs` 형태의 payload 생성

## 프론트엔드 구성

첫 화면은 프로젝트 소개가 아니라 실제 분석 화면으로 이어지도록 구성했습니다.

- 영상 업로드 영역
- 저장된 `video/` 폴더 파일 목록
- Colab API Base 입력
- 현장 상황 설명 입력
- 분석 단계 표시
- 사고 유형, 부상자 수, 신뢰도, 원인 결과
- `accident_analysis_payload.json` 미리보기

## 백엔드 API

| API | 역할 |
| --- | --- |
| `GET /api/health` | 서버 상태 확인 |
| `GET /api/videos` | `video/` 폴더의 영상 목록 조회 |
| `POST /api/videos` | 업로드 영상 저장 |
| `GET /api/videos/{filename}` | 저장된 영상 파일 제공 |
| `POST /api/analyze` | VL Agent 분석 실행 |

## 모델 추천

Colab Pro 기준 1순위는 `Qwen/Qwen2.5-VL-32B-Instruct`입니다.

추천 이유:

- 사고 전후 장면 비교처럼 복합적인 시각 추론에 7B보다 안정적입니다.
- 원인 흐름, 부상자 수, evidence 작성에서 문맥 유지가 좋습니다.
- JSON 구조를 지키는 능력이 더 낫습니다.

VRAM이 부족하면 `Qwen/Qwen2.5-VL-7B-Instruct`를 fallback으로 사용합니다.

## 결과물

```text
agent/output/accident_overview_sheet.jpg
agent/output/accident_moment_detection.json
agent/output/accident_moment_sheet.jpg
agent/output/accident_judgment_result.json
agent/output/accident_analysis_payload.json
agent/output/pt_detection_result.json
```

## 강조 포인트

이 프로젝트의 핵심은 모델 호출 자체가 아니라, 현장 영상 업로드부터 사고 분석과 서비스 전달 payload까지 연결하는 파이프라인을 설계했다는 점입니다.

특히 원인 판단을 위해 사고 전후 프레임을 비교하고, 모델 응답을 검증 가능한 JSON과 payload로 정리한 점을 강조할 수 있습니다.

## 확장 방향

- YOLO/PT 모델과 tracking을 결합해 작업자 식별 안정성 향상
- pose estimation으로 추락/낙상 근거 강화
- 사고 구간 탐지 정확도 개선
- RAG 법령 Agent와 연결해 관련 규정, 재발 방지책, 보고서 초안 생성
