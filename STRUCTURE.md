# VL Folder Structure

이 문서는 포트폴리오용 `VL` 폴더에 포함된 파일과 제외한 파일을 설명합니다.

## 포함한 파일

```text
VL/
  README.md
  STRUCTURE.md
  agent/
    __init__.py
    env.py
    pt_detector.py
    run_remote_judgment.py
    save_judgment.py
    schemas.py
    test_json_payload.py
    .env.example
    Judgement_Agent_qwen25_vl_7b_colab_server.ipynb
    ORIGINAL_AGENT_README.md
```

## 파일별 역할

| 파일 | 역할 |
| --- | --- |
| `run_remote_judgment.py` | mp4 프레임 추출, contact sheet 생성, Qwen2.5-VL 호출, 사고 판단 결과 저장 |
| `save_judgment.py` | Qwen 판단 JSON을 SPilot ERD 영상 파트 payload로 변환 |
| `pt_detector.py` | YOLO `.pt` 모델 입력 여부 확인 및 추론 결과 JSON 생성 |
| `schemas.py` | Agent 출력 데이터 구조 정의 |
| `env.py` | Agent 전용 `.env` 로더 |
| `test_json_payload.py` | raw judgment와 schema payload 검증용 테스트 스크립트 |
| `.env.example` | Colab/ngrok LLM 서버 연결값 예시 |
| `Judgement_Agent_qwen25_vl_7b_colab_server.ipynb` | Colab에서 Qwen2.5-VL 서버를 띄우는 노트북 |
| `ORIGINAL_AGENT_README.md` | 원본 `backend/agent/README.md` 사본 |

## 제외한 파일

아래 파일/폴더는 포트폴리오용 사본에서 제외했습니다.

```text
backend/agent/.env
backend/agent/output/
backend/agent/video/
backend/agent/models/
backend/agent/__pycache__/
```

제외 이유:

- `.env`: 개인 ngrok 주소나 API 키가 들어갈 수 있음
- `output`: 실행 결과 JSON/JPG이므로 재현 가능 산출물
- `video`: 테스트 mp4 원본, 용량 및 저작권/공유 이슈 가능
- `models`: `.pt` 모델 파일은 Git 공유 대상 아님
- `__pycache__`: Python 캐시 파일

## 원본 프로젝트와의 관계

```text
원본 실행 위치:
backend/agent/

포트폴리오 사본:
VL/agent/
```

원본 프로젝트에서는 Django DB, 프론트 대시보드, 법령 RAG Agent와 연결됩니다.  
`VL/` 폴더는 해당 기능 중 Vision-Language 사고 판단 Agent 부분만 설명하기 위해 분리한 자료입니다.

## 데이터 흐름

```text
1. mp4 입력
2. 프레임 추출
3. 사고 후보 구간 contact sheet 생성
4. Colab Qwen2.5-VL 서버 호출
5. 사고 유형/details/red_zone_analysis JSON 생성
6. SPilot ERD payload 변환
7. cctv_events/evidence_photos/incidents/reports 저장 흐름으로 전달
```

## 포트폴리오에서 강조할 점

- 영상 기반 사고 판단 Agent를 직접 설계하고 구현
- 단순 이미지 분류가 아니라 시간 흐름 기반 프레임 비교 수행
- Red Zone을 사고 트리거가 아닌 증거 데이터로 분리
- Qwen2.5-VL과 로컬 Django/DB 구조를 연결
- 사고 판단 결과를 ERD 기반 payload로 변환
- 실행 산출물과 코드 자산을 분리해 협업 충돌을 줄임
