# SPilot VL Accident Judgment Agent

건설현장 CCTV 영상을 프레임 단위로 분석해 사고 발생 여부, 사고 유형, 사고 경위, Red Zone 관련 근거를 판단하는 Vision-Language Agent 프로토타입입니다.

이 폴더는 포트폴리오 제출용으로 기존 SPilot 프로젝트의 `backend/agent` 구현을 분리한 사본입니다. 실제 서비스 연동 코드는 원본 프로젝트에 남아 있으며, 이 폴더는 VL Agent 설계와 구현 흐름을 독립적으로 설명하기 위한 용도입니다.

## 목표

- CCTV 영상에서 사고 발생 순간 탐지
- 사고 유형 분류: 추락, 낙상, 화재, 기타
- Red Zone 진입 로그를 사고 판단의 증거로 활용
- 이동식 비계 작업 중 사고 경위 생성
- Qwen2.5-VL 서버와 연동해 사고 판단 JSON 생성
- SPilot ERD의 영상 파트 테이블 payload로 변환

## 핵심 아이디어

원본 mp4 전체를 VL 모델에 직접 넣는 방식이 아니라, 로컬 Agent가 영상에서 의미 있는 프레임을 추출해 contact sheet 이미지를 만들고, Qwen2.5-VL이 그 contact sheet를 보고 판단합니다.

```text
mp4 영상
-> 프레임 추출
-> 사고 전후 contact sheet 생성
-> Qwen2.5-VL 판단
-> 사고 유형/details JSON 생성
-> SPilot DB payload 변환
```

## 주요 기능

### 1. 사고 순간 탐지

`--auto-moment` 옵션을 사용하면 VL이 전체 영상의 대표 프레임을 먼저 보고 사고 발생 시점 후보를 찾습니다.

판단 기준:

- 사람이 높은 위치에서 아래로 급격히 이동하는지
- 비계 또는 구조물이 기울거나 전도되는지
- 단순 Red Zone 진입인지, 실제 추락/낙상 사고인지
- 사고 전후 프레임 사이에 명확한 위치 변화가 있는지

### 2. 사고 유형 판단

VL 판단 결과는 아래 유형으로 정리됩니다.

- `fall_from_height`: 추락
- `slip_and_fall`: 낙상
- `fire_explosion`: 화재
- `other`: 기타

이동식 비계, 작업발판, 사다리 등 높은 위치에서 바닥 방향으로 떨어지는 단서가 있으면 추락으로 분류합니다. 단순히 사람이 바닥에 누워 있다는 이유만으로 추락으로 단정하지 않도록 프롬프트를 구성했습니다.

### 3. Red Zone 증거 활용

Red Zone 진입은 사고 자체가 아니라 경고 로그 또는 사고 경위의 증거로 사용합니다.

예시 판단 흐름:

```text
Red Zone 진입
-> 비계 하부 접근 또는 조작 가능성
-> 비계 이동/기울어짐/전도 위험
-> 상부 작업자 추락
```

단, 승인 여부, 교육 여부, 사망 여부처럼 영상에서 직접 확인할 수 없는 내용은 단정하지 않도록 제한했습니다.

### 4. 사고 경위 details 생성

최종 JSON에는 `details`가 포함됩니다.

예시:

```text
[사고 경위]
Qwen2.5-VL 사고 순간 탐지 결과, 16초 전후 실제 사고 발생이 감지되었습니다.
현장 상황은 이동식 비계 작업 중이며 비계 임의 이동이 금지된 상태입니다.
RED ZONE 진입 경고 로그 이후 사람이 높은 작업 위치에서 바닥 방향으로 급격히 이동하는 장면이 확인되므로,
동일 평면 낙상이 아니라 이동식 비계 작업 중 상부 작업자의 추락 사고로 판단합니다.
```

### 5. DB payload 변환

Qwen 판단 결과는 바로 DB row가 아니라 SPilot ERD에 맞는 payload로 변환됩니다.

주요 매핑:

| Agent 결과 | DB 테이블/컬럼 |
| --- | --- |
| 사고 판단 | `cctv_events.agent_verdict` |
| 사고 경위 | `cctv_events.agent_summary` |
| 영상 경로 | `cctv_events.clip_path` |
| 사고 구간 시작 | `cctv_events.clip_start_offset` |
| 사고 구간 종료 | `cctv_events.clip_end_offset` |
| contact sheet | `evidence_photos.photo_url` |

## 파일 구성

```text
VL/
  README.md
  STRUCTURE.md
  agent/
    run_remote_judgment.py
    save_judgment.py
    pt_detector.py
    schemas.py
    env.py
    test_json_payload.py
    Judgement_Agent_qwen25_vl_7b_colab_server.ipynb
    .env.example
    ORIGINAL_AGENT_README.md
```

## 실행 흐름

### 1. Colab에서 Qwen2.5-VL 서버 실행

`agent/Judgement_Agent_qwen25_vl_7b_colab_server.ipynb`를 Colab에서 실행합니다.

서버가 실행되면 ngrok 주소가 출력됩니다.

```text
LLM_API_BASE=https://xxxxx.ngrok-free.app/v1
```

### 2. 환경변수 설정

`.env.example`을 참고해 `.env`를 구성합니다.

```env
LLM_PROVIDER=remote_openai
LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
LLM_API_BASE=https://xxxxx.ngrok-free.app/v1
LLM_API_KEY=dummy
```

### 3. 로컬 Agent 실행

원본 프로젝트 기준 실행 예시는 아래와 같습니다.

```powershell
cd C:\spilot\backend
uv run python -m agent.run_remote_judgment --video C:\spilot\backend\agent\video\accident_video.mp4 --auto-moment --insert-db
```

이 포트폴리오 폴더는 독립 실행용 패키징이 아니라 구현 사본이므로, 실제 실행은 원본 SPilot 프로젝트의 backend 환경에서 수행하는 것을 기준으로 합니다.

## 산출물

실행 시 원본 프로젝트에서는 아래 파일들이 자동 생성됩니다.

```text
backend/agent/output/accident_moment_sheet.jpg
backend/agent/output/spilot_judgment_result.json
backend/agent/output/judgement_agent_payload.json
backend/agent/output/pt_detection_result.json
```

이 파일들은 실행 결과이므로 Git에 포함하지 않습니다. 팀원이 pull해도 각자 실행하면 자동으로 다시 생성됩니다.

## 구현 포인트

- 영상 전체를 모델에 직접 넣지 않고 contact sheet 방식으로 VL 입력을 경량화
- Red Zone 이벤트와 사고 판단을 분리
- 사고 판단은 VL이 수행하고, Red Zone은 증거 로그로 사용
- Qwen 출력값을 그대로 쓰지 않고 ERD payload로 변환
- output 산출물은 Git 추적에서 제외해 다른 개발자의 결과와 충돌하지 않게 정리
- 실제 DB 저장 흐름과 포트폴리오 설명 흐름을 분리

## 한계 및 개선 방향

- 현재는 특정 테스트 영상에 맞춘 사고 구간 후보 추출 로직이 포함되어 있습니다.
- 실제 운영에서는 YOLO 추론 결과와 VL 프레임 판단을 더 긴 시간 윈도우로 안정화해야 합니다.
- 작업자 A/B/C 식별은 pt 모델 또는 tracking 모델과 결합해야 더 정확해집니다.
- Red Zone 다각형 좌표는 현재 입력값 기반이며, 향후 화면 설정값과 자동 동기화할 수 있습니다.
- 사고 판단 이후 법령 분석은 별도의 14B RAG Agent가 담당합니다.
