# Construction Accident VL Agent

건설현장 CCTV 사고 영상을 업로드하면, 주요 프레임을 contact sheet로 압축하고 Vision-Language 모델이 사고 유형, 부상자 수, 사고 원인을 구조화해 판단하는 포트폴리오 프로젝트입니다.

핵심은 단순 모델 호출이 아니라 `mp4 업로드 -> video 폴더 저장 -> 프레임 추출 -> Qwen VL 판단 -> JSON 검증 -> 분석 payload 변환`까지 이어지는 파이프라인입니다.

## 목표

- CCTV 영상에서 사고 발생 구간을 찾습니다.
- 사고 유형을 `추락`, `낙상`, `화재`, `기타`로 분류합니다.
- 영상에서 보이는 행동, 구조물 변화, 사람의 위치 변화를 근거로 사고 원인을 판단합니다.
- 사고와 직접 관련된 부상자 수를 보수적으로 추정합니다.
- Qwen2.5-VL 서버와 연동해 사고 판단 JSON을 생성합니다.
- 판단 결과를 서비스 DB에 전달 가능한 payload 형태로 변환합니다.

## 기술 스택

- Frontend: React + Vite + TypeScript
- Backend: FastAPI
- VL Server: Colab + Qwen2.5-VL
- Video Processing: OpenCV
- Optional Detector: YOLO `.pt` adapter

## 핵심 아이디어

원본 mp4 전체를 모델에 직접 넣지 않고, 로컬 Agent가 영상에서 의미 있는 프레임을 추출해 contact sheet 이미지를 만듭니다. VL 모델은 이 contact sheet를 보고 시간순 변화를 비교합니다.

```text
mp4 영상
-> video 폴더 저장
-> 프레임 추출
-> 사고 전후 contact sheet 생성
-> Qwen2.5-VL 판단
-> 사고 유형 / 부상자 수 / 원인 JSON 생성
-> 분석 payload 변환
```

## 주요 기능

### 1. 영상 업로드

프론트엔드에서 mp4, mov, avi, mkv 파일을 선택하면 FastAPI 백엔드가 루트 `video/` 폴더에 저장합니다. 이미 저장된 영상 목록도 UI에서 다시 선택할 수 있습니다.

### 2. 사고 순간 탐지

`--auto-moment` 옵션을 사용하면 먼저 overview contact sheet를 만들고, VL 모델이 사고가 실제로 시작되는 시간대를 찾습니다.

판단 기준:

- 사람이 높은 위치에서 아래로 급격히 이동하는지
- 구조물이 이동, 기울어짐, 전도되는지
- 같은 바닥면에서 미끄러지거나 넘어진 상황인지
- 사고 전후 프레임 사이에 명확한 위치 변화가 있는지

### 3. 사고 유형 및 원인 판단

최종 판단은 다음 정보를 JSON으로 생성합니다.

- `primary_type`: 낙상, 추락, 화재, 기타
- `injured_count`: 사고와 직접 관련된 부상자 수
- `cause`: 관찰 가능한 원인 흐름
- `timeline`: 시간대별 장면 변화
- `evidence`: 판단에 사용한 시각 근거
- `details`: 사고 경위 설명

사고 원인은 법적 책임이나 과실을 단정하지 않고, 영상에서 관찰 가능한 변화만 근거로 작성합니다.

### 4. 분석 payload 변환

Qwen 출력은 바로 DB row로 저장하지 않고, `agent/save_judgment.py`에서 서비스 전달용 payload로 변환합니다.

| Agent 결과 | Payload 필드 |
| --- | --- |
| 사고 판단 | `judgment.agent_verdict` |
| 사고 유형 | `judgment.accident_type` |
| 사고 경위 | `judgment.details` |
| 영상 경로 | `video_part_tables.cctv_events[].clip_path` |
| 사고 구간 | `clip_start_offset`, `clip_end_offset` |
| 증거 이미지 | `video_part_tables.evidence_photos[].photo_url` |

## 실행 방법

### 1. Conda 환경 생성

```powershell
conda create -n construction-vl python=3.11 -y
conda activate construction-vl
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 프론트엔드 설치

```powershell
npm install
```

### 3. Colab Qwen 서버 실행

`agent/Judgement_Agent_qwen25_vl_32b_colab_server.ipynb`를 Colab에서 실행합니다.

Colab Pro 환경에서는 `Qwen/Qwen2.5-VL-32B-Instruct`를 우선 추천합니다. VRAM이 부족하면 `Qwen/Qwen2.5-VL-7B-Instruct`로 낮춰 실행합니다.

서버가 실행되면 ngrok 주소를 확인합니다.

```text
https://xxxxx.ngrok-free.app/v1
```

### 4. 환경 변수 설정

`agent/.env.example`을 참고해 `agent/.env`를 만듭니다.

```env
LLM_PROVIDER=remote_openai
LLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct
LLM_API_BASE=https://xxxxx.ngrok-free.app/v1
LLM_API_KEY=dummy
```

### 5. 백엔드 실행

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### 6. 프론트엔드 실행

```powershell
npm run dev
```

브라우저에서 `http://127.0.0.1:5173`을 엽니다.

## 주요 산출물

실행 후 `agent/output/`에 다음 파일이 생성됩니다.

```text
agent/output/accident_overview_sheet.jpg
agent/output/accident_moment_detection.json
agent/output/accident_moment_sheet.jpg
agent/output/accident_judgment_result.json
agent/output/accident_analysis_payload.json
agent/output/pt_detection_result.json
```

## 구현 포인트

- 영상 업로드와 분석 실행을 프론트엔드에서 하나의 흐름으로 연결했습니다.
- mp4 원본을 `video/` 폴더에 저장하고, 저장된 파일을 기준으로 분석합니다.
- VL 판단은 사고 유형보다 원인 설명에 더 높은 우선순위를 둡니다.
- Qwen 응답은 JSON 검증과 fallback 처리를 거쳐 안정화합니다.
- 분석 결과를 서비스 DB에 맞는 payload로 변환해 후속 시스템과 연결할 수 있게 했습니다.

## 확장 방향

- YOLO/PT 모델과 tracking을 결합해 작업자 식별과 시간 안정성을 높입니다.
- pose estimation을 추가해 추락, 낙상, 충돌 상황의 근거를 강화합니다.
- 사고 판단 이후 RAG 법령 Agent와 연결해 관련 규정과 예방 조치를 추천합니다.
