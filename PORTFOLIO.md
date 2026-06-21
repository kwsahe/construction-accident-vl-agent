# SPilot VL Accident Judgment Agent

## 프로젝트 개요

SPilot VL Accident Judgment Agent는 건설현장 CCTV 영상을 프레임 단위로 분석해 사고 발생 여부, 사고 유형, 사고 경위, Red Zone 관련 근거를 판단하는 Vision-Language Agent 프로토타입입니다.

원본 프로젝트의 `backend/agent` 구현 중 VL 사고 판단 파이프라인을 포트폴리오 제출용으로 분리했습니다. 핵심 목표는 단순히 이미지를 분류하는 것이 아니라, CCTV 영상의 시간순 변화를 근거로 사고 경위를 JSON으로 구조화하고 SPilot 서비스의 ERD payload로 변환하는 것입니다.

- 원본 참고: https://github.com/Focus-Report/SPliot
- 프론트엔드 페이지: `index.html`
- 주요 코드: `agent/run_remote_judgment.py`, `agent/save_judgment.py`, `agent/pt_detector.py`, `agent/schemas.py`

## 문제 정의

건설현장 CCTV 사고 영상은 사람이 직접 전체 영상을 확인해야 사고 순간과 사고 유형을 판단할 수 있습니다. 특히 이동식 비계 사고처럼 Red Zone 진입, 구조물 이동, 상부 작업자 추락이 시간순으로 이어지는 경우에는 마지막 장면만 보고 단순 낙상으로 잘못 판단할 수 있습니다.

이 프로젝트는 영상 전체를 VL 모델에 그대로 입력하지 않고, 로컬 Agent가 사고 판단에 필요한 프레임을 추출해 contact sheet로 만들고, Qwen2.5-VL이 그 이미지에서 시간순 변화를 비교하도록 설계했습니다.

```text
mp4 영상
-> 프레임 추출
-> 사고 전후 contact sheet 생성
-> Qwen2.5-VL 판단
-> 사고 유형/details JSON 생성
-> SPilot DB payload 변환
```

## 내 역할

- CCTV 영상 기반 사고 판단 Agent 흐름 설계
- Qwen2.5-VL용 사고 순간 탐지 프롬프트 작성
- 사고 유형, Red Zone, 구조물 변화, 작업자 위치 변화 판단 프롬프트 작성
- mp4 프레임 추출 및 contact sheet 생성 흐름 구현
- Colab/ngrok Qwen2.5-VL 서버와 로컬 Agent 연동
- Qwen 판단 결과 JSON 검증 및 fallback 처리
- SPilot ERD의 영상 파트 테이블 payload 변환 구현
- YOLO `.pt` 모델 결합을 위한 입력 adapter 구현

## 핵심 구현

### 1. Contact Sheet 기반 VL 입력 경량화

영상 전체를 모델에 직접 전달하지 않고, 로컬 Agent가 대표 프레임을 추출해 시간 라벨이 포함된 contact sheet 이미지를 생성합니다.

이 방식은 모델 입력을 줄이면서도 사고 전후의 위치 변화, 구조물 변화, 작업자 움직임을 한 화면에서 비교할 수 있게 합니다.

### 2. 사고 순간 탐지와 최종 판단 분리

`--auto-moment` 옵션을 사용하면 먼저 overview sheet를 보고 사고 발생 시점 후보를 찾습니다.

이후 사고 전후 구간의 contact sheet를 다시 생성해 최종 판단을 수행합니다.

```text
overview sheet
-> accident_moment_detection.json
-> accident_moment_sheet.jpg
-> spilot_judgment_result.json
```

### 3. Prompt Engineering

프롬프트는 영상에 보이는 사실만 근거로 판단하도록 제한했습니다.

중요한 판단 기준:

- 높은 위치에서 아래로 떨어지면 `추락`
- 같은 바닥면에서 미끄러지거나 넘어지면 `낙상`
- 단순 Red Zone 진입만으로는 사고로 판단하지 않음
- 구조물 이동, 기울어짐, 전도 여부를 시간순으로 비교
- 교육 여부, 승인 여부, 사망 여부처럼 영상에서 직접 확인할 수 없는 내용은 단정하지 않음

출력 JSON 주요 구조:

```json
{
  "primary_type": "추락",
  "secondary_type": "전도",
  "confidence": 0.85,
  "structure_change": {
    "16s": "이동/기울어짐 또는 추락 발생"
  },
  "timeline": [
    {
      "time": "16초",
      "description": "상부 작업자의 추락 사고 순간이 감지되었습니다.",
      "structure_state": "이동/기울어짐 또는 추락 발생"
    }
  ],
  "red_zone_analysis": {
    "entry_detected": true,
    "zone_relation_to_accident": "직접 관련"
  },
  "details": "[사고 경위]\n시간순 사고 경위"
}
```

### 4. Red Zone 판단 분리

Red Zone 진입은 사고 자체가 아니라 경고 로그 또는 사고 경위의 근거로 사용했습니다.

예시 판단 흐름:

```text
Red Zone 진입
-> 비계 하부 접근 또는 조작 가능성
-> 비계 이동/기울어짐/전도 위험
-> 상부 작업자 추락
```

이 설계 덕분에 단순 위험구역 진입 이벤트와 실제 사고 이벤트를 분리할 수 있습니다.

### 5. ERD Payload 변환

Qwen 출력은 DB row로 바로 저장하지 않고, `save_judgment.py`에서 SPilot ERD 구조에 맞게 변환합니다.

| Agent 결과 | DB 테이블/컬럼 |
| --- | --- |
| 사고 판단 | `cctv_events.agent_verdict` |
| 사고 경위 | `cctv_events.agent_summary` |
| 사고 라벨 | `cctv_events.label` |
| 영상 경로 | `cctv_events.clip_path` |
| 사고 구간 시작 | `cctv_events.clip_start_offset` |
| 사고 구간 종료 | `cctv_events.clip_end_offset` |
| contact sheet | `evidence_photos.photo_url` |
| Red Zone 경고 | `tts_alert_logs.message` |

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `agent/run_remote_judgment.py` | mp4 프레임 추출, contact sheet 생성, Qwen2.5-VL 호출, 사고 판단 JSON 저장 |
| `agent/save_judgment.py` | Qwen 판단 JSON 검증 및 SPilot ERD payload 변환 |
| `agent/pt_detector.py` | YOLO `.pt` 모델 입력 검증 및 선택적 추론 결과 연결 |
| `agent/schemas.py` | Agent 출력 데이터 계약 정의 |
| `agent/test_json_payload.py` | raw judgment와 payload JSON 검증 |
| `agent/Judgement_Agent_qwen25_vl_7b_colab_server.ipynb` | Colab에서 Qwen2.5-VL 서버 실행 |

## 실행 예시

Colab에서 Qwen2.5-VL 서버를 실행한 뒤 ngrok 주소를 `.env`에 설정합니다.

```env
LLM_PROVIDER=remote_openai
LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
LLM_API_BASE=https://xxxxx.ngrok-free.app/v1
LLM_API_KEY=dummy
```

로컬 Agent 실행 예시:

```powershell
python -m agent.run_remote_judgment `
  --video C:\spilot\backend\agent\video\accident_video.mp4 `
  --auto-moment `
  --pt-model C:\spilot\backend\model\best.pt `
  --run-pt `
  --insert-db
```

생성 산출물:

```text
agent/output/accident_overview_sheet.jpg
agent/output/accident_moment_detection.json
agent/output/accident_moment_sheet.jpg
agent/output/spilot_judgment_result.json
agent/output/pt_detection_result.json
agent/output/judgement_agent_payload.json
```


## 프론트엔드 구성 추천

포트폴리오 제출용이라면 현재처럼 정적 HTML/CSS/JS만으로도 충분합니다. 다만 프로젝트처럼 보이게 하려면 첫 화면을 랜딩 페이지가 아니라 `영상 업로드 -> 분석 상태 -> 사고 판단 결과 -> JSON payload` 워크스페이스로 구성하는 편이 좋습니다.

실제 제품화까지 고려한다면 추천 스택은 `React + Vite + TypeScript`입니다.

- `Vite`: 빠른 dev server, 간단한 빌드, 정적 배포에 적합
- `React`: 업로드 상태, 분석 job 상태, 결과 JSON, 탭 UI처럼 상태가 많은 화면에 적합
- `TypeScript`: Agent payload, `cctv_events`, `evidence_photos`, `tts_alert_logs` 타입을 프론트에서 명확히 관리 가능
- `Tailwind CSS` 또는 CSS Modules: 운영 도구형 UI를 빠르게 만들기 좋음
- `TanStack Query`: 분석 요청, job polling, 결과 캐싱에 적합
- `React Router`: 대시보드, 분석 상세, 이력 페이지가 생길 때 사용

Next.js는 로그인, 권한, 서버 렌더링, 백오피스 페이지가 커질 때 좋습니다. 현재 VL 분석 화면처럼 브라우저에서 영상을 넣고 백엔드 API로 분석 job을 보내는 구조라면 Vite 기반 SPA가 더 가볍고 포트폴리오 설명도 명확합니다.

권장 화면 구조:

```text
/analyze
  영상 업로드
  분석 옵션 선택
  진행 상태 표시
  사고 유형/details/evidence 표시
  judgement_agent_payload.json 미리보기

/history
  이전 분석 목록
  사고 유형/신뢰도/시간 조회

/events/:id
  contact sheet, 원본 영상 구간, DB payload 상세
```

## React/Vite 실행 방법

현재 프론트엔드는 `React + Vite + TypeScript` 구조로 구성했습니다. 루트의 `index.html`은 Vite 엔트리이고, 실제 화면 로직은 `src/App.tsx`, 스타일은 `src/styles.css`에 있습니다. 기존 단일 HTML 버전은 `legacy-index.html`로 보존했습니다.

```powershell
npm install
npm run dev
npm run build
```

개발 서버 기본 주소:

```text
http://127.0.0.1:5173
```

## 프론트엔드/백엔드 실행

현재 구조는 React 프론트엔드와 FastAPI 백엔드로 분리되어 있습니다.

```powershell
conda create -n vl-agent python=3.11.15 -y
conda activate vl-agent
pip install -r requirements.txt

uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

다른 터미널에서 프론트엔드를 실행합니다.

```powershell
npm install
npm run dev
```

사용 흐름:

```text
React mp4 업로드
-> FastAPI가 video/ 폴더에 저장
-> 저장된 mp4를 agent.run_remote_judgment가 분석
-> Colab Qwen VL 서버 호출
-> 사고 유형, 부상자 수, 원인, payload 반환
```

Colab 서버 모델 추천:

- 1순위: `Qwen/Qwen2.5-VL-32B-Instruct`
- VRAM 부족 시: `Qwen/Qwen2.5-VL-7B-Instruct`

Colab 노트북: `agent/Judgement_Agent_qwen25_vl_7b_colab_server.ipynb`
## 기술 스택

- Python
- OpenCV
- Qwen2.5-VL-7B-Instruct
- Colab + ngrok
- JSON schema/dataclass 기반 payload
- Django ORM 또는 REST API ingest 연동
- YOLO `.pt` 모델 adapter
- HTML/CSS 정적 포트폴리오 페이지

## 성과

- 영상 기반 사고 판단 과정을 모델 호출, 프레임 처리, DB 저장까지 하나의 파이프라인으로 구성
- VL 모델이 마지막 장면만 보고 사고를 오판하지 않도록 시간순 프롬프트와 contact sheet 입력 방식 적용
- Red Zone 이벤트를 사고 자체가 아닌 증거 로그로 분리해 판단 정확도와 서비스 데이터 구조를 개선
- Qwen 출력과 서비스 DB 사이에 mapper를 두어 LLM 출력 변동성을 줄임
- 향후 YOLO/PT 모델, tracking, 법령 RAG Agent와 연결 가능한 구조로 확장성 확보

## 한계 및 개선 방향

- 현재 사고 구간 후보 추출은 테스트 영상 중심으로 보정된 부분이 있습니다.
- 실제 운영에서는 더 긴 시간 윈도우와 다중 카메라 상황을 고려해야 합니다.
- 작업자 A/B/C 식별은 tracking 모델과 결합해야 안정화할 수 있습니다.
- Red Zone 좌표는 현재 입력값 기반이며, 향후 화면 설정값과 자동 동기화할 수 있습니다.
- 사고 판단 이후 법령 분석은 별도의 RAG Agent와 연결하는 구조가 필요합니다.

