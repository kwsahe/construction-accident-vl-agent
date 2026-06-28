# VL Folder Structure

이 문서는 `Construction Accident VL Agent` 저장소의 주요 파일과 실행 산출물을 설명합니다.

## 포함 파일

```text
VL/
  README.md
  PORTFOLIO.md
  STRUCTURE.md
  index.html
  package.json
  requirements.txt
  src/
    App.tsx
    styles.css
  backend/
    __init__.py
    main.py
  agent/
    __init__.py
    env.py
    pt_detector.py
    run_remote_judgment.py
    save_judgment.py
    schemas.py
    test_json_payload.py
    .env.example
    Judgement_Agent_qwen25_vl_32b_colab_server.ipynb
```

## 파일별 역할

| 파일 | 역할 |
| --- | --- |
| `src/App.tsx` | React 기반 영상 업로드, 분석 실행, 결과 표시 UI |
| `src/styles.css` | 포트폴리오용 프론트엔드 스타일 |
| `backend/main.py` | FastAPI 영상 업로드, 목록 조회, 분석 실행 API |
| `agent/run_remote_judgment.py` | mp4 프레임 추출, contact sheet 생성, Qwen VL 호출, 결과 저장 |
| `agent/save_judgment.py` | Qwen 판단 JSON 검증 및 분석 payload 변환 |
| `agent/pt_detector.py` | YOLO `.pt` 모델 입력 검증 및 선택적 추론 결과 연결 |
| `agent/schemas.py` | Agent 출력 데이터 구조 정의 |
| `agent/env.py` | Agent 전용 `.env` 로더 |
| `agent/test_json_payload.py` | raw judgment와 payload JSON 검증용 테스트 스크립트 |
| `agent/.env.example` | Colab/ngrok LLM 서버 연결값 예시 |
| `agent/Judgement_Agent_qwen25_vl_32b_colab_server.ipynb` | Colab에서 Qwen2.5-VL 서버를 띄우는 노트북 |

## 실행 중 생성되는 파일

아래 파일과 폴더는 실행 결과물이므로 Git 추적 대상에서 제외합니다.

```text
video/
agent/.env
agent/output/
agent/models/
agent/__pycache__/
backend/__pycache__/
dist/
node_modules/
```

제외 이유:

- `video/`: 테스트 mp4 원본이 저장되며 용량이 큽니다.
- `agent/.env`: 개인 ngrok 주소와 API 설정이 들어갈 수 있습니다.
- `agent/output/`: 실행할 때마다 바뀌는 JSON/JPG 결과물입니다.
- `agent/models/`: `.pt` 모델 파일은 크기가 크고 별도 관리가 필요합니다.
- `node_modules/`, `dist/`: 프론트엔드 설치 및 빌드 산출물입니다.

## 데이터 흐름

```text
1. 프론트엔드에서 mp4 선택
2. FastAPI가 video/ 폴더에 저장
3. 저장된 영상에서 프레임 추출
4. 사고 후보 구간 contact sheet 생성
5. Colab Qwen2.5-VL 서버 호출
6. 사고 유형 / 부상자 수 / 원인 JSON 생성
7. 분석 payload 변환
8. 후속 서비스 또는 DB 저장 흐름으로 전달
```

## 구현에서 강조한 점

- 단순 이미지 분류가 아니라 영상 시간 흐름 기반 사고 판단을 수행합니다.
- 사고 유형 분류와 사고 원인 설명을 분리했습니다.
- 원인 판단은 영상에서 관찰 가능한 근거만 사용하도록 프롬프트를 제한했습니다.
- 업로드, 분석, payload 변환을 분리해 프로젝트 구조를 확장하기 쉽게 구성했습니다.
- YOLO/PT, tracking, RAG 법령 Agent를 추가할 수 있는 확장 지점을 남겼습니다.
