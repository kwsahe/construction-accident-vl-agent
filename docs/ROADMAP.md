# Construction Accident VL Agent Roadmap

## 1. 현재 프로젝트 방향성

이 프로젝트의 방향성은 **건설현장 사고 영상을 입력받아 사고 유형, 부상자 수, 사고 원인, 사고보고서 초안, 재발 방지 조치까지 생성하는 Vision-Language 사고 분석 Agent**입니다.

단순히 VL 모델을 호출하는 데서 끝내지 않고, 영상 입력부터 분석 결과 구조화와 프론트엔드 시각화까지 이어지는 end-to-end 파이프라인을 목표로 합니다.

```text
mp4 업로드 또는 YouTube URL 입력
-> video 폴더 저장
-> 프레임 추출
-> contact sheet 생성
-> YOLO pretrained 보조 evidence 생성
-> Qwen3-VL 사고 분석
-> 사고 유형 / 부상자 수 / 원인 판단
-> 사고보고서 초안 생성
-> 재발 방지 조치 추천
-> JSON payload 저장 및 프론트엔드 시각화
```

## 2. 핵심 설계 원칙

### 2.1 YOLO는 학습하지 않고 보조 evidence로 사용

사고 영상은 데이터 수집과 라벨링이 어렵고, 사고 순간이 다양하기 때문에 커스텀 YOLO 학습은 현재 프로젝트 범위를 벗어납니다.

따라서 YOLO는 사고 판단 모델이 아니라 **pretrained detector 기반 보조 관찰자**로 사용합니다.

역할:

- 사람 bbox 탐지
- 장비, 차량, 구조물 후보 탐지
- 프레임별 객체 수 변화 기록
- 사람 위치 변화 요약
- VL 프롬프트에 보조 evidence 제공

예시:

```json
{
  "frame_time": "16s",
  "detections": [
    {
      "label": "person",
      "bbox": [120, 80, 170, 210],
      "confidence": 0.82
    }
  ],
  "summary": "작업자 bbox가 이전 프레임보다 아래쪽으로 크게 이동했습니다."
}
```

YOLO의 출력은 최종 판단이 아니라 Qwen3-VL이 사고 원인을 추론할 때 참고하는 근거로 사용합니다.

### 2.2 Qwen3-VL 32B를 주 reasoning 모델로 사용

Qwen3-VL 32B는 사고 전후 장면 비교, 구조물 변화, 작업자 위치 변화, 원인-결과 흐름 생성에 사용합니다.

모델 구성:

```text
primary: Qwen3-VL-32B-Instruct
fallback: Qwen3-VL-8B 또는 Qwen2.5-VL-32B-Instruct
```

역할:

- 사고 발생 여부 판단
- 사고 유형 분류
- 부상자 수 추정
- 사고 원인 후보 생성
- 시각 근거 정리
- 사고보고서 초안 작성
- 재발 방지 조치 추천

### 2.3 법령 RAG는 제외

현 단계에서는 법령 검색과 법적 판단은 제외합니다. 대신 VL 모델이 영상에서 관찰 가능한 사실을 바탕으로 사고보고서 초안과 재발 방지 조치까지만 생성합니다.

제외하는 내용:

- 법령 조항 검색
- 법적 책임 판단
- 과실 단정
- 행정 처분 판단

포함하는 내용:

- 사고 개요
- 사고 경위
- 부상자 수 추정
- 관찰 가능한 원인
- 불확실한 점
- 재발 방지 조치

## 3. 입력 방식 확장

### 3.1 로컬 mp4 업로드

현재 구조를 유지합니다.

```text
프론트엔드 파일 선택
-> FastAPI 업로드
-> video/ 폴더 저장
-> 분석 실행
```

### 3.2 YouTube URL 입력

영상 파일을 직접 다운로드하기 어려운 경우를 위해 YouTube URL 입력을 지원합니다.

```text
YouTube URL 입력
-> backend /api/videos/youtube
-> yt-dlp로 mp4 다운로드
-> video/ 폴더 저장
-> 기존 분석 파이프라인 실행
```

주의 사항:

- 사용자가 분석 권한을 가진 영상만 입력하도록 안내합니다.
- 다운로드한 영상은 Git에 포함하지 않습니다.
- `video/` 폴더는 계속 `.gitignore` 대상입니다.
- 포트폴리오 설명에서는 연구/데모 목적의 입력 방식으로 명시합니다.

프론트엔드 입력 방식:

```text
[파일 업로드] [YouTube URL]
```

## 4. 사고보고서 초안 생성

VL 분석 결과를 바탕으로 사고보고서 초안을 생성합니다.

목표 JSON:

```json
{
  "report_draft": {
    "title": "건설현장 추락 사고 분석 보고서 초안",
    "overview": "영상에서 작업자가 높은 위치에서 아래로 이동하는 사고 정황이 관찰되었습니다.",
    "accident_type": "추락",
    "injured_count": 1,
    "timeline": [
      {
        "time": "14s",
        "event": "작업자가 높은 위치에서 작업 중인 것으로 보입니다."
      },
      {
        "time": "16s",
        "event": "작업자 위치가 급격히 아래쪽으로 이동합니다."
      },
      {
        "time": "18s",
        "event": "사고 이후 작업자가 바닥 근처에 위치한 것으로 보입니다."
      }
    ],
    "cause_analysis": {
      "primary_cause": "구조물 불안정 또는 작업 중 위치 변화로 인한 추락 가능성",
      "evidence": [
        "사고 전후 작업자 bbox 위치가 크게 변했습니다.",
        "contact sheet에서 작업자의 수직 이동이 관찰됩니다."
      ],
      "uncertain_points": [
        "구조물 조작 주체는 영상만으로 단정할 수 없습니다.",
        "안전장비 착용 여부는 프레임 해상도에 따라 불확실합니다."
      ]
    }
  }
}
```

## 5. 재발 방지 조치 추천

법령 기반 판단이 아니라, 영상에서 관찰된 사고 유형과 원인 후보를 바탕으로 일반적인 예방 조치를 생성합니다.

예시:

```json
{
  "prevention_actions": [
    {
      "priority": "high",
      "action": "고소작업 전 작업발판과 구조물 고정 상태를 점검합니다.",
      "reason": "영상에서 높은 위치 작업 중 추락 가능성이 관찰되었습니다."
    },
    {
      "priority": "high",
      "action": "작업 중 구조물 이동 또는 임의 조작을 제한합니다.",
      "reason": "사고 전후 구조물 또는 작업자 위치 변화가 사고 원인 후보로 보입니다."
    },
    {
      "priority": "medium",
      "action": "사고 위험 작업 구간에는 접근 통제와 신호 담당자를 배치합니다.",
      "reason": "사고 전후 작업자 간 위치 관계를 명확히 관리할 필요가 있습니다."
    }
  ]
}
```

## 6. 정확도 평가 계획

사고 영상 데이터셋을 구성해 모델과 프롬프트별 성능을 수치화합니다.

### 6.1 평가 대상

비교 대상:

- Qwen2.5-VL-32B 기본 프롬프트
- Qwen2.5-VL-32B 원인 강화 프롬프트
- Qwen3-VL-32B 기본 프롬프트
- Qwen3-VL-32B 원인 강화 프롬프트
- Qwen3-VL-32B + YOLO pretrained evidence

### 6.2 평가 데이터 구조

`eval/eval_cases.json` 형태로 관리합니다.

```json
[
  {
    "case_id": "fall_001",
    "video_path": "video/eval/fall_001.mp4",
    "youtube_url": "",
    "expected": {
      "accident_detected": true,
      "accident_type": "추락",
      "injured_count": 1,
      "cause_keywords": ["고소작업", "추락", "구조물 불안정"],
      "accident_time_sec": 16.0
    }
  }
]
```

### 6.3 평가 지표

모델과 프롬프트별로 다음 지표를 산출합니다.

| 지표 | 설명 |
| --- | --- |
| Accident Detection Accuracy | 사고 발생 여부를 맞춘 비율 |
| Accident Type Accuracy | 추락, 낙상, 화재, 기타 분류 정확도 |
| Injured Count Accuracy | 부상자 수가 정답과 일치한 비율 |
| Cause Keyword Recall | 정답 원인 키워드 중 모델 응답에 포함된 비율 |
| Timeline Error | 예측 사고 시점과 정답 시점의 평균 오차 |
| JSON Valid Rate | 응답이 schema를 통과한 비율 |
| Report Completeness | 보고서 초안 필수 섹션 생성 비율 |
| Average Latency | 영상 1개당 평균 분석 시간 |

### 6.4 점수 산식

종합 점수는 다음과 같이 계산합니다.

```text
total_score =
  accident_detection_accuracy * 0.20
  + accident_type_accuracy * 0.20
  + injured_count_accuracy * 0.15
  + cause_keyword_recall * 0.25
  + json_valid_rate * 0.10
  + report_completeness * 0.10
```

이 프로젝트에서는 사고 원인 분석이 핵심이므로 `cause_keyword_recall`의 비중을 가장 높게 둡니다.

## 7. 시각화 계획

Python 평가 스크립트에서 `pandas`, `matplotlib`, `seaborn`을 사용해 모델별 점수를 시각화합니다.

생성할 차트:

- 모델별 종합 점수 bar chart
- 사고 유형별 confusion matrix
- 프롬프트별 cause keyword recall 비교
- 모델별 JSON valid rate 비교
- 분석 latency box plot
- 사고 유형별 성능 radar chart

예시 산출물:

```text
eval/output/model_score_bar.png
eval/output/confusion_matrix.png
eval/output/cause_recall_by_prompt.png
eval/output/json_valid_rate.png
eval/output/latency_boxplot.png
eval/output/eval_summary.csv
eval/output/eval_summary.json
```

## 8. 프론트엔드 탑재 계획

평가 결과를 프론트엔드에 탑재해 포트폴리오에서 모델 개선 과정을 보여줍니다.

추가할 화면:

```text
Analyze
Reports
Evaluation
```

### 8.1 Evaluation Dashboard

프론트엔드에 다음 컴포넌트를 추가합니다.

- 모델 선택 dropdown
- 프롬프트 버전 선택 dropdown
- 종합 점수 카드
- 사고 유형 정확도 카드
- 원인 keyword recall 카드
- JSON valid rate 카드
- matplotlib로 생성한 평가 차트 이미지
- 평가 결과 테이블

표시 예시:

```text
Qwen3-VL-32B + Cause Prompt + YOLO Evidence

Total Score: 0.84
Accident Type Accuracy: 0.88
Cause Keyword Recall: 0.79
JSON Valid Rate: 0.96
Average Latency: 42.3s
```

### 8.2 분석 상세 페이지와 연결

개별 사고 분석 결과 페이지에는 다음 정보를 함께 표시합니다.

- contact sheet
- YOLO bbox evidence 요약
- VL 사고 원인 판단
- 사고보고서 초안
- 재발 방지 조치
- raw judgment JSON
- analysis payload JSON

## 9. 개발 우선순위

### Phase 1. Qwen3-VL 기반 분석 고도화

- Qwen3-VL-32B Colab 노트북 추가
- 모델명을 `.env`에서 쉽게 변경 가능하게 정리
- Qwen2.5-VL과 Qwen3-VL 결과를 같은 schema로 저장
- 원인 강화 프롬프트 버전 관리

### Phase 2. YouTube URL 입력

- `yt-dlp` 의존성 추가
- `/api/videos/youtube` API 추가
- URL 입력 UI 추가
- 다운로드 진행 상태 표시
- 다운로드 파일을 `video/` 폴더에 저장

### Phase 3. 사고보고서 초안과 재발 방지 조치

- `report_draft` schema 추가
- `prevention_actions` schema 추가
- 프롬프트에 보고서 생성 task 추가
- 프론트엔드에 보고서 탭 추가

### Phase 4. Pretrained YOLO Evidence

- 최신 Ultralytics pretrained 모델 다운로드
- 커스텀 학습 없이 inference만 수행
- 사람 bbox 변화 요약
- YOLO 결과를 Qwen3-VL 프롬프트에 보조 정보로 주입

### Phase 5. 평가 자동화

- `eval/eval_cases.json` 작성
- 모델/프롬프트별 batch evaluation 실행
- `eval/run_evaluation.py` 구현
- `matplotlib`, `seaborn` 기반 score chart 생성
- `eval_summary.csv`, `eval_summary.json` 저장

### Phase 6. Evaluation Dashboard

- 프론트엔드 Evaluation 메뉴 추가
- 평가 요약 API 추가
- chart 이미지와 score JSON 표시
- 모델/프롬프트별 비교 UI 추가

## 10. 최종 포트폴리오 메시지

최종적으로 이 프로젝트는 다음 메시지를 전달하는 것을 목표로 합니다.

> 건설현장 사고 영상을 업로드하거나 YouTube URL로 입력하면, pretrained YOLO evidence와 Qwen3-VL reasoning을 결합해 사고 유형, 부상자 수, 사고 원인, 사고보고서 초안, 재발 방지 조치를 생성하고, 모델/프롬프트별 정확도 평가 결과까지 시각화하는 Vision-Language 사고 분석 Agent.

