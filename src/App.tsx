import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from 'react';

type AnalyzeState = 'idle' | 'uploading' | 'ready' | 'running' | 'done' | 'error';

type VideoMeta = {
  filename: string;
  path: string;
  url: string;
  size: number;
  modified_at: string;
};

type AnalysisSummary = {
  accident_type?: string;
  accident_type_ko?: string;
  agent_verdict?: string;
  confidence?: number;
  injured_count?: number;
  cause?: string;
  details?: string;
  clip_start_offset?: number;
  clip_end_offset?: number;
};

type AnalysisResponse = {
  video: VideoMeta;
  analysis: AnalysisSummary;
  payload: Record<string, unknown>;
  raw_judgment: Record<string, unknown>;
  logs: string;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000';
const stages = ['video 폴더 저장', '프레임 추출', 'VL 사고 판단', 'ERD payload'];

function formatBytes(bytes: number) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [status, setStatus] = useState<AnalyzeState>('idle');
  const [activeStage, setActiveStage] = useState(-1);
  const [uploadedVideo, setUploadedVideo] = useState<VideoMeta | null>(null);
  const [videos, setVideos] = useState<VideoMeta[]>([]);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState('');
  const [apiBase, setApiBase] = useState('');
  const [cameraId, setCameraId] = useState('Camera 15');
  const [zoneName, setZoneName] = useState('비계 하부 RED ZONE');
  const [sceneContext, setSceneContext] = useState('이동식 비계 작업 중이며, 현재 고정 작업 중입니다. 비계 임의 이동은 금지된 상태입니다.');

  useEffect(() => {
    refreshVideos().catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl('');
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const jsonPreview = useMemo(() => {
    if (result) return JSON.stringify(result.payload, null, 2);
    if (status === 'running' || status === 'uploading') {
      return JSON.stringify(
        {
          status,
          current_stage: stages[Math.max(activeStage, 0)],
          message: 'mp4 저장 및 VL 사고 분석을 진행하고 있습니다.',
        },
        null,
        2,
      );
    }
    return JSON.stringify(
      {
        status: uploadedVideo ? 'ready' : 'waiting_for_video',
        video_folder: 'video/',
        message: uploadedVideo ? 'video 폴더에 저장된 mp4를 분석할 수 있습니다.' : 'mp4를 업로드하면 video 폴더에 저장됩니다.',
      },
      null,
      2,
    );
  }, [activeStage, result, status, uploadedVideo]);

  const refreshVideos = async () => {
    const response = await fetch(`${API_BASE}/api/videos`);
    if (!response.ok) return;
    const data = await response.json() as { videos: VideoMeta[] };
    setVideos(data.videos);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setUploadedVideo(null);
    setResult(null);
    setError('');
    setActiveStage(-1);
    setStatus(selected ? 'ready' : 'idle');
  };

  const uploadVideo = async () => {
    if (!file) throw new Error('업로드할 mp4를 선택하세요.');
    setStatus('uploading');
    setActiveStage(0);
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE}/api/videos`, { method: 'POST', body: formData });
    if (!response.ok) throw new Error(await readError(response));
    const data = await response.json() as { video: VideoMeta };
    setUploadedVideo(data.video);
    await refreshVideos();
    return data.video;
  };

  const runAnalysis = async (event?: FormEvent) => {
    event?.preventDefault();
    if (status === 'running' || status === 'uploading') return;

    try {
      setError('');
      setResult(null);
      const video = uploadedVideo ?? await uploadVideo();
      setStatus('running');
      setActiveStage(1);
      window.setTimeout(() => setActiveStage(2), 400);

      const formData = new FormData();
      formData.append('filename', video.filename);
      formData.append('api_base', apiBase);
      formData.append('camera_id', cameraId);
      formData.append('zone_name', zoneName);
      formData.append('scene_context', sceneContext);

      const response = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(await readError(response));
      const data = await response.json() as AnalysisResponse;
      setActiveStage(stages.length);
      setResult(data);
      setStatus('done');
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const selectExistingVideo = (video: VideoMeta) => {
    setFile(null);
    setPreviewUrl(`${API_BASE}${video.url}`);
    setUploadedVideo(video);
    setResult(null);
    setError('');
    setStatus('ready');
    setActiveStage(0);
  };

  return (
    <>
      <header className="topbar">
        <nav className="shell nav" aria-label="주요 메뉴">
          <div className="brand"><span className="mark">VL</span> SPilot Accident Judgment Agent</div>
          <div className="links">
            <a href="#workspace">Analyze</a>
            <a href="#pipeline">Pipeline</a>
            <a href="#prompt">Prompt</a>
            <a href="#schema">Schema</a>
            <a href="#portfolio">Portfolio</a>
          </div>
        </nav>
      </header>

      <main>
        <Hero />
        <AnalyzeWorkspace
          activeStage={activeStage}
          apiBase={apiBase}
          cameraId={cameraId}
          error={error}
          file={file}
          jsonPreview={jsonPreview}
          onApiBaseChange={setApiBase}
          onCameraIdChange={setCameraId}
          onFileChange={handleFileChange}
          onRunAnalysis={runAnalysis}
          onSceneContextChange={setSceneContext}
          onSelectVideo={selectExistingVideo}
          onZoneNameChange={setZoneName}
          previewUrl={previewUrl}
          result={result}
          sceneContext={sceneContext}
          status={status}
          uploadedVideo={uploadedVideo}
          videos={videos}
          zoneName={zoneName}
        />
        <Pipeline />
        <Features />
        <PromptSection />
        <SchemaSection />
        <PortfolioSection />
      </main>

      <footer className="footer">
        <div className="shell">
          Source reference: <a href="https://github.com/Focus-Report/SPliot">Focus-Report/SPliot</a> · React + FastAPI VL analysis app
        </div>
      </footer>
    </>
  );
}

async function readError(response: Response) {
  try {
    const data = await response.json();
    return typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail ?? data, null, 2);
  } catch {
    return response.statusText;
  }
}

function Hero() {
  return (
    <section className="shell hero">
      <div>
        <p className="eyebrow">Vision-Language Safety AI</p>
        <h1>SPilot VL Accident Judgment Agent</h1>
        <p className="lead">
          CCTV 사고 영상을 넣으면 프레임 추출, contact sheet 생성, Qwen2.5-VL 판단,
          사고 유형·부상자 수·원인 분석과 SPilot ERD payload 변환까지 이어지는 워크플로우입니다.
        </p>
        <div className="actions">
          <a className="button" href="#workspace">영상 분석 UI 보기</a>
          <a className="button secondary" href="/PORTFOLIO.md">PORTFOLIO.md</a>
        </div>
        <div className="stats" aria-label="프로젝트 핵심 수치">
          <div className="stat"><strong>mp4</strong><span>프론트 업로드 후 video 폴더 저장</span></div>
          <div className="stat"><strong>VL</strong><span>사고 유형·부상자·원인 판단</span></div>
          <div className="stat"><strong>ERD</strong><span>SPilot 영상 파트 payload 변환</span></div>
        </div>
      </div>
      <ContactSheetPreview />
    </section>
  );
}

function ContactSheetPreview() {
  return (
    <aside className="monitor" aria-label="Contact sheet 판단 예시">
      <div className="monitor-head">
        <span>accident_moment_sheet.jpg</span>
        <div className="lights" aria-hidden="true"><span /><span /><span /></div>
      </div>
      <div className="sheet">
        {['4s', '8s', '14s', '16s', '17s', '18s'].map((time, index) => (
          <div key={time} className={`frame ${index >= 2 && index < 4 ? 'tilt' : ''} ${index >= 4 ? 'fall' : ''}`} data-time={time}>
            <span className="red-zone" />
            <span className="scaffold" />
            <span className="person" />
          </div>
        ))}
      </div>
      <div className="monitor-body">
        <div className="verdict">
          <div>
            <strong>판단: 추락 / 전도</strong>
            <span className="muted">Red Zone 진입은 사고 원인이 아니라 판단 근거 로그로 분리</span>
          </div>
          <span className="badge">confidence 0.85</span>
        </div>
      </div>
    </aside>
  );
}

type WorkspaceProps = {
  activeStage: number;
  apiBase: string;
  cameraId: string;
  error: string;
  file: File | null;
  jsonPreview: string;
  onApiBaseChange: (value: string) => void;
  onCameraIdChange: (value: string) => void;
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onRunAnalysis: (event?: FormEvent) => void;
  onSceneContextChange: (value: string) => void;
  onSelectVideo: (video: VideoMeta) => void;
  onZoneNameChange: (value: string) => void;
  previewUrl: string;
  result: AnalysisResponse | null;
  sceneContext: string;
  status: AnalyzeState;
  uploadedVideo: VideoMeta | null;
  videos: VideoMeta[];
  zoneName: string;
};

function AnalyzeWorkspace(props: WorkspaceProps) {
  const {
    activeStage, apiBase, cameraId, error, file, jsonPreview, onApiBaseChange, onCameraIdChange,
    onFileChange, onRunAnalysis, onSceneContextChange, onSelectVideo, onZoneNameChange, previewUrl,
    result, sceneContext, status, uploadedVideo, videos, zoneName,
  } = props;

  const statusLabel = {
    idle: '영상을 기다리는 중',
    uploading: 'video 폴더에 mp4 저장 중',
    ready: '분석 준비 완료',
    running: `${stages[Math.max(activeStage, 0)]} 중`,
    done: '사고 분석 완료',
    error: '분석 오류',
  }[status];

  return (
    <section id="workspace" className="workspace-band">
      <div className="shell">
        <div className="section-title">
          <div>
            <h2>영상 입력에서 사고 분석까지 이어지는 화면 구조</h2>
            <p>사용자가 mp4를 넣으면 백엔드가 루트 <code>video/</code> 폴더에 저장하고, 저장된 영상을 기준으로 VL 사고 분석을 실행합니다.</p>
          </div>
          <p className="muted">Colab 서버의 <code>LLM_API_BASE</code>를 넣으면 Qwen VL 서버로 분석 요청이 전달됩니다. 비워두면 백엔드의 <code>agent/.env</code> 값을 사용합니다.</p>
        </div>

        <div className="workspace">
          <form className="upload-card" aria-label="영상 입력" onSubmit={onRunAnalysis}>
            <div className="panel-head">
              <div>
                <strong>1. 영상 업로드</strong>
                <span>프론트에서 선택한 mp4를 백엔드의 video 폴더에 저장</span>
              </div>
              <span className="pill">mp4 · mov · avi</span>
            </div>
            <label className={`video-drop ${file || uploadedVideo ? 'has-file' : ''}`} htmlFor="videoInput">
              <input id="videoInput" type="file" accept="video/*" onChange={onFileChange} />
              <span className="drop-icon">+</span>
              <strong>{file?.name ?? uploadedVideo?.filename ?? '분석할 사고 영상을 선택하세요'}</strong>
              <small>{file ? `${formatBytes(file.size)} · ${file.type || 'video file'}` : uploadedVideo ? `video/${uploadedVideo.filename}` : '파일을 넣으면 video 폴더 저장과 분석 버튼이 활성화됩니다.'}</small>
            </label>
            {previewUrl && <video className="video-preview" src={previewUrl} controls />}
            <div className="control-grid">
              <label><span>카메라</span><input value={cameraId} onChange={(event) => onCameraIdChange(event.target.value)} aria-label="카메라 ID" /></label>
              <label><span>구역</span><input value={zoneName} onChange={(event) => onZoneNameChange(event.target.value)} aria-label="구역 이름" /></label>
              <label><span>Colab API Base</span><input value={apiBase} onChange={(event) => onApiBaseChange(event.target.value)} placeholder="https://xxxxx.ngrok-free.app/v1" aria-label="Colab API Base" /></label>
              <label><span>분석 질문</span><select aria-label="분석 질문"><option>사고 유형 + 부상자 수 + 원인</option><option>사고 유형 + 원인만</option></select></label>
            </div>
            <label className="textarea-control">
              <span>현장 상황 설명</span>
              <textarea value={sceneContext} onChange={(event) => onSceneContextChange(event.target.value)} rows={3} />
            </label>
            <button className="button analyze-button" type="submit" disabled={(!file && !uploadedVideo) || status === 'running' || status === 'uploading'}>mp4 저장 및 사고 분석 시작</button>
            {videos.length > 0 && (
              <div className="video-list">
                <strong>video 폴더 mp4</strong>
                {videos.slice(0, 4).map((video) => (
                  <button key={video.filename} type="button" onClick={() => onSelectVideo(video)}>{video.filename} <span>{formatBytes(video.size)}</span></button>
                ))}
              </div>
            )}
          </form>

          <section className="analysis-panel" aria-label="사고 분석 결과">
            <div className="panel-head">
              <div>
                <strong>2. 사고 분석 Pipeline</strong>
                <span>{statusLabel}</span>
              </div>
              <span className={`pill ${status}`}>{status.toUpperCase()}</span>
            </div>
            <div className="stages" aria-label="분석 단계">
              {stages.map((stage, index) => (
                <div key={stage} className={`stage ${activeStage === index ? 'active' : ''} ${activeStage > index ? 'complete' : ''}`}>
                  <b>{String(index + 1).padStart(2, '0')}</b>
                  <span>{stage}</span>
                </div>
              ))}
            </div>
            <div className="result-grid">
              <div className="result-box"><span>사고 유형</span><strong>{result?.analysis.accident_type_ko ?? (status === 'running' ? '분석 중' : '대기')}</strong></div>
              <div className="result-box"><span>부상자 수</span><strong>{result ? `${result.analysis.injured_count ?? 0}명` : '-'}</strong></div>
              <div className="result-box"><span>신뢰도</span><strong>{typeof result?.analysis.confidence === 'number' ? result.analysis.confidence.toFixed(2) : '-'}</strong></div>
            </div>
            {result?.analysis.cause && <div className="cause-box"><span>판단 원인</span><p>{result.analysis.cause}</p></div>}
            {error && <div className="error-box">{error}</div>}
            <div className="code-panel result-json">
              <div className="code-title">judgement_agent_payload.json preview</div>
              <pre><code>{jsonPreview}</code></pre>
            </div>
          </section>
        </div>
      </div>
    </section>
  );
}

function Pipeline() {
  const items = [
    ['mp4 업로드', 'React 프론트에서 선택한 영상을 FastAPI 백엔드로 전송합니다.'],
    ['video 폴더 저장', '백엔드가 루트 video 폴더에 mp4 파일을 저장하고 목록으로 관리합니다.'],
    ['VL 판단', 'Colab Qwen VL 서버가 사고 유형, 부상자 수, 원인을 JSON으로 판단합니다.'],
    ['JSON 검증', 'primary_type, injured_count, cause, details, evidence를 schema 기준으로 확인합니다.'],
    ['DB payload', 'SPilot ERD의 cctv_events, evidence_photos, tts_alert_logs로 변환합니다.'],
  ];

  return (
    <section id="pipeline" className="shell">
      <div className="section-title">
        <div>
          <h2>영상 전체를 모델에 던지지 않고, 판단 가능한 장면으로 압축했습니다.</h2>
          <p>백엔드가 mp4에서 대표 프레임을 뽑고 contact sheet를 만든 뒤, VL 모델이 시간순 변화를 비교합니다.</p>
        </div>
        <p className="muted">목표는 단순 이미지 분류가 아니라 사고 유형, 부상자 수, 원인을 JSON으로 구조화하는 것입니다.</p>
      </div>
      <div className="pipeline">
        {items.map(([title, body], index) => (
          <article className="step" key={title}><b>{index + 1}</b><h3>{title}</h3><p className="muted">{body}</p></article>
        ))}
      </div>
    </section>
  );
}

function Features() {
  const features = [
    ['Video Upload', 'video 폴더 저장', '프론트엔드에서 업로드한 mp4를 백엔드가 루트 video 폴더에 저장하고 목록으로 관리합니다.'],
    ['Accident Analysis', '사고 유형·부상자·원인 판단', 'VL 모델이 추락, 낙상, 화재, 기타 사고 유형과 부상자 수, 원인 흐름을 JSON으로 생성합니다.'],
    ['Red Zone', '위험구역과 사고 판단 분리', 'Red Zone 진입은 사고 자체가 아니라 사고 경위의 증거 로그 또는 경고 이벤트로 사용합니다.'],
    ['Colab Server', 'Qwen2.5-VL 32B 추천', 'Colab Pro에서는 Qwen2.5-VL-32B-Instruct를 1순위로 쓰고, VRAM 부족 시 7B로 fallback합니다.'],
    ['Fallback', 'VL 응답 안정화', '응답이 JSON이 아니거나 반복 토큰으로 깨지는 경우를 감지해 재시도 또는 fallback 판단을 적용합니다.'],
    ['React + FastAPI', '프론트/백엔드 분리', '업로드 상태, 분석 상태, 결과 payload를 React와 FastAPI API로 연결했습니다.'],
  ];

  return (
    <section className="shell">
      <div className="section-title">
        <div>
          <h2>구현 포인트</h2>
          <p>사고 판단 모델을 서비스 DB와 바로 섞지 않고, 업로드, 관찰, 판단, 저장 payload를 분리했습니다.</p>
        </div>
      </div>
      <div className="grid-3">
        {features.map(([tag, title, body]) => (
          <article className="feature" key={title}>
            <span className="tag">{tag}</span>
            <h3>{title}</h3>
            <p className="muted">{body}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function PromptSection() {
  return (
    <section id="prompt" className="shell">
      <div className="section-title">
        <div>
          <h2>프롬프트는 시각 근거 기반 판단에 초점을 맞췄습니다.</h2>
          <p>이동식 비계 사고에서 낙상과 추락을 혼동하지 않도록, 구조물 상태와 작업자 위치 변화를 시간순으로 보게 했습니다.</p>
        </div>
      </div>
      <div className="grid-2">
        <div className="code-panel">
          <div className="code-title">VL 판단 스키마 일부</div>
          <pre><code>{`{
  "primary_type": "낙상|추락|화재|기타",
  "injured_count": 1,
  "cause": "하부 작업자의 비계 이동/조작 -> 비계 전도 -> 상부 작업자 추락",
  "confidence": 0.0,
  "timeline": [{ "time": "16s", "structure_state": "정상|이동|기울어짐|전도|불확실" }],
  "details": "[사고 경위] 시간순 원인-결과 흐름"
}`}</code></pre>
        </div>
        <div>
          <div className="quote">단순 RED ZONE 진입만으로 사고라고 쓰지 말고, 실제 전도/추락이 보일 때만 사고 발생으로 쓰세요.</div>
          <p className="muted spacious">이 제한 덕분에 위험구역 진입 경고와 실제 사고 판단이 분리됩니다.</p>
        </div>
      </div>
    </section>
  );
}

function SchemaSection() {
  const rows = [
    ['primary_type', '사고 유형을 worker_fall_from_height, worker_slip_and_fall, fire_or_smoke 라벨로 변환'],
    ['injured_count', 'raw_judgment와 workers 정보를 기준으로 부상자 수 요약'],
    ['cause', '사고 원인 흐름을 분석 요약 및 agent_summary에 반영'],
    ['contact sheet', 'snapshot_path, evidence_photos.photo_url에 증거 이미지로 연결'],
    ['red_zone_analysis', 'tts_alert_logs 메시지와 사고 경위 근거로 사용'],
  ];

  return (
    <section id="schema" className="shell">
      <div className="section-title">
        <div>
          <h2>Qwen 출력은 그대로 저장하지 않고 SPilot ERD payload로 변환합니다.</h2>
          <p>LLM은 판단 JSON을 만들고, 로컬 mapper가 서비스 테이블에 맞는 안정적인 row 형태로 바꿉니다.</p>
        </div>
      </div>
      <div className="grid-2">
        <div className="table" role="table" aria-label="Agent 결과와 DB 매핑">
          {rows.map(([key, value]) => <div className="row" key={key}><strong>{key}</strong><span>{value}</span></div>)}
        </div>
        <div className="code-panel">
          <div className="code-title">백엔드 실행 예시</div>
          <pre><code>{`uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
npm run dev`}</code></pre>
        </div>
      </div>
    </section>
  );
}

function PortfolioSection() {
  return (
    <section id="portfolio" className="shell">
      <div className="section-title">
        <div>
          <h2>포트폴리오에서 강조할 점</h2>
          <p>단순 모델 호출보다 현장 영상 업로드부터 사고 분석, 서비스 DB payload까지 연결하는 파이프라인을 설계했다는 점이 핵심입니다.</p>
        </div>
      </div>
      <div className="grid-3">
        <article className="artifact"><h3>문제 정의</h3><p className="muted">건설현장 사고 영상을 사람이 매번 확인하지 않아도, 사고 발생 구간과 경위를 구조화합니다.</p></article>
        <article className="artifact"><h3>내 역할</h3><p className="muted">업로드 API, VL 프롬프트 설계, contact sheet 생성 흐름, Qwen 서버 연동, JSON 검증, ERD payload 변환 구현.</p></article>
        <article className="artifact"><h3>확장 방향</h3><p className="muted">YOLO/PT 모델과 tracking을 결합해 작업자 식별과 시간 안정성을 높이고, 이후 RAG 법령 Agent와 연결합니다.</p></article>
      </div>
    </section>
  );
}

export default App;
