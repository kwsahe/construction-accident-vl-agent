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

type ModelScore = {
  name: string;
  prompt: string;
  total: number;
  typeAccuracy: number;
  causeRecall: number;
  jsonValid: number;
  latency: number;
};

type EvaluationSummary = {
  updated_at?: string;
  dataset?: string;
  best_model?: string;
  scores: ModelScore[];
  charts?: string[];
};

type LlmStatus = {
  live: boolean;
  model: string;
  api_base: string;
  message: string;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000';
const stages = ['video 폴더 저장', '프레임 추출', 'VL 사고 판단', '분석 payload'];

const fallbackModelScores: ModelScore[] = [
  { name: 'Qwen3-VL-32B', prompt: 'Cause Prompt + YOLO Evidence', total: 0.86, typeAccuracy: 0.89, causeRecall: 0.82, jsonValid: 0.97, latency: 43.2 },
  { name: 'InternVL3-38B', prompt: 'Cause Prompt', total: 0.81, typeAccuracy: 0.86, causeRecall: 0.76, jsonValid: 0.94, latency: 51.8 },
  { name: 'LLaVA-OneVision-2-8B', prompt: 'Cause Prompt', total: 0.74, typeAccuracy: 0.79, causeRecall: 0.68, jsonValid: 0.91, latency: 24.5 },
  { name: 'MiniCPM-V 4.5', prompt: 'Fast Video Prompt', total: 0.71, typeAccuracy: 0.76, causeRecall: 0.64, jsonValid: 0.90, latency: 18.7 },
];

function formatBytes(bytes: number) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function formatElapsed(seconds: number) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [status, setStatus] = useState<AnalyzeState>('idle');
  const [activeStage, setActiveStage] = useState(-1);
  const [analysisStartedAt, setAnalysisStartedAt] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [uploadedVideo, setUploadedVideo] = useState<VideoMeta | null>(null);
  const [videos, setVideos] = useState<VideoMeta[]>([]);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState('');
  const [apiBase, setApiBase] = useState('');
  const [cameraId, setCameraId] = useState('Camera 15');
  const [evaluationSummary, setEvaluationSummary] = useState<EvaluationSummary>({ scores: fallbackModelScores });
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [selectedModel, setSelectedModel] = useState('qwen3_vl_32b');
  const [runYolo, setRunYolo] = useState(false);
  const [yoloModel, setYoloModel] = useState('yolo26n.pt');
  const [llmStatus, setLlmStatus] = useState<LlmStatus>({ live: false, model: '', api_base: '', message: 'not checked' });
  const [sceneContext, setSceneContext] = useState(
    '건설현장 CCTV 사고 영상입니다. 영상에 보이는 행동, 구조물 변화, 사람의 위치 변화를 근거로 사고 유형과 원인을 판단합니다.',
  );

  useEffect(() => {
    refreshVideos().catch(() => undefined);
    refreshEvaluationSummary().catch(() => undefined);
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      refreshLlmStatus().catch(() => undefined);
    }, 400);
    return () => window.clearTimeout(timer);
  }, [apiBase]);

  useEffect(() => {
    if (!file) {
      setPreviewUrl('');
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  useEffect(() => {
    if (!analysisStartedAt || (status !== 'running' && status !== 'uploading')) return;
    const updateElapsed = () => setElapsedSeconds(Math.floor((Date.now() - analysisStartedAt) / 1000));
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(timer);
  }, [analysisStartedAt, status]);

  const jsonPreview = useMemo(() => {
    if (result) return JSON.stringify(result.payload, null, 2);
    if (status === 'running' || status === 'uploading') {
      return JSON.stringify(
        {
          status,
          current_stage: stages[Math.max(activeStage, 0)],
          message: 'mp4 저장과 VL 사고 분석을 진행하고 있습니다.',
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

  const refreshEvaluationSummary = async () => {
    const response = await fetch(`${API_BASE}/api/evaluation/summary`);
    if (!response.ok) return;
    const data = await response.json() as EvaluationSummary;
    if (Array.isArray(data.scores) && data.scores.length) {
      setEvaluationSummary(data);
    }
  };

  const refreshLlmStatus = async () => {
    const query = apiBase.trim() ? `?api_base=${encodeURIComponent(apiBase.trim())}` : '';
    const response = await fetch(`${API_BASE}/api/llm/status${query}`);
    if (!response.ok) return;
    const data = await response.json() as LlmStatus;
    setLlmStatus(data);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setUploadedVideo(null);
    setResult(null);
    setError('');
    setActiveStage(-1);
    setAnalysisStartedAt(null);
    setElapsedSeconds(0);
    setStatus(selected ? 'ready' : 'idle');
  };

  const uploadVideo = async () => {
    if (!file) throw new Error('업로드할 mp4를 선택하세요.');
    setAnalysisStartedAt((current) => current ?? Date.now());
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

  const downloadYoutubeVideo = async () => {
    if (!youtubeUrl.trim()) throw new Error('YouTube URL을 입력하세요.');
    setAnalysisStartedAt((current) => current ?? Date.now());
    setStatus('uploading');
    setActiveStage(0);
    const formData = new FormData();
    formData.append('url', youtubeUrl.trim());
    const response = await fetch(`${API_BASE}/api/videos/youtube`, { method: 'POST', body: formData });
    if (!response.ok) throw new Error(await readError(response));
    const data = await response.json() as { video: VideoMeta };
    setUploadedVideo(data.video);
    setFile(null);
    setPreviewUrl(`${API_BASE}${data.video.url}`);
    await refreshVideos();
    return data.video;
  };

  const runAnalysis = async (event?: FormEvent) => {
    event?.preventDefault();
    if (status === 'running' || status === 'uploading') return;

    try {
      setError('');
      setResult(null);
      const startedAt = Date.now();
      setAnalysisStartedAt(startedAt);
      setElapsedSeconds(0);
      const video = uploadedVideo ?? (youtubeUrl.trim() && !file ? await downloadYoutubeVideo() : await uploadVideo());
      setStatus('running');
      setActiveStage(1);
      window.setTimeout(() => setActiveStage(2), 400);

      const formData = new FormData();
      formData.append('filename', video.filename);
      formData.append('api_base', apiBase);
      formData.append('camera_id', cameraId);
      formData.append('scene_context', sceneContext);
      formData.append('model_key', selectedModel);
      formData.append('run_yolo', String(runYolo));
      formData.append('yolo_model', yoloModel);

      const response = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(await readError(response));
      const data = await response.json() as AnalysisResponse;
      setActiveStage(stages.length);
      setResult(data);
      setStatus('done');
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
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
    setAnalysisStartedAt(null);
    setElapsedSeconds(0);
  };

  return (
    <>
      <header className="topbar">
        <nav className="shell nav" aria-label="주요 메뉴">
          <div className="brand"><span className="mark">VL</span> Construction Accident VL Agent</div>
          <div className="links">
            <a href="#workspace">Analyze</a>
            <a href="#dashboard">Dashboard</a>
            <a href="#reports">Reports</a>
            <a href="#models">Models</a>
            <a href="#schema">Schema</a>
          </div>
        </nav>
      </header>

      <main>
        <Hero />
        <AnalyzeWorkspace
          activeStage={activeStage}
          apiBase={apiBase}
          cameraId={cameraId}
          elapsedSeconds={elapsedSeconds}
          error={error}
          file={file}
          jsonPreview={jsonPreview}
          onApiBaseChange={setApiBase}
          onCameraIdChange={setCameraId}
          onFileChange={handleFileChange}
          onRunAnalysis={runAnalysis}
          onSceneContextChange={setSceneContext}
          onSelectVideo={selectExistingVideo}
          onSelectedModelChange={setSelectedModel}
          onRunYoloChange={setRunYolo}
          onYoloModelChange={setYoloModel}
          onYoutubeUrlChange={setYoutubeUrl}
          previewUrl={previewUrl}
          result={result}
          sceneContext={sceneContext}
          selectedModel={selectedModel}
          status={status}
          uploadedVideo={uploadedVideo}
          videos={videos}
          runYolo={runYolo}
          yoloModel={yoloModel}
          youtubeUrl={youtubeUrl}
          llmStatus={llmStatus}
          onRefreshLlmStatus={refreshLlmStatus}
        />
        <EvaluationDashboard summary={evaluationSummary} />
        <ReportDashboard result={result} />
        <ModelRecommendation />
        <Pipeline />
        <SchemaSection />
      </main>

      <footer className="footer">
        <div className="shell">
          Construction Accident VL Agent · React + FastAPI accident analysis app
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
        <h1>Accident Video to Report Draft</h1>
        <p className="lead">
          mp4 또는 YouTube 영상 입력에서 시작해 사고 유형, 부상자 수, 원인 흐름, 사고보고서 초안,
          재발 방지 조치, 모델별 평가 대시보드까지 연결하는 VL 사고 분석 프로젝트입니다.
        </p>
        <div className="actions">
          <a className="button" href="#workspace">영상 분석 시작</a>
          <a className="button secondary" href="#dashboard">평가 대시보드 보기</a>
        </div>
        <div className="stats" aria-label="프로젝트 핵심 지표">
          <div className="stat"><strong>VL</strong><span>Qwen3-VL 중심 사고 reasoning</span></div>
          <div className="stat"><strong>YOLO</strong><span>학습 없이 pretrained evidence 활용</span></div>
          <div className="stat"><strong>Eval</strong><span>모델·프롬프트별 정확도 시각화</span></div>
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
            <span className="scaffold" />
            <span className="person" />
          </div>
        ))}
      </div>
      <div className="monitor-body">
        <div className="verdict">
          <div>
            <strong>판단: 추락 / 구조물 불안정</strong>
            <span className="muted">사고 전후 프레임 변화로 원인 흐름을 추론</span>
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
  elapsedSeconds: number;
  error: string;
  file: File | null;
  jsonPreview: string;
  onApiBaseChange: (value: string) => void;
  onCameraIdChange: (value: string) => void;
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onRunAnalysis: (event?: FormEvent) => void;
  onSceneContextChange: (value: string) => void;
  onSelectVideo: (video: VideoMeta) => void;
  onSelectedModelChange: (value: string) => void;
  onRunYoloChange: (value: boolean) => void;
  onYoloModelChange: (value: string) => void;
  onYoutubeUrlChange: (value: string) => void;
  previewUrl: string;
  result: AnalysisResponse | null;
  sceneContext: string;
  selectedModel: string;
  status: AnalyzeState;
  uploadedVideo: VideoMeta | null;
  videos: VideoMeta[];
  runYolo: boolean;
  yoloModel: string;
  youtubeUrl: string;
  llmStatus: LlmStatus;
  onRefreshLlmStatus: () => void;
};

function AnalyzeWorkspace(props: WorkspaceProps) {
  const {
    activeStage, apiBase, cameraId, elapsedSeconds, error, file, jsonPreview, onApiBaseChange, onCameraIdChange,
    onFileChange, onRunAnalysis, onSceneContextChange, onSelectVideo, previewUrl,
    result, sceneContext, status, uploadedVideo, videos, selectedModel, onSelectedModelChange,
    runYolo, onRunYoloChange, yoloModel, onYoloModelChange, youtubeUrl, onYoutubeUrlChange,
    llmStatus, onRefreshLlmStatus,
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
            <h2>영상 입력부터 사고 원인 분석까지</h2>
            <p>사용자가 mp4를 넣으면 백엔드가 루트 <code>video/</code> 폴더에 저장하고, 저장된 영상을 기준으로 사고 유형과 원인 흐름을 분석합니다.</p>
          </div>
          <p className="muted">다음 단계에서는 YouTube URL 입력과 Qwen3-VL 모델 선택, pretrained YOLO evidence 옵션을 이 영역에 추가합니다.</p>
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
              <label><span>Colab API Base</span><input value={apiBase} onChange={(event) => onApiBaseChange(event.target.value)} placeholder="https://xxxxx.ngrok-free.app/v1" aria-label="Colab API Base" /></label>
              <label><span>VL 모델</span><select value={selectedModel} onChange={(event) => onSelectedModelChange(event.target.value)} aria-label="VL 모델 선택">
                <option value="qwen3_vl_32b">Qwen3-VL-32B</option>
                <option value="internvl3">InternVL3</option>
                <option value="llava_onevision_2_8b">LLaVA-OneVision-2-8B</option>
                <option value="minicpm_v_4_5">MiniCPM-V 4.5</option>
                <option value="qwen25_vl_32b">Qwen2.5-VL-32B</option>
              </select></label>
              <label><span>분석 질문</span><select aria-label="분석 질문"><option>사고 유형 + 부상자 수 + 원인</option><option>사고보고서 초안까지 생성</option></select></label>
              <label><span>YouTube URL</span><input value={youtubeUrl} onChange={(event) => onYoutubeUrlChange(event.target.value)} placeholder="https://www.youtube.com/watch?v=..." aria-label="YouTube URL" /></label>
              <label><span>YOLO 모델 파일</span><input value={yoloModel} onChange={(event) => onYoloModelChange(event.target.value)} aria-label="YOLO 모델 파일" /></label>
            </div>
            <div className="workspace-options">
              <label className="check-control">
                <input type="checkbox" checked={runYolo} onChange={(event) => onRunYoloChange(event.target.checked)} />
                <span>pretrained YOLO evidence 사용</span>
              </label>
              <button className="mini-button" type="button" onClick={onRefreshLlmStatus}>Colab 연결 확인</button>
              <span className={`live-badge ${llmStatus.live ? 'on' : 'off'}`}>
                {llmStatus.live ? 'LIVE' : 'OFF'} {llmStatus.model || selectedModel}
              </span>
            </div>
            <label className="textarea-control">
              <span>현장 상황 설명</span>
              <textarea value={sceneContext} onChange={(event) => onSceneContextChange(event.target.value)} rows={3} />
            </label>
            <button className="button analyze-button" type="submit" disabled={(!file && !uploadedVideo && !youtubeUrl.trim()) || status === 'running' || status === 'uploading'}>영상 저장 및 사고 분석 시작</button>
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
              <div className="status-stack">
                <span className={`pill ${status}`}>{status.toUpperCase()}</span>
                {(status === 'running' || status === 'uploading' || elapsedSeconds > 0) && (
                  <span className="elapsed-badge">elapsed {formatElapsed(elapsedSeconds)}</span>
                )}
              </div>
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
              <div className="code-title">accident_analysis_payload.json preview</div>
              <pre><code>{jsonPreview}</code></pre>
            </div>
          </section>
        </div>
      </div>
    </section>
  );
}

function EvaluationDashboard({ summary }: { summary: EvaluationSummary }) {
  const scores = summary.scores.length ? summary.scores : fallbackModelScores;
  const best = scores.find((item) => item.name === summary.best_model) ?? scores[0];
  const maxScore = Math.max(...scores.map((item) => item.total));
  const charts = summary.charts?.length ? summary.charts : [
    'eval/output/confusion_matrix.png',
    'eval/output/cause_recall_by_prompt.png',
    'eval/output/latency_boxplot.png',
    'eval/output/model_score_bar.png',
  ];

  return (
    <section id="dashboard" className="shell">
      <div className="section-title">
        <div>
          <h2>Evaluation Dashboard</h2>
          <p>사고 영상 데이터셋으로 모델과 프롬프트별 정확도를 비교하고, matplotlib 산출물을 프론트엔드에 탑재하는 영역입니다.</p>
        </div>
        <p className="muted">현재 데이터셋: <code>{summary.dataset ?? 'sample_accident_eval'}</code> · 갱신: <code>{summary.updated_at ?? 'sample'}</code></p>
      </div>

      <div className="score-strip">
        <MetricCard label="Best Model" value={best.name} detail={best.prompt} />
        <MetricCard label="Total Score" value={best.total.toFixed(2)} detail="weighted score" />
        <MetricCard label="Cause Recall" value={`${Math.round(best.causeRecall * 100)}%`} detail="원인 키워드 회수율" />
        <MetricCard label="JSON Valid" value={`${Math.round(best.jsonValid * 100)}%`} detail="schema pass rate" />
      </div>

      <div className="dashboard-grid">
        <div className="dashboard-card wide">
          <div className="card-title">모델·프롬프트별 종합 점수</div>
          <div className="bar-chart" aria-label="model score chart">
            {scores.map((item) => (
              <div className="bar-row" key={`${item.name}-${item.prompt}`}>
                <span>{item.name}</span>
                <div className="bar-track"><i style={{ width: `${(item.total / maxScore) * 100}%` }} /></div>
                <b>{item.total.toFixed(2)}</b>
              </div>
            ))}
          </div>
        </div>
        <div className="dashboard-card">
          <div className="card-title">평가 지표</div>
          <div className="mini-table">
            <span>Type Accuracy</span><strong>{Math.round(best.typeAccuracy * 100)}%</strong>
            <span>Cause Keyword Recall</span><strong>{Math.round(best.causeRecall * 100)}%</strong>
            <span>Average Latency</span><strong>{best.latency.toFixed(1)}s</strong>
          </div>
        </div>
        <div className="dashboard-card">
          <div className="card-title">추가할 matplotlib 산출물</div>
          <ul className="plain-list">
            {charts.map((chart) => <li key={chart}>{chart}</li>)}
          </ul>
        </div>
      </div>
    </section>
  );
}

function MetricCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function ReportDashboard({ result }: { result: AnalysisResponse | null }) {
  const report = {
    title: '건설현장 추락 사고 분석 보고서 초안',
    overview: result?.analysis.details || '사고 전후 프레임에서 작업자 위치 변화와 구조물 불안정 가능성을 확인하고, 사고 유형과 원인 후보를 정리합니다.',
    cause: result?.analysis.cause || '구조물 이동 또는 작업 위치 변화로 인한 추락 가능성',
    actions: [
      '고소작업 전 구조물 고정 상태를 점검합니다.',
      '작업 중 구조물 이동 또는 임의 조작을 제한합니다.',
      '사고 위험 작업 구간에 접근 통제와 신호 담당자를 배치합니다.',
    ],
  };

  return (
    <section id="reports" className="workspace-band">
      <div className="shell">
        <div className="section-title">
          <div>
            <h2>Report Draft Dashboard</h2>
            <p>VL 판단 결과를 사람이 바로 검토할 수 있는 사고보고서 초안과 재발 방지 조치로 변환합니다.</p>
          </div>
          <p className="muted">마지막 목표인 논문형 자동 작성은 이 보고서 데이터를 기반으로 abstract, method, experiment, result 형태로 확장합니다.</p>
        </div>
        <div className="report-layout">
          <article className="report-paper">
            <p className="eyebrow">Auto Draft</p>
            <h3>{report.title}</h3>
            <h4>1. 사고 개요</h4>
            <p>{report.overview}</p>
            <h4>2. 원인 분석</h4>
            <p>{report.cause}</p>
            <h4>3. 재발 방지 조치</h4>
            <ol>
              {report.actions.map((action) => <li key={action}>{action}</li>)}
            </ol>
          </article>
          <div className="dashboard-card">
            <div className="card-title">논문형 자동 작성 구조</div>
            <div className="paper-flow">
              <span>Abstract</span>
              <span>Method</span>
              <span>Dataset</span>
              <span>Evaluation</span>
              <span>Result</span>
              <span>Limitations</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function ModelRecommendation() {
  const models = [
    {
      name: 'Qwen3-VL-32B',
      tag: '1순위 reasoning 모델',
      role: '긴 영상 이해와 원인-결과 설명에 가장 잘 맞는 주 모델 후보입니다.',
      notebook: 'server/Qwen3_VL_32B_colab_server.ipynb',
      useCase: '최종 보고서 초안, 사고 원인 분석, 고정밀 평가 기준',
    },
    {
      name: 'InternVL3',
      tag: '강한 open-source MLLM',
      role: 'perception과 reasoning 성능 비교군으로 적합합니다.',
      notebook: 'server/InternVL3_colab_server.ipynb',
      useCase: 'Qwen3-VL과 perception/reasoning 비교',
    },
    {
      name: 'LLaVA-OneVision-2',
      tag: '8B급 video baseline',
      role: '속도와 성능의 균형을 비교하기 좋은 경량 video baseline입니다.',
      notebook: 'server/LLaVA_OneVision_2_8B_colab_server.ipynb',
      useCase: '저비용 baseline, latency 비교',
    },
    {
      name: 'MiniCPM-V 4.5',
      tag: '효율형 video 모델',
      role: '긴 영상 token 압축과 빠른 실험용 fallback 후보입니다.',
      notebook: 'server/MiniCPM_V_4_5_colab_server.ipynb',
      useCase: '빠른 반복 실험, fallback 비교',
    },
  ];

  return (
    <section id="models" className="shell">
      <div className="section-title">
        <div>
          <h2>Qwen3-VL 외 비교 모델 후보</h2>
          <p>최종 성능은 같은 사고 영상 데이터셋에서 모델과 프롬프트별로 수치화해 비교합니다.</p>
        </div>
        <p className="muted">VL 모델은 사고 원인 reasoning 담당, YOLO는 학습 없이 pretrained evidence 담당으로 역할을 나눕니다.</p>
      </div>
      <div className="grid-4">
        {models.map((model) => (
          <article className="feature model-card" key={model.name}>
            <span className="tag">{model.tag}</span>
            <h3>{model.name}</h3>
            <p className="muted">{model.role}</p>
            <dl className="model-meta">
              <dt>Notebook</dt>
              <dd>{model.notebook}</dd>
              <dt>Use case</dt>
              <dd>{model.useCase}</dd>
            </dl>
          </article>
        ))}
      </div>
    </section>
  );
}

function Pipeline() {
  const items = [
    ['mp4 / YouTube 입력', '파일 업로드와 URL 입력을 같은 video 저장 파이프라인으로 통합합니다.'],
    ['contact sheet', '주요 프레임을 시간 라벨과 함께 압축해 VL 모델 입력으로 사용합니다.'],
    ['YOLO evidence', 'pretrained 모델로 사람 bbox와 위치 변화 힌트를 생성합니다.'],
    ['VL reasoning', 'Qwen3-VL이 사고 유형, 부상자 수, 원인 흐름을 판단합니다.'],
    ['report & eval', '보고서 초안과 평가 대시보드로 결과를 설명 가능하게 만듭니다.'],
  ];

  return (
    <section id="pipeline" className="shell">
      <div className="section-title">
        <div>
          <h2>서비스형 사고 분석 파이프라인</h2>
          <p>모델 호출보다 중요한 부분은 영상 입력, 근거 생성, 판단, 보고서, 평가가 이어지는 구조입니다.</p>
        </div>
        <p className="muted">정확도는 사고 영상 테스트셋으로 모델·프롬프트별로 산출하고 프론트엔드에 표시합니다.</p>
      </div>
      <div className="pipeline">
        {items.map(([title, body], index) => (
          <article className="step" key={title}><b>{index + 1}</b><h3>{title}</h3><p className="muted">{body}</p></article>
        ))}
      </div>
    </section>
  );
}

function SchemaSection() {
  const rows = [
    ['analysis', '사고 유형, 부상자 수, confidence, cause를 요약'],
    ['report_draft', '사고 개요, 경위, 원인 분석, 재발 방지 조치 생성'],
    ['evaluation', '모델별 score, confusion matrix, cause recall, latency 저장'],
    ['paper_draft', '실험 결과를 논문형 abstract/method/result로 자동 작성'],
  ];

  return (
    <section id="schema" className="shell">
      <div className="section-title">
        <div>
          <h2>다음 schema는 분석, 보고서, 평가, 논문형 작성까지 확장합니다.</h2>
          <p>Qwen 출력은 그대로 저장하지 않고 프론트에서 검토 가능한 구조화 payload로 변환합니다.</p>
        </div>
      </div>
      <div className="grid-2">
        <div className="table" role="table" aria-label="확장 payload 매핑">
          {rows.map(([key, value]) => <div className="row" key={key}><strong>{key}</strong><span>{value}</span></div>)}
        </div>
        <div className="code-panel">
          <div className="code-title">확장 JSON 예시</div>
          <pre><code>{`{
  "analysis": { "accident_type": "추락", "injured_count": 1 },
  "report_draft": { "overview": "...", "prevention_actions": [] },
  "evaluation": { "model": "Qwen3-VL-32B", "total_score": 0.86 },
  "paper_draft": { "abstract": "...", "method": "...", "result": "..." }
}`}</code></pre>
        </div>
      </div>
    </section>
  );
}

export default App;
