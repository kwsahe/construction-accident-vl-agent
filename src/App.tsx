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
  pt_result?: YoloResult;
  annotated_video_url?: string;
  annotated_sheet_url?: string;
  run_id?: number;
  eval_score?: Record<string, number | string> | null;
  logs: string;
};

type ModelScore = {
  name: string;
  prompt: string;
  total: number;
  typeAccuracy: number;
  causeRecall: number;
  semanticScore?: number;
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

type EvaluationCase = {
  video_id: string;
  accident_detected: boolean;
  accident_type: string;
  injured_count: number;
  cause_keywords: string[];
  accident_time_range: [number, number];
  required_evidence: string[];
};

type LlmStatus = {
  live: boolean;
  model: string;
  api_base: string;
  message: string;
};

type YoloResult = {
  status: string;
  model_path?: string;
  video_path?: string;
  annotated_video_path?: string;
  detections: Array<Record<string, unknown>>;
  labels: string[];
  confidence?: number;
  message: string;
};

type YoloResponse = {
  video: VideoMeta;
  result: YoloResult;
  annotated_video_url: string;
  annotated_sheet_url?: string;
  result_url: string;
};

type OllamaModel = {
  name: string;
  model?: string;
  size?: number;
  modified_at?: string;
};

type OllamaStatus = {
  live: boolean;
  base_url: string;
  models: OllamaModel[];
  message: string;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000';
const stages = ['video 폴더 저장', '프레임 추출', 'VL 사고 판단', '분석 payload'];

const fallbackModelScores: ModelScore[] = [
  { name: 'Qwen3-VL-32B', prompt: 'Cause Prompt + YOLO Evidence', total: 0.86, typeAccuracy: 0.89, causeRecall: 0.82, semanticScore: 0.84, jsonValid: 0.97, latency: 43.2 },
  { name: 'InternVL3-38B', prompt: 'Cause Prompt', total: 0.81, typeAccuracy: 0.86, causeRecall: 0.76, semanticScore: 0.79, jsonValid: 0.94, latency: 51.8 },
  { name: 'LLaVA-OneVision-2-8B', prompt: 'Cause Prompt', total: 0.74, typeAccuracy: 0.79, causeRecall: 0.68, semanticScore: 0.70, jsonValid: 0.91, latency: 24.5 },
  { name: 'MiniCPM-V 4.5', prompt: 'Fast Video Prompt', total: 0.71, typeAccuracy: 0.76, causeRecall: 0.64, semanticScore: 0.67, jsonValid: 0.90, latency: 18.7 },
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

function splitCsv(value: string) {
  return value.split(',').map((item) => item.trim()).filter(Boolean);
}

function buildReportDraft(result: AnalysisResponse | null) {
  const type = result?.analysis.accident_type_ko || result?.analysis.accident_type || '사고 유형 분석 대기';
  const injured = typeof result?.analysis.injured_count === 'number' ? `${result.analysis.injured_count}명` : '확인 필요';
  const confidence = typeof result?.analysis.confidence === 'number' ? result.analysis.confidence.toFixed(2) : '-';
  const cause = result?.analysis.cause || '분석 완료 후 사고 원인 후보가 표시됩니다.';
  const detail = result?.analysis.details || '영상 프레임 변화, 작업자 위치 변화, 구조물 상태 변화를 근거로 사고 개요를 정리합니다.';
  const clip =
    typeof result?.analysis.clip_start_offset === 'number' && typeof result?.analysis.clip_end_offset === 'number'
      ? `${result.analysis.clip_start_offset}s ~ ${result.analysis.clip_end_offset}s`
      : '분석 구간 확인 필요';

  return {
    title: `건설현장 ${type} 사고 분석 보고서 초안`,
    overview: `분석 구간 ${clip}에서 ${type} 사고가 의심됩니다. 부상자 수는 ${injured}, 신뢰도는 ${confidence}입니다.`,
    simple: cause,
    details: detail,
    actions: [
      '사고 구간의 작업자 위치 변화와 구조물 상태 변화를 원본 영상으로 재확인합니다.',
      '고소작업 또는 이동식 구조물 사용 시 고정 상태와 임의 이동 여부를 점검합니다.',
      '동일 작업 전 접근 통제, 작업자 간 신호 체계, 추락 방지 조치를 재점검합니다.',
    ],
  };
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
  const [inferenceProvider, setInferenceProvider] = useState<'colab' | 'ollama'>('colab');
  const [selectedModel, setSelectedModel] = useState('qwen3_vl_32b');
  const [selectedOllamaModel, setSelectedOllamaModel] = useState('minicpm-v4.6:q4_K_M');
  const [ollamaBaseUrl, setOllamaBaseUrl] = useState('http://127.0.0.1:11434');
  const [runYolo, setRunYolo] = useState(false);
  const [fastMode, setFastMode] = useState(true);
  const [yoloModel, setYoloModel] = useState('yolo26n.pt');
  const [yoloStatus, setYoloStatus] = useState<AnalyzeState>('idle');
  const [yoloResult, setYoloResult] = useState<YoloResponse | null>(null);
  const [yoloError, setYoloError] = useState('');
  const [selectedYoloLabels, setSelectedYoloLabels] = useState<string[]>(['person']);
  const [llmStatus, setLlmStatus] = useState<LlmStatus>({ live: false, model: '', api_base: '', message: 'not checked' });
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatus>({ live: false, base_url: 'http://127.0.0.1:11434', models: [], message: 'not checked' });
  const [sceneContext, setSceneContext] = useState(
    '건설현장 CCTV 사고 영상입니다. 영상에 보이는 행동, 구조물 변화, 사람의 위치 변화를 근거로 사고 유형과 원인을 판단합니다.',
  );

  useEffect(() => {
    refreshVideos().catch(() => undefined);
    refreshEvaluationSummary().catch(() => undefined);
    refreshOllamaStatus().catch(() => undefined);
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

  const refreshOllamaStatus = async () => {
    const response = await fetch(`${API_BASE}/api/ollama/status`);
    if (!response.ok) return;
    const data = await response.json() as OllamaStatus;
    setOllamaStatus(data);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setUploadedVideo(null);
    setResult(null);
    setYoloResult(null);
    setYoloError('');
    setSelectedYoloLabels(['person']);
    setYoloStatus(selected ? 'ready' : 'idle');
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
      formData.append('inference_provider', inferenceProvider);
      formData.append('model_key', selectedModel);
      formData.append('ollama_base_url', ollamaBaseUrl);
      formData.append('ollama_model', selectedOllamaModel);
      formData.append('run_yolo', String(runYolo));
      formData.append('yolo_model', yoloModel);
      formData.append('selected_yolo_labels', selectedYoloLabels.join(','));
      formData.append('fast_mode', String(fastMode));

      const response = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(await readError(response));
      const data = await response.json() as AnalysisResponse;
      setActiveStage(stages.length);
      setResult(data);
      setStatus('done');
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
      if (data.eval_score) {
        await refreshEvaluationSummary();
      }
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const runYoloOnly = async () => {
    if (yoloStatus === 'running' || status === 'running' || status === 'uploading') return;

    try {
      setYoloStatus('running');
      setYoloError('');
      setYoloResult(null);
      const video = uploadedVideo ?? (youtubeUrl.trim() && !file ? await downloadYoutubeVideo() : await uploadVideo());
      const formData = new FormData();
      formData.append('filename', video.filename);
      formData.append('yolo_model', yoloModel);

      const response = await fetch(`${API_BASE}/api/yolo/analyze`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(await readError(response));
      const data = await response.json() as YoloResponse;
      setYoloResult(data);
      setSelectedYoloLabels(data.result.labels.includes('person') ? ['person'] : data.result.labels.slice(0, 1));
      setYoloStatus('done');
      setStatus('ready');
      setActiveStage(1);
    } catch (err) {
      setYoloStatus('error');
      setYoloError(err instanceof Error ? err.message : String(err));
      setStatus((current) => current === 'uploading' ? 'error' : current);
    }
  };

  const buildYoloEvidence = async () => {
    if (!uploadedVideo || !selectedYoloLabels.length) return;
    try {
      setYoloStatus('running');
      setYoloError('');
      const formData = new FormData();
      formData.append('filename', uploadedVideo.filename);
      formData.append('selected_labels', selectedYoloLabels.join(','));
      const response = await fetch(`${API_BASE}/api/yolo/evidence`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(await readError(response));
      const data = await response.json() as YoloResponse;
      setYoloResult(data);
      setYoloStatus('done');
    } catch (err) {
      setYoloStatus('error');
      setYoloError(err instanceof Error ? err.message : String(err));
    }
  };

  const selectExistingVideo = (video: VideoMeta) => {
    setFile(null);
    setPreviewUrl(`${API_BASE}${video.url}`);
    setUploadedVideo(video);
    setResult(null);
    setYoloResult(null);
    setYoloError('');
    setSelectedYoloLabels(['person']);
    setYoloStatus('ready');
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
          onRunYoloOnly={runYoloOnly}
          onBuildYoloEvidence={buildYoloEvidence}
          onSceneContextChange={setSceneContext}
          onSelectVideo={selectExistingVideo}
          onSelectedModelChange={setSelectedModel}
          inferenceProvider={inferenceProvider}
          onInferenceProviderChange={setInferenceProvider}
          selectedOllamaModel={selectedOllamaModel}
          onSelectedOllamaModelChange={setSelectedOllamaModel}
          ollamaBaseUrl={ollamaBaseUrl}
          onOllamaBaseUrlChange={setOllamaBaseUrl}
          onRunYoloChange={setRunYolo}
          onFastModeChange={setFastMode}
          onYoloModelChange={setYoloModel}
          onYoutubeUrlChange={setYoutubeUrl}
          previewUrl={previewUrl}
          result={result}
          sceneContext={sceneContext}
          selectedModel={selectedModel}
          ollamaStatus={ollamaStatus}
          status={status}
          uploadedVideo={uploadedVideo}
          videos={videos}
          runYolo={runYolo}
          fastMode={fastMode}
          yoloModel={yoloModel}
          yoloResult={yoloResult}
          selectedYoloLabels={selectedYoloLabels}
          onSelectedYoloLabelsChange={setSelectedYoloLabels}
          yoloStatus={yoloStatus}
          yoloError={yoloError}
          youtubeUrl={youtubeUrl}
          llmStatus={llmStatus}
          onRefreshOllamaStatus={refreshOllamaStatus}
          onRefreshLlmStatus={refreshLlmStatus}
        />
        <EvaluationDashboard summary={evaluationSummary} result={result} onSummaryChange={setEvaluationSummary} />
        <ReportDashboard result={result} />
        <ModelRecommendation
          llmStatus={llmStatus}
          ollamaStatus={ollamaStatus}
          onRefreshColab={refreshLlmStatus}
          onRefreshOllama={refreshOllamaStatus}
        />
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
  onRunYoloOnly: () => void;
  onBuildYoloEvidence: () => void;
  onSceneContextChange: (value: string) => void;
  onSelectVideo: (video: VideoMeta) => void;
  onSelectedModelChange: (value: string) => void;
  inferenceProvider: 'colab' | 'ollama';
  onInferenceProviderChange: (value: 'colab' | 'ollama') => void;
  selectedOllamaModel: string;
  onSelectedOllamaModelChange: (value: string) => void;
  ollamaBaseUrl: string;
  onOllamaBaseUrlChange: (value: string) => void;
  onRunYoloChange: (value: boolean) => void;
  onFastModeChange: (value: boolean) => void;
  onYoloModelChange: (value: string) => void;
  onYoutubeUrlChange: (value: string) => void;
  previewUrl: string;
  result: AnalysisResponse | null;
  sceneContext: string;
  selectedModel: string;
  ollamaStatus: OllamaStatus;
  status: AnalyzeState;
  uploadedVideo: VideoMeta | null;
  videos: VideoMeta[];
  runYolo: boolean;
  fastMode: boolean;
  yoloModel: string;
  yoloResult: YoloResponse | null;
  selectedYoloLabels: string[];
  onSelectedYoloLabelsChange: (value: string[]) => void;
  yoloStatus: AnalyzeState;
  yoloError: string;
  youtubeUrl: string;
  llmStatus: LlmStatus;
  onRefreshLlmStatus: () => void;
  onRefreshOllamaStatus: () => void;
};

function AnalyzeWorkspace(props: WorkspaceProps) {
  const {
    activeStage, apiBase, cameraId, elapsedSeconds, error, file, jsonPreview, onApiBaseChange, onCameraIdChange,
    onFileChange, onRunAnalysis, onRunYoloOnly, onBuildYoloEvidence, onSceneContextChange, onSelectVideo, previewUrl,
    result, sceneContext, status, uploadedVideo, videos, selectedModel, onSelectedModelChange,
    inferenceProvider, onInferenceProviderChange, selectedOllamaModel, onSelectedOllamaModelChange, ollamaBaseUrl, onOllamaBaseUrlChange,
    runYolo, onRunYoloChange, fastMode, onFastModeChange, yoloModel, onYoloModelChange, yoloResult, selectedYoloLabels, onSelectedYoloLabelsChange, yoloStatus, yoloError, youtubeUrl, onYoutubeUrlChange,
    llmStatus, ollamaStatus, onRefreshLlmStatus, onRefreshOllamaStatus,
  } = props;
  const ollamaOptions = ollamaStatus.models.length
    ? ollamaStatus.models.map((item) => item.name)
    : ['minicpm-v4.6:q4_K_M', 'qwen3.5:4b', 'qwen3:4b', 'qwen2.5:3b', 'gemma3:4b-it-qat'];

  const statusLabel = {
    idle: '영상을 기다리는 중',
    uploading: 'video 폴더에 mp4 저장 중',
    ready: '분석 준비 완료',
    running: `${stages[Math.max(activeStage, 0)]} 중`,
    done: '사고 분석 완료',
    error: '분석 오류',
  }[status];
  const reportDraft = buildReportDraft(result);
  const toggleYoloLabel = (label: string) => {
    const next = selectedYoloLabels.includes(label)
      ? selectedYoloLabels.filter((item) => item !== label)
      : [...selectedYoloLabels, label];
    onSelectedYoloLabelsChange(next);
  };

  return (
    <section id="workspace" className="workspace-band">
      <div className="shell">
        <div className="section-title">
          <div>
            <h2>영상 입력부터 사고 원인 분석까지</h2>
            <p>사용자가 mp4를 넣으면 백엔드가 루트 <code>video/</code> 폴더에 저장하고, 저장된 영상을 기준으로 사고 유형과 원인 흐름을 분석합니다.</p>
          </div>
          <p className="muted">YouTube URL, VL 모델 선택, YOLO-only preview, pretrained YOLO evidence 옵션을 한 화면에서 제어합니다.</p>
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
              <label><span>실행 위치</span><select value={inferenceProvider} onChange={(event) => onInferenceProviderChange(event.target.value as 'colab' | 'ollama')} aria-label="실행 위치 선택">
                <option value="colab">Colab 서버</option>
                <option value="ollama">Local Ollama</option>
              </select></label>
              {inferenceProvider === 'colab' ? (
                <label><span>Colab API Base</span><input value={apiBase} onChange={(event) => onApiBaseChange(event.target.value)} placeholder="https://xxxxx.ngrok-free.app/v1" aria-label="Colab API Base" /></label>
              ) : (
                <label><span>Ollama Base URL</span><input value={ollamaBaseUrl} onChange={(event) => onOllamaBaseUrlChange(event.target.value)} placeholder="http://127.0.0.1:11434" aria-label="Ollama Base URL" /></label>
              )}
              {inferenceProvider === 'colab' ? (
              <label><span>Colab VL 모델</span><select value={selectedModel} onChange={(event) => onSelectedModelChange(event.target.value)} aria-label="VL 모델 선택">
                <option value="qwen3_vl_32b">Qwen3-VL-32B</option>
                <option value="internvl3">InternVL3</option>
                <option value="llava_onevision_2_8b">LLaVA-OneVision-2-8B</option>
                <option value="minicpm_v_4_5">MiniCPM-V 4.5</option>
                <option value="qwen25_vl_32b">Qwen2.5-VL-32B</option>
              </select></label>
              ) : (
              <label><span>Ollama 모델</span><select value={selectedOllamaModel} onChange={(event) => onSelectedOllamaModelChange(event.target.value)} aria-label="Ollama 모델 선택">
                {ollamaOptions.map((model) => <option key={model} value={model}>{model}</option>)}
              </select></label>
              )}
              <label><span>분석 질문</span><select aria-label="분석 질문"><option>사고 유형 + 부상자 수 + 원인</option><option>사고보고서 초안까지 생성</option></select></label>
              <label><span>YouTube URL</span><input value={youtubeUrl} onChange={(event) => onYoutubeUrlChange(event.target.value)} placeholder="https://www.youtube.com/watch?v=..." aria-label="YouTube URL" /></label>
              <label><span>YOLO 모델 파일</span><input value={yoloModel} onChange={(event) => onYoloModelChange(event.target.value)} aria-label="YOLO 모델 파일" /></label>
            </div>
            <div className="workspace-options">
              <label className="check-control">
                <input type="checkbox" checked={runYolo} onChange={(event) => onRunYoloChange(event.target.checked)} />
                <span>pretrained YOLO evidence 사용</span>
              </label>
              <label className="check-control">
                <input type="checkbox" checked={fastMode} onChange={(event) => onFastModeChange(event.target.checked)} />
                <span>빠른 분석 모드</span>
              </label>
              <button className="mini-button" type="button" onClick={onRefreshLlmStatus}>Colab 연결 확인</button>
              <button className="mini-button" type="button" onClick={onRefreshOllamaStatus}>Ollama 연결 확인</button>
              <span className={`live-badge ${llmStatus.live ? 'on' : 'off'}`}>
                {llmStatus.live ? 'LIVE' : 'OFF'} {llmStatus.model || selectedModel}
              </span>
              <span className={`live-badge ${ollamaStatus.live ? 'on' : 'off'}`}>
                {ollamaStatus.live ? 'OLLAMA ON' : 'OLLAMA OFF'} {inferenceProvider === 'ollama' ? selectedOllamaModel : ''}
              </span>
            </div>
            <p className="mode-note">
              {fastMode
                ? '빠른 모드: auto-moment를 생략하고 고정 프레임 6장만 VL에 전달합니다.'
                : '정밀 모드: overview sheet로 사고 순간을 먼저 찾은 뒤 최종 판단을 실행합니다.'}
            </p>
            <div className="yolo-runner">
              <button
                className="button secondary"
                type="button"
                onClick={onRunYoloOnly}
                disabled={(!file && !uploadedVideo && !youtubeUrl.trim()) || yoloStatus === 'running' || status === 'running' || status === 'uploading'}
              >
                {yoloStatus === 'running' ? 'YOLO 적용 중...' : 'YOLO만 실행'}
              </button>
              <span>VL 분석 전에 사람/장비 bbox evidence와 어노테이션 영상을 먼저 확인합니다.</span>
            </div>
            {(yoloResult || yoloError) && (
              <div className="yolo-result">
                <div className="panel-head compact">
                  <div>
                    <strong>YOLO-only preview</strong>
                    <span>{yoloResult ? `${yoloResult.result.detections.length} detections · ${yoloResult.result.labels.join(', ') || 'label 없음'}` : 'YOLO 실행 오류'}</span>
                  </div>
                  <span className={`pill ${yoloStatus}`}>{yoloStatus.toUpperCase()}</span>
                </div>
                {yoloResult?.annotated_sheet_url && (
                  <img className="annotated-sheet" src={`${API_BASE}${yoloResult.annotated_sheet_url}`} alt="YOLO annotated contact sheet" />
                )}
                {!yoloResult?.annotated_sheet_url && yoloResult?.annotated_video_url && (
                  <video className="video-preview yolo-video" src={`${API_BASE}${yoloResult.annotated_video_url}`} controls />
                )}
                {yoloResult && (
                  <>
                  <div className="label-filter">
                    <strong>VL evidence로 사용할 라벨 선택</strong>
                    <div>
                      {yoloResult.result.labels.map((label) => (
                        <label key={label}>
                          <input type="checkbox" checked={selectedYoloLabels.includes(label)} onChange={() => toggleYoloLabel(label)} />
                          <span>{label}</span>
                        </label>
                      ))}
                    </div>
                    <button className="mini-button" type="button" onClick={onBuildYoloEvidence} disabled={!selectedYoloLabels.length || yoloStatus === 'running'}>
                      선택 라벨로 evidence 생성
                    </button>
                  </div>
                  <div className="yolo-meta">
                    <span>model</span><strong>{yoloModel}</strong>
                    <span>confidence</span><strong>{typeof yoloResult.result.confidence === 'number' ? yoloResult.result.confidence.toFixed(2) : '-'}</strong>
                    <span>selected</span><strong>{selectedYoloLabels.join(', ') || '-'}</strong>
                  </div>
                  </>
                )}
                {yoloError && <div className="error-box">{yoloError}</div>}
              </div>
            )}
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
            {(result?.annotated_sheet_url || result?.annotated_video_url) && (
              <div className="detecting-preview">
                <div className="panel-head compact">
                  <div>
                    <strong>YOLO annotated contact sheet</strong>
                    <span>{result.pt_result?.detections.length ?? 0} detections · {result.pt_result?.labels.join(', ') || 'label 없음'}</span>
                  </div>
                  <span className="pill done">DETECTED</span>
                </div>
                {result.annotated_sheet_url && (
                  <img className="annotated-sheet" src={`${API_BASE}${result.annotated_sheet_url}`} alt="YOLO annotated contact sheet" />
                )}
                {!result.annotated_sheet_url && result.annotated_video_url && (
                  <video className="video-preview yolo-video" src={`${API_BASE}${result.annotated_video_url}`} controls />
                )}
              </div>
            )}
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

function EvaluationDashboard({
  summary,
  result,
  onSummaryChange,
}: {
  summary: EvaluationSummary;
  result: AnalysisResponse | null;
  onSummaryChange: (summary: EvaluationSummary) => void;
}) {
  const [evalCase, setEvalCase] = useState<EvaluationCase>({
    video_id: 'accident_001',
    accident_detected: true,
    accident_type: '추락',
    injured_count: 1,
    cause_keywords: ['비계', '이동', '전도', '상부 작업자 추락'],
    accident_time_range: [14, 18],
    required_evidence: ['작업자 위치 변화', '구조물 이동', '추락 후 바닥 접촉'],
  });
  const [evalSaveMessage, setEvalSaveMessage] = useState('');
  const [latestEvalScore, setLatestEvalScore] = useState<Record<string, unknown> | null>(result?.eval_score ?? null);
  useEffect(() => {
    if (!result?.video.filename) return;
    setEvalCase((current) => ({
      ...current,
      video_id: result.video.filename.replace(/\.[^.]+$/, ''),
      accident_type: result.analysis.accident_type_ko || current.accident_type,
      injured_count: result.analysis.injured_count ?? current.injured_count,
      accident_time_range: [
        result.analysis.clip_start_offset ?? current.accident_time_range[0],
        result.analysis.clip_end_offset ?? current.accident_time_range[1],
      ],
    }));
  }, [result]);
  useEffect(() => {
    if (result?.eval_score) setLatestEvalScore(result.eval_score);
  }, [result]);
  const scores = summary.scores.length ? summary.scores : fallbackModelScores;
  const best = scores.find((item) => item.name === summary.best_model) ?? scores[0];
  const maxScore = Math.max(...scores.map((item) => item.total));
  const charts = summary.charts?.length ? summary.charts : [
    'eval/output/confusion_matrix.png',
    'eval/output/cause_recall_by_prompt.png',
    'eval/output/latency_boxplot.png',
    'eval/output/model_score_bar.png',
  ];

  const saveEvalCase = async () => {
    setEvalSaveMessage('저장 중...');
    const response = await fetch(`${API_BASE}/api/evaluation/cases`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(evalCase),
    });
    if (!response.ok) {
      setEvalSaveMessage(await readError(response));
      return;
    }
    const data = await response.json() as { summary?: EvaluationSummary; score?: Record<string, unknown> | null };
    if (data.summary?.scores?.length) onSummaryChange(data.summary);
    setLatestEvalScore(data.score ?? null);
    setEvalSaveMessage(data.score ? '정답 라벨 저장 및 최신 분석 결과 채점 완료' : '정답 라벨 저장 완료. 같은 video_id 분석 후 자동 채점됩니다.');
  };

  return (
    <section id="dashboard" className="shell">
      <div className="section-title">
        <div>
          <h2>Evaluation Dashboard</h2>
          <p>사고 영상 데이터셋으로 모델과 프롬프트별 정확도를 비교하고, matplotlib 산출물을 프론트엔드에 탑재하는 영역입니다.</p>
        </div>
        <p className="muted">현재 데이터셋: <code>{summary.dataset ?? 'sample_accident_eval'}</code> · 갱신: <code>{summary.updated_at ?? 'sample'}</code></p>
      </div>

      <div className="eval-labeler">
        <div className="dashboard-card wide">
          <div className="card-title">정답 라벨 입력</div>
          <div className="eval-form">
            <label><span>Video ID</span><input value={evalCase.video_id} onChange={(event) => setEvalCase({ ...evalCase, video_id: event.target.value })} /></label>
            <label><span>사고 여부</span><select value={String(evalCase.accident_detected)} onChange={(event) => setEvalCase({ ...evalCase, accident_detected: event.target.value === 'true' })}>
              <option value="true">사고 발생</option>
              <option value="false">사고 아님</option>
            </select></label>
            <label><span>사고 유형</span><select value={evalCase.accident_type} onChange={(event) => setEvalCase({ ...evalCase, accident_type: event.target.value })}>
              <option>추락</option>
              <option>낙상</option>
              <option>화재</option>
              <option>충돌</option>
              <option>끼임</option>
              <option>붕괴</option>
              <option>기타</option>
            </select></label>
            <label><span>부상자 수</span><input type="number" min="0" value={evalCase.injured_count} onChange={(event) => setEvalCase({ ...evalCase, injured_count: Number(event.target.value) })} /></label>
            <label><span>사고 시작초</span><input type="number" min="0" value={evalCase.accident_time_range[0]} onChange={(event) => setEvalCase({ ...evalCase, accident_time_range: [Number(event.target.value), evalCase.accident_time_range[1]] })} /></label>
            <label><span>사고 종료초</span><input type="number" min="0" value={evalCase.accident_time_range[1]} onChange={(event) => setEvalCase({ ...evalCase, accident_time_range: [evalCase.accident_time_range[0], Number(event.target.value)] })} /></label>
            <label className="full"><span>원인 키워드</span><input value={evalCase.cause_keywords.join(', ')} onChange={(event) => setEvalCase({ ...evalCase, cause_keywords: splitCsv(event.target.value) })} /></label>
            <label className="full"><span>필수 근거</span><textarea rows={3} value={evalCase.required_evidence.join(', ')} onChange={(event) => setEvalCase({ ...evalCase, required_evidence: splitCsv(event.target.value) })} /></label>
          </div>
          <div className="eval-actions">
            <button className="mini-button" type="button" onClick={saveEvalCase}>DB에 정답 라벨 저장</button>
            <span>{evalSaveMessage || '저장하면 같은 video_id의 최신 분석 결과를 자동 채점합니다.'}</span>
          </div>
        </div>
        <div className="code-panel eval-json">
          <div className="code-title">eval_cases.json label preview</div>
          <pre><code>{JSON.stringify(evalCase, null, 2)}</code></pre>
        </div>
      </div>

      <div className="score-strip">
        <MetricCard label="Best Model" value={best.name} detail={best.prompt} />
        <MetricCard label="Total Score" value={best.total.toFixed(2)} detail="weighted score" />
        <MetricCard label="Cause Recall" value={`${Math.round(best.causeRecall * 100)}%`} detail="원인 키워드 회수율" />
        <MetricCard label="Semantic" value={`${Math.round((best.semanticScore ?? 0) * 100)}%`} detail="qwen2.5:3b 의미 채점" />
        <MetricCard label="JSON Valid" value={`${Math.round(best.jsonValid * 100)}%`} detail="schema pass rate" />
      </div>

      {latestEvalScore && (
        <div className="eval-score-card">
          <div className="card-title">최신 채점 결과</div>
          <div className="mini-table">
            <span>Total</span><strong>{Number(latestEvalScore.total_score ?? 0).toFixed(2)}</strong>
            <span>Type</span><strong>{Number(latestEvalScore.type_score ?? 0).toFixed(2)}</strong>
            <span>Cause Recall</span><strong>{Number(latestEvalScore.cause_recall ?? 0).toFixed(2)}</strong>
            <span>Semantic</span><strong>{Number(latestEvalScore.semantic_score ?? 0).toFixed(2)}</strong>
            <span>Time IoU</span><strong>{Number(latestEvalScore.time_iou ?? 0).toFixed(2)}</strong>
          </div>
          <p>{String(latestEvalScore.semantic_reason ?? 'qwen2.5:3b 의미 채점 결과가 여기에 표시됩니다.')}</p>
        </div>
      )}

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
            <span>Semantic Score</span><strong>{Math.round((best.semanticScore ?? 0) * 100)}%</strong>
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
  const report = buildReportDraft(result);

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
            <textarea value={report.overview} readOnly rows={3} aria-label="사고 개요" />
            <h4>2. 원인 분석</h4>
            <textarea value={report.simple} readOnly rows={4} aria-label="원인 분석" />
            <h4>3. 세부 설명</h4>
            <textarea value={report.details} readOnly rows={6} aria-label="세부 설명" />
            <h4>4. 재발 방지 조치</h4>
            <textarea value={report.actions.map((action, index) => `${index + 1}. ${action}`).join('\n')} readOnly rows={5} aria-label="재발 방지 조치" />
          </article>
          <div className="dashboard-card">
            <div className="card-title">보고서 구성 필드</div>
            <div className="report-fields">
              <div><span>사고 유형</span><strong>{result?.analysis.accident_type_ko ?? '분석 대기'}</strong></div>
              <div><span>부상자 수</span><strong>{typeof result?.analysis.injured_count === 'number' ? `${result.analysis.injured_count}명` : '확인 필요'}</strong></div>
              <div><span>신뢰도</span><strong>{typeof result?.analysis.confidence === 'number' ? result.analysis.confidence.toFixed(2) : '-'}</strong></div>
              <div><span>분석 구간</span><strong>{typeof result?.analysis.clip_start_offset === 'number' && typeof result?.analysis.clip_end_offset === 'number' ? `${result.analysis.clip_start_offset}s ~ ${result.analysis.clip_end_offset}s` : '확인 필요'}</strong></div>
            </div>
            <div className="card-title spacious">다음 확장</div>
            <div className="paper-flow">
              <span>검토자 수정</span>
              <span>PDF Export</span>
              <span>평가 결과 첨부</span>
              <span>논문형 요약</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function ModelRecommendation({
  llmStatus,
  ollamaStatus,
  onRefreshColab,
  onRefreshOllama,
}: {
  llmStatus: LlmStatus;
  ollamaStatus: OllamaStatus;
  onRefreshColab: () => void;
  onRefreshOllama: () => void;
}) {
  const models = [
    {
      name: 'Qwen3-VL-32B',
      tag: '1순위 reasoning 모델',
      role: '긴 영상 이해와 원인-결과 설명에 가장 잘 맞는 주 모델 후보입니다.',
      notebook: 'server/Qwen3_VL_32B_colab_server.ipynb',
      useCase: '최종 보고서 초안, 사고 원인 분석, 고정밀 평가 기준',
      colabModel: 'Qwen/Qwen3-VL-32B-Instruct',
      ollamaModel: '',
    },
    {
      name: 'InternVL3',
      tag: '강한 open-source MLLM',
      role: 'perception과 reasoning 성능 비교군으로 적합합니다.',
      notebook: 'server/InternVL3_colab_server.ipynb',
      useCase: 'Qwen3-VL과 perception/reasoning 비교',
      colabModel: 'OpenGVLab/InternVL3-38B',
      ollamaModel: '',
    },
    {
      name: 'LLaVA-OneVision-2',
      tag: '8B급 video baseline',
      role: '속도와 성능의 균형을 비교하기 좋은 경량 video baseline입니다.',
      notebook: 'server/LLaVA_OneVision_2_8B_colab_server.ipynb',
      useCase: '저비용 baseline, latency 비교',
      colabModel: 'lmms-lab/LLaVA-OneVision-2-8B-ov',
      ollamaModel: '',
    },
    {
      name: 'MiniCPM-V 4.5',
      tag: '효율형 video 모델',
      role: '긴 영상 token 압축과 빠른 실험용 fallback 후보입니다.',
      notebook: 'server/MiniCPM_V_4_5_colab_server.ipynb',
      useCase: '빠른 반복 실험, fallback 비교',
      colabModel: 'openbmb/MiniCPM-V-4_5',
      ollamaModel: '',
    },
    {
      name: 'MiniCPM-V 4.6 local',
      tag: '3050 4GB 로컬 추천',
      role: '로컬 Ollama에서 contact sheet 이미지 기반 빠른 sanity check용으로 가장 현실적인 후보입니다.',
      notebook: 'ollama pull minicpm-v4.6:q4_K_M',
      useCase: '저사양 fallback, 로컬 빠른 검증',
      colabModel: '',
      ollamaModel: 'minicpm-v4.6:q4_K_M',
    },
    {
      name: 'Qwen3.5 2B local',
      tag: '로컬 reasoning 비교',
      role: '3050 4GB에서 시도 가능한 소형 vision/reasoning 비교군입니다.',
      notebook: 'ollama pull qwen3.5:2b-q4_K_M',
      useCase: '로컬 원인 추론 비교',
      colabModel: '',
      ollamaModel: 'qwen3.5:2b-q4_K_M',
    },
    {
      name: 'Gemma3 4B local',
      tag: '로컬 보조 비교군',
      role: '4GB VRAM에서는 빡빡하지만 QAT 버전으로 보조 실험 가치가 있습니다.',
      notebook: 'ollama pull gemma3:4b-it-qat',
      useCase: '로컬 multimodal 비교',
      colabModel: '',
      ollamaModel: 'gemma3:4b-it-qat',
    },
  ];
  const installedOllamaNames = new Set(
    ollamaStatus.models.flatMap((item) => [item.name, item.model]).filter(Boolean).map((name) => String(name).toLowerCase()),
  );
  const connectedColabModel = llmStatus.model.toLowerCase();

  return (
    <section id="models" className="shell">
      <div className="section-title">
        <div>
          <h2>Qwen3-VL 외 비교 모델 후보</h2>
          <p>최종 성능은 같은 사고 영상 데이터셋에서 모델과 프롬프트별로 수치화해 비교합니다.</p>
        </div>
        <div className="model-check-actions">
          <span className={`live-badge ${llmStatus.live ? 'on' : 'off'}`}>{llmStatus.live ? 'Colab 연결됨' : 'Colab 미연결'}</span>
          <span className={`live-badge ${ollamaStatus.live ? 'on' : 'off'}`}>{ollamaStatus.live ? 'Ollama 연결됨' : 'Ollama 미연결'}</span>
          <button className="mini-button" type="button" onClick={onRefreshColab}>Colab 체크</button>
          <button className="mini-button" type="button" onClick={onRefreshOllama}>Ollama 체크</button>
        </div>
      </div>
      <div className="model-status-grid">
        {models.map((model) => {
          const ollamaInstalled = model.ollamaModel
            ? installedOllamaNames.has(model.ollamaModel.toLowerCase())
            : null;
          const colabConnected = model.colabModel
            ? llmStatus.live && connectedColabModel.includes(model.colabModel.toLowerCase())
            : null;
          return (
          <article className="feature model-card" key={model.name}>
            <span className="tag">{model.tag}</span>
            <h3>{model.name}</h3>
            <p className="muted">{model.role}</p>
            <div className="model-checks">
              <div className={`model-check ${colabConnected ? 'ok' : 'off'} ${colabConnected === null ? 'muted-check' : ''}`}>
                <span>Colab</span>
                <strong>{colabConnected === null ? '해당 없음' : colabConnected ? '연결됨' : '미연결'}</strong>
              </div>
              <div className={`model-check ${ollamaInstalled ? 'ok' : 'off'} ${ollamaInstalled === null ? 'muted-check' : ''}`}>
                <span>Ollama</span>
                <strong>{ollamaInstalled === null ? '해당 없음' : ollamaInstalled ? '설치됨' : '미설치'}</strong>
              </div>
            </div>
            <dl className="model-meta">
              <dt>{model.ollamaModel ? 'Install' : 'Notebook'}</dt>
              <dd>{model.notebook}</dd>
              <dt>Use case</dt>
              <dd>{model.useCase}</dd>
              {model.colabModel && <><dt>Colab model</dt><dd>{model.colabModel}</dd></>}
              {model.ollamaModel && <><dt>Ollama model</dt><dd>{model.ollamaModel}</dd></>}
            </dl>
          </article>
          );
        })}
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
