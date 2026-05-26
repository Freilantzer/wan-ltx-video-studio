import {
  Activity,
  Clock3,
  FolderOpen,
  Gauge,
  Image,
  Layers3,
  Play,
  RefreshCw,
  Save,
  Settings,
  Sparkles,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

type SeedPolicy = "fixed" | "increment" | "none";
type PromptMode = "shared" | "perSegment";

type LoraSelection = {
  id: string;
  name: string;
  role: "workflow" | "creative";
  strength: number;
  enabled: boolean;
};

type FormState = {
  width: number;
  height: number;
  segmentCount: number;
  fps: number;
  chunkSeconds: number;
  startImage: string;
  prompt: string;
  promptMode: PromptMode;
  segmentPrompts: string[];
  negativePrompt: string;
  seed: number;
  seedPolicy: SeedPolicy;
  baseModel: string;
  loras: LoraSelection[];
  pixelBudget: number;
  motionFrames: number;
  motionAmplitude: number;
};

type SegmentPlan = {
  index: number;
  inputFrames: number;
  outputFrames: number;
  inputDurationSeconds: number;
  outputDurationSeconds: number;
  seed: number | null;
  prompt: string;
  continuity: {
    source: string;
    trimStartFrames: number;
    motionFrames: number;
    motionAmplitude: number;
    previousSegmentIndex: number | null;
    startImage: string | null;
  };
};

type PlanResponse = {
  targetTimelineFrames: number;
  requestedChunkFrames: number;
  actualOutputFrames: number;
  extraOutputFrames: number;
  targetDurationSeconds: number;
  actualOutputDurationSeconds: number;
  pixels: number;
  engine: {
    baseModel: string;
    loras: Array<{
      name: string;
      role: string;
      strength: number;
      enabled: boolean;
    }>;
  };
  segments: SegmentPlan[];
};

const modelProfiles = [
  {
    id: "wan22_i2v_a14b_fp8_original",
    label: "WAN 2.2 I2V A14B FP8",
    note: "Original",
    family: "WAN 2.2",
    includesLightning: false,
    targetVramGb: 25,
  },
  {
    id: "wan22_i2v_a14b_fp8_lightning_workflow",
    label: "WAN 2.2 I2V A14B FP8 + Lightning",
    note: "Workflow",
    family: "WAN 2.2",
    includesLightning: true,
    targetVramGb: 25,
  },
  {
    id: "wan22_ti2v_5b_fp16",
    label: "WAN 2.2 TI2V 5B",
    note: "5B",
    family: "WAN 2.2",
    includesLightning: false,
    targetVramGb: 16,
  },
  {
    id: "wan22_i2v_a14b_q8_gguf",
    label: "WAN 2.2 I2V A14B Q8 GGUF",
    note: "GGUF",
    family: "WAN 2.2",
    includesLightning: false,
    targetVramGb: 22,
  },
  {
    id: "ltx23_dev_distilled",
    label: "LTX 2.3 Dev Distilled",
    note: "LTX",
    family: "LTX",
    includesLightning: false,
    targetVramGb: 24,
  },
];

const defaultPrompt = "cinematic POV drive through a dense city at dusk, natural camera motion, detailed streets";

const defaultLoras: LoraSelection[] = [
  {
    id: "wan_lightning_high_noise",
    name: "WAN Lightning high-noise",
    role: "workflow",
    strength: 1,
    enabled: false,
  },
  {
    id: "wan_lightning_low_noise",
    name: "WAN Lightning low-noise",
    role: "workflow",
    strength: 1,
    enabled: false,
  },
  {
    id: "cinematic_motion",
    name: "Cinematic motion",
    role: "creative",
    strength: 0.65,
    enabled: false,
  },
];

const defaultForm: FormState = {
  width: 1280,
  height: 720,
  segmentCount: 3,
  fps: 16,
  chunkSeconds: 5,
  startImage: "inputs/start_frames_1280x720/woman_black_sand_beach.png",
  prompt: defaultPrompt,
  promptMode: "shared",
  segmentPrompts: [defaultPrompt, defaultPrompt, defaultPrompt],
  negativePrompt: "blur, low quality, flicker, warped hands, extra limbs, text",
  seed: 1234,
  seedPolicy: "fixed",
  baseModel: "wan22_i2v_a14b_fp8_original",
  loras: defaultLoras,
  pixelBudget: 2_100_000,
  motionFrames: 10,
  motionAmplitude: 1.15,
};

const pixelProfiles = [
  { label: "720p", value: 1_100_000 },
  { label: "5090", value: 2_100_000 },
  { label: "Max", value: 2_400_000 },
];

export function App() {
  const [form, setForm] = useState<FormState>(defaultForm);
  const [plan, setPlan] = useState<PlanResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const totalSeconds = form.segmentCount * form.chunkSeconds;
  const pixelMegapixels = useMemo(() => form.width * form.height / 1_000_000, [form.width, form.height]);
  const activeModel = modelProfiles.find((model) => model.id === form.baseModel) ?? modelProfiles[0];

  async function refreshPlan(nextForm = form) {
    setIsLoading(true);
    try {
      const response = await fetch("/api/plan", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(toPlanPayload(nextForm)),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error ?? "Planning failed");
      }
      setPlan(payload);
      setError(null);
    } catch (currentError) {
      setPlan(null);
      setError(currentError instanceof Error ? currentError.message : "Planning failed");
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      refreshPlan(form);
    }, 180);
    return () => window.clearTimeout(timeout);
  }, [form]);

  function updateField<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function updateSegmentCount(value: number) {
    const segmentCount = clamp(Math.round(value || 1), 1, 5);
    setForm((current) => ({
      ...current,
      segmentCount,
      segmentPrompts: resizePrompts(current.segmentPrompts, segmentCount, current.prompt),
    }));
  }

  function updateSegmentPrompt(index: number, value: string) {
    setForm((current) => {
      const segmentPrompts = resizePrompts(current.segmentPrompts, current.segmentCount, current.prompt);
      segmentPrompts[index] = value;
      return { ...current, segmentPrompts };
    });
  }

  function updateLora(id: string, patch: Partial<LoraSelection>) {
    setForm((current) => ({
      ...current,
      loras: current.loras.map((lora) => (lora.id === id ? { ...lora, ...patch } : lora)),
    }));
  }

  return (
    <main className="app-shell">
      <header className="top-bar">
        <div className="brand">
          <Sparkles size={20} aria-hidden="true" />
          <div>
            <h1>WAN/LTX Video Studio</h1>
            <span>{activeModel.family} planner</span>
          </div>
        </div>
        <div className="top-actions">
          <StatusPill label={error ? "Plan blocked" : "Plan ready"} tone={error ? "danger" : "ok"} />
          <button className="icon-button" title="Refresh plan" onClick={() => refreshPlan()}>
            <RefreshCw size={18} aria-hidden="true" />
          </button>
          <button className="icon-button" title="Open renders">
            <FolderOpen size={18} aria-hidden="true" />
          </button>
          <button className="icon-button" title="Settings">
            <Settings size={18} aria-hidden="true" />
          </button>
        </div>
      </header>

      <section className="studio-grid">
        <aside className="panel left-panel" aria-label="Shot settings">
          <PanelHeader icon={<Image size={17} />} title="Shot" />
          <div className="segmented">
            <button
              className={form.promptMode === "shared" ? "selected" : ""}
              onClick={() => updateField("promptMode", "shared")}
            >
              Shared
            </button>
            <button
              className={form.promptMode === "perSegment" ? "selected" : ""}
              onClick={() => updateField("promptMode", "perSegment")}
            >
              Per Segment
            </button>
          </div>
          {form.promptMode === "shared" ? (
            <label className="field">
              <span>Positive</span>
              <textarea
                value={form.prompt}
                onChange={(event) => updateField("prompt", event.target.value)}
                rows={7}
              />
            </label>
          ) : (
            <div className="segment-prompt-list">
              {Array.from({ length: form.segmentCount }, (_, index) => (
                <label className="field" key={index}>
                  <span>Segment {index + 1}</span>
                  <textarea
                    value={form.segmentPrompts[index] ?? form.prompt}
                    onChange={(event) => updateSegmentPrompt(index, event.target.value)}
                    rows={3}
                  />
                </label>
              ))}
            </div>
          )}
          <label className="field">
            <span>Negative</span>
            <textarea
              value={form.negativePrompt}
              onChange={(event) => updateField("negativePrompt", event.target.value)}
              rows={4}
            />
          </label>
          <label className="field">
            <span>Start image</span>
            <input
              value={form.startImage}
              onChange={(event) => updateField("startImage", event.target.value)}
            />
          </label>
          <div className="field-row">
            <label className="field">
              <span>Seed</span>
              <input
                type="number"
                value={form.seed}
                onChange={(event) => updateField("seed", Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Policy</span>
              <select
                value={form.seedPolicy}
                onChange={(event) => updateField("seedPolicy", event.target.value as SeedPolicy)}
              >
                <option value="fixed">Fixed</option>
                <option value="increment">Increment</option>
                <option value="none">Random</option>
              </select>
            </label>
          </div>
        </aside>

        <section className="workspace">
          <div className="preview-panel">
            <div className="preview-frame">
              <div className="preview-mark">
                <Play size={28} aria-hidden="true" />
                <span>{form.width} x {form.height}</span>
              </div>
            </div>
            <div className="preview-metrics">
              <Metric icon={<Clock3 size={16} />} label="Duration" value={formatSeconds(plan?.actualOutputDurationSeconds)} />
              <Metric icon={<Layers3 size={16} />} label="Segments" value={String(form.segmentCount)} />
              <Metric icon={<Activity size={16} />} label="Frames" value={plan ? String(plan.actualOutputFrames) : "-"} />
              <Metric icon={<Gauge size={16} />} label="Pixels" value={`${pixelMegapixels.toFixed(2)} MP`} />
            </div>
          </div>

          <div className="panel timeline-panel">
            <PanelHeader icon={<Activity size={17} />} title="Segments" />
            {error ? <div className="error-box">{error}</div> : <Timeline plan={plan} />}
          </div>

          <div className="panel queue-panel">
            <PanelHeader icon={<Play size={17} />} title="Queue" />
            <div className="queue-row">
              <span>{activeModel.label}</span>
              <button className="primary-button" title="Queue render" disabled>
                <Play size={16} aria-hidden="true" />
                Queue
              </button>
            </div>
          </div>
        </section>

        <aside className="panel right-panel" aria-label="Engine settings">
          <PanelHeader icon={<Settings size={17} />} title="Engine" />
          <div className="segmented">
            <button className={activeModel.family === "WAN 2.2" ? "selected" : ""}>WAN 2.2</button>
            <button className={activeModel.family === "LTX" ? "selected" : ""}>LTX</button>
          </div>
          <label className="field">
            <span>Base model</span>
            <select
              value={form.baseModel}
              onChange={(event) => updateField("baseModel", event.target.value)}
            >
              {modelProfiles.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
          </label>
          <div className="model-chip-row">
            <span className="model-chip">{activeModel.note}</span>
            <span className="model-chip">720p target ~{activeModel.targetVramGb} GB VRAM</span>
            {activeModel.includesLightning ? <span className="model-chip">Lightning in workflow</span> : null}
          </div>
          <div className="field-grid">
            <label className="field">
              <span>Width</span>
              <input
                type="number"
                step={16}
                value={form.width}
                onChange={(event) => updateField("width", Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Height</span>
              <input
                type="number"
                step={16}
                value={form.height}
                onChange={(event) => updateField("height", Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Segments</span>
              <input
                type="number"
                min={1}
                max={5}
                step={1}
                value={form.segmentCount}
                onChange={(event) => updateSegmentCount(Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Seconds/seg</span>
              <input
                type="number"
                step={0.25}
                value={form.chunkSeconds}
                onChange={(event) => updateField("chunkSeconds", Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>FPS</span>
              <input
                type="number"
                step={1}
                value={form.fps}
                onChange={(event) => updateField("fps", Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Motion</span>
              <input
                type="number"
                step={1}
                value={form.motionFrames}
                onChange={(event) => updateField("motionFrames", Number(event.target.value))}
              />
            </label>
          </div>
          <label className="field range-field">
            <span>Amplitude {form.motionAmplitude.toFixed(2)}</span>
            <input
              type="range"
              min={1}
              max={2}
              step={0.05}
              value={form.motionAmplitude}
              onChange={(event) => updateField("motionAmplitude", Number(event.target.value))}
            />
          </label>
          <div className="segmented">
            {pixelProfiles.map((profile) => (
              <button
                key={profile.value}
                className={form.pixelBudget === profile.value ? "selected" : ""}
                onClick={() => updateField("pixelBudget", profile.value)}
              >
                {profile.label}
              </button>
            ))}
          </div>
          <div className="lora-list">
            {form.loras.map((lora) => (
              <div className="lora-row" key={lora.id}>
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={activeModel.includesLightning && lora.role === "workflow" ? true : lora.enabled}
                    disabled={activeModel.includesLightning && lora.role === "workflow"}
                    onChange={(event) => updateLora(lora.id, { enabled: event.target.checked })}
                  />
                  <span>{lora.name}</span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={1.5}
                  step={0.05}
                  value={lora.strength}
                  disabled={!lora.enabled || (activeModel.includesLightning && lora.role === "workflow")}
                  onChange={(event) => updateLora(lora.id, { strength: Number(event.target.value) })}
                />
                <strong>{activeModel.includesLightning && lora.role === "workflow" ? "built in" : lora.strength.toFixed(2)}</strong>
              </div>
            ))}
          </div>
          <div className="summary-grid">
            <Metric label="Target" value={plan ? `${plan.targetTimelineFrames} f` : "-"} />
            <Metric label="Chunk" value={plan ? `${plan.requestedChunkFrames} f` : "-"} />
            <Metric label="Total" value={formatSeconds(totalSeconds)} />
            <Metric label="VRAM" value={`~${activeModel.targetVramGb} GB`} />
          </div>
          <div className="button-row">
            <button className="secondary-button" title="Save preset">
              <Save size={16} aria-hidden="true" />
              Save
            </button>
            <button className="primary-button" title="Refresh plan" onClick={() => refreshPlan()}>
              <RefreshCw size={16} aria-hidden="true" />
              Plan
            </button>
          </div>
        </aside>
      </section>
    </main>
  );
}

function toPlanPayload(form: FormState) {
  const activeModel = modelProfiles.find((model) => model.id === form.baseModel) ?? modelProfiles[0];
  const loras = form.loras
    .filter((lora) => lora.enabled && !(activeModel.includesLightning && lora.role === "workflow"))
    .map((lora) => ({
      name: lora.id,
      role: lora.role,
      strength: lora.strength,
      enabled: lora.enabled,
    }));

  return {
    width: form.width,
    height: form.height,
    totalSeconds: form.segmentCount * form.chunkSeconds,
    fps: form.fps,
    chunkSeconds: form.chunkSeconds,
    startImage: form.startImage,
    prompt: form.prompt,
    segmentPrompts: form.promptMode === "perSegment" ? resizePrompts(form.segmentPrompts, form.segmentCount, form.prompt) : [],
    negativePrompt: form.negativePrompt,
    seed: form.seed,
    seedPolicy: form.seedPolicy,
    baseModel: form.baseModel,
    loras,
    pixelBudget: form.pixelBudget,
    motionFrames: form.motionFrames,
    motionAmplitude: form.motionAmplitude,
  };
}

function PanelHeader({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <div className="panel-header">
      {icon}
      <h2>{title}</h2>
    </div>
  );
}

function StatusPill({ label, tone }: { label: string; tone: "ok" | "danger" }) {
  return <span className={`status-pill ${tone}`}>{label}</span>;
}

function Metric({ icon, label, value }: { icon?: ReactNode; label: string; value: string }) {
  return (
    <div className="metric">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Timeline({ plan }: { plan: PlanResponse | null }) {
  if (!plan) {
    return <div className="empty-state">-</div>;
  }

  return (
    <div className="timeline">
      <div className="timeline-bars">
        {plan.segments.map((segment) => (
          <div
            key={segment.index}
            className="segment-bar"
            style={{ flexGrow: segment.outputFrames }}
            title={`Segment ${segment.index + 1}`}
          >
            <span>S{segment.index + 1}</span>
            <strong>{segment.outputFrames}f</strong>
          </div>
        ))}
      </div>
      <div className="segment-table">
        {plan.segments.map((segment) => (
          <div className="segment-row" key={segment.index}>
            <span>Segment {segment.index + 1}</span>
            <span>{segment.inputFrames} in</span>
            <span>{segment.outputFrames} out</span>
            <span>{segment.continuity.source}</span>
            <span>{segment.seed ?? "-"}</span>
            <span title={segment.prompt}>{segment.prompt}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function resizePrompts(prompts: string[], segmentCount: number, fallback: string) {
  return Array.from({ length: segmentCount }, (_, index) => prompts[index] ?? fallback);
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function formatSeconds(value: number | undefined) {
  if (value === undefined) {
    return "-";
  }
  return `${value.toFixed(2)}s`;
}
