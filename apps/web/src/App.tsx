import {
  Activity,
  Clock3,
  FolderOpen,
  Gauge,
  Image,
  Play,
  RefreshCw,
  Save,
  Settings,
  Sparkles,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

type SeedPolicy = "fixed" | "increment" | "none";

type FormState = {
  width: number;
  height: number;
  totalSeconds: number;
  fps: number;
  chunkSeconds: number;
  startImage: string;
  prompt: string;
  negativePrompt: string;
  seed: number;
  seedPolicy: SeedPolicy;
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
  segments: SegmentPlan[];
};

const defaultForm: FormState = {
  width: 1280,
  height: 720,
  totalSeconds: 15,
  fps: 16,
  chunkSeconds: 5,
  startImage: "inputs/start_frames_1280x720/woman_black_sand_beach.png",
  prompt: "cinematic handheld shot, natural motion, detailed environment",
  negativePrompt: "blur, low quality, flicker, warped hands, extra limbs, text",
  seed: 1234,
  seedPolicy: "fixed",
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

  const pixelMegapixels = useMemo(() => form.width * form.height / 1_000_000, [form.width, form.height]);

  async function refreshPlan(nextForm = form) {
    setIsLoading(true);
    try {
      const response = await fetch("/api/plan", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(nextForm),
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

  return (
    <main className="app-shell">
      <header className="top-bar">
        <div className="brand">
          <Sparkles size={20} aria-hidden="true" />
          <div>
            <h1>WAN/LTX Video Studio</h1>
            <span>Planner engine</span>
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
          <label className="field">
            <span>Positive</span>
            <textarea
              value={form.prompt}
              onChange={(event) => updateField("prompt", event.target.value)}
              rows={7}
            />
          </label>
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
              <span>Idle</span>
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
            <button className="selected">WAN 2.2</button>
            <button>LTX</button>
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
              <span>Seconds</span>
              <input
                type="number"
                step={0.25}
                value={form.totalSeconds}
                onChange={(event) => updateField("totalSeconds", Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Chunk</span>
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
          <div className="summary-grid">
            <Metric label="Target" value={plan ? `${plan.targetTimelineFrames} f` : "-"} />
            <Metric label="Chunk" value={plan ? `${plan.requestedChunkFrames} f` : "-"} />
            <Metric label="Extra" value={plan ? `${plan.extraOutputFrames} f` : "-"} />
            <Metric label="Status" value={isLoading ? "Planning" : "Ready"} />
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
          </div>
        ))}
      </div>
    </div>
  );
}

function formatSeconds(value: number | undefined) {
  if (value === undefined) {
    return "-";
  }
  return `${value.toFixed(2)}s`;
}
