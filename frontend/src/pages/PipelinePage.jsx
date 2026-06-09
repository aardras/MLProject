import { useState } from "react";

const API = "http://localhost:8000";

export default function PipelinePage({ columnRoles, onBack, onDone }) {
  const [config, setConfig] = useState({
    encoding: "onehot",
    scaling: "standard",
    missing_numeric_strategy: "mean",
    missing_categorical_strategy: "most_frequent",
    test_size: 0.2,
    stratify: false,
    random_state: 42,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [logs, setLogs] = useState([]);

  const addLog = (msg) => {
    const ts = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, { ts, msg }]);
  };

  const set = (key, val) => setConfig((prev) => ({ ...prev, [key]: val }));

  const runPipeline = async () => {
    setLoading(true);
    setError("");
    setLogs([]);
    addLog("Starting pipeline run…");

    const payload = {
      column_roles: columnRoles,
      ...config,
    };

    try {
      addLog("Running preprocessing & train/test split…");
      const res = await fetch(`${API}/api/pipeline/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || "Pipeline failed");
      }
      const result = await res.json();
      addLog(`✓ Split complete — X_train: ${result.shapes.X_train}, X_test: ${result.shapes.X_test}`);
      addLog(`✓ Features after encoding: ${result.n_features}`);
      addLog(`✓ Target: ${result.target_column}`);
      addLog("Pipeline ready!");
      setTimeout(() => onDone(result, config), 600);
    } catch (e) {
      setError(e.message);
      addLog(`✗ Error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const trainPct = Math.round((1 - config.test_size) * 100);
  const testPct = Math.round(config.test_size * 100);

  return (
    <div>
      <div className="page-title">Pipeline Configuration</div>
      <div className="page-sub">Configure preprocessing and split parameters before training.</div>

      {error && <div className="alert alert-error">⚠ {error}</div>}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        {/* Preprocessing */}
        <div className="card">
          <div className="card-title">🔧 Preprocessing</div>

          <div className="form-group" style={{ marginBottom: 16 }}>
            <label>Encoding Strategy</label>
            <select value={config.encoding} onChange={(e) => set("encoding", e.target.value)}>
              <option value="onehot">One-Hot Encoding</option>
              <option value="label">Label Encoding</option>
            </select>
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              {config.encoding === "onehot" ? "Creates binary columns for each category." : "Assigns integer codes to categories."}
            </span>
          </div>

          <div className="form-group" style={{ marginBottom: 16 }}>
            <label>Scaling Strategy</label>
            <select value={config.scaling} onChange={(e) => set("scaling", e.target.value)}>
              <option value="standard">Standard Scaler (Z-score)</option>
              <option value="minmax">MinMax Scaler (0–1)</option>
              <option value="none">None</option>
            </select>
          </div>

          <div className="form-group" style={{ marginBottom: 16 }}>
            <label>Numeric Imputation</label>
            <select value={config.missing_numeric_strategy} onChange={(e) => set("missing_numeric_strategy", e.target.value)}>
              <option value="mean">Mean</option>
              <option value="median">Median</option>
              <option value="most_frequent">Mode</option>
              <option value="constant">Constant (0)</option>
            </select>
          </div>

          <div className="form-group">
            <label>Categorical Imputation</label>
            <select value={config.missing_categorical_strategy} onChange={(e) => set("missing_categorical_strategy", e.target.value)}>
              <option value="most_frequent">Most Frequent</option>
              <option value="constant">Constant ("missing")</option>
            </select>
          </div>
        </div>

        {/* Split */}
        <div className="card">
          <div className="card-title">✂️ Train / Test Split</div>

          <div className="form-group" style={{ marginBottom: 16 }}>
            <label>Split Ratio — Train {trainPct}% / Test {testPct}%</label>
            <input
              type="range"
              min={0.1} max={0.4} step={0.05}
              value={config.test_size}
              onChange={(e) => set("test_size", parseFloat(e.target.value))}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-muted)" }}>
              <span>60/40</span><span>90/10</span>
            </div>
          </div>

          <div className="form-group" style={{ marginBottom: 16 }}>
            <label>Random Seed</label>
            <input
              type="number"
              value={config.random_state}
              onChange={(e) => set("random_state", parseInt(e.target.value) || 42)}
            />
          </div>

          <div className="form-group">
            <label>Stratified Split</label>
            <div className="toggle-row">
              <input
                type="checkbox"
                id="stratify"
                checked={config.stratify}
                onChange={(e) => set("stratify", e.target.checked)}
              />
              <label htmlFor="stratify" style={{ textTransform: "none", letterSpacing: 0, color: "var(--text)", fontWeight: 400, cursor: "pointer" }}>
                Maintain class distribution across splits
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Log panel */}
      {logs.length > 0 && (
        <div className="card">
          <div className="card-title">📟 Pipeline Log</div>
          <div className="log-panel">
            {logs.map((l, i) => (
              <div key={i} className="log-line">
                <span className="log-ts">[{l.ts}]</span>{l.msg}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="btn-row">
        <button className="btn btn-secondary" onClick={onBack} disabled={loading}>← Back</button>
        <button className="btn btn-primary" onClick={runPipeline} disabled={loading}>
          {loading ? <><span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} /> Running…</> : "▶ Run Pipeline →"}
        </button>
      </div>
    </div>
  );
}
