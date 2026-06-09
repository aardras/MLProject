import { useState, useEffect } from "react";

const API = "http://localhost:8000";

const MODEL_DESCRIPTIONS = {
  logistic_regression: "Great baseline for binary classification. Fast and interpretable.",
  decision_tree: "Highly interpretable tree-based model. Prone to overfitting without depth limits.",
  random_forest: "Ensemble of decision trees. Robust and accurate with minimal tuning.",
  knn: "Instance-based learner. Simple but sensitive to feature scaling and dimensionality.",
};

export default function ModelPage({ pipelineResult, onBack, onDone }) {
  const [registry, setRegistry] = useState(null);
  const [selected, setSelected] = useState([]);
  const [configs, setConfigs] = useState({});
  const [cvK, setCvK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [logs, setLogs] = useState([]);
  const [activeTab, setActiveTab] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/models/registry`)
      .then((r) => r.json())
      .then((data) => {
        setRegistry(data);
        // Init configs with defaults
        const initConfigs = {};
        Object.entries(data).forEach(([key, entry]) => {
          initConfigs[key] = { ...entry.defaults };
        });
        setConfigs(initConfigs);
      })
      .catch(() => setError("Failed to load model registry."));
  }, []);

  const toggleModel = (key) => {
    setSelected((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    );
    if (!activeTab) setActiveTab(key);
  };

  const updateParam = (modelKey, paramKey, value) => {
    setConfigs((prev) => ({
      ...prev,
      [modelKey]: { ...prev[modelKey], [paramKey]: value },
    }));
  };

  const resetDefaults = (modelKey) => {
    setConfigs((prev) => ({
      ...prev,
      [modelKey]: { ...registry[modelKey].defaults },
    }));
  };

  const addLog = (msg) => {
    const ts = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, { ts, msg }]);
  };

  const trainModels = async () => {
    if (selected.length === 0) {
      setError("Select at least one model.");
      return;
    }
    setLoading(true);
    setError("");
    setLogs([]);
    addLog(`Training ${selected.length} model(s) with cv_k=${cvK}…`);

    const payload = {
      models: selected.map((key) => ({ model_name: key, params: configs[key] })),
      cv_k: cvK,
    };

    try {
      const res = await fetch(`${API}/api/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || "Training failed");
      }
      const data = await res.json();
      data.results.forEach((r) => {
        if (r.error) addLog(`✗ ${r.model_name}: ${r.error}`);
        else addLog(`✓ ${r.label} — F1: ${r.f1}, Accuracy: ${r.accuracy} (${r.training_time_s}s)`);
      });
      if (data.best_model) addLog(`🏆 Best model: ${registry[data.best_model]?.label}`);
      setTimeout(() => onDone(data), 500);
    } catch (e) {
      setError(e.message);
      addLog(`✗ Error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (!registry) return <div style={{ textAlign: "center", paddingTop: 60 }}><div className="spinner" style={{ margin: "0 auto" }} /></div>;

  return (
    <div>
      <div className="page-title">Model Selection</div>
      <div className="page-sub">Choose classifiers and tune their hyperparameters. All values can be adjusted from defaults.</div>

      {pipelineResult && (
        <div className="alert alert-info" style={{ marginBottom: 20 }}>
          Pipeline ready — {pipelineResult.shapes.X_train[0]} training samples, {pipelineResult.n_features} features
        </div>
      )}

      {error && <div className="alert alert-error">⚠ {error}</div>}

      {/* Model selector cards */}
      <div className="card">
        <div className="card-title">🤖 Select Models</div>
        <div className="model-selector">
          {Object.entries(registry).map(([key, entry]) => (
            <div
              key={key}
              className={`model-card ${selected.includes(key) ? "selected" : ""}`}
              onClick={() => toggleModel(key)}
            >
              <input type="checkbox" checked={selected.includes(key)} readOnly style={{ marginTop: 2 }} />
              <div className="model-card-info">
                <h4>{entry.label}</h4>
                <p>{MODEL_DESCRIPTIONS[key]}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Hyperparameter tabs */}
      {selected.length > 0 && (
        <div className="card">
          <div className="card-title">⚙️ Hyperparameter Configuration</div>
          <div className="tab-bar">
            {selected.map((key) => (
              <button
                key={key}
                className={`tab-btn ${activeTab === key ? "active" : ""}`}
                onClick={() => setActiveTab(key)}
              >
                {registry[key].label}
              </button>
            ))}
          </div>

          {selected.map((key) =>
            activeTab === key ? (
              <HyperparamPanel
                key={key}
                modelKey={key}
                entry={registry[key]}
                config={configs[key]}
                onChange={(pk, val) => updateParam(key, pk, val)}
                onReset={() => resetDefaults(key)}
              />
            ) : null
          )}
        </div>
      )}

      {/* CV config */}
      <div className="card">
        <div className="card-title">🔄 Cross-Validation</div>
        <div className="form-group" style={{ maxWidth: 240 }}>
          <label>Number of Folds (k)</label>
          <input type="number" min={2} max={20} value={cvK} onChange={(e) => setCvK(parseInt(e.target.value) || 5)} />
        </div>
      </div>

      {/* Log */}
      {logs.length > 0 && (
        <div className="card">
          <div className="card-title">📟 Training Log</div>
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
        <button className="btn btn-primary" onClick={trainModels} disabled={loading || selected.length === 0}>
          {loading
            ? <><span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} /> Training…</>
            : `🚀 Train ${selected.length} Model${selected.length !== 1 ? "s" : ""} →`}
        </button>
      </div>
    </div>
  );
}

function HyperparamPanel({ modelKey, entry, config, onChange, onReset }) {
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 16 }}>
        <button className="btn btn-secondary" style={{ fontSize: 12, padding: "6px 12px" }} onClick={onReset}>
          ↺ Reset to Defaults
        </button>
      </div>
      <div className="form-row">
        {Object.entries(entry.param_schema).map(([pk, schema]) => (
          <ParamControl key={pk} paramKey={pk} schema={schema} value={config[pk]} onChange={(v) => onChange(pk, v)} />
        ))}
      </div>
    </div>
  );
}

function ParamControl({ paramKey, schema, value, onChange }) {
  const label = paramKey.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className="form-group">
      <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {label}
        <div className="tooltip-wrap">
          <span style={{ fontSize: 11, color: "var(--text-muted)", cursor: "help" }}>ⓘ</span>
          <div className="tooltip-box">{schema.description}</div>
        </div>
      </label>

      {schema.type === "select" && (
        <select value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
          {schema.options.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      )}

      {schema.type === "bool" && (
        <div className="toggle-row">
          <input
            type="checkbox"
            id={paramKey}
            checked={!!value}
            onChange={(e) => onChange(e.target.checked)}
          />
          <label htmlFor={paramKey} style={{ textTransform: "none", letterSpacing: 0, color: "var(--text)", fontWeight: 400, cursor: "pointer" }}>
            {value ? "Enabled" : "Disabled"}
          </label>
        </div>
      )}

      {(schema.type === "int" || schema.type === "float") && (
        <>
          <input
            type="number"
            min={schema.min} max={schema.max} step={schema.step}
            value={value ?? schema.min}
            onChange={(e) => {
              const v = schema.type === "int" ? parseInt(e.target.value) : parseFloat(e.target.value);
              onChange(isNaN(v) ? schema.min : v);
            }}
          />
          <input
            type="range"
            min={schema.min} max={schema.max} step={schema.step}
            value={value ?? schema.min}
            onChange={(e) => {
              const v = schema.type === "int" ? parseInt(e.target.value) : parseFloat(e.target.value);
              onChange(v);
            }}
          />
        </>
      )}

      {schema.type === "int_nullable" && (
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <input
            type="number"
            min={schema.min} max={schema.max} step={schema.step}
            value={value === null ? "" : value}
            placeholder="None (unlimited)"
            onChange={(e) => {
              const v = parseInt(e.target.value);
              onChange(isNaN(v) ? null : v);
            }}
            style={{ flex: 1 }}
          />
          {value !== null && (
            <button className="btn btn-secondary" style={{ fontSize: 11, padding: "4px 8px", flex: "none" }} onClick={() => onChange(null)}>
              None
            </button>
          )}
        </div>
      )}
    </div>
  );
}
