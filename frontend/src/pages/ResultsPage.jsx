import { useState } from "react";

export default function ResultsPage({ results, onBack, onRestart }) {
  const [activeModel, setActiveModel] = useState(
    results?.results?.find((r) => !r.error)?.model_name || null
  );

  if (!results) return <div>No results available.</div>;

  const successful = results.results.filter((r) => !r.error);
  const failed = results.results.filter((r) => r.error);
  const best = results.best_model;
  const active = successful.find((r) => r.model_name === activeModel);

  const downloadResults = () => {
    const rows = [
      ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV Mean", "CV Std", "Train Time (s)"],
      ...results.comparison.map((r) => [
        r.label, r.accuracy, r.precision, r.recall, r.f1,
        r.roc_auc ?? "N/A", r.cv_mean, r.cv_std, r.training_time_s,
      ]),
    ];
    const csv = rows.map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "ml_results.csv"; a.click();
  };

  return (
    <div>
      <div className="page-title">Results Dashboard</div>
      <div className="page-sub">Training complete. Review metrics, confusion matrices, and model comparison below.</div>

      {failed.length > 0 && (
        <div className="alert alert-error">
          {failed.map((r) => `${r.model_name}: ${r.error}`).join(" | ")}
        </div>
      )}

      {best && (
        <div className="alert alert-success">
          🏆 Best model by F1: <strong>{successful.find((r) => r.model_name === best)?.label}</strong>
        </div>
      )}

      {/* Comparison summary table */}
      {results.comparison.length > 1 && (
        <div className="card">
          <div className="card-title">📊 Model Comparison</div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>ROC-AUC</th>
                  <th>CV Mean ± Std</th>
                  <th>Time (s)</th>
                </tr>
              </thead>
              <tbody>
                {results.comparison.map((r) => (
                  <tr key={r.model_name}>
                    <td>
                      <span style={{ fontWeight: 700 }}>{r.label}</span>
                      {r.model_name === best && (
                        <span style={{ marginLeft: 6, fontSize: 11, color: "var(--warning)" }}>🏆 Best</span>
                      )}
                    </td>
                    <td><MetricPill value={r.accuracy} /></td>
                    <td><MetricPill value={r.precision} /></td>
                    <td><MetricPill value={r.recall} /></td>
                    <td><MetricPill value={r.f1} highlight={r.model_name === best} /></td>
                    <td>{r.roc_auc != null ? <MetricPill value={r.roc_auc} /> : <span style={{ color: "var(--text-muted)" }}>—</span>}</td>
                    <td style={{ fontFamily: "monospace", fontSize: 12 }}>{r.cv_mean} ± {r.cv_std}</td>
                    <td style={{ color: "var(--text-muted)" }}>{r.training_time_s}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* F1 bar chart */}
          <div style={{ marginTop: 20 }}>
            <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13 }}>F1 Score Comparison</div>
            <div className="bar-chart">
              {results.comparison.map((r) => (
                <div key={r.model_name} className="bar-row">
                  <div className="bar-name">{r.label}</div>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{
                        width: `${r.f1 * 100}%`,
                        background: r.model_name === best ? "var(--accent2)" : "var(--accent)",
                      }}
                    />
                  </div>
                  <div className="bar-val">{r.f1}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Per-model detail tabs */}
      <div className="card">
        <div className="card-title">🔬 Per-Model Results</div>
        <div className="tab-bar">
          {successful.map((r) => (
            <button
              key={r.model_name}
              className={`tab-btn ${activeModel === r.model_name ? "active" : ""}`}
              onClick={() => setActiveModel(r.model_name)}
            >
              {r.label} {r.model_name === best ? "🏆" : ""}
            </button>
          ))}
        </div>

        {active && <ModelDetail result={active} />}
      </div>

      <div className="btn-row">
        <button className="btn btn-secondary" onClick={onBack}>← Back</button>
        <button className="btn btn-secondary" onClick={downloadResults}>⬇ Download CSV</button>
        <button className="btn btn-primary" onClick={onRestart}>🔄 Start Over</button>
      </div>
    </div>
  );
}

function ModelDetail({ result }) {
  return (
    <div>
      {/* Key metrics */}
      <div className="metrics-grid" style={{ marginBottom: 24 }}>
        <MetricCard label="Accuracy" value={result.accuracy} />
        <MetricCard label="Precision" value={result.precision} />
        <MetricCard label="Recall" value={result.recall} />
        <MetricCard label="F1 Score" value={result.f1} highlight />
        {result.roc_auc != null && <MetricCard label="ROC-AUC" value={result.roc_auc} />}
        <MetricCard label="CV Mean" value={result.cv_mean} />
        <MetricCard label="CV Std" value={result.cv_std} />
        <MetricCard label="Train Time" value={`${result.training_time_s}s`} isText />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        {/* Confusion matrix */}
        <div>
          <div style={{ fontWeight: 700, marginBottom: 12, fontSize: 13 }}>Confusion Matrix</div>
          <ConfusionMatrix matrix={result.confusion_matrix} labels={result.class_labels} />
        </div>

        {/* CV scores */}
        <div>
          <div style={{ fontWeight: 700, marginBottom: 12, fontSize: 13 }}>Cross-Validation Scores</div>
          <div className="bar-chart">
            {result.cv_scores.map((s, i) => (
              <div key={i} className="bar-row">
                <div className="bar-name">Fold {i + 1}</div>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${s * 100}%`, background: "var(--accent)" }} />
                </div>
                <div className="bar-val">{s.toFixed(3)}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 8, fontSize: 12, color: "var(--text-muted)" }}>
            Mean: <strong>{result.cv_mean}</strong> ± {result.cv_std}
          </div>
        </div>
      </div>

      {/* Classification report */}
      <div style={{ marginTop: 20 }}>
        <div style={{ fontWeight: 700, marginBottom: 8, fontSize: 13 }}>Classification Report</div>
        <pre style={{
          background: "#0a0c14", border: "1px solid var(--border)",
          borderRadius: 8, padding: 16, fontSize: 12,
          fontFamily: "'Fira Code', monospace",
          color: "var(--text)", overflowX: "auto",
        }}>
          {result.classification_report}
        </pre>
      </div>

      {/* Params used */}
      <details style={{ marginTop: 16 }}>
        <summary style={{ cursor: "pointer", fontWeight: 600, fontSize: 12, color: "var(--text-muted)" }}>
          Parameters used
        </summary>
        <pre style={{
          marginTop: 8, background: "#0a0c14", border: "1px solid var(--border)",
          borderRadius: 8, padding: 12, fontSize: 11, fontFamily: "monospace", color: "var(--text-muted)",
        }}>
          {JSON.stringify(result.params_used, null, 2)}
        </pre>
      </details>
    </div>
  );
}

function ConfusionMatrix({ matrix, labels }) {
  if (!matrix || matrix.length === 0) return null;
  const max = Math.max(...matrix.flat());

  return (
    <div>
      {/* Column headers */}
      <div style={{ display: "flex", gap: 3, marginBottom: 3, marginLeft: 56 }}>
        {labels.map((l) => (
          <div key={l} className="conf-label" style={{ width: 64 }}>Pred {l}</div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 3, flexDirection: "column" }}>
        {matrix.map((row, i) => (
          <div key={i} style={{ display: "flex", gap: 3, alignItems: "center" }}>
            <div className="conf-label" style={{ width: 52, textAlign: "right", paddingRight: 4 }}>
              Act {labels[i]}
            </div>
            {row.map((val, j) => {
              const intensity = max > 0 ? val / max : 0;
              const isDiag = i === j;
              const bg = isDiag
                ? `rgba(108,99,255,${0.2 + intensity * 0.7})`
                : `rgba(255,107,107,${intensity * 0.5})`;
              return (
                <div
                  key={j}
                  className="conf-cell"
                  style={{ background: bg, color: intensity > 0.6 ? "#fff" : "var(--text)" }}
                >
                  {val}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: "var(--text-muted)" }}>
        Diagonal = correct predictions (purple). Off-diagonal = errors (red).
      </div>
    </div>
  );
}

function MetricCard({ label, value, highlight, isText }) {
  return (
    <div className={`metric-card ${highlight ? "best" : ""}`}>
      <div className="metric-value" style={{ fontSize: isText ? 16 : 26 }}>
        {isText ? value : typeof value === "number" ? (value * 100).toFixed(1) + "%" : value}
      </div>
      <div className="metric-label">{label}</div>
    </div>
  );
}

function MetricPill({ value, highlight }) {
  const pct = (value * 100).toFixed(1);
  const color = value >= 0.9 ? "var(--success)" : value >= 0.7 ? "var(--accent2)" : value >= 0.5 ? "var(--warning)" : "var(--danger)";
  return (
    <span style={{ fontWeight: 700, color: highlight ? "var(--accent2)" : color }}>
      {pct}%
    </span>
  );
}
