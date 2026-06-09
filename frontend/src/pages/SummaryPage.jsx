export default function SummaryPage({ schemaData, columnRoles, setColumnRoles, onBack, onNext }) {
  const targetCol = columnRoles.find((r) => r.role === "target")?.name;
  const featureCols = columnRoles.filter((r) => r.role === "feature");
  const ignoreCols = columnRoles.filter((r) => r.role === "ignore");

  // Class distribution from preview data
  const classDist = {};
  if (targetCol) {
    schemaData.preview.forEach((row) => {
      const val = row[targetCol];
      classDist[val] = (classDist[val] || 0) + 1;
    });
  }
  const distTotal = Object.values(classDist).reduce((a, b) => a + b, 0);

  // Null summary
  const nullSummary = schemaData.columns
    .filter((c) => schemaData.null_counts[c] > 0)
    .map((c) => ({ col: c, count: schemaData.null_counts[c] }));

  const totalNulls = Object.values(schemaData.null_counts).reduce((a, b) => a + b, 0);

  return (
    <div>
      <div className="page-title">Data Summary</div>
      <div className="page-sub">Review your dataset configuration before running the pipeline.</div>

      {/* Stats row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 12, marginBottom: 20 }}>
        <StatCard icon="📊" label="Total Rows" value={schemaData.total_rows} />
        <StatCard icon="📐" label="Total Columns" value={schemaData.columns.length} />
        <StatCard icon="⚙️" label="Feature Columns" value={featureCols.length} color="var(--accent)" />
        <StatCard icon="🎯" label="Target Column" value={targetCol || "—"} color="var(--accent2)" isText />
        <StatCard icon="🚫" label="Ignored Columns" value={ignoreCols.length} />
        <StatCard icon="❓" label="Total Nulls" value={totalNulls} color={totalNulls > 0 ? "var(--warning)" : "var(--success)"} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        {/* Column roles */}
        <div className="card">
          <div className="card-title">🏷 Column Assignments</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {columnRoles.map((r) => (
              <div key={r.name} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 0", borderBottom: "1px solid var(--border)" }}>
                <span style={{ fontWeight: 600, fontSize: 13 }}>{r.name}</span>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  {r.dtype_override && (
                    <span className={`badge ${r.dtype_override === "numeric" ? "badge-numeric" : "badge-categorical"}`}>
                      {r.dtype_override}
                    </span>
                  )}
                  <span className={`badge badge-${r.role}`}>{r.role}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Class distribution */}
          {targetCol && (
            <div className="card">
              <div className="card-title">📊 Class Distribution (preview)</div>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 12 }}>
                Based on first 10 rows — actual distribution may differ.
              </div>
              {Object.entries(classDist).map(([cls, cnt]) => (
                <div key={cls} className="dist-row">
                  <div className="dist-label">
                    <span style={{ fontWeight: 600 }}>{cls}</span>
                    <span style={{ color: "var(--text-muted)" }}>{cnt} / {distTotal}</span>
                  </div>
                  <div className="dist-bar">
                    <div className="dist-fill" style={{ width: `${(cnt / distTotal) * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Null summary */}
          <div className="card">
            <div className="card-title">❓ Null Value Summary</div>
            {nullSummary.length === 0 ? (
              <div className="alert alert-success" style={{ marginBottom: 0 }}>✓ No null values detected</div>
            ) : (
              nullSummary.map(({ col, count }) => (
                <div key={col} className="dist-row">
                  <div className="dist-label">
                    <span>{col}</span>
                    <span style={{ color: "var(--warning)" }}>{count} nulls</span>
                  </div>
                  <div className="dist-bar">
                    <div className="dist-fill" style={{ width: `${(count / schemaData.total_rows) * 100}%`, background: "var(--warning)" }} />
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="btn-row">
        <button className="btn btn-secondary" onClick={onBack}>← Back</button>
        <button className="btn btn-primary" onClick={onNext}>Next: Pipeline Config →</button>
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, color, isText }) {
  return (
    <div className="card" style={{ padding: 16, marginBottom: 0, textAlign: "center" }}>
      <div style={{ fontSize: 22 }}>{icon}</div>
      <div style={{ fontSize: isText ? 14 : 22, fontWeight: 800, color: color || "var(--text)", marginTop: 4, wordBreak: "break-all" }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", marginTop: 2 }}>{label}</div>
    </div>
  );
}
