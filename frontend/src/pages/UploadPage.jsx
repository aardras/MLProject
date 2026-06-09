import { useState, useRef } from "react";

const API = "http://localhost:8000";

const ROLE_OPTIONS = ["feature", "target", "ignore"];
const DTYPE_OPTIONS = ["auto", "numeric", "categorical"];

export default function UploadPage({ onDone }) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [schemaData, setSchemaData] = useState(null);
  const [roles, setRoles] = useState([]);
  const fileRef = useRef();

  const handleFile = async (file) => {
    if (!file || !file.name.endsWith(".csv")) {
      setError("Please upload a CSV file.");
      return;
    }
    setError("");
    setLoading(true);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API}/api/upload`, { method: "POST", body: form });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || "Upload failed");
      }
      const data = await res.json();
      setSchemaData(data);

      // Default: last column = target, rest = feature
      const defaultRoles = data.columns.map((col, i) => ({
        name: col,
        role: i === data.columns.length - 1 ? "target" : "feature",
        dtype_override: null,
      }));
      setRoles(defaultRoles);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const setRole = (col, role) => {
    // Only one target allowed
    setRoles((prev) =>
      prev.map((r) => {
        if (r.name === col) return { ...r, role };
        if (role === "target" && r.role === "target") return { ...r, role: "feature" };
        return r;
      })
    );
  };

  const setDtype = (col, dtype) => {
    setRoles((prev) =>
      prev.map((r) =>
        r.name === col
          ? { ...r, dtype_override: dtype === "auto" ? null : dtype }
          : r
      )
    );
  };

  const targetCount = roles.filter((r) => r.role === "target").length;

  const handleNext = () => {
    if (targetCount !== 1) {
      setError("Exactly one column must be assigned the 'target' role.");
      return;
    }
    onDone(schemaData, roles);
  };

  return (
    <div>
      <div className="page-title">Upload Dataset</div>
      <div className="page-sub">Upload a CSV file. We'll detect the schema and let you configure column roles.</div>

      {error && <div className="alert alert-error">⚠ {error}</div>}

      {!schemaData && (
        <div
          className={`dropzone ${dragging ? "over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
          onClick={() => fileRef.current.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {loading ? (
            <><div className="spinner" style={{ margin: "0 auto 12px" }} /><div>Analysing schema…</div></>
          ) : (
            <>
              <div className="dropzone-icon">📂</div>
              <div style={{ fontWeight: 700, fontSize: 16 }}>Drop your CSV here</div>
              <div className="dropzone-hint">or click to browse</div>
            </>
          )}
        </div>
      )}

      {schemaData && (
        <>
          <div className="card">
            <div className="card-title">📋 Schema Overview</div>
            <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginBottom: 16 }}>
              <Stat label="Rows" value={schemaData.total_rows} />
              <Stat label="Columns" value={schemaData.columns.length} />
              <Stat label="Total Nulls" value={Object.values(schemaData.null_counts).reduce((a, b) => a + b, 0)} />
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Inferred Type</th>
                    <th>Null Count</th>
                    <th>Override Type</th>
                    <th>Role</th>
                  </tr>
                </thead>
                <tbody>
                  {schemaData.columns.map((col) => {
                    const row = roles.find((r) => r.name === col) || {};
                    const dtype = schemaData.datatypes[col];
                    const nulls = schemaData.null_counts[col];
                    const isNumeric = dtype.includes("int") || dtype.includes("float");
                    return (
                      <tr key={col}>
                        <td style={{ fontWeight: 600 }}>{col}</td>
                        <td>
                          <span className={`badge ${isNumeric ? "badge-numeric" : "badge-categorical"}`}>
                            {dtype}
                          </span>
                        </td>
                        <td>
                          <span style={{ color: nulls > 0 ? "var(--warning)" : "var(--text-muted)" }}>
                            {nulls}
                          </span>
                        </td>
                        <td>
                          <select
                            value={row.dtype_override || "auto"}
                            onChange={(e) => setDtype(col, e.target.value)}
                            style={{ width: 130 }}
                          >
                            {DTYPE_OPTIONS.map((o) => <option key={o}>{o}</option>)}
                          </select>
                        </td>
                        <td>
                          <select
                            value={row.role || "feature"}
                            onChange={(e) => setRole(col, e.target.value)}
                            style={{ width: 120 }}
                          >
                            {ROLE_OPTIONS.map((o) => <option key={o}>{o}</option>)}
                          </select>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {targetCount !== 1 && (
              <div className="alert alert-error" style={{ marginTop: 12 }}>
                {targetCount === 0 ? "Select exactly one target column." : "Only one column can be the target."}
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-title">👁 Data Preview (first 10 rows)</div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {schemaData.columns.map((c) => <th key={c}>{c}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {schemaData.preview.map((row, i) => (
                    <tr key={i}>
                      {schemaData.columns.map((c) => <td key={c}>{row[c]}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="btn-row">
            <button className="btn btn-secondary" onClick={() => { setSchemaData(null); setRoles([]); }}>
              ↩ Re-upload
            </button>
            <button
              className="btn btn-primary"
              disabled={targetCount !== 1}
              onClick={handleNext}
            >
              Next: Data Summary →
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: 22, fontWeight: 800, color: "var(--accent)" }}>{value}</div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase" }}>{label}</div>
    </div>
  );
}
