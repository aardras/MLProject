import { useState } from "react";
import UploadPage from "./pages/UploadPage";
import SummaryPage from "./pages/SummaryPage";
import PipelinePage from "./pages/PipelinePage";
import ModelPage from "./pages/ModelPage";
import ResultsPage from "./pages/ResultsPage";
import "./App.css";

const STEPS = [
  { id: 0, label: "Upload & Schema" },
  { id: 1, label: "Data Summary" },
  { id: 2, label: "Pipeline Config" },
  { id: 3, label: "Model Selection" },
  { id: 4, label: "Results" },
];

export default function App() {
  const [step, setStep] = useState(0);
  const [schemaData, setSchemaData] = useState(null);       // from /api/upload
  const [columnRoles, setColumnRoles] = useState([]);       // user assignments
  const [pipelineResult, setPipelineResult] = useState(null); // from /api/pipeline/run
  const [pipelineConfig, setPipelineConfig] = useState(null);
  const [trainingResults, setTrainingResults] = useState(null); // from /api/train

  const goTo = (s) => setStep(s);

  return (
    <div className="app">
      {/* Stepper */}
      <header className="app-header">
        <div className="brand">⚗️ ML Pipeline</div>
        <nav className="stepper">
          {STEPS.map((s) => (
            <div
              key={s.id}
              className={`step-item ${step === s.id ? "active" : ""} ${step > s.id ? "done" : ""}`}
            >
              <div className="step-circle">{step > s.id ? "✓" : s.id + 1}</div>
              <span className="step-label">{s.label}</span>
              {s.id < STEPS.length - 1 && <div className="step-line" />}
            </div>
          ))}
        </nav>
      </header>

      <main className="app-main">
        {step === 0 && (
          <UploadPage
            onDone={(data, roles) => {
              setSchemaData(data);
              setColumnRoles(roles);
              goTo(1);
            }}
          />
        )}
        {step === 1 && (
          <SummaryPage
            schemaData={schemaData}
            columnRoles={columnRoles}
            setColumnRoles={setColumnRoles}
            onBack={() => goTo(0)}
            onNext={() => goTo(2)}
          />
        )}
        {step === 2 && (
          <PipelinePage
            columnRoles={columnRoles}
            onBack={() => goTo(1)}
            onDone={(result, config) => {
              setPipelineResult(result);
              setPipelineConfig(config);
              goTo(3);
            }}
          />
        )}
        {step === 3 && (
          <ModelPage
            pipelineResult={pipelineResult}
            onBack={() => goTo(2)}
            onDone={(results) => {
              setTrainingResults(results);
              goTo(4);
            }}
          />
        )}
        {step === 4 && (
          <ResultsPage
            results={trainingResults}
            onBack={() => goTo(3)}
            onRestart={() => {
              setSchemaData(null);
              setColumnRoles([]);
              setPipelineResult(null);
              setPipelineConfig(null);
              setTrainingResults(null);
              goTo(0);
            }}
          />
        )}
      </main>
    </div>
  );
}
