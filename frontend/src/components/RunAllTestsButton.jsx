import React, { useState, useEffect } from "react";
import api from "../api.js";
import BenchmarkCharts from "./BenchmarkCharts.jsx";

// This class helps with applying a run buton to the front end and interacts with the backend to run all the files
const RunAllTestsButton = ({ promptsReady }) => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [progress, setProgress] = useState(null);

  useEffect(() => {
    if (!loading || !promptsReady) return; 

    const interval = setInterval(async () => {
      try {
        const res = await api.get("/benchmark_progress");
        setProgress(res.data);
      } catch (err) {
        console.error("Progress polling failed", err);
      }
    }, 2000); //2000 is once per 2 seconds

    return () => clearInterval(interval);
  }, [loading, promptsReady]);

  const handleRun = async () => {
    if (!promptsReady) return; 
    setLoading(true);
    setErrorMsg("");
    setResults(null);
    setProgress({ current: 0, total: 0, status: "running" });
    try {
      const res = await api.post("/run_all_tests");
      setResults(res.data);

    } catch (err) {
      console.error("Error running benchmark", err);
      const backendMsg =
    err?.response?.data?.detail ||
    err?.message ||
    "Failed to run benchmark. Check backend console.";
    setErrorMsg(backendMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="run-tests-container">
      <button 
        className="run-tests-button"
        onClick={handleRun}
        disabled={loading|| !promptsReady}
      >
        {loading ? "Running..." : "Run Benchmark"}
      </button>
      {!promptsReady && (
        <p style={{ opacity: 0.7 }}>
          Upload prompts before running benchmarks.
        </p>
      )}
      {loading && progress && progress.total > 0 && (
        <div className="progress-container">
          <div className="progress-label">
            {progress.current} / {progress.total} prompts completed
          </div>
          <div className="progress-bar">
            <div
              className="progress-bar-fill"
              style={{
                width: `${(progress.current / progress.total) * 100}%`,
              }}
            />
          </div>
        </div>
      )}

      {errorMsg && <p style={{ color: "red" }}>{errorMsg}</p>}

      {results && <BenchmarkCharts results={results} />}
    </div>
  );
};

export default RunAllTestsButton;
