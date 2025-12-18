// BenchmarkCharts.jsx
import React from "react";

const formatSeconds = (v) => (typeof v === "number" ? v.toFixed(4) : "N/A");
const formatEnergy = (v) => (typeof v === "number" ? v.toExponential(4) : "N/A");
const formatPct = (v) => (typeof v === "number" ? (v * 100).toFixed(2) + "%" : "N/A");


//This component is used to display charts and data recieved from the backend to send to the frontend
const BenchmarkCharts = ({ results }) => {
  // Expecting results = backend payload (NOT wrapped), like:
  // { models_tested, results: [...], plots: {...}, summary: {...} }
  if (!results || !results.results) {
    console.error("Missing required benchmark fields:", results);
    return null;
  }

  const modelRuns = results.results; // array of per-model benchmark objects
  const plots = results.plots || {};
  const summary = results.summary || {};
  const perModelPlots = results.per_model_plots || {};
  


  return (
    <div className="charts">
        <h2>LLM Benchmark Charts</h2>
       <div className="section"> 
        <h3>Energy per prompt</h3>
        {/* Top-level comparison plots (like the bar+line row in Charts) */}
        
        <div className="chart-row">
            <div className="chart-card">
            <h4>Energy / Prompt Token (kWh/token)</h4>
            {plots.energy_per_prompt_token_bar ? (
                <img
                src={`data:image/png;base64,${plots.energy_per_prompt_token_bar}`}
                alt="Energy per prompt token by model"
                />
            ) : (
                <p>No energy-per-prompt-token plot available.</p>
            )}
            </div>

            <div className="chart-card">
                <h4>Energy / Generated Token (kWh/token)</h4>
                {plots.energy_per_generated_token_bar ? (
                    <img
                    src={`data:image/png;base64,${plots.energy_per_generated_token_bar}`}
                    alt="Energy per generated token by model"
                    />
                ) : (
                    <p>No energy-per-generated-token plot available.</p>
                )}
                </div>
            </div>
      </div>

      <div className="section"> 
      <h3>Times per prompt</h3>
      <div className="chart-row">
        <div className="chart-card">
            <h4>Time / Prompt Token (s/token)</h4>
            {plots.time_per_prompt_token_bar ? (
            <img
                src={`data:image/png;base64,${plots.time_per_prompt_token_bar}`}
                alt="Time per prompt token by model"
            />
            ) : (
            <p>No time-per-prompt-token plot available.</p>
            )}
        </div>

        <div className="chart-card">
            <h4>Time / Generated Token (s/token)</h4>
            {plots.time_per_generated_token_bar ? (
            <img
                src={`data:image/png;base64,${plots.time_per_generated_token_bar}`}
                alt="Time per generated token by model"
            />
            ) : (
            <p>No time-per-generated-token plot available.</p>
            )}
        </div>
        </div>
        </div>


      <div className="section">
      <h3>Totals</h3>
      <div className="chart-row">
        <div className="chart-card">
          <h4>Total Time (s)</h4>
          {plots.total_time_bar ? (
            <img
              src={`data:image/png;base64,${plots.total_time_bar}`}
              alt="Total time by model"
            />
          ) : (
            <p>No total-time plot available.</p>
          )}
        </div>
        <div className="chart-card">
            <h4>Total Energy (kWh)</h4>
            {plots.total_energy_bar ? (
            <img
                src={`data:image/png;base64,${plots.total_energy_bar}`}
                alt="Total energy by model"
            />
            ) : (
            <p>No total-energy plot available.</p>
            )}
        </div>
        </div>
      </div>

      
        
      <div className="section"> 
      <h3>Other</h3>
        <div className="chart-row">
        <div className="chart-card">
            <h4>Accuracy by Model</h4>
            {plots.accuracy_bar ? (
                <img
                src={`data:image/png;base64,${plots.accuracy_bar}`}
                alt="Accuracy by model"
                />
            ) : (
                <p>No accuracy plot available.</p>
            )}
            </div>
            </div>
        </div>


      {/* --- Per-model sections (sections -> Model sections) */}
      <div className="charts-grid">
        {modelRuns.map((m) => {
          const modelName = m.model || "Unknown model";
          const correctnessPlot = perModelPlots.correctness_by_prompt?.[modelName];
          const timeByPromptPlot = perModelPlots.time_by_prompt?.[modelName];
          
          // Aggregate totals from per-prompt rows (works even if summary is missing)
          const perPrompt = Array.isArray(m.results) ? m.results : [];

          const totalTime = perPrompt.reduce(
            (acc, r) => acc + (typeof r.total_time === "number" ? r.total_time : 0),
            0
          );

          const totalEnergy = perPrompt.reduce(
            (acc, r) => acc + (typeof r.energy_total_kwh === "number" ? r.energy_total_kwh : 0),
            0
          );

          const promptTokens = perPrompt.reduce(
            (acc, r) => acc + (typeof r.prompt_tokens === "number" ? r.prompt_tokens : 0),
            0
          );

          const genTokens = perPrompt.reduce(
            (acc, r) => acc + (typeof r.generated_tokens === "number" ? r.generated_tokens : 0),
            0
          );

          const prefillEnergy = perPrompt.reduce(
            (acc, r) => acc + (typeof r.energy_prefill_kwh === "number" ? r.energy_prefill_kwh : 0),
            0
          );

          const decodeEnergy = perPrompt.reduce(
            (acc, r) => acc + (typeof r.energy_decode_kwh === "number" ? r.energy_decode_kwh : 0),
            0
          );

          const prefillTime = perPrompt.reduce(
            (acc, r) => acc + (typeof r.prefill_time === "number" ? r.prefill_time : 0),
            0
          );

          const decodeTime = perPrompt.reduce(
            (acc, r) => acc + (typeof r.decode_time === "number" ? r.decode_time : 0),
            0
          );

          const energyPerPromptTok = promptTokens ? prefillEnergy / promptTokens : 0;
          const energyPerGenTok = genTokens ? decodeEnergy / genTokens : 0;
          const timePerPromptTok = promptTokens ? prefillTime / promptTokens : 0;
          const timePerGenTok = genTokens ? decodeTime / genTokens : 0;
        
          
          return (
            <section key={modelName} className="section">
              <h3>{modelName}</h3>

              <div className="chart-row">
                <div className="chart-card">
                <h4>Correct / Incorrect by Prompt</h4>
                    {correctnessPlot ? (
                        <img
                        src={`data:image/png;base64,${correctnessPlot}`}
                        alt={`${modelName} correctness by prompt`}
                        />
                    ) : (
                        <p>No correctness-by-prompt plot available.</p>
                    )}
                </div>

                <div className="chart-card">
                    <h4>Total Time by Prompt (seconds)</h4>
                    {timeByPromptPlot ? (
                        <img
                        src={`data:image/png;base64,${timeByPromptPlot}`}
                        alt={`${modelName} time by prompt`}
                        />
                    ) : (
                        <p>No time-by-prompt plot available.</p>
                    )}
                    </div>

                </div>


              <div className="summary">
                <div className="stat-group">
                    <h4>Prompt tokens</h4>
                    <p><strong>Total prompt tokens:</strong> {promptTokens}</p>
                    <p><strong>Total prefill time:</strong> {formatSeconds(prefillTime)} s</p>
                    <p><strong>Time / prompt token:</strong> {formatSeconds(timePerPromptTok)} s/token</p>
                    <p><strong>Energy / prompt token:</strong> {formatEnergy(energyPerPromptTok)} kWh/token</p>
                </div>

                <div className="stat-group">
                    <h4>Generated tokens</h4>
                    <p><strong>Total generated tokens:</strong> {genTokens}</p>
                    <p><strong>Total decode time:</strong> {formatSeconds(decodeTime)} s</p>
                    <p><strong>Time / generated token:</strong> {formatSeconds(timePerGenTok)} s/token</p>
                    <p><strong>Energy / generated token:</strong> {formatEnergy(energyPerGenTok)} kWh/token</p>
                </div>

                <div className="stat-group">
                    <h4>Other</h4>
                    <p><strong>Accuracy:</strong> {formatPct(m.accuracy)}</p>
                    <p><strong>Total time (all prompts):</strong> {formatSeconds(totalTime)} s</p>
                    <p><strong>Total energy (all prompts):</strong> {formatEnergy(totalEnergy)} kWh</p>
                </div>
                </div>
            </section>
          );
        })}
      </div>
    </div>
  );
};

export default BenchmarkCharts;
