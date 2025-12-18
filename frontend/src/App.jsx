import React, { useState } from "react";
import "./App.css";

import ModelManager from "./components/ModelManager.jsx";
import PromptsUpload from "./components/PromptsUpload.jsx";
import RunAllTestsButton from "./components/RunAllTestsButton.jsx";

function App() {

  const [promptsReady, setPromptsReady] = useState(false);

  return (
    <div className="app-root">
      <h1 className="page-title">LLM Benchmark</h1>

      <div className="connection-grid">
        <section className="connection-card">
          <h2>Models (GGUF)</h2>
          <ModelManager />
        </section>

        <section className="connection-card">
          <h2>LLM Prompts to test</h2>
          
          
          <PromptsUpload setPromptsReady={setPromptsReady} />
        </section>
      </div>

      <RunAllTestsButton promptsReady={promptsReady} />
    </div>
  );
}

export default App;