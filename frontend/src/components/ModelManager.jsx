import React, { useEffect, useState } from "react";
import api from "../api.js";
import ModelUploadForm from "./ModelUploadForm";


// This component saves the gguf files to the backend
//It also allows the user to delete models
const ModelManager = () => {
  const [models, setModels] = useState([]);
  const [uploadMessage, setUploadMessage] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  const fetchModels = async () => {
    try {
      const res = await api.get("/models");
      setModels(res.data?.models || []);
    } catch (error) {
      console.error("Error fetching models", error);
    }
  };

  const uploadModels = async (files) => {
    setUploadMessage("");
    setIsUploading(true);

    const formData = new FormData();
    // MUST match FastAPI parameter name: models: List[UploadFile] = File(...)
    files.forEach((file) => formData.append("models", file));

    try {
      const res = await api.post("/upload-models", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setUploadMessage(
        `Uploaded: ${res.data.count} model(s)`
      );
      await fetchModels();
    } catch (error) {
      console.error("Error uploading models", error);
      setUploadMessage("Error uploading models");
    } finally {
      setIsUploading(false);
    }
  };

  const deleteModel = async (filename) => {
    try {
      await api.delete(`/models/${encodeURIComponent(filename)}`);
      await fetchModels();
    } catch (error) {
      console.error("Error deleting model", error);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  return (
    <div>
      <p>
        <strong>Models Loaded:</strong>{" "}
        {models.length ? models.length : "None"}
      </p>

      <ModelUploadForm onUploadModels={uploadModels} isUploading={isUploading} />

      {uploadMessage && <p className="upload-message">{uploadMessage}</p>}

      <div className="file-upload">
        <p><strong>Uploaded Models</strong></p>

        {models.length === 0 ? (
          <p>No models uploaded yet.</p>
        ) : (
          <ul>
            {models.map((m) => (
              <li key={m.filename}>
                <span>
                  {m.filename}
                  {m.size_mb != null ? ` (${m.size_mb} MB)` : ""}
                </span>

                {/* optional delete button */}
                <button
                  type="button"
                  onClick={() => deleteModel(m.filename)}
                  style={{ marginLeft: "10px" }}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default ModelManager;
