import React, { useEffect, useState } from "react";

//This component uploads files to the frontend
const ModelUploadForm = ({ onUploadModels, isUploading }) => {
  const [files, setFiles] = useState([]);

  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files || []);
    setFiles(selected);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!files.length) return;
    await onUploadModels(files);
    setFiles([]);
    // clear the native input
    e.target.reset();
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Upload GGUF Models:
        <input
          type="file"
          accept=".gguf"
          multiple
          onChange={handleFileChange}
        />
      </label>

      <button type="submit" disabled={isUploading || files.length === 0}>
        {isUploading ? "Uploading..." : "Upload"}
      </button>
    </form>
  );
};

export default ModelUploadForm;
