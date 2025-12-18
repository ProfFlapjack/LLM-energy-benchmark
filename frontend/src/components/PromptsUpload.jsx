import React, { useEffect, useState } from 'react';
import api from "../api.js";

// THis component gets the prompt csv file and sends them to the backend
const PromptsUpload = ({ setPromptsReady }) => {
    const [promptFile, setPromptFile] = useState(null);
    const [uploadMessage, setUploadMessage] = useState("");


// gets the files
// handle file selection
    const handleFileChange = (e) => {
        setPromptFile(e.target.files[0] || null);
        setUploadMessage("");
    };

  // send file to /uploadrdmsfile/
    const handleUploadFile = async () => {
        if (!promptFile) return;

        const formData = new FormData();
        // field name MUST match `file_upload` in FastAPI
        formData.append("file_upload", promptFile);

        try {
        const res = await api.post("/uploadpromptfile/", formData, {
            headers: { "Content-Type": "multipart/form-data" },
        });
        setUploadMessage(`Uploaded: ${res.data.count} prompts loaded`);
        if (res.data.count > 0) {
          setPromptsReady(true);   // tell parent prompts are ready
        } else {
          setPromptsReady(false);
        }
        } catch (error) {
          setPromptsReady(false);
        console.error("Error uploading prompt file", error);
        setUploadMessage("Error uploading file");
        }
    };



    return (
        <div>
          
    
          {/* File upload block */}
          <div className="file-upload">
            <label>
              Upload Prompt CSV:
              <input type="file" accept=".csv" onChange={handleFileChange} />
            </label>
            <button type="button" onClick={handleUploadFile}>
              Upload Prompt File
            </button>
            {uploadMessage && <p className="upload-message">{uploadMessage}</p>}
          </div>
        </div>
    );
};

export default PromptsUpload;
