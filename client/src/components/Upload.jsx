import React, { useState } from 'react';
import { uploadVideo } from '../api';
import './Upload.css';

export default function Upload({ onSuccess }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type.startsWith('video/')) {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please select a valid video file');
      setFile(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a video file');
      return;
    }

    setLoading(true);
    setError(null);
    setProgress(0);

    try {
      const response = await uploadVideo(file);
      console.log('Upload response:', response.data);
      setProgress(100);
      onSuccess(response.data);
      setFile(null);
      setTimeout(() => setProgress(0), 2000);
    } catch (err) {
      console.error('Upload error:', err);
      const errorMessage = err.response?.data?.error || err.message || 'Upload failed';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-section">
      <div className="upload-card">
        <div className="upload-header">
          <h2>Choose your video</h2>
          <p>Select a video file to blur faces automatically</p>
        </div>

        <div className="upload-area">
          <div className="upload-zone">
            <input
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              disabled={loading}
              id="video-input"
              className="file-input"
            />
            <label htmlFor="video-input" className="file-label">
              <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="16 16 12 12 8 16"></polyline>
                <line x1="12" y1="12" x2="12" y2="21"></line>
                <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path>
              </svg>
              <span className="upload-text">
                {file ? `✓ ${file.name}` : 'Click to select or drag video'}
              </span>
              <span className="upload-hint">MP4, WebM, MOV • Max 100MB</span>
            </label>
          </div>
        </div>

        {error && (
          <div className="error-message">
            <span className="error-icon">⚠</span>
            {error}
          </div>
        )}

        {loading && (
          <div className="progress-container">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }}></div>
            </div>
            <span className="progress-text">{progress}%</span>
          </div>
        )}

        <button
          className={`upload-button ${loading ? 'loading' : ''} ${!file || loading ? 'disabled' : ''}`}
          onClick={handleUpload}
          disabled={!file || loading}
        >
          {loading ? 'Processing...' : 'Start Processing'}
        </button>
      </div>
    </div>
  );
}
