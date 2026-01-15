import React, { useState, useEffect } from 'react';
import { getJobStatus, downloadVideo } from '../api';
import './VideoPreview.css';

export default function VideoPreview({ uploadData }) {
  const [jobStatus, setJobStatus] = useState(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    if (!uploadData || !uploadData.jobId) return;

    const pollStatus = async () => {
      try {
        const response = await getJobStatus(uploadData.jobId);
        setJobStatus(response.data);
      } catch (err) {
        console.error('Status check error:', err);
      }
    };

    // Poll every 2 seconds
    const interval = setInterval(pollStatus, 2000);
    pollStatus(); // Check immediately

    return () => clearInterval(interval);
  }, [uploadData]);

  const handleDownload = async () => {
    if (!jobStatus?.outputFile) {
      alert('Output file not yet available');
      return;
    }

    setDownloading(true);
    try {
      const response = await downloadVideo(jobStatus.outputFile);
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', jobStatus.outputFile);
      document.body.appendChild(link);
      link.click();
      link.parentElement.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download error:', err);
      alert('Failed to download video');
    } finally {
      setDownloading(false);
    }
  };

  if (!uploadData) return null;

  const isComplete = jobStatus?.status === 'completed';
  const isFailed = jobStatus?.status === 'failed';
  // Show processing until we get a final status
  const isProcessing = !isComplete && !isFailed;
  const progress = jobStatus?.progress || 0;

  return (
    <div className="preview-section">
      <div className="preview-card">
        <div className="upload-success">
          <div className="success-icon">✓</div>
          <h2>Video uploaded</h2>
          <p className="success-filename">{uploadData.file}</p>
        </div>

        {/* Progress Status - Always Show During Processing */}
        {isProcessing && (
          <div className="status-section">
            <div className="status-header">
              <h3>Processing status</h3>
              <span className="processing-badge">In progress</span>
            </div>

            <div className="progress-container">
              <div className="progress-track">
                <div className="progress-bar" style={{ width: `${progress}%` }}></div>
              </div>
              <span className="progress-percentage">{progress}%</span>
            </div>

            {jobStatus && (
              <div className="status-info">
                <p className="status-detail">
                  <span className="status-label">Job ID:</span>
                  <span className="status-value">{uploadData.jobId}</span>
                </p>
                {jobStatus.currentStep && (
                  <p className="status-detail">
                    <span className="status-label">Current step:</span>
                    <span className="status-value">{jobStatus.currentStep}</span>
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {/* Processing Steps */}
        {isProcessing && (
          <div className="processing-steps">
            <h4>What's happening</h4>
            <ul>
              <li className="step">
                <span className="step-icon">1</span>
                <span className="step-text">Extracting frames from video</span>
              </li>
              <li className="step">
                <span className="step-icon">2</span>
                <span className="step-text">Detecting faces in each frame</span>
              </li>
              <li className="step">
                <span className="step-icon">3</span>
                <span className="step-text">Blurring detected faces</span>
              </li>
              <li className="step">
                <span className="step-icon">4</span>
                <span className="step-text">Reconstructing video</span>
              </li>
            </ul>
            <p className="processing-hint">Typically takes 10-60 seconds. Processing runs locally on our server.</p>
          </div>
        )}

        {/* Download Section */}
        {isComplete && (
          <div className="completion-section">
            <div className="completion-icon">✓</div>
            <h3>Processing complete</h3>
            <p className="completion-message">Your video has been processed and is ready for download.</p>
            <button
              className="download-button"
              onClick={handleDownload}
              disabled={downloading}
            >
              {downloading ? (
                <>
                  <span className="button-icon">⏳</span>
                  <span>Downloading...</span>
                </>
              ) : (
                <>
                  <span className="button-icon">↓</span>
                  <span>Download</span>
                </>
              )}
            </button>
            <p className="output-filename">{jobStatus.outputFile}</p>
          </div>
        )}

        {/* Error Section */}
        {isFailed && (
          <div className="error-section">
            <div className="error-icon">⚠</div>
            <h3>Processing failed</h3>
            <p className="error-message">{jobStatus?.error || 'An error occurred during processing'}</p>
          </div>
        )}
      </div>
    </div>
  );
}
