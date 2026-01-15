import React, { useState, useEffect } from 'react';
import { checkHealth } from './api';
import Upload from './components/Upload';
import VideoPreview from './components/VideoPreview';
import './App.css';

function App() {
  const [uploadData, setUploadData] = useState(null);
  const [serverStatus, setServerStatus] = useState('checking...');

  useEffect(() => {
    const checkServer = async () => {
      try {
        const response = await checkHealth();
        setServerStatus(response.data.status);
      } catch (err) {
        setServerStatus('Server offline');
      }
    };
    checkServer();
  }, []);

  const handleUploadSuccess = (data) => {
    setUploadData(data);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <h1 className="logo">
              <span className="logo-icon">🎬</span>
              Blurify
            </h1>
            <p className="tagline">AI-powered face blurring for videos</p>
          </div>
          <div className="header-right">
            <span className={`status-badge ${serverStatus === 'Server running' ? 'online' : 'offline'}`}>
              {serverStatus === 'Server running' ? 'Connected' : 'Offline'}
            </span>
          </div>
        </div>
      </header>

      <main className="main-container">
        <Upload onSuccess={handleUploadSuccess} />
        {uploadData && <VideoPreview uploadData={uploadData} />}
      </main>

      <footer className="footer">
        <p>🔒 Your videos are processed locally and auto-deleted after processing</p>
      </footer>
    </div>
  );
}

export default App;
