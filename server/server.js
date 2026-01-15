const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs-extra');
require('dotenv').config();
const { execSync } = require('child_process');
const ffmpegStatic = require('ffmpeg-static');
const { cleanupOldOutputVideos, getStorageUsage } = require('./controllers/processVideo');

const app = express();
const PORT = process.env.PORT || 5000;

// Resolve ffmpeg executable from bundled static binary
function resolveFfmpegExecutable() {
  return ffmpegStatic || null;
}

// Check if FFmpeg is available
function checkFFmpegAvailable() {
  try {
    if (!ffmpegStatic) return false;
    execSync(`"${ffmpegStatic}" -version`, { stdio: 'pipe' });
    return true;
  } catch (err) {
    return false;
  }
}

const ffmpegAvailable = checkFFmpegAvailable();
if (!ffmpegAvailable) {
  console.warn('\n⚠️  WARNING: FFmpeg is NOT installed on this system!');
  console.warn('📥 Without FFmpeg, video processing will FAIL.');
  console.warn('\n📖 To install FFmpeg:');
  console.warn('   Windows: Download from https://ffmpeg.org/download.html');
  console.warn('   Or use: choco install ffmpeg (if using Chocolatey)');
  console.warn('   macOS: brew install ffmpeg');
  console.warn('   Linux: apt-get install ffmpeg\n');
} else {
  const exe = resolveFfmpegExecutable();
  console.log('✅ FFmpeg is available');
  if (exe) console.log(`🔧 Using FFmpeg at: ${exe}`);
}

// Ensure required directories exist
const requiredDirs = [
  path.join(__dirname, 'uploads'),
  path.join(__dirname, 'outputs'),
  path.join(__dirname, 'temp')
];

requiredDirs.forEach(dir => {
  fs.ensureDirSync(dir);
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ limit: '100mb', extended: true }));
app.use('/outputs', express.static(path.join(__dirname, 'outputs')));

// Routes
app.use('/api/video', require('./routes/video.routes'));
// Root info page to avoid "Cannot GET /" confusion
app.get('/', (req, res) => {
  res.send('Blurify API is running. Open http://localhost:3000 for the UI.');
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'Server is running',
    timestamp: new Date().toISOString(),
    version: '2.0.0'
  });
});

// Info endpoint
app.get('/api/info', async (req, res) => {
  const storage = await getStorageUsage();
  res.json({
    name: 'Blurify Video Processing API',
    version: '2.0.0',
    status: 'Active',
    features: [
      'Video upload',
      'Face detection (Python MediaPipe)',
      'Face blurring',
      'Video processing',
      'Async job handling',
      'Auto cleanup of temp files',
      'Storage management'
    ],
    storage: {
      files: storage.files,
      totalSizeMB: storage.totalSizeMB
    },
    endpoints: {
      upload: 'POST /api/video/upload',
      download: 'GET /api/video/download/:filename',
      status: 'GET /api/video/status/:jobId',
      jobs: 'GET /api/video/jobs',
      files: 'GET /api/video/files',
      storage: 'GET /api/video/storage',
      cleanup: 'POST /api/video/cleanup?hours=24',
      deleteFile: 'DELETE /api/video/:filename',
      health: 'GET /api/health',
      info: 'GET /api/info'
    }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('❌ Error:', err);
  res.status(err.status || 500).json({
    error: err.message || 'Internal server error',
    status: 'error'
  });
});

// Start server
app.listen(PORT, async () => {
  // Auto-cleanup old videos on startup (older than 48 hours)
  console.log('\n🧹 Running initial cleanup of old videos...');
  const cleanupResult = await cleanupOldOutputVideos(48);
  if (cleanupResult.deleted > 0) {
    console.log(`   ✅ Removed ${cleanupResult.deleted} old files (${cleanupResult.totalSizeMB} MB freed)`);
  } else {
    console.log('   ✅ No old files to clean up');
  }

  console.log(`\n${'='.repeat(60)}`);
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`${'='.repeat(60)}`);
  console.log(`\n📚 Features:`);
  console.log(`   ✅ Video Upload (Multer)`);
  console.log(`   ✅ Face Detection (Python MediaPipe)`);
  console.log(`   ✅ Face Blurring (Sharp)`);
  console.log(`   ✅ Video Processing (FFmpeg)`);
  console.log(`   ✅ Async Job Handling`);
  console.log(`   ✅ Progress Tracking`);
  console.log(`   ✅ AUTO CLEANUP (temp files deleted after processing)`);
  console.log(`\n📡 Main Endpoints:`);
  console.log(`   POST   /api/video/upload         - Upload and process video`);
  console.log(`   GET    /api/video/download/:id   - Download processed video`);
  console.log(`   GET    /api/video/status/:jobId  - Get processing status`);
  console.log(`   GET    /api/video/jobs           - List active jobs`);
  console.log(`   GET    /api/video/files          - List available files`);
  console.log(`   GET    /api/video/storage        - Get storage usage`);
  console.log(`   POST   /api/video/cleanup        - Clean old videos (48+ hours)`);
  console.log(`   DELETE /api/video/:filename      - Delete specific video`);
  console.log(`   GET    /api/health               - Health check`);
  console.log(`   GET    /api/info                 - API information`);
  console.log(`\n${'='.repeat(60)}\n`);
});
