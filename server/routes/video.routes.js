const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs-extra');
const { v4: uuidv4 } = require('uuid');
const { execSync } = require('child_process');
const ffmpegStatic = require('ffmpeg-static');
const { processVideo, getJobStatus, getActiveJobs, cleanupOldOutputVideos, getStorageUsage } = require('../controllers/processVideo');

const router = express.Router();

// Resolve ffmpeg executable from bundled static binary
function resolveFfmpegExecutable() {
  return ffmpegStatic || null;
}

// Check FFmpeg availability
function checkFFmpeg() {
  try {
    if (!ffmpegStatic) return false;
    execSync(`"${ffmpegStatic}" -version`, { stdio: 'pipe' });
    return true;
  } catch (_) {
    return false;
  }
}

// Multer configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../uploads');
    fs.ensureDir(uploadDir).then(() => cb(null, uploadDir)).catch(err => cb(err));
  },
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${file.originalname}`;
    cb(null, uniqueName);
  },
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    const allowedMimes = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
    if (allowedMimes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Only video files are allowed'));
    }
  },
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB
});

/**
 * POST /api/video/upload
 * Upload and process video
 */
router.post('/upload', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // Check if FFmpeg is available
    if (!checkFFmpeg()) {
      return res.status(503).json({ 
        error: 'FFmpeg is not installed on the server',
        status: 'error',
        message: 'Server misconfiguration: FFmpeg is required for video processing'
      });
    }

    const jobId = uuidv4();
    const videoPath = path.join(__dirname, '../uploads', req.file.filename);
    const outputFilename = `blurred_${Date.now()}.mp4`;
    const outputPath = path.join(__dirname, '../outputs', outputFilename);

    // Ensure outputs directory exists
    await fs.ensureDir(path.join(__dirname, '../outputs'));

    // Parse options from request (optimized for speed)
    const options = {
      fps: req.body.fps || 1,
      confidence: parseFloat(req.body.confidence) || 0.6,
      blurStrength: parseInt(req.body.blurStrength) || 20,
      quality: parseInt(req.body.quality) || 28
    };

    // Send immediate response with jobId
    res.json({
      status: 'processing',
      message: 'Video received and processing started',
      jobId: jobId,
      file: req.file.filename,
      outputFile: outputFilename,
      uploadedPath: `/uploads/${req.file.filename}`,
      options
    });

    // Start async processing (don't wait for completion) - pass jobId
    processVideo(videoPath, outputPath, options, jobId)
      .then((result) => {
        console.log(`✅ Video processing succeeded for ${req.file.filename}`);
        // Optionally delete original after successful processing
        fs.remove(videoPath).catch(err => console.error('Failed to delete upload:', err));
      })
      .catch((error) => {
        console.error(`❌ Video processing failed: ${error.message}`);
      });

  } catch (error) {
    res.status(500).json({ 
      error: error.message,
      status: 'error'
    });
  }
});

/**
 * GET /api/video/download/:filename
 * Download processed video
 */
router.get('/download/:filename', async (req, res) => {
  try {
    const filename = req.params.filename;
    
    // Validate filename (prevent directory traversal)
    if (filename.includes('..') || filename.includes('/')) {
      return res.status(400).json({ error: 'Invalid filename' });
    }

    const filepath = path.join(__dirname, '../outputs', filename);

    // Check if file exists
    if (!await fs.pathExists(filepath)) {
      return res.status(404).json({ error: 'File not found' });
    }

    // Send file
    res.download(filepath, filename);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/video/status/:jobId
 * Get processing status
 */
router.get('/status/:jobId', (req, res) => {
  try {
    const jobId = req.params.jobId;
    const status = getJobStatus(jobId);
    
    if (status.status === 'not_found') {
      return res.status(404).json(status);
    }

    res.json(status);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/video/jobs
 * Get all active jobs
 */
router.get('/jobs', (req, res) => {
  try {
    const jobs = getActiveJobs();
    res.json({
      total: jobs.length,
      jobs
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/video/files
 * List available output files
 */
router.get('/files', async (req, res) => {
  try {
    const outputDir = path.join(__dirname, '../outputs');
    
    if (!await fs.pathExists(outputDir)) {
      return res.json({ files: [] });
    }

    const files = await fs.readdir(outputDir);
    const fileStats = await Promise.all(
      files.map(async (file) => {
        const filepath = path.join(outputDir, file);
        const stat = await fs.stat(filepath);
        return {
          name: file,
          size: stat.size,
          sizeInMB: (stat.size / (1024 * 1024)).toFixed(2),
          created: stat.birthtime
        };
      })
    );

    res.json({
      total: fileStats.length,
      files: fileStats.sort((a, b) => b.created - a.created)
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/video/storage
 * Get current storage usage
 */
router.get('/storage', async (req, res) => {
  try {
    const usage = await getStorageUsage();
    res.json(usage);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/video/cleanup
 * Delete old output videos
 * Query params: ?hours=24 (default: delete videos older than 24 hours)
 */
router.post('/cleanup', async (req, res) => {
  try {
    const hours = parseInt(req.query.hours) || 24;
    const result = await cleanupOldOutputVideos(hours);
    res.json({
      status: 'success',
      message: `Cleanup complete`,
      ...result
    });
  } catch (error) {
    res.status(500).json({ 
      status: 'error',
      error: error.message 
    });
  }
});

/**
 * DELETE /api/video/:filename
 * Delete a specific output video file
 */
router.delete('/:filename', async (req, res) => {
  try {
    const filename = req.params.filename;
    
    // Prevent directory traversal attacks
    if (filename.includes('..') || filename.includes('/')) {
      return res.status(400).json({ error: 'Invalid filename' });
    }

    const filePath = path.join(__dirname, '../outputs', filename);
    
    // Verify file exists
    if (!await fs.pathExists(filePath)) {
      return res.status(404).json({ error: 'File not found' });
    }

    // Delete the file
    const stat = await fs.stat(filePath);
    await fs.remove(filePath);
    
    console.log(`🗑️  User deleted: ${filename} (${(stat.size / (1024 * 1024)).toFixed(2)} MB)`);
    
    res.json({
      status: 'success',
      message: `Deleted ${filename}`,
      sizeFreed: (stat.size / (1024 * 1024)).toFixed(2) + ' MB'
    });
  } catch (error) {
    res.status(500).json({ 
      status: 'error',
      error: error.message 
    });
  }
});

module.exports = router;
