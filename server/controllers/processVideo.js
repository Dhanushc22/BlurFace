const fs = require('fs-extra');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const { extractFrames, buildVideoFromFrames, getVideoMetadata } = require('../utils/ffmpeg');
const { initializeFaceDetector, processFramesForFaceDetection, saveFaceData } = require('../utils/faceDetection');
const { blurAllFrames } = require('../utils/blurFrame');

// Store active processing jobs
const processingJobs = new Map();

/**
 * Main video processing pipeline
 * @param {string} videoPath - Path to uploaded video
 * @param {string} outputPath - Path to save processed video
 * @param {Object} options - Processing options
 * @param {string} jobId - Optional job ID (will generate if not provided)
 * @returns {Promise<Object>} Processing result
 */
async function processVideo(videoPath, outputPath, options = {}, jobId = null) {
  if (!jobId) {
    jobId = uuidv4();
  }
  const startTime = Date.now();
  
  try {
    // Create temporary directories
    const tempDir = path.join(__dirname, '../temp', jobId);
    const frameDir = path.join(tempDir, 'frames');
    const blurredDir = path.join(tempDir, 'blurred');
    const metadataFile = path.join(tempDir, 'face_data.json');

    // Register job
    processingJobs.set(jobId, {
      status: 'starting',
      progress: 0,
      message: 'Initializing...'
    });

    console.log(`\n📹 [${jobId}] Starting video processing...`);

    // ============================================
    // STEP 1: Get video metadata
    // ============================================
    console.log(`\n[1/5] 📊 Analyzing video...`);
    updateJob(jobId, { status: 'processing', progress: 10, currentStep: 'Analyzing video...', message: 'Analyzing video metadata...' });

    const metadata = await getVideoMetadata(videoPath);
    console.log(`  ✅ Video Info:`);
    console.log(`     Duration: ${metadata.duration.toFixed(2)}s`);
    console.log(`     FPS: ${metadata.fps}`);
    console.log(`     Resolution: ${metadata.width}x${metadata.height}`);
    console.log(`     Total Frames: ${metadata.totalFrames}`);

    // ============================================
    // STEP 2: Extract frames
    // ============================================
    console.log(`\n[2/5] 🎬 Extracting frames...`);
    updateJob(jobId, { status: 'processing', progress: 25, currentStep: 'Extracting frames from video', message: 'Extracting video frames...' });

    // Extract at original FPS to preserve exact video duration
    const extractFps = metadata.fps;
    const frames = await extractFrames(videoPath, frameDir, extractFps);
    console.log(`  ✅ Extracted ${frames.length} frames at ${extractFps} FPS`);

    // ============================================
    // STEP 3: Face detection
    // ============================================
    console.log(`\n[3/5] 🤖 Detecting faces...`);
    updateJob(jobId, { status: 'processing', progress: 40, currentStep: 'Detecting faces in each frame', message: 'Detecting faces in frames...' });

    const faceModel = await initializeFaceDetector();
    const confidence = options.confidence || 0.5;
    const faceData = await processFramesForFaceDetection(frameDir, faceModel, confidence);

    // Save face data for reference
    await saveFaceData(faceData, metadataFile);
    const framesWithFaces = Object.keys(faceData).length;
    console.log(`  ✅ Detected faces in ${framesWithFaces}/${frames.length} frames`);

    // ============================================
    // STEP 4: Apply blur
    // ============================================
    console.log(`\n[4/5] 🌀 Blurring faces...`);
    updateJob(jobId, { status: 'processing', progress: 60, currentStep: 'Blurring detected faces', message: 'Applying blur to detected faces...' });

    const blurStrength = options.blurStrength || 25;
    
    // Process all frames (blur faces where detected, copy others)
    await blurAllFrames(frameDir, faceData, blurredDir, blurStrength);
    console.log(`  ✅ Blur applied`);

    // ============================================
    // STEP 5: Reconstruct video
    // ============================================
    console.log(`\n[5/5] 🎥 Reconstructing video...`);
    updateJob(jobId, { status: 'processing', progress: 80, currentStep: 'Reconstructing video', message: 'Reconstructing video from frames...' });

    const crf = options.quality || 15; // 0=best, 51=worst (lower is better, 15=high quality)
    await buildVideoFromFrames(blurredDir, outputPath, metadata.fps, crf, videoPath);
    console.log(`  ✅ Video saved: ${outputPath}`);

    // ============================================
    // STEP 6: Cleanup temporary files & input video
    // ============================================
    console.log(`\n[6/6] 🗑️  Cleaning up temporary files...`);
    updateJob(jobId, { status: 'processing', progress: 95, currentStep: 'Cleaning up...', message: 'Cleaning up...' });

    // Delete ALL temporary directories and files
    await fs.remove(tempDir);
    console.log(`  ✅ Temp directory removed: ${tempDir}`);

    // Delete uploaded input video after successful processing (save storage!)
    try {
      await fs.remove(videoPath);
      console.log(`  ✅ Input video removed: ${videoPath}`);
    } catch (err) {
      console.warn(`  ⚠️  Could not remove input video: ${err.message}`);
    }

    console.log(`  ✅ Cleanup complete`);

    // ============================================
    // Complete
    // ============================================
    const duration = (Date.now() - startTime) / 1000;
    console.log(`\n✅ [${jobId}] Processing complete in ${duration.toFixed(2)}s`);

    const result = {
      success: true,
      jobId,
      outputPath,
      outputFile: path.basename(outputPath),
      duration: duration.toFixed(2),
      statistics: {
        inputFrames: frames.length,
        framesWithFaces,
        blurStrength,
        quality: crf,
        fps: metadata.fps
      }
    };

    updateJob(jobId, { 
      status: 'completed', 
      progress: 100, 
      message: 'Processing complete!',
      outputFile: path.basename(outputPath),
      result 
    });

    return result;

  } catch (error) {
    const duration = (Date.now() - startTime) / 1000;
    console.error(`\n❌ [${jobId}] Error: ${error.message}`);
    
    // Check if it's an FFmpeg error
    const isFFmpegError = error.message.includes('FFmpeg') || 
                         error.message.includes('ffmpeg') ||
                         error.message.includes('not found') ||
                         error.message.includes('ENOENT');

    updateJob(jobId, { 
      status: 'failed', 
      progress: 0, 
      message: isFFmpegError 
        ? 'FFmpeg is not installed or not in system PATH' 
        : error.message,
      error: error.message,
      isFFmpegError
    });

    throw error;
  }
}

/**
 * Update job status
 * @param {string} jobId - Job ID
 * @param {Object} update - Status update object
 */
function updateJob(jobId, update) {
  if (processingJobs.has(jobId)) {
    const job = processingJobs.get(jobId);
    processingJobs.set(jobId, { ...job, ...update });
  }
}

/**
 * Get job status
 * @param {string} jobId - Job ID
 * @returns {Object} Job status
 */
function getJobStatus(jobId) {
  return processingJobs.get(jobId) || { status: 'not_found', error: 'Job not found' };
}

/**
 * Get all active jobs
 * @returns {Array} Array of active jobs
 */
function getActiveJobs() {
  return Array.from(processingJobs.entries()).map(([jobId, status]) => ({
    jobId,
    ...status
  }));
}

/**
 * Clean up old job records (older than 1 hour)
 */
function cleanupOldJobs() {
  const oneHourAgo = Date.now() - (60 * 60 * 1000);
  
  for (const [jobId, job] of processingJobs.entries()) {
    if (job.createdAt && job.createdAt < oneHourAgo) {
      processingJobs.delete(jobId);
    }
  }
}

/**
 * Clean up old output videos (older than specified hours)
 * @param {number} hoursOld - Delete files older than this many hours (default: 24)
 */
async function cleanupOldOutputVideos(hoursOld = 24) {
  try {
    const outputDir = path.join(__dirname, '../outputs');
    
    // Check if outputs dir exists
    if (!fs.existsSync(outputDir)) {
      return { deleted: 0, message: 'No outputs directory found' };
    }

    const files = await fs.readdir(outputDir);
    const now = Date.now();
    const cutoffTime = now - (hoursOld * 60 * 60 * 1000);
    let deleted = 0;
    let totalSize = 0;

    for (const file of files) {
      const filePath = path.join(outputDir, file);
      const stats = await fs.stat(filePath);

      if (stats.mtimeMs < cutoffTime) {
        const sizeInMB = (stats.size / (1024 * 1024)).toFixed(2);
        await fs.remove(filePath);
        deleted++;
        totalSize += stats.size;
        console.log(`🗑️  Deleted old output: ${file} (${sizeInMB} MB)`);
      }
    }

    const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2);
    console.log(`✅ Cleanup: Deleted ${deleted} files (${totalSizeMB} MB freed)`);
    
    return { deleted, totalSizeMB, message: 'Cleanup complete' };
  } catch (error) {
    console.error(`Error cleaning up old videos: ${error.message}`);
    return { deleted: 0, error: error.message };
  }
}

/**
 * Get current storage usage of outputs directory
 */
async function getStorageUsage() {
  try {
    const outputDir = path.join(__dirname, '../outputs');
    
    if (!fs.existsSync(outputDir)) {
      return { files: 0, totalSizeMB: 0 };
    }

    const files = await fs.readdir(outputDir);
    let totalSize = 0;

    for (const file of files) {
      const filePath = path.join(outputDir, file);
      const stats = await fs.stat(filePath);
      totalSize += stats.size;
    }

    const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2);
    return { 
      files: files.length, 
      totalSizeMB, 
      directory: outputDir 
    };
  } catch (error) {
    console.error(`Error getting storage usage: ${error.message}`);
    return { files: 0, totalSizeMB: 0, error: error.message };
  }
}

module.exports = {
  processVideo,
  updateJob,
  getJobStatus,
  getActiveJobs,
  cleanupOldJobs,
  cleanupOldOutputVideos,
  getStorageUsage
};
