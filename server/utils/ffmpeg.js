const ffmpeg = require('fluent-ffmpeg');
const ffmpegPath = require('ffmpeg-static');
const ffprobePath = require('ffprobe-static').path;
const path = require('path');
const fs = require('fs-extra');

// Configure fluent-ffmpeg to use bundled static binaries
ffmpeg.setFfmpegPath(ffmpegPath);
ffmpeg.setFfprobePath(ffprobePath);
console.log(`🔧 Using bundled FFmpeg at: ${ffmpegPath}`);
console.log(`🔧 Using bundled FFprobe at: ${ffprobePath}`);

/**
 * Extract frames from video file (optimized for speed)
 * @param {string} videoPath - Path to input video
 * @param {string} outputDir - Directory to save extracted frames
 * @param {number} fps - Frames per second to extract (use original video FPS)
 * @returns {Promise<Array>} Array of frame filenames
 */
async function extractFrames(videoPath, outputDir, fps) {
  return new Promise((resolve, reject) => {
    // Ensure output directory exists
    fs.ensureDirSync(outputDir);

    // Use JPEG for MUCH faster extraction (10x faster than PNG)
    // Quality 95 is visually lossless
    const framePattern = path.join(outputDir, 'frame_%04d.jpg');

    const cmd = ffmpeg(videoPath);
    
    // Optimized settings for MAXIMUM speed extraction
    const outputOpts = [
      '-threads 0',            // Use all CPU threads
      '-q:v 6',                // Faster JPEG extraction (6 is still good quality)
      '-vsync drop',           // Drop duplicate frames
      '-an'                    // No audio during extraction (faster)
    ];
    
    if (fps) {
      outputOpts.unshift(`-vf fps=${fps}`);
    }
    
    cmd.outputOptions(outputOpts);

    cmd
      .output(framePattern)
      .on('start', (commandLine) => {
        console.log('▶️ Extracting frames...');
      })
      .on('end', () => {
        const frames = fs.readdirSync(outputDir)
          .filter(f => f.startsWith('frame_') && f.endsWith('.jpg'))
          .sort();
        
        console.log(`✅ Extracted ${frames.length} frames`);
        resolve(frames);
      })
      .on('error', (err) => {
        reject(new Error(`FFmpeg error: ${err.message}`));
      })
      .run();
  });
}

/**
 * Build video from frames
 * @param {string} frameDir - Directory containing frame images
 * @param {string} outputPath - Path to save output video
 * @param {number} fps - Frames per second for output video
 * @param {number} crf - Quality (0-51, 0=best, 51=worst, default 23)
 * @returns {Promise<string>} Output video path
 */
async function buildVideoFromFrames(frameDir, outputPath, fps = 30, crf = 18, sourceVideoPath = null) {
  return new Promise((resolve, reject) => {
    // Ensure output directory exists
    const outputDir = path.dirname(outputPath);
    fs.ensureDirSync(outputDir);

    const framePattern = path.join(frameDir, 'frame_%04d.jpg');
    
    // Validate frames exist (support both jpg and png)
    const frames = fs.readdirSync(frameDir)
      .filter(f => f.startsWith('frame_') && (f.endsWith('.jpg') || f.endsWith('.png')))
      .sort();
    if (frames.length === 0) {
      return reject(new Error(`No frames found in ${frameDir}. Cannot build video.`));
    }

    // Detect format from first frame
    const frameExt = path.extname(frames[0]);
    const actualPattern = path.join(frameDir, `frame_%04d${frameExt}`);

    // For image sequences, set framerate as an input option and set output rate explicitly.
    // If sourceVideoPath is provided, pull audio from it; otherwise output will be silent.
    const cmd = ffmpeg()
      .input(actualPattern)
      .inputOptions([`-framerate ${fps}`, '-f image2', '-start_number 1']);

    if (sourceVideoPath) {
      cmd.input(sourceVideoPath);
    }

    // Get dimensions from first frame to ensure exact match
    const firstFramePath = path.join(frameDir, frames[0]);
    
    // x264 requires dimensions divisible by 2
    // Only pad if necessary (adds 1px black border if odd dimension)
    let outputOpts = [
      '-vf pad=ceil(iw/2)*2:ceil(ih/2)*2',  // Pad to even dimensions (no scaling!)
      `-r ${fps}`,
      `-crf ${crf}`,
      '-c:v libx264',
      '-preset ultrafast',    // FASTEST encoding
      '-tune zerolatency',    // Minimize latency
      '-pix_fmt yuv420p',
      '-threads 0',           // Use all CPU threads
      '-movflags +faststart', // Better streaming
      '-y'
    ];

    if (sourceVideoPath) {
      // Map video from frames (input 0) and audio from source (input 1)
      outputOpts = [
        '-map 0:v:0',           // Video from frames
        '-map 1:a:0?',          // Audio from source video (if exists)
        '-vf pad=ceil(iw/2)*2:ceil(ih/2)*2',
        `-r ${fps}`,
        `-crf ${crf}`,
        '-c:v libx264',
        '-preset ultrafast',    // FASTEST encoding
        '-tune zerolatency',    // Minimize latency
        '-pix_fmt yuv420p',
        '-threads 0',           // Use all CPU threads
        '-c:a copy',            // Copy audio without re-encoding (MUCH faster)
        '-movflags +faststart',
        '-shortest',            // Match shortest stream duration
        '-y'
      ];
    }

    cmd
      .outputOptions(outputOpts)
      .output(outputPath)
      .on('start', (commandLine) => {
        console.log('▶️ FFmpeg start:', commandLine);
        console.log(`📷 Frames to encode: ${frames.length} from ${frameDir}`);
      })
      .on('stderr', (line) => {
        // Helpful for diagnosing Windows path or pattern issues
        console.log('ffmpeg stderr:', line);
      })
      .on('end', () => {
        console.log(`✅ Video reconstruction complete: ${outputPath}`);
        resolve(outputPath);
      })
      .on('error', (err) => {
        reject(new Error(`FFmpeg error: ${err.message}`));
      })
      .run();
  });
}

/**
 * Get video metadata (duration, FPS, resolution, codec)
 * @param {string} videoPath - Path to video file
 * @returns {Promise<Object>} Video metadata
 */
async function getVideoMetadata(videoPath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(videoPath, (err, metadata) => {
      if (err) {
        return reject(new Error(`FFprobe error: ${err.message}`));
      }

      const videoStream = metadata.streams.find(s => s.codec_type === 'video');
      
      if (!videoStream) {
        return reject(new Error('No video stream found'));
      }

      const info = {
        duration: metadata.format.duration,
        fps: eval(videoStream.r_frame_rate) || 30, // Parse fps like "30/1"
        width: videoStream.width,
        height: videoStream.height,
        codec: videoStream.codec_name,
        bitrate: metadata.format.bit_rate,
        totalFrames: Math.floor(metadata.format.duration * (eval(videoStream.r_frame_rate) || 30))
      };

      resolve(info);
    });
  });
}

/**
 * Get total number of frames in video
 * @param {string} videoPath - Path to video file
 * @returns {Promise<number>} Total frame count
 */
async function getFrameCount(videoPath) {
  const metadata = await getVideoMetadata(videoPath);
  return metadata.totalFrames;
}

module.exports = {
  extractFrames,
  buildVideoFromFrames,
  getVideoMetadata,
  getFrameCount
};
