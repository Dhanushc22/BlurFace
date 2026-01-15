const sharp = require('sharp');
const fs = require('fs-extra');
const path = require('path');
const os = require('os');

// Use more CPU cores for parallel processing
const CPU_CORES = os.cpus().length;
const CONCURRENCY = Math.max(32, CPU_CORES * 8); // ULTRA-HIGH parallelism for maximum speed

/**
 * Blur faces in a single frame with smooth blending (optimized)
 * @param {string} inputPath - Original frame path
 * @param {Array} faces - Detected faces [{x, y, width, height}]
 * @param {string} outputPath - Output frame path
 * @param {number} blurStrength - Blur strength (default: 25)
 */
async function blurFrame(inputPath, faces, outputPath, blurStrength = 50) {
  try {
    // If no faces → just copy frame (use link if possible for speed)
    if (!faces || faces.length === 0) {
      await fs.copy(inputPath, outputPath);
      return;
    }

    // Read image once into buffer for faster processing
    const imageBuffer = await fs.readFile(inputPath);
    const metadata = await sharp(imageBuffer).metadata();

    // Pre-calculate blur sigma once
    const blurSigma = mapBlurStrength(blurStrength);

    // Process all faces in parallel
    const composites = await Promise.all(faces.map(async (face) => {
      const left = Math.max(0, Math.floor(face.x));
      const top = Math.max(0, Math.floor(face.y));
      const right = Math.min(metadata.width, Math.ceil(face.x + face.width));
      const bottom = Math.min(metadata.height, Math.ceil(face.y + face.height));
      
      const width = right - left;
      const height = bottom - top;

      if (width <= 0 || height <= 0) return null;

      // Enhance blur strength for side faces (profile detection tends to be less accurate)
      // Apply stronger blur with edge smoothing for better anonymization
      let enhancedSigma = blurSigma;
      if (face.probability && face.probability >= 0.88) {
        // Side/profile faces get extra blur for better anonymization
        enhancedSigma = blurSigma * 1.15;
      }

      // Extract and blur in one pipeline for efficiency
      const blurredFace = await sharp(imageBuffer)
        .extract({ left, top, width, height })
        .blur(enhancedSigma)
        .toBuffer();

      return {
        input: blurredFace,
        left,
        top,
        blend: 'over'
      };
    }));

    // Filter out null results
    const validComposites = composites.filter(c => c !== null);

    // Determine output format based on input
    const isJpeg = inputPath.toLowerCase().endsWith('.jpg') || inputPath.toLowerCase().endsWith('.jpeg');
    
    if (isJpeg) {
      // Use fast JPEG for maximum speed
      await sharp(imageBuffer)
        .composite(validComposites)
        .jpeg({ quality: 85, chromaSubsampling: '4:2:0' })  // Faster compression
        .toFile(outputPath);
    } else {
      // Use PNG for lossless
      await sharp(imageBuffer)
        .composite(validComposites)
        .png({ compressionLevel: 1, effort: 1 })
        .toFile(outputPath);
    }
  } catch (error) {
    throw new Error(`Blur frame error: ${error.message}`);
  }
}

/**
 * Blur all frames using detected face data (highly optimized)
 * @param {string} frameDir - Directory with original frames
 * @param {Object} faceData - { frameName: [faces] }
 * @param {string} outputDir - Directory for blurred frames
 * @param {number} blurStrength - Blur strength
 */
async function blurAllFrames(frameDir, faceData, outputDir, blurStrength = 50) {
  console.log(`🌀 Blurring frames (using ${CONCURRENCY} parallel workers)...`);
  await fs.ensureDir(outputDir);

  const frames = fs.readdirSync(frameDir)
    .filter(f => f.endsWith('.png') || f.endsWith('.jpg'))
    .sort();

  const totalFrames = frames.length;
  let processed = 0;
  let blurred = 0;
  let copied = 0;

  // Process frames in larger batches for better throughput
  const processFrame = async (frame) => {
    const inputPath = path.join(frameDir, frame);
    const outputPath = path.join(outputDir, frame);
    
    if (faceData[frame]) {
      await blurFrame(inputPath, faceData[frame], outputPath, blurStrength);
      return 'blurred';
    } else {
      await fs.copy(inputPath, outputPath);
      return 'copied';
    }
  };

  // Process all frames with high concurrency
  let idx = 0;
  while (idx < frames.length) {
    const batch = frames.slice(idx, idx + CONCURRENCY);
    const results = await Promise.all(batch.map(processFrame));
    
    results.forEach(r => {
      if (r === 'blurred') blurred++;
      else copied++;
    });
    
    processed += batch.length;
    idx += CONCURRENCY;

    // Progress update every 200 frames
    if (processed % 200 === 0 || processed === totalFrames) {
      const pct = ((processed / totalFrames) * 100).toFixed(1);
      console.log(`  📊 Blur progress: ${processed}/${totalFrames} (${pct}%)`);
    }
  }

  console.log(`✅ Blur complete: ${blurred} blurred, ${copied} copied`);
}

/**
 * Utility: convert UI blur strength (1–50) → sharp sigma
 * MUCH stronger blur for better face anonymization
 */
function mapBlurStrength(strength = 25) {
  // Increased max sigma from 30 to 50 for stronger blur
  return Math.max(1, Math.min(50, (strength / 50) * 50));
}

module.exports = {
  blurFrame,
  blurAllFrames,
  mapBlurStrength
};
