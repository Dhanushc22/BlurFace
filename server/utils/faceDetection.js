// Fast face detection using Python (MediaPipe or OpenCV)
const { spawn } = require('child_process');
const fs = require('fs-extra');
const path = require('path');

// Process every 3rd frame for speed (Python handles interpolation)
const DETECTION_STRIDE = 3;

async function initializeFaceDetector() {
  // Check if Python is available
  try {
    console.log('🐍 Checking Python installation...');
    
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    
    return new Promise((resolve, reject) => {
      const pythonCheck = spawn(pythonCmd, ['--version']);
      
      pythonCheck.on('close', (code) => {
        if (code === 0) {
          console.log('✅ Python is available');
          resolve(pythonCmd);
        } else {
          reject(new Error('Python is not installed or not in PATH'));
        }
      });
      
      pythonCheck.on('error', () => {
        reject(new Error('Python is not installed or not in PATH'));
      });
    });
  } catch (error) {
    throw new Error(`Failed to initialize Python: ${error.message}`);
  }
}

async function processFramesForFaceDetection(frameDir, pythonCmd, confidence = 0.5) {
  console.log('🔍 Processing frames for face detection using Python...');

  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, 'face_detector.py');
    
    // Spawn Python process
    const pythonProcess = spawn(pythonCmd, [
      scriptPath,
      frameDir,
      confidence.toString(),
      DETECTION_STRIDE.toString()
    ]);

    let stdoutData = '';
    let stderrData = '';

    // Collect stdout (JSON output)
    pythonProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });

    // Collect stderr (progress logs)
    pythonProcess.stderr.on('data', (data) => {
      const output = data.toString().trim();
      if (output) {
        console.log(output);
      }
      stderrData += output;
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python face detection failed with code ${code}\n${stderrData}`));
        return;
      }

      try {
        // Parse JSON output
        const faceData = JSON.parse(stdoutData);
        resolve(faceData);
      } catch (error) {
        reject(new Error(`Failed to parse Python output: ${error.message}\nOutput: ${stdoutData}`));
      }
    });

    pythonProcess.on('error', (error) => {
      reject(new Error(`Failed to spawn Python process: ${error.message}`));
    });
  });
}

async function saveFaceData(faceData, outputPath) {
  await fs.writeJson(outputPath, faceData, { spaces: 2 });
  console.log(`✅ Face data saved to ${outputPath}`);
}

module.exports = {
  initializeFaceDetector,
  processFramesForFaceDetection,
  saveFaceData
};
