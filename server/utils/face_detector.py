#!/usr/bin/env python3
"""
Fast face detection using MediaPipe or OpenCV
Processes video frames and outputs face coordinates as JSON
"""

import cv2
import json
import os
import sys
from pathlib import Path
import numpy as np

# Try to import MediaPipe (faster and more accurate)
USE_MEDIAPIPE = False
mp = None
try:
    import mediapipe.python.solutions as mp_solutions
    mp = mp_solutions
    USE_MEDIAPIPE = True
    print("✅ MediaPipe loaded successfully", file=sys.stderr)
except (ImportError, AttributeError) as e:
    print(f"⚠️  MediaPipe not available ({e}), falling back to OpenCV Haar Cascades", file=sys.stderr)


class FaceDetector:
    def __init__(self, confidence=0.5):
        self.confidence = confidence
        self.use_mediapipe = USE_MEDIAPIPE
        
        if USE_MEDIAPIPE and mp:
            # MediaPipe Face Detection - high accuracy mode
            try:
                # Model 1: Full-range (within 5 meters) - better for all distances
                # Using higher confidence to reduce false positives
                self.detector = mp.face_detection.FaceDetection(
                    model_selection=1,  # Full-range model for ALL distances
                    min_detection_confidence=0.55  # Balanced - catches side faces too
                )
                print("✅ Using MediaPipe Face Detection (full-range, balanced)", file=sys.stderr)
            except Exception as e:
                print(f"⚠️  MediaPipe initialization failed ({e}), falling back to OpenCV", file=sys.stderr)
                self.use_mediapipe = False
                self._init_opencv()
        else:
            self._init_opencv()
    
    def _init_opencv(self):
        """Initialize OpenCV Haar Cascade detectors (frontal + profile + alt)"""
        try:
            # Load multiple frontal face detectors for better coverage
            frontal_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            frontal_alt_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            self.frontal_detector = cv2.CascadeClassifier(frontal_path)
            self.frontal_alt_detector = cv2.CascadeClassifier(frontal_alt_path)
            
            # Load profile face detector for side angles
            profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            self.profile_detector = cv2.CascadeClassifier(profile_path)
            
            if self.frontal_detector.empty():
                raise Exception("Failed to load frontal Haar Cascade classifier")
            
            print("✅ Using OpenCV Haar Cascade (frontal + frontal_alt + profile)", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to initialize OpenCV: {e}", file=sys.stderr)
            sys.exit(1)
    
    def detect_faces_mediapipe(self, image):
        """Detect faces using MediaPipe - accurate single pass"""
        h, w = image.shape[:2]
        
        # Single pass detection
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(image_rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                # Balanced confidence for both front and side faces
                if detection.score[0] < 0.55:
                    continue
                
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Skip detections that are too small (likely false positives)
                if width < 35 or height < 35:
                    continue
                
                # Validate aspect ratio (faces are roughly square, not too wide/tall)
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue
                
                # Add small margins for smooth tracking (reduced from 20% to 10%)
                margin_x = int(width * 0.10)
                margin_y = int(height * 0.10)
                
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                width = min(w - x, width + margin_x * 2)
                height = min(h - y, height + margin_y * 2)
                
                faces.append({
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'probability': float(detection.score[0])
                })
        
        return faces
    
    def detect_faces_opencv(self, image):
        """Detect faces using OpenCV Haar Cascade with strict settings to avoid false positives"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Enhance contrast for better detection
        gray = cv2.equalizeHist(gray)
        h, w = image.shape[:2]
        
        all_faces = []
        
        # Detect frontal faces with default detector - STRICT settings
        frontal_rects = self.frontal_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,     # Standard scale for accuracy
            minNeighbors=8,      # Very high = minimal false positives
            minSize=(50, 50),    # Larger minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, fw, fh) in frontal_rects:
            # Validate aspect ratio (reject non-face shapes)
            aspect_ratio = fw / fh if fh > 0 else 0
            if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                continue
            
            # Small margins (10%) for smooth tracking
            margin_x = int(fw * 0.10)
            margin_y = int(fh * 0.10)
            
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            fw = min(w - x, fw + margin_x * 2)
            fh = min(h - y, fh + margin_y * 2)
            
            all_faces.append({
                'x': int(x),
                'y': int(y),
                'width': int(fw),
                'height': int(fh),
                'probability': 0.9
            })
        
        # Detect with alt detector (catches different face angles) - STRICT settings
        if hasattr(self, 'frontal_alt_detector') and not self.frontal_alt_detector.empty():
            alt_rects = self.frontal_alt_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=7,      # Very strict to avoid false positives
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, fw, fh) in alt_rects:
                # Validate aspect ratio
                aspect_ratio = fw / fh if fh > 0 else 0
                if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                    continue
                
                margin_x = int(fw * 0.10)
                margin_y = int(fh * 0.10)
                
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                fw = min(w - x, fw + margin_x * 2)
                fh = min(h - y, fh + margin_y * 2)
                
                all_faces.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(fw),
                    'height': int(fh),
                    'probability': 0.88
                })
        
        # Detect profile faces (left-facing) - MULTI-SCALE for better accuracy
        if hasattr(self, 'profile_detector') and not self.profile_detector.empty():
            # First pass: detect larger/closer profiles
            profile_rects_close = self.profile_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,       # Finer scale for close faces
                minNeighbors=4,        # Balanced for profiles
                minSize=(60, 60),      # Larger faces
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Second pass: detect smaller/distant profiles
            profile_rects_far = self.profile_detector.detectMultiScale(
                gray,
                scaleFactor=1.05,      # Very fine scale for distant faces
                minNeighbors=5,        # Stricter for small detections
                minSize=(35, 35),      # Smaller faces
                maxSize=(80, 80),      # Cap to avoid overlap with first pass
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Combine both passes
            all_profile_rects = list(profile_rects_close) + list(profile_rects_far)
            
            for (x, y, fw, fh) in all_profile_rects:
                # Validate aspect ratio for profile faces (slightly wider range for profiles)
                aspect_ratio = fw / fh if fh > 0 else 0
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue
                
                # Profile faces need slightly larger margins to cover the full side
                margin_x = int(fw * 0.20)
                margin_y = int(fh * 0.15)
                
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                fw = min(w - x, fw + margin_x * 2)
                fh = min(h - y, fh + margin_y * 2)
                
                all_faces.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(fw),
                    'height': int(fh),
                    'probability': 0.87  # Good confidence for profiles
                })
            
            # Detect right-facing profiles by flipping image
            flipped = cv2.flip(gray, 1)
            
            # First pass: close right-facing profiles
            profile_rects_flip_close = self.profile_detector.detectMultiScale(
                flipped,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Second pass: distant right-facing profiles
            profile_rects_flip_far = self.profile_detector.detectMultiScale(
                flipped,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(35, 35),
                maxSize=(80, 80),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Combine both passes for flipped
            all_profile_flip = list(profile_rects_flip_close) + list(profile_rects_flip_far)
            
            for (x, y, fw, fh) in all_profile_flip:
                # Mirror x coordinate back
                x = w - x - fw
                
                # Validate aspect ratio
                aspect_ratio = fw / fh if fh > 0 else 0
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue
                
                margin_x = int(fw * 0.20)
                margin_y = int(fh * 0.15)
                
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                fw = min(w - x, fw + margin_x * 2)
                fh = min(h - y, fh + margin_y * 2)
                
                all_faces.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(fw),
                    'height': int(fh),
                    'probability': 0.87
                })
        
        # Remove duplicate/overlapping detections
        return self._remove_duplicates(all_faces)
    
    def _remove_duplicates(self, faces):
        """Remove overlapping face detections"""
        if len(faces) <= 1:
            return faces
        
        # Sort by probability (higher first)
        faces = sorted(faces, key=lambda f: f.get('probability', 0), reverse=True)
        
        unique = []
        for face in faces:
            is_duplicate = False
            for existing in unique:
                # Check overlap (IoU)
                x1 = max(face['x'], existing['x'])
                y1 = max(face['y'], existing['y'])
                x2 = min(face['x'] + face['width'], existing['x'] + existing['width'])
                y2 = min(face['y'] + face['height'], existing['y'] + existing['height'])
                
                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = face['width'] * face['height']
                    area2 = existing['width'] * existing['height']
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > 0.3:  # 30% overlap = duplicate (stricter)
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique.append(face)
        
        return unique
    
    def detect_faces(self, image):
        """Detect faces using the available method"""
        if self.use_mediapipe:
            return self.detect_faces_mediapipe(image)
        else:
            return self.detect_faces_opencv(image)
    
    def __del__(self):
        if self.use_mediapipe and hasattr(self, 'detector'):
            try:
                self.detector.close()
            except:
                pass


def process_frames(frame_dir, confidence=0.5, stride=1):
    """
    Process frames FAST while maintaining accuracy for all face types
    
    Args:
        frame_dir: Directory containing frame images
        confidence: Minimum detection confidence (0.0 to 1.0)
        stride: Process every Nth frame (3 = fast with interpolation)
    
    Returns:
        Dictionary mapping frame filenames to detected faces
    """
    # Balanced confidence for front + side faces
    detector = FaceDetector(0.55)
    
    # Get all frame files (support both PNG and JPG)
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
    total_frames = len(frame_files)
    
    # Use stride=1 to detect EVERY frame (ensures continuous blur for moving faces)
    effective_stride = 1
    
    print(f"🔍 Processing {total_frames} frames (stride={effective_stride})...", file=sys.stderr)
    
    # First pass: detect faces on ALL frames for maximum coverage
    raw_detections = {}
    processed = 0
    
    for idx, frame_file in enumerate(frame_files):
        # Only process every Nth frame for speed
        if idx % effective_stride != 0:
            continue
            
        frame_path = os.path.join(frame_dir, frame_file)
        
        # Read image
        image = cv2.imread(frame_path)
        if image is None:
            continue
        
        # Detect faces on this frame
        faces = detector.detect_faces(image)
        
        if faces and len(faces) > 0:
            raw_detections[idx] = (frame_file, faces)
        
        processed += 1
        if processed % 30 == 0:
            print(f"  📊 Progress: {processed} frames scanned ({len(raw_detections)} with faces)", file=sys.stderr)
    
    print(f"  📊 Scan complete: {processed} frames scanned, {len(raw_detections)} with faces", file=sys.stderr)
    
    # ENHANCED: Apply temporal smoothing to fill gaps and stabilize side face blur
    # This prevents blinking by maintaining face presence across brief detection gaps
    face_data = apply_temporal_smoothing(frame_files, raw_detections, total_frames)
    
    print(f"✅ Detection complete! Found {len(face_data)} frames with faces", file=sys.stderr)
    return face_data


def apply_temporal_smoothing(frame_files, raw_detections, total_frames):
    """
    Apply temporal smoothing with VELOCITY-BASED TRACKING.
    Tracks face movement direction to predict position during brief detection gaps.
    Only blurs verified face positions - never extends to unverified areas.
    """
    face_data = {}
    
    # Tracking parameters - conservative to avoid false positives
    PERSISTENCE_FRAMES = 5   # Maintain blur for 5 frames without detection
    MATCH_DISTANCE = 250     # Stricter matching - faces must be close to be same track
    VELOCITY_DAMPING = 0.7   # How much to reduce velocity prediction each frame
    
    # Track faces with velocity for smooth following
    # Each track: {x, y, width, height, vx, vy, last_seen, frames_since_seen, confidence_streak}
    tracked_faces = []
    
    for idx in range(total_frames):
        frame_file = frame_files[idx]
        current_faces = []
        
        # Get detected faces for this frame (if any)
        detected_faces = []
        if idx in raw_detections:
            detected_faces = raw_detections[idx][1]
        
        # Update tracked faces with new detections
        used_detections = set()
        
        for track in tracked_faces:
            # Predict where face should be based on velocity
            predicted_x = track['x'] + track.get('vx', 0)
            predicted_y = track['y'] + track.get('vy', 0)
            predicted_cx = predicted_x + track['width'] / 2
            predicted_cy = predicted_y + track['height'] / 2
            
            # Find best matching detection near predicted position
            best_match = None
            best_dist = float('inf')
            best_det_idx = -1
            
            for det_idx, det in enumerate(detected_faces):
                if det_idx in used_detections:
                    continue
                    
                det_cx = det['x'] + det['width'] / 2
                det_cy = det['y'] + det['height'] / 2
                
                # Distance from PREDICTED position (better for tracking moving faces)
                dist = ((predicted_cx - det_cx) ** 2 + (predicted_cy - det_cy) ** 2) ** 0.5
                
                if dist < best_dist and dist < MATCH_DISTANCE:
                    best_dist = dist
                    best_match = det
                    best_det_idx = det_idx
            
            if best_match:
                # Calculate velocity (movement from last position)
                old_cx = track['x'] + track['width'] / 2
                old_cy = track['y'] + track['height'] / 2
                new_cx = best_match['x'] + best_match['width'] / 2
                new_cy = best_match['y'] + best_match['height'] / 2
                
                # Update velocity with smoothing
                new_vx = new_cx - old_cx
                new_vy = new_cy - old_cy
                track['vx'] = track.get('vx', 0) * 0.3 + new_vx * 0.7  # Smooth velocity
                track['vy'] = track.get('vy', 0) * 0.3 + new_vy * 0.7
                
                # Update position directly to detected position (accurate tracking)
                track['x'] = best_match['x']
                track['y'] = best_match['y']
                track['width'] = best_match['width']
                track['height'] = best_match['height']
                track['last_seen'] = idx
                track['frames_since_seen'] = 0
                track['confidence_streak'] = track.get('confidence_streak', 0) + 1
                track['probability'] = best_match.get('probability', 0.9)
                used_detections.add(best_det_idx)
            else:
                # No match found - only use velocity prediction if we have a good track
                track['frames_since_seen'] = idx - track['last_seen']
                
                # Only predict position if we had a stable track (seen for multiple frames)
                if track.get('confidence_streak', 0) >= 3 and track['frames_since_seen'] <= 2:
                    # Apply velocity prediction with damping
                    track['x'] = int(track['x'] + track.get('vx', 0) * VELOCITY_DAMPING)
                    track['y'] = int(track['y'] + track.get('vy', 0) * VELOCITY_DAMPING)
                    track['vx'] = track.get('vx', 0) * VELOCITY_DAMPING
                    track['vy'] = track.get('vy', 0) * VELOCITY_DAMPING
                else:
                    # Don't predict if track is not stable - this prevents jumping to wrong areas
                    track['vx'] = 0
                    track['vy'] = 0
        
        # Add new tracked faces for unmatched detections
        for det_idx, det in enumerate(detected_faces):
            if det_idx not in used_detections:
                tracked_faces.append({
                    'x': det['x'],
                    'y': det['y'],
                    'width': det['width'],
                    'height': det['height'],
                    'vx': 0,
                    'vy': 0,
                    'last_seen': idx,
                    'frames_since_seen': 0,
                    'confidence_streak': 1,
                    'probability': det.get('probability', 0.9)
                })
        
        # Collect faces to blur - ONLY from verified tracks
        for track in tracked_faces:
            # Only blur if:
            # 1. Recently seen (within persistence window)
            # 2. Had at least 2 consecutive detections (filters one-off false positives)
            if track['frames_since_seen'] <= PERSISTENCE_FRAMES and track.get('confidence_streak', 0) >= 2:
                current_faces.append({
                    'x': track['x'],
                    'y': track['y'],
                    'width': track['width'],
                    'height': track['height'],
                    'probability': track.get('probability', 0.9)
                })
        
        # Remove stale tracks
        tracked_faces = [t for t in tracked_faces if t['frames_since_seen'] <= PERSISTENCE_FRAMES + 2]
        
        if current_faces:
            face_data[frame_file] = current_faces
    
    # Skip gap filling - rely only on velocity-based tracking for accuracy
    # This prevents blur from jumping to wrong areas
    
    return face_data


def fill_gaps_with_interpolation(frame_files, face_data, max_gap):
    """
    Fill remaining gaps between face detections with smooth interpolation.
    Ensures continuous blur even during brief detection failures.
    """
    # Find frames with faces and gaps between them
    frame_indices = {f: i for i, f in enumerate(frame_files)}
    detected_frames = sorted([frame_indices[f] for f in face_data.keys()])
    
    if len(detected_frames) < 2:
        return face_data
    
    # Fill gaps between consecutive detected frames
    for i in range(len(detected_frames) - 1):
        prev_idx = detected_frames[i]
        next_idx = detected_frames[i + 1]
        gap = next_idx - prev_idx
        
        # Only fill reasonable gaps
        if gap <= 1 or gap > max_gap:
            continue
        
        prev_file = frame_files[prev_idx]
        next_file = frame_files[next_idx]
        prev_faces = face_data[prev_file]
        next_faces = face_data[next_file]
        
        # Fill each frame in the gap
        for fill_idx in range(prev_idx + 1, next_idx):
            fill_file = frame_files[fill_idx]
            if fill_file in face_data:
                continue  # Already has faces
            
            t = (fill_idx - prev_idx) / gap
            interpolated = []
            
            for pf in prev_faces:
                # Find closest matching face in next frame
                best_match = None
                best_dist = float('inf')
                
                pf_cx = pf['x'] + pf['width'] / 2
                pf_cy = pf['y'] + pf['height'] / 2
                
                for nf in next_faces:
                    nf_cx = nf['x'] + nf['width'] / 2
                    nf_cy = nf['y'] + nf['height'] / 2
                    dist = ((pf_cx - nf_cx) ** 2 + (pf_cy - nf_cy) ** 2) ** 0.5
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_match = nf
                
                if best_match and best_dist < 600:
                    # Smooth interpolation with easing
                    ease_t = t * t * (3 - 2 * t)  # Smoothstep easing
                    interpolated.append({
                        'x': int(pf['x'] + (best_match['x'] - pf['x']) * ease_t),
                        'y': int(pf['y'] + (best_match['y'] - pf['y']) * ease_t),
                        'width': int(pf['width'] + (best_match['width'] - pf['width']) * ease_t),
                        'height': int(pf['height'] + (best_match['height'] - pf['height']) * ease_t),
                        'probability': pf.get('probability', 0.9)
                    })
                else:
                    # No good match - maintain previous position (fading)
                    fade = 1.0 - (t * 0.3)  # Slight size reduction as we move away
                    interpolated.append({
                        'x': pf['x'],
                        'y': pf['y'],
                        'width': int(pf['width'] * fade),
                        'height': int(pf['height'] * fade),
                        'probability': pf.get('probability', 0.9)
                    })
            
            if interpolated:
                face_data[fill_file] = interpolated
    
    return face_data


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python face_detector.py <frame_directory> [confidence] [stride]", file=sys.stderr)
        sys.exit(1)
    
    frame_dir = sys.argv[1]
    confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    if not os.path.isdir(frame_dir):
        print(f"❌ Error: Directory not found: {frame_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Process frames and detect faces
    face_data = process_frames(frame_dir, confidence, stride)
    
    # Output JSON to stdout
    print(json.dumps(face_data, indent=2))


if __name__ == '__main__':
    main()
