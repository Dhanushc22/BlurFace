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
            # MediaPipe Face Detection - use BOTH models for maximum coverage
            try:
                # Model 0: Short-range (within 2 meters) - better for close faces
                # Model 1: Full-range (within 5 meters) - better for distant faces
                # We'll use model 1 (full-range) with VERY low confidence
                self.detector = mp.face_detection.FaceDetection(
                    model_selection=1,  # Full-range model for ALL distances
                    min_detection_confidence=0.4  # Balanced threshold - no false positives
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
        """Detect faces using MediaPipe - FAST single pass"""
        h, w = image.shape[:2]
        
        # Single pass detection (fast)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(image_rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                if detection.score[0] < 0.4:
                    continue
                
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Add margins for movement coverage
                margin_x = int(width * 0.20)
                margin_y = int(height * 0.20)
                
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
        """Detect ALL faces using OpenCV Haar Cascade (frontal + alt + profile, multi-scale)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Enhance contrast for better detection
        gray = cv2.equalizeHist(gray)
        h, w = image.shape[:2]
        
        all_faces = []
        
        # Detect frontal faces with default detector
        frontal_rects = self.frontal_detector.detectMultiScale(
            gray,
            scaleFactor=1.05,    # Finer scale for better accuracy
            minNeighbors=3,      # Lower for more detections
            minSize=(20, 20),    # Smaller minimum for distant faces
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, fw, fh) in frontal_rects:
            margin_x = int(fw * 0.25)
            margin_y = int(fh * 0.25)
            
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
        
        # Detect with alt detector (catches different face angles)
        if hasattr(self, 'frontal_alt_detector') and not self.frontal_alt_detector.empty():
            alt_rects = self.frontal_alt_detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, fw, fh) in alt_rects:
                margin_x = int(fw * 0.25)
                margin_y = int(fh * 0.25)
                
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
        
        # Detect profile faces (left-facing)
        if hasattr(self, 'profile_detector') and not self.profile_detector.empty():
            profile_rects = self.profile_detector.detectMultiScale(
                gray,
                scaleFactor=1.03,      # Finer scale for better side face detection
                minNeighbors=1,        # Even lower for side angles (catches more profile faces)
                minSize=(15, 15),      # Smaller minimum for profile faces at distance
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, fw, fh) in profile_rects:
                margin_x = int(fw * 0.40)  # Larger margin for profiles - covers more area
                margin_y = int(fh * 0.35)  # Increased top/bottom margin
                
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                fw = min(w - x, fw + margin_x * 2)
                fh = min(h - y, fh + margin_y * 2)
                
                all_faces.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(fw),
                    'height': int(fh),
                    'probability': 0.90  # Higher confidence for profile faces
                })
            
            # Detect right-facing profiles by flipping image
            flipped = cv2.flip(gray, 1)
            profile_rects_flip = self.profile_detector.detectMultiScale(
                flipped,
                scaleFactor=1.03,      # Finer scale for better side face detection
                minNeighbors=1,        # Even lower for side angles
                minSize=(15, 15),      # Smaller minimum for profile faces
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, fw, fh) in profile_rects_flip:
                # Mirror x coordinate back
                x = w - x - fw
                margin_x = int(fw * 0.40)  # Larger margin for profiles - covers more area
                margin_y = int(fh * 0.35)  # Increased top/bottom margin
                
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                fw = min(w - x, fw + margin_x * 2)
                fh = min(h - y, fh + margin_y * 2)
                
                all_faces.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(fw),
                    'height': int(fh),
                    'probability': 0.90  # Higher confidence for profile faces
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
                    
                    if iou > 0.4:  # 40% overlap = duplicate (allow some overlap for side faces)
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
    # Balanced confidence - catch real faces, no false positives
    detector = FaceDetector(0.4)
    
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
    Apply temporal smoothing to stabilize face blur and prevent blinking.
    Uses face tracking with persistence to maintain blur across detection gaps.
    """
    detection_indices = sorted(raw_detections.keys())
    face_data = {}
    
    # Track faces across frames with persistence
    # Each tracked face has: position, last_seen, persistence_frames
    tracked_faces = []
    PERSISTENCE_FRAMES = 12  # Keep blurring for 12 frames after losing detection
    MAX_GAP_FILL = 25  # Fill gaps up to 25 frames for smooth continuous blur
    MATCH_DISTANCE = 600  # Faces within 600px are considered same face
    
    for idx in range(total_frames):
        frame_file = frame_files[idx]
        current_faces = []
        
        # Get detected faces for this frame (if any)
        detected_faces = []
        if idx in raw_detections:
            detected_faces = raw_detections[idx][1]
        
        # Update tracked faces with new detections
        used_detections = set()
        
        for track_idx, track in enumerate(tracked_faces):
            # Find best matching detection for this tracked face
            best_match = None
            best_dist = float('inf')
            best_det_idx = -1
            
            for det_idx, det in enumerate(detected_faces):
                if det_idx in used_detections:
                    continue
                # Calculate center-to-center distance
                track_cx = track['x'] + track['width'] / 2
                track_cy = track['y'] + track['height'] / 2
                det_cx = det['x'] + det['width'] / 2
                det_cy = det['y'] + det['height'] / 2
                dist = ((track_cx - det_cx) ** 2 + (track_cy - det_cy) ** 2) ** 0.5
                
                if dist < best_dist and dist < MATCH_DISTANCE:
                    best_dist = dist
                    best_match = det
                    best_det_idx = det_idx
            
            if best_match:
                # Update tracked face with smooth interpolation (reduces jitter)
                alpha = 0.6  # Smoothing factor - higher = more responsive, lower = smoother
                track['x'] = int(track['x'] * (1 - alpha) + best_match['x'] * alpha)
                track['y'] = int(track['y'] * (1 - alpha) + best_match['y'] * alpha)
                track['width'] = int(track['width'] * (1 - alpha) + best_match['width'] * alpha)
                track['height'] = int(track['height'] * (1 - alpha) + best_match['height'] * alpha)
                track['last_seen'] = idx
                track['frames_since_seen'] = 0
                track['probability'] = best_match.get('probability', 0.9)
                used_detections.add(best_det_idx)
            else:
                # No match - increment frames since last seen
                track['frames_since_seen'] = idx - track['last_seen']
        
        # Add new tracked faces for unmatched detections
        for det_idx, det in enumerate(detected_faces):
            if det_idx not in used_detections:
                tracked_faces.append({
                    'x': det['x'],
                    'y': det['y'],
                    'width': det['width'],
                    'height': det['height'],
                    'last_seen': idx,
                    'frames_since_seen': 0,
                    'probability': det.get('probability', 0.9)
                })
        
        # Collect faces to blur (active tracks within persistence window)
        for track in tracked_faces:
            if track['frames_since_seen'] <= PERSISTENCE_FRAMES:
                current_faces.append({
                    'x': track['x'],
                    'y': track['y'],
                    'width': track['width'],
                    'height': track['height'],
                    'probability': track.get('probability', 0.9)
                })
        
        # Remove stale tracks (not seen for too long)
        tracked_faces = [t for t in tracked_faces if t['frames_since_seen'] <= PERSISTENCE_FRAMES + 5]
        
        if current_faces:
            face_data[frame_file] = current_faces
    
    # Second pass: fill remaining gaps with interpolation for even smoother results
    face_data = fill_gaps_with_interpolation(frame_files, face_data, MAX_GAP_FILL)
    
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
