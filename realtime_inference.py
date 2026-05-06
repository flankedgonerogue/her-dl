"""
realtime_inference.py
=====================
Real-Time Webcam Emotion Recognition
-------------------------------------
Integrates the trained CNN with OpenCV for live webcam-based emotion detection.

Pipeline (as described in the project proposal):
  1. Capture frame from webcam via cv2.VideoCapture
  2. Convert frame to grayscale
  3. Detect faces using Haar Cascade classifier (haarcascade_frontalface_default.xml)
  4. For each detected face:
     a. Crop and resize face ROI to 48×48 pixels
     b. Normalize pixel values to [0, 1]
     c. Feed to trained CNN model
     d. Display top-1 prediction + confidence bar overlay
  5. Show annotated frame via cv2.imshow

Controls:
  Q  — quit
  S  — save current frame as screenshot
  P  — toggle pause
  +  — increase detection sensitivity
  -  — decrease detection sensitivity

Usage:
    python realtime_inference.py --model models/best_model.keras
    python realtime_inference.py --model models/best_model.keras --camera 1
    python realtime_inference.py --model models/best_model.keras --demo   # demo mode, no webcam

References:
    Viola, P. & Jones, M. (2001). Rapid Object Detection using a Boosted Cascade.
    OpenCV documentation — Cascade Classifier
    Goodfellow et al. (2013) — FER2013 emotion classes
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

# Suppress TF logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 48

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# BGR colors for each emotion (OpenCV uses BGR)
EMOTION_COLORS_BGR = {
    'Angry':    (0,   68,  255),   # red
    'Disgust':  (19, 69,  139),    # brown
    'Fear':     (166, 90, 155),    # purple
    'Happy':    (0,  211, 241),    # yellow
    'Sad':      (220, 77, 52),     # blue
    'Surprise': (0,  126, 230),    # orange
    'Neutral':  (150,165, 149)     # gray
}

# Emoji-style text overlay per emotion
EMOTION_EMOJI = {
    'Angry':    '(>_<)',
    'Disgust':  '(*_*)',
    'Fear':     '(O_O)',
    'Happy':    '^_^  ',
    'Sad':      "(T_T)",
    'Surprise': '(o_O)',
    'Neutral':  '(-_-)'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Real-Time Emotion Recognition via Webcam',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='models/best_model.keras',
                        help='Path to trained .keras model')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (0=default webcam)')
    parser.add_argument('--scale_factor', type=float, default=1.1,
                        help='Haar Cascade scaleFactor (1.05-1.4)')
    parser.add_argument('--min_neighbors', type=int, default=5,
                        help='Haar Cascade minNeighbors (3-8)')
    parser.add_argument('--min_face_size', type=int, default=60,
                        help='Minimum face size in pixels')
    parser.add_argument('--demo', action='store_true',
                        help='Demo mode: uses test images instead of webcam')
    parser.add_argument('--save_video', type=str, default=None,
                        help='Save output to video file (e.g., output.avi)')
    return parser.parse_args()


def load_model(model_path):
    """Load trained Keras model with fallback to untrained architecture."""
    if os.path.exists(model_path):
        print(f"[Inference] Loading trained model: {model_path}")
        model = keras.models.load_model(model_path)
        print(f"[Inference] Model loaded. Parameters: {model.count_params():,}")
        return model, True
    else:
        print(f"[Warning] Model not found at '{model_path}'.")
        print("[Warning] Using UNTRAINED model — predictions will be random!")
        print("[Info] Train first: python train_model.py --synthetic --epochs 5")
        from model import build_emotion_model, compile_model
        model = build_emotion_model()
        model = compile_model(model)
        return model, False


def load_face_detector():
    """
    Load OpenCV Haar Cascade face detector.
    Falls back to the bundled OpenCV file if a local copy isn't present.
    The Viola-Jones cascade was trained with 6000+ features across 38 stages.
    """
    # Try built-in OpenCV cascades first
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_path):
        print(f"[Inference] Haar Cascade loaded: {cascade_path}")
        return cv2.CascadeClassifier(cascade_path)

    # Fallback to local file
    local_path = 'haarcascade_frontalface_default.xml'
    if os.path.exists(local_path):
        print(f"[Inference] Haar Cascade loaded: {local_path}")
        return cv2.CascadeClassifier(local_path)

    print("[Error] Haar Cascade XML not found. Install OpenCV with data.")
    sys.exit(1)


def preprocess_face(face_roi_gray):
    """
    Prepare a face ROI for model input.
    Steps:
      1. Resize to 48×48 (FER2013 input size)
      2. Normalize pixels to [0, 1]
      3. Add batch + channel dimensions
    """
    face_resized = cv2.resize(face_roi_gray, (IMG_SIZE, IMG_SIZE),
                              interpolation=cv2.INTER_AREA)
    face_norm = face_resized.astype(np.float32) / 255.0

    # Apply histogram equalization for lighting robustness
    face_uint8 = (face_norm * 255).astype(np.uint8)
    face_equalized = cv2.equalizeHist(face_uint8)
    face_final = face_equalized.astype(np.float32) / 255.0

    return face_final[np.newaxis, ..., np.newaxis]  # (1, 48, 48, 1)


def draw_face_annotation(frame, x, y, w, h, emotion, confidence, all_probs):
    """
    Draw rich annotation overlay on the detected face:
      - Bounding box in emotion color
      - Emotion label + confidence
      - Mini probability bar chart for top 3 emotions
    """
    color = EMOTION_COLORS_BGR.get(emotion, (255, 255, 255))
    emoji = EMOTION_EMOJI.get(emotion, '')

    # ── Main bounding box ────────────────────────────────────────────────
    thickness = 2
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

    # ── Corner accents ───────────────────────────────────────────────────
    corner_len = min(w, h) // 6
    corners = [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]
    for (cx, cy) in corners:
        dx = corner_len if cx == x else -corner_len
        dy = corner_len if cy == y else -corner_len
        cv2.line(frame, (cx, cy), (cx+dx, cy), color, 3)
        cv2.line(frame, (cx, cy), (cx, cy+dy), color, 3)

    # ── Label background ─────────────────────────────────────────────────
    label_text = f"{emoji} {emotion}: {confidence*100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)

    label_y = max(y - 10, th + 10)
    cv2.rectangle(frame,
                  (x, label_y - th - 5),
                  (x + tw + 8, label_y + baseline),
                  color, -1)

    # Text (white on colored bg)
    cv2.putText(frame, label_text,
                (x + 4, label_y - 2),
                font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    # ── Mini probability bars (top 3) ────────────────────────────────────
    top3_idx = np.argsort(all_probs)[::-1][:3]
    bar_x = x + w + 8
    bar_y = y

    if bar_x + 120 < frame.shape[1]:
        for rank, cls_idx in enumerate(top3_idx):
            bar_label  = EMOTION_LABELS[cls_idx]
            bar_prob   = all_probs[cls_idx]
            bar_color  = EMOTION_COLORS_BGR.get(bar_label, (200, 200, 200))
            bar_width  = int(bar_prob * 80)
            bar_height = 14

            by = bar_y + rank * (bar_height + 4)
            if by + bar_height > frame.shape[0]:
                break

            # Background
            cv2.rectangle(frame, (bar_x, by), (bar_x + 80, by + bar_height),
                          (50, 50, 50), -1)
            # Filled bar
            if bar_width > 0:
                cv2.rectangle(frame, (bar_x, by), (bar_x + bar_width, by + bar_height),
                              bar_color, -1)
            # Label text
            cv2.putText(frame,
                        f"{bar_label[:3]} {bar_prob*100:.0f}%",
                        (bar_x + 2, by + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_hud(frame, fps, n_faces, model_trained, paused=False):
    """Draw a HUD (heads-up display) with FPS and status info."""
    h, w = frame.shape[:2]

    # Semi-transparent HUD background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 95), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    status_color = (0, 255, 0) if model_trained else (0, 165, 255)
    model_str = "Trained Model" if model_trained else "Untrained (Demo)"

    cv2.putText(frame, f"EmotionNet | FPS: {fps:5.1f}", (8, 20),
                font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Faces detected: {n_faces}", (8, 42),
                font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Model: {model_str}", (8, 62),
                font, 0.45, status_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "Q=Quit | S=Screenshot | P=Pause", (8, 82),
                font, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

    if paused:
        cv2.putText(frame, "|| PAUSED", (w//2 - 50, h//2),
                    font, 1.5, (0, 255, 255), 3, cv2.LINE_AA)

    return frame


def generate_demo_frame(width=640, height=480):
    """
    Generate a synthetic grayscale frame with a face-like region for demo mode.
    Simulates what a webcam frame looks like — useful for headless testing.
    """
    frame = np.ones((height, width, 3), dtype=np.uint8) * 30

    # Draw a simple oval "face" in the center
    cx, cy = width // 2, height // 2
    face_w, face_h = 120, 150
    cv2.ellipse(frame, (cx, cy), (face_w//2, face_h//2), 0, 0, 360, (180, 160, 140), -1)
    # Eyes
    cv2.circle(frame, (cx - 30, cy - 20), 12, (50, 50, 50), -1)
    cv2.circle(frame, (cx + 30, cy - 20), 12, (50, 50, 50), -1)
    # Smile
    cv2.ellipse(frame, (cx, cy + 20), (40, 25), 0, 0, 180, (50, 50, 50), 2)

    cv2.putText(frame, "DEMO MODE - Simulated Frame", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)

    return frame


def run_realtime(args):
    """Main real-time inference loop."""
    model, model_trained = load_model(args.model)
    face_cascade = load_face_detector()

    # Set up video capture
    if args.demo:
        cap = None
        print("[Inference] Demo mode: using synthetic frames (no webcam required)")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[Error] Cannot open camera {args.camera}")
            print("[Info] Try: python realtime_inference.py --demo")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"[Inference] Webcam {args.camera} opened successfully")

    # Video writer
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (640, 480))
        print(f"[Inference] Recording to: {args.save_video}")

    print("\n[Inference] Starting real-time emotion recognition...")
    print("  Press Q to quit, S to screenshot, P to pause\n")

    fps = 0.0
    frame_count = 0
    fps_buffer = []
    paused = False
    screenshot_count = 0
    scale_factor = args.scale_factor
    min_neighbors = args.min_neighbors

    os.makedirs('screenshots', exist_ok=True)

    try:
        while True:
            t_frame_start = time.perf_counter()

            # Get frame
            if args.demo:
                frame = generate_demo_frame()
                time.sleep(0.033)  # Simulate 30 FPS
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[Warning] Frame capture failed, retrying...")
                    continue

            if paused:
                draw_hud(frame, fps, 0, model_trained, paused=True)
                cv2.imshow('EmotionNet — Real-Time Recognition', frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:
                    break
                if key == ord('p'):
                    paused = False
                continue

            # ── Face Detection ──────────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE for better detection under variable lighting
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_enhanced = clahe.apply(gray)

            faces = face_cascade.detectMultiScale(
                gray_enhanced,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(args.min_face_size, args.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            n_faces = len(faces) if isinstance(faces, np.ndarray) else 0

            # ── Emotion Prediction for each face ─────────────────────────
            if n_faces > 0:
                # Batch all faces for efficient inference
                face_inputs = []
                valid_faces = []

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    face_input = preprocess_face(face_roi)
                    face_inputs.append(face_input[0])  # remove batch dim
                    valid_faces.append((x, y, w, h))

                if face_inputs:
                    batch = np.stack(face_inputs, axis=0)  # (N, 48, 48, 1)
                    probs_batch = model(batch, training=False).numpy()

                    for i, (x, y, w, h) in enumerate(valid_faces):
                        probs = probs_batch[i]
                        pred_idx = np.argmax(probs)
                        emotion = EMOTION_LABELS[pred_idx]
                        confidence = probs[pred_idx]

                        frame = draw_face_annotation(
                            frame, x, y, w, h, emotion, confidence, probs
                        )

            # ── HUD overlay ──────────────────────────────────────────────
            frame = draw_hud(frame, fps, n_faces, model_trained)

            # ── FPS computation ──────────────────────────────────────────
            t_frame_end = time.perf_counter()
            frame_time = t_frame_end - t_frame_start
            fps_buffer.append(1.0 / max(frame_time, 1e-9))
            if len(fps_buffer) > 30:
                fps_buffer.pop(0)
            fps = np.mean(fps_buffer)
            frame_count += 1

            # ── Display & write ──────────────────────────────────────────
            cv2.imshow('EmotionNet — Real-Time Recognition', frame)
            if video_writer:
                video_writer.write(frame)

            # ── Key handling ─────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[Inference] Quit signal received.")
                break
            elif key == ord('s'):
                path = f"screenshots/screenshot_{screenshot_count:04d}.png"
                cv2.imwrite(path, frame)
                print(f"[Inference] Screenshot saved: {path}")
                screenshot_count += 1
            elif key == ord('p'):
                paused = True
                print("[Inference] Paused. Press P to resume.")
            elif key == ord('+'):
                min_neighbors = max(1, min_neighbors - 1)
                print(f"[Inference] minNeighbors → {min_neighbors}")
            elif key == ord('-'):
                min_neighbors = min(15, min_neighbors + 1)
                print(f"[Inference] minNeighbors → {min_neighbors}")

    except KeyboardInterrupt:
        print("\n[Inference] Interrupted by user.")
    finally:
        if cap is not None:
            cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"\n[Inference] Session summary:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Screenshots: {screenshot_count}")


def run_on_image(model_path, image_path, output_path=None):
    """
    Run emotion recognition on a single static image.
    Useful for testing without a webcam.
    """
    model, model_trained = load_model(model_path)
    face_cascade = load_face_detector()

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Error] Cannot read image: {image_path}")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                           minSize=(30, 30))

    print(f"[Inference] Detected {len(faces)} face(s) in {image_path}")

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_input = preprocess_face(face_roi)
        probs = model(face_input, training=False).numpy()[0]
        pred_idx = np.argmax(probs)
        emotion = EMOTION_LABELS[pred_idx]
        confidence = probs[pred_idx]
        print(f"  Face at ({x},{y}): {emotion} ({confidence*100:.1f}%)")
        frame = draw_face_annotation(frame, x, y, w, h, emotion, confidence, probs)

    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"[Inference] Annotated image saved: {output_path}")
    else:
        cv2.imshow('Emotion Recognition', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    print("=" * 60)
    print("  Real-Time Human Emotion Recognition")
    print("  Haar Cascade + CNN Pipeline")
    print("=" * 60)
    print(f"  OpenCV version: {cv2.__version__}")
    print(f"  TensorFlow version: {tf.__version__}")
    print()

    run_realtime(args)


if __name__ == '__main__':
    main()