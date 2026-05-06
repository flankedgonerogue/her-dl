"""
app.py
======
Flask Web Application — Browser-based Emotion Recognition
----------------------------------------------------------
Provides a web interface for emotion recognition that works in any browser,
without requiring a GUI/display. Routes:

  GET  /           — Main dashboard
  GET  /api/status — Model & system status JSON
  POST /api/predict — Upload image, get emotion predictions JSON
  GET  /api/demo   — Demo predictions on synthetic face data

No webcam streaming in the web app (webcam access is in realtime_inference.py).
The web app focuses on image upload + prediction visualization.

Usage:
    python app.py
    # Then open: http://localhost:5000
"""

import os
import io
import time
import base64
import json
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, Response

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

from model import EMOTION_LABELS, IMG_SIZE

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Global model + cascade (loaded once at startup)
_model = None
_face_cascade = None
_model_trained = False

EMOTION_COLORS_HEX = {
    'Angry':    '#FF4444',
    'Disgust':  '#8B4513',
    'Fear':     '#9B59B6',
    'Happy':    '#F1C40F',
    'Sad':      '#3498DB',
    'Surprise': '#E67E22',
    'Neutral':  '#95A5A6'
}


def load_resources():
    """Load model and Haar cascade at startup."""
    global _model, _face_cascade, _model_trained

    # Load face detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    _face_cascade = cv2.CascadeClassifier(cascade_path)
    print(f"[App] Haar Cascade loaded")

    # Load model
    model_path = 'models/best_model.keras'
    if os.path.exists(model_path):
        _model = keras.models.load_model(model_path)
        _model_trained = True
        print(f"[App] Trained model loaded: {model_path}")
    else:
        from model import build_emotion_model, compile_model
        _model = build_emotion_model()
        _model = compile_model(_model)
        _model_trained = False
        print("[App] Warning: using untrained model. Train first with train_model.py")


def preprocess_face(face_gray):
    """Preprocess a face ROI for CNN inference."""
    resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)
    norm = equalized.astype(np.float32) / 255.0
    return norm[np.newaxis, ..., np.newaxis]


def decode_image(file_bytes):
    """Decode uploaded image bytes to numpy BGR array."""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def annotate_image(img, faces_data):
    """Draw emotion annotations on image, return as JPEG base64."""
    for face_info in faces_data:
        x, y, w, h = face_info['bbox']
        emotion = face_info['emotion']
        confidence = face_info['confidence']

        # Get color in BGR
        hex_color = EMOTION_COLORS_HEX.get(emotion, '#FFFFFF')
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        bgr = (b, g, r)

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x+w, y+h), bgr, 3)

        # Label background
        label = f"{emotion}: {confidence*100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thick = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        label_y = max(y - 10, th + 10)
        cv2.rectangle(img, (x, label_y - th - 8), (x + tw + 10, label_y + 4), bgr, -1)
        cv2.putText(img, label, (x + 5, label_y - 2), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)

        # Corner accents
        cl = min(w, h) // 5
        corners = [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]
        for (cx, cy) in corners:
            dx = cl if cx == x else -cl
            dy = cl if cy == y else -cl
            cv2.line(img, (cx, cy), (cx+dx, cy), bgr, 4)
            cv2.line(img, (cx, cy), (cx, cy+dy), bgr, 4)

    # Encode to JPEG base64
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html',
                           model_trained=_model_trained,
                           emotion_labels=EMOTION_LABELS,
                           emotion_colors=EMOTION_COLORS_HEX)


@app.route('/api/status')
def api_status():
    """Return system status as JSON."""
    return jsonify({
        'status': 'running',
        'model_trained': _model_trained,
        'model_params': int(_model.count_params()) if _model else 0,
        'tensorflow_version': tf.__version__,
        'opencv_version': cv2.__version__,
        'emotion_labels': EMOTION_LABELS,
        'input_size': f'{IMG_SIZE}x{IMG_SIZE}',
        'timestamp': time.time()
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Predict emotions from uploaded image.

    Request: multipart/form-data with 'image' file
    Response: JSON with detected faces and their emotions
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        t_start = time.perf_counter()

        # Decode image
        img_bytes = file.read()
        img = decode_image(img_bytes)
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400

        h_orig, w_orig = img.shape[:2]

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enh = clahe.apply(gray)

        # Detect faces
        faces = _face_cascade.detectMultiScale(
            gray_enh, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        results = []

        if len(faces) > 0:
            # Batch inference
            face_inputs = []
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                face_inputs.append(preprocess_face(roi)[0])

            batch = np.stack(face_inputs, axis=0)
            probs_batch = _model(batch, training=False).numpy()

            for i, (x, y, w, h) in enumerate(faces):
                probs = probs_batch[i]
                pred_idx = int(np.argmax(probs))
                emotion = EMOTION_LABELS[pred_idx]

                results.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'emotion': emotion,
                    'confidence': float(probs[pred_idx]),
                    'color': EMOTION_COLORS_HEX[emotion],
                    'all_probs': {
                        label: float(p)
                        for label, p in zip(EMOTION_LABELS, probs)
                    }
                })

        # Annotate and encode image
        img_annotated_b64 = annotate_image(img.copy(), results)

        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000

        return jsonify({
            'faces_detected': len(results),
            'faces': results,
            'annotated_image': img_annotated_b64,
            'image_size': {'width': w_orig, 'height': h_orig},
            'inference_ms': round(inference_ms, 2),
            'model_trained': _model_trained
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo')
def api_demo():
    """
    Demo prediction on a synthetically generated face image.
    Useful for testing the API without a real image.
    """
    # Generate synthetic 48x48 "face" image
    synthetic_face = np.random.randint(100, 220, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Add some face-like structure
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    cv2.ellipse(synthetic_face, (cx, cy), (15, 18), 0, 0, 360, 200, -1)
    cv2.circle(synthetic_face, (cx-5, cy-4), 3, 100, -1)
    cv2.circle(synthetic_face, (cx+5, cy-4), 3, 100, -1)

    face_norm = synthetic_face.astype(np.float32) / 255.0
    face_input = face_norm[np.newaxis, ..., np.newaxis]

    probs = _model(face_input, training=False).numpy()[0]
    pred_idx = int(np.argmax(probs))
    emotion = EMOTION_LABELS[pred_idx]

    return jsonify({
        'demo': True,
        'emotion': emotion,
        'confidence': float(probs[pred_idx]),
        'all_probs': {label: float(p) for label, p in zip(EMOTION_LABELS, probs)},
        'note': 'Demo uses synthetic face data for API testing'
    })


if __name__ == '__main__':
    print("=" * 60)
    print("  EmotionNet — Web Application")
    print("=" * 60)
    load_resources()
    print("\n[App] Starting Flask server at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)