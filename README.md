# 🧠 Real-Time Human Emotion Recognition using Deep CNNs

**Project for Human-Computer Interaction / Deep Learning Course**  
**Group Members:** Abdur Rahman Goraya (2022035) · Ahsan Naqvi (2022073)

---

## 📋 Background Research Summary

### 1. The FER2013 Dataset (Goodfellow et al., 2013)

The FER2013 dataset was introduced by Ian Goodfellow, Mehdi Mirza, and colleagues at the **30th International Conference on Machine Learning (ICML 2013)** in Atlanta, Georgia. It was part of a "Challenges in Representation Learning" competition alongside two other tasks.

**Dataset characteristics:**
- **35,887 total images** (28,709 training · 3,589 public test · 3,589 private test)
- **48×48 pixel grayscale images** collected by scraping Google Image Search
- **7 emotion classes:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Severe class imbalance:** Happy (8,989) vs Disgust (547) — 16× ratio
- **Noise:** Images were automatically labeled using Amazon Mechanical Turk, leading to significant label noise. Human accuracy on this dataset is estimated at only **65.5%**

The competition winner, Yichuan Tang, used an SVM loss function in a deep neural network, achieving **71.2% accuracy** — a landmark result at the time. Most early baselines were in the 40-60% range, showing the difficulty of "in-the-wild" recognition.

**Citation:** Goodfellow, I.J., et al. (2013). *Challenges in Representation Learning: A report on three machine learning contests.* Proceedings of ICNIP. [arXiv:1307.0414]

---

### 2. CNN Evolution for FER: From HOG/SVM to Deep Learning

**Traditional methods (pre-2013):**
- **Haar Cascades** (Viola & Jones, 2001): Fast face detection using rectangular feature differences across image regions. OpenCV's implementation uses 6,000+ features across 38 cascading stages — only ~10 features are evaluated per non-face region, making it extremely fast.
- **HOG + SVM:** Histogram of Oriented Gradients describes local gradient distributions; SVM classifies. Robust for controlled conditions but fails on variations in pose, lighting, and subtle expressions.

**Deep learning breakthrough:**
CNNs learn hierarchical spatial features automatically via backpropagation:
- **Layer 1-2:** Edge detectors (Gabor-like wavelets)
- **Layer 3-4:** Part detectors (eye corners, mouth shapes)
- **Layer 5+:** High-level expression detectors

Key CNN results on FER2013:
| Model | Accuracy | Year |
|---|---|---|
| Baseline SVM | ~44% | 2013 |
| Tang (2013) SVM-loss CNN | 71.2% | 2013 |
| VGG-16 (Simonyan) | ~72.7% | 2014 |
| Ensemble CNNs | ~74.3% | 2018 |
| EfficientNet V2 | ~76%+ | 2021 |
| **Human accuracy** | **~65.5%** | — |

---

### 3. MobileNet Architecture (Howard et al., 2017)

Standard convolution on a DxD input with N filters = **D × D × M × N × Dk × Dk** multiply-adds.

**Depthwise Separable Convolution** factorizes this into two steps:
1. **Depthwise conv:** Apply one Dk×Dk filter per input channel → D × D × M × Dk × Dk
2. **Pointwise conv (1×1):** Project channels → D × D × M × N

**Reduction ratio:** 1/N + 1/Dk² ≈ **8-9× fewer operations** for Dk=3

This enables real-time inference on mobile/edge devices. Our EmotionNet uses this principle across 4 stages (64→128→256→512 filters), keeping total parameters ~1.2M versus VGG16's 138M.

**Citation:** Howard, A.G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861.

---

### 4. The Affective Computing & HCI Context

Emotion recognition sits at the intersection of **affective computing** (Picard, 1997) and HCI. The "affective gap" is the inability of machines to respond to human emotional states, causing frustration in interactions. Key application domains:

- **Mental health monitoring:** Longitudinal tracking of mood via passive facial analysis
- **Intelligent tutoring:** Adjusting curriculum difficulty based on detected confusion/frustration  
- **Customer analytics:** Real-time sentiment in retail and service environments
- **Accessibility:** Helping individuals with ASD interpret others' emotional states

Paul Ekman's foundational work (1970s) established **6 universal expressions** (the 7th — neutral — was added later): anger, disgust, fear, happiness, sadness, surprise. FER2013 follows this taxonomy directly.

---

## 🏗️ Project Architecture

```
emotion_recognition/
├── model.py                  # Baseline CNN + MobileNetV2 transfer-learning models
├── data_loader.py            # FER2013 loading, preprocessing, augmentation
├── train_model.py            # Baseline training pipeline
├── train_transfer.py         # Two-stage transfer-learning pipeline
├── test_model.py             # Evaluation, comparison, benchmarks
├── realtime_inference.py     # Live webcam + metadata-aware inference
├── app.py                    # Flask web application
├── templates/
│   └── index.html            # Web UI
├── fer2013/                  # Dataset folder tree (train/test/class)
├── models/                   # Saved model weights
├── logs/                     # Training curves, reports
└── screenshots/              # Saved webcam frames
```

---

## 🧪 Model Architecture Detail

```
INPUT: 48×48×1 (grayscale face, normalized to [0,1])
│
├── Stem (Standard Conv Blocks)
│   ├── Conv2D(32, 3×3) → BN → ReLU6
│   └── Conv2D(32, 3×3) → BN → ReLU6
│   MaxPool2D(2×2)  ─── 48×48 → 24×24
│
├── Stage 1 (Depthwise Separable)
│   ├── DSConv(64) + Dropout(0.10)
│   └── DSConv(64) + Dropout(0.10)
│   MaxPool2D(2×2)  ─── 24×24 → 12×12
│
├── Stage 2
│   ├── DSConv(128) + Dropout(0.15)
│   └── DSConv(128) + Dropout(0.15)
│   MaxPool2D(2×2)  ─── 12×12 → 6×6
│
├── Stage 3
│   ├── DSConv(256) + Dropout(0.20)
│   └── DSConv(256) + Dropout(0.20)
│   MaxPool2D(2×2)  ─── 6×6 → 3×3
│
├── Stage 4
│   └── DSConv(512) + Dropout(0.25)
│
├── Global Average Pooling  ─── 3×3 → 1×1 (no Flatten)
│
├── Dense(256) → BN → Dropout(0.50)
├── Dense(128) → BN → Dropout(0.30)
└── Dense(7, Softmax)  ─── 7 emotion classes
```

**Regularization strategy:**
- L2 weight decay (1e-4) on all conv layers
- Progressive Dropout (0.10 → 0.50 in classifier head)
- Label smoothing (0.1): avoids overconfidence, helpful given FER2013's noisy labels
- Balanced class weighting: compensates for Disgust underrepresentation

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Install dependencies

```bash
cd emotion_recognition/
pip install tensorflow keras numpy opencv-python-headless matplotlib seaborn scikit-learn flask pillow
```

### (Optional) Download FER2013

For real training, download the dataset from Kaggle:
```
https://www.kaggle.com/datasets/msambare/fer2013
```
Place the extracted `fer2013/` folder in the project directory so it contains `train/<class>` and `test/<class>` subfolders.

---

## 📖 Usage

### 1. Test the architecture (no data needed)

```bash
uv run test_model.py --arch_only
```

Expected output:
```
[ArchTest] Building model...
Total parameters: ~1,200,000
Batch=1: output=(1, 7) ✓
Batch=4: output=(4, 7) ✓
Inference Speed Test:
  Batch Size |  Mean (ms) |  Std (ms) |      FPS
           1 |       8.50 |      1.20 |    117.6
           8 |      25.30 |      2.10 |    316.2
          32 |      89.40 |      5.30 |    357.9
Real-time feasibility (≥25 FPS at batch=1): ✓ YES
```

### 2. Train with synthetic data (quick smoke test)

```bash
uv run train_model.py --synthetic --synthetic_n 5000 --epochs 10
```

### 3. Train on real FER2013

```bash
uv run train_model.py --data fer2013 --epochs 100 --batch_size 64
```

Training output files:
- `models/best_model.keras` — best checkpoint by val_accuracy
- `logs/training_curves.png` — accuracy & loss plots
- `logs/training_history.csv` — per-epoch metrics
- `logs/training_metadata.json` — full run metadata

### 4. Full evaluation

```bash
# With trained model + real data:
uv run test_model.py --model models/best_model.keras --data fer2013

# With synthetic data:
uv run test_model.py --model models/best_model.keras --synthetic
```

Outputs:
- `logs/confusion_matrix.png` — absolute + normalized confusion matrices
- `logs/per_class_metrics.png` — per-class precision/recall/F1 bars
- `logs/classification_report.txt` — full sklearn report

### 5. Real-time webcam inference

```bash
# With webcam:
uv run realtime_inference.py --model models/best_model.keras

# Demo mode (no webcam):
uv run realtime_inference.py --demo

# Save to video:
uv run realtime_inference.py --model models/best_model.keras --save_video output.avi
```

Controls: `Q` quit · `S` screenshot · `P` pause · `+/-` adjust sensitivity

### 6. Web application

```bash
uv run app.py
# Open: http://localhost:5000
```

The web app lets you:
- Upload any face image
- See emotion predictions with probability bars
- View annotated output image with bounding boxes
- Use the `/api/demo` endpoint for API testing

---

## 📊 Expected Results on Real FER2013

| Metric | Expected Value | Notes |
|---|---|---|
| Top-1 Accuracy | 65–70% | Matches human performance |
| Top-2 Accuracy | 85–90% | Two most likely emotions |
| Happy (F1) | ~85% | Most samples, easiest class |
| Disgust (F1) | ~40% | Only 547 training samples |
| Inference (1 face) | <15ms | Real-time capable |
| FPS (webcam) | >25 FPS | Smooth real-time |

### Challenging aspects of FER2013:
1. **Label noise:** Many images are mislabeled by Amazon Mechanical Turk workers
2. **Class imbalance:** Disgust has 16× fewer samples than Happy
3. **In-the-wild:** Uncontrolled lighting, pose, occlusion
4. **Low resolution:** 48×48 loses subtle micro-expressions

---

## 🔬 Inference Pipeline Detail

```
Webcam Frame (640×480, BGR)
        │
        ▼
Grayscale conversion (cv2.cvtColor)
        │
        ▼
CLAHE enhancement (local contrast normalization)
        │
        ▼
Haar Cascade face detection (haarcascade_frontalface_default.xml)
  - scaleFactor=1.1 (image pyramid step)
  - minNeighbors=5 (false positive suppression)
  - minSize=(60,60) pixels
        │
        ▼
For each detected face ROI:
  ├── Resize to 48×48 (INTER_AREA)
  ├── Histogram equalization (lighting normalization)
  ├── Normalize to [0,1]
  └── Shape: (1, 48, 48, 1)
        │
        ▼
Batch all faces → EmotionNet CNN
        │
        ▼
Softmax probabilities (7 classes)
        │
        ▼
Draw bounding box + label + probability bars on frame
        │
        ▼
Display with FPS counter
```

---

## 🔁 Transfer-Learning Workflow

The new transfer path keeps the baseline CNN available while adding a stronger
MobileNetV2 option for higher accuracy.

### Train the transfer model

```bash
uv run train_transfer.py --data fer2013 --head_epochs 10 --fine_tune_epochs 10 --model_path models/transfer_best_model.keras
```

### Evaluate baseline vs transfer

```bash
uv run test_model.py --model models/best_model.keras --data fer2013
uv run test_model.py --model models/transfer_best_model.keras --data fer2013
uv run test_model.py --model models/best_model.keras --compare_model models/transfer_best_model.keras --data fer2013
```

### Run real-time inference

```bash
uv run realtime_inference.py --model models/best_model.keras
uv run realtime_inference.py --model models/transfer_best_model.keras
```

### Model comparison summary

| Model | Input | Preprocessing | Expected tradeoff |
|---|---|---|---|
| Baseline CNN | 48×48×1 | Grayscale normalization | Fastest inference |
| Transfer model | 96×96×3 | MobileNetV2 preprocessing | Better accuracy, heavier compute |

The runtime reads model metadata automatically, so the same inference and
Flask app entry points can serve either model without manual preprocessing
changes.

## 📚 References

1. **Goodfellow, I.J., et al. (2013).** *Challenges in Representation Learning: A report on three machine learning contests.* Proceedings of the International Conference on Neural Information Processing. [arXiv:1307.0414]

2. **Lopes, A.T., et al. (2017).** *Facial expression recognition with convolutional neural networks: Coping with few data and the training of a specific fully convolutional network.* Pattern Recognition, 61, 650–662.

3. **Simonyan, K. & Zisserman, A. (2014).** *Very Deep Convolutional Networks for Large-Scale Image Recognition.* arXiv:1409.1556. [VGG]

4. **Howard, A.G., et al. (2017).** *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861.

5. **Viola, P. & Jones, M. (2001).** *Rapid Object Detection using a Boosted Cascade of Simple Features.* Proceedings of CVPR, 2001.

6. **Ekman, P. & Friesen, W.V. (1971).** *Constants across cultures in the face and emotion.* Journal of Personality and Social Psychology, 17(2), 124–129.

7. **He, K., et al. (2016).** *Deep Residual Learning for Image Recognition (ResNet).* Proceedings of CVPR, 2016.

---

## 🤝 Acknowledgments

Dataset: FER2013 — Kaggle / Goodfellow et al.  
Framework: TensorFlow / Keras  
Face Detection: OpenCV Haar Cascades (Viola & Jones)  
Architecture inspiration: MobileNet (Howard et al.)
