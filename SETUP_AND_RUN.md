# 🚀 Setup & Run Guide — Real-Time Human Emotion Recognition

This guide provides step-by-step instructions to install, train, evaluate, and run the emotion recognition project.

---

## Quick Start (with Synthetic Data)

```bash
# 1. Navigate to project directory
cd /home/ahsan/her-dl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model (using synthetic data, no dataset download needed)
python train_model.py --synthetic --epochs 20

# 4. Test/evaluate the model
python test_model.py --model models/best_model.keras --synthetic

# 5. Run real-time webcam inference
python realtime_inference.py --model models/best_model.keras

# 6. Start web application
python app.py
# Then open: http://localhost:5000
```

---

## Full Training (with Real FER2013 Data)

```bash
# 1. Download FER2013 dataset from Kaggle
# https://www.kaggle.com/datasets/msambare/fer2013
# Download fer2013.csv and place it in the project directory

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model on real data (100 epochs)
python train_model.py --data fer2013.csv --epochs 100 --batch_size 64

# 4. Evaluate on test set
python test_model.py --model models/best_model.keras --data fer2013.csv

# 5. Run real-time webcam inference
python realtime_inference.py --model models/best_model.keras

# 6. Launch web app
python app.py
```

---

## Detailed Command Reference

### Training Options

```bash
# Train with synthetic data (quick test, ~5 minutes)
python train_model.py --synthetic --epochs 20

# Train with real FER2013 data (2-3 hours on GPU)
python train_model.py --data fer2013.csv --epochs 100 --batch_size 64

# Resume from checkpoint
python train_model.py --data fer2013.csv --resume models/best_model.keras --epochs 100

# Disable data augmentation
python train_model.py --data fer2013.csv --no_augment --epochs 50

# Custom learning rate and batch size
python train_model.py --data fer2013.csv --lr 0.0001 --batch_size 32 --epochs 100

# Generate more synthetic samples
python train_model.py --synthetic --synthetic_n 10000 --epochs 50
```

### Testing & Evaluation

```bash
# Evaluate on FER2013 test set
python test_model.py --model models/best_model.keras --data fer2013.csv

# Test with synthetic data
python test_model.py --model models/best_model.keras --synthetic

# Test architecture only (no model file needed)
python test_model.py --arch_only
```

### Real-Time Webcam Inference

```bash
# Use default webcam (camera 0)
python realtime_inference.py --model models/best_model.keras

# Use specific camera device (e.g., camera 1)
python realtime_inference.py --model models/best_model.keras --camera 1

# Demo mode (no webcam required, uses synthetic face data)
python realtime_inference.py --model models/best_model.keras --demo
```

**Keyboard Controls:**
- `Q` — Quit the application
- `S` — Save current frame as screenshot
- `P` — Pause/unpause video
- `+` — Increase detection sensitivity
- `-` — Decrease detection sensitivity

### Web Application

```bash
# Start Flask web server (runs on http://localhost:5000)
python app.py
```

**Features:**
- Image upload and emotion prediction
- Real-time prediction visualization
- System status and model information
- Demo predictions on synthetic data

---

## Installation Requirements

### System Requirements
- Python 3.8+
- NVIDIA GPU (optional, but recommended for faster training)
- Webcam (for real-time inference, not required for web app)

### Dependencies

Install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Key packages:**
- TensorFlow >= 2.12.0
- OpenCV (cv2)
- Flask
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

---

## Download FER2013 Dataset

The FER2013 dataset is required for training on real data. To download:

1. Go to [Kaggle Datasets: FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
2. Download `fer2013.csv`
3. Place it in the project root directory: `/home/ahsan/her-dl/fer2013.csv`
4. Run training: `python train_model.py --data fer2013.csv --epochs 100`

**Dataset Info:**
- **35,887 total images** (28,709 training · 3,589 public test · 3,589 private test)
- **7 emotion classes:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Image size:** 48×48 pixels (grayscale)
- **Human accuracy baseline:** 65.5%

---

## Expected Results

### Training Performance
- **Synthetic data:** ~80-85% accuracy (5 epochs, 5 minutes)
- **Real FER2013 data:** ~65-70% accuracy (100 epochs, 2-3 hours on GPU)

### Output Files

After training and evaluation, the following files are generated:

```
models/
├── best_model.keras          # Trained model weights

logs/
├── training_curves.png       # Accuracy and loss graphs
├── training_metadata.json    # Training configuration and results
├── confusion_matrix.png      # Evaluation confusion matrix
├── classification_report.txt # Per-class metrics (precision, recall, F1)

screenshots/
└── <timestamp>_emotion.png   # Saved webcam frames (during real-time inference)
```

---

## Project Structure

```
emotion_recognition/
├── model.py                   # CNN architecture (MobileNet-inspired)
├── data_loader.py             # FER2013 loading, preprocessing, augmentation
├── train_model.py             # Full training pipeline
├── test_model.py              # Evaluation suite
├── realtime_inference.py      # Live webcam emotion recognition
├── app.py                     # Flask web application
├── index.html                 # Web UI
├── requirements.txt           # Python dependencies
├── README.md                  # Project background & methodology
├── SETUP_AND_RUN.md          # This file
├── models/                    # Saved model weights
├── logs/                      # Training curves, reports
└── screenshots/               # Saved webcam frames
```

---

## Troubleshooting

### Issue: `pip install -r requirements.txt` fails

**Solution:** Updates pip and try again
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Cannot find `fer2013.csv`

**Solution:** Download from Kaggle or use synthetic data
```bash
python train_model.py --synthetic --epochs 20  # Test with synthetic data
```

### Issue: Webcam not detected during real-time inference

**Solution:** Use demo mode instead
```bash
python realtime_inference.py --model models/best_model.keras --demo
```

### Issue: CUDA/GPU not detected by TensorFlow

**Solution:** TensorFlow will automatically fall back to CPU. For GPU support:
```bash
pip install tensorflow[and-cuda]
# or install CUDA/cuDNN separately and configure your system
```

### Issue: ModuleNotFoundError for tensorflow/opencv

**Solution:** Reinstall packages
```bash
pip install --force-reinstall tensorflow opencv-python
```

---

## Command Workflow Examples

### Workflow 1: Complete Training & Evaluation (Synthetic)
```bash
# Quick test without downloading data (~10 minutes total)
python train_model.py --synthetic --epochs 20
python test_model.py --model models/best_model.keras --synthetic
python realtime_inference.py --model models/best_model.keras --demo
```

### Workflow 2: Complete Training & Evaluation (Real Data)
```bash
# Full training on FER2013 (~3-4 hours total)
python train_model.py --data fer2013.csv --epochs 100 --batch_size 64
python test_model.py --model models/best_model.keras --data fer2013.csv
python realtime_inference.py --model models/best_model.keras
python app.py
```

### Workflow 3: Fine-tune Pre-trained Model
```bash
# Resume from existing checkpoint and train more
python train_model.py --data fer2013.csv --resume models/best_model.keras --epochs 150
python test_model.py --model models/best_model.keras --data fer2013.csv
```

### Workflow 4: Web App Deployment
```bash
# Just run the web app (assumes model already trained)
python app.py
# Open: http://localhost:5000
```

---

## Python Version & Environment Setup (Optional)

If using conda:

```bash
# Create a new conda environment
conda create -n emotion_recognition python=3.10

# Activate the environment
conda activate emotion_recognition

# Install dependencies
pip install -r requirements.txt
```

If using venv:

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Next Steps

1. **Train a model** using `train_model.py`
2. **Evaluate performance** using `test_model.py`
3. **Experiment with webcam** using `realtime_inference.py`
4. **Deploy web app** using `app.py`
5. **Fine-tune hyperparameters** for better accuracy

---

## References

- Goodfellow, I.J., et al. (2013). *Challenges in Representation Learning: A report on three machine learning contests.* ICML. [arXiv:1307.0414]
- Howard, A.G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* [arXiv:1704.04861]
- Simonyan, K. & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* [arXiv:1409.1556]
- Viola, P. & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade.*

---

**For more details, see [README.md](README.md)**
