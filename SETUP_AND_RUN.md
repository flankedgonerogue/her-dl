# 🚀 Setup & Run Guide — Real-Time Human Emotion Recognition

This guide covers installation, dataset layout, baseline training, transfer-learning training, evaluation, and runtime inference.

---

## 1) Install dependencies

```bash
cd /home/arg/Documents/her-dl
pip install -r requirements.txt
```

> If you use a virtual environment, activate it first.

---

## 2) FER-2013 dataset layout

The project expects the Kaggle FER-2013 folder structure:

```text
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Download the dataset from Kaggle and place the extracted `fer2013/` directory in the project root.

---

## 3) Baseline CNN training

Train the existing grayscale CNN:

```bash
python train_model.py --data fer2013 --epochs 100 --batch_size 64
```

Useful options:

```bash
python train_model.py --synthetic --epochs 10
python train_model.py --data fer2013 --resume models/best_model.keras --epochs 150
python train_model.py --data fer2013 --lr 0.0001 --batch_size 32 --epochs 100
```

Outputs:

- `models/best_model.keras`
- `models/best_model.metadata.json`
- `logs/training_curves.png`
- `logs/training_metadata.json`
- `logs/training_history.csv`

---

## 4) Transfer-learning training

Train the MobileNetV2 transfer-learning model in two stages:

```bash
python train_transfer.py --data fer2013 --head_epochs 10 --fine_tune_epochs 10 --model_path models/transfer_best_model.keras
```

Useful options:

```bash
python train_transfer.py --synthetic --synthetic_n 4000 --head_epochs 3 --fine_tune_epochs 3
python train_transfer.py --data fer2013 --fine_tune_at -20 --head_epochs 8 --fine_tune_epochs 8
python train_transfer.py --data fer2013 --input_size 96 --batch_size 32 --head_lr 0.001 --fine_tune_lr 0.00001
```

Outputs:

- `models/transfer_best_model.keras`
- `models/transfer_best_model.metadata.json`
- `logs/transfer/training_curves.png`
- `logs/transfer/training_metadata.json`
- `logs/transfer/stage1/history.csv`
- `logs/transfer/stage2/history.csv`

---

## 5) Evaluation and comparison

Evaluate a single model:

```bash
python test_model.py --model models/best_model.keras --data fer2013
python test_model.py --model models/transfer_best_model.keras --data fer2013
```

Compare both models side by side:

```bash
python test_model.py --model models/best_model.keras --compare_model models/transfer_best_model.keras --data fer2013
```

Evaluation outputs:

- accuracy
- top-2 accuracy
- classification report
- confusion matrix
- per-class precision/recall/F1
- inference-speed benchmark
- optional single-image prediction

Generated files are saved under `logs/`.

---

## 6) Real-time webcam inference

Run the baseline model:

```bash
python realtime_inference.py --model models/best_model.keras
```

Run the transfer-learning model:

```bash
python realtime_inference.py --model models/transfer_best_model.keras
```

Demo mode without webcam:

```bash
python realtime_inference.py --demo
```

Keyboard controls:

- `Q` — quit
- `S` — save screenshot
- `P` — pause/resume

The runtime reads model metadata automatically and adapts preprocessing for the baseline and transfer model.

---

## 7) Web application

Launch the Flask app:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

The app can serve either model type as long as the corresponding `.keras` file and `.metadata.json` sidecar are present.

---

## 8) Expected comparison

| Model | Input | Speed | Accuracy goal |
|---|---|---|---|
| Baseline CNN | 48×48×1 | Fastest | Solid baseline |
| Transfer model | 96×96×3 | Slower | Best accuracy |

---

## 9) Troubleshooting

### `pip install -r requirements.txt` fails

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset folder not found

Make sure the folder is named `fer2013/` and contains `train/` and `test/` subfolders.

### Webcam does not open

Use demo mode:

```bash
python realtime_inference.py --demo
```

### TensorFlow does not see the GPU

TensorFlow will fall back to CPU automatically. Check your CUDA / cuDNN install if you need GPU acceleration.

---

## 10) Recommended workflow

1. Train the baseline model.
2. Train the transfer-learning model.
3. Compare both with `test_model.py`.
4. Run real-time inference on the better model.
5. Deploy the Flask app if you want browser-based predictions.
