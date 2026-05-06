"""
test_model.py
=============
Evaluation & Testing Suite for Emotion Recognition CNN
-------------------------------------------------------
Comprehensive evaluation including:
  - Per-class precision, recall, F1
  - Confusion matrix (normalized & absolute)
  - Top-K accuracy
  - Inference speed benchmarking
  - Single image prediction demo
  - Misclassification analysis

Usage:
    # Evaluate on FER2013 test set:
    python test_model.py --model models/best_model.keras --data path/to/fer2013.csv

    # Run on synthetic data (no CSV needed):
    python test_model.py --model models/best_model.keras --synthetic

    # Run a quick architecture test (no model file needed):
    python test_model.py --arch_only
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix,
    top_k_accuracy_score, accuracy_score
)

from model import build_emotion_model, compile_model, EMOTION_LABELS, IMG_SIZE
from data_loader import (
    load_fer2013_csv, preprocess_images, generate_synthetic_fer2013,
    NUM_CLASSES, EMOTION_COLORS
)

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Emotion Recognition CNN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='models/best_model.keras',
                        help='Path to trained .keras model')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to fer2013.csv')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for evaluation')
    parser.add_argument('--arch_only', action='store_true',
                        help='Only test architecture (no model file needed)')
    parser.add_argument('--n_synthetic', type=int, default=2000,
                        help='Synthetic sample count')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a single image for prediction demo')
    return parser.parse_args()


def plot_confusion_matrix(y_true, y_pred, save_path='logs/confusion_matrix.png'):
    """Plot both normalized and absolute confusion matrices side by side."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Confusion Matrix — Emotion Recognition CNN', fontsize=14, fontweight='bold')

    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('Absolute Counts', fontsize=11)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Normalized (recall per class)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title('Normalized (Recall per Class)', fontsize=11)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Evaluation] Confusion matrix saved: {save_path}")
    plt.close()
    return cm


def plot_per_class_metrics(report_dict, save_path='logs/per_class_metrics.png'):
    """Horizontal bar chart of precision, recall, F1 per emotion class."""
    classes = EMOTION_LABELS
    precision = [report_dict[c]['precision'] for c in classes]
    recall    = [report_dict[c]['recall']    for c in classes]
    f1        = [report_dict[c]['f1-score']  for c in classes]
    support   = [report_dict[c]['support']   for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Per-Class Performance Metrics', fontsize=14, fontweight='bold')

    # Precision / Recall / F1 bars
    ax = axes[0]
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x,         recall,    width, label='Recall',    color='#FF5722', alpha=0.8)
    bars3 = ax.bar(x + width, f1,        width, label='F1-Score',  color='#4CAF50', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Precision / Recall / F1 per Class')
    ax.legend()
    ax.axhline(y=0.655, color='gray', ls=':', lw=1.5, label='Human Acc.')
    ax.grid(axis='y', alpha=0.3)

    # Support (# samples per class)
    ax2 = axes[1]
    colors = [EMOTION_COLORS[c] for c in classes]
    ax2.barh(classes, support, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Test Samples')
    ax2.set_title('Test Set Class Support')
    ax2.grid(axis='x', alpha=0.3)
    for i, (s, c) in enumerate(zip(support, classes)):
        ax2.text(s + 5, i, str(s), va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Evaluation] Per-class metrics saved: {save_path}")
    plt.close()


def benchmark_inference_speed(model, n_runs=200, batch_sizes=(1, 8, 32)):
    """Benchmark model inference speed for real-time feasibility check."""
    print("\n[Benchmark] Inference Speed Test")
    print("-" * 50)
    print(f"{'Batch Size':>12} | {'Mean (ms)':>10} | {'Std (ms)':>8} | {'FPS':>8}")
    print("-" * 50)

    results = {}
    for bs in batch_sizes:
        dummy = np.random.randn(bs, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)
        # Warm-up
        for _ in range(10):
            _ = model(dummy, training=False)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy, training=False)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        mean_ms = np.mean(times)
        std_ms  = np.std(times)
        fps     = bs / (mean_ms / 1000)

        print(f"{bs:>12} | {mean_ms:>10.2f} | {std_ms:>8.2f} | {fps:>8.1f}")
        results[bs] = {'mean_ms': mean_ms, 'std_ms': std_ms, 'fps': fps}

    print("-" * 50)
    single_fps = results[1]['fps']
    real_time_ok = single_fps >= 25.0
    print(f"\n  Real-time feasibility (≥25 FPS at batch=1): {'✓ YES' if real_time_ok else '✗ NO'}")
    return results


def predict_single_image(model, image_path):
    """
    Run prediction on a single image file.
    Handles color images by converting to grayscale and resizing.
    """
    import cv2
    if not os.path.exists(image_path):
        print(f"[Error] Image not found: {image_path}")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[Error] Could not read image: {image_path}")
        return

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_input = img_norm[np.newaxis, ..., np.newaxis]  # (1, 48, 48, 1)

    probs = model(img_input, training=False).numpy()[0]
    pred_class = np.argmax(probs)

    print(f"\n[Prediction] Image: {image_path}")
    print(f"  Predicted emotion: {EMOTION_LABELS[pred_class]} ({probs[pred_class]*100:.1f}%)")
    print(f"\n  All class probabilities:")
    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, probs)):
        bar = '█' * int(prob * 30)
        marker = ' ←' if i == pred_class else ''
        print(f"    {label:>10}: {prob*100:5.1f}%  {bar}{marker}")


def run_full_evaluation(model, X_test, y_test_ohe, y_test_raw):
    """Run complete evaluation suite."""
    print("\n[Evaluation] Running comprehensive evaluation...")

    # Get predictions
    test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(64)
    y_probs = model.predict(test_ds, verbose=1)
    y_pred  = np.argmax(y_probs, axis=1)
    y_true  = y_test_raw

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    top2_acc = top_k_accuracy_score(y_true, y_probs, k=2)
    top3_acc = top_k_accuracy_score(y_true, y_probs, k=3)

    print(f"\n{'='*50}")
    print("  EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Top-1 Accuracy:  {acc*100:.2f}%")
    print(f"  Top-2 Accuracy:  {top2_acc*100:.2f}%")
    print(f"  Top-3 Accuracy:  {top3_acc*100:.2f}%")
    print(f"  Human Accuracy:  ~65.5% (FER2013 benchmark)")
    print(f"{'='*50}")

    # Per-class report
    print("\n[Evaluation] Classification Report:")
    report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS, digits=4)
    print(report)

    # Save report
    with open('logs/classification_report.txt', 'w') as f:
        f.write("Real-Time Human Emotion Recognition — Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Top-1 Accuracy: {acc*100:.2f}%\n")
        f.write(f"Top-2 Accuracy: {top2_acc*100:.2f}%\n")
        f.write(f"Top-3 Accuracy: {top3_acc*100:.2f}%\n\n")
        f.write(report)
    print("[Evaluation] Report saved: logs/classification_report.txt")

    # Confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred)

    # Per-class metrics chart
    report_dict = classification_report(
        y_true, y_pred, target_names=EMOTION_LABELS, output_dict=True
    )
    plot_per_class_metrics(report_dict)

    # Most confused pairs
    print("\n[Evaluation] Top confused class pairs:")
    np.fill_diagonal(cm, 0)
    for _ in range(5):
        max_idx = np.unravel_index(cm.argmax(), cm.shape)
        true_cls, pred_cls = max_idx
        count = cm[true_cls, pred_cls]
        if count == 0:
            break
        print(f"  True={EMOTION_LABELS[true_cls]:>10} → Pred={EMOTION_LABELS[pred_cls]:>10}: {count} cases")
        cm[true_cls, pred_cls] = 0

    return acc, top2_acc, report_dict


def test_architecture_only():
    """Test model builds, compiles, and runs a forward pass without any data."""
    print("[ArchTest] Building model...")
    model = build_emotion_model()
    model = compile_model(model)
    model.summary()

    print(f"\n[ArchTest] Parameters: {model.count_params():,}")

    # Forward pass test with various batch sizes
    for bs in [1, 4, 16]:
        dummy = np.random.randn(bs, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)
        out = model(dummy, training=False)
        probs_sum = tf.reduce_sum(out, axis=-1)
        assert out.shape == (bs, NUM_CLASSES), f"Wrong output shape: {out.shape}"
        assert np.allclose(probs_sum.numpy(), 1.0, atol=1e-5), "Probs don't sum to 1"
        print(f"  Batch={bs}: output={out.shape} ✓ (probs sum={probs_sum.numpy().mean():.6f})")

    # Test inference speed
    benchmark_inference_speed(model)

    print("\n✓ Architecture test PASSED")
    return model


def main():
    args = parse_args()

    print("=" * 60)
    print("  Emotion Recognition CNN — Evaluation & Testing")
    print("=" * 60)
    print()

    # Architecture-only test mode
    if args.arch_only:
        test_architecture_only()
        return

    # Load model
    if not os.path.exists(args.model):
        print(f"[Warning] Model not found at '{args.model}'.")
        print("[Info] Running architecture test instead (train first to get a model).")
        test_architecture_only()
        return

    print(f"[Evaluation] Loading model: {args.model}")
    model = keras.models.load_model(args.model)
    print(f"[Evaluation] Model loaded. Parameters: {model.count_params():,}")

    # Load data
    if args.synthetic or args.data is None:
        print("[Evaluation] Using synthetic test data...")
        X_tr, y_tr, X_v, y_v, X_te, y_te = generate_synthetic_fer2013(args.n_synthetic)
    elif os.path.exists(args.data):
        X_tr, y_tr, X_v, y_v, X_te, y_te = load_fer2013_csv(args.data)
    else:
        print(f"[Error] CSV not found: {args.data}")
        sys.exit(1)

    (_, _, _,
     _, _, _,
     X_test, y_test_ohe, y_test_raw) = preprocess_images(X_tr, y_tr, X_v, y_v, X_te, y_te)

    # Full evaluation
    acc, top2_acc, report = run_full_evaluation(model, X_test, y_test_ohe, y_test_raw)

    # Inference speed benchmark
    benchmark_inference_speed(model)

    # Single image prediction (if provided)
    if args.image:
        predict_single_image(model, args.image)

    print(f"\n✓ Evaluation complete!")
    print(f"  Final test accuracy: {acc*100:.2f}%")


if __name__ == '__main__':
    main()