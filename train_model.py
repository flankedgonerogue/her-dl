"""
train_model.py
==============
Training Script — Real-Time Human Emotion Recognition CNN
----------------------------------------------------------
Full training pipeline for the lightweight CNN on FER2013.

Usage:
    # With real FER2013 CSV (download from Kaggle):
    python train_model.py --data path/to/fer2013.csv --epochs 100 --batch_size 64

    # Quick test with synthetic data (no CSV needed):
    python train_model.py --synthetic --epochs 5

    # Resume from checkpoint:
    python train_model.py --data path/to/fer2013.csv --resume models/best_model.keras

Training strategy:
    Phase 1 (epochs 1-30):   LR=1e-3, Adam, full model train
    Phase 2 (epochs 31-70):  LR reduces via ReduceLROnPlateau
    Phase 3 (epochs 71-100): Fine-tuning with LR=1e-5

Expected results on real FER2013:
    Test accuracy: ~65-70% (human accuracy on FER2013 is ~65.5%)
    Top-2 accuracy: ~85-90%

References:
    - Goodfellow et al. (2013) — FER2013 dataset & challenge
    - Lopes et al. (2017) — FER with CNNs, coping with few data
    - Howard et al. (2017) — MobileNets architecture
    - Simonyan & Zisserman (2014) — VGG: very deep conv networks
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Local modules
from model import build_emotion_model, compile_model, get_callbacks, EMOTION_LABELS
from data_loader import (
    load_fer2013_csv, preprocess_images, compute_class_weights,
    create_tf_datasets, generate_synthetic_fer2013, visualize_samples, NUM_CLASSES
)

os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Emotion Recognition CNN on FER2013',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data', type=str, default=None,
                        help='Path to fer2013.csv')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing (no CSV needed)')
    parser.add_argument('--synthetic_n', type=int, default=5000,
                        help='Number of synthetic samples')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to .keras model to resume training from')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--model_path', type=str, default='models/best_model.keras',
                        help='Where to save the best model')
    return parser.parse_args()


def load_data(args):
    """Load data from CSV or generate synthetic."""
    if args.synthetic or args.data is None:
        if args.data is not None and not os.path.exists(args.data):
            print(f"[Warning] FER2013 CSV not found at '{args.data}', using synthetic data.")
        else:
            print("[Info] Using synthetic data (--synthetic flag set).")

        X_tr, y_tr, X_v, y_v, X_te, y_te = generate_synthetic_fer2013(args.synthetic_n)
    else:
        if not os.path.exists(args.data):
            print(f"[Error] CSV not found: {args.data}")
            print("Download FER2013 from: https://www.kaggle.com/datasets/msambare/fer2013")
            sys.exit(1)
        X_tr, y_tr, X_v, y_v, X_te, y_te = load_fer2013_csv(args.data)

    return preprocess_images(X_tr, y_tr, X_v, y_v, X_te, y_te)


def plot_training_history(history, save_path='logs/training_curves.png'):
    """Plot accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History — EmotionNet', fontsize=14, fontweight='bold')

    # Accuracy
    ax = axes[0]
    ax.plot(history.history['accuracy'],     label='Train Accuracy',  color='#2196F3', lw=2)
    ax.plot(history.history['val_accuracy'], label='Val Accuracy',    color='#FF5722', lw=2, ls='--')
    if 'top_2_accuracy' in history.history:
        ax.plot(history.history['top_2_accuracy'],
                label='Train Top-2 Acc', color='#4CAF50', lw=1.5, alpha=0.7)
        ax.plot(history.history['val_top_2_accuracy'],
                label='Val Top-2 Acc',   color='#FF9800', lw=1.5, ls='--', alpha=0.7)
    ax.axhline(y=0.655, color='gray', ls=':', lw=1.5, label='Human Accuracy (65.5%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Loss
    ax = axes[1]
    ax.plot(history.history['loss'],     label='Train Loss', color='#2196F3', lw=2)
    ax.plot(history.history['val_loss'], label='Val Loss',   color='#FF5722', lw=2, ls='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Training] Curves saved to: {save_path}")
    plt.close()


def save_training_metadata(args, history, test_results, save_path='logs/training_metadata.json'):
    """Save all training metadata for reproducibility."""
    best_val_acc = max(history.history.get('val_accuracy', [0]))
    metadata = {
        'model': 'EmotionNet_MobileInspired',
        'dataset': 'FER2013' if not args.synthetic else 'Synthetic',
        'epochs_run': len(history.history['accuracy']),
        'max_epochs': args.epochs,
        'batch_size': args.batch_size,
        'initial_lr': args.lr,
        'best_val_accuracy': float(best_val_acc),
        'test_loss': float(test_results[0]),
        'test_accuracy': float(test_results[1]),
        'emotion_classes': EMOTION_LABELS,
        'architecture': {
            'input_size': '48x48x1',
            'depthwise_separable_convolutions': True,
            'global_average_pooling': True,
            'regularization': 'L2 + Dropout',
            'label_smoothing': 0.1,
        },
        'training_time_note': 'See logs/training_history.csv for per-epoch data'
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[Training] Metadata saved to: {save_path}")
    return metadata


def main():
    args = parse_args()

    print("=" * 60)
    print("  Real-Time Human Emotion Recognition — Training")
    print("  Group: Abdur Rahman Goraya (2022035), Ahsan Naqvi (2022073)")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print()

    # ── 1. Load & preprocess data ────────────────────────────────────────
    print("[Step 1/5] Loading and preprocessing data...")
    (X_train, y_train, y_train_raw,
     X_val,   y_val,   y_val_raw,
     X_test,  y_test,  y_test_raw) = load_data(args)

    # Visualize samples
    try:
        visualize_samples(X_train, y_train_raw, save_path='logs/sample_grid.png')
    except Exception as e:
        print(f"[Warning] Could not save sample grid: {e}")

    # ── 2. Build / load model ────────────────────────────────────────────
    print("\n[Step 2/5] Building model...")
    if args.resume and os.path.exists(args.resume):
        print(f"[Training] Resuming from: {args.resume}")
        model = keras.models.load_model(args.resume)
    else:
        model = build_emotion_model()
        model = compile_model(model, learning_rate=args.lr)

    model.summary()
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,} (~{total_params*4/1e6:.1f} MB)")

    # ── 3. Compute class weights ─────────────────────────────────────────
    print("\n[Step 3/5] Computing class weights...")
    class_weights = compute_class_weights(y_train_raw)

    # ── 4. Create tf.data pipelines ──────────────────────────────────────
    print("\n[Step 4/5] Building data pipelines...")
    train_ds, val_ds = create_tf_datasets(X_train, y_train, X_val, y_val,
                                           batch_size=args.batch_size)

    # ── 5. Train ─────────────────────────────────────────────────────────
    print(f"\n[Step 5/5] Training for up to {args.epochs} epochs...")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Model saved:   {args.model_path}")
    print()

    callbacks = get_callbacks(model_save_path=args.model_path)
    start_time = time.time()

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    elapsed = time.time() - start_time
    print(f"\n[Training] Completed in {elapsed/60:.1f} minutes")

    # ── Evaluation on test set ───────────────────────────────────────────
    print("\n[Evaluation] Loading best model weights for final test evaluation...")
    if os.path.exists(args.model_path):
        best_model = keras.models.load_model(args.model_path)
    else:
        best_model = model

    print("[Evaluation] Evaluating on held-out test set...")
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)
    test_results = best_model.evaluate(test_ds, verbose=1)

    print("\n" + "=" * 40)
    print("  FINAL TEST RESULTS")
    print("=" * 40)
    metric_names = best_model.metrics_names
    for name, val in zip(metric_names, test_results):
        print(f"  {name:>20}: {val:.4f}")
    print(f"\n  Human accuracy on FER2013: ~65.5%")
    print("=" * 40)

    # ── Save artifacts ───────────────────────────────────────────────────
    plot_training_history(history)
    metadata = save_training_metadata(args, history, test_results)

    # Also save in TF SavedModel format for serving
    saved_model_path = 'models/emotion_model_savedmodel'
    best_model.export(saved_model_path)
    print(f"[Saving] SavedModel exported to: {saved_model_path}")

    print("\n✓ Training complete!")
    print(f"  Best model: {args.model_path}")
    print(f"  Training curves: logs/training_curves.png")
    print(f"  Metadata: logs/training_metadata.json")


if __name__ == '__main__':
    main()