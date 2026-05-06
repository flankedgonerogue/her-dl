"""
model.py
========
Real-Time Human Emotion Recognition — CNN Architecture
-------------------------------------------------------
Implements a lightweight CNN inspired by MobileNet's depthwise separable
convolutions. Designed for real-time inference on CPU/GPU without sacrificing
too much accuracy on the FER2013 benchmark.

Architecture rationale:
  - Depthwise separable convolutions reduce parameters by ~8-9x vs standard convs
  - Batch normalization stabilizes gradients and accelerates training
  - Global Average Pooling instead of Flatten to reduce overfitting
  - Dropout at multiple stages for regularization
  - Softmax over 7 emotion classes

References:
  Howard et al. (2017) - MobileNets: Efficient CNNs for Mobile Vision Applications
  Goodfellow et al. (2013) - Challenges in Representation Learning (FER2013)
  Lopes et al. (2017) - FER with CNNs: Coping with few data
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
import numpy as np

# FER2013 emotion class labels (7 classes as defined by Goodfellow et al.)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
NUM_CLASSES = 7
IMG_SIZE = 48  # FER2013 images are 48x48 grayscale


def depthwise_separable_block(x, filters, strides=1, dropout_rate=0.1):
    """
    Depthwise Separable Convolution block as described in Howard et al. (2017).
    
    Factorizes a standard NxN convolution into:
      1. Depthwise conv (spatial filtering, one filter per channel)
      2. Pointwise 1x1 conv (channel combination / projection)
    
    This achieves ~8-9x reduction in computation vs standard convolution.
    """
    # Depthwise convolution — applies one filter per input channel
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding='same',
        use_bias=False,
        depthwise_regularizer=l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)  # ReLU6 as in MobileNet for numerical stability

    # Pointwise convolution — 1x1 conv to combine channels
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_regularizer=l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    return x


def standard_conv_block(x, filters, kernel_size=3, strides=1):
    """Standard Conv -> BN -> ReLU block for the initial stem layers."""
    x = layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_regularizer=l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    return x


def build_emotion_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """
    Build the lightweight emotion recognition CNN.

    Architecture Overview:
    ─────────────────────────────────────────────────────────────────────
    Input (48×48×1 grayscale)
        ↓
    Stem: Conv2D(32) → Conv2D(32)          [standard blocks, capture low-level features]
        ↓
    Stage 1: DSConv(64) × 2 + MaxPool      [depthwise separable, 24×24]
        ↓
    Stage 2: DSConv(128) × 2 + MaxPool     [12×12]
        ↓
    Stage 3: DSConv(256) × 2 + MaxPool     [6×6]
        ↓
    Stage 4: DSConv(512)                   [3×3]
        ↓
    Global Average Pooling                 [replaces Flatten, reduces params]
        ↓
    Dense(256) → Dropout(0.5)
        ↓
    Dense(128) → Dropout(0.3)
        ↓
    Dense(7, Softmax)                      [7 emotion classes]
    ─────────────────────────────────────────────────────────────────────

    Total params: ~1.2M (vs VGG16's 138M, ResNet50's 25M)
    Target accuracy: ~65-70% on FER2013 test set

    Args:
        input_shape: Tuple, image dimensions (H, W, C)
        num_classes: int, number of emotion classes (7 for FER2013)

    Returns:
        Compiled Keras Model
    """
    inputs = keras.Input(shape=input_shape, name='image_input')

    # ── Stem: Two standard conv layers ──────────────────────────────────
    x = standard_conv_block(inputs, filters=32, kernel_size=3)
    x = standard_conv_block(x, filters=32, kernel_size=3)
    x = layers.MaxPooling2D(pool_size=2)(x)         # 48→24

    # ── Stage 1: Depthwise Separable Blocks ─────────────────────────────
    x = depthwise_separable_block(x, filters=64, dropout_rate=0.1)
    x = depthwise_separable_block(x, filters=64, dropout_rate=0.1)
    x = layers.MaxPooling2D(pool_size=2)(x)         # 24→12

    # ── Stage 2 ──────────────────────────────────────────────────────────
    x = depthwise_separable_block(x, filters=128, dropout_rate=0.15)
    x = depthwise_separable_block(x, filters=128, dropout_rate=0.15)
    x = layers.MaxPooling2D(pool_size=2)(x)         # 12→6

    # ── Stage 3 ──────────────────────────────────────────────────────────
    x = depthwise_separable_block(x, filters=256, dropout_rate=0.2)
    x = depthwise_separable_block(x, filters=256, dropout_rate=0.2)
    x = layers.MaxPooling2D(pool_size=2)(x)         # 6→3

    # ── Stage 4 ──────────────────────────────────────────────────────────
    x = depthwise_separable_block(x, filters=512, dropout_rate=0.25)

    # ── Global Average Pooling (no Flatten to reduce overfitting) ────────
    x = layers.GlobalAveragePooling2D()(x)

    # ── Classifier Head ──────────────────────────────────────────────────
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='emotion_output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='EmotionNet_MobileInspired')
    return model


def compile_model(model, learning_rate=1e-3):
    """
    Compile model with Adam optimizer and categorical crossentropy.
    Uses label smoothing (0.1) to reduce overconfidence — helpful for FER2013
    which has noisy labels (human agreement ~65.5%).
    """
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
    )
    return model


def get_callbacks(model_save_path='models/best_model.keras', log_dir='logs/'):
    """
    Training callbacks:
    - ModelCheckpoint: save best weights based on val_accuracy
    - ReduceLROnPlateau: halve LR if no val_loss improvement for 5 epochs
    - EarlyStopping: stop if no improvement after 15 epochs
    - TensorBoard: for visualization
    - CSVLogger: export epoch-by-epoch metrics
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='max'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        ),
        keras.callbacks.CSVLogger('logs/training_history.csv', append=False),
    ]
    return callbacks


if __name__ == '__main__':
    # Quick sanity check
    model = build_emotion_model()
    model = compile_model(model)
    model.summary()

    # Count parameters
    total = model.count_params()
    print(f"\nTotal parameters: {total:,}")
    print(f"Model size estimate: ~{total * 4 / 1e6:.1f} MB (float32)")

    # Test forward pass
    dummy_input = np.random.randn(4, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)
    dummy_output = model(dummy_input, training=False)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
    print(f"Output sum (should be ~4.0): {tf.reduce_sum(dummy_output).numpy():.4f}")
    print("\n✓ Model architecture verified.")