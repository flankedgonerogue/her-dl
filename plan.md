# Transfer Learning Plan for FER-2013

## Objective
Implement a transfer-learning pipeline for FER-2013, then fine-tune the model on the dataset in `her-dl/fer2013` while keeping the current custom CNN baseline intact.

## Research notes
- FER-2013 is a 7-class emotion recognition dataset with 48x48 grayscale face images.
- The Kaggle FER layout typically uses `train/<class>` and `test/<class>` folders.
- The dataset is imbalanced, especially for `disgust`, so class weighting and augmentation are important.
- Keras transfer-learning guidance recommends:
  1. load a pretrained base model,
  2. freeze it,
  3. train a new classifier head,
  4. unfreeze the top layers,
  5. fine-tune with a very low learning rate.
- `MobileNetV2` is a good first backbone because it is lightweight and fits the real-time goal.

## Implementation plan

### 1) Add a transfer-learning model path
- Add a new model builder in `model.py` for a pretrained backbone such as `MobileNetV2`.
- Use `include_top=False` and ImageNet weights.
- Replace the head with:
  - global average pooling,
  - batch normalization,
  - dropout,
  - dense classifier for 7 emotions.
- Keep the existing custom CNN as the baseline.

### 2) Add transfer-specific preprocessing
- Create a preprocessing path for transfer learning in `data_loader.py`.
- Convert grayscale FER images to RGB.
- Resize to the backbone input size, likely `96x96` or `128x128`.
- Apply the backbone-specific preprocessing function.
- Keep augmentation moderate:
  - horizontal flip,
  - small rotation,
  - small zoom,
  - small translation,
  - light brightness/contrast jitter.

### 3) Add a two-stage training flow
- Stage 1: feature extraction
  - freeze the backbone,
  - train only the new head,
  - use a moderate learning rate,
  - keep class weights enabled.
- Stage 2: fine-tuning
  - unfreeze only the top layers of the backbone,
  - keep earlier layers frozen,
  - recompile the model,
  - train with a much lower learning rate.
- Freeze BatchNorm layers during fine-tuning unless there is a clear reason not to.

### 4) Add a dedicated training entry point
- Prefer a new script such as `train_transfer.py` so the current `train_model.py` stays stable.
- Add CLI options for:
  - backbone choice,
  - input size,
  - head-training epochs,
  - fine-tuning epochs,
  - fine-tuning start layer,
  - learning rates.
- Save checkpoints, logs, and metadata separately from the baseline model.

### 5) Update evaluation
- Extend `test_model.py` so it can evaluate both:
  - the custom grayscale CNN,
  - the transfer-learning model.
- Report:
  - accuracy,
  - top-2 accuracy,
  - classification report,
  - confusion matrix,
  - per-class precision/recall/F1,
  - inference speed.
- Compare the transfer model against the custom baseline.

### 6) Update inference paths
- Update `realtime_inference.py` and `app.py` to handle the new model input format.
- Save model metadata so inference can determine:
  - expected input size,
  - number of channels,
  - preprocessing method.
- Keep the custom model available for faster grayscale inference.

### 7) Document the workflow
- Update `README.md` and `SETUP_AND_RUN.md` with:
  - dataset layout,
  - training commands,
  - evaluation commands,
  - realtime inference commands,
  - baseline vs transfer-learning comparison.

## Suggested implementation order
1. Add the transfer-learning model builder.
2. Add preprocessing and dataset support for RGB transfer inputs.
3. Add a new training script for feature extraction and fine-tuning.
4. Run a short smoke test.
5. Run full training on FER-2013.
6. Update evaluation and inference.
7. Update documentation.

## Risks and mitigations
- **Overfitting during fine-tuning**: use low learning rates, early stopping, and dropout.
- **Class imbalance**: keep class weights and track per-class F1.
- **Slower inference**: start with MobileNetV2 and a smaller input size.
- **Grayscale to RGB mismatch**: explicitly convert and preprocess images for the pretrained backbone.

## Expected outcome
A transfer-learning model that matches or exceeds the custom CNN baseline while remaining practical for evaluation and real-time inference.