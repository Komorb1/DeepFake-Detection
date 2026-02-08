# DeepFake-Detection

## Overview

This repository contains the implementation of a deepfake video detection system using neural networks that combine RGB spatial features and Discrete Cosine Transform (DCT) frequency-domain features. The project is based on a research paper titled "Analysis of Deepfake video detection using Neural Networks on Discrete cosine transform and RGB features" from Istanbul Arel University.

The model detects manipulated videos (deepfakes) by analyzing facial regions in videos, extracting RGB and DCT features, and classifying them as real or fake using a hybrid neural network architecture. It was trained and evaluated on the FaceForensics++ dataset (C23 compression) and tested for cross-dataset robustness on Celeb-DF.

### Key Features
- **Dual-Domain Approach**: Combines spatial (RGB) and frequency (DCT) features for improved detection robustness.
- **Dataset**: FaceForensics++ (400 videos: 200 real, 200 fake) with splits: 70% train, 15% validation, 15% test.
- **Model Architecture**: EfficientNetB0 backbone for RGB branch + MLP for DCT branch, fused for binary classification.
- **Performance**: Achieves ~0.79 accuracy on FaceForensics++ test set; ~0.64 on Celeb-DF (cross-dataset).
- **Explainability**: Includes Grad-CAM visualizations for interpreting model decisions.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- OpenCV (cv2)
- dlib (for face detection)
- NumPy
- SciPy (for DCT computation)
- Matplotlib (for visualizations)

Install dependencies:
```
pip install tensorflow keras opencv-python dlib numpy scipy matplotlib
```

Note: Due to hardware constraints in the original experiments, training was performed on CPU. GPU acceleration is recommended for faster training.

## Dataset

- **FaceForensics++**: Download from the official repository (DeepFakes subset, C23 compression). Place videos in a directory structured as `real/` and `fake/`.
- **Celeb-DF**: Download for cross-dataset evaluation. Used for testing only (no retraining).

The dataset is preprocessed to extract 15 frames per video, detect faces, resize to 224x224 (RGB) and 64x64 (DCT), and aggregate features.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Komorb1/DeepFake-Detection.git
   cd DeepFake-Detection
   ```

2. Navigate to the model directory:
   ```
   cd model
   ```

3. Set up your environment and install requirements as above.

## Usage

### Preprocessing
Run the preprocessing script to prepare RGB and DCT features from videos:
```python
# Example: preprocess.py (adapt paths as needed)
import preprocess  # Assuming preprocess module exists
preprocess.process_videos(input_dir='path/to/videos', output_dir='path/to/processed')
```
This generates RGB images (224x224x3) and DCT vectors (4096-dim) for training.

### Training
Train the hybrid model:
```python
# Example: train.py
from model import build_hybrid_model
model = build_hybrid_model()
model.fit(train_data, epochs=20, validation_data=val_data)
model.save('deepfake_detector.h5')
```
- Uses Adam optimizer (lr=1e-5), binary cross-entropy loss.
- Augmentation: Random horizontal flips and brightness adjustments on RGB.
- Fine-tuning: Unfreeze last layers of EfficientNetB0.

### Evaluation
Evaluate on test set:
```python
# Example: evaluate.py
from model import load_model
model = load_model('deepfake_detector.h5')
metrics = model.evaluate(test_data)
print(metrics)  # Accuracy, Precision, Recall, F1, AUC
```
- Threshold tuning: Use balanced accuracy on validation set (e.g., τ=0.49).
- Cross-dataset: Apply the same model and threshold to Celeb-DF.

### Inference
Detect deepfakes on new videos:
```python
# Example: infer.py
from inference import detect_deepfake
result = detect_deepfake(model, 'path/to/video.mp4')
print("Fake" if result > 0.5 else "Real")
```
- Processes video frames, extracts features, and averages predictions.

### Explainability (Grad-CAM)
Generate heatmaps:
```python
# Example: gradcam.py
from explain import generate_gradcam
heatmap = generate_gradcam(model, image_path='path/to/frame.jpg')
# Visualize overlay
```

## Model Details

- **RGB Branch**: EfficientNetB0 (pre-trained on ImageNet), input: 224x224x3 (center frame).
- **DCT Branch**: 64x64 DCT maps averaged over 15 frames, flattened to 4096-dim, processed by MLP (512 units x2 + Dropout).
- **Fusion**: Concatenate embeddings (1280 + 512 = 1792-dim), Dense-128 + Dropout, sigmoid output.
- **Parameters**: ~6.6M total, ~3.9M trainable.
- **Training Strategy**: Baseline (frozen backbone) + Fine-tuning (unfreeze last layers).
- **Inference Optimization**: Frame averaging (e.g., 3 frames) for stability.

## Results

### FaceForensics++ Test (τ=0.49)
| Class   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| REAL    | 0.65      | 0.61   | 0.63     |
| FAKE    | 0.63      | 0.67   | 0.65     |
| Accuracy| -         | -      | 0.64     |

### Celeb-DF Cross-Dataset (τ=0.54)
| Class   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| REAL    | 0.68      | 0.53   | 0.60     |
| FAKE    | 0.61      | 0.76   | 0.68     |
| Accuracy| -         | -      | 0.64     |

Confusion matrices and Grad-CAM examples are available in the paper.

## Limitations and Ethics

- **Limitations**: Performance drops on cross-dataset due to domain shift. Sensitive to compression and post-processing.
- **Ethics**: This tool is for research/decision support only. Avoid misuse for falsely accusing media. Report probabilities and keep humans in the loop.

## References

See the full list in the paper (e.g., FaceForensics++, EfficientNet, Grad-CAM).

## Contributors

- Muhammad Akbar (210303894)
- Omar Al-Kebsi (220303916)
- Khaled M.S Ahmad (210303882)

For questions, open an issue or contact via GitHub.

---

This README is based on the research paper provided. Update paths, scripts, and details as needed for the actual repo structure.
