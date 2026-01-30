# Baseball Swing Anomaly Detection Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A semi-supervised anomaly detection system for identifying faulty baseball swings from motion sensor data. This project uses a **1D Convolutional Autoencoder** trained exclusively on "good" swings to learn normal movement patterns, then detects anomalous swings through reconstruction error analysis and an ensemble of machine learning methods.

## Key Results

| Method | Recall | Specificity | F1 Score | Accuracy |
|--------|--------|-------------|----------|----------|
| Threshold (95%) | 33.3% | 80.0% | 0.40 | 62.5% |
| Isolation Forest | 33.3% | 80.0% | 0.40 | 62.5% |
| One-Class SVM | 66.7% | 40.0% | 0.44 | 50.0% |
| **Ensemble (2/3)** | 33.3% | 80.0% | 0.40 | 62.5% |

---

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [Technical Background & Theory](#technical-background--theory)
   - [Autoencoders for Anomaly Detection](#autoencoders-for-anomaly-detection)
   - [1D Convolutional Neural Networks](#1d-convolutional-neural-networks)
   - [Group Normalization](#group-normalization)
   - [Curse of Dimensionality](#curse-of-dimensionality)
   - [Anomaly Detection Methods](#anomaly-detection-methods)
3. [Model Architecture](#model-architecture)
4. [Installation & Requirements](#installation--requirements)
5. [Usage Guide](#usage-guide)
6. [Results & Evaluation](#results--evaluation)
7. [Limitations & Future Work](#limitations--future-work)
8. [References](#references)
9. [License](#license)

---

## Dataset Description

### Data Source

The dataset consists of motion capture sensor data recorded during baseball batting practice. Each trial captures the electrical resistance changes from 8 flexible sensors attached to a smart glove or sleeve worn by the batter.

### Data Specifications

| Property | Value |
|----------|-------|
| Sampling Rate | 220 Hz |
| Number of Channels | 8 sensors (labeled 25-32) |
| Trial Duration | Variable (~4-5 seconds per swing) |
| Good Trials | 27 |
| Bad Trials | 3 |
| Data Format | CSV |

### Directory Structure

```
baseball_data/
├── full_swing_ball/
│   ├── good/
│   │   ├── Trial06.csv, Trial07.csv, ...
│   │   └── processed_data/
│   │       └── Trial06.csv, Trial07.csv, ...  (preprocessed)
│   └── bad/
│       ├── Trial18.csv, Trial20.csv, Trial32.csv
│       └── processed_data/
│           └── Trial18.csv, Trial20.csv, Trial32.csv
├── main.ipynb          # Main analysis notebook
├── plots/              # Generated visualizations
└── README.md
```

### Data Format

**Raw Data** (e.g., `full_swing_ball/good/Trial06.csv`):
- Contains voltage readings from the DAQ system
- Columns: `Frame`, `Sub Frame`, `25`, `26`, `27`, `28`, `29`, `30`, `31`, `32`
- Values are in Volts (V)

**Preprocessed Data** (e.g., `full_swing_ball/good/processed_data/Trial06.csv`):
- Scaled resistance values centered around zero
- Columns: `time`, `25`, `26`, `27`, `28`, `29`, `30`, `31`, `32`
- Values normalized to approximately [-1, 1] range

### What Defines a "Bad" Swing?

Bad swings in this dataset represent batting motions with significant deviations from proper technique, such as:
- Incorrect grip or hand positioning
- Poor timing in the swing mechanics
- Unusual body movement patterns
- Incomplete or interrupted swings

---

## Technical Background & Theory

### Autoencoders for Anomaly Detection

An **autoencoder** is a type of neural network that learns to compress data into a lower-dimensional representation (encoding) and then reconstruct the original input from this compressed form (decoding).

```
Input → [Encoder] → Latent Space → [Decoder] → Reconstructed Output
  X                     z                            X̂
```

#### Why Autoencoders Work for Anomaly Detection

The key insight is that when an autoencoder is trained **only on normal data**, it learns to reconstruct normal patterns well. When presented with an anomaly (data different from the training distribution), the reconstruction quality degrades, resulting in higher **reconstruction error**.

**Training Phase:**
1. Feed only "good" swing data to the autoencoder
2. The model learns the underlying patterns of proper swings
3. The encoder compresses temporal patterns into a latent representation
4. The decoder learns to reconstruct these patterns accurately

**Inference Phase:**
1. Feed a new (unseen) swing to the trained model
2. Calculate reconstruction error: `MSE = mean((X - X̂)²)`
3. High reconstruction error → likely anomaly
4. Low reconstruction error → likely normal

This approach is **semi-supervised**: we only need labels to separate training data (all normal) from test data (mixed), not for training the model itself.

> **Further Reading:** [Keras: Timeseries Anomaly Detection using Autoencoder](https://keras.io/examples/timeseries/timeseries_anomaly_detection/)

---

### 1D Convolutional Neural Networks

Traditional dense (fully-connected) networks treat each input feature independently. For time series data, this ignores the crucial **temporal relationships** between consecutive samples.

**1D Convolutional Neural Networks** address this by using learnable filters (kernels) that slide across the time dimension:

```
Input Signal:  [x₁, x₂, x₃, x₄, x₅, x₆, x₇, ...]
                 └─kernel─┘
                    ↓
Convolution:   [y₁, y₂, y₃, y₄, y₅, ...]
```

#### Advantages for Time Series

1. **Local Pattern Detection**: Kernels capture local temporal patterns (e.g., the acceleration phase of a swing)
2. **Translation Invariance**: A learned pattern is detected regardless of when it occurs
3. **Parameter Efficiency**: Shared kernel weights reduce model complexity
4. **Hierarchical Features**: Stacked conv layers learn increasingly abstract features

#### Architecture Components in This Project

- **Conv1D**: Applies filters across the temporal dimension
- **AvgPool1d**: Downsamples by averaging, reducing temporal resolution while preserving overall shape
- **Upsample**: Increases temporal resolution for the decoder
- **Kernel Sizes**: Larger kernels (k=7, k=5) capture broader patterns; smaller (k=3) for fine details

---

### Group Normalization

Normalization layers stabilize training by normalizing activations to have zero mean and unit variance. **Batch Normalization** (BN) is the most common choice, but it has a critical limitation for small batch sizes.

#### The Batch Size Problem

Batch Normalization computes mean and variance across the **batch dimension**:

```python
# Batch Norm: statistics computed over batch
mean = x.mean(dim=0)  # Average over samples in batch
variance = x.var(dim=0)
```

With small batches (like 16 samples as in our training), these statistics are noisy estimates of the true population statistics. With batch size 1, variance becomes zero, causing numerical instability.

#### How Group Normalization Works

**Group Normalization** (Wu & He, ECCV 2018) divides channels into groups and normalizes within each group, **independent of batch size**:

```
Channels: [c₁, c₂, c₃, c₄, c₅, c₆, c₇, c₈]
           └─Group 1─┘  └─Group 2─┘ ...

For each group: normalize over (channels in group, time)
```

In this project, we use `num_groups=min(8, num_channels)`, meaning channels are divided into up to 8 groups.

#### Why This Matters

| Batch Size | Batch Norm Error | Group Norm Error |
|------------|------------------|------------------|
| 32 | 22.0% | 22.4% |
| 2 | 31.9% | 23.0% |

Group Normalization maintains stable performance regardless of batch size, making it ideal for training with limited data.

> **Reference:** [Group Normalization (arXiv:1803.08494)](https://arxiv.org/abs/1803.08494)

---

### Curse of Dimensionality

The **curse of dimensionality** refers to problems that arise when working with high-dimensional data with limited samples.

#### The Problem in This Project

An earlier version of the anomaly detection extracted 49 features:
- Per-channel statistics (mean, std, max MSE × 8 channels = 24 features)
- Temporal statistics (25 additional features)

With only 17 training samples and 49 features, the **sample-to-feature ratio** was 17/49 ≈ 0.35:1.

This caused several issues:
1. **Sparse Data**: In high dimensions, data points become increasingly distant from each other
2. **Unreliable Distance Metrics**: All points appear equidistant, making similarity-based methods fail
3. **Overfitting**: Models memorize noise rather than learning patterns

#### The Solution: Feature Reduction

The improved approach uses only **3 features**:

| Feature | Description |
|---------|-------------|
| Overall MSE | Mean squared reconstruction error across all channels and timesteps |
| Max Channel Error | Highest per-channel MSE (identifies which sensor had worst reconstruction) |
| Channel Error Variance | Standard deviation of per-channel errors (captures if error is uniform or concentrated) |

This improves the sample-to-feature ratio to 17/3 ≈ 5.7:1, making the anomaly detection methods more reliable.

> **Further Reading:** [Curse of Dimensionality in Machine Learning](https://www.datacamp.com/blog/curse-of-dimensionality-machine-learning)

---

### Anomaly Detection Methods

This project implements an **ensemble** of three complementary anomaly detection methods:

#### Method 1: Threshold-Based Detection

The simplest approach: flag samples with reconstruction error above a threshold.

```python
threshold = np.percentile(train_mse, 95)  # 95th percentile of training MSE
prediction = "anomaly" if sample_mse > threshold else "normal"
```

**Pros:** Simple, interpretable, no training required
**Cons:** Sensitive to threshold choice, ignores feature relationships

#### Method 2: Isolation Forest

[Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) isolates anomalies by randomly selecting features and split values. The key insight: **anomalies are easier to isolate** (require fewer splits).

```
                    Root
                   /    \
              [Split]   ...
               /    \
          [Split]  ANOMALY (isolated early!)
           /    \
        Normal  Normal (requires more splits)
```

The **anomaly score** is based on the average path length across many random trees. Shorter paths = more anomalous.

**Pros:** Handles high dimensions well, fast, no distance metrics
**Cons:** May miss clustered anomalies

#### Method 3: One-Class SVM

[One-Class SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) learns a boundary around the normal data in a high-dimensional kernel space.

```
Feature Space:              Kernel Space:
    ●●●                        ╭─────────────╮
   ●●●●●  → RBF Kernel →      │   ●●●●●●●   │ (normal inside)
    ●●●                        │     ●●●     │
      ✕                        ╰─────✕───────╯ (anomaly outside)
```

The `nu` parameter (set to 0.1) controls the upper bound on the fraction of outliers expected.

**Pros:** Effective for small datasets, captures non-linear boundaries
**Cons:** Sensitive to kernel parameters, slower for large datasets

#### Ensemble Voting

To improve robustness, predictions are combined using **majority voting**:

```python
def ensemble_predict(X, threshold):
    vote_threshold = X[:, 0] > threshold  # Method 1
    vote_iforest = isolation_forest.predict(X)  # Method 2
    vote_svm = one_class_svm.predict(X)  # Method 3

    votes = sum([vote_threshold, vote_iforest, vote_svm])
    return "anomaly" if votes >= 2 else "normal"  # 2/3 majority
```

This reduces the impact of any single method's weaknesses and provides more consistent predictions.

> **Further Reading:** [Anomaly Detection with One-Class SVM and Isolation Forest](https://ai.plainenglish.io/detecting-anomalies-a-comprehensive-guide-with-one-class-svm-and-isolation-forest-230336f0988a)

---

## Model Architecture

### CAE1D: 1D Convolutional Autoencoder

The model architecture follows an encoder-decoder structure with skip connections implicitly learned through the bottleneck:

```
Input (B, 8, T)
    │
    ▼
┌─────────────────────────────────────────┐
│  ENCODER                                │
│                                         │
│  Conv1D(8→32, k=7) + GroupNorm + ReLU   │
│  AvgPool1d(2) → T/2                     │
│                                         │
│  Conv1D(32→64, k=5) + GroupNorm + ReLU  │
│  AvgPool1d(2) → T/4                     │
│                                         │
│  Conv1D(64→128, k=5) + GroupNorm + ReLU │
│  AvgPool1d(2) → T/8                     │
│                                         │
│  Conv1D(128→128, k=3) + GroupNorm + ReLU│
│  AvgPool1d(2) → T/16                    │
└─────────────────────────────────────────┘
    │
    ▼
  Latent Space (B, 128, T/16)
    │
    ▼
┌─────────────────────────────────────────┐
│  DECODER                                │
│                                         │
│  Upsample(2) + Conv1D(128→128, k=3)     │
│  + GroupNorm + ReLU                     │
│                                         │
│  Upsample(2) + Conv1D(128→64, k=5)      │
│  + GroupNorm + ReLU                     │
│                                         │
│  Upsample(2) + Conv1D(64→32, k=5)       │
│  + GroupNorm + ReLU                     │
│                                         │
│  Upsample(2) + Conv1D(32→8, k=7)        │
│  + GroupNorm + ReLU                     │
│                                         │
│  Conv1D(8→8, k=1) [Output projection]   │
└─────────────────────────────────────────┘
    │
    ▼
Output (B, 8, T)
```

### Architecture Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Normalization | GroupNorm | Stable with small batch sizes (16) |
| Pooling | AvgPool1d | Smoother downsampling than MaxPool |
| Upsampling | Nearest neighbor | Simple, avoids checkerboard artifacts |
| Activation | ReLU | Standard, efficient |
| Kernel sizes | 7,5,5,3 | Decreasing for hierarchical features |
| Latent dim | 128 | Balance between compression and capacity |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Batch Size | 16 |
| Epochs | 400 |
| Loss Function | Masked MSE |

### Data Pipeline

```
Raw CSV Files (Voltage readings)
        │
        ▼
┌─────────────────────────────┐
│ load_processed_trials()     │  Load preprocessed CSVs
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ TimeSeriesDataset           │  Convert to PyTorch tensors
│ - Drop time column          │
│ - Convert to float32        │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ pad_collate()               │  Batch collation
│ - Pad to max length         │
│ - Create attention mask     │
│ - Transpose (T,C) → (C,T)   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ fit_channel_stats()         │  Compute normalization stats
│ normalize_batch()           │  Z-score normalization
└─────────────────────────────┘
        │
        ▼
    Model Input (B, C, T)
```

### Anomaly Detection Pipeline

```
Trained CAE Model
        │
        ▼
┌─────────────────────────────────────────┐
│ extract_simple_features()               │
│                                         │
│ - Feed trial through encoder + decoder  │
│ - Compute reconstruction error          │
│ - Extract 3 features:                   │
│   1. Overall MSE                        │
│   2. Max channel error                  │
│   3. Channel error variance             │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ ENSEMBLE DETECTION                      │
│                                         │
│  Method 1: Threshold (95th percentile)  │
│  Method 2: Isolation Forest             │
│  Method 3: One-Class SVM                │
│                                         │
│  → Majority Vote (2/3 agree = anomaly)  │
└─────────────────────────────────────────┘
        │
        ▼
  Prediction: Normal (+1) or Anomaly (-1)
```

---

## Installation & Requirements

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Dependencies

```txt
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
seaborn>=0.11.0
```

### Installation Steps

1. **Clone or download the repository:**
   ```bash
   git clone <repository-url>
   cd baseball_data
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch pandas numpy matplotlib scikit-learn seaborn
   ```

4. **Verify installation:**
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

---

## Usage Guide

### Running the Analysis

1. **Open the notebook:**
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Update data paths** (if necessary):
   The notebook expects data in:
   ```python
   full_swing_good_ds = load_processed_trials(
       "C:/Users/TonyZheng/Desktop/baseball_data/full_swing_ball/good/processed_data"
   )
   ```
   Modify this path to match your directory structure.

3. **Run all cells** sequentially (Kernel → Restart & Run All)

### Expected Outputs

The notebook generates the following:

**Console Output:**
- Data loading progress
- Training progress (loss every 50 epochs)
- Anomaly detection metrics

**Visualizations** (saved to `plots/`):
- `good_vs_bad_trials.png` - Comparison of sensor signals
- `training_curves.png` - Loss over epochs
- `comprehensive_evaluation.png` - Full evaluation dashboard

### Customizing Parameters

Key parameters to adjust:

```python
# Training
BATCH_SIZE = 16        # Increase if more data available
EPOCHS = 400           # Reduce for faster iteration
VAL_FRACTION = 0.2     # Validation split ratio

# Model
latent_dim = 128       # Compression level (lower = more compression)

# Anomaly Detection
threshold_percentile = 95  # More sensitive = lower percentile
contamination = 0.1        # Expected anomaly fraction for Isolation Forest
nu = 0.1                   # Upper bound on outliers for One-Class SVM
```

---

## Results & Evaluation

### Training Curves

The model shows steady convergence over 400 epochs:

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1 | 9.227 | 6.462 |
| 50 | 3.433 | 2.771 |
| 100 | 2.040 | 1.779 |
| 200 | 1.049 | 1.256 |
| 400 | 0.653 | 0.935 |

The gap between train and validation loss is moderate, suggesting the model generalizes well without severe overfitting.

### Reconstruction Quality

The trained autoencoder successfully reconstructs normal swing patterns. Visualizations show input signals closely matched by their reconstructions across all 8 channels.

### Anomaly Detection Performance

**Dataset Split:**
- Training: 17 good trials
- Validation: 5 good trials
- Test: 5 good trials + 3 bad trials = 8 total

**Evaluation Metrics:**

| Method | Recall (Sensitivity) | Specificity | Precision | F1 Score |
|--------|----------------------|-------------|-----------|----------|
| Threshold (95%) | 33.3% | 80.0% | 50.0% | 0.40 |
| Isolation Forest | 33.3% | 80.0% | 50.0% | 0.40 |
| One-Class SVM | 66.7% | 40.0% | 40.0% | 0.44 |
| Ensemble (2/3) | 33.3% | 80.0% | 50.0% | 0.40 |

**Interpretation:**
- **Recall**: Proportion of bad swings correctly identified (higher = fewer missed anomalies)
- **Specificity**: Proportion of good swings correctly identified (higher = fewer false alarms)
- **Ensemble**: Provides balanced detection by requiring agreement from multiple methods

### Feature Space Visualization

The 3D feature space (MSE vs Max Channel Error vs Channel Error Variance) shows separation between good and bad trials, though with overlap due to the limited dataset size.

---

## Limitations & Future Work

### Current Limitations

1. **Small Dataset Size**
   - Only 27 good and 3 bad trials
   - Limited statistical power for evaluation
   - High variance in performance estimates

2. **Class Imbalance**
   - 9:1 ratio of good to bad trials
   - Difficult to reliably train supervised methods
   - Metrics may be misleading

3. **Subject Variability**
   - Data may be from single or few subjects
   - Model may not generalize to other batters

### Potential Improvements

1. **Data Augmentation**
   - Time warping
   - Noise injection
   - Magnitude scaling
   - Mixup between similar trials

2. **Architecture Enhancements**
   - LSTM/GRU layers for temporal modeling
   - Transformer attention mechanisms
   - Variational Autoencoder (VAE) for probabilistic encoding

3. **Training Strategies**
   - Cross-validation for robust evaluation
   - Ensemble of multiple autoencoder models
   - Progressive training with curriculum learning

4. **Feature Engineering**
   - Frequency domain features (FFT)
   - Wavelet decomposition
   - Peak detection and timing features

### Future Directions

- **Real-time Detection**: Optimize for inference speed on edge devices
- **Feedback System**: Integrate with visual/haptic feedback for training
- **Multi-sport Generalization**: Extend to golf swings, tennis serves, etc.

---

## References

### Deep Learning & Autoencoders

1. [Keras: Timeseries Anomaly Detection using Autoencoder](https://keras.io/examples/timeseries/timeseries_anomaly_detection/) - Official Keras tutorial

2. [Deep Learning for Time Series Anomaly Detection: A Survey](https://arxiv.org/html/2211.05244v3) - Comprehensive survey paper

3. [Temporal Convolutional Autoencoder for Unsupervised Anomaly Detection](https://www.sciencedirect.com/science/article/abs/pii/S1568494621006724) - TCN-AE approach

### Normalization

4. [Group Normalization](https://arxiv.org/abs/1803.08494) - Wu & He, ECCV 2018

5. [Comparing Normalization Methods for Limited Batch](https://arxiv.org/pdf/2011.11559) - Analysis of normalization methods

### Anomaly Detection

6. [scikit-learn: Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) - Official documentation

7. [scikit-learn: One-Class SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) - Official documentation

8. [Detecting Anomalies: A Guide with One-Class SVM and Isolation Forest](https://ai.plainenglish.io/detecting-anomalies-a-comprehensive-guide-with-one-class-svm-and-isolation-forest-230336f0988a) - Practical guide

### Curse of Dimensionality

9. [Curse of Dimensionality in Machine Learning](https://www.datacamp.com/blog/curse-of-dimensionality-machine-learning) - DataCamp explanation

10. [Wikipedia: Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) - General overview

---

## License

This project is available under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{baseball_swing_anomaly_detection,
  title = {Baseball Swing Anomaly Detection Using Deep Learning},
  author = {Tony Zheng},
  year = {2024},
  url = {https://github.com/username/baseball_data}
}
```

---

## Contact

For questions or collaboration opportunities, please open an issue in this repository.
