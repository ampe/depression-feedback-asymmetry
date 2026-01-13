# Depression-Feedback-Asymmetry

## EEG-Based Emotion Recognition in Major Depressive Disorder

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project investigates brain asymmetry patterns and emotional processing in Major Depressive Disorder (MDD) using EEG data and machine learning. The goal is to identify neural markers that distinguish emotional responses between MDD patients and healthy controls, with a focus on detecting "bifurcation points" where neutral emotional states transition to positive or negative states.

## 🧠 Research Overview

### Scientific Questions
- How do MDD patients process positive vs. negative feedback differently from healthy controls?
- Can EEG-based features predict emotional state changes during decision-making tasks?
- What are the neural signatures of emotional state transitions (bifurcation points)?

### Dataset
- **120 participants**: 60 MDD patients + 60 healthy controls
- **Task**: Antarctic search-and-rescue decision-making with emotional feedback
- **Conditions**: Positive feedback (mostly wins) vs. Negative feedback (mostly losses)
- **EEG**: 17 channels, 500 Hz sampling rate

## 📊 Features Extracted

| Feature Type | Count | Description |
|-------------|-------|-------------|
| Differential Entropy (DE) | 45 | Entropy in frequency bands (δ, θ, α, β, γ) |
| Power Spectral Density (PSD) | 135 | Absolute, relative, and peak power |
| Statistical | 45 | Mean, std, skewness, kurtosis, Hjorth parameters |
| Wavelet Transform | 75 | DWT coefficients and energy |
| Connectivity | 15 | Correlation, PLV between channels |
| Asymmetry | 18 | Frontal, central, parietal alpha asymmetry |
| Band Ratios | 9 | θ/β, θ/α, α/β ratios |
| **Total** | **342** | Comprehensive feature set |

## 🔬 Methods

### Feature Extraction
Based on literature review of successful EEG emotion recognition studies:
- **Differential Entropy** - most popular feature in DEAP/SEED benchmarks
- **Alpha Asymmetry** - key biomarker for emotional valence
- **Wavelet features** - captures non-stationary dynamics
- **Connectivity** - inter-hemispheric communication patterns

### Machine Learning Pipeline
- Logistic Regression
- Random Forest
- SVM (RBF kernel)
- Gradient Boosting
- *Coming soon: CNN, LSTM, EEGNet*

### Validation
- GroupKFold cross-validation (subjects don't leak between train/test)
- ROC-AUC for binary classification
- Trial-level and subject-level analysis

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/ampe/depression-feedback-asymmetry.git
cd depression-feedback-asymmetry
pip install -r requirements.txt
```

### Requirements
```
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
PyWavelets>=1.1.0
```

### Usage
```python
# Configure data path in config.py
DATA_ROOT = Path("path/to/your/data")

# Run classification pipeline
python trial_level_classification.py
```

## 📁 Project Structure
```
depression-feedback-asymmetry/
├── config.py                      # Data paths and parameters
├── data_loader.py                 # EEGLAB file loading utilities
├── feature_extraction.py          # Comprehensive feature extraction
├── trial_level_classification.py  # ML classification pipeline
├── analysis.py                    # Statistical analysis
├── ml_classification.py           # Subject-level classification
└── README.md
```

## 📈 Results

### Classification Performance

| Task | Best Model | Accuracy | AUC |
|------|-----------|----------|-----|
| Feedback Type (all data) | SVM | 60.5% | 0.626 |
| Feedback Type (MDD only) | SVM | 59.1% | 0.616 |
| MDD vs Healthy | Gradient Boosting | 55.3% | 0.578 |

*Results with 342 features - experiments in progress*

### Key Findings
- Reduced P300 amplitude in MDD during feedback processing (p=0.038)
- Frontal alpha asymmetry differences between conditions
- Parietal asymmetry shows trend for positive vs. negative feedback (p=0.076)

## 🗺️ Roadmap

- [x] Data loading pipeline for EEGLAB format
- [x] ERP feature extraction
- [x] Comprehensive spectral features (DE, PSD, Wavelet)
- [x] Connectivity and asymmetry features
- [x] Traditional ML classifiers
- [ ] Deep learning models (CNN, LSTM, EEGNet)
- [ ] Bifurcation point detection
- [ ] Cross-dataset validation
- [ ] Publication-ready figures

## 📚 References

Key papers informing this work:
1. Su & Liu (2021) - Residual Contraction Network for EEG emotion (94.8% on SEED)
2. Liu et al. (2021) - 3DCANN spatio-temporal features (96.4% on SEED)
3. Du et al. (2020) - ATDD-LSTM for emotion recognition (94.7%)

## 👤 Author

**Ruslan** - Neuroscience Researcher

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data collection: [Your University/Lab]
- EEG preprocessing: EEGLAB/ERPLAB
