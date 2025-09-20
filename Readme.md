# Parkinson's Disease Detection using Voice Biomarkers

A comprehensive machine learning project for detecting Parkinson's disease using voice recordings and acoustic features. This project implements subject-independent cross-validation to ensure clinical validity and prevent data leakage.

## 🎯 Project Overview

This project uses machine learning to detect Parkinson's disease from voice recordings by analyzing acoustic features like jitter, shimmer, fundamental frequency, and other voice quality measures. The key innovation is the use of **subject-independent cross-validation**, ensuring that no individual appears in both training and testing sets, which is crucial for clinical applications.

### Key Features

- **Subject-Independent Validation**: Prevents data leakage by ensuring no person appears in both training and testing
- **Multiple Model Configurations**: Various Random Forest configurations for performance comparison
- **Clinical Metrics**: Sensitivity, specificity, and AUC-ROC for medical evaluation
- **Feature Importance Analysis**: Identifies which voice features are most predictive
- **Comprehensive Evaluation**: Detailed confusion matrices and performance metrics

## 📊 Dataset

The project uses the Parkinson's Disease Detection dataset containing voice recordings from 31 people (23 with Parkinson's, 8 healthy controls). Each person has multiple recordings, resulting in 195 total voice samples with 22 acoustic features.

### Dataset Features

- **MDVP Features**: Fundamental frequency measures (Fo, Fhi, Flo)
- **Jitter Measures**: Voice instability indicators (%, Abs, RAP, PPQ, DDP)
- **Shimmer Measures**: Amplitude variation indicators (%, dB, APQ3, APQ5, APQ, DDA)
- **Harmonic Features**: NHR (Noise-to-Harmonics Ratio), HNR (Harmonics-to-Noise Ratio)
- **Nonlinear Features**: RPDE, DFA, spread1, spread2, D2, PPE
- **Target**: Binary classification (0 = Healthy, 1 = Parkinson's)

## 🏗️ Project Structure

```
Parkinsonstrackingthing/
├── Readme.md                    # This comprehensive documentation
├── requirements.txt             # Python dependencies
├── parkinsons.data             # Main dataset (CSV format)
├── est100_d10_3cv_s2_l1_rs42/  # 100 trees, depth 10, 3-fold CV
│   ├── main.py
│   └── readme.md
├── est100_d10_4cv_s2_l1_rs42/  # 100 trees, depth 10, 4-fold CV
│   └── main.py
├── est100_d10_10cv_s2_l1_rs42/ # 100 trees, depth 10, 10-fold CV
│   ├── main.py
│   └── readme.md
├── est100_d10_20cv_s2_l1_rs42/ # 100 trees, depth 10, 20-fold CV
│   └── main.py
├── est200_d10_3cv_s2_l1_rs42/  # 200 trees, depth 10, 3-fold CV
│   └── main.py
└── est200_d10_4cv_s2_l1_rs42/  # 200 trees, depth 10, 4-fold CV
    └── main.py
```

## 🔧 Model Configurations

Each folder represents a different Random Forest configuration:

### Naming Convention
- `est{100|200}`: Number of decision trees (estimators)
- `d{10}`: Maximum tree depth
- `{3|4|10|20}cv`: Number of cross-validation folds
- `s2`: min_samples_split (default: 2)
- `l1`: min_samples_leaf (default: 1)
- `rs42`: Random state for reproducibility

### Configuration Details

| Configuration | Trees | Max Depth | CV Folds | Expected Performance |
|---------------|-------|-----------|----------|---------------------|
| est100_d10_3cv | 100 | 10 | 3 | ~88% accuracy |
| est100_d10_4cv | 100 | 10 | 4 | ~89% accuracy |
| est100_d10_10cv | 100 | 10 | 10 | ~91% accuracy |
| est100_d10_20cv | 100 | 10 | 20 | ~91% accuracy |
| est200_d10_3cv | 200 | 10 | 3 | ~88% accuracy |
| est200_d10_4cv | 200 | 10 | 4 | ~89% accuracy |

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip or pip3

### Installation

1. **Clone or download the project**
   ```bash
   cd Parkinsonstrackingthing
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # For Linux/Mac users:
   pip3 install -r requirements.txt
   ```

3. **Run a model configuration**
   ```bash
   cd est100_d10_10cv_s2_l1_rs42
   python main.py
   ```

### Expected Output

```
============================================================
PARKINSON'S DISEASE DETECTION - ML ANALYSIS
============================================================
Loaded: 195 recordings, 22 features
Subjects: 31 people (6.3 recordings each)
Labels: 48 healthy, 147 Parkinson's
Ready: X=(195, 22), y=(195,)

========================================
CROSS-VALIDATION SETUP
========================================
Validating CV splits...
  Fold 1: 28 train subjects, 3 test subjects
  Fold 2: 28 train subjects, 3 test subjects
  ...
✓ No data leakage - subject-independent CV validated

========================================
MODEL TRAINING & EVALUATION
========================================
Performing subject-independent cross-validation...
  ACCURACY: 0.910 ± 0.045
  PRECISION: 0.920 ± 0.040
  RECALL: 0.910 ± 0.045
  F1: 0.915 ± 0.042
  ROC_AUC: 0.945 ± 0.035

========================================
FEATURE IMPORTANCE ANALYSIS
========================================
Top 10 most important voice features:
  spread2: 0.156
  PPE: 0.142
  spread1: 0.128
  ...

========================================
CLINICAL EVALUATION
========================================
Confusion Matrix:
                Predicted
Actual    Healthy  PD
Healthy      42     6
PD           12   135

Clinical Metrics:
  Sensitivity (PD Detection Rate): 0.918
  Specificity (Healthy Detection Rate): 0.875
  AUC-ROC: 0.945
```

## 📈 Performance Analysis

### Cross-Validation Impact

The project demonstrates the importance of cross-validation strategy:

- **3-Fold CV**: ~88% accuracy (each model sees ~67% of data)
- **10-Fold CV**: ~91% accuracy (each model sees ~90% of data)
- **20-Fold CV**: ~91% accuracy (diminishing returns)

### Why More Folds Improve Performance

1. **More Training Data**: Each model sees more data during training
2. **Reduced Bias**: Better pattern recognition with larger training sets
3. **Better Generalization**: Models trained on nearly the entire dataset
4. **Statistical Robustness**: More reliable performance estimates

## 🔬 Clinical Significance

### Key Metrics

- **Sensitivity**: Percentage of Parkinson's patients correctly identified
- **Specificity**: Percentage of healthy individuals correctly identified
- **AUC-ROC**: Overall discriminative ability (0.5 = random, 1.0 = perfect)

### Medical Interpretation

- **High Sensitivity**: Important for early detection and treatment
- **High Specificity**: Reduces false alarms and unnecessary treatment
- **Subject Independence**: Ensures model works on new patients

## 🛠️ Technical Implementation

### Core Components

1. **Data Loading**: CSV parsing with subject ID extraction
2. **Preprocessing**: StandardScaler for feature normalization
3. **Cross-Validation**: StratifiedGroupKFold for subject independence
4. **Model Training**: RandomForestClassifier with balanced classes
5. **Evaluation**: Comprehensive metrics and feature analysis

### Key Classes

- `ParkinsonDetectionModel`: Main model class
- `extract_subject_id()`: Extracts person ID from recording names
- `setup_subject_independent_cv()`: Validates no data leakage
- `clinical_evaluation()`: Calculates medical metrics

## 📚 Usage Examples

### Running Different Configurations

```bash
# High accuracy configuration (recommended)
cd est100_d10_10cv_s2_l1_rs42
python main.py

# Quick testing with fewer folds
cd est100_d10_3cv_s2_l1_rs42
python main.py

# More trees for potential better performance
cd est200_d10_10cv_s2_l1_rs42
python main.py
```

### Custom Analysis

```python
from est100_d10_10cv_s2_l1_rs42.main import ParkinsonDetectionModel

# Initialize model
model = ParkinsonDetectionModel('path/to/parkinsons.data')

# Run complete analysis
feature_cols = model.load_and_prepare_data()
results = model.train_and_evaluate()
importance = model.get_feature_importance(feature_cols)
clinical = model.clinical_evaluation()
```

## 🔍 Feature Importance

The most predictive voice features for Parkinson's detection:

1. **spread2**: Second spread measure
2. **PPE**: Pitch Period Entropy
3. **spread1**: First spread measure
4. **DFA**: Detrended Fluctuation Analysis
5. **RPDE**: Recurrence Period Density Entropy

These features capture voice instability and nonlinear dynamics characteristic of Parkinson's disease.

## ⚠️ Important Notes

### Data Leakage Prevention

- **Critical**: The model uses subject-independent cross-validation
- **Why Important**: Prevents overoptimistic results from same person in train/test
- **Clinical Relevance**: Ensures model works on completely new patients

### Class Imbalance

- **Dataset**: 3:1 ratio (147 Parkinson's : 48 healthy)
- **Solution**: `class_weight='balanced'` in RandomForestClassifier
- **Impact**: Prevents model bias toward majority class

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different configurations
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please cite the original dataset paper when using this work.

## 📖 References

### Dataset Citation

If you use this dataset, please cite:

Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM.  
"Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection."  
*BioMedical Engineering OnLine*, 2007, 6:23 (26 June 2007).  
DOI: [10.1186/1475-925X-6-23](https://doi.org/10.1186/1475-925X-6-23)

### Additional Resources

- [UCI Machine Learning Repository - Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/Parkinsons)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Cross-Validation Best Practices](https://scikit-learn.org/stable/modules/cross_validation.html)

## 🆘 Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure `parkinsons.data` is in the correct location
2. **ImportError**: Install all requirements with `pip install -r requirements.txt`
3. **Memory Issues**: Use smaller configurations (3-fold CV) for limited resources

### Performance Tips

- Use 10-fold CV for best accuracy
- Monitor memory usage with larger configurations
- Consider feature selection for faster training

---

**Note**: This project is designed for research and educational purposes. For clinical applications, additional validation and regulatory approval would be required.