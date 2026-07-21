# 🩺 Fetal Health Classification

> An end-to-end Machine Learning project that classifies fetal health status from cardiotocography (CTG) measurements into **Normal**, **Suspect**, or **Pathological** categories.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7-green.svg)](https://lightgbm.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.14-blue.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Feature Importance](#-feature-importance)
- [Technologies Used](#-technologies-used)
- [Disclaimer](#-disclaimer)
- [Contributing](#-contributing)

---

## 🔍 Overview

Cardiotocography (CTG) is a widely used technique for monitoring fetal heart rate and uterine contractions during pregnancy. This project leverages machine learning to classify fetal health outcomes based on CTG features, potentially assisting healthcare professionals in identifying at-risk pregnancies.

The project follows a rigorous, reproducible ML workflow that includes:

- **Data quality audits** — detecting duplicates, missing values, and target validity
- **Leakage-safe evaluation** — strict train/test splitting before any modeling
- **Baseline model comparison** — benchmarking 7 algorithms via stratified cross-validation
- **Hyperparameter optimization** — Optuna-driven Bayesian search for the best model
- **Experiment tracking** — full MLflow integration for reproducibility
- **Model interpretation** — permutation importance analysis
- **Model persistence** — serialized model with metadata for production deployment

---

## 🏆 Key Results

| Metric              | Score    |
|----------------------|----------|
| **Test Accuracy**        | 96.22%   |
| **Test Balanced Accuracy** | 92.23%   |
| **Test Macro F1-Score**  | 92.95%   |
| **Test Macro Recall**    | 92.23%   |

**Best Model:** LightGBM (tuned with Optuna, 20 trials)

### Per-Class Performance (Holdout Test Set)

| Class          | Precision | Recall  | F1-Score | Support |
|----------------|-----------|---------|----------|---------|
| **Normal**         | 0.9703    | 0.9909  | 0.9805   | 330     |
| **Suspect**        | 0.9375    | 0.7759  | 0.8491   | 58      |
| **Pathological**   | 0.9211    | 1.0000  | 0.9589   | 35      |

---

## 📁 Project Structure

```
Fetal Health Classifier/
├── 📂 artifacts/              # Trained models & metadata
│   ├── best_fetal_health_lightgbm.joblib
│   └── best_fetal_health_lightgbm_metadata.json
├── 📂 configs/                # Configuration files
├── 📂 data/
│   ├── 📂 external/          # Third-party data sources
│   ├── 📂 processed/         # Cleaned/transformed data
│   └── 📂 raw/               # Original dataset
│       └── fetal_health.csv
├── 📂 models/                 # Model registry
├── 📂 notebooks/
│   ├── EDA.ipynb              # Exploratory data analysis
│   └── Fetal_Health_Professional_ML_Project.ipynb  # Main notebook
├── 📂 reports/                # Generated reports & figures
├── 📂 src/                    # Source code modules
│   ├── 📂 data/              # Data loading & processing
│   ├── 📂 features/          # Feature engineering
│   ├── 📂 models/            # Model training & evaluation
│   ├── 📂 utils/             # Utility functions
│   └── 📂 visualization/     # Plotting utilities
├── 📂 tests/                  # Unit tests
├── .gitignore
├── README.md
├── requirements.txt
└── mlflow.db                  # Experiment tracking database
```

---

## 📊 Dataset

The dataset is sourced from the [UCI Machine Learning Repository — Cardiotocography](https://archive.ics.uci.edu/ml/datasets/cardiotocography) and contains **2,126 records** with **21 features** extracted from CTG exams.

### Target Classes

| Code | Label          | Count  | Percentage |
|------|----------------|--------|------------|
| 1    | Normal         | 1,655  | 77.85%     |
| 2    | Suspect        | 295    | 13.88%     |
| 3    | Pathological   | 176    | 8.28%      |

> ⚠️ **Class imbalance:** The dataset is heavily imbalanced (~78% Normal), which is why we optimize for **macro F1-score** rather than accuracy.

### Features

The dataset includes 21 features derived from CTG measurements:

| Feature | Description |
|---------|-------------|
| `baseline value` | Fetal heart rate baseline (beats per minute) |
| `accelerations` | Number of accelerations per second |
| `fetal_movement` | Number of fetal movements per second |
| `uterine_contractions` | Number of uterine contractions per second |
| `light_decelerations` | Number of light decelerations per second |
| `severe_decelerations` | Number of severe decelerations per second |
| `prolongued_decelerations` | Number of prolonged decelerations per second |
| `abnormal_short_term_variability` | Percentage of time with abnormal short-term variability |
| `mean_value_of_short_term_variability` | Mean value of short-term variability |
| `percentage_of_time_with_abnormal_long_term_variability` | Percentage of time with abnormal long-term variability |
| `mean_value_of_long_term_variability` | Mean value of long-term variability |
| `histogram_width` | Width of the FHR histogram |
| `histogram_min` | Minimum of the FHR histogram |
| `histogram_max` | Maximum of the FHR histogram |
| `histogram_number_of_peaks` | Number of peaks in the FHR histogram |
| `histogram_number_of_zeroes` | Number of zeroes in the FHR histogram |
| `histogram_mode` | Mode of the FHR histogram |
| `histogram_mean` | Mean of the FHR histogram |
| `histogram_median` | Median of the FHR histogram |
| `histogram_variance` | Variance of the FHR histogram |
| `histogram_tendency` | Tendency of the FHR histogram |

---

## 🔬 Methodology

The workflow follows four core principles:

1. **Remove exact duplicates** before splitting to reduce leakage risk (13 duplicates removed)
2. **Stratified holdout test set** (80/20 split) — untouched until final evaluation
3. **5-fold stratified cross-validation** on training set only for model comparison
4. **Macro F1-score optimization** — prevents accuracy from hiding poor minority-class performance

### Pipeline Steps

```
1. Data loading & validation
       ↓
2. Duplicate removal & cleaning
       ↓
3. Exploratory Data Analysis (EDA)
       ↓
4. Stratified train/test split (80/20)
       ↓
5. Baseline model comparison (7 models, 5-fold CV)
       ↓
6. Hyperparameter tuning (Optuna, 20 trials)
       ↓
7. Final evaluation on holdout test set
       ↓
8. Model interpretation (permutation importance)
       ↓
9. Model persistence (joblib + metadata JSON)
```

### Models Compared (Baseline)

| Model                          | Macro F1 (Mean) | Macro F1 (Std) | Balanced Acc | Accuracy |
|--------------------------------|-----------------|----------------|--------------|----------|
| **LightGBM** ✅                | 0.9014          | 0.0237         | 0.8895       | 0.9450   |
| Histogram Gradient Boosting    | 0.8972          | 0.0193         | 0.8847       | 0.9444   |
| Random Forest                  | 0.8835          | 0.0195         | 0.8954       | 0.9302   |
| Extra Trees                    | 0.8606          | 0.0146         | 0.8348       | 0.9278   |
| Decision Tree                  | 0.8435          | 0.0158         | 0.8429       | 0.9095   |
| Logistic Regression            | 0.7807          | 0.0298         | 0.8501       | 0.8639   |
| Dummy (majority class)         | 0.2919          | 0.0002         | 0.3333       | 0.7787   |

### Optimized Hyperparameters

| Parameter          | Value     |
|--------------------|-----------|
| `max_depth`        | 10        |
| `n_estimators`     | 305       |
| `learning_rate`    | 0.0193    |
| `num_leaves`       | 92        |
| `min_child_samples`| 5         |
| `subsample`        | 0.7897    |
| `colsample_bytree` | 0.7246    |
| `reg_alpha`        | 0.0031    |
| `reg_lambda`       | 0.0002    |
| `min_split_gain`   | 0.1400    |

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.11+
- pip or conda

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/al-rafi-DSC/Fatal_Health_Classifier.git
   cd Fatal_Health_Classifier
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the environment**

   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

6. **Open the main notebook**
   Navigate to `notebooks/Fetal_Health_Professional_ML_Project.ipynb`

---

## 🚀 Usage

### Running the Full Pipeline

Open and run all cells in `notebooks/Fetal_Health_Professional_ML_Project.ipynb`. The notebook is self-contained and will:

1. Load and validate the raw data
2. Clean duplicates and prepare features
3. Split data into train/test sets
4. Compare baseline models
5. Tune the best model with Optuna
6. Evaluate on holdout test set
7. Generate interpretation plots
8. Save the trained model and metadata

### Loading the Pre-Trained Model

```python
import joblib
import json
import pandas as pd

# Load model and metadata
model = joblib.load("artifacts/best_fetal_health_lightgbm.joblib")

with open("artifacts/best_fetal_health_lightgbm_metadata.json") as f:
    metadata = json.load(f)

# View class mapping
print(metadata["class_mapping"])
# {'1': 'Normal', '2': 'Suspect', '3': 'Pathological'}

# View feature order
print(metadata["feature_order"])

# Make predictions on new data
# Ensure features are in the correct order
new_data = pd.DataFrame(...)  # Your CTG data here
predictions = model.predict(new_data[metadata["feature_order"]])

# Map predictions to labels
labels = [metadata["class_mapping"][str(p)] for p in predictions]
```

### Using MLflow for Experiment Tracking

```bash
# Launch the MLflow UI to explore experiment runs
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://localhost:5000` in your browser.

---

## 📈 Model Performance

The final tuned LightGBM model achieves excellent performance across all three classes:

- **Normal class:** Near-perfect precision (97.03%) and recall (99.09%)
- **Pathological class:** Perfect recall (100%) with strong precision (92.11%)
- **Suspect class:** Strong precision (93.75%) with good recall (77.59%)

> The Suspect class is the hardest to identify, which aligns with clinical intuition — borderline cases are inherently ambiguous.

---

## 🔑 Feature Importance

The model uses **permutation importance** to rank features by their impact on macro F1-score. Key predictors include:

- `abnormal_short_term_variability` — Most influential feature
- `mean_value_of_short_term_variability`
- `percentage_of_time_with_abnormal_long_term_variability`
- `prolongued_decelerations`
- `accelerations`

> 💡 Feature importance reflects predictive power, not causal relationships.

---

## 🛠 Technologies Used

| Technology     | Purpose                                   |
|----------------|-------------------------------------------|
| **Python 3.11** | Core programming language                |
| **pandas**      | Data manipulation and analysis           |
| **NumPy**       | Numerical computing                      |
| **scikit-learn** | Model training, evaluation, and pipelines |
| **LightGBM**    | Gradient boosting classifier (best model)|
| **Optuna**       | Bayesian hyperparameter optimization     |
| **MLflow**       | Experiment tracking and model registry   |
| **Matplotlib**   | Static data visualizations               |
| **Seaborn**      | Statistical data visualizations          |
| **joblib**       | Model serialization and persistence      |

---

## ⚠️ Disclaimer

> **This project is for educational and research purposes only.** A model trained on this public dataset must **not** be used for clinical decisions without external validation, calibration assessment, governance, and review by qualified healthcare professionals.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

