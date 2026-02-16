# Explainable AI Analysis: Hospital Readmission Prediction

## Overview

This project applies **Explainable AI (XAI) methods** to predict hospital readmissions for diabetic patients. Beyond building a predictive model, the focus is on **understanding why** the model makes certain predictions — making it transparent and trustworthy for clinical use.

Two XAI methods are compared:
- **eli5 Permutation Importance** — measures global feature importance by performance drop
- **SHAP (SHapley Additive exPlanations)** — provides both global and local explanations based on game theory

The XAI insights are then used for **feature selection**, reducing model complexity while maintaining predictive performance.

## Key Results

| Model | Features | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|----------|---------|
| Baseline (XGBoost) | 52 | 0.619 | 0.547 | 0.658 |
| Tuned (RandomizedSearchCV) | 52 | 0.621 | 0.553 | 0.660 |
| XAI-Improved (Feature Selection) | 32 | 0.617 | 0.548 | 0.657 |

**Main finding**: Healthcare utilization history (`number_inpatient`, `number_emergency`, `number_outpatient`) is the dominant predictor of readmission. XAI-guided feature selection reduced features by 38% with no performance loss.

## Dataset

- **Source**: [Hospital Readmissions Dataset (Kaggle)](https://www.kaggle.com/datasets/dansbecker/hospital-readmissions/data)
- **Samples**: 25,000 patient records
- **Features**: 64 (clinical, demographic, treatment)
- **Target**: Hospital readmission (binary, 54/46 split)

> **Note**: The dataset is not included in this repository. Download it from the link above, name the CSV file hospital_readmissions.csv and place it in the project root directory to run the notebook.

### Feature Categories

| Category | Examples |
|----------|----------|
| Utilization History | Prior inpatient stays, emergency visits, outpatient visits |
| Clinical Complexity | Number of medications, lab procedures, diagnoses |
| Demographics | Age group, race, gender |
| Treatment | Medication flags, insulin usage, medication changes |

## Project Structure

```
├── xai_hospital_readmission.ipynb    # Main analysis notebook
├── requirements.txt                  # Python dependencies
└── README.md
```

## Setup

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook xai_hospital_readmission.ipynb
```

## Notebook Structure

1. **Introduction** — Project goal and dataset overview
2. **Setup & Data Loading** — Imports and data inspection
3. **Exploratory Data Analysis** — Distributions, correlations, class differences
4. **Data Preprocessing** — Variance threshold, train-test split
5. **Feature Engineering** — Utilization score, treatment intensity ratios
6. **Model Training & Evaluation** — Baseline XGBoost, hyperparameter tuning
7. **XAI Analysis** — Permutation Importance, SHAP (global + local explanations)
8. **XAI-Guided Feature Selection** — Improved model with top 32 features
9. **Conclusion** — Results, limitations, potential improvements

## Tech Stack

- **Model**: XGBoost
- **XAI**: eli5, SHAP
- **ML Pipeline**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Data**: pandas, numpy
