# Sepsis Early Warning System
### Explainable Machine Learning for ICU Clinical Decision Support

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sepsis-prediction-ai.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)](https://xgboost.ai)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.953-brightgreen)](https://sepsis-prediction-ai.streamlit.app)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection, responsible for over 11 million deaths annually worldwide. Early identification remains one of the most critical — and difficult — challenges in intensive care medicine.

This project develops and deploys a clinically-oriented machine learning system for **early sepsis prediction**, trained on 40,336 real ICU patients from the PhysioNet/Computing in Cardiology Challenge 2019 dataset. The system achieves an AUC-ROC of 0.953 and integrates SHAP-based explainability to provide transparent, actionable predictions that clinicians can interpret and trust.

**Live Demo →** [https://sepsis-prediction-ai.streamlit.app](https://sepsis-prediction-ai.streamlit.app)

---

## Clinical Motivation

Standard sepsis screening tools such as SOFA and qSOFA rely on threshold-based rules that are applied reactively and miss early-stage deterioration. Machine learning models trained on continuous ICU time series data can detect subtle physiological patterns hours before clinical manifestation, enabling preemptive intervention and significantly improving patient outcomes.

This system is designed with two principles in mind:

- **Performance** — achieving AUC-ROC competitive with published literature
- **Explainability** — ensuring every prediction is interpretable by clinical staff via SHAP feature attribution

---

## Dataset

**PhysioNet/Computing in Cardiology Challenge 2019**

| Property | Value |
|----------|-------|
| Total patients | 40,336 |
| Total records | 1,552,210 |
| Clinical variables | 40 (vitals, labs, demographics) |
| Measurement frequency | Hourly |
| Sepsis definition | Sepsis-3 |
| Sepsis prevalence | 1.80% |

---

## Methodology

### Data Preprocessing
Missing values — totalling 43,512,155 entries — were handled using a two-step imputation strategy: per-patient forward-fill and backward-fill to preserve temporal continuity, followed by global median imputation for any remaining gaps.

### Feature Engineering
Seven clinically motivated features were derived from existing variables:

| Feature | Clinical Rationale |
|---------|-------------------|
| Shock Index (HR / SBP) | Indicator of hemodynamic instability |
| Pulse Pressure (SBP − DBP) | Marker of arterial compliance |
| MAP/HR Ratio | Combined cardiovascular stress indicator |
| Resp/O2 Ratio | Measure of ventilatory efficiency |
| HR Rolling Mean (3h) | Captures sustained tachycardia trends |
| HR Rolling Std (3h) | Captures heart rate variability |
| Resp Rolling Mean (3h) | Captures respiratory trend over time |

### Model
An XGBoost gradient boosted classifier was trained with `scale_pos_weight` to address the severe class imbalance (54:1 ratio). Decision threshold was optimised using the F1 score on the Precision-Recall curve.

### Explainability
SHAP (SHapley Additive exPlanations) TreeExplainer was applied to a 5,000-patient sample to produce feature importance rankings and beeswarm plots, identifying the most influential clinical predictors.

---

## Results

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.953** |
| Sepsis Recall | **0.88** |
| Average Precision | **0.42** |
| Optimal F1 Threshold | **0.84** |
| Training set size | 1,241,768 records |
| Test set size | 310,442 records |

### Key SHAP Findings

- **ICU Length of Stay (ICULOS)** is the dominant predictor, consistent with clinical understanding that sepsis risk accumulates over time
- **Temperature** and **FiO2** are among the most discriminative features, reflecting fever and oxygenation requirements
- Engineered features **MAP_HR_ratio** and **Resp_rolling_mean** appear in the top 15, validating the clinical relevance of the feature engineering approach

---

## Project Structure

```
sepsis-prediction-ai/
├── dashboard.py          # Streamlit clinical decision support application
├── sepsis_model.pkl      # Trained XGBoost classifier
├── scaler.pkl            # StandardScaler fitted on training data
└── requirements.txt      # Python dependencies
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| ML Framework | XGBoost |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## Limitations and Future Work

- The model was trained and evaluated on a single challenge dataset; external validation on independent hospital cohorts is required before clinical deployment
- The current implementation does not model temporal dependencies across patient time steps; future work will explore LSTM and Transformer architectures for sequential modelling
- Fairness analysis across demographic subgroups has not been conducted and represents an important area for further investigation

---

## Author

**Kashiruddin Shaik**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kashiruddin-shaik/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/Kashiruddinshaik)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-c9a84c?logo=google-chrome&logoColor=white)](https://kashiruddinshaik.github.io/Portfolio/)

---

*This project is intended for research and educational purposes. Clinical deployment requires prospective validation, regulatory approval, and institutional review.*
