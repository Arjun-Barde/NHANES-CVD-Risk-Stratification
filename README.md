# NHANES CVD Risk Stratification

Cardiovascular disease symptom risk stratification using NHANES 2017–2020 federal survey data. Merges 6 XPT data files across 4 NHANES components, engineers clinical features, benchmarks 4 ML classifiers, and applies SHAP explainability to identify key predictors of CVD symptoms in U.S. adults aged 40+.

---

## Project Overview

**Dataset:** CDC NHANES 2017–March 2020 Pre-Pandemic Public Use Files  
**Population:** 6,429 adults aged 40+ (nationally representative U.S. sample)  
**Target:** CVD symptom — chest pain on exertion (CDQ001, positive rate: 29.5%)  
**Features:** 22 features including 4 engineered interaction terms  

---

## Data Sources

All data is publicly available from the CDC NHANES program. Download directly from:  
[https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020)

| File | Component | Description |
|------|-----------|-------------|
| P_DEMO.XPT | Demographics | Age, sex, race/ethnicity, income, survey weights |
| P_BMX.XPT | Examination | BMI, waist circumference |
| P_BPXO.XPT | Examination | Systolic and diastolic blood pressure |
| P_TCHOL.XPT | Laboratory | Total cholesterol |
| P_DIQ.XPT | Questionnaire | Diabetes diagnosis |
| P_CDQ.XPT | Questionnaire | Cardiovascular symptom questionnaire (target) |

> XPT files are not included in this repo per CDC data use guidelines. Place downloaded files in the `data/` directory before running the notebook.

---

## Methods

**Preprocessing**
- Merged 6 XPT files on participant ID (SEQN) using left joins from CDQ base
- Restricted to adults 40+ (CDQ administration criteria)
- Median imputation for continuous variables (BMI 9.7% missing, BP 16.9% missing, cholesterol 14.0% missing)
- One-hot encoded race/ethnicity (5 categories)

**Feature Engineering**
- `pulse_pressure` = SBP − DBP (arterial stiffness marker)
- `age_x_diabetes` = age × diabetes status (interaction term)
- `bmi_x_diabetes` = BMI × diabetes status (interaction term)
- `sbp_x_chol` = systolic BP × total cholesterol (interaction term)
- Binary flags: `hypertension` (SBP ≥ 130), `obese` (BMI ≥ 30), `high_chol` (≥ 200 mg/dL)

**Models**
All models evaluated using 5-fold stratified cross-validation + held-out test set (80/20 split).

| Model | CV AUC | Test AUC |
|-------|--------|----------|
| Logistic Regression | 0.6968 | 0.6879 |
| **Random Forest** | **0.7001** | **0.6935** |
| XGBoost (tuned) | 0.6865 | 0.6747 |
| MLP Neural Network | 0.5977 | 0.5831 |

**Best model: Random Forest (Test AUC = 0.6935)**

---

## SHAP Explainability

SHAP TreeExplainer applied to the Random Forest model to identify patient-level prediction drivers.

![SHAP Beeswarm](outputs/shap_beeswarm.png)

**Top predictors by mean |SHAP|:**

| Feature | Mean \|SHAP\| | Interpretation |
|---------|--------------|----------------|
| `sob` | 0.0916 | Shortness of breath — strongest predictor; correlated cardiopulmonary symptom |
| `race_eth_NH_Asian` | 0.0145 | Race/ethnicity differential in symptom reporting |
| `income_pir` | 0.0131 | Lower income associated with higher CVD symptom risk |
| `bmi_x_diabetes` | 0.0089 | Compounded metabolic risk: obesity + diabetes interaction |
| `age_x_diabetes` | 0.0074 | Age amplifies diabetes-associated CVD symptom risk |

![Model Comparison](outputs/model_comparison.png)

---

## Key Findings

- Shortness of breath (SOB) was by far the dominant predictor of CVD symptoms, consistent with clinical literature on correlated cardiopulmonary presentations
- Income-to-poverty ratio emerged as a significant social determinant — lower income independently associated with higher symptom risk
- Engineered interaction terms (BMI × diabetes, age × diabetes) contributed meaningful signal beyond raw features alone
- Random Forest outperformed XGBoost on this dataset, likely due to the moderate sample size and mixed feature types

---

## Repository Structure
