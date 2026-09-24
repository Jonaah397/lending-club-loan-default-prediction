# Machine Learning-Based Loan Default Prediction and Credit Risk Assessment

**Evidence from LendingClub Consumer Loans**

Machine learning project comparing Logistic Regression, Random Forest, and XGBoost for predicting loan defaults on LendingClub data.

---

##  Overview

This project investigates whether machine learning models can predict loan defaults more accurately than traditional statistical methods, using the LendingClub Consumer Loans dataset (2007–2018). Three models are compared — **Logistic Regression**, **Random Forest**, and **XGBoost** — with a strong focus on handling severe class imbalance and delivering interpretable, financially meaningful results.

---

##  Problem Statement

> Can machine learning models outperform traditional statistical models in predicting loan defaults, based on the information available at loan application time?

Accurate prediction matters because:
- **For lenders**: Missed defaults cause direct financial losses.
- **For borrowers**: Wrongly rejected applications deny fair access to credit.

---

##  Dataset

- **Source**: [LendingClub Consumer Loans (2007–2018)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Size**: 2,260,701 loans, 151 columns (raw)
- **Modeling subset**: 1,345,310 loans with final outcome
- **Target**: Binary — `1` = Charged Off (default), `0` = Fully Paid (no default)
- **Class imbalance**: ~80% non-default, ~20% default

### Features Used (10)
`loan_amnt`, `term`, `int_rate`, `grade`, `sub_grade`, `emp_length`, `annual_inc`, `dti`, `fico_range_low`, `fico_range_high`

---

##  Methodology

1. **Data Cleaning**: Ordinal encoding for `term`, `grade`, `sub_grade`, `emp_length`; median imputation for missing values (`emp_length`: 78,511; `dti`: 374).
2. **Train/Test Split**: 80/20 stratified split to preserve class distribution.
3. **Scaling**: `StandardScaler` applied **only** for Logistic Regression; tree models trained on raw features.
4. **Class Imbalance Handling**: `scale_pos_weight = 4.01` used in XGBoost.
5. **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, feature importance, and SHAP.

---

##  Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Time (s) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8002 | 0.4971 | 0.0666 | 0.1174 | 0.7013 | 3.38 |
| Random Forest | 0.7964 | 0.4537 | 0.0972 | 0.1601 | 0.6801 | 307.83 |
| **XGBoost ** | 0.6376 | 0.3131 | **0.6829** | **0.4294** | **0.7125** | 18.83 |

### Key Finding
**XGBoost caught 68.3% of defaults** — a ~7× improvement over Random Forest and ~10× over Logistic Regression — while training in under 19 seconds.

The trade-off: Precision dropped to 31.3%, but this is appropriate in credit risk, where a missed default (False Negative) costs ~10× more than a false alarm (False Positive).

---

##  Confusion Matrices

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 211,733 | 3,617 | 50,137 | 3,575 |
| Random Forest | 209,065 | 6,285 | 48,492 | 5,220 |
| **XGBoost** | 134,886 | 80,464 | **17,031** | **36,681** |

XGBoost missed only 17,031 defaults — a dramatic reduction from ~50,000 in the other models.

---

##  Robustness Check

A second XGBoost model was trained **without `grade` and `sub_grade`** — features that accounted for 91.4% of the original model's importance. Performance barely dropped:

| Model | Recall | ROC-AUC |
|---|---|---|
| XGBoost (full) | 0.6829 | 0.7125 |
| XGBoost (no grade) | **0.6709** | **0.7088** |

This confirms the model is **not over-reliant** on LendingClub's proprietary grading.

---

##  Interpretability (SHAP)

SHAP analysis confirmed that the model's decisions align with credit risk theory:
- **Higher** `grade`, `sub_grade`, `dti`, `int_rate`, `loan_amnt` → push toward **default**
- **Higher** `annual_inc`, `fico_range_low` → push toward **no default**

This shows the model learns **real financial patterns**, not spurious correlations — a critical requirement for regulated lending environments.

---

##  Project Structure

- `Loan_default_prediction.ipynb` — Full analysis notebook (data loading, cleaning, modeling, evaluation, SHAP)

---

##  How to Run

Open the notebook in Google Colab and run all cells. The dataset downloads automatically via `kagglehub`. On first run, you'll be prompted to authenticate with a **free Kaggle account**.

---

##  Limitations

- **Accepted loans only** — selection bias (rejected applications excluded).
- **Temporal drift** — data spans the 2008 financial crisis and changing underwriting policies.
- **Self-reported income** — `annual_inc` may be inaccurate.
- **Limited features** — real credit models use hundreds of variables.

---

##  Future Work

- Temporal validation (train on old vintages, test on new)
- Fairness analysis across demographic groups
- Comparison with LightGBM and neural networks
- Real-time API deployment

---

##  Author

**Jonathan Matura**
[GitHub](https://github.com/Jonaah397)

---

##  License

This project is for educational and portfolio purposes.
