# Customer Churn Prediction

A machine learning system that predicts which customers of a subscription-based telecom service are likely to churn (cancel their subscription), deployed as an interactive Streamlit web app that returns a churn probability for a given customer profile.

## Overview

Customer churn is one of the costliest problems for subscription businesses. This project builds an end-to-end pipeline — from raw, messy customer data to a deployed prediction app — that identifies customers at risk of leaving so that retention efforts can be targeted before they churn.

The final model is a tuned XGBoost classifier that achieves a **ROC-AUC of 0.846** and correctly identifies roughly **80% of customers who go on to churn**.

## Dataset

- **Source:** Telco Customer Churn dataset (IBM public dataset)
- **Size:** 7,043 customer records, 21 original columns
- **Target variable:** `Churn` (Yes/No), converted to binary (1/0)
- **Class distribution:** ~27% churn, ~73% retained — a moderately imbalanced classification problem

## Project Workflow

1. **Data Cleaning**
   - Converted `TotalCharges` from string to numeric, coercing invalid entries to null
   - Dropped `customerID` (no predictive value)
   - Mapped `Churn` to binary values (Yes → 1, No → 0)
   - Filled resulting nulls in `TotalCharges` with 0 (corresponded to customers with 0 tenure)

2. **Exploratory Data Analysis**
   - Reviewed churn rate against contract type, tenure group, and internet service type
   - Surfaced early signal that contract length and service type are strong churn predictors, later confirmed by the models

3. **Feature Engineering**
   - Numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) standardized with `StandardScaler`, fit on the training set only to avoid data leakage
   - Categorical features one-hot encoded with `pd.get_dummies(drop_first=True)`, expanding the dataset to 30 model-ready features
   - 80/20 train/test split, stratified on the target to preserve class balance

4. **Modeling**
   Three models were trained and compared on the same train/test split, each using class-balancing techniques (`class_weight="balanced"` for Logistic Regression and Random Forest, `scale_pos_weight` for XGBoost) to counter class imbalance:

   | Metric              | Logistic Regression | Random Forest | XGBoost (Weighted) |
   |---------------------|---------------------|----------------|---------------------|
   | Accuracy            | 0.739                | 0.771          | 0.762               |
   | Precision (Churn)   | 0.505                | 0.560          | 0.541               |
   | Recall (Churn)      | 0.783                | 0.634          | 0.685               |
   | F1-Score (Churn)    | 0.614                | 0.595          | 0.605               |
   | ROC-AUC             | 0.842                | 0.825          | 0.826               |

5. **Hyperparameter Tuning**
   XGBoost was tuned with `GridSearchCV` (3-fold cross-validation, scoring on ROC-AUC) across `n_estimators`, `max_depth`, `learning_rate`, and `subsample` — 72 candidate combinations, 216 total fits.

   **Best parameters:** `learning_rate=0.05`, `max_depth=3`, `n_estimators=200`, `subsample=0.8`

   **Tuned XGBoost — final test performance:**

   | Metric              | Score |
   |---------------------|-------|
   | Accuracy            | 0.742 |
   | Precision (Churn)   | 0.509 |
   | Recall (Churn)      | 0.802 |
   | F1-Score            | 0.623 |
   | ROC-AUC             | 0.846 |

   The tuned XGBoost model was selected as final — it achieved the best ROC-AUC and the best recall of any model tested. High recall was prioritized because, in a churn use case, failing to identify a customer about to leave (a false negative) is typically more costly than a false alarm.

6. **Explainability (SHAP)**
   `shap.TreeExplainer` was used to generate SHAP values for the tuned XGBoost model. Key findings:
   - **Tenure** is the single strongest predictor — low-tenure customers churn far more often than long-tenured ones
   - **Contract type** matters heavily: one-year and two-year contracts strongly reduce churn risk relative to month-to-month
   - **Fiber optic internet service** is associated with higher churn risk
   - **Electronic check** as a payment method is associated with higher churn risk
   - **No internet service** is associated with lower churn risk
   - **Higher monthly charges** modestly increase churn likelihood
   - Lacking add-on services (online security, tech support, streaming, etc.) slightly increases churn risk

   These results align with published research on telecom churn, supporting confidence that the model is learning genuine signal rather than noise.

7. **Deployment**
   The final model, its fitted `StandardScaler`, and the exact training column layout were serialized with `joblib` and served through a Streamlit web app (`app.py`):
   - Interactive form collecting customer attributes (demographics, account details, subscribed services, contract, billing, payment method)
   - Input is transformed to match the model's exact preprocessing pipeline (one-hot encoding layout + scaling)
   - Outputs a churn prediction (Likely to Churn / Likely to Stay) with the associated probability

## Tech Stack

- **Language:** Python
- **Data handling:** pandas
- **Modeling:** scikit-learn (Logistic Regression, Random Forest, StandardScaler, GridSearchCV), XGBoost
- **Explainability:** SHAP
- **Deployment:** Streamlit
- **Persistence:** joblib

## Repository Structure

```
.
├── CustomerChurnModel.ipynb   # Full data pipeline: cleaning, EDA, feature engineering, modeling, tuning, SHAP
├── app.py                     # Streamlit app for interactive churn prediction
├── churn_model.pkl            # Serialized tuned XGBoost model
├── model_columns.pkl          # Training column layout, used to align app input with the model's expected features
├── scaler.pkl                 # Fitted StandardScaler for numeric features
├── requirements.txt           # Project dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

`requirements.txt` should include:

```
pandas
scikit-learn
xgboost
shap
matplotlib
seaborn
streamlit
joblib
```

### Reproducing the model

Open and run `CustomerChurnModel.ipynb` end to end. It will regenerate `churn_model.pkl`, `model_columns.pkl`, and `scaler.pkl`.

### Running the app

```bash
streamlit run app.py
```

This launches a local web app where you can fill in a customer's attributes and get a churn prediction with its probability.

## Results Summary

This project delivers a complete, working churn prediction pipeline — from raw, messy CSV data through cleaning, feature engineering, model comparison, hyperparameter tuning, explainability analysis, and a deployed interactive app. The final tuned XGBoost model achieves a ROC-AUC of 0.846 and correctly identifies roughly 80% of customers who go on to churn, with explainability results that match established domain knowledge about telecom churn drivers.

## Author

Buchi
