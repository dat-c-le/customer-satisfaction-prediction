# Customer Satisfaction Prediction

A predictive modeling project that helps an online retailer identify potentially churning customers by predicting their satisfaction scores using machine learning techniques in Python.

## Overview

An online retailer collected satisfaction survey responses along with customer behavioral and transactional data. This project builds two models:

1. **Linear Regression (Lasso)** — predicts a customer's satisfaction score (1–7 scale)
2. **Logistic Regression** — classifies customers as *satisfied* (score ≥ 6) or *unsatisfied* (score < 6), enabling the retailer to proactively reach out to at-risk customers

## Dataset

Five CSV files are used across the analysis:

| File | Description |
|---|---|
| `customer1.csv`, `customer2.csv` | Customer demographics: number of devices, addresses, saved cards |
| `activity.csv` | Recent 2-month activity: visit hours, support tickets, coupons used, orders, spending |
| `survey.csv` | Customer satisfaction scores (1 = very unsatisfied, 7 = very satisfied) |
| `customer_to_predict.csv` | Customers with unknown satisfaction scores to be predicted |

## Project Structure

```
├── Exam.ipynb                  # Main Jupyter Notebook with full analysis
├── customer1.csv
├── customer2.csv
├── activity.csv
├── survey.csv
├── customer_to_predict.csv
└── README.md
```

## Methodology

### 1. Data Preparation
- Concatenated `customer1.csv` and `customer2.csv`
- Merged customer, activity, and survey data on `CustomerID`
- Handled three data quality issues:
  - Dropped rows with missing values
  - Removed records where satisfaction score was incorrectly recorded as `0`
  - Removed records where spending > 0 but number of orders was incorrectly recorded as `0`
- Final clean dataset: **5,307 customers, 10 features**
- 80/20 train-test split (random state = 42)

### 2. Exploratory Data Analysis
- Analyzed the relationship between customer spending and satisfaction level
- Used both summary-based bar charts and raw-data Seaborn bar plots
- Finding: customers who spend more tend to report higher satisfaction

### 3. Linear Regression (OLS)
- Built an OLS model using `statsmodels` with 4 hand-picked features: `VisitHours`, `NumCouponUsed`, `NumOrder`, `Spending`
- Identified significant positive predictors: **VisitHours**, **NumCouponUsed**, **Spending**

### 4. Lasso Regression — Feature Selection
- Ran Lasso Regression with 5 different alpha values (0.001, 0.01, 0.1, 0.3, 0.5) to observe feature shrinkage
- Used `LassoCV` with 5-fold cross-validation to select the optimal regularization strength
- Evaluated model R² on both training and test sets
- Visualized ranked feature importance by absolute coefficient value

### 5. Logistic Regression — Churn Classification
- Defined binary target: `Unsatisfied = 1` if satisfaction < 6, else `0`
- Used features selected by the best Lasso model
- Trained a logistic regression model (no regularization) with `sklearn`
- Evaluated on test data:
  - Overall accuracy
  - Confusion matrix heatmap
  - Classification report (precision, recall, F1-score)

### 6. Prediction on New Customers
- Applied the trained logistic regression model to `customer_to_predict.csv`
- Identified CustomerIDs predicted to be unsatisfied (at churn risk)

## Key Results

- **Lasso CV** selected the optimal alpha via 5-fold cross-validation
- Top predictors of satisfaction: **NumDevice** and **Spending** (by coefficient magnitude after regularization)
- Logistic regression successfully classified unsatisfied customers with interpretable results
- **8 customers** from the prediction set were flagged as potentially unsatisfied/churning

## Tech Stack

- Python 3
- `pandas` — data manipulation
- `matplotlib` / `seaborn` — visualization
- `statsmodels` — OLS regression with statistical summary
- `scikit-learn` — Lasso, LassoCV, Logistic Regression, train-test split, metrics

## How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas matplotlib seaborn statsmodels scikit-learn
   ```
3. Open `Exam.ipynb` in Jupyter and run all cells top to bottom
