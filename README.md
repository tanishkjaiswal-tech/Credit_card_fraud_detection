# Credit Card Fraud Detection with XGBoost

## Overview

This project builds a machine learning model to detect fraudulent credit card transactions using **XGBoost**.

The notebook uses the `creditcard.csv` dataset, handles the strong class imbalance with **SMOTE (Synthetic Minority Over-sampling Technique)**, trains an XGBoost classifier, evaluates its performance, tunes the classification threshold using F1-score, and visualizes feature importance.

> **Note:** This is a machine learning/educational project and should not be treated as a production fraud-detection system without further validation.

## Dataset

The notebook uses a `creditcard.csv` dataset.

Dataset shape in the notebook:

- **263,383 rows**
- **31 columns**
- `Class` is the target column
- `Class = 0` → Normal transaction
- `Class = 1` → Fraudulent transaction

The dataset contains:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

The notebook reports the following class distribution:

| Class | Count |
|---|---:|
| Normal (0) | 262,903 |
| Fraud (1) | 479 |

This shows a strong imbalance between normal and fraudulent transactions.

## Project Workflow

```text
Credit Card Dataset
        ↓
Data Loading
        ↓
Remove Missing Target Values
        ↓
Separate Features (X) and Target (y)
        ↓
Train-Test Split (80:20)
        ↓
Apply SMOTE to Training Data
        ↓
Train XGBoost Classifier
        ↓
Generate Predictions & Probabilities
        ↓
Evaluate Model
        ↓
Tune Classification Threshold
        ↓
Feature Importance Analysis
```

## Data Preparation

The target column `Class` is separated from the input features:

```python
X = df.drop('Class', axis=1)
y = df['Class']
```

Rows with missing values in the target are removed:

```python
df = df.dropna(subset=['Class'])
```

The dataset is split using:

- Test size: **20%**
- Random state: **42**
- Stratified split: **Yes**

The resulting split contains:

- Training set: **210,705 samples**
  - Normal: 210,322
  - Fraud: 383
- Test set: **52,677 samples**
  - Normal: 52,581
  - Fraud: 96

## Handling Class Imbalance with SMOTE

Because fraudulent transactions are much less common than normal transactions, the notebook applies **SMOTE** to the training data.

Before SMOTE:

```text
Normal: 210,322
Fraud:     383
```

After SMOTE:

```text
Normal: 210,322
Fraud:  210,322
```

SMOTE is applied only to the training set so that the original test distribution remains unchanged.

## Model

The project uses an **XGBoost Classifier**.

Model configuration:

```python
XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)
```

### Main Parameters

| Parameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 5 |
| `learning_rate` | 0.1 |
| `eval_metric` | logloss |
| `random_state` | 42 |

## Model Evaluation

The notebook evaluates the model using:

- Confusion Matrix
- Precision
- Recall
- F1-score
- Accuracy
- ROC-AUC

### Initial Results

Confusion matrix:

```text
[[52543    38]
 [    9    87]]
```

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Normal (0) | 0.9998 | 0.9993 | 0.9996 | 52,581 |
| Fraud (1) | 0.6960 | 0.9062 | 0.7873 | 96 |

Overall accuracy:

```text
0.9991
```

ROC-AUC:

```text
0.98257
```

## Threshold Tuning

Instead of using only the default classification threshold, the notebook calculates precision and recall at different thresholds and selects the threshold with the highest F1-score.

The notebook reports:

```text
Best threshold: 0.99256456
Best F1 score: 0.91011236
```

Using this threshold, the fraud-class metrics become:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Normal (0) | 0.9997 | 1.0000 | 0.9998 |
| Fraud (1) | 0.9878 | 0.8438 | 0.9101 |

The notebook also plots **Precision and Recall vs. Classification Threshold** to visualize this trade-off.

## Feature Importance

The final section uses XGBoost's feature-importance functionality with:

```python
plot_importance(
    model,
    max_num_features=15,
    importance_type='gain'
)
```

This produces a visualization of the **top 15 features based on gain**.

## Technologies Used

- **Python**
- **Pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **Scikit-learn** — train/test split and evaluation metrics
- **XGBoost** — fraud classification model
- **imbalanced-learn** — SMOTE
- **Matplotlib** — visualizations
- **Seaborn** — installed in the notebook environment

## Installation

Install the required libraries with:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

## How to Run

### Google Colab

1. Open the notebook in Google Colab.
2. Upload `creditcard.csv`.
3. Run the cells from top to bottom.
4. Review the model evaluation, threshold-tuning results, and feature-importance plot.

### Jupyter Notebook

1. Install the dependencies.
2. Place `creditcard.csv` in the same directory as the notebook.
3. Open the notebook.
4. Run all cells sequentially.

The notebook expects the dataset at:

```python
df = pd.read_csv("creditcard.csv")
```

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── CreditCard FraudDetection with XGBoost.ipynb
├── creditcard.csv
└── README.md
```

## Key Takeaways

- The dataset has a severe imbalance between normal and fraudulent transactions.
- SMOTE is used to balance the training data.
- XGBoost is trained on the resampled training set.
- The initial model achieves a ROC-AUC of approximately **0.983** on the test set.
- Threshold tuning improves the reported fraud-class F1-score from **0.7873** to **0.9101** on the test set.
- Feature importance is analyzed using XGBoost's gain-based importance.

## Future Improvements

Possible extensions to the project include:

- Compare XGBoost with Logistic Regression, Random Forest, and other classifiers.
- Perform cross-validation.
- Tune XGBoost hyperparameters.
- Compare SMOTE with other imbalance-handling techniques.
- Evaluate PR-AUC in addition to ROC-AUC.
- Analyze false positives and false negatives in more detail.
- Save the trained model for later predictions.
- Build a simple interface/API for transaction prediction.
- Test the model on a separate unseen dataset.

## Author

**Tanishk Jaiswal**

B.Tech — Computer Science Engineering (AI & ML)
