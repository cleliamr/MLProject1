# Heart Attack Prediction - Machine Learning Project

This Machine Learning project aims to predict coronary heart disease (`_MICHD` - Myocardial Infarct / Coronary Heart Disease) using a medical dataset (BRFSS survey) of over 320,000 individuals and 321 features. This work is implemented using **pure NumPy**, without relying on external Machine Learning libraries (in accordance with the project guidelines). 

The task is a highly imbalanced binary classification problem (only 8.8 % of the respondents are positive), so models are compared with the F1 score of the positive class rather than accuracy.

---

## Project Structure

The project directory is organized as follows:

```text
MLProject1/
├── helpers.py             # Data loading function and submission file creation
├── implementations.py     # ML models & functions shared between the models such as cross-validation and metrics, all in NumPy
├── run.ipynb              # Full pipeline: preprocessing, model selection, predictions
├── model_comparison.pdf   # CV comparison of all tested models
├── top4_models.pdf        # Detailed CV results of the 4 best models
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

```
The dataset (x_train.csv, y_train.csv, x_test.csv) is not included in the repository.

## Prerequisites & Installation
Requirements: Python 3.x

```bash
git clone https://github.com/cleliamr/MLProject1.git
cd MLProject1
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: numpy, matplotlib (comparison plots only) and jupyter
## Usage & Execution

1. Download the dataset folder and place the folder named dataset/ at the root of the repository (should have x_train.csv, y_train.csv and x_test.csv inside it).
2. Launch the notebook and run all cells from top to bottom:

```bash
jupyter notebook run.ipynb
```
3. The notebook preprocesses the data, runs cross-validation, trains the final
   model and writes the submission `.csv` file.


Two switches at the top of the notebook control the preprocessing:

- USE_ONEHOT: One-hot encode categorical features (True) or keep the raw codes (False, baseline)
- USE_EXTRA_COLS:	Keep BPMEDS, AVEDRNK2, MAXDRNKS, DRNK3GE5 despite >50 % missing values

All random seeds are fixed (seed=42 for the folds), so results are reproducible.

## Code Overview

### `helpers.py`
- `load_csv_data(data_path)` loads the train/test CSVs and returns features, labels and ids.
- `create_csv_submission(ids, y_pred, name)` writes the submission file (predictions are `{-1, 1}`).

### `implementations.py`

**Required methods** (each returns `(w, loss)`):

| Function | Description |
|----------|-------------|
| `mean_squared_error_gd` | Linear regression, gradient descent |
| `mean_squared_error_sgd` | Linear regression, stochastic gradient descent (batch size 1) |
| `least_squares` | Least squares via normal equations |
| `ridge_regression` | Ridge regression via normal equations |
| `logistic_regression` | Logistic regression, gradient descent |
| `reg_logistic_regression` | L2-regularized logistic regression, gradient descent |

**Additional models:**
- `build_tree` / `predict_tree`: decision tree (Gini impurity, quantile-binned features)
- `build_forest` / `predict_forest`: random forest (bootstrap samples + random feature subsets)
- `linear_svm`: linear SVM trained with mini-batch SGD, with optional class weighting

**Evaluation tools:**
- `stratified_kfold_indices`: stratified folds that preserve the ~8.8% positive rate
- `classification_scores`: precision, recall and F1 for the positive class
- `cross_validate`: generic stratified k-fold CV that also selects the best decision threshold by mean F1
- `best_f1`: best F1 over a grid of thresholds


### `run.ipynb`
## Preprocessing 

All statistics (sentinel codes, medians, categories, mean/std) are computed on the **training set only** and applied identically to the test set.

| Step | Description | Features left |
|------|-------------|---------------|
| Raw data | 328,135 training rows, 109,379 test rows | 321 |
| Administrative columns | Drop IDs, interview dates, survey weights, sampling and phone-related columns | 284 |
| BRFSS missing codes | For count questions, `88` means "none" and is mapped to `0`. Then `7, 9, 77, 88, 99, 777, 999, 7777, 9999` are converted to `NaN` when they lie clearly above the bulk of a column's values (> 99th percentile + 1) | 284 |
| High missingness | Drop 96 columns with ≥ 90 % `NaN` (mostly follow-up questions for people with a given condition), then columns with > 50 % `NaN`, except `BPMEDS`, `AVEDRNK2`, `MAXDRNKS`, `DRNK3GE5` whose `NaN` come from skip patterns | 151 |
| Skip-pattern columns | `NaN` → `0` for the drinking questions (non-drinkers) and for `BPMEDS` (not asked = own category) | 151 |
| Imputation | Remaining `NaN` replaced by the training median of each column | 151 |
| Correlated features | Groups of features with \|corr\| > 0.9 are found; one feature per group is kept (the one with the fewest missing values, `_AGEG5YR` forced) | 119 |
| One-hot encoding | Columns with 3 to 10 distinct values are one-hot encoded (first category dropped). Recoded duplicates (`_RACEGR3`, `_RACE_G1`, `_INCOMG`) are removed, and columns that are exact linear combinations of others are removed with a QR decomposition, so that `XᵀX` is invertible | 181 |
| Standardization | Zero mean, unit variance; a bias column is added before training | 181 (+1 bias) |

---

## Model Selection

Every model is evaluated with **stratified 5-fold cross-validation** (same folds for all models). Because of the class imbalance, the classification threshold is not fixed at 0.5: for each model it is chosen as the threshold that maximizes the mean F1 over the folds. Hyperparameters were chosen as follows:

| Model | Hyperparameters |
|-------|-----------------|
| Linear regression (GD) | `gamma = 1/L` (L = largest eigenvalue of `XᵀX/N`), 200 iterations |
| Linear regression (SGD) | `gamma = 1e-5`, 600,000 iterations, batch size 1 |
| Least squares | none |
| Ridge regression | `lambda = 1e-3` (grid `1e-5 … 1e2`) |
| Logistic regression | `gamma = 4/L`, 1000 iterations |
| Regularized logistic regression | `lambda = 1e-4` (grid `1e-6 … 1e-2`), `gamma = 1/(L/4 + 2·lambda)`, 1000 iterations |
| Linear SVM | `lambda = 1e-5`, learning rate `1e-2`, 10 epochs, batch size 256, class weights |
| Decision tree | max depth 8, min leaf size 200, 32 quantile bins |
| Random forest | 50 trees, max depth 12, min leaf size 50, √(#features) features per split |

---

## Results

5-fold stratified CV, mean ± std over the folds, at the best threshold of each model:

| Model | Threshold | F1 | Precision | Recall |
|-------|-----------|----|-----------|--------|
| **Regularized logistic regression** | 0.20 | **0.4272 ± 0.0038** | 0.3527 ± 0.0032 | 0.5418 ± 0.0060 |
| Logistic regression | 0.20 | 0.4271 ± 0.0037 | 0.3532 ± 0.0030 | 0.5402 ± 0.0059 |
| Linear SVM | 0.90 (raw score) | 0.4231 ± 0.0037 | 0.3536 ± 0.0050 | 0.5274 ± 0.0175 |
| Random forest | 0.20 | 0.4160 ± 0.0052 | 0.3452 ± 0.0040 | 0.5234 ± 0.0078 |
| Linear regression (GD) | 0.21 | 0.4135 ± 0.0038 | 0.3340 ± 0.0027 | 0.5425 ± 0.0060 |
| Ridge regression | 0.21 | 0.4134 ± 0.0040 | 0.3343 ± 0.0029 | 0.5416 ± 0.0063 |
| Least squares | 0.21 | 0.4133 ± 0.0041 | 0.3339 ± 0.0030 | 0.5421 ± 0.0065 |
| Linear regression (SGD) | 0.21 | 0.4123 ± 0.0028 | 0.3338 ± 0.0039 | 0.5396 ± 0.0144 |
| Decision tree | 0.21 | 0.4060 ± 0.0038 | 0.3290 ± 0.0068 | 0.5305 ± 0.0118 |
| *Trivial baseline (always predict +1)* | - | *0.1623* | *0.0883* | *1.0000* |

**Final model:** regularized logistic regression (`lambda = 1e-4`, threshold 0.20), trained on all training rows. Its advantage over plain logistic regression (0.0001 F1) is far below the fold-to-fold standard deviation, so the two are effectively tied. Both outperform the linear regression family and the tree-based models; the margin over the linear SVM (about 0.004 F1) is of the order of one standard deviation.

Plots: `model_comparison.pdf` (all models) and `top4_models.pdf` (best four).

## Authors & Contributions

- Valeria Gonzalez Perez: Initial function implementation of mandatory ML models, Training of Least squares, Ridge regression, Linear SVM, Report (Abstract, Discussion and Conclusion)
- Ella Aeby: Preprocessing, Training of Logistic regression, regularized logistic regression, Random Forest, Report (Results)
- Clelia Messin-Roizard: Preprocessing, Training of linear regression with GD and SGD, Decision Tree, README file, Report (Introduction and Preprocessing)