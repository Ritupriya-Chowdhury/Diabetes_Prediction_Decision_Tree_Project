# Diabetes Prediction with a Decision Tree

A Jupyter notebook that classifies patients as **non-diabetic (N)**, **pre-diabetic (P)**, or **diabetic (Y)** from routine blood-test and demographic data, using a shallow, interpretable decision tree built with scikit-learn.

**Result:** 96.37% accuracy on a held-out test set, using a tree only three levels deep whose rules can be read and checked by hand.

---

## Table of Contents

- [Dataset](#dataset)
- [Workflow](#workflow)
- [Results](#results)
- [What I Have Done](#-what-i-have-done)
- [What I Learned](#-what-i-learned)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Limitations](#limitations)
- [Resources & Links](#-resources--links)
- [Conclusion](#-conclusion)
- [Disclaimer](#disclaimer)

---

## Dataset

The notebook expects a CSV named `Dataset of Diabetes .csv` (note the space before the extension) in the same folder. It is **not included** in this repository, so you will need to download it separately and keep that exact filename, or edit the `pd.read_csv(...)` call.

The raw file contains **1,000 records and 14 columns**, with no missing values.

| Column | Description |
| --- | --- |
| `ID`, `No_Pation` | Record and patient identifiers (dropped before modelling) |
| `Gender` | `M` / `F` |
| `AGE` | Age in years |
| `Urea`, `Cr` | Kidney-function markers (urea, creatinine) |
| `HbA1c` | Glycated hemoglobin |
| `Chol`, `TG`, `HDL`, `LDL`, `VLDL` | Cholesterol, triglycerides, and lipoprotein levels |
| `BMI` | Body mass index |
| `CLASS` | **Target:** `N` = non-diabetic, `P` = pre-diabetic, `Y` = diabetic |

## Workflow

1. **Load and inspect**: `info()`, `head()`, and summary statistics for the 10 numeric features.
2. **Clean the data**
   - Drop the identifier columns `ID` and `No_Pation`.
   - Normalize `Gender` casing (one stray lowercase `f`).
   - Strip trailing whitespace from `CLASS` (which had created spurious labels such as `'Y '` and `'N '`).
   - Remove **174 duplicate rows**, leaving **826 rows**.
3. **Explore**: class distribution, box plots of every numeric feature by class, gender vs. class, and an HbA1c vs. BMI scatter plot.
4. **Prepare**: label-encode `Gender` (F = 0, M = 1) and split 70/30 into train (578 rows) and test (248 rows) with `random_state=42`.
5. **Train** a `DecisionTreeClassifier` with:
   - `max_depth=3`
   - `criterion='gini'`
   - `class_weight='balanced'` (to compensate for the imbalanced classes)
   - `random_state=42`
6. **Evaluate**: accuracy, classification report, and a confusion-matrix heatmap.
7. **Visualize** the tree as text rules and as a plot, saved to `decision_tree.png`.

## Results

**Class distribution after cleaning** (imbalanced):

| Class | Share |
| --- | --- |
| Y (diabetic) | 83.5% |
| N (non-diabetic) | 11.6% |
| P (pre-diabetic) | 4.8% |

**Test-set performance (248 samples):**

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| N | 0.91 | 0.94 | 0.93 | 34 |
| P | 0.71 | 1.00 | 0.83 | 10 |
| Y | 0.99 | 0.97 | 0.98 | 204 |
| **Accuracy** | | | **0.96** | 248 |
| Macro avg | 0.87 | 0.97 | 0.91 | 248 |
| Weighted avg | 0.97 | 0.96 | 0.96 | 248 |

**Learned decision rules:**

```
HbA1c <= 5.65
|   BMI <= 24.80
|   |   VLDL <= 1.45  ->  N
|   |   VLDL >  1.45  ->  Y
|   BMI >  24.80      ->  Y
HbA1c >  5.65
|   HbA1c <= 6.45
|   |   AGE <= 55.50  ->  P
|   |   AGE >  55.50  ->  Y
|   HbA1c >  6.45     ->  Y
```

**HbA1c is by far the dominant predictor**, with BMI, VLDL, and age refining the borderline cases. This is consistent with HbA1c being a standard clinical marker for diabetes.

## What I Have Done

- [x] **Dataset Loading & Initial Inspection:** Loaded the 1,000 × 14 dataset, checked shape and column data types with `info()`, and confirmed there were no missing values.
- [x] **Descriptive Statistics:** Summary statistics (count, mean, std, min, quartiles, max) for all 10 numerical features.
- [x] **Data Cleaning:** Dropped identifier columns, fixed inconsistent `Gender` casing, stripped stray whitespace from `CLASS` labels, and removed 174 duplicate rows.
- [x] **Target Distribution Analysis:** Measured class proportions and visualized them with a count plot to expose the class imbalance.
- [x] **Feature vs. Target Analysis:** Box plots of all 10 numeric features by class, a gender vs. class count plot, and an HbA1c vs. BMI scatter plot colored by class.
- [x] **Preprocessing:** Label-encoded `Gender` and created a 70/30 train/test split.
- [x] **Model Training:** Trained a depth-3, class-weighted Decision Tree classifier.
- [x] **Model Evaluation:** Accuracy, per-class precision/recall/F1 report, and a confusion-matrix heatmap.
- [x] **Model Interpretation:** Exported the tree as readable text rules and a saved plot (`decision_tree.png`).

---

## What I Learned

- [x] **Data Cleaning:** Inconsistent labels can silently corrupt a target. Trailing spaces turned a 3-class problem (`N`, `P`, `Y`) into 5 apparent classes, and 174 of 1,000 rows (about 17%) turned out to be duplicates.
- [x] **Handling Imbalanced Classes:** With about 84% of records in one class, accuracy alone is misleading. `class_weight='balanced'` and per-class metrics give a more honest picture.
- [x] **Interpretable Machine Learning:** A shallow decision tree can perform well while staying fully readable. `export_text` and `plot_tree` make it easy to explain *why* a prediction was made.
- [x] **Model Evaluation:** Reading precision, recall, F1, and the confusion matrix together, and recognizing that small classes (only 10 pre-diabetic test samples) produce unstable metrics.
- [x] **Analytical & Domain Insights:**
   * `HbA1c` is the dominant predictor. The tree's split points (5.65 and 6.45) sit close to commonly used clinical thresholds for pre-diabetes and diabetes.
   * `BMI`, `VLDL`, and `AGE` help refine borderline cases, such as separating pre-diabetic from diabetic patients in the middle HbA1c range.
   * Because `HbA1c` is itself a diagnostic marker, high accuracy should be interpreted with care, since the model may partly be re-learning the diagnostic criteria.

---

## Getting Started

### Tech Stack & Tools

- **Language:** Python
- **Packages**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`

## Repository Structure

```
.
├── diabetes_prediction.ipynb    # Main analysis notebook
├── Dataset of Diabetes .csv     # Input data (add this yourself)
├── decision_tree.png            # Tree plot (generated by the notebook)
└── README.md
```

## Limitations

- **Small test set for the minority class.** The pre-diabetic class has only 10 test samples, so its metrics (especially the 1.00 recall and 0.71 precision) are noisy and should not be over-interpreted.
- **Single train/test split.** The split is random and not stratified, so results may vary with a different seed. Cross-validation would give a more reliable estimate.
- **Possible label leakage.** `HbA1c` is itself used clinically to define diabetes status, so the model may largely be re-learning the diagnostic criteria rather than discovering independent risk factors.

---

## Resources & Links

- ☁️ **Google Colab Notebook:** [View Interactive Code](https://colab.research.google.com/drive/1lN8AZP9TXKRVM2HRHs4QXkjPrdzBLZxr)


---

## Conclusion

A small, interpretable decision tree can separate non-diabetic, pre-diabetic, and diabetic patients with about **96% test accuracy**, driven mainly by `HbA1c` and supported by `BMI`, `VLDL`, and `AGE`. The results are promising, but the small pre-diabetic sample, the single train/test split, and the diagnostic role of `HbA1c` mean they should be validated further before drawing strong conclusions.

**Next Goals:**
* **Robust Validation:** Use stratified splitting and k-fold cross-validation, and tune hyperparameters (`max_depth`, `min_samples_leaf`, `ccp_alpha`).
* **Model Comparison:** Benchmark against Random Forest, Logistic Regression, KNN and SVM.
* **Feature Analysis:** Study feature importances and test how the model performs with `HbA1c` removed.
* **Imbalance Handling:** Try resampling techniques (e.g. SMOTE) for the pre-diabetic class.
* **Deployment:** Save the trained model with `joblib` and wrap it in a simple prediction app.
* **Build an app:** Make a Streamlit where the user enters values and gets the predicted class with probabilities.

---

