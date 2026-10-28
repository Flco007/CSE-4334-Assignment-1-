# Project 2 Code Walkthrough

**Course:** CSE 4334/5334 Data Mining, Fall 2026, Project 2
**Team:** Joseph Fitchet, Rodrigo Pleitez

This document goes through the submitted `final_submission.ipynb` **cell by cell**. Every code block is copied exactly from the notebook, so cell numbers match what you'll see on screen. Each block is followed by:

* **What each function and argument does**
* **The choice that was made**, and the alternative that was turned down
* **How it aligns** with the assignment spec (quoted) and with `Project2_Code_Explainer.md` (§ = explainer section, Q = explainer question)

Use the **Code Explainer** to talk about the project. Use this **Walkthrough** when someone points at a line of code and asks "why is that there?"

---

## 0. Map: notebook → spec → explainer

| Cells | What | Spec requirement | Explainer |
|---|---|---|---|
| 0–2 | Install, imports, shared constants | Environment: Scikit-Learn / Imbalanced-Learn | §3.1 |
| 4, 13–15 | `load_data`, `make_preprocessor`, checks | Task 1: Unified Preprocessing | §3.2, Q4–Q7 |
| 6, 17–19 | `evaluate`, `show`, baselines | Task 2: Stratified Cross-Validation | §3.3–3.4, Q8–Q10 |
| 21–25 | Decision tree grid search and validation curve | Task 3: DT tuning | §3.5, Q11–Q14 |
| 27–33 | k-NN grid search and validation curve | Task 3: k-NN tuning | §3.6, Q15–Q18 |
| 11, 35–36 | RIPPER wrapper, evaluation, rules | Rule-Based models; Challenge C | §3.7, Q19–Q21 |
| 41–44 | SMOTE pipelines and grid searches | Task 4: SMOTE inside `imblearn.pipeline.Pipeline` | §3.8, Q22–Q24 |
| 8–9, 45–47, 38 | `cost_sweep`, out-of-fold probabilities, cost plot | Task 4: threshold sweep and total cost | §3.9, Q25–Q26 |
| 39–40 | Summary table, save | Report: model comparison tables | §2.1 |
| 48 | Debugged AI pipeline | Part II: AI Prompt Audit | §3.10, Q27 |

---

## 1. Setup (cells 0–2)

> **Spec:** "Environment: Python 3.8+ / Scikit-Learn / Imbalanced-Learn."  **Explainer:** §3.1

**Cell 0**

```python
# this is for RIPPER
%pip install wittgenstein
```

| Line | What it does | Why |
|---|---|---|
| `%pip install wittgenstein` | A Jupyter *magic* that runs pip inside **the Python the kernel is using** | Plain `!pip` or a terminal `pip` can install into a different environment. That's exactly the `ModuleNotFoundError` you hit earlier. scikit-learn has no RIPPER, so this library supplies it. |

**Cell 1**

```python
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
```

* **`os`, `joblib`:** used only by the save cell (40).
* **`DummyClassifier`:** the do-nothing baseline.
* **`GridSearchCV`:** the Task 3 tuner.
* **`DecisionTreeClassifier`:** the tree model.

**Cell 2**

```python
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 21
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
COST_FN, COST_FP = 100, 10          # C(+,-), C(-,+)

DATA_PATH = "data/adult.data"

# the raw file has no header so we name the columns ourselves order comes from adult.names
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex",
           "capital-gain", "capital-loss", "hours-per-week", "native-country",
           "income"]
NUMERIC_COLS = ["age", "fnlwgt", "education-num", "capital-gain",
                "capital-loss", "hours-per-week"]
CATEGORICAL_COLS = ["workclass", "education", "marital-status", "occupation",
                    "relationship", "race", "sex", "native-country"]

# we keep all 14 attributes to match the spec, so nothing is dropped for now
DROP_COLS = []
```

| Name | Value | Why it's defined once, here |
|---|---|---|
| `RANDOM_STATE` | `21` | One seed for every model, split and sampler, so results are reproducible. The value itself doesn't matter; using the same one everywhere does. |
| `CV` | `StratifiedKFold(n_splits=5, shuffle=True, random_state=21)` | **The most important line in the notebook.** It's one splitter object passed to `evaluate()`, every `GridSearchCV`, and `cross_val_predict`, so every model is trained and scored on the **same 5 splits**. `n_splits=5` is the spec's 5-fold. Stratified keeps the 24/76 class ratio in every fold. `shuffle=True` with a seed mixes the rows randomly but repeatably. |
| `SCORING` | 5 metric names | Exactly the spec's five metrics. scikit-learn recognizes these strings directly. |
| `COST_FN, COST_FP` | `100, 10` | The spec's C(+,−) = $100 and C(−,+) = $10 |
| `DATA_PATH` | `"data/adult.data"` | A relative path, so the notebook runs from the project folder on any machine |
| `COLUMNS` | 15 names | `adult.data` has no header row. The names and their order come from `adult.names`. |
| `NUMERIC_COLS`, `CATEGORICAL_COLS` | 6 + 8 names | Tells the `ColumnTransformer` which branch each column goes through. Listing them explicitly is safer than auto-detecting by dtype, which changed between pandas 2 and 3 (see the warning in cell 48's output). |
| `DROP_COLS` | `[]` | Keeps all 14 attributes, as in the spec's "14 mixed continuous and categorical attributes". It's the single place to drop `fnlwgt` later if you wanted to. |

**Alternative not taken:** setting the seed separately in every cell, which makes it easy to end up with mismatched folds.

---

## 2. Task 1: Unified Preprocessing (cells 4, 13–15)

> **Spec:** "Construct a master ColumnTransformer: Continuous features: SimpleImputer(strategy='median') + StandardScaler(). Categorical features: SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')." Also: "missing values marked as ? in categorical attributes."  **Explainer:** §3.2, Q4–Q7

**Cell 4**

```python
def load_data(path=DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Read the raw file and apply STATELESS cleaning only.
    - assign COLUMNS (file has no header)
    - strip leading whitespace from values
    - replace '?' with NaN
    - encode target: '>50K' -> 1, '<=50K' -> 0
    - drop columns listed in DROP_COLS
    Returns X (features) and y (0/1 target)."""

    df = pd.read_csv(path, names=COLUMNS, header=None,
                     skipinitialspace=True, na_values="?")

    # we encode the target as 1 for >50K and 0 for <=50K
    y = (df["income"].str.rstrip(".") == ">50K").astype(int)

    # we split off the features by dropping the target and any columns we chose to drop
    X = df.drop(columns=["income"] + DROP_COLS)

    return X, y


def make_preprocessor() -> ColumnTransformer:
    """Return an UNFITTED Task 1 ColumnTransformer.
    Never call .fit() here. It is fitted inside each CV fold."""

    numeric_features = [c for c in NUMERIC_COLS if c not in DROP_COLS]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler())]
    )

    categorical_features = [c for c in CATEGORICAL_COLS if c not in DROP_COLS]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
```

### `load_data()`: stateless cleaning

| Call / argument | What it does | Why this choice |
|---|---|---|
| `pd.read_csv(path, ...)` | Reads a comma-separated text file into a DataFrame | `.data` is just CSV with no header |
| `names=COLUMNS, header=None` | Use our names; treat line 1 as data | Without `header=None`, the first person becomes the column names |
| `skipinitialspace=True` | Strips the space after each comma | Raw values look like `" Private"` and `" ?"`. Without this, `"?"` wouldn't match `na_values`. |
| `na_values="?"` | Turns every `?` into a real `NaN` | `SimpleImputer` only fills `NaN`. Left as text, `?` would become its own category: 1,836 + 1,843 + 583 values. |
| `df["income"].str.rstrip(".")` | Removes a trailing period | `adult.test` labels are `>50K.`, so the same function works on both files |
| `== ">50K"` then `.astype(int)` | True/False becomes 1/0 | Positive class = 1. scikit-learn's precision, recall and F1 default to `pos_label=1`. With text labels, `scoring="f1"` crashes (AI audit Bug 2). |
| `df.drop(columns=["income"] + DROP_COLS)` | Features only | Keeps the target out of X |
| Returns `X, y` | Features DataFrame, target Series | Kept separate so scikit-learn can split them in parallel |

**Why it's "stateless":** nothing here learns from the data. It names columns, strips spaces and recodes values, so it's safe to run once on all 32,561 rows.

### `make_preprocessor()`: stateful recipe, unfitted

| Object | What it does | Why |
|---|---|---|
| `SimpleImputer(strategy="median")` | Learns each numeric column's median on the training fold and fills `NaN` with it | Spec choice. The median isn't dragged around by capital-gain's extreme values. |
| `StandardScaler()` | Learns the mean and std, then outputs (x − mean) / std | Spec choice. Puts every numeric feature on mean 0 and std 1, so k-NN distances aren't dominated by capital-gain (Challenge B). |
| `SimpleImputer(strategy="most_frequent")` | Learns each categorical column's mode and fills `NaN` with it | Spec choice. The only simple strategy that works on text. |
| `OneHotEncoder(handle_unknown="ignore")` | One 0/1 column per category. An unseen category becomes all zeros. | Spec choice. `Holand-Netherlands` appears in **1 row**, so in its test fold the encoder has never seen it. `"ignore"` prevents a crash. |
| `Pipeline(steps=[...])` | Runs steps **in order** on the same columns | Imputing must happen before scaling or encoding |
| `ColumnTransformer(transformers=[...])` | Runs the two pipelines **side by side** on different columns, then stacks the results | Numbers and text need different treatment. Output is 6 + 99 = 105 columns. |
| `[c for c in ... if c not in DROP_COLS]` | Filters the column lists | If `DROP_COLS` ever changes, the preprocessor follows automatically instead of crashing on a missing column |
| Returned **unfitted** | No `.fit()` inside | Each fold fits its own copy on training rows only. That's the leakage prevention (Q1). |

**Step names** (`imputer`, `scaler`, `encoder`, `num`, `cat`) match scikit-learn's "Column Transformer with Mixed Types" example, so nested parameter names look like `prep__num__imputer__strategy`.

**Alternatives not taken:**
* mean imputation, which outliers pull around
* `RobustScaler`, which the spec doesn't ask for
* `OrdinalEncoder`, which implies a fake order between categories
* doing all of this with pandas before CV, which leaks

**Cell 13**

```python
#load the data and check it matches the spec
X, y = load_data()

print("Rows:", len(X))                      # expect 32561
print("Share >50K:", round(y.mean(), 3))    # expect ~0.241
```

* **`len(X)`** should be 32,561, matching the spec's dataset size.
* **`y.mean()`** gives the share of 1s, which should be 0.241, matching the spec's "24.1%".

These checks prove the right file was loaded the right way.

**Cell 14**

```python
# checking to see what has missing values
X.isna().sum()
```

* **`.isna().sum()`** counts the `NaN`s per column.
* **Expected result:** workclass 1,836, occupation 1,843, native-country 583, and zero everywhere else.
* **If every count were 0,** the `?` conversion would have failed.

**Cell 15**

```python
# inspecting the shape 
print(make_preprocessor().fit_transform(X).shape)
```

* **Inspection only.** This fits on all rows *just to look at the output shape*, and the result is never used for modeling.
* **Output:** (32561, 105), which is 6 numeric columns plus 99 one-hot columns. The 99 breaks down as 8 + 16 + 7 + 14 + 6 + 5 + 2 + 41 categories.

---

## 3. Task 2: Evaluation harness and baselines (cells 6, 17–19)

> **Spec:** "Benchmark Decision Trees, k-NN, and Rule-Based models using 5-Fold StratifiedKFold. Compute Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Generate comparative bar plots with standard deviation error bars."  **Explainer:** §3.3–3.4, Q8–Q10

**Cell 6**

```python
def evaluate(pipeline, X, y) -> dict:

    fold_scores = {m: [] for m in SCORING}

    # one out-of-fold probability per row, filled in as each fold is scored
    oof_proba = np.zeros(len(y))

    # this loop is where cross-validation happens: 5 folds, each one a train/test split
    for train_idx, test_idx in CV.split(X, y):

        # we clone so every fold starts from a fresh, unfitted copy of the pipeline
        model = clone(pipeline)

        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        # we score on the held-out rows the model never saw
        y_true = y.iloc[test_idx]
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (proba > 0.5).astype(int)   # default 0.5 threshold; Task 4 moves this

        fold_scores["accuracy"].append(accuracy_score(y_true, pred))
        fold_scores["precision"].append(precision_score(y_true, pred, zero_division=0))
        fold_scores["recall"].append(recall_score(y_true, pred, zero_division=0))
        fold_scores["f1"].append(f1_score(y_true, pred, zero_division=0))
        fold_scores["roc_auc"].append(roc_auc_score(y_true, proba))

        oof_proba[test_idx] = proba

    metrics = {m: (np.mean(v), np.std(v)) for m, v in fold_scores.items()}
    return {"metrics": metrics, "oof_proba": oof_proba}
```

| Line | What it does | Why |
|---|---|---|
| `fold_scores = {m: [] for m in SCORING}` | One empty list per metric | Collects 5 scores per metric, so we can report mean **and std** for the spec's error bars |
| `oof_proba = np.zeros(len(y))` | One slot per row | Filled in fold by fold. Ends with an out-of-fold probability for every row. |
| `for train_idx, test_idx in CV.split(X, y)` | Yields 5 pairs of row-index arrays | **This loop is the cross-validation.** `y` is passed so the splits can be stratified. |
| `model = clone(pipeline)` | A fresh, **unfitted** copy with the same settings | So fold 2 doesn't start from fold 1's fitted model |
| `model.fit(X.iloc[train_idx], y.iloc[train_idx])` | Fits the **whole pipeline**: imputers, scaler, encoder, SMOTE if present, then the model | Training rows only, so there's no leakage. `.iloc` selects by position, which is what `CV.split` returns. |
| `model.predict_proba(...)[:, 1]` | Column 1 = P(class 1) = P(>50K) | Probabilities, not just labels, so ROC-AUC and the cost sweep are possible |
| `pred = (proba > 0.5).astype(int)` | Converts probabilities to labels at the default threshold | 0.5 is the standard default. Task 4 then moves the threshold. |
| `precision_score(..., zero_division=0)` | Returns 0 instead of warning when nothing is predicted positive | The dummy model never predicts positive, so precision would be 0/0 |
| `roc_auc_score(y_true, proba)` | Uses the **probabilities** | AUC measures ranking, so it needs scores, not labels |
| `oof_proba[test_idx] = proba` | Stores this fold's predictions in the right rows | Each row gets a probability from a model that never trained on it |
| `np.mean(v), np.std(v)` | Mean and spread across the 5 folds | The ± values, and the spec's error bars |
| Returns a dict | `{"metrics": ..., "oof_proba": ...}` | One consistent shape for every model, so they all stack into one table |

**Why a hand-written loop instead of `cross_validate` plus `cross_val_predict`?** Those two would fit every model **twice**: once for the scores and once for the probabilities. That doubles the cost of slow models like RIPPER and k-NN. The explicit loop also shows the leak-free design line by line, which helps in the demo.

**Cell 17**

```python
# we collect every model's results here
results = {}

# small helper so every model prints the same way mean ± std for each metric
def show(name, res):
    print(name)
    for m, (mean, std) in res["metrics"].items():
        print(f"  {m:<10} {mean:.3f} ± {std:.3f}")
```

* **`results`** is a dict of `{model name: evaluate() output}`. The summary table (cell 39) and a Task 2 bar plot would both read from it.
* **`show()`** prints each metric as `mean ± std` with an f-string. `{m:<10}` left-aligns the metric names in a 10-character column.

**Cell 18**

```python
#baseline pipeline 
dummy_pipe = Pipeline(
    steps=[
        ("prep", make_preprocessor()),
        ("clf", DummyClassifier(strategy="most_frequent"))])

results["Dummy (majority)"] = evaluate(dummy_pipe, X, y)

show("Dummy (majority)", results["Dummy (majority)"])
```

| Piece | Why |
|---|---|
| `DummyClassifier(strategy="most_frequent")` | Always predicts the majority class (≤50K). It's a floor: any real model must beat it. |
| Same `make_preprocessor()` step | Keeps the comparison fair. The only difference between models is the `clf` step. |
| Result: accuracy 0.759, F1 0 | The **accuracy paradox** (Challenge A) in your own numbers |

**Cell 19**

```python
#default tree
dt_default = Pipeline(
    steps=[
        ("prep", make_preprocessor()),
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])

results["Decision Tree (default)"] = evaluate(dt_default, X, y)

show("Decision Tree (default)", results["Decision Tree (default)"])
```

* **`DecisionTreeClassifier(random_state=RANDOM_STATE)`** uses all defaults: `max_depth=None` and `min_samples_split=2`. So the tree grows until every leaf is pure (explainer §3.4).
* **`random_state`** fixes how tied splits are broken, so the tree is reproducible.
* **Result:** F1 0.621, ROC-AUC 0.751. This is the "before tuning" number Task 3 improves on.

---

## 4. Task 3: Decision tree tuning (cells 21–25)

> **Spec:** "Execute systematic tuning via GridSearchCV: Decision Trees: max_depth (3 to 20), min_samples_split, criterion ('gini' vs 'entropy')."  **Explainer:** §3.5, Q11–Q14

**Cell 21**

```python
dt_pipe = Pipeline(
    steps=[
        ("prep", make_preprocessor()),
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])

param_grid = {
    "clf__max_depth": list(range(3, 21)),                  
    "clf__min_samples_split": [2, 10, 50, 100, 200],       
    "clf__criterion": ["gini", "entropy"],
}

print("Candidates:", 18 * 5 * 2, "x 5 folds =", 18 * 5 * 2 * 5, "fits")
```

| Key | Values | Why |
|---|---|---|
| `clf__` prefix | — | A Pipeline exposes its step `clf`'s parameters as `clf__<param>` (Q3) |
| `clf__max_depth` | `list(range(3, 21))`, which is 3 through 20 | Exactly the spec's range. `range` stops *before* 21. |
| `clf__min_samples_split` | `[2, 10, 50, 100, 200]` | The spec gives no values. These run from the default (2, no limit) to about 1% of a 26k-row training fold, roughly evenly spread on a log scale. |
| `clf__criterion` | `["gini", "entropy"]` | The spec's comparison |
| `print(...)` | 18 × 5 × 2 = 180 candidates, × 5 folds = 900 fits | Shows the search is exhaustive, which is what *Grid*SearchCV means |

**Cell 22**

```python
#winner by F1 
dt_search = GridSearchCV(
    dt_pipe,
    param_grid=param_grid,
    scoring=SCORING,
    refit="f1",
    cv=CV,
    n_jobs=-1,
    return_train_score=True,
)
dt_search.fit(X, y)  

print("Best parameters:", dt_search.best_params_)
print("Best mean CV F1:", round(dt_search.best_score_, 4))
```

| Argument | What it does | Why this choice |
|---|---|---|
| `dt_pipe` | The pipeline to tune | Preprocessing is refit inside every fold of the search too |
| `param_grid` | Every combination gets tried | Spec: "systematic tuning via GridSearchCV" |
| `scoring=SCORING` | Records all 5 metrics for every candidate | Lets you compare candidates on AUC or accuracy later without rerunning |
| `refit="f1"` | Picks the winner by mean F1, then refits it on all data as `best_estimator_` | **Not accuracy**: accuracy rewards predicting ≤50K (the paradox). ROC-AUC was the alternative (Q11). |
| `cv=CV` | Same 5 folds as `evaluate()` | Tuning and scoring see identical splits |
| `n_jobs=-1` | Uses every CPU core | 900 fits, about 5 minutes on 2 cores |
| `return_train_score=True` | Also stores the training-fold scores | Needed for the dashed training lines in the validation curve. Off by default because it costs time. |
| `dt_search.fit(X, y)` | Runs the search | — |
| `best_params_`, `best_score_` | The winning settings and their mean CV F1 | entropy, depth 14, split 50, F1 0.6847 |

**Cell 23**

```python
#top 10 candidates by F1, with the other metrics alongside
cv_table = pd.DataFrame(dt_search.cv_results_)
cols = ["param_clf__criterion", "param_clf__max_depth", "param_clf__min_samples_split",
        "mean_test_f1", "std_test_f1", "mean_train_f1", "mean_test_roc_auc", "mean_test_accuracy"]
cv_table.sort_values("rank_test_f1")[cols].head(10).round(4)
```

* **`cv_results_`** is a dict of arrays with one entry per candidate. `pd.DataFrame(...)` turns it into a table with 180 rows.
* **Column names follow a pattern:** `param_clf__max_depth` holds the setting, and `mean_test_f1`, `std_test_f1` and `mean_train_f1` hold the scores. Scores get a metric suffix because `scoring` is a list.
* **`sort_values("rank_test_f1")`** puts rank 1 first. `.head(10).round(4)` keeps the table readable.
* **What it shows:** the top 8 are within 0.004 F1 of each other, less than one std (about 0.015). The winner is only slightly ahead, which is the selection-bias point in Q14.

**Cell 24**

```python
best_split = dt_search.best_params_["clf__min_samples_split"]
fig, ax = plt.subplots(figsize=(9, 5))

for criterion, color in [("gini", "tab:blue"), ("entropy", "tab:orange")]:
    rows = cv_table[(cv_table["param_clf__criterion"] == criterion) &
                    (cv_table["param_clf__min_samples_split"] == best_split)].sort_values("param_clf__max_depth")
    depth = rows["param_clf__max_depth"].astype(int)
    #test folds with a ±1 std band, training folds dashed
    ax.plot(depth, rows["mean_test_f1"], "-", color=color, label=f"{criterion} (test)")
    ax.fill_between(depth, rows["mean_test_f1"] - rows["std_test_f1"],
                    rows["mean_test_f1"] + rows["std_test_f1"], color=color, alpha=0.15)
    ax.plot(depth, rows["mean_train_f1"], "--", color=color, alpha=0.7, label=f"{criterion} (train)")

best_depth = dt_search.best_params_["clf__max_depth"]
ax.axvline(best_depth, linestyle=":", color="gray")
ax.annotate(f"best: depth {best_depth}", (best_depth + 0.2, dt_search.best_score_ - 0.03))
ax.set_xticks(range(3, 21))
ax.set_xlabel("max_depth")
ax.set_ylabel("F1 (positive class >50K)")
ax.set_title(f"Decision tree F1 vs. depth (min_samples_split={best_split})")
ax.legend()
plt.show()
```

| Call | What it does | Why |
|---|---|---|
| `best_split = dt_search.best_params_[...]` | Fixes `min_samples_split` at the winning value (50) | Shows one line per criterion against depth, at the best split value |
| `fig, ax = plt.subplots(figsize=(9, 5))` | The object-oriented matplotlib style: one figure, one axes | Every call names the axes it draws on, which is clearer than `plt.` global state |
| `cv_table[(...) & (...)]` | Boolean mask selecting one criterion at one split value | `&` means AND for pandas, and each condition needs its own parentheses |
| `.sort_values("param_clf__max_depth")` | Orders the points by depth | Otherwise the line zig-zags |
| `.astype(int)` | The depth column comes out as generic objects | Makes it a clean number for plotting |
| `ax.plot(..., "-")` / `ax.plot(..., "--")` | Solid line for test folds, dashed for training folds | The gap between them is the overfitting signal |
| `ax.fill_between(mean − std, mean + std, alpha=0.15)` | Shaded ±1 std band | Shows fold-to-fold variability. `alpha` keeps the lines visible through it. |
| `ax.axvline(best_depth, linestyle=":")` | Vertical line at depth 14 | Marks the chosen value |
| `ax.annotate(text, (x, y))` | Text label near the best point | Placed in data coordinates |
| `ax.set_xticks(range(3, 21))` | A tick at every whole depth | Without it, matplotlib shows 2.5, 5.0, and so on |
| `set_xlabel/ylabel/title`, `legend`, `plt.show()` | Labels and render | — |

**Reading it:**
* **Depth 3–8:** both lines are low and close together. That's underfitting.
* **Depth 13–14:** test F1 peaks.
* **Past that:** training F1 keeps rising while test F1 drifts down. That's overfitting.
* **Gini vs. entropy:** the two overlap almost everywhere, so the criterion barely matters (Q13).

The structure is adapted from scikit-learn's "Demonstration of multi-metric evaluation on cross_val_score and GridSearchCV" example, which also uses a decision tree.

**Cell 25**

```python
#evaluate() clones the pipeline, so this re-fits the best settings fold by fold
dt_best = dt_search.best_estimator_
results["Decision Tree (tuned)"] = evaluate(dt_best, X, y)
show("Decision Tree (tuned)", results["Decision Tree (tuned)"])
```

* **`best_estimator_`** is the winning pipeline, already refit on all the data.
* **`evaluate()` clones it back to unfitted** and re-scores it fold by fold through the same harness. That way its row in the comparison table is produced exactly like every other model's.
* **Result:** F1 0.685, ROC-AUC 0.895.
* **Caveat:** the score is slightly optimistic, because the same folds chose the settings (Q14, model-selection bias).

---

## 5. Task 3: k-NN tuning (cells 27–33)

> **Spec:** "k-NN: n_neighbors (k ∈ {1, 3, 7, 15, 31}), weights ('uniform' vs 'distance'), metric ('euclidean' vs 'manhattan')."  **Explainer:** §3.6, Q15–Q18

**Cell 27**

```python
from sklearn.neighbors import KNeighborsClassifier
```

**Cell 28**

```python
# k-NN pipeline and grid search
knn_pipeline = Pipeline(
    steps=[
        ("prep", make_preprocessor()),
        ("model", KNeighborsClassifier())])

knn_grid = {
    "model__n_neighbors": [1, 3, 7, 15, 31],
    "model__weights": ['uniform', 'distance'],
    "model__metric": ["euclidean", "manhattan"]
}

knn_search = GridSearchCV(
    knn_pipeline,
    param_grid=knn_grid,
    scoring=SCORING,
    refit="f1",
    cv=CV,
    n_jobs=-1,
    return_train_score=True
)
```

| Piece | What it does | Why |
|---|---|---|
| `("prep", make_preprocessor())` | The same Task 1 preprocessor | Scaling is **essential** for k-NN, because distances use every column (Challenge B) |
| `("model", KNeighborsClassifier())` | The k-NN step, named `model` | Because of the name, its grid keys are `model__...` rather than the tree's `clf__...`. Both work. |
| `model__n_neighbors: [1, 3, 7, 15, 31]` | How many neighbors vote | Exactly the spec's values. Odd numbers avoid tied votes between two classes. |
| `model__weights: ['uniform', 'distance']` | Every neighbor gets an equal vote, or closer neighbors count more | Spec |
| `model__metric: ["euclidean", "manhattan"]` | Straight-line distance vs. sum of absolute differences | Spec |
| `GridSearchCV(...)` | Same settings as the tree: `SCORING`, `refit="f1"`, `cv=CV`, `n_jobs=-1`, `return_train_score=True` | Consistency: same folds and same selection rule |

20 candidates × 5 folds = 100 fits. k-NN is slow because it has no real training step: each prediction searches the whole training fold.

**Cell 29**

```python
knn_search.fit(X, y)
```

**Cell 30**

```python
print("Best parameters:", knn_search.best_params_)
print("Best mean CV F1:", round(knn_search.best_score_, 4))
```

* **`knn_search.fit(X, y)`** runs the search.
* **The winner:** euclidean, k = 31, distance weighting, CV F1 0.650.
* **k = 31 is the largest value in the grid.** The true best might be even larger (Q16).

**Cell 31**

```python
#top 10 candidates by F1, with the other metrics alongside
cv_table2 = pd.DataFrame(knn_search.cv_results_)
cols2 = ["param_model__metric", "param_model__n_neighbors", "param_model__weights",
        "mean_test_f1", "std_test_f1", "mean_train_f1", "mean_test_roc_auc", "mean_test_accuracy"]
cv_table2.sort_values("rank_test_f1")[cols2].head(10).round(4)
```

* **Same table as cell 23,** with `param_model__` columns.
* **`mean_train_f1 = 1.0` for every `distance` row.** Each training point is its own neighbor at distance 0, so it always votes for itself. That says nothing about generalization (Q17).

**Cell 32**

```python
# score the tuned k-NN through the same harness
best_split = knn_search.best_params_["model__weights"]
fig, ax = plt.subplots(figsize=(9, 5))

for criterion, color in [("euclidean", "tab:blue"), ("manhattan", "tab:orange")]:
    rows = cv_table2[(cv_table2["param_model__metric"] == criterion) &
                    (cv_table2["param_model__weights"] == best_split)].sort_values("param_model__n_neighbors")
    neighbors = rows["param_model__n_neighbors"].astype(int)
    #test folds with a ±1 std band, training folds dashed
    ax.plot(neighbors, rows["mean_test_f1"], "-", color=color, label=f"{criterion} (test)")
    ax.fill_between(neighbors, rows["mean_test_f1"] - rows["std_test_f1"],
                    rows["mean_test_f1"] + rows["std_test_f1"], color=color, alpha=0.15)
    ax.plot(neighbors, rows["mean_train_f1"], "--", color=color, alpha=0.7, label=f"{criterion} (train)")

best_neighbors = knn_search.best_params_["model__n_neighbors"]
ax.axvline(best_neighbors, linestyle=":", color="gray")
ax.annotate(f"best: neighbors {best_neighbors}", (best_neighbors + 0.2, knn_search.best_score_ - 0.03))
ax.set_xticks([1, 3, 7, 15, 31])
ax.set_xlabel("n_neighbors")
ax.set_ylabel("F1 (positive class >50K)")
ax.set_title(f"KNN search F1 vs. neighbors (min_samples_split={best_split})")
ax.legend()
plt.show()
```

* **Same plot pattern as cell 24.** The x-axis is `n_neighbors`, with one line per metric, fixed at the winning `weights` value.
* **`set_xticks([1, 3, 7, 15, 31])`** puts ticks only at the tested k values.
* **The title has a copy-paste label.** It says `min_samples_split=`, but the value shown is the `weights` setting. The variable `best_split` is also reused here, which overwrites the tree's value from cell 24. Both are cosmetic (explainer §6).

**Cell 33**

```python
knn_best = knn_search.best_estimator_
results["KNN Search (tuned)"] = evaluate(knn_best, X, y)
show("KNN Search (tuned)", results["KNN Search (tuned)"])
```

* **Same as cell 25:** the winning pipeline goes back through `evaluate()`.
* **Result:** F1 0.650, ROC-AUC 0.895, the same AUC as the tuned tree.

---

## 6. Rule-based model: RIPPER (cells 11, 35–36)

> **Spec:** "Rule-Based / Cost-Sensitive Models — Sequential covering, FOIL Information Gain…"; Task 2: "Benchmark … Rule-Based models"; Challenge C: "FOIL Information Gain in RIPPER."  **Explainer:** §3.7, Q19–Q21

**Cell 11**

```python
import numpy as np
import pandas as pd
import wittgenstein as lw
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted



# Ripper wrapper for skicit
class RipperClassifier(ClassifierMixin, BaseEstimator):


    def __init__(self, k=2, prune_size=0.33, dl_allowance=64, n_discretize_bins=10,
                 max_rules=None, max_rule_conds=None, random_state=None):
        self.k = k
        self.prune_size = prune_size
        self.dl_allowance = dl_allowance
        self.n_discretize_bins = n_discretize_bins
        self.max_rules = max_rules
        self.max_rule_conds = max_rule_conds
        self.random_state = random_state

    def _to_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X)
        return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

    def fit(self, X, y):
        X = self._to_frame(X)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y have different numbers of rows")

        # we store the classes seen during fit predict_proba columns follow this order
        self.classes_ = unique_labels(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("RipperClassifier expects a binary 0/1 target")
        self.n_features_in_ = X.shape[1]

        # we build a fresh RIPPER every fit so a refit never reuses old rules
        self.model_ = lw.RIPPER(
            k=self.k,
            prune_size=self.prune_size,
            dl_allowance=self.dl_allowance,
            n_discretize_bins=self.n_discretize_bins,
            max_rules=self.max_rules,
            max_rule_conds=self.max_rule_conds,
            random_state=self.random_state,
        )
        self.model_.fit(X, y, pos_class=1)
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        return np.asarray(self.model_.predict_proba(self._to_frame(X)), dtype=float)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def ruleset_(self):
        check_is_fitted(self)
        return self.model_.ruleset_


# rule preproccessor 
def make_rule_preprocessor() -> ColumnTransformer:
    
    num_cols = [c for c in NUMERIC_COLS if c not in DROP_COLS]
    cat_cols = [c for c in CATEGORICAL_COLS if c not in DROP_COLS]
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", SimpleImputer(strategy="most_frequent"), cat_cols)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_ripper_pipeline() -> Pipeline:
    # Unfitted pipeline: rule preprocessor + RIPPER, using our fixed step names
    return Pipeline(steps=[
        ("prep", make_rule_preprocessor()),
        ("clf", RipperClassifier(random_state=RANDOM_STATE))])
```

### The class: `RipperClassifier`

This is a **wrapper** (adapter): it holds a `wittgenstein.RIPPER` inside and exposes the interface scikit-learn expects. It follows scikit-learn's "Developing scikit-learn estimators" template.

| Code | What it does | Why it's written that way |
|---|---|---|
| `class RipperClassifier(ClassifierMixin, BaseEstimator)` | Inherits scikit-learn's classifier behavior and parameter machinery | **Mixin on the left.** Reversed, `is_classifier()` silently returns False and scikit-learn stops using stratified folds by default. `BaseEstimator` supplies `get_params`/`set_params`, so `clone()` works. |
| `__init__(self, k=2, prune_size=0.33, ...)` | Only stores each argument under the same name | `clone()` rebuilds the object from these. `set_params()` changes them **without** calling `__init__`, so any logic in `__init__` would be skipped. |
| Defaults `k=2, prune_size=0.33, dl_allowance=64, n_discretize_bins=10` | Mirror wittgenstein's own defaults | k = 2 optimization passes and a ⅓ prune split match Cohen's RIPPER2. `dl_allowance=64` is the MDL stopping rule. RIPPER isn't tuned, since Task 3 only requires tuning DT and k-NN. |
| `_to_frame(X)` | DataFrame → reset the index. Sparse → dense. Array → DataFrame with column names. | wittgenstein needs a DataFrame, and only bins columns that are numeric. The leading `_` marks it as a private helper. |
| `fit`: `len(X) != len(y)` check | Raises `ValueError` | scikit-learn's expected error for mismatched input |
| `self.classes_ = unique_labels(y)` | Sorted labels seen in training: `[0, 1]` | `predict_proba` columns must follow this order |
| Binary check | Rejects anything but 0/1 | wittgenstein is binary-only |
| `self.n_features_in_` | Number of input columns | A standard fitted attribute |
| `self.model_ = lw.RIPPER(...)` **inside `fit`** | Builds a **new** RIPPER on every fit | A refit never reuses old rules, and every parameter change reaches the real model. The trailing `_` marks it as learned state. |
| `self.model_.fit(X, y, pos_class=1)` | Learns rules for class 1 (>50K) | **Rarest-first**: RIPPER only writes rules for the positive class, and here that's the minority |
| `return self` | — | wittgenstein's `fit` returns `None`, which breaks `.fit(...).predict(...)` and pipelines |
| `predict_proba`: `check_is_fitted(self)` | Clear error if called before `fit` | Passes because `model_` and `classes_` exist |
| `np.asarray(..., dtype=float)` | Returns a float NumPy array of `[P(0), P(1)]` | Matches `classes_` order, which `evaluate()`'s `[:, 1]` relies on |
| `predict`: `classes_[np.argmax(...)]` | Picks the more likely class | The standard scikit-learn convention |
| `@property ruleset_` | Read-only view of the learned rules | Reads from `model_` live, so it never goes stale |

### `make_rule_preprocessor()` and `make_ripper_pipeline()`

| Code | What it does | Why |
|---|---|---|
| Only two `SimpleImputer`s (median, most_frequent) | Same Task 1 imputation, **no scaling, no one-hot** | RIPPER bins numbers itself (10 bins), so scaling changes nothing. It tests categories directly (`relationship=Husband`), so one-hot would only add 99 columns. On fold 1, imputation-only scored F1 0.556 vs. 0.523. |
| `verbose_feature_names_out=False` | Keeps names like `education-num` | Otherwise they'd be `num__education-num`, and the rules would be harder to read |
| `.set_output(transform="pandas")` | The transformer returns a DataFrame, not a NumPy array | Keeps real column names **and** numeric dtypes, so wittgenstein knows which columns to bin |
| `Pipeline(steps=[("prep", ...), ("clf", RipperClassifier(random_state=RANDOM_STATE))])` | Same `prep`/`clf` naming as the tree | Plugs straight into `evaluate()` |

**Where the code came from:** scikit-learn has no rule learner, so there's no scikit-learn skeleton for this. It's built from wittgenstein's README plus scikit-learn's estimator template.

**Cell 35**

```python
ripper_pipe = make_ripper_pipeline()

results["RIPPER"] = evaluate(ripper_pipe, X, y)
show("RIPPER", results["RIPPER"])
```

* **Exactly like every other model:** pipeline in, `evaluate()` out.
* **Runtime:** about 1–2 minutes per fold.
* **Result:** precision 0.718, which matches the tuned tree, but recall only 0.433, so F1 is 0.539. Rows no rule covers fall to ≤50K (Q21).

**Cell 36**

```python
ripper_full = make_ripper_pipeline().fit(X, y)
rules = ripper_full.named_steps["clf"].ruleset_

print("Number of rules:", len(rules))
print("Each rule predicts >50K; anything no rule covers is predicted <=50K\n")
for rule in rules[:10]:
    print(" ", rule)
```

* **Inspection only.** It fits on all rows *to read the rules*, and the result is never used for a score.
* **`named_steps["clf"]`** reaches inside the pipeline to the wrapper, and **`.ruleset_`** gets the rules.
* **Each rule is an AND of conditions** that predicts >50K. Anything no rule covers is ≤50K.
* **No saved output in the submitted notebook.** From the earlier run: 53 rules, nearly all starting with `marital-status=Married-civ-spouse`.

---

## 7. Task 4: SMOTE (cells 41–44)

> **Spec:** "Integrate SMOTE within imblearn.pipeline.Pipeline to ensure zero data leakage."  **Explainer:** §3.8, Q22–Q24

**Cell 41**

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
```

* **`ImbPipeline`** is imblearn's Pipeline, imported under a different name so it doesn't clash with scikit-learn's `Pipeline`.
* **Why it's needed:** scikit-learn's Pipeline only accepts steps that `transform` X without changing the row count. SMOTE *adds rows* through `fit_resample(X, y)`. imblearn's Pipeline calls `fit_resample` during `fit` and **skips** samplers during `predict`, so test data is never resampled.

**Cell 42**

```python
# SMOTE pipeline
smote_knn_pipeline = ImbPipeline([
    ("preprocessor", make_preprocessor()),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("model", KNeighborsClassifier())
])

tree_grid = {
    "model__max_depth": list(range(3, 21)),
    "model__min_samples_split": [2, 5, 10, 20],
    "model__criterion": ["gini", "entropy"]
}
smote_tree_pipeline = ImbPipeline([
    ("preprocessor", make_preprocessor()),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
])

smote_knn_search = GridSearchCV(
    smote_knn_pipeline,
    param_grid=knn_grid,
    scoring=SCORING,
    refit="f1",
    cv=CV,
    n_jobs=-1,
    return_train_score=True,
)

smote_tree_search = GridSearchCV(
    smote_tree_pipeline,
    param_grid=tree_grid,
    scoring=SCORING,
    refit="f1",
    cv=CV,
    n_jobs=-1,
    return_train_score=True,
)
```

| Piece | What it does | Why |
|---|---|---|
| Order: `preprocessor` → `smote` → `model` | Impute, scale and encode, then oversample, then fit | SMOTE needs numbers to compute distances, so it must come after encoding. It must come before the model so the model trains on the balanced data. |
| `SMOTE(random_state=RANDOM_STATE)` | Defaults: `k_neighbors=5`, `sampling_strategy="auto"` | `"auto"` means resample the minority class until it matches the majority, so each training fold goes 24/76 → 50/50. That's about 13,500 synthetic rows per fold. |
| `smote_knn_pipeline` + `knn_grid` | k-NN with SMOTE, reusing cell 28's grid | Same search space as without SMOTE, so the comparison is fair |
| `smote_tree_pipeline` + `tree_grid` | Tree with SMOTE | `min_samples_split` is `[2, 5, 10, 20]` here, not the earlier `[2, 10, 50, 100, 200]`. It's a narrower search, so the with/without-SMOTE tree comparison isn't perfectly like-for-like. |
| Step names `preprocessor` / `model` | — | Different from the `prep`/`clf` used elsewhere. Grid keys must match, hence `model__max_depth`. |
| `GridSearchCV(..., refit="f1", cv=CV, ...)` | Same settings as Task 3 | SMOTE runs inside **every** inner fold, because the whole pipeline is refit each time |

**Known limitation:** SMOTE after one-hot encoding interpolates 0/1 columns into fractions like `sex_Male = 0.37`. SMOTENC, applied before encoding, avoids that (explainer Q23).

**Cell 43**

```python
smote_tree_search.fit(X, y)
```

**Cell 44**

```python
smote_knn_search.fit(X, y)
```

* **Running the searches.** The tree with SMOTE won, with CV F1 0.676 vs. 0.685 without SMOTE.
* **SMOTE doesn't raise F1 at 0.5. It shifts the trade-off:** more recall, less precision (Q24).
* **These models aren't passed through `evaluate()`,** so they have no rows in the comparison table (explainer §6).

---

## 8. Task 4: Threshold sweep and cost (cells 8–9, 45–47, 38)

> **Spec:** "Perform probability threshold sweeping (t ∈ [0, 1]) and evaluate total cost under asymmetric penalties: False Negative C(+,-) = $100, False Positive C(-,+) = $10." Challenge A: the 0.091 threshold.  **Explainer:** §3.9, Q25–Q26

**Cell 8**

```python
from sklearn.metrics import confusion_matrix
```

**Cell 9**

```python
def cost_sweep(y, oof_proba, thresholds=np.linspace(0, 1, 101)) -> pd.DataFrame:
    """For each threshold t, predict + when P(+|x) > t and compute
    total cost = FN * COST_FN + FP * COST_FP.
    Returns a DataFrame with columns: threshold, FN, FP, total_cost.
    Theoretical optimum: COST_FP / (COST_FP + COST_FN) = 10/110 ≈ 0.091."""
    
    cost_results = []
    
    for threshold in thresholds:
        predictions = (oof_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
        
        total_cost = (100 * fn) + (10 * fp) 
        
        cost_results.append({
        "threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "total_cost": total_cost,
        "cost_per_person": total_cost / len(y),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0)
        })
        
    cost_dataframe = pd.DataFrame(cost_results)
    
    return cost_dataframe
```

| Line | What it does | Why / note |
|---|---|---|
| `thresholds=np.linspace(0, 1, 101)` | 0.00, 0.01, …, 1.00 | Covers the spec's t ∈ [0, 1] in steps of 0.01, so 0.09 is in the grid |
| `predictions = (oof_proba >= threshold).astype(int)` | Positive when P ≥ t | The docstring says `>`. At these thresholds the difference only matters for rows exactly on a threshold, which is negligible. |
| `confusion_matrix(y, predictions, labels=[0, 1]).ravel()` | Counts in the order **TN, FP, FN, TP** | `labels=[0, 1]` forces a 2×2 matrix even when one class is never predicted (t = 0 or 1). Without it, `.ravel()` would return the wrong number of values. |
| `total_cost = (100 * fn) + (10 * fp)` | The spec's cost formula | Hard-coded. `COST_FN * fn + COST_FP * fp` would be cleaner. Same result. |
| `cost_per_person` | Total ÷ 32,561 | About $4.19 at the optimum. Easier to interpret than the total. |
| precision, recall, F1 per threshold | — | Shows what the cost-optimal threshold does to the usual metrics |
| `pd.DataFrame(cost_results)` | One row per threshold | Easy to find the minimum and to plot |

**Cell 45**

```python
from sklearn.model_selection import cross_val_predict
```

**Cell 46**

```python
#cost sweep and cost-vs-threshold plot
best_search = max([smote_knn_search, smote_tree_search],
                  key=lambda search: search.best_score_)

best_model = best_search.best_estimator_

print("Selected model:", type(best_model.named_steps["model"]).__name__)
print("Best CV F1:", best_search.best_score_)

# Obtain out-of-fold probabilities for threshold selection.
# Each example's probability comes from a model that did not train on it.
oof_probabilities = cross_val_predict(
    best_model,
    X,
    y,
    cv=CV,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

results = cost_sweep(y, oof_probabilities)
```

| Line | What it does | Why |
|---|---|---|
| `max([...], key=lambda search: search.best_score_)` | Picks whichever SMOTE search had the higher CV F1 | An automatic, reproducible choice. Result: the decision tree. |
| `best_search.best_estimator_` | The winning pipeline | — |
| `type(best_model.named_steps["model"]).__name__` | Prints the model's class name | Confirms which model was picked |
| `cross_val_predict(..., cv=CV, method="predict_proba")[:, 1]` | Out-of-fold P(>50K) for every row | Each probability comes from a model that didn't train on that row. SMOTE is refit inside each fold. This is the honest input for threshold selection. |
| `results = cost_sweep(y, oof_probabilities)` | The cost table | **Overwrites** the `results` dict of model metrics. Everything that needed it already ran, but a separate name like `cost_df` is cleaner (explainer §6). |

**Cell 38**

```python
#plot_metrics(results)

def plot_metrics(results):
    best_row = results.loc[results["total_cost"].idxmin()]
    print("Optimal theshold on OOF predictsions:")
    print(best_row.to_string())
    
    plt.figure(figsize=(9, 5))
    plt.plot(
        results["threshold"],
        results["total_cost"]
    )
    plt.axvline(
        best_row["threshold"],
        linestyle="--",
        label=f'Best threshold = {best_row["threshold"]:.2f}'
    )
    plt.xlabel("Classification threshold")
    plt.ylabel("Total misclassification cost ($)")
    plt.title("Threshold Selection by Total Cost")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

| Line | What it does |
|---|---|
| `results.loc[results["total_cost"].idxmin()]` | `idxmin()` gives the row label of the lowest cost, and `.loc` fetches that row |
| `best_row.to_string()` | Prints every column of the optimal row |
| `plt.plot(threshold, total_cost)` | Draws the cost curve |
| `plt.axvline(best_threshold, "--")` | Marks the minimum |
| `plt.tight_layout()`, `plt.grid(alpha=0.3)` | Spacing and a light grid |

**Note:** despite its name and its position under "task 2 benchmark bar plot," this function draws the **Task 4 cost curve**. The Task 2 grouped bar chart with error bars isn't in the notebook (explainer §6).

**Cell 47**

```python
plot_metrics(results)
```

| Result | Value |
|---|---|
| Best threshold | **0.12** (theory: 10/110 = 0.091) |
| FN / FP | 424 / 9,410 |
| Total cost | **$136,500** (vs. $784,100 contacting nobody, $247,200 contacting everyone) |
| Recall / precision | 0.946 / 0.441 |

**Why 0.12 instead of 0.091:** the theory assumes calibrated probabilities. SMOTE and tree leaf frequencies aren't calibrated (Q25).

---

## 9. Summary table and save (cells 39–40)

> **Spec (report):** "model comparison tables."  **Explainer:** §2.1

**Cell 39**

```python
#one row per model, mean ± std from the same 5 folds
summary = pd.DataFrame({
    name: {m: f"{mean:.3f} ± {std:.3f}" for m, (mean, std) in res["metrics"].items()}
    for name, res in results.items()
}).T
summary
```

* **A nested dict comprehension:** for each model, format each metric as `"mean ± std"`.
* **`pd.DataFrame({...}).T`** transposes the table so models are rows and metrics are columns.
* **It's built from the `results` dict,** so every row was produced by the same `evaluate()` call.
* **No saved output in the submitted notebook.** The same numbers are printed by `show()` in each model's cell.

**Cell 40**

```python
#we save the results (including out-of-fold probabilities) so Task 4 and the final notebook can reuse them
os.makedirs("artifacts", exist_ok=True)
joblib.dump({"results": results, "dt_best_params": dt_search.best_params_},
            "artifacts/a_results.joblib")
print("saved artifacts/a_results.joblib")
```

* **`joblib.dump`** saves the results, including every model's out-of-fold probabilities, to disk.
* **Why it exists:** it was written for the two-notebook workflow, so the partner's notebook could reuse the results. In the merged final notebook it isn't needed.
* **`os.makedirs(..., exist_ok=True)`** creates the folder if it's missing and doesn't error if it already exists.

---

## 10. Part II: AI Prompt Audit, the debugged pipeline (cell 48)

> **Spec:** "provide the unedited AI code, their final debugged pipeline, and audit whether the AI placed SMOTE inside the imblearn.pipeline.Pipeline or mistakenly applied SMOTE globally prior to cross-validation."  **Explainer:** §3.10, Q27

Only the changed part is shown. The rest of the cell is the AI's original code, unchanged:

**Cell 48**

```python
# final debugged pipeline: the AI's code with only the data loading fixed
...
# -------------------------------------------------
# Load dataset
# -------------------------------------------------
# FIX for bug 1 and bug 2: load_data() names the columns, turns ? into NaN, and encodes >50K as 1

# COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num",
#            "marital-status", "occupation", "relationship", "race", "sex",
#            "capital-gain", "capital-loss", "hours-per-week", "native-country",
#            "income"]

# def load_data(path="data/adult.data") -> tuple[pd.DataFrame, pd.Series]:

#     df = pd.read_csv(path, names=COLUMNS, header=None,
#                      skipinitialspace=True, na_values="?")

#     # we encode the target as 1 for >50K and 0 for <=50K
#     y = (df["income"].str.rstrip(".") == ">50K").astype(int)

#     # we split off the features by dropping the target and any columns we chose to drop
#     X = df.drop(columns=["income"])

#     return X, y

X, y = load_data()

# ... everything below is the AI's original code, unchanged ...
```

| What changed | Why |
|---|---|
| The AI's `pd.read_csv("data.csv")` and `target = "target"` lines were replaced by `X, y = load_data()` | **Bug 1:** the AI's loading can't read Adult. There's no header row, `?` isn't treated as missing, and `data.csv` doesn't exist. |
| (the same change) | **Bug 2:** the AI's `scoring="f1"` crashes on the text labels `>50K`/`<=50K`. `load_data()` encodes >50K as 1, which F1 needs. |
| The commented-out block | Shows what `load_data()` does, so the cell reads on its own |
| Everything else left as the AI wrote it | Keeps the before/after comparison clean: two fixes, nothing else |

**The audit verdict:** the AI placed SMOTE **correctly**. Its pipeline is:

```python
ImbPipeline([("preprocessor", preprocessor), ("smote", SMOTE(...)), ("classifier", DecisionTreeClassifier(...))])
```

`GridSearchCV` refits this whole pipeline on each training fold, so SMOTE never sees validation data. There's no global SMOTE before CV, and no leakage.

**Output:** CV F1 0.668. On the AI's 80/20 test split, F1 is 0.67 and recall 0.85. The pandas warning in the output appears because `select_dtypes(include=["object"])` relies on behavior that's deprecated in pandas 3. It still worked.

---

## 11. Assignment compliance matrix

| Spec requirement | Where | Status |
|---|---|---|
| Preprocessing inside Pipeline and ColumnTransformer (no leakage) | Cells 4, 6, and every pipeline | ✅ |
| Task 1: median + StandardScaler / most_frequent + OneHotEncoder(handle_unknown='ignore') | Cell 4 | ✅ |
| `?` handled as missing | Cell 4 (`na_values="?"`), checked in cell 14 | ✅ |
| Task 2: 5-fold StratifiedKFold | Cell 2 (`CV`), cell 6 | ✅ |
| Task 2: Accuracy, Precision, Recall, F1, ROC-AUC | Cell 6 | ✅ |
| Task 2: benchmark DT, k-NN, rule-based | Cells 19, 25, 33, 35 | ✅ |
| Task 2: comparative bar plots with std error bars | — | ❌ Not in the notebook. `mean ± std` is printed by `show()`. |
| Task 3: DT `max_depth` 3–20, `min_samples_split`, criterion via GridSearchCV | Cells 21–22 | ✅ |
| Task 3: k-NN k ∈ {1, 3, 7, 15, 31}, weights, metric via GridSearchCV | Cell 28 | ✅ |
| Rule-based: sequential covering / FOIL gain (RIPPER) | Cells 11, 35 | ✅ (rules cell 36 has no saved output) |
| Task 4: SMOTE inside `imblearn.pipeline.Pipeline` | Cells 41–44 | ✅ |
| Task 4: threshold sweep t ∈ [0, 1], cost with FN = $100, FP = $10 | Cells 9, 46–47 | ✅ (best t = 0.12) |
| Report: model comparison table | Cell 39 | ⚠️ Code present, no saved output |
| Report: ROC/PR plots with captions | — | ❌ Not in the notebook. `oof_proba` is stored for each model. |
| Part II: unedited AI code | Report Part 2 / `Ai_integration.ipynb` | ✅ |
| Part II: final debugged pipeline | Cell 48 | ✅ |
| Part II: SMOTE placement audit | Cell 48 comment; explainer Q27 | ⚠️ Say it out loud in the demo, since it isn't written in the report |

The honest answers for every ❌ and ⚠️ are in **explainer §6**.

---

## 12. Quick function reference

| Function / class | Library | One-line meaning |
|---|---|---|
| `pd.read_csv` | pandas | Read a CSV text file into a DataFrame |
| `SimpleImputer(strategy=...)` | sklearn.impute | Fill missing values with a learned statistic |
| `StandardScaler` | sklearn.preprocessing | Rescale to mean 0, std 1 (standardization) |
| `OneHotEncoder` | sklearn.preprocessing | One 0/1 column per category |
| `ColumnTransformer` | sklearn.compose | Different transformers for different columns, run side by side |
| `Pipeline` | sklearn.pipeline | Chain steps; fit them together, on training data only |
| `StratifiedKFold` | sklearn.model_selection | K splits that keep the class ratio |
| `clone` | sklearn.base | Unfitted copy of an estimator with the same settings |
| `GridSearchCV` | sklearn.model_selection | Try every parameter combination with CV and keep the best |
| `cross_val_predict(method="predict_proba")` | sklearn.model_selection | Out-of-fold probabilities for every row |
| `confusion_matrix(...).ravel()` | sklearn.metrics | TN, FP, FN, TP |
| `precision_score`, `recall_score`, `f1_score`, `accuracy_score`, `roc_auc_score` | sklearn.metrics | The five Task 2 metrics |
| `DummyClassifier` | sklearn.dummy | Baseline that ignores the features |
| `DecisionTreeClassifier` | sklearn.tree | Splits on Gini/entropy |
| `KNeighborsClassifier` | sklearn.neighbors | Votes of the k nearest training points |
| `ClassifierMixin`, `BaseEstimator` | sklearn.base | Base classes that make a custom model scikit-learn-compatible |
| `unique_labels`, `check_is_fitted` | sklearn.utils | Sorted class labels; error if not yet fitted |
| `lw.RIPPER` | wittgenstein | RIPPER rule learner |
| `SMOTE` | imblearn.over_sampling | Synthetic minority rows by interpolation |
| `ImbPipeline` | imblearn.pipeline | Pipeline that can hold samplers and skips them at predict time |
| `np.linspace(0, 1, 101)` | NumPy | 101 evenly spaced thresholds |
| `plt.subplots`, `ax.plot`, `ax.fill_between`, `ax.axvline`, `ax.annotate` | matplotlib | Figure/axes, lines, shaded band, vertical marker, text label |

---

## 13. Sources

* scikit-learn: [Column Transformer with Mixed Types](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html); [Multi-metric evaluation with GridSearchCV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html); [GridSearchCV API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html); [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html); [Developing scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html); [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html); [cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html)
* imbalanced-learn: [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html); [Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html)
* wittgenstein: [GitHub README](https://github.com/imoscovitz/wittgenstein) (v0.3.5)
* matplotlib: [Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html)
* pandas: [read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
* Project spec: `project2_spec_v3.pdf`; companion: `Project2_Code_Explainer.md`
