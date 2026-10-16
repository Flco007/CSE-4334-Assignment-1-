# CSE 4334/5334 Project 2: Team Contract

**Project:** Classification Models & Imbalanced Evaluation (UCI Adult Income)
**Team:** Person A: ____________  Person B: ____________  Project Manager: ____________
**Due date:** ____________  **Demo date:** ____________

This contract defines who owns what, how our code fits together, and the rules we both build to. Neither person changes the shared interface (`common.py`) without agreement from the other.

---

## 1. Ownership

### Person A: Trees, Rules, and the AI Audit

| Work package | Rubric area |
|---|---|
| Task 1: `load_data()` and `make_preprocessor()` in `common.py` | Implementation |
| Task 3 (DT): `GridSearchCV` over `max_depth` (3–20), `min_samples_split`, `criterion` (`gini` / `entropy`) | Implementation |
| Rule-based model: RIPPER (sequential covering, FOIL gain), e.g. via `wittgenstein`, wrapped as a scikit-learn-compatible estimator | Implementation |
| Challenge C: Gini/entropy concavity vs. FOIL gain; RIPPER's rarest-first class ordering | Report |
| AI appendix: run the required prompt, log it unedited, audit SMOTE placement, document fixes | AI Appendix (10%) |
| Demo: explain preprocessing, DT, and RIPPER | Demo |

### Person B: k-NN, Evaluation, and Cost

| Work package | Rubric area |
|---|---|
| Task 2: `evaluate()` harness in `common.py`. 5-fold `StratifiedKFold`; Accuracy, Precision, Recall, F1, ROC-AUC; bar plots with std error bars; out-of-fold probabilities | Implementation |
| Task 3 (k-NN): `GridSearchCV` over `n_neighbors` ∈ {1, 3, 7, 15, 31}, `weights`, `metric` | Implementation |
| Task 4: SMOTE inside `imblearn.pipeline.Pipeline`; threshold sweep t ∈ [0, 1]; total cost at C(+,−) = $100, C(−,+) = $10. Includes `cost_sweep()` in `common.py` | Implementation |
| ROC and PR curve plots; model comparison table | Report |
| Challenge A: accuracy paradox; derivation of the 0.091 threshold | Report |
| Challenge B: scale distortion in Euclidean distance; distance concentration from one-hot sparsity | Report |
| Demo: explain the harness, k-NN, SMOTE, and cost analysis | Demo |

### Shared

- Final report assembly and formatting (owned by the Project Manager)
- Final notebook assembly
- Demo rehearsal
- **Cross-review for data leakage:** A reviews B's SMOTE pipeline. B reviews A's DT and RIPPER pipelines.

---

## 2. Order of Work

1. **Day 1 (together):** Kickoff meeting (Section 3). Agree on this contract.
2. **Parallel:** A builds `load_data()` and `make_preprocessor()`. B builds `evaluate()` against a stub preprocessor (Section 6).
3. **Parallel:** A does DT tuning and RIPPER. B does k-NN tuning.
4. **Task 4 (B):** Runs on the tuned models. `cost_sweep()` works on any model's out-of-fold probabilities, so it does not wait on RIPPER.
5. **Together:** Results go into the report, then cross-review, then demo rehearsal.

**Critical path:** preprocessing → harness → tuned models → cost analysis → results tables. A delay on any of these delays the whole project.

**Highest-risk item:** RIPPER, owned by A. Start it early. If `wittgenstein` will not run inside a `Pipeline`, wrap it in a small custom estimator class (Section 5, rule 2).

---

## 3. Day 1 Kickoff Meeting (30 min)

| Time | Item | Output |
|---|---|---|
| 5 min | Repo setup | Repo created with the layout in Section 4 |
| 5 min | Data decisions | Section 7 filled in |
| 10 min | Function signatures | `common.py` stubs committed (Section 5) |
| 5 min | Conventions | Section 5 rules agreed |
| 5 min | Definition of done and check-ins | Section 9 filled in |

**Rest of Day 1, in parallel:**

| Person A | Person B |
|---|---|
| Implement `load_data()` and `make_preprocessor()` for real | Implement `evaluate()` using the stub preprocessor |
| Push to `common.py` once they pass a smoke test | Test it on a default `DecisionTreeClassifier` to confirm the output dict and plots work |
| Start on the RIPPER wrapper | Swap in A's real preprocessor once it lands. No code changes needed. |

**End-of-Day-1 checklist:**
- [ ] Repo exists with the agreed layout
- [ ] `common.py` stubs agreed and committed
- [ ] Data decisions recorded in Section 7
- [ ] Real `load_data()` and `make_preprocessor()` working (A)
- [ ] `evaluate()` working on a test model (B)

---

## 4. Repository Layout

```
common.py               ← shared contract (edit only by agreement)
a_trees_rules.ipynb     ← Person A's workspace
b_knn_cost.ipynb        ← Person B's workspace
final_submission.ipynb  ← assembled at the end; imports common.py
README.md               ← decisions log (copy of Section 7)
```

Shared code lives in `.py` files because notebooks merge badly in Git. Each person works only in their own notebook.

### How the layers fit

| Layer | Contents | Owner |
|---|---|---|
| `common.py` | load, preprocess, evaluate, cost sweep | A: first two. B: last two. |
| `a_trees_rules.ipynb` | DT pipeline + grid search, RIPPER wrapper + pipeline, AI-audit pipeline | A |
| `b_knn_cost.ipynb` | k-NN pipeline + grid search, SMOTE pipeline, threshold sweep plots, ROC/PR curves | B |
| `final_submission.ipynb` | Runs everything through `common.py`, produces report figures | Both |

`common.py` is the shared foundation, roughly 15–20% of the code. Its job is to guarantee that every model was cleaned, split, and scored the same way.

---

## 5. The Interface Contract

### `common.py` signatures

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- Shared constants ----------
RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
COST_FN, COST_FP = 100, 10          # C(+,-), C(-,+)

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex",
           "capital-gain", "capital-loss", "hours-per-week", "native-country",
           "income"]
NUMERIC_COLS = ["age", "fnlwgt", "education-num", "capital-gain",
                "capital-loss", "hours-per-week"]        # adjust per Section 7
CATEGORICAL_COLS = ["workclass", "education", "marital-status", "occupation",
                    "relationship", "race", "sex", "native-country"]  # adjust per Section 7


# ---------- Owner: Person A ----------
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Read the raw file and apply STATELESS cleaning only.
    - assign COLUMNS (file has no header)
    - strip leading/trailing whitespace from string values
    - replace '?' with NaN
    - encode target: '>50K' -> 1, '<=50K' -> 0
    - drop columns agreed in Section 7
    Returns X (features) and y (0/1 target)."""


def make_preprocessor() -> ColumnTransformer:
    """Return an UNFITTED Task 1 ColumnTransformer.
    Never call .fit() here. It is fitted inside each CV fold."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_COLS),
                              ("cat", categorical, CATEGORICAL_COLS)])


# ---------- Owner: Person B ----------
def evaluate(pipeline, X, y) -> dict:
    """5-fold stratified CV on the FULL pipeline (preprocessor + model).
    Returns:
      {'metrics': {metric_name: (mean, std)},
       'oof_proba': np.ndarray of P(+|x) for every row, out-of-fold}"""
    scores = cross_validate(pipeline, X, y, cv=CV, scoring=SCORING)
    oof_proba = cross_val_predict(pipeline, X, y, cv=CV,
                                  method="predict_proba")[:, 1]
    metrics = {m: (scores[f"test_{m}"].mean(), scores[f"test_{m}"].std())
               for m in SCORING}
    return {"metrics": metrics, "oof_proba": oof_proba}


def cost_sweep(y, oof_proba, thresholds=np.linspace(0, 1, 101)) -> pd.DataFrame:
    """For each threshold t, predict + when P(+|x) > t and compute
    total cost = FN * COST_FN + FP * COST_FP.
    Returns a DataFrame with columns: threshold, FN, FP, total_cost.
    Theoretical optimum: COST_FP / (COST_FP + COST_FN) = 10/110 ≈ 0.091."""
```

### Conventions (both people follow these)

1. **Fixed pipeline step names:** `"prep"`, `"smote"`, `"clf"`. Grid parameters are always written as `clf__<param>` (e.g. `clf__max_depth`), and SMOTE can be inserted into any pipeline without renaming.
2. **Every model must support `fit`, `predict`, and `predict_proba`** (scikit-learn estimator API). This is what lets `cost_sweep()` work on any model. The RIPPER wrapper must also expose `classes_` and `get_params` / `set_params` (inherit from `BaseEstimator` and `ClassifierMixin`).
3. **`evaluate()` is the only source of reported numbers.** No ad hoc `cross_val_score` calls for report results. Every table row is produced the same way.
4. **`GridSearchCV` always uses `cv=CV`**, so tuning and evaluation split the data identically.
5. **`GridSearchCV` refit metric is not accuracy.** Use the metric agreed in Section 7. Tuning on accuracy reproduces the accuracy paradox.
6. **Shared plot style:** one fixed color per model across all figures.
7. **Positive class is 1 (`>50K`)** for every metric, threshold, and cost calculation.

---

## 6. Key Concepts (reference)

### Stateless vs. stateful preprocessing

| | Stateless | Stateful |
|---|---|---|
| Learns from data? | No | Yes (medians, means, stds, most-frequent values) |
| Examples | Column names, whitespace strip, `?` → NaN, target encoding | `SimpleImputer`, `StandardScaler`, `OneHotEncoder` |
| Where it goes | `load_data()`, applied once to all rows | `make_preprocessor()`, fitted inside each CV fold |
| Leakage risk | None | High if fitted on the full dataset before CV |

`make_preprocessor()` returns an unfitted recipe. It does not clean anything when called.

### Where cross-validation happens

CV happens inside `evaluate()` and `GridSearchCV`, never anywhere else. On each of the 5 folds, the **whole pipeline** (preprocessor and model) is fitted on 4/5 of the data and scored on the remaining 1/5. Because the preprocessor is inside the pipeline, it is refit every fold. This is what "zero data leakage" means for this project.

### Building a model pipeline

Models come straight from scikit-learn. The contract passes **pipelines**, not models. Each person assembles their own in their notebook:

```python
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from common import load_data, make_preprocessor, evaluate

X, y = load_data()
dt = Pipeline([("prep", make_preprocessor()),
               ("clf", DecisionTreeClassifier(random_state=42))])
results = evaluate(dt, X, y)
```

With SMOTE (Task 4), use the imblearn `Pipeline` so SMOTE runs only on each fold's training data:

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

dt_smote = ImbPipeline([("prep", make_preprocessor()),
                        ("smote", SMOTE(random_state=42)),
                        ("clf", DecisionTreeClassifier(random_state=42))])
```

Order matters: `prep` → `smote` → `clf`. SMOTE must come after encoding and must never be applied to the full dataset before CV.

### Stub preprocessor

A stub is a deliberately simple stand-in with the **same signature** as the real function, so B can test `evaluate()` before A's version exists:

```python
def make_preprocessor():   # TEMPORARY STUB, replaced by A's real version
    return ColumnTransformer([("num", StandardScaler(), NUMERIC_COLS)])
```

It drops the categorical columns, so results are worse, but it runs. When A's real version lands, it replaces the stub and B changes nothing.

### Test model vs. real model

- B uses a default `DecisionTreeClassifier` only to prove the harness works. Its score does not matter.
- A owns the real, tuned decision tree and its write-up.
- B owns the real, tuned k-NN.

---

## 7. Data Decisions (fill in at kickoff)

| Decision | Choice | Reason |
|---|---|---|
| Data file(s) used | `adult.data` (32,561 rows) / also `adult.test`? | |
| Keep or drop `fnlwgt` (census sampling weight, not a person attribute) | | |
| Keep `education`, `education-num`, or both (they encode the same thing) | | |
| `GridSearchCV` refit metric (F1 or ROC-AUC) | | |
| RIPPER library (`wittgenstein` or other) | | |
| SMOTE variant (`SMOTE` after encoding, or `SMOTENC`) | | |
| Other | | |

Note: `adult.test` labels end with a period (`>50K.`), so strip it if that file is used.

---

## 8. Known Risks

| Risk | Mitigation | Owner |
|---|---|---|
| Demo rubric requires "active participation of all 3 team members." | If the team is down to two, email Dr. Diaz now, not at demo time. | PM |
| RIPPER does not fit cleanly into a `Pipeline`. | Wrap it in a custom estimator class. Start early. | A |
| SMOTE on one-hot columns creates fractional dummy values. | Use `SMOTENC`, or keep the `prep` → `smote` → `clf` order and note it in the report. | B |
| Tuning on accuracy reproduces the accuracy paradox. | Refit on the metric chosen in Section 7. | A, B |
| The required AI prompt lists components in an order that invites SMOTE leakage. | Audit placement carefully for the appendix. | A |
| k-NN grid search is slow (20 combos × 5 folds on ~100 one-hot dimensions). | Run with `n_jobs=-1`. Start it early. | B |

---

## 9. Definition of Done and Check-ins

**A model is done when:**
- [ ] It runs through `evaluate()` without error
- [ ] Its grid search has finished and best parameters are recorded
- [ ] Its results are in the comparison table
- [ ] The other person has reviewed it for leakage

**A report section is done when:**
- [ ] It answers every bullet in its challenge
- [ ] Every figure has a caption
- [ ] The PM has merged it into the report

**Check-ins:**

| Check-in | Date | Goal |
|---|---|---|
| Kickoff | | This contract agreed |
| Mid-point integration | | Both notebooks run end to end against the real `common.py` |
| Pre-demo rehearsal | | Each person explains their pipeline live and answers questions |

---

**Agreed by:**

Person A: ____________  Date: ______
Person B: ____________  Date: ______
