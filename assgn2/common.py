import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- Shared constants ----------
RANDOM_STATE = 21
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
COST_FN, COST_FP = 100, 10          # C(+,-), C(-,+)

DATA_PATH = "data/adult.data"

# the raw file has no header so we name the columns ourselves (order comes from adult.names)
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


# ---------- Task 1: loading and preprocessing ----------
def load_data(path=DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Read the raw file and apply STATELESS cleaning only.
    - assign COLUMNS (file has no header)
    - strip leading whitespace from values
    - replace '?' with NaN
    - encode target: '>50K' -> 1, '<=50K' -> 0
    - drop columns listed in DROP_COLS
    Returns X (features) and y (0/1 target)."""

    # we read the file with no header, strip the space after each comma,
    # and turn every ? into a real missing value so the imputer can see it
    df = pd.read_csv(path, names=COLUMNS, header=None,
                     skipinitialspace=True, na_values="?")

    # we encode the target as 1 for >50K and 0 for <=50K
    # rstrip(".") lets this also work on adult.test, where labels end with a period
    y = (df["income"].str.rstrip(".") == ">50K").astype(int)

    # we split off the features by dropping the target and any columns we chose to drop
    X = df.drop(columns=["income"] + DROP_COLS)

    return X, y


def make_preprocessor() -> ColumnTransformer:
    """Return an UNFITTED Task 1 ColumnTransformer.
    Never call .fit() here. It is fitted inside each CV fold."""

    # numerical part of the pipeline: fill missing with the median, then z-score scale
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]
    )

    # categorical part of the pipeline: fill missing with the most common value, then one-hot encode
    # handle_unknown="ignore" keeps a category seen only in a test fold from crashing the model
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    # the column transformer runs both pipelines side by side on their own columns
    # and glues the outputs together into one feature matrix
    return ColumnTransformer(
        transformers=[
            ("num", numeric, [c for c in NUMERIC_COLS if c not in DROP_COLS]),
            ("cat", categorical, [c for c in CATEGORICAL_COLS if c not in DROP_COLS])]
    )


# ---------- Task 2: evaluation harness (step 2) ----------
def evaluate(pipeline, X, y) -> dict:
    """5-fold stratified CV on the FULL pipeline (preprocessor + model).
    Returns:
      {'metrics': {metric_name: (mean, std)},
       'oof_proba': np.ndarray of P(+|x) for every row, out-of-fold}"""

    # we keep one score per fold for each metric so we can report mean and std
    fold_scores = {m: [] for m in SCORING}

    # one out-of-fold probability per row, filled in as each fold is scored
    oof_proba = np.zeros(len(y))

    # this loop is where cross-validation happens: 5 folds, each one a train/test split
    for train_idx, test_idx in CV.split(X, y):

        # we clone so every fold starts from a fresh, unfitted copy of the pipeline
        model = clone(pipeline)

        # fitting the whole pipeline fits the imputers, scaler and encoder
        # (and SMOTE, if present) on this fold's training rows only, so nothing leaks
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

        # we save these probabilities for the ROC curves and the Task 4 cost sweep
        oof_proba[test_idx] = proba

    metrics = {m: (np.mean(v), np.std(v)) for m, v in fold_scores.items()}
    return {"metrics": metrics, "oof_proba": oof_proba}


# ---------- Task 4: cost analysis (step 7) ----------
def cost_sweep(y, oof_proba, thresholds=np.linspace(0, 1, 101)) -> pd.DataFrame:
    """For each threshold t, predict + when P(+|x) > t and compute
    total cost = FN * COST_FN + FP * COST_FP.
    Returns a DataFrame with columns: threshold, FN, FP, total_cost.
    Theoretical optimum: COST_FP / (COST_FP + COST_FN) = 10/110 ≈ 0.091."""
    raise NotImplementedError("step 7")
