
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

from common import NUMERIC_COLS, CATEGORICAL_COLS, DROP_COLS, RANDOM_STATE


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
