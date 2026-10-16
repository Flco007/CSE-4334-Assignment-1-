"""
verify_env.py — CSE 4334/5334 Project 1
Checks versions and runs a minimal pipeline to confirm the scikit-learn API
surface this project depends on. Exits non-zero if anything is wrong.
"""
import sys
import platform

FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


print("=" * 62)
print("ENVIRONMENT")
print("=" * 62)
print(f"  python       {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")

try:
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import matplotlib
except ImportError as exc:
    print(f"\n  MISSING PACKAGE: {exc.name}\n  Re-run the conda install step in SETUP.md.")
    sys.exit(1)

print(f"  numpy        {np.__version__}")
print(f"  pandas       {pd.__version__}")
print(f"  scipy        {scipy.__version__}")
print(f"  scikit-learn {sklearn.__version__}")
print(f"  matplotlib   {matplotlib.__version__}")

for name in ("openpyxl", "pyarrow", "joblib"):
    try:
        mod = __import__(name)
        print(f"  {name:<12} {getattr(mod, '__version__', 'ok')}")
    except ImportError:
        print(f"  {name:<12} MISSING  <-- required")
        FAILURES.append(name)

print("\n" + "=" * 62)
print("API CHECKS")
print("=" * 62)

check("Python >= 3.11 (required by scikit-learn 1.9)",
      sys.version_info[:2] >= (3, 11),
      f"got {sys.version_info.major}.{sys.version_info.minor}")

sk = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
check("scikit-learn >= 1.4", sk >= (1, 4), f"got {sklearn.__version__}")

from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    OneHotEncoder(sparse_output=True)
    check("OneHotEncoder accepts sparse_output=", True)
except TypeError:
    check("OneHotEncoder accepts sparse_output=", False, "version too old")

try:
    OneHotEncoder(min_frequency=10)
    check("OneHotEncoder accepts min_frequency=", True)
except TypeError:
    check("OneHotEncoder accepts min_frequency=", False)

try:
    KBinsDiscretizer(strategy="quantile", quantile_method="averaged_inverted_cdf")
    check("KBinsDiscretizer accepts quantile_method=", True)
except (TypeError, ValueError):
    check("KBinsDiscretizer accepts quantile_method=", False,
          "sklearn < 1.7 -- bin edges will differ from teammates")

print("\n" + "=" * 62)
print("END-TO-END PIPELINE SMOKE TEST")
print("=" * 62)

rng = np.random.default_rng(42)
n = 500
frame = pd.DataFrame({
    "Quantity": rng.integers(-1000, 1000, n).astype(float),
    "UnitPrice": np.abs(rng.normal(3, 50, n)),
    "Country": rng.choice(["United Kingdom", "France", "Germany"], n),
})
frame.loc[rng.random(n) < 0.1, "Quantity"] = np.nan
frame.loc[rng.random(n) < 0.1, "Country"] = None

numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
binned = Pipeline([("impute", SimpleImputer(strategy="median")),
                   ("bin", KBinsDiscretizer(n_bins=4, encode="ordinal",
                                            strategy="quantile",
                                            quantile_method="averaged_inverted_cdf",
                                            subsample=None))])
categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("encode", OneHotEncoder(handle_unknown="ignore",
                                                 sparse_output=True))])

pre = ColumnTransformer([("num", numeric, ["Quantity"]),
                         ("bin", binned, ["UnitPrice"]),
                         ("cat", categorical, ["Country"])], remainder="drop")

train, test = frame.iloc[:400], frame.iloc[400:]
pre.fit(train)
out_train, out_test = pre.transform(train), pre.transform(test)

check("pipeline fits and transforms", out_train.shape[0] == 400)
check("train/test widths match", out_train.shape[1] == out_test.shape[1],
      f"{out_train.shape[1]} vs {out_test.shape[1]}")

unseen = test.copy()
unseen["Country"] = "Atlantis"
pre.transform(unseen)
check("unseen categories do not raise", True)

check("get_feature_names_out works",
      len(pre.get_feature_names_out()) == out_train.shape[1])

print("\n" + "=" * 62)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {', '.join(FAILURES)}")
    print("See the Troubleshooting section of SETUP.md.")
    sys.exit(1)
print("ALL CHECKS PASSED - environment is ready.")
print("=" * 62)


