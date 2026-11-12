"""Shared functions for CSE 4334/5334 Project 3 (Person A: data, mining, benchmark).

Person B imports these to get encoded datasets; at the Nov 2 merge this whole
file is pasted into the Setup section of the final notebook so it runs alone.

Code reuse (noted per our contract):
  - Assignment 1 (assgn1/final.ipynb cells 2 and 27): reading Online_Retail.xlsx
    with pd.read_excel, the missing-value check, the "InvoiceNo starts with C"
    cancellation test, and the Description whitespace cleanup. The A1 regex
    r'[\\s+]' matched ONE whitespace or a literal '+', so it is fixed here to
    r'\\s+'. A1's lowercasing and punctuation stripping are dropped, because
    item labels should stay readable and match the original product names.
  - Assignment 2 (assgn2/final_submission.ipynb cell 2): constants at the top
    of the module (RANDOM_STATE = 21, paths, grids) and docstrings that say
    what each function returns.
"""

import csv
import signal
import statistics
import time
import tracemalloc

import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# reused layout from Assignment 2: every setting lives here, nothing hard-coded below
RANDOM_STATE = 21
RETAIL_PATH = "data/Online_Retail.xlsx"
GROCERIES_PATH = "data/groceries.csv"

MINSUP_GRID = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]   # contract Section 3
# "apriori" is mlxtend's default (fast, vectorized, memory hungry);
# "apriori_lowmem" is apriori(low_memory=True), which the mlxtend docstring calls
# about 3-6x slower but builds candidates one at a time instead of all at once
ALGOS = ["apriori", "apriori_lowmem", "fpgrowth"]
TIMEOUT_S = 300          # contract: a run past 5 minutes is recorded as DNF
REPEAT_UNDER_S = 60      # contract: runs under 60 s repeat 3 times, median reported
REPEATS = 3
MEMORY_CAP_GB = 4        # default Apriori is skipped when its level-2 arrays would pass this

# a real product code is 5 digits plus an optional letter suffix (UCI variable notes);
# everything else (POST, DOT, M, C2, D, S, BANK CHARGES, AMAZONFEE, gift vouchers, ...)
# is postage, fees, discounts, or manual adjustments, not a product
PRODUCT_CODE = r"^\d{5}[A-Za-z]*$"
NORMAL_INVOICE = r"^\d{6}$"     # drops cancellations (C...) and bad-debt adjustments (A...)


# ---------------------------------------------------------------------------
# Task 3.1: loading and cleaning
# ---------------------------------------------------------------------------

def load_online_retail(path=RETAIL_PATH, drop_no_customer=True):
    """Clean Online Retail into baskets, one per invoice.

    Returns (transactions, labels, log):
      transactions : list[list[str]], the StockCodes in each invoice
      labels       : dict StockCode -> readable description (unique per code)
      log          : DataFrame with rows and invoices left after each step
    """
    # reused from Assignment 1: read the xlsx; codes as strings so 85123A etc. stay intact
    df = pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})

    log = []

    def keep(frame, mask, step):
        frame = frame[mask]
        log.append({"step": step, "rows": len(frame), "invoices": frame["InvoiceNo"].nunique()})
        return frame

    log.append({"step": "raw file", "rows": len(df), "invoices": df["InvoiceNo"].nunique()})

    # reused idea from Assignment 1 (IsCancellation): C invoices are cancellations,
    # here we also drop the 3 A invoices, which are bad-debt adjustments
    df = keep(df, df["InvoiceNo"].str.strip().str.match(NORMAL_INVOICE),
              "drop cancellations (C) and adjustments (A)")

    # zero or negative quantity or price lines are returns, damages, or stock notes
    df = keep(df, (df["Quantity"] > 0) & (df["UnitPrice"] > 0),
              "drop Quantity <= 0 or UnitPrice <= 0")

    # spec: handle missing item IDs
    df = keep(df, df["Description"].notna() & df["StockCode"].notna(),
              "drop missing Description or StockCode")

    # spec: remove non-product noise
    df = keep(df, df["StockCode"].str.strip().str.match(PRODUCT_CODE),
              "keep product StockCodes only")

    # invoices with no CustomerID average about 95 distinct items vs about 21 for
    # identified customers, so they do not look like one shopper's basket
    if drop_no_customer:
        df = keep(df, df["CustomerID"].notna(), "drop invoices with no CustomerID")

    df = df.copy()
    df["StockCode"] = df["StockCode"].str.strip().str.upper()
    # reused from Assignment 1 with the regex fixed: trim and collapse repeated spaces
    df["Description"] = df["Description"].str.strip().str.replace(r"\s+", " ", regex=True)

    # one code can have spelling variants, so each code gets its most common description;
    # when two codes share that description we add the code so columns never merge
    top = df.groupby("StockCode")["Description"].agg(lambda s: s.value_counts().index[0])
    shared = top.duplicated(keep=False)
    top[shared] = top[shared] + " [" + top.index[shared] + "]"
    labels = top.to_dict()

    # a set per invoice: the same product on two lines of one invoice counts once
    transactions = df.groupby("InvoiceNo")["StockCode"].agg(lambda s: sorted(set(s))).tolist()
    return transactions, labels, pd.DataFrame(log)


def load_groceries(path=GROCERIES_PATH):
    """Read the Groceries basket file (one transaction per line, items comma-separated).

    Returns list[list[str]]. It is already clean: 9,835 transactions, 169 items.
    """
    with open(path, newline="") as f:
        return [sorted({item.strip() for item in row if item.strip()}) for row in csv.reader(f)]


def encode(transactions):
    """One-hot encode baskets with mlxtend's TransactionEncoder.

    Returns a boolean DataFrame: one row per transaction, one column per item.
    """
    te = TransactionEncoder()
    onehot = te.fit(transactions).transform(transactions)
    return pd.DataFrame(onehot, columns=te.columns_)


def profile(name, X):
    """One summary row for the report: size, basket length, and density."""
    basket = X.sum(axis=1)
    return {
        "dataset": name,
        "transactions": X.shape[0],
        "items": X.shape[1],
        "mean basket": round(basket.mean(), 1),
        "median basket": int(basket.median()),
        "density": round(basket.mean() / X.shape[1], 4),
        "dense MB": round(X.memory_usage(deep=True).sum() / 1e6, 1),
    }


# ---------------------------------------------------------------------------
# Task 3.2: mining and benchmarking
# ---------------------------------------------------------------------------

def mine(X, algo, min_support):
    """Frequent itemsets from mlxtend; algo is "apriori", "apriori_lowmem" or "fpgrowth"."""
    if algo == "apriori":
        return apriori(X, min_support=min_support, use_colnames=True)
    if algo == "apriori_lowmem":
        return apriori(X, min_support=min_support, use_colnames=True, low_memory=True)
    if algo == "fpgrowth":
        return fpgrowth(X, min_support=min_support, use_colnames=True)
    raise ValueError(f"unknown algo {algo!r}")


def apriori_level2_gb(X, min_support):
    """Memory default Apriori needs to count its 2-item candidates.

    mlxtend evaluates every candidate at once with np.all(X[:, combin], axis=2),
    a bool array of rows x candidates x 2 plus a rows x candidates result, so
    about rows * C2 * 3 bytes, where C2 = n(n-1)/2 for n frequent single items.
    """
    n = int((X.mean() >= min_support).sum())
    return X.shape[0] * (n * (n - 1) // 2) * 3 / 1e9


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout


def _timed(X, algo, min_support, timeout_s, trace):
    """One run under a wall-clock limit. Returns (seconds, peak MB or None, itemsets)."""
    # SIGALRM works on macOS and Linux in the main thread, which is where Jupyter runs cells
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    try:
        if trace:
            tracemalloc.start()
        start = time.perf_counter()
        itemsets = mine(X, algo, min_support)
        seconds = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] / 1e6 if trace else None
        return seconds, peak, itemsets
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        if trace:
            tracemalloc.stop()


def benchmark(X, algo, min_support, timeout_s=TIMEOUT_S):
    """Runtime and peak memory for one (algorithm, minsup) setting.

    Timing runs never use tracemalloc, because tracing slows Python down; peak
    memory comes from one extra traced run. Returns a dict with runtime_s,
    peak_mb, n_itemsets, max_len, status ("ok", "DNF (time)" or "DNF (memory)").
    """
    row = {"algo": algo, "min_support": min_support, "runtime_s": None,
           "peak_mb": None, "n_itemsets": None, "max_len": None, "status": "DNF (time)"}
    # default Apriori can ask for more RAM than the machine has; numpy then either
    # refuses outright or the OS kills the kernel, so we skip it above the cap
    if algo == "apriori" and apriori_level2_gb(X, min_support) > MEMORY_CAP_GB:
        row["status"] = "DNF (memory)"
        return row
    try:
        first, _, itemsets = _timed(X, algo, min_support, timeout_s, trace=False)
        times = [first]
        if first < REPEAT_UNDER_S:
            for _ in range(REPEATS - 1):
                times.append(_timed(X, algo, min_support, timeout_s, trace=False)[0])
    except _Timeout:
        return row
    row.update(runtime_s=statistics.median(times), n_itemsets=len(itemsets),
               max_len=int(itemsets["itemsets"].map(len).max()), status="ok")
    # tracing slows a run down, so a setting that finished can still time out here;
    # we keep its runtime and only lose the memory number
    try:
        row["peak_mb"] = _timed(X, algo, min_support, timeout_s, trace=True)[1]
    except _Timeout:
        row["status"] = "ok (memory trace timed out)"
    return row


def run_grid(X, dataset, grid=MINSUP_GRID, algos=ALGOS, timeout_s=TIMEOUT_S, verbose=True):
    """Benchmark every algorithm across the minsup grid, highest minsup first.

    Lowering minsup can only add frequent itemsets (anti-monotone property), so
    once an algorithm runs out of time or memory every lower minsup would too;
    those settings are recorded as DNF without running them.
    """
    rows = []
    for algo in algos:
        timed_out = False
        for ms in sorted(grid, reverse=True):
            if timed_out:
                row = {"algo": algo, "min_support": ms, "status": "DNF (skipped)"}
            else:
                row = benchmark(X, algo, ms, timeout_s)
                timed_out = row["status"].startswith("DNF")
            row["dataset"] = dataset
            rows.append(row)
            if verbose:
                rt = f"{row['runtime_s']:.2f}s" if row.get("runtime_s") is not None else row["status"]
                print(f"{dataset:<13} {algo:<9} minsup={ms:<6} {rt}", flush=True)
    cols = ["dataset", "algo", "min_support", "status", "runtime_s", "peak_mb", "n_itemsets", "max_len"]
    return pd.DataFrame(rows).reindex(columns=cols)
