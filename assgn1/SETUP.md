# Environment Setup — CSE 4334/5334 Project 1

Follow this once per machine. All three team members must end up on identical versions, or
the notebook will produce different numbers for different people.

Estimated time: 15 minutes, most of it waiting on `conda`.

---

## Why the versions are pinned

The spec says "Python 3.8+ (Anaconda)." Do not install 3.8. Current scikit-learn (1.9.0,
released June 2026) requires **Python 3.11 or newer**, and the project depends on several API
features that only exist in recent versions.

Three specific things break when versions drift:

| Change | Version | Consequence if you're on the wrong side |
|---|---|---|
| `OneHotEncoder(sparse=)` renamed to `sparse_output=` | renamed 1.2, old name **removed** 1.4 | `TypeError` on any code using the old name |
| `OneHotEncoder(min_frequency=)` added | 1.1 | `TypeError`; no way to cap one-hot cardinality |
| `KBinsDiscretizer(quantile_method=)` default changed `"linear"` → `"averaged_inverted_cdf"` | added 1.7, default changed 1.9 | **Different bin edges.** Silent, not an error. |

That last one is the dangerous one — it produces no warning on 1.9 and different results on
1.7/1.8 unless the parameter is passed explicitly. The notebook passes it explicitly for this
reason, but you should still pin.

---

## Step 1 — Install Miniconda

If you already have Anaconda or Miniconda, skip this.

Miniconda is preferred over full Anaconda: it's ~400 MB instead of ~3 GB and installs only
what you ask for. Download from `https://www.anaconda.com/download/success` (Miniconda section)
and run the installer for your OS.

Verify:

```bash
conda --version
```

---

## Step 2 — Create the environment

**Option A (preferred) — from the shared file.** Save the block below as `environment.yml` in
the repo root, then create from it. This is the version that gets committed, and it's what
keeps the team in sync.

```yaml
name: cse4334
channels:
  - conda-forge
dependencies:
  - python=3.12
  - scikit-learn=1.9
  - pandas
  - numpy
  - scipy
  - matplotlib
  - seaborn
  - jupyterlab
  - ipykernel
  - openpyxl      # required to read the .xlsx source file
  - pyarrow       # required for the Parquet cache
  - joblib        # persisting the fitted preprocessor
  - nbstripout    # keeps notebook diffs readable in git
```

```bash
conda env create -f environment.yml
conda activate cse4334
```

**Option B — one-liner**, if you'd rather not create the file first. Note this is written on
a single line: the `\` continuation character used in most shell examples is POSIX-only and
will break in Windows CMD (which uses `^`) and PowerShell (which uses a backtick).

```bash
conda create -n cse4334 -c conda-forge python=3.12 scikit-learn=1.9 pandas numpy scipy matplotlib seaborn jupyterlab ipykernel openpyxl pyarrow joblib nbstripout -y
conda activate cse4334
```

**Windows: `conda activate` may fail the first time** with a message about your shell not
being initialized. Either use the **Anaconda Prompt** (installed alongside Miniconda) instead
of CMD/PowerShell, or run `conda init powershell` once and restart the terminal.

Python 3.12 is the recommendation over 3.13/3.14 — it's a year into its support window, so
every package in the list has stable prebuilt wheels for it.

---

## Step 3 — Register the Jupyter kernel

Without this, JupyterLab won't offer the environment as a kernel choice and you'll silently
run the notebook against your base Python.

```bash
python -m ipykernel install --user --name cse4334 --display-name "CSE 4334"
```

Confirm it registered:

```bash
jupyter kernelspec list
```

You should see `cse4334` in the output. When you open the notebook, check the kernel name in
the top-right corner reads **CSE 4334**.

---

## Step 4 — Get the dataset

The UCI Online Retail dataset is `id=352`. Two routes:

**Manual download** (simplest): go to `https://archive.ics.uci.edu/dataset/352/online+retail`,
download the archive, and place `Online Retail.xlsx` in `data/`.

**Programmatic**:

```bash
pip install ucimlrepo
```

The notebook's loader falls back to `fetch_ucirepo(id=352)` automatically if it doesn't find
the `.xlsx`.

**Verify you have the right file.** The correct dataset is **541,909 rows × 8 columns**,
covering 2010-12-01 to 2011-12-09. If you get 1,067,371 rows you downloaded *Online Retail II*
(id=502), which is a different, larger dataset spanning 2009–2011. Every number in your report
depends on getting this right — the notebook asserts the shape in §2 for exactly this reason.

The dataset is CC BY 4.0. Cite it in the report:

> Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5BW33

---

## Step 5 — Verify

Save the following as `verify_env.py` in the repo root and run it. It checks versions, confirms
each API feature the project needs actually exists, and runs a miniature end-to-end pipeline.

```bash
python verify_env.py
```

```python
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
```

Do not start on the notebook until this exits with `ALL CHECKS PASSED`.

---

## Working directory vs. environment — they're independent

Two things people conflate. The conda environment is **shell state**, not directory state:
once you `conda activate cse4334`, it stays active no matter where you `cd`. There is no
"enter the project folder to pick up the environment" — that's a Python `venv`/`node_modules`
convention, not how conda works. Conversely, `cd`-ing into the repo does **not** activate
anything.

So: the environment doesn't care where you are. But several commands do.

| Command | Needs the right directory? | Notes |
|---|---|---|
| `conda activate cse4334` | no | Works from anywhere; persists across `cd` |
| `git` anything | **yes** | Must be somewhere inside the repo tree |
| `nbstripout --install` | **yes** | Configures *this* repo's `.git/config` |
| `python verify_env.py` | yes, or pass a path | It's just a script location |
| `jupyter lab` | **yes — repo root** | Sets the file-browser root; see below |
| `conda env create -f environment.yml` | yes, or pass a full path | |

**Launch `jupyter lab` from the repo root.** If you launch it from inside `notebooks/`, the
file browser can't see `data/` or `report/` — Jupyter won't let you navigate above the
directory it was started in.

```bash
cd ~/path/to/project1     # repo root, where environment.yml lives
conda activate cse4334
jupyter lab
```

**The subtle part: the kernel's working directory is the notebook's folder, not where you
launched from.** A notebook in `notebooks/` gets a kernel whose cwd is `notebooks/`, so a
relative `Path("data")` inside the notebook resolves to `notebooks/data/` and fails. The
notebook handles this by walking upward to find `environment.yml` and anchoring all paths to
the repo root it finds — which is another reason that file belongs in the root and should be
committed early. Section 2.0 of the notebook prints both paths so you can confirm.

**Activation is per-terminal.** Every new terminal window, tab, reboot, or VS Code integrated
terminal starts in `base`. "It worked yesterday" is almost always this.

---

## Step 6 — Repository layout

```
project1/
├── data/                       # gitignored
│   ├── Online Retail.xlsx      # 23 MB source, downloaded once
│   └── online_retail.parquet   # generated cache, regenerates itself
├── artifacts/                  # gitignored; fitted preprocessor + report CSVs
├── notebooks/
│   └── project1_pipeline_scaffold.ipynb
├── report/
│   ├── part1_critical_thinking.md
│   └── ai_prompt_log.md        # start this TODAY, not at the end
├── environment.yml
├── verify_env.py
├── .gitignore
├── .gitattributes              # line-ending policy; see Step 7
└── README.md
```

`.gitignore`:

```gitignore
data/
artifacts/
.ipynb_checkpoints/
__pycache__/
*.pyc
.DS_Store
.venv/
```

Never commit the dataset. It's 23 MB of binary, it's freely redownloadable, and git will keep
every version of it forever.

---

## Step 7 — Notebook hygiene for a 3-person team

Jupyter stores cell outputs, execution counts, and metadata inside the `.ipynb` JSON. Two
people editing the same notebook produce merge conflicts in base64-encoded PNG data, which is
unresolvable in practice.

### Line endings — do this before anyone commits

If your team spans Windows and Unix, git's `core.autocrlf` will rewrite line endings on
checkout. A `.ipynb` is a single JSON file, so a line-ending conversion changes **every line**,
and git reports the whole notebook as modified even when nobody touched it. Merges become
impossible.

Commit this `.gitattributes` in the repo root before the first push:

```gitattributes
* text=auto eol=lf
*.ipynb text eol=lf
*.py    text eol=lf
*.md    text eol=lf
*.yml   text eol=lf
*.xlsx  binary
*.parquet binary
```

This forces LF in the repository regardless of platform. If someone has already committed with
CRLF, fix it once with `git add --renormalize .` and commit the result.

### Output stripping

Install the strip filter once per clone:

```bash
nbstripout --install
```

This clears outputs on commit, so diffs show only the code and prose that actually changed.
Your local copy keeps its outputs — only what git sees is stripped.

**Before submitting**, re-run everything so the graded notebook has outputs:

```
Kernel → Restart Kernel and Run All Cells
```

Confirm zero exceptions. The spec requires it to execute natively top-to-bottom.

### Working in parallel without conflicts

The strip filter reduces conflicts but doesn't eliminate them. The reliable approach is to
avoid simultaneous edits entirely:

- **Split by section, not by file.** Agree who owns §7 (scaling), §8 (binning), §9 (encoding)
  and merge one at a time.
- **Or move logic into `src/preprocessing.py`** and have the notebook import it. Three people
  can edit a `.py` concurrently; three people cannot edit an `.ipynb` concurrently.
- **Announce before you push.** For a two-week project, a message in the group chat is cheaper
  than a merge strategy.

---

## Step 8 — Freeze the environment when it works

Once `verify_env.py` passes on all three machines, one person regenerates the lockfile and
commits it:

```bash
conda env export --from-history > environment.yml
```

Use `--from-history`. A plain `conda env export` pins exact build strings for your specific OS
and architecture (`py312h4f0b9e3_0` and similar), which makes the file unusable for a teammate
on a different platform — a real problem if anyone is on Apple Silicon while others are on
Windows.

One caveat: `--from-history` records only what you explicitly asked conda to install. It does
**not** capture anything installed with `pip`, so if you used `pip install ucimlrepo`, note it
separately in the README or add it as a `pip:` subsection in `environment.yml`.

---

## Platform notes — ARM Mac, Windows, Linux

The package set works on all three. conda-forge ships scikit-learn 1.9.0 builds for `win-64`,
`osx-64`, `osx-arm64`, `linux-64`, and `linux-aarch64`, and every other dependency in
`environment.yml` is available on those platforms. Nothing in the notebook uses
platform-specific code — file paths go through `pathlib`, so `data/Online Retail.xlsx` resolves
correctly on Windows without any change.

The differences are in the surrounding tooling, not the libraries.

### Apple Silicon (M1–M4)

- **Download the arm64 Miniconda installer**, not the x86_64 one. If you install the Intel
  build it runs under Rosetta — functional, but noticeably slower and it makes architecture
  mismatches possible later. Check with `python -c "import platform; print(platform.machine())"`,
  which should print `arm64`.
- **Use the `conda-forge` channel**, as `environment.yml` does. Anaconda's `defaults` channel
  has historically lagged on `osx-arm64` coverage; conda-forge builds it as a first-class
  platform.
- Native arm64 is generally the fastest of the three for this workload, since scikit-learn
  links against Accelerate/OpenBLAS builds tuned for the platform.

### Windows

- `conda activate` requires either the **Anaconda Prompt** or a one-time `conda init powershell`.
  This is the single most common Windows stumble.
- Backslash line continuations in shell examples will not work. Use the single-line commands
  given above.
- `nbstripout --install` writes a git filter that points at your Python executable. It works,
  but if your conda environment lives under a path containing spaces, the filter can silently
  fail to run. Verify with `git config --get filter.nbstripout.clean` and confirm the path
  looks sane.
- The `.gitattributes` file in Step 7 is not optional on a mixed-platform team.

### Linux

- No special handling required. If `conda activate` isn't recognized, run `conda init bash`
  (or `zsh`) once and restart the shell.

### Will all three produce identical numbers?

Effectively yes, but not bit-for-bit, and it's worth knowing the difference before someone
raises it as a bug.

Order statistics are exactly reproducible across platforms: **median imputation values and
quantile bin edges will match precisely**, because they're computed by sorting and selecting,
not by arithmetic that can reassociate.

Floating-point *reductions* — the mean and standard deviation that `StandardScaler` learns —
may differ in the last few bits. Different BLAS implementations and different SIMD widths
(AVX on x86 vs. NEON on ARM) sum long arrays in different orders, and floating-point addition
isn't associative. Expect agreement to roughly 1e-12 relative error.

Practically: any number you round to four decimals for the report will match across all three
machines. If you write an equality assertion comparing teammates' outputs, use
`np.allclose(a, b)` rather than `a == b`. A disagreement in the **third** decimal place is not
floating point — that's a version or configuration difference, and you should compare
`verify_env.py` output.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError: __init__() got an unexpected keyword argument 'sparse_output'` | scikit-learn < 1.2 | Upgrade. The parameter is `sparse` on old versions, but don't use it — it was removed in 1.4. |
| `TypeError: ... unexpected keyword argument 'sparse'` | scikit-learn ≥ 1.4 running old code | Use `sparse_output=`. Common in AI-generated snippets — note it in the Part II critique. |
| `FutureWarning: The current default behavior of quantile_method...` | sklearn 1.7/1.8, parameter not passed | Pass `quantile_method="averaged_inverted_cdf"` explicitly. |
| `ImportError: Missing optional dependency 'openpyxl'` | `read_excel` needs it | `conda install -c conda-forge openpyxl` |
| `ImportError: Unable to find a usable engine ... pyarrow` | Parquet cache needs it | `conda install -c conda-forge pyarrow` |
| Kernel "CSE 4334" not listed in JupyterLab | Step 3 skipped | Re-run the `ipykernel install` command, then reload the browser tab. |
| Notebook runs but imports fail | Wrong kernel selected | Check the kernel name top-right. Switch via `Kernel → Change Kernel`. |
| `read_excel` takes 30–60 s every restart | Reading the `.xlsx` each time | Expected on first run only. The notebook caches to Parquet; subsequent loads are <1 s. |
| Different bin edges than a teammate | Version drift, or `subsample` randomness | Compare `verify_env.py` output. The notebook sets `subsample=None` to force exact edges. |
| `MemoryError` during one-hot encoding | Encoding a high-cardinality column densely | Confirm `sparse_output=True` and `min_frequency` are set. See notebook §3.2 and §10.1a. |
| Merge conflict inside a `.ipynb` | Two people edited the same notebook | `nbstripout --install`, then split work by section. |
| Whole notebook shows as changed, but nobody edited it | CRLF/LF conversion on a mixed-platform team | Add the `.gitattributes` from Step 7, then `git add --renormalize .` |
| Windows: `conda: command not found` or activate fails | Shell not initialized | Use Anaconda Prompt, or `conda init powershell` and restart the terminal. |
| Windows: `\` at end of line causes an error | POSIX continuation in CMD/PowerShell | Use the single-line command in Step 2. |
| Mac: `platform.machine()` prints `x86_64` on an M-series Mac | Intel Miniconda installed, running under Rosetta | Reinstall using the arm64 installer. |
| Teammates' scaler means differ around the 12th decimal | Different BLAS/SIMD summation order | Expected. Compare with `np.allclose`, not `==`. |

---

## Checklist

- [ ] `conda activate cse4334` works
- [ ] `jupyter kernelspec list` shows `cse4334`
- [ ] `data/Online Retail.xlsx` exists and loads to **541,909 × 8**
- [ ] `python verify_env.py` prints `ALL CHECKS PASSED`
- [ ] `jupyter lab` launched from the repo root; notebook §2.0 reports the correct repo root
- [ ] `nbstripout --install` run in the clone
- [ ] `.gitattributes` committed **before** the first push (mixed-platform teams)
- [ ] `environment.yml` committed; `data/` and `artifacts/` gitignored
- [ ] `report/ai_prompt_log.md` created and receiving entries
- [ ] All three members completed the above and compared `verify_env.py` output
