# Git Workflow & CI Reference

**CSE 4334/5334 Project 1**

Everything here assumes you've activated the environment first:

```bash
conda activate datamining-p1
```

---

## 1. One-time setup (each person, once)

```bash
git clone <repo-url>
cd project1

conda env create -f environment.yml
conda activate datamining-p1

nbstripout --install          # optional, see contract
git config user.name  "Your Name"
git config user.email "you@mavs.uta.edu"
```

Then download the dataset to `data/online_retail.xlsx`. It is **not** in the repo — `data/` is
gitignored because the file is ~23 MB and GitHub warns above 50 MB.

Verify you match your teammates:

```bash
python -c "import sklearn, pandas, numpy; print(sklearn.__version__, pandas.__version__, numpy.__version__)"
```

---

## 2. Daily workflow

Start of every session:

```bash
git pull                      # get teammates' work before you start
conda activate datamining-p1
jupyter lab
```

End of every session:

```bash
git add 02_numeric.ipynb ai_log.md
git commit -m "Task 2: RobustScaler comparison + variance plots"
git push
```

**Add specific files, not `git add .`** — that sweeps up `.ipynb_checkpoints/`, stray CSVs, and
whatever else is lying around.

---

## 3. Branching

For a project this size with two or three people on separate files, you can work directly on `main`
and it will mostly be fine. Branches are worth it only for the merge notebook.

**Minimum viable strategy:**

| Branch | Who | Purpose |
|---|---|---|
| `main` | everyone | Each person's own notebook, committed directly |
| `merge` | assembler | `final.ipynb` assembly — keeps a broken merge off `main` |

```bash
git switch -c merge           # create and switch
# ...assemble final.ipynb...
git add final.ipynb
git commit -m "Assemble final pipeline"
git switch main
git merge merge               # bring it back once it runs clean
```

This works because you each own a different file. The rule that makes it work is in the contract:
**nobody edits a notebook they don't own.**

---

## 4. Notebook merge conflicts

If two people edit the same `.ipynb`, git shows you a conflict in raw JSON. It is close to
unreadable and hand-editing it usually corrupts the file.

**Don't try to resolve it manually.** Pick a side:

```bash
git checkout --ours   final.ipynb    # keep your version
git checkout --theirs final.ipynb    # keep theirs
git add final.ipynb
git commit
```

Then manually copy the missing cells across in Jupyter. Ugly, but it always works and takes five
minutes. Trying to merge notebook JSON by hand takes an hour and often produces a file that won't
open.

Prevention beats cure: one notebook per person.

---

## 5. Command reference

### Everyday

| Command | What it does |
|---|---|
| `git status` | What's changed, what's staged. Run this constantly. |
| `git pull` | Fetch and merge teammates' commits |
| `git add <file>` | Stage a specific file |
| `git commit -m "msg"` | Record staged changes |
| `git push` | Send commits to GitHub |
| `git log --oneline -10` | Last 10 commits, one line each |
| `git diff` | Unstaged changes |
| `git diff --staged` | Staged changes, before committing |

### Branching

| Command | What it does |
|---|---|
| `git branch` | List branches, `*` marks current |
| `git switch -c <name>` | Create branch and switch to it |
| `git switch <name>` | Switch to existing branch |
| `git merge <name>` | Merge that branch into current one |
| `git branch -d <name>` | Delete a merged branch |

### Undoing

| Command | What it does |
|---|---|
| `git restore <file>` | Discard uncommitted changes to a file |
| `git restore --staged <file>` | Unstage, keep the changes |
| `git revert <hash>` | New commit that undoes an old one — safe, keeps history |
| `git reset --soft HEAD~1` | Undo last commit, keep the changes staged |
| `git stash` / `git stash pop` | Shelve changes temporarily, then restore |

⚠️ Avoid `git reset --hard` and `git push --force`. Both destroy work, and `--force` destroys your
teammates' work too. If something is broken, ask the group before reaching for either.

### Inspecting

| Command | What it does |
|---|---|
| `git log --oneline --graph --all` | Visual branch history |
| `git log -p <file>` | History of one file with diffs |
| `git blame <file>` | Who last changed each line |
| `git show <hash>` | Full contents of one commit |

---

## 6. `.gitignore`

Create this at the repo root and commit it before anything else:

```gitignore
# Data — download separately, too large for git
data/

# Jupyter
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Python
__pycache__/
*.py[cod]
.Python

# Conda / venv
.conda/
venv/
env/

# Generated outputs — regenerate, don't commit
figures/
*.pkl
*.joblib

# OS
.DS_Store
Thumbs.db

# Editors
.vscode/
.idea/
*.swp
```

**Keep `ai_log.md` tracked.** It's 10% of the grade.

---

## 7. CI pipeline

The workflow lives at `.github/workflows/ci.yml`. It runs automatically on every push and pull
request to `main`.

### What it checks

| Job | Checks | Needs data? |
|---|---|---|
| `environment` | `environment.yml` resolves; all imports succeed; versions match the pins | No |
| `notebooks` | Every notebook executes top-to-bottom with no exceptions | Yes |

The second job is the one that matters. Your spec requires a notebook "executing natively from
top-to-bottom without runtime exceptions" — CI verifies that on a clean machine, every push. It
catches the classic failure where a notebook only runs because of a variable defined in a cell you
deleted an hour ago.

### The data problem

`data/` is gitignored, so CI has no dataset. Three ways to handle it, in order of preference:

**A. Commit a sample.** Have the data owner export `df.sample(5000, random_state=42)` to
`data/sample.parquet` (~200 KB, fine for git, remove it from `.gitignore`). Notebooks read from an
env var:

```python
import os
DATA_PATH = os.environ.get("DATA_PATH", "data/online_retail.xlsx")
```

CI sets `DATA_PATH=data/sample.parquet`. Fast, reliable, and it verifies logic rather than scale.
This is what the workflow is configured for.

**B. Download in CI.** Add a step that fetches the file from the UCI archive. Verify the direct
download URL yourself before relying on it — archive URLs move, and a broken CI that cries wolf gets
ignored within a week.

**C. Environment job only.** Delete the `notebooks` job. You still catch version drift, which is the
most common breakage. Least effort, still worth having.

### Enabling it

```bash
mkdir -p .github/workflows
# put ci.yml there
git add .github/workflows/ci.yml
git commit -m "Add CI"
git push
```

Results appear under the **Actions** tab on GitHub. A red X means someone pushed a notebook that
doesn't run — fix it then, not the night before the demo.

### Reading a failure

Click the failed run → click the failed job → expand the red step. For notebook failures the
traceback is the same one you'd see in Jupyter, including the cell number. If it passes locally but
fails in CI, the cause is almost always execution order: re-run yours with **Kernel → Restart and
Run All** and you'll usually reproduce it.

---

## 8. Suggested commit rhythm

Commit when something works, not when you're done for the day. Small commits give you somewhere to
return to.

Reasonable messages:

```
Task 2: median imputer + StandardScaler baseline
Task 2: add MinMax and Robust comparison plots
Task 3: quantile binning, n_bins=5
Fix: convert whitespace-only Descriptions to NaN
Challenge B: geometric proof draft
```

Not this: `update`, `stuff`, `asdf`, `final`, `final2`, `FINAL_actually`.
