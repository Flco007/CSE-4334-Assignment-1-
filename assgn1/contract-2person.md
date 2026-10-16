# Project 1 Design Contract — Two-Person Team

**CSE 4334/5334 Data Mining — Fall 2026 — Dr. Elizabeth D. Diaz**
Project 1: Data Preprocessing & Pipeline Construction

| | |
|---|---|
| Team | A: `__________` · B: `__________` |
| Agreed on | `__________` |
| Status | **FROZEN** — changes require both members to agree, in writing, in §9 |

> ⚠️ **Get approval before using this.** The spec states "Assigned Teams of 3," and the Live Team
> Demonstration rubric awards 20% for "active and equal participation of all 3 members." A
> two-person team needs written sign-off from Dr. Diaz, ideally with confirmation of how the demo
> rubric will be applied. Do that first — record the answer in §10.

---

## 0. Dataset

**File:** UCI **Online Retail** — transactions from 01/12/2010 to 09/12/2011, 541,909 rows, 8 columns.
Source: <https://archive.ics.uci.edu/dataset/352/online+retail>

⚠️ **Not** "Online Retail II" (01/12/2009–09/12/2011, ~1.07M rows). Both appear in search results.
Verify after loading: `df.shape == (541909, 8)`. If it doesn't match, stop — wrong file.

The source file is `.xlsx`, which is why `openpyxl` is in the environment.

Store at `data/online_retail.xlsx`. `data/` is gitignored — each of you downloads your own copy.

---

## 1. Roles

Split by **branch ownership**, not by task number. `ColumnTransformer` is a set of independent
sub-pipelines, which is what makes parallel work possible.

| | Person A — Data, Categorical & Assembly | Person B — Numeric & Discretization |
|---|---|---|
| **Spec tasks** | Cleaning, split, Task 1 (categorical), Task 4, Task 5 | Task 1 (numeric), Task 2, Task 3 |
| **Challenges** | **A** (MCAR/MNAR) + **C** (SMC vs Jaccard) | **B** (scaling geometry) + measurement-scale section |
| **Notebooks** | `01_data.ipynb`, `03_categorical.ipynb`, `final.ipynb` | `02_numeric.ipynb` |
| **Admin** | Git repo, notebook merge, PDF compile | AI log audit, demo timing |

**Why this pairing:** each person writes about what they measured. A computes the missingness
statistics Challenge A needs and works in the sparse binary space Challenge C reasons about. B runs
the scaler comparison that supplies Challenge B's evidence and the discretization that grounds the
measurement-scale section.

**Load balance.** A has three code units to B's two, but B's units are the heavy ones — Task 2 is
three scalers and Task 3 is three binning strategies, meaning six fitted variants with plots and
writeups. If it turns out lopsided in practice, the adjustment is to move Task 3 (discretization) to
A, along with the measurement-scale section that pairs with it. Log the change in §9.

**Demo:** 10 minutes, split evenly. Each person presents their own branches. Both attend
cross-review at Merge 2 so either can field a question about any component.

---

## 2. Data shape decisions

### 2.1 Negative Quantity → **KEEP AND FLAG**

Derive `IsCancellation = InvoiceNo.str.upper().str.startswith('C')` **before** dropping `InvoiceNo`.
Keep all cancellation rows.

**Rationale:** the spec's dataset table identifies negative quantities as cancellations and states
that extreme values distort standard scaling. Dropping them collapses Quantity's range and leaves
Task 2's scaler comparison with no anomaly to demonstrate.

### 2.2 Missing CustomerID → **KEEP ROWS, INDICATOR ONLY**

Keep all ~135,080 rows with missing `CustomerID`.
Derive `HasCustomerID = CustomerID.notna()`.
**Do not** one-hot encode the raw `CustomerID` — drop the column after deriving the indicator.

**Rationale:** dropping the rows removes ~25% of the data and leaves Challenge A with nothing to
analyze. Encoding ~4,300 identifiers adds no generalization value, and unseen test customers all
collapse to zero vectors under `handle_unknown='ignore'`. The indicator is Challenge A's recommended
fix, implemented — report and code agree.

⚠️ Person A owns both this code and Challenge A, so the risk of report/implementation contradiction
is lower here than on a three-person team — but the argument still has to run *against*
constant-imputing identifiers, not for it.

### 2.3 Null convention → **`np.nan` EVERYWHERE**

`SimpleImputer` defaults to `missing_values=np.nan`. Empty and whitespace-only strings pass straight
through into the encoder as real categories.

Person A owns both cleaning and the categorical branch, so this is a single-person consistency
check: convert `''` and whitespace-only to `np.nan` during cleaning, then re-check after text
standardization, since punctuation-only descriptions can become `''`.

---

## 3. Column routing

| Column | Decision | Goes to | Reason |
|---|---|---|---|
| `InvoiceNo` | Drop after deriving `IsCancellation` | — | Identifier, ~25k uniques |
| `StockCode` | Drop | — | Redundant with Description |
| `Description` | Keep | **Categorical (A)** | ~4,200 uniques after standardization |
| `Quantity` | Keep | **Numeric (B)** | Spec-mandated |
| `InvoiceDate` | Derive `Month`, `DayOfWeek`; drop raw | **Categorical (A)** | Raw timestamp → ~23k useless columns |
| `UnitPrice` | Keep | **Numeric (B)** *and* **Discretized (B)** | Feeds both branches — see §4 |
| `CustomerID` | Drop raw; derive `HasCustomerID` | **Categorical (A)** | See §2.2 |
| `Country` | Keep | **Categorical (A)** | 38 uniques, clean |

**Derived features are Person A's responsibility, created during cleaning, before the split:**
`IsCancellation`, `HasCustomerID`, `Month`, `DayOfWeek`

**Final column lists (both build against exactly these):**

```python
NUMERIC_COLS     = ['Quantity', 'UnitPrice']
DISCRETIZE_COLS  = ['UnitPrice']
CATEGORICAL_COLS = ['Description', 'Country', 'Month', 'DayOfWeek',
                    'IsCancellation', 'HasCustomerID']
```

---

## 4. Pipeline architecture

Three branches, two owners. `UnitPrice` appears twice — intentional and legal.

```python
ColumnTransformer([
    ('num',  numeric_pipe,     NUMERIC_COLS),      # Person B
    ('disc', discretize_pipe,  DISCRETIZE_COLS),   # Person B
    ('cat',  categorical_pipe, CATEGORICAL_COLS),  # Person A
])
```

Prefixes `num` / `disc` / `cat` are fixed. They keep `get_feature_names_out()` readable so scaled
price is distinguishable from binned price — expect to be asked about this in the demo.

**Branch contents:**

```python
# B
numeric_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  <chosen scaler — see §5>),
])

# B
discretize_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('bin',    KBinsDiscretizer(n_bins=5, encode='ordinal',
                                strategy=<chosen — see §5>,
                                random_state=42, subsample=None)),
])

# A
categorical_pipe = Pipeline([
    ('standardize', FunctionTransformer(clean_text)),   # stateless — no leakage
    ('impute',      SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encode',      OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
])
```

**Interface:** B hands A two fitted, self-contained `Pipeline` objects, each accepting a DataFrame
slice of its assigned columns. A swaps them into the skeleton at Merge 1.

**Parallelism:** B develops against a 500-row `df.sample()` matching the schema above. B never waits
for A's cleaned data, and A builds the skeleton with placeholder transformers.

---

## 5. Imputation and transformer defaults

| Decision | Choice | Reason |
|---|---|---|
| Numeric imputation | `strategy='median'` | Spec-mandated; resists outlier distortion |
| Categorical imputation | `strategy='constant', fill_value='Unknown'` | `most_frequent` would fabricate ~1,450 purchases of a specific real product. A constant is honest and encodes as its own column. |
| Scaler (**assumed default**) | `StandardScaler` | Provisional — B's Task 2 experiment decides |
| Binning (**assumed default**) | `strategy='quantile'`, `n_bins=5` | Provisional — B's Task 3 experiment decides. Uniform will likely put nearly all rows in bin 0 because £38,970 sets the width. |
| Encoder | `handle_unknown='ignore'`, `sparse_output=True` | Spec-mandated |
| Split | `train_test_split(..., test_size=0.2, random_state=42)` | Fixed seed = reproducible across both machines |

The two "assumed defaults" are **not frozen** — they are placeholders so report writing can proceed
in parallel with the experiments. If a result changes them, it's a one-paragraph edit, logged in §9.

**Text standardization (`clean_text`, Person A):** lowercase → strip punctuation → collapse
whitespace → strip. Must be a stateless `FunctionTransformer`. Stateless means no fitted parameters,
therefore no leakage — this is a defensible design point in the demo, so know it.

---

## 6. Hard rules

1. **Split before any `fit()`.** Every mean, standard deviation, bin edge, and category list comes
   from `X_train` only. This is 40% of the code grade.
2. **Never call `.toarray()`** on the full encoded matrix. ~4,200 Description categories ×
   541,909 rows dense float64 is multiple GB and will kill the kernel. Inspect with `.shape`,
   `.nnz`, and `.getnnz()`. Slice a few rows if you need to eyeball values.
3. **Nobody edits a notebook they don't own.** A merges into `final.ipynb`.
4. **Log AI prompts as you go** in `ai_log.md`. Retroactive reconstruction is impossible and it's
   10% of the grade. Template in that file.
5. **Same scikit-learn version.** Run `import sklearn; sklearn.__version__` and confirm you match
   before leaving Meeting 1. CI enforces this on every push.

---

## 7. Report assignments

| Section | Owner | Must contain |
|---|---|---|
| **Challenge A** — Imputation bias | A | Formal MCAR/MAR/MNAR definitions in this transactional context; the mathematical bias from collapsing ~135k distinct non-customers into one constant; the behavioral bias in a CLV model (phantom mega-customer with enormous aggregate spend); missingness indicator as the recommended alternative — and note it's implemented per §2.2 |
| **Challenge B** — Scaling geometry | B | Geometric proof that with Quantity differences ~10⁴ and UnitPrice differences ~10⁰, squared terms differ by ~10⁸, so Quantity determines 3-NN neighbor selection almost deterministically; why dividing by σ equalizes contribution; how outliers inflate σ and compress normal points — illustrated with B's own Task 2 numbers |
| **Challenge C** — SMC vs Jaccard | A | With 50,000 items and ~50 purchased each, show f₀₀ ≈ 49,900 drives SMC to ~0.998 regardless of real overlap; Jaccard excludes f₀₀ and returns something meaningful; frame against the supermarket line in the prompt. **Pure algebra — do not build a basket matrix.** |
| **Measurement scale theory** | B | The rubric lists this but no challenge asks for it. Short section: `KBinsDiscretizer` converts ratio-scale price to ordinal, what information is lost, why that's acceptable for tree-based and mining algorithms. |

**Sequencing note for A:** Challenge C has zero dependencies — no code, no data, no teammate. Write
it first, during the build phase, since A carries the heavier report load.

**Known spec inconsistency:** Challenge C says 50,000 distinct items; the actual dataset has ~4,000
unique StockCodes. C is written as a hypothetical, so answer it as posed — but mention the
discrepancy in one sentence so it's clear you noticed.

---

## 8. Schedule

| Milestone | Date | Definition of done |
|---|---|---|
| Approval | `______` | Dr. Diaz has signed off on a two-person team (§10) |
| Meeting 1 | `______` | This file committed. Environment installed, versions confirmed matching. |
| Merge 1 (~50%) | `______` | B's two pipelines dropped into A's skeleton. Runs end-to-end on full data. |
| Merge 2 (~85%) | `______` | Code frozen. Cross-review walkthrough. PDF compiled. |
| Rehearsal | `______` | Full 10-min demo, timed, both speaking. |
| **Submission** | `______` | Late penalty is 20% per 24 hours. |

⚠️ **Two people means no slack.** On a three-person team, one person falling behind is absorbed.
Here it's a schedule slip. Treat Merge 1 as a real checkpoint: if either branch isn't running by
that date, cut scope (drop the third binning strategy, reduce `n_bins` exploration) rather than
compressing the rehearsal.

---

## 9. Amendments

| Date | Change | Agreed by |
|---|---|---|
| | | |

---

## 10. Questions for office hours

**1. Is a two-person team permitted, and how is the 20% demo rubric applied?**
The rubric specifies "all 3 members." Get this in writing.

Asked by: `______` · Date: `______` · Answer: `______`

**2. Should `CustomerID` be one-hot encoded, or is an indicator sufficient?**
§2.2 chooses the indicator, but the spec's Task 1 language nudges toward constant-imputing the
identifier. This is the only technical decision here that changes both the pipeline architecture
*and* a report section.

Asked by: `______` · Date: `______` · Answer: `______`
